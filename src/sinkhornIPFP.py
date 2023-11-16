# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This source code is modified from the original source code.
# See https://github.com/ott-jax/ott/blob/main/src/ott/solvers/linear/sinkhorn.py for the original source code.

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import jax
import jax.experimental
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from ott import utils
from ott.geometry import geometry
from ott.initializers.linear import initializers as init_lib
from ott.math import fixed_point_loop
from ott.math import unbalanced_functions as uf
from ott.math import utils as mu
from ott.problems.linear import linear_problem, potentials
from ott.solvers.linear import acceleration
from ott.solvers.linear import implicit_differentiation as implicit_lib
from ott.solvers.linear.sinkhorn import Sinkhorn, SinkhornState, SinkhornOutput, solve

# if TYPE_CHECKING:
#     from ott.solvers.linear.sinkhorn_lr import LRISinkhornOutput


# ProgressCallbackFn_t = Callable[[Tuple[np.ndarray, np.ndarray, np.ndarray, "SinkhornState"]], None]


class IPFPState(NamedTuple):
    """Holds the state variables used to solve OT with IPFP."""

    errors: Optional[jnp.ndarray] = None
    fu: Optional[jnp.ndarray] = None
    gv: Optional[jnp.ndarray] = None
    old_fus: Optional[jnp.ndarray] = None
    old_mapped_fus: Optional[jnp.ndarray] = None

    def set(self, **kwargs: Any) -> "SinkhornState":
        """Return a copy of self, with potential overwrites."""
        return self._replace(**kwargs)

    def solution_error(
        self,
        ot_prob: linear_problem.LinearProblem,
        norm_error: Sequence[int],
        *,
        lse_mode: bool,
        parallel_dual_updates: bool,
        recenter: bool,
    ) -> jnp.ndarray:
        """State dependent function to return error."""
        fu, gv = self.fu, self.gv
        if recenter and lse_mode:
            fu, gv = self.recenter(fu, gv, ot_prob=ot_prob)

        return solution_error(
            fu,
            gv,
            ot_prob,
            norm_error=norm_error,
            lse_mode=lse_mode,
            parallel_dual_updates=parallel_dual_updates,
        )

    def compute_kl_reg_cost(  # noqa: D102
        self, ot_prob: linear_problem.LinearProblem, lse_mode: bool
    ) -> float:
        return compute_kl_reg_cost(self.fu, self.gv, ot_prob, lse_mode)

    def recenter(
        self,
        f: jnp.ndarray,
        g: jnp.ndarray,
        ot_prob: linear_problem.LinearProblem,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Re-center dual potentials.

        If the ``ot_prob`` is balanced, the ``f`` potential is zero-centered.
        Otherwise, use prop. 2 of :cite:`sejourne:22` re-center the potentials iff
        ``tau_a < 1`` and ``tau_b < 1``.

        Args:
          f: The first dual potential.
          g: The second dual potential.
          ot_prob: Linear OT problem.

        Returns:
          The centered potentials.
        """
        if ot_prob.is_balanced:
            # center the potentials for numerical stability
            is_finite = jnp.isfinite(f)
            shift = jnp.sum(jnp.where(is_finite, f, 0.0)) / jnp.sum(is_finite)
            return f - shift, g + shift

        if ot_prob.tau_a == 1.0 or ot_prob.tau_b == 1.0:
            # re-centering wasn't done during the lse-step, ignore
            return f, g

        rho_a = uf.rho(ot_prob.epsilon, ot_prob.tau_a)
        rho_b = uf.rho(ot_prob.epsilon, ot_prob.tau_b)
        tau = rho_a * rho_b / (rho_a + rho_b)

        shift = tau * (
            mu.logsumexp(-f / rho_a, b=ot_prob.a) - mu.logsumexp(-g / rho_b, b=ot_prob.b)
        )
        return f + shift, g - shift


def solution_error(
    f_u: jnp.ndarray,
    g_v: jnp.ndarray,
    ot_prob: linear_problem.LinearProblem,
    *,
    norm_error: Sequence[int],
    lse_mode: bool,
    parallel_dual_updates: bool,
) -> jnp.ndarray:
    """Given two potential/scaling solutions, computes deviation to optimality.

    When the ``ot_prob`` problem is balanced and the usual IPFP updates are
    used, this is simply deviation of the coupling's marginal to ``ot_prob.b``.
    This is the case because the second (and last) update of the IPFP
    algorithm equalizes the row marginal of the coupling to ``ot_prob.a``. To
    simplify the logic, this is parameterized by checking whether
    `parallel_dual_updates = False`.

    When that flag is `True`, or when the problem is unbalanced,
    additional quantities to qualify optimality must be taken into account.

    Args:
      f_u: jnp.ndarray, potential or scaling
      g_v: jnp.ndarray, potential or scaling
      ot_prob: linear OT problem
      norm_error: int, p-norm used to compute error.
      lse_mode: True if log-sum-exp operations, False if kernel vector products.
      parallel_dual_updates: Whether potentials/scalings were computed in
        parallel.

    Returns:
      a positive number quantifying how far from optimality current solution is.
    """
    if ot_prob.is_balanced and not parallel_dual_updates:
        return marginal_error(f_u, g_v, ot_prob.b, ot_prob.geom, 0, norm_error, lse_mode)

    # In the unbalanced case, we compute the norm of the gradient.
    # the gradient is equal to the marginal of the current plan minus
    # the gradient of < z, rho_z(exp^(-h/rho_z) -1> where z is either a or b
    # and h is either f or g. Note this is equal to z if rho_z → inf, which
    # is the case when tau_z → 1.0
    if lse_mode:
        grad_a = uf.grad_of_marginal_fit(ot_prob.a, f_u, ot_prob.tau_a, ot_prob.epsilon)
        grad_b = uf.grad_of_marginal_fit(ot_prob.b, g_v, ot_prob.tau_b, ot_prob.epsilon)
    else:
        u = ot_prob.geom.potential_from_scaling(f_u)
        v = ot_prob.geom.potential_from_scaling(g_v)
        grad_a = uf.grad_of_marginal_fit(ot_prob.a, u, ot_prob.tau_a, ot_prob.epsilon)
        grad_b = uf.grad_of_marginal_fit(ot_prob.b, v, ot_prob.tau_b, ot_prob.epsilon)
    err = marginal_error(f_u, g_v, grad_a, ot_prob.geom, 1, norm_error, lse_mode)
    err += marginal_error(f_u, g_v, grad_b, ot_prob.geom, 0, norm_error, lse_mode)
    return err


def marginal_error(
    f_u: jnp.ndarray,
    g_v: jnp.ndarray,
    target: jnp.ndarray,
    geom: geometry.Geometry,
    axis: int = 0,
    norm_error: Sequence[int] = (1,),
    lse_mode: bool = True,
) -> jnp.asarray:
    """Output how far IPFP solution is w.r.t target.

    Args:
      f_u: a vector of potentials or scalings for the first marginal.
      g_v: a vector of potentials or scalings for the second marginal.
      target: target marginal.
      geom: Geometry object.
      axis: axis (0 or 1) along which to compute marginal.
      norm_error: (tuple of int) p's to compute p-norm between marginal/target
      lse_mode: whether operating on scalings or potentials

    Returns:
      Array of floats, quantifying difference between target / marginal.
    """
    if lse_mode:
        marginal = geom.marginal_from_potentials(f_u, g_v, axis=axis)
    else:
        marginal = geom.marginal_from_scalings(f_u, g_v, axis=axis)
    norm_error = jnp.asarray(norm_error)
    if axis == 0:
        return jnp.sum(
            jnp.abs(marginal + f_u**2 - target) ** norm_error[:, jnp.newaxis], axis=1
        ) ** (1.0 / norm_error)
    else:
        return jnp.sum(
            jnp.abs(marginal + g_v**2 - target) ** norm_error[:, jnp.newaxis], axis=1
        ) ** (1.0 / norm_error)


def compute_kl_reg_cost(
    f: jnp.ndarray, g: jnp.ndarray, ot_prob: linear_problem.LinearProblem, lse_mode: bool
) -> float:
    r"""Compute objective of IPFP for OT problem given dual solutions.

    The objective is evaluated for dual solution ``f`` and ``g``, using
    information contained in  ``ot_prob``. The objective is the regularized
    optimal transport cost (i.e. the cost itself plus entropic and unbalanced
    terms). Situations where marginals ``a`` or ``b`` in ot_prob have zero
    coordinates are reflected in minus infinity entries in their corresponding
    dual potentials. To avoid NaN that may result when multiplying 0's by infinity
    values, ``jnp.where`` is used to cancel these contributions.

    Args:
      f: jnp.ndarray, potential
      g: jnp.ndarray, potential
      ot_prob: linear optimal transport problem.
      lse_mode: bool, whether to compute total mass in lse or kernel mode.

    Returns:
      The regularized transport cost.
    """
    supp_a = ot_prob.a > 0
    supp_b = ot_prob.b > 0
    fa = ot_prob.geom.potential_from_scaling(ot_prob.a)
    if ot_prob.tau_a == 1.0:
        div_a = jnp.sum(jnp.where(supp_a, ot_prob.a * (f - fa), 0.0))
    else:
        rho_a = uf.rho(ot_prob.epsilon, ot_prob.tau_a)
        div_a = -jnp.sum(jnp.where(supp_a, ot_prob.a * uf.phi_star(-(f - fa), rho_a), 0.0))

    gb = ot_prob.geom.potential_from_scaling(ot_prob.b)
    if ot_prob.tau_b == 1.0:
        div_b = jnp.sum(jnp.where(supp_b, ot_prob.b * (g - gb), 0.0))
    else:
        rho_b = uf.rho(ot_prob.epsilon, ot_prob.tau_b)
        div_b = -jnp.sum(jnp.where(supp_b, ot_prob.b * uf.phi_star(-(g - gb), rho_b), 0.0))

    # Using https://arxiv.org/pdf/1910.12958.pdf (24)
    if lse_mode:
        total_sum = jnp.sum(ot_prob.geom.marginal_from_potentials(f, g))
    else:
        # u = ot_prob.geom.scaling_from_potential(f)
        # v = ot_prob.geom.scaling_from_potential(g)
        # total_sum = jnp.sum(ot_prob.geom.marginal_from_scalings(u, v))
        total_sum = jnp.sum(ot_prob.geom.marginal_from_scalings(f, g))
    return div_a + div_b + ot_prob.epsilon * (jnp.sum(ot_prob.a) * jnp.sum(ot_prob.b) - total_sum)


@jax.tree_util.register_pytree_node_class
class IPFP(Sinkhorn):
    r"""IPFP solver."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(
        self,
        ot_prob: linear_problem.LinearProblem,
        init: Tuple[Optional[jnp.ndarray], Optional[jnp.ndarray]] = (None, None),
        rng: Optional[jax.random.PRNGKeyArray] = None,
    ) -> SinkhornOutput:
        """Run IPFP algorithm.

        Args:
          ot_prob: Linear OT problem.
          init: Initial dual potentials/scalings f_u and g_v, respectively.
            Any `None` values will be initialized using the initializer.
          rng: Random number generator key for stochastic initialization.

        Returns:
          The IPFP output.
        """
        assert self.lse_mode == False, "IPFP only supports kernel mode."
        rng = utils.default_prng_key(rng)
        initializer = self.create_initializer()
        init_dual_a, init_dual_b = initializer(ot_prob, *init, lse_mode=self.lse_mode, rng=rng)
        return run(ot_prob, self, (init_dual_a, init_dual_b))

    def lse_step(
        self, ot_prob: linear_problem.LinearProblem, state: SinkhornState, iteration: int
    ) -> SinkhornState:
        """IPFP LSE update."""

        def k(tau_i: float, tau_j: float) -> float:
            num = -tau_j * (tau_a - 1) * (tau_b - 1) * (tau_i - 1)
            denom = (tau_j - 1) * (tau_a * (tau_b - 1) + tau_b * (tau_a - 1))
            return num / denom

        def xi(tau_i: float, tau_j: float) -> float:
            k_ij = k(tau_i, tau_j)
            return k_ij / (1.0 - k_ij)

        def smin(potential: jnp.ndarray, marginal: jnp.ndarray, tau: float) -> float:
            rho = uf.rho(ot_prob.epsilon, tau)
            return -rho * mu.logsumexp(-potential / rho, b=marginal)

        # only for an unbalanced problems with `tau_{a,b} < 1`
        recenter = self.recenter_potentials and ot_prob.tau_a < 1.0 and ot_prob.tau_b < 1.0
        w = self.momentum.weight(state, iteration)
        tau_a, tau_b = ot_prob.tau_a, ot_prob.tau_b
        old_fu, old_gv = state.fu, state.gv

        if recenter:
            k11, k22 = k(tau_a, tau_a), k(tau_b, tau_b)
            xi12, xi21 = xi(tau_a, tau_b), xi(tau_b, tau_a)

        # update g potential
        new_gv = tau_b * ot_prob.geom.update_potential(
            old_fu, old_gv, jnp.log(ot_prob.b), iteration, axis=0
        )
        if recenter:
            new_gv -= k22 * smin(old_fu, ot_prob.a, tau_a)
            new_gv += xi21 * smin(new_gv, ot_prob.b, tau_b)
        gv = self.momentum(w, old_gv, new_gv, self.lse_mode)

        if not self.parallel_dual_updates:
            old_gv = gv

        # update f potential
        new_fu = tau_a * ot_prob.geom.update_potential(
            old_fu, old_gv, jnp.log(ot_prob.a), iteration, axis=1
        )
        if recenter:
            new_fu -= k11 * smin(old_gv, ot_prob.b, tau_b)
            new_fu += xi12 * smin(new_fu, ot_prob.a, tau_a)
        fu = self.momentum(w, old_fu, new_fu, self.lse_mode)

        return state.set(fu=fu, gv=gv)

    def kernel_step(
        self, ot_prob: linear_problem.LinearProblem, state: SinkhornState, iteration: int
    ) -> SinkhornState:
        """IPFP multiplicative update."""
        w = self.momentum.weight(state, iteration)
        old_gv = state.gv
        new_gv = (
            ot_prob.geom.update_scaling(state.fu, ot_prob.b, iteration, axis=0) ** ot_prob.tau_b
        )
        gv = self.momentum(w, state.gv, new_gv, self.lse_mode)
        new_fu = (
            ot_prob.geom.update_scaling(
                old_gv if self.parallel_dual_updates else gv, ot_prob.a, iteration, axis=1
            )
            ** ot_prob.tau_a
        )
        fu = self.momentum(w, state.fu, new_fu, self.lse_mode)
        return state.set(fu=fu, gv=gv)

    def one_iteration(
        self,
        ot_prob: linear_problem.LinearProblem,
        state: SinkhornState,
        iteration: int,
        compute_error: bool,
    ) -> SinkhornState:
        """Carries out one IPFP iteration.

        Depending on lse_mode, these iterations can be either in:

          - log-space for numerical stability.
          - scaling space, using standard kernel-vector multiply operations.

        Args:
          ot_prob: the transport problem definition
          state: SinkhornState named tuple.
          iteration: the current iteration of the IPFP loop.
          compute_error: flag to indicate this iteration computes/stores an error

        Returns:
          The updated state.
        """
        # When running updates in parallel (Gauss-Seidel mode), old_g_v will be
        # used to update f_u, rather than the latest g_v computed in this loop.
        # Unused otherwise.
        if self.anderson:
            state = self.anderson.update(state, iteration, ot_prob, self.lse_mode)

        if self.lse_mode:  # In lse_mode, run additive updates.
            state = self.lse_step(ot_prob, state, iteration)
        else:
            state = self.kernel_step(ot_prob, state, iteration)

        if self.anderson:
            state = self.anderson.update_history(state, ot_prob, self.lse_mode)

        # re-computes error if compute_error is True, else set it to inf.
        err = jax.lax.cond(
            jnp.logical_or(
                iteration == self.max_iterations - 1,
                jnp.logical_and(compute_error, iteration >= self.min_iterations),
            ),
            lambda state, prob: state.solution_error(
                prob,
                self.norm_error,
                lse_mode=self.lse_mode,
                parallel_dual_updates=self.parallel_dual_updates,
                recenter=self.recenter_potentials,
            )[0],
            lambda *_: jnp.inf,
            state,
            ot_prob,
        )
        errors = state.errors.at[iteration // self.inner_iterations, :].set(err)
        state = state.set(errors=errors)

        if self.progress_fn is not None:
            jax.experimental.io_callback(
                self.progress_fn,
                None,
                (iteration, self.inner_iterations, self.max_iterations, state),
            )
        return state

    def _converged(self, state: SinkhornState, iteration: int) -> bool:
        err = state.errors[iteration // self.inner_iterations - 1, 0]
        return jnp.logical_and(iteration > 0, err < self.threshold)

    def _diverged(self, state: SinkhornState, iteration: int) -> bool:
        err = state.errors[iteration // self.inner_iterations - 1, 0]
        return jnp.logical_not(jnp.isfinite(err))

    def _continue(self, state: SinkhornState, iteration: int) -> bool:
        """Continue while not(converged) and not(diverged)."""
        return jnp.logical_and(
            jnp.logical_not(self._diverged(state, iteration)),
            jnp.logical_not(self._converged(state, iteration)),
        )

    @property
    def outer_iterations(self) -> int:
        """Upper bound on number of times inner_iterations are carried out.

        This integer can be used to set constant array sizes to track the algorithm
        progress, notably errors.
        """
        return np.ceil(self.max_iterations / self.inner_iterations).astype(int)

    def init_state(
        self, ot_prob: linear_problem.LinearProblem, init: Tuple[jnp.ndarray, jnp.ndarray]
    ) -> SinkhornState:
        """Return the initial state of the loop."""
        fu, gv = init
        errors = -jnp.ones((self.outer_iterations, len(self.norm_error)), dtype=fu.dtype)
        state = IPFPState(errors=errors, fu=fu, gv=gv)
        return self.anderson.init_maps(ot_prob, state) if self.anderson else state

    def output_from_state(
        self, ot_prob: linear_problem.LinearProblem, state: SinkhornState
    ) -> SinkhornOutput:
        """Create an output from a loop state.

        Note:
          When differentiating the regularized OT cost, and assuming IPFP has
          run to convergence, Danskin's (or the envelope)
          `theorem <https://en.wikipedia.org/wiki/Danskin%27s_theorem>`_
          :cite:`danskin:67,bertsekas:71`
          states that the resulting OT cost as a function of the inputs
          (``geometry``, ``a``, ``b``) behaves locally as if the dual optimal
          potentials were frozen and did not vary with those inputs.

          Notice this is only valid, as when using ``implicit_differentiation``
          mode, if the IPFP algorithm outputs potentials that are near optimal.
          namely when the threshold value is set to a small tolerance.

          The flag ``use_danskin`` controls whether that assumption is made. By
          default, that flag is set to the value of ``implicit_differentiation`` if
          not specified. If you wish to compute derivatives of order 2 and above,
          set ``use_danskin`` to ``False``.

        Args:
          ot_prob: the transport problem.
          state: a SinkhornState.

        Returns:
          A SinkhornOutput.
        """
        geom = ot_prob.geom

        f = state.fu if self.lse_mode else geom.potential_from_scaling(state.fu)
        g = state.gv if self.lse_mode else geom.potential_from_scaling(state.gv)
        if self.recenter_potentials:
            f, g = state.recenter(f, g, ot_prob=ot_prob)

        # By convention, the algorithm is said to have converged if the algorithm
        # has not nan'ed during iterations (notice some errors might be infinite,
        # this convention is used when the error is not recomputed), and if the
        # last recorded error is lower than the threshold. Note that this will be
        # the case if either the algorithm terminated earlier (in which case the
        # last state.errors[-1] = -1 by convention) or if the algorithm carried out
        # the maximal number of iterations and its last recorded error (at -1
        # position) is lower than the threshold.

        converged = jnp.logical_and(
            jnp.logical_not(jnp.any(jnp.isnan(state.errors))), state.errors[-1] < self.threshold
        )[0]

        return SinkhornOutput(
            f=f,
            g=g,
            errors=state.errors[:, 0],
            threshold=jnp.array(self.threshold),
            converged=converged,
        )

    @property
    def norm_error(self) -> Tuple[int, ...]:
        """Powers used to compute the p-norm between marginal/target."""
        # To change momentum adaptively, one needs errors in ||.||_1 norm.
        # In that case, we add this exponent to the list of errors to compute,
        # notably if that was not the error requested by the user.
        if self.momentum and self.momentum.start > 0 and self._norm_error != 1:
            return self._norm_error, 1
        return (self._norm_error,)

    # TODO(michalk8): in the future, enforce this (+ in GW) via abstract method
    def create_initializer(self) -> init_lib.SinkhornInitializer:  # noqa: D102
        if isinstance(self.initializer, init_lib.SinkhornInitializer):
            return self.initializer
        if self.initializer == "default":
            return init_lib.DefaultInitializer()
        if self.initializer == "gaussian":
            return init_lib.GaussianInitializer()
        if self.initializer == "sorting":
            return init_lib.SortingInitializer(**self.kwargs_init)
        if self.initializer == "subsample":
            return init_lib.SubsampleInitializer(**self.kwargs_init)
        raise NotImplementedError(f"Initializer `{self.initializer}` is not yet implemented.")

    def tree_flatten(self):  # noqa: D102
        aux = vars(self).copy()
        aux["norm_error"] = aux.pop("_norm_error")
        aux.pop("threshold")
        return [self.threshold], aux

    @classmethod
    def tree_unflatten(cls, aux_data, children):  # noqa: D102
        return cls(**aux_data, threshold=children[0])


def run(
    ot_prob: linear_problem.LinearProblem, solver: IPFP, init: Tuple[jnp.ndarray, ...]
) -> SinkhornOutput:
    """Run loop of the solver, outputting a state upgraded to an output."""
    iter_fun = _iterations_implicit if solver.implicit_diff else iterations
    out = iter_fun(ot_prob, solver, init)
    # Be careful here, the geom and the cost are injected at the end, where it
    # does not interfere with the implicit differentiation.
    out = out.set_cost(ot_prob, solver.lse_mode, solver.use_danskin)
    return out.set(ot_prob=ot_prob)


def iterations(
    ot_prob: linear_problem.LinearProblem, solver: IPFP, init: Tuple[jnp.ndarray, ...]
) -> SinkhornOutput:
    """Jittable IPFP loop. args contain initialization variables."""

    def cond_fn(
        iteration: int, const: Tuple[linear_problem.LinearProblem, IPFP], state: SinkhornState
    ) -> bool:
        _, solver = const
        return solver._continue(state, iteration)

    def body_fn(
        iteration: int,
        const: Tuple[linear_problem.LinearProblem, IPFP],
        state: SinkhornState,
        compute_error: bool,
    ) -> SinkhornState:
        ot_prob, solver = const
        return solver.one_iteration(ot_prob, state, iteration, compute_error)

    # Run the IPFP loop. Choose either a standard fixpoint_iter loop if
    # differentiation is implicit, otherwise switch to the backprop friendly
    # version of that loop if unrolling to differentiate.
    if solver.implicit_diff:
        fix_point = fixed_point_loop.fixpoint_iter
    else:
        fix_point = fixed_point_loop.fixpoint_iter_backprop

    const = ot_prob, solver
    state = solver.init_state(ot_prob, init)
    state = fix_point(
        cond_fn,
        body_fn,
        solver.min_iterations,
        solver.max_iterations,
        solver.inner_iterations,
        const,
        state,
    )
    return solver.output_from_state(ot_prob, state)


def _iterations_taped(
    ot_prob: linear_problem.LinearProblem, solver: IPFP, init: Tuple[jnp.ndarray, ...]
) -> Tuple[SinkhornOutput, Tuple[jnp.ndarray, jnp.ndarray, linear_problem.LinearProblem, IPFP]]:
    """Run forward pass of the IPFP algorithm storing side information."""
    state = iterations(ot_prob, solver, init)
    return state, (state.f, state.g, ot_prob, solver)


def _iterations_implicit_bwd(res, gr):
    """Run IPFP in backward mode, using implicit differentiation.

    Args:
      res: residual data sent from fwd pass, used for computations below. In this
        case consists in the output itself, as well as inputs against which we
        wish to differentiate.
      gr: gradients w.r.t outputs of fwd pass, here w.r.t size f, g, errors. Note
        that differentiability w.r.t. errors is not handled, and only f, g is
        considered.

    Returns:
      a tuple of gradients: PyTree for geom, one jnp.ndarray for each of a and b.
    """
    f, g, ot_prob, solver = res
    gr = gr[:2]
    return (*solver.implicit_diff.gradient(ot_prob, f, g, solver.lse_mode, gr), None, None)


# sets threshold, norm_errors, geom, a and b to be differentiable, as those are
# non-static. Only differentiability w.r.t. geom, a and b will be used.
_iterations_implicit = jax.custom_vjp(iterations)
_iterations_implicit.defvjp(_iterations_taped, _iterations_implicit_bwd)


@utils.deprecate(alt="Please use `ott.solvers.linear.solve()` instead.")
def solve(*args: Any, **kwargs: Any) -> Union[SinkhornOutput, "LRSinkhornOutput"]:
    """Solve linear regularized OT problem using IPFP iterations.

    Args:
      args: Position arguments for :func:`ott.solvers.linear.solve`.
      kwargs: Keyword arguments for :func:`ott.solvers.linear.solve`.

    Returns:
      The IPFP output.
    """
    from ott.solvers.linear import solve

    return solve(*args, **kwargs)
