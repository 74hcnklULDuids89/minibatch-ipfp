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
# See https://github.com/ott-jax/ott/blob/main/src/ott/geometry/pointcloud.py for the original source code.

from typing import Optional

import jax
import jax.numpy as jnp

from ott.geometry import geometry
from ott.geometry.pointcloud import PointCloud, _apply_kernel_xy

from ott.geometry.costs import CostFn


@jax.tree_util.register_pytree_node_class
class DotProd(CostFn):
    """dot product distance cost function."""

    def __init__(self):
        super().__init__()

    def pairwise(self, x: jnp.ndarray, y: jnp.ndarray) -> float:
        """dot product between factor vectors x and y"""
        return -jnp.vdot(x, y) / 2.0


@jax.tree_util.register_pytree_node_class
class PointCloudIPFP(PointCloud):
    """Defines geometry for Mini-Batch IPFP.
    Note that the IPFP iterations is implemented in only kernel update mode (lse_mode=False).
    The only difference is that the apply_kernel_ipfp and update_scaling methods are overwritten.
    """

    def apply_kernel_ipfp(  # noqa: D102
        self, scaling: jnp.ndarray, eps: Optional[float] = None, axis: int = 0
    ) -> jnp.ndarray:
        if eps is None:
            eps = self.epsilon

        def body0(carry, i: int):
            eps, scaling = carry
            y = jax.lax.dynamic_slice(
                self.y, (i * self.batch_size, 0), (self.batch_size, self.y.shape[1])
            )
            # scaling_sub = jax.lax.dynamic_slice(
            #     scaling, (i * self.batch_size,), (self.batch_size,)
            # )
            # g_ = jax.lax.dynamic_slice(g, (i * self.batch_size,), (self.batch_size,))
            if self._axis_norm is None:
                norm_y = self._norm_y
            else:
                norm_y = jax.lax.dynamic_slice(
                    self._norm_y, (i * self.batch_size,), (self.batch_size,)
                )
            h = app(
                self.x,
                y,
                self._norm_x,
                norm_y,
                scaling,
                eps,
                self.cost_fn,
                self.inv_scale_cost,
            )
            return carry, h

        def body1(carry, i: int):
            eps, scaling = carry
            x = jax.lax.dynamic_slice(
                self.x, (i * self.batch_size, 0), (self.batch_size, self.x.shape[1])
            )
            # scaling_sub = jax.lax.dynamic_slice(
            #     scaling, (i * self.batch_size,), (self.batch_size,)
            # )
            # f_ = jax.lax.dynamic_slice(f, (i * self.batch_size,), (self.batch_size,))
            if self._axis_norm is None:
                norm_x = self._norm_x
            else:
                norm_x = jax.lax.dynamic_slice(
                    self._norm_x, (i * self.batch_size,), (self.batch_size,)
                )
            h = app(
                self.y,
                x,
                self._norm_y,
                norm_x,
                scaling,
                eps,
                self.cost_fn,
                self.inv_scale_cost,
            )
            return carry, h

        def finalize(i: int):
            if axis == 0:
                norm_y = self._norm_y if self._axis_norm is None else self._norm_y[i:]
                return app(
                    self.x,
                    self.y[i:],
                    self._norm_x,
                    norm_y,
                    scaling,
                    eps,
                    self.cost_fn,
                    self.inv_scale_cost,
                )
            norm_x = self._norm_x if self._axis_norm is None else self._norm_x[i:]
            return app(
                self.y,
                self.x[i:],
                self._norm_y,
                norm_x,
                scaling,
                eps,
                self.cost_fn,
                self.inv_scale_cost,
            )

        if not self.is_online:
            return super().apply_kernel(scaling, eps, axis)

        app = jax.vmap(
            _apply_kernel_xy,
            in_axes=[None, 0, None, self._axis_norm, None, None, None, None],
        )

        if axis == 0:
            fun = body0
            n = self._y_nsplit
        elif axis == 1:
            fun = body1
            n = self._x_nsplit
        else:
            raise ValueError(axis)

        _, h = jax.lax.scan(fun, init=(eps, scaling), xs=jnp.arange(n))
        h = jnp.concatenate(h)
        h_rest = finalize(n * self.batch_size)
        # h_res = jnp.concatenate([h, h_rest])
        # h_sign = jnp.concatenate([h_sign, h_sign_rest])

        return jnp.concatenate([h, h_rest])

    def marginal_from_scalings(
        self,
        u: jnp.ndarray,
        v: jnp.ndarray,
        axis: int = 0,
    ) -> jnp.ndarray:
        """Output marginal of transportation matrix from scalings."""
        u, v = (v, u) if axis == 0 else (u, v)
        return u * self.apply_kernel_ipfp(v, eps=self.epsilon, axis=axis)

    def update_scaling(
        self,
        scaling: jnp.ndarray,
        marginal: jnp.ndarray,
        iteration: Optional[int] = None,
        axis: int = 0,
    ) -> jnp.ndarray:
        """Carry out one Sinkhorn update for scalings, using kernel directly.

        Args:
        scaling: jnp.ndarray of num_a or num_b positive values.
        marginal: targeted marginal
        iteration: used to compute epsilon from schedule, if provided.
        axis: axis along which the update should be carried out.

        Returns:
        new scaling vector, of size num_b if axis=0, num_a if axis is 1.
        """

        eps = self._epsilon.at(iteration)
        app_kernel = self.apply_kernel_ipfp(scaling, eps, axis=axis) / 2.0  # S
        return jnp.sqrt(app_kernel**2 + marginal) - app_kernel


@jax.tree_util.register_pytree_node_class
class GeometryIPFP(geometry.Geometry):
    """Defines geometry for IPFP."""

    def update_scaling(
        self,
        scaling: jnp.ndarray,
        marginal: jnp.ndarray,
        iteration: Optional[int] = None,
        axis: int = 0,
    ) -> jnp.ndarray:
        """Carry out one Sinkhorn update for scalings, using kernel directly.

        Args:
        scaling: jnp.ndarray of num_a or num_b positive values.
        marginal: targeted marginal
        iteration: used to compute epsilon from schedule, if provided.
        axis: axis along which the update should be carried out.

        Returns:
        new scaling vector, of size num_b if axis=0, num_a if axis is 1.
        """
        # jax.debug.print("vector = {x}", x=scaling)

        eps = self._epsilon.at(iteration)
        app_kernel = self.apply_kernel(scaling, eps, axis=axis) / 2.0  # S
        return jnp.sqrt(app_kernel**2 + marginal) - app_kernel
