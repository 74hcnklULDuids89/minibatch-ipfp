import os
import pickle
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from ipfp_method import ipfp
from subprocess_exp import solve_ott


def save_data(method: str, size: int, **data: dict[str, Any]) -> None:
    logs_dir = f"/workspace/logs/{method}/{size}"
    os.makedirs(logs_dir, exist_ok=True)

    for name, value in data.items():
        file_path = os.path.join(logs_dir, f"{name}.pkl")
        with open(file_path, "wb") as file:
            pickle.dump(value, file)


def evaluate_rankings(
    match_probs: np.ndarray,
    pref_x: np.ndarray,
    pref_y: np.ndarray,
    v_cand: Callable,
    v_job: Callable,
    method: str,
    runs: int = 1000,
    Pc_sim: list[np.ndarray] | None = None,
    preference_boost: float = 1.0,
) -> tuple[float, list[int]]:
    """evaluate methods by running monte carlo simulation of following steps:
    1. Candidates apply to jobs based on their preferences and examination function.
    2. Given an application from candidates, jobs apply interviews based on their preferences and examination function.
    3. Calculate the number of matches (=interviews).
    For SW method, we use the precomputed Pc_sim, which is the ranking policy for each candidate.

    Args:
        match_probs (np.ndarray): matching probabilities matrix of shape (num_candidates, num_employers)
        pref_x (np.ndarray): preference probabilities matrix of shape (num_candidates, num_employers)
        pref_y (np.ndarray): preference probabilities of shape (num_employers, num_candidates)
        v_cand (Callable): examination function for candidates
        v_job (Callable): examination function for jobs
        runs (int, optional): monte calro simulation run times. Defaults to 1000.
        Pc_sim (list[np.ndarray], optional): ranking policy for each candidate. Defaults to None.
        preference_boost (float, optional): factor to boost probability of match action. Defaults to 1.0.

    Returns:
        tuple[float, float]: elapsed time and average matches
    """
    # clip preference matrices
    pref_x = np.clip(pref_x * preference_boost, 0, 1)
    pref_y = np.clip(pref_y * preference_boost, 0, 1)

    start_time = time.time()
    num_candidates, num_employers = pref_x.shape
    mc_matches = []

    for _ in tqdm(range(runs), total=runs):
        if method == "SW":
            v_cand_array = v_cand(np.arange(num_employers) + 1)
            exam_prob = np.array([np.dot(ranking_policy, v_cand_array) for ranking_policy in Pc_sim])
            apply_probs = np.clip(np.multiply(pref_x, exam_prob), 0, 1)
            v_job_array = v_job(np.arange(num_candidates) + 1)
        else:
            rankings_for_candidate = np.argsort(-match_probs, axis=1)
            apply_probs = pref_x * v_cand(rankings_for_candidate + 1)

        apply_actions = np.random.binomial(np.ones_like(apply_probs).astype(int), apply_probs)
        if method == "SW":
            appl_rel_table = np.multiply(apply_actions.T, pref_y)  # for SW

        total_matches = []
        for job_id in range(num_employers):
            if method == "SW":
                bool_filter = appl_rel_table[job_id] > 0
                # Correcting the view to match applications recieved and their
                # relevance ordering
                view_ord = np.argsort(-appl_rel_table[job_id])
                view_corr = np.argsort(view_ord)
                # v_job for job j based on the applications and their relevance
                v_job_temp = np.where(bool_filter, v_job_array[view_corr], 0)
                interview_probs = np.multiply(pref_y[job_id], v_job_temp)
                # Sometimes precision can cause errors when sampling so clip values
                interview_probs = np.clip(interview_probs, 0, 1)
            else:
                # get applied candidates to job_id
                applied_candidates = np.where(apply_actions[:, job_id] == 1)[0]
                if len(applied_candidates) == 0:
                    continue
                job_prefs = pref_y[job_id, :]
                # get rankings of applied candidates
                rankings_for_job = np.argsort(-match_probs[applied_candidates, job_id])
                interview_probs = job_prefs[applied_candidates] * v_job(rankings_for_job + 1)

            interview_actions = np.random.binomial(np.ones_like(interview_probs).astype(int), interview_probs)
            total_matches.append(np.sum(interview_actions))
        mc_matches.append(np.sum(total_matches))

    elapsed_time = time.time() - start_time
    return elapsed_time, np.mean(mc_matches)


def run_method(
    pref_x: np.ndarray,
    pref_y: np.ndarray,
    method: str,
    factor_x_u: np.ndarray | None = None,
    factor_x_v: np.ndarray | None = None,
    factor_y_u: np.ndarray | None = None,
    factor_y_v: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Execute the given method on preferences.

    Args:
    - pref_x (numpy.ndarray): Preference array for x.
    - pref_y (numpy.ndarray): Preference array for y.
    - method (str): Method to be used ('torch', 'numpy').

    Returns:
    - tuple: Elapsed time, mean loop time, and length of loop time list.
    """
    start = time.time()
    loop_time_list = []
    print(f"running {method}")
    if method == "IPFP-Debug":
        # call pytorch implementation for debug
        mu_xy, res_list, loop_time_list, u, v, residual = ipfp(pref_x, pref_y, is_torch=True)
        elapsed_time = time.time() - start
        mean_loop_time = np.mean(loop_time_list)
        converged_step = len(loop_time_list)
        print(mu_xy)
    elif method == "batch-IPFP":
        # call modified jax-ott library
        a = jnp.ones((pref_x.shape[0],), dtype=jnp.float32) * pref_y.shape[0]
        b = jnp.ones((pref_y.shape[0],), dtype=jnp.float32) * pref_x.shape[0]
        a = jax.device_put(a, jax.devices("gpu")[0])
        b = jax.device_put(b, jax.devices("gpu")[0])
        pref_x = jax.device_put(jnp.array(pref_x), jax.devices("gpu")[0])
        pref_y = jax.device_put(jnp.array(pref_y), jax.devices("gpu")[0])

        result = solve_ott(a, b, pref_x, pref_y, threshold=1e-6, epsilon=1.0, factorize=False, batch_size=None)
        elapsed_time = time.time() - start
        mu_xy = result.matrix
        res_list = result.errors
        # potential -> scaling transfrom because we use jax-ott kernel mode
        u = np.exp(result.f)
        v = np.exp(result.g)
        residual = result.reg_ot_cost
        converged_step = int(result.n_iters)
        mean_loop_time = elapsed_time / converged_step
        print(result.matrix)
    elif method == "minibatch-IPFP":
        a = jnp.ones((pref_x.shape[0],), dtype=jnp.float32) * pref_y.shape[0]
        b = jnp.ones((pref_y.shape[0],), dtype=jnp.float32) * pref_x.shape[0]
        a = jax.device_put(a, jax.devices("gpu")[0])
        b = jax.device_put(b, jax.devices("gpu")[0])

        if factor_x_u is None:
            # run matrix_factorization to preference matrices
            # note that this operation causes differences from original preference matrices
            from sklearn.decomposition import NMF

            model_x = NMF(n_components=pref_x.shape[0] // 2, init="random", random_state=0)
            W_x = model_x.fit_transform(pref_x)
            H_x = model_x.components_

            model_y = NMF(n_components=pref_y.shape[0] // 2, init="random", random_state=0)
            W_y = model_y.fit_transform(pref_y)
            H_y = model_y.components_

            factor_x = jnp.concatenate((jnp.array(W_x), jnp.array(H_y.T)), axis=1)
            factor_y = jnp.concatenate((jnp.array(H_x.T), jnp.array(W_y)), axis=1)
        else:
            # use given factor vectors
            factor_x = jnp.concatenate((jnp.array(factor_x_u), jnp.array(factor_y_v)), axis=1)
            factor_y = jnp.concatenate((jnp.array(factor_x_v), jnp.array(factor_y_u)), axis=1)

        # Run factor vector based IPFP
        result = solve_ott(a, b, factor_x, factor_y, threshold=1e-6, epsilon=1.0, factorize=True, batch_size=None)
        elapsed_time = time.time() - start
        mu_xy = result.matrix
        res_list = result.errors
        # potential -> scaling transfrom because we use jax-ott kernel mode
        u = np.exp(result.f)
        v = np.exp(result.g)
        residual = result.reg_ot_cost
        converged_step = int(result.n_iters)
        mean_loop_time = elapsed_time / converged_step

    save_data(
        method=method,
        size=pref_x.shape[0],
        loop_time_list=loop_time_list,
        converged_step=len(loop_time_list),
    )

    results_data = {
        "elapsed_time": elapsed_time,
        "mean_loop_time": mean_loop_time,
        "mu_xy": np.array(mu_xy),
        "res_list": res_list,
        "mu_x0": np.array(u**2),
        "mu_y0": np.array(v**2),
        "residual": residual,
        "pref_x": np.array(pref_x),
        "pref_y": np.array(pref_y),
        "converged_step": converged_step,
    }

    return results_data
