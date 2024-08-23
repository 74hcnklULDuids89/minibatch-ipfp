import argparse
import os
import pickle
from typing import Callable

import implicit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from sklearn.decomposition import NMF

from evaluate_method import evaluate_rankings, run_method, save_data
from generate_preference import generate_preference_matrices, load_real_data, load_real_data_Amask
from libimseti_preprocessing import pmf_solve
# from sw_method import run_sw_method
from visualization import compare_expected_matches, compare_expected_matches_realdata, setup_visualization

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.set_palette("husl")


def compare_methods(
    size: int,
    crowding: float | None,
    exam_type: str,
    methods: list[str],
    pref_x: np.ndarray,
    pref_y: np.ndarray,
    factor_x_u: np.ndarray,
    factor_x_v: np.ndarray,
    factor_y_u: np.ndarray,
    factor_y_v: np.ndarray,
    exam_threshold: int = 3,
    preference_boost: float = 1.0,
) -> list[dict]:
    """Compare different methods on the same preference matrices.

    Args:
        size (int): size of the preference matrices (job)
        crowding (float | None): crowding level of the preference matrices. It is used just for logging purposes.
        exam_type (str): type of examination function to use
        methods (list[str]): list of methods to compare
        pref_x (np.ndarray): preference matrix of the candidates
        pref_y (np.ndarray): preference matrix of the jobs
        factor_x_u (np.ndarray | None, optional): factor vectors for the candidates. Defaults to None.
        factor_x_v (np.ndarray | None, optional): factor vectors for the candidates. Defaults to None.
        factor_y_u (np.ndarray | None, optional): factor vectors for the jobs. Defaults to None.
        factor_y_v (np.ndarray | None, optional): factor vectors for the jobs. Defaults to None.
        exam_threshold (int, optional): threshold for examination function `threshold`. Defaults to 3.

    Returns:
        list[dict]: list of results for each method
    """
    v_cand, v_job = get_examination_functions(exam_type, exam_threshold)

    results = []

    for method in methods:
        print(f"Running size {size} with method {method}, examination {exam_type}")
        if method == "random":
            _, total_matches = evaluate_rankings(
                np.random.rand(pref_x.shape[0], pref_x.shape[1]),
                pref_x,
                pref_y,
                v_cand,
                v_job,
                method=method,
                preference_boost=preference_boost,
            )
            elapsed_time = 0
        elif method == "naive":
            _, total_matches = evaluate_rankings(
                factor_x_u @ factor_x_v.T,  # use x -> y preference matrix as matching policy
                pref_x,
                pref_y,
                v_cand,
                v_job,
                method=method,
                preference_boost=preference_boost,
            )
            elapsed_time = 0
        elif method == "reciprocal":
            reconstrcted_pref_x = factor_x_u @ factor_x_v.T
            reconstrcted_pref_y = factor_y_u @ factor_y_v.T
            reciprocal_scores = reconstrcted_pref_x * reconstrcted_pref_y.T
            _, total_matches = evaluate_rankings(
                reciprocal_scores,
                pref_x,
                pref_y,
                v_cand,
                v_job,
                method=method,
                preference_boost=preference_boost,
            )
            elapsed_time = 0
        elif method == "CR":
            reconstrcted_pref_x = factor_x_u @ factor_x_v.T
            reconstrcted_pref_y = factor_y_u @ factor_y_v.T
            clip_pref_x = np.clip(reconstrcted_pref_x, 0.0, 1.0)
            clip_pref_y = np.clip(reconstrcted_pref_y, 0.0, 1.0)
            reciprocal_scores = clip_pref_x * clip_pref_y.T
            inv_reciprocal_scores = (1 - clip_pref_x) * (1 - clip_pref_y.T)
            cr_norm = reciprocal_scores / (reciprocal_scores + inv_reciprocal_scores)
            _, total_matches = evaluate_rankings(
                cr_norm,
                pref_x,
                pref_y,
                v_cand,
                v_job,
                method=method,
                preference_boost=preference_boost,
            )
            elapsed_time = 0
        elif method == "SW":
            if size > 50:
                print("Skipping SW method for large matrices due to performance issues")
                continue  # Skip SW method for large matrices due to performance issues
            reconstrcted_pref_x = factor_x_u @ factor_x_v.T
            reconstrcted_pref_y = factor_y_u @ factor_y_v.T
            result = run_sw_method(reconstrcted_pref_x, reconstrcted_pref_y, exam_type, exam_type)
            _, total_matches = evaluate_rankings(
                None,
                pref_x,
                pref_y,
                v_cand,
                v_job,
                method=method,
                Pc_sim=result["Pc"],
                preference_boost=preference_boost,
            )
            elapsed_time = result["elapsed_time"]
            save_data(method, size, **result)
        else:
            # Run the IPFP methods
            reconstrcted_pref_x = factor_x_u @ factor_x_v.T
            reconstrcted_pref_y = factor_y_u @ factor_y_v.T
            result = run_method(
                reconstrcted_pref_x, reconstrcted_pref_y, method, factor_x_u, factor_x_v, factor_y_u, factor_y_v
            )
            _, total_matches = evaluate_rankings(
                result["mu_xy"],
                pref_x,
                pref_y,
                v_cand,
                v_job,
                method=method,
                preference_boost=preference_boost,
            )
            elapsed_time = result["elapsed_time"]
            save_data(method, size, **result)
        results.append(
            {
                "Method": method,
                "n": size,
                "Crowding": crowding,
                "Examination": exam_type,
                "Time per step (s)": elapsed_time,
                "Expected number of total matches": total_matches,
            }
        )
    return results


def synthetic_data_experiment(
    sizes: list[int],
    methods: list[str] = ["naive", "reciprocal", "TU", "Ours"],
    crowding_levels: list[float] = [0.5],
    examination_types: list[str] = ["inv"],
    n_runs: int = 1,
    exam_threshold: int = 3,
    visualize: bool = False,
    mf_method: str = "ALS",
    mask_ratio: float = 0.0,
) -> pd.DataFrame:
    """
    Run the main process of the algorithm for different matrix sizes, methods, crowding levels, and examination types.

    Args:
        sizes (list): List of matrix sizes to run the algorithm on.
        methods (list): List of methods to use (default: ["naive", "reciprocal", "SW", "TU", "Ours"]).
        crowding_levels (list): List of crowding levels to consider (default: [0.5]).
        examination_types (list): List of examination types to consider (default: ["inv"]).

    Returns:
        pd.DataFrame: A dataframe containing the results of the algorithm runs.
    """
    if not visualize:
        results = []
        for seed in range(n_runs):
            for crowding in crowding_levels:
                for exam_type in examination_types:
                    for size in sizes:
                        num_jobs = int(size)
                        num_candidates = int(size * 2.0)
                        pref_x, pref_y = generate_preference_matrices(num_jobs, num_candidates, crowding, seed=seed)

                        if mf_method == "ALS":
                            print("Sampling preference matrix and reconstruct by iALS")
                            # Sampling
                            pref_obs_x = np.random.binomial(1, pref_x)
                            mask = np.random.rand(*pref_x.shape) < mask_ratio
                            pref_obs_x[mask] = 0.0
                            als_x = implicit.als.AlternatingLeastSquares(factors=50, random_state=seed, iterations=100)
                            als_x.fit(scipy.sparse.csr_matrix(pref_obs_x))
                            factor_x_u = als_x.user_factors
                            factor_x_v = als_x.item_factors

                            pref_obs_y = np.random.binomial(1, pref_y)
                            mask = np.random.rand(*pref_y.shape) < mask_ratio
                            pref_obs_y[mask] = 0.0
                            als_y = implicit.als.AlternatingLeastSquares(factors=50, random_state=seed, iterations=100)
                            als_y.fit(scipy.sparse.csr_matrix(pref_obs_y))
                            factor_y_u = als_y.user_factors
                            factor_y_v = als_y.item_factors

                        elif mf_method == "NMF":
                            print("Masking preference matrix and reconstruct by NMF")
                            # Randomly masking the preference matrices (for observation)
                            mask = np.random.rand(*pref_x.shape) < 0.5
                            pref_x_masked = pref_x.copy()
                            pref_x_masked[mask] = 0.0
                            mask = np.random.rand(*pref_y.shape) < 0.5
                            pref_y_masked = pref_y.copy()
                            pref_y_masked[mask] = 0.0

                            nmf_x = NMF(n_components=50, random_state=seed, init="random")
                            factor_x_u = nmf_x.fit_transform(pref_x_masked)
                            factor_x_v = nmf_x.components_.T
                            nmf_y = NMF(n_components=50, random_state=seed, init="random")
                            factor_y_u = nmf_y.fit_transform(pref_y_masked)
                            factor_y_v = nmf_y.components_.T
                        else:
                            raise ValueError(f"Invalid matrix factorization method: {mf_method}")

                        results += compare_methods(
                            size,
                            crowding,
                            exam_type,
                            methods,
                            pref_x,
                            pref_y,
                            factor_x_u,
                            factor_x_v,
                            factor_y_u,
                            factor_y_v,
                        )

        df = pd.DataFrame(results)
        print(df)
        df.to_csv(f"/workspace/logs/synthetic_data_results.csv", index=False)
    else:
        # visualize
        df = pd.read_csv("/workspace/logs/synthetic_data_results.csv")
        setup_visualization(df, crowding_levels, examination_types)
        compare_expected_matches(df, examination_types[0], size=sizes[0])

    return df


def real_data_experiment(
    sizes: list[int],
    male_data_path: str,
    female_data_path: str,
    methods: list[str] = ["naive", "reciprocal", "CR", "TU", "Ours"],
    n_runs: int = 1,
    visualize: bool = False,
) -> pd.DataFrame:

    dirname = os.path.dirname(male_data_path).split("/")[-1]

    preference_boost = 1.0

    if not visualize:
        results = []
        for seed in range(n_runs):
            np.random.seed(seed)
            for size in sizes:
                m_A, m_mask, f_A, f_mask = load_real_data_Amask(male_data_path, female_data_path, size)

                # exciplit nonnegative matrix factorization with ALS
                pref_x, factor_x_u, factor_x_v = pmf_solve(m_A, m_mask, 6, 1e-2, seed=seed)
                pref_y, factor_y_u, factor_y_v = pmf_solve(f_A, f_mask, 6, 1e-2, seed=seed)

                # cut off the preference matrices
                pref_x = pref_x[:size, :size]
                pref_y = pref_y[:size, :size]
                factor_x_u = factor_x_u[:size]
                factor_x_v = factor_x_v[:size]
                factor_y_u = factor_y_u[:size]
                factor_y_v = factor_y_v[:size]

                results += compare_methods(
                    size,
                    None,
                    "exp",
                    methods,
                    pref_x,
                    pref_y,
                    factor_x_u,
                    factor_x_v,
                    factor_y_u,
                    factor_y_v,
                    preference_boost=preference_boost,
                )
        df = pd.DataFrame(results)
        # get male_data_path foldername
        dirname = os.path.dirname(male_data_path).split("/")[-1]
        if not os.path.exists(f"/workspace/logs/{dirname}"):
            os.makedirs(f"/workspace/logs/{dirname}")
        df.to_csv(f"/workspace/logs/{dirname}/real_data_results.csv", index=False)
    else:
        dirname = os.path.dirname(male_data_path).split("/")[-1]
        df = pd.read_csv(f"/workspace/logs/{dirname}/real_data_results.csv")
        compare_expected_matches_realdata(df, exam_type="exp", dirname=dirname)
    return df


def get_examination_functions(exam_type: str, thres: int = 3) -> tuple[Callable, Callable]:
    """
    Get the examination functions based on the examination type.

    Args:
        exam_type (str): The examination type ("inv", "exp", or "log").

    Returns:
        tuple: A tuple containing the candidate and job examination functions.
    """
    if exam_type == "inv":
        v_cand = lambda x: 1 / x
        v_job = lambda x: 1 / x
    elif exam_type == "exp":
        v_cand = lambda x: 1 / np.exp(x - 1)
        v_job = lambda x: 1 / np.exp(x - 1)
    elif exam_type == "log":
        v_cand = lambda x: 1 / np.log(x + 2)
        v_job = lambda x: 1 / np.log(x + 2)
    elif exam_type == "threshold":
        v_cand = lambda x: x <= thres
        v_job = lambda x: x <= thres

    return v_cand, v_job


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IPFP algorithm on different matrix sizes.")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[500],
        # default=[500],
        help="List of matrix sizes to run the algorithm on. Example: --sizes 10 100 1000 10000",
    )
    parser.add_argument("--use_real_data", action="store_true", help="Use real data for the experiment.")
    parser.add_argument(
        "--male_data_path",
        type=str,
        default="./data/libimseti/male_to_female",
        help="Path to the male preference data file.",
    )
    parser.add_argument(
        "--female_data_path",
        type=str,
        default="./data/libimseti/female_to_male",
        help="Path to the female preference data file.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the results of the experiment.",
        default=False,
    )
    args = parser.parse_args()

    os.makedirs("/workspace/logs", exist_ok=True)

    if args.use_real_data:
        real_data_experiment(
            sizes=args.sizes,
            male_data_path=args.male_data_path,
            female_data_path=args.female_data_path,
            # methods=["minibatch-IPFP"],
            methods=["naive", "reciprocal", "CR", "batch-IPFP", "minibatch-IPFP"],
            visualize=args.visualize,
            n_runs=10,
        )
    else:
        synthetic_data_experiment(
            sizes=args.sizes,
            # methods=["minibatch-IPFP"],
            methods=["naive", "reciprocal", "CR", "batch-IPFP", "minibatch-IPFP"],
            # crowding_levels=[0.75],
            crowding_levels=[0.0, 0.25, 0.5, 0.75],
            examination_types=["exp"],
            n_runs=10,
            exam_threshold=5,
            visualize=args.visualize,
            mf_method="ALS",
        )
