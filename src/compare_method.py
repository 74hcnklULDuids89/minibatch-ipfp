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
from sympy import factor

from evaluate_method import evaluate_rankings, run_method, save_data
from generate_preference import generate_preference_matrices, load_real_data
from sw_method import run_sw_method
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
    factor_x_u: np.ndarray | None = None,
    factor_x_v: np.ndarray | None = None,
    factor_y_u: np.ndarray | None = None,
    factor_y_v: np.ndarray | None = None,
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
                pref_x,
                pref_x,
                pref_y,
                v_cand,
                v_job,
                method=method,
                preference_boost=preference_boost,
            )
            elapsed_time = 0
        elif method == "reciprocal":
            reciprocal_scores = pref_x * pref_y.T
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
            clip_pref_x = np.clip(pref_x, 0.0, 1.0)
            clip_pref_y = np.clip(pref_y, 0.0, 1.0)
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
            result = run_sw_method(pref_x, pref_y, exam_type, exam_type)
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
            result = run_method(pref_x, pref_y, method, factor_x_u, factor_x_v, factor_y_u, factor_y_v)
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
                        num_candidates = int(size * 1.5)
                        pref_x, pref_y = generate_preference_matrices(num_jobs, num_candidates, crowding, seed=seed)
                        results += compare_methods(size, crowding, exam_type, methods, pref_x, pref_y)

        df = pd.DataFrame(results)
        print(df)
        df.to_csv("/workspace/logs/synthetic_data_results.csv", index=False)
    else:
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

    if dirname == "taichi":
        preference_boost = 1000.0
    else:
        preference_boost = 100.0

    if not visualize:
        results = []
        for seed in range(n_runs):
            np.random.seed(seed)
            for size in sizes:
                # pref_x, pref_y, factor_x_u, factor_x_v, factor_y_u, factor_y_v = load_real_data(
                #     male_data_path, female_data_path, size
                # )
                user_A = scipy.sparse.load_npz(f"{male_data_path}_rel.npz")
                job_A = scipy.sparse.load_npz(f"{female_data_path}_rel.npz")

                # decompose the preference matrices by ALS
                als_x = implicit.als.AlternatingLeastSquares(factors=50, random_state=seed, iterations=100)
                als_x.fit(user_A)
                factor_x_u = als_x.user_factors
                factor_x_v = als_x.item_factors

                als_y = implicit.als.AlternatingLeastSquares(factors=50, random_state=seed, iterations=100)
                als_y.fit(job_A)
                factor_y_u = als_y.user_factors
                factor_y_v = als_y.item_factors

                pref_x = np.dot(factor_x_u, factor_x_v.T)
                pref_y = np.dot(factor_y_u, factor_y_v.T)

                # cut of the preference matrices
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
            methods=["random", "naive", "reciprocal", "CR", "batch-IPFP", "minibatch-IPFP"],
            crowding_levels=[0.0, 0.25, 0.5, 0.75, 1.0],
            examination_types=["exp"],
            n_runs=10,
            exam_threshold=5,
            visualize=args.visualize,
        )
