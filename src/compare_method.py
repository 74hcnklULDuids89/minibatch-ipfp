import argparse
import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
                np.random.rand(pref_x.shape[0], pref_x.shape[1]), pref_x, pref_y, v_cand, v_job, method=method
            )
            elapsed_time = 0
        elif method == "naive":
            _, total_matches = evaluate_rankings(pref_x, pref_x, pref_y, v_cand, v_job, method=method)
            elapsed_time = 0
        elif method == "reciprocal":
            reciprocal_scores = pref_x * pref_y.T
            _, total_matches = evaluate_rankings(reciprocal_scores, pref_x, pref_y, v_cand, v_job, method=method)
            elapsed_time = 0
        elif method == "SW":
            if size > 50:
                print("Skipping SW method for large matrices due to performance issues")
                continue  # Skip SW method for large matrices due to performance issues
            result = run_sw_method(pref_x, pref_y, exam_type, exam_type)
            _, total_matches = evaluate_rankings(
                None, pref_x, pref_y, v_cand, v_job, method=method, Pc_sim=result["Pc"]
            )
            elapsed_time = result["elapsed_time"]
            save_data(method, size, **result)
        else:
            # Run the IPFP methods
            result = run_method(pref_x, pref_y, method, factor_x_u, factor_x_v, factor_y_u, factor_y_v)
            _, total_matches = evaluate_rankings(result["mu_xy"], pref_x, pref_y, v_cand, v_job, method=method)
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
    methods: list[str] = ["naive", "reciprocal", "TU", "Ours"],
    n_runs: int = 1,
    visualize: bool = False,
) -> pd.DataFrame:
    if not visualize:
        results = []
        for seed in range(n_runs):
            np.random.seed(seed)
            for size in sizes:
                pref_x, pref_y, factor_x_u, factor_x_v, factor_y_u, factor_y_v = load_real_data(
                    male_data_path, female_data_path, size
                )
                v_cand, v_job = get_examination_functions("exp")
                results += compare_methods(
                    size, None, "exp", methods, pref_x, pref_y, factor_x_u, factor_x_v, factor_y_u, factor_y_v
                )
        df = pd.DataFrame(results)
        df.to_csv("/workspace/logs/real_data_results.csv", index=False)
    else:
        df = pd.read_csv("/workspace/logs/real_data_results.csv")
        compare_expected_matches_realdata(df, exam_type="exp")
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
        # default=[10],
        default=[500],
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
            methods=["naive", "reciprocal", "batch-IPFP", "minibatch-IPFP"],
            visualize=args.visualize,
        )
    else:
        synthetic_data_experiment(
            sizes=args.sizes,
            # methods=["minibatch-IPFP"],
            methods=["random", "naive", "reciprocal", "batch-IPFP", "minibatch-IPFP"],
            crowding_levels=[0.0, 0.25, 0.5, 0.75, 1.0],
            examination_types=["exp"],
            n_runs=10,
            exam_threshold=5,
            visualize=args.visualize,
        )
