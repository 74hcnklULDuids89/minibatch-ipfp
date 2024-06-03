import pickle
import warnings

import numpy as np


def load_real_data(
    male_data_path: str, female_data_path: str, size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load real data from the specified paths"""
    with open(male_data_path + "_rel_500.pkl", "rb") as fp:
        male_pref = pickle.load(fp)
    with open(female_data_path + "_rel_500.pkl", "rb") as fp:
        female_pref = pickle.load(fp)
    with open(male_data_path + "_u.pkl", "rb") as fp:
        factor_m_u = pickle.load(fp)
    with open(male_data_path + "_v.pkl", "rb") as fp:
        factor_m_v = pickle.load(fp)
    with open(female_data_path + "_u.pkl", "rb") as fp:
        factor_f_u = pickle.load(fp)
    with open(female_data_path + "_v.pkl", "rb") as fp:
        factor_f_v = pickle.load(fp)

    if male_pref.shape[0] < size or female_pref.shape[0] < size:
        warnings.warn(
            f"The number of data points in the real data is less than the specified size {size}. Using the available data."
        )
        size = min(male_pref.shape[0], female_pref.shape[0])

    return (
        male_pref[:size, :size],
        female_pref[:size, :size],
        factor_m_u[:size],
        factor_m_v[:size],
        factor_f_u[:size],
        factor_f_v[:size],
    )


def generate_preference_matrices(
    num_jobs: int, num_candidates: int, crowding: float, seed: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Generate preference matrices for the synthetic data

    Args:
        num_jobs (int): Size of the job set
        num_candidates (int): Size of the candidate set
        crowding (float): Preference crowding level
        seed (int, optional): Random seed. Defaults to 1.

    Returns:
        tuple[np.ndarray, np.ndarray]: Preference matrices for candidates and jobs
    """
    np.random.seed(seed)
    pref_x_random = np.random.rand(num_candidates, num_jobs).astype(np.float32)
    pref_y_random = np.random.rand(num_jobs, num_candidates).astype(np.float32)

    pref_x_crowded = np.zeros((num_candidates, num_jobs), dtype=np.float32)
    pref_y_crowded = np.zeros((num_jobs, num_candidates), dtype=np.float32)

    for i in range(num_candidates):
        pref_x_crowded[i, :] = 1 - np.linspace(0, 1, num_jobs)
    for i in range(num_jobs):
        pref_y_crowded[i, :] = 1 - np.linspace(0, 1, num_candidates)

    pref_x = (1 - crowding) * pref_x_random + crowding * pref_x_crowded
    pref_y = (1 - crowding) * pref_y_random + crowding * pref_y_crowded

    return pref_x, pref_y
