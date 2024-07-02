""" Preprocess the libimseti dataset to create a 1000x1000 matrix of ratings between top"""

import pickle

import implicit
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm
from typer import Typer

app = Typer()


def pmf_solve(
    A: np.ndarray, mask: np.ndarray, k: int, mu: float, epsilon: float = 1e-3, max_iterations: int = 100
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve probabilistic matrix factorization using alternating least squares.
    Since loss function is non-convex, each attempt at ALS starts from a
    random initialization and returns a local optimum.
    [ Salakhutdinov and Mnih 2008 ]
    [ Hu, Koren, and Volinksy 2009 ]
    Parameters:
    -----------
    A : m x n array
        matrix to complete
    mask : m x n array
        matrix with entries zero (if missing) or one (if present)
    k : integer
        how many factors to use
    mu : float
        hyper-parameter penalizing norm of factored U, V
    epsilon : float
        convergence condition on the difference between iterative results
    max_iterations: int
        hard limit on maximum number of iterations
    Returns:
    --------
    X: m x n array
        completed matrix
    """
    m, n = A.shape

    U = np.random.randn(m, k)
    V = np.random.randn(n, k)

    C_u = [np.diag(row) for row in mask]
    C_v = [np.diag(col) for col in mask.T]

    prev_X = np.dot(U, V.T)

    for _ in tqdm(range(max_iterations)):

        for i in range(m):
            U[i] = np.linalg.solve(
                np.linalg.multi_dot([V.T, C_u[i], V]) + mu * np.eye(k), np.linalg.multi_dot([V.T, C_u[i], A[i, :]])
            )

        for j in range(n):
            V[j] = np.linalg.solve(
                np.linalg.multi_dot([U.T, C_v[j], U]) + mu * np.eye(k), np.linalg.multi_dot([U.T, C_v[j], A[:, j]])
            )

        X = np.dot(U, V.T)

        mean_diff = np.linalg.norm(X - prev_X) / m / n
        if mean_diff < epsilon:
            break
        prev_X = X

    return X, U, V


@app.command()
def preprocess_libimseti() -> None:
    """Preprocess the libimseti dataset to create a 1000x1000 matrix of ratings between"""
    gender = pd.read_csv("./data/libimseti/gender.dat")
    gender.columns = ["ID", "Gender"]

    ratings = pd.read_csv("./data/libimseti/ratings.dat")
    ratings.columns = ["UserID", "ProfileID", "Rating"]

    M_ids = np.array(
        ratings.loc[ratings["UserID"].isin(gender.loc[gender["Gender"] == "M", "ID"]), "UserID"]
        .value_counts()
        .index[:1000]
    )

    F_ids = np.array(
        ratings.loc[ratings["UserID"].isin(gender.loc[gender["Gender"] == "F", "ID"]), "UserID"]
        .value_counts()
        .index[:1000]
    )

    mid_to_row = {mid: row for row, mid in enumerate(M_ids)}
    fid_to_row = {fid: row for row, fid in enumerate(F_ids)}

    m_ratings = ratings.loc[(ratings["UserID"].isin(M_ids)) & (ratings["ProfileID"].isin(F_ids))]
    m_ratings["UserID"] = m_ratings["UserID"].apply(lambda x: mid_to_row[x])
    m_ratings["ProfileID"] = m_ratings["ProfileID"].apply(lambda x: fid_to_row[x])

    m_A = np.zeros((1000, 1000), dtype=np.float32)
    m_mask = np.zeros((1000, 1000), dtype=np.int8)

    male_A = csr_matrix((m_ratings["Rating"].values / 10, (m_ratings["UserID"].values, m_ratings["ProfileID"].values)))
    save_npz("./data/libimseti/male_to_female_rel.npz", male_A)

    for _, r in tqdm(m_ratings.iterrows(), total=m_ratings.shape[0]):
        row, col, val = r["UserID"], r["ProfileID"], r["Rating"]
        m_mask[row, col] = 1
        m_A[row, col] = val / 10.0

    f_ratings = ratings.loc[(ratings["UserID"].isin(F_ids)) & (ratings["ProfileID"].isin(M_ids))]
    f_ratings["UserID"] = f_ratings["UserID"].apply(lambda x: fid_to_row[x])
    f_ratings["ProfileID"] = f_ratings["ProfileID"].apply(lambda x: mid_to_row[x])

    female_A = csr_matrix(
        (f_ratings["Rating"].values / 10, (f_ratings["UserID"].values, f_ratings["ProfileID"].values))
    )
    save_npz("./data/libimseti/female_to_male_rel.npz", female_A)

    return

    f_A = np.zeros((1000, 1000), dtype=np.float32)
    f_mask = np.zeros((1000, 1000), dtype=np.int8)

    for _, r in tqdm(f_ratings.iterrows(), total=f_ratings.shape[0]):
        row, col, val = r["UserID"], r["ProfileID"], r["Rating"]
        f_mask[row, col] = 1
        f_A[row, col] = val / 10.0

    imputed_m, factor_m_u, factor_m_v = pmf_solve(m_A, m_mask, 6, 1e-2)
    imputed_f, factor_f_u, factor_f_v = pmf_solve(f_A, f_mask, 6, 1e-2)

    # clipped_f = np.clip(imputed_f, 0, 1)
    # clipped_m = np.clip(imputed_m, 0, 1)

    with open("./data/libimseti/female_to_male_rel_500.pkl", "wb") as fp:
        pickle.dump(imputed_f[:500, :500], fp)

    with open("./data/libimseti/male_to_female_rel_500.pkl", "wb") as fp:
        pickle.dump(imputed_m[:500, :500], fp)

    with open("./data/libimseti/female_to_male_u.pkl", "wb") as fp:
        pickle.dump(factor_f_u[:500], fp)

    with open("./data/libimseti/female_to_male_v.pkl", "wb") as fp:
        pickle.dump(factor_f_v[:500], fp)

    with open("./data/libimseti/male_to_female_u.pkl", "wb") as fp:
        pickle.dump(factor_m_u[:500], fp)

    with open("./data/libimseti/male_to_female_v.pkl", "wb") as fp:
        pickle.dump(factor_m_v[:500], fp)


@app.command()
def preprocess_taichi() -> None:
    """Preprocess the Taichi dataset to create a 1000x1000 matrix of ratings between top"""
    threshold_filter_cv = 10  # filter candidates with less than threshold_filter "delivered" actions
    threshold_filter_jd = 10  # filter jobs with less than threshold_filter "satisfied" actions

    n_factor_cv = 50
    n_factor_jd = 50

    action = pd.read_csv("data/taichi/table3_action.txt", sep="\t")

    # user_id
    user_id = action["user_id"].unique()
    print("all number of candidates", len(user_id))
    job_id = action["jd_no"].unique()
    print("all number of jobs", len(job_id))

    # filter jobs with less than threshold_filter "satisfied" actions given
    action_top_jd = action.groupby("jd_no").filter(lambda x: x["delivered"].sum() >= threshold_filter_jd)
    top_jd_ids = action_top_jd["jd_no"].unique()
    # filter candidates with less than threshold_filter "delivered" actions to filtered jobs
    action_top_cv = action.groupby("user_id").filter(lambda x: x["satisfied"].sum() >= threshold_filter_cv)
    top_cv_ids = action_top_cv["user_id"].unique()

    print("filtered number of candidates", len(top_cv_ids))
    print("filtered number of jobs", len(top_jd_ids))
    uid_to_row = {mid: row for row, mid in enumerate(top_cv_ids)}
    jid_to_row = {fid: row for row, fid in enumerate(top_jd_ids)}

    # filter actions to only include top_cv_ids and top_jd_ids
    action = action[(action["user_id"].isin(top_cv_ids)) & (action["jd_no"].isin(top_jd_ids))]

    action["user_id"] = action["user_id"].apply(lambda x: uid_to_row[x])
    action["jd_no"] = action["jd_no"].apply(lambda x: jid_to_row[x])

    # create a sparse matrix of ratings
    actions_with_delivered = action[action["delivered"] == 1]
    rows = actions_with_delivered["user_id"].values
    cols = actions_with_delivered["jd_no"].values
    vals = actions_with_delivered["delivered"].values

    user_A = csr_matrix((vals, (rows, cols)), shape=(len(top_cv_ids), len(top_jd_ids)))
    save_npz("./data/taichi/user_to_job_rel.npz", user_A)
    # als_user = implicit.als.AlternatingLeastSquares(factors=n_factor_cv, random_state=2)
    # als_user.fit(user_A)

    # imputed_user = als_user.user_factors
    # imputed_job = als_user.item_factors

    # imputed_preference = np.dot(imputed_user, imputed_job.T)

    # with open("./data/taichi/user_to_job_rel.pkl", "wb") as fp:
    #     pickle.dump(imputed_preference, fp)
    # with open("./data/taichi/user_to_job_u.pkl", "wb") as fp:
    #     pickle.dump(imputed_user, fp)
    # with open("./data/taichi/user_to_job_v.pkl", "wb") as fp:
    #     pickle.dump(imputed_job, fp)

    actions_with_satisfied = action[action["satisfied"] == 1]
    rows = actions_with_satisfied["jd_no"].values
    cols = actions_with_satisfied["user_id"].values
    vals = actions_with_satisfied["satisfied"].values

    job_A = csr_matrix((vals, (rows, cols)), shape=(len(top_jd_ids), len(top_cv_ids)))
    save_npz("./data/taichi/job_to_user_rel.npz", job_A)

    return
    als_job = implicit.als.AlternatingLeastSquares(factors=n_factor_jd, random_state=2)
    als_job.fit(job_A)

    imputed_job = als_job.user_factors
    imputed_user = als_job.item_factors

    imputed_preference = np.dot(imputed_job, imputed_user.T)

    with open("./data/taichi/job_to_user_rel.pkl", "wb") as fp:
        pickle.dump(imputed_preference, fp)

    with open("./data/taichi/job_to_user_u.pkl", "wb") as fp:
        pickle.dump(imputed_job, fp)

    with open("./data/taichi/job_to_user_v.pkl", "wb") as fp:
        pickle.dump(imputed_user, fp)


if __name__ == "__main__":
    app()
