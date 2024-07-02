import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
sns.set_palette("husl")


def setup_visualization(df: pd.DataFrame, crowding_levels: list[float], examination_types: list[str]) -> None:
    """
    Visualize the performance of different methods for each crowding level and examination type.

    Args:
        df (pandas.DataFrame): The input dataframe containing the results.
        crowding_levels (list): A list of crowding levels to visualize.
        examination_types (list): A list of examination types to visualize.
    """
    for crowding in crowding_levels:
        for exam_type in examination_types:
            # Create a figure and set up the axes
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_facecolor("#F5F5F5")

            # Set up the color palette for different methods
            color_palette = sns.color_palette("husl", len(df["Method"].unique()))

            # Plot the results for each method
            for i, (method, group) in enumerate(
                df[(df["Crowding"] == crowding) & (df["Examination"] == exam_type)].groupby("Method")
            ):
                ax.scatter(
                    group["Time per step (s)"],
                    group["Expected number of total matches"],
                    label=method,
                    s=100,
                    alpha=0.8,
                    color=color_palette[i],
                    edgecolors="black",
                    linewidths=1.5,
                )

                # Add annotations for the number of matches (n)
                for _, row in group.iterrows():
                    ax.annotate(
                        row["n"],
                        (row["Time per step (s)"], row["Expected number of total matches"]),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        fontsize=14,
                    )

            # Set labels, title, and legend
            ax.set_xlabel("Time per step (s)", fontsize=18)
            ax.set_ylabel("Expected number of total matches", fontsize=18)
            ax.set_xscale("log")
            ax.set_title(f"Crowding: {crowding}, Examination: {exam_type}", fontsize=20)
            ax.xaxis.labelpad = 15
            ax.yaxis.labelpad = 15
            ax.title.set_position([0.5, 1.05])
            leg = ax.legend(fontsize=16, loc="upper left", frameon=True, fancybox=True, framealpha=0.8)
            leg.get_frame().set_edgecolor("black")
            leg.get_frame().set_linewidth(1.5)

            # Save the figure and results
            save_results(fig, df, crowding, exam_type)


def save_results(fig: matplotlib.figure, df: pd.DataFrame, crowding: float, exam_type: str) -> None:
    """
    Save the visualization and results for a specific crowding level and examination type.

    Args:
        fig (matplotlib.figure.Figure): The figure to save.
        df (pandas.DataFrame): The input dataframe containing the results.
        crowding (str): The crowding level.
        exam_type (str): The examination type.
    """
    logs_dir = f"/workspace/logs/crowding_{crowding}_examination_{exam_type}"
    os.makedirs(logs_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"{logs_dir}/performance_vs_time.png", dpi=300, bbox_inches="tight")

    with open(f"{logs_dir}/results.pkl", "wb") as f:
        pickle.dump(df[(df["Crowding"] == crowding) & (df["Examination"] == exam_type)], f)

    plt.close(fig)


def compare_expected_matches(
    df: pd.DataFrame,
    exam_type: str,
    size: int = None,
    crowding: float = None,
) -> None:
    """
    Compare the expected number of total matches for different methods and number of matches (n).

    Args:
        df (pandas.DataFrame): The input dataframe containing the results.
        crowding (str): The crowding level.
        exam_type (str): The examination type.
    """

    method_replace = {
        "random": "Random",
        "naive": "Naive",
        "reciprocal": "Reciprocal",
        "Cross Ratio": "CR",
        "batch-IPFP": "IPFP",
        "minibatch-IPFP": "Mini-batch",
    }
    df["Method"] = df["Method"].replace(method_replace)
    # remove Crowding = 1.0
    df = df[df["Crowding"] != 1.0]
    # remove Random
    df = df[df["Method"] != "Random"]

    unique_n = sorted(df["n"].unique())
    unique_crowding = sorted(df["Crowding"].unique())

    if crowding is not None:
        graph_in_row = unique_n
        param_name = "n"
        condition_name = "Crowding"
        condition_value = crowding
    elif size is not None:
        graph_in_row = unique_crowding
        param_name = "Crowding"
        condition_name = "n"
        condition_value = size
    else:
        raise ValueError("Either sizes or crowding should be provided")
    num_subplots = len(graph_in_row)

    fig, axes = plt.subplots(1, num_subplots, figsize=(18, 6), sharey=True)
    plt.subplots_adjust(wspace=0.1)

    for i, n in enumerate(graph_in_row):
        if num_subplots == 1:
            ax = axes
        else:
            ax = axes[i]
        subset_df = df[
            (df[condition_name] == condition_value) & (df["Examination"] == exam_type) & (df[param_name] == n)
        ]
        sns.barplot(
            x="Method",
            y="Expected number of total matches",
            data=subset_df,
            ax=ax,
            palette="viridis",
            hue="Method",
            legend=False,
        )
        ax.set_title(f"{param_name} = {n}", fontsize=16)
        ax.set_xlabel("")

        if i == 0:
            ax.set_ylabel("Expected number of total matches", fontsize=18)
        else:
            ax.set_ylabel("")

        ax.set_ylim(0, max(df["Expected number of total matches"]) * 1.1)

    plt.tight_layout()

    logs_dir = f"/workspace/logs/synthetic_examination_{exam_type}"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    plt.savefig(
        f"{logs_dir}/expected_matches_comparison_{condition_name}={condition_value}.png", dpi=300, bbox_inches="tight"
    )
    plt.savefig(
        f"{logs_dir}/expected_matches_comparison_{condition_name}={condition_value}.eps",
        format="eps",
        bbox_inches="tight",
    )
    plt.close(fig)


def compare_expected_matches_realdata(df: pd.DataFrame, exam_type: str, dirname: str) -> None:
    """
    Compare the expected number of total matches for different methods and number of matches (n).

    Args:
        df (pandas.DataFrame): The input dataframe containing the results.
        crowding (str): The crowding level.
        exam_type (str): The examination type.
    """
    unique_n = sorted(df["n"].unique())
    num_subplots = len(unique_n)
    method_replace = {
        "random": "Random",
        "naive": "Naive",
        "reciprocal": "Reciprocal",
        "batch-IPFP": "Batch IPFP",
        "minibatch-IPFP": "Mini-batch IPFP",
    }
    df["Method"] = df["Method"].replace(method_replace)
    # remove Random
    df = df[df["Method"] != "Random"]
    fig, axes = plt.subplots(1, num_subplots, figsize=(12, 6), sharey=True)
    plt.subplots_adjust(wspace=0.1)

    for i, n in enumerate(unique_n):
        if num_subplots == 1:
            ax = axes
        else:
            ax = axes[i]
        subset_df = df[(df["Examination"] == exam_type) & (df["n"] == n)]
        sns.barplot(
            x="Method",
            y="Expected number of total matches",
            data=subset_df,
            ax=ax,
            palette="viridis",
            hue="Method",
            legend=False,
        )
        if dirname == "taichi":
            datasetname = "Zhilian"
            n_x = 1060
            n_y = 154
        elif dirname == "libimseti":
            datasetname = "Libimseti"
            n_x = 500
            n_y = 500
        ax.set_title(f"{datasetname} dataset; data size = ({n_x}, {n_y})", fontsize=24)
        ax.set_xlabel("")
        # method name fontsize
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)

        if i == 0:
            ax.set_ylabel("Expected number of total matches", fontsize=18)
        else:
            ax.set_ylabel("")

        ax.set_ylim(0, max(df["Expected number of total matches"]) * 1.1)

    plt.tight_layout()

    logs_dir = f"/workspace/logs/{dirname}/realdata_examination_{exam_type}"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    plt.show()
    plt.savefig(f"{logs_dir}/expected_matches_comparison_real.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{logs_dir}/expected_matches_comparison_real.eps", format="eps", bbox_inches="tight")
    plt.close(fig)
