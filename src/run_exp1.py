import itertools
import os
import subprocess
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

SAVE_DIR = "exp1"
TIME_OUT = 100000  # sec
LOG_FILENAME = "exp.csv"


if __name__ == "__main__":
    problem_sizes = [100, 1000, 10000]
    method_settings = [
        {"method": "Batch IPFP", "factorize": False},
        # {"method": "Mini-Batch IPFP", "factorize": True},
    ]

    # touch log file
    if not os.path.exists(f"logs/{SAVE_DIR}/profile"):
        os.makedirs(f"logs/{SAVE_DIR}/profile")
    with open(f"logs/{SAVE_DIR}/{LOG_FILENAME}", "w") as f:
        f.write(f"method,device,size,batch_size,exec_time,max_mem\n")

    for size, device, method in itertools.product(problem_sizes, ["gpu"], method_settings):
        try:
            subprocess.run(
                [
                    "python",
                    "src/subprocess_exp.py",
                    "--method",
                    str(method["method"]),
                    "--size",
                    str(size),
                    "--device",
                    device,
                    "--factorize",
                    "True" if method["factorize"] else "",
                    "--batch_size",
                    str(100),
                    "--save_dir",
                    SAVE_DIR,
                ],
                timeout=TIME_OUT,
            )
        except subprocess.TimeoutExpired:
            print(f"Timeout: {method['method']} ({device}) ({size})")
        finally:
            time.sleep(1)

    # read result csv file
    result_table = pd.read_csv(f"logs/{SAVE_DIR}/{LOG_FILENAME}")

    # Plotting execution times
    plt.figure(figsize=(12, 8))
    plt.style.use("ggplot")
    plt.xscale("log")
    plt.yscale("log")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.LogFormatter(base=10.0, labelOnlyBase=True))
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.xlabel("Sample Size (n)", fontsize=36)
    plt.ylabel("Time per Step (s)", fontsize=36)
    plt.title("Execution Time", fontsize=36)

    # Plot for each method and device
    for group, table in result_table.groupby(["method", "device"]):
        print(group)
        if group[0] == "Batch IPFP" and group[1] == "cpu":
            formatted_label = "(a) Batch (CPU)"
        elif group[0] == "Batch IPFP" and group[1] == "gpu":
            formatted_label = "(b) Batch (GPU)"
        elif group[0] == "Mini-Batch IPFP" and group[1] == "cpu":
            formatted_label = "(c) Mini-Batch (CPU)"
        elif group[0] == "Mini-Batch IPFP" and group[1] == "gpu":
            formatted_label = "(d) Mini-Batch (GPU)"

        plt.plot(table["size"], table["exec_time"], "o-", label=formatted_label, markersize=16)

    plt.legend(fontsize=28)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()

    os.makedirs(f"/workspace/logs/{SAVE_DIR}", exist_ok=True)
    # Save the figure
    plt.savefig(f"/workspace/logs/{SAVE_DIR}/exp1_execution_times.eps", format="eps")
    plt.savefig(f"/workspace/logs/{SAVE_DIR}/exp1_execution_times.png", dpi=300)

    # Plotting memory usage
    plt.figure(figsize=(12, 8))
    plt.style.use("ggplot")
    plt.xscale("log")
    plt.yscale("log")
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.LogFormatter(base=10.0, labelOnlyBase=True))
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.xlabel("Sample Size (n)", fontsize=36)
    plt.ylabel("Memory Usage (MB)", fontsize=36)
    plt.title("Memory Usage", fontsize=36)

    for group, table in result_table.groupby(["method", "device"]):
        if group[0] == "Batch IPFP" and group[1] == "cpu":
            formatted_label = "(a) Batch (CPU)"
        elif group[0] == "Batch IPFP" and group[1] == "gpu":
            formatted_label = "(b) Batch (GPU)"
        elif group[0] == "Mini-Batch IPFP" and group[1] == "cpu":
            formatted_label = "(c) Mini-Batch (CPU)"
        elif group[0] == "Mini-Batch IPFP" and group[1] == "gpu":
            formatted_label = "(d) Mini-Batch (GPU)"

        plt.plot(
            table["size"],
            table["max_mem"] / 1024 / 1024,
            "o-",
            label=formatted_label,
            markersize=16,
        )

    plt.legend(fontsize=28)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()

    plt.savefig(f"/workspace/logs/{SAVE_DIR}/exp1_memory_usages.eps", format="eps")
    plt.savefig(f"/workspace/logs/{SAVE_DIR}/exp1_memory_usages.png", dpi=300)
