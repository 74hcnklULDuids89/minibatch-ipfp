import itertools
import os
import subprocess
import time

import matplotlib.pyplot as plt
import pandas as pd

SAVE_DIR = "exp_minibatch"
TIME_OUT = 100000  # sec
LOG_FILENAME = "exp.csv"


if __name__ == "__main__":
    problem_sets = [
        {"size": 100, "batch_size": [100]},
        {"size": 1000, "batch_size": [100]},
        {"size": 10000, "batch_size": [100]},
        {"size": 100000, "batch_size": [100]},
        {"size": 1000000, "batch_size": [100]},
    ]
    method_settings = [
        {"method": "MiniBatch IPFP", "factorize": True},
    ]

    # touch log file
    if not os.path.exists(f"logs/{SAVE_DIR}/profile"):
        os.makedirs(f"logs/{SAVE_DIR}/profile")

    with open(f"logs/{SAVE_DIR}/{LOG_FILENAME}", "w") as f:
        f.write(f"method,device,size,batch_size,exec_time,max_mem\n")

    for device, method in itertools.product(["gpu"], method_settings):
        for problem_set in problem_sets:
            size = problem_set["size"]
            for batch_size in problem_set["batch_size"]:
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
                            str(batch_size),
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
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.xlabel("Sample Size (n)", fontsize=36)
    plt.ylabel("Time per Step (s)", fontsize=36)
    plt.title("Execution Time", fontsize=36)

    # Plot for each method and device
    for group, table in result_table.groupby(["method", "device", "batch_size"]):
        if group[2] == 10000:
            continue
        print(group)
        if group[0] == "Batch IPFP":
            formatted_label = f"{group[0]} ({group[1]})".replace("(cpu)", "(CPU)").replace("(gpu)", "(GPU)")
        else:
            formatted_label = f"Mini-Batch ({group[1]}) size={group[2]}".replace("(cpu)", "(CPU)").replace(
                "(gpu)", "(GPU)"
            )
        plt.plot(table["size"], table["exec_time"], "o-", label=formatted_label, markersize=16)

    plt.legend(fontsize=28)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()

    os.makedirs(f"/workspace/logs/{SAVE_DIR}", exist_ok=True)
    # Save the figure
    plt.savefig(f"/workspace/logs/{SAVE_DIR}/exp2_execution_times.eps", format="eps")
    plt.savefig(f"/workspace/logs/{SAVE_DIR}/exp2_execution_times.png", dpi=300)

    # Plotting memory usage
    plt.figure(figsize=(12, 8))
    plt.style.use("ggplot")
    plt.xscale("log")
    plt.yscale("log")
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    plt.xlabel("Sample Size (n)", fontsize=36)
    plt.ylabel("Memory Usage (MB)", fontsize=36)
    plt.title("Memory Usage", fontsize=36)

    for group, table in result_table.groupby(["method", "device", "batch_size"]):
        if group[2] == 10000:
            continue
        print(group)
        if group[0] == "Batch IPFP":
            formatted_label = f"{group[0]} ({group[1]})".replace("(cpu)", "(CPU)").replace("(gpu)", "(GPU)")
        else:
            formatted_label = f"Mini-Batch ({group[1]}) size={group[2]}".replace("(cpu)", "(CPU)").replace(
                "(gpu)", "(GPU)"
            )
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

    plt.savefig(f"/workspace/logs/{SAVE_DIR}/exp2_memory_usages.eps", format="eps")
    plt.savefig(f"/workspace/logs/{SAVE_DIR}/exp2_memory_usages.png", dpi=300)
