""" Run subprocess experiment for Batch / MiniBatch IPFP """
import gc
import subprocess
import time
from functools import partial
import re
import argparse

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from ott.problems.linear import linear_problem
from sinkhornIPFP import IPFP
from jax import profiler

from geomIPFP import PointCloudIPFP, GeometryIPFP, DotProd

# Configuration and constants
MAX_STEPS = 100
DIMENSION = 100
EPSILON_SCALE = 0.1
FIXED_SEED = 42
LOG_FILENAME = "exp.csv"


def get_memory_usage(filename: str) -> float:
    # run pprof to get memory usage
    output = subprocess.run(
        ["pprof", "--text", "--unit=B", filename],
        stdout=subprocess.PIPE,
        text=True,
    ).stdout
    # get max memory usage in float
    max_mem = output.split("\n")[1].split()[-2]
    matches = re.search(r"(\d+(\.\d+)?)", max_mem)
    max_mem = float(matches.group(1))
    return max_mem


def sample_points(rng, num_points, dimension):
    """Sample uniform factor vectors and create marginal vectors."""
    rng, *rngs = jax.random.split(rng, 5)
    # the factor matrices F,G,K,L in our paper are concatenated into a single matrix x,y.
    x = jax.random.uniform(rngs[0], (num_points, dimension)) / jax.lax.sqrt(float(dimension))
    y = jax.random.uniform(rngs[1], (num_points, dimension)) / jax.lax.sqrt(float(dimension))
    # marginal vectors a,b are all ones * num_points
    a = b = jnp.ones((num_points,)) * num_points
    # Stop gradient computation for constants
    x, y, a, b = map(jax.lax.stop_gradient, (x, y, a, b))
    return a, b, x, y


@partial(jax.jit, static_argnames=["factorize", "batch_size"])
def solve_ott(a, b, x, y, epsilon, threshold, factorize, batch_size=None):
    """Solve stable matching using IPFP algorithm"""
    if factorize:
        print(f"batch_size: {batch_size}")
        geom = PointCloudIPFP(x, y, cost_fn=DotProd(), epsilon=epsilon, batch_size=batch_size)
    else:
        cost_matrix = -jnp.dot(x, y.T) / 2.0
        geom = GeometryIPFP(cost_matrix=cost_matrix, epsilon=epsilon)

    problem = linear_problem.LinearProblem(geom, a=a, b=b)
    solver = IPFP(
        threshold=threshold,
        max_iterations=MAX_STEPS,
        lse_mode=False,
    )
    solution = solver(problem)
    return solution


def run_simulation(rng, size, epsilon, threshold, factorize=False, batch_size=None, device="cpu"):
    """Run simulation on a specified device (CPU or GPU) and measure the execution time and memory usage."""
    a, b, x, y = sample_points(rng, size, DIMENSION)
    a = jax.device_put(a, jax.devices(device)[0])
    b = jax.device_put(b, jax.devices(device)[0])
    x = jax.device_put(x, jax.devices(device)[0])
    y = jax.device_put(y, jax.devices(device)[0])
    assert x.dtype == "float32", "Input data type should be float32."

    if device == "cpu":
        # a, b, x, y = map(np.array, (a, b, x, y))

        # Measure execution time and memory usage
        start_time = time.time()
        result = solve_ott(a, b, x, y, epsilon, threshold, factorize, batch_size)
        end_time = time.time()
        exec_time = end_time - start_time

        print(f"size {len(x)} calc time: {exec_time} sec")
        return end_time - start_time, result
    else:
        # Measure only execution time for GPU
        start_time = time.time()
        result = solve_ott(a, b, x, y, epsilon, threshold, factorize, batch_size)
        end_time = time.time()
        exec_time = end_time - start_time
        print(f"size {len(x)} calc time: {exec_time} sec")

    return exec_time, result


def check_matrix(size, out_geo, out_pc, device):
    # Assert convergence and compare results
    # This function is only for debugging
    print(f"on-memory matrix ({device}):", out_geo.matrix)
    print(f"minibatch matrix ({device}):", out_pc.matrix)
    assert np.allclose(out_geo.matrix, out_pc.matrix, rtol=1e-1), "Result matrices do not match."

    # Visualize and save transport matrices as images
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(out_geo.matrix, cmap="Purples")
    ax[0].set_title(f"On-Memory Matrix ({device})")
    ax[1].imshow(out_pc.matrix, cmap="Purples")
    ax[1].set_title(f"Minibatch Matrix ({device})")
    plt.savefig(f"/workspace/results/{size}_{device}.png")
    plt.close(fig)


def main(method, size, device, factorize, batch_size, save_dir):
    """Run single experiment with specified parameters.

    Args:
        method (str): method name
        size (int): data size
        device (str): cpu or gpu
        factorize (bool): if True, use minibatch IPFP, else use on-memory IPFP
        batch_size (int): batch size for minibatch IPFP
        save_dir (str): directory name to save results
    """
    rng = jax.random.PRNGKey(FIXED_SEED)

    epsilon = 1.0 * EPSILON_SCALE
    threshold_n = 0.0  # set threshold to 0 to run all steps
    print(f"\nSize: {size}, Epsilon: {epsilon:.5f}, Threshold: {threshold_n:.5f}")

    # Run Batch / Mini-Batch based IPFP
    batch_size_min = min(batch_size, size)
    time_, out_ = run_simulation(
        rng,
        size,
        epsilon,
        threshold_n,
        factorize=factorize,
        batch_size=batch_size_min,
        device=device,
    )
    print(f"Calculation Time ({device.upper()}):", time_)

    # save max memory usage
    filename = f"logs/{save_dir}/profile/memory_{device}_{size}_{factorize}.prof"
    profiler.save_device_memory_profile(filename, backend=device)
    max_mem = get_memory_usage(filename)

    # write execution time and memory usage to csv file last line
    with open(f"logs/{save_dir}/{LOG_FILENAME}", "a") as f:
        f.write(f"{method},{device},{size},{batch_size_min},{time_ / MAX_STEPS},{max_mem}\n")

    del rng, time_, out_
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--factorize", type=bool, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()
    print(args)
    main(args.method, args.size, args.device, args.factorize, args.batch_size, args.save_dir)
