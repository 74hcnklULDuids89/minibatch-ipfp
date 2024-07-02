""" Run subprocess experiment for Batch / MiniBatch IPFP """

import argparse
import gc
import os
import re
import subprocess
import time
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import config, profiler
from jax._src.typing import Array
from ott.problems.linear import linear_problem
from ott.solvers.linear.sinkhorn import SinkhornOutput

from geomIPFP import DotProd, GeometryIPFP, PointCloudIPFP
from sinkhornIPFP import IPFP

# config.update("jax_enable_x64", True)


# Configuration and constants
MAX_STEPS = 100
DIMENSION = 50 * 2
EPSILON_SCALE = 1.0
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


def sample_points(
    rng: Array, num_points: int, dimension: int, sample_type: str = "random"
) -> tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
    """Sample uniform factor vectors and create marginal vectors."""
    rng, *rngs = jax.random.split(rng, 3)
    # the factor matrices F,G,K,L in our paper are concatenated into a single matrix x,y.
    if sample_type == "random":
        x = jax.random.uniform(rngs[0], (num_points, dimension)) / jax.lax.sqrt(float(dimension))
        y = jax.random.uniform(rngs[1], (num_points, dimension)) / jax.lax.sqrt(float(dimension))

    elif sample_type == "sin_wave":
        # experiment with sine wave, no random sampling
        x_lin = jnp.array(
            [jnp.ones(dimension) * i for i in jnp.linspace(0, 1, num_points)],
            dtype=jnp.float32,
        ) / jnp.sqrt(dimension)
        y_lin = jnp.array(
            [jnp.ones(dimension) * i for i in jnp.linspace(0, 1, num_points)],
            dtype=jnp.float32,
        ) / jnp.sqrt(dimension)
        x = x_lin
        y = y_lin

        # plot preference matrix
        if num_points <= 1000:
            plt.figure(figsize=(12, 8))
            plt.imshow(np.matmul(x, y.T), cmap="Purples")
            plt.colorbar()
            plt.savefig("pref_reciprocal.png")

    else:
        raise ValueError("Invalid sample type")

    # marginal vectors a,b are all ones * num_points
    a = b = jnp.ones((num_points,), dtype=jnp.float32) * num_points
    # Stop gradient computation for constantsa
    x, y, a, b = map(jax.lax.stop_gradient, (x, y, a, b))
    return a, b, x, y


@partial(jax.jit, static_argnames=["factorize", "batch_size"])
def solve_ott(
    a: Array,
    b: Array,
    x: Array,
    y: Array,
    epsilon: float,
    threshold: float,
    factorize: bool,
    batch_size: int | None = None,
) -> SinkhornOutput:
    """Solve stable matching using IPFP algorithm"""
    if factorize:
        if batch_size is not None:
            print(f"batch_size: {batch_size}")
            geom: PointCloudIPFP | GeometryIPFP = PointCloudIPFP(
                x, y, cost_fn=DotProd(), epsilon=epsilon, batch_size=batch_size
            )
        else:
            cost_matrix = (-jnp.dot(x, y.T)) / 2.0
            geom = GeometryIPFP(cost_matrix=cost_matrix, epsilon=epsilon)
    else:
        # preference matrix is given as input
        cost_matrix = -(x + y.T) / 2.0
        geom = GeometryIPFP(cost_matrix=cost_matrix, epsilon=epsilon)

    problem = linear_problem.LinearProblem(geom, a=a, b=b)

    print("solve start")
    solver = IPFP(
        threshold=threshold,
        max_iterations=MAX_STEPS,
        lse_mode=False,
        inner_iterations=1,
    )
    result = solver(problem)
    print("solve end")
    return result


def run_simulation(
    rng: Array,
    size: int,
    epsilon: float,
    threshold: float,
    factorize: bool = False,
    batch_size: int | None = None,
    device: str = "cpu",
    sample_type: str = "random",
    dimension: int = DIMENSION,
) -> tuple[float, SinkhornOutput]:
    """Run simulation on a specified device (CPU or GPU) and measure the execution time and memory usage."""
    a, b, x, y = sample_points(rng, size, dimension=dimension, sample_type=sample_type)

    a = jax.device_put(a, jax.devices(device)[0])
    b = jax.device_put(b, jax.devices(device)[0])
    x = jax.device_put(x, jax.devices(device)[0])
    y = jax.device_put(y, jax.devices(device)[0])

    # Measure execution time and memory usage
    start_time = time.time()
    result = solve_ott(a, b, x, y, epsilon, threshold, factorize, batch_size)
    end_time = time.time()
    exec_time = end_time - start_time

    print(f"size {len(x)} calc time: {exec_time} sec")

    return exec_time, result


def check_matrix(size: int, out_geo: GeometryIPFP, out_pc: PointCloudIPFP, device: int) -> None:
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


def main(
    method: str,
    size: int,
    device: str,
    factorize: bool,
    batch_size: int,
    save_dir: str,
    sample_type: str = "random",
    dimension: int = DIMENSION,
) -> None:
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
    print(f"save to logs/{save_dir}/error_{device}_{size}_{factorize}.png")

    # Run Batch / Mini-Batch based IPFP
    batch_size_min = min(batch_size, size)
    time_, result = run_simulation(
        rng,
        size,
        epsilon,
        threshold_n,
        factorize=factorize,
        batch_size=batch_size_min,
        device=device,
        sample_type=sample_type,
        dimension=dimension,
    )
    print(f"Calculation Time ({device.upper()}):", time_)

    # save max memory usage
    filename = f"/workspace/logs/{save_dir}/profile/memory_{device}_{size}_{factorize}.prof"
    profiler.save_device_memory_profile(filename, backend=device)
    max_mem = get_memory_usage(filename)

    # write execution time and memory usage to csv file last line
    with open(f"logs/{save_dir}/{LOG_FILENAME}", "a") as f:
        f.write(f"{method},{device},{size},{batch_size_min},{time_ / MAX_STEPS},{max_mem},{dimension}\n")

    # save result.erros to csv file
    with open(f"logs/{save_dir}/error_{device}_{size}_{factorize}.csv", "w") as f:
        f.write("errors\n")
        for error in result.errors:
            f.write(f"{error}\n")

    # plot errors
    plt.figure(figsize=(12, 8))
    plt.plot(result.errors)
    plt.title(f"Error {method} size={size} ({device.upper()})")
    print(f"logs/{save_dir}/error_{device}_{size}_{factorize}.png")
    plt.savefig(f"logs/{save_dir}/error_{device}_{size}_{factorize}.png")

    # plot errors (10~)
    plt.figure(figsize=(12, 8))
    plt.plot(result.errors[10:])
    plt.title(f"Error {method} size={size} ({device.upper()}) epoch 10~")
    print(f"logs/{save_dir}/error_{device}_{size}_{factorize}.png")
    plt.savefig(f"logs/{save_dir}/error_{device}_{size}_{factorize}_after10epoch.png")

    # plot errors (100~)
    plt.figure(figsize=(12, 8))
    plt.plot(result.errors[100:])
    plt.title(f"Error {method} size={size} ({device.upper()}) epoch 100~")
    print(f"logs/{save_dir}/error_{device}_{size}_{factorize}.png")
    plt.savefig(f"logs/{save_dir}/error_{device}_{size}_{factorize}_after100epoch.png")

    # plot errors (200~)
    plt.figure(figsize=(12, 8))
    plt.plot(result.errors[200:])
    plt.title(f"Error {method} size={size} ({device.upper()}) epoch 200~")
    print(f"logs/{save_dir}/error_{device}_{size}_{factorize}.png")
    plt.savefig(f"logs/{save_dir}/error_{device}_{size}_{factorize}_after200epoch.png")

    # plot matrix for small size
    if size <= 1000:
        plt.figure(figsize=(12, 8))
        plt.imshow(result.matrix, cmap="Purples")
        plt.colorbar()
        plt.title(f"Matrix {method} size={size} ({device.upper()})")
        plt.savefig(f"logs/{save_dir}/matrix_{device}_{size}_{factorize}.png")

    # print("result errors: ", result.errors)

    del rng, time_, result
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--factorize", type=bool, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--sample_type", type=str, default="random")
    parser.add_argument("--dimension", type=int, default=DIMENSION)
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.save_dir + "/profile"):
        os.makedirs(f"{args.save_dir}/profile")
    main(
        args.method,
        args.size,
        args.device,
        args.factorize,
        args.batch_size,
        args.save_dir,
        args.sample_type,
        args.dimension,
    )
