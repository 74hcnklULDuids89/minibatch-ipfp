# Mini-Batch IPFP

This repository contains a tool for benchmarking various implementations of the Iterative Proportional Fitting Procedure (IPFP) algorithm within a Docker environment. We have encapsulated the dependencies and runtime into a Docker container to streamline the setup process. Currently, we support benchmarking using [ott-jax](https://github.com/ott-jax/ott).

**This software includes the work that is distributed in the Apache License 2.0.**

The following source codes are modified from original sources:
- `src/sinkhornIPFP.py` is modified from [ott-jax sinkhorn.py](https://github.com/ott-jax/ott/blob/main/src/ott/solvers/linear/sinkhorn.py)
- `src/geomIPFP.py` is modified from [ott-jax pointcloud.py](https://github.com/ott-jax/ott/blob/main/src/ott/geometry/pointcloud.py)

## Features

- Compare computation times across different IPFP implementations.
- Visualize the average computation time and memory usage with respect to input sizes.

## Prerequisites

- Docker (ensure your Docker can handle GPU workloads if you plan on utilizing GPUs)

## Usage

1. Clone this repository:
    ```bash
    git clone https://github.com/KNakadas/tumatching-sinkhorn.git
    cd tumatching-sinkhorn
    ```

2. Build the Docker image:
    ```bash
    docker build --no-cache -t tu-matching-gpu .
    ```

3. Run the container utilizing your GPU (ensure your Docker setup supports `--gpus` flag):
    ```bash
    docker run --gpus all -it --rm -v $HOME/tumatching-sinkhorn/:/workspace/ tu-matching-gpu /bin/bash
    ```

4. Inside the container, you can execute the main script:
    ```bash
    python src/run_exp1.py
    ```
    or
    ```bash
    python src/run_exp_minibatch.py
    ```


5. After running, view the generated plots to compare different IPFP methods. The plots will be stored in the mounted volume (specified by `-v` in the `docker run` command), and you can view them on your host machine.

## Notes

- The `-v` flag in the `docker run` command mounts a directory from your host system into the Docker container to ensure that logs or output files generated within the Docker environment are persisted and accessible even after the container is stopped or removed.
- The `--gpus all` flag allocates all available GPUs to your Docker container. Ensure that your Docker version and hardware support this feature. You might need to set up NVIDIA Docker runtime or make adjustments depending on your system configuration.
