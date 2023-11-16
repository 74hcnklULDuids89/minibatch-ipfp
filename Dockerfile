FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

WORKDIR /workspace

# Install Python and pip
RUN apt-get update && apt-get install -y python3 golang-go python3-pip && rm -rf /var/lib/apt/lists/*
# Install JAX Memory profiler
RUN go install github.com/google/pprof@latest
# Install JAX with CUDA support
RUN pip install --upgrade pip
RUN pip install --default-timeout=1000 --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install -U --pre jaxlib==0.4.20+cuda12.cudnn89 -f https://storage.googleapis.com/jax-releases/jaxlib_nightly_cuda12_releases.html

# Install other Python dependencies
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Define environment variable to enable JAX to see the GPU
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda
# Disable JAX's GPU memory preallocation
ENV XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Symlink python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

# add go to PATH
ENV PATH $PATH:/root/go/bin
