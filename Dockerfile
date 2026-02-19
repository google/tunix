# Base image with Python 3.12
FROM python:3.12-slim

# Set environment variables to non-interactive to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies, including Python 3 and pip
RUN apt-get update && \
    apt-get install -y git python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install nano for easier file editing
RUN apt-get update && apt-get install -y nano

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Create a virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

RUN pip install git+https://github.com/ayaka14732/jax-smi.git
RUN pip install git+https://github.com/AI-Hypercomputer/pathways-utils.git
# RUN pip uninstall -y pathwaysutils
# If you encounter a checkpoint issue, try using following old version of pathways-utils.
# RUN pip install git+https://github.com/AI-Hypercomputer/pathways-utils.git@b72729bb152b7b3426299405950b3af300d765a9#egg=pathwaysutils
RUN pip install gcsfs
RUN pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

RUN pip install --upgrade wandb


# Set the working directory
WORKDIR /app

# Copy the project files to the image
COPY . .

# Install the project in editable mode
RUN pip install  --force-reinstall .

# Set a directory to clone sglang-jax into
# WORKDIR /usr/src
# Clone the repository using HTTPS
# RUN rm -rf sglang-jax && git clone https://github.com/sgl-project/sglang-jax.git 
# WORKDIR /usr/src
# Install the package in editable mode
# The -e flag means the installation links to the source code in /usr/src/sglang-jax
# RUN cd sglang-jax/python && pip install --force-reinstall --no-cache-dir  .

# # Install vllm
# RUN pip install vllm-tpu
# vllm dependencies
RUN pip install vllm==0.15.1
WORKDIR /usr/src
RUN rm -rf tpu-inference && git clone https://github.com/vllm-project/tpu-inference.git
RUN cd tpu-inference && git fetch origin pull/1466/head:pr-1466 && git checkout pr-1466; pip install -e .

WORKDIR /app


# Set the default command to bash
CMD ["bash"]