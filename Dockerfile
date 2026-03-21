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


RUN pip uninstall wandb
RUN pip install wandb==0.24.2


# Set the working directory
WORKDIR /app

# Copy the project files to the image
COPY . .

# Install the project in editable mode
RUN pip install  --force-reinstall -e .

ARG ENGINE
# Set a directory to clone sglang-jax or vllm into
WORKDIR /usr/src

RUN if [ "$ENGINE" = "sglang_jax" ]; then \
        rm -rf sglang-jax && git clone https://github.com/sgl-project/sglang-jax.git && \
        cd sglang-jax/python && pip install --force-reinstall --no-cache-dir .; \
    elif [ "$ENGINE" = "vllm" ]; then \
        rm -rf vllm && git clone https://github.com/wang2yn84/vllm.git && cd vllm && git checkout lance-ds && \
        pip uninstall torch torch-xla -y && pip install -r requirements/tpu.txt && VLLM_TARGET_DEVICE="tpu" python -m pip install -e . && \
        cd .. && \
        rm -rf tpu-inference && git clone https://github.com/vllm-project/tpu-inference.git && \
        cd tpu-inference && git checkout lance-ds && pip install -e .; \
    else \
        echo "Skip engine installation or unsupported engine: $ENGINE"; \
    fi

WORKDIR /app
# RUN pip install --force-reinstall flax==0.12.4
# RUN pip install "numpy>=2.0,<2.3"
RUN pip install --force-reinstall protobuf==6.33.5


# Set the default command to bash
CMD ["bash"]