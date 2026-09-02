# Base image with Python 3.12
FROM python:3.12-slim

# Set environment variables to non-interactive to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies, including Python 3 and pip
RUN apt-get update && \
    apt-get install -y build-essential curl git python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Create a virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Install uv
RUN pip install uv

RUN pip install git+https://github.com/ayaka14732/jax-smi.git
# If you encounter a checkpoint issue, try using following old version of pathways-utils.
# RUN pip install git+https://github.com/AI-Hypercomputer/pathways-utils.git@b72729bb152b7b3426299405950b3af300d765a9#egg=pathwaysutils
RUN pip install gcsfs
RUN pip install wandb

# Set the working directory
WORKDIR /app

# Copy scripts and requirements first for cached dependency installation
COPY requirements requirements
COPY scripts scripts
RUN bash /app/scripts/install_tunix_vllm_requirement.sh

# Build argument to conditionally install MaxText dependencies
ARG INSTALL_MAXTEXT=false

# Install MaxText specific dependencies conditionally
RUN if [ "$INSTALL_MAXTEXT" = "true" ]; then \
      uv pip install -r /app/requirements/maxtext_requirements.txt --torch-backend=cpu; \
    fi

# Build argument to conditionally install Raiden weight sync dependencies
ARG INSTALL_RAIDEN=false
ARG RAIDEN_WHEEL_DIR=/app/raiden_wheels

# Install Raiden specific dependencies conditionally
RUN if [ "$INSTALL_RAIDEN" = "true" ]; then \
    if [ -d "$RAIDEN_WHEEL_DIR" ] && ls "$RAIDEN_WHEEL_DIR"/*.whl 1>/dev/null 2>&1; then \
      pip install --force-reinstall --no-deps "$RAIDEN_WHEEL_DIR"/*.whl; \
    else \
      pip install keyrings.google-artifactregistry-auth && \
      pip install tpu-raiden-jax --extra-index-url https://us-python.pkg.dev/cloud-tpu-inference-test/tpu-raiden/simple/; \
    fi; \
fi

# Build argument to conditionally install DeepSWE evaluation dependencies
ARG INSTALL_DEEPSWE_DEPS=false

# Install DeepSWE specific dependencies and apply runtime patches conditionally
RUN if [ "$INSTALL_DEEPSWE_DEPS" = "true" ]; then \
      pip install kubernetes gym swebench==3.0.2 && \
      pip install --no-deps git+https://github.com/r2e-gym/r2e-gym.git@0d94c4eb9431cd195c55a7ea3abd54006c9a1735 && \
      sed -i 's/create_repo, upload_folder, HfFolder/create_repo, upload_folder/' /opt/venv/lib/python3.12/site-packages/r2egym/agenthub/utils/utils.py && \
      sed -i 's/self.commit = ParsedCommit(\*\*json.loads(self.commit_json))/self.commit = ParsedCommit(\*\*(json.loads(self.commit_json) if isinstance(self.commit_json, str) else self.commit_json))/' /opt/venv/lib/python3.12/site-packages/r2egym/agenthub/runtime/docker.py; \
    fi

# Patch tpu_inference to ensure random keys are scoped to CPU device on Pathways
RUN python3 /app/scripts/patch_tpu_runner.py

# Copy the project files to the image (cache buster ensures fresh code is always copied)
ARG CACHEBUST=1
COPY . .

# Install the project in editable mode
RUN pip install -e .

# Set the default command to bash
CMD ["bash"]
