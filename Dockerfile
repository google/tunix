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

# Copy scripts and requirements first to leverage Docker cache
COPY scripts/install_tunix_vllm_requirement.sh scripts/
COPY requirements/ requirements/

RUN bash scripts/install_tunix_vllm_requirement.sh

# Copy pyproject.toml and README.md to install dependencies first
COPY pyproject.toml README.md /app/
RUN mkdir /app/tunix && touch /app/tunix/__init__.py
RUN uv pip install .

# Install SFT/MaxText dependencies (unconditional)
RUN uv pip install --upgrade flax && \
    uv pip install torchax aqtp tokamax math_verify drjax && \
    uv pip install --no-deps git+https://github.com/google/maxtext.git

# Build argument to conditionally install MaxText dependencies
ARG INSTALL_MAXTEXT=false

# Install MaxText specific dependencies conditionally
RUN if [ "$INSTALL_MAXTEXT" = "true" ]; then \
      uv pip install -r /app/requirements/maxtext_requirements.txt --torch-backend=cpu; \
    else \
      uv pip install numpy==2.3.5; \
fi

# Build argument to conditionally install Raiden weight sync dependencies
ARG INSTALL_RAIDEN=false
ARG RAIDEN_WHEEL_DIR=/app/raiden_wheels

# Install Raiden specific dependencies conditionally
COPY raiden_wheels/ ${RAIDEN_WHEEL_DIR}/
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
      uv pip install kubernetes gym swebench==3.0.2 && \
      uv pip install --no-deps git+https://github.com/kubernetes-sigs/agent-sandbox.git#subdirectory=clients/python/agentic-sandbox-client && \
      uv pip install --no-deps git+https://github.com/kubernetes-sigs/agent-sandbox.git#subdirectory=examples/agent-sandbox-rl && \
      uv pip install --no-deps git+https://github.com/r2e-gym/r2e-gym.git@0d94c4eb9431cd195c55a7ea3abd54006c9a1735 && \
      sed -i 's/create_repo, upload_folder, HfFolder/create_repo, upload_folder/' /opt/venv/lib/python3.12/site-packages/r2egym/agenthub/utils/utils.py && \
      sed -i 's/self.commit = ParsedCommit(\*\*json.loads(self.commit_json))/self.commit = ParsedCommit(\*\*(json.loads(self.commit_json) if isinstance(self.commit_json, str) else self.commit_json))/' /opt/venv/lib/python3.12/site-packages/r2egym/agenthub/runtime/docker.py; \
    fi

# Build argument to conditionally install Kubernetes tools
ARG INSTALL_K8S_TOOLS=false

# Install gcloud, kubectl, k9s
RUN if [ "$INSTALL_K8S_TOOLS" = "true" ]; then \
      apt-get update && \
      apt-get install -y vim lsof procps apt-transport-https ca-certificates gnupg && \
      (echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list) && \
      (curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --batch --yes --no-tty --dearmor -o /usr/share/keyrings/cloud.google.gpg) && \
      apt-get update && apt-get install -y google-cloud-cli google-cloud-cli-gke-gcloud-auth-plugin kubectl && \
      (curl -sS https://webinstall.dev/k9s | bash) && \
      rm -rf /var/lib/apt/lists/*; \
    fi

# Copy the rest of the project files
COPY . .

# Compile proto buffer for discovery service, this has to be the last step.
RUN uv pip install grpcio-tools
RUN cd /app && find tunix/experimental/distributed -name "*.proto" -exec python -m grpc_tools.protoc -I/app --python_out=/app --grpc_python_out=/app {} +

# Install Tunix in editable mode
RUN uv pip install --no-deps -e .



# Set the default command to bash
CMD ["bash"]
