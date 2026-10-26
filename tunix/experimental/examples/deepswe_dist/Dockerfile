# Experimental DeepSWE distributed RL Dockerfile with pinned JAX/Flax/NumPy
# runtime constraints and optional MaxText / tpu-inference commit overrides.
# Build from any working directory via:
#   bash tunix/experimental/examples/deepswe_dist/build_docker.sh ...
# TODO(tunix-dev): [legacy-maxtext-a2cf3bc8b] Retire this experimental
# Dockerfile once main's root Dockerfile upgrades its dependency stack and
# commit overrides (--maxtext-commit / --tpu-inference-commit) are no longer
# required for the 35B DeepSWE recipe.

# Base image with Python 3.12
FROM python:3.12-slim

# Set environment variables to non-interactive to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies, including Python 3 and pip
RUN apt-get update && \
    apt-get install -y build-essential cmake curl git ninja-build python3 python3-pip && \
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

# Copy pyproject.toml, README.md, and CMakeLists.txt to install dependencies first
COPY pyproject.toml README.md CMakeLists.txt /app/
RUN mkdir /app/tunix && touch /app/tunix/__init__.py
RUN uv pip install .

# Single source of truth for runtime pins enforced across all subsequent layers.
# Note: flax 0.12.7 matches the validated TPU RL runtime environment.
ARG NUMPY_VERSION=2.3.5
ARG JAX_VERSION=0.11.0
ARG FLAX_VERSION=0.12.7
# Keep in sync with requirements/maxtext_requirements.txt.
ARG DEFAULT_MAXTEXT_COMMIT=bc35b839da747ffbce998eb8da5354e5367f7d66

# Install SFT/MaxText dependencies (unconditional) in one constrained resolution
# so unpinned packages (e.g. tokamax) backtrack to versions compatible with the
# pinned JAX/Flax/NumPy runtime rather than upgrading JAX.
RUN printf 'numpy==%s\njax==%s\njaxlib==%s\nflax==%s\n' \
      "${NUMPY_VERSION}" "${JAX_VERSION}" "${JAX_VERSION}" "${FLAX_VERSION}" \
      > /opt/venv/constraints.txt && \
    uv pip install -c /opt/venv/constraints.txt \
      "numpy==${NUMPY_VERSION}" \
      "jax==${JAX_VERSION}" \
      "jaxlib==${JAX_VERSION}" \
      "flax==${FLAX_VERSION}" \
      torchax aqtp tokamax math_verify drjax && \
    uv pip install --no-deps \
      "git+https://github.com/AI-Hypercomputer/maxtext.git@${DEFAULT_MAXTEXT_COMMIT}"

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

# Build argument to conditionally install DeepSWE evaluation dependencies
ARG INSTALL_DEEPSWE_DEPS=false

# Install DeepSWE specific dependencies and apply runtime patches conditionally
RUN if [ "$INSTALL_DEEPSWE_DEPS" = "true" ]; then \
      uv pip install -c /opt/venv/constraints.txt kubernetes gym swebench==3.0.2 && \
      uv pip install --no-deps git+https://github.com/kubernetes-sigs/agent-sandbox.git@9afee7039342f18fa9ad402929cabf8b66eebaa7#subdirectory=clients/python/agentic-sandbox-client && \
      uv pip install --no-deps git+https://github.com/kubernetes-sigs/agent-sandbox.git@9afee7039342f18fa9ad402929cabf8b66eebaa7#subdirectory=examples/agent-sandbox-rl && \
      uv pip install --no-deps git+https://github.com/r2e-gym/r2e-gym.git@0d94c4eb9431cd195c55a7ea3abd54006c9a1735 && \
      sed -i 's/create_repo, upload_folder, HfFolder/create_repo, upload_folder/' /opt/venv/lib/python3.12/site-packages/r2egym/agenthub/utils/utils.py && \
      sed -i 's/self.commit = ParsedCommit(\*\*json.loads(self.commit_json))/self.commit = ParsedCommit(\*\*(json.loads(self.commit_json) if isinstance(self.commit_json, str) else self.commit_json))/' /opt/venv/lib/python3.12/site-packages/r2egym/agenthub/runtime/docker.py; \
    fi

# Build arguments to conditionally install MaxText dependencies and/or override
# maxtext and tpu-inference commits (--no-deps; commits must be compatible with
# the pinned vLLM/JAX/libtpu stack). Note: MAXTEXT_COMMIT also installs
# maxtext-vllm-adapter (a `vllm.general_plugins` entry point active on every
# vLLM start) even when INSTALL_MAXTEXT=false.
ARG INSTALL_MAXTEXT=false
ARG MAXTEXT_COMMIT=""
ARG TPU_INFERENCE_COMMIT=""

# Install MaxText specific dependencies and optional commit overrides
RUN if [ "$INSTALL_MAXTEXT" = "true" ]; then \
      uv pip install -c /opt/venv/constraints.txt -r /app/requirements/maxtext_requirements.txt --torch-backend=cpu; \
    fi && \
    if [ -n "$MAXTEXT_COMMIT" ]; then \
      uv pip install --no-deps --force-reinstall \
        "git+https://github.com/AI-Hypercomputer/maxtext.git@${MAXTEXT_COMMIT}" \
        "git+https://github.com/AI-Hypercomputer/maxtext.git@${MAXTEXT_COMMIT}#subdirectory=src/maxtext/integration/vllm"; \
    fi && \
    if [ -n "$TPU_INFERENCE_COMMIT" ]; then \
      uv pip install --no-deps --force-reinstall \
        "git+https://github.com/vllm-project/tpu-inference.git@${TPU_INFERENCE_COMMIT}"; \
    fi

# Build argument to conditionally install Raiden weight sync dependencies
ARG INSTALL_RAIDEN=false
ARG RAIDEN_WHEEL_DIR=/app/raiden_wheels

# Install Raiden specific dependencies conditionally. Off by default, so a
# plain build never needs access to the wheel. The installer prefers a locally
# built wheel from ${RAIDEN_WHEEL_DIR} (see scripts/build_raiden_wheel.sh) and
# otherwise fetches the wheel pinned in scripts/install_raiden.sh.
COPY scripts/install_raiden.sh /app/scripts/
COPY raiden_wheels/ ${RAIDEN_WHEEL_DIR}/
RUN if [ "$INSTALL_RAIDEN" = "true" ]; then \
      RAIDEN_WHEEL_DIR="$RAIDEN_WHEEL_DIR" bash /app/scripts/install_raiden.sh; \
    fi

# Re-apply runtime pins and assert numpy, jax, jaxlib, and flax versions.
RUN uv pip install -c /opt/venv/constraints.txt \
      "numpy==${NUMPY_VERSION}" \
      "jax==${JAX_VERSION}" \
      "jaxlib==${JAX_VERSION}" \
      "flax==${FLAX_VERSION}" && \
    NUMPY_VERSION="${NUMPY_VERSION}" JAX_VERSION="${JAX_VERSION}" FLAX_VERSION="${FLAX_VERSION}" \
    python -c 'import os, flax, jax, jaxlib, numpy; exp = (os.environ["NUMPY_VERSION"], os.environ["JAX_VERSION"], os.environ["JAX_VERSION"], os.environ["FLAX_VERSION"]); got = (numpy.__version__, jax.__version__, jaxlib.__version__, flax.__version__); assert got == exp, (exp, got)'

# Copy the rest of the project files
COPY . .

# Compile proto buffer for discovery service, this has to be the last step.
RUN uv pip install grpcio-tools
RUN cd /app && find tunix/dist_rl/distributed -name "*.proto" -exec python -m grpc_tools.protoc -I/app --python_out=/app --grpc_python_out=/app {} +

# Install Tunix in editable mode
RUN uv pip install --no-deps -e .

# Set the default command to bash
CMD ["bash"]
