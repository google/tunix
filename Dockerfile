# syntax=docker/dockerfile:1
# ==============================================================================
# Tunix Container Image (`FROM python:3.12-slim`)
#
# Why we build from `python:3.12-slim` instead of `FROM vllm/vllm-tpu`:
#   Upstream `vllm/vllm-tpu` (`vllm-project/tpu-inference/docker/Dockerfile`)
#   starts from `python:3.12-slim-bookworm`, installs packages globally via
#   standard `pip` (without `/opt/venv` or `uv`), resolves `vllm`'s
#   `requirements/tpu.txt` + `lm-eval` + `tpu-inference`'s `requirements.txt`
#   and `requirements_benchmarking.txt` without `--no-deps`, and freezes a
#   single `vllm` + `tpu-inference` commit pair (`ENTRYPOINT ["/entrypoint.sh"]`)
#   oriented around standalone inference serving.
#
#   Building directly from `python:3.12-slim` with `uv` into `/opt/venv` gives
#   explicit control over layer ordering and declarative dependency resolution:
#     1. Isolates the heavy vLLM + TPU-Inference C++ compilation layer
#        (`install_tunix_vllm_requirement.sh`) BEFORE `COPY pyproject.toml`,
#        so routine edits to `pyproject.toml`, `MAXTEXT_REF`, or Tunix source
#        never invalidate the ~15-minute base compilation layer.
#     2. Consolidates all PyPI dependencies, git integration groups
#        (`[dependency-groups]`), and resolver overrides (`[tool.uv]`)
#        declaratively in `pyproject.toml`, installed in a fast (~15s) cached
#        wheel layer (`uv pip install -e "/app[tpu,cli,test,experimental]"`).
#     3. Splits MaxText into a cached PyPI wheel layer
#        (`uv pip install -e "/app[maxtext]"`) and a ~15-second `--no-deps`
#        git source layer (`--group maxtext-git`, or overridden via
#        `--build-arg MAXTEXT_REF=<ref>`).
#     4. Prevents transitive dependency clobbering automatically via
#        `[tool.uv] override-dependencies` in `pyproject.toml` (`flax>=0.12.5`,
#        `numpy==2.3.5`, `protobuf>=7.35.1`) and `ENV UV_TORCH_BACKEND=cpu`.
#     5. Zero model weights, datasets, or checkpoints are baked into the image
#        so `tunix/mlperf` stays generic and fast to pull (~13s container init);
#        CI workflows mount cached weights/datasets into `/root/.cache` at runtime.
# ==============================================================================

ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV VLLM_TARGET_DEVICE=tpu

# Install OS build dependencies (`zstd` for fast CI cache archiving) and
# initialize `/opt/venv` with `uv`.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      git \
      libnuma-dev \
      libomp-dev \
      libopenmpi-dev \
      ninja-build \
      python3 \
      python3-pip \
      python3-venv \
      zstd && \
    rm -rf /var/lib/apt/lists/* && \
    python3.12 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
# Pre-compile `.pyc` bytecode during `uv pip install` so container startups in
# CI and production do not re-parse thousands of `.py` files on first import,
# and default `uv` to CPU PyTorch wheels so installing packages that depend on
# `torch` never pulls 4+ GB of NVIDIA CUDA wheels into a TPU container.
ENV UV_COMPILE_BYTECODE=1
ENV UV_TORCH_BACKEND=cpu

RUN pip install --upgrade pip uv

WORKDIR /app

# ---------------------------------------------------------------------------
# 1. Base vLLM + TPU-Inference Compilation Layer (~15 min cold, cached in GAR)
# ---------------------------------------------------------------------------
# Copy ONLY `install_tunix_vllm_requirement.sh` before `pyproject.toml` so
# routine edits to `pyproject.toml` or Tunix source never invalidate this layer.
ARG VLLM_COMMIT=""
ARG TPU_INFERENCE_COMMIT=""
COPY scripts/install_tunix_vllm_requirement.sh /app/scripts/install_tunix_vllm_requirement.sh
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    VLLM_COMMIT="${VLLM_COMMIT}" \
    TPU_INFERENCE_COMMIT="${TPU_INFERENCE_COMMIT}" \
    bash /app/scripts/install_tunix_vllm_requirement.sh

# ---------------------------------------------------------------------------
# 2. Declarative Tunix (`pyproject.toml`) Runtime & Test Dependencies Layer
# ---------------------------------------------------------------------------
COPY pyproject.toml README.md /app/
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    mkdir -p /app/tunix && touch /app/tunix/__init__.py && \
    uv pip install -e "/app[tpu,cli,test,experimental]"

# ---------------------------------------------------------------------------
# 3. Optional Kubernetes CLI Tools (`INSTALL_K8S_TOOLS=true`)
# ---------------------------------------------------------------------------
ARG INSTALL_K8S_TOOLS=false
COPY scripts/install_k8s_tools.sh /app/scripts/install_k8s_tools.sh
RUN if [ "$INSTALL_K8S_TOOLS" = "true" ]; then \
      bash /app/scripts/install_k8s_tools.sh; \
    fi

# ---------------------------------------------------------------------------
# 4. Optional MaxText + Adapter Split-Layer Caching (`INSTALL_MAXTEXT=true`)
# ---------------------------------------------------------------------------
ARG INSTALL_MAXTEXT=false

# 4a. Cached third-party MaxText wheels (`pyproject.toml` `[maxtext]`, not
# invalidated when `--build-arg MAXTEXT_REF=<ref>` is overridden).
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_MAXTEXT" = "true" ]; then \
      uv pip install -e "/app[maxtext]"; \
    fi

# 4b. Install MaxText + adapter source (`~15s` rebuild).
# By default (`MAXTEXT_REF=""`), installs the pinned commit from `pyproject.toml`
# (`--group maxtext-git`) so the commit SHA lives in a single place. Pass
# `--build-arg MAXTEXT_REF=<commit-or-branch>` to test a custom MaxText ref.
ARG MAXTEXT_REF=""
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_MAXTEXT" = "true" ]; then \
      if [ -n "$MAXTEXT_REF" ]; then \
        uv pip install --no-deps --reinstall --no-cache \
          "maxtext @ git+https://github.com/AI-Hypercomputer/maxtext.git@${MAXTEXT_REF}" \
          "maxtext-vllm-adapter @ git+https://github.com/AI-Hypercomputer/maxtext.git@${MAXTEXT_REF}#subdirectory=src/maxtext/integration/vllm"; \
      else \
        uv pip install --directory /app --no-deps --reinstall --no-cache --group maxtext-git; \
      fi; \
    fi

# ---------------------------------------------------------------------------
# 5. Copy Tunix Repository & Run Recipe/Component Installers
# ---------------------------------------------------------------------------
COPY . /app

# Optional DeepSWE recipe dependencies (`examples/deepswe/install_deepswe.sh`).
ARG INSTALL_DEEPSWE_DEPS=false
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_DEEPSWE_DEPS" = "true" ]; then \
      bash /app/examples/deepswe/install_deepswe.sh; \
    fi

# Optional Raiden (`tpu_sync_jax`) installation (`scripts/install_raiden.sh`).
# Uses `./raiden_wheels/*.whl` if present in the build context or fetches
# `RAIDEN_WHEEL_URL` via GCE metadata server / `gcloud`.
ARG INSTALL_RAIDEN=false
ARG RAIDEN_WHEEL_DIR=/app/raiden_wheels
ARG RAIDEN_WHEEL_URL=""
ARG RAIDEN_WHEEL_SHA256=""
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_RAIDEN" = "true" ]; then \
      RAIDEN_WHEEL_DIR="${RAIDEN_WHEEL_DIR}" \
      RAIDEN_WHEEL_URL="${RAIDEN_WHEEL_URL}" \
      RAIDEN_WHEEL_SHA256="${RAIDEN_WHEEL_SHA256}" \
      bash /app/scripts/install_raiden.sh; \
    fi

# Compile the explicit gRPC protobuf definition for distributed orchestrator
# discovery and install Tunix in editable mode (`--no-deps`).
RUN test -f /app/tunix/experimental/distributed/runtime/discovery/discovery_service.proto && \
    python3 -m grpc_tools.protoc -I/app --python_out=/app --grpc_python_out=/app \
      /app/tunix/experimental/distributed/runtime/discovery/discovery_service.proto && \
    uv pip install --no-deps -e /app

# Build-time environment verification (`scripts/verify_environment.py`).
# When `INSTALL_MAXTEXT=true`, verifies not only top-level `maxtext` (whose
# `__init__.py` is minimal) but also `maxtext.checkpoint_conversion.to_maxtext`
# and `maxtext_vllm_adapter` so any missing third-party dependency in `[maxtext]`
# fails immediately during `docker build`.
RUN JAX_PLATFORMS=cpu python3 /app/scripts/verify_environment.py \
      --check-tunix-version \
      --packages jax jaxlib flax optax orbax.checkpoint qwix datasets vllm tpu_inference tunix \
      tunix.experimental.distributed.runtime.discovery.discovery_service_pb2 \
      $(if [ "$INSTALL_MAXTEXT" = "true" ]; then echo "maxtext maxtext.checkpoint_conversion.to_maxtext maxtext_vllm_adapter"; fi) \
      $(if [ "$INSTALL_RAIDEN" = "true" ]; then echo "tpu_sync"; fi)

CMD ["bash"]
