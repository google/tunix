# syntax=docker/dockerfile:1
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG BASE_IMAGE=python:3.12-slim

# -----------------------------------------------------------------------------
# Stage 1: Split `pyproject.toml` into isolated layer manifests so that:
#   1. Editing PyPI extras or `maxtext-git` does NOT invalidate the expensive
#      `vllm-tpu` + `tpu-inference` layer.
#   2. Editing `maxtext-git` does NOT invalidate the PyPI wheel layers
#      (`.[dev,cli]` and `.[maxtext]`).
# -----------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS manifest-extractor
COPY pyproject.toml /tmp/pyproject.toml
RUN python3 - <<'PY'
import pathlib, re, tomllib

src = pathlib.Path("/tmp/pyproject.toml").read_text()
data = tomllib.loads(src)
dg = data.get("dependency-groups", {})
uv_overrides = data.get("tool", {}).get("uv", {}).get("override-dependencies", [])

vllm_lines = ["[dependency-groups]"]
for group in ("vllm-tpu", "tpu-inference"):
  vllm_lines.append(f"{group} = {dg.get(group, [])!r}")
vllm_lines.append("[tool.uv]")
vllm_lines.append(f"override-dependencies = {uv_overrides!r}")
vllm_dir = pathlib.Path("/tmp/vllm_groups")
vllm_dir.mkdir(parents=True, exist_ok=True)
(vllm_dir / "pyproject.toml").write_text("\n".join(vllm_lines) + "\n")

pypi_only = re.sub(r"(?ms)^\[dependency-groups\].*?(?=^\[|\Z)", "", src)
pypi_dir = pathlib.Path("/tmp/pypi_only")
pypi_dir.mkdir(parents=True, exist_ok=True)
(pypi_dir / "pyproject.toml").write_text(pypi_only)
PY

# -----------------------------------------------------------------------------
# Stage 2: Tunix Runtime & CI Image (`tunix/mlperf`)
# -----------------------------------------------------------------------------
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      g++ \
      gcc \
      git \
      libgl1 \
      libglib2.0-0 \
      ninja-build \
      pkg-config \
      procps \
      psmisc \
      zstd && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.2 /uv /uvx /bin/

RUN uv venv --python python3.12 --seed /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PIP_NO_BUILD_ISOLATION=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=120

WORKDIR /app

# Layer 1: Pinned `vllm-tpu` and `tpu-inference` commits (`pyproject.toml`).
# Cached unless `vllm-tpu`, `tpu-inference`, `override-dependencies`, or
# `scripts/install_tunix_vllm_requirement.sh` changes.
COPY scripts/install_tunix_vllm_requirement.sh /app/scripts/install_tunix_vllm_requirement.sh
COPY --from=manifest-extractor /tmp/vllm_groups/pyproject.toml /app/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    bash /app/scripts/install_tunix_vllm_requirement.sh

# Layer 2: Tunix core, `[dev]`, and `[cli]` PyPI dependencies.
# Cached across Tunix Python code edits and `maxtext-git` commit bumps.
COPY --from=manifest-extractor /tmp/pypi_only/pyproject.toml /app/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    mkdir -p /app/tunix && \
    printf '__version__ = "0.0.0"\n' > /app/tunix/__init__.py && \
    printf '# Tunix\n' > /app/README.md && \
    uv pip install --torch-backend=cpu ".[dev,cli]" \
      "git+https://github.com/ayaka14732/jax-smi.git" \
      math_verify && \
    rm -rf /app/tunix /app/README.md

# Layer 3a: MaxText PyPI wheel dependencies (`[maxtext]`).
# Cached across MaxText source commit bumps.
ARG INSTALL_MAXTEXT=false
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "${INSTALL_MAXTEXT}" = "true" ]; then \
      mkdir -p /app/tunix && \
      printf '__version__ = "0.0.0"\n' > /app/tunix/__init__.py && \
      printf '# Tunix\n' > /app/README.md && \
      uv pip install --torch-backend=cpu ".[maxtext]" && \
      rm -rf /app/tunix /app/README.md; \
    fi

# Layer 3b: MaxText source package (`maxtext-git` or custom `MAXTEXT_REF`).
# Installed with `--no-deps` so transitive metadata cannot downgrade
# `protobuf>=7.35.1` or `flax>=0.12.5`.
COPY pyproject.toml /app/pyproject.toml
ARG MAXTEXT_REF=""
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "${INSTALL_MAXTEXT}" = "true" ]; then \
      if [ -n "${MAXTEXT_REF}" ]; then \
        uv pip install --no-deps --reinstall \
          "maxtext @ git+https://github.com/AI-Hypercomputer/maxtext.git@${MAXTEXT_REF}" \
          "maxtext-vllm-adapter @ git+https://github.com/AI-Hypercomputer/maxtext.git@${MAXTEXT_REF}#subdirectory=src/maxtext/integration/vllm" \
          "protobuf>=7.35.1"; \
      else \
        uv pip install --no-deps --reinstall --group maxtext-git; \
      fi; \
    fi

# Layer 4: Raiden TPU weight-sync library (`tpu_sync_jax`).
# Cached across Tunix source edits. Uses a local wheel from `raiden_wheels/`
# when present in the build context, and otherwise fetches the wheel pinned
# in `scripts/install_raiden.sh`.
ARG INSTALL_RAIDEN=true
ARG RAIDEN_WHEEL_DIR=/app/raiden_wheels
COPY scripts/install_raiden.sh /app/scripts/install_raiden.sh
COPY scripts/install_raiden.sh raiden_wheel[s]/*.whl ${RAIDEN_WHEEL_DIR}/
RUN if [ "${INSTALL_RAIDEN}" = "true" ]; then \
      RAIDEN_WHEEL_DIR="${RAIDEN_WHEEL_DIR}" bash /app/scripts/install_raiden.sh && \
      rm -rf "${RAIDEN_WHEEL_DIR}"; \
    fi

# Build argument to conditionally install Kubernetes tools
ARG INSTALL_K8S_TOOLS=false
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
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$INSTALL_DEEPSWE_DEPS" = "true" ]; then \
      uv pip install kubernetes gym swebench==3.0.2 && \
      uv pip install --no-deps git+https://github.com/kubernetes-sigs/agent-sandbox.git#subdirectory=clients/python/agentic-sandbox-client && \
      uv pip install --no-deps git+https://github.com/kubernetes-sigs/agent-sandbox.git#subdirectory=examples/agent-sandbox-rl && \
      uv pip install --no-deps git+https://github.com/r2e-gym/r2e-gym.git@0d94c4eb9431cd195c55a7ea3abd54006c9a1735 && \
      sed -i 's/create_repo, upload_folder, HfFolder/create_repo, upload_folder/' /opt/venv/lib/python3.12/site-packages/r2egym/agenthub/utils/utils.py && \
      sed -i 's/self.commit = ParsedCommit(\*\*json.loads(self.commit_json))/self.commit = ParsedCommit(\*\*(json.loads(self.commit_json) if isinstance(self.commit_json, str) else self.commit_json))/' /opt/venv/lib/python3.12/site-packages/r2egym/agenthub/runtime/docker.py; \
    fi

# Copy the rest of the project files, compile gRPC protos, and install Tunix.
COPY . /app

RUN cd /app && \
    rm -rf /app/raiden_wheels && \
    find tunix/experimental/distributed -name "*.proto" -exec \
      python3 -m grpc_tools.protoc -I/app --python_out=/app --grpc_python_out=/app {} + && \
    uv pip install --no-deps -e . && \
    uv pip install --no-deps "numpy==2.3.5"

# Build-time smoke check: fail fast if core packages, version pins, or
# compiled protobuf modules are broken before publishing to Artifact Registry.
RUN JAX_PLATFORMS=cpu TPU_SKIP_MDS_QUERY=1 INSTALL_MAXTEXT="${INSTALL_MAXTEXT}" INSTALL_RAIDEN="${INSTALL_RAIDEN}" python3 - <<'PY'
import importlib, os, numpy, google.protobuf, tunix

for mod in (
    "jax", "jaxlib", "flax", "optax", "orbax.checkpoint", "qwix",
    "datasets", "vllm", "tpu_inference", "tunix",
    "tunix.experimental.distributed.runtime.discovery.discovery_service_pb2",
):
  importlib.import_module(mod)

assert numpy.__version__ == "2.3.5", f"Expected numpy==2.3.5, got {numpy.__version__}"
assert tunix.__version__ != "0.0.0.dev0", "Tunix package metadata not set"

if os.environ.get("INSTALL_MAXTEXT") == "true":
  for mod in ("maxtext", "maxtext.checkpoint_conversion.to_maxtext", "maxtext_vllm_adapter"):
    importlib.import_module(mod)
  pb_major = int(google.protobuf.__version__.split(".")[0])
  assert pb_major >= 6, f"MaxText protos require protobuf>=6, got {google.protobuf.__version__}"

if os.environ.get("INSTALL_RAIDEN") == "true":
  importlib.import_module("tpu_sync")

print("Verified Tunix container environment:", tunix.__version__)
PY

CMD ["bash"]
