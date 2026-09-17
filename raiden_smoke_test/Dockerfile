# Base image with Python 3.12 following tunix/Dockerfile pattern
FROM python:3.12-slim

# Set environment variables to non-interactive to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies, including Python 3, pip, curl, git, and build tools
RUN apt-get update && \
    apt-get install -y build-essential curl git python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Create a virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install uv
RUN pip install --upgrade pip uv

# Install common utilities matching tunix/Dockerfile
RUN pip install git+https://github.com/ayaka14732/jax-smi.git gcsfs wandb

# 1. Install JAX / jaxlib / libtpu to 0.11.0 / 0.0.44 to match Raiden wheel
RUN pip install --no-cache-dir -U "jax[tpu]==0.11.0" "jaxlib==0.11.0" "libtpu==0.0.44" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

WORKDIR /app

# Pinned commits for vLLM and tpu-inference.
#
# 2026-09-14: VLLM_COMMIT was 96242aa50 here, but NO working image was ever
# built from it. The last good image (v9, which ran weight syncs fine) actually
# contained vllm 51da0ca66 -- 437 commits NEWER -- because whoever built it had
# VLLM_COMMIT exported in their shell, and this line is only a `${VAR:-default}`
# fallback. Rebuilding from a clean shell therefore silently downgraded vLLM and
# the rollout died with:
#     ImportError: cannot import name 'ACT' from
#                  vllm.model_executor.layers.quantization.modelopt
# because the local tpu-inference checkout (origin/main, 9bfc2520) imports ACT,
# WEIGHT, CkptCtx, KNvfp4Dynamic, KNvfp4Static, Shapes from modelopt, and none of
# those exist in 96242aa50.
#
# 51da0ca66 + tpu-inference 9bfc2520 is the empirically validated pair: it is
# what v9/v10 ran. Do not "tidy" this back to an older commit, and if you bump
# one of the two, bump the other and re-run the import check at the end of this
# file.
ARG VLLM_COMMIT=51da0ca66c8065619c79e35dff97aa99aeaf5644
ARG TPU_INFERENCE_COMMIT=0e68df7f2259b8468ef5e0391c8cc065e8fff76e

# 2. Copy tunix (including raiden_wheels) and install base requirements with tpu-inference@main and vllm@LKG
COPY tunix /app
RUN ln -sf /app/raiden_wheels /raiden_wheels && \
    (cd /app/tunix && ln -sf . tunix) && \
    uv pip install -e /app"[experimental]" && \
    echo "vllm @ git+https://github.com/vllm-project/vllm.git@${VLLM_COMMIT}" > /tmp/vllm_req.txt && \
    echo "tpu-inference @ git+https://github.com/vllm-project/tpu-inference.git@${TPU_INFERENCE_COMMIT}" > /tmp/tpu_inf_req.txt && \
    REQ_FILE=/tmp/vllm_req.txt SPECIAL_REQ_FILE=/tmp/tpu_inf_req.txt bash /app/scripts/install_tunix_vllm_requirement.sh && \
    rm -f /tmp/vllm_req.txt /tmp/tpu_inf_req.txt

# 3. Install MaxText workflow dependencies from maxtext_requirements.txt
# (filtering out git clone lines so our local maxtext directory is preserved)
RUN grep -v '^maxtext' /app/requirements/maxtext_requirements.txt | uv pip install -r /dev/stdin --torch-backend=cpu && \
    uv pip install torchax aqtp tokamax math_verify drjax

# Copy local maxtext and install in editable mode
COPY maxtext /app/maxtext
RUN pip install --no-deps -e /app/maxtext && \
    (pip install --no-deps -e /app/maxtext/src/maxtext/integration/vllm 2>/dev/null || true) && \
    mkdir -p /app/maxtext/maxtext && ln -sf /app/maxtext/src /app/maxtext/maxtext/src

# 4. Install local tunix in editable mode, compile protos, and install Raiden wheel
RUN pip install --no-deps -e /app && \
    pip install --no-deps /app/raiden_wheels/*.whl && \
    pip install "grpcio-tools==$(python -c 'import grpc; print(grpc.__version__)')" && \
    python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. tunix/experimental/distributed/runtime/discovery/discovery_service.proto 2>/dev/null || true && \
    cp -rf /app/tunix /opt/venv/lib/python3.12/site-packages/ && \
    cp -rf /app/maxtext/src/maxtext /opt/venv/lib/python3.12/site-packages/

# Copy local tpu-inference and overlay over site-packages
COPY tpu-inference /app/tpu-inference
RUN cp -rf /app/tpu-inference/tpu_inference /opt/venv/lib/python3.12/site-packages/

ENV PYTHONPATH="/app/maxtext/src:/app:${PYTHONPATH:-}"

# Fail the BUILD if the overlaid tpu-inference is incompatible with the vLLM we
# installed. Without this the mismatch only shows up ~25 min into a cluster run,
# as a rollout CrashLoopBackOff, after the TPUs are already allocated -- which is
# exactly how 2026-09-14 was lost. This import chain is the one the rollout node
# walks (tpu_platform.check_and_update_config -> layers.vllm -> quantization
# -> nvfp4 -> vllm ... modelopt), so if it resolves here it resolves there.
RUN python3 -c "\
import vllm.model_executor.layers.quantization.modelopt as m; \
missing = [n for n in ('ACT','WEIGHT','CkptCtx','KNvfp4Dynamic','KNvfp4Static','ModelOptNvFp4Config','ModelOptNvFp4FusedMoE','Shapes') if not hasattr(m, n)]; \
assert not missing, 'vLLM/tpu-inference MISMATCH: modelopt is missing %s -- bump VLLM_COMMIT' % missing; \
import tpu_inference.layers.vllm.quantization.nvfp4; \
import importlib.metadata as md; \
print('BUILD CHECK OK: vllm', md.version('vllm'), '<-> tpu-inference overlay')"

WORKDIR /app
CMD ["bash"]
