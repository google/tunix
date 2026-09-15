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

# MaxText enters this image from a git ref in two places, and both were pinned
# to a FLOATING branch inside a static RUN string. Docker's cache key for that
# layer therefore never changed while the branch moved underneath it, so two
# builds weeks apart could contain different MaxText -- or the same stale one --
# with nothing in the image recording which. Making the ref a build argument
# fixes both halves: the cache key now moves with the ref, and the resolved ref
# is written into the image. MLPerf requires an immutable commit hash rather
# than a branch name regardless (see the submission README), so prefer a SHA.
ARG MAXTEXT_REPO=https://github.com/AI-Hypercomputer/maxtext.git
ARG MAXTEXT_REF=niting-rl_mlperf_temp

# Install SFT/MaxText dependencies (unconditional)
RUN uv pip install --upgrade flax && \
    uv pip install torchax aqtp tokamax math_verify drjax && \
    printf 'MAXTEXT_REPO=%s\nMAXTEXT_REF=%s\n' "${MAXTEXT_REPO}" "${MAXTEXT_REF}" > /app/.maxtext_ref && \
    uv pip install --no-deps "maxtext @ git+${MAXTEXT_REPO}@${MAXTEXT_REF}"

# Build argument to conditionally install MaxText dependencies
ARG INSTALL_MAXTEXT=false

# Install MaxText specific dependencies conditionally. The requirements file
# still carries the historical default pin; rewrite it to the build arg so the
# adapter and maxtext can never come from two different refs.
RUN if [ "$INSTALL_MAXTEXT" = "true" ]; then \
      sed -e "s|https://github.com/AI-Hypercomputer/maxtext.git@niting-rl_mlperf_temp|${MAXTEXT_REPO}@${MAXTEXT_REF}|g" \
          /app/requirements/maxtext_requirements.txt > /tmp/maxtext_requirements.txt && \
      echo "--- resolved maxtext requirements ---" && head -2 /tmp/maxtext_requirements.txt && \
      uv pip install -r /tmp/maxtext_requirements.txt --torch-backend=cpu; \
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
      uv pip install "numpy<2.3" kubernetes gym swebench==3.0.2 && \
      uv pip install --no-deps git+https://github.com/kubernetes-sigs/agent-sandbox.git#subdirectory=clients/python/agentic-sandbox-client && \
      uv pip install --no-deps git+https://github.com/kubernetes-sigs/agent-sandbox.git#subdirectory=examples/agent-sandbox-rl && \
      uv pip install --no-deps git+https://github.com/r2e-gym/r2e-gym.git@0d94c4eb9431cd195c55a7ea3abd54006c9a1735 && \
      sed -i 's/create_repo, upload_folder, HfFolder/create_repo, upload_folder/' /opt/venv/lib/python3.12/site-packages/r2egym/agenthub/utils/utils.py && \
      sed -i 's/self.commit = ParsedCommit(\*\*json.loads(self.commit_json))/self.commit = ParsedCommit(\*\*(json.loads(self.commit_json) if isinstance(self.commit_json, str) else self.commit_json))/' /opt/venv/lib/python3.12/site-packages/r2egym/agenthub/runtime/docker.py && \
      sed -i "s/e.body = e.body.decode('utf-8') if six.PY3 else e.body/e.body = (e.body.decode('utf-8') if hasattr(e.body, 'decode') else str(e.body)) if six.PY3 else e.body/" /opt/venv/lib/python3.12/site-packages/kubernetes/client/api_client.py && \
      python3 -c "p='/opt/venv/lib/python3.12/site-packages/agent_sandbox_rl/resources.py'; c=open(p).read(); t='\"resources\": {\"requests\": {\n                \"cpu\": template.resources.cpu,\n                \"memory\": template.resources.memory,\n            }},'; open(p,'w').write(c.replace(t, '\"resources\": {},'))"; \
      uv pip install "numpy<2.3"; \
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

# Verify critical runtime dependencies
RUN python3 -c "import numpy, numba; assert numpy.__version__ < '2.3', f'NumPy {numpy.__version__} too new for Numba'; print('Verified NumPy + Numba OK:', numpy.__version__)"

# Set the default command to bash
CMD ["bash"]
