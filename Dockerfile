# Base image with Python 3.12
FROM python:3.12-slim

# Set environment variables to non-interactive to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies, including Python 3 and pip
RUN apt-get update && \
    apt-get install -y build-essential git curl python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Create a virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:/root/.local/bin:/root/.cargo/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

RUN pip install git+https://github.com/ayaka14732/jax-smi.git
# If you encounter a checkpoint issue, try using following old version of pathways-utils.
# RUN pip install git+https://github.com/AI-Hypercomputer/pathways-utils.git@b72729bb152b7b3426299405950b3af300d765a9#egg=pathwaysutils
RUN pip install gcsfs
RUN pip install wandb
RUN pip install kubernetes

# Copy the project files to a temp directory
COPY . /app/g3_sources

# Restructure to match OSS layout
RUN mkdir -p /app/tunix && \
    cp -a /app/g3_sources/* /app/tunix/ && \
    mv /app/tunix/oss/pyproject.toml /app/pyproject.toml && \
    mv /app/tunix/oss/requirements /app/requirements && \
    mv /app/tunix/oss/scripts /app/scripts && \
    mv /app/tunix/oss/examples /app/examples && \
    mv /app/tunix/tests /app/tests && \
    rm -rf /app/tunix/pyproject.toml /app/tunix/requirements /app/tunix/scripts /app/tunix/examples && \
    rm -rf /app/g3_sources

# Set the working directory
WORKDIR /app

# Apply OSS replacements to avoid internal imports on GCP
RUN python3 /app/scripts/apply_oss_replacements.py /app/tunix

# Install the project in editable mode
RUN pip install -e .

RUN bash /app/scripts/install_tunix_vllm_requirement.sh
RUN pip install gym swebench==3.0.2
RUN pip install --no-deps git+https://github.com/r2e-gym/r2e-gym.git@0d94c4eb9431cd195c55a7ea3abd54006c9a1735
RUN pip install --upgrade flax
RUN pip install aqtp tokamax
RUN pip install git+https://github.com/google/maxtext.git

# Patch r2egym bugs
RUN sed -i 's/create_repo, upload_folder, HfFolder/create_repo, upload_folder/' /opt/venv/lib/python3.12/site-packages/r2egym/agenthub/utils/utils.py
RUN sed -i 's/self.commit = ParsedCommit(\*\*json.loads(self.commit_json))/self.commit = ParsedCommit(\*\*(json.loads(self.commit_json) if isinstance(self.commit_json, str) else self.commit_json))/' /opt/venv/lib/python3.12/site-packages/r2egym/agenthub/runtime/docker.py
RUN sed -i 's/"karpenter.sh\/nodepool": "bigcpu-standby"/"cloud.google.com\/gke-nodepool": "cpu-np"/' /opt/venv/lib/python3.12/site-packages/r2egym/agenthub/runtime/docker.py


# Set the default command to bash
CMD ["bash"]
