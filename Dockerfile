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

RUN pip install git+https://github.com/ayaka14732/jax-smi.git
RUN pip install gcsfs
RUN pip install wandb
RUN pip install aqtp==0.9.0 tokamax==0.0.12 git+https://github.com/AI-Hypercomputer/maxtext.git@bc72cc7a9455a5dfa5143fc71a67a31c186954e7

# Install vllm/tpu dependencies
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install vllm==0.8.3.post1 torchax==0.0.4 tpu-inference==0.0.4.dev20250428+jax -f https://storage.googleapis.com/libtpu-releases/index.html -f https://storage.googleapis.com/libtpu-wheels/index.html

# Install DeepSWE specific dependencies
RUN pip install kubernetes gym swebench==3.0.2 && \
    pip install --no-deps git+https://github.com/r2e-gym/r2e-gym.git@0d94c4eb9431cd195c55a7ea3abd54006c9a1735 && \
    sed -i "s/create_repo, upload_folder, HfFolder/create_repo, upload_folder/" /opt/venv/lib/python3.12/site-packages/r2egym/agenthub/utils/utils.py && \
    sed -i "s/self.commit = ParsedCommit(\*\*json.loads(self.commit_json))/self.commit = ParsedCommit(\*\*(json.loads(self.commit_json) if isinstance(self.commit_json, str) else self.commit_json))/" /opt/venv/lib/python3.12/site-packages/r2egym/agenthub/runtime/docker.py

# Set the working directory
WORKDIR /app

# Cache buster to ensure local code changes are always copied
ARG CACHEBUST=1

# Copy the project files to the image
COPY . .

# Install the project in editable mode
RUN pip install -e .

# Set the default command to bash
CMD ["bash"]
