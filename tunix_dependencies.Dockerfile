FROM python:3.11-slim

WORKDIR /app

ARG INSTALL_VLLM=false

# Install git and upgrade pip
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    pip install --no-cache-dir --upgrade pip && \
    rm -rf /var/lib/apt/lists/*

# Install base python dependencies
RUN pip install --no-cache-dir \
    jax \
    jaxtyping \
    sentencepiece \
    tensorboardX \
    tqdm \
    grain \
    kagglehub \
    tensorflow \
    datasets \
    git+https://github.com/google/qwix

# Uninstall default flax and install from git
RUN pip uninstall -y flax && \
    pip install --no-cache-dir git+https://github.com/google/flax.git

# Conditionally install vLLM using the provided script
COPY install_vllm.sh .
RUN if [ "$INSTALL_VLLM" = "true" ] ; then chmod +x install_vllm.sh && ./install_vllm.sh ; fi

# Copy the project files
COPY . .

# Install the project itself
RUN pip install .
