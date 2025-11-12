# Start FROM a base image
FROM ubuntu:22.04

# Set environment variables to non-interactive to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Use the official Python 3.12 image based on Debian Stable (Bookworm)
FROM python:3.12-slim

# Python 3.12, pip, and venv are already included.
# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install other system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        openjdk-21-jdk \
        maven \
    && rm -rf /var/lib/apt/lists/*


# Create a virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install git+https://github.com/ayaka14732/jax-smi.git
RUN pip install git+https://github.com/AI-Hypercomputer/pathways-utils.git
RUN pip install gcsfs
RUN pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
RUN pip install --upgrade wandb

# Set a directory to clone sglang-jax into
WORKDIR /usr/src
# Clone the repository using HTTPS
RUN git clone https://github.com/sgl-project/sglang-jax.git
# Install the package in editable mode
# The -e flag means the installation links to the source code in /usr/src/sglang-jax
RUN cd sglang-jax/python && pip install --no-cache-dir -e .

# Reset the working directory to /app
WORKDIR /app

# Copy the project files to the image
COPY . .

# Install the project in editable mode
RUN pip install  --force-reinstall .

# Set the default command to bash
CMD ["bash"]