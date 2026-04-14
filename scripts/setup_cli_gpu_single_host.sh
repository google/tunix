#!/bin/bash
# Stop the script if any command fails
set -e

echo "=== Starting Tunix GPU Environment Setup ==="

# 1. Create a project specific environment
echo "Creating virtual environment in .venv..."
python3 -m venv .venv

# 2. Activate the Environment
echo "Activating virtual environment..."
source .venv/bin/activate

# 3. Install Tunix dependency
echo "Upgrading pip to the latest version..."
pip install --upgrade pip

echo "Installing Tunix core dependencies..."
# Install editable core without hardware acceleration first
pip install -e .

echo "Installing JAX for GPU..."
# Installs GPU support. Note: You may need to specify a CUDA version
# explicitly (e.g., jax[cuda12]) if this default does not match your driver.
pip install jax[gpu]

echo "Installing gcsfs..."
pip install gcsfs

echo "=== GPU Setup Complete ==="
echo "To activate this environment in the future, run: source .venv/bin/activate"