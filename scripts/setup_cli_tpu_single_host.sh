#!/bin/bash
# Stop the script if any command fails
set -e

echo "=== Starting Tunix CLI TPU Environment Setup ==="

# 1. Create a project specific environment
echo "Creating virtual environment in .venv..."
python3 -m venv .venv

# 2. Activate the Environment
echo "Activating virtual environment..."
source .venv/bin/activate

# 3. Install Tunix dependency
echo "Upgrading pip to the latest version..."
pip install --upgrade pip

echo "Installing Tunix with TPU dependencies (prod)..."
# The [prod] extra includes specific versions of JAX for TPU
pip install -e .[prod]

echo "Installing gcsfs..."
pip install gcsfs

echo "=== TPU Setup Complete ==="
echo "To activate this environment in the future, run: source .venv/bin/activate"