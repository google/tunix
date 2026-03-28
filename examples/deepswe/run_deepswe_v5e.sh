#!/bin/bash

# Exit immediately if a command fails
set -e

echo "Starting DeepSWE training..."

# Run the training script
python tunix/examples/deepswe/train_deepswe_nb.py \
    --model_version="Qwen3-1.7B" \
    --node_selector_val="deepswe-worker-pool" \
    --max_turns=10 \
    

echo "Process finished."