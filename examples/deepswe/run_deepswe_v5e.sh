#!/bin/bash

# Exit immediately if a command fails
set -e

echo "Starting DeepSWE training..."

# Run the training script
export SKIP_JAX_PRECOMPILE=True
export ROLLOUT_ENGINE=vllm
export JAX_RANDOM_WEIGHTS=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export NEW_MODEL_DESIGN=1
export TPU_MIN_LOG_LEVEL=0
export TF_CPP_MIN_LOG_LEVEL=0
export TPU_STDERR_LOG_LEVEL=0
export NUM_SLICES=1

python examples/deepswe/train_deepswe_nb.py \
    --model_version="Qwen3-1.7B" \
    --node_selector_val="deepswe-worker-pool" \
    --max_turns=10 \
    

echo "Process finished."
