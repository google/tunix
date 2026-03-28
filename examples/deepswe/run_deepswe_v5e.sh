#!/bin/bash

# Exit immediately if a command fails
set -e

echo "Starting DeepSWE training..."

# Run the training script
SKIP_JAX_PRECOMPILE=True ROLLOUT_ENGINE=vllm JAX_RANDOM_WEIGHTS=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 NEW_MODEL_DESIGN=1 TPU_MIN_LOG_LEVEL=0 TF_CPP_MIN_LOG_LEVEL=0 TPU_STDERR_LOG_LEVEL=0 NUM_SLICES=1
python tunix/examples/deepswe/train_deepswe_nb.py \
    --model_version="Qwen3-1.7B" \
    --node_selector_val="deepswe-worker-pool" \
    --max_turns=10 \
    

echo "Process finished."