#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

# Resolve directory path
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

TRAINER_PORT=20000
ROLLOUT_PORT=20001

# Parse engine type (default to vllm)
ENGINE=${1:-vllm}

if [[ "$ENGINE" != "vllm" && "$ENGINE" != "vanilla" ]]; then
  echo "Error: Unknown engine type '$ENGINE'."
  echo "Usage: $0 [vllm|vanilla]"
  exit 1
fi

echo "=================================================="
echo "Starting Distributed GSM8K GRPO training locally..."
echo "Using $ENGINE Engine for Rollout Generation Node."
echo "=================================================="

# 1. Start Trainer Worker on TPU chips 0,1
echo "Launching Trainer Node on TPU chips 0,1..."
python3 "${DIR}/run_trainer_node.py" \
  --port=$TRAINER_PORT \
  --tpu_chips="0,1" > "${DIR}/trainer.log" 2>&1 &
TRAINER_PID=$!

# 2. Start Rollout Worker on TPU chips 2,3 (configures engine type)
echo "Launching Rollout Node ($ENGINE) on TPU chips 2,3..."
python3 "${DIR}/run_rollout_node.py" \
  --port=$ROLLOUT_PORT \
  --tpu_chips="2,3" \
  --engine="$ENGINE" > "${DIR}/rollout.log" 2>&1 &
ROLLOUT_PID=$!

# Function to clean up background processes on exit
cleanup() {
  echo "Cleaning up worker processes (PIDs: $TRAINER_PID, $ROLLOUT_PID)..."
  kill $TRAINER_PID $ROLLOUT_PID || true
  wait $TRAINER_PID $ROLLOUT_PID 2>/dev/null || true
  echo "Workers stopped."
}
trap cleanup EXIT

# 3. Wait for workers to start gRPC services
echo "Waiting for gRPC servers to initialize (TPU compilation/loading weights might take a moment)..."
python3 -c "
import socket, time
for p in ($TRAINER_PORT, $ROLLOUT_PORT):
  while True:
    try:
      socket.create_connection(('localhost', p), timeout=1)
      print(f'Port {p} is ready.')
      break
    except OSError:
      time.sleep(2)
"

# 4. Start Orchestrator on CPU
echo "Launching Orchestrator..."
python3 "${DIR}/run_gsm8k_dist_grpo.py" \
  --trainer_addr="localhost:$TRAINER_PORT" \
  --rollout_addr="localhost:$ROLLOUT_PORT" \
  --batch_size=4 \
  --mini_batch_size=2 \
  --max_steps=20

echo "Distributed GSM8K GRPO training run ($ENGINE) finished successfully!"
