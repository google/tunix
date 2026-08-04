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

"""Trainer worker process runner. Loads Qwen3-1.7B with LoRA and starts gRPC service."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

# Setup paths to import tunix
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

import jax
# Enforce TPU platforms for JAX before initializing JAX
os.environ["JAX_PLATFORMS"] = "tpu"

from flax import nnx
from jax import numpy as jnp
from jax.sharding import Mesh
import numpy as np
import optax
from transformers import AutoTokenizer

from tunix.cli.utils import model as model_utils
from tunix.models.qwen3 import model as qwen3_model_lib
from tunix.models.qwen3 import params as qwen3_params_lib
from tunix.rl import rl_cluster as rl_engine_lib

from tunix.experimental.worker import jax_trainer_worker
from tunix.experimental.worker import remote_execution

# ====== Args ======
parser = argparse.ArgumentParser(description="JAX Trainer Worker Process")
parser.add_argument("--port", type=int, default=20000)
parser.add_argument("--tpu_chips", type=str, default="0,1")
args, _ = parser.parse_known_args()

# Pin to specified TPU chips
os.environ["TPU_VISIBLE_CHIPS"] = args.tpu_chips

# Initialize JAX
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [TrainerNode] %(message)s")
logging.info("Initializing JAX on TPU chips: %s", args.tpu_chips)
logging.info("Visible devices: %s", jax.devices())

# ====== Configuration ======
MODEL_NAME = "Qwen3-1.7B"
MODEL_ID = f"Qwen/{MODEL_NAME}"
MODEL_DTYPE = jnp.bfloat16

LORA_RANK = 64
LORA_ALPHA = 64.0


def main():
  # Set up Mesh on local TPU devices (shape 1,2)
  local_devices = jax.devices()
  mesh = Mesh(np.array(local_devices).reshape(1, len(local_devices)), ("data", "fsdp"))
  
  # Configure logical sharding rules
  logical_rules = qwen3_params_lib.LogicalAxisRules.get_default_rules()
  role_rules = {
      rl_engine_lib.Role.ACTOR: logical_rules,
      rl_engine_lib.Role.REFERENCE: logical_rules,
  }

  # Load Actor model with LoRA
  logging.info("Loading Actor model (with LoRA)...")
  actor_model = model_utils.create_model_from_safe_tensors(
      MODEL_ID,
      mesh=mesh,
      data_type=MODEL_DTYPE,
      sharding_config=qwen3_params_lib.ShardingConfig.get_default_sharding(),
  )
  lora_config = {
      "module_path": (
          ".*q_proj|.*k_proj|.*v_proj|.*o_proj|"
          ".*gate_proj|.*down_proj|.*up_proj"
      ),
      "rank": LORA_RANK,
      "alpha": LORA_ALPHA,
  }
  actor_model = model_utils.apply_lora_to_model(
      actor_model, mesh=mesh, lora_config=lora_config
  )

  # Load Reference model (frozen, no LoRA)
  logging.info("Loading Reference model...")
  reference_model = model_utils.create_model_from_safe_tensors(
      MODEL_ID,
      mesh=mesh,
      data_type=MODEL_DTYPE,
      sharding_config=qwen3_params_lib.ShardingConfig.get_default_sharding(),
  )

  # Optimizer
  optimizer = optax.adamw(learning_rate=2.0e-7)

  tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

  # Build local RLEngine
  cluster_config = rl_engine_lib.ClusterConfig(
      role_to_mesh={
          rl_engine_lib.Role.ACTOR: mesh,
          rl_engine_lib.Role.REFERENCE: mesh,
      },
      role_to_logical_axis_rule=role_rules,
      rollout_engine="vanilla",
      offload_to_cpu=False,
      training_config=rl_engine_lib.RLTrainingConfig(
          actor_optimizer=optimizer,
          train_micro_batch_size=1,
          compute_logps_micro_batch_size=1,
      ),
  )

  logging.info("Initializing RLEngine...")
  rl_engine = rl_engine_lib.RLEngine(
      actor=actor_model,
      reference=reference_model,
      tokenizer=tokenizer,
      cluster_config=cluster_config,
  )

  # Create remote execution wrapper using generic JaxTrainerWorkerService
  worker_service = jax_trainer_worker.JaxTrainerWorkerService(rl_engine)
  server = remote_execution.GrpcRemoteExecutionServer(worker_service)

  logging.info("Starting Trainer gRPC Server on port %d...", args.port)
  asyncio.run(server.start_serving_async(args.port))


if __name__ == "__main__":
  main()
