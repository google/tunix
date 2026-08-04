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

"""Rollout worker process runner. Supports JAX VanillaRollout or JAX in-process vLLM engine."""

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

from transformers import AutoTokenizer
from tunix.experimental.worker import remote_execution

# ====== Args ======
parser = argparse.ArgumentParser(description="JAX Rollout Worker Process")
parser.add_argument("--port", type=int, default=20001)
parser.add_argument("--tpu_chips", type=str, default="2,3")
parser.add_argument("--engine", type=str, default="vllm", choices=["vanilla", "vllm"])
args, _ = parser.parse_known_args()

# Pin to specified TPU chips
os.environ["TPU_VISIBLE_CHIPS"] = args.tpu_chips

# Initialize JAX
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [RolloutNode] %(message)s")
logging.info("Initializing JAX on TPU chips: %s using engine: %s", args.tpu_chips, args.engine)
logging.info("Visible devices: %s", jax.devices())

# ====== Configuration ======
MODEL_NAME = "Qwen3-1.7B"
MODEL_ID = f"Qwen/{MODEL_NAME}"
LORA_RANK = 64
LORA_ALPHA = 64.0

MAX_PROMPT_LENGTH = 512
MAX_RESPONSE_LENGTH = 512
KV_CACHE_SIZE = MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 128


def main():
  tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

  if args.engine == "vllm":
    # Lazily import vLLM dependencies
    from tunix.generate import vllm_sampler
    from tunix.generate import mappings as mappings_lib
    from tunix.models.qwen3 import mapping_vllm_jax
    from tunix.experimental.rollout import legacy_vllm_sampler_adapter
    from tunix.experimental.worker import vllm_rollout_worker

    # Configure LoRA weight mappings for vLLM JAX backend
    mapping_config = mappings_lib.MappingConfig(
        lora_to_hf_mappings=mapping_vllm_jax.LORA_TO_HF_MAPPINGS
    )

    vllm_config = vllm_sampler.VllmConfig(
        tensor_parallel_size=jax.device_count(),
        lora_config={
            "max_lora_rank": LORA_RANK,
            "max_loras": 1,
        },
        mapping_config=mapping_config,
        engine_kwargs={"model": MODEL_ID},
    )

    logging.info("Initializing in-process vLLM sampler...")
    sampler_adapter = legacy_vllm_sampler_adapter.LegacyVllmSamplerAdapter(
        server_id="vllm-rollout-0",
        tokenizer=tokenizer,
        config=vllm_config,
    )
    sampler_adapter.initialize()

    worker_service = vllm_rollout_worker.VllmRolloutWorkerService(sampler_adapter)

  else:  # vanilla
    # Lazily import JAX models/parameters dependencies
    from jax.sharding import Mesh
    import numpy as np
    from tunix.cli.utils import model as model_utils
    from tunix.models.qwen3 import params as qwen3_params_lib
    from tunix.rl.rollout import vanilla_rollout
    from tunix.rl.rollout import base_rollout
    from tunix.experimental.worker import jax_rollout_worker

    # Set up Mesh on local TPU devices (shape 1,2)
    local_devices = jax.devices()
    mesh = Mesh(np.array(local_devices).reshape(1, len(local_devices)), ("data", "fsdp"))
    logical_rules = qwen3_params_lib.LogicalAxisRules.get_default_rules()

    # Load Actor model with LoRA
    logging.info("Loading Rollout Actor model (with LoRA)...")
    model = model_utils.create_model_from_safe_tensors(
        MODEL_ID,
        mesh=mesh,
        data_type=jax.numpy.bfloat16,
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
    model = model_utils.apply_lora_to_model(
        model, mesh=mesh, lora_config=lora_config
    )

    logging.info("Initializing VanillaRollout...")
    rollout_config = base_rollout.RolloutConfig(
        max_prompt_length=MAX_PROMPT_LENGTH,
        max_tokens_to_generate=MAX_RESPONSE_LENGTH,
        return_logprobs=True,
        kv_cache_size=KV_CACHE_SIZE,
    )

    rollout_engine = vanilla_rollout.VanillaRollout(
        model=model,
        tokenizer=tokenizer,
        rollout_config=rollout_config,
        mesh=mesh,
        logical_axis_rules=logical_rules,
    )

    worker_service = jax_rollout_worker.JaxRolloutWorkerService(
        rollout_engine=rollout_engine,
        max_prompt_length=MAX_PROMPT_LENGTH,
        kv_cache_size=KV_CACHE_SIZE,
    )

  # Start gRPC Remote Execution Server
  server = remote_execution.GrpcRemoteExecutionServer(worker_service)

  logging.info("Starting Rollout gRPC Server (engine: %s) on port %d...", args.engine, args.port)
  asyncio.run(server.start_serving_async(args.port))


if __name__ == "__main__":
  main()
