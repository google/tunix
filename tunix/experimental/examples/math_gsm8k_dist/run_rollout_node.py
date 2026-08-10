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

"""vLLM rollout worker process runner for the distributed GRPO demo."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import pickle
import sys

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from transformers import AutoTokenizer

from tunix.experimental.rollout import legacy_vllm_sampler_adapter
from tunix.experimental.worker import remote_execution
from tunix.experimental.worker import rollout_worker
from tunix.generate import mappings as mappings_lib
from tunix.generate import vllm_sampler
from tunix.models.qwen3 import mapping_vllm_jax


REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)


def create_tokenizer(args):
  tokenizer_path = args.tokenizer_path or args.model_dir or args.model_id
  logging.info("Loading tokenizer from %s...", tokenizer_path)

  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
  if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
  return tokenizer

def create_vllm_worker(args, mesh, tokenizer):
  logging.info("Creating vLLM mapping config...")

  mapping_config = mappings_lib.MappingConfig(
      lora_to_hf_mappings=mapping_vllm_jax.LORA_TO_HF_MAPPINGS
  )
  vllm_model = args.model_dir or args.model_id
  rollout_mesh = mesh
  max_model_len = args.max_prompt_length + args.max_response_length
  logging.info(
      "Creating vLLM config for model=%s mesh=%s tensor_parallel_size=%d "
      "max_model_len=%d...",
      vllm_model,
      rollout_mesh,
      jax.device_count(),
      max_model_len,
  )
  vllm_config = vllm_sampler.VllmConfig(
      mesh=rollout_mesh,
      tensor_parallel_size=jax.device_count(),
      data_parallel_size=1,
      return_logprobs=True,
      lora_config=(
          {
              "max_lora_rank": args.lora_rank,
              "max_loras": 1,
          }
          if args.use_lora
          else None
      ),
      mapping_config=mapping_config,
      engine_kwargs={
          "model": vllm_model,
          "max_model_len": max_model_len,
      },
  )
  sampler_adapter = legacy_vllm_sampler_adapter.LegacyVllmSamplerAdapter(
      server_id=args.worker_id,
      tokenizer=tokenizer,
      config=vllm_config,
  )
  logging.info("Creating RolloutWorker wrapper...")
  config = rollout_worker.RolloutConfig(
      sampler_type="legacy_vllm",
      max_prompt_length=args.max_prompt_length,
      max_tokens_to_generate=args.max_response_length,
      temperature=1.0,
      top_p=1.0,
      return_logprobs=True,
      rollout_vllm_model_version=vllm_model,
  )
  return rollout_worker.RolloutWorker(
      worker_id=args.worker_id,
      config=config,
      sampler=sampler_adapter,
      tokenizer=tokenizer,
      # Direct prompt sampling does not use chat_parser. Pass a real parser when
      # enabling multi-turn agentic trajectories on this worker.
      chat_parser=tokenizer,
      max_concurrency=64,
  )

def main(argv: list[str], context: Any = None) -> None:
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - [RolloutNode] %(message)s",
      force=True,
  )

  parser = argparse.ArgumentParser(description="vLLM rollout worker process")
  parser.add_argument("--port", type=int, default=20001)
  parser.add_argument("--worker_id", type=str, default="vllm-rollout-0")
  parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
  parser.add_argument("--model_dir", type=str, default=os.getenv("MODEL_DIR", ""))
  parser.add_argument("--tokenizer_path", type=str, default="")
  parser.add_argument("--max_prompt_length", type=int, default=512)
  parser.add_argument("--max_response_length", type=int, default=128)
  parser.add_argument("--use_lora", action="store_true")
  parser.add_argument("--lora_rank", type=int, default=16)
  parser.add_argument("--lora_alpha", type=float, default=16.0)

  # 1. parse args
  args = parser.parse_args(argv)
  logging.info("Parsed args: %s", args)

  # 2. init JAX, vLLM, import PATH
  if context:
    context.jax.initialize()
  os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
  os.environ.setdefault("VLLM_TPU_RPA_VERSION", "2")
  os.environ.setdefault("DISABLE_MOSAIC_ATTN", "1")
  if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
  logging.info("Repo root inserted into sys.path: %s", REPO_ROOT)

  # 3. load tokenizer
  tokenizer = create_tokenizer(args)

  # 4. create worker
  logging.info("Creating rollout worker service...")
  worker_service = create_vllm_worker(
    args=args,
    mesh=Mesh(
      mesh_utils.create_device_mesh((1, jax.device_count()), jax.devices()),
      axis_names=("dp", "tp")),
    tokenizer=tokenizer)

  # 5. start rpc endpoint
  async def grpc_server_main() -> None:
    logging.info("Starting rollout gRPC server...")
    server = remote_execution.GrpcRemoteExecutionServer(worker_service)
    await server.start_serving_async(args.port)
    logging.info("Serving vLLM rollout worker on port %d.", args.port)

    # 6. register
    if context and context.ipc and context.ipc.discovery:
      context.ipc.discovery.register(
          metadata=pickle.dumps({
              "service_type": "rollout",
              "service_port": args.port,
              "worker_id": args.worker_id,
              "model_id": args.model_id,
          })
      )
      logging.info("Rollout worker is registered.")
    else:
      logging.info("Rollout worker is not registered: process context doesn't support.")

    try:
      while True:
        await asyncio.sleep(1)
    except asyncio.CancelledError:
      pass
    finally:
      await server.stop_serving()
  asyncio.run(grpc_server_main())

if __name__ == "__main__":
  main(sys.argv[1:])
