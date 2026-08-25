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
from typing import Any

import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from transformers import AutoTokenizer
from tunix.experimental.examples.math_gsm8k_dist import gsm8k
from tunix.experimental.examples.math_gsm8k_dist import models
from tunix.experimental.rollout import inprocess_vllm_sampler_adapter
from tunix.experimental.rollout import vanilla_sampler_adapter
from tunix.experimental.weight_sync import raiden_weight_sync_delegate
from tunix.experimental.weight_sync import weight_sync
from tunix.experimental.worker import remote_execution
from tunix.experimental.worker import rollout_worker
from tunix.generate import mappings as mappings_lib
from tunix.generate import tokenizer_adapter as tokenizer_adapter_lib
from tunix.generate import vllm_sampler
from tunix.models.qwen3 import mapping_vllm_jax
from tunix.rl.agentic.parser.chat_template_parser import parser as chat_parser_lib

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)

SAMPLERS = ("inprocess_vllm",)

CHAT_PARSERS = {
    "qwen": chat_parser_lib.QwenChatTemplateParser,
    "llama": chat_parser_lib.LlamaChatTemplateParser,
    "gemma": chat_parser_lib.GemmaChatTemplateParser,
}


def _chat_parser_for(model_id: str, tokenizer):
  """Selects the chat template parser by model family."""
  name = model_id.lower()
  for family, parser_cls in CHAT_PARSERS.items():
    if family in name:
      return parser_cls(tokenizer, enable_thinking=False)
  return chat_parser_lib.DefaultChatTemplateParser(
      tokenizer, enable_thinking=False
  )


def _parse_args(argv: list[str]) -> argparse.Namespace:
  """Parses command line arguments for the rollout worker process."""
  parser = argparse.ArgumentParser(description="vLLM rollout worker process")
  parser.add_argument("--port", type=int, default=20001)
  parser.add_argument("--worker_id", type=str, default="vllm-rollout-0")
  parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
  parser.add_argument(
      "--model_dir", type=str, default=os.getenv("MODEL_DIR", "")
  )
  parser.add_argument("--tokenizer_path", type=str, default="")
  parser.add_argument("--max_prompt_length", type=int, default=512)
  parser.add_argument("--max_response_length", type=int, default=128)
  parser.add_argument("--use_lora", action="store_true")
  parser.add_argument("--lora_rank", type=int, default=16)
  parser.add_argument("--lora_alpha", type=float, default=16.0)
  parser.add_argument(
      "--model_name", type=str, default=os.getenv("MODEL_NAME", "Qwen3-1.7B")
  )
  parser.add_argument(
      "--sampler",
      type=str,
      default=os.getenv("SAMPLER", "inprocess_vllm"),
      choices=["inprocess_vllm", "vanilla"],
  )
  parser.add_argument(
      "--weight_sync_mode",
      type=weight_sync.WeightSyncMode,
      default=weight_sync.WeightSyncMode(
          os.getenv("WEIGHT_SYNC_MODE", "raiden")
      ),
      choices=list(weight_sync.WeightSyncMode),
      help="Weight sync mode (e.g. raiden, fallback).",
  )
  return parser.parse_args(argv)


def _create_rollout_mesh() -> Mesh:
  shape = (1, jax.device_count())
  devices = mesh_utils.create_device_mesh(shape, jax.devices())
  return Mesh(devices, axis_names=("fsdp", "tp"))


def _create_vanilla_worker(args, tokenizer):
  """Creates a vanilla sampler rollout worker instance."""
  logging.info("Creating native sampler on the rollout mesh...")
  mesh = _create_rollout_mesh()
  with mesh:
    model = models.create_model(
        args.model_name, args.model_dir or args.model_id, mesh
    )
  raiden_delegate = (
      raiden_weight_sync_delegate.RaidenWeightSyncDelegate()
      if args.weight_sync_mode == weight_sync.WeightSyncMode.RAIDEN
      else None
  )
  config = rollout_worker.RolloutConfig(
      sampler_type="vanilla",
      weight_sync_mode=args.weight_sync_mode,
      max_prompt_length=args.max_prompt_length,
      max_tokens_to_generate=args.max_response_length,
      temperature=1.0,
      top_p=1.0,
      return_logprobs=True,
      env_name=gsm8k.GSM8K_ENV_NAME,
      agent_name=gsm8k.GSM8K_AGENT_NAME,
  )
  sampler_adapter = vanilla_sampler_adapter.VanillaSamplerAdapter(
      server_id=args.worker_id,
      transformer=model,
      tokenizer=tokenizer,
      cache_config=args.max_prompt_length + args.max_response_length,
      config=config,
      raiden_sync_delegate=raiden_delegate,
  )

  rollout_tokenizer = tokenizer_adapter_lib.TokenizerAdapter(tokenizer)
  chat_parser = chat_parser_lib.QwenChatTemplateParser(
      tokenizer, enable_thinking=False
  )
  return rollout_worker.RolloutWorker(
      worker_id=args.worker_id,
      config=config,
      sampler=sampler_adapter,
      tokenizer=rollout_tokenizer,
      chat_parser=chat_parser,
      max_concurrency=64,
  )


def _create_vllm_worker(args, tokenizer):
  """Creates an in-process vLLM sampler rollout worker instance."""
  logging.info("Creating vLLM mapping config...")
  mapping_config = mappings_lib.MappingConfig(
      lora_to_hf_mappings=mapping_vllm_jax.LORA_TO_HF_MAPPINGS
  )
  vllm_model = args.model_dir or args.model_id
  rollout_mesh = _create_rollout_mesh()
  max_model_len = args.max_prompt_length + args.max_response_length
  logging.info(
      "Creating vLLM config for model=%s mesh=%s tensor_parallel_size=%d "
      "max_model_len=%d...",
      vllm_model,
      rollout_mesh,
      jax.device_count(),
      max_model_len,
  )
  lora_config = None
  if args.use_lora:
    lora_config = {
        "max_lora_rank": args.lora_rank,
        "max_loras": 1,
    }
  vllm_config = vllm_sampler.VllmConfig(
      mesh=rollout_mesh,
      tensor_parallel_size=jax.device_count(),
      data_parallel_size=1,
      return_logprobs=True,
      lora_config=lora_config,
      mapping_config=mapping_config,
      engine_kwargs={
          "model": vllm_model,
          "max_model_len": max_model_len,
      },
  )
  raiden_delegate = (
      raiden_weight_sync_delegate.RaidenWeightSyncDelegate()
      if args.weight_sync_mode == weight_sync.WeightSyncMode.RAIDEN
      else None
  )
  sampler_adapter = inprocess_vllm_sampler_adapter.InprocessVllmSamplerAdapter(
      server_id=args.worker_id,
      tokenizer=tokenizer,
      config=vllm_config,
      raiden_sync_delegate=raiden_delegate,
  )
  rollout_tokenizer = tokenizer_adapter_lib.TokenizerAdapter(tokenizer)
  chat_parser = chat_parser_lib.QwenChatTemplateParser(
      tokenizer, enable_thinking=False
  )
  logging.info("Creating RolloutWorker wrapper...")
  config = rollout_worker.RolloutConfig(
      sampler_type="inprocess_vllm",
      weight_sync_mode=args.weight_sync_mode,
      max_prompt_length=args.max_prompt_length,
      max_tokens_to_generate=args.max_response_length,
      temperature=1.0,
      top_p=1.0,
      return_logprobs=True,
      rollout_vllm_model_version=vllm_model,
      env_name=gsm8k.GSM8K_ENV_NAME,
      agent_name=gsm8k.GSM8K_AGENT_NAME,
  )
  return rollout_worker.RolloutWorker(
      worker_id=args.worker_id,
      config=config,
      sampler=sampler_adapter,
      tokenizer=rollout_tokenizer,
      chat_parser=chat_parser,
      max_concurrency=64,
  )


def main(argv: list[str], context: Any = None) -> None:
  if context and context.ipc and context.ipc.discovery:
    pass
  else:
    raise RuntimeError(
        "Require discovery API, but process context doesn't support."
    )

  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - [RolloutNode] %(message)s",
      force=True,
  )

  args = _parse_args(argv)
  logging.info("Parsed args: %s", args)

  if context:
    context.jax.initialize()
  os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
  os.environ.setdefault("VLLM_TPU_RPA_VERSION", "2")
  os.environ.setdefault("DISABLE_MOSAIC_ATTN", "1")
  if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
  logging.info("Repo root inserted into sys.path: %s", REPO_ROOT)

  tokenizer_path = args.tokenizer_path or args.model_dir or args.model_id
  logging.info("Loading tokenizer from %s...", tokenizer_path)
  tokenizer: Any = AutoTokenizer.from_pretrained(
      tokenizer_path, trust_remote_code=True
  )
  if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

  async def grpc_server_main() -> None:
    logging.info("Creating rollout worker service...")
    if args.sampler == "vanilla":
      worker_service = _create_vanilla_worker(args, tokenizer)
    else:
      worker_service = _create_vllm_worker(args, tokenizer)

    logging.info("Creating rollout gRPC server...")
    server = remote_execution.GrpcRemoteExecutionServer(worker_service)
    await server.start_serving_async(args.port)
    logging.info("Serving vLLM rollout worker on port %d.", args.port)

    context.ipc.discovery.register(
        metadata=pickle.dumps({
            "service_type": "rollout",
            "service_port": args.port,
            "worker_id": args.worker_id,
        })
    )
    logging.info("Rollout worker is registered.")

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
