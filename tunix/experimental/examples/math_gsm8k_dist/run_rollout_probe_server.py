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

"""Standalone rollout worker server for Pathways smoke tests.

This starts the same rollout worker implementation used by the GRPO demo, but
without requiring discovery registration or an orchestrator process.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from typing import Sequence

from tunix.experimental.examples.math_gsm8k_dist import run_rollout_node


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
  return run_rollout_node._parse_args(list(argv))


def _standalone_init_with_random_weights() -> bool | None:
  value = os.getenv("STANDALONE_INIT_WITH_RANDOM_WEIGHTS", "")
  if not value:
    return None
  normalized = value.strip().lower()
  if normalized in ("1", "true", "yes", "on"):
    return True
  if normalized in ("0", "false", "no", "off"):
    return False
  raise ValueError(
      "STANDALONE_INIT_WITH_RANDOM_WEIGHTS must be one of "
      "0/1/false/true/no/yes/off/on"
  )


def _create_inprocess_vllm_worker_for_standalone(args, tokenizer):
  """Creates an in-process vLLM rollout worker with smoke-test-only overrides."""
  vllm_sampler = run_rollout_node._import_vllm_sampler()
  from tunix.experimental.rollout import (  # pylint: disable=g-import-not-at-top
      inprocess_vllm_sampler_adapter,
  )
  from tunix.experimental.worker import (  # pylint: disable=g-import-not-at-top
      rollout_worker,
  )
  from tunix.generate import (  # pylint: disable=g-import-not-at-top
      mappings as mappings_lib,
  )
  from tunix.generate import (  # pylint: disable=g-import-not-at-top
      tokenizer_adapter as tokenizer_adapter_lib,
  )
  from tunix.models.qwen3 import (  # pylint: disable=g-import-not-at-top
      mapping_vllm_jax,
  )

  logging.info("Creating standalone vLLM mapping config...")
  mapping_config = mappings_lib.MappingConfig(
      lora_to_hf_mappings=mapping_vllm_jax.LORA_TO_HF_MAPPINGS
  )
  vllm_model = (
      args.model_dir
      if (
          args.model_dir
          and os.path.exists(args.model_dir)
          and any(os.scandir(args.model_dir))
      )
      else args.model_id
  )
  rollout_mesh = run_rollout_node._create_rollout_mesh(args)
  max_model_len = args.max_prompt_length + args.max_response_length
  init_with_random_weights = _standalone_init_with_random_weights()
  if init_with_random_weights is None:
    init_with_random_weights = True
  logging.info(
      "Creating standalone vLLM config for model=%s mesh=%s tensor_parallel_size=%d "
      "data_parallel_size=%d max_model_len=%d init_with_random_weights=%s...",
      vllm_model,
      rollout_mesh,
      args.mesh_tp,
      args.mesh_fsdp,
      max_model_len,
      init_with_random_weights,
  )
  lora_config = None
  if args.use_lora:
    lora_config = {
        "max_lora_rank": args.lora_rank,
        "max_loras": 1,
    }
  vllm_config = vllm_sampler.VllmConfig(
      mesh=rollout_mesh,
      tensor_parallel_size=args.mesh_tp,
      data_parallel_size=args.mesh_fsdp,
      init_with_random_weights=init_with_random_weights,
      return_logprobs=True,
      lora_config=lora_config,
      mapping_config=mapping_config,
      engine_kwargs={
          "model": vllm_model,
          "max_model_len": max_model_len,
      },
  )
  sampler_adapter = inprocess_vllm_sampler_adapter.InprocessVllmSamplerAdapter(
      server_id=args.worker_id,
      tokenizer=tokenizer,
      config=vllm_config,
      weight_sync_mode=args.weight_sync_mode,
  )
  rollout_config = rollout_worker.RolloutConfig(
      sampler_type="inprocess_vllm",
      weight_sync_mode=args.weight_sync_mode,
      max_prompt_length=args.max_prompt_length,
      max_tokens_to_generate=args.max_response_length,
      temperature=1.0,
      top_p=1.0,
      return_logprobs=True,
      rollout_vllm_model_version=vllm_model,
      env_name=run_rollout_node.gsm8k.GSM8K_ENV_NAME,
      agent_name=run_rollout_node.gsm8k.GSM8K_AGENT_NAME,
  )
  rollout_tokenizer = tokenizer_adapter_lib.TokenizerAdapter(tokenizer)
  chat_parser = run_rollout_node._chat_parser_for(
      args.model_id or args.model_name, tokenizer
  )
  return rollout_worker.RolloutWorker(
    worker_id=args.worker_id,
    config=rollout_config,
    sampler=sampler_adapter,
    tokenizer=rollout_tokenizer,
    chat_parser=chat_parser,
    max_concurrency=64,
  )


async def _serve(args: argparse.Namespace) -> None:
  os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
  os.environ.setdefault("VLLM_TPU_RPA_VERSION", "2")
  os.environ.setdefault("DISABLE_MOSAIC_ATTN", "1")
  os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
  if args.maxtext_model_name:
    os.environ.setdefault("NEW_MODEL_DESIGN", "1")
  if run_rollout_node.REPO_ROOT not in sys.path:
    sys.path.insert(0, run_rollout_node.REPO_ROOT)

  from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top
  from tunix.experimental.worker import (  # pylint: disable=g-import-not-at-top
      remote_execution,
  )

  tokenizer_path = args.tokenizer_path or args.model_dir or args.model_id
  logging.info("Loading tokenizer from %s...", tokenizer_path)
  tokenizer = AutoTokenizer.from_pretrained(
      tokenizer_path, trust_remote_code=True
  )
  if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token

  logging.info("Creating rollout worker service without discovery...")
  if args.sampler == "vanilla":
    worker_service = run_rollout_node._create_vanilla_worker(args, tokenizer)
  elif args.sampler == "inprocess_vllm":
    worker_service = _create_inprocess_vllm_worker_for_standalone(
        args, tokenizer
    )
  else:
    worker_service = run_rollout_node._create_vllm_worker(args, tokenizer)

  server = remote_execution.GrpcRemoteExecutionServer(worker_service)
  await server.start_serving_async(args.port)
  logging.info("Serving standalone rollout worker on port %d.", args.port)

  if args.sampler != "vanilla":
    logging.info("Eagerly starting sampler engine...")
    await worker_service.sampler.start()
    logging.info("Sampler engine started.")
    if hasattr(worker_service.sampler, "bind_weight_sync"):
      logging.info("Eagerly warming up weight sync delegate...")
      await worker_service.sampler.bind_weight_sync()
      logging.info("Weight sync delegate warmed up.")

  try:
    while True:
      await asyncio.sleep(1)
  except asyncio.CancelledError:
    logging.info("Standalone rollout worker cancelled; shutting down.")
    raise
  finally:
    await server.stop_serving(5)


def main(argv: Sequence[str]) -> None:
  args = _parse_args(argv)
  logging.basicConfig(
      level=logging.DEBUG if args.debug else logging.INFO,
      format="%(asctime)s - [RolloutProbeServer] %(message)s",
      force=True,
  )
  logging.info("Parsed args: %s", args)
  asyncio.run(_serve(args))


if __name__ == "__main__":
  main(sys.argv[1:])