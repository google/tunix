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
import importlib
import logging
import os
import pickle
import sys
from typing import Any

from tunix.experimental.examples.math_gsm8k_dist import gsm8k
from tunix.experimental.examples.math_gsm8k_dist import models
from tunix.experimental.weight_sync import weight_sync as weight_sync_lib
from tunix.rl.agentic.parser.chat_template_parser import parser as chat_parser_lib

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)

# This must be set before the first vLLM import. Keep vLLM imports lazy so the
# rollout process can start with non-vLLM samplers in environments where vLLM is
# not installed.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

CHAT_PARSERS = {
    "qwen": chat_parser_lib.QwenChatTemplateParser,
    "llama": chat_parser_lib.LlamaChatTemplateParser,
    "gemma": chat_parser_lib.GemmaChatTemplateParser,
}


def _import_vllm_sampler():
  logging.info(
      "Importing tunix.generate.vllm_sampler before rollout adapters..."
  )
  vllm_sampler = importlib.import_module("tunix.generate.vllm_sampler")
  logging.info("Finished importing tunix.generate.vllm_sampler.")
  return vllm_sampler


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
  parser.add_argument("--mesh_fsdp", type=int, default=1)
  parser.add_argument("--mesh_tp", type=int, default=2)
  parser.add_argument("--max_prompt_length", type=int, default=1024)
  parser.add_argument("--max_response_length", type=int, default=1024)
  parser.add_argument("--use_lora", action="store_true")
  parser.add_argument("--lora_rank", type=int, default=64)
  parser.add_argument("--lora_alpha", type=float, default=64.0)
  parser.add_argument(
      "--model_name", type=str, default=os.getenv("MODEL_NAME", "Qwen3-1.7B")
  )
  parser.add_argument(
      "--sampler",
      type=str,
      default=os.getenv("SAMPLER", "inprocess_vllm"),
      choices=["vllm", "inprocess_vllm", "vanilla"],
      help="Rollout sampler backend: vllm, inprocess_vllm, or vanilla.",
  )
  parser.add_argument("--sampler_mesh_tp", type=int, default=4)
  parser.add_argument(
      "--maxtext_model_name",
      type=str,
      default="",
      help=(
          "MaxText model name (e.g. qwen3-0.6b) to load via"
          " maxtext_vllm_adapter's MaxTextForCausalLM."
      ),
  )
  parser.add_argument(
      "--maxtext_attention",
      type=str,
      default="",
      help=(
          "Override MaxText inference attention kernel (e.g."
          " vllm_batched_rpa)."
      ),
  )
  parser.add_argument(
      "--debug",
      action="store_true",
      help="Enable debug logging for rollout worker.",
  )

  parser.add_argument(
      "--weight_sync_mode",
      type=weight_sync_lib.WeightSyncMode,
      default=weight_sync_lib.WeightSyncMode(
          os.getenv("WEIGHT_SYNC_MODE", "none")
      ),
      choices=list(weight_sync_lib.WeightSyncMode),
      help="Weight sync mode (none, fallback, or raiden).",
  )
  return parser.parse_args(argv)


def _create_rollout_mesh(args) -> Any:
  import jax  # pylint: disable=g-import-not-at-top
  from jax.experimental import mesh_utils  # pylint: disable=g-import-not-at-top
  from jax.sharding import Mesh  # pylint: disable=g-import-not-at-top

  shape = (args.mesh_fsdp, args.mesh_tp)
  if args.mesh_fsdp * args.mesh_tp != jax.device_count():
    raise ValueError(
        "Rollout mesh dimensions must match visible device count: "
        f"mesh_fsdp={args.mesh_fsdp} mesh_tp={args.mesh_tp} "
        f"device_count={jax.device_count()}"
    )

  devices = mesh_utils.create_device_mesh(shape, jax.devices())
  mesh = Mesh(devices, axis_names=("fsdp", "tp"))
  logging.info("Rollout mesh: %s", mesh)
  return mesh


def _create_vanilla_worker(args, tokenizer):
  """Creates a vanilla sampler rollout worker instance."""
  from tunix.experimental.rollout import (  # pylint: disable=g-import-not-at-top
      vanilla_sampler_adapter,
  )
  from tunix.experimental.worker import (  # pylint: disable=g-import-not-at-top
      rollout_worker,
  )
  from tunix.generate import (  # pylint: disable=g-import-not-at-top
      tokenizer_adapter as tokenizer_adapter_lib,
  )

  logging.info("Creating native sampler on the rollout mesh...")
  mesh = _create_rollout_mesh(args)
  with mesh:
    model = models.create_model(
        args.model_name, args.model_dir or args.model_id, mesh
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
  )

  rollout_tokenizer = tokenizer_adapter_lib.TokenizerAdapter(tokenizer)
  chat_parser = _chat_parser_for(args.model_id or args.model_name, tokenizer)
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
  from tunix.experimental.worker import (  # pylint: disable=g-import-not-at-top
      rollout_worker,
  )
  from tunix.generate import (  # pylint: disable=g-import-not-at-top
      tokenizer_adapter as tokenizer_adapter_lib,
  )

  if args.sampler == "vllm":
    sampler_adapter, rollout_config = _create_vllm_sampler(args)
  else:
    sampler_adapter, rollout_config = _create_inprocess_vllm_sampler(
        args, tokenizer
    )

  rollout_tokenizer = tokenizer_adapter_lib.TokenizerAdapter(tokenizer)
  chat_parser = _chat_parser_for(args.model_id or args.model_name, tokenizer)
  logging.info("Creating RolloutWorker wrapper...")
  return rollout_worker.RolloutWorker(
      worker_id=args.worker_id,
      config=rollout_config,
      sampler=sampler_adapter,
      tokenizer=rollout_tokenizer,
      chat_parser=chat_parser,
      max_concurrency=64,
  )


def _create_inprocess_vllm_sampler(args, tokenizer):
  """Creates an in-process vLLM sampler rollout worker instance."""
  vllm_sampler = _import_vllm_sampler()
  import jax  # pylint: disable=g-import-not-at-top
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

  logging.info("Creating vLLM mapping config...")
  mapping_config = mappings_lib.MappingConfig(
      lora_to_hf_mappings=mapping_vllm_jax.LORA_TO_HF_MAPPINGS
  )
  vllm_model = args.model_dir or args.model_id
  rollout_mesh = _create_rollout_mesh(args)
  max_model_len = args.max_prompt_length + args.max_response_length
  logging.info(
      "Creating vLLM config for model=%s mesh=%s tensor_parallel_size=%d "
      "data_parallel_size=%d max_model_len=%d...",
      vllm_model,
      rollout_mesh,
      args.mesh_tp,
      args.mesh_fsdp,
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
      tensor_parallel_size=args.mesh_tp,
      data_parallel_size=args.mesh_fsdp,
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
  return sampler_adapter, config


def _create_vllm_sampler(args):
  """Creates a vLLM sampler rollout worker instance."""
  from tunix.experimental.rollout import vllm_sampler_adapter  # pylint: disable=g-import-not-at-top
  from tunix.experimental.worker import rollout_worker  # pylint: disable=g-import-not-at-top
  from vllm.engine.arg_utils import AsyncEngineArgs  # pylint: disable=g-import-not-at-top

  vllm_model = (
      args.model_dir
      if (
          args.model_dir
          and os.path.exists(args.model_dir)
          and any(os.scandir(args.model_dir))
      )
      else args.model_id
  )
  max_model_len = args.max_prompt_length + args.max_response_length
  logging.info(
      "Creating vLLM RLVllmSampler config for model=%s tensor_parallel_size=%d "
      "max_model_len=%d...",
      vllm_model,
      args.sampler_mesh_tp,
      max_model_len,
  )
  engine_kwargs = dict(
      model=vllm_model,
      tokenizer=args.tokenizer_path or vllm_model,
      tensor_parallel_size=args.sampler_mesh_tp,
      max_model_len=max_model_len,
      trust_remote_code=True,
      dtype="bfloat16",
      enable_lora=args.use_lora,
      max_lora_rank=args.lora_rank if args.use_lora else None,
      max_loras=1 if args.use_lora else None,
  )
  if args.maxtext_model_name:
    logging.info(
        "Loading MaxText model %r natively via maxtext_vllm_adapter's"
        " MaxTextForCausalLM (architectures override).",
        args.maxtext_model_name,
    )
    engine_kwargs["hf_overrides"] = {"architectures": ["MaxTextForCausalLM"]}
    # MaxText inference config. prefuse_moe_weights is left False so rollout
    # variable names match unfused trainer parameters during weight sync.
    maxtext_config_overrides = {
        "model_name": args.maxtext_model_name,
        "model_call_mode": "inference",
        "enable_dp_attention": False,
        "allow_split_physical_axes": True,
        "log_config": False,
        "weight_dtype": "bfloat16",
    }
    if args.maxtext_attention:
      maxtext_config_overrides["attention"] = args.maxtext_attention
    engine_kwargs["additional_config"] = {
        "maxtext_config": maxtext_config_overrides
    }
  engine_args = AsyncEngineArgs(**engine_kwargs)
  sampler_adapter = vllm_sampler_adapter.VllmSamplerAdapter(
      server_id=args.worker_id,
      engine_args=engine_args,
      model_name=vllm_model,
      weight_sync_mode=args.weight_sync_mode,
  )
  config = rollout_worker.RolloutConfig(
      sampler_type="vllm",
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
  return sampler_adapter, config


def main(argv: list[str], context: Any = None) -> None:
  if context and context.ipc and context.ipc.discovery:
    pass
  else:
    raise RuntimeError(
        "Require discovery API, but process context doesn't support."
    )

  args = _parse_args(argv)
  logging.basicConfig(
      level=logging.DEBUG if args.debug else logging.INFO,
      format="%(asctime)s - [RolloutNode] %(message)s",
      force=True,
  )
  logging.info("Parsed args: %s", args)

  if context and args.sampler != "vllm":
    context.jax.initialize()
  os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
  os.environ.setdefault("VLLM_TPU_RPA_VERSION", "2")
  os.environ.setdefault("DISABLE_MOSAIC_ATTN", "1")
  os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
  if args.maxtext_model_name:
    os.environ.setdefault("NEW_MODEL_DESIGN", "1")
  if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
  logging.info("Repo root inserted into sys.path: %s", REPO_ROOT)

  from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

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

    from tunix.experimental.worker import (  # pylint: disable=g-import-not-at-top
        remote_execution,
    )

    logging.info("Creating rollout gRPC server...")
    server = remote_execution.GrpcRemoteExecutionServer(worker_service)
    await server.start_serving_async(args.port)
    logging.info("Serving vLLM rollout worker on port %d.", args.port)

    if args.sampler != "vanilla":
      # Eagerly start the sampler engine so all pods in a multihost rollout
      # jobset join the JAX distributed group at startup rather than lazily.
      logging.info("Eagerly starting sampler engine...")
      await worker_service.sampler.start()
      logging.info("Sampler engine started.")
      if hasattr(worker_service.sampler, "bind_weight_sync"):
        logging.info("Eagerly warming up Raiden weight sync...")
        await worker_service.sampler.bind_weight_sync()
        logging.info("Raiden weight sync warmed up.")

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
