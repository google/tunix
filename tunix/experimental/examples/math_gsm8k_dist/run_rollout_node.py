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
from tunix.rl.agentic.parser.chat_template_parser import parser as chat_parser_lib

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)

# This must be set before the first vLLM import. Keep vLLM imports lazy so the
# runtime context and logging are already configured if TPU/vLLM initialization
# fails.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

CHAT_PARSERS = {
    "qwen": chat_parser_lib.QwenChatTemplateParser,
    "llama": chat_parser_lib.LlamaChatTemplateParser,
    "gemma": chat_parser_lib.GemmaChatTemplateParser,
}


def _import_vllm_sampler():
  """Imports Tunix's vLLM sampler before rollout-side sync adapters."""
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


def _str_to_bool(value: str | bool) -> bool:
  if isinstance(value, bool):
    return value
  normalized = value.lower()
  if normalized in ("1", "true", "t", "yes", "y"):
    return True
  if normalized in ("0", "false", "f", "no", "n"):
    return False
  raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
  """Parses command line arguments for the rollout worker process."""
  from tunix.experimental.weight_sync import (  # pylint: disable=g-import-not-at-top
      weight_sync,
  )

  parser = argparse.ArgumentParser(description="vLLM rollout worker process")
  parser.add_argument("--port", type=int, default=20001)
  parser.add_argument("--worker_id", type=str, default="vllm-rollout-0")
  parser.add_argument("--model_id", type=str, default=os.getenv("MODEL_ID", ""))
  models.add_model_source_args(parser)
  parser.add_argument(
      "--model_dir", type=str, default=os.getenv("MODEL_DIR", "")
  )
  parser.add_argument("--tokenizer_path", type=str, default="")
  parser.add_argument("--mesh_fsdp", type=int, default=1)
  parser.add_argument("--mesh_tp", type=int, default=2)
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
  parser.add_argument(
      "--vllm_init_with_random_weights",
      type=_str_to_bool,
      default=_str_to_bool(os.getenv("VLLM_INIT_WITH_RANDOM_WEIGHTS", "false")),
  )
  return parser.parse_args(argv)


def _create_rollout_mesh(args):
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


def _tokenizer_path(args) -> str:
  if args.tokenizer_path:
    return args.tokenizer_path
  if models.is_maxtext_source(args.model_source):
    return args.model_id
  return args.model_dir or args.model_id


def _register_maxtext_vllm_adapter() -> None:
  try:
    from maxtext.integration.vllm import (  # pylint: disable=g-import-not-at-top
        maxtext_vllm_adapter,
    )
  except ImportError as exc:
    raise RuntimeError(
        "MaxText rollout requires maxtext.integration.vllm. Install MaxText "
        "with its vLLM integration before running MODEL_SOURCE=maxtext."
    ) from exc

  maxtext_vllm_adapter.register()
  logging.info(
      "Registered %s with vLLM.", models.MAXTEXT_VLLM_ARCHITECTURE
  )


def _create_vanilla_worker(args, tokenizer):
  """Creates a vanilla sampler rollout worker instance."""
  from tunix.experimental.rollout import (  # pylint: disable=g-import-not-at-top
      vanilla_sampler_adapter,
  )
  from tunix.experimental.weight_sync import (  # pylint: disable=g-import-not-at-top
      raiden_weight_sync_delegate,
  )
  from tunix.experimental.weight_sync import (  # pylint: disable=g-import-not-at-top
      weight_sync,
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
  vllm_sampler = _import_vllm_sampler()
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

  rollout_mesh = _create_rollout_mesh(args)
  is_maxtext = models.is_maxtext_source(args.model_source)
  if is_maxtext:
    _register_maxtext_vllm_adapter()

  logging.info("Creating vLLM mapping config...")
  mapping_config = (
      mappings_lib.MappingConfig()
      if is_maxtext
      else mappings_lib.MappingConfig(
          lora_to_hf_mappings=mapping_vllm_jax.LORA_TO_HF_MAPPINGS
      )
  )
  vllm_model = (
      _tokenizer_path(args) if is_maxtext else args.model_dir or args.model_id
  )
  max_model_len = args.max_prompt_length + args.max_response_length
  engine_kwargs = {
      "model": vllm_model,
      "max_model_len": max_model_len,
  }
  additional_config = {}
  if is_maxtext:
    engine_kwargs.update({
        "tokenizer": _tokenizer_path(args),
        "dtype": args.maxtext_dtype,
        "hf_overrides": {
            "architectures": [models.MAXTEXT_VLLM_ARCHITECTURE]
        },
    })
    additional_config = models.maxtext_vllm_additional_config(
        mesh=rollout_mesh,
        max_prompt_length=args.max_prompt_length,
        max_response_length=args.max_response_length,
        model_name=args.model_name,
        model_dir=args.model_dir,
        dtype=args.maxtext_dtype,
    )
  logging.info(
      "Creating vLLM config for model=%s mesh=%s tensor_parallel_size=%d "
      "data_parallel_size=%d max_model_len=%d model_source=%s...",
      vllm_model,
      rollout_mesh,
      args.mesh_tp,
      args.mesh_fsdp,
      max_model_len,
      args.model_source,
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
      # MaxText's vLLM inference config uses singleton attn_dp mesh axes in its
      # logical axis rules. This asks tpu-inference to materialize those axes.
      enable_dp_attention=is_maxtext,
      return_logprobs=True,
      init_with_random_weights=args.vllm_init_with_random_weights,
      lora_config=lora_config,
      mapping_config=mapping_config,
      additional_config=additional_config,
      engine_kwargs=engine_kwargs,
  )
  setattr(vllm_config, "weight_sync_mode", args.weight_sync_mode)
  sampler_adapter = inprocess_vllm_sampler_adapter.InprocessVllmSamplerAdapter(
      server_id=args.worker_id,
      tokenizer=tokenizer,
      config=vllm_config,
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
  if args.sampler == "vanilla" and models.is_maxtext_source(args.model_source):
    raise ValueError(
        "MODEL_SOURCE=maxtext is not supported with SAMPLER=vanilla. The "
        "MaxText Tunix adapter is intended for trainer/full-forward use and "
        "MaxText vLLM rollout, but the vanilla sampler runs a compiled "
        "autoregressive decode loop that is incompatible with MaxText's "
        "mutable NNX state updates. Use SAMPLER=inprocess_vllm for MaxText "
        "rollout, or use MODEL_SOURCE=safetensors with SAMPLER=vanilla."
    )
  if context:
    context.jax.initialize()
  os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
  os.environ.setdefault("VLLM_TPU_RPA_VERSION", "2")
  os.environ.setdefault("DISABLE_MOSAIC_ATTN", "1")
  os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
  if models.is_maxtext_source(args.model_source):
    os.environ.setdefault("NEW_MODEL_DESIGN", "1")
  if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
  logging.info("Repo root inserted into sys.path: %s", REPO_ROOT)

  from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

  tokenizer_path = _tokenizer_path(args)
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
