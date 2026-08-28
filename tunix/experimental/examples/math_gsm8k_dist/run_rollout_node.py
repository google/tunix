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

# Load Raiden's native extension before vLLM/torch pull in their own. Importing
# it afterwards aborts the process with `free(): invalid pointer` during
# `tpu_inference.rl.raiden_worker_sync`, which imports this same module. Keep
# this above every other import; it is load-bearing, not stylistic.
try:
  from tpu_sync.api.jax import weight_synchronizer as _  # pylint: disable=unused-import
except ImportError:
  pass

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
      choices=["inprocess_vllm", "vanilla", "vllm"],
      help=(
          "'vllm' runs tpu-inference's RLVllmSampler, which binds Raiden"
          " inside the EngineCore subprocess where the live TPU arrays are."
          " Pair it with --maxtext_model_name: only the MaxText-native"
          " rollout produces tensor names the MaxText trainer can match."
      ),
  )
  parser.add_argument(
      "--worker_index",
      type=int,
      default=0,
      help=(
          "Distinguishes this replica's Raiden work-unit registration from"
          " every other rollout replica's. raiden_worker_sync.py derives"
          " job_replica_id from this (falling back to '' when it's 0), so"
          " leaving every replica at the default of 0 makes them all register"
          " under the same empty job_replica_id -- the second replica to"
          " register silently overwrites the first's endpoint, and only one"
          " actually receives each subsequent transfer. Must be nonzero and"
          " unique per replica when running more than one."
      ),
  )
  parser.add_argument(
      "--tensor_parallel_size",
      type=int,
      default=None,
      help=(
          "Number of chips vLLM shards over; defaults to the chips this"
          " process was given. Resolved after parsing by"
          " _default_tensor_parallel_size() rather than as a default= here,"
          " because an eager jax.device_count() opens the TPU in this parent"
          " process while the argument parser is still being built. The"
          " --sampler=vllm path then hands those same chips to a separate"
          " EngineCore process, which cannot reopen them"
          " ('open(/dev/vfio/N): Device or resource busy'), or inherits them"
          " half-owned via fork and hangs on its first compile."
      ),
  )
  parser.add_argument(
      "--maxtext_model_name",
      type=str,
      default=os.getenv("MAXTEXT_MODEL_NAME", ""),
      help=(
          "MaxText model_name (e.g. qwen3-0.6b). When set, loads the model"
          " via maxtext_vllm_adapter's MaxTextForCausalLM instead of"
          " tpu-inference's own reimplementation, so trainer and rollout"
          " share Raiden weight-sync tensor names."
      ),
  )
  parser.add_argument(
      "--maxtext_attention",
      type=str,
      default=os.getenv("MAXTEXT_ATTENTION", ""),
      help=(
          "Override MaxText's inference attention kernel (e.g."
          " vllm_batched_rpa instead of the maxtext_vllm_adapter default"
          " vllm_rpa). Passed through maxtext_config; USE_BATCHED_RPA_KERNEL"
          " must also be set as an env var for vllm_batched_rpa, since"
          " tpu_inference.layers.common.attention_interface reads it at"
          " module-import time."
      ),
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


def _default_tensor_parallel_size() -> int:
  """Chip count for this process, preferring env over touching JAX.

  The launcher already tells us which chips we own, so read that instead of
  calling jax.device_count(); see --tensor_parallel_size on why initializing
  the TPU backend in this process breaks the --sampler=vllm path. The
  jax.device_count() fallback is only reached when neither variable is set,
  and the other sampler paths open the TPU here anyway.
  """
  for var in ("TPU_VISIBLE_CHIPS", "TPU_VISIBLE_DEVICES"):
    chips = [c for c in (os.getenv(var) or "").split(",") if c.strip()]
    if chips:
      return len(chips)
  return jax.device_count()


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
      weight_sync_mode=args.weight_sync_mode,
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


def _create_rl_vllm_worker(args, tokenizer):
  """Creates a tpu-inference RLVllmSampler rollout worker instance.

  Unlike the inprocess_vllm path, Raiden binds inside the EngineCore
  subprocess that owns the live TPU arrays, so only plain metadata crosses
  the RPC boundary.
  """
  from vllm.engine.arg_utils import AsyncEngineArgs  # pylint: disable=g-import-not-at-top
  from tunix.experimental.rollout import vllm_sampler_adapter  # pylint: disable=g-import-not-at-top

  vllm_model = args.model_dir or args.model_id
  max_model_len = args.max_prompt_length + args.max_response_length
  logging.info(
      "Creating vLLM RLVllmSampler config for model=%s tp_size=%d"
      " max_model_len=%d...",
      vllm_model,
      args.tensor_parallel_size,
      max_model_len,
  )
  engine_kwargs = dict(
      model=vllm_model,
      tokenizer=args.tokenizer_path or vllm_model,
      tensor_parallel_size=args.tensor_parallel_size,
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
    # MaxText's configs/inference/vllm.yml declares
    # mesh_axes = [data, attn_dp, model, expert, attn_dp_expert] and writes
    # logical_axis_rules over them, but tpu-inference only builds that mesh
    # under NEW_MODEL_DESIGN; its default 2D mesh is just ('data', 'model'),
    # so model loading dies with "Resource axis: attn_dp ... is not found in
    # mesh". MaxText sets this itself for its own vLLM entrypoints (see
    # tests/post_training/integration/single_host_train_rl_test.py); we are
    # the "direct vllm serve" case that has to set it. Scoped to the MaxText
    # branch so the HF/Mode-1 path keeps the 2D mesh its MoE kernel wants.
    os.environ.setdefault("NEW_MODEL_DESIGN", "1")
    # enable_dp_attention is explicitly False: attention DP is a trainer-side
    # setting, not a rollout one. allow_split_physical_axes governs how mesh
    # axes may split/share across the physical device topology.
    #
    # Deliberately NOT setting prefuse_moe_weights=True: it fuses wi_0/wi_1
    # into a single tensor on the rollout side, but the trainer never sets
    # it, so weight-sync preflight fails with "source variable
    # [...]['wi_0'].value has no destination counterpart" -- the two sides
    # must agree, since it is a wire-format shape difference, not a perf knob.
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

  sampler_adapter = vllm_sampler_adapter.VllmSamplerAdapter(
      server_id=args.worker_id,
      engine_args=AsyncEngineArgs(**engine_kwargs),
      model_name=vllm_model,
      worker_index=args.worker_index,
  )
  rollout_tokenizer = tokenizer_adapter_lib.TokenizerAdapter(tokenizer)
  chat_parser = chat_parser_lib.QwenChatTemplateParser(
      tokenizer, enable_thinking=False
  )
  logging.info("Creating RolloutWorker wrapper...")
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
  if args.tensor_parallel_size is None:
    args.tensor_parallel_size = _default_tensor_parallel_size()
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
    elif args.sampler == "vllm":
      worker_service = _create_rl_vllm_worker(args, tokenizer)
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
