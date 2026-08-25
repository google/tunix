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
from tunix.experimental.rollout import legacy_vllm_sampler_adapter
from tunix.experimental.rollout import raiden_sampler_adapter
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
  return chat_parser_lib.DefaultChatTemplateParser(tokenizer, enable_thinking=False)

def _parse_args(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="vLLM rollout worker process")
  parser.add_argument("--port", type=int, default=20001)
  parser.add_argument("--worker_id", type=str, default="vllm-rollout-0")
  parser.add_argument(
      "--worker_index",
      type=int,
      default=0,
      help=(
          "Distinguishes this replica's Raiden work-unit registration from"
          " every other rollout replica's. raiden_worker_sync.py derives"
          " job_replica_id from this (falling back to '' when it's 0), so"
          " leaving every replica at the class default of 0 makes them all"
          " register under the same empty job_replica_id -- the second"
          " replica to register silently overwrites the first's endpoint in"
          " the coordinator's registry, and only one replica actually"
          " receives each subsequent weight-sync transfer. Must be nonzero"
          " and unique per replica; k8s_launcher.sh passes replica_index+1."
      ),
  )
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
      default=os.getenv("SAMPLER", "legacy_vllm"),
      choices=["legacy_vllm", "vanilla"],
  )
  parser.add_argument("--tensor_parallel_size", type=int, default=4)
  parser.add_argument(
      "--sampler_type",
      type=str,
      default="vllm",
      choices=["vllm", "legacy_vllm"],
  )
  parser.add_argument(
      "--maxtext_model_name",
      type=str,
      default="",
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
      default="",
      help=(
          "Override MaxText's inference attention kernel (e.g."
          " vllm_batched_rpa instead of the maxtext_vllm_adapter default"
          " vllm_rpa). Passed through maxtext_config; USE_BATCHED_RPA_KERNEL"
          " must also be set as a pod env var for vllm_batched_rpa, since"
          " tpu_inference.layers.common.attention_interface reads it at"
          " module-import time -- MaxText's own os.environ set (on model"
          " construction) happens too late for that check."
      ),
  )
  return parser.parse_args(argv)


def _create_rollout_mesh() -> Mesh:
  shape = (1, jax.device_count())
  devices = mesh_utils.create_device_mesh(shape, jax.devices())
  return Mesh(devices, axis_names=("fsdp", "tp"))

def _create_vanilla_worker(args, tokenizer):
  logging.info("Creating native sampler on the rollout mesh...")
  mesh = _create_rollout_mesh()
  with mesh:
    model = models.create_model(
        args.model_name, args.model_dir or args.model_id, mesh
    )
  sampler_adapter = raiden_sampler_adapter.RaidenSamplerAdapter(
      server_id=args.worker_id,
      transformer=model,
      tokenizer=tokenizer,
      cache_config=args.max_prompt_length + args.max_response_length,
  )

  rollout_tokenizer = tokenizer_adapter_lib.TokenizerAdapter(tokenizer)
  chat_parser = chat_parser_lib.QwenChatTemplateParser(
      tokenizer, enable_thinking=False
  )
  # TODO: select the chat template parser by model family instead of hardcoding. 
  config = rollout_worker.RolloutConfig(
      sampler_type="raiden_vanilla",
      max_prompt_length=args.max_prompt_length,
      max_tokens_to_generate=args.max_response_length,
      temperature=1.0,
      top_p=1.0,
      return_logprobs=True,
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


def _create_vllm_worker(args, tokenizer):
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

  if getattr(args, "sampler_type", "vllm") == "vllm":
    from vllm.engine.arg_utils import AsyncEngineArgs  # pylint: disable=g-import-not-at-top
    from tunix.experimental.rollout import vllm_sampler_adapter  # pylint: disable=g-import-not-at-top

    tp_size = args.tensor_parallel_size
    logging.info(
        "Creating vLLM RLVllmSampler config for model=%s tp_size=%d "
        "max_model_len=%d...",
        vllm_model,
        tp_size,
        max_model_len,
    )
    engine_kwargs = dict(
        model=vllm_model,
        tokenizer=args.tokenizer_path or vllm_model,
        tensor_parallel_size=tp_size,
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
      # Matches maxtext/src/maxtext/trainers/post_train/rl/scripts/run_qwen3_30b_rl.sh's
      # working vllm_additional_config for Qwen3-30B-A3B: enable_dp_attention
      # is explicitly False here (attn DP is a trainer-side setting, not a
      # rollout one), allow_split_physical_axes governs how mesh axes can
      # split/share across the physical device topology.
      #
      # Deliberately NOT setting prefuse_moe_weights=True here (unlike that
      # script): it fuses wi_0/wi_1 into a single tensor on the rollout side,
      # but the trainer (run_trainer_node.py) never sets it, so Raiden's
      # weight-sync preflight fails with "source variable [...]['wi_0'].value
      # has no destination counterpart" -- the two sides must agree on this
      # since it's a wire-format shape difference, not just a perf knob.
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
        worker_index=args.worker_index,
    )
    rollout_config = rollout_worker.RolloutConfig(
        sampler_type="vllm",
        max_prompt_length=args.max_prompt_length,
        max_tokens_to_generate=args.max_response_length,
        temperature=1.0,
        top_p=1.0,
        return_logprobs=True,
        rollout_vllm_model_version=vllm_model,
    )
  else:
    logging.info("Creating vLLM mapping config...")
    mapping_config = mappings_lib.MappingConfig(
        lora_to_hf_mappings=mapping_vllm_jax.LORA_TO_HF_MAPPINGS
    )
    rollout_mesh = _create_rollout_mesh()
    logging.info(
        "Creating legacy vLLM config for model=%s mesh=%s tensor_parallel_size=%d "
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
    rollout_config = rollout_worker.RolloutConfig(
        sampler_type="legacy_vllm",
        max_prompt_length=args.max_prompt_length,
        max_tokens_to_generate=args.max_response_length,
        temperature=1.0,
        top_p=1.0,
        return_logprobs=True,
        rollout_vllm_model_version=vllm_model,
    )

  rollout_tokenizer = tokenizer_adapter_lib.TokenizerAdapter(tokenizer)
  chat_parser = chat_parser_lib.QwenChatTemplateParser(
      tokenizer, enable_thinking=False
  )
  logging.info("Creating RolloutWorker wrapper...")
  return rollout_worker.RolloutWorker(
      worker_id=args.worker_id,
      config=rollout_config,
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

  if context and getattr(args, "sampler_type", "vllm") != "vllm":
    context.jax.initialize()
  os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
  os.environ.setdefault("VLLM_TPU_RPA_VERSION", "2")
  os.environ.setdefault("DISABLE_MOSAIC_ATTN", "1")
  if args.maxtext_model_name:
    # the MaxText adapter's logical_axis_rules need the 7-axis mesh that
    # tpu-inference only builds under NEW_MODEL_DESIGN; must be set before
    # vllm/tpu_inference is imported, hence here rather than in _create_vllm_worker
    os.environ.setdefault("NEW_MODEL_DESIGN", "1")
  if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
  logging.info("Repo root inserted into sys.path: %s", REPO_ROOT)


  tokenizer_path = args.tokenizer_path or args.model_dir or args.model_id
  logging.info("Loading tokenizer from %s...", tokenizer_path)
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
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

    if args.sampler != "vanilla":
      # A multihost rollout jobset spans multiple pods, but the orchestrator
      # only ever dispatches sample requests to one of them (its discovery
      # future only ever resolves to the first-registered address). The
      # sampler's underlying engine (and its JAX/TPU backend rendezvous)
      # would otherwise only be started lazily on that one pod's first
      # request, which hangs forever waiting for the other pod's hosts to
      # join a JAX-distributed group they were never told to enter. Starting
      # every pod's engine eagerly here, before registering, mirrors how the
      # MaxText trainer eagerly builds its engine at node startup so every
      # pod of a multihost jobset joins JAX-distributed together.
      logging.info("Eagerly starting sampler engine...")
      await worker_service.sampler.start()
      logging.info("Sampler engine started.")

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
