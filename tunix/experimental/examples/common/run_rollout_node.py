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

"""Rollout worker process runner shared by distributed RL examples."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import os
import pickle
import signal
import sys
from typing import Any

from tunix.experimental.examples.common import models
from tunix.experimental.weight_sync import raiden_preload
from tunix.experimental.weight_sync import weight_sync as weight_sync_lib
from tunix.rl.agentic.parser.chat_template_parser import parser as chat_parser_lib
from tunix.utils import maxtext_utils

# Import Raiden before any other libraries to ensure correct JAX compilation.
raiden_preload.import_raiden()

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
DEFAULT_REGISTRY_MODULE = "tunix.experimental.examples.math_gsm8k_dist.gsm8k"
DEFAULT_ENV_NAME = "gsm8kenv"
DEFAULT_AGENT_NAME = "gsm8kagent"


def _import_vllm_sampler():
  logging.info(
      "Importing tunix.generate.vllm_sampler before rollout adapters..."
  )
  vllm_sampler = importlib.import_module("tunix.generate.vllm_sampler")
  logging.info("Finished importing tunix.generate.vllm_sampler.")
  return vllm_sampler


def _chat_parser_for(
    model_id: str,
    tokenizer: Any,
    mode: str = "auto",
    *,
    enable_thinking: bool = False,
):
  """Selects the chat parser: `raw` text, or the model family's template.

  Args:
    model_id: Model name used to pick the family-specific template parser.
    tokenizer: Tokenizer handed to the parser.
    mode: Either `auto` (model family's template) or `raw` (no template).
    enable_thinking: Whether the template parser opens a thinking block.

  Returns:
    The chat parser to use for this rollout worker.
  """
  if mode == "raw":
    return chat_parser_lib.RawTextParser(tokenizer)
  name = model_id.lower()
  for family, parser_cls in CHAT_PARSERS.items():
    if family in name:
      return parser_cls(tokenizer, enable_thinking=enable_thinking)
  return chat_parser_lib.DefaultChatTemplateParser(
      tokenizer, enable_thinking=enable_thinking
  )


def _str2bool(v: str | bool) -> bool:
  """Converts string representations of booleans to bool."""
  if isinstance(v, bool):
    return v
  if v.lower() in ("yes", "true", "t", "y", "1"):
    return True
  if v.lower() in ("no", "false", "f", "n", "0"):
    return False
  raise argparse.ArgumentTypeError(f"Boolean value expected, got {v}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
  """Parses command line arguments for the rollout worker process."""
  parser = argparse.ArgumentParser(description="Distributed rollout worker")
  parser.add_argument("--port", type=int, default=20001)
  parser.add_argument("--worker_id", type=str, default="vllm-rollout-0")
  parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
  parser.add_argument(
      "--model_dir", type=str, default=os.getenv("MODEL_DIR", "")
  )
  parser.add_argument("--tokenizer_path", type=str, default="")
  parser.add_argument("--mesh_fsdp", type=int, default=1)
  parser.add_argument("--mesh_tp", type=int, default=2)
  # Expert parallelism for the SAMPLER. The 35b convergence study found this
  # is the axis that moves oob_ratio (78.12% -> 53.12% per-seq); trainer-side
  # expert parallelism measured as no impact. EP is carved out of the same
  # device budget, so mesh_fsdp * mesh_tp * mesh_ep must equal the device
  # count.
  parser.add_argument("--mesh_ep", type=int, default=1)
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
      "--chat_parser",
      type=str,
      default=os.getenv("CHAT_PARSER", "auto"),
      choices=["auto", "raw"],
      help=(
          "auto: the model family's chat template parser (Qwen/Llama/Gemma);"
          " raw: feed message contents verbatim with no template, for"
          " completion-style prompts the model is meant to continue."
      ),
  )
  parser.add_argument(
      "--debug",
      action="store_true",
      help="Enable debug logging for rollout worker.",
  )
  parser.add_argument(
      "--registry_module",
      type=str,
      default=os.getenv("ROLLOUT_REGISTRY_MODULE", DEFAULT_REGISTRY_MODULE),
      help=(
          "Module imported at startup to register rollout env/agent classes."
      ),
  )
  parser.add_argument(
      "--env_name",
      type=str,
      default=os.getenv("ROLLOUT_ENV_NAME", DEFAULT_ENV_NAME),
      help="Registered rollout environment name.",
  )
  parser.add_argument(
      "--agent_name",
      type=str,
      default=os.getenv("ROLLOUT_AGENT_NAME", DEFAULT_AGENT_NAME),
      help="Registered rollout agent name.",
  )
  parser.add_argument(
      "--agent_config_json",
      type=str,
      default=os.getenv("ROLLOUT_AGENT_CONFIG_JSON", "{}"),
      help="JSON object passed to the registered agent constructor.",
  )
  parser.add_argument(
      "--max_concurrency",
      type=int,
      default=int(os.getenv("ROLLOUT_MAX_CONCURRENCY", "64")),
      help="Maximum concurrent trajectory collections inside this worker.",
  )
  parser.add_argument(
      "--enable_thinking",
      action=argparse.BooleanOptionalAction,
      default=False,
      help="Enable the model family's thinking chat-template mode.",
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
  parser.add_argument(
      "--prefuse_moe_weights",
      type=_str2bool,
      default=True,
      nargs="?",
      const=True,
      help="Prefuse MoE weights for MaxText inference in vLLM.",
  )
  parser.add_argument(
      "--enable_prefix_caching",
      type=_str2bool,
      default=False,
      nargs="?",
      const=True,
      help="Enable KV prefix caching in vLLM sampler.",
  )
  parser.add_argument(
      "--return_routed_experts",
      type=_str2bool,
      default=False,
      nargs="?",
      const=True,
      help=(
          "Record the MoE expert ids each sampled token was routed through so"
          " training can replay them instead of re-routing (MoE router"
          " replay). Only meaningful for MoE models."
      ),
  )
  parser.add_argument(
      "--tensor_parallel_size",
      type=int,
      default=None,
      help=(
          "Explicit tensor parallel size for vLLM sampler (defaults to"
          " mesh_tp)."
      ),
  )
  args = parser.parse_args(argv)
  _get_tensor_parallel_size(args)
  return args


def _get_tensor_parallel_size(args: argparse.Namespace) -> int:
  """Derives tensor_parallel_size from mesh_tp if not explicitly set."""
  tp = getattr(args, "tensor_parallel_size", None)
  if tp is None:
    tp = getattr(args, "sampler_mesh_tp", None) or getattr(args, "mesh_tp", 1)
    args.tensor_parallel_size = tp
    if tp > 1:
      logging.info("Auto-derived tensor_parallel_size=%d from mesh_tp", tp)
  return tp


def _agent_config(args: argparse.Namespace) -> dict[str, Any]:
  try:
    config = json.loads(args.agent_config_json or "{}")
  except json.JSONDecodeError as exc:
    raise ValueError("--agent_config_json must be a valid JSON object.") from exc
  if not isinstance(config, dict):
    raise ValueError("--agent_config_json must decode to a JSON object.")
  return config


def _rollout_config_kwargs(args: argparse.Namespace) -> dict[str, Any]:
  return {
      "weight_sync_mode": args.weight_sync_mode,
      "max_prompt_length": args.max_prompt_length,
      "max_tokens_to_generate": args.max_response_length,
      "temperature": 1.0,
      "top_p": 1.0,
      "return_logprobs": True,
      "return_routed_experts": args.return_routed_experts,
      "env_name": args.env_name,
      "agent_name": args.agent_name,
      "agent_config": _agent_config(args),
  }


def _create_rollout_mesh(args) -> Any:
  import jax  # pylint: disable=g-import-not-at-top
  from jax.experimental import mesh_utils  # pylint: disable=g-import-not-at-top
  from jax.sharding import Mesh  # pylint: disable=g-import-not-at-top

  # Expert parallelism is NOT an axis of *this* mesh: tpu-inference builds its
  # own mesh from `sharding_strategy`, and this mesh only supplies the device
  # list. But EP is a real, device-consuming axis over there --
  # ShardingConfigManager computes
  #     total_devices = prod(tensor, expert, sequence, data, attn_dp, ...)
  # and asserts it equals len(device_indexes) (tpu_inference/layers/common/
  # sharding.py:199-202). Since device_indexes is this whole mesh, the product
  # dp * tp * ep must equal mesh_fsdp * mesh_tp. So ep > 1 only works if dp or
  # tp is divided down -- see the `data_parallel_size=-1` below, which makes
  # resolve_parallelism_sizes() infer dp = devices // (tp * ep).
  shape = (args.mesh_fsdp, args.mesh_tp)
  if args.mesh_fsdp * args.mesh_tp != jax.device_count():
    raise ValueError(
        "Rollout mesh dimensions must match visible device count: "
        f"mesh_fsdp={args.mesh_fsdp} mesh_tp={args.mesh_tp} "
        f"device_count={jax.device_count()}"
    )

  mesh_ep = getattr(args, "mesh_ep", 1) or 1
  if jax.device_count() % mesh_ep != 0:
    raise ValueError(
        f"mesh_ep={mesh_ep} must divide device_count={jax.device_count()}"
    )

  devices = mesh_utils.create_device_mesh(shape, jax.devices())
  mesh = Mesh(devices, axis_names=("fsdp", "tp"))
  logging.info("Rollout mesh: %s (expert_parallelism=%d)", mesh, mesh_ep)
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
      **_rollout_config_kwargs(args),
  )
  sampler_adapter = vanilla_sampler_adapter.VanillaSamplerAdapter(
      server_id=args.worker_id,
      transformer=model,
      tokenizer=tokenizer,
      cache_config=args.max_prompt_length + args.max_response_length,
      config=config,
  )

  rollout_tokenizer = tokenizer_adapter_lib.TokenizerAdapter(tokenizer)
  chat_parser = _chat_parser_for(
      args.model_id or args.model_name,
      tokenizer,
      args.chat_parser,
      enable_thinking=args.enable_thinking,
  )
  return rollout_worker.RolloutWorker(
      worker_id=args.worker_id,
      config=config,
      sampler=sampler_adapter,
      tokenizer=rollout_tokenizer,
      chat_parser=chat_parser,
      max_concurrency=args.max_concurrency,
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
  chat_parser = _chat_parser_for(
      args.model_id or args.model_name,
      tokenizer,
      args.chat_parser,
      enable_thinking=args.enable_thinking,
  )
  logging.info("Creating RolloutWorker wrapper...")
  return rollout_worker.RolloutWorker(
      worker_id=args.worker_id,
      config=rollout_config,
      sampler=sampler_adapter,
      tokenizer=rollout_tokenizer,
      chat_parser=chat_parser,
      max_concurrency=args.max_concurrency,
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
      **mapping_vllm_jax.VLLM_JAX_MAPPING
  )
  vllm_model = (
      args.model_dir
      if (
          args.model_dir
          and os.path.isdir(args.model_dir)
          and bool(os.listdir(args.model_dir))
      )
      else args.model_id
  )
  max_model_len = args.max_prompt_length + args.max_response_length

  multihost_backend = os.environ.get("TPU_MULTIHOST_BACKEND", "")
  if multihost_backend:
    assert (
        multihost_backend != "ray" or args.mesh_tp is not None
    ), "Must set --mesh_tp when using Ray backend."

  engine_kwargs = {
      "model": vllm_model,
      "max_model_len": max_model_len,
      "enable_prefix_caching": args.enable_prefix_caching,
  }
  # Select MaxText's `MaxTextForCausalLM` as rollout model.
  # `additional_config` must be set on VllmConfig (not engine_kwargs): the
  # sampler overwrites args["additional_config"] from the VllmConfig field.
  maxtext_additional_config = None
  if args.maxtext_model_name:
    logging.info(
        "Loading MaxText model %r natively via maxtext_vllm_adapter's"
        " MaxTextForCausalLM (architectures override).",
        args.maxtext_model_name,
    )
    engine_kwargs["hf_overrides"] = dict(
        maxtext_utils.VLLM_MAXTEXT_HF_OVERRIDES
    )
    maxtext_additional_config = (
        maxtext_utils.build_vllm_maxtext_additional_config(
            args.maxtext_model_name,
            attention=args.maxtext_attention,
            prefuse_moe_weights=args.prefuse_moe_weights,
        )
    )

  if multihost_backend:
    engine_kwargs["distributed_executor_backend"] = multihost_backend
  server_mode = True if multihost_backend else None
  rollout_mesh = None if multihost_backend else _create_rollout_mesh(args)

  tp_size = _get_tensor_parallel_size(args)
  mesh_ep = getattr(args, "mesh_ep", 1) or 1
  # tpu-inference asserts dp * tp * ep == len(device_indexes), and
  # device_indexes is the whole rollout mesh (mesh_fsdp * mesh_tp). Passing
  # mesh_fsdp as dp therefore overshoots by exactly a factor of ep. Pass -1 so
  # resolve_parallelism_sizes() infers dp = devices // (tp * ep) instead.
  # When ep == 1 the inferred value equals mesh_fsdp, so this is a no-op there;
  # we keep the explicit value anyway to avoid changing the ep == 1 path.
  dp_arg = -1 if mesh_ep > 1 else args.mesh_fsdp
  logging.info(
      "Creating vLLM config for model=%s mesh=%s tensor_parallel_size=%d "
      "data_parallel_size=%s expert_parallel_size=%d max_model_len=%d...",
      vllm_model,
      rollout_mesh,
      tp_size,
      "infer" if dp_arg == -1 else dp_arg,
      mesh_ep,
      max_model_len,
  )
  lora_config = None
  if args.use_lora:
    lora_config = {
        "max_lora_rank": args.lora_rank,
        "max_loras": 1,
    }
  # The sampler is normally DUMMY-INITIALISED: vllm_sampler.py:333 sets
  # load_format="dummy" whenever init_with_random_weights is true (the default),
  # because "model weights are synced from trainer later on". So in the steady
  # state every sampler weight arrives over Raiden, and the sampler's own
  # checkpoint is irrelevant.
  #
  # That is only safe if Raiden's coverage is COMPLETE. Any tensor Raiden does
  # not bind keeps its random initialisation, and nothing warns: the engine runs,
  # generates plausible text, and quietly samples from a different policy than
  # the trainer scores. The symptom we are chasing is exactly that shape --
  # KL(sampler||trainer) ~= 0.07 nats/token at step 0, before any optimizer
  # update, against 0.003 when one checkpoint is loaded into both engines
  # offline. (The mean log ratio is -KL by construction, which is why
  # seq_geomean is below 1.0 in every run and no sequence ever lands above the
  # band.)
  #
  # Setting ROLLOUT_MAXTEXT_CKPT loads real weights first, which turns the
  # question into a single-variable experiment:
  #   seq_geomean unchanged (~0.93) -> Raiden coverage is complete; the gap is
  #     engine numerics at production's sharding, not weight transport.
  #   seq_geomean jumps toward 1.0  -> Raiden was leaving tensors at their random
  #     init, and those are the tensors to go find.
  #
  # init_with_random_weights must be false for the checkpoint to be read at all:
  # under load_format="dummy" the MaxText adapter force-nulls
  # load_parameters_path (maxtext_vllm_adapter/adapter.py:97-102), which then
  # fails MaxTextConfig's pydantic string validation.
  #
  # Costs a slower bootstrap (~70 GB from GCS). Unset the env to restore the
  # fast dummy-init path exactly.
  rollout_ckpt = os.environ.get("ROLLOUT_MAXTEXT_CKPT", "")
  vllm_config = vllm_sampler.VllmConfig(
      server_mode=server_mode,
      mesh=rollout_mesh,
      tensor_parallel_size=tp_size,
      data_parallel_size=dp_arg,
      expert_parallel_size=mesh_ep,
      return_logprobs=True,
      return_routed_experts=args.return_routed_experts,
      init_with_random_weights=not rollout_ckpt,
      lora_config=lora_config,
      mapping_config=mapping_config,
      additional_config=maxtext_additional_config,
      engine_kwargs=engine_kwargs,
  )
  logging.info(
      "rollout sampler init_with_random_weights=%s (ROLLOUT_MAXTEXT_CKPT=%r)",
      not rollout_ckpt,
      rollout_ckpt,
  )
  sampler_adapter = inprocess_vllm_sampler_adapter.InprocessVllmSamplerAdapter(
      server_id=args.worker_id,
      tokenizer=tokenizer,
      config=vllm_config,
      weight_sync_mode=args.weight_sync_mode,
  )
  config = rollout_worker.RolloutConfig(
      sampler_type="inprocess_vllm",
      rollout_vllm_model_version=vllm_model,
      **_rollout_config_kwargs(args),
  )
  return sampler_adapter, config


def _create_vllm_sampler(args):
  """Creates a vLLM sampler rollout worker instance."""
  from tunix.experimental.rollout import (  # pylint: disable=g-import-not-at-top
      vllm_sampler_adapter,
  )
  from tunix.experimental.worker import (  # pylint: disable=g-import-not-at-top
      rollout_worker,
  )
  from vllm.engine.arg_utils import (  # pylint: disable=g-import-not-at-top
      AsyncEngineArgs,
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
  max_model_len = args.max_prompt_length + args.max_response_length
  tp_size = _get_tensor_parallel_size(args)
  logging.info(
      "Creating vLLM RLVllmSampler config for model=%s tensor_parallel_size=%d "
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
      enable_prefix_caching=args.enable_prefix_caching,
  )
  if args.maxtext_model_name:
    logging.info(
        "Loading MaxText model %r natively via maxtext_vllm_adapter's"
        " MaxTextForCausalLM (architectures override).",
        args.maxtext_model_name,
    )
    engine_kwargs["hf_overrides"] = dict(
        maxtext_utils.VLLM_MAXTEXT_HF_OVERRIDES
    )
    engine_kwargs["additional_config"] = (
        maxtext_utils.build_vllm_maxtext_additional_config(
            args.maxtext_model_name,
            attention=args.maxtext_attention,
            prefuse_moe_weights=args.prefuse_moe_weights,
        )
    )
  engine_args = AsyncEngineArgs(**engine_kwargs)  # pytype: disable=bad-argument-type  # type: ignore[arg-type]
  sampler_adapter = vllm_sampler_adapter.VllmSamplerAdapter(  # pytype: disable=bad-instantiation  # type: ignore[abstract]
      server_id=args.worker_id,
      engine_args=engine_args,
      model_name=vllm_model,
      weight_sync_mode=args.weight_sync_mode,
  )
  config = rollout_worker.RolloutConfig(
      sampler_type="vllm",
      rollout_vllm_model_version=vllm_model,
      **_rollout_config_kwargs(args),
  )
  return sampler_adapter, config


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

  if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
  logging.info("Repo root inserted into sys.path: %s", REPO_ROOT)

  logging.info("Importing rollout registry module: %s", args.registry_module)
  importlib.import_module(args.registry_module)

  if context and args.sampler == "vanilla":
    context.jax.initialize()
  os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
  os.environ.setdefault("VLLM_TPU_RPA_VERSION", "2")
  os.environ.setdefault("DISABLE_MOSAIC_ATTN", "1")
  os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
  if args.maxtext_model_name:
    os.environ.setdefault("NEW_MODEL_DESIGN", "1")

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

    # Shut down gracefully on SIGTERM/SIGINT.
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
      try:
        loop.add_signal_handler(sig, stop_event.set)
      except NotImplementedError:
        pass

    try:
      await stop_event.wait()
    except asyncio.CancelledError:
      pass
    finally:
      logging.info("Draining rollout worker...")
      try:
        # Cancel in-flight trajectory collections
        worker_service.stop()
        if hasattr(worker_service.sampler, "stop"):
          # Shut down vLLM engine
          await worker_service.sampler.stop()
        logging.info("Rollout worker drained.")
      except Exception:
        logging.exception("Failed to drain rollout worker cleanly.")
      await server.stop_serving()

  asyncio.run(grpc_server_main())


if __name__ == "__main__":
  main(sys.argv[1:])
