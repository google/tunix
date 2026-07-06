#!/usr/bin/env python3
"""Clean DeepSWE/R2E-Gym data by filtering degenerate reward groups.

For each dataset example, this script runs ``num_generations`` independent SWE
rollouts. If all attempts in the group receive reward 1, or all attempts receive
reward 0, the example is removed from the cleaned dataset. Mixed-reward groups
are kept because they provide useful GRPO/RLOO training signal.

The rollout/model/env setup intentionally mirrors ``eval_deepswe.py`` and the
DeepSWE training recipe, but this script never runs actor training.

Example:

  python3 examples/deepswe/clean_deepswe_dataset.py \
    --dataset_name=R2E-Gym/R2E-Gym-Subset \
    --dataset_split=train \
    --num_generations=8 \
    --output_dir=/tmp/deepswe_clean
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import os
import sys
import threading
import time
from typing import Any


def _bootstrap_paths() -> None:
  script_dir = os.path.dirname(os.path.abspath(__file__))
  repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
  workspace_root = os.path.dirname(repo_root)
  candidate_roots = [
      script_dir,
      repo_root,
      workspace_root,
      os.path.join(repo_root, "tunix"),
      os.path.join(repo_root, "pathways-utils"),
      os.path.join(repo_root, "r2egym"),
      os.path.join(workspace_root, "tunix"),
      os.path.join(workspace_root, "pathways-utils"),
      os.path.join(workspace_root, "r2egym"),
      "/usr/github/rllm",
      "/usr/github/pathways-utils",
  ]
  for root in candidate_roots:
    if root and root not in sys.path:
      sys.path.insert(0, root)


_bootstrap_paths()

from datasets import load_dataset
from guarded_swe_env import GuardedSWEEnv
from huggingface_hub import snapshot_download
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from kubernetes import client
from kubernetes import config as k8s_config
import numpy as np
from r2egym_runtime_patch import apply_repoenv_kubernetes_watch_patch
from swe_agent import SWEAgent
from swe_env import SWEEnv
from transformers import AutoTokenizer
from tunix.generate import tokenizer_adapter as tok_adapter
from tunix.models.automodel import call_model_config
from tunix.models.qwen3 import params as params_lib
from tunix.rl.agentic import utils as agentic_utils
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.parser.chat_template_parser import parser
from tunix.rl.agentic.pipeline.rollout_orchestrator import RolloutOrchestrator
from tunix.rl.agentic.trajectory import trajectory_collect_engine
from tunix.sft import utils as sft_utils


logger = logging.getLogger("deepswe_data_clean")
Counter = collections.Counter


class PromptTooLongError(ValueError):
  """Raised when a prompt exceeds the model context limit before sampling."""


def _setup_logging(level: str) -> None:
  logging.basicConfig(
      stream=sys.stdout,
      level=getattr(logging, level.upper()),
      format="%(asctime)s - %(levelname)s - [%(name)s] %(message)s",
      datefmt="%Y-%m-%d %H:%M:%S",
      force=True,
  )


def _parse_args() -> argparse.Namespace:
  parser_arg = argparse.ArgumentParser(
      description=(
          "Clean DeepSWE/R2E-Gym data by removing examples whose"
          " num_generations rollouts all get reward 0 or all get reward 1."
      )
  )

  # Data.
  parser_arg.add_argument(
      "--dataset_name", type=str, default="R2E-Gym/R2E-Gym-Subset"
  )
  parser_arg.add_argument("--dataset_split", type=str, default="train")
  parser_arg.add_argument(
      "--dataset_cache",
      type=str,
      default=os.path.join(os.getcwd(), "dataset_cache"),
  )
  parser_arg.add_argument(
      "--output_dir",
      type=str,
      default=os.path.join(
          os.path.dirname(__file__),
          "data_clean_results",
          time.strftime("%Y%m%d_%H%M%S"),
      ),
  )
  parser_arg.add_argument(
      "--tasks_limit",
      type=int,
      default=0,
      help="Maximum number of candidate examples to process. 0 means all.",
  )
  parser_arg.add_argument(
      "--start_index",
      type=int,
      default=0,
      help="Start offset within the filtered candidate examples.",
  )
  parser_arg.add_argument(
      "--end_index",
      type=int,
      default=0,
      help="End offset within filtered candidates. 0 means no upper bound.",
  )
  parser_arg.add_argument(
      "--require_docker_image",
      action=argparse.BooleanOptionalAction,
      default=True,
      help="Filter out dataset rows without docker_image before cleaning.",
  )

  # Cleaning criterion.
  parser_arg.add_argument("--num_generations", type=int, default=8)
  parser_arg.add_argument(
      "--reward_tol",
      type=float,
      default=1e-6,
      help="Tolerance for treating rewards as exactly 0 or exactly 1.",
  )

  # Model / rollout.
  parser_arg.add_argument("--models_base_dir", type=str, default="models")
  parser_arg.add_argument("--model_version", type=str, default="Qwen3-32B")
  parser_arg.add_argument("--max_model_len", type=int, default=32768)
  parser_arg.add_argument("--max_prompt_length", type=int, default=4096)
  parser_arg.add_argument("--max_response_length", type=int, default=8192)
  parser_arg.add_argument("--max_concurrent", type=int, default=128)
  parser_arg.add_argument("--temperature", type=float, default=1.0)
  parser_arg.add_argument("--top_p", type=float, default=None)
  parser_arg.add_argument("--top_k", type=int, default=None)
  parser_arg.add_argument(
      "--rollout_engine",
      type=str,
      default="vllm",
      choices=["vanilla", "vllm", "sglang_jax"],
  )
  parser_arg.add_argument("--vllm_hbm_utilization", type=float, default=0.4)
  parser_arg.add_argument(
      "--vllm_init_random_weights",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser_arg.add_argument(
      "--vllm_server_mode",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser_arg.add_argument("--vllm_max_num_seqs", type=int, default=None)
  parser_arg.add_argument("--vllm_max_batched_tokens", type=int, default=None)
  parser_arg.add_argument(
      "--sglang_mem_fraction_static", type=float, default=0.4
  )
  parser_arg.add_argument(
      "--sglang_init_random_weights",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser_arg.add_argument("--sglang_max_running_requests", type=int, default=1)

  # Mesh.
  parser_arg.add_argument("--mesh_fsdp", type=int, default=1)
  parser_arg.add_argument(
      "--mesh_tp",
      type=int,
      default=None,
      help=(
          "TP size for eval/cleaning mesh. Defaults to min(num_devices,"
          " num_kv_heads)."
      ),
  )

  # Env.
  parser_arg.add_argument("--max_turns", type=int, default=50)
  parser_arg.add_argument("--episode_timeout_secs", type=float, default=3 * 60 * 60)
  parser_arg.add_argument("--step_timeout_secs", type=int, default=30 * 60)
  parser_arg.add_argument("--reward_timeout_secs", type=int, default=30 * 60)
  parser_arg.add_argument("--backend", type=str, default="kubernetes")
  parser_arg.add_argument(
      "--node_selector_key",
      type=str,
      default="cloud.google.com/gke-nodepool",
  )
  parser_arg.add_argument(
      "--node_selector_val", type=str, default="deepswe-cpu-pool"
  )
  parser_arg.add_argument(
      "--enable_guard", action=argparse.BooleanOptionalAction, default=False
  )
  parser_arg.add_argument(
      "--scaffold",
      type=str,
      default="r2egym",
      choices=["r2egym", "sweagent"],
  )
  parser_arg.add_argument(
      "--verbose_env", action=argparse.BooleanOptionalAction, default=False
  )
  parser_arg.add_argument(
      "--delete_image", action=argparse.BooleanOptionalAction, default=False
  )

  # Runtime.
  parser_arg.add_argument(
      "--skip_k8s_check", action=argparse.BooleanOptionalAction, default=False
  )
  parser_arg.add_argument(
      "--logging_level",
      type=str,
      default="INFO",
      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
  )
  return parser_arg.parse_args()


def _model_id(model_version: str) -> str:
  return model_version if "/" in model_version else f"Qwen/{model_version}"


def _model_name_for_config(model_version: str) -> str:
  return model_version.split("/", 1)[1] if "/" in model_version else model_version


def _jsonify_list_fields(entry: dict[str, Any]) -> dict[str, Any]:
  out = dict(entry)
  for key, value in out.items():
    if isinstance(value, list):
      out[key] = json.dumps(value)
  return out


def _json_default(value: Any):
  if isinstance(value, np.ndarray):
    return value.tolist()
  if isinstance(value, np.generic):
    return value.item()
  raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_dataset_jsonl(dataset, path: str) -> None:
  with open(path, "w", encoding="utf-8") as f:
    for entry in dataset:
      f.write(json.dumps(dict(entry), ensure_ascii=False, default=_json_default))
      f.write("\n")


def _as_float_reward(value: Any) -> float:
  if value is None:
    return 0.0
  try:
    return float(value)
  except (TypeError, ValueError):
    return 0.0


def _is_prompt_overflow_error(exc: Exception) -> bool:
  message = str(exc)
  return (
      "maximum input length" in message
      or "context length is only" in message
      or "Prompt too long before sampler call" in message
      or ("input_tokens" in message and "max_model_len" in message)
  )


class CleanTrajectoryCollectEngine(
    trajectory_collect_engine.TrajectoryCollectEngine
):
  """Trajectory engine that records prompt-overflow attempts as reward 0."""

  async def _one_step(self) -> bool:
    try:
      return await super()._one_step()
    except PromptTooLongError as exc:
      logger.warning(
          "[group=%s pair=%s instance=%s] terminating due to prompt overflow:"
          " %s",
          self.env.extra_kwargs.get("group_id"),
          self.env.extra_kwargs.get("pair_index"),
          self.env.entry.get("instance_id", "unknown"),
          exc,
      )
      self.agent.trajectory.status = (
          agent_types.TrajectoryStatus.MAX_CONTEXT_LIMIT_REACHED
      )
      self._skip_final_reward = True
      if self.agent.trajectory.steps:
        self.agent.trajectory.steps[-1].done = True
      return True

  async def _append_final_reward(self):
    if getattr(self, "_skip_final_reward", False):
      return
    await super()._append_final_reward()

  def compute_trajectory_reward(self):
    if getattr(self, "_skip_final_reward", False):
      self.agent.trajectory.reward = 0.0
      return self.agent.trajectory
    return super().compute_trajectory_reward()


class _CleanLoggingEnvMixin:
  """Adds group-level reset/step logs for data-clean debugging."""

  def reset(self):
    group_id = self.extra_kwargs.get("group_id")
    pair_index = self.extra_kwargs.get("pair_index")
    instance_id = self.entry.get("instance_id", "unknown")
    logger.info(
        "[group=%s pair=%s instance=%s] reset start",
        group_id,
        pair_index,
        instance_id,
    )
    t0 = time.time()
    obs, info = super().reset()
    logger.info(
        "[group=%s pair=%s instance=%s] reset end in %.1fs",
        group_id,
        pair_index,
        instance_id,
        time.time() - t0,
    )
    return obs, info

  def step(self, action):
    group_id = self.extra_kwargs.get("group_id")
    pair_index = self.extra_kwargs.get("pair_index")
    instance_id = self.entry.get("instance_id", "unknown")
    action_name = action
    if isinstance(action, str):
      action_name = action.split("\n", 1)[0][:120]
    logger.info(
        "[group=%s pair=%s instance=%s] env.step start step=%s action=%s",
        group_id,
        pair_index,
        instance_id,
        self.step_count + 1,
        action_name,
    )
    t0 = time.time()
    obs, reward, done, info = super().step(action)
    logger.info(
        "[group=%s pair=%s instance=%s] env.step end reward=%s done=%s in %.1fs",
        group_id,
        pair_index,
        instance_id,
        reward,
        done,
        time.time() - t0,
    )
    return obs, reward, done, info


class CleanSWEEnv(_CleanLoggingEnvMixin, SWEEnv):
  pass


class CleanGuardedSWEEnv(_CleanLoggingEnvMixin, GuardedSWEEnv):
  pass


def _load_model_and_sampler(args: argparse.Namespace):
  model_id = _model_id(args.model_version)
  model_name = _model_name_for_config(args.model_version)
  model_path = os.path.join(args.models_base_dir, model_id)

  if not os.path.isdir(model_path) or not os.listdir(model_path):
    os.makedirs(model_path, exist_ok=True)
    logger.info("Downloading model %s to %s", model_id, model_path)
    snapshot_download(
        repo_id=model_id,
        local_dir=model_path,
        local_dir_use_symlinks=False,
    )
  else:
    logger.info("Using local model at %s", model_path)

  tokenizer = AutoTokenizer.from_pretrained(
      model_path, local_files_only=True, trust_remote_code=True
  )
  tokenizer_for_agentic = tok_adapter.TokenizerAdapter(tokenizer)
  chat_parser = parser.QwenChatTemplateParser(tokenizer, enable_thinking=False)
  qwen_eos_tokens = [tokenizer.encode("<|im_end|>")[0]]

  model_config = call_model_config(model_name)
  model_config.dtype = jnp.bfloat16
  devices = jax.devices()
  total_devices = len(devices)
  mesh_tp = args.mesh_tp or min(
      total_devices // args.mesh_fsdp,
      getattr(model_config, "num_kv_heads", total_devices),
  )
  mesh_shape = (args.mesh_fsdp, mesh_tp)
  num_mesh_devices = int(np.prod(mesh_shape))
  if num_mesh_devices > total_devices:
    raise ValueError(
        f"Requested mesh {mesh_shape} uses {num_mesh_devices} devices, but only"
        f" {total_devices} JAX devices are available."
    )
  mesh_devices = np.array(devices[:num_mesh_devices]).reshape(mesh_shape)
  mesh = Mesh(mesh_devices, axis_names=("fsdp", "tp"))
  logger.info("Using mesh shape fsdp=%d tp=%d", mesh_shape[0], mesh_shape[1])

  logger.info("Loading model weights from %s", model_path)
  model = params_lib.create_model_from_safe_tensors(
      model_path, model_config, mesh, dtype=jnp.float32
  )
  sft_utils.show_hbm_usage()

  if args.rollout_engine == "vanilla":
    from tunix.generate import sampler as sampler_lib

    sampler = sampler_lib.Sampler(
        model,
        tokenizer,
        sampler_lib.CacheConfig(
            cache_size=16384,
            num_layers=model_config.num_layers,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        ),
    )
  elif args.rollout_engine == "vllm":
    from flax import nnx
    from tunix.generate import mappings
    from tunix.generate.vllm_sampler import VllmConfig, VllmSampler

    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    kv_cache_size = args.max_prompt_length + args.max_response_length + 256
    max_num_seqs = args.vllm_max_num_seqs or args.max_concurrent
    max_batched_tokens = (
        args.vllm_max_batched_tokens
        if args.vllm_max_batched_tokens is not None
        else (max_num_seqs * kv_cache_size) // 8
    )
    mapping_config = mappings.MappingConfig.build(
        mapping_obj=None,
        model=model,
        backend="vllm_jax",
    )
    vllm_config = VllmConfig(
        mesh=mesh,
        hbm_utilization=args.vllm_hbm_utilization,
        init_with_random_weights=args.vllm_init_random_weights,
        tpu_backend_type="jax",
        server_mode=args.vllm_server_mode,
        tensor_parallel_size=mesh.shape["tp"],
        data_parallel_size=mesh.shape["fsdp"],
        mapping_config=mapping_config,
        engine_kwargs={
            "model": model_path,
            "max_model_len": args.max_model_len,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_batched_tokens,
            "enable_prefix_caching": True,
            "kv_cache_metrics": True,
            "disable_log_stats": False,
        },
    )
    sampler = VllmSampler(tokenizer=tokenizer, config=vllm_config)
    sampler.load_checkpoint(nnx.state(model))
    logger.info("Synced model weights to vLLM engine.")
  elif args.rollout_engine == "sglang_jax":
    from flax import nnx
    from tunix.generate import mappings
    from tunix.generate.sglang_jax_sampler import (
        SglangJaxConfig,
        SglangJaxSampler,
    )

    mapping_config = mappings.MappingConfig.build(
        mapping_obj=None,
        model=model,
        backend="sglang_jax",
    )
    sampler = SglangJaxSampler(
        tokenizer=tokenizer,
        config=SglangJaxConfig(
            mesh=mesh,
            mapping_config=mapping_config,
            model_version=model_path,
            context_length=args.max_model_len,
            mem_fraction_static=args.sglang_mem_fraction_static,
            init_with_random_weights=args.sglang_init_random_weights,
            disable_radix_cache=True,
            enable_deterministic_sampling=False,
            precompile_token_paddings=[8192, 16384],
            precompile_bs_paddings=[1],
            max_running_requests=args.sglang_max_running_requests,
        ),
    )
    if args.sglang_init_random_weights:
      sampler.load_checkpoint(nnx.state(model))
      logger.info("Synced model weights to sglang_jax engine.")
  else:
    raise ValueError(f"Unsupported rollout_engine={args.rollout_engine!r}")

  return sampler, tokenizer, tokenizer_for_agentic, chat_parser, qwen_eos_tokens


def _build_model_call(
    *,
    sampler,
    tokenizer,
    chat_parser,
    qwen_eos_tokens,
    args: argparse.Namespace,
):
  sampler_lock = None
  if args.rollout_engine == "vanilla" or (
      args.rollout_engine == "vllm" and not args.vllm_server_mode
  ):
    sampler_lock = threading.Lock()

  def model_call(chat_completions, env_unused, max_generation_steps=None):
    pair_index = None
    group_id = None
    instance_id = "unknown"
    if env_unused is not None:
      extra_kwargs = getattr(env_unused, "extra_kwargs", {})
      pair_index = extra_kwargs.get("pair_index")
      group_id = extra_kwargs.get("group_id")
      instance_id = getattr(env_unused, "entry", {}).get(
          "instance_id", "unknown"
      )

    prompt = chat_parser.parse(
        chat_completions,
        add_generation_prompt=True,
        is_first_msg=True,
    )
    prompt_token_count = len(tokenizer.encode(prompt))
    logger.info(
        "[group=%s pair=%s instance=%s] model_call start prompt_tokens=%d",
        group_id,
        pair_index,
        instance_id,
        prompt_token_count,
    )
    if prompt_token_count >= args.max_model_len:
      raise PromptTooLongError(
          "Prompt too long before sampler call:"
          f" prompt_tokens={prompt_token_count}, max_model_len={args.max_model_len}"
      )

    generation_steps = max_generation_steps or args.max_response_length
    generation_steps = max(1, min(generation_steps, args.max_response_length))
    t0 = time.time()
    try:
      if sampler_lock is None:
        out = sampler(
            prompt,
            max_generation_steps=generation_steps,
            echo=False,
            eos_tokens=qwen_eos_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
      else:
        with sampler_lock:
          out = sampler(
              prompt,
              max_generation_steps=generation_steps,
              echo=False,
              eos_tokens=qwen_eos_tokens,
              temperature=args.temperature,
              top_p=args.top_p,
              top_k=args.top_k,
          )
    except Exception as exc:
      if _is_prompt_overflow_error(exc):
        raise PromptTooLongError(str(exc)) from exc
      raise
    logger.info(
        "[group=%s pair=%s instance=%s] model_call end response_chars=%d in %.1fs",
        group_id,
        pair_index,
        instance_id,
        len(out.text[0]) if out.text else 0,
        time.time() - t0,
    )
    return out

  return model_call


def _group_decision(rewards: list[float], reward_tol: float) -> tuple[bool, str]:
  all_zero = all(abs(reward) <= reward_tol for reward in rewards)
  all_one = all(abs(reward - 1.0) <= reward_tol for reward in rewards)
  if all_zero:
    return True, "all_zero_rewards"
  if all_one:
    return True, "all_one_rewards"
  return False, "mixed_rewards"


def _trajectory_attempt_record(item, entries_env: dict[int, dict[str, Any]]):
  traj = item.traj
  dataset_index = int(item.group_id)
  entry = entries_env[dataset_index]
  guard_reasons = sorted({
      (getattr(step, "info", {}) or {}).get("guard_reason", "unknown")
      for step in traj.steps
      if (getattr(step, "info", {}) or {}).get("guard_blocked")
  })
  return {
      "dataset_index": dataset_index,
      "pair_index": int(item.pair_index),
      "instance_id": entry.get("instance_id", dataset_index),
      "reward": _as_float_reward(getattr(traj, "reward", 0.0)),
      "num_steps": len(traj.steps),
      "status": getattr(traj.status, "name", str(traj.status)),
      "guard_blocked_steps": sum(
          1
          for step in traj.steps
          if (getattr(step, "info", {}) or {}).get("guard_blocked")
      ),
      "guard_reasons": guard_reasons,
  }


async def _run_cleaning(
    *,
    args: argparse.Namespace,
    candidate_indices: list[int],
    entries_env: dict[int, dict[str, Any]],
    model_call,
    tokenizer_for_agentic,
    chat_parser,
    output_dir: str,
) -> dict[str, Any]:
  env_cls = CleanGuardedSWEEnv if args.enable_guard else CleanSWEEnv

  def pairs_generator():
    for dataset_index in candidate_indices:
      entry = entries_env[dataset_index]
      for pair_index in range(args.num_generations):
        agent = SWEAgent()
        env = env_cls(
            entry=entry,
            group_id=dataset_index,
            pair_index=pair_index,
            max_steps=args.max_turns,
            step_timeout=args.step_timeout_secs,
            reward_timeout=args.reward_timeout_secs,
            backend=args.backend,
            delete_image=args.delete_image,
            verbose=args.verbose_env,
            scaffold=args.scaffold,
        )
        yield agent, env

  orchestrator = RolloutOrchestrator(
      engine_cls=CleanTrajectoryCollectEngine,
      engine_kwargs=dict(
          model_call=model_call,
          timeout=args.episode_timeout_secs,
          max_response_length=args.max_response_length,
          tokenizer=tokenizer_for_agentic,
          chat_parser=chat_parser,
      ),
      max_concurrency=args.max_concurrent,
      rollout_sync_lock=agentic_utils.RolloutSyncLock(),
  )

  result_path = os.path.join(output_dir, "group_rollout_results.jsonl")
  kept_path = os.path.join(output_dir, "kept_groups.jsonl")
  removed_path = os.path.join(output_dir, "removed_groups.jsonl")

  kept_indices = []
  removed_indices = []
  status_counts = Counter()
  decision_counts = Counter()
  start_time = time.time()

  producer = asyncio.create_task(
      orchestrator.run_producers_from_stream(
          pairs_stream=pairs_generator(),
          group_size=args.num_generations,
          group_key_fn=lambda i, env, traj: env.extra_kwargs["group_id"],
          collect_mode="Trajectory",
      )
  )

  await asyncio.sleep(0)

  with open(result_path, "w", encoding="utf-8") as result_f, open(
      kept_path, "w", encoding="utf-8"
  ) as kept_f, open(removed_path, "w", encoding="utf-8") as removed_f:
    processed_groups = 0
    async for batch in orchestrator.yield_batches(
        batch_size=args.num_generations
    ):
      attempts = [
          _trajectory_attempt_record(item, entries_env)
          for item in sorted(batch, key=lambda x: x.pair_index)
      ]
      if not attempts:
        continue
      dataset_index = attempts[0]["dataset_index"]
      rewards = [attempt["reward"] for attempt in attempts]
      remove, decision = _group_decision(rewards, args.reward_tol)
      if remove:
        removed_indices.append(dataset_index)
      else:
        kept_indices.append(dataset_index)
      decision_counts[decision] += 1
      status_counts.update(attempt["status"] for attempt in attempts)

      group_record = {
          "dataset_index": dataset_index,
          "instance_id": attempts[0]["instance_id"],
          "remove": remove,
          "decision": decision,
          "rewards": rewards,
          "attempts": attempts,
      }
      result_f.write(json.dumps(group_record, ensure_ascii=False) + "\n")
      (removed_f if remove else kept_f).write(
          json.dumps(group_record, ensure_ascii=False) + "\n"
      )
      result_f.flush()

      processed_groups += 1
      logger.info(
          "[%d/%d] dataset_index=%s instance=%s decision=%s rewards=%s elapsed=%.0fs",
          processed_groups,
          len(candidate_indices),
          dataset_index,
          attempts[0]["instance_id"],
          decision,
          rewards,
          time.time() - start_time,
      )

  await producer

  return {
      "kept_indices": kept_indices,
      "removed_indices": removed_indices,
      "decision_counts": dict(decision_counts),
      "status_counts": dict(status_counts),
      "result_path": result_path,
      "kept_path": kept_path,
      "removed_path": removed_path,
      "elapsed_secs": time.time() - start_time,
  }


def main() -> None:
  args = _parse_args()
  _setup_logging(args.logging_level)

  os.environ["TOKENIZERS_PARALLELISM"] = "true"
  os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
  os.environ["KUBECONFIG"] = os.path.expanduser(
      os.environ.get("KUBECONFIG", "~/.kube/config")
  )
  os.environ["NODE_SELECTOR_KEY"] = args.node_selector_key
  os.environ["NODE_SELECTOR_VAL"] = args.node_selector_val

  if os.getenv("JAX_PLATFORMS", None) == "proxy":
    import pathwaysutils  # pytype: disable=import-error

    pathwaysutils.initialize()

  apply_repoenv_kubernetes_watch_patch()

  if not args.skip_k8s_check:
    k8s_config.load_kube_config()
    k8s_client = client.CoreV1Api()
    k8s_client.list_namespace(timeout_seconds=5)
    logger.info("Kubernetes connection verified.")

  os.makedirs(args.output_dir, exist_ok=True)
  os.makedirs(args.dataset_cache, exist_ok=True)

  logger.info(
      "Loading dataset %s split=%s cache=%s",
      args.dataset_name,
      args.dataset_split,
      args.dataset_cache,
  )
  dataset = load_dataset(
      args.dataset_name,
      split=args.dataset_split,
      cache_dir=args.dataset_cache,
      trust_remote_code=True,
  )
  all_indices = list(range(len(dataset)))
  if args.require_docker_image:
    all_indices = [
        index for index in all_indices if "docker_image" in dataset[index]
    ]
  end_index = args.end_index if args.end_index > 0 else len(all_indices)
  candidate_indices = all_indices[args.start_index : end_index]
  if args.tasks_limit > 0:
    candidate_indices = candidate_indices[: args.tasks_limit]
  if not candidate_indices:
    raise ValueError("No candidate dataset entries selected for cleaning.")

  logger.info(
      "Selected %d candidate groups out of %d dataset rows.",
      len(candidate_indices),
      len(dataset),
  )
  entries_env = {
      index: _jsonify_list_fields(dataset[index]) for index in candidate_indices
  }

  sampler, tokenizer, tokenizer_for_agentic, chat_parser, qwen_eos_tokens = (
      _load_model_and_sampler(args)
  )
  model_call = _build_model_call(
      sampler=sampler,
      tokenizer=tokenizer,
      chat_parser=chat_parser,
      qwen_eos_tokens=qwen_eos_tokens,
      args=args,
  )

  clean_stats = asyncio.run(
      _run_cleaning(
          args=args,
          candidate_indices=candidate_indices,
          entries_env=entries_env,
          model_call=model_call,
          tokenizer_for_agentic=tokenizer_for_agentic,
          chat_parser=chat_parser,
          output_dir=args.output_dir,
      )
  )

  kept_indices = sorted(clean_stats["kept_indices"])
  removed_indices = sorted(clean_stats["removed_indices"])
  cleaned_dataset = dataset.select(kept_indices)
  removed_dataset = dataset.select(removed_indices)
  cleaned_dataset_dir = os.path.join(args.output_dir, "cleaned_dataset")
  removed_dataset_dir = os.path.join(args.output_dir, "removed_dataset")
  cleaned_dataset.save_to_disk(cleaned_dataset_dir)
  removed_dataset.save_to_disk(removed_dataset_dir)
  cleaned_jsonl = os.path.join(args.output_dir, "cleaned_dataset.jsonl")
  removed_jsonl = os.path.join(args.output_dir, "removed_dataset.jsonl")
  _write_dataset_jsonl(cleaned_dataset, cleaned_jsonl)
  _write_dataset_jsonl(removed_dataset, removed_jsonl)

  summary = {
      "dataset_name": args.dataset_name,
      "dataset_split": args.dataset_split,
      "num_generations": args.num_generations,
      "reward_tol": args.reward_tol,
      "num_candidates": len(candidate_indices),
      "num_kept": len(kept_indices),
      "num_removed": len(removed_indices),
      "kept_indices": kept_indices,
      "removed_indices": removed_indices,
      "decision_counts": clean_stats["decision_counts"],
      "status_counts": clean_stats["status_counts"],
      "cleaned_dataset_dir": cleaned_dataset_dir,
      "removed_dataset_dir": removed_dataset_dir,
      "cleaned_dataset_jsonl": cleaned_jsonl,
      "removed_dataset_jsonl": removed_jsonl,
      "group_rollout_results": clean_stats["result_path"],
      "kept_groups_path": clean_stats["kept_path"],
      "removed_groups_path": clean_stats["removed_path"],
      "elapsed_secs": clean_stats["elapsed_secs"],
  }
  summary_path = os.path.join(args.output_dir, "summary.json")
  with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

  logger.info("=" * 60)
  logger.info("DeepSWE data cleaning complete")
  logger.info("Candidates: %d", summary["num_candidates"])
  logger.info("Kept:       %d", summary["num_kept"])
  logger.info("Removed:    %d", summary["num_removed"])
  logger.info("Decisions:  %s", summary["decision_counts"])
  logger.info("Statuses:   %s", summary["status_counts"])
  logger.info("Cleaned dataset: %s", cleaned_dataset_dir)
  logger.info("Removed dataset: %s", removed_dataset_dir)
  logger.info("Cleaned JSONL: %s", cleaned_jsonl)
  logger.info("Removed JSONL: %s", removed_jsonl)
  logger.info("Summary: %s", summary_path)
  logger.info("=" * 60)


if __name__ == "__main__":
  main()
