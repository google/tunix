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

"""Minimal standalone reproduction of Mamba prefix caching (align mode) NaN / '!' collapse.

Reuses the DeepSWE dataset generator (`load_deepswe_dataset` / `build_prompt_item`),
`SWEAgent`, `TrajectoryCollectorEngine` (with `exact_token_continuity=True`), and
`VllmSamplerAdapter` (`RLVllmSampler`) with the exact configuration from
`mlperf_35b_128_v5p.sh` + `mlperf_base.sh`, without any Docker/K8s sandboxing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys
import time
from typing import Any, Dict, Mapping

from absl import logging
import numpy as np
import transformers
import tunix

_TUNIX_REPO_ROOT = str(pathlib.Path(tunix.__file__).resolve().parent.parent)
if _TUNIX_REPO_ROOT not in sys.path:
  sys.path.insert(0, _TUNIX_REPO_ROOT)

from examples.deepswe import swe_agent as swe_agent_lib
from tunix.experimental.common import datatypes
from tunix.experimental.examples.deepswe_dist import deepswe
from tunix.experimental.rollout import collector
from tunix.experimental.rollout import vllm_sampler_adapter
from tunix.generate import tokenizer_adapter as tokenizer_adapter_lib
from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.parser.chat_template_parser import parser as parser_lib


SYNTHETIC_OBSERVATION_BLOCK = (
    "Repository status check:\n"
    "On branch main\n"
    "Listing relevant files in repository:\n"
    "  src/main.py\n"
    "  src/utils.py\n"
    "  tests/test_main.py\n"
    "File excerpt (src/main.py lines 1-40):\n"
    "def process_items(items, config=None):\n"
    "    if config is None:\n"
    "        config = {'strict': True, 'max_retries': 3}\n"
    "    results = []\n"
    "    for idx, item in enumerate(items):\n"
    "        if not isinstance(item, dict):\n"
    "            if config.get('strict'):\n"
    "                raise TypeError(f'Expected dict at index {idx}, got {type(item).__name__}')\n"
    "            continue\n"
    "        results.append({k: v for k, v in item.items() if v is not None})\n"
    "    return results\n"
    "[Command finished with exit code 0]\n"
)


class DummySWEEnv(base_environment.BaseTaskEnv):
  """Sandbox-free environment that returns synthetic tool output to drive multi-turn rollouts."""

  def __init__(
      self,
      entry: Dict[str, Any],
      max_steps: int = 8,
      obs_repeats: int = 2,
      stop_on_collapse: bool = False,
      **kwargs,
  ):
    super().__init__(task=entry, max_steps=max_steps, **kwargs)
    self.entry = entry
    self.obs_repeats = obs_repeats
    self.stop_on_collapse = stop_on_collapse
    self.total_steps = 0

  def _initial_observation(self) -> Any:
    self.total_steps = 0
    return str(
        self.entry.get("problem_statement")
        or self.entry.get("instruction")
        or ""
    )

  def _step_impl(self, action: Any) -> base_environment.EnvStepResult:
    self.total_steps += 1
    done = self.total_steps >= self.max_steps
    action_str = str(action or "").strip()
    obs_body = (
        f"Executed action at step {self.total_steps}:\n"
        f"{action_str[:400]}\n\n"
        + (SYNTHETIC_OBSERVATION_BLOCK * self.obs_repeats)
    )
    return base_environment.EnvStepResult(
        observation=obs_body,
        reward=0.0,
        done=done,
        info={"max_steps": self.max_steps},
    )


def _find_collapse_token(tokens: np.ndarray, logprobs: np.ndarray | None) -> tuple[bool, int | None, str]:
  """Detects NaN logprob or repeated token 0 ('!') collapse and returns first token index."""
  if logprobs is not None and len(logprobs) > 0:
    nan_mask = ~np.isfinite(logprobs)
    if np.any(nan_mask):
      first_nan = int(np.argmax(nan_mask))
      return True, first_nan, f"NaN/Inf logprob at token {first_nan} (token_id={int(tokens[first_nan]) if first_nan < len(tokens) else -1})"

  if len(tokens) >= 16:
    # Check for run of >= 16 consecutive token_id == 0 ('!')
    is_zero = (tokens == 0).astype(np.int32)
    # Convolution to find 16 consecutive zeros
    window = 16
    conv = np.convolve(is_zero, np.ones(window, dtype=np.int32), mode="valid")
    hits = np.where(conv == window)[0]
    if len(hits) > 0:
      first_zero = int(hits[0])
      return True, first_zero, f"Token 0 ('!') run of >=16 starting at token {first_zero}"

  return False, None, "ok"


def build_vllm_sampler(args: argparse.Namespace) -> vllm_sampler_adapter.VllmSamplerAdapter:
  """Builds VllmSamplerAdapter matching mlperf_35b_128_v5p.sh + mlperf_base.sh + run_rollout_node.py."""
  from vllm.engine.arg_utils import AsyncEngineArgs  # pylint: disable=g-import-not-at-top
  from tunix.utils import maxtext_utils  # pylint: disable=g-import-not-at-top

  prefuse_moe = "prefused" in (args.maxtext_ckpt or "") or not args.maxtext_ckpt
  base_additional = maxtext_utils.build_vllm_maxtext_additional_config(
      args.maxtext_model_name,
      attention="",
      prefuse_moe_weights=prefuse_moe,
  )
  maxtext_cfg = dict(base_additional["maxtext_config"])
  if args.maxtext_ckpt:
    maxtext_cfg["load_parameters_path"] = args.maxtext_ckpt

  additional_config = {
      "sharding": {
          "sharding_strategy": {
              "expert_parallelism": args.expert_parallelism,
              "tensor_parallelism": args.tensor_parallel_size,
              "enable_dp_attention": args.enable_dp_attention,
          }
      },
      "custom_mamba_cache_multiplier": args.custom_mamba_cache_multiplier,
      "maxtext_config": maxtext_cfg,
  }

  engine_kwargs = dict(
      model=args.model_id,
      tokenizer=args.tokenizer_path,
      tensor_parallel_size=args.tensor_parallel_size,
      data_parallel_size=1,
      enable_expert_parallel=True,
      max_model_len=args.max_model_len,
      max_num_batched_tokens=args.max_num_batched_tokens,
      max_num_seqs=args.max_num_seqs,
      gpu_memory_utilization=args.gpu_memory_utilization,
      trust_remote_code=True,
      dtype="bfloat16",
      enable_lora=False,
      enable_prefix_caching=args.enable_prefix_caching,
      prefix_cache_retention_interval=0,
      mamba_cache_mode=args.mamba_cache_mode,
      kv_cache_dtype="bfloat16",
      block_size=args.block_size,
      async_scheduling=args.async_scheduling,
      enable_chunked_prefill=args.enable_chunked_prefill,
      language_model_only=True,
      reasoning_parser="qwen3",
      limit_mm_per_prompt={"image": 0, "video": 0},
      enable_return_routed_experts=args.return_routed_experts,
      hf_overrides=dict(maxtext_utils.VLLM_MAXTEXT_HF_OVERRIDES),
      additional_config=additional_config,
  )
  logging.info("Initializing AsyncEngineArgs with:\n%s", json.dumps(engine_kwargs, indent=2, default=str))
  engine_args = AsyncEngineArgs(**engine_kwargs)
  sampler = vllm_sampler_adapter.VllmSamplerAdapter(
      server_id="repro-rollout-0",
      engine_args=engine_args,
      model_name=args.model_id,
      weight_sync_mode="none",
  )
  return sampler


async def run_single_trajectory(
    prompt_item: Dict[str, Any],
    group_index: int,
    sampler: vllm_sampler_adapter.VllmSamplerAdapter,
    tokenizer: Any,
    chat_parser: Any,
    eos_ids: list[int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
  """Runs one multi-turn trajectory using TrajectoryCollectorEngine and DummySWEEnv."""
  prompt_id = str(prompt_item["prompt_id"])
  traj_id = f"{prompt_id}_g{group_index}"
  entry = dict(prompt_item["metadata"]["env_config"]["entry"])

  env = DummySWEEnv(
      entry=entry,
      max_steps=args.max_turns,
      obs_repeats=args.obs_repeats,
      extra_kwargs={"group_id": prompt_id, "pair_index": group_index},
  )
  agent = swe_agent_lib.SWEAgent(
      scaffold="openhands",
  )

  generation_kwargs = {
      "temperature": args.temperature,
      "top_p": args.top_p,
      "top_k": args.top_k,
      "eos_tokens": eos_ids,
      "return_logprobs": True,
      "return_routed_experts": args.return_routed_experts,
  }
  if args.max_tokens_per_turn > 0:
    generation_kwargs["max_generation_steps"] = args.max_tokens_per_turn

  req = datatypes.RolloutRequest(
      request_id=f"req_{traj_id}",
      prompt=prompt_item["prompt"],
      prompt_id=prompt_id,
      group_index=group_index,
      target_policy_version=0,
      generation_kwargs=generation_kwargs,
      max_turns=args.max_turns,
      max_response_length=args.max_response_length,
      exact_token_continuity=True,
      metadata={"episode_timeout": 1800.0},
  )

  turn_summaries = []
  first_collapse_turn = None
  first_collapse_global_token = None
  first_collapse_turn_token = None
  first_collapse_reason = None
  cumulative_asst_tokens = 0

  class _ObservedSampler:
    """Thin per-trajectory wrapper around sampler to log each turn immediately."""

    supports_token_input = True

    def __getattr__(self, name: str) -> Any:
      return getattr(sampler, name)

    async def sample(self, sampling_req: Any, **s_kwargs: Any) -> Any:
      nonlocal first_collapse_turn, first_collapse_global_token, first_collapse_turn_token, first_collapse_reason, cumulative_asst_tokens
      res = await sampler.sample(sampling_req, **s_kwargs)
      r = res[0] if isinstance(res, (list, tuple)) and len(res) == 1 else res
      turn_idx = len(turn_summaries)
      gen_tokens = np.asarray(getattr(r, "token_ids", []), dtype=np.int32)
      gen_lps = getattr(r, "logprobs", None)
      if gen_lps is not None:
        gen_lps = np.asarray(gen_lps, dtype=np.float32)
      prompt_ids = getattr(r, "prompt_token_ids", None)
      prompt_toks = len(prompt_ids) if prompt_ids is not None else 0
      comp_toks = len(gen_tokens)

      collapsed, tok_idx, reason = _find_collapse_token(gen_tokens, gen_lps)
      if collapsed and first_collapse_turn is None:
        first_collapse_turn = turn_idx
        first_collapse_turn_token = tok_idx
        first_collapse_global_token = cumulative_asst_tokens + (tok_idx or 0)
        first_collapse_reason = reason
        if args.stop_env_on_collapse:
          env.max_steps = 0

      preview = repr(getattr(r, "text", "")[:120])
      turn_info = {
          "turn": turn_idx,
          "prompt_tokens": prompt_toks,
          "completion_tokens": comp_toks,
          "collapsed": collapsed,
          "collapse_token_in_turn": tok_idx,
          "reason": reason,
          "preview": preview,
      }
      turn_summaries.append(turn_info)
      logging.info(
          "[%s] Turn %d: prompt_tokens=%d, completion_tokens=%d, collapsed=%s (%s), preview=%s",
          traj_id,
          turn_idx,
          prompt_toks,
          comp_toks,
          collapsed,
          reason,
          preview,
      )
      cumulative_asst_tokens += comp_toks
      return res

  engine = collector.TrajectoryCollectorEngine(
      traj_id=traj_id,
      request=req,
      sampler=_ObservedSampler(),  # pytype: disable=wrong-arg-types
      env_client=env,
      agent=agent,
      tokenizer=tokenizer,
      chat_parser=chat_parser,
      eos_ids=eos_ids,
  )

  traj_item = await engine.run_episode()

  status = traj_item.traj.get("status", "UNKNOWN") if isinstance(traj_item.traj, Mapping) else "UNKNOWN"
  clipped = traj_item.metadata.get("clipped", False)
  raw_length = traj_item.metadata.get("raw_length", 0)
  return {
      "traj_id": traj_id,
      "group_index": group_index,
      "status": status,
      "clipped": clipped,
      "raw_length": raw_length,
      "turns": len(turn_summaries),
      "collapsed": first_collapse_turn is not None,
      "first_collapse_turn": first_collapse_turn,
      "first_collapse_turn_token": first_collapse_turn_token,
      "first_collapse_global_token": first_collapse_global_token,
      "first_collapse_reason": first_collapse_reason,
      "turn_summaries": turn_summaries,
  }


async def async_main(args: argparse.Namespace) -> int:
  logging.set_verbosity(logging.INFO)
  t0 = time.time()

  logging.info("Loading tokenizer %s...", args.tokenizer_path)
  tokenizer = transformers.AutoTokenizer.from_pretrained(
      args.tokenizer_path, trust_remote_code=True
  )
  rollout_tokenizer = tokenizer_adapter_lib.TokenizerAdapter(tokenizer)
  chat_parser = parser_lib.QwenChatTemplateParser(
      tokenizer, enable_thinking=False
  )
  eos_ids = [int(x) for x in args.eos_tokens.split(",") if x.strip()]

  cache_file = f"/tmp/deepswe_prompt_items_n{args.num_prompts}.json"
  if os.path.exists(cache_file):
    logging.info("Loading cached DeepSWE prompt items from %s...", cache_file)
    with open(cache_file, "r") as f:
      prompt_items = json.load(f)
  else:
    logging.info(
        "Loading DeepSWE dataset from %s (num_prompts=%d)...",
        args.dataset_path,
        args.num_prompts,
    )
    dataset = deepswe.load_deepswe_dataset(
        dataset_path=args.dataset_path,
        dataset_split="train",
        shuffle=True,
        seed=42,
    )
    prompt_items = list(
        deepswe.iter_prompt_items(
            dataset=dataset,
            max_steps=1,
            batch_size=args.num_prompts,
            max_turns=args.max_turns,
            max_response_length=args.max_response_length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            step_timeout_secs=300,
            reward_timeout_secs=600,
            env_backend="dummy",
            use_agent_sandbox=False,
            scaffold="openhands",
            env_verbose=False,
            exact_token_continuity=True,
        )
    )
    with open(cache_file, "w") as f:
      json.dump(prompt_items, f, default=str)
  logging.info("Loaded %d prompt items: %s", len(prompt_items), [p["prompt_id"] for p in prompt_items])

  # Pre-flight check of DummySWEEnv and SWEAgent before loading the 35B model.
  test_entry = dict(prompt_items[0]["metadata"]["env_config"]["entry"])
  test_env = DummySWEEnv(entry=test_entry, max_steps=args.max_turns, obs_repeats=args.obs_repeats)
  test_agent = swe_agent_lib.SWEAgent(scaffold="openhands")
  test_obs, _ = test_env.reset()
  test_agent.reset()
  test_agent.update_from_env(observation=test_obs, reward=0.0, done=False, info={})
  logging.info("Pre-flight agent/env check passed (initial messages=%d).", len(test_agent.chat_completions))

  sampler = build_vllm_sampler(args)
  await sampler.start()
  try:
    logging.info("Sampler started in %.1fs. Launching %d x %d = %d concurrent trajectories...",
                 time.time() - t0, len(prompt_items), args.num_generations,
                 len(prompt_items) * args.num_generations)

    tasks = []
    for p_item in prompt_items:
      for g_idx in range(args.num_generations):
        tasks.append(
            run_single_trajectory(
                prompt_item=p_item,
                group_index=g_idx,
                sampler=sampler,
                tokenizer=rollout_tokenizer,
                chat_parser=chat_parser,
                eos_ids=eos_ids,
                args=args,
            )
        )

    results = await asyncio.gather(*tasks)
  finally:
    await sampler.stop()

  print("\n" + "=" * 100)
  print(
      f"REPRODUCTION SUMMARY (enable_prefix_caching={args.enable_prefix_caching}, "
      f"mamba_cache_mode={args.mamba_cache_mode})"
  )
  print("=" * 100)
  collapsed_count = 0
  for r in results:
    if r["collapsed"]:
      collapsed_count += 1
    print(
        f"  {r['traj_id']:<16} | status={r['status']:<26} | turns={r['turns']:<2} "
        f"| raw_len={r['raw_length']:<6} | collapsed={str(r['collapsed']):<5} "
        f"| turn={str(r['first_collapse_turn']):<4} "
        f"| turn_tok={str(r['first_collapse_turn_token']):<5} "
        f"| global_tok={str(r['first_collapse_global_token']):<6} "
        f"| {r['first_collapse_reason'] or 'clean'}"
    )
  print("-" * 100)
  print(
      f"Total trajectories: {len(results)} | Collapsed (NaN / '!'): {collapsed_count}/{len(results)} "
      f"| Clean: {len(results) - collapsed_count}/{len(results)}"
  )
  print("=" * 100 + "\n")

  if args.output_json:
    with open(args.output_json, "w") as f:
      json.dump(results, f, indent=2, default=str)
    logging.info("Wrote detailed results to %s", args.output_json)

  return 0


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Minimal Mamba prefix cache NaN collapse reproducer")
  p.add_argument("--dataset_path", type=str, default="gs://mlperf_dataset/benchmark-r2e-gym-easy")
  p.add_argument("--model_id", type=str, default="Qwen/Qwen3.5-35B-A3B")
  p.add_argument("--tokenizer_path", type=str, default="Qwen/Qwen3.5-35B-A3B")
  p.add_argument("--maxtext_model_name", type=str, default="qwen3.5-35b-a3b")
  p.add_argument(
      "--maxtext_ckpt",
      type=str,
      default="gs://hengtaoguo-maxtext-logs/checkpoints/qwen3.5-35b-a3b/unscanned/2026-06-10-15-55/0/items",
  )
  p.add_argument("--eos_tokens", type=str, default="248046,248044")
  p.add_argument("--num_prompts", type=int, default=1)
  p.add_argument("--num_generations", type=int, default=16)
  p.add_argument("--max_turns", type=int, default=5)
  p.add_argument("--obs_repeats", type=int, default=2)
  p.add_argument("--max_tokens_per_turn", type=int, default=1024)
  p.add_argument("--max_response_length", type=int, default=4096)
  p.add_argument("--stop_env_on_collapse", action=argparse.BooleanOptionalAction, default=True)
  p.add_argument("--enable_prefix_caching", action=argparse.BooleanOptionalAction, default=True)
  p.add_argument("--mamba_cache_mode", type=str, default="align")
  p.add_argument("--tensor_parallel_size", type=int, default=1)
  p.add_argument("--expert_parallelism", type=int, default=4)
  p.add_argument("--enable_dp_attention", action=argparse.BooleanOptionalAction, default=True)
  p.add_argument("--custom_mamba_cache_multiplier", type=int, default=16)
  p.add_argument("--max_model_len", type=int, default=65536)
  p.add_argument("--max_num_batched_tokens", type=int, default=2048)
  p.add_argument("--max_num_seqs", type=int, default=16)
  p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
  p.add_argument("--block_size", type=int, default=256)
  p.add_argument("--async_scheduling", action=argparse.BooleanOptionalAction, default=True)
  p.add_argument("--enable_chunked_prefill", action=argparse.BooleanOptionalAction, default=True)
  p.add_argument("--return_routed_experts", action=argparse.BooleanOptionalAction, default=True)
  p.add_argument("--temperature", type=float, default=1.0)
  p.add_argument("--top_p", type=float, default=1.0)
  p.add_argument("--top_k", type=int, default=20)
  p.add_argument("--output_json", type=str, default="/tmp/repro_mamba_prefix_cache.json")
  return p.parse_args()


if __name__ == "__main__":
  os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
  raise SystemExit(asyncio.run(async_main(parse_args())))
