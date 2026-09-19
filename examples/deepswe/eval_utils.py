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

"""Common evaluation utilities, trajectory engines, and metrics for DeepSWE."""

from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import contextlib
import json
import logging
import math
import os
import sys
import threading
import time
from typing import Any, Callable, Optional

_DEEPSWE_DIR = os.path.dirname(os.path.abspath(__file__))
if _DEEPSWE_DIR not in sys.path:
  sys.path.insert(0, _DEEPSWE_DIR)

try:
  import guarded_swe_env  # type: ignore
  import swe_env  # type: ignore
except ImportError:
  from examples.deepswe import guarded_swe_env
  from examples.deepswe import swe_env

from tunix.rl.agentic import utils as agentic_utils
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.pipeline.rollout_orchestrator import RolloutOrchestrator
from tunix.rl.agentic.trajectory import trajectory_collect_engine

Counter = collections.Counter

ANSI_RED = "\033[31m"
ANSI_RESET = "\033[0m"


class PromptTooLongError(ValueError):
  """Raised when a prompt exceeds the model context limit before sampling."""


def _is_prompt_overflow_error(exc: Exception) -> bool:
  message = str(exc)
  return (
      "maximum input length" in message
      or "context length is only" in message
      or "Prompt too long before sampler call" in message
      or ("input_tokens" in message and "max_model_len" in message)
  )


def create_model_call(
    sampler: Any,
    tokenizer: Any,
    chat_parser: Any,
    max_response_length: int,
    max_context_limit: int,
    sampler_kwargs: Optional[dict[str, Any]] = None,
    sampler_lock: Optional[threading.Lock] = None,
    logger: Optional[logging.Logger] = None,
) -> Callable[..., Any]:
  """Creates a model inference callable for TrajectoryCollectEngine."""
  log = logger or logging.getLogger("deepswe_eval")
  call_kwargs = dict(sampler_kwargs) if sampler_kwargs else {}

  def model_call(
      chat_completions: Any,
      env: Any = None,
      max_generation_steps: Optional[int] = None,
      **kwargs: Any,
  ) -> Any:
    """Model inference via tunix sampler."""
    max_gen_steps = min(max_generation_steps or max_response_length, 4096)
    pair_index = None
    instance_id = "unknown"
    if env is not None:
      pair_index = getattr(env, "extra_kwargs", {}).get("pair_index")
      instance_id = getattr(env, "entry", {}).get("instance_id", "unknown")

    prompt = chat_parser.parse(
        chat_completions,
        add_generation_prompt=True,
        is_first_msg=True,
    )
    prompt_token_count = len(tokenizer.encode(prompt))
    log.info(
        "[pair=%s instance=%s] model_call start prompt_chars=%d prompt_tokens=%d"
        " max_context_limit=%d",
        pair_index,
        instance_id,
        len(prompt),
        prompt_token_count,
        max_context_limit,
    )
    remaining_context = max_context_limit - prompt_token_count
    if remaining_context <= 0:
      raise PromptTooLongError(
          "Prompt too long before sampler call:"
          f" prompt_tokens={prompt_token_count},"
          f" max_context_limit={max_context_limit}"
      )
    max_gen_steps = min(max_gen_steps, remaining_context)
    t0 = time.time()
    try:
      if sampler_lock is None:
        out = sampler(
            prompt,
            max_generation_steps=max_gen_steps,
            echo=False,
            **call_kwargs,
        )
      else:
        with sampler_lock:
          out = sampler(
              prompt,
              max_generation_steps=max_gen_steps,
              echo=False,
              **call_kwargs,
          )
    except Exception as exc:
      if _is_prompt_overflow_error(exc):
        raise PromptTooLongError(str(exc)) from exc
      raise
    log.info(
        "[pair=%s instance=%s] model_call end response_chars=%d (%.1fs)",
        pair_index,
        instance_id,
        len(out.text[0]) if out.text else 0,
        time.time() - t0,
    )
    return out

  return model_call


class _EvalLoggingEnvMixin:
  """Adds phase-level reset/step logs and optional observation sanitization."""

  enforce_xml_function_check: bool = False
  clip_obs_len: Optional[int] = None

  def __init__(
      self,
      *args: Any,
      enforce_xml_function_check: Optional[bool] = None,
      clip_obs_len: Optional[int] = None,
      **kwargs: Any,
  ):
    super().__init__(*args, **kwargs)
    if enforce_xml_function_check is not None:
      self.enforce_xml_function_check = enforce_xml_function_check
    if clip_obs_len is not None:
      self.clip_obs_len = clip_obs_len

  def reset(self):
    """Resets the environment and logs the timing."""
    log = logging.getLogger("deepswe_eval")
    pair_index = self.extra_kwargs.get("pair_index")
    instance_id = self.entry.get("instance_id", "unknown")
    log.info("[pair=%s instance=%s] reset start", pair_index, instance_id)
    t0 = time.time()
    obs, info = super().reset()
    log.info(
        "[pair=%s instance=%s] reset end (%.1fs)",
        pair_index,
        instance_id,
        time.time() - t0,
    )
    return obs, info

  def step(self, action: Any):
    """Steps the environment and logs the action and timing."""
    log = logging.getLogger("deepswe_eval")
    pair_index = self.extra_kwargs.get("pair_index")
    instance_id = self.entry.get("instance_id", "unknown")
    step_idx = self.step_count + 1
    action_name = action
    if isinstance(action, str):
      action_name = action.split("\n", 1)[0][:120]
    log.info(
        "[pair=%s instance=%s] env.step start step=%s action=%s",
        pair_index,
        instance_id,
        step_idx,
        action_name,
    )
    t0 = time.time()
    obs, reward, done, info = super().step(action)

    if getattr(self, "enforce_xml_function_check", False):
      has_valid_fn = bool(
          action
          and isinstance(action, str)
          and "<function=" in action
          and not action.startswith("<function=>")
      )
      if not has_valid_fn and not done:
        obs = (
            "[ACTION GUARD] Your previous response did not include a valid"
            " function call. You must output exactly one tool call in the"
            " required XML format. For example:\n<function=execute_bash>\n"
            "<parameter=command>git status</parameter>\n</function>\nDo NOT"
            " call submit until you have modified code and verified the fix."
        )
      elif not obs and not done:
        obs = "(Command executed successfully with no output.)"

    clip_len = getattr(self, "clip_obs_len", None)
    if clip_len and isinstance(obs, str) and len(obs) > clip_len:
      half = clip_len // 2
      obs = obs[:half] + "\n...<response clipped>...\n" + obs[-half:]

    log.info(
        "[pair=%s instance=%s] env.step end step=%s reward=%.1f done=%s"
        " (%.1fs)",
        pair_index,
        instance_id,
        step_idx,
        reward,
        done,
        time.time() - t0,
    )
    return obs, reward, done, info


class LoggedSWEEnv(_EvalLoggingEnvMixin, swe_env.SWEEnv):
  pass


class LoggedGuardedSWEEnv(_EvalLoggingEnvMixin, guarded_swe_env.GuardedSWEEnv):
  pass


class EvalTrajectoryCollectEngine(
    trajectory_collect_engine.TrajectoryCollectEngine
):
  """Trajectory engine that converts prompt overflows and env errors into per-trajectory termination."""

  skip_final_reward_on_overflow: bool = False

  async def _reset(self):
    log = logging.getLogger("deepswe_eval")
    try:
      await super()._reset()
    except Exception as exc:
      log.exception(
          "[pair=%s instance=%s] unexpected exception in _reset, terminating"
          " trajectory gracefully: %s",
          self.env.extra_kwargs.get("pair_index"),
          self.env.entry.get("instance_id", "unknown"),
          exc,
      )
      self._reset_failed = True
      self._skip_final_reward = True

  async def _one_step(self) -> bool:
    log = logging.getLogger("deepswe_eval")
    if getattr(self, "_reset_failed", False):
      self.agent.trajectory.status = agent_types.TrajectoryStatus.TIMEOUT
      return True
    try:
      return await super()._one_step()
    except PromptTooLongError as exc:
      log.warning(
          "[pair=%s instance=%s] terminating trajectory due to prompt"
          " overflow: %s",
          self.env.extra_kwargs.get("pair_index"),
          self.env.entry.get("instance_id", "unknown"),
          exc,
      )
      self.agent.trajectory.status = (
          agent_types.TrajectoryStatus.MAX_CONTEXT_LIMIT_REACHED
      )
      self._skip_final_reward = self.skip_final_reward_on_overflow
      if self.agent.trajectory.steps:
        self.agent.trajectory.steps[-1].done = True
      return True
    except Exception as exc:
      log.exception(
          "[pair=%s instance=%s] unexpected exception in _one_step, terminating"
          " trajectory gracefully: %s",
          self.env.extra_kwargs.get("pair_index"),
          self.env.entry.get("instance_id", "unknown"),
          exc,
      )
      if self.agent.trajectory.steps:
        self.agent.trajectory.steps[-1].done = True
      return True

  async def _append_final_reward(self):
    if getattr(self, "_skip_final_reward", False):
      return
    log = logging.getLogger("deepswe_eval")
    pair_index = self.env.extra_kwargs.get("pair_index")
    instance_id = self.env.entry.get("instance_id", "unknown")
    log.info(
        "[pair=%s instance=%s] final_reward_fn start (steps=%d status=%s)",
        pair_index,
        instance_id,
        len(self.agent.trajectory.steps),
        getattr(self.agent.trajectory, "status", "UNKNOWN"),
    )
    t0 = time.time()
    await super()._append_final_reward()
    last_step = self.agent.get_current_step()
    rew = last_step.reward if last_step is not None else 0.0
    log.info(
        "[pair=%s instance=%s] final_reward_fn end reward=%.1f (%.1fs)",
        pair_index,
        instance_id,
        rew,
        time.time() - t0,
    )

  def compute_trajectory_reward(self):
    if getattr(self, "_skip_final_reward", False):
      self.agent.trajectory.reward = 0.0
      return self.agent.trajectory
    return super().compute_trajectory_reward()


async def run_evaluation(
    entries: list[dict[str, Any]],
    pairs_stream: Any,
    model_call: Callable[..., Any],
    tokenizer_for_agentic: Any,
    chat_parser: Any,
    timeout: float,
    max_concurrent: int,
    output_dir: str,
    engine_cls: type[
        trajectory_collect_engine.TrajectoryCollectEngine
    ] = EvalTrajectoryCollectEngine,
    logger: Optional[logging.Logger] = None,
    num_rollouts_per_instance: int = 1,
    max_response_length: Optional[int] = None,
    use_custom_executor: bool = True,
) -> list[dict[str, Any]]:
  """Runs evaluation with orchestrator-managed task-level parallelism."""
  log = logger or logging.getLogger("deepswe_eval")
  if not output_dir.startswith("gs://"):
    os.makedirs(output_dir, exist_ok=True)

  if use_custom_executor:
    loop = asyncio.get_running_loop()
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max(max_concurrent, 32),
        thread_name_prefix="model_call_worker",
    )
    loop.set_default_executor(executor)

  engine_kwargs: dict[str, Any] = dict(
      model_call=model_call,
      timeout=timeout,
      tokenizer=tokenizer_for_agentic,
      chat_parser=chat_parser,
  )
  if max_response_length is not None:
    engine_kwargs["max_response_length"] = max_response_length

  orchestrator = RolloutOrchestrator(
      engine_cls=engine_cls,
      engine_kwargs=engine_kwargs,
      max_concurrency=max_concurrent,
      rollout_sync_lock=agentic_utils.RolloutSyncLock(),
  )

  results = []
  start_time = time.time()

  producer = asyncio.create_task(
      orchestrator.run_producers_from_stream(
          pairs_stream=pairs_stream,
          num_generations=1,
          group_key_fn=lambda i, env, traj: env.extra_kwargs["group_id"],
          collect_mode="Trajectory",
      )
  )

  await asyncio.sleep(0)

  try:
    async for batch in orchestrator.yield_batches(batch_size=1):
      for item in batch:
        traj = item.traj
        entry_index = item.group_index // num_rollouts_per_instance
        entry = entries[entry_index]
        step_actions = [
            getattr(step, "action", "").split("\n", 1)[0][:80]
            for step in traj.steps
        ]
        guard_reasons = sorted({
            (getattr(step, "info", {}) or {}).get("guard_reason", "unknown")
            for step in traj.steps
            if (getattr(step, "info", {}) or {}).get("guard_blocked")
        })
        guard_blocked_steps = sum(
            1
            for step in traj.steps
            if (getattr(step, "info", {}) or {}).get("guard_blocked")
        )
        result = {
            "pair_index": item.group_index,
            "entry_index": entry_index,
            "instance_id": entry.get("instance_id", entry_index),
            "reward": float(traj.reward),
            "num_steps": len(traj.steps),
            "status": getattr(traj.status, "name", str(traj.status)),
            "step_actions": step_actions,
            "guard_blocked_steps": guard_blocked_steps,
            "guard_reasons": guard_reasons,
        }
        results.append(result)
        elapsed = time.time() - start_time
        log.info(
            "[%d/%d] Instance %s: reward=%.1f, steps=%d, status=%s (%.0fs"
            " elapsed)",
            len(results),
            len(entries) * num_rollouts_per_instance,
            result["instance_id"],
            result["reward"],
            result["num_steps"],
            result["status"],
            elapsed,
        )
        log.info(
            "%s[%s] FINAL TRAJECTORY REWARD=%.1f%s",
            ANSI_RED,
            result["instance_id"],
            result["reward"],
            ANSI_RESET,
        )

    await producer
    return results
  finally:
    if not producer.done():
      producer.cancel()
      with contextlib.suppress(asyncio.CancelledError):
        await producer


def _estimate_pass_at_k(n: int, c: int, k: int) -> Optional[float]:
  """Unbiased estimator for pass@k given n samples and c correct."""
  if n < k:
    return None
  if n - c < k:
    return 1.0
  return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def compute_pass_at_k(
    results: list[dict[str, Any]],
    ks: tuple[int, ...] = (1, 4),
    log_guard_stats: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
  """Computes and logs evaluation metrics such as Pass@k and average reward."""
  log = logger or logging.getLogger("deepswe_eval")
  total = len(results)
  if total == 0:
    log.warning("No results to evaluate.")
    return

  correct = sum(1 for r in results if r["reward"] > 0)
  total_reward = sum(float(r["reward"]) for r in results)
  total_steps = sum(r["num_steps"] for r in results)
  status_counts = Counter(r["status"] for r in results)

  instance_groups = collections.defaultdict(list)
  for r in results:
    instance_groups[r["instance_id"]].append(r)

  num_instances = len(instance_groups)
  resolved_instances = sum(
      1
      for inst_results in instance_groups.values()
      if any(r["reward"] > 0 for r in inst_results)
  )

  pass_at_k_metrics = {}
  for k in ks:
    scores = []
    for inst_results in instance_groups.values():
      n = len(inst_results)
      c = sum(1 for r in inst_results if r["reward"] > 0)
      score = _estimate_pass_at_k(n, c, k)
      if score is not None:
        scores.append(score)
    pass_at_k_metrics[k] = sum(scores) / len(scores) if scores else None

  avg_reward = total_reward / total
  avg_steps = total_steps / total

  log.info("=" * 50)
  log.info("Evaluation Results")
  log.info("=" * 50)
  if total != num_instances:
    log.info("Total instances:  %d (rollouts: %d)", num_instances, total)
    log.info("Resolved:         %d (rollouts: %d)", resolved_instances, correct)
  else:
    log.info("Total instances:  %d", num_instances)
    log.info("Resolved:         %d", resolved_instances)
  for k in ks:
    if k == 1:
      val = (
          pass_at_k_metrics[1]
          if pass_at_k_metrics.get(1) is not None
          else correct / total
      )
      log.info("Pass@1:           %.4f", val)
    elif pass_at_k_metrics.get(k) is not None:
      log.info("Pass@%d:           %.4f", k, pass_at_k_metrics[k])
    else:
      log.info("Pass@%d:           N/A", k)

  log.info("Avg reward:       %.4f", avg_reward)
  log.info("Avg steps:        %.2f", avg_steps)
  log.info("Status counts:    %s", dict(status_counts))

  total_guard_blocks = sum(r.get("guard_blocked_steps", 0) for r in results)
  if log_guard_stats or total_guard_blocks > 0:
    guard_blocked_trajectories = sum(
        1 for r in results if r.get("guard_blocked_steps", 0) > 0
    )
    guard_reason_counts = Counter()
    for r in results:
      for reason in r.get("guard_reasons", []):
        guard_reason_counts[reason] += 1
    log.info(
        "Guarded trajs:    %d/%d (%.2f%%)",
        guard_blocked_trajectories,
        total,
        100.0 * guard_blocked_trajectories / total,
    )
    log.info("Guard blocks:     %d", total_guard_blocks)
    if guard_reason_counts:
      log.info("Guard reasons:    %s", dict(guard_reason_counts))

  log.info("=" * 50)


def save_results(
    results: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    output_dir: str,
    filename_prefix: str = "eval",
    include_pair_index: bool = True,
    include_step_actions: bool = True,
    include_guard_stats: bool = False,
    logger: Optional[logging.Logger] = None,
) -> str:
  """Saves the evaluation results to a JSONL file and uploads to GCS if needed."""
  log = logger or logging.getLogger("deepswe_eval")
  timestamp = time.strftime("%Y%m%d_%H%M%S")
  filename = f"{filename_prefix}_{timestamp}.jsonl"

  local_dir = (
      "/tmp/eval_results" if output_dir.startswith("gs://") else output_dir
  )
  os.makedirs(local_dir, exist_ok=True)
  output_file = os.path.join(local_dir, filename)

  with open(output_file, "w") as f:
    for r in results:
      entry_idx = r.get("entry_index", r.get("pair_index", 0))
      entry = entries[entry_idx]
      record: dict[str, Any] = {}
      if include_pair_index:
        record["pair_index"] = r.get("pair_index", -1)
      record["instance_id"] = entry.get("instance_id", r.get("instance_id"))
      record["docker_image"] = entry.get("docker_image", "")
      record["reward"] = r["reward"]
      record["num_steps"] = r["num_steps"]
      record["status"] = r["status"]
      if include_step_actions:
        record["step_actions"] = r.get("step_actions", [])
      if include_guard_stats:
        record["guard_blocked_steps"] = r.get("guard_blocked_steps", 0)
        record["guard_reasons"] = r.get("guard_reasons", [])
      f.write(json.dumps(record) + "\n")

  log.info("Results saved to %s", output_file)

  if output_dir.startswith("gs://"):
    from google.cloud import storage  # pytype: disable=import-error

    gcs_path = output_dir[5:]
    bucket_name, *prefix_parts = gcs_path.split("/")
    blob_prefix = "/".join(prefix_parts)
    blob_name = (
        os.path.join(blob_prefix, filename) if blob_prefix else filename
    )
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(output_file)
    log.info("Uploaded results to gs://%s/%s", bucket_name, blob_name)

  return output_file
