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

"""Rollout-only distributed DeepSWE evaluation; no trainer or RL learner.

The controller uses ActorHandle DispatchTask/PollResponses, and the worker
uses RolloutWorker/RolloutManager. Heavy dependencies are imported only in
the process that needs them.
"""

import argparse
import asyncio
import collections
import datetime
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
import uuid

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"


def boolean(value):
  if isinstance(value, bool):
    return value
  if value.lower() in ("true", "1"):
    return True
  if value.lower() in ("false", "0"):
    return False
  raise argparse.ArgumentTypeError("Expected true/false or 1/0")


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument(
      "--role", choices=("controller", "worker"), default="controller"
  )
  p.add_argument(
      "--worker_id",
      default="deepswe-eval",
      help=(
          "Worker identifier; numeric replica suffix offsets the vLLM engine"
          " seed modulo 2**32."
      ),
  )
  p.add_argument("--worker_addresses", nargs="+", default=["localhost:20001"])
  p.add_argument("--port", type=int, default=20001)
  p.add_argument("--model_id", default=MODEL_ID)
  p.add_argument("--tokenizer_path", default=MODEL_ID)
  p.add_argument("--model_absolute_path", default="")
  p.add_argument("--maxtext_model_name", default="qwen3.5-35b-a3b")
  p.add_argument(
      "--scan_layers",
      type=boolean,
      default=False,
      help=(
          "Whether the input checkpoint uses scanned layers; when true,"
          " converts to unscanned weights in-memory at worker startup."
      ),
  )
  p.add_argument(
      "--mesh_fsdp",
      type=int,
      default=1,
      help="Rollout fully sharded data parallel size (FSDP).",
  )
  p.add_argument(
      "--mesh_dp",
      type=int,
      default=None,
      help=(
          "Rollout data parallel size (DP). If unset, defaults to"
          " VLLM_DATA_PARALLEL_SIZE env var or 1."
      ),
  )
  p.add_argument("--mesh_tp", type=int, default=2)
  p.add_argument(
      "--mesh_expert",
      type=int,
      default=1,
      help="Rollout expert parallel size.",
  )
  p.add_argument("--vllm_utilization", type=float, default=0.70)
  p.add_argument("--max_model_len", type=int, default=16384)
  p.add_argument(
      "--max_context_limit",
      type=int,
      default=0,
      help="Cumulative response token budget across turns (0 disables).",
  )
  p.add_argument(
      "--max_response_length",
      type=int,
      default=12288,
      help="Per-turn generation cap.",
  )
  p.add_argument(
      "--max_steps", type=int, default=30, help="Agent turns per attempt."
  )
  p.add_argument("--max_concurrent", type=int, default=128)
  p.add_argument(
      "--batch_size",
      type=int,
      default=16,
      help="Number of tasks in the active warmpool prewarming window.",
  )
  p.add_argument("--vllm_max_num_seqs", type=int, default=128)
  p.add_argument("--vllm_max_num_batched_tokens", type=int, default=32768)
  p.add_argument("--timeout", type=float, default=1800)
  p.add_argument("--reward_timeout", type=int, default=60)
  p.add_argument("--step_timeout", type=int, default=60)
  p.add_argument("--startup_timeout", type=float, default=7200)
  p.add_argument("--temperature", type=float, default=0.7)
  p.add_argument("--top_p", type=float, default=1.0)
  p.add_argument("--top_k", type=int, default=0)
  p.add_argument(
      "--seed",
      type=int,
      default=42,
      help=(
          "Base engine RNG seed; offset by the numeric replica suffix of"
          " --worker_id modulo 2**32."
      ),
  )
  p.add_argument("--enable_thinking", type=boolean, default=False)
  p.add_argument("--enable_prefix_caching", type=boolean, default=False)
  p.add_argument("--exact_token_continuity", type=boolean, default=True)
  p.add_argument("--checkpoint_storage_use_ocdbt", type=boolean, default=True)
  p.add_argument("--checkpoint_storage_use_zarr3", type=boolean, default=False)
  p.add_argument("--checkpoint_storage_concurrent_gb", type=int, default=96)
  p.add_argument(
      "--use_ocdbt_with_pathways",
      type=boolean,
      default=True,
      help="Use standard Orbax ArrayHandler for OCDBT restore on the head.",
  )
  p.add_argument("--dataset_name", default="R2E-Gym/SWE-Bench-Verified")
  p.add_argument("--dataset_path", default="")
  p.add_argument("--dataset_split", default="test")
  p.add_argument("--tasks_limit", type=int, default=0)
  p.add_argument("--num_rollouts_per_instance", type=int, default=4)
  p.add_argument(
      "--scaffold",
      choices=("openhands", "r2egym", "sweagent"),
      default="openhands",
  )
  p.add_argument(
      "--use_agent_sandbox", type=boolean, nargs="?", const=True, default=True
  )
  p.add_argument("--max_warmpool_size", type=int, default=1)
  p.add_argument("--output_dir", default="eval_results")
  p.add_argument(
      "--rcp_logging",
      type=boolean,
      nargs="?",
      const=True,
      default=os.environ.get("RCP_LOGGING", "0").lower() in ("1", "true"),
      help="Enable MLPerf RCP (mllog) compliance logging.",
  )
  p.add_argument(
      "--metric_logger_dir",
      default=os.environ.get("METRIC_LOGGER_DIR", ""),
      help="Directory or GCS URI for MLPerf RCP output (seed_<seed>.out).",
  )
  p.add_argument(
      "--target_accuracy",
      type=float,
      default=float(os.environ.get("TARGET_ACCURACY", "0.69")),
      help="Target evaluation accuracy for MLPerf RCP compliance logging.",
  )
  p.add_argument(
      "--checkpoint_step",
      type=int,
      default=int(os.environ.get("CHECKPOINT_STEP", "0")),
      help="Optimizer step corresponding to the evaluated checkpoint.",
  )
  p.add_argument(
      "--checkpoint_timestamp_ms",
      type=int,
      default=(
          int(os.environ["CHECKPOINT_TIMESTAMP_MS"])
          if os.environ.get("CHECKPOINT_TIMESTAMP_MS", "").strip()
          else None
      ),
      help=(
          "Training epoch timestamp (ms) when the checkpoint weights were "
          "updated, used to backdate run_stop."
      ),
  )
  p.add_argument(
      "--samples_count",
      type=int,
      default=int(os.environ.get("SAMPLES_COUNT", "0")),
      help="Cumulative training samples at checkpoint_step.",
  )
  p.add_argument(
      "--is_last_checkpoint",
      type=boolean,
      nargs="?",
      const=True,
      default=os.environ.get("IS_LAST_CHECKPOINT", "0").lower()
      in ("1", "true"),
      help="Whether this checkpoint is the final checkpoint in the manifest.",
  )
  a = p.parse_args(argv)
  if a.mesh_dp is None:
    a.mesh_dp = int(os.environ.get("VLLM_DATA_PARALLEL_SIZE", 0)) or 1
  for name in (
      "mesh_fsdp",
      "mesh_dp",
      "mesh_tp",
      "mesh_expert",
      "max_model_len",
      "max_response_length",
      "max_steps",
      "max_concurrent",
      "batch_size",
      "vllm_max_num_seqs",
      "vllm_max_num_batched_tokens",
      "timeout",
      "startup_timeout",
      "reward_timeout",
      "step_timeout",
      "num_rollouts_per_instance",
      "max_warmpool_size",
      "checkpoint_storage_concurrent_gb",
  ):
    if getattr(a, name) <= 0:
      p.error(f"--{name} must be positive")
  if (
      not 0 < a.vllm_utilization < 1
      or not 0 < a.top_p <= 1
      or a.temperature < 0
  ):
    p.error("Invalid HBM utilization or sampling probabilities")
  if a.tasks_limit < 0 or a.seed < 0 or a.max_context_limit < 0:
    p.error("--tasks_limit, --seed, and --max_context_limit must be nonnegative")
  if not a.model_absolute_path:
    p.error("MaxText eval requires --model_absolute_path (Orbax .../items)")
  if len(set(a.worker_addresses)) != len(a.worker_addresses):
    p.error("Duplicate worker addresses")
  if a.max_concurrent < len(a.worker_addresses):
    p.error("--max_concurrent must be at least the number of workers")
  return a


def model_profile(a):
  """Settings the controller verifies against each ready worker."""
  names = (
      "model_id",
      "tokenizer_path",
      "model_absolute_path",
      "maxtext_model_name",
      "scan_layers",
      "mesh_fsdp",
      "mesh_dp",
      "mesh_tp",
      "mesh_expert",
      "max_model_len",
      "enable_thinking",
      "enable_prefix_caching",
      "vllm_utilization",
      "vllm_max_num_seqs",
      "vllm_max_num_batched_tokens",
      "scaffold",
      "use_agent_sandbox",
      "max_concurrent",
      "seed",
  )
  return {name: getattr(a, name) for name in names}


def maxtext_config(a):
  """Native adapter restores the checkpoint directly, never dummy weights."""
  cfg = {
      "model_name": a.maxtext_model_name,
      "load_parameters_path": a.model_absolute_path,
      "scan_layers": False,
      "model_call_mode": "inference",
      "attention": "vllm_rpa",
      "allow_split_physical_axes": True,
      "enable_dp_attention": False,
      "remat_policy": "none",
      "weight_dtype": "bfloat16",
      "prefuse_moe_weights": True,
      "use_multimodal": False,
      "skip_jax_distributed_system": True,
      "log_config": False,
      "checkpoint_storage_use_ocdbt": a.checkpoint_storage_use_ocdbt,
      "checkpoint_storage_use_zarr3": a.checkpoint_storage_use_zarr3,
      "checkpoint_storage_concurrent_gb": a.checkpoint_storage_concurrent_gb,
  }
  if os.environ.get("FLOAT32_GATE_LOGITS"):
    cfg["float32_gate_logits"] = boolean(os.environ["FLOAT32_GATE_LOGITS"])
  if os.environ.get("FLOAT32_LOGITS"):
    cfg["float32_logits"] = boolean(os.environ["FLOAT32_LOGITS"])
  raw_add_cfg = os.environ.get("VLLM_ADDITIONAL_CONFIG", "").strip()
  if raw_add_cfg:
    try:
      extra_mt_cfg = json.loads(raw_add_cfg).get("maxtext_config")
      if isinstance(extra_mt_cfg, dict) and "fp8_moe" in extra_mt_cfg:
        cfg["rollout_fp8_moe"] = bool(extra_mt_cfg["fp8_moe"])
    except Exception:  # pylint: disable=broad-exception-caught
      pass
  return cfg


def request_fields(a, entry, index, attempt):
  """Construct a wire request without importing JAX on the controller."""
  # template is stdlib-only, so this keeps the controller JAX-free.
  from examples.deepswe import template  # pylint: disable=import-outside-toplevel

  instance_id = str(entry["instance_id"])
  prompt_id = f"eval_{index}"
  max_context_limit = getattr(a, "max_context_limit", 0)
  return {
      "request_id": f"{prompt_id}_{attempt}",
      "prompt_id": prompt_id,
      "group_index": attempt,
      "prompt": str(entry["problem_statement"]),
      "max_turns": a.max_steps,
      "max_response_length": max_context_limit if max_context_limit > 0 else None,
      "exact_token_continuity": getattr(a, "exact_token_continuity", True),
      "generation_kwargs": {
          "max_generation_steps": a.max_response_length,
          "temperature": a.temperature,
          "top_p": a.top_p,
          "top_k": None if a.top_k < 0 else a.top_k,
          # TPU inference uses the engine RNG, not per-request seeds.
          "return_logprobs": False,
      },
      "metadata": {
          "instance_id": instance_id,
          "episode_timeout": a.timeout,
          "overlong_filter": False,
          "exact_token_continuity": getattr(a, "exact_token_continuity", True),
          "env_config": {
              "entry": entry,
              "prompt_id": prompt_id,
              "max_steps": a.max_steps,
              "num_generations": a.num_rollouts_per_instance,
              "step_timeout": a.step_timeout,
              "reward_timeout": a.reward_timeout,
              "backend": "kubernetes" if a.use_agent_sandbox else "docker",
              "use_agent_sandbox": a.use_agent_sandbox,
              "scaffold": a.scaffold,
              "verbose": False,
          },
          "agent_name": template.get_agent_name(a.scaffold),
          "agent_config": {"scaffold": a.scaffold},
      },
  }


def summarize(rows, instance_ids, attempts, entry_by_iid=None):
  """Conservative metrics: failed/missing attempts stay in the denominator."""
  grouped = {key: [] for key in instance_ids}
  seen = set()
  for row in rows:
    key = (row["instance_id"], row["attempt"])
    if key in seen or key[0] not in grouped or not 0 <= key[1] < attempts:
      raise ValueError(f"Invalid/duplicate evaluation result: {key}")
    seen.add(key)
    grouped[key[0]].append(row)
  total = len(instance_ids) * attempts
  solved = sum(row["resolved"] for row in rows)
  per_instance_resolved_counts = {
      iid: int(sum(bool(row["resolved"]) for row in grouped[iid]))
      for iid in instance_ids
  }
  per_instance_completed_counts = {
      iid: len(grouped[iid]) for iid in instance_ids
  }
  completed_instance_ids = [
      iid for iid in instance_ids if len(grouped[iid]) == attempts
  ]
  touched_instance_ids = [
      iid for iid in instance_ids if len(grouped[iid]) > 0
  ]
  resolved_instance_ids = [
      iid for iid in instance_ids if per_instance_resolved_counts[iid] > 0
  ]
  unresolved_instance_ids = [
      iid for iid in instance_ids if per_instance_resolved_counts[iid] == 0
  ]
  resolved_instances = len(resolved_instance_ids)
  unresolved_instances = len(unresolved_instance_ids)
  instance_pass_rate = (
      resolved_instances / len(instance_ids) if instance_ids else 0.0
  )
  pass_at_k = {}
  completed_pass_at_k = {}
  ks = {1, attempts}
  if attempts >= 4:
    ks.add(4)
  for k in sorted(ks):
    values = []
    comp_values = []
    for iid, group in grouped.items():
      c = sum(row["resolved"] for row in group)
      val = 1 - math.comb(attempts - c, k) / math.comb(attempts, k)
      values.append(val)
      if len(group) == attempts:
        comp_values.append(val)
    pass_at_k[str(k)] = sum(values) / len(values)
    completed_pass_at_k[str(k)] = (
        sum(comp_values) / len(comp_values) if comp_values else 0.0
    )
  per_difficulty = {}
  if entry_by_iid:
    by_lvl = collections.defaultdict(list)
    for iid in instance_ids:
      ent = entry_by_iid.get(iid) or {}
      lvl = ent.get("difficulty_level") or ent.get("difficulty") or "unknown"
      by_lvl[lvl].append(iid)
    for lvl, lvl_iids in sorted(by_lvl.items()):
      lvl_rows = [r for iid in lvl_iids for r in grouped[iid]]
      lvl_comp_iids = [iid for iid in lvl_iids if len(grouped[iid]) == attempts]
      lvl_res_iids = [iid for iid in lvl_iids if per_instance_resolved_counts[iid] > 0]
      lvl_solved = sum(bool(r["resolved"]) for r in lvl_rows)
      per_difficulty[lvl] = {
          "total_instances": len(lvl_iids),
          "completed_instances": len(lvl_comp_iids),
          "touched_instances": sum(1 for iid in lvl_iids if grouped[iid]),
          "resolved_instances": len(lvl_res_iids),
          "completed_attempts": len(lvl_rows),
          "resolved_attempts": int(lvl_solved),
          "pass_at_1_so_far": (
              lvl_solved / len(lvl_rows) if lvl_rows else 0.0
          ),
          "instance_pass_rate_on_completed": (
              sum(1 for iid in lvl_comp_iids if per_instance_resolved_counts[iid] > 0)
              / len(lvl_comp_iids)
              if lvl_comp_iids
              else 0.0
          ),
      }
  return {
      "instances": len(instance_ids),
      "completed_instances": len(completed_instance_ids),
      "touched_instances": len(touched_instance_ids),
      "resolved_instances": resolved_instances,
      "unresolved_instances": unresolved_instances,
      "instance_pass_rate": instance_pass_rate,
      "completed_instance_pass_rate": (
          sum(1 for iid in completed_instance_ids if per_instance_resolved_counts[iid] > 0)
          / len(completed_instance_ids)
          if completed_instance_ids
          else 0.0
      ),
      "resolved_instance_ids": resolved_instance_ids,
      "unresolved_instance_ids": unresolved_instance_ids,
      "per_instance_resolved_counts": per_instance_resolved_counts,
      "per_instance_completed_counts": per_instance_completed_counts,
      "attempts_per_instance": attempts,
      "expected_attempts": total,
      "completed_attempts": len(rows),
      "missing_attempts": total - len(rows),
      "error_attempts": sum(bool(row.get("error")) for row in rows),
      "resolved_attempts": solved,
      "avg_at_k": solved / total,
      "avg_at_k_completed_attempts": (
          solved / len(rows) if rows else 0.0
      ),
      "pass_at_k": pass_at_k,
      "completed_instances_pass_at_k": completed_pass_at_k,
      "per_difficulty_summary": per_difficulty,
      "mean_reward": sum(row["reward"] for row in rows) / total,
      "status_counts": dict(collections.Counter(row["status"] for row in rows)),
      "complete": len(rows) == total,
  }


def compact_result(response):
  if response.error is not None or response.payload is None:
    result = {
        "reward": 0.0,
        "resolved": False,
        "status": response.status,
        "error": str(response.error or "Missing trajectory"),
    }
  else:
    traj = response.payload.traj
    reward = float(traj["trajectory_reward"])
    if not math.isfinite(reward):
      raise ValueError("Non-finite trajectory reward")
    model_time = traj.get("model_time") if isinstance(traj.get("model_time"), dict) else {}
    reward_time = traj.get("reward_time") if isinstance(traj.get("reward_time"), dict) else {}
    env_time = traj.get("env_time") if isinstance(traj.get("env_time"), dict) else {}
    step_lats = model_time.get("step_latency")
    steps = len(step_lats) if isinstance(step_lats, (list, tuple)) else None
    result = {
        "reward": reward,
        "resolved": reward > 0,
        "status": str(traj.get("status", "UNKNOWN")),
        "error": None,
    }
    if steps is not None:
      result["steps"] = int(steps)
    if "total_latency" in model_time:
      result["model_time_s"] = round(float(model_time["total_latency"]), 3)
    if "total_completion_tokens" in model_time:
      result["completion_tokens"] = int(model_time["total_completion_tokens"])
    if "reward_latency" in reward_time:
      result["reward_time_s"] = round(float(reward_time["reward_latency"]), 3)
    if env_time:
      result["env_time_s"] = round(
          float(env_time.get("reset_latency", 0.0))
          + float(env_time.get("total_step_latency", 0.0))
          + float(env_time.get("close_latency", 0.0)),
          3,
      )
    if "total_time" in traj and traj["total_time"] is not None:
      result["total_time_s"] = round(float(traj["total_time"]), 3)
  # Worker-reported metadata (success and error); evaluate_worker falls back to
  # the requested agent_name if the worker did not report one.
  if response.metadata.get("agent_name"):
    result["agent_name"] = str(response.metadata["agent_name"])
  return result


async def evaluate_worker(handle, jobs, limit, timeout, write_record):
  """Bounded dispatch with short RPCs; episode duration is not an RPC deadline."""
  pending = {}
  timed_out = set()
  rows = []
  exhausted = False
  while pending or not exhausted:
    while not exhausted and len(pending) < limit:
      try:
        fields = next(jobs)
      except StopIteration:
        exhausted = True
        break
      key = fields["request_id"]
      ack = await handle.dispatch_task(key, "evaluate", fields)
      if ack != key:
        raise RuntimeError(f"Unexpected dispatch acknowledgement: {ack}")
      pending[key] = (fields, time.monotonic())
    if not pending:
      break
    reply = await handle.poll_responses(timeout_s=5)
    if reply is not None:
      if reply.request_id in timed_out:
        timed_out.discard(reply.request_id)
        continue
      if reply.request_id not in pending:
        raise RuntimeError(f"Unrecognized/duplicate result: {reply.request_id}")
      fields, start = pending.pop(reply.request_id)
      try:
        row = reply.unwrap()
      except (
          Exception
      ) as exc:  # Record the failure; don't silently drop a task.
        row = {
            "reward": 0.0,
            "resolved": False,
            "status": "ERROR",
            "error": str(exc),
        }
      requested_agent = fields["metadata"].get("agent_name")
      if requested_agent and not row.get("agent_name"):
        row["agent_name"] = requested_agent
      entry_obj = fields["metadata"].get("env_config", {}).get("entry") or {}
      if entry_obj.get("difficulty_level") and not row.get("difficulty_level"):
        row["difficulty_level"] = str(entry_obj["difficulty_level"])
      if entry_obj.get("repo") and not row.get("repo"):
        row["repo"] = str(entry_obj["repo"])
      row.update(
          instance_id=fields["metadata"]["instance_id"],
          attempt=fields["group_index"],
          request_id=reply.request_id,
          wall_seconds=time.monotonic() - start,
      )
      await asyncio.to_thread(write_record, row)
      rows.append(row)
    now = time.monotonic()
    expired = [
        req_id
        for req_id, (_, start) in pending.items()
        if now - start > timeout
    ]
    for req_id in expired:
      fields, start = pending.pop(req_id)
      timed_out.add(req_id)
      logging.warning(
          "Request %s (instance_id=%s) exceeded controller deadline"
          " (%.1fs > %.1fs); recording TIMEOUT.",
          req_id,
          fields["metadata"]["instance_id"],
          now - start,
          timeout,
      )
      row = {
          "reward": 0.0,
          "resolved": False,
          "status": "TIMEOUT",
          "error": None,
          "instance_id": fields["metadata"]["instance_id"],
          "attempt": fields["group_index"],
          "request_id": req_id,
          "wall_seconds": now - start,
      }
      entry_obj = fields["metadata"].get("env_config", {}).get("entry") or {}
      if entry_obj.get("difficulty_level"):
        row["difficulty_level"] = str(entry_obj["difficulty_level"])
      if entry_obj.get("repo"):
        row["repo"] = str(entry_obj["repo"])
      requested_agent = fields["metadata"].get("agent_name")
      if requested_agent:
        row["agent_name"] = requested_agent
      await asyncio.to_thread(write_record, row)
      rows.append(row)
  return rows


class ResultWriter:
  """Each attempt is durably written separately, including on GCS."""

  def __init__(self, output, mirror_output=None):
    import fsspec  # pylint: disable=import-outside-toplevel

    self.fs, self.root = fsspec.core.url_to_fs(output)
    self.fs.makedirs(self.root, exist_ok=True)
    self.mirror_fs = None
    self.mirror_root = None
    if mirror_output and mirror_output.rstrip("/") != output.rstrip("/"):
      try:
        self.mirror_fs, self.mirror_root = fsspec.core.url_to_fs(mirror_output)
        self.mirror_fs.makedirs(self.mirror_root, exist_ok=True)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        logging.warning("Could not initialize mirror output %s: %s", mirror_output, exc)
        self.mirror_fs = None
        self.mirror_root = None

  def write(self, name, value):
    path = self.root + "/" + name
    self.fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
    with self.fs.open(path, "w") as stream:
      json.dump(value, stream, ensure_ascii=False, allow_nan=False)
    if self.mirror_fs is not None and self.mirror_root is not None:
      try:
        mpath = self.mirror_root + "/" + name
        self.mirror_fs.makedirs(mpath.rsplit("/", 1)[0], exist_ok=True)
        with self.mirror_fs.open(mpath, "w") as stream:
          json.dump(value, stream, ensure_ascii=False, allow_nan=False)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        logging.warning("Mirror write failed for %s: %s", name, exc)

  def write_text(self, name, text):
    path = self.root + "/" + name
    self.fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
    with self.fs.open(path, "w") as stream:
      stream.write(text)
    if self.mirror_fs is not None and self.mirror_root is not None:
      try:
        mpath = self.mirror_root + "/" + name
        self.mirror_fs.makedirs(mpath.rsplit("/", 1)[0], exist_ok=True)
        with self.mirror_fs.open(mpath, "w") as stream:
          stream.write(text)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        logging.warning("Mirror write_text failed for %s: %s", name, exc)

  def record(self, row):
    self.write("attempts/" + row["request_id"] + ".json", row)

  def load_existing_attempts(self):
    rows = []
    try:
      files = sorted(self.fs.glob(self.root + "/attempts/*.json"))
      for p in files:
        try:
          with self.fs.open(p, "r") as f:
            rows.append(json.load(f))
        except Exception as exc:  # pylint: disable=broad-exception-caught
          logging.warning("Could not read existing attempt %s: %s", p, exc)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      logging.warning("Could not glob existing attempts in %s: %s", self.root, exc)
    return rows


def build_per_task_jsonl(rows, entries, attempts):
  grouped = collections.defaultdict(list)
  for r in rows:
    grouped[r["instance_id"]].append(r)
  lines = []
  for idx, entry in enumerate(entries):
    iid = str(entry["instance_id"])
    atts = sorted(grouped.get(iid, []), key=lambda x: x.get("attempt", 0))
    if not atts:
      continue
    meta = entry.get("meta") or {}
    if isinstance(meta, str):
      try:
        meta = json.loads(meta)
      except Exception:  # pylint: disable=broad-exception-caught
        meta = {}
    resolved_cnt = sum(1 for a in atts if a.get("resolved"))
    steps_list = [a["steps"] for a in atts if isinstance(a.get("steps"), int)]
    rec = {
        "task_index": idx,
        "instance_id": iid,
        "repo": entry.get("repo"),
        "difficulty_level": entry.get("difficulty_level"),
        "num_modified_files": meta.get("num_modified_files"),
        "num_modified_lines": meta.get("num_modified_lines"),
        "expected_attempts": attempts,
        "completed_attempts": len(atts),
        "complete": len(atts) == attempts,
        "resolved_count": resolved_cnt,
        "pass_at_1_so_far": round(resolved_cnt / len(atts), 4),
        "resolved_any": 1 if resolved_cnt > 0 else 0,
        "avg_steps": (
            round(sum(steps_list) / len(steps_list), 1) if steps_list else None
        ),
        "status_counts": dict(
            collections.Counter(a.get("status", "UNKNOWN") for a in atts)
        ),
    }
    lines.append(json.dumps(rec, ensure_ascii=False))
  return "\n".join(lines) + ("\n" if lines else "")


def load_entries(a):
  from examples.deepswe import harbor_utils
  from tunix.experimental.examples.deepswe_dist import deepswe

  if a.tasks_limit and not a.dataset_path:
    from datasets import load_dataset

    # A smoke test should not download and materialize the entire split.
    dataset = load_dataset(
        a.dataset_name, split=a.dataset_split, streaming=True
    )
    entries = [
        deepswe._jsonify_lists(dict(entry))
        for entry in dataset.take(a.tasks_limit)
    ]
  elif a.dataset_path and harbor_utils.is_jsonl_or_harbor_path(
      a.dataset_path
  ):
    entries = harbor_utils.load_dataset_records(
        a.dataset_path,
        max_rows=a.tasks_limit if a.tasks_limit > 0 else None,
    )
  else:
    dataset = deepswe.load_deepswe_dataset(
        dataset_name=a.dataset_name,
        dataset_split=a.dataset_split,
        dataset_path=a.dataset_path,
        shuffle=False,
    )
    count = min(a.tasks_limit, len(dataset)) if a.tasks_limit else len(dataset)
    entries = [dict(dataset[i]) for i in range(count)]
  ids = []
  normalized_entries = []
  for raw_entry in entries:
    entry = (
        harbor_utils.ensure_harbor_entry(raw_entry)
        if harbor_utils.is_harbor_entry(raw_entry)
        else raw_entry
    )
    # R2E-Gym releases identify tasks by their versioned Docker image;
    # SWE-Bench releases provide instance_id directly.
    if not entry.get("instance_id"):
      entry["instance_id"] = entry.get("docker_image")
    if not all(
        entry.get(key)
        for key in ("instance_id", "problem_statement", "docker_image")
    ):
      raise ValueError(
          "Each eval row needs instance_id, problem_statement and docker_image"
      )
    ids.append(str(entry["instance_id"]))
    normalized_entries.append(entry)
  if not ids or len(set(ids)) != len(ids):
    raise ValueError(
        "Evaluation split must be nonempty with unique instance_id values"
    )
  return normalized_entries


def setup_rcp_logging(a):
  """Points mllog at the RCP log; fails fast if mlperf_logging is missing."""
  if not a.rcp_logging:
    return
  from tunix.utils import mllog_utils

  # Without mlperf_logging every mllog call is a silent no-op, so fail before
  # the eval runs instead of dropping its eval_* / run_stop events.
  if mllog_utils.mllogger is None:
    raise RuntimeError(
        "--rcp_logging requires the mlperf_logging package, which is not"
        " installed."
    )
  if a.metric_logger_dir:
    mllog_utils.configure_logger(
        metric_logger_dir=a.metric_logger_dir,
        seed=a.seed,
        append=True,
    )


async def run_controller(a):
  from tunix.experimental.worker import remote_execution
  from tunix.utils import mllog_utils

  setup_rcp_logging(a)

  entries = load_entries(a)
  instance_ids = [str(e["instance_id"]) for e in entries]
  entry_by_iid = {str(e["instance_id"]): e for e in entries}
  run_id = os.environ.get("EVAL_RUN_ID", "").strip() or (
      datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
      + "-"
      + uuid.uuid4().hex[:8]
  )
  output = a.output_dir.rstrip("/") + "/" + run_id
  mirror_base = os.environ.get("EVAL_MIRROR_OUTPUT_DIR", "").strip()
  mirror_output = (
      mirror_base.rstrip("/") + "/" + run_id if mirror_base else None
  )
  writer = ResultWriter(output, mirror_output=mirror_output)
  existing_rows = (
      writer.load_existing_attempts()
      if os.environ.get("EVAL_RUN_ID", "").strip()
      else []
  )
  seen_keys = set()
  all_rows = []
  for r in existing_rows:
    k = (r.get("instance_id"), r.get("attempt"))
    if (
        k not in seen_keys
        and k[0] in entry_by_iid
        and isinstance(k[1], int)
        and 0 <= k[1] < a.num_rollouts_per_instance
    ):
      seen_keys.add(k)
      all_rows.append(r)
  if all_rows:
    logging.info(
        "Resuming evaluation run %s with %d existing completed attempts.",
        run_id,
        len(all_rows),
    )
  writer.write(
      "config.json",
      {
          **vars(a),
          "run_id": run_id,
          "mirror_output": mirror_output,
          "dataset_sha256": (
              hashlib.sha256(
                  json.dumps(entries, sort_keys=True).encode()
              ).hexdigest()
          ),
          "instance_ids": instance_ids,
      },
  )
  logging.info("Evaluation output: %s (mirror: %s)", output, mirror_output)
  handles = [
      remote_execution.ActorHandle.from_address(
          address if address.startswith("grpc://") else "grpc://" + address,
          rpc_timeout_s=60,
      )
      for address in a.worker_addresses
  ]
  tasks = []
  failure = None
  fleet = None
  entry_stream = entries
  t_eval_start = None
  eval_start_time_ms = None
  last_checkpoint_time = [time.monotonic()]

  def flush_checkpoint():
    try:
      partial_summary = summarize(
          all_rows,
          instance_ids,
          a.num_rollouts_per_instance,
          entry_by_iid=entry_by_iid,
      )
      writer.write("summary.json", partial_summary)
      writer.write_text(
          "per_task_results.jsonl",
          build_per_task_jsonl(all_rows, entries, a.num_rollouts_per_instance),
      )
      last_checkpoint_time[0] = time.monotonic()
    except Exception as exc:  # pylint: disable=broad-exception-caught
      logging.warning("Incremental checkpoint write failed: %s", exc)

  def record(row):
    writer.record(row)
    all_rows.append(row)
    now_t = time.monotonic()
    if len(all_rows) % 16 == 0 or (now_t - last_checkpoint_time[0]) >= 30.0:
      flush_checkpoint()

  try:

    async def ready(handle, timeout_s=None, stop_event=None):
      deadline = (
          time.monotonic() + timeout_s if timeout_s is not None else None
      )
      while True:
        if stop_event is not None and stop_event.is_set():
          return False
        try:
          profile = await handle.asubmit("evaluation_info")
          break
        except Exception:
          if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("Rollout worker did not become ready")
          await asyncio.sleep(5)
      if profile != model_profile(a):
        raise ValueError(f"Worker/controller model settings differ: {profile}")
      return True

    await ready(handles[0], timeout_s=a.startup_timeout)
    t_eval_start = time.monotonic()
    eval_start_time_ms = time.time_ns() // 1_000_000
    indexed_entries = [
        (idx, entry)
        for idx, entry in enumerate(entries)
        if any(
            (str(entry["instance_id"]), att) not in seen_keys
            for att in range(a.num_rollouts_per_instance)
        )
    ]
    active_entries = [entry for _, entry in indexed_entries]
    orig_idx_by_iid = {str(entry["instance_id"]): idx for idx, entry in indexed_entries}
    if a.use_agent_sandbox and active_entries:
      from examples.deepswe import sandbox_utils  # pylint: disable=import-outside-toplevel

      fleet = sandbox_utils.init_global_fleet(
          tasks=active_entries,
          max_concurrency=a.max_concurrent,
          num_generations=a.num_rollouts_per_instance,
          batch_size=a.batch_size,
          max_warmpool_replicas=a.max_warmpool_size,
          scaffold=a.scaffold,
      )
      entry_stream = sandbox_utils.PrewarmDatasetIterator(
          active_entries,
          fleet=fleet,
          num_generations=a.num_rollouts_per_instance,
          batch_size=a.batch_size,
          max_warmpool_replicas=a.max_warmpool_size,
          unwarm_on_exhaustion=True,
          scaffold=a.scaffold,
          wait_initial=True,
      )
    else:
      entry_stream = active_entries
    raw_jobs = (
        request_fields(a, entry, orig_idx_by_iid[str(entry["instance_id"])], attempt)
        for entry in entry_stream
        for attempt in range(a.num_rollouts_per_instance)
        if (str(entry["instance_id"]), attempt) not in seen_keys
    )
    jobs_done = asyncio.Event()

    def _iter_jobs_with_done_signal():
      for item in raw_jobs:
        yield item
      jobs_done.set()

    jobs = _iter_jobs_with_done_signal()
    effective_concurrency = a.max_concurrent
    if a.use_agent_sandbox:
      effective_concurrency = max(
          len(handles),
          min(a.max_concurrent, a.batch_size * a.num_rollouts_per_instance),
      )
    deadline = a.timeout + a.reward_timeout + 600

    async def _run_worker_slot(index, handle, limit):
      if index > 0:
        became_ready = await ready(
            handle, timeout_s=None, stop_event=jobs_done
        )
        if not became_ready:
          return []
        logging.info(
            "Rollout worker slot %d (%s) became ready and joined evaluation.",
            index,
            a.worker_addresses[index],
        )
      return await evaluate_worker(handle, jobs, limit, deadline, record)

    for index, handle in enumerate(handles):
      limit = effective_concurrency // len(handles) + (
          index < effective_concurrency % len(handles)
      )
      tasks.append(
          asyncio.create_task(_run_worker_slot(index, handle, limit))
      )
    await asyncio.gather(*tasks)
  except BaseException as exc:
    failure = f"{type(exc).__name__}: {exc}"
    raise
  finally:
    for task in tasks:
      task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    if hasattr(entry_stream, "close"):
      entry_stream.close()
    if fleet is not None:
      from examples.deepswe import sandbox_utils  # pylint: disable=import-outside-toplevel

      await asyncio.to_thread(sandbox_utils.teardown_global_fleet)
    validation_time = (
        time.monotonic() - t_eval_start if t_eval_start is not None else None
    )
    summary = summarize(
        all_rows,
        instance_ids,
        a.num_rollouts_per_instance,
        entry_by_iid=entry_by_iid,
    )
    summary["fatal_error"] = failure
    expected_attempts = int(summary["expected_attempts"])
    error_attempts = int(summary["error_attempts"])
    eval_ok = (
        failure is None
        and bool(summary["complete"])
        and (expected_attempts > 0 and error_attempts < expected_attempts)
    )
    pass_at_k = summary["pass_at_k"]
    eval_accuracy = float(
        pass_at_k["4"]
        if "4" in pass_at_k
        else pass_at_k[str(a.num_rollouts_per_instance)]
    )
    target_acc = float(a.target_accuracy)
    target_reached = bool(eval_ok and eval_accuracy > target_acc)
    rcp_logged = False
    if a.rcp_logging and eval_ok:
      mllog_utils.start_eval(
          step=int(a.checkpoint_step),
          samples_count=int(a.samples_count),
          time_ms=eval_start_time_ms,
      )
      target_reached = mllog_utils.log_offline_eval_step(
          step=int(a.checkpoint_step),
          samples_count=int(a.samples_count),
          eval_accuracy=eval_accuracy,
          target_accuracy=target_acc,
          checkpoint_timestamp_ms=a.checkpoint_timestamp_ms,
          is_last_checkpoint=bool(a.is_last_checkpoint),
          validation_time=validation_time,
          emit_start_eval=False,
      )
      rcp_logged = True
    summary["rcp_logged"] = rcp_logged
    summary["target_accuracy"] = target_acc
    summary["target_reached"] = bool(target_reached)
    summary["checkpoint_step"] = int(a.checkpoint_step)
    summary["checkpoint_timestamp_ms"] = a.checkpoint_timestamp_ms
    summary["samples_count"] = int(a.samples_count)
    try:
      writer.write("summary.json", summary)
      writer.write_text(
          "per_task_results.jsonl",
          build_per_task_jsonl(all_rows, entries, a.num_rollouts_per_instance),
      )
    finally:
      await asyncio.gather(
          *(
              h.asubmit("shutdown")
              for addr, h in zip(a.worker_addresses, handles)
              if not addr.startswith((
                  "localhost:",
                  "127.0.0.1:",
                  "grpc://localhost:",
                  "grpc://127.0.0.1:",
              ))
          ),
          return_exceptions=True,
      )
      await asyncio.gather(
          *(h.close() for h in handles), return_exceptions=True
      )
    logging.info("Evaluation summary: %s", json.dumps(summary))
  if summary["error_attempts"] or not summary["complete"]:
    raise RuntimeError(
        f"Evaluation contains execution failures; inspect {output}"
    )


def main(argv=None):
  a = parse_args(argv)
  logging.basicConfig(
      level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
  )
  sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
  if a.role == "worker":
    try:
      from tunix.experimental.common import gcs_cache  # pylint: disable=g-import-not-at-top
      gcs_cache.restore_jax_cache(role="rollout")
    except Exception as e:  # pylint: disable=broad-except
      logging.warning("[jax_cache] Failed to restore JAX cache: %s", e)

    # Run this file directly so Pathways initializes before tunix.__init__
    # imports JAX and the model stack.
    if "proxy" in os.environ.get("JAX_PLATFORMS", "").split(","):
      import pathwaysutils

      pathwaysutils.initialize()
    from tunix.experimental.examples.deepswe_dist import eval_worker

    try:
      asyncio.run(eval_worker.serve(a))
    finally:
      try:
        from tunix.experimental.common import gcs_cache  # pylint: disable=g-import-not-at-top
        gcs_cache.save_jax_cache(role="rollout")
      except Exception as e:  # pylint: disable=broad-except
        logging.warning("[jax_cache] Failed to save JAX cache: %s", e)
  else:
    # Must precede importing the DTOs, registry, or dataset wrapper.
    os.environ["JAX_PLATFORMS"] = "cpu"
    asyncio.run(run_controller(a))


if __name__ == "__main__":
  main()
