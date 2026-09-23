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
  p.add_argument("--worker_addresses", nargs="+", default=["localhost:20001"])
  p.add_argument("--port", type=int, default=20001)
  p.add_argument("--model_id", default=MODEL_ID)
  p.add_argument("--tokenizer_path", default=MODEL_ID)
  p.add_argument("--model_absolute_path", default="")
  p.add_argument("--maxtext_model_name", default="qwen3.5-35b-a3b")
  p.add_argument("--scan_layers", type=boolean, default=False)
  p.add_argument(
      "--mesh_fsdp", type=int, default=32, help="Rollout data parallel size."
  )
  p.add_argument("--mesh_tp", type=int, default=2)
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
  p.add_argument("--timeout", type=float, default=3600)
  p.add_argument("--reward_timeout", type=int, default=1800)
  p.add_argument("--step_timeout", type=int, default=600)
  p.add_argument("--startup_timeout", type=float, default=7200)
  p.add_argument("--temperature", type=float, default=0.7)
  p.add_argument("--top_p", type=float, default=1.0)
  p.add_argument("--top_k", type=int, default=0)
  p.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Engine RNG seed; concurrent request ordering can affect samples.",
  )
  p.add_argument("--enable_thinking", type=boolean, default=True)
  p.add_argument("--enable_prefix_caching", type=boolean, default=False)
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
  p.add_argument(
      "--docker_image_prefix",
      default="",
      help="Optional registry prefix to rewrite task Docker images.",
  )
  p.add_argument("--output_dir", default="eval_results")
  a = p.parse_args(argv)
  for name in (
      "mesh_fsdp",
      "mesh_tp",
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
  if a.docker_image_prefix:
    os.environ["IMAGE_REWRITE_PREFIX"] = a.docker_image_prefix
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
      "mesh_tp",
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
  return {
      "model_name": a.maxtext_model_name,
      "load_parameters_path": a.model_absolute_path,
      "scan_layers": a.scan_layers,
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


def request_fields(a, entry, index, attempt):
  """Construct a wire request without importing JAX on the controller."""
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
          "agent_config": {"scaffold": a.scaffold},
      },
  }


def summarize(rows, instance_ids, attempts):
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
  pass_at_k = {}
  for k in sorted({1, attempts}):
    values = []
    for group in grouped.values():
      c = sum(row["resolved"] for row in group)
      values.append(1 - math.comb(attempts - c, k) / math.comb(attempts, k))
    pass_at_k[str(k)] = sum(values) / len(values)
  return {
      "instances": len(instance_ids),
      "attempts_per_instance": attempts,
      "expected_attempts": total,
      "completed_attempts": len(rows),
      "missing_attempts": total - len(rows),
      "error_attempts": sum(bool(row.get("error")) for row in rows),
      "resolved_attempts": solved,
      "avg_at_k": solved / total,
      "pass_at_k": pass_at_k,
      "mean_reward": sum(row["reward"] for row in rows) / total,
      "status_counts": dict(collections.Counter(row["status"] for row in rows)),
      "complete": len(rows) == total,
  }


def compact_result(response):
  if response.error is not None or response.payload is None:
    return {
        "reward": 0.0,
        "resolved": False,
        "status": response.status,
        "error": str(response.error or "Missing trajectory"),
    }
  traj = response.payload.traj
  reward = float(traj["trajectory_reward"])
  if not math.isfinite(reward):
    raise ValueError("Non-finite trajectory reward")
  return {
      "reward": reward,
      "resolved": reward > 0,
      "status": str(traj.get("status", "UNKNOWN")),
      "error": None,
  }


async def evaluate_worker(handle, jobs, limit, timeout, write_record):
  """Bounded dispatch with short RPCs; episode duration is not an RPC deadline."""
  pending = {}
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
      row.update(
          instance_id=fields["metadata"]["instance_id"],
          attempt=fields["group_index"],
          request_id=reply.request_id,
          wall_seconds=time.monotonic() - start,
      )
      await asyncio.to_thread(write_record, row)
      rows.append(row)
    if any(time.monotonic() - start > timeout for _, start in pending.values()):
      raise TimeoutError("Worker exceeded collection and cleanup deadline")
  return rows


class ResultWriter:
  """Each attempt is durably written separately, including on GCS."""

  def __init__(self, output):
    import fsspec  # pylint: disable=import-outside-toplevel

    self.fs, self.root = fsspec.core.url_to_fs(output)
    self.fs.makedirs(self.root, exist_ok=True)

  def write(self, name, value):
    path = self.root + "/" + name
    self.fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
    with self.fs.open(path, "w") as stream:
      json.dump(value, stream, ensure_ascii=False, allow_nan=False)

  def record(self, row):
    self.write("attempts/" + row["request_id"] + ".json", row)


def load_entries(a):
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
  for entry in entries:
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
  if not ids or len(set(ids)) != len(ids):
    raise ValueError(
        "Evaluation split must be nonempty with unique instance_id values"
    )
  return entries


async def run_controller(a):
  from tunix.experimental.worker import remote_execution

  entries = load_entries(a)
  run_id = (
      datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
      + "-"
      + uuid.uuid4().hex[:8]
  )
  output = a.output_dir.rstrip("/") + "/" + run_id
  writer = ResultWriter(output)
  writer.write(
      "config.json",
      {
          **vars(a),
          "run_id": run_id,
          "dataset_sha256": (
              hashlib.sha256(
                  json.dumps(entries, sort_keys=True).encode()
              ).hexdigest()
          ),
          "instance_ids": [str(e["instance_id"]) for e in entries],
      },
  )
  logging.info("Evaluation output: %s", output)
  handles = [
      remote_execution.ActorHandle.from_address(
          address if address.startswith("grpc://") else "grpc://" + address,
          rpc_timeout_s=60,
      )
      for address in a.worker_addresses
  ]
  all_rows = []
  tasks = []
  failure = None
  fleet = None
  entry_stream = entries

  def record(row):
    writer.record(row)
    all_rows.append(row)

  try:

    async def ready(handle):
      deadline = time.monotonic() + a.startup_timeout
      while True:
        try:
          profile = await handle.asubmit("evaluation_info")
          break
        except Exception:
          if time.monotonic() >= deadline:
            raise TimeoutError("Rollout worker did not become ready")
          await asyncio.sleep(5)
      if profile != model_profile(a):
        raise ValueError(f"Worker/controller model settings differ: {profile}")

    await asyncio.gather(*(ready(h) for h in handles))
    if a.use_agent_sandbox:
      from examples.deepswe import sandbox_utils  # pylint: disable=import-outside-toplevel

      if a.docker_image_prefix:
        os.environ["IMAGE_REWRITE_PREFIX"] = a.docker_image_prefix
      fleet = sandbox_utils.init_global_fleet(
          tasks=entries,
          max_concurrency=a.max_concurrent,
          num_generations=a.num_rollouts_per_instance,
          batch_size=a.batch_size,
          max_warmpool_replicas=a.max_warmpool_size,
          scaffold=a.scaffold,
      )
      entry_stream = sandbox_utils.PrewarmDatasetIterator(
          entries,
          fleet=fleet,
          num_generations=a.num_rollouts_per_instance,
          batch_size=a.batch_size,
          max_warmpool_replicas=a.max_warmpool_size,
          unwarm_on_exhaustion=True,
          scaffold=a.scaffold,
          wait_initial=True,
      )
    jobs = iter(
        request_fields(a, entry, index, attempt)
        for index, entry in enumerate(entry_stream)
        for attempt in range(a.num_rollouts_per_instance)
    )
    effective_concurrency = a.max_concurrent
    if a.use_agent_sandbox:
      effective_concurrency = max(
          len(handles),
          min(a.max_concurrent, a.batch_size * a.num_rollouts_per_instance),
      )
    deadline = a.timeout + a.reward_timeout + 360
    for index, handle in enumerate(handles):
      limit = effective_concurrency // len(handles) + (
          index < effective_concurrency % len(handles)
      )
      tasks.append(
          asyncio.create_task(
              evaluate_worker(handle, jobs, limit, deadline, record)
          )
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
    summary = summarize(
        all_rows,
        [str(e["instance_id"]) for e in entries],
        a.num_rollouts_per_instance,
    )
    summary["fatal_error"] = failure
    try:
      writer.write("summary.json", summary)
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
    # Run this file directly so Pathways initializes before tunix.__init__
    # imports JAX and the model stack.
    if "proxy" in os.environ.get("JAX_PLATFORMS", "").split(","):
      import pathwaysutils

      pathwaysutils.initialize()
    from tunix.experimental.examples.deepswe_dist import eval_worker

    asyncio.run(eval_worker.serve(a))
  else:
    # Must precede importing the DTOs, registry, or dataset wrapper.
    os.environ["JAX_PLATFORMS"] = "cpu"
    asyncio.run(run_controller(a))


if __name__ == "__main__":
  main()
