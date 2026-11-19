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

"""Evaluate DeepSWE tasks with the distributed rollout worker, without training.

The defaults match examples/deepswe/eval_deepswe.py: one greedy Qwen3-32B
trajectory per SWE-Bench Verified task, a 30-turn limit, and Pass@1 reporting.
`--num_rollouts_per_instance` samples several attempts per task and adds
Pass@k. Run locally on one TPU host, or run worker and controller roles
separately. Results go to a local or fsspec (e.g. `gs://`) `--output_dir`.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from collections.abc import Iterator
import json
import logging
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[4]
# Model context tokens kept free of responses, so the prompt still fits.
_CONTEXT_HEADROOM_TOKENS = 256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--role", choices=("local", "controller", "worker"),
                      default="local")
  parser.add_argument("--worker_addresses", nargs="+", default=None)
  parser.add_argument("--port", type=int, default=20001)
  parser.add_argument("--model_id", default="Qwen/Qwen3-32B")
  parser.add_argument(
      "--model_dir", default="",
      help="Local Hugging Face model directory; otherwise use model_id.",
  )
  parser.add_argument("--tokenizer_path", default="")
  parser.add_argument("--mesh_fsdp", type=int, default=1)
  parser.add_argument("--mesh_tp", type=int, default=8)
  parser.add_argument("--max_model_len", type=int, default=32768)
  parser.add_argument(
      "--max_context_limit",
      type=int,
      default=0,
      help=(
          "Cumulative response budget; zero uses model length minus"
          f" {_CONTEXT_HEADROOM_TOKENS}."
      ),
  )
  parser.add_argument("--max_response_length", type=int, default=8192,
                      help="Maximum generated tokens per agent turn.")
  parser.add_argument("--max_concurrent", type=int, default=256)
  parser.add_argument("--timeout", type=float, default=600,
                      help="Maximum seconds per trajectory.")
  parser.add_argument("--startup_timeout", type=float, default=1800)
  parser.add_argument("--dataset_name", default="R2E-Gym/SWE-Bench-Verified")
  parser.add_argument("--dataset_split", default="test")
  parser.add_argument("--dataset_cache", default="")
  parser.add_argument("--tasks_limit", type=int, default=0,
                      help="Zero evaluates the complete split.")
  parser.add_argument("--max_steps", type=int, default=30)
  parser.add_argument("--num_rollouts_per_instance", type=int, default=1,
                      help="Attempts per task; Pass@k uses k = this value.")
  parser.add_argument("--temperature", type=float, default=0.0)
  parser.add_argument("--top_p", type=float, default=1.0)
  parser.add_argument("--top_k", type=int, default=-1,
                      help="Top-k sampling; -1 disables it.")
  parser.add_argument("--enable_guard", action="store_true")
  parser.add_argument("--disable_thinking", action="store_true",
                      help="Agentic eval enables the Qwen thinking template.")
  parser.add_argument("--vllm_hbm_utilization", type=float, default=0.4)
  parser.add_argument("--vllm_max_num_seqs", type=int, default=128)
  parser.add_argument("--vllm_max_batched_tokens", type=int, default=165888)
  parser.add_argument("--output_dir", default="eval_results")
  args = parser.parse_args(argv)
  if args.worker_addresses is None:
    args.worker_addresses = [f"127.0.0.1:{args.port}"]
  for name in ("port", "mesh_fsdp", "mesh_tp", "max_model_len",
               "max_response_length", "max_concurrent", "timeout",
               "startup_timeout", "max_steps", "num_rollouts_per_instance",
               "vllm_max_num_seqs", "vllm_max_batched_tokens"):
    if getattr(args, name) <= 0:
      parser.error(f"--{name} must be positive")
  if (args.tasks_limit < 0 or args.max_context_limit < 0
      or not 0 < args.vllm_hbm_utilization < 1):
    parser.error(
        "Task/context limits must be nonnegative and HBM utilization in (0, 1)"
    )
  if args.temperature < 0 or not 0 < args.top_p <= 1:
    parser.error("--temperature must be nonnegative and --top_p in (0, 1]")
  if args.num_rollouts_per_instance > 1 and args.temperature == 0:
    parser.error("--num_rollouts_per_instance > 1 requires --temperature > 0")
  if args.top_k != -1 and args.top_k <= 0:
    parser.error("--top_k must be positive, or -1 to disable it")
  if args.max_response_length > args.max_model_len - _CONTEXT_HEADROOM_TOKENS:
    parser.error(
        f"max_response_length must leave {_CONTEXT_HEADROOM_TOKENS} context"
        " tokens"
    )
  if len(set(args.worker_addresses)) != len(args.worker_addresses):
    parser.error("worker_addresses must be unique")
  if args.max_concurrent < len(args.worker_addresses):
    parser.error("max_concurrent must cover every rollout worker")
  return args


def model_profile(args: argparse.Namespace) -> dict[str, object]:
  """Reject a worker serving a different model or rollout configuration."""
  names = ("model_id", "model_dir", "tokenizer_path", "mesh_fsdp", "mesh_tp",
           "max_model_len", "max_context_limit", "max_response_length", "max_steps",
           "temperature", "top_p", "top_k", "enable_guard", "disable_thinking")
  return {name: getattr(args, name) for name in names}


def load_entries(args: argparse.Namespace) -> list[dict]:
  from datasets import load_dataset  # pylint: disable=g-import-not-at-top

  dataset = load_dataset(
      args.dataset_name,
      split=args.dataset_split,
      cache_dir=args.dataset_cache or None,
  )
  entries = [dict(entry) for entry in dataset if entry.get("docker_image")]
  if args.tasks_limit:
    entries = entries[:args.tasks_limit]
  if not entries:
    raise ValueError("Evaluation dataset contains no Docker-backed tasks")
  return entries


def response_budget(args: argparse.Namespace) -> int:
  """Returns the cumulative response budget of one trajectory."""
  return args.max_context_limit or args.max_model_len - _CONTEXT_HEADROOM_TOKENS


def _request_id(args: argparse.Namespace, index: int, attempt: int) -> str:
  """Returns `eval_<task>`, or `eval_<task>_<attempt>` with several attempts."""
  if args.num_rollouts_per_instance == 1:
    return f"eval_{index}"
  return f"eval_{index}_{attempt}"


def request_fields(
    args: argparse.Namespace,
    entry: dict[str, object],
    index: int,
    attempt: int = 0,
) -> dict[str, object]:
  """Use the same per-trajectory limits as agentic eval; greedy by default."""
  generation_kwargs = {
      "max_generation_steps": args.max_response_length,
      "temperature": args.temperature,
      "return_logprobs": False,
  }
  if args.top_p < 1.0:
    generation_kwargs["top_p"] = args.top_p
  if args.top_k > 0:
    generation_kwargs["top_k"] = args.top_k
  return {
      "request_id": _request_id(args, index, attempt),
      "prompt_id": f"eval_{index}",
      "group_index": attempt,
      "prompt": str(entry.get("problem_statement", "")),
      "max_turns": args.max_steps,
      "max_response_length": response_budget(args),
      "generation_kwargs": generation_kwargs,
      "metadata": {
          "episode_timeout": args.timeout,
          "record_episode_summary": True,
          "env_config": {
              "entry": entry,
              "max_steps": args.max_steps,
              "group_id": index,
              "pair_index": index,
          },
          "agent_config": {"scaffold": "r2egym"},
      },
  }


def compact_result(response) -> dict:
  """Return only the fields used by the existing agentic evaluation report."""
  if response.error is not None or response.payload is None:
    message = str(getattr(response.error, "message", "Missing trajectory"))
    overflow = any(text in message.lower() for text in (
        "maximum input length", "context length is only",
        "max_model_len", "prompt too long",
    ))
    return {
        "reward": 0.0,
        "num_steps": 0,
        "status": "MAX_CONTEXT_LIMIT_REACHED" if overflow else "ERROR",
        "guard_blocked_steps": 0,
        "guard_reasons": [],
        "error": None if overflow else message,
    }
  traj = response.payload.traj
  reward = float(traj["trajectory_reward"])
  if not math.isfinite(reward):
    raise ValueError("Non-finite trajectory reward")
  details = traj["episode_summary"]
  return {
      "reward": reward,
      "num_steps": details["num_steps"],
      "status": traj["status"],
      "guard_blocked_steps": details["guard_blocked_steps"],
      "guard_reasons": details["guard_reasons"],
      "error": None,
  }


def summarize(rows: list[dict[str, object]], expected: int, attempts: int = 1) -> dict[str, object]:
  """Summarizes `expected` tasks with `attempts` attempts each.

  Failed and missing attempts count as unresolved. `pass_at_1` is the resolved
  fraction of all attempts; with several attempts, `pass_at_k` adds Pass@k for
  k = `attempts`.

  Args:
    rows: One compact result per finished attempt.
    expected: Number of tasks evaluated.
    attempts: Attempts per task.

  Returns:
    The summary metrics.
  """
  total = expected * attempts
  resolved = sum(row["reward"] > 0 for row in rows)
  summary = {
      "total_instances": expected,
      "attempts_per_instance": attempts,
      "total_attempts": total,
      "completed_attempts": len(rows),
      "resolved": resolved,
      "pass_at_1": resolved / total,
      "average_reward": sum(row["reward"] for row in rows) / total,
      "average_steps": sum(row["num_steps"] for row in rows) / total,
      "status_counts": dict(Counter(row["status"] for row in rows)),
      "guard_blocked_trajectories": sum(
          row["guard_blocked_steps"] > 0 for row in rows
      ),
      "guard_blocks": sum(row["guard_blocked_steps"] for row in rows),
      "guard_reasons": dict(Counter(
          reason for row in rows for reason in row["guard_reasons"]
      )),
      "errors": sum(bool(row["error"]) for row in rows),
  }
  if attempts > 1:
    # For k equal to the attempts per task, the unbiased Pass@k estimate is the
    # fraction of tasks with at least one resolved attempt. Tasks are keyed by
    # their dataset index, which is unique even when instance ids are not.
    resolved_tasks = {row["pair_index"] for row in rows if row["reward"] > 0}
    summary["pass_at_k"] = {str(attempts): len(resolved_tasks) / expected}
  return summary


class ResultWriter:
  """Writes one evaluation run to a local or fsspec (e.g. `gs://`) directory.

  Each finished attempt is written as `<run>/attempts/<request_id>.json` as soon
  as it completes, so partial results survive a failure. The run config goes to
  `<run>/config.json`, and the final rows and summary to `<run>.jsonl` and
  `<run>.summary.json`.
  """

  def __init__(self, output_dir: str, run_name: str):
    import fsspec  # pylint: disable=g-import-not-at-top
    from fsspec.implementations import local as fsspec_local  # pylint: disable=g-import-not-at-top

    self._fs, root = fsspec.core.url_to_fs(output_dir)
    self._is_local = isinstance(self._fs, fsspec_local.LocalFileSystem)
    self.prefix = f"{root.rstrip('/')}/{run_name}"
    self._lock = threading.Lock()

  def _write(self, path: str, text: str) -> None:
    with self._lock:
      if self._is_local:
        # Object stores have no directories; creating a bucket path can fail.
        self._fs.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
      with self._fs.open(path, "w", encoding="utf-8") as stream:
        stream.write(text)

  def write_json(self, name: str, value: object) -> None:
    """Writes `value` to `<run>/<name>`."""
    self._write(
        f"{self.prefix}/{name}",
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2),
    )

  def write_attempt(self, row: dict[str, object]) -> None:
    self.write_json(f"attempts/{row['request_id']}.json", row)

  def write_results(
      self, rows: list[dict[str, object]], summary: dict[str, object]
  ) -> None:
    self._write(f"{self.prefix}.jsonl", "".join(
        json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        for row in rows
    ))
    self._write(f"{self.prefix}.summary.json",
                json.dumps(summary, indent=2) + "\n")


def _attempts(
    jobs: Iterator[tuple[int, dict[str, object]]], attempts: int
) -> Iterator[tuple[int, dict[str, object], int]]:
  """Expands each `(index, entry)` task taken from `jobs` into its attempts."""
  for index, entry in jobs:
    for attempt in range(attempts):
      yield index, entry, attempt


async def collect_worker(handle, jobs, limit, args, on_result):
  """Bound in-flight work without applying the RPC deadline to episodes.

  Each task taken from the shared `jobs` iterator is dispatched as
  `--num_rollouts_per_instance` attempts to this worker. `on_result` runs in a
  thread so that remote result writes do not block the event loop.
  """
  requests = _attempts(jobs, args.num_rollouts_per_instance)
  pending = {}
  exhausted = False
  while pending or not exhausted:
    while not exhausted and len(pending) < limit:
      try:
        index, entry, attempt = next(requests)
      except StopIteration:
        exhausted = True
        break
      fields = request_fields(args, entry, index, attempt)
      request_id = fields["request_id"]
      ack = await handle.dispatch_task(request_id, "evaluate", fields)
      if ack != request_id:
        raise RuntimeError(f"Unexpected dispatch acknowledgement: {ack}")
      pending[request_id] = (index, entry, attempt, time.monotonic())
    if not pending:
      break
    reply = await handle.poll_responses(timeout_s=5)
    if reply is not None:
      if reply.request_id not in pending:
        raise RuntimeError(f"Unexpected evaluation result: {reply.request_id}")
      index, entry, attempt, started = pending.pop(reply.request_id)
      try:
        result = reply.unwrap()
      except Exception as exc:  # A task failure must remain in Pass@1's denominator.
        result = {
            "reward": 0.0, "num_steps": 0, "status": "ERROR",
            "guard_blocked_steps": 0, "guard_reasons": [], "error": str(exc),
        }
      result.update(
          request_id=reply.request_id,
          pair_index=index,
          attempt=attempt,
          instance_id=entry.get("instance_id", index),
          docker_image=entry["docker_image"],
          wall_seconds=time.monotonic() - started,
      )
      await asyncio.to_thread(on_result, result)
    if any(time.monotonic() - started > args.timeout + 1800
           for _, _, _, started in pending.values()):
      raise TimeoutError("A rollout exceeded its episode and cleanup deadline")


async def run_controller(args: argparse.Namespace) -> dict[str, object]:
  from tunix.dist_rl.worker import remote_execution

  entries = load_entries(args)
  attempts = args.num_rollouts_per_instance
  writer = ResultWriter(
      args.output_dir, f"eval_deepswe_{time.strftime('%Y%m%d_%H%M%S')}"
  )
  writer.write_json("config.json", {
      **vars(args),
      "instance_ids": [entry.get("instance_id", index)
                       for index, entry in enumerate(entries)],
  })
  handles = [remote_execution.ActorHandle.from_address(
      address if address.startswith("grpc://") else f"grpc://{address}",
      rpc_timeout_s=60,
  ) for address in args.worker_addresses]
  rows = []
  rows_lock = threading.Lock()
  failure = None
  try:
    async def wait_ready(handle):
      deadline = time.monotonic() + args.startup_timeout
      while True:
        try:
          actual = await handle.asubmit("evaluation_info")
          break
        except Exception:
          if time.monotonic() >= deadline:
            raise TimeoutError("Rollout worker did not become ready") from None
          await asyncio.sleep(5)
      if actual != model_profile(args):
        raise ValueError(f"Worker model settings differ: {actual}")

    await asyncio.gather(*(wait_ready(handle) for handle in handles))
    jobs = iter(enumerate(entries))

    def record(row: dict[str, object]) -> None:
      with rows_lock:
        rows.append(row)
        done = len(rows)
      try:
        writer.write_attempt(row)
      except OSError as exc:
        logging.warning(
            "Failed to write attempt %s: %s", row["request_id"], exc
        )
      logging.info(
          "[%d/%d] %s reward=%.1f steps=%d status=%s",
          done, len(entries) * attempts, row["request_id"], row["reward"],
          row["num_steps"], row["status"],
      )

    workers = len(handles)
    await asyncio.gather(*(
        collect_worker(
            handle, jobs,
            args.max_concurrent // workers
            + (index < args.max_concurrent % workers),
            args, record,
        )
        for index, handle in enumerate(handles)
    ))
  except BaseException as exc:
    failure = f"{type(exc).__name__}: {exc}"
    raise
  finally:
    await asyncio.gather(*(handle.close() for handle in handles),
                         return_exceptions=True)
    summary = summarize(rows, len(entries), attempts)
    summary["fatal_error"] = failure
    writer.write_results(rows, summary)

  logging.info("Evaluation results: %s.jsonl", writer.prefix)
  logging.info("Pass@1: %.4f (%d/%d)", summary["pass_at_1"],
               summary["resolved"], summary["total_attempts"])
  if attempts > 1:
    logging.info("Pass@%d: %.4f", attempts, summary["pass_at_k"][str(attempts)])
  if (summary["errors"]
      or summary["completed_attempts"] != summary["total_attempts"]):
    raise RuntimeError(f"Evaluation contains failures; inspect {writer.prefix}")
  return summary


def register_guarded_env(deepswe_module) -> str:
  """Register the optional action-guard variant of the DeepSWE environment."""
  from examples.deepswe.action_guard import ActionGuard, GuardConfig
  from tunix.dist_rl.common import registry
  from tunix.rl.agentic.environments.base_environment import EnvStepResult

  @registry.register_env("deepswe_eval_guarded")
  class GuardedEvalEnv(deepswe_module.DeepSWEEnv):
    """Apply the agentic eval's optional action guard to the registry env."""

    def __init__(self, *env_args, **env_kwargs):
      self.guard = ActionGuard(GuardConfig())
      super().__init__(*env_args, **env_kwargs)

    def _initial_observation(self):
      self.guard.reset()
      return super()._initial_observation()

    def _step_impl(self, action):
      if isinstance(action, str):
        function, _ = self.guard._parse_action(  # pylint: disable=protected-access
            action
        )
        if not function:
          return EnvStepResult(
              observation=(
                  "[ACTION GUARD] Your previous response did not include a"
                  " valid function call. You must output exactly one tool"
                  " call in the required XML format."
              ),
              reward=0.0,
              done=False,
              info={"guard_blocked": True,
                    "guard_reason": "missing_function_call"},
          )
        verdict = self.guard.evaluate(action)
        if verdict.blocked:
          return EnvStepResult(
              observation=verdict.message, reward=0.0, done=False,
              info={"guard_blocked": True, "guard_reason": verdict.reason},
          )
      result = super()._step_impl(action)
      if isinstance(action, str):
        self.guard.record_outcome(action, str(result.observation))
      return result

  return "deepswe_eval_guarded"


async def run_worker(args: argparse.Namespace):
  # Keep vLLM imports before rollout adapters, as in the training rollout node.
  os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
  os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
  # Evaluation must load checkpoint weights even after a training shell export.
  os.environ.pop("JAX_RANDOM_WEIGHTS", None)
  from tunix.generate import vllm_sampler  # pylint: disable=g-import-not-at-top,unused-import
  import jax
  from jax.experimental import mesh_utils
  from jax.sharding import Mesh
  from transformers import AutoTokenizer
  from tunix.dist_rl.common import datatypes
  from tunix.dist_rl.examples.deepswe_dist import deepswe
  from tunix.dist_rl.rollout import inprocess_vllm_sampler
  from tunix.dist_rl.worker import remote_execution, rollout_worker
  from tunix.generate import tokenizer_adapter
  from tunix.rl.agentic.parser.chat_template_parser import parser

  if args.mesh_fsdp * args.mesh_tp != jax.device_count():
    raise ValueError("mesh_fsdp * mesh_tp must equal visible TPU chips")
  model = args.model_dir or args.model_id
  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or model)
  if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
  eos = tokenizer.encode("<|im_end|>", add_special_tokens=False)
  if len(eos) != 1:
    raise ValueError("<|im_end|> must encode to one Qwen token")
  mesh = Mesh(mesh_utils.create_device_mesh(
      (args.mesh_fsdp, args.mesh_tp), jax.devices()), ("fsdp", "tp"))
  env_name = deepswe.DEEPSWE_ENV_NAME
  if args.enable_guard:
    env_name = register_guarded_env(deepswe)
  # Greedy by default; `--temperature`, `--top_p`, and `--top_k` sample instead.
  config = rollout_worker.RolloutConfig(
      weight_sync_mode="none",
      max_tokens_to_generate=args.max_response_length,
      max_response_length=response_budget(args),
      kv_cache_size=args.max_model_len,
      episode_timeout=float(args.timeout),
      temperature=args.temperature,
      top_p=args.top_p,
      top_k=args.top_k if args.top_k > 0 else None,
      return_logprobs=False,
      eos_tokens=eos,
      data_parallel_size=args.mesh_fsdp,
      tensor_parallel_size=args.mesh_tp,
      rollout_vllm_model_version=model,
      rollout_vllm_server_mode=True,
      rollout_vllm_init_with_random_weights=False,
      rollout_vllm_hbm_utilization=args.vllm_hbm_utilization,
      rollout_vllm_tpu_backend_type="jax",
      rollout_vllm_max_num_seqs=args.vllm_max_num_seqs,
      rollout_vllm_max_num_batched_tokens=args.vllm_max_batched_tokens,
      rollout_vllm_kwargs={
          "tokenizer": args.tokenizer_path or model,
          "enable_prefix_caching": True,
          "dtype": "bfloat16",
      },
  )
  sampler = inprocess_vllm_sampler.InprocessVllmSampler(
      server_id="deepswe-eval",
      tokenizer=tokenizer,
      config=config,
      mesh=mesh,
      max_concurrency=args.max_concurrent,
  )

  stop = asyncio.Event()

  class EvaluationWorker(rollout_worker.RolloutWorker):

    def evaluation_info(self):
      return model_profile(args)

    async def evaluate(self, fields):
      response = await self.generate(datatypes.RolloutRequest(**fields))
      return compact_result(response)

    def shutdown(self):
      stop.set()
      return True

  worker = EvaluationWorker(
      worker_id="deepswe-eval",
      config=config,
      sampler=sampler,
      env_name=env_name,
      agent_name=deepswe.DEEPSWE_AGENT_NAME,
      tokenizer=tokenizer_adapter.TokenizerAdapter(tokenizer),
      chat_parser=parser.QwenChatTemplateParser(
          tokenizer, enable_thinking=not args.disable_thinking
      ),
      max_concurrency=args.max_concurrent,
  )
  server = remote_execution.GrpcRemoteExecutionServer(worker)
  try:
    worker.initialize()
    await sampler.start()
    await server.start_serving_async(args.port)
    logging.info("DeepSWE eval worker ready on port %d", args.port)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
      loop.add_signal_handler(sig, stop.set)
    await stop.wait()
  finally:
    worker.stop()
    await server.stop_serving()
    await sampler.stop()


def run_local(argv: list[str], args: argparse.Namespace) -> int:
  if args.worker_addresses != [f"127.0.0.1:{args.port}"]:
    raise ValueError(
        "Local mode starts one worker; use separate roles for remote workers"
    )
  worker = controller = None
  try:
    worker = subprocess.Popen(
        [sys.executable, __file__, *argv, "--role", "worker"],
        start_new_session=True,
    )
    controller = subprocess.Popen(
        [sys.executable, __file__, *argv, "--role", "controller"],
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        start_new_session=True,
    )
    while controller.poll() is None:
      if worker.poll() is not None:
        raise RuntimeError(f"Rollout worker exited: {worker.returncode}")
      time.sleep(1)
    return controller.returncode
  finally:
    for process in (controller, worker):
      if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
          process.wait(timeout=30)
        except subprocess.TimeoutExpired:
          os.killpg(process.pid, signal.SIGKILL)
          process.wait()


def main(argv: list[str] | None = None) -> int:
  argv = list(sys.argv[1:] if argv is None else argv)
  args = parse_args(argv)
  logging.basicConfig(level=logging.INFO,
                      format="%(asctime)s %(levelname)s %(message)s")
  sys.path.insert(0, str(ROOT))
  if args.role == "local":
    return run_local(argv, args)
  if args.role == "controller":
    os.environ["JAX_PLATFORMS"] = "cpu"
    asyncio.run(run_controller(args))
  else:
    asyncio.run(run_worker(args))
  return 0


if __name__ == "__main__":
  sys.exit(main())
