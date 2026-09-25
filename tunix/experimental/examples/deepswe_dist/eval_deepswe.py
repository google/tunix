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
Run locally on one TPU host, or run worker and controller roles separately.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import logging
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[4]


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
      "--max_context_limit", type=int, default=0,
      help="Cumulative response budget; zero uses model length minus 256.",
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
               "startup_timeout", "max_steps", "vllm_max_num_seqs",
               "vllm_max_batched_tokens"):
    if getattr(args, name) <= 0:
      parser.error(f"--{name} must be positive")
  if (args.tasks_limit < 0 or args.max_context_limit < 0
      or not 0 < args.vllm_hbm_utilization < 1):
    parser.error(
        "Task/context limits must be nonnegative and HBM utilization in (0, 1)"
    )
  if args.max_response_length > args.max_model_len - 256:
    parser.error("max_response_length must leave 256 context tokens")
  if len(set(args.worker_addresses)) != len(args.worker_addresses):
    parser.error("worker_addresses must be unique")
  if args.max_concurrent < len(args.worker_addresses):
    parser.error("max_concurrent must cover every rollout worker")
  return args


def model_profile(args: argparse.Namespace) -> dict[str, object]:
  """Reject a worker serving a different model or rollout configuration."""
  names = ("model_id", "model_dir", "tokenizer_path", "mesh_fsdp", "mesh_tp",
           "max_model_len", "max_context_limit", "max_response_length", "max_steps",
           "enable_guard", "disable_thinking")
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


def request_fields(args: argparse.Namespace, entry: dict, index: int) -> dict:
  """Use the same one-trajectory limits and greedy sampling as agentic eval."""
  return {
      "request_id": f"eval_{index}",
      "prompt_id": f"eval_{index}",
      "group_index": 0,
      "prompt": str(entry.get("problem_statement", "")),
      "max_turns": args.max_steps,
      "max_response_length": (
          args.max_context_limit or args.max_model_len - 256
      ),
      "generation_kwargs": {
          "max_generation_steps": args.max_response_length,
          "temperature": 0.0,
          "return_logprobs": False,
      },
      "metadata": {
          "episode_timeout": args.timeout,
          "exact_token_continuity": False,
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


def summarize(rows: list[dict], expected: int) -> dict:
  """Pass@1 keeps failed or missing tasks in the denominator."""
  resolved = sum(row["reward"] > 0 for row in rows)
  return {
      "total_instances": expected,
      "completed_instances": len(rows),
      "resolved": resolved,
      "pass_at_1": resolved / expected,
      "average_reward": sum(row["reward"] for row in rows) / expected,
      "average_steps": sum(row["num_steps"] for row in rows) / expected,
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


async def collect_worker(handle, jobs, limit, args, on_result):
  """Bound in-flight work without applying the RPC deadline to episodes."""
  pending = {}
  exhausted = False
  while pending or not exhausted:
    while not exhausted and len(pending) < limit:
      try:
        index, entry = next(jobs)
      except StopIteration:
        exhausted = True
        break
      fields = request_fields(args, entry, index)
      request_id = fields["request_id"]
      ack = await handle.dispatch_task(request_id, "evaluate", fields)
      if ack != request_id:
        raise RuntimeError(f"Unexpected dispatch acknowledgement: {ack}")
      pending[request_id] = (index, entry, time.monotonic())
    if not pending:
      break
    reply = await handle.poll_responses(timeout_s=5)
    if reply is not None:
      if reply.request_id not in pending:
        raise RuntimeError(f"Unexpected evaluation result: {reply.request_id}")
      index, entry, started = pending.pop(reply.request_id)
      try:
        result = reply.unwrap()
      except Exception as exc:  # A task failure must remain in Pass@1's denominator.
        result = {
            "reward": 0.0, "num_steps": 0, "status": "ERROR",
            "guard_blocked_steps": 0, "guard_reasons": [], "error": str(exc),
        }
      result.update(
          pair_index=index,
          instance_id=entry.get("instance_id", index),
          docker_image=entry["docker_image"],
          wall_seconds=time.monotonic() - started,
      )
      on_result(result)
    if any(time.monotonic() - started > args.timeout + 1800
           for _, _, started in pending.values()):
      raise TimeoutError("A rollout exceeded its episode and cleanup deadline")


async def run_controller(args: argparse.Namespace) -> dict:
  from tunix.experimental.worker import remote_execution

  entries = load_entries(args)
  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  output = output_dir / f"eval_deepswe_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
  handles = [remote_execution.ActorHandle.from_address(
      address if address.startswith("grpc://") else f"grpc://{address}",
      rpc_timeout_s=60,
  ) for address in args.worker_addresses]
  rows = []
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
    with output.open("w", encoding="utf-8") as stream:
      def record(row):
        rows.append(row)
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        logging.info(
            "[%d/%d] %s reward=%.1f steps=%d status=%s",
            len(rows), len(entries), row["instance_id"], row["reward"],
            row["num_steps"], row["status"],
        )

      workers = len(handles)
      await asyncio.gather(*(
          collect_worker(
              handle, jobs,
              args.max_concurrent // workers + (index < args.max_concurrent % workers),
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
    summary = summarize(rows, len(entries))
    summary["fatal_error"] = failure
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

  logging.info("Evaluation results: %s", output)
  logging.info("Pass@1: %.4f (%d/%d)", summary["pass_at_1"],
               summary["resolved"], summary["total_instances"])
  if summary["errors"] or summary["completed_instances"] != len(entries):
    raise RuntimeError(f"Evaluation contains failures; inspect {output}")
  return summary


def register_guarded_env(deepswe_module) -> str:
  """Register the optional action-guard variant of the DeepSWE environment."""
  from examples.deepswe.action_guard import ActionGuard, GuardConfig
  from tunix.experimental.rl.agentic import registry
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
  from tunix.generate import vllm_sampler
  import jax
  from jax.experimental import mesh_utils
  from jax.sharding import Mesh
  from transformers import AutoTokenizer
  from tunix.experimental.common import datatypes
  from tunix.experimental.examples.deepswe_dist import deepswe
  from tunix.experimental.rollout import inprocess_vllm_sampler_adapter
  from tunix.experimental.worker import remote_execution, rollout_worker
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
  config = vllm_sampler.VllmConfig(
      mesh=mesh,
      server_mode=True,
      init_with_random_weights=False,
      hbm_utilization=args.vllm_hbm_utilization,
      data_parallel_size=args.mesh_fsdp,
      tensor_parallel_size=args.mesh_tp,
      eos_tokens=eos,
      engine_kwargs={
          "model": model,
          "tokenizer": args.tokenizer_path or model,
          "max_model_len": args.max_model_len,
          "max_num_seqs": args.vllm_max_num_seqs,
          "max_num_batched_tokens": args.vllm_max_batched_tokens,
          "enable_prefix_caching": True,
          "dtype": "bfloat16",
      },
  )
  sampler = inprocess_vllm_sampler_adapter.InprocessVllmSamplerAdapter(
      server_id="deepswe-eval", tokenizer=tokenizer, config=config,
      weight_sync_mode="none", max_concurrency=args.max_concurrent,
  )
  env_name = deepswe.DEEPSWE_ENV_NAME
  if args.enable_guard:
    env_name = register_guarded_env(deepswe)

  stop = asyncio.Event()

  class EvaluationWorker(rollout_worker.RolloutWorker):

    def evaluation_info(self):
      return model_profile(args)

    async def evaluate(self, fields):
      response = await self.generate(datatypes.RolloutRequest(**fields))
      # generate also publishes a copy for streaming RL consumers.
      await self.pop_next_completed()
      return compact_result(response)

    def shutdown(self):
      stop.set()
      return True

  worker = EvaluationWorker(
      worker_id="deepswe-eval",
      config=rollout_worker.RolloutConfig(
          sampler_type="inprocess_vllm", weight_sync_mode="none",
          env_name=env_name, agent_name=deepswe.DEEPSWE_AGENT_NAME,
          eos_tokens=eos,
      ),
      sampler=sampler,
      tokenizer=tokenizer_adapter.TokenizerAdapter(tokenizer),
      chat_parser=parser.QwenChatTemplateParser(
          tokenizer, enable_thinking=not args.disable_thinking
      ),
      max_concurrency=args.max_concurrent,
  )
  server = remote_execution.GrpcRemoteExecutionServer(worker)
  try:
    await sampler.start()
    worker.initialize()
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
