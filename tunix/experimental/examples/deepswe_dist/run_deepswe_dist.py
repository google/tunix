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

"""CPU control-plane for the experimental distributed DeepSWE GRPO demo."""

from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import sys
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # pylint: disable=g-import-not-at-top
from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

# pylint: disable=g-import-not-at-top
from tunix.experimental.common import datatypes
from tunix.experimental.distributed.runtime import context as runtime_context
from tunix.experimental.examples.deepswe_dist import deepswe
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import batch_assembly
from tunix.experimental.orchestrator import orchestrator
from tunix.experimental.orchestrator import rl_program
from tunix.experimental.weight_sync import weight_sync
from tunix.experimental.worker import remote_execution
from examples.deepswe import swe_env
from tunix.sft import metrics_logger as metrics_logger_lib
# pylint: enable=g-import-not-at-top


ProcessContext = runtime_context.ProcessContext


def _parse_args(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Orchestrator V2 DeepSWE distributed GRPO demo."
  )
  parser.add_argument("--batch_size", type=int, default=1)
  parser.add_argument("--num_generations", type=int, default=2)
  parser.add_argument("--max_steps", type=int, default=1)
  parser.add_argument("--max_prompt_length", type=int, default=1024)
  parser.add_argument("--max_response_length", type=int, default=1024)
  parser.add_argument("--train_micro_batch_size", type=int, default=1)
  parser.add_argument(
      "--max_seq_token_per_tpu",
      type=int,
      default=None,
      help=(
          "Maximum sequence tokens per TPU for sequence packing. When"
          " configured, SequencePackedBatchAssembler is used instead of"
          " PaddedBatchAssembler."
      ),
  )
  parser.add_argument(
      "--max_segments_per_packed_row",
      type=int,
      default=None,
      help="Maximum segments per packed row when sequence packing is enabled.",
  )
  # TODO(tunix-dev): Clean up worker specific configuration to orchestrator.
  parser.add_argument(
      "--trainer_fsdp",
      type=int,
      default=None,
      help=(
          "Trainer FSDP mesh dimension size for sequence packing pack_size"
          " computation."
      ),
  )
  parser.add_argument(
      "--trainer_dp",
      type=int,
      default=None,
      help=(
          "Trainer DP mesh dimension size for sequence packing pack_size"
          " computation."
      ),
  )
  parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
  parser.add_argument("--tokenizer_path", type=str, default="")
  parser.add_argument("--temperature", type=float, default=1.0)
  parser.add_argument("--top_p", type=float, default=1.0)
  parser.add_argument("--top_k", type=int, default=-1)
  parser.add_argument("--beta", type=float, default=0.0)
  parser.add_argument("--epsilon", type=float, default=0.2)
  parser.add_argument(
      "--use_rollout_logps",
      action=argparse.BooleanOptionalAction,
      default=True,
      help=(
          "Use rollout sampler log-probs as old_per_token_logps (off-policy /"
          " sampler importance ratio). Default True matches the"
          " non-experimental GRPOConfig; pass --no-use_rollout_logps for"
          " on-policy ratio=1."
      ),
  )
  parser.add_argument(
      "--offpolicy",
      "--max_staleness",
      dest="max_staleness",
      type=int,
      default=0,
  )
  parser.add_argument(
      "--weight_sync_mode",
      type=weight_sync.WeightSyncMode,
      default=weight_sync.WeightSyncMode(os.getenv("WEIGHT_SYNC_MODE", "none")),
      choices=list(weight_sync.WeightSyncMode),
  )
  parser.add_argument("--dataset_path", type=str, default="")
  parser.add_argument(
      "--dataset_name", type=str, default=deepswe.DEFAULT_DATASET_NAME
  )
  parser.add_argument("--dataset_split", type=str, default="train")
  parser.add_argument(
      "--dataset_cache_dir",
      type=str,
      default=os.getenv("DATASET_CACHE_DIR", ""),
  )
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
      "--shuffle", action=argparse.BooleanOptionalAction, default=True
  )
  parser.add_argument("--max_turns", type=int, default=50)
  parser.add_argument("--step_timeout_secs", type=int, default=30 * 60)
  parser.add_argument("--reward_timeout_secs", type=int, default=30 * 60)
  parser.add_argument("--env_backend", type=str, default="kubernetes")
  parser.add_argument(
      "--scaffold", choices=("r2egym", "sweagent"), default="r2egym"
  )
  parser.add_argument("--use_agent_sandbox", action="store_true")
  parser.add_argument(
      "--max_warmpool_replicas",
      type=int,
      default=None,
      help=(
          "Maximum replicas per SandboxWarmPool (defaults to num_generations)."
      ),
  )
  parser.add_argument(
      "--max_concurrency",
      type=int,
      default=128,
      help="Maximum concurrency for SandboxFleet.",
  )
  parser.add_argument("--env_verbose", action="store_true")
  parser.add_argument(
      "--flush_every_n_steps",
      type=int,
      default=1,
      help="Frequency in steps to flush metrics logger.",
  )
  parser.add_argument(
      "--log_dir",
      type=str,
      default=os.getenv("LOG_DIR", "/tmp/trellis_deepswe"),
      help="Directory for local event logging (TensorBoard/CLU).",
  )
  parser.add_argument(
      "--wandb_project",
      type=str,
      default=os.getenv("WANDB_PROJECT", "trellis-deepswe"),
      help="W&B project name.",
  )
  parser.add_argument(
      "--wandb_run_name",
      type=str,
      default=os.getenv("WANDB_RUN_NAME", ""),
      help="W&B run name. Defaults to timestamp-based name if unset.",
  )
  parser.add_argument("--rpc_timeout_s", type=float, default=1800.0)
  parser.add_argument("--init_timeout_s", type=float, default=None)
  parser.add_argument("--stop_workers_on_exit", action="store_true")
  parser.add_argument("--debug", action="store_true")
  return parser.parse_args(argv)


def _build_algo(args: argparse.Namespace) -> algorithm_adapter.GRPOAdapter:
  return algorithm_adapter.GRPOAdapter(
      group_size=args.num_generations,
      mini_batch_size=args.batch_size,
      train_micro_batch_size=args.train_micro_batch_size,
      max_turns=args.max_turns,
      max_packed_len=(
          args.max_seq_token_per_tpu
          if args.max_seq_token_per_tpu is not None
          else args.max_prompt_length + args.max_response_length
      ),
      clip_epsilon=args.epsilon,
      beta_kl=args.beta,
      temperature=args.temperature,
      use_rollout_logps=args.use_rollout_logps,
  )


def _configure_trainer_loss(
    trainer_handle: remote_execution.ActorHandle,
    *,
    algo: algorithm_adapter.GRPOAdapter,
    pad_id: int,
    eos_id: int,
) -> None:
  logging.info(
      "Configuring trainer-side GRPO loss via TrainerWorker RPC (beta=%s, "
      "epsilon=%s).",
      algo.beta_kl,
      algo.clip_epsilon,
  )
  trainer_handle.submit("with_loss_fn", algo.loss_fn(), has_aux=True)
  trainer_handle.submit(
      "with_gen_model_input_fn",
      algo.build_gen_model_input_fn(pad_id=pad_id, eos_id=eos_id),
  )


def main(argv: list[str], context: ProcessContext | None = None) -> None:
  assert (
      context and context.ipc and context.ipc.discovery
  ), "Require discovery API, but process context doesn't support."

  args = _parse_args(argv)
  logging.basicConfig(
      level=logging.DEBUG if args.debug else logging.INFO,
      format="%(asctime)s - [DeepSWEOrchestrator] %(message)s",
      force=True,
  )

  if args.num_generations <= 1:
    raise ValueError("num_generations must be greater than 1 for GRPO.")
  if args.batch_size <= 0:
    raise ValueError("batch_size must be positive.")
  if args.train_micro_batch_size <= 0:
    raise ValueError("train_micro_batch_size must be positive.")
  if args.max_staleness < 0:
    raise ValueError("offpolicy/max_staleness must be non-negative.")

  logging.info("=== Starting Distributed DeepSWE GRPO Orchestrator ===")
  logging.info(
      "Configuration: model_id=%s, batch_size=%d prompt group(s), "
      "num_generations=%d, max_steps=%d, max_turns=%d, train_micro=%d, "
      "beta=%.4f, env_backend=%s, use_agent_sandbox=%s, weight_sync_mode=%s.",
      args.model_id,
      args.batch_size,
      args.num_generations,
      args.max_steps,
      args.max_turns,
      args.train_micro_batch_size,
      args.beta,
      args.env_backend,
      args.use_agent_sandbox,
      args.weight_sync_mode,
  )
  logging.info("Control-plane JAX backend: %s", jax.default_backend())

  tokenizer_path = (
      args.tokenizer_path or os.getenv("MODEL_DIR") or args.model_id
  )
  tokenizer = AutoTokenizer.from_pretrained(
      tokenizer_path, trust_remote_code=True
  )
  if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
  pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
  eos_id = (
      tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
  )
  logging.info(
      "Loaded tokenizer from %s (vocab_size=%d, pad_id=%d, eos_id=%d).",
      tokenizer_path,
      len(tokenizer),
      pad_id,
      eos_id,
  )

  dataset = deepswe.load_deepswe_dataset(
      dataset_name=args.dataset_name,
      dataset_split=args.dataset_split,
      dataset_path=args.dataset_path,
      cache_dir=args.dataset_cache_dir or None,
      shuffle=args.shuffle,
      seed=args.seed,
  )
  logging.info(
      "Loaded DeepSWE dataset: source=%s split=%s size=%d.",
      args.dataset_path or args.dataset_name,
      args.dataset_split,
      len(dataset),
  )

  cluster = orchestrator.ClusterOrchestrator(
      weight_sync_mode=args.weight_sync_mode,
  )
  context.ipc.discovery.on_register(
      functools.partial(
          cluster.register_worker_from_hostname,
          rpc_timeout_s=args.rpc_timeout_s,
      )
  )

  cluster.wait_for_workers(
      min_workers={
          datatypes.Role.ACTOR: 1,
          datatypes.Role.ROLLOUT: 1,
          datatypes.Role.REFERENCE: 1 if args.beta != 0.0 else 0,
      },
      timeout=args.init_timeout_s,
      poll_interval_s=1.0,
  )
  logging.info("Registered workers: %s", cluster.worker_infos())

  algo = _build_algo(args)
  trainer_handles = cluster.worker_handles(datatypes.Role.ACTOR)
  if len(trainer_handles) != 1:
    raise ValueError(f"Expected 1 trainer worker, got {len(trainer_handles)}.")
  _configure_trainer_loss(
      trainer_handles[0],
      algo=algo,
      pad_id=pad_id,
      eos_id=eos_id,
  )

  metrics_logging_options = metrics_logger_lib.MetricsLoggerOptions(
      log_dir=args.log_dir,
      project_name=args.wandb_project,
      run_name=args.wandb_run_name,
      flush_every_n_steps=args.flush_every_n_steps,
      backend_kwargs={"wandb": {"config": vars(args)}},
  )

  fleet = None
  if args.use_agent_sandbox:
    fleet = swe_env._init_global_fleet(  # pylint: disable=protected-access
        tasks=dataset,
        max_concurrency=args.max_concurrency,
        num_generations=args.num_generations,
        batch_size=args.batch_size,
        max_warmpool_replicas=args.max_warmpool_replicas,
    )

  prompt_stream = deepswe.iter_prompt_items(
      dataset=dataset,
      max_steps=args.max_steps,
      batch_size=args.batch_size,
      max_turns=args.max_turns,
      max_response_length=args.max_response_length,
      temperature=args.temperature,
      top_p=args.top_p,
      top_k=None if args.top_k < 0 else args.top_k,
      step_timeout_secs=args.step_timeout_secs,
      reward_timeout_secs=args.reward_timeout_secs,
      env_backend=args.env_backend,
      use_agent_sandbox=args.use_agent_sandbox,
      scaffold=args.scaffold,
      env_verbose=args.env_verbose,
  )
  if args.use_agent_sandbox:
    prompt_stream = swe_env.PrewarmDatasetIterator(
        prompt_stream,
        fleet=fleet,
        num_generations=args.num_generations,
        batch_size=args.batch_size,
        max_warmpool_replicas=args.max_warmpool_replicas,
    )

  def _save_and_log_trajectories(
      step: int,
      trajectories: Any,
      source: str = "Rollouts",
  ) -> None:
    if not trajectories:
      logging.info("No trajectories recorded for step %d (%s).", step, source)
      return

    traj_records = []
    for idx, item in enumerate(trajectories):
      prompt_id = getattr(item, "prompt_id", f"item_{idx}")
      metadata = dict(getattr(item, "metadata", {}) or {})
      traj = getattr(item, "traj", None)
      reward = float(
          getattr(traj, "reward", metadata.get("reward", 0.0)) or 0.0
      )
      status = str(getattr(traj, "status", metadata.get("status", "UNKNOWN")))
      steps_history = metadata.get("steps_history") or []
      if not steps_history and traj and hasattr(traj, "steps"):
        for s in traj.steps:
          steps_history.append({
              "model_response": str(
                  getattr(s, "message", getattr(s, "model_response", ""))
              ),
              "thought": str(getattr(s, "thought", "")),
              "action": str(getattr(s, "action", "")),
              "observation": str(getattr(s, "observation", "")),
              "reward": float(getattr(s, "reward", 0.0) or 0.0),
          })

      record = {
          "step": step,
          "index": idx,
          "prompt_id": prompt_id,
          "reward": reward,
          "status": status,
          "num_turns": len(steps_history),
          "steps": steps_history,
          "metadata": {
              k: str(v)
              for k, v in metadata.items()
              if k not in ("steps_history", "prompt_tokens", "tokens")
          },
      }
      traj_records.append(record)

      error_msg = metadata.get("error") or getattr(item, "error", None)
      if error_msg:
        logging.error(
            "[Trajectory Dump - %s] Item %d Error: %s", source, idx, error_msg
        )
      logging.info(
          "[Trajectory Dump - %s] Step %d | Item %d | PromptID: %s |"
          " Reward: %.2f | Status: %s | Turns: %d",
          source,
          step,
          idx,
          prompt_id,
          reward,
          status,
          len(steps_history),
      )
      if idx < 2 or args.env_verbose:
        for turn_idx, turn in enumerate(steps_history):
          thought = str(turn.get("thought") or "")
          model_resp = str(
              turn.get("model_response") or turn.get("message") or ""
          )
          action = str(turn.get("action") or "")
          obs = str(turn.get("observation") or "")
          if thought:
            logging.info(
                "[Turn %d Thought]:\n%s",
                turn_idx,
                thought[:3000] + ("..." if len(thought) > 3000 else ""),
            )
          if model_resp:
            logging.info(
                "[Turn %d Model Response]:\n%s",
                turn_idx,
                model_resp[:3000] + ("..." if len(model_resp) > 3000 else ""),
            )
          if action:
            logging.info(
                "[Turn %d Action]:\n%s",
                turn_idx,
                action[:1000] + ("..." if len(action) > 1000 else ""),
            )
          if obs:
            logging.info(
                "[Turn %d Environment Observation]:\n%s",
                turn_idx,
                obs[:3000] + ("..." if len(obs) > 3000 else ""),
            )

    dump_dirs = [
        "/tmp/deepswe_trajectories",
        "/app/checkpoints/trajectories",
    ]
    for out_dir in dump_dirs:
      try:
        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, f"trajectories_step_{step}.json")
        with open(file_path, "w", encoding="utf-8") as f:
          json.dump(traj_records, f, indent=2)
        logging.info(
            "Saved %d trajectories to %s", len(traj_records), file_path
        )
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning(
            "Note: could not save trajectories to %s: %s", out_dir, e
        )

  def _on_rollouts_ready(step: int, trajectories: Any) -> None:
    logging.info(
        ">>> Rollouts ready for step %d (%d items)", step, len(trajectories)
    )
    _save_and_log_trajectories(step, trajectories, source="RolloutsReady")

  def _dump_and_log_trajectories(
      step: int,
      step_result: Any,
      trajectories: Any = None,
  ) -> None:
    logging.info(
        "<<< DeepSWE step %d finished | train_result=%s", step, step_result
    )
    _save_and_log_trajectories(step, trajectories, source="StepEnd")

  program = rl_program.StandardRLProgram(
      algo=algo,
      dataset=prompt_stream,
      max_steps=args.max_steps,
      generation_args=datatypes.GenerationArgs(
          max_response_length=args.max_response_length,
          temperature=args.temperature,
          top_p=args.top_p,
          top_k=None if args.top_k < 0 else args.top_k,
          return_logprobs=True,
      ),
      reward_fns=[],
      batch_config=batch_assembly.BatchConfig(
          pad_id=pad_id,
          max_prompt_length=args.max_prompt_length,
          max_response_length=args.max_response_length,
          max_seq_token_per_tpu=args.max_seq_token_per_tpu,
          max_segments_per_packed_row=args.max_segments_per_packed_row,
          trainer_fsdp=args.trainer_fsdp,
          trainer_dp=args.trainer_dp,
      ),
      metrics_logging_options=metrics_logging_options,
      max_staleness=args.max_staleness,
      sync_weights=(args.weight_sync_mode != weight_sync.WeightSyncMode.NONE),
      on_step_begin=lambda step: logging.info(
          ">>> DeepSWE step %d starting | policy_version=%d",
          step,
          step,
      ),
      on_step_end=_dump_and_log_trajectories,
      on_rollouts_ready=_on_rollouts_ready,
  )

  try:
    logging.info("Bringing up remote workers through ClusterOrchestrator...")
    cluster.bring_up_workers(dummy_data=None)
    logging.info("Starting DeepSWE StandardRLProgram execution...")
    cluster.run(
        program=program,
        num_steps=args.max_steps,
        bring_up=False,
    )
  finally:
    program.close()
    if fleet is not None:
      logging.info("Tearing down SandboxFleet on orchestrator...")
      try:
        fleet.teardown()
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning("Fleet teardown note: %s", e)
    if args.stop_workers_on_exit:
      logging.info("Shutting down cluster workers...")
      cluster.shutdown()
    else:
      cluster.monitor.close()

  result = program.last_step_result
  if result is None:
    logging.info("=== DeepSWE GRPO pipeline finished without step result ===")
    return

  logging.info(
      "=== DeepSWE GRPO pipeline finished ===\n"
      "  Final step: %d\n"
      "  Final policy version: %d\n"
      "  Rollouts in final step: %d\n"
      "  Microbatches in final step: %d\n"
      "  Final reward: mean=%.4f std=%.4f",
      result.step,
      result.policy_version,
      result.num_rollouts,
      result.num_microbatches,
      result.reward_mean,
      result.reward_std,
  )


if __name__ == "__main__":
  main(sys.argv[1:])
