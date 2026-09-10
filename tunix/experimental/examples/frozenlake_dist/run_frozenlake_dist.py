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

"""CPU control-plane for the distributed FrozenLake GRPO recipe."""

from __future__ import annotations

import argparse
import functools
import logging
import os
import sys

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
from tunix.experimental.examples.frozenlake_dist import frozenlake
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import batch_assembly
from tunix.experimental.orchestrator import orchestrator
from tunix.experimental.orchestrator import rl_program
from tunix.experimental.weight_sync import weight_sync
from tunix.experimental.worker import remote_execution
from tunix.sft import metrics_logger as metrics_logger_lib

# pylint: enable=g-import-not-at-top


ProcessContext = runtime_context.ProcessContext


def _parse_args(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Orchestrator V2 Qwen3 FrozenLake distributed GRPO demo."
  )
  parser.add_argument("--batch_size", type=int, default=64)
  parser.add_argument(
      "--mini_batch_size",
      type=int,
      default=64,
      help="Number of prompt groups per optimizer update.",
  )
  parser.add_argument("--num_generations", type=int, default=8)
  parser.add_argument("--max_steps", type=int, default=450)
  parser.add_argument("--max_turns", type=int, default=8)
  parser.add_argument("--dataset_size", type=int, default=10000)
  parser.add_argument("--num_batches", type=int, default=150)
  parser.add_argument("--num_epochs", type=int, default=3)
  parser.add_argument("--eval_dataset_size", type=int, default=100)
  parser.add_argument("--eval_every_n_steps", type=int, default=10)
  parser.add_argument("--max_prompt_length", type=int, default=2048)
  parser.add_argument("--max_response_length", type=int, default=2048)
  parser.add_argument("--train_micro_batch_size", type=int, default=4)
  parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-8B")
  parser.add_argument("--tokenizer_path", type=str, default="")
  parser.add_argument("--temperature", type=float, default=0.7)
  parser.add_argument("--top_p", type=float, default=1.0)
  parser.add_argument("--top_k", type=int, default=0)
  parser.add_argument("--beta", type=float, default=0.0)
  parser.add_argument("--epsilon", type=float, default=0.003)
  parser.add_argument("--epsilon_high", type=float, default=0.005)
  parser.add_argument("--loss_algo", type=str, default="gspo-token")
  parser.add_argument(
      "--loss_agg_mode", type=str, default="sequence-mean-token-mean"
  )
  parser.add_argument("--kl_loss_mode", type=str, default="low_var_kl")
  parser.add_argument(
      "--advantage_estimator",
      choices=("grpo", "rloo", "drgrpo"),
      default="rloo",
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
      default=weight_sync.WeightSyncMode(
          os.getenv("WEIGHT_SYNC_MODE", "raiden")
      ),
      choices=list(weight_sync.WeightSyncMode),
  )
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
      "--shuffle", action=argparse.BooleanOptionalAction, default=True
  )
  parser.add_argument(
      "--is_slippery", action=argparse.BooleanOptionalAction, default=False
  )
  parser.add_argument(
      "--use_multistep_prompt",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser.add_argument("--episode_timeout_secs", type=int, default=600)
  parser.add_argument(
      "--use_rollout_logps",
      action=argparse.BooleanOptionalAction,
      default=True,
  )
  parser.add_argument(
      "--sampler_is", choices=("none", "token"), default="token"
  )
  parser.add_argument("--sampler_is_threshold", type=float, default=2.0)
  parser.add_argument(
      "--max_seq_token_per_tpu",
      type=int,
      default=None,
      help="Enables sequence-packed batches when set.",
  )
  parser.add_argument("--max_segments_per_packed_row", type=int, default=None)
  parser.add_argument("--trainer_fsdp", type=int, default=None)
  parser.add_argument("--trainer_dp", type=int, default=None)
  parser.add_argument(
      "--log_dir",
      type=str,
      default=os.getenv("LOG_DIR", "/tmp/trellis_frozenlake"),
  )
  parser.add_argument(
      "--wandb_project",
      type=str,
      default=os.getenv("WANDB_PROJECT", "trellis-frozenlake"),
  )
  parser.add_argument(
      "--wandb_run_name",
      type=str,
      default=os.getenv("WANDB_RUN_NAME", ""),
  )
  parser.add_argument("--flush_every_n_steps", type=int, default=1)
  parser.add_argument("--rpc_timeout_s", type=float, default=1800.0)
  parser.add_argument("--init_timeout_s", type=float, default=None)
  parser.add_argument("--stop_workers_on_exit", action="store_true")
  parser.add_argument("--debug", action="store_true")
  return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
  if args.num_generations <= 1:
    raise ValueError("num_generations must be greater than 1 for GRPO.")
  if args.batch_size <= 0:
    raise ValueError("batch_size must be positive.")
  if args.mini_batch_size <= 0:
    raise ValueError("mini_batch_size must be positive.")
  if args.batch_size % args.mini_batch_size != 0:
    raise ValueError("batch_size must be divisible by mini_batch_size.")
  if args.train_micro_batch_size <= 0:
    raise ValueError("train_micro_batch_size must be positive.")
  update_size = args.mini_batch_size * args.num_generations
  if update_size % args.train_micro_batch_size != 0:
    raise ValueError(
        "mini_batch_size * num_generations must be divisible by "
        "train_micro_batch_size."
    )
  if args.max_steps <= 0 or args.max_turns <= 0:
    raise ValueError("max_steps and max_turns must be positive.")
  if min(
      args.dataset_size,
      args.num_batches,
      args.num_epochs,
      args.eval_dataset_size,
  ) <= 0:
    raise ValueError("dataset and evaluation sizes must be positive.")
  if args.eval_every_n_steps < 0:
    raise ValueError("eval_every_n_steps must be non-negative.")
  if args.max_staleness < 0:
    raise ValueError("offpolicy/max_staleness must be non-negative.")
  if args.epsilon_high < args.epsilon:
    raise ValueError("epsilon_high must be greater than or equal to epsilon.")
  if args.loss_algo not in ("grpo", "gspo-token"):
    raise ValueError("loss_algo must be either grpo or gspo-token.")
  if args.episode_timeout_secs <= 0:
    raise ValueError("episode_timeout_secs must be positive.")
  if args.sampler_is_threshold <= 0:
    raise ValueError("sampler_is_threshold must be positive.")
  if args.weight_sync_mode == weight_sync.WeightSyncMode.FALLBACK:
    raise ValueError(
        "weight_sync_mode=fallback does not transfer weights; use none for a "
        "smoke test or raiden for multi-step training."
    )


def _build_algo(args: argparse.Namespace) -> algorithm_adapter.GRPOAdapter:
  return algorithm_adapter.GRPOAdapter(
      group_size=args.num_generations,
      mini_batch_size=args.mini_batch_size,
      train_micro_batch_size=args.train_micro_batch_size,
      max_turns=args.max_turns,
      max_packed_len=(
          args.max_seq_token_per_tpu
          if args.max_seq_token_per_tpu is not None
          else args.max_prompt_length + args.max_response_length
      ),
      max_response_length=args.max_response_length,
      clip_epsilon=args.epsilon,
      epsilon_high=args.epsilon_high,
      beta_kl=args.beta,
      temperature=args.temperature,
      loss_algo=args.loss_algo,
      policy_loss_fn="grpo",
      advantage_estimator=args.advantage_estimator,
      loss_agg_mode=args.loss_agg_mode,
      kl_loss_mode=args.kl_loss_mode,
      use_rollout_logps=args.use_rollout_logps,
      sampler_is=None if args.sampler_is == "none" else args.sampler_is,
      sampler_is_threshold=args.sampler_is_threshold,
  )


def _configure_trainer_loss(
    trainer_handle: remote_execution.ActorHandle,
    *,
    algo: algorithm_adapter.GRPOAdapter,
    pad_id: int,
    eos_id: int,
) -> None:
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
  _validate_args(args)
  logging.basicConfig(
      level=logging.DEBUG if args.debug else logging.INFO,
      format="%(asctime)s - [FrozenLakeOrchestrator] %(message)s",
      force=True,
  )
  logging.info(
      "Starting distributed FrozenLake: model=%s full_batch=%d "
      "mini_batch=%d generations=%d steps=%d turns=%d weight_sync=%s",
      args.model_id,
      args.batch_size,
      args.mini_batch_size,
      args.num_generations,
      args.max_steps,
      args.max_turns,
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

  dataset = frozenlake.prepare_dataset(
      frozenlake.create_dataset(size=args.dataset_size, seed=args.seed),
      shuffle=args.shuffle,
      seed=args.seed,
      batch_size=args.batch_size,
      num_batches=args.num_batches,
      num_epochs=args.num_epochs,
  )
  eval_dataset = frozenlake.prepare_dataset(
      frozenlake.create_dataset(size=args.eval_dataset_size, seed=123),
      shuffle=args.shuffle,
      seed=args.seed,
      batch_size=args.batch_size,
      num_batches=2,
      num_epochs=1,
  )
  logging.info(
      "Prepared %d training and %d held-out FrozenLake configurations.",
      len(dataset),
      len(eval_dataset),
  )

  cluster = orchestrator.ClusterOrchestrator(
      weight_sync_mode=args.weight_sync_mode
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

  algo = _build_algo(args)
  trainer_handles = cluster.worker_handles(datatypes.Role.ACTOR)
  if len(trainer_handles) != 1:
    raise ValueError(f"Expected 1 trainer worker, got {len(trainer_handles)}.")
  _configure_trainer_loss(
      trainer_handles[0], algo=algo, pad_id=pad_id, eos_id=eos_id
  )

  metrics_options = metrics_logger_lib.MetricsLoggerOptions(
      log_dir=args.log_dir,
      project_name=args.wandb_project,
      run_name=args.wandb_run_name,
      flush_every_n_steps=args.flush_every_n_steps,
      backend_kwargs={"wandb": {"config": vars(args)}},
  )
  program = rl_program.StandardRLProgram(
      algo=algo,
      dataset=frozenlake.iter_prompt_items(
          dataset=dataset,
          max_steps=args.max_steps,
          batch_size=args.batch_size,
          max_turns=args.max_turns,
          max_response_length=args.max_response_length,
          episode_timeout_secs=args.episode_timeout_secs,
          temperature=args.temperature,
          top_p=args.top_p,
          top_k=args.top_k,
          is_slippery=args.is_slippery,
          use_multistep_prompt=args.use_multistep_prompt,
      ),
      max_steps=args.max_steps,
      reward_fns=[],
      batch_size=args.batch_size,
      batch_config=batch_assembly.BatchConfig(
          pad_id=pad_id,
          max_prompt_length=args.max_prompt_length,
          max_response_length=args.max_response_length,
          max_seq_token_per_tpu=args.max_seq_token_per_tpu,
          max_segments_per_packed_row=args.max_segments_per_packed_row,
          trainer_fsdp=args.trainer_fsdp,
          trainer_dp=args.trainer_dp,
      ),
      metrics_logging_options=metrics_options,
      max_staleness=args.max_staleness,
      sync_weights=(args.weight_sync_mode != weight_sync.WeightSyncMode.NONE),
      evaluation_dataset=list(
          frozenlake.iter_prompt_items(
              dataset=eval_dataset,
              max_steps=1,
              batch_size=args.batch_size,
              max_turns=args.max_turns,
              max_response_length=args.max_response_length,
              episode_timeout_secs=args.episode_timeout_secs,
              temperature=args.temperature,
              top_p=args.top_p,
              top_k=args.top_k,
              is_slippery=args.is_slippery,
              use_multistep_prompt=args.use_multistep_prompt,
              prompt_id_prefix="frozenlake_eval",
              num_prompt_items=len(eval_dataset),
          )
      ) if args.eval_every_n_steps else None,
      eval_every_n_steps=args.eval_every_n_steps or None,
      success_reward_threshold=0.1,
      on_step_begin=lambda step: logging.info(
          ">>> FrozenLake step %d starting", step
      ),
      on_step_end=lambda step, result: logging.info(
          "<<< FrozenLake step %d finished | %s", step, result
      ),
  )

  try:
    cluster.bring_up_workers(dummy_data=None)
    cluster.run(program=program, num_steps=args.max_steps, bring_up=False)
  finally:
    program.close()
    if args.stop_workers_on_exit:
      cluster.shutdown()
    else:
      cluster.monitor.close()

  result = program.last_step_result
  if result is not None:
    logging.info(
        "FrozenLake finished: step=%d policy_version=%d rollouts=%d "
        "microbatches=%d reward_mean=%.4f reward_std=%.4f",
        result.step,
        result.policy_version,
        result.num_rollouts,
        result.num_microbatches,
        result.reward_mean,
        result.reward_std,
    )


if __name__ == "__main__":
  main(sys.argv[1:])
