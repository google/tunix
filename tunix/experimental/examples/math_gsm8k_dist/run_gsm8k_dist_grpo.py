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

"""CPU control-plane for a minimal Orchestrator V2 GSM8K GRPO demo.

The TPU worker processes host the expensive pieces:
  1. a TrainerWorker backed by experimental PeftTrainer V2,
  2. a vLLM RolloutWorker,
  3. optionally an InferenceWorker for frozen reference log-probs.

This process only owns Orchestrator V2 control flow. It registers remote worker
handles with ClusterOrchestrator, configures the GRPO loss on the trainer worker,
and executes StandardRLProgram through ClusterOrchestrator.run_program().
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import functools
import logging
import os
import sys
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # pylint: disable=g-import-not-at-top
from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
if REPO_ROOT not in sys.path:
  sys.path.insert(0, REPO_ROOT)

from tunix.experimental.common import datatypes  # pylint: disable=g-import-not-at-top
from tunix.experimental.distributed.runtime import context as runtime_context  # pylint: disable=g-import-not-at-top
from tunix.experimental.examples.math_gsm8k_dist import gsm8k  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import algorithm_adapter  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import batch_assembly  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import orchestrator  # pylint: disable=g-import-not-at-top
from tunix.experimental.orchestrator import rl_program  # pylint: disable=g-import-not-at-top
from tunix.experimental.weight_sync import weight_sync  # pylint: disable=g-import-not-at-top
from tunix.experimental.worker import remote_execution  # pylint: disable=g-import-not-at-top
from tunix.sft import metrics_logger as metrics_logger_lib  # pylint: disable=g-import-not-at-top

ProcessContext = runtime_context.ProcessContext

def _parse_args(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Orchestrator V2 Qwen3 GSM8K GRPO demo."
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=4,
      help="Number of prompt groups per step.",
  )
  parser.add_argument("--num_generations", type=int, default=8)
  parser.add_argument("--max_steps", type=int, default=1)
  parser.add_argument("--max_prompt_length", type=int, default=1024)
  parser.add_argument("--max_response_length", type=int, default=1024)
  parser.add_argument("--train_micro_batch_size", type=int, default=1)
  parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
  parser.add_argument("--tokenizer_path", type=str, default="")
  parser.add_argument("--temperature", type=float, default=1.0)
  parser.add_argument("--top_p", type=float, default=1.0)
  parser.add_argument("--top_k", type=int, default=-1)
  parser.add_argument(
      "--beta",
      type=float,
      default=0.0,
      help=(
          "KL coefficient. Set to 0.04 with a reference inference worker to "
          "match the Qwen3 GSM8K recipe."
      ),
  )
  parser.add_argument("--epsilon", type=float, default=0.2)
  parser.add_argument(
      "--offpolicy",
      "--max_staleness",
      dest="max_staleness",
      type=int,
      default=0,
      help=(
          "Maximum policy-version lag accepted by the async rollout queue. "
          "0 means queue-level on-policy training."
      ),
  )
  parser.add_argument(
      "--weight_sync_mode",
      type=weight_sync.WeightSyncMode,
      default=weight_sync.WeightSyncMode(os.getenv("WEIGHT_SYNC_MODE", "none")),
      choices=list(weight_sync.WeightSyncMode),
      help=(
          "Weight synchronization mode. 'none' disables post-update sync, "
          "'raiden' uses Raiden, and 'fallback' runs protocol-only sync."
      ),
  )
  parser.add_argument(
      "--reward_mode",
      choices=("env", "exact"),
      default="env",
      help=(
          "env uses rollout environment rewards; exact recomputes the same "
          "GSM8K reward in the orchestrator from returned trajectory text."
      ),
  )
  parser.add_argument(
      "--tfds_data_dir",
      type=str,
      default=os.getenv("TFDS_DATA_DIR", "/tmp/gsm8k_data"),
  )
  parser.add_argument("--tfds_split", type=str, default="train")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
      "--shuffle", action=argparse.BooleanOptionalAction, default=True
  )
  parser.add_argument(
      "--log_dir",
      type=str,
      default=os.getenv("LOG_DIR", "/tmp/trellis_gsm8k"),
      help="Directory for local event logging (TensorBoard/CLU).",
  )
  parser.add_argument(
      "--wandb_project",
      type=str,
      default=os.getenv("WANDB_PROJECT", "trellis-gsm8k"),
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
  parser.add_argument("--inference_addr", type=str, default="")
  parser.add_argument("--stop_workers_on_exit", action="store_true")
  parser.add_argument(
      "--debug",
      action="store_true",
      help="Enable debug logging and print full sampler responses.",
  )
  return parser.parse_args(argv)



def _grpo_model_input(
    train_example: Any,
    *,
    algo_config: Any,
    pad_id: int,
    eos_id: int,
) -> dict[str, Any]:
  """Maps an RLTrainerPayload microbatch to algo_core.grpo_loss_fn kwargs."""
  return {
      "train_example": train_example,
      "algo_config": algo_config,
      "pad_id": pad_id,
      "eos_id": eos_id,
  }


def _build_algo(args: argparse.Namespace) -> algorithm_adapter.GRPOAdapter:
  algo = algorithm_adapter.GRPOAdapter(
      group_size=args.num_generations,
      # StandardRLProgram consumes this many prompt groups per trainer update.
      mini_batch_size=args.batch_size,
      max_packed_len=args.max_prompt_length + args.max_response_length,
      clip_epsilon=args.epsilon,
      beta_kl=args.beta,
  )
  return algo


def _get_config_attr(config: Any, key: str, default: Any = None) -> Any:
  if config is None:
    return default
  if isinstance(config, dict):
    return config.get(key, default)
  return getattr(config, key, default)


def _build_grpo_config(args: argparse.Namespace) -> Any:
  return SimpleNamespace(
      beta=args.beta,
      epsilon=args.epsilon,
      loss_algo="grpo",
      loss_agg_mode="sequence-mean-token-mean",
      temperature=args.temperature,
      kl_loss_mode="mse_kl",
      kl_clamp_value=None,
  )


def _configure_trainer_loss(
    trainer_handle: remote_execution.ActorHandle,
    *,
    algo: algorithm_adapter.GRPOAdapter,
    grpo_config: Any,
    pad_id: int,
    eos_id: int,
) -> None:
  beta = _get_config_attr(grpo_config, "beta", "N/A")
  epsilon = _get_config_attr(grpo_config, "epsilon", "N/A")
  loss_algo = _get_config_attr(grpo_config, "loss_algo", "N/A")
  logging.info(
      "Configuring trainer-side GRPO loss via TrainerWorker RPC (beta=%s, "
      "epsilon=%s, loss_algo=%s).",
      beta,
      epsilon,
      loss_algo,
  )
  trainer_handle.submit("with_loss_fn", algo.loss_fn(), has_aux=True)
  trainer_handle.submit(
      "with_gen_model_input_fn",
      functools.partial(
          _grpo_model_input,
          algo_config=grpo_config,
          pad_id=pad_id,
          eos_id=eos_id,
      ),
  )


def _build_prompt_item(
    *,
    example: dict[str, Any],
    prompt_idx: int,
    max_response_length: int,
    temperature: float,
    top_p: float,
    top_k: int | None,
) -> dict[str, Any]:
  prompt = gsm8k.as_text(example["prompts"])
  question = gsm8k.as_text(example["question"])
  answer = gsm8k.normalize_example_value(example["answer"])
  prompt_id = f"prompt_{prompt_idx}"
  return {
      "prompt": prompt,
      "prompt_id": prompt_id,
      "generation_kwargs": {
          "max_generation_steps": max_response_length,
          "temperature": temperature,
          "top_p": top_p,
          "top_k": top_k,
          "return_logprobs": True,
      },
      "metadata": {
          "answer": answer,
          "gold_answer": answer,
          "question": question,
          "prefix_hash": prompt_id,
          "env_config": {
              "prompt": prompt,
              "prompts": prompt,
              "question": question,
              "answer": answer,
              "gold_answer": answer,
              "max_steps": 1,
          },
      },
  }


def _iter_prompt_items(
    args: argparse.Namespace,
) -> Iterator[dict[str, Any]]:
  top_k = None if args.top_k < 0 else args.top_k
  dataset = gsm8k.load_gsm8k_dataset(
      split=args.tfds_split,
      data_dir=args.tfds_data_dir,
      shuffle=args.shuffle,
      seed=args.seed,
  )
  dataset_size = len(dataset)
  if dataset_size == 0:
    raise ValueError("GSM8K dataset is empty.")
  for prompt_idx in range(args.max_steps * args.batch_size):
    example = dataset[prompt_idx % dataset_size]
    assert example is not None
    yield _build_prompt_item(
        example=example,
        prompt_idx=prompt_idx,
        max_response_length=args.max_response_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=top_k,
    )


def main(argv: list[str], context: ProcessContext | None = None) -> None:
  assert (
      context and context.ipc and context.ipc.discovery
  ), "Require discovery API, but process context doesn't support."

  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - [Orchestrator] %(message)s",
      force=True,
  )

  args = _parse_args(argv)
  if args.num_generations <= 1:
    raise ValueError("num_generations must be greater than 1 for GRPO.")
  if args.batch_size <= 0:
    raise ValueError("batch_size must be positive.")
  if args.train_micro_batch_size <= 0:
    raise ValueError("train_micro_batch_size must be positive.")
  if args.max_staleness < 0:
    raise ValueError("offpolicy/max_staleness must be non-negative.")

  logging.info("=== Starting Distributed GSM8K GRPO Orchestrator ===")
  logging.info(
      "Configuration: model_id=%s, batch_size=%d (prompt groups), "
      "num_generations=%d (%d rollouts/step), max_steps=%d, "
      "train_micro_batch_size=%d, beta=%.4f, epsilon=%.2f, reward_mode=%s, "
      "max_staleness=%d, weight_sync_mode=%s.",
      args.model_id,
      args.batch_size,
      args.num_generations,
      args.batch_size * args.num_generations,
      args.max_steps,
      args.train_micro_batch_size,
      args.beta,
      args.epsilon,
      args.reward_mode,
      args.max_staleness,
      args.weight_sync_mode,
  )
  logging.info("Control-plane JAX backend: %s", jax.default_backend())
  logging.info(
      "Dataset: GSM8K split=%s data_dir=%s reward_mode=%s.",
      args.tfds_split,
      args.tfds_data_dir,
      args.reward_mode,
  )

  tokenizer_path = args.tokenizer_path or os.getenv("MODEL_DIR") or args.model_id
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
  if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
    tokenizer.pad_token = tokenizer.eos_token
  pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
  eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
  logging.info(
      "Loaded tokenizer from %s (vocab_size=%d, pad_id=%d, eos_id=%d).",
      tokenizer_path,
      len(tokenizer),
      pad_id,
      eos_id,
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

  logging.info("Waiting for workers to register via discovery service...")
  cluster.wait_for_workers(
      min_workers={
          datatypes.Role.ACTOR: 1,
          datatypes.Role.ROLLOUT: 1,
          datatypes.Role.REFERENCE: 1 if args.beta != 0.0 else 0,
      },
      timeout=args.init_timeout_s,
      poll_interval_s=1.0,
  )
  logging.info("Registered Orchestrator V2 workers: %s", cluster.worker_infos())

  algo = _build_algo(args)
  grpo_config = _build_grpo_config(args)
  trainer_handles = cluster.worker_handles(datatypes.Role.ACTOR)
  assert (
      len(trainer_handles) == 1
  ), f"Expected 1 trainer worker, got {len(trainer_handles)}."
  _configure_trainer_loss(
      trainer_handles[0],
      algo=algo,
      grpo_config=grpo_config,
      pad_id=pad_id,
      eos_id=eos_id,
  )

  metrics_logging_options = metrics_logger_lib.MetricsLoggerOptions(
      log_dir=args.log_dir,
      project_name=args.wandb_project,
      run_name=args.wandb_run_name,
      flush_every_n_steps=1,
      backend_kwargs={
          "wandb": {
              "config": vars(args),
          }
      },
  )

  reward_fns = (
      [gsm8k.make_gsm8k_reward_fn(debug=args.debug)]
      if args.reward_mode == "exact"
      else []
  )
  program = rl_program.StandardRLProgram(
      algo=algo,
      dataset=_iter_prompt_items(args),
      max_steps=args.max_steps,
      reward_fns=reward_fns,
      assembler=batch_assembly.PaddedBatchAssembler(
          batch_size=args.train_micro_batch_size,
          max_prompt_length=args.max_prompt_length,
          max_response_length=args.max_response_length,
          pad_id=pad_id,
          group_size=algo.group_size,
          mini_batch_size=algo.mini_batch_size,
      ),
      metrics_logging_options=metrics_logging_options,
      max_staleness=args.max_staleness,
      sync_weights=(args.weight_sync_mode != "none"),
      on_step_begin=lambda step: logging.info(
          ">>> Step %d starting | Policy Version: %d",
          step,
          step,
      ),
      on_step_end=lambda step, result: logging.info(
          "<<< Step %d finished | Advanced to Policy Version: %d",
          step,
          step + 1,
      ),
  )

  try:
    logging.info("Bringing up remote workers through ClusterOrchestrator...")
    cluster.bring_up_workers(dummy_data=None)
    logging.info(
        "Cluster workers ready: %s. Starting StandardRLProgram execution...",
        [w.worker_id for w in cluster.worker_infos()],
    )
    cluster.run_program(
        program=program,
        num_steps=args.max_steps,
        bring_up=False,
    )
  finally:
    program.close()
    if args.stop_workers_on_exit:
      logging.info("Shutting down cluster workers...")
      cluster.shutdown()
    else:
      cluster.monitor.close()

  result = program.last_step_result
  if result is not None:
    logging.info(
        "=== GRPO Training Finished Successfully ===\n"
        "  Final step: %d\n"
        "  Final policy version: %d\n"
        "  Total rollouts: %d\n"
        "  Total microbatches: %d\n"
        "  Final step reward: mean=%.4f, std=%.4f",
        result.step,
        result.policy_version,
        result.num_rollouts,
        result.num_microbatches,
        result.reward_mean,
        result.reward_std,
    )
  else:
    logging.info("=== GRPO Training Finished (No step results) ===")


if __name__ == "__main__":
  main(sys.argv[1:])
