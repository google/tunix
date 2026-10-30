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
  1. a TrainerWorker backed by PeftTrainer V2,
  2. a vLLM RolloutWorker,
  3. optionally an InferenceWorker for frozen reference log-probs.

This process only owns Orchestrator V2 control flow. It registers remote worker
handles with ClusterOrchestrator, configures the GRPO loss on the trainer
worker,
and executes StandardRLProgram through ClusterOrchestrator.run().
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Iterator
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

from tunix.dist_rl.common import datatypes  # pylint: disable=g-import-not-at-top
from tunix.dist_rl.distributed.runtime import context as runtime_context  # pylint: disable=g-import-not-at-top
from tunix.dist_rl.examples.math_gsm8k_dist import gsm8k  # pylint: disable=g-import-not-at-top
from tunix.dist_rl.orchestrator import algorithm_adapter  # pylint: disable=g-import-not-at-top
from tunix.dist_rl.orchestrator import batch_assembly  # pylint: disable=g-import-not-at-top
from tunix.dist_rl.orchestrator import dataset_utils  # pylint: disable=g-import-not-at-top
from tunix.dist_rl.orchestrator import orchestrator  # pylint: disable=g-import-not-at-top
from tunix.dist_rl.orchestrator import rl_program  # pylint: disable=g-import-not-at-top
from tunix.dist_rl.weight_sync import weight_sync  # pylint: disable=g-import-not-at-top
from tunix.dist_rl.worker import remote_execution  # pylint: disable=g-import-not-at-top
from tunix.rl import algorithm_config  # pylint: disable=g-import-not-at-top
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
  parser.add_argument(
      "--mini_batch_size",
      type=int,
      default=None,
      help=(
          "Number of prompt groups per optimizer update. Defaults to"
          " batch_size."
      ),
  )
  parser.add_argument("--num_generations", type=int, default=8)
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
  parser.add_argument(
      "--segment_alignment_boundary",
      type=int,
      default=1,
      help=(
          "Token boundary every packed segment after the first in a row starts"
          " on when sequence packing is enabled. 1 packs segments back to"
          " back; use 64 for Qwen3.5 GatedDeltaNet (gdn_chunk_size=64)."
      ),
  )
  parser.add_argument(
      "--num_data_shards",
      type=int,
      default=None,
      help=(
          "Total number of trainer data-parallel shards (e.g. fsdp * dp *"
          " expert) used as the sequence packing pack_size."
      ),
  )
  parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-1.7B")
  parser.add_argument(
      "--model_dir", type=str, default=os.getenv("MODEL_DIR", "")
  )
  parser.add_argument("--tokenizer_path", type=str, default="")
  parser.add_argument("--temperature", type=float, default=1.0)
  parser.add_argument("--top_p", type=float, default=1.0)
  parser.add_argument("--top_k", type=int, default=-1)
  parser.add_argument(
      "--opd_teacher_url",
      default="",
      help="Enable on-policy distillation with this SGLang teacher URL.",
  )
  parser.add_argument(
      "--opd_teacher_tokenizer_path",
      default="",
      help=(
          "Explicit teacher tokenizer directory or HF ID, required with OPD."
          " Must match the tokenizer served at opd_teacher_url."
      ),
  )
  parser.add_argument("--opd_coefficient", type=float, default=1.0)
  parser.add_argument(
      "--opd_teacher_revision",
      default="",
      help=(
          "Caller-declared immutable teacher snapshot/revision, required to"
          " resume positive-coefficient OPD checkpoints. This does not verify"
          " which weights the remote server actually serves."
      ),
  )
  parser.add_argument(
      "--opd_task_reward_coefficient",
      type=float,
      default=0.0,
      help=(
          "Scale the normalized task advantage before adding OPD; 0 selects"
          " pure distillation, 1 combines the original task advantage."
      ),
  )
  parser.add_argument("--opd_teacher_timeout_seconds", type=float, default=60.0)
  parser.add_argument("--opd_max_concurrency", type=int, default=8)
  parser.add_argument("--opd_models_manifest", default="")
  parser.add_argument("--opd_eval_output_dir", default="")
  parser.add_argument("--opd_require_resume", action="store_true")
  parser.add_argument(
      "--opd_eval_config",
      default="",
      help="JSON configuration for baseline/final or checkpoint evaluation.",
  )
  parser.add_argument(
      "--opd_eval_only",
      action="store_true",
      help="Restore and evaluate an existing OPD checkpoint without training.",
  )
  parser.add_argument(
      "--in_flight_weight_updates",
      type=_str2bool,
      default=_str2bool(os.getenv("IN_FLIGHT_WEIGHT_UPDATES", "false")),
      help=(
          "Describe the rollout worker setting for OPD validation; the worker"
          " config controls execution. OPD requires false."
      ),
  )
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
      default=int(os.getenv("MAX_STALENESS", "0")),
      help=(
          "Maximum policy-version lag accepted by the async rollout queue. "
          "0 means queue-level on-policy training."
      ),
  )
  parser.add_argument(
      "--trajectory_group_order",
      choices=("trajectory_completion", "prompt_arrival"),
      default=os.getenv("TRAJECTORY_GROUP_ORDER", "trajectory_completion"),
      help="Trajectory group order.",
  )
  parser.add_argument(
      "--rollout_replicas",
      type=int,
      default=1,
      help="Number of rollout worker replicas to wait for.",
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
      choices=("env", "reward_fn"),
      default="env",
      help=(
          "Reward mode. 'env' uses rollout environment trajectory rewards; "
          "'reward_fn' evaluates orchestrator reward_fns."
      ),
  )
  parser.add_argument(
      "--tfds_data_dir",
      type=str,
      default=os.getenv("TFDS_DATA_DIR", "/tmp/gsm8k_data"),
  )
  parser.add_argument("--tfds_split", type=str, default="train")
  parser.add_argument("--tfds_dataset", type=str, default="gsm8k")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument(
      "--generation_seed",
      type=int,
      default=None,
      help=(
          "Optional base rollout seed; request seeds also depend on prompt and"
          " group index."
      ),
  )
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
      "--trajectory_log_dir",
      type=str,
      default=os.getenv("TRAJECTORY_LOG_DIR", None),
      help=(
          "Directory for trajectory logging. Defaults to "
          "<log_dir>/trajectories when log_dir is set."
      ),
  )
  parser.add_argument(
      "--flush_metrics_every_n_steps",
      type=int,
      default=1,
      help="Frequency in steps to flush metrics logger.",
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
      "--debug",
      action="store_true",
      help="Enable debug logging and print full sampler responses.",
  )
  parser.add_argument(
      "--run_id",
      type=str,
      default=os.getenv("RUN_ID", ""),
      help="Run identifier shared between orchestrator and rollout workers.",
  )
  parser.add_argument(
      "--enable_trajectory_store",
      type=_str2bool,
      default=False,
      nargs="?",
      const=True,
      help="Whether to enable TrajectoryStore.",
  )
  parser.add_argument(
      "--trajectory_store_backend",
      type=str,
      default="file",
      help="TrajectoryStore backend name (e.g. 'file').",
  )
  parser.add_argument(
      "--trajectory_store_dir",
      type=str,
      default="",
      help="Root directory for the file-backed TrajectoryStore.",
  )
  parser.add_argument(
      "--trainer_precompile",
      type=_str2bool,
      default=_str2bool(os.getenv("TRAINER_PRECOMPILE", "true")),
      nargs="?",
      const=True,
      help=(
          "Precompile the trainer step executables (fwd_bwd, update, eval)"
          " ahead of time during worker configuration."
      ),
  )
  return parser.parse_args(argv)


def _str2bool(v: str | bool) -> bool:
  """Converts string representations of booleans to bool."""
  if isinstance(v, bool):
    return v
  if v.lower() in ("yes", "true", "t", "y", "1"):
    return True
  if v.lower() in ("no", "false", "f", "n", "0"):
    return False
  raise argparse.ArgumentTypeError(f"Boolean value expected, got {v}")


def _validate_opd_args(args: argparse.Namespace) -> None:
  """Rejects unsupported OPD sampling and worker schedules before setup."""
  if not args.opd_teacher_url:
    return
  if not args.opd_teacher_tokenizer_path:
    raise ValueError(
        "--opd_teacher_tokenizer_path is required with --opd_teacher_url."
    )
  if args.temperature != 1.0 or args.top_p != 1.0 or args.top_k not in (-1, 0):
    raise ValueError(
        "OPD requires temperature=1, top_p=1 and top_k=-1 or 0 (disabled)."
    )
  if not args.use_rollout_logps:
    raise ValueError("OPD requires --use_rollout_logps.")
  if args.max_staleness != 0 or args.in_flight_weight_updates:
    raise ValueError(
        "OPD requires max_staleness=0 and in_flight_weight_updates=false."
    )
  if (
      args.opd_coefficient > 0
      and args.weight_sync_mode != weight_sync.WeightSyncMode.RAIDEN
  ):
    raise ValueError(
        "OPD requires --weight_sync_mode=raiden; none/fallback do not update"
        " rollout weights."
    )


def _opd_program_kwargs(
    args: argparse.Namespace, student_tokenizer: Any
) -> dict[str, Any]:
  """Creates teacher resources only when OPD was explicitly enabled."""
  if not args.opd_teacher_url:
    return {}
  from tunix.dist_rl.distillation import identity  # pylint: disable=g-import-not-at-top
  from tunix.dist_rl.distillation import on_policy  # pylint: disable=g-import-not-at-top
  from tunix.dist_rl.distillation import teacher  # pylint: disable=g-import-not-at-top

  config = on_policy.OnPolicyDistillationConfig(
      coefficient=args.opd_coefficient,
      task_reward_coefficient=args.opd_task_reward_coefficient,
  )
  teacher_tokenizer = AutoTokenizer.from_pretrained(
      args.opd_teacher_tokenizer_path, trust_remote_code=True
  )
  if (
      teacher_tokenizer.pad_token_id is None
      and teacher_tokenizer.eos_token is not None
  ):
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
  scorer = teacher.SGLangTeacherScorer(
      args.opd_teacher_url,
      student_tokenizer=student_tokenizer,
      teacher_tokenizer=teacher_tokenizer,
      timeout_seconds=args.opd_teacher_timeout_seconds,
      max_concurrency=args.opd_max_concurrency,
  )
  logging.info(
      "OPD teacher=%s tokenizer=%s coefficient=%s task_advantage_scale=%s.",
      args.opd_teacher_url,
      args.opd_teacher_tokenizer_path,
      config.coefficient,
      config.task_reward_coefficient,
  )
  return {
      "opd_config": config,
      "teacher_scorer": scorer,
      "opd_teacher_identity": identity.TeacherIdentity(
          declared_revision=args.opd_teacher_revision or None,
          tokenizer_fingerprint=scorer.tokenizer_fingerprint,
      ),
  }


def _load_opd_models_manifest(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
  """Checks the control-plane settings against prepared immutable snapshots."""
  if not args.opd_models_manifest:
    if args.opd_eval_config or args.opd_eval_only or args.opd_require_resume:
      raise ValueError("OPD evaluation requires --opd_models_manifest.")
    return None
  if not args.opd_teacher_url:
    raise ValueError("--opd_models_manifest requires OPD to be enabled.")
  from tunix.dist_rl.examples.on_policy_distillation import models  # pylint: disable=g-import-not-at-top

  manifest = models.load_manifest(args.opd_models_manifest)
  student, teacher_record = manifest["student"], manifest["teacher"]
  if args.model_id != student["repo_id"]:
    raise ValueError("model_id differs from the prepared student manifest.")
  tokenizer_path = args.tokenizer_path or args.model_dir or args.model_id
  for actual, expected, name in (
      (args.model_dir, student["snapshot_path"], "student weights"),
      (tokenizer_path, student["tokenizer_path"], "student tokenizer"),
      (
          args.opd_teacher_tokenizer_path,
          teacher_record["tokenizer_path"],
          "teacher tokenizer",
      ),
  ):
    if os.path.realpath(actual) != os.path.realpath(expected):
      raise ValueError(f"{name} differs from the prepared model manifest.")
  if args.opd_teacher_revision != teacher_record["declared_revision"]:
    raise ValueError(
        "Teacher revision differs from the prepared model manifest."
    )
  return manifest


def _load_opd_evaluation(args: argparse.Namespace):
  """Loads a separate generation-evaluation config before any worker request."""
  if not args.opd_eval_config:
    if args.opd_eval_only:
      raise ValueError("--opd_eval_only requires --opd_eval_config.")
    return None
  if (
      not args.opd_teacher_url
      or args.weight_sync_mode != weight_sync.WeightSyncMode.RAIDEN
  ):
    raise ValueError(
        "OPD evaluation requires OPD and actual Raiden synchronization."
    )
  from tunix.dist_rl.examples.on_policy_distillation import evaluation  # pylint: disable=g-import-not-at-top

  with open(args.opd_eval_config, encoding="utf-8") as config_file:
    return evaluation.EvaluationConfig.from_dict(json.load(config_file))


class _EvaluatedOPDProgram(rl_program.StandardRLProgram):
  """Adds generation evaluation without replacing training or restore logic."""

  def __init__(
      self,
      *,
      evaluation_config,
      evaluation_examples,
      evaluation_output_dir,
      model_identity,
      eval_only=False,
      require_resume=False,
      **kwargs,
  ):
    super().__init__(**kwargs)
    self._evaluation_config = evaluation_config
    self._evaluation_examples = evaluation_examples
    self._evaluation_output_dir = evaluation_output_dir
    self._evaluation_model_identity = model_identity
    self._eval_only = eval_only
    self._require_resume = require_resume

  async def _evaluate(self, label: str):
    from tunix.dist_rl.examples.on_policy_distillation import evaluation  # pylint: disable=g-import-not-at-top

    assert self.engine is not None
    model_identity = {
        **self._evaluation_model_identity,
        "evaluation_source": "live_synced_rollout",
        "program_step": self.step,
        "checkpoint_root_directory": os.getenv("CHECKPOINT_ROOT_DIRECTORY", ""),
        "opd_manifest": copy.deepcopy(self._opd_manifest),
        "restored_checkpoint_metadata": copy.deepcopy(
            self.engine.restored_checkpoint_metadata
        ),
    }
    return await evaluation.evaluate_engine(
        self.engine,
        self._evaluation_examples,
        config=self._evaluation_config,
        output_path=os.path.join(self._evaluation_output_dir, f"{label}.json"),
        label=label,
        model_identity=model_identity,
        policy_version=self.policy_version,
    )

  async def _resume_from_checkpoint(self) -> None:
    # This validates the saved OPD objective after the real engine restore and
    # resync, before any generation for evaluation or training.
    await super()._resume_from_checkpoint()
    if self._eval_only:
      if self.step <= 0:
        raise ValueError(
            "OPD evaluation-only requires an existing nonzero checkpoint."
        )
      return
    if self._require_resume and self.step <= 0:
      raise ValueError("OPD resume requires an existing nonzero checkpoint.")
    if self._evaluation_config is None:
      return
    if self.step == 0:
      await self.engine.prepare_rollout_policy(
          role=datatypes.Role.ACTOR,
          sync_weights=True,
          policy_version=self.policy_version,
      )
      await self._evaluate("baseline")

  async def run_async(self, engine, **kwargs) -> None:
    if self._eval_only:
      self.engine = engine
      await self._resume_from_checkpoint()
      await self._evaluate("checkpoint")
      return
    await super().run_async(engine, **kwargs)
    if self._evaluation_config is not None:
      await self._evaluate("final")


def _trajectory_store_config(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
  """Builds TrajectoryStore config dict from parsed CLI flags."""
  if not getattr(args, "enable_trajectory_store", False):
    return None
  config: dict[str, Any] = {
      "backend": getattr(args, "trajectory_store_backend", "file") or "file",
  }
  root_dir = getattr(args, "trajectory_store_dir", "")
  if root_dir:
    config["root_dir"] = root_dir
  run_id = getattr(args, "run_id", "")
  if run_id:
    config["run_id"] = run_id
  return config


def _build_algo(args: argparse.Namespace) -> algorithm_adapter.GRPOAdapter:
  """Builds the GRPO algorithm adapter from CLI arguments."""
  algo_config = algorithm_config.GRPOConfig(
      num_generations=args.num_generations,
      epsilon=args.epsilon,
      beta=args.beta,
      use_rollout_logps=args.use_rollout_logps,
      reward_mode=args.reward_mode,
  )
  return algorithm_adapter.GRPOAdapter(
      algo_config=algo_config,
      mini_batch_size=args.mini_batch_size,
      train_micro_batch_size=args.train_micro_batch_size,
      max_packed_len=(
          args.max_seq_token_per_tpu
          if args.max_seq_token_per_tpu is not None
          else args.max_prompt_length + args.max_response_length
      ),
      max_response_length=args.max_response_length,
  )


def _build_prompt_item(
    *,
    example: dict[str, Any],
    prompt_idx: int,
) -> dict[str, Any]:
  prompt = gsm8k.as_text(example["prompts"])
  question = gsm8k.as_text(example["question"])
  answer = gsm8k.normalize_example_value(example["answer"])
  prompt_id = f"prompt_{prompt_idx}"
  return {
      "prompt": prompt,
      "prompt_id": prompt_id,
      "metadata": {
          "answer": answer,
          "gold_answer": answer,
          "question": question,
          "env_config": {
              "prompt": prompt,
              "prompts": prompt,
              "prompt_id": prompt_id,
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
  dataset = gsm8k.load_gsm8k_dataset(
      split=args.tfds_split,
      data_dir=args.tfds_data_dir,
      shuffle=args.shuffle,
      seed=args.seed,
      dataset_name=args.tfds_dataset,
  )
  dataset_size = len(dataset)
  if dataset_size == 0:
    raise ValueError("GSM8K dataset is empty.")
  total_prompts = dataset_utils.total_prompt_groups(
      args.max_steps, args.batch_size, args.max_staleness
  )
  for prompt_idx in range(total_prompts):
    example = dataset[prompt_idx % dataset_size]
    assert example is not None
    yield _build_prompt_item(
        example=example,
        prompt_idx=prompt_idx,
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
  if args.debug:
    # Enable canonical debug logging and print full sampler responses
    logging.getLogger().setLevel(logging.DEBUG)
  if args.mini_batch_size is None:
    args.mini_batch_size = args.batch_size
  if args.num_generations <= 1:
    raise ValueError("num_generations must be greater than 1 for GRPO.")
  if args.batch_size <= 0:
    raise ValueError("batch_size must be positive.")
  if args.max_staleness < 0:
    raise ValueError("offpolicy/max_staleness must be non-negative.")
  _validate_opd_args(args)
  models_manifest = _load_opd_models_manifest(args)
  eval_config = _load_opd_evaluation(args)

  logging.info("=== Starting Distributed GSM8K GRPO Orchestrator ===")
  logging.info(
      "Configuration: model_id=%s, batch_size=%d (prompt groups), "
      "mini_batch_size=%d, num_generations=%d (%d rollouts/step), "
      "max_steps=%d, train_micro_batch_size=%d, beta=%.4f, epsilon=%.2f, "
      "reward_mode=%s, max_staleness=%d, weight_sync_mode=%s.",
      args.model_id,
      args.batch_size,
      args.mini_batch_size,
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

  tokenizer_path = args.tokenizer_path or args.model_dir or args.model_id
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
  opd_kwargs = _opd_program_kwargs(args, tokenizer)
  if models_manifest is not None:
    if (
        opd_kwargs["opd_teacher_identity"].tokenizer_fingerprint
        != models_manifest["tokenizer_fingerprint"]
    ):
      raise ValueError(
          "Tokenizer fingerprint differs from the prepared model manifest."
      )

  trajectory_store_config = _trajectory_store_config(args)
  cluster = orchestrator.ClusterOrchestrator(
      weight_sync_mode=args.weight_sync_mode,
      trajectory_store_config=trajectory_store_config,
  )
  context.ipc.discovery.on_register(
      functools.partial(
          cluster.registry.register_from_hostname,
          rpc_timeout_s=args.rpc_timeout_s,
      )
  )

  logging.info("Waiting for workers to register via discovery service...")
  cluster.registry.wait_for_workers(
      min_workers={
          datatypes.Role.ACTOR: 1,
          datatypes.Role.ROLLOUT: args.rollout_replicas,
          datatypes.Role.REFERENCE: 1 if args.beta != 0.0 else 0,
      },
      timeout=args.init_timeout_s,
  )
  logging.info(
      "Registered Orchestrator V2 workers: %s", cluster.registry.infos()
  )

  algo = _build_algo(args)

  metrics_logging_options = metrics_logger_lib.MetricsLoggerOptions(
      log_dir=args.log_dir,
      project_name=args.wandb_project,
      run_name=args.wandb_run_name,
      flush_every_n_steps=args.flush_metrics_every_n_steps,
      backend_kwargs={
          "wandb": {
              "config": vars(args),
          }
      },
  )

  reward_fns = (
      [gsm8k.make_gsm8k_reward_fn(debug=args.debug)]
      if args.reward_mode == "reward_fn"
      else []
  )
  generation_args = datatypes.GenerationArgs(
      temperature=args.temperature,
      top_p=args.top_p,
      top_k=0 if args.top_k < 0 else args.top_k,
      return_logprobs=True,
      seed=args.generation_seed,
  )
  program_class = rl_program.StandardRLProgram
  evaluation_kwargs = {}
  if eval_config is not None or args.opd_require_resume:
    from tunix.dist_rl.examples.on_policy_distillation import evaluation  # pylint: disable=g-import-not-at-top

    program_class = _EvaluatedOPDProgram
    evaluation_kwargs = {
        "evaluation_config": eval_config,
        "evaluation_examples": (
            evaluation.load_examples(args.tfds_data_dir, eval_config)
            if eval_config
            else []
        ),
        "evaluation_output_dir": (
            args.opd_eval_output_dir or os.path.join(args.log_dir, "evaluation")
        ),
        "model_identity": models_manifest["student"],
        "eval_only": args.opd_eval_only,
        "require_resume": args.opd_require_resume,
    }
  program = program_class(
      algo=algo,
      dataset=_iter_prompt_items(args),
      max_steps=args.max_steps,
      reward_fns=reward_fns,
      generation_args=generation_args,
      batch_size=args.batch_size,
      batch_config=batch_assembly.BatchConfig(
          pad_id=pad_id,
          max_prompt_length=args.max_prompt_length,
          max_response_length=args.max_response_length,
          max_seq_token_per_tpu=args.max_seq_token_per_tpu,
          max_segments_per_packed_row=args.max_segments_per_packed_row,
          segment_alignment_boundary=args.segment_alignment_boundary,
          num_data_shards=args.num_data_shards,
      ),
      metrics_logging_options=metrics_logging_options,
      trajectory_log_dir=args.trajectory_log_dir,
      trajectory_store=cluster.trajectory_store,
      max_staleness=args.max_staleness,
      group_order=args.trajectory_group_order,
      sync_weights=(args.weight_sync_mode != "none"),
      precompile_trainer=args.trainer_precompile,
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
      **opd_kwargs,
      **evaluation_kwargs,
  )

  try:
    logging.info("Bringing up remote workers through ClusterOrchestrator...")
    cluster.bring_up_workers(dummy_data=None)
    logging.info(
        "Cluster workers ready: %s. Starting StandardRLProgram execution...",
        [w.worker_id for w in cluster.registry.infos()],
    )
    cluster.run(
        program=program,
        num_steps=args.max_steps,
        bring_up=False,
    )
  except BaseException as exc:
    logging.exception("FATAL: StandardRLProgram execution failed: %s", exc)
    raise
  finally:
    program.close()
    if args.stop_workers_on_exit:
      logging.info("Shutting down cluster workers...")
      cluster.shutdown()
    else:
      cluster.monitor.close()
      if cluster.trajectory_store is not None:
        cluster.trajectory_store.close()

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
