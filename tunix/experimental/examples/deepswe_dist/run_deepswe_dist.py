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

"""CPU control-plane for the experimental distributed DeepSWE GRPO recipe."""

from __future__ import annotations

import argparse
import functools
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
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.common import timeouts
from tunix.dist_rl.distributed.runtime import context as runtime_context
from tunix.dist_rl.orchestrator import algorithm_adapter
from tunix.dist_rl.orchestrator import batch_assembly
from tunix.dist_rl.orchestrator import orchestrator
from tunix.dist_rl.orchestrator import rl_program
from tunix.dist_rl.weight_sync import weight_sync
from tunix.dist_rl.weight_sync import weight_sync_coordinator
from tunix.dist_rl.worker import remote_execution
from tunix.experimental.examples.deepswe_dist import deepswe
from tunix.experimental.examples.deepswe_dist import sandbox_utils
from tunix.rl import algorithm_config
from tunix.sft import metrics_logger as metrics_logger_lib

# pylint: enable=g-import-not-at-top


ProcessContext = runtime_context.ProcessContext


def _parse_args(argv: list[str]) -> argparse.Namespace:
  """Parses CLI flags for the DeepSWE distributed GRPO orchestrator."""
  parser = argparse.ArgumentParser(
      description="Orchestrator V2 DeepSWE distributed GRPO recipe."
  )
  parser.add_argument("--batch_size", type=int, default=1)
  parser.add_argument(
      "--mini_batch_size",
      type=int,
      default=None,
      help=(
          "Number of prompt groups per optimizer update. Defaults to"
          " batch_size."
      ),
  )
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
  parser.add_argument("--tokenizer_path", type=str, default="")
  parser.add_argument("--temperature", type=float, default=1.0)
  parser.add_argument("--top_p", type=float, default=1.0)
  parser.add_argument("--top_k", type=int, default=-1)
  parser.add_argument("--beta", type=float, default=0.0)
  parser.add_argument("--epsilon", type=float, default=0.2)
  parser.add_argument(
      "--epsilon_high",
      type=_optional_float,
      default=_optional_float(os.getenv("EPSILON_HIGH")),
      help=(
          "Upper clip bound for asymmetric PPO ratio clipping. Defaults to"
          " epsilon when unset."
      ),
  )
  parser.add_argument(
      "--advantage_estimator",
      type=str,
      default=os.getenv("ADVANTAGE_ESTIMATOR", "grpo"),
      help="Advantage estimator ('grpo' or 'grpo-loo').",
  )
  parser.add_argument(
      "--loss_agg_mode",
      type=str,
      default=os.getenv("LOSS_AGG_MODE", "sequence-mean-token-mean"),
      help="Loss aggregation mode (e.g. 'token-mean').",
  )
  parser.add_argument(
      "--sampler_is",
      type=str,
      default=os.getenv("SAMPLER_IS") or None,
      help="Sampler importance-sampling level ('token', 'sequence', or None).",
  )
  parser.add_argument(
      "--sampler_is_threshold",
      type=_optional_float,
      default=_optional_float(os.getenv("SAMPLER_IS_THRESHOLD", "2.0")),
      help="Upper truncation threshold for sampler IS ratio, or None.",
  )
  parser.add_argument(
      "--sampler_rs",
      type=str,
      default=os.getenv("SAMPLER_RS") or None,
      help=(
          "Sampler rejection-sampling mode ('token', 'sequence', 'geometric',"
          " or None)."
      ),
  )
  parser.add_argument(
      "--sampler_rs_min",
      type=_optional_float,
      default=_optional_float(os.getenv("SAMPLER_RS_MIN")),
      help="Lower acceptance threshold for sampler rejection sampling.",
  )
  parser.add_argument(
      "--sampler_rs_max",
      type=_optional_float,
      default=_optional_float(os.getenv("SAMPLER_RS_MAX")),
      help="Upper acceptance threshold for sampler rejection sampling.",
  )
  parser.add_argument(
      "--seq_logprob_error_threshold",
      type=_optional_float,
      default=_optional_float(os.getenv("SEQ_LOGPROB_ERROR_THRESHOLD")),
      help=(
          "Max per-sequence average absolute logprob drift before masking out"
          " the sequence."
      ),
  )
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
      "--return_routed_experts",
      type=_str2bool,
      default=_str2bool(os.getenv("RETURN_ROUTED_EXPERTS", "false")),
      nargs="?",
      const=True,
      help="Capture MoE router indices during rollout for trainer replay.",
  )
  parser.add_argument(
      "--routed_experts_encode_mode",
      choices=datatypes.ROUTED_EXPERTS_ENCODE_MODES,
      default="auto",
      help="Compact encoding of routed experts in trainer batches.",
  )
  parser.add_argument(
      "--routed_experts_num_experts",
      type=int,
      default=None,
      help=(
          "Experts per MoE layer. Set it with --routed_experts_encode_mode=auto"
          " when 256 <= E <= 1023 so every batch encodes to the same tier and"
          " the trainer step does not recompile."
      ),
  )
  parser.add_argument(
      "--offpolicy",
      "--max_staleness",
      dest="max_staleness",
      type=int,
      default=int(os.getenv("MAX_STALENESS", "0")),
  )
  parser.add_argument(
      "--trajectory_group_order",
      choices=("trajectory_completion", "prompt_arrival", "prompt_batch"),
      default=os.getenv("TRAJECTORY_GROUP_ORDER", "trajectory_completion"),
      help="Trajectory group order.",
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
  parser.add_argument(
      "--step_timeout_secs",
      type=int,
      default=int(os.getenv("STEP_TIMEOUT_SECS", "60")),
  )
  parser.add_argument(
      "--reward_timeout_secs",
      type=int,
      default=int(os.getenv("REWARD_TIMEOUT_SECS", "60")),
  )
  parser.add_argument(
      "--episode_timeout_secs",
      type=int,
      default=int(os.getenv("EPISODE_TIMEOUT_SECS", str(30 * 60))),
  )
  parser.add_argument("--env_backend", type=str, default="kubernetes")
  parser.add_argument(
      "--scaffold",
      choices=("r2egym", "sweagent", "openhands"),
      default="r2egym",
  )
  parser.add_argument("--use_agent_sandbox", action="store_true")
  parser.add_argument(
      "--rollout_replicas",
      type=int,
      default=1,
      help="Number of rollout worker replicas to wait for before step 0.",
  )
  parser.add_argument(
      "--max_warmpool_replicas",
      type=int,
      default=None,
      help=(
          "Maximum replicas per SandboxWarmPool managed by the orchestrator's"
          " PrewarmDatasetIterator. Defaults to num_generations when unset."
      ),
  )
  parser.add_argument(
      "--max_concurrency",
      type=int,
      default=128,
      help="Maximum concurrent sandbox claims across the fleet.",
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
      "--trajectory_log_dir",
      type=str,
      default=os.getenv("TRAJECTORY_LOG_DIR", None),
      help=(
          "Directory for trajectory logging. Defaults to "
          "<log_dir>/trajectories when log_dir is set."
      ),
  )
  parser.add_argument(
      "--trajectory_logging",
      action=argparse.BooleanOptionalAction,
      default=_str2bool(os.getenv("TRAJECTORY_LOGGING", "true")),
      help=(
          "Whether to write monolithic JSON trajectory logs to"
          " trajectory_log_dir."
      ),
  )
  parser.add_argument(
      "--wandb_entity",
      type=str,
      default=os.getenv("WANDB_ENTITY", ""),
      help="W&B entity (team or user) name.",
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
  parser.add_argument(
      "--overlong_filter",
      action=argparse.BooleanOptionalAction,
      default=False,
      help=(
          "Zero out the policy loss mask, final reward, and advantage inclusion"
          " of overlong (max-steps or max-context-limit) trajectories, rather"
          " than training on them."
      ),
  )
  parser.add_argument("--rpc_timeout_s", type=float, default=None)
  parser.add_argument("--init_timeout_s", type=float, default=None)
  parser.add_argument("--stop_workers_on_exit", action="store_true")
  parser.add_argument("--debug", action="store_true")
  parser.add_argument(
      "--run_id",
      type=str,
      default="",
      help="Optional run identifier for TrajectoryStore.",
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
  args = parser.parse_args(argv)
  return args


def _optional_float(v: str | float | None) -> float | None:
  """Parses an optional float CLI argument where 'none'/'null'/'' maps to None."""
  if v is None:
    return None
  if isinstance(v, (int, float)):
    return float(v)
  s = v.strip()
  if not s or s.lower() in ("none", "null"):
    return None
  return float(s)


def _str2bool(v: str | bool) -> bool:
  """Converts string representations of booleans to bool."""
  if isinstance(v, bool):
    return v
  if v.lower() in ("yes", "true", "t", "y", "1"):
    return True
  if v.lower() in ("no", "false", "f", "n", "0"):
    return False
  raise argparse.ArgumentTypeError(f"Boolean value expected, got {v}")


def _trajectory_store_config(
    args: argparse.Namespace,
) -> dict[str, Any] | None:
  """Builds TrajectoryStore config dict from parsed CLI flags."""
  if not args.enable_trajectory_store:
    return None
  config: dict[str, Any] = {
      "backend": args.trajectory_store_backend or "file",
  }
  if args.trajectory_store_dir:
    config["root_dir"] = args.trajectory_store_dir
  if args.run_id:
    config["run_id"] = args.run_id
  return config


def _build_algo(args: argparse.Namespace) -> algorithm_adapter.GRPOAdapter:
  sampler_is = (
      None
      if not args.sampler_is or args.sampler_is.lower() in ("none", "null")
      else args.sampler_is
  )
  sampler_rs = (
      None
      if not args.sampler_rs or args.sampler_rs.lower() in ("none", "null")
      else args.sampler_rs
  )
  algo_config = algorithm_config.GRPOConfig(
      num_generations=args.num_generations,
      epsilon=args.epsilon,
      epsilon_high=args.epsilon_high,
      beta=args.beta,
      advantage_estimator=args.advantage_estimator,
      loss_agg_mode=args.loss_agg_mode,
      use_rollout_logps=args.use_rollout_logps,
      sampler_is=sampler_is,
      sampler_is_threshold=args.sampler_is_threshold,
      sampler_rs=sampler_rs,
      sampler_rs_min=args.sampler_rs_min,
      sampler_rs_max=args.sampler_rs_max,
      seq_logprob_error_threshold=args.seq_logprob_error_threshold,
      routed_experts_encode_mode=args.routed_experts_encode_mode,
      routed_experts_num_experts=args.routed_experts_num_experts,
  )
  return algorithm_adapter.GRPOAdapter(
      algo_config=algo_config,
      mini_batch_size=args.mini_batch_size,
      train_micro_batch_size=args.train_micro_batch_size,
      max_turns=args.max_turns,
      max_packed_len=(
          args.max_seq_token_per_tpu
          if args.max_seq_token_per_tpu is not None
          else args.max_prompt_length + args.max_response_length
      ),
      max_response_length=args.max_response_length,
  )


def _configure_trainer_loss(
    trainer_handle: remote_execution.ActorHandle,
    *,
    algo: algorithm_adapter.GRPOAdapter,
    pad_id: int,
    eos_id: int,
) -> None:
  """Configures the trainer-side GRPO loss via TrainerWorker RPC."""
  logging.info(
      "Configuring trainer-side GRPO loss via TrainerWorker RPC (beta=%s, "
      "epsilon=%s).",
      algo.algo_config.beta,
      algo.algo_config.epsilon,
  )
  trainer_handle.submit("with_loss_fn", algo.loss_fn(), has_aux=True)
  trainer_handle.submit(
      "with_gen_model_input_fn",
      algo.build_gen_model_input_fn(pad_id=pad_id, eos_id=eos_id),
  )


def _validate_args(
    args: argparse.Namespace,
) -> weight_sync_coordinator.PhaseTimeouts:
  """Validates DeepSWE arguments and derives aligned `PhaseTimeouts`."""
  if args.trajectory_group_order == "prompt_batch":
    args.trajectory_group_order = "prompt_arrival"
  if args.mini_batch_size is None:
    args.mini_batch_size = args.batch_size
  if args.num_generations <= 1:
    raise ValueError("num_generations must be greater than 1 for GRPO.")
  if args.batch_size <= 0:
    raise ValueError("batch_size must be positive.")
  if args.max_staleness < 0:
    raise ValueError("offpolicy/max_staleness must be non-negative.")
  if args.use_agent_sandbox and args.max_staleness > 1:
    raise ValueError(
        "use_agent_sandbox only supports max_staleness <= 1, got"
        f" {args.max_staleness}."
    )
  if args.step_timeout_secs <= 0:
    raise ValueError("step_timeout_secs must be positive.")
  if args.reward_timeout_secs <= 0:
    raise ValueError("reward_timeout_secs must be positive.")
  if args.episode_timeout_secs <= 0:
    raise ValueError("episode_timeout_secs must be positive.")
  if args.episode_timeout_secs < args.step_timeout_secs:
    raise ValueError(
        f"episode_timeout_secs ({args.episode_timeout_secs}) must be >="
        f" step_timeout_secs ({args.step_timeout_secs})."
    )
  weight_sync_timeouts = (
      weight_sync_coordinator.PhaseTimeouts.for_drain_timeout(
          timeouts.drain_timeout_for(args.episode_timeout_secs)
      )
  )
  min_rpc_s = weight_sync_timeouts.max_rpc_phase_s()
  if args.rpc_timeout_s is None:
    args.rpc_timeout_s = max(1800.0, min_rpc_s)
  else:
    timeouts.validate_positive_finite(args.rpc_timeout_s, "rpc_timeout_s")
    if args.rpc_timeout_s < min_rpc_s:
      raise ValueError(
          f"rpc_timeout_s ({args.rpc_timeout_s}) must be >="
          f" weight_sync_timeouts.max_rpc_phase_s() ({min_rpc_s})."
      )
  return weight_sync_timeouts


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

  weight_sync_timeouts = _validate_args(args)

  logging.info("=== Starting Distributed DeepSWE GRPO Orchestrator ===")
  logging.info(
      "Configuration: model_id=%s, batch_size=%d prompt group(s), "
      "mini_batch_size=%d, num_generations=%d, max_steps=%d, max_turns=%d, "
      "train_micro=%d, beta=%.4f, env_backend=%s, use_agent_sandbox=%s, "
      "weight_sync_mode=%s.",
      args.model_id,
      args.batch_size,
      args.mini_batch_size,
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

  trajectory_store_config = _trajectory_store_config(args)
  cluster = orchestrator.ClusterOrchestrator(
      weight_sync_mode=args.weight_sync_mode,
      trajectory_store_config=trajectory_store_config,
      weight_sync_timeouts=weight_sync_timeouts,
  )
  context.ipc.discovery.on_register(
      functools.partial(
          cluster.registry.register_from_hostname,
          rpc_timeout_s=args.rpc_timeout_s,
      )
  )

  cluster.registry.wait_for_workers(
      min_workers={
          datatypes.Role.ACTOR: 1,
          datatypes.Role.ROLLOUT: max(1, args.rollout_replicas),
          datatypes.Role.REFERENCE: 1 if args.beta != 0.0 else 0,
      },
      timeout=args.init_timeout_s,
  )
  logging.info("Registered workers: %s", cluster.registry.infos())

  algo = _build_algo(args)
  trainer_handles = cluster.registry.handles(datatypes.Role.ACTOR)
  if len(trainer_handles) != 1:
    raise ValueError(f"Expected 1 trainer worker, got {len(trainer_handles)}.")

  wandb_backend_kwargs: dict[str, Any] = {"config": vars(args)}
  if args.wandb_entity:
    os.environ["WANDB_ENTITY"] = args.wandb_entity
    wandb_backend_kwargs["entity"] = args.wandb_entity
  metrics_logging_options = metrics_logger_lib.MetricsLoggerOptions(
      log_dir=args.log_dir,
      project_name=args.wandb_project,
      run_name=args.wandb_run_name,
      flush_every_n_steps=args.flush_every_n_steps,
      backend_kwargs={"wandb": wandb_backend_kwargs},
  )

  prompt_iter = deepswe.iter_prompt_items(
      dataset=dataset,
      max_steps=args.max_steps,
      batch_size=args.batch_size,
      max_turns=args.max_turns,
      max_response_length=args.max_response_length,
      temperature=args.temperature,
      top_p=args.top_p,
      top_k=0 if args.top_k < 0 else args.top_k,
      step_timeout_secs=args.step_timeout_secs,
      reward_timeout_secs=args.reward_timeout_secs,
      episode_timeout_secs=args.episode_timeout_secs,
      env_backend=args.env_backend,
      use_agent_sandbox=args.use_agent_sandbox,
      scaffold=args.scaffold,
      env_verbose=args.env_verbose,
      overlong_filter=args.overlong_filter,
      max_staleness=args.max_staleness,
      return_routed_experts=args.return_routed_experts,
  )
  orchestrator_fleet = None
  prewarm_iter = None
  if args.use_agent_sandbox:
    job_id = os.environ["ORCHESTRATOR_ID"]
    max_warmpool_replicas = (
        args.max_warmpool_replicas
        if args.max_warmpool_replicas is not None
        else args.num_generations
    )
    logging.info(
        "Initializing orchestrator SandboxFleet + PrewarmDatasetIterator "
        "(job_id=%s, batch_size=%d, num_generations=%d, max_concurrency=%d, "
        "max_warmpool_replicas=%d, max_staleness=%d).",
        job_id,
        args.batch_size,
        args.num_generations,
        args.max_concurrency,
        max_warmpool_replicas,
        args.max_staleness,
    )
    orchestrator_fleet = sandbox_utils.init_global_fleet(
        tasks=None,
        max_concurrency=args.max_concurrency,
        num_generations=args.num_generations,
        batch_size=args.batch_size,
        max_warmpool_replicas=max_warmpool_replicas,
        scaffold=args.scaffold,
        job_id=job_id,
        warm_on_init=False,
        teardown_on_exit=True,
    )
    prewarm_iter = sandbox_utils.PrewarmDatasetIterator(
        prompt_iter,
        fleet=orchestrator_fleet,
        num_generations=args.num_generations,
        batch_size=args.batch_size,
        max_warmpool_replicas=max_warmpool_replicas,
        max_staleness=args.max_staleness,
        unwarm_on_exhaustion=True,
        scaffold=args.scaffold,
    )
    prompt_iter = prewarm_iter

  program = rl_program.StandardRLProgram(
      algo=algo,
      dataset=prompt_iter,
      max_steps=args.max_steps,
      reward_fns=[],
      generation_args=datatypes.GenerationArgs(
          temperature=args.temperature,
          top_p=args.top_p,
          top_k=0 if args.top_k < 0 else args.top_k,
          return_logprobs=args.use_rollout_logps,
          return_routed_experts=args.return_routed_experts,
      ),
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
      trajectory_log_dir=(
          args.trajectory_log_dir if args.trajectory_logging else ""
      ),
      trajectory_store=cluster.trajectory_store,
      max_staleness=args.max_staleness,
      group_order=args.trajectory_group_order,
      sync_weights=(args.weight_sync_mode != weight_sync.WeightSyncMode.NONE),
      precompile_trainer=args.trainer_precompile,
      on_step_begin=lambda step: logging.info(
          ">>> DeepSWE step %d starting | policy_version=%d",
          step,
          step,
      ),
      on_step_end=lambda step, result: logging.info(
          "<<< DeepSWE step %d finished | train_result=%s",
          step,
          result,
      ),
  )
  _configure_trainer_loss(
      trainer_handles[0],
      algo=algo,
      pad_id=pad_id,
      eos_id=eos_id,
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
  except Exception:
    logging.exception("Fatal error in DeepSWE orchestrator execution.")
    raise
  finally:
    if prewarm_iter is not None:
      prewarm_iter.close()
    if orchestrator_fleet is not None:
      sandbox_utils.teardown_global_fleet()
    program.close()
    if args.stop_workers_on_exit:
      logging.info("Shutting down cluster workers...")
      cluster.shutdown()
    else:
      cluster.monitor.close()
      if cluster.trajectory_store is not None:
        cluster.trajectory_store.close()

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
