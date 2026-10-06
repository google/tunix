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

"""Multi-stage reinforcement learning programs.

Provides modular program abstractions for orchestrating  RL training
pipelines.
"""

import abc
import asyncio
from collections.abc import Callable, Iterable, Mapping, Sequence, Sized
import copy
import ctypes
import dataclasses
import gc
import os
import threading
import json
import time
from typing import Any

from absl import logging
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.common import logging_utils
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import batch_assembly
from tunix.experimental.orchestrator import rl_engine_interface
from tunix.experimental.queue_manager import trajectory_queue_manager
from tunix.experimental.trajectory import store as trajectory_store_lib
from tunix.rl import common as rl_common
from tunix.sft import metrics_logger as metrics_logger_lib
from tunix.utils import trajectory_logger

MetricsLogger = metrics_logger_lib.MetricsLogger
MetricsLoggerOptions = metrics_logger_lib.MetricsLoggerOptions
Mode = metrics_logger_lib.Mode
BatchConfig = batch_assembly.BatchConfig


_EXHAUSTED = object()


def _release_host_memory() -> None:
  """Forces Python GC and returns freed glibc heap pages to the OS."""
  gc.collect()
  try:
    libc = ctypes.CDLL("libc.so.6")
    libc.malloc_trim(0)
  except (OSError, AttributeError):
    pass


def _raised_by_worker_method(exc: BaseException) -> bool:
  """Whether a worker call failed in the worker's method, not on the way.

  An actor handle re-raises the method's exception as a RuntimeError whose
  message starts "RemoteExecutionError" (`ExecutionResponse.unwrap`); a
  transport failure (a gRPC status, a reset connection, a timeout) is raised
  as itself.

  Args:
    exc: What the call raised.
  """
  return isinstance(exc, RuntimeError) and str(exc).startswith(
      "RemoteExecutionError"
  )


def _next_or_exhausted(iterator: Any) -> Any:
  """`next()` that returns `_EXHAUSTED` instead of raising StopIteration.

  StopIteration cannot cross `asyncio.to_thread` (it becomes a TypeError).
  """
  try:
    return next(iterator)
  except StopIteration:
    return _EXHAUSTED


def _extract_scalar(val: Any, name: str | None = None) -> float | None:
  """Extracts a step scalar from a metric value, reducing multi-microbatch arrays by suffix."""
  if val is None:
    return None
  if hasattr(val, "compute") and callable(val.compute):
    try:
      val = val.compute()
    except Exception:  # pylint: disable=broad-exception-caught
      return None
  if hasattr(val, "unreduced_sum") and hasattr(val, "denominator"):
    return metrics_logger_lib.extract_scalar(val)
  try:
    arr = np.asarray(val, dtype=np.float64)
    if arr.size == 1:
      return float(arr.item())
    if arr.size > 1:
      finite = arr[np.isfinite(arr)]
      if finite.size == 0:
        return float(arr.flat[0])
      if name is not None and name.endswith(("_max", "/max")):
        return float(np.max(finite))
      if name is not None and name.endswith(("_min", "/min")):
        return float(np.min(finite))
      return float(np.mean(finite))
  except Exception:  # pylint: disable=broad-exception-caught
    pass
  return metrics_logger_lib.extract_scalar(val)

def _prompt_coordinates(prompt_idx: int, full_batch_size: int) -> dict[str, int]:
  """Returns the dataset coordinates of the prompt at `prompt_idx`.

  The coordinates are derived purely from dataset position, so they are stable
  across restarts and identical on retry. `batch_idx` is the prompt batch a
  trajectory belongs to, which is what a batch-ordered queue manager routes and
  orders on; `intra_batch_idx` identifies the prompt inside that batch.

  Args:
    prompt_idx: Position of the prompt in the dataset.
    full_batch_size: Prompts per prompt batch (groups per trainer iteration).
  """
  return {
      "prompt_idx": prompt_idx,
      "batch_idx": prompt_idx // full_batch_size,
      "intra_batch_idx": prompt_idx % full_batch_size,
  }


def _tag_prompt(prompt_item: Any, coordinates: Mapping[str, int]) -> Any:
  """Stamps `coordinates` onto a copy of `prompt_item`'s metadata.

  The dataset item itself is never mutated: dicts are copied, and other items
  are shallow-copied before their `metadata` is replaced. Coordinates win over
  any same-named key already present, because dataset position is the only
  authority on them.

  Args:
    prompt_item: A prompt dict, or any object the engine accepts.
    coordinates: The tags to stamp, from `_prompt_coordinates`.

  Returns:
    The tagged item, or `prompt_item` unchanged when its metadata cannot be
    replaced (logged, since a batch-ordered queue will later reject it).
  """
  if isinstance(prompt_item, Mapping):
    tagged = dict(prompt_item)
    tagged["metadata"] = {**(tagged.get("metadata") or {}), **coordinates}
    return tagged

  metadata = {**(getattr(prompt_item, "metadata", None) or {}), **coordinates}
  if dataclasses.is_dataclass(prompt_item) and any(
      field.name == "metadata" for field in dataclasses.fields(prompt_item)
  ):
    # Covers frozen dataclasses, whose attributes cannot be assigned.
    return dataclasses.replace(prompt_item, metadata=metadata)  # pytype: disable=wrong-arg-types

  tagged = copy.copy(prompt_item)
  try:
    tagged.metadata = metadata
  except (AttributeError, TypeError):
    logging.warning(
        "Cannot tag prompt coordinates onto a %s; batch-ordered consumption"
        " requires a `metadata` mapping on every prompt item.",
        type(prompt_item).__name__,
    )
    return prompt_item
  return tagged


def _generation_metrics(
    groups: Sequence[Sequence[Any]],
) -> dict[str, float]:
  """Computes the step's `rollout/completions/*` metrics, grouped by prompt.

  Matches `agentic_grpo_learner.GRPOLearner._process_results`: the two means are
  per group, not per rollout, so a short group still weighs the same as a full
  one. Max and min are unaffected by grouping.

  `clipped` / `raw_length` come from the collector, which derives them with
  `collector.response_budget_facts` and records them on
  `TrajectoryItem.metadata`, because only the producer knows the budget a
  rollout was held to -- `DistributedRLEngine` lets a dataset item override
  `max_response_length`. Raw length counts env and tool tokens, per the
  rLLM/VERL `response_length` convention.

  Args:
    groups: Trajectory items for the step, grouped by prompt.

  Returns:
    Fully-qualified metric name to step value; empty when no rollout carried
    the annotations.
  """
  clip_ratios: list[float] = []
  group_mean_lengths: list[float] = []
  all_lengths: list[int] = []
  for group in groups:
    clipped = 0
    lengths: list[int] = []
    for item in group:
      # Token mode yields a dict; Trajectory mode yields a dataclass, which
      # never carries these. Both keys are required: a rollout annotated with
      # only one of them is a producer bug, and dropping it keeps that bug from
      # taking down the training step over a metric.
      traj = item.traj
      if not isinstance(traj, dict):
        continue
      meta = getattr(item, "metadata", None)
      if not isinstance(meta, dict):
        continue
      if meta.get("clipped") is None or meta.get("raw_length") is None:
        continue
      # A zero-length response is a rollout that ran and produced nothing; it
      # stays in the denominator, matching the agentic learner.
      lengths.append(int(meta["raw_length"]))
      clipped += int(meta["clipped"])
    if not lengths:
      continue
    clip_ratios.append(clipped / len(lengths))
    group_mean_lengths.append(float(np.mean(lengths)))
    all_lengths.extend(lengths)

  if not all_lengths:
    return {}
  return {
      "rollout/completions/clip_ratio": float(np.mean(clip_ratios)),
      "rollout/completions/mean_raw_length": float(
          np.mean(group_mean_lengths)
      ),
      "rollout/completions/max_raw_length": float(np.max(all_lengths)),
      "rollout/completions/min_raw_length": float(np.min(all_lengths)),
  }


def _extract_reward(item: Any) -> float:
  """Extracts the scalar trajectory reward from a TrajectoryItem.

  The orchestrator consumes only "Token"-mode trajectories, which are plain
  dicts carrying `trajectory_reward` (see
  `tunix.rl.agentic.trajectory.trajectory_collect_engine`). Anything else is
  rejected rather than coerced.

  There is deliberately no default to avoid a silent failure.

  Args:
    item: A `TrajectoryItem`, or a raw trajectory mapping.

  Returns:
    The trajectory reward as a float.

  Raises:
    TypeError: If the trajectory is not a mapping.
    KeyError: If the mapping has no `trajectory_reward` entry.
  """
  traj = getattr(item, "traj", item)
  if not isinstance(traj, dict):
    raise TypeError(
        f"Expected a Token-mode trajectory mapping, got {type(traj).__name__}."
    )
  if "trajectory_reward" not in traj:
    raise KeyError(
        "Trajectory is missing the 'trajectory_reward' key; present"
        f" keys are {sorted(traj)}. Rollout backends and critique stage must"
        " stamp the trajectory reward under 'trajectory_reward'."
    )
  return float(traj["trajectory_reward"])


def _invoke_reward_fn(
    fn: Callable[[str, Mapping[str, Any]], float],
    item: datatypes.TrajectoryItem,
) -> float:
  """Scores `item`'s assistant completion string with `fn(completion, metadata)`."""
  if not isinstance(item.traj, dict):
    raise TypeError(
        "Expected a Token-mode trajectory mapping, got"
        f" {type(item.traj).__name__}."
    )
  completion = datatypes.assistant_text(item.traj.get("conversation_text", ""))
  return float(fn(completion, item.metadata))


def _format_rollout_completion(conversation: Any) -> str:
  """Renders post-prompt interaction (assistant + env) for trajectory logging.

  In single-turn tasks (e.g. GSM8K), this yields the assistant's completion.
  In multi-turn agentic environments (e.g. ToolAgent, DeepSWE), this preserves
  both the assistant's turns and subsequent environment/tool feedback while
  excluding the initial prompt.

  Args:
    conversation: Either a list of chat messages or an already-rendered string.

  Returns:
    The rendered completion string for trajectory logging.
  """
  if not isinstance(conversation, list):
    return str(conversation)

  # Find the end of the initial prompt (first user turn).
  prompt_end_idx = 0
  for idx, msg in enumerate(conversation):
    if isinstance(msg, dict) and msg.get("role") == "user":
      prompt_end_idx = idx + 1
      break

  post_prompt = conversation[prompt_end_idx:]
  if not post_prompt:
    return datatypes.assistant_text(conversation)

  # For a single assistant turn (e.g. single-turn math), return raw content directly.
  if len(post_prompt) == 1 and isinstance(post_prompt[0], dict):
    return str(post_prompt[0].get("content", ""))

  # For multi-turn interactions, include role labels so env observations and assistant actions are distinguishable.
  lines = []
  for msg in post_prompt:
    if not isinstance(msg, dict):
      continue
    role = msg.get("role", "unknown")
    content = str(msg.get("content", ""))
    label = "environment" if role == "user" else role
    lines.append(f"[{label}]: {content}")
  return "\n".join(lines)


@dataclasses.dataclass(kw_only=True)
class RLStepResult:
  """Summary of a completed RL training step."""

  step: int
  policy_version: int
  num_rollouts: int
  num_microbatches: int
  reward_mean: float
  reward_std: float
  advantage_mean: float = 0.0
  advantage_std: float = 0.0
  train_result: Any = None


@dataclasses.dataclass(eq=False)
class _StepState:
  """One step's intake and train-side counters, for `_train_loop`.

  The groups taken from `scored_q`, what they were packed into, and the
  per-step counters. With `pipeline_step_boundary`, a step's state can be
  filled while the previous step still trains (`_prepack`); the step adopts
  it when it starts.
  """

  uncommitted_groups: list[Any] = dataclasses.field(default_factory=list)
  all_step_items: list[Any] = dataclasses.field(default_factory=list)
  step_rewards: list[float] = dataclasses.field(default_factory=list)
  step_advantages: list[float] = dataclasses.field(default_factory=list)
  num_rollouts: int = 0
  groups_consumed: int = 0
  current_batch_idx: int | None = None
  # What the queue last returned; `[]` once it has no further group.
  scored_items: list[Any] = dataclasses.field(default_factory=list)
  packing_time_sec: float = 0.0
  exposed_generation_time: float = 0.0
  # Packed but not yet sent to the trainer, in packing order.
  ready: list[batch_assembly.AssembledBatch] = dataclasses.field(
      default_factory=list
  )
  num_microbatches: int = 0
  padding_stats: list[batch_assembly.PaddingStats] = dataclasses.field(
      default_factory=list
  )
  sampler_agreement: dict[str, tuple[Any, list[float]]] = dataclasses.field(
      default_factory=dict
  )
  step_begun: bool = False
  start_time: float | None = None
  # The first micro-batch, already on the trainer host (`stage_train_step`):
  # the micro-batch without its payload, and the id that runs it.
  staged: tuple[batch_assembly.AssembledBatch, str] | None = None


class RLProgram(abc.ABC):
  """Base class for multi-stage DAG workflows."""

  def __init__(self):
    self._is_running = False
    self._step = 0
    self.policy_version = 0
    self.last_step_result: RLStepResult | None = None
    self.engine: rl_engine_interface.AbstractRLEngine | None = None

  @property
  def step(self) -> int:
    return self._step

  @abc.abstractmethod
  def run(
      self,
      engine: rl_engine_interface.AbstractRLEngine,
      **kwargs: Any,
  ) -> None:
    """Entry point running all stages on an event loop."""
    raise NotImplementedError("Subclasses must implement run.")

  def close(self) -> None:
    """Closes and releases program resources."""
    pass


class StandardRLProgram(RLProgram):
  """Standard RL program handling common multi-stage training workflows asynchronously.

  Runs 4 concurrent stages:
  1. Rollout dispatch stage: Fire-and-forget requests across worker pool.
  2. Polling stage: Long-polls completed rollout responses into grouping queue.
  3. Critique stage: Scores rewards, PRMs, and reference KL logprobs.
  4. Train stage: Streaming gradient accumulation over microbatches.
  """

  def __init__(
      self,
      algo: algorithm_adapter.AlgorithmAdapter,
      dataset: Iterable[Any] | None = None,
      max_steps: int | None = None,
      reward_fns: Sequence[Callable[..., Any]] | None = None,
      assembler: batch_assembly.BatchAssembler | None = None,
      batch_config: batch_assembly.BatchConfig | None = None,
      generation_args: datatypes.GenerationArgs | None = None,
      batch_size: int | None = None,
      max_staleness: int = 0,
      sync_weights: bool = True,
      async_weight_sync: bool = False,
      partial_rollout: bool = False,
      metrics_logging_options: MetricsLoggerOptions | None = None,
      trajectory_log_dir: str | None = None,
      trajectory_store: (
          trajectory_store_lib.TrajectoryStore
          | Sequence[trajectory_store_lib.TrajectoryStore | None]
          | None
      ) = None,
      metrics_prefix: str = "",
      mode: Mode | str = Mode.TRAIN,
      group_order: (
          trajectory_queue_manager.GroupOrder | str
      ) = trajectory_queue_manager.GroupOrder.ARRIVAL,
      on_train_start: Callable[[int], None] | None = None,
      on_step_begin: Callable[[int], None] | None = None,
      on_step_end: Callable[[int, Any], None] | None = None,
      val_start_step: int | None = None,
      on_checkpoint_saved: Callable[[dict[str, Any]], None] | None = None,
      checkpoint_optimizer_interval_steps: int = 1,
      pipeline_train_microbatches: bool = False,
      rollout_priority_scheduling: bool = False,
      initial_dispatch_stagger_s: float = 0.0,
      initial_dispatch_first_stagger_s: float | None = None,
      trainer_warmup_payload: datatypes.RLTrainerPayload | None = None,
      num_generations_to_dispatch: int | None = None,
      prefer_valid_rollouts: bool = True,
      pipeline_step_boundary: bool = False,
      stage_next_microbatch: bool = False,
      stage_before_update: bool = True,
  ):
    super().__init__()
    self.engine: rl_engine_interface.AbstractRLEngine | None = None
    if max_staleness < 0:
      raise ValueError("max_staleness must be non-negative.")
    if initial_dispatch_stagger_s < 0:
      raise ValueError("initial_dispatch_stagger_s must be non-negative.")
    if (
        initial_dispatch_first_stagger_s is not None
        and initial_dispatch_first_stagger_s < 0
    ):
      raise ValueError("initial_dispatch_first_stagger_s must be non-negative.")
    if stage_next_microbatch and not pipeline_step_boundary:
      raise ValueError(
          "stage_next_microbatch requires pipeline_step_boundary: the"
          " micro-batch it stages is the one the pipelined boundary takes"
          " ahead."
      )
    if checkpoint_optimizer_interval_steps < 0:
      raise ValueError(
          "checkpoint_optimizer_interval_steps must be non-negative."
      )
    self.dataset = dataset
    self.max_steps = max_steps
    self.algo = algo
    algo_config = getattr(self.algo, "algo_config", None)
    algo_max_response_length = getattr(self.algo, "max_response_length", None)
    if algo_max_response_length is None and algo_config is not None:
      algo_max_response_length = getattr(
          algo_config, "max_response_length", None
      )
    self.max_response_length = algo_max_response_length
    self.generation_args = generation_args or datatypes.GenerationArgs()

    gen_temp = self.generation_args.temperature
    if gen_temp is not None:
      self.algo.algo_config.temperature = gen_temp

    # The sampler's per-token log-probabilities are the PPO ratio's
    # denominator, and separately the sampler side of every sequence-level gate
    # and sampler-vs-trainer diagnostic. A recipe may recompute the denominator
    # on the trainer and still want those, so ask the rollout for the
    # log-probabilities when any of them does.
    self.generation_args = dataclasses.replace(
        self.generation_args,
        return_logprobs=(
            getattr(algo_config, "use_rollout_logps", True)
            or getattr(algo_config, "truncated_importance_sampling_type", None)
            is not None
            or getattr(algo_config, "seq_logprob_error_threshold", None)
            is not None
            or getattr(algo_config, "sampler_is_length_buckets", None)
            is not None
        ),
    )
    self.sampler_is = getattr(self.algo.algo_config, "sampler_is", None)
    self.sampler_is_threshold = getattr(
        self.algo.algo_config, "sampler_is_threshold", 2.0
    )
    raw_seq_err_thresh = getattr(
        self.algo.algo_config, "seq_logprob_error_threshold", None
    )
    self.seq_logprob_error_threshold = (
        float(raw_seq_err_thresh)
        if isinstance(raw_seq_err_thresh, (int, float))
        else None
    )

    self.reward_fns = list(reward_fns) if reward_fns else []
    self.num_generations = algo.num_generations
    self.mini_batch_size = algo.mini_batch_size
    if self.mini_batch_size <= 0 or self.num_generations <= 0:
      raise ValueError("mini_batch_size and num_generations must be positive.")
    if (
        num_generations_to_dispatch is not None
        and num_generations_to_dispatch < self.num_generations
    ):
      raise ValueError(
          f"num_generations_to_dispatch ({num_generations_to_dispatch}) must be"
          f" >= num_generations ({self.num_generations})."
      )
    self.num_generations_to_dispatch = num_generations_to_dispatch
    self.prefer_valid_rollouts = prefer_valid_rollouts
    self._effective_generations_to_dispatch = (
        self.num_generations_to_dispatch
        if self.num_generations_to_dispatch is not None
        else self.num_generations
    )
    self.full_batch_size = (
        self.mini_batch_size if batch_size is None else batch_size
    )
    # Keep batch_size as a public alias for callers that use recipe naming.
    self.batch_size = self.full_batch_size
    if self.full_batch_size <= 0:
      raise ValueError("batch_size must be positive.")
    if self.full_batch_size % self.mini_batch_size != 0:
      raise ValueError(
          "batch_size must be divisible by mini_batch_size; got "
          f"batch_size={self.full_batch_size}, "
          f"mini_batch_size={self.mini_batch_size}."
      )
    use_rollout_logps = getattr(
        algo,
        "use_rollout_logps",
        getattr(getattr(algo, "algo_config", None), "use_rollout_logps", True),
    )
    if not use_rollout_logps and self.full_batch_size != self.mini_batch_size:
      raise ValueError(
          "use_rollout_logps=False requires batch_size == mini_batch_size "
          "(exactly one optimizer update per rollout batch); got "
          f"batch_size={self.full_batch_size}, "
          f"mini_batch_size={self.mini_batch_size}."
      )
    self.batch_config = batch_config or batch_assembly.BatchConfig()
    if self.batch_config.max_response_length is None:
      self.batch_config = dataclasses.replace(
          self.batch_config,
          max_response_length=self.max_response_length,
      )
    if assembler is not None:
      self.assembler = assembler
      self.assembler.num_generations = self.num_generations
      self.assembler.mini_batch_size = self.mini_batch_size
    else:
      self.assembler = batch_assembly.create_batch_assembler(
          num_generations=self.num_generations,
          mini_batch_size=self.mini_batch_size,
          train_micro_batch_size=getattr(algo, "train_micro_batch_size", 1),
          batch_config=self.batch_config,
      )
    self.max_staleness = max_staleness
    self.sync_weights = sync_weights
    # Packs the next microbatch (in a worker thread) while the previous one's
    # train_step RPC is in flight, so orchestrator-side packing overlaps
    # trainer compute. At most one train_step is ever outstanding, so the
    # gradient-accumulation order is unchanged.
    self.pipeline_train_microbatches = pipeline_train_microbatches
    self._pending_train: asyncio.Task[tuple[Any, float]] | None = None
    # Pipelines the step boundary (see `_train_loop`): step k+1's groups are
    # taken and packed while step k's last micro-batches train, step k+1's
    # first micro-batch goes to the trainer as soon as step k's weight sync
    # has staged, and step k's metrics, logging and host-memory release run
    # once that call has returned, while the device computes it. The trainer
    # gets the same calls, with the same inputs, in the same order as without
    # it. What moves with that bookkeeping: step k's `Train step` line,
    # StepMetrics and `on_step_end` follow a `[pipeline] SENT_AHEAD step=k` line
    # (step k trained, step k+1 started), so a wedge in step k+1's first call
    # stops the log there; `on_step_begin(k+1)` fires after step k+1's first
    # train call returned; and under ARRIVAL order the dispatch window, keyed on
    # the step count, opens that one call later (PROMPT_BATCH opens it at the
    # commit).
    self.pipeline_step_boundary = pipeline_step_boundary
    # With `pipeline_step_boundary`: also send that micro-batch's data to the
    # trainer host while the device still computes step k's last micro-batch,
    # so the train call that later runs it carries no data. Needs an engine
    # with `stage_train_step` and a trainer that supports it
    # (`trainer_staging_support`); otherwise the boundary runs unstaged.
    self.stage_next_microbatch = stage_next_microbatch
    # Where that stage goes. With `stage_before_update` and a trainer that can
    # defer a step's update (`stage_before_update` in its staging support):
    # between step k's last train call, sent with its update deferred, and
    # that update (`_stage_next_before_update`). The last call then returns
    # once the trainer has dispatched the last micro-batch, which under
    # MaxText's `retry_when_tokens_dropped` is when the micro-batch before it
    # finished, so the transfer runs while the device computes the last one.
    # Otherwise right after the last call, which runs the update and returns
    # at its dispatch (`_maybe_stage_next_microbatch`); under
    # `retry_when_tokens_dropped` that dispatch waits until the last
    # micro-batch has finished, so the transfer finds the device idle and hides
    # nothing, which the latch below catches at the first boundary except on
    # Pathways (see `_STAGING_HIDDEN_WAIT_S`). Only matters with
    # `stage_next_microbatch`.
    self.stage_before_update = stage_before_update
    # Takes groups for a step ahead of it (`_prepack`), filling a `_StepState`
    # the step adopts when it starts.
    self._intake_task: asyncio.Task[None] | None = None
    # Set once a stage is seen not to hide behind the trainer's device (its
    # device had gone idle before the transfer finished; see
    # `_STAGING_HIDDEN_WAIT_S` for where this can be seen): the trainer is then
    # fed slower than it computes, and staging only delays each step's sync.
    # Also set when the trainer cannot stage at all.
    self._staging_latched_off = False
    # What the trainer said it supports of staging; asked once, lazily.
    self._staging_supported: bool | None = None
    self._staging_support: dict[str, bool] | None = None
    # Whether it can also defer a step's update past its last micro-batch, so
    # the stage goes in between (`_trainer_can_stage_before_update`).
    self._stage_before_update_supported: bool | None = None
    # The optimizer step the trainer last reported (`train_step` of an update
    # call), from which `_stage_planned` predicts the next one.
    self._last_optimizer_step: int | None = None
    # A cancelled asyncio.to_thread leaves its worker running, so the error
    # path's reset() must wait for any in-flight feed/flush to finish.
    self._assembler_lock = threading.Lock()

    # Run each step's weight sync in the background instead of holding the
    # trainer for the whole round. The trainer waits only for the source to
    # snapshot its weights; the dispatch window stays shut until the round
    # commits, so rollouts never run further behind than `max_staleness`.
    # Requires a source that transfers from a host snapshot (raiden:
    # RAIDEN_FFI_USE_DIRECT_DEVICE_BUFFER=0), since the next train step
    # overwrites the device buffers while the transfer is still reading.
    self.async_weight_sync = async_weight_sync
    self._pending_sync: asyncio.Task[None] | None = None
    # Trained steps whose weights have not reached the rollouts yet. Held out
    # of the dispatch window; stays set if the round fails.
    self._unsynced_steps = 0
    # Set from the optimizer step until that step's round starts. The trainer
    # can spend minutes in between (checkpoint save), and the previous round
    # committing in that gap must not reopen the window: anything dispatched
    # then is drained on the old weights before the next transfer.
    self._round_due = False
    # Rollouts pause and resume in-flight requests across a sync instead of
    # draining them, so dispatch need not wait out a round; each batch is
    # gated only on the version the rollouts have committed.
    self.partial_rollout = partial_rollout
    # Stamp each rollout request with its prompt batch index as its sampler
    # priority, so rollouts that schedule by priority (vLLM
    # `scheduling_policy="priority"`) serve the oldest in-flight batch first
    # under `max_staleness > 0`. Off, every request gets priority 0 and the
    # rollouts serve requests in arrival order.
    self.rollout_priority_scheduling = rollout_priority_scheduling
    # Spread the start-up prompt batches out in time: batch 1 is held until
    # `initial_dispatch_first_stagger_s` (defaults to
    # `initial_dispatch_stagger_s`) after batch 0 first dispatches, and each
    # later batch `initial_dispatch_stagger_s` after the one before it. Applies
    # only before the trainer consumes its first batch; both 0 disables it.
    self.initial_dispatch_stagger_s = float(initial_dispatch_stagger_s)
    self.initial_dispatch_first_stagger_s = (
        self.initial_dispatch_stagger_s
        if initial_dispatch_first_stagger_s is None
        else float(initial_dispatch_first_stagger_s)
    )
    # One synthetic micro-batch to compile the trainer against before
    # `on_train_start`, after `configure_worker` (which discards any compile
    # done at bring-up). None keeps the lazy compile on the first micro-batch.
    self.trainer_warmup_payload = trainer_warmup_payload
    self._first_train_payload_logged = False
    self._initial_dispatch_t0: float | None = None
    self._initial_stagger_logged: set[int] = set()
    self._sync_error: BaseException | None = None
    self.metrics_logger: MetricsLogger = MetricsLogger(metrics_logging_options)
    # Trajectory logging is disabled on mlperf to prevent GCS write timeouts
    # and queue stalls (b/565475624). Rollout data is handled by TrajectoryStore.
    self.trajectory_logger = None
    logging.info("Trajectory logging disabled in mlperf branch.")
    # Received, not built: the orchestrator running this program owns the
    # Trajectory Store's construction and lifecycle (ClusterOrchestrator, one
    # per process), since a store's lifetime should span the whole
    # orchestrator process rather than just one program run. This program
    # only uses it; close() below does not close it.
    # TODO(sizhi): Wire active trajectory store reads/writes in pipeline stages
    # in follow-up CLs.
    self._trajectory_store = trajectory_store
    self.metrics_prefix = metrics_prefix
    self.mode = mode if isinstance(mode, Mode) else Mode(mode)
    self.group_order = (
        group_order
        if isinstance(group_order, trajectory_queue_manager.GroupOrder)
        else trajectory_queue_manager.GroupOrder(group_order)
    )
    self.on_train_start = on_train_start
    self.on_step_begin = on_step_begin
    self.on_step_end = on_step_end
    self.val_start_step = val_start_step
    self.on_checkpoint_saved = on_checkpoint_saved
    # Checkpoints carry the optimizer state only every this many optimizer
    # steps, and on the last step; the rest hold just the model params, which
    # is all eval reads. The optimizer state can be several times the params'
    # size, too much to write every step. 0 never writes it, not even on the
    # last step, so such a run cannot be resumed.
    self.checkpoint_optimizer_interval_steps = (
        checkpoint_optimizer_interval_steps
    )
    self.last_step_timestamp_ms: int | None = None
    self._in_flight_rollouts = 0
    self._pending_cancel_tasks: set[asyncio.Task[int]] = set()
    self._window_release = asyncio.Event()
    self._dispatch_done = asyncio.Event()

    self.scored_q: (
        trajectory_queue_manager.TrajectoryQueueManager
        | trajectory_queue_manager.BatchOrderedQueueManager
    )
    on_drop: Callable[[int, int], None] | None = None
    if self.group_order == trajectory_queue_manager.GroupOrder.PROMPT_BATCH:
      ordered_q = trajectory_queue_manager.BatchOrderedQueueManager.create(
          num_generations=self.num_generations,
          full_batch_size=self.full_batch_size,
          max_staleness=max_staleness,
          current_policy_version=lambda: self.policy_version,
          on_cursor_advance=lambda _: self._release_window(),
      )
      self.scored_q = ordered_q
      on_drop = ordered_q.dropped
    else:
      self.scored_q = trajectory_queue_manager.TrajectoryQueueManager.create(
          num_generations=self.num_generations,
          max_staleness=max_staleness,
          current_policy_version=lambda: self.policy_version,
      )
    self.raw_q = trajectory_queue_manager.TrajectoryQueueManager.create(
        num_generations=self.num_generations,
        num_generations_to_dispatch=self._effective_generations_to_dispatch,
        prefer_valid_rollouts=self.prefer_valid_rollouts,
        max_staleness=max_staleness,
        current_policy_version=lambda: self.policy_version,
        on_drop=on_drop,
    )

  @property
  def trajectory_store(
      self,
  ) -> (
      trajectory_store_lib.TrajectoryStore
      | Sequence[trajectory_store_lib.TrajectoryStore | None]
      | None
  ):
    return self._trajectory_store

  def close(self) -> None:
    """Flushes and closes the metrics logger and associated resources.

    Does not close `self._trajectory_store`: this program does not own it
    (see `__init__`), and closing a store the orchestrator may still be
    using — e.g. across a second `run_program()` call — would be wrong.
    """
    if self.trajectory_logger is not None:
      self.trajectory_logger.stop()
    if self.metrics_logger is not None:
      self.metrics_logger.close()

  @property
  def _next_batch(self) -> int:
    """Index of the next prompt batch the trainer will consume.

    The window's lower edge. Under `ARRIVAL` ordering, batches are cut by
    arrival order so the count of batches trained (`_step`) is also the index
    of the next one. Under `PROMPT_BATCH` ordering, `scored_q.next_batch_idx`
    is the cursor, which runs ahead of `_step` by the number of empty batches
    skipped.
    """
    if isinstance(
        self.scored_q, trajectory_queue_manager.BatchOrderedQueueManager
    ):
      return self.scored_q.next_batch_idx
    return self._step

  def _release_window(self) -> None:
    """Wakes the dispatcher after `_next_batch` advances."""
    self._window_release.set()

  async def _wait_for_dispatch_window(self, batch_idx: int) -> None:
    """Blocks until `batch_idx` is inside the policy-staleness window.

    The dispatcher may run `max_staleness` prompt batches past the batch the
    trainer is about to consume, and no further (R1).

    Args:
      batch_idx: The prompt batch the dispatcher is about to emit into.
    """
    # Checked up front too: a batch already inside the window would otherwise
    # skip the loop and keep dispatching after a sync round failed.
    if self._sync_error is not None:
      raise self._sync_error
    await self._hold_initial_stagger(batch_idx)
    if self.partial_rollout:
      await self._wait_for_partial_dispatch_window(batch_idx)
      return
    # Nothing goes out while a background round is in flight. The rollouts
    # cannot start a request until the round commits anyway, and one that
    # lands before they close admission is drained on the old weights,
    # idling every other worker until it finishes.
    while (
        self._unsynced_steps
        or self._round_due
        or batch_idx
        > self._next_batch - self._unsynced_steps + self.max_staleness
    ):
      if self._sync_error is not None:
        raise self._sync_error
      # Wait first, clear second. The reverse loses a release that lands
      # between the test above and the clear, and nothing would set it again.
      await self._window_release.wait()
      self._window_release.clear()

  def _initial_dispatch_offset(self, batch_idx: int) -> float:
    """Seconds after batch 0's first dispatch at which start-up batch is due.

    Batch 0 is due at once, batch 1 after `initial_dispatch_first_stagger_s`,
    and batch k > 1 after `first + (k - 1) * initial_dispatch_stagger_s`.

    Args:
      batch_idx: The prompt batch index.

    Returns:
      The offset in seconds.
    """
    if batch_idx <= 0:
      return 0.0
    return (
        self.initial_dispatch_first_stagger_s
        + (batch_idx - 1) * self.initial_dispatch_stagger_s
    )

  async def _hold_initial_stagger(self, batch_idx: int) -> None:
    """Holds start-up batch `batch_idx` until its staggered dispatch time.

    Batch k (k > 0) is held until `_initial_dispatch_offset(k)` after batch 0
    first dispatched. No-op when both staggers are 0, and once the trainer has
    consumed a batch (`_next_batch > 0`).

    Args:
      batch_idx: The prompt batch the dispatcher is about to emit into.
    """
    if (
        self.initial_dispatch_stagger_s <= 0
        and self.initial_dispatch_first_stagger_s <= 0
    ) or self._next_batch > 0:
      return
    if batch_idx == 0:
      if self._initial_dispatch_t0 is None:
        self._initial_dispatch_t0 = time.monotonic()
      return
    t0 = self._initial_dispatch_t0
    if t0 is None:
      return
    due = t0 + self._initial_dispatch_offset(batch_idx)
    now = time.monotonic()
    if now >= due:
      return
    if batch_idx not in self._initial_stagger_logged:
      self._initial_stagger_logged.add(batch_idx)
      logging.info(
          "[pipeline] STAGGER batch_idx=%d held %.1f s"
          " (initial_dispatch_first_stagger_s=%.1f"
          " initial_dispatch_stagger_s=%.1f)",
          batch_idx,
          due - now,
          self.initial_dispatch_first_stagger_s,
          self.initial_dispatch_stagger_s,
      )
    await asyncio.sleep(due - now)
    if self._sync_error is not None:
      raise self._sync_error

  async def _wait_for_partial_dispatch_window(self, batch_idx: int) -> None:
    """Per-batch dispatch gate for partial rollout.

    A paused request resumes on the new weights rather than draining on the
    old ones, so a round in flight is no reason to hold dispatch. Batch
    `batch_idx` goes out once the rollouts have committed version
    `batch_idx - max_staleness`: its trajectories are then stamped no older
    than that, which the staleness filter keeps until step `batch_idx`
    consumes them.

    Args:
      batch_idx: The prompt batch the dispatcher is about to emit into.
    """
    while (
        batch_idx > self._next_batch + self.max_staleness
        or batch_idx - self.max_staleness
        > self.policy_version - self._unsynced_steps
    ):
      if self._sync_error is not None:
        raise self._sync_error
      await self._window_release.wait()
      self._window_release.clear()

  async def _resume_from_checkpoint(self) -> None:
    """Realigns program orchestration state with the engine's restored checkpoint.

    Delegates the mesh-level work (restoring the trainer checkpoint and, when
    `sync_weights` is enabled, resyncing rollout worker weights to the restored
    policy) to the engine, then translates the restored step into program
    orchestration state: the train-loop bound (`_step`) and the dataset prefix
    to skip (resumed `_step` if any).
    """
    assert self.engine is not None
    restored = await self.engine.resume_from_checkpoint(
        role=datatypes.Role.ACTOR,
        resync_rollout_weights=self.sync_weights,
    )
    if isinstance(restored, tuple):
      restored_step, restored_next_batch_idx = restored
    else:
      restored_step = restored
      engine_next_batch = getattr(self.engine, "restored_next_batch_idx", None)
      restored_next_batch_idx = (
          engine_next_batch
          if isinstance(engine_next_batch, int)
          and not isinstance(engine_next_batch, bool)
          else restored_step
      )
    if restored_step <= 0:
      return
    restored_next_batch_idx = max(restored_step, restored_next_batch_idx)
    self._step = restored_step
    self.policy_version = restored_step
    if isinstance(
        self.scored_q, trajectory_queue_manager.BatchOrderedQueueManager
    ):
      self.scored_q.skip(0, restored_next_batch_idx)
    logging.info(
        "Resuming from checkpoint: step=%d policy_version=%d (skipping %d"
        " already-trained dataset items).",
        restored_step,
        self.policy_version,
        self._next_batch * self.full_batch_size,
    )

  async def rollout_dispatch_stage(self) -> None:
    """Stage 1A: Dispatches rollout requests across workers asynchronously.

    Ensures that all dataset items carry unique, collision-free `prompt_id`s
    (e.g., `f"prompt_{prompt_idx}"`) before dispatching to the engine layer,
    satisfying the engine's strict `prompt_id` contract.

    Also stamps each prompt's dataset coordinates (`prompt_idx`, `batch_idx`,
    `intra_batch_idx`) into its metadata, where they ride along untouched to the
    queue managers. Consumers that order or account per prompt batch read them;
    nothing upstream of the dispatcher knows dataset position.
    """
    assert self.engine is not None
    if self.dataset is None:
      raise ValueError(
          "StandardRLProgram requires a dataset either at init or in run()."
      )
    already_consumed = self._next_batch * self.full_batch_size
    last_coordinates: dict[str, int] | None = None
    dataset_iter = iter(self.dataset)
    dataset_len = len(self.dataset) if isinstance(self.dataset, Sized) else None

    try:
      prompt_idx = 0
      while True:
        if prompt_idx < already_consumed:
          try:
            next(dataset_iter)
          except StopIteration:
            break
          prompt_idx += 1
          continue

        if (dataset_len is not None and prompt_idx >= dataset_len) or (
            hasattr(dataset_iter, "has_next") and not dataset_iter.has_next()
        ):
          try:
            next(dataset_iter)
          except StopIteration:
            break

        coordinates = _prompt_coordinates(prompt_idx, self.full_batch_size)
        await self._wait_for_dispatch_window(coordinates["batch_idx"])
        # Off the event loop: dataset iterators such as the DeepSWE
        # PrewarmDatasetIterator make blocking K8s calls here, which would
        # otherwise stall the train loop and the polling stage.
        prompt_item = await asyncio.to_thread(_next_or_exhausted, dataset_iter)
        if prompt_item is _EXHAUSTED:
          break
        # Again: the fetch yielded, and the train loop may have started a
        # round meanwhile. Nothing below awaits before the dispatch, so the
        # window cannot shut between this check and the request going out.
        await self._wait_for_dispatch_window(coordinates["batch_idx"])
        prompt_idx += 1
        last_coordinates = coordinates
        if isinstance(prompt_item, dict):
          prompt_item = dict(prompt_item)
          prompt_item.setdefault(
              "prompt_id", f"prompt_{coordinates['prompt_idx']}"
          )
          if self.max_response_length is not None:
            prompt_item.setdefault(
                "max_response_length", self.max_response_length
            )
        elif not hasattr(prompt_item, "prompt_id"):
          prompt_item = {
              "prompt": prompt_item,
              "prompt_id": f"prompt_{coordinates['prompt_idx']}",
          }
          if self.max_response_length is not None:
            prompt_item["max_response_length"] = self.max_response_length

        prompt_item = _tag_prompt(prompt_item, coordinates)
        # What the rollouts actually hold. While a background sync is in
        # flight the trainer is already a version ahead of them.
        rollout_policy_version = self.policy_version - self._unsynced_steps
        logging.info(
            "[pipeline] DISPATCH prompt_id=%s prompt_idx=%d batch_idx=%d"
            " intra_batch_idx=%d policy_version=%d next_batch=%d",
            prompt_item.get("prompt_id")
            if isinstance(prompt_item, dict)
            else getattr(prompt_item, "prompt_id", ""),
            coordinates["prompt_idx"],
            coordinates["batch_idx"],
            coordinates["intra_batch_idx"],
            rollout_policy_version,
            self._next_batch,
        )
        self._in_flight_rollouts += self._effective_generations_to_dispatch
        dispatch_kwargs: dict[str, Any] = {
            "num_generations": self._effective_generations_to_dispatch,
            "policy_version": rollout_policy_version,
            # Lower is served first, so with `max_staleness > 0` the oldest
            # in-flight batch (the window's lower edge) keeps its sampler
            # slots, including for its later turns, which re-enter the
            # sampler queue behind newer batches' first turns. Under
            # PROMPT_BATCH ordering this is exactly the batch the trainer
            # consumes next; under ARRIVAL ordering it drains the oldest
            # prompts first, keeping consumed staleness near the bound.
            "priority": (
                coordinates["batch_idx"]
                if self.rollout_priority_scheduling
                else 0
            ),
            "exact_token_continuity": getattr(
                self.algo.algo_config, "exact_token_continuity", True
            ),
        }
        if self.generation_args is not None:
          dispatch_kwargs["generation_args"] = self.generation_args
        try:
          await self.engine.dispatch_rollouts(
              [prompt_item],
              **dispatch_kwargs,
          )
        except Exception:
          self._in_flight_rollouts = max(
              0,
              self._in_flight_rollouts - self._effective_generations_to_dispatch,
          )
          raise
      if last_coordinates is not None and isinstance(
          self.scored_q, trajectory_queue_manager.BatchOrderedQueueManager
      ):
        self.scored_q.expect(
            last_coordinates["batch_idx"],
            last_coordinates["intra_batch_idx"] + 1,
        )
    finally:
      self._dispatch_done.set()

  async def polling_stage(self) -> None:
    """Stage 1B: Long-polls completed worker rollout responses into the queue."""
    assert self.engine is not None
    try:
      while not self._dispatch_done.is_set() or self._in_flight_rollouts > 0:
        if self._in_flight_rollouts <= 0:
          await asyncio.sleep(0.001)
          continue

        try:
          completed = await self.engine.poll_rollouts()
          if isinstance(completed, list) and completed:
            # TODO: Fault-tolerance must either decrement `_in_flight_rollouts` for failed
            # requests or retry them internally. Otherwise, a dropped RPC will cause
            # `_in_flight_rollouts` to never reach 0, hanging the EOF cascade.
            self._in_flight_rollouts -= len(completed)
            for item in completed:
              group_completed = await self.raw_q.put(item)
              if (
                  group_completed
                  and self._effective_generations_to_dispatch
                  > self.num_generations
                  and item.prompt_id
              ):
                cancel_task = asyncio.create_task(
                    self.engine.cancel_rollouts(str(item.prompt_id))
                )
                self._pending_cancel_tasks.add(cancel_task)
                cancel_task.add_done_callback(
                    self._pending_cancel_tasks.discard
                )
        except datatypes.NoHealthyRolloutWorkersError:
          raise
        except Exception as exc:  # pylint: disable=broad-exception-caught
          logging.warning("Error in polling_stage: %s", exc)
          await asyncio.sleep(0.01)
    finally:
      if self._pending_cancel_tasks:
        await asyncio.gather(
            *list(self._pending_cancel_tasks), return_exceptions=True
        )
      # NB: We currently assume it's safe to silently drop partial groups upon EOF.
      await self.raw_q.close()

  async def critique_stage(self) -> None:
    """Stage 2: Scores rewards, PRMs, and reference KL logprobs."""
    assert self.engine is not None
    try:
      while True:
        try:
          group = await self.raw_q.get_group()
          if not group:
            break
        except Exception:
          break

        rewards = []
        for item in group:
          # Skip reward evaluation or extraction for failed, env-timed-out, or
          # masked-out trajectories (`not item.is_valid`): although the payload
          # still goes through trainer fwd/bwd to keep static batch shapes, its
          # advantage and completion_mask are zeroed out, so scoring it is
          # wasted work (and an aborted trajectory may lack trajectory_reward).
          # Episode-budget timeouts are valid and arrive with reward 0.
          if not item.is_valid:
            r = 0.0
          elif self.reward_fns:
            r = sum(_invoke_reward_fn(fn, item) for fn in self.reward_fns)
          else:
            r = _extract_reward(item)
          rewards.append(float(r))

        trainer_payloads = self.algo.create_trainer_payloads(
            group, rewards=rewards
        )
        for idx, payload in enumerate(trainer_payloads):
          reward_val = rewards[idx] if idx < len(rewards) else 0.0
          src_item = group[idx] if idx < len(group) else None
          src_traj = getattr(src_item, "traj", None)
          if isinstance(src_traj, dict):
            raw_status = src_traj.get("status") or getattr(
                src_item, "status", None
            )
            raw_steps = src_traj.get("steps")
            traj_dict = dict(src_traj)
          elif src_traj is not None:
            raw_status = getattr(src_traj, "status", None) or getattr(
                src_item, "status", None
            )
            raw_steps = getattr(src_traj, "steps", None)
            traj_dict = (
                src_traj.to_dict() if hasattr(src_traj, "to_dict") else {}
            )
          else:
            raw_status = getattr(src_item, "status", None)
            raw_steps = []
            traj_dict = {}
          if isinstance(raw_status, datatypes.TrajectoryStatus):
            status = raw_status
          elif isinstance(raw_status, str):
            status = getattr(
                datatypes.TrajectoryStatus,
                raw_status.upper(),
                datatypes.TrajectoryStatus.SUCCEEDED
                if raw_status.upper() in ("COMPLETED", "SUCCESS")
                else datatypes.TrajectoryStatus.RUNNING,
            )
          else:
            status = datatypes.TrajectoryStatus.RUNNING
          steps = raw_steps if isinstance(raw_steps, list) else []
          src_metadata = getattr(src_item, "metadata", None)
          if isinstance(src_metadata, dict):
            src_metadata.pop("routed_experts", None)
          metadata = dict(src_metadata) if src_metadata else {}
          metadata.pop("routed_experts", None)
          traj_dict["trajectory_reward"] = reward_val
          traj_dict["status"] = status
          traj_dict["steps"] = steps
          traj_dict["conversation_masks"] = payload.completion_mask
          # `payload.routed_experts` now holds the sequence-aligned routing
          # array; drop the raw rollout reference from `traj_dict`, `metadata`,
          # and the post-critique `TrajectoryItem` so we do not retain two
          # copies per trajectory across `scored_q` and `all_step_items`.
          traj_dict.pop("routed_experts", None)
          if isinstance(src_traj, dict):
            src_traj.pop("routed_experts", None)
          if src_item is not None:
            src_item.routed_experts = None
          item = datatypes.TrajectoryItem(
              prompt_id=getattr(src_item, "prompt_id", ""),
              group_index=getattr(src_item, "group_index", 0),
              start_step=0,
              traj=traj_dict,
              is_valid=getattr(src_item, "is_valid", True),
              prompt_tokens=getattr(src_item, "prompt_tokens", None),
              completion_tokens=getattr(src_item, "completion_tokens", None),
              action_mask=getattr(src_item, "action_mask", None),
              routed_experts=None,
              policy_version=getattr(src_item, "policy_version", 0),
              metadata=metadata,
              # TODO: b/552087289 - Stream RLTrainerPayload directly instead of
              # re-wrapping in TrajectoryItem.
          )
          item.payload = payload  # pyrefly: ignore[missing-attribute]
          await self.scored_q.put(item)
        if group:
          first_meta = getattr(group[0], "metadata", None) or {}
          logging.info(
              "[pipeline] ARRIVE prompt_id=%s prompt_idx=%s batch_idx=%s"
              " intra_batch_idx=%s policy_version=%d worker_id=%s",
              getattr(group[0], "prompt_id", ""),
              first_meta.get("prompt_idx"),
              first_meta.get("batch_idx"),
              first_meta.get("intra_batch_idx"),
              getattr(group[0], "policy_version", 0),
              first_meta.get("worker_id"),
          )
        group = None
        trainer_payloads = None
        src_item = None
        payload = None
        item = None
    finally:
      await self.scored_q.close()

  _SAMPLER_IS_ATTR_PREFIXES = (
      "sampler_is/hist_frac/",
      "sampler_is/hist_mass/",
      "sampler_is/pos_signed/",
      "sampler_is/pos_absmean/",
      "sampler_is/conf_signed/",
      "sampler_is/conf_frac/",
      "sampler_is/lenscale/",
  )

  def _log_metric(
      self,
      metric_name: str,
      scalar_value: float | np.ndarray,
      step: int,
      *,
      prefix: str | None = None,
  ) -> None:
    """Logs a scalar metric using RLEngine-compatible '<prefix>/<mode>/<name>' layout."""
    if prefix is None:
      if "/" in metric_name:
        prefix, metric_name = metric_name.split("/", maxsplit=1)
      else:
        prefix = "global"
    if self.metrics_prefix:
      prefix = (
          f"{self.metrics_prefix}/{prefix}"
          if prefix
          else self.metrics_prefix
      )
    self.metrics_logger.log(prefix, metric_name, scalar_value, self.mode, step)

  async def _drain_filtered_groups(
      self,
  ) -> list[list[datatypes.TrajectoryItem]]:
    """Drains and returns all filtered groups across raw and scored queues."""
    filtered: list[list[datatypes.TrajectoryItem]] = []
    if hasattr(self.raw_q, "get_filtered_groups"):
      filtered.extend(await self.raw_q.get_filtered_groups())
    if hasattr(self.scored_q, "get_filtered_groups"):
      filtered.extend(await self.scored_q.get_filtered_groups())
    return filtered

  def _collect_and_log_step_metrics(
      self,
      *,
      all_step_items: Sequence[datatypes.TrajectoryItem],
      step_rewards: Sequence[float],
      step_advantages: Sequence[float] | None = None,
      generation_metrics: Mapping[str, float] | None = None,
      step_result: Any = None,
      trainer_metrics: Any = None,
      num_rollouts: int,
      num_microbatches: int,
      padding_stats: Sequence[batch_assembly.PaddingStats] = (),
      packing_time_sec: float = 0.0,
      step_time_sec: float,
      consumed_policy_version: int,
      log_step: int,
      sampler_agreement: dict[str, tuple[Any, list[float]]] | None = None,
      policy_training_time: float = 0.0,
      exposed_generation_time: float = 0.0,
      weight_sync_time: float = 0.0,
      metrics_fetch_time: float = 0.0,
      filtered_groups: Sequence[Sequence[datatypes.TrajectoryItem]] = (),
  ) -> dict[str, Any]:
    """Logs rollout, reward, trainer, and orchestrator metrics.

    TODO: b/552087289 - All metrics in this program are currently aggregated
    and flushed at the trainer's global step T boundary, which relies on an
    ON-POLICY assumption.
    In off-policy / asynchronous RL:
     1. Rollout/critique workers run asynchronously and may produce or buffer
        data across multiple steps; data generated during global step T might
        only be consumed at later steps (T+1, T+2, ...) or discarded.
     2. The batch consumed at step T directly updates the policy to pi_{T+1},
        but logging worker/rollout metrics at step T couples worker generation
        timelines to the trainer clock.
    Future work under b/552087289 will decouple metric logging so workers emit
    their own generation metrics independently (keyed by policy version or
    sample count), rather than forcing all worker metrics into the trainer's
    global step boundary.
    """
    # --- 1. Rollout metrics & Ingestion Staleness ---
    prompt_lengths = []
    completion_lengths = []
    total_lengths = []
    turns_list = []
    successes = []
    staleness_list = []
    partial_rollouts = []
    for item in all_step_items:
      p_len = None
      prompt_tokens = getattr(item, "prompt_tokens", None)
      if prompt_tokens is not None:
        p_len = len(prompt_tokens)
      elif hasattr(item, "payload"):
        segment_ids = getattr(item.payload, "segment_ids", None)
        prompt_mask = getattr(item.payload, "prompt_mask", None)
        completion_mask = getattr(item.payload, "completion_mask", None)
        if segment_ids is not None and completion_mask is not None:
          p_len = int(
              np.sum(
                  (np.asarray(segment_ids) > 0)
                  & (np.asarray(completion_mask) == 0)
              )
          )
        elif prompt_mask is not None and np.size(prompt_mask):
          p_len = int(np.sum(np.asarray(prompt_mask) > 0))

      c_len = None
      completion_tokens = getattr(item, "completion_tokens", None)
      if completion_tokens is not None:
        c_len = len(completion_tokens)
      elif (
          hasattr(item, "payload")
          and getattr(item.payload, "completion_mask", None) is not None
      ):
        c_len = int(np.sum(np.asarray(item.payload.completion_mask) > 0))

      if p_len is not None:
        prompt_lengths.append(p_len)
      if c_len is not None:
        completion_lengths.append(c_len)
      if p_len is not None and c_len is not None:
        total_lengths.append(p_len + c_len)

      traj = getattr(item, "traj", None)
      if isinstance(traj, dict):
        steps = traj.get("steps")
        status = traj.get("status") or getattr(item, "status", None)
      elif traj is not None:
        steps = getattr(traj, "steps", None)
        status = getattr(traj, "status", None) or getattr(item, "status", None)
      else:
        steps = getattr(item, "steps", None)
        status = getattr(item, "status", None)
      if steps and len(steps) > 0:
        turns_list.append(len(steps))
      if status is not None:
        if isinstance(status, datatypes.TrajectoryStatus):
          if status != datatypes.TrajectoryStatus.RUNNING:
            is_succ = status == datatypes.TrajectoryStatus.SUCCEEDED
            successes.append(1.0 if is_succ else 0.0)
        elif isinstance(status, str):
          status_str = status.upper()
          if status_str != "RUNNING":
            is_succ = status_str in ("COMPLETED", "SUCCEEDED", "SUCCESS")
            successes.append(1.0 if is_succ else 0.0)

      # Batch ingestion staleness: consumed_policy_version - item.policy_version
      pol_ver = getattr(item, "policy_version", None)
      if (
          pol_ver is None
          and hasattr(item, "metadata")
          and isinstance(item.metadata, dict)
      ):
        pol_ver = item.metadata.get("policy_version")
      if pol_ver is not None:
        staleness_list.append(float(max(0, consumed_policy_version - pol_ver)))

      turn_vers = item.metadata.get("turn_policy_versions")
      if turn_vers is None and isinstance(traj, dict):
        turn_vers = traj.get("turn_policy_versions")
      if isinstance(turn_vers, (list, tuple)) and turn_vers:
        partial_rollouts.append(
            1.0 if len({int(v) for v in turn_vers}) > 1 else 0.0
        )

    if prompt_lengths:
      self._log_metric(
          "rollout/prompts/mean_length",
          float(np.mean(prompt_lengths)),
          log_step,
      )
      self._log_metric(
          "rollout/prompts/max_length",
          float(np.max(prompt_lengths)),
          log_step,
      )
      self._log_metric(
          "rollout/prompts/min_length",
          float(np.min(prompt_lengths)),
          log_step,
      )
    if completion_lengths:
      self._log_metric(
          "rollout/completions/mean_length",
          float(np.mean(completion_lengths)),
          log_step,
      )
      self._log_metric(
          "rollout/completions/max_length",
          float(np.max(completion_lengths)),
          log_step,
      )
      self._log_metric(
          "rollout/completions/min_length",
          float(np.min(completion_lengths)),
          log_step,
      )
    if total_lengths:
      self._log_metric(
          "rollout/total_tokens_mean",
          float(np.mean(total_lengths)),
          log_step,
      )
      self._log_metric(
          "rollout/global_valid_toks",
          float(np.sum(total_lengths)),
          log_step,
      )
    elif completion_lengths:
      self._log_metric(
          "rollout/global_valid_toks",
          float(np.sum(completion_lengths)),
          log_step,
      )
    if all_step_items:
      self._log_metric(
          "rollout/global_valid_seqs",
          float(len(all_step_items)),
          log_step,
      )
    if turns_list:
      self._log_metric(
          "rollout/num_turns_mean",
          float(np.mean(turns_list)),
          log_step,
      )
    if successes:
      self._log_metric(
          "rollout/success_rate",
          float(np.mean(successes)),
          log_step,
      )
    if staleness_list:
      staleness_stats = {
          "staleness_mean": float(np.mean(staleness_list)),
          "staleness_max": float(np.max(staleness_list)),
          "staleness_min": float(np.min(staleness_list)),
      }
      for tag, val in staleness_stats.items():
        self._log_metric(f"rollout/{tag}", val, log_step)
    if partial_rollouts:
      self._log_metric(
          "rollout/partial_rollout_fraction",
          float(np.mean(partial_rollouts)),
          log_step,
      )

    # Generation/rollout metrics grouped under rollout/*.
    if generation_metrics:
      for tag, val in generation_metrics.items():
        if tag.startswith("generation/"):
          tag = "rollout/" + tag.removeprefix("generation/")
        self._log_metric(tag, val, log_step)

    filtered_groups_count = len(filtered_groups)
    filtered_trajectories_count = sum(len(g) for g in filtered_groups)
    self._log_metric(
        "rollout/filtered_groups_count",
        float(filtered_groups_count),
        log_step,
    )
    self._log_metric(
        "rollout/filtered_trajectories_count",
        float(filtered_trajectories_count),
        log_step,
    )

    # --- 2. Reward Metrics ---
    rewards_to_log = step_rewards
    if all_step_items:
      valid_flags = [item.is_valid for item in all_step_items]
      self._log_metric(
          "rollout/invalid_trajectory_frac",
          1.0 - float(np.mean(valid_flags)),
          log_step,
      )
      rewards_to_log = [
          reward
          for reward, is_valid in zip(step_rewards, valid_flags)
          if is_valid
      ]

    reward_mean = float(np.mean(rewards_to_log)) if rewards_to_log else 0.0
    reward_std = float(np.std(rewards_to_log)) if rewards_to_log else 0.0
    reward_min = float(np.min(rewards_to_log)) if rewards_to_log else 0.0
    reward_max = float(np.max(rewards_to_log)) if rewards_to_log else 0.0
    reward_sum = float(np.sum(rewards_to_log)) if rewards_to_log else 0.0
    if rewards_to_log:
      reward_stats = {
          "mean": reward_mean,
          "std": reward_std,
          "min": reward_min,
          "max": reward_max,
          "sum": reward_sum,
      }
      for tag, val in reward_stats.items():
        self._log_metric(f"rewards/{tag}", val, log_step)

    # --- Advantage Metrics ---
    advantage_mean = float(np.mean(step_advantages)) if step_advantages else 0.0
    advantage_std = float(np.std(step_advantages)) if step_advantages else 0.0
    advantage_min = float(np.min(step_advantages)) if step_advantages else 0.0
    advantage_max = float(np.max(step_advantages)) if step_advantages else 0.0
    advantage_abs_mean = (
        float(np.mean(np.abs(step_advantages))) if step_advantages else 0.0
    )
    advantage_nonzero_frac = (
        float(np.mean(np.abs(step_advantages) > 1e-8))
        if step_advantages
        else 0.0
    )
    if step_advantages:
      advantage_stats = {
          "mean": advantage_mean,
          "max": advantage_max,
          "min": advantage_min,
          "std": advantage_std,
          "abs_mean": advantage_abs_mean,
          "nonzero_frac": advantage_nonzero_frac,
      }
      for tag, val in advantage_stats.items():
        self._log_metric(f"rewards/advantage/{tag}", val, log_step)

    # --- 3. Perf & Orchestrator Metrics ---
    self._log_metric("perf/global_step_time", float(step_time_sec), log_step)
    orchestrator_stats = {
        "policy_version": float(self.policy_version),
        "num_rollouts": float(num_rollouts),
        "num_microbatches": float(num_microbatches),
        "step_time_sec": float(step_time_sec),
        "policy_training_time": float(policy_training_time),
        "exposed_generation_time": float(exposed_generation_time),
        "weight_sync_time": float(weight_sync_time),
        "metrics_fetch_time": float(metrics_fetch_time),
    }
    for tag, val in orchestrator_stats.items():
      self._log_metric(f"orchestrator/{tag}", val, log_step)
    if padding_stats:
      for tag, val in batch_assembly.summarize_padding_stats(
          padding_stats
      ).items():
        self._log_metric(f"efficiency/padding/{tag}", val, log_step)
      packing_stats = {
          "time_sec_total": float(packing_time_sec),
          "time_sec_mean": float(packing_time_sec) / len(padding_stats),
      }
      for tag, val in packing_stats.items():
        self._log_metric(f"efficiency/packing/{tag}", val, log_step)

    # --- 4. Actor Trainer Metrics ---
    loss_val = None
    perplexity_val = None
    grad_norm_val = None
    if trainer_metrics is None:
      if isinstance(step_result, dict):
        trainer_metrics = step_result.get("metrics")
      elif step_result is not None:
        trainer_metrics = step_result

    if trainer_metrics is not None:
      scalar_metrics = {}
      weighted_metrics = {}
      if hasattr(trainer_metrics, "scalar_metrics"):
        scalar_metrics.update(getattr(trainer_metrics, "scalar_metrics", {}))
      if hasattr(trainer_metrics, "weighted_metrics"):
        weighted_metrics.update(
            getattr(trainer_metrics, "weighted_metrics", {})
        )
      if isinstance(trainer_metrics, dict):
        if (
            "scalar_metrics" in trainer_metrics
            or "weighted_metrics" in trainer_metrics
        ):
          scalar_metrics.update(trainer_metrics.get("scalar_metrics") or {})
          weighted_metrics.update(trainer_metrics.get("weighted_metrics") or {})
        else:
          for k, v in trainer_metrics.items():
            if k == "metrics":
              continue
            scalar_metrics[k] = v

      # Loss & Perplexity
      raw_loss = scalar_metrics.pop(
          "loss",
          scalar_metrics.pop(
              "trainer/loss", scalar_metrics.pop("actor/loss", None)
          ),
      )
      if raw_loss is None:
        if "loss" in weighted_metrics:
          raw_loss = weighted_metrics.pop("loss")
        elif "trainer/loss" in weighted_metrics:
          raw_loss = weighted_metrics.pop("trainer/loss")
        elif "actor/loss" in weighted_metrics:
          raw_loss = weighted_metrics.pop("actor/loss")

      loss_val = _extract_scalar(raw_loss)
      if loss_val is not None:
        self._log_metric("loss", loss_val, log_step, prefix="actor")
        perplexity_val = float(np.exp(loss_val))
        self._log_metric(
            "perplexity", perplexity_val, log_step, prefix="actor"
        )

      # Learning Rate
      raw_lr = scalar_metrics.pop(
          "learning_rate",
          scalar_metrics.pop(
              "trainer/learning_rate",
              scalar_metrics.pop("actor/learning_rate", None),
          ),
      )
      lr_val = _extract_scalar(raw_lr)
      if lr_val is not None:
        self._log_metric("learning_rate", lr_val, log_step, prefix="actor")

      # Grad Norm
      # ``gradient_norm`` is the MaxText training engine's name for it
      # (``MaxTextTrainingEngine.update`` records it under that key). Read the
      # MaxText key via ``.get`` rather than ``.pop`` so ``gradient_norm``
      # continues to be published by the auxiliary scalar loop below while also
      # populating ``actor/grad_norm`` and the console step summary.
      raw_gn = scalar_metrics.pop("grad_norm", None)
      if raw_gn is None:
        raw_gn = scalar_metrics.pop("trainer/grad_norm", None)
      if raw_gn is None:
        raw_gn = scalar_metrics.pop("actor/grad_norm", None)
      if raw_gn is None:
        raw_gn = scalar_metrics.get("gradient_norm")
      if raw_gn is None:
        raw_gn = scalar_metrics.get("trainer/gradient_norm")
      if raw_gn is None:
        raw_gn = scalar_metrics.get("actor/gradient_norm")
      gn_val = _extract_scalar(raw_gn)
      grad_norm_val = gn_val
      if gn_val is not None:
        self._log_metric("grad_norm", gn_val, log_step, prefix="actor")

      # Auxiliary weighted metrics
      for k, v in weighted_metrics.items():
        val = _extract_scalar(v, k)
        if val is not None:
          clean_key = (
              k.removeprefix("trainer/").removeprefix("actor/")
          )
          self._log_metric(clean_key, val, log_step, prefix="actor")

      # Auxiliary scalar metrics
      for k, v in scalar_metrics.items():
        if k in ("perplexity", "trainer/perplexity", "actor/perplexity"):
          continue
        val = _extract_scalar(v, k)
        if val is not None:
          clean_key = (
              k.removeprefix("trainer/").removeprefix("actor/")
          )
          if clean_key.startswith(self._SAMPLER_IS_ATTR_PREFIXES):
            self._log_metric(
                clean_key.removeprefix("sampler_is/"),
                val,
                log_step,
                prefix="sampler_is_attr",
            )
          elif clean_key.startswith((
              "sampler_trainer/",
              "sampler_is/",
              "sampler_rs/",
          )):
            self._log_metric(clean_key, val, log_step)
          else:
            self._log_metric(clean_key, val, log_step, prefix="actor")

      # Every trainer-side metric for the step on one line, the sequence gate's
      # included (``sample_mask/kept_frac``, ``sample_mask/mult_prob_error_*``,
      # ``tis/is_oob_ratio``). The `metrics_logger` calls above only reach the
      # configured backends, so without this the gate is invisible to anyone
      # reading the run's log -- which is where it is read when a run is being
      # triaged and the dashboards are not to hand.
      step_trainer_metrics = {
          k: _extract_scalar(v, k)
          for k, v in {**weighted_metrics, **scalar_metrics}.items()
      }
      if (
          grad_norm_val is not None
          and "grad_norm" not in step_trainer_metrics
          and "trainer/grad_norm" not in step_trainer_metrics
      ):
        step_trainer_metrics["grad_norm"] = grad_norm_val
      logging.info(
          "[StepMetrics step=%d] trainer_metrics=%s",
          log_step,
          step_trainer_metrics,
      )

    # --- 5. Sampler/Trainer Agreement Metrics ---
    # Names are already namespaced (``sampler_trainer/*``, ``sampler_is/*``,
    # ``sampler_rs/*``) by the shared helper; reduce each metric's
    # per-microbatch values with the aggregation fn the helper paired with it.
    if sampler_agreement:
      for name, (agg_fn, values) in sampler_agreement.items():
        if not values:
          continue
        if name.startswith(self._SAMPLER_IS_ATTR_PREFIXES):
          self._log_metric(
              name.removeprefix("sampler_is/"),
              float(agg_fn(values)),
              log_step,
              prefix="sampler_is_attr",
          )
        else:
          self._log_metric(name, float(agg_fn(values)), log_step)

    return {
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "advantage_mean": advantage_mean,
        "advantage_std": advantage_std,
        "loss_val": loss_val,
        "perplexity_val": perplexity_val,
        "grad_norm_val": grad_norm_val,
        "filtered_groups_count": len(filtered_groups),
        "filtered_trajectories_count": sum(len(g) for g in filtered_groups),
    }

  async def _apply_sampler_trainer_agreement(
      self,
      batch: datatypes.RLTrainerPayload,
      accumulator: dict[str, tuple[Any, list[float]]],
  ) -> datatypes.RLTrainerPayload:
    """Records sampler-vs-trainer agreement and feeds TIS weights into a batch.

    Recomputes per-token log-probs under the trainer's live (actor) weights and
    compares them against the sampler's recorded ``old_per_token_logps`` to
    quantify sampler-vs-trainer drift, appending the resulting metrics to
    ``accumulator`` (keyed by the shared helper's already-namespaced names).
    When ``sampler_is == "token"`` it also writes truncated importance-sampling
    weights and overwrites ``old_per_token_logps`` with the trainer logps so the
    policy loss can correct for off-policy drift, matching the agentic learner.

    Args:
      batch: The microbatch to score; must carry ``old_per_token_logps``.
      accumulator: Per-step map of metric name -> (agg_fn, values) to extend.

    Returns:
      The batch, updated with TIS weights / trainer logps when
      ``sampler_is == "token"``; otherwise returned unchanged.
    """
    assert self.engine is not None
    gen_temp = getattr(self.generation_args, "temperature", None)
    logps_req = datatypes.LogprobsRequest(
        prompt_tokens=batch.prompt_ids,
        completion_tokens=batch.completion_ids,
        temperature=gen_temp if gen_temp is not None else 1.0,
        model_role="actor",
        pad_id=self.batch_config.pad_id,
        eos_id=getattr(self.assembler, "eos_id", self.batch_config.pad_id),
        segment_ids=batch.segment_ids,
        segment_positions=batch.segment_positions,
        routed_experts=batch.routed_experts,
    )
    trainer_logps = await self.engine.per_token_logps(
        datatypes.Role.ACTOR, items=logps_req
    )
    trainer_logps = np.asarray(trainer_logps.per_token_logps, dtype=np.float32)
    sa_metrics, sampler_is_weights, filtered_completion_mask = (
        rl_common.sampler_trainer_agreement(
            batch.old_per_token_logps,
            trainer_logps,
            batch.completion_mask,
            sampler_is=self.sampler_is,
            sampler_is_threshold=self.sampler_is_threshold,
            seq_logprob_error_threshold=self.seq_logprob_error_threshold,
            segment_ids=batch.segment_ids,
        )
    )
    for name, (value, agg_fn) in sa_metrics.items():
      accumulator.setdefault(name, (agg_fn, []))[1].append(float(value))

    updates: dict[str, Any] = {}
    if self.seq_logprob_error_threshold is not None:
      updates["completion_mask"] = filtered_completion_mask
    if sampler_is_weights is not None:
      updates["sampler_is_weights"] = sampler_is_weights
    if (
        self.sampler_is == "token"
        or self.seq_logprob_error_threshold is not None
    ):
      updates["old_per_token_logps"] = trainer_logps
    if updates:
      batch = dataclasses.replace(batch, **updates)
    return batch

  def _log_consumed_trajectories(
      self,
      all_step_items: Sequence[datatypes.TrajectoryItem],
      *,
      log_step: int,
      consumed_policy_version: int,
  ) -> None:
    """Persists one row per consumed rollout when trajectory logging is enabled."""
    if self.trajectory_logger is None:
      return
    for item in all_step_items:
      metadata = dict(getattr(item, "metadata", None) or {})
      env_config = metadata.get("env_config")
      if not isinstance(env_config, dict):
        env_config = {}
      # Logging must never take down a training step, so read defensively
      # rather than reusing the strict `_extract_reward` above.
      traj = getattr(item, "traj", None) or {}
      status = traj.get("status", None)
      reward = traj.get("trajectory_reward", None)
      if isinstance(status, datatypes.TrajectoryStatus):
        status = status.name
      prompt_id = getattr(item, "prompt_id", "")
      group_index = getattr(item, "group_index", 0)
      worker_id = (
          metadata.get("worker_id")
          or getattr(item, "worker_id", None)
          or f"worker{group_index}"
      )
      traj_id = (
          getattr(item, "traj_id", None)
          or f"traj_{prompt_id}_g{group_index}"
      )
      row = {
          "global_step": log_step,
          "consumed_policy_version": consumed_policy_version,
          "prompt_id": prompt_id,
          "group_index": group_index,
          "worker_id": worker_id,
          "traj_id": traj_id,
          "rollout_policy_version": getattr(item, "policy_version", 0),
          "status": status or "UNKNOWN",
          "reward": float(reward) if reward is not None else None,
          "question": metadata.get("question", env_config.get("question", "")),
          "prompt": metadata.get("prompt", env_config.get("prompt", "")),
          "completion": _format_rollout_completion(
              traj.get("conversation_text", "")
          ),
          "gold_answer": metadata.get(
              "gold_answer",
              metadata.get("answer", env_config.get("gold_answer", "")),
          ),
          "prompt_tokens": getattr(item, "prompt_tokens", None),
          "completion_tokens": getattr(item, "completion_tokens", None),
          "metadata": metadata,
          # The JSON logger writes only the text and timing fields. Leaving out
          # the arrays (routed_experts alone is ~1.2 KB/token) keeps a
          # backlogged queue from pinning a full batch of them in memory.
          # `ndim` catches numpy and jax arrays alike but keeps 0-d scalars
          # such as an np.float32 trajectory_reward, which the logger writes.
          "trajectory": {
              k: v for k, v in traj.items() if getattr(v, "ndim", 0) == 0
          },
      }
      self.trajectory_logger.log_item_async(row)

  async def _assemble(
      self,
      fn: Callable[..., list[batch_assembly.AssembledBatch]],
      *args: Any,
  ) -> list[batch_assembly.AssembledBatch]:
    """Runs a CPU-heavy assembler call off the event loop, under the lock."""

    def _locked() -> list[batch_assembly.AssembledBatch]:
      with self._assembler_lock:
        return fn(*args)

    return await asyncio.to_thread(_locked)

  async def _reset_assembler(self) -> None:
    """Resets the assembler once any in-flight feed/flush has finished."""

    def _locked() -> None:
      with self._assembler_lock:
        self.assembler.reset()

    # Off the event loop: the lock can be held by a worker thread.
    await asyncio.to_thread(_locked)

  async def _timed_train_step(
      self,
      batch: datatypes.RLTrainerPayload | None,
      *,
      apply_optimizer: bool,
      staged_request_id: str | None = None,
      wait_for_update: bool = True,
      defer_update: bool = False,
  ) -> tuple[Any, float]:
    assert self.engine is not None
    start = time.monotonic()
    # Passed only when set, so engines without staging see the call unchanged.
    kwargs = {}
    if staged_request_id is not None:
      kwargs["staged_request_id"] = staged_request_id
    if not wait_for_update:
      # Only with staging, i.e. to a trainer that supports it
      # (`_trainer_can_stage`): its worker's fused update then returns at
      # the update's dispatch (`TrainerWorker.fwd_bwd`).
      kwargs["wait_for_update"] = False
    if defer_update:
      # Only to a trainer that can (`_trainer_can_stage_before_update`): the
      # update is left to `apply_deferred_update`.
      kwargs["defer_update"] = True
    result = await self.engine.train_step(
        batch,
        role=datatypes.Role.ACTOR,
        accumulate_gradients=True,
        apply_optimizer=apply_optimizer,
        **kwargs,
    )
    return result, time.monotonic() - start

  async def _log_trainer_step_timing(self, step: int) -> None:
    """Logs `TRAINER_STEP_TIMING` for `step` (one line, machine-readable).

    Format::

      TRAINER_STEP_TIMING step=<N> policy_version=<v> rtt_s=<r> timing=<json>

    `timing` is the trainer worker's step snapshot (schema `v`): per
    microbatch `mb[i].{dispatch_begin,dispatch_end,ready}_ago_s`, the same for
    `update`, and `prepare.{recv,done}_ago_s` for this round's
    `prepare_weight_sync`. Every `*_ago_s` is seconds before the worker handled
    the request, measured on the trainer's monotonic clock. On this log's
    clock an instant is `t(line) - rtt_s / 2 - ago_s`. `ready` is when the
    device finished the work (None if not yet observed); `dispatch_*` bracket
    the RPC handler, which returns at dispatch. Never raises: a worker without
    the RPC logs nothing.
    """
    assert self.engine is not None
    get_timing = getattr(self.engine, "get_step_timing", None)
    if get_timing is None:
      return
    try:
      timing, rtt_s = await get_timing(role=datatypes.Role.ACTOR)
    except Exception as exc:  # pylint: disable=broad-except
      logging.warning("TRAINER_STEP_TIMING step=%d unavailable: %s", step, exc)
      return
    if timing is None:
      return
    logging.info(
        "TRAINER_STEP_TIMING step=%d policy_version=%d rtt_s=%.3f timing=%s",
        step,
        self.policy_version,
        rtt_s,
        json.dumps(timing, separators=(",", ":")),
    )

  async def _await_pending_sync(self) -> None:
    """Waits for the background weight sync, if any; raises if it failed."""
    task, self._pending_sync = self._pending_sync, None
    if task is not None:
      await task

  async def _run_background_sync(
      self, target_version: int, staged: asyncio.Event
  ) -> None:
    """Runs one weight sync round and records its outcome.

    Completion is handled here rather than in a done callback, so the state
    updates land before anything awaiting the task resumes.
    """
    assert self.engine is not None
    start_time = time.monotonic()
    try:
      new_version = await self.engine.sync_weights(
          role=datatypes.Role.ACTOR,
          policy_version=target_version,
          source_staged=staged,
      )
    except asyncio.CancelledError:
      raise
    except Exception as exc:
      logging.error(
          "Background weight sync to policy_version=%d failed after %.2fs: %s",
          target_version,
          time.monotonic() - start_time,
          exc,
      )
      self._sync_error = exc
      # The dispatcher has to wake to see `_sync_error`.
      self._release_window()
      raise
    if new_version is not None and new_version != target_version:
      logging.warning(
          "Background weight sync committed policy_version=%d, expected %d.",
          new_version,
          target_version,
      )
      self.policy_version = new_version
    logging.info(
        "Background weight sync committed policy_version=%d in %.2fs.",
        target_version,
        time.monotonic() - start_time,
    )
    self._unsynced_steps = 0
    self._release_window()

  async def _start_background_sync(self) -> None:
    """Launches this step's weight sync and returns once the source staged.

    Any previous round is awaited first, since rounds must not overlap.
    """
    await self._await_pending_sync()
    target_version = self.policy_version + 1
    staged = asyncio.Event()
    task = asyncio.create_task(
        self._run_background_sync(target_version, staged)
    )
    self._pending_sync = task
    # Before commit advances `_next_batch`, so the dispatcher cannot slip a
    # batch through against the old weights.
    self._unsynced_steps = 1
    self._round_due = False
    # The trainer's weights are this version from here on; rollouts catch up
    # when the round commits.
    self.policy_version = target_version
    staged_wait = asyncio.create_task(staged.wait())
    try:
      await asyncio.wait(
          {task, staged_wait}, return_when=asyncio.FIRST_COMPLETED
      )
    finally:
      staged_wait.cancel()
    if task.done():
      # Failed (or finished) before signalling; surface a failure now.
      await self._await_pending_sync()

  async def train_stage(self) -> None:
    """Stage 3: Streaming gradient accumulation with RLTrainerPayloads."""
    try:
      await self._train_loop()
      # The last round must land before the run is reported done.
      await self._await_pending_sync()
    finally:
      if self._pending_sync is not None and not self._pending_sync.done():
        self._pending_sync.cancel()
      if self._intake_task is not None and not self._intake_task.done():
        self._intake_task.cancel()

  async def _train_loop(self) -> None:
    """Trains one optimizer step per full batch of scored groups.

    One loop for both step boundaries. Without `pipeline_step_boundary`, step
    k+1 starts once step k's bookkeeping is done. With it, the trainer gets
    the same calls, with the same inputs, in the same order -- every
    micro-batch of step k (the last one also runs step k's update), then the
    metrics fetch, checkpoint save and weight-sync staging, then step k+1's
    first micro-batch -- and the assembler sees the same feeds, so the
    micro-batches, their order, and the weights each is computed on are
    unchanged. What moves is the orchestrator's own work between step k's
    update and step k+1's first micro-batch:

    * Packing. Once step k's last group is fed, step k+1's groups are taken
      and packed (`_prepack`) while step k's last micro-batches train.
    * Bookkeeping. Step k+1's first micro-batch is sent right after step k's
      sync stages, its batch commits and, when deferred, its metrics are
      fetched (`_send_first_microbatch_ahead`); step k's step metrics,
      logging, `on_step_end` and host-memory release run once that call is on
      the trainer.
    * Transfer (`stage_next_microbatch`). That micro-batch's data goes to the
      trainer host while the device still computes step k's last micro-batch
      (`_stage_planned`): between step k's last train call, sent with its
      update deferred, and that update (`_stage_next_before_update`), when
      the trainer can defer it; else right after the last call, which then
      returns at the update's dispatch (`_maybe_stage_next_microbatch`). The
      stage call, and with a deferred update the update call, are the only
      calls added on the trainer.
    """
    assert self.engine is not None
    ordered_q = isinstance(
        self.scored_q, trajectory_queue_manager.BatchOrderedQueueManager
    )
    # The next step's state while it is taken ahead (`_prepack`); only ever
    # set with `pipeline_step_boundary`.
    next_state: _StepState | None = None

    while self.max_steps is None or self._step < self.max_steps:
      current_step = self._step
      is_last_step = (
          self.max_steps is not None and current_step + 1 >= self.max_steps
      )
      consumed_policy_version = self.policy_version
      st = next_state if next_state is not None else _StepState()
      next_state = None
      # The step's clock starts at the step boundary in every mode: at the send
      # when its first micro-batch went ahead, and otherwise here, before any
      # wait for its groups, as without the pipelined boundary. So step_time
      # spans the same work either way, and the wait also counts as exposed
      # generation time.
      if st.start_time is None:
        st.start_time = time.monotonic()
      if self._intake_task is not None:
        # Still taking this step's groups ahead; from here the wait is exposed.
        intake, self._intake_task = self._intake_task, None
        t_wait = time.monotonic()
        await intake
        st.exposed_generation_time += time.monotonic() - t_wait
      step_start_time = st.start_time
      if st.groups_consumed == 0:
        # Nothing was taken ahead (the next batch was a hole, or the queue
        # closed): start exactly as without taking ahead.
        st.current_batch_idx = None
        st.scored_items = []
      elif not st.step_begun:
        st.step_begun = True
        if self.on_step_begin:
          self.on_step_begin(current_step)

      step_result = None
      trainer_metrics = None
      checkpoint_saved = False
      final_minibatch_completed = False
      pending_metrics_fetch = False
      queue_dry = False
      prepack_started = False
      policy_training_time = 0.0
      weight_sync_time = 0.0
      # The end-of-step get_metrics() deferred past the weight sync and commit,
      # kept out of policy_training_time because it no longer runs in training.
      metrics_fetch_time = 0.0

      async def _await_pending_train() -> None:
        nonlocal step_result, policy_training_time
        task, self._pending_train = self._pending_train, None
        if task is not None:
          step_result, elapsed = await task
          policy_training_time += elapsed

      async def _maybe_save_checkpoint() -> None:
        nonlocal checkpoint_saved
        await self._save_step_checkpoint(st, step_result)
        checkpoint_saved = True

      while True:
        if (
            self.pipeline_step_boundary
            and not prepack_started
            and not is_last_step
            and st.groups_consumed > 0
            and (queue_dry or st.groups_consumed >= self.full_batch_size)
        ):
          # Every group of this step is fed. Take the next step's ahead while
          # this step's last micro-batches train. Under PROMPT_BATCH order
          # that is the next batch by index: `next_batch_idx` does not move
          # before this batch commits, and a hole there just leaves nothing
          # taken.
          prepack_started = True
          next_state = _StepState(
              current_batch_idx=(
                  st.current_batch_idx + 1
                  if ordered_q and st.current_batch_idx is not None
                  else None
              )
          )
          self._intake_task = asyncio.create_task(self._prepack(next_state))

        # Send what is packed, in order: a feed's micro-batches go out before
        # the next group is taken.
        while st.ready:
          mb = st.ready.pop(0)
          is_final_batch = mb.is_final_batch
          # The reference model is static and on its own worker, so this
          # overlaps the in-flight actor train_step.
          batch = await self._with_reference_logps(mb.payload)
          rescored = self._needs_sampler_agreement(batch)
          if rescored:
            await _await_pending_train()
            batch = await self._apply_sampler_trainer_agreement(
                batch, st.sampler_agreement
            )
          self._record_microbatch(st, mb)
          mb = None
          if not self._first_train_payload_logged:
            self._first_train_payload_logged = True
            logging.info(
                "First trainer micro-batch: %s",
                batch_assembly.describe_payload(batch),
            )
          await _await_pending_train()
          # Save only at a resumable full-batch boundary. An optimizer step
          # can occur earlier when a full batch contains multiple mini
          # batches, but the dataset resume cursor advances in full batches.
          # (Known before the call: it changes neither count.)
          full_batch_complete = is_final_batch and (
              st.groups_consumed >= self.full_batch_size or not st.scored_items
          )
          # Whether the next step's first micro-batch is staged right after
          # this call. The call runs the update too, and then returns at the
          # update's dispatch rather than once it landed, so the transfer
          # hides behind the device's remaining work; the stage confirms the
          # update instead. Off the flag, the call is exactly as without it.
          stage_after = (
              full_batch_complete
              and self.stage_next_microbatch
              and await self._stage_planned(next_state, rescored=rescored)
          )
          # Or, when the trainer can, the update is left out of this call and
          # the stage goes before it (`_stage_next_before_update`).
          defer_update = (
              stage_after and await self._trainer_can_stage_before_update()
          )
          train_step = self._timed_train_step(
              batch,
              apply_optimizer=is_final_batch,
              wait_for_update=not stage_after,
              defer_update=defer_update,
          )
          batch = None
          if self.pipeline_train_microbatches and not is_final_batch:
            self._pending_train = asyncio.create_task(train_step)
            continue
          step_result, elapsed = await train_step
          policy_training_time += elapsed
          if is_final_batch:
            # While an update is deferred the worker refuses get_metrics, a
            # checkpoint and a weight sync, so `_stage_next_before_update`,
            # which always applies the update, runs before any of them. A
            # result without `update_deferred` means the trainer ran the
            # update inside the call after all (the engine reports that as a
            # fused call): `stage_after` is then still set, and the stage goes
            # after the update below, as with `--no-stage_before_update`.
            if (
                defer_update
                and isinstance(step_result, dict)
                and step_result.get("update_deferred")
            ):
              assert next_state is not None
              # The trainer has this step's last micro-batch and no update
              # yet: stage the next step's first micro-batch under it, then
              # run the update. Training time, as the fused call's wait for
              # the device would have been.
              deferred_start = time.monotonic()
              step_result = await self._stage_next_before_update(
                  next_state, step_result
              )
              policy_training_time += time.monotonic() - deferred_start
              # This boundary's stage was made, or refused, before the
              # update. Skip the hook below: staging after the update's
              # dispatch is the order this path replaces, and under dropless
              # replay it would find the device idle and hide nothing.
              stage_after = False
            if (
                isinstance(step_result, dict)
                and step_result.get("train_step") is not None
            ):
              self._last_optimizer_step = int(step_result["train_step"])
            update_ready = bool(
                isinstance(step_result, dict)
                and step_result.get("update_ready")
            )
            if stage_after:
              assert next_state is not None
              # The update is dispatched and the device has this step's last
              # micro-batch and the update ahead of it.
              if await self._maybe_stage_next_microbatch(
                  next_state, step_result
              ):
                update_ready = True
            # The checkpoint is stamped as the weight-update time and saved
            # next, so the update must have landed on device first. Unless the
            # trainer has said it did (`update_ready`, from this call or from
            # the stage right after it), fetching the metrics is what waits
            # for it, so the fetch stays ahead of the save, counted as
            # training time, as with a separate update RPC.
            pending_metrics_fetch = full_batch_complete and update_ready
            if not pending_metrics_fetch:
              _t_metrics = time.monotonic()
              trainer_metrics = await self.engine.get_metrics(
                  role=datatypes.Role.ACTOR
              )
              policy_training_time += time.monotonic() - _t_metrics
            final_minibatch_completed = True
            if self.sync_weights and self.async_weight_sync:
              self._round_due = True
            # TODO(tunix-dev): Configurable checkpointing frequency. Today we
            # checkpoint at the same frequency as the weight update.
            # TODO(tunix-dev): For now any failures in save_checkpoint will
            # abort the entire program. Make it configurable on whether to fail
            # or continue.
            if full_batch_complete:
              await _maybe_save_checkpoint()

        if queue_dry:
          if not checkpoint_saved and final_minibatch_completed:
            await _maybe_save_checkpoint()
          break
        if st.groups_consumed >= self.full_batch_size:
          break
        if not await self._take_group(st, current_step=current_step):
          await self._flush_step(st)
          queue_dry = True

      await _await_pending_train()
      if not st.all_step_items:
        logging.info(
            "Dataset exhausted at step %d before max_steps.", current_step
        )
        break

      if self.sync_weights and self._pending_cancel_tasks:
        await asyncio.gather(
            *list(self._pending_cancel_tasks), return_exceptions=True
        )
      if self.sync_weights and self.async_weight_sync:
        # Only the time the trainer is actually held: the previous round, if
        # it outlived this step, plus this round's source snapshot.
        _t_sync = time.monotonic()
        await self._start_background_sync()
        weight_sync_time = time.monotonic() - _t_sync
      elif self.sync_weights:
        _t_sync = time.monotonic()
        new_version = await self.engine.sync_weights(role=datatypes.Role.ACTOR)
        weight_sync_time = time.monotonic() - _t_sync
        self.policy_version = (
            new_version if new_version is not None else self.policy_version + 1
        )
      # The source has staged (or the round ran), so the trainer's update is
      # complete: fetch and log its device-side timeline for this step.
      await self._log_trainer_step_timing(current_step)

      # Before `commit()`, which will eventually take ownership of the groups.
      generation_metrics = _generation_metrics(st.uncommitted_groups)
      logging.info(
          "[pipeline] COMMIT step=%d batch_idx=%s consumed_version=%d"
          " rollout_versions=%s groups=%s",
          current_step,
          st.current_batch_idx,
          consumed_policy_version,
          [
              getattr(g[0], "policy_version", 0)
              for g in st.uncommitted_groups
              if g
          ],
          [
              (
                  getattr(g[0], "prompt_id", ""),
                  (getattr(g[0], "metadata", None) or {}).get("batch_idx"),
              )
              for g in st.uncommitted_groups
              if g
          ],
      )
      if ordered_q:
        assert st.current_batch_idx is not None
        self.scored_q.commit_batch(st.current_batch_idx)
      else:
        self.scored_q.commit(current_step, groups=st.uncommitted_groups)
      # The commit may have opened the dispatch window. Yield so the
      # dispatcher refills the just-synced rollout workers now, rather than
      # after this step's bookkeeping and the next step's first packing.
      await asyncio.sleep(0)

      if pending_metrics_fetch:
        _t_metrics = time.monotonic()
        trainer_metrics = await self.engine.get_metrics(
            role=datatypes.Role.ACTOR
        )
        metrics_fetch_time += time.monotonic() - _t_metrics

      step_time_sec = time.monotonic() - step_start_time

      filtered_groups = await self._drain_filtered_groups()

      if next_state is not None:
        # Step k+1's first micro-batch goes to the trainer now. This step's
        # remaining bookkeeping waits until that call is on the trainer, so
        # its transfer is not slowed by it and the device computes while it
        # runs: step k's `Train step` line, StepMetrics and `on_step_end`
        # therefore follow the `[pipeline] SENT_AHEAD` line by one train
        # call's return (one transfer unstaged, ~a round trip staged).
        sent_ahead = await self._send_first_microbatch_ahead(next_state)
        if sent_ahead is not None:
          logging.info(
              "[pipeline] SENT_AHEAD step=%d first microbatch of step %d is on"
              " its way to the trainer; step %d's metrics follow once it is"
              " there.",
              current_step,
              current_step + 1,
              current_step,
          )
          # Not collected here: its result is step k+1's, and a failure
          # surfaces when step k+1 collects it.
          await asyncio.wait({sent_ahead})

      metrics_summary = self._collect_and_log_step_metrics(
          all_step_items=st.all_step_items,
          step_rewards=st.step_rewards,
          step_advantages=st.step_advantages,
          generation_metrics=generation_metrics,
          step_result=step_result,
          trainer_metrics=trainer_metrics,
          num_rollouts=st.num_rollouts,
          num_microbatches=st.num_microbatches,
          padding_stats=st.padding_stats,
          packing_time_sec=st.packing_time_sec,
          step_time_sec=step_time_sec,
          consumed_policy_version=consumed_policy_version,
          log_step=current_step,
          sampler_agreement=st.sampler_agreement,
          policy_training_time=policy_training_time,
          exposed_generation_time=st.exposed_generation_time,
          weight_sync_time=weight_sync_time,
          metrics_fetch_time=metrics_fetch_time,
          filtered_groups=filtered_groups,
      )
      self._log_consumed_trajectories(
          st.all_step_items,
          log_step=current_step,
          consumed_policy_version=consumed_policy_version,
      )
      filtered_groups.clear()
      st.all_step_items.clear()
      st.uncommitted_groups.clear()
      st.scored_items.clear()

      self.last_step_result = RLStepResult(
          step=current_step,
          policy_version=self.policy_version,
          num_rollouts=st.num_rollouts,
          num_microbatches=st.num_microbatches,
          reward_mean=metrics_summary["reward_mean"],
          reward_std=metrics_summary["reward_std"],
          advantage_mean=metrics_summary["advantage_mean"],
          advantage_std=metrics_summary["advantage_std"],
          train_result=step_result,
      )

      loss_val = metrics_summary["loss_val"]
      perplexity_val = metrics_summary["perplexity_val"]
      grad_norm_val = metrics_summary["grad_norm_val"]
      if self.mode == Mode.TRAIN:
        logging.info(
            "Train step %d - loss: %s - reward_mean: %.4f - advantage_mean:"
            " %.4f - grad_norm: %s - perplexity: %s - step_time: %.2fs",
            current_step,
            f"{loss_val:.4f}" if loss_val is not None else "N/A",
            metrics_summary["reward_mean"],
            metrics_summary["advantage_mean"],
            f"{grad_norm_val:.4g}" if grad_norm_val is not None else "N/A",
            f"{perplexity_val:.4f}" if perplexity_val is not None else "N/A",
            step_time_sec,
        )

      if self.on_step_end:
        self.on_step_end(current_step, step_result)
      self._step += 1
      # Strictly after the increment: the dispatcher re-reads `_next_batch` on
      # waking, and waking it on the old value would park it again with no one
      # left to set the event.
      self._release_window()
      _release_host_memory()

  # The pieces of `_train_loop` that work on a `_StepState`, so that a step's
  # state can be filled ahead of it (`_prepack`) and its first micro-batch
  # sent (`_send_first_microbatch_ahead`) or staged
  # (`_maybe_stage_next_microbatch`) early under `pipeline_step_boundary`.

  async def _take_group(
      self, st: _StepState, *, current_step: int | None
  ) -> bool:
    """Takes the step's next group from `scored_q` and feeds it to the assembler.

    `current_step` is None when taking ahead for a step that has not started
    (`_prepack`): the queue wait and `on_step_begin` are then left to the
    step, which does both on adopting the state.

    Args:
      st: State of the step the group is for.
      current_step: The step taking the group, or None when taking ahead.

    Returns:
      False, leaving `st.scored_items == []`, when the queue has no further
      group for the step.
    """
    t_gen = time.monotonic()
    if isinstance(
        self.scored_q, trajectory_queue_manager.BatchOrderedQueueManager
    ):
      ordered = await self.scored_q.get_ordered_group(
          batch_idx=st.current_batch_idx
      )
      if ordered is None:
        scored_items = []
      else:
        st.current_batch_idx, scored_items = ordered
    else:
      scored_items = await self.scored_q.get_group_batch(num_groups=1)
    # TODO(sanbao): Track accurate generation time by matching rollout
    # requests/responses or reading from the rollout worker. Currently this
    # measures time spent waiting on the scored queue (which includes
    # rollout + reward calculation), as MLPerf does not strictly validate
    # per-component timing accuracy.
    if current_step is not None:
      st.exposed_generation_time += time.monotonic() - t_gen
    st.scored_items = scored_items
    if not scored_items:
      return False
    if current_step is not None and not st.step_begun:
      st.step_begun = True
      if self.on_step_begin:
        self.on_step_begin(current_step)

    st.groups_consumed += 1
    st.uncommitted_groups.append(scored_items)
    st.all_step_items.extend(scored_items)
    st.num_rollouts += len(scored_items)
    for item in scored_items:
      st.step_rewards.append(_extract_reward(item))
      payload = getattr(item, "payload", None)
      if (
          item.is_valid
          and payload is not None
          and payload.advantages is not None
          and np.asarray(payload.advantages).size > 0
      ):
        st.step_advantages.append(float(np.mean(payload.advantages)))

    payloads = []
    for item in scored_items:
      payload = getattr(item, "payload", None)
      if isinstance(payload, datatypes.RLTrainerPayload):
        payload = dataclasses.replace(
            payload,
            metadata={
                **payload.metadata,
                "traj_id": item.traj_id,
            },
        )
      payloads.append(payload)
    packing_start_time = time.perf_counter()
    assembled_batches = list(
        await self._assemble(
            self.assembler.feed,
            payloads,  # pyrefly: ignore[bad-argument-type]
        )
    )
    st.packing_time_sec += time.perf_counter() - packing_start_time
    # `self.assembler` has now copied or packed `routed_experts`; release
    # the per-trajectory references on `payloads` and `scored_items`
    # (which remain in `all_step_items` until step-end metric logging) so
    # unbatched routing arrays do not coexist with packed microbatches and
    # weight sync. Lightweight 1D mask fields on `item.payload` are kept
    # for `_collect_and_log_step_metrics`.
    payloads.clear()
    for item in scored_items:
      item.routed_experts = None
      if item.metadata is not None:
        item.metadata.pop("routed_experts", None)
      if isinstance(item.traj, dict):
        item.traj.pop("routed_experts", None)
      item_payload = getattr(item, "payload", None)
      if (
          isinstance(item_payload, datatypes.RLTrainerPayload)
          and item_payload.routed_experts is not None
      ):
        item.payload = dataclasses.replace(item_payload, routed_experts=None)
    st.ready.extend(assembled_batches)
    return True

  async def _flush_step(self, st: _StepState) -> None:
    """Packs whatever the assembler still holds for the step."""
    packing_start_time = time.perf_counter()
    st.ready.extend(list(await self._assemble(self.assembler.flush)))
    st.packing_time_sec += time.perf_counter() - packing_start_time

  async def _prepack(self, st: _StepState) -> None:
    """Takes groups for a step ahead of it, until one micro-batch is packed.

    Also stops once the step has all its groups, or the queue has no further
    group for it. The step takes over exactly where this stops, so the
    assembler sees the same feeds in the same order as without taking ahead.

    Args:
      st: State of the step being taken ahead.
    """
    while not st.ready and st.groups_consumed < self.full_batch_size:
      if not await self._take_group(st, current_step=None):
        return

  def _intake_done(self) -> bool:
    """Whether the intake task finished without error (and is not consumed)."""
    task = self._intake_task
    return (
        task is not None
        and task.done()
        and not task.cancelled()
        and task.exception() is None
    )

  async def _with_reference_logps(self, batch: Any) -> Any:
    """Adds the reference policy's log-probs when the algorithm needs them."""
    if not getattr(self.algo, "requires_reference_kl", False):
      return batch
    if not isinstance(batch, datatypes.RLTrainerPayload):
      raise TypeError(
          "Reference KL requires an assembler that returns "
          "datatypes.RLTrainerPayload microbatches; got "
          f"{type(batch).__name__}."
      )
    assert self.engine is not None
    # The reference model is static and on its own worker.
    ref_logps = await self.engine.per_token_logps(
        datatypes.Role.REFERENCE, items=batch
    )
    return batch_assembly.with_ref_per_token_logps(batch, ref_logps)

  def _needs_sampler_agreement(self, batch: Any) -> bool:
    """Whether `batch` is rescored under the trainer's live weights first."""
    algo_config = getattr(self.algo, "algo_config", None)
    return bool(
        isinstance(batch, datatypes.RLTrainerPayload)
        and batch.old_per_token_logps is not None
        and algo_config is not None
        and algo_config.use_rollout_logps
    )

  def _record_microbatch(
      self, st: _StepState, mb: batch_assembly.AssembledBatch
  ) -> None:
    """Counts a micro-batch about to be trained, and logs what it packs."""
    st.num_microbatches += 1
    if mb.padding_stats is not None:
      st.padding_stats.append(mb.padding_stats)
      logging.info(
          "Packed %d trajectories into microbatch: %s (padding_ratio=%.3f,"
          " row_imbalance=%.3f)",
          len(mb.trajectory_ids),
          logging_utils.summarize_list(list(mb.trajectory_ids)),
          mb.padding_stats.padding_ratio,
          mb.padding_stats.row_imbalance,
      )
    else:
      logging.info(
          "Packed %d trajectories into microbatch: %s",
          len(mb.trajectory_ids),
          logging_utils.summarize_list(list(mb.trajectory_ids)),
      )

  def _optimizer_step_after(self, step_result: Any) -> int:
    """The optimizer step a step's final train call completed."""
    if (
        isinstance(step_result, dict)
        and step_result.get("train_step") is not None
    ):
      return int(step_result["train_step"])
    return self.step + 1

  def _writes_scored_checkpoint(self, step_result: Any) -> bool:
    """Whether this step's checkpoint is one an MLPerf eval may be scored on."""
    return (
        self.val_start_step is not None
        and self._optimizer_step_after(step_result) >= self.val_start_step
    )

  async def _save_step_checkpoint(
      self, st: _StepState, step_result: Any
  ) -> None:
    """Stamps the step's checkpoint time and saves it unless it is unscored.

    The stamp is taken here, after the barrier that proves the step's update
    landed (MLPerf: a checkpoint's timestamp follows its last weight update):
    `get_metrics`, or the trainer's `update_ready` on the step's last train
    call or on the stage right after it.

    Args:
      st: The step's state.
      step_result: The step's final train call result.
    """
    assert self.engine is not None
    ckpt_ts_ms = time.time_ns() // 1_000_000
    self.last_step_timestamp_ms = ckpt_ts_ms
    optimizer_step = self._optimizer_step_after(step_result)
    if self.val_start_step is not None and optimizer_step < self.val_start_step:
      return
    if isinstance(
        self.scored_q,
        trajectory_queue_manager.BatchOrderedQueueManager,
    ):
      next_batch_idx = (
          self.scored_q.next_batch_after(st.current_batch_idx)
          if st.current_batch_idx is not None
          else self.scored_q.next_batch_idx
      )
    else:
      next_batch_idx = self.step + 1
    save_kwargs = {}
    interval = self.checkpoint_optimizer_interval_steps
    if interval == 0 or (
        optimizer_step % interval != 0
        and (self.max_steps is None or self.step + 1 < self.max_steps)
    ):
      # Sent only when skipping, so an interval of 1 leaves the request,
      # and trainers that predate the option, untouched.
      save_kwargs["save_optimizer_state"] = False
    save_resp = await self.engine.save_checkpoint(
        role=datatypes.Role.ACTOR,
        metadata={
            "step": optimizer_step,
            "global_step": self.step + 1,
            "next_batch_idx": next_batch_idx,
            "policy_version": self.policy_version + 1,
            "num_rollouts": st.num_rollouts,
            "num_microbatches": st.num_microbatches,
        },
        **save_kwargs,
    )
    if self.on_checkpoint_saved is not None:
      # `TrainerWorker.save_checkpoint` always returns the saved path in
      # `Response.metadata["checkpoint_path"]`. The callback runs in a
      # worker thread (it may do blocking file/GCS I/O) and is awaited, so
      # calls never overlap and its exceptions still propagate.
      ckpt_info: dict[str, Any] = {
          "step": optimizer_step,
          "timestamp_ms": ckpt_ts_ms,
          "checkpoint_path": save_resp.metadata["checkpoint_path"],
      }
      await asyncio.to_thread(self.on_checkpoint_saved, ckpt_info)

  # How long a trainer must still have waited for its device, once a staged
  # micro-batch was in place, for the stage to count as hidden behind the
  # device's remaining work (`_staging_hidden`). A shorter wait is read as a
  # device that had already gone idle, and switches staging off. That catches a
  # stage that did not hide only where waiting on an idle device returns in well
  # under this time. On Pathways the wait is a proxy round trip that reads about
  # 2.7-2.9 s even on an idle device, above this threshold, so there this check
  # never switches staging off.
  _STAGING_HIDDEN_WAIT_S = 0.05

  def _staging_hidden(self, device_wait_s: float | None) -> bool:
    """Whether a stage hid behind the trainer's device (see `_maybe_stage_...`).

    Args:
      device_wait_s: How long the trainer still waited for the work dispatched
        before the stage, once the staged micro-batch was in place; None when
        it cannot tell.
    """
    return (
        device_wait_s is not None
        and device_wait_s >= self._STAGING_HIDDEN_WAIT_S
    )

  async def _trainer_can_stage(self) -> bool:
    """Whether the actor's trainer supports staging; asks it once.

    Staging needs a trainer that holds a micro-batch ahead of its fwd_bwd and
    reports whether doing so hid behind its device (`device_wait_s`, without
    which the latch below cannot tell a feed-bound trainer). An older trainer
    image, or a trainer without the report, gets the boundary unstaged.
    """
    if self._staging_supported is None:
      probe = getattr(self.engine, "trainer_staging_support", None)
      if (
          probe is None
          or getattr(self.engine, "stage_train_step", None) is None
      ):
        support = {}
      else:
        support = await probe(role=datatypes.Role.ACTOR)
      self._staging_support = dict(support)
      self._staging_supported = bool(
          support.get("stage_fwd_bwd") and support.get("reports_device_wait")
      )
      if not self._staging_supported:
        self._staging_latched_off = True
        logging.warning(
            "[pipeline] stage_next_microbatch is set but the trainer cannot"
            " stage a microbatch and report whether that hid behind its"
            " device (%s). Running the pipelined boundary unstaged.",
            support,
        )
    return self._staging_supported

  async def _trainer_can_stage_before_update(self) -> bool:
    """Whether to stage between the last call and the update; asks once.

    Needs `stage_before_update`, an engine with `apply_deferred_update`, and a
    trainer that can defer a step's update (`stage_before_update` in the
    staging support `_trainer_can_stage` asked it for). Only asked when a
    stage is planned, i.e. once the trainer is known to stage at all.
    """
    if not self.stage_before_update:
      return False
    if self._stage_before_update_supported is None:
      if not await self._trainer_can_stage():
        return False
      support = self._staging_support or {}
      self._stage_before_update_supported = bool(
          support.get("stage_before_update")
          and getattr(self.engine, "apply_deferred_update", None) is not None
      )
      if not self._stage_before_update_supported:
        logging.warning(
            "[pipeline] The trainer cannot defer a step's update past its last"
            " microbatch (%s): staging right after the last call instead,"
            " which hides nothing when the trainer's update waits for that"
            " microbatch (MaxText retry_when_tokens_dropped).",
            support,
        )
    return self._stage_before_update_supported

  async def _stage_planned(
      self, st: _StepState | None, *, rescored: bool
  ) -> bool:
    """Whether to stage the next step's first micro-batch after the last call.

    Decided before this step's last call is sent: it runs the update too; when
    a stage follows it must return at the update's dispatch
    (`wait_for_update=False`) rather than once the update landed, or the
    transfer could not hide behind the device. Planning a stage that then
    does not happen costs that step the deferred metrics fetch: the update
    is not confirmed, so get_metrics() is its barrier again, ahead of the
    checkpoint save. So this rules out now whatever `_maybe_stage_next_...`
    would refuse after the call: the flag, the latch, a step that writes a
    scored checkpoint (the update this call runs is the one after the last
    the trainer reported; before any, `self.step + 1`, exact with one update
    per step as the recipes run), rescored micro-batches, an already packed
    micro-batch that closes its step, and a trainer that cannot stage. What
    is left open is whether the micro-batch is packed by the time the call
    returns.

    Args:
      st: The next step's state, or None when nothing is taken ahead.
      rescored: This step's last micro-batch is rescored under the live
        weights (`_needs_sampler_agreement`). The next step's carry the same
        rollout log-probs, so its first would be too, and those go out whole.

    Returns:
      Whether to stage: send the step's last call with its update deferred
      and call `_stage_next_before_update` once it returns, when the trainer
      can (`_trainer_can_stage_before_update`); else send it with
      `wait_for_update=False` and call `_maybe_stage_next_microbatch` once it
      returns.
    """
    if (
        not self.stage_next_microbatch
        or st is None
        or self._staging_latched_off
        or st.staged is not None
        or rescored
    ):
      return False
    predicted_step = (
        self._last_optimizer_step + 1
        if self._last_optimizer_step is not None
        else self.step + 1
    )
    if self._writes_scored_checkpoint({"train_step": predicted_step}):
      return False
    if self._intake_done():
      # Taking ahead is over: only what it packed could go, and it has to be
      # neither the step's last micro-batch nor rescored.
      if not st.ready:
        return False
      first = st.ready[0]
      if first.is_final_batch or self._needs_sampler_agreement(first.payload):
        return False
    return await self._trainer_can_stage()

  async def _maybe_stage_next_microbatch(
      self,
      st: _StepState,
      step_result: Any,
  ) -> bool:
    """Sends the next step's first micro-batch to the trainer host ahead of time.

    Called right after the current step's last train call returns. That call
    ran the update too and, staging being planned (`_stage_planned`), came
    back at the update's dispatch, while the trainer's device still computes
    the step's last micro-batch and update, so the transfer hides behind that
    work. The train call that later runs it (`_send_first_microbatch_ahead`)
    names it instead of carrying it, and lands on the trainer at the same
    point as without staging. Skipped when the micro-batch is not packed yet,
    closes its step, or is rescored under the live weights first
    (`use_rollout_logps`): those go out whole.

    Also skipped when this step writes a scored checkpoint: its timestamp is
    taken once the update lands (`get_metrics`), and a transfer longer than
    the device's remaining work -- several trainer copies fed through one
    controller -- would push that timestamp back.

    And switched off for the rest of the run the first time a stage is seen not
    to hide. The trainer answers once the micro-batch is in place and its device
    has finished the work dispatched before it, and says how long it waited for
    the latter. A wait under `_STAGING_HIDDEN_WAIT_S` means the device had
    already gone idle: the transfer outlasted the step's remaining device work
    and delayed this step's metrics, stamp and weight sync. The trainer is then
    fed slower than it computes, so staging only moves a transfer from after the
    sync to before it. (A step's train calls returning cannot tell this: MaxText
    queues `max_inflight_computations` of them, and a step starts on a device
    drained by the weight sync, so its first calls return at the transfer's pace
    even when the device is the bottleneck.) On Pathways even an idle device's
    wait reads above that threshold, so there such a stage is not seen and
    staging stays on.

    A stage is optional, so one that fails on the way to the trainer, or in
    the reference log-probs it needs, falls back: the micro-batch goes back
    to the front of `st.ready` and is sent whole in the next step's first
    train call, as without staging, and this step keeps its get_metrics()
    barrier, checkpoint and weight sync. Staging is then off for the rest of
    the run. A stage the trainer ran and that raised there ends the run with
    its error, as a failed train call does: past its refusals,
    `TrainerWorker.stage_fwd_bwd` puts the worker in ERROR, so no checkpoint
    could follow anyway.

    Args:
      st: The next step's state, filled by `_prepack`.
      step_result: The current step's final train call result.

    Returns:
      Whether the trainer confirmed, once the micro-batch was staged, that
      the step's update has landed (`StagedTrainStep.update_ready`): the
      caller may then stamp and save the checkpoint without a get_metrics()
      first, as after a fused call that waited for the update. False when
      nothing was staged.
    """
    if self._staging_latched_off:
      return False
    if self._writes_scored_checkpoint(step_result):
      return False
    if not self._intake_done() or st.staged is not None:
      return False
    if not st.ready or st.ready[0].is_final_batch:
      return False
    if self._needs_sampler_agreement(st.ready[0].payload):
      return False
    if not await self._trainer_can_stage():
      return False
    sent = await self._send_stage(st)
    if sent is None:
      return False
    staged, stage_s = sent
    device_wait_s = staged.device_wait_s
    logging.info(
        "[pipeline] STAGED next step's first microbatch on the trainer in"
        " %.2fs; its device then still computed for %s (request_id=%s)",
        stage_s,
        "unknown" if device_wait_s is None else f"{device_wait_s:.2f}s",
        staged.request_id,
    )
    if not self._staging_hidden(device_wait_s):
      self._staging_latched_off = True
      logging.warning(
          "[pipeline] Staging a microbatch took %.2fs and the trainer's device"
          " was idle before it finished (device wait %s): the trainer is fed"
          " slower than it computes, so staging only delays each step's"
          " weight sync. Not staging for the rest of the run.",
          stage_s,
          "unknown" if device_wait_s is None else f"{device_wait_s:.3f}s",
      )
    return staged.update_ready

  async def _send_stage(self, st: _StepState) -> tuple[Any, float] | None:
    """Stages `st.ready[0]` on the trainer; the callers checked it may go.

    A stage is optional, so one that fails on the way to the trainer, or in
    the reference log-probs it needs, falls back: the micro-batch goes back
    to the front of `st.ready` and staging is off for the rest of the run. A
    stage the trainer ran and that raised there is re-raised.

    Args:
      st: The next step's state, filled by `_prepack`.

    Returns:
      The engine's `StagedTrainStep` and how long the call took, or None when
      it fell back.
    """
    intake, self._intake_task = self._intake_task, None
    mb = st.ready.pop(0)
    calling_trainer = False
    try:
      batch = await self._with_reference_logps(mb.payload)
      start = time.monotonic()
      calling_trainer = True
      staged = await (
          self.engine.stage_train_step  # pyrefly: ignore[missing-attribute]
      )(batch, role=datatypes.Role.ACTOR)
      stage_s = time.monotonic() - start
    except Exception as exc:  # pylint: disable=broad-exception-caught
      if calling_trainer and _raised_by_worker_method(exc):
        raise
      # Undone exactly, so `_send_first_microbatch_ahead` sends it whole. Had
      # the trainer received it after all (a reply lost on the way back), it
      # holds it unused: with staging off nothing names it, and a successor's
      # bring-up drops it.
      st.ready.insert(0, mb)
      self._intake_task = intake
      self._staging_latched_off = True
      logging.warning(
          "[pipeline] Staging failed on the way to the trainer or in the"
          " reference log-probs (%r); sending the microbatch whole in step"
          " %d's first train call instead, and not staging for the rest of"
          " the run.",
          exc,
          self._step + 1,
      )
      return None
    batch = None
    st.staged = (mb._replace(payload=None), staged.request_id)
    return staged, stage_s

  async def _stage_next_before_update(
      self, st: _StepState, step_result: dict[str, Any]
  ) -> dict[str, Any]:
    """Stages the next step's first micro-batch, then runs the step's update.

    Called once the step's last train call, sent with its update deferred
    (`_stage_planned`, `_trainer_can_stage_before_update`), has returned: the
    trainer has dispatched that micro-batch and holds the update back. Under
    MaxText's `retry_when_tokens_dropped` the call returns once the
    micro-batch before it has finished (it reads that one's overflow flag),
    so the device computes the whole last micro-batch while the stage's
    transfer runs. The update then goes out when the last micro-batch is done,
    where `update()` would have dispatched it anyway (it reads that one's flag
    first), so nothing is delayed while the transfer is shorter than the
    micro-batch.

    The micro-batch is staged under the refusals of
    `_maybe_stage_next_microbatch`: not when it is not packed yet, closes its
    step, or is rescored under the live weights first (those go out whole),
    and not once staging is off. The scored-checkpoint refusal was applied by
    `_stage_planned`, from the predicted optimizer step. A stage that fails on
    the way falls back as there (`_send_stage`); the update still runs.

    The update call waits for the device first and says how long
    (`device_wait_s`): how long the last micro-batch still ran once the staged
    one was in place. Under `_STAGING_HIDDEN_WAIT_S`, the transfer outlasted
    it and delayed the update, metrics, checkpoint stamp and weight sync: the
    trainer is fed slower than it computes, so staging is switched off for the
    rest of the run, as there. Also as there, this does not fire on Pathways,
    where even an idle device's wait reads above that threshold.

    A replay of the last micro-batch (MaxText reruns an overflowed one,
    dropless, inside `update()`) leaves the staged micro-batch as it is: its
    inputs are mapped from its own payload alone (`prepare_fwd_bwd_input`
    reads no parameter, gradient or optimizer state), and the rerun reads the
    inputs it kept for itself.

    Args:
      st: The next step's state, filled by `_prepack`.
      step_result: The last train call's result (`update_deferred`).

    Returns:
      `step_result` with the update call's result over it: `updated`,
      `train_step`, `update_ready` (the trainer confirmed the update landed,
      so the caller may stamp and save the checkpoint without a get_metrics()
      first) and `device_wait_s`.
    """
    assert self.engine is not None
    sent = None
    if (
        not self._staging_latched_off
        and self._intake_done()
        and st.staged is None
        and st.ready
        and not st.ready[0].is_final_batch
        and not self._needs_sampler_agreement(st.ready[0].payload)
    ):
      sent = await self._send_stage(st)
    update = await (
        self.engine.apply_deferred_update  # pyrefly: ignore[missing-attribute]
    )(role=datatypes.Role.ACTOR)
    if sent is not None:
      staged, stage_s = sent
      device_wait_s = update.get("device_wait_s")
      logging.info(
          "[pipeline] STAGED next step's first microbatch on the trainer"
          " before the update in %.2fs; the last microbatch then still"
          " computed for %s (request_id=%s)",
          stage_s,
          "unknown" if device_wait_s is None else f"{device_wait_s:.2f}s",
          staged.request_id,
      )
      if not self._staging_hidden(device_wait_s):
        self._staging_latched_off = True
        logging.warning(
            "[pipeline] Staging a microbatch took %.2fs and the trainer's"
            " device was idle before it finished (device wait %s): the"
            " trainer is fed slower than it computes, so staging only delays"
            " each step's update and weight sync. Not staging for the rest of"
            " the run.",
            stage_s,
            "unknown" if device_wait_s is None else f"{device_wait_s:.3f}s",
        )
    return {**step_result, **update}

  async def _send_first_microbatch_ahead(
      self, st: _StepState
  ) -> asyncio.Task[tuple[Any, float]] | None:
    """Sends the next step's first micro-batch, taken ahead by `_prepack`.

    Called once the current step's weight sync has staged and its batch is
    committed: the point at which `_train_loop` first sends any of the next
    step's micro-batches. The train call is left in `_pending_train` for the
    next step to collect as its own. The next step's groups then keep being
    taken, so its second micro-batch is packed while the first is in flight.

    Args:
      st: The next step's state.

    Returns:
      The train call's task, or None when nothing was ready to send.
    """
    assert self._pending_train is None
    if st.staged is not None:
      mb, staged_request_id = st.staged
      st.staged = None
      st.start_time = time.monotonic()
      batch = None
    elif self._intake_done() and st.ready and not st.ready[0].is_final_batch:
      self._intake_task = None
      mb = st.ready.pop(0)
      staged_request_id = None
      st.start_time = time.monotonic()
      batch = await self._with_reference_logps(mb.payload)
      if self._needs_sampler_agreement(batch):
        batch = await self._apply_sampler_trainer_agreement(
            batch, st.sampler_agreement
        )
    else:
      return None
    self._record_microbatch(st, mb)
    mb = None
    self._pending_train = asyncio.create_task(
        self._timed_train_step(
            batch,
            apply_optimizer=False,
            staged_request_id=staged_request_id,
        )
    )
    batch = None
    if (
        not st.ready
        and st.scored_items
        and st.groups_consumed < self.full_batch_size
    ):
      self._intake_task = asyncio.create_task(self._prepack(st))
    return self._pending_train

  async def run_async(
      self,
      engine: rl_engine_interface.AbstractRLEngine,
      **kwargs: Any,
  ) -> None:
    """Launches all stages concurrently on event loop."""
    del kwargs
    self.engine = engine
    # Must happen before any stage starts: train_stage reads `_step` for its
    # loop bound and rollout_dipatch_stage reads resumed `_step` to skip
    # the dataset prefix the previous run already consumed.
    await self._resume_from_checkpoint()
    logging.info("Starting StandardRLProgram concurrent stages...")

    engine.configure_worker(
        role=datatypes.Role.ACTOR,
        algo=self.algo,
        assembler=self.assembler,
    )

    if self.sync_weights and self._step == 0:
      await engine.prepare_rollout_policy(
          role=datatypes.Role.ACTOR,
          sync_weights=True,
          policy_version=self.policy_version,
      )

    if self.trainer_warmup_payload is not None:
      warmup_payload, self.trainer_warmup_payload = (
          self.trainer_warmup_payload,
          None,
      )
      logging.info(
          "Trainer warm-up compile against synthetic micro-batch: %s",
          batch_assembly.describe_payload(warmup_payload),
      )
      _t_compile = time.monotonic()
      await engine.compile_trainer(  # pytype: disable=attribute-error
          warmup_payload, role=datatypes.Role.ACTOR
      )
      del warmup_payload
      logging.info(
          "Trainer warm-up compile finished in %.1fs.",
          time.monotonic() - _t_compile,
      )

    if self.on_train_start is not None:
      self.on_train_start(self._step)

    train_task = asyncio.create_task(self.train_stage())
    tasks = [
        asyncio.create_task(self.rollout_dispatch_stage()),
        asyncio.create_task(self.polling_stage()),
        asyncio.create_task(self.critique_stage()),
        train_task,
    ]

    pending_tasks = set(tasks)
    try:
      while not train_task.done() and pending_tasks:
        done, pending_tasks = await asyncio.wait(
            pending_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=0.05
        )
        for task in done:
          if task.exception():
            raise task.exception()  # pyrefly: ignore[bad-raise]
      if train_task.exception():
        raise train_task.exception()  # pyrefly: ignore[bad-raise]
    except Exception as exc:
      logging.error("Exception in StandardRLProgram execution: %s", exc)
      await self.raw_q.abort(exc)
      await self.scored_q.abort(exc)
      await self._reset_assembler()
      raise
    finally:
      for task in tasks:
        if not task.done():
          task.cancel()
      if self._pending_train is not None and not self._pending_train.done():
        self._pending_train.cancel()
      if self._intake_task is not None and not self._intake_task.done():
        self._intake_task.cancel()

  def run(
      self,
      engine: rl_engine_interface.AbstractRLEngine,
      **kwargs: Any,
  ) -> None:
    """Synchronous entry point running all stages on an event loop."""
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      loop = None

    def _retrieve_task_exception(t: asyncio.Task[Any]) -> None:
      try:
        t.result()
      except Exception:  # pylint: disable=broad-except
        # Exception is already logged inside run_async, we just need to
        # retrieve it so asyncio doesn't complain about unretrieved exceptions.
        pass

    if loop and loop.is_running():
      self._bg_task = asyncio.create_task(self.run_async(engine, **kwargs))
      self._bg_task.add_done_callback(_retrieve_task_exception)
    else:
      asyncio.run(self.run_async(engine, **kwargs))
