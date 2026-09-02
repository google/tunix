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
from collections.abc import Callable, Iterable, Sequence
import dataclasses
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
from tunix.sft import metrics_logger as metrics_logger_lib

MetricsLogger = metrics_logger_lib.MetricsLogger
MetricsLoggerOptions = metrics_logger_lib.MetricsLoggerOptions
Mode = metrics_logger_lib.Mode
_extract_scalar = metrics_logger_lib.extract_scalar


@dataclasses.dataclass(kw_only=True)
class RLStepResult:
  """Summary of a completed RL training step."""

  step: int
  policy_version: int
  num_rollouts: int
  num_microbatches: int
  reward_mean: float
  reward_std: float
  train_result: Any = None


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
      group_size: int = 8,
      mini_batch_size: int = 4,
      max_staleness: int = 0,
      sync_weights: bool = True,
      metrics_logging_options: MetricsLoggerOptions | None = None,
      metrics_prefix: str = "",
      mode: Mode | str = Mode.TRAIN,
      on_step_begin: Callable[[int], None] | None = None,
      on_step_end: Callable[[int, Any], None] | None = None,
  ):
    super().__init__()
    self.engine: rl_engine_interface.AbstractRLEngine | None = None
    if max_staleness < 0:
      raise ValueError("max_staleness must be non-negative.")
    self.dataset = dataset
    self.max_steps = max_steps
    self.algo = algo
    self.reward_fns = list(reward_fns) if reward_fns else []
    self.group_size = getattr(algo, "group_size", group_size)
    self.mini_batch_size = getattr(algo, "mini_batch_size", mini_batch_size)
    if self.mini_batch_size <= 0 or self.group_size <= 0:
      raise ValueError("mini_batch_size and group_size must be positive.")
    self.assembler = assembler or batch_assembly.SequencePackedBatchAssembler(
        group_size=self.group_size,
        max_packed_len=getattr(algo, "max_packed_len", 8192),
    )
    self.assembler.group_size = self.group_size
    rollouts_per_mini_batch_step = self.mini_batch_size * self.group_size
    assembly_batch_size = self.assembler.assembly_batch_size
    if rollouts_per_mini_batch_step % assembly_batch_size != 0:
      raise ValueError(
          f"Total rollouts per mini-batch step ({rollouts_per_mini_batch_step}) is "
          f"not divisible by assembler assembly_batch_size ({assembly_batch_size})."
      )
    self.max_staleness = max_staleness
    self.sync_weights = sync_weights
    self.metrics_logger: MetricsLogger = MetricsLogger(metrics_logging_options)
    self.metrics_prefix = metrics_prefix
    self.mode = mode if isinstance(mode, Mode) else Mode(mode)
    self.on_step_begin = on_step_begin
    self.on_step_end = on_step_end
    self._in_flight_rollouts = 0
    self._dispatch_capacity: asyncio.Semaphore | None = None
    self._dispatch_done = asyncio.Event()

    self.raw_q = trajectory_queue_manager.TrajectoryQueueManager.create(
        group_size=self.group_size,
        max_staleness=max_staleness,
        current_policy_version=lambda: self.policy_version,
    )
    self.scored_q = trajectory_queue_manager.TrajectoryQueueManager.create(
        group_size=self.group_size
    )

  def close(self) -> None:
    """Flushes and closes the metrics logger and associated resources."""
    if self.metrics_logger is not None:
      self.metrics_logger.close()

  async def _wait_for_dispatch_window(self) -> None:
    """Applies policy-staleness backpressure utilizing token buckets."""
    assert (
        self._dispatch_capacity is not None
    ), "run_async must initialize capacity."
    await self._dispatch_capacity.acquire()

  async def rollout_dispatch_stage(self) -> None:
    """Stage 1A: Dispatches rollout requests across workers asynchronously.

    Ensures that all dataset items carry unique, collision-free `prompt_id`s
    (e.g., `f"prompt_{prompt_idx}"`) before dispatching to the engine layer,
    satisfying the engine's strict `prompt_id` contract.
    """
    assert self.engine is not None
    # TODO(tunix-dev): Skip already trained datasets when resuming from
    # checkpoints.
    if self.dataset is None:
      raise ValueError(
          "StandardRLProgram requires a dataset either at init or in run()."
      )

    try:
      for prompt_idx, prompt_item in enumerate(self.dataset):
        await self._wait_for_dispatch_window()
        if isinstance(prompt_item, dict):
          prompt_item = dict(prompt_item)
          prompt_item.setdefault("prompt_id", f"prompt_{prompt_idx}")
        elif not hasattr(prompt_item, "prompt_id"):
          prompt_item = {
              "prompt": prompt_item,
              "prompt_id": f"prompt_{prompt_idx}",
          }

        self._in_flight_rollouts += self.group_size
        await self.engine.dispatch_rollouts(
            [prompt_item],
            group_size=self.group_size,
            policy_version=self.policy_version,
        )
    finally:
      self._dispatch_done.set()

  async def polling_stage(self) -> None:
    """Stage 1B: Long-polls completed worker rollout responses into the queue."""
    assert self.engine is not None
    try:
      while not self._dispatch_done.is_set() or self._in_flight_rollouts > 0:

        try:
          completed = await self.engine.poll_rollouts()
          if isinstance(completed, list) and completed:
            # TODO: Fault-tolerance must either decrement `_in_flight_rollouts` for failed
            # requests or retry them internally. Otherwise, a dropped RPC will cause
            # `_in_flight_rollouts` to never reach 0, hanging the EOF cascade.
            self._in_flight_rollouts -= len(completed)
            for item in completed:
              await self.raw_q.put(item)
        except Exception as exc:  # pylint: disable=broad-exception-caught
          logging.warning("Error in polling_stage: %s", exc)
          await asyncio.sleep(0.01)
    finally:
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
        except Exception as exc:
          logging.exception("Error in critique_stage get_group: %s", exc)
          break

        rewards = []
        for item in group:
          if self.reward_fns:
            r = sum(fn(item) for fn in self.reward_fns)
          else:
            r = getattr(item.traj, "reward", 0.0)
          rewards.append(float(r))

        trainer_payloads = self.algo.create_trainer_payloads(
            group, rewards=rewards
        )
        for idx, payload in enumerate(trainer_payloads):
          reward_val = rewards[idx] if idx < len(rewards) else 0.0
          src_item = group[idx] if idx < len(group) else None
          src_traj = getattr(src_item, "traj", None)
          raw_status = (
              getattr(src_traj, "status", None) if src_traj else None
          ) or getattr(src_item, "status", None)
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
          raw_steps = getattr(src_traj, "steps", None) if src_traj else None
          steps = raw_steps if isinstance(raw_steps, list) else []
          src_metadata = getattr(src_item, "metadata", None)
          metadata = dict(src_metadata) if src_metadata else {}
          item = datatypes.TrajectoryItem(
              prompt_id=getattr(src_item, "prompt_id", ""),
              group_index=getattr(src_item, "group_index", 0),
              start_step=0,
              traj=datatypes.Trajectory(
                  reward=reward_val,
                  status=status,
                  steps=steps,
              ),
              prompt_tokens=getattr(src_item, "prompt_tokens", None),
              completion_tokens=getattr(src_item, "completion_tokens", None),
              action_mask=getattr(src_item, "action_mask", None),
              policy_version=getattr(src_item, "policy_version", 0),
              metadata=metadata,
              # TODO: b/552087289 - Stream RLTrainerPayload directly instead of
              # re-wrapping in TrajectoryItem.
          )
          item.payload = payload  # pyrefly: ignore[missing-attribute]
          await self.scored_q.put(item)
    finally:
      await self.scored_q.close()

  def _collect_and_log_step_metrics(
      self,
      *,
      all_step_items: Sequence[datatypes.TrajectoryItem],
      step_rewards: Sequence[float],
      step_result: Any = None,
      trainer_metrics: Any = None,
      num_rollouts: int,
      num_microbatches: int,
      step_time_sec: float,
      consumed_policy_version: int,
      log_step: int,
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
    for item in all_step_items:
      p_len = None
      prompt_tokens = getattr(item, "prompt_tokens", None)
      if prompt_tokens is not None:
        p_len = len(prompt_tokens)
      elif (
          hasattr(item, "payload")
          and getattr(item.payload, "token_ids", None) is not None
      ):
        token_mask = getattr(item.payload, "token_mask", None)
        loss_mask = getattr(item.payload, "loss_mask", None)
        if token_mask is not None and loss_mask is not None:
          p_len = int(np.sum((token_mask > 0) & (loss_mask == 0)))
        elif token_mask is not None:
          p_len = int(np.sum(token_mask > 0))

      c_len = None
      completion_tokens = getattr(item, "completion_tokens", None)
      if completion_tokens is not None:
        c_len = len(completion_tokens)
      elif (
          hasattr(item, "payload")
          and getattr(item.payload, "loss_mask", None) is not None
      ):
        c_len = int(np.sum(item.payload.loss_mask > 0))

      if p_len is not None:
        prompt_lengths.append(p_len)
      if c_len is not None:
        completion_lengths.append(c_len)
      if p_len is not None and c_len is not None:
        total_lengths.append(p_len + c_len)

      traj = getattr(item, "traj", None)
      steps = getattr(traj, "steps", None) if traj else None
      if steps and len(steps) > 0:
        turns_list.append(len(steps))

      status = (getattr(traj, "status", None) if traj else None) or getattr(
          item, "status", None
      )
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

    if prompt_lengths:
      self.metrics_logger.log(
          self.metrics_prefix,
          "rollout/prompt_length_mean",
          float(np.mean(prompt_lengths)),
          self.mode,
          log_step,
      )
    if completion_lengths:
      self.metrics_logger.log(
          self.metrics_prefix,
          "rollout/completion_length_mean",
          float(np.mean(completion_lengths)),
          self.mode,
          log_step,
      )
    if total_lengths:
      self.metrics_logger.log(
          self.metrics_prefix,
          "rollout/total_tokens_mean",
          float(np.mean(total_lengths)),
          self.mode,
          log_step,
      )
    if turns_list:
      self.metrics_logger.log(
          self.metrics_prefix,
          "rollout/num_turns_mean",
          float(np.mean(turns_list)),
          self.mode,
          log_step,
      )
    if successes:
      self.metrics_logger.log(
          self.metrics_prefix,
          "rollout/success_rate",
          float(np.mean(successes)),
          self.mode,
          log_step,
      )
    if staleness_list:
      staleness_stats = {
          "staleness_mean": float(np.mean(staleness_list)),
          "staleness_max": float(np.max(staleness_list)),
          "staleness_min": float(np.min(staleness_list)),
      }
      for tag, val in staleness_stats.items():
        self.metrics_logger.log(
            self.metrics_prefix, f"rollout/{tag}", val, self.mode, log_step
        )

    # --- 2. Reward Metrics ---
    reward_mean = float(np.mean(step_rewards)) if step_rewards else 0.0
    reward_std = float(np.std(step_rewards)) if step_rewards else 0.0
    reward_min = float(np.min(step_rewards)) if step_rewards else 0.0
    reward_max = float(np.max(step_rewards)) if step_rewards else 0.0
    reward_sum = float(np.sum(step_rewards)) if step_rewards else 0.0
    if step_rewards:
      reward_stats = {
          "mean": reward_mean,
          "std": reward_std,
          "min": reward_min,
          "max": reward_max,
          "sum": reward_sum,
      }
      for tag, val in reward_stats.items():
        self.metrics_logger.log(
            self.metrics_prefix, f"rewards/{tag}", val, self.mode, log_step
        )

    # --- 3. Orchestrator Metrics ---
    orchestrator_stats = {
        "policy_version": float(self.policy_version),
        "num_rollouts": float(num_rollouts),
        "num_microbatches": float(num_microbatches),
        "step_time_sec": float(step_time_sec),
    }
    for tag, val in orchestrator_stats.items():
      self.metrics_logger.log(
          self.metrics_prefix, f"orchestrator/{tag}", val, self.mode, log_step
      )

    # --- 4. Trainer Metrics ---
    loss_val = None
    perplexity_val = None
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
          "loss", scalar_metrics.pop("trainer/loss", None)
      )
      if raw_loss is None and "loss" in weighted_metrics:
        raw_loss = weighted_metrics.pop("loss")
      elif raw_loss is None and "trainer/loss" in weighted_metrics:
        raw_loss = weighted_metrics.pop("trainer/loss")

      loss_val = _extract_scalar(raw_loss)
      if loss_val is not None:
        self.metrics_logger.log(
            self.metrics_prefix, "trainer/loss", loss_val, self.mode, log_step
        )
        perplexity_val = float(np.exp(loss_val))
        self.metrics_logger.log(
            self.metrics_prefix,
            "trainer/perplexity",
            perplexity_val,
            self.mode,
            log_step,
        )

      # Learning Rate
      raw_lr = scalar_metrics.pop(
          "learning_rate", scalar_metrics.pop("trainer/learning_rate", None)
      )
      lr_val = _extract_scalar(raw_lr)
      if lr_val is not None:
        self.metrics_logger.log(
            self.metrics_prefix,
            "trainer/learning_rate",
            lr_val,
            self.mode,
            log_step,
        )

      # Grad Norm
      raw_gn = scalar_metrics.pop(
          "grad_norm", scalar_metrics.pop("trainer/grad_norm", None)
      )
      gn_val = _extract_scalar(raw_gn)
      if gn_val is not None:
        self.metrics_logger.log(
            self.metrics_prefix,
            "trainer/grad_norm",
            gn_val,
            self.mode,
            log_step,
        )

      # Auxiliary weighted metrics
      for k, v in weighted_metrics.items():
        val = _extract_scalar(v)
        if val is not None:
          metric_key = k if k.startswith("trainer/") else f"trainer/{k}"
          self.metrics_logger.log(
              self.metrics_prefix, metric_key, val, self.mode, log_step
          )

      # Auxiliary scalar metrics
      for k, v in scalar_metrics.items():
        if k in ("perplexity", "trainer/perplexity"):
          continue
        val = _extract_scalar(v)
        if val is not None:
          metric_key = k if k.startswith("trainer/") else f"trainer/{k}"
          self.metrics_logger.log(
              self.metrics_prefix, metric_key, val, self.mode, log_step
          )

    return {
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "loss_val": loss_val,
        "perplexity_val": perplexity_val,
    }

  async def train_stage(self) -> None:
    """Stage 3: Streaming gradient accumulation with RLTrainerPayloads."""
    assert self.engine is not None

    while self.max_steps is None or self._step < self.max_steps:
      current_step = self._step
      step_start_time = time.monotonic()
      consumed_policy_version = self.policy_version

      uncommitted_groups = []
      step_result = None
      trainer_metrics = None
      step_rewards = []
      num_microbatches = 0
      num_rollouts = 0
      all_step_items = []
      scored_items = []
      groups_consumed = 0

      groups_per_assembly_batch = self.assembler.groups_per_assembly_batch

      while groups_consumed < self.mini_batch_size:
        groups_to_fetch = min(
            groups_per_assembly_batch, self.mini_batch_size - groups_consumed
        )
        scored_items = await self.scored_q.get_batch(num_groups=groups_to_fetch)
        if not scored_items:
          break

        if groups_consumed == 0 and self.on_step_begin:
          self.on_step_begin(current_step)

        groups_consumed += groups_to_fetch
        uncommitted_groups.append(scored_items)
        all_step_items.extend(scored_items)
        num_rollouts += len(scored_items)
        for item in scored_items:
          step_rewards.append(float(getattr(item.traj, "reward", 0.0)))

        prompt_ids = list(
            dict.fromkeys(
                item.prompt_id
                for item in scored_items
                if getattr(item, "prompt_id", None)
            )
        )
        payloads = [getattr(item, "payload", None) for item in scored_items]
        # TODO(tunix-dev): Implement streaming microbatch assembly to overlap
        # packing with trainer execution.
        microbatches = self.assembler.pack(payloads)  # pyrefly: ignore[bad-argument-type]
        logging.info(
            "Packed %d prompt groups into %d microbatches (total_rollouts=%d)."
            " All prompt groups ids packed: %s",
            len(prompt_ids),
            len(microbatches),
            len(payloads),
            logging_utils.summarize_list(prompt_ids),
        )
        if getattr(self.algo, "requires_reference_kl", False):
          scored_microbatches = []
          for batch in microbatches:
            if not isinstance(batch, datatypes.RLTrainerPayload):
              raise TypeError(
                  "Reference KL requires an assembler that returns "
                  "datatypes.RLTrainerPayload microbatches; got "
                  f"{type(batch).__name__}."
              )
            ref_logps = await self.engine.per_token_logps(
                datatypes.Role.REFERENCE, items=batch
            )
            scored_microbatches.append(
                batch_assembly.with_ref_per_token_logps(batch, ref_logps)
            )
          microbatches = scored_microbatches

        num_microbatches += len(microbatches)
        is_final_group = groups_consumed >= self.mini_batch_size
        for batch_idx, batch in enumerate(microbatches):
          is_final_batch = is_final_group and batch_idx == len(microbatches) - 1
          step_result = await self.engine.train_step(
              batch,
              role=datatypes.Role.ACTOR,
              accumulate_gradients=True,
              apply_optimizer=is_final_batch,
          )
          if is_final_batch:
            # TODO(tunix-dev): Current checkpoint and metrics logic only works
            # for fully on-policy. We need to come up with a solution for
            # semi-off-policy where a single full batch has multiple mini
            # batches.
            trainer_metrics = await self.engine.get_metrics(
                role=datatypes.Role.ACTOR
            )
            # TODO(tunix-dev): Configurable checkpointing frequency. Today we
            # checkpoint at the same frequency as the weight update.
            # TODO(tunix-dev): For now any failures in save_checkpoint will
            # abort the entire program. Make it configurable on whether to fail
            # or continue.
            await self.engine.save_checkpoint(
                role=datatypes.Role.ACTOR,
                metadata={
                    "step": self.step + 1,
                    "policy_version": self.policy_version,
                    "num_rollouts": num_rollouts,
                    "num_microbatches": num_microbatches,
                },
            )

      if not scored_items:
        # TODO: We currently silently drop in-progress partial microbatch accumulators if
        # the dataset ends early. We may need to force-apply gradients here instead.
        logging.info(
            "Dataset exhausted at step %d before max_steps.", current_step
        )
        break

      if self.sync_weights:
        new_version = await self.engine.sync_weights(role=datatypes.Role.ACTOR)
        self.policy_version = (
            new_version if new_version is not None else self.policy_version + 1
        )
      else:
        self.policy_version += 1

      self.scored_q.commit(current_step, groups=uncommitted_groups)

      assert (
          self._dispatch_capacity is not None
      ), "run_async must initialize capacity."
      for _ in range(groups_consumed):
        self._dispatch_capacity.release()

      step_time_sec = time.monotonic() - step_start_time

      metrics_summary = self._collect_and_log_step_metrics(
          all_step_items=all_step_items,
          step_rewards=step_rewards,
          step_result=step_result,
          trainer_metrics=trainer_metrics,
          num_rollouts=num_rollouts,
          num_microbatches=num_microbatches,
          step_time_sec=step_time_sec,
          consumed_policy_version=consumed_policy_version,
          log_step=current_step,
      )

      self.last_step_result = RLStepResult(
          step=current_step,
          policy_version=self.policy_version,
          num_rollouts=num_rollouts,
          num_microbatches=num_microbatches,
          reward_mean=metrics_summary["reward_mean"],
          reward_std=metrics_summary["reward_std"],
          train_result=step_result,
      )

      loss_val = metrics_summary["loss_val"]
      perplexity_val = metrics_summary["perplexity_val"]
      if self.mode == Mode.TRAIN:
        logging.info(
            "Train step %d - loss: %s - reward_mean: %.4f - perplexity: %s -"
            " step_time: %.2fs",
            current_step,
            f"{loss_val:.4f}" if loss_val is not None else "N/A",
            metrics_summary["reward_mean"],
            f"{perplexity_val:.4f}" if perplexity_val is not None else "N/A",
            step_time_sec,
        )

      if self.on_step_end:
        self.on_step_end(current_step, step_result)
      self._step += 1

  async def run_async(
      self,
      engine: rl_engine_interface.AbstractRLEngine,
      **kwargs: Any,
  ) -> None:
    """Launches all stages concurrently on event loop."""
    del kwargs
    self.engine = engine
    logging.info("Starting StandardRLProgram concurrent stages...")

    if self.sync_weights:
      await engine.prepare_rollout_policy(
          role=datatypes.Role.ACTOR,
          sync_weights=True,
          policy_version=self.policy_version,
      )

    max_groups_ahead = self.mini_batch_size * (self.max_staleness + 1)
    self._dispatch_capacity = asyncio.Semaphore(max_groups_ahead)

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
      raise
    finally:
      for task in tasks:
        if not task.done():
          task.cancel()

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
