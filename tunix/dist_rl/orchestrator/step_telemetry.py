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

"""Per-step telemetry for `StandardRLProgram`.

Types and pure functions that turn a train step's phase times and scored
rollouts into the metrics and log lines its train stage emits. None of them
touch the program's state, so they are tested here rather than through a
running program.
"""

import collections
from collections.abc import Mapping, Sequence
import dataclasses
from typing import Any

import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.common import lineage


@dataclasses.dataclass(kw_only=True)
class StepTimes:
  """Seconds the train stage waited on each phase, logged as `perf/*`.

  `exposed_generation_time`, `policy_training_time` and `weight_sync_time`
  match the MLPerf branch so dashboards compare across branches. Each is how
  long the orchestrator awaited that phase, not how long its work ran. The
  trainer answers `train_step` once its programs are dispatched (JAX runs them
  asynchronously), so device time lands in whichever later call first blocks
  on the trainer's results: with `PeftTrainerV2`, `TrainerWorker.update` at the
  end of `train_step(..., apply_optimizer=True)` (it flushes and reads the
  step's losses and gradient norm), counted in `policy_training_time`, or the
  actor logprob pass of a standalone sampler
  agreement check. Training that runs while the stage waits for scored groups
  or packs is hidden behind that waiting and packing, and `save_checkpoint`
  and `sync_weights` absorb training only when nothing before them blocked.
  So the phase split shows where the orchestrator waited, not where the
  device spent its time. The phases need not add up to `step_time_sec`: a
  microbatch's `train_step` keeps running in the background while the next
  microbatch is packed, reference-scored or awaited, and packing
  (`efficiency/packing/*`), committing and metric logging belong to no phase.
  Times are wall clock on the event loop, so they also absorb any stall from
  other stages sharing it.

  Attributes:
    exposed_generation_time: Waiting for scored rollout groups.
    policy_training_time: Union of (possibly overlapping) microbatch
      `train_step` intervals, each instant counted once (including `update` and
      metric flush on the final microbatch).
    reference_logps_time: Reference-model logprob passes for the KL penalty.
    sampler_agreement_time: `StandardRLProgram`'s standalone sampler-agreement
      pass, which computes trainer logprobs to compare with the sampler's.
    checkpoint_time: The `save_checkpoint` call. With async checkpointing that
      is only its blocking part (waiting out a previous save, copying the state
      off device); the write itself runs in the background, uncounted. About 0
      on steps the save policy skips.
    weight_sync_time: Syncing the new policy to the rollout workers.
  """

  exposed_generation_time: float = 0.0
  policy_training_time: float = 0.0
  reference_logps_time: float = 0.0
  sampler_agreement_time: float = 0.0
  checkpoint_time: float = 0.0
  weight_sync_time: float = 0.0


@dataclasses.dataclass(frozen=True, kw_only=True)
class RolloutWindow:
  """Dispatch, receive, and finish times of a rollout, and its worker.

  Attributes:
    dispatch_time_s: Unix time of the rollout's `engine.dispatch` event.
    finish_time_s: Unix time of its last `worker.rollout` event.
    worker_id: The rollout worker that handed the trajectory back.
    received_time_s: Unix time of its last `worker.received` event from
      `worker_id`, or None when not stamped.
    execution_time_s: Worker-side episode execution duration in seconds, or None
      when not stamped.
    batch_idx: Prompt cohort index from `engine.dispatch` or item metadata, or
      None when untagged.
  """

  dispatch_time_s: float
  finish_time_s: float
  worker_id: str
  received_time_s: float | None = None
  execution_time_s: float | None = None
  batch_idx: int | None = None

  @property
  def duration_s(self) -> float:
    """Seconds from dispatch to finish, including cross-host clock skew."""
    return self.finish_time_s - self.dispatch_time_s


def _rollout_window(item: datatypes.TrajectoryItem) -> RolloutWindow | None:
  """Reads a rollout's dispatch, received, and finish times from its lineage.

  `DistributedRLEngine` records `engine.dispatch` when it builds the request,
  the rollout worker records `worker.received` when `generate()` receives the
  request, and records `worker.rollout` when it hands the finished trajectory
  back. `dispatch_time_s` and `received_time_s` / `finish_time_s` come from
  different hosts' clocks, so `duration_s` and dispatch delay include their
  skew; `received_time_s` and `finish_time_s` share the winning worker's clock,
  and `execution_time_s` (when stamped by the worker) measures episode execution
  on the worker's own monotonic clock.

  Args:
    item: A scored rollout.

  Returns:
    The rollout's window, or None when it carries no lineage or either
    `engine.dispatch` or `worker.rollout` is missing.
  """
  lineage_ctx: lineage.LineageContext | None = item.metadata.get("lineage")
  if lineage_ctx is None:
    return None
  dispatched = [
      event
      for event in lineage_ctx.events
      if event.component == "engine.dispatch"
  ]
  finished = [
      event
      for event in lineage_ctx.events
      if event.component == "worker.rollout"
  ]
  if not dispatched or not finished:
    return None
  dispatch_attrs = dispatched[0].attributes
  finish_attrs = finished[-1].attributes
  worker_id = str(finish_attrs["worker_id"])
  received = [
      event
      for event in lineage_ctx.events
      if event.component == "worker.received"
      and event.attributes.get("worker_id") is not None
      and str(event.attributes["worker_id"]) == worker_id
  ]
  raw_exec = finish_attrs.get("execution_time_s")
  raw_batch_idx = dispatch_attrs.get("batch_idx")
  if raw_batch_idx is None:
    raw_batch_idx = item.metadata.get("batch_idx")
  return RolloutWindow(
      dispatch_time_s=dispatched[0].timestamp_s,
      finish_time_s=finished[-1].timestamp_s,
      worker_id=worker_id,
      received_time_s=received[-1].timestamp_s if received else None,
      execution_time_s=float(raw_exec) if raw_exec is not None else None,
      batch_idx=int(raw_batch_idx) if raw_batch_idx is not None else None,
  )


def _env_time_s(traj: Mapping[str, Any]) -> float | None:
  """Sums `reset_latency`, `step_latency`, and `close_latency` from `env_time`."""
  env_time = traj.get("env_time")
  if env_time is None:
    return None
  return float(
      env_time["reset_latency"]
      + sum(env_time["step_latency"])
      + env_time["close_latency"]
  )


def _reward_time_s(traj: Mapping[str, Any]) -> float | None:
  """Reads `reward_latency` from `reward_time`."""
  reward_time = traj.get("reward_time")
  if reward_time is None:
    return None
  return float(reward_time["reward_latency"])


def _num_turns(traj: Mapping[str, Any]) -> int | None:
  """Counts the assistant turns in a Token-mode rollout's message list.

  Args:
    traj: A scored rollout's trajectory dict.

  Returns:
    The number of assistant messages in `conversation_text`, or None when the
    rollout carries no message list.
  """
  conversation = traj.get("conversation_text")
  if conversation is None:
    return None
  return sum(1 for message in conversation if message["role"] == "assistant")


@dataclasses.dataclass(frozen=True, kw_only=True)
class RolloutFacts:
  """What one scored rollout contributes to its step's telemetry.

  Read once per rollout, so the health metrics and the slowest-rollouts line
  share the lineage scan and turn count.

  Attributes:
    traj_id: The rollout's trajectory id.
    status: Name of the `TrajectoryStatus` the rollout ended in.
    num_turns: Assistant turns in `conversation_text`, or None without one.
    turn_lengths: Tokens sampled per assistant turn, from the
      `assistant_turn_lengths` the rollout collector records, or None without
      them. They do not depend on the training mask, so rollouts that were
      masked out still have them.
    window: Dispatch and finish times from lineage, or None without them.
    env_time_s: Total environment interaction wall time (`reset` + `step` +
      `close`) in seconds, or None without `env_time`.
    reward_time_s: Final reward evaluation wall time in seconds, or None without
      `reward_time`.
  """

  traj_id: str
  status: str
  num_turns: int | None
  turn_lengths: Sequence[int] | None
  window: RolloutWindow | None
  env_time_s: float | None = None
  reward_time_s: float | None = None

  @classmethod
  def of(cls, item: datatypes.TrajectoryItem) -> "RolloutFacts":
    """Reads the facts off a rollout as `critique_stage` emits it.

    Engine error placeholders carry no conversation and lineage is optional
    telemetry, so those facts may be None; a missing input is never read as
    zero.

    Args:
      item: A scored rollout.

    Returns:
      The rollout's facts.

    Raises:
      KeyError: If the rollout has no `status`, which `critique_stage` always
        sets.
    """
    traj = item.traj
    return cls(
        traj_id=item.traj_id,
        status=traj["status"].name,
        num_turns=_num_turns(traj),
        turn_lengths=traj.get("assistant_turn_lengths"),
        window=_rollout_window(item),
        env_time_s=_env_time_s(traj),
        reward_time_s=_reward_time_s(traj),
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class RolloutHealth:
  """A step's rollout status, turn and latency metrics.

  Attributes:
    metrics: Fully-qualified metric name to step value; see `rollout_health`.
    status_counts: Rollouts per `TrajectoryStatus` name, every status listed in
      enum order, zeros included.
  """

  metrics: dict[str, float]
  status_counts: dict[str, int]


def _add_percentiles(
    metrics: dict[str, float],
    prefix: str,
    values: Sequence[float | int] | np.ndarray,
    percentiles: Sequence[int] = (50, 90, 99),
    *,
    include_mean: bool = False,
) -> None:
  """Records `prefix/{mean,pXX,...,max}` summary statistics into `metrics`."""
  if include_mean:
    metrics[f"{prefix}/mean"] = float(np.mean(values))
  for p, val in zip(
      percentiles, np.percentile(values, percentiles), strict=True
  ):
    metrics[f"{prefix}/p{p}"] = float(val)
  metrics[f"{prefix}/max"] = float(np.max(values))


def rollout_health(facts: Sequence[RolloutFacts]) -> RolloutHealth:
  """Computes the step's rollout status, turn and latency metrics.

  Covers the rollouts the step consumed; filtered groups are not included.

    * `rollout/status/<status>_frac`: share of rollouts that ended in each
      `TrajectoryStatus`, all reported every step so dashboard series stay
      continuous. `critique_stage` maps a missing or unrecognized producer
      status to RUNNING, so a nonzero `running_frac` points at the producer.
      Unlike `rollout/success_rate`, these keep such rollouts in the
      denominator.
    * `rollout/success_rate`: share of non-`RUNNING` rollouts that ended in
      `SUCCEEDED`.
    * `rollout/num_turns_{mean,max}`: assistant turns per rollout, counted in
      `conversation_text`.
    * `rollout/turn_length/{mean,p50,p99,max}`: tokens sampled per assistant
      turn, pooled over the step.
    * `rollout/env_time_sec/{mean,p50,p99,max}` and
      `rollout/reward_time_sec/{mean,p50,p99,max}`: environment (`reset` +
      `step` + `close`) and final-reward wall time per rollout.
    * `rollout/duration_sec/{p50,p90,p99,max}`: seconds from the engine
      building a rollout's request to its worker handing the trajectory back,
      as the orchestrator sees it: queueing before the worker starts, pauses
      for weight sync and clock skew between hosts are all included.
      `rollout/execution_sec/{p50,p90,p99,max}` is the worker-side episode
      execution duration, and `rollout/queue_wait_sec/{p50,p90,p99,max}` is
      `max(0, duration_sec - execution_sec)`.
      `rollout/dispatch_delay_sec/{p50,p90,p99,max}` is `max(0, received_time_s
      - dispatch_time_s)`, capturing transport transit and any pre-reroute wait
      time on failed attempts before the winning worker received the request,
      and `rollout/worker_queue_wait_sec/{p50,p90,p99,max}` is
      `max(0, (finish_time_s - received_time_s) - execution_sec)` measured on
      the winning worker's clock (`duration ≈ dispatch_delay +
      worker_queue_wait + execution`). `rollout/tail_sec` is max - p50, and
      `rollout/batch_wall_sec` is the maximum cohort span from first dispatch
      to last finish within each `batch_idx` cohort.

  The turn and duration families skip rollouts that lack their inputs and are
  omitted when no rollout has them.

  Args:
    facts: The step's scored rollouts, read with `RolloutFacts.of`.

  Returns:
    The metrics, and the rollout count per status they were derived from.
  """
  status_counts = {status.name: 0 for status in datatypes.TrajectoryStatus}
  turns: list[int] = []
  turn_lengths: list[int] = []
  windows: list[RolloutWindow] = []
  env_times: list[float] = []
  reward_times: list[float] = []
  for fact in facts:
    status_counts[fact.status] += 1
    if fact.num_turns is not None:
      turns.append(fact.num_turns)
    if fact.turn_lengths is not None:
      turn_lengths.extend(fact.turn_lengths)
    if fact.window is not None:
      windows.append(fact.window)
    if fact.env_time_s is not None:
      env_times.append(fact.env_time_s)
    if fact.reward_time_s is not None:
      reward_times.append(fact.reward_time_s)

  metrics: dict[str, float] = {}
  if facts:
    for name, count in status_counts.items():
      metrics[f"rollout/status/{name.lower()}_frac"] = count / len(facts)
    finished = (
        len(facts) - status_counts[datatypes.TrajectoryStatus.RUNNING.name]
    )
    if finished > 0:
      metrics["rollout/success_rate"] = (
          status_counts[datatypes.TrajectoryStatus.SUCCEEDED.name] / finished
      )
  if turns:
    metrics["rollout/num_turns_mean"] = float(np.mean(turns))
    metrics["rollout/num_turns_max"] = float(np.max(turns))
  if turn_lengths:
    _add_percentiles(
        metrics,
        "rollout/turn_length",
        turn_lengths,
        (50, 99),
        include_mean=True,
    )
  if env_times:
    _add_percentiles(
        metrics, "rollout/env_time_sec", env_times, (50, 99), include_mean=True
    )
  if reward_times:
    _add_percentiles(
        metrics,
        "rollout/reward_time_sec",
        reward_times,
        (50, 99),
        include_mean=True,
    )
  if windows:
    durations = np.array([window.duration_s for window in windows])
    _add_percentiles(metrics, "rollout/duration_sec", durations)
    metrics["rollout/tail_sec"] = (
        metrics["rollout/duration_sec/max"]
        - metrics["rollout/duration_sec/p50"]
    )
    cohorts: collections.defaultdict[int | None, list[RolloutWindow]] = (
        collections.defaultdict(list)
    )
    for window in windows:
      cohorts[window.batch_idx].append(window)
    metrics["rollout/batch_wall_sec"] = float(
        max(
            max(w.finish_time_s for w in cohort)
            - min(w.dispatch_time_s for w in cohort)
            for cohort in cohorts.values()
        )
    )
    dispatch_delays = [
        max(0.0, window.received_time_s - window.dispatch_time_s)
        for window in windows
        if window.received_time_s is not None
    ]
    if dispatch_delays:
      _add_percentiles(metrics, "rollout/dispatch_delay_sec", dispatch_delays)
    exec_durations = [
        window.execution_time_s
        for window in windows
        if window.execution_time_s is not None
    ]
    if exec_durations:
      _add_percentiles(metrics, "rollout/execution_sec", exec_durations)
      queue_waits = [
          max(0.0, window.duration_s - window.execution_time_s)
          for window in windows
          if window.execution_time_s is not None
      ]
      _add_percentiles(metrics, "rollout/queue_wait_sec", queue_waits)
    worker_queue_waits = [
        max(
            0.0,
            (window.finish_time_s - window.received_time_s)
            - window.execution_time_s,
        )
        for window in windows
        if window.received_time_s is not None
        and window.execution_time_s is not None
    ]
    if worker_queue_waits:
      _add_percentiles(
          metrics, "rollout/worker_queue_wait_sec", worker_queue_waits
      )
  return RolloutHealth(metrics=metrics, status_counts=status_counts)


def slowest_rollouts_summary(
    facts: Sequence[RolloutFacts], k: int
) -> str | None:
  """Describes the `k` rollouts that took longest from dispatch to finish.

  Args:
    facts: The step's scored rollouts, read with `RolloutFacts.of`.
    k: How many rollouts to describe.

  Returns:
    A one-line summary, or None when no rollout carries lineage.
  """
  timed: list[tuple[RolloutWindow, RolloutFacts]] = []
  for fact in facts:
    if fact.window is not None:
      timed.append((fact.window, fact))
  if not timed:
    return None
  timed.sort(key=lambda pair: pair[0].duration_s, reverse=True)
  described = []
  for window, fact in timed[:k]:
    fields = [
        fact.traj_id,
        f"dur={window.duration_s:.1f}s",
    ]
    if window.execution_time_s is not None:
      fields.append(f"exec={window.execution_time_s:.1f}s")
    if fact.env_time_s is not None:
      fields.append(f"env={fact.env_time_s:.1f}s")
    fields.append(f"status={fact.status}")
    if fact.num_turns is not None:
      fields.append(f"turns={fact.num_turns}")
    if fact.turn_lengths:
      fields.append(f"max_turn_length={max(fact.turn_lengths)}")
    fields.append(f"worker={window.worker_id}")
    described.append(" ".join(fields))
  return f"slowest {len(described)} of {len(timed)}: " + " | ".join(described)
