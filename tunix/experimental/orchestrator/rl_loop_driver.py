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

"""The synchronous RL loop driver: one training step end to end.

Ties the control plane, data plane, algorithm adapter, and sequencer into a
single step over the (fake or real) workers resolved from the registry:

  requests -> acquire credits + open groups -> rollout generate -> admit ->
  drain complete groups -> (staleness gate) -> postprocess -> enqueue ->
  release credits -> plan micro-steps -> fwd_bwd x N -> update (x mu replays).

This is the v1 synchronous shape (staleness gating optional; one rollout worker
and one trainer worker). Weight sync, metrics pumping, eval, and checkpointing
are separate stages layered on later; the loss is configured on the trainer by
the caller before stepping.
"""

import dataclasses
from typing import Any

from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import algorithm_adapter
from tunix.experimental.orchestrator import dispatch_credits
from tunix.experimental.orchestrator import group_assembler
from tunix.experimental.orchestrator import micro_batch_sequencer
from tunix.experimental.orchestrator import train_batch_queue
from tunix.experimental.orchestrator import worker_registry

_UNBOUNDED = 1 << 30


@dataclasses.dataclass(kw_only=True)
class StepOutcome:
  """Result of one training step.

  Attributes:
    step: The step index that just ran.
    num_groups_trained: Micro-batches (groups) fed to the trainer this step.
    num_groups_dropped: Complete groups dropped by the staleness gate.
    num_updates: Optimizer updates applied (one per micro-batch iteration).
    update_results: The trainer's UpdateResult per iteration.
  """

  step: int
  num_groups_trained: int
  num_groups_dropped: int
  num_updates: int
  update_results: list[datatypes.UpdateResult]


class RLLoopDriver:
  """Drives one synchronous RL step over the registered workers."""

  def __init__(
      self,
      *,
      registry: worker_registry.WorkerRegistry,
      adapter: algorithm_adapter.AlgorithmAdapter,
      tokenizer_info: datatypes.TokenizerInfo,
      shape_config: datatypes.ShapeConfig,
      group_size: int,
      loss_agg_mode: str = "token-mean",
      num_microbatch_iterations: int = 1,
      dispatch_credit_capacity: int | None = None,
      train_queue_size: int | None = None,
      max_staleness: int | None = None,
  ):
    self._registry = registry
    self._adapter = adapter
    self._tokenizer_info = tokenizer_info
    self._shape_config = shape_config
    self._group_size = group_size
    self._loss_agg_mode = loss_agg_mode
    self._num_iterations = num_microbatch_iterations
    self._max_staleness = max_staleness
    self._assembler = group_assembler.GroupAssembler(min_group_size=group_size)
    self._credits = dispatch_credits.DispatchCredits(
        capacity=dispatch_credit_capacity
        if dispatch_credit_capacity is not None
        else _UNBOUNDED
    )
    self._queue = train_batch_queue.TrainBatchQueue(
        maxsize=train_queue_size if train_queue_size is not None else _UNBOUNDED
    )
    self._step = 0
    self._policy_version = 0

  @property
  def step(self) -> int:
    return self._step

  @property
  def credits(self) -> dispatch_credits.DispatchCredits:
    return self._credits

  def _single_worker(self, role: str):
    members = self._registry.group(role).members()
    if not members:
      raise ValueError(f"no worker registered for role {role!r}")
    return members[0]

  async def run_step(self, rows: list[dict[str, Any]]) -> StepOutcome:
    """Runs one full training step over `rows` (a batch of prompt dicts)."""
    rollout = self._single_worker("rollout")
    trainer = self._single_worker("trainer")

    # 1. Algorithm builds one group of G requests per prompt.
    groups = self._adapter.make_trajectory_requests(rows, self._step)

    # 2. Acquire a credit per request, register each group, gather requests.
    requests = []
    for records in groups:
      if not self._credits.try_acquire(len(records)):
        raise RuntimeError("out of dispatch credits")
      self._assembler.open_group(records)
      requests.extend(record.request for record in records)

    # 3. Generate and admit every result (dedup + incarnation gate in-ledger).
    results = await rollout.generate(requests)
    for result in results:
      self._assembler.admit(result)

    # 4. Drain complete groups; gate staleness; postprocess; enqueue; release.
    num_dropped = 0
    for group in self._assembler.drain_ready():
      terminal_credits = len(group)
      if self._max_staleness is not None and group_assembler.is_group_stale(
          group,
          current_version=self._policy_version,
          max_staleness=self._max_staleness,
      ):
        self._credits.release(terminal_credits)
        num_dropped += 1
        continue
      train_example = self._adapter.postprocess_group(
          group,
          tokenizer_info=self._tokenizer_info,
          shape_config=self._shape_config,
      )
      if not self._queue.put(train_example):
        raise RuntimeError("train batch queue is full")
      self._credits.release(terminal_credits)

    # 5. Drain the queue into the mini-batch's micro-batches.
    micro_batches = []
    while (item := self._queue.try_get()) is not None:
      micro_batches.append(item)

    # 6. mu replays, each a full grad-accum cycle over the micro-batches (I7).
    update_results: list[datatypes.UpdateResult] = []
    if micro_batches:
      for iteration in range(self._num_iterations):
        accum_id = f"{self._step}/{iteration}"
        micro_steps = micro_batch_sequencer.plan_micro_steps(
            micro_batches, accum_id=accum_id, loss_agg_mode=self._loss_agg_mode
        )
        for micro_step in micro_steps:
          trainer.fwd_bwd(
              micro_step.payload,
              accum_id=micro_step.accum_id,
              micro_index=micro_step.micro_index,
              loss_scale=micro_step.loss_scale,
          )
        update_results.append(
            trainer.update(
                accum_id=accum_id, expected_micro_steps=len(micro_steps)
            )
        )

    outcome = StepOutcome(
        step=self._step,
        num_groups_trained=len(micro_batches),
        num_groups_dropped=num_dropped,
        num_updates=len(update_results),
        update_results=update_results,
    )
    self._step += 1
    return outcome
