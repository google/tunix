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

"""A contract-conformant fake TrainerWorker implementing the receipt protocol.

Does the gradient-accumulation *bookkeeping* with no real math: per-accum_id
micro-step dedup, {0..N-1} verification on update, a cached UpdateResult on
retry, and reset_accumulation. It lets the orchestrator (and the TrainerWorker
contract suite) exercise the accumulation protocol before the real trainer lands.
"""

from typing import Any

from tunix.experimental.common import datatypes
from tunix.experimental.metrics import metrics
from tunix.experimental.worker import abstract_worker


class FakeTrainerWorker(abstract_worker.Worker):
  """Deterministic fake TrainerWorker (receipt protocol, no math)."""

  def __init__(self, worker_id: str = "fake-trainer"):
    self._worker_id = worker_id
    self._running = False
    self._step = 0
    self._receipts: dict[str, dict[int, datatypes.StepReceipt]] = {}
    self._updates: dict[str, datatypes.UpdateResult] = {}
    self._checkpoint: dict[str, Any] = {}

  def initialize(self) -> None:
    pass

  def compile(self, shape_config: datatypes.ShapeConfig) -> None:
    del shape_config

  def start(self) -> None:
    self._running = True

  def stop(self) -> None:
    self._running = False

  def health(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(
        state="READY" if self._running else "STOPPED", policy_version=self._step
    )

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self._worker_id, roles=frozenset({"trainer"})
    )

  def with_loss_fn(self, loss_fn, has_aux: bool = False) -> "FakeTrainerWorker":
    del loss_fn, has_aux
    return self

  def fwd_bwd(
      self,
      payload: datatypes.TrainExample,
      *,
      accum_id: str,
      micro_index: int,
      loss_scale: float = 1.0,
  ) -> datatypes.StepReceipt:
    del payload
    per_accum = self._receipts.setdefault(accum_id, {})
    if micro_index in per_accum:
      # Duplicate (accum_id, micro_index): return the existing receipt without
      # re-accumulating.
      return per_accum[micro_index]
    receipt = datatypes.StepReceipt(
        accum_id=accum_id,
        micro_index=micro_index,
        applied=False,
        micro_loss=float(micro_index),
        denominator=float(loss_scale),
    )
    per_accum[micro_index] = receipt
    return receipt

  def update(
      self, *, accum_id: str, expected_micro_steps: int
  ) -> datatypes.UpdateResult:
    if accum_id in self._updates:
      # Retried update: return the cached result; never apply twice.
      return self._updates[accum_id]
    if accum_id not in self._receipts:
      raise KeyError(f"update() for unknown accum_id: {accum_id!r}")
    received = set(self._receipts[accum_id])
    expected = set(range(expected_micro_steps))
    if received != expected:
      raise ValueError(
          f"accum_id {accum_id!r} has micro-steps {sorted(received)}, "
          f"expected {sorted(expected)}"
      )
    self._step += 1
    result = datatypes.UpdateResult(step=self._step, applied=True, grad_norm=0.0)
    self._updates[accum_id] = result
    return result

  def reset_accumulation(self, accum_id: str | None = None) -> None:
    if accum_id is None:
      self._receipts.clear()
      self._updates.clear()
    else:
      self._receipts.pop(accum_id, None)
      self._updates.pop(accum_id, None)

  def eval_step(
      self, payload: datatypes.TrainExample, **kwargs
  ) -> metrics.MetricsBuffer:
    del payload, kwargs
    return metrics.MetricsBuffer(id=f"eval-{self._step}", mode="eval")

  def save_checkpoint(self, metadata: dict[str, Any], **kwargs) -> None:
    del kwargs
    self._checkpoint = {"step": self._step, **metadata}

  def restore_checkpoint(self, **kwargs) -> dict[str, Any]:
    del kwargs
    return dict(self._checkpoint) if self._checkpoint else {"step": self._step}

  def prepare_weight_sync(
      self, spec: datatypes.WeightSyncSpec
  ) -> datatypes.WeightSyncMetadata:
    return datatypes.WeightSyncMetadata(version=spec.version, method=spec.method)

  def get_metrics(self) -> metrics.MetricsBuffer:
    return metrics.MetricsBuffer(id=self._step)
