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

"""A tiny, real AbstractTrainer for tests and parameter-update equivalence.

A pure-optax linear "policy" over a small token vocabulary: the score of a
completion token is a learned weight indexed by its id, and the loss is the
advantage-weighted, completion-masked token-mean of those scores. It implements
real gradient accumulation (`fwd_bwd` sums caller-scaled grads under an
`accum_id`; `update` applies the sum), so N micro-batches carrying token-mean
`loss_scale`s reproduce the full-batch update exactly. No transformer and no I/O
— just enough to exercise the trainer contract and the orchestrator's dispatch
schedule.
"""

from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax

from tunix.experimental.common import datatypes
from tunix.experimental.metrics import metrics
from tunix.experimental.train import abstract_trainer
from tunix.rl import common


class ToyAbstractTrainer(abstract_trainer.AbstractTrainer):
  """Minimal real trainer: a linear score over a small token vocabulary."""

  def __init__(self, config: Any = None):
    config = config or {}
    self._vocab_size = int(config.get("vocab_size", 16))
    self._optimizer = optax.sgd(float(config.get("learning_rate", 0.1)))
    self._params = {"w": jnp.zeros((self._vocab_size,), dtype=jnp.float32)}
    self._opt_state = self._optimizer.init(self._params)
    self._step = 0
    self._accum: dict[str, Any] = {}
    self._receipts: dict[str, dict[int, datatypes.StepReceipt]] = {}
    self._updates: dict[str, datatypes.UpdateResult] = {}
    self._checkpoint: dict[str, Any] | None = None

  def with_loss_fn(self, loss_fn: Callable[..., Any], has_aux: bool = False):
    # The toy uses a fixed built-in loss; the call is accepted only to honor the
    # "set the loss before compile()" contract.
    del loss_fn, has_aux
    return self

  def compile(self, shape_config: datatypes.ShapeConfig) -> None:
    del shape_config  # Tiny model; XLA compiles lazily on the first step.

  def _loss(self, params, ex: common.TrainExample):
    scores = params["w"][ex.completion_ids]  # [B, C] gather by token id.
    advantages = ex.advantages
    if advantages.ndim == 1:
      advantages = advantages[:, None]
    mask = ex.completion_mask.astype(jnp.float32)
    mask_sum = jnp.sum(mask)
    loss = jnp.sum(-(advantages * scores) * mask) / jnp.maximum(mask_sum, 1.0)
    return loss, mask_sum

  def fwd_bwd(
      self,
      payload: common.TrainExample,
      *,
      accum_id: str,
      micro_index: int,
      loss_scale: float = 1.0,
  ) -> datatypes.StepReceipt:
    per_accum = self._receipts.setdefault(accum_id, {})
    if micro_index in per_accum:
      # Duplicate micro-step: return the existing receipt without re-adding grads.
      return per_accum[micro_index]
    (loss, mask_sum), grad = jax.value_and_grad(self._loss, has_aux=True)(
        self._params, payload
    )
    scaled = jax.tree.map(lambda g: g * loss_scale, grad)
    if accum_id in self._accum:
      self._accum[accum_id] = jax.tree.map(
          lambda a, b: a + b, self._accum[accum_id], scaled
      )
    else:
      self._accum[accum_id] = scaled
    receipt = datatypes.StepReceipt(
        accum_id=accum_id,
        micro_index=micro_index,
        applied=False,
        micro_loss=float(loss),
        denominator=float(mask_sum),
    )
    per_accum[micro_index] = receipt
    return receipt

  def update(
      self, *, accum_id: str, expected_micro_steps: int
  ) -> datatypes.UpdateResult:
    if accum_id in self._updates:
      return self._updates[accum_id]
    if accum_id not in self._accum:
      raise KeyError(f"update() for unknown accum_id: {accum_id!r}")
    received = set(self._receipts[accum_id])
    expected = set(range(expected_micro_steps))
    if received != expected:
      raise ValueError(
          f"accum_id {accum_id!r} has micro-steps {sorted(received)}, "
          f"expected {sorted(expected)}"
      )
    summed = self._accum[accum_id]
    updates, self._opt_state = self._optimizer.update(
        summed, self._opt_state, self._params
    )
    self._params = optax.apply_updates(self._params, updates)
    self._step += 1
    grad_sq = sum(jnp.vdot(g, g) for g in jax.tree.leaves(summed))
    result = datatypes.UpdateResult(
        step=self._step,
        applied=True,
        grad_norm=float(jnp.sqrt(grad_sq)),
    )
    self._updates[accum_id] = result
    return result

  def reset_accumulation(self, accum_id: str | None = None) -> None:
    if accum_id is None:
      self._accum.clear()
      self._receipts.clear()
      self._updates.clear()
    else:
      self._accum.pop(accum_id, None)
      self._receipts.pop(accum_id, None)
      self._updates.pop(accum_id, None)

  def eval_step(
      self, payload: common.TrainExample, **kwargs
  ) -> metrics.MetricsBuffer:
    del kwargs
    loss, _ = self._loss(self._params, payload)
    return metrics.MetricsBuffer(
        id=f"eval-{self._step}",
        scalar_metrics={"loss": float(loss)},
        mode="eval",
    )

  def save_checkpoint(self, metadata: dict[str, Any], **kwargs) -> None:
    del kwargs
    self._checkpoint = {
        "params": jax.tree.map(jnp.array, self._params),
        "opt_state": self._opt_state,
        "step": self._step,
        "metadata": dict(metadata),
    }

  def restore_checkpoint(self, **kwargs) -> dict[str, Any]:
    del kwargs
    if self._checkpoint is None:
      return {"step": self._step}
    self._params = jax.tree.map(jnp.array, self._checkpoint["params"])
    self._opt_state = self._checkpoint["opt_state"]
    self._step = self._checkpoint["step"]
    return {"step": self._step, **self._checkpoint["metadata"]}

  def prepare_weight_sync(
      self, spec: datatypes.WeightSyncSpec
  ) -> datatypes.WeightSyncMetadata:
    # In-process transport: the locator is the live params handle (the one
    # documented wire carve-out).
    return datatypes.WeightSyncMetadata(
        version=spec.version, method="in_process", locator=self._params
    )

  def get_metrics(self) -> metrics.MetricsBuffer:
    return metrics.MetricsBuffer(id=self._step)

  def close(self) -> None:
    pass
