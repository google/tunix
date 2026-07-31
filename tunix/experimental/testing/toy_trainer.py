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

"""A tiny, real trainer for tests that need exact parameter arithmetic.

A pure-optax linear "policy" over a small token vocabulary: the score of a
token is a learned weight indexed by its id, and the loss is the
advantage-weighted, mask-weighted token-mean of those scores. It reaches the
payload's fields through a `gen_model_input_fn` adapter, so it accepts any
payload shape, and it accumulates real gradients across `fwd_bwd` calls so
that N equal-sized micro-batches reproduce the single full-batch update
exactly.

No transformer, no I/O, no jit: every update is a closed-form arithmetic step,
which is what makes it usable as a numeric anchor when comparing two training
paths parameter-for-parameter.
"""

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import optax

from tunix.experimental.common import datatypes
from tunix.experimental.metrics import metrics
from tunix.experimental.train import abstract_trainer


class ToyAbstractTrainer(abstract_trainer.AbstractTrainer):
  """Minimal real trainer: a linear score over a small token vocabulary."""

  def __init__(self, config: Any = None):
    config = config or {}
    self._vocab_size = int(config.get("vocab_size", 16))
    self._optimizer = optax.sgd(float(config.get("learning_rate", 0.1)))
    self._params = {"w": jnp.zeros((self._vocab_size,), dtype=jnp.float32)}
    self._opt_state = self._optimizer.init(self._params)
    self._gen_model_input_fn = self._default_gen_model_input_fn
    self._step = 0
    self._accum: Optional[dict[str, Any]] = None
    self._accum_count = 0
    self._micro_losses: list[float] = []
    self._checkpoint: Optional[dict[str, Any]] = None
    self.last_eval_loss: Optional[float] = None
    self.staged_weights: Optional[dict[str, Any]] = None

  @property
  def params(self) -> dict[str, Any]:
    """The live parameters, for parameter-equality assertions."""
    return self._params

  @property
  def train_steps(self) -> int:
    """Optimizer updates applied so far."""
    return self._step

  def with_loss_fn(self, loss_fn: Callable[..., Any], has_aux: bool = False):
    # The toy's loss is built in -- that is the point of the anchor. The call
    # is accepted so callers can honor the "set the loss before compile()"
    # ordering contract.
    del loss_fn, has_aux
    return self

  def with_gen_model_input_fn(
      self, gen_model_input_fn: Callable[[Any], dict[str, Any]]
  ):
    self._gen_model_input_fn = gen_model_input_fn
    return self

  def compile(self, dummy_data: Any = None) -> None:
    del dummy_data  # Tiny and unjitted; nothing to warm up.

  def _default_gen_model_input_fn(
      self, payload: datatypes.RLTrainerPayload
  ) -> dict[str, Any]:
    return {
        "token_ids": jnp.asarray(payload.token_ids),
        "loss_mask": jnp.asarray(payload.loss_mask),
        "advantages": jnp.asarray(payload.advantages, dtype=jnp.float32),
    }

  def _loss(self, params, *, token_ids, loss_mask, advantages):
    scores = params["w"][token_ids]  # [B, T] gather by token id.
    if advantages.ndim == 1:
      advantages = advantages[:, None]
    mask = loss_mask.astype(jnp.float32)
    weighted = -(advantages * scores) * mask
    return jnp.sum(weighted) / jnp.maximum(jnp.sum(mask), 1.0)

  def fwd_bwd(self, payload: datatypes.TrainerPayload, **kwargs) -> None:
    """Accumulates one micro-batch's gradients without applying them."""
    del kwargs
    inputs = self._gen_model_input_fn(payload)
    loss, grad = jax.value_and_grad(self._loss)(self._params, **inputs)
    if self._accum is None:
      self._accum = grad
    else:
      self._accum = jax.tree.map(lambda a, b: a + b, self._accum, grad)
    self._accum_count += 1
    self._micro_losses.append(float(loss))

  def update(self, **kwargs) -> int:
    """Applies the mean of the accumulated gradients as one optimizer step."""
    del kwargs
    if self._accum is None:
      raise RuntimeError("update() called with no accumulated gradients.")
    mean_grad = jax.tree.map(lambda g: g / self._accum_count, self._accum)
    updates, self._opt_state = self._optimizer.update(
        mean_grad, self._opt_state, self._params
    )
    self._params = optax.apply_updates(self._params, updates)
    self._step += 1
    self._accum = None
    self._accum_count = 0
    return self._step

  def eval_step(self, payload: datatypes.TrainerPayload, **kwargs) -> None:
    """Scores a batch without touching parameters or accumulation state."""
    del kwargs
    loss = self._loss(self._params, **self._gen_model_input_fn(payload))
    self.last_eval_loss = float(loss)

  def save_checkpoint(self, metadata: Any, **kwargs) -> None:
    del kwargs
    self._checkpoint = {
        "params": jax.tree.map(jnp.array, self._params),
        "opt_state": self._opt_state,
        "step": self._step,
        "metadata": dict(metadata or {}),
    }

  def restore_checkpoint(self, **kwargs) -> Any:
    del kwargs
    if self._checkpoint is None:
      return {"step": self._step}
    self._params = jax.tree.map(jnp.array, self._checkpoint["params"])
    self._opt_state = self._checkpoint["opt_state"]
    self._step = self._checkpoint["step"]
    return {"step": self._step, **self._checkpoint["metadata"]}

  def prepare_weight_sync(self, **kwargs) -> None:
    """Stages the live parameters for an in-process hand-off."""
    del kwargs
    self.staged_weights = jax.tree.map(jnp.array, self._params)

  def get_metrics(self) -> metrics.MetricsBuffer:
    buffer = metrics.MetricsBuffer(
        id=self._step,
        scalar_metrics={
            "loss": (
                sum(self._micro_losses) / len(self._micro_losses)
                if self._micro_losses
                else 0.0
            )
        },
    )
    self._micro_losses = []
    return buffer

  def close(self) -> None:
    pass
