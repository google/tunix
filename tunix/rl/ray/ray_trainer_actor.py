# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ray remote actor that owns the trainer (actor + reference models).

``TrainerActor`` is intended to be instantiated via ``ray.remote`` so that it
runs in its own Python process with exclusive access to a dedicated set of
accelerator devices.  It mirrors the trainer-side responsibilities of
``RLCluster``:

  * Holds the actor model (policy being trained) and reference model.
  * Runs training steps (gradient updates via ``update_actor``).
  * Computes reference log-probabilities for KL-divergence terms.
  * Exports current policy weights as numpy arrays for weight synchronisation.

The actor is intentionally kept **thin** – it delegates the heavy lifting to
the existing ``RLCluster`` internals rather than re-implementing them, so that
future improvements to those internals are automatically inherited.
"""

from __future__ import annotations

from typing import Any, Callable

from absl import logging
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np


class TrainerActor:
  """Owns the trainer model and reference model; runs training steps.

  This class is designed to be wrapped with ``ray.remote`` by
  ``RayRLCluster``, but it is also usable as a plain Python object for
  single-process testing (see ``JaxDevicePutSync``).

  Args:
    cluster_factory: A zero-argument callable that constructs and returns a
      fully initialised ``RLCluster`` (trainer-side only – the rollout engine
      inside the cluster will be unused/idle in this actor).  The factory is
      called **inside** the actor process, so JAX device assignment happens
      correctly.
    cluster_config: The ``ClusterConfig`` used to build the cluster.  Stored
      for reference (the factory may capture it via closure).
  """

  def __init__(
      self,
      cluster_factory: Callable[[], Any],
  ):
    """Initialise inside the Ray actor process."""
    logging.info("TrainerActor: initialising cluster inside actor process.")
    self._cluster = cluster_factory()
    logging.info("TrainerActor: cluster initialised. JAX devices: %s", jax.devices())

  # ---------------------------------------------------------------------------
  # Training
  # ---------------------------------------------------------------------------

  def update_actor(
      self,
      train_ds: list[Any],
      eval_ds: list[Any] | None,
      skip_jit: bool = False,
  ) -> None:
    """Run one actor gradient update step."""
    self._cluster.update_actor(train_ds, eval_ds, skip_jit)

  def update_critic(
      self,
      train_ds: list[Any],
      eval_ds: list[Any] | None,
      skip_jit: bool = False,
  ) -> None:
    """Run one critic gradient update step (PPO-style only)."""
    self._cluster.update_critic(train_ds, eval_ds, skip_jit)

  # ---------------------------------------------------------------------------
  # Reference log-probs
  # ---------------------------------------------------------------------------

  def get_ref_per_token_logps(
      self,
      prompt_tokens: np.ndarray,
      completion_tokens: np.ndarray,
      pad_id: int,
      eos_id: int,
      micro_batch_size: int | None = None,
      completion_mask: np.ndarray | None = None,
  ) -> np.ndarray:
    """Compute reference model per-token log-probabilities.

    Accepts and returns **numpy** arrays so they can cross Ray process
    boundaries without serialisation issues.
    """
    prompt_jax = jnp.asarray(prompt_tokens)
    completion_jax = jnp.asarray(completion_tokens)
    mask_jax = jnp.asarray(completion_mask) if completion_mask is not None else None
    result = self._cluster.get_ref_per_token_logps(
        prompt_tokens=prompt_jax,
        completion_tokens=completion_jax,
        pad_id=pad_id,
        eos_id=eos_id,
        micro_batch_size=micro_batch_size,
        completion_mask=mask_jax,
    )
    return np.asarray(result)

  # ---------------------------------------------------------------------------
  # Weight export
  # ---------------------------------------------------------------------------

  def get_weights_numpy(self) -> Any:
    """Export current policy parameters as a pytree of numpy arrays.

    Returns a structure that mirrors ``nnx.state(actor_model)`` but with
    every leaf converted to ``np.ndarray`` so that it can be safely
    transferred across Ray process boundaries.
    """
    from flax.nnx import filterlib  # pylint: disable=g-import-not-at-top
    from tunix.sft import utils as sft_utils  # pylint: disable=g-import-not-at-top

    filter_types = (
        nnx.LoRAParam
        if sft_utils.is_lora_enabled(self._cluster.actor_trainer.model)
        else nnx.Param,
    )
    state = nnx.state(self._cluster.actor_trainer.model, filter_types)
    numpy_state = jax.tree_util.tree_map(np.asarray, state)
    return numpy_state

  def get_weights_numpy_ref(self) -> Any:
    """Like ``get_weights_numpy`` but returns a Ray ObjectRef.

    Calling ``ray.put(weights)`` inside the actor avoids a redundant copy
    through the orchestrator process when using ``RayObjectStoreSync``.
    """
    import ray  # pylint: disable=g-import-not-at-top
    weights = self.get_weights_numpy()
    return ray.put(weights)

  # ---------------------------------------------------------------------------
  # Metrics & misc
  # ---------------------------------------------------------------------------

  def buffer_metrics_async(
      self,
      metrics: dict[str, Any],
      mode: Any,
      step: int | None = None,
  ) -> None:
    self._cluster.buffer_metrics_async(metrics, mode=mode, step=step)

  def flush_metrics(self) -> None:
    self._cluster.close()

  @property
  def global_steps(self) -> int:
    return self._cluster.global_steps

  @global_steps.setter
  def global_steps(self, value: int) -> None:
    self._cluster.global_steps = value

  def get_global_steps(self) -> int:
    return self._cluster.global_steps

  def set_global_steps(self, value: int) -> None:
    self._cluster.global_steps = value

  # Expose trainer properties needed by the learner.
  def get_actor_trainer_state(self) -> dict[str, Any]:
    """Return serialisable metadata about the actor trainer."""
    t = self._cluster.actor_trainer
    return {
        "train_steps": t.train_steps,
        "iter_steps": t.iter_steps,
        "restored_global_step": t.restored_global_step(),
        "is_managed_externally": t.is_managed_externally,
    }

  def set_actor_trainer_managed_externally(self, value: bool) -> None:
    self._cluster.actor_trainer.is_managed_externally = value

  def pad_id(self) -> int:
    return self._cluster.rollout.pad_id()

  def eos_id(self) -> int:
    return self._cluster.rollout.eos_id()

  def close(self) -> None:
    self._cluster.close()
