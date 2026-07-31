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

"""A cluster surface for an orchestrator that owns no models.

The worker-backed cluster routes compute to handles but is built around a real
in-process cluster, which it needs for everything it does not route: the
config, the tokenizer, padding ids, the step counter, the metrics logger, and
a fallback for any handle that is absent. That base loads actual models, so
until now "distributed" meant routed compute with a full copy of the models
still sitting in the orchestrator process.

This is the other half. It answers the same reads from configuration and from
what the workers declared about themselves, and constructs no parameters at
all. Two consequences are deliberate:

  * every compute primitive raises. With no models there is nothing to fall
    back to, so a missing handle is a misconfiguration to report rather than a
    silent detour through weights that should not exist here.
  * the model accessors return None. The learner uses them to ask whether the
    trainer and the sampler share weights; across processes they cannot, and
    None is how that question is answered honestly rather than by fabricating
    a model to compare.
"""

from __future__ import annotations

import contextlib
from typing import Any, Mapping, Optional

from tunix.rl import rl_cluster as rl_cluster_lib


class HeadlessClusterError(RuntimeError):
  """A compute primitive was asked of a cluster that holds no models."""


class _NoOpSpan:
  """Stands in for a profiling span; records nothing."""

  def async_end(self, *args, **kwargs) -> None:
    del args, kwargs


class _NoOpPerf:
  """The profiling surface the learner reaches for, doing nothing."""

  @contextlib.contextmanager
  def span(self, *args, **kwargs):
    del args, kwargs
    yield _NoOpSpan()

  @contextlib.contextmanager
  def span_group(self, *args, **kwargs):
    del args, kwargs
    yield


class _HeadlessRollout:
  """Sampler-shaped view backed by declared configuration, not a model."""

  def __init__(self, pad_id: int, eos_id: int):
    self._pad_id = pad_id
    self._eos_id = eos_id

  def pad_id(self) -> int:
    return self._pad_id

  def eos_id(self) -> int:
    return self._eos_id

  def model(self) -> None:
    """No sampler model lives here; see the module docstring."""
    return None


class _HeadlessTrainer:
  """Trainer-shaped view: accepts the wiring, owns no parameters.

  The learner installs a loss and an input adapter on the trainer at
  construction, and reads back a few bookkeeping values. Those calls are
  accepted and remembered so the learner can be built, but nothing here can
  execute them -- training is the trainer worker's job, and what it needs of
  this wiring has to reach it some other way.
  """

  def __init__(self, restored_step: int = 0):
    self.is_managed_externally = False
    self.iter_steps = 0
    self.train_steps = 0
    self.loss_fn: Optional[Any] = None
    self.has_aux = False
    self.gen_model_input_fn: Optional[Any] = None
    self.rl_metrics_to_log: Optional[Any] = None
    self._restored_step = restored_step

  @property
  def model(self) -> None:
    """No trainer model lives here; see the module docstring."""
    return None

  def restored_global_step(self) -> int:
    return self._restored_step

  def with_loss_fn(self, loss_fn: Any, has_aux: bool = False):
    self.loss_fn = loss_fn
    self.has_aux = has_aux
    return self

  def with_gen_model_input_fn(self, gen_model_input_fn: Any):
    self.gen_model_input_fn = gen_model_input_fn
    return self

  def with_rl_metrics_to_log(self, metrics: Any):
    self.rl_metrics_to_log = metrics
    return self


class HeadlessCluster:
  """The cluster surface, served from configuration instead of models."""

  def __init__(
      self,
      *,
      cluster_config: Any,
      tokenizer: Any,
      pad_id: int,
      eos_id: int,
      metrics_logger: Any = None,
      initial_global_step: int = 0,
  ):
    """Initializes the cluster.

    Args:
      cluster_config: The run's configuration, as the in-process cluster would
        hold it.
      tokenizer: Used by loops that encode or decode orchestrator-side.
      pad_id: Padding token id, as declared by the workers.
      eos_id: End-of-sequence token id, as declared by the workers.
      metrics_logger: Receives buffered metrics; optional.
      initial_global_step: Step count to resume from.
    """
    self.cluster_config = cluster_config
    self.tokenizer = tokenizer
    self.global_steps = initial_global_step
    self.rollout = _HeadlessRollout(pad_id, eos_id)
    self.actor_trainer = _HeadlessTrainer(initial_global_step)
    self.perf_v2 = _NoOpPerf()
    self._metrics_logger = metrics_logger
    self.buffered_metrics: list[Any] = []

  @property
  def r2m(self) -> Mapping[Any, Any]:
    """Role-to-mesh map. Empty meshes: the devices are in other processes."""
    return getattr(self.cluster_config, "role_to_mesh", {})

  def get_rollout_config(self, mode: Any) -> Any:
    """Returns the rollout config for `mode`, dict-keyed or not."""
    rollout_config = self.cluster_config.rollout_config
    if isinstance(rollout_config, dict):
      return rollout_config[mode]
    return rollout_config

  # --- Metrics --------------------------------------------------------------

  def buffer_metrics(self, metrics: Any, mode: Any = None, **kwargs) -> None:
    del kwargs
    self.buffered_metrics.append(metrics)
    if self._metrics_logger is not None:
      self._metrics_logger.buffer_metrics(metrics, mode)

  def buffer_metrics_async(self, metrics: Any, mode: Any = None, **kwargs):
    self.buffer_metrics(metrics, mode, **kwargs)

  # --- Compute: not here ----------------------------------------------------

  def generate(self, *args, **kwargs):
    raise self._no_models("generate", "rollout")

  def update_actor(self, *args, **kwargs):
    raise self._no_models("update_actor", "trainer")

  def update_critic(self, *args, **kwargs):
    raise self._no_models("update_critic", "trainer")

  def get_ref_per_token_logps(self, *args, **kwargs):
    raise self._no_models("get_ref_per_token_logps", "inference")

  def get_actor_per_token_logps(self, *args, **kwargs):
    raise self._no_models("get_actor_per_token_logps", "trainer")

  def sync_weights(self, *args, **kwargs):
    raise self._no_models("sync_weights", "weight sync")

  def _no_models(self, primitive: str, role: str) -> HeadlessClusterError:
    return HeadlessClusterError(
        f"{primitive} was called on an orchestrator that holds no models, and"
        f" no {role} handle is attached to route it to. Attach one; there is"
        " deliberately nothing here to fall back to."
    )

  def close(self) -> None:
    pass


def resources_from(cluster_config: Any, tokenizer_hash: str, **extra) -> dict:
  """Builds the resource map an orchestrator expects workers to match.

  Args:
    cluster_config: The run's configuration.
    tokenizer_hash: Identifies the vocabulary this run assumes.
    **extra: Additional declared keys.

  Returns:
    The map to compare against what each worker reports.
  """
  rollout_config = cluster_config.rollout_config
  if isinstance(rollout_config, dict):
    rollout_config = rollout_config[rl_cluster_lib.Mode.TRAIN]
  declared = {
      "tokenizer_hash": tokenizer_hash,
      "temperature": rollout_config.temperature,
  }
  declared.update(extra)
  return declared
