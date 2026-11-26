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

"""MaxText trainer adapter for ICI weight synchronization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import gc
import time
from typing import Any

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
from tunix.dist_rl.common import env_utils
from tunix.dist_rl.weight_sync import ici_weight_sync_delegate
from tunix.dist_rl.weight_sync import weight_sync
from tunix.utils import maxtext_utils
from tunix.utils import sharding_utils
from tunix.utils import subworld_checkpoint


def _extract_maxtext_params(engine: Any) -> Any:
  """Extracts pure parameter arrays from a `MaxTextTrainingEngine`."""
  # pylint: disable-next=protected-access
  return engine._get_trainable_params_state()


def _unscan_maxtext_layers(
    params: Any,
    *,
    num_layers: int,
    scan_axis: int,
    cycle_interval: int,
) -> Any:
  """Unscans stacked MaxText decoder layers into per-layer rollout leaves."""
  _, maxtext_engine, _ = maxtext_utils.maxtext_modules()
  return maxtext_engine.raiden_unscan.unscan_layers(
      params,
      num_layers=num_layers,
      scan_axis=scan_axis,
      cycle_interval=cycle_interval,
  )


def _convert_maxtext_params_for_rollout(engine: Any, params_state: Any) -> Any:
  """Converts MaxText trainer params into the rollout layout and dtype."""
  # pylint: disable=protected-access
  if engine._use_weight_converter:
    converter = engine._weight_converter
    if converter is None:
      raise RuntimeError(
          f"{type(engine).__name__} has _use_weight_converter=True but"
          " _weight_converter is None."
      )
    return converter.convert(params_state, target_state=engine._target_state)

  cfg = engine._config
  # pylint: enable=protected-access
  # TODO(tunix-dev): [legacy-maxtext-a2cf3bc8b] Read `cfg.rollout_weight_dtype`
  # directly once MaxText images predating the field are retired.
  rollout_dtype_raw = (
      getattr(cfg, "rollout_weight_dtype", None)
      or maxtext_utils.DEFAULT_ROLLOUT_WEIGHT_DTYPE
  )
  rollout_dtype = jnp.dtype(rollout_dtype_raw)
  converted = jax.tree_util.tree_map(
      lambda x: (
          x.astype(rollout_dtype)
          if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
          else x
      ),
      params_state,
  )
  if cfg.scan_layers:
    converted = _unscan_maxtext_layers(
        converted,
        num_layers=cfg.num_decoder_layers,
        scan_axis=cfg.param_scan_axis,
        cycle_interval=cfg.inhomogeneous_layer_cycle_interval,
    )
  return converted


class IciMaxTextTrainerAdapter:
  """Wraps a `MaxTextTrainingEngine` to serve weights over ICI.

  Delegates training, evaluation, checkpointing, and metric calls to the
  underlying engine, while replacing `prepare_weight_sync`,
  `release_weight_sync`, `ici_check`, and `ici_transfer` so converted rollout
  weights stay on device and transfer via `IciSourceSynchronizer`.
  """

  def __init__(
      self,
      engine: Any,
      *,
      job_name: str,
  ):
    """Initializes the adapter around `engine`.

    Args:
      engine: Underlying `MaxTextTrainingEngine` instance.
      job_name: Logical work-unit job name published to `IciHandler`.

    Raises:
      TypeError: If `engine` has neither `with_fwd_only_fn` nor a callable
        `fwd_only`.
    """
    # TODO(tunix-dev): [legacy-maxtext-a2cf3bc8b] Remove once the TPU image's
    # MaxText provides with_fwd_only_fn.
    self._legacy_fwd_only = not hasattr(engine, "with_fwd_only_fn")
    has_fwd_only = callable(getattr(engine, "fwd_only", None))
    if self._legacy_fwd_only and not has_fwd_only:
      raise TypeError(
          f"{type(engine).__name__} has neither with_fwd_only_fn nor a"
          " callable fwd_only."
      )
    self._engine = engine
    self._job_name = job_name
    self._ici_sync: ici_weight_sync_delegate.IciSourceSynchronizer | None = None
    self._last_staged_step: int | None = None
    self._staged_metadata: list[weight_sync.WorkUnitMetadata] | None = None
    self._active_processes = subworld_checkpoint.mesh_process_indices(
        engine._mesh  # pylint: disable=protected-access
    )
    if jax.process_count() > 1 and not self._active_processes:
      raise RuntimeError(
          f"{type(engine).__name__}._mesh yielded no JAX process indices in a"
          f" {jax.process_count()}-process world; cannot scope multi-host"
          " batch sharding."
      )
    self._install_multihost_batch_sharding()

  def _install_multihost_batch_sharding(self) -> None:
    """Places multi-host MaxText global host batches via callback.

    When the orchestrator fans a global micro-batch out to S > 1 McJAX trainer
    processes, `MaxTextTrainingEngine._place_batch_inputs` uses `jax.device_put`
    (which rejects multi-host shardings) and compiled `fwd_bwd` passes raw NumPy
    arrays directly. Wrapping `_prepare_batch` converts global host array leaves
    into globally-sharded `jax.Array`s right after `gen_model_input_fn` runs.
    """
    if len(self._active_processes) <= 1:
      return
    # pylint: disable=protected-access
    orig_prepare = self._engine._prepare_batch
    leaf_sharding = self._engine._leaf_data_sharding
    input_spec = self._engine._input_data_spec

    def _prepare_batch_multihost(payload: Any) -> Any:
      batch = orig_prepare(payload)
      data_spec = input_spec()

      def _place(leaf: Any) -> Any:
        if not isinstance(leaf, (jax.Array, np.ndarray)):
          return leaf
        return sharding_utils.place_array_on_target(
            leaf, leaf_sharding(leaf, data_spec)
        )

      return jax.tree.map(_place, batch)

    self._engine._prepare_batch = _prepare_batch_multihost
    # pylint: enable=protected-access

  def __getattr__(self, name: str) -> Any:
    return getattr(self._engine, name)

  @property
  def model(self) -> Any:
    """Returns the wrapped engine's model."""
    return self._engine.model

  def with_loss_fn(
      self, loss_fn: Any, /, has_aux: bool = False
  ) -> IciMaxTextTrainerAdapter:
    """Forwards `with_loss_fn` while keeping the adapter wrapper intact."""
    self._engine.with_loss_fn(loss_fn, has_aux=has_aux)
    return self

  def with_gen_model_input_fn(
      self, gen_model_input_fn: Any, /
  ) -> IciMaxTextTrainerAdapter:
    """Forwards `with_gen_model_input_fn` while keeping the adapter wrapper."""
    self._engine.with_gen_model_input_fn(gen_model_input_fn)
    return self

  def compile(self, dummy_data: Any = None) -> None:
    """Forwards ahead-of-time compilation to the wrapped engine."""
    self._engine.compile(dummy_data)

  def fwd_bwd(self, payload: Any) -> None:
    """Forwards a forward/backward micro-step to the wrapped engine."""
    self._engine.fwd_bwd(payload)

  def update(self) -> int:
    """Applies accumulated gradients on the wrapped engine."""
    return self._engine.update()

  def eval_step(self, payload: Any) -> None:
    """Forwards an evaluation step to the wrapped engine."""
    self._engine.eval_step(payload)

  def set_target_state(self, target_state: Any) -> None:
    """Forwards target state registration to the wrapped engine."""
    self._engine.set_target_state(target_state)

  def get_metrics(self) -> Any:
    """Returns and clears metrics collected by the wrapped engine."""
    return self._engine.get_metrics()

  def with_fwd_only_fn(self, fn: Any, /) -> IciMaxTextTrainerAdapter:
    """Forwards `with_fwd_only_fn` while keeping the adapter wrapper intact.

    Legacy MaxText engines without `with_fwd_only_fn` skip the registration;
    `fwd_only` then raises `NotImplementedError`.
    """
    if self._legacy_fwd_only:
      logging.warning(
          "%s predates with_fwd_only_fn; skipping registration. Per-token"
          " log-probs via fwd_only are unsupported on this MaxText.",
          type(self._engine).__name__,
      )
      return self
    self._engine.with_fwd_only_fn(fn)
    return self

  def fwd_only(self, payload: Any) -> Any:
    """Runs forward-only evaluation on the wrapped MaxText engine.

    Raises:
      NotImplementedError: If the engine predates `with_fwd_only_fn`.
    """
    if self._legacy_fwd_only:
      raise NotImplementedError(
          f"{type(self._engine).__name__} predates with_fwd_only_fn, so it"
          " cannot compute trainer-side per-token log-probs. Use a MaxText"
          " image with with_fwd_only_fn (>= aee266d4), or a config that does"
          " not need them."
      )
    if len(self._active_processes) > 1:
      if (
          dataclasses.is_dataclass(payload)
          and "metadata" in payload.__dataclass_fields__
      ):
        payload = dataclasses.replace(payload, metadata={})
      # pylint: disable-next=protected-access
      shardings = self._engine._fwd_only_data_shardings(payload)
      payload = jax.tree.map(
          sharding_utils.place_array_leaf, payload, shardings
      )
    return self._engine.fwd_only(payload)

  def save_checkpoint(self, *args: Any, **kwargs: Any) -> Any:
    """Saves a MaxText checkpoint on the wrapped engine."""
    return self._engine.save_checkpoint(*args, **kwargs)

  def restore_checkpoint(self, *args: Any, **kwargs: Any) -> Any:
    """Restores a MaxText checkpoint and invalidates any staged ICI cache."""
    self._invalidate_staged()
    return self._engine.restore_checkpoint(*args, **kwargs)

  def _invalidate_staged(
      self, *, sync_request: Any = None, collect_gc: bool = False
  ) -> None:
    self._last_staged_step = None
    self._staged_metadata = None
    if self._ici_sync is not None and self._ici_sync.bound:
      if sync_request is not None:
        logging.vlog(
            1, "ICI weight sync metrics: %s", self._ici_sync.metrics()
        )
      self._ici_sync.release(sync_request)
      if collect_gc:
        gc.collect()

  def _ensure_sync(self) -> ici_weight_sync_delegate.IciSourceSynchronizer:
    if self._ici_sync is None:
      self._ici_sync = ici_weight_sync_delegate.IciSourceSynchronizer(
          job_name=self._job_name
      )
    return self._ici_sync

  def _can_reuse_staged(self, train_step: int) -> bool:
    return (
        self._ici_sync is not None
        and self._ici_sync.bound
        and self._staged_metadata is not None
        and self._last_staged_step == train_step
    )

  def _stage_params(
      self,
      sync: ici_weight_sync_delegate.IciSourceSynchronizer,
      *,
      train_step: int,
      started: float,
  ) -> list[weight_sync.WorkUnitMetadata]:
    """Waits for compute, converts params on device, binds `sync`, and logs."""
    t0 = time.monotonic()
    self._engine._throttler.wait_for_all()  # pylint: disable=protected-access
    throttler_wait_s = time.monotonic() - t0

    t1 = time.monotonic()
    if sync.bound:
      sync.release()
      gc.collect()
    purge_s = time.monotonic() - t1

    t2 = time.monotonic()
    params_state = _extract_maxtext_params(self._engine)
    converted_state = _convert_maxtext_params_for_rollout(
        self._engine, params_state
    )
    del params_state
    convert_s = time.monotonic() - t2

    t3 = time.monotonic()
    sync.bind(converted_state)
    del converted_state
    bind_s = time.monotonic() - t3

    checksum_s = 0.0
    if env_utils.parse_env_bool("VERIFY_WEIGHTS"):
      t_ck = time.monotonic()
      logging.info("Source weights checksums: %s", sync.checksums())
      checksum_s = time.monotonic() - t_ck

    t4 = time.monotonic()
    metadata = [sync.work_unit_metadata()]
    metadata_s = time.monotonic() - t4
    total_vars = sum(len(m.variables) for m in metadata)
    logging.info(
        "TRAINER_WEIGHT_SYNC_TIMING step=prepare_weight_sync train_step=%s"
        " transport=ici throttler_wait_s=%.3f purge_buffers_s=%.3f"
        " convert_s=%.3f bind_s=%.3f checksum_s=%.3f metadata_s=%.3f"
        " total_s=%.3f work_units=%d variables=%d",
        train_step,
        throttler_wait_s,
        purge_s,
        convert_s,
        bind_s,
        checksum_s,
        metadata_s,
        time.monotonic() - started,
        len(metadata),
        total_vars,
    )
    return metadata

  def prepare_weight_sync(
      self, *, sync_request: Any = None
  ) -> Sequence[weight_sync.WorkUnitMetadata]:
    """Converts trainable params on device and binds them for ICI transfer."""
    del sync_request
    started = time.monotonic()
    train_step = self._engine.train_step
    if self._can_reuse_staged(train_step):
      assert self._staged_metadata is not None
      total_vars = sum(len(m.variables) for m in self._staged_metadata)
      logging.info(
          "TRAINER_WEIGHT_SYNC_TIMING step=prepare_weight_sync train_step=%s"
          " transport=ici reused=True total_s=%.3f work_units=%d variables=%d",
          train_step,
          time.monotonic() - started,
          len(self._staged_metadata),
          total_vars,
      )
      return list(self._staged_metadata)

    with jax.profiler.TraceAnnotation("prepare_weight_sync_ici"):
      metadata = self._stage_params(
          self._ensure_sync(), train_step=train_step, started=started
      )

    self._last_staged_step = train_step
    self._staged_metadata = metadata
    return list(metadata)

  def _require_ici_sync(
      self,
  ) -> ici_weight_sync_delegate.IciSourceSynchronizer:
    if self._ici_sync is None:
      raise RuntimeError(
          "ICI weight sync needs prepare_weight_sync to have run first on"
          f" {self._job_name}."
      )
    return self._ici_sync

  def ici_check(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validates this trainer's part of an ICI transfer plan."""
    # Phase 1 world discovery (`plan=None`) can arrive before
    # `prepare_weight_sync`, so create the synchronizer lazily here.
    return self._ensure_sync().check(request)

  def ici_transfer(self, request: Mapping[str, Any]) -> Mapping[str, float]:
    """Enters the ICI transfer collectives as the source."""
    return self._require_ici_sync().transfer(request)

  def release_weight_sync(self, *, sync_request: Any = None) -> bool:
    """Drops converted weight references after the ICI transfer completes."""
    started = time.monotonic()
    self._invalidate_staged(sync_request=sync_request, collect_gc=True)
    logging.info(
        "TRAINER_WEIGHT_SYNC_TIMING step=release_weight_sync train_step=%s"
        " transport=ici total_s=%.3f",
        self._engine.train_step,
        time.monotonic() - started,
    )
    return True

  def close(self) -> None:
    """Releases ICI references and closes the wrapped MaxText engine."""
    self._last_staged_step = None
    self._staged_metadata = None
    try:
      if self._ici_sync is not None:
        self._ici_sync.close()
        self._ici_sync = None
    finally:
      self._engine.close()
