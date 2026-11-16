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

"""Coordinator-side `WeightSyncHandler` validating staged checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import functools
import threading

from etils import epath
from orbax import checkpoint as ocp
from tunix.dist_rl.weight_sync import weight_sync
from tunix.experimental.weight_sync import gcs_synchronizer


def _resolve_registered_artifact(
    units: Mapping[weight_sync.WorkUnitId, weight_sync.WorkUnitMetadata],
    fresh_sources: set[weight_sync.WorkUnitId],
    src_units: Sequence[weight_sync.WorkUnitId],
    dst_units: Sequence[weight_sync.WorkUnitId],
) -> tuple[str | None, str]:
  """Validates registered units and returns `(artifact_uri, error_message)`."""
  if not src_units:
    return None, "GCSHandler.transfer requires at least one source work unit."
  if not dst_units:
    return (
        None,
        "GCSHandler.transfer requires at least one destination work unit.",
    )
  missing_dst = [unit for unit in dst_units if unit not in units]
  if missing_dst:
    return None, f"Unregistered destination work unit(s): {missing_dst!r}"
  missing_src = [unit for unit in src_units if unit not in units]
  if missing_src:
    return None, f"Unregistered source work unit(s): {missing_src!r}"
  stale_src = [unit for unit in src_units if unit not in fresh_sources]
  if stale_src:
    return (
        None,
        (
            "Source work unit(s) did not register a fresh artifact_uri for"
            f" this round: {stale_src!r}"
        ),
    )

  artifact_uris: set[str] = {
      uri for unit in src_units if (uri := units[unit].artifact_uri) is not None
  }
  if len(artifact_uris) != 1:
    return (
        None,
        (
            "Source work units published conflicting artifact_uri values:"
            f" {sorted(artifact_uris)!r}"
        ),
    )
  return next(iter(artifact_uris)), ""


def _probe_finalized_artifact(artifact_uri: str) -> tuple[bool, bool]:
  """Probes `(exists, is_finalized)` for Safetensors or Orbax step paths."""
  # Observed on GCS: probing `index.json` via `resolve_safetensors_files` first
  # avoids an extra directory prefix-list RPC (~60-100ms) on every sync while
  # still rejecting `.tmp` directories, uncommitted shards, and invalid indexes.
  ckpt_path = epath.Path(artifact_uri)
  try:
    files = gcs_synchronizer.resolve_safetensors_files(ckpt_path)
  except FileNotFoundError:
    return ckpt_path.exists(), False
  if files is not None:
    return True, True
  return True, ocp.path.step.is_path_finalized(ckpt_path)


def _check_finalized_artifact(artifact_uri: str) -> tuple[bool, str]:
  """Verifies `artifact_uri` exists and passes Safetensors/Orbax finalization."""
  try:
    exists, is_finalized = gcs_synchronizer.run_blocking(
        functools.partial(_probe_finalized_artifact, artifact_uri)
    )
  except Exception as exc:  # pylint: disable=broad-exception-caught
    return (
        False,
        (
            "Failed checking checkpoint artifact finalization for"
            f" {artifact_uri}: {exc}"
        ),
    )
  if not exists:
    return False, f"Checkpoint artifact does not exist: {artifact_uri}"
  if not is_finalized:
    return False, f"Checkpoint artifact is not finalized: {artifact_uri}"
  return True, ""


class GCSHandler(weight_sync.WeightSyncHandler):
  """Coordinator-side `WeightSyncHandler` gating H2D on finalized checkpoints.

  Tracks per-round `WorkUnitMetadata` published by Trainer (source) and
  Rollout/Sampler (destination) workers. On each :meth:`transfer` call, it
  verifies that:

    1. All requested source and destination work units are registered.
    2. Every source unit published a fresh `artifact_uri` for this round (no
       stale replay from a previous policy version).
    3. All source units agree on one canonical `artifact_uri`.
    4. The staged checkpoint directory exists and is finalized (contains
       `model.safetensors.index.json` / `model.safetensors` for Safetensors, or
       finalized Orbax metadata markers, with no `.tmp` suffix).

  Example:
    ```python
    from tunix.experimental.weight_sync import gcs_handler

    handler = gcs_handler.GCSHandler()
    # 1. Once at startup, register destination (Sampler/Rollout) work units.
    handler.register_work_unit(sampler_sync.work_unit_metadata())

    # 2. After Trainer finishes `trainer_sync.d2h(req)`, publish fresh metadata
    #    and validate finalization before triggering destination `h2d()`.
    src_meta = trainer_sync.work_unit_metadata()
    handler.register_work_unit(src_meta)
    result = handler.transfer(
        src_units=[src_meta.unit],
        dst_units=[sampler_sync.work_unit_metadata().unit],
        req_id="step_1",
    )
    if not result.success:
      raise RuntimeError(f"Weight sync barrier failed: {result.message}")
    ```
  """

  def __init__(self) -> None:
    """Initializes empty coordinator registration tables."""
    self._lock = threading.Lock()
    self._units: dict[weight_sync.WorkUnitId, weight_sync.WorkUnitMetadata] = {}
    self._fresh_sources: set[weight_sync.WorkUnitId] = set()

  def register_work_unit(self, metadata: weight_sync.WorkUnitMetadata) -> None:
    """Registers or replaces `metadata` for `metadata.unit`.

    Marks `metadata.unit` as having a fresh staged artifact whenever
    `metadata.artifact_uri` is non-None (typically after `d2h()`).

    Args:
      metadata: Wire-safe work-unit metadata produced by
        `GCSSynchronizer.work_unit_metadata()`.

    Example:
      ```python
      handler.register_work_unit(trainer_sync.work_unit_metadata())
      ```
    """
    with self._lock:
      self._units[metadata.unit] = metadata
      if metadata.artifact_uri is not None:
        self._fresh_sources.add(metadata.unit)
      else:
        self._fresh_sources.discard(metadata.unit)

  def transfer(
      self,
      src_units: Sequence[weight_sync.WorkUnitId],
      dst_units: Sequence[weight_sync.WorkUnitId],
      req_id: str | None = None,
      generation: int | None = None,
  ) -> weight_sync.TransferResult:
    """Verifies the source round's staged checkpoint exists and is finalized.

    Consumes the freshness bit on `src_units` so the same round cannot be
    replayed accidentally without a new `register_work_unit()` call after the
    next `d2h()`.

    Args:
      src_units: Non-empty sequence of source (Trainer) work-unit IDs.
      dst_units: Non-empty sequence of destination (Sampler) work-unit IDs.
      req_id: Optional correlation ID echoed in the returned `TransferResult`.
      generation: Unused generation counter kept for `WeightSyncHandler` parity.

    Returns:
      A `weight_sync.TransferResult` whose `success` is True iff all units are
      registered, sources agree on a fresh `artifact_uri`, and storage reports
      the checkpoint as finalized.

    Example:
      ```python
      res = handler.transfer(
          src_units=[src_meta.unit],
          dst_units=[dst_meta.unit],
          req_id="policy_v2",
      )
      assert res.success, res.message
      ```
    """
    del generation
    resolved_req_id = "" if req_id is None else req_id
    with self._lock:
      artifact_uri, err_msg = _resolve_registered_artifact(
          self._units, self._fresh_sources, src_units, dst_units
      )
      if artifact_uri is None:
        return weight_sync.TransferResult(
            req_id=resolved_req_id, success=False, message=err_msg
        )

    ok, finalize_msg = _check_finalized_artifact(artifact_uri)
    if not ok:
      return weight_sync.TransferResult(
          req_id=resolved_req_id, success=False, message=finalize_msg
      )
    with self._lock:
      current_uri, concurrent_err = _resolve_registered_artifact(
          self._units, self._fresh_sources, src_units, dst_units
      )
      if current_uri != artifact_uri:
        msg = concurrent_err
        if not msg:
          msg = (
              "Source artifact changed concurrently during verification"
              f" ({artifact_uri!r} -> {current_uri!r})."
          )
        return weight_sync.TransferResult(
            req_id=resolved_req_id, success=False, message=msg
        )
      self._fresh_sources.difference_update(src_units)
    return weight_sync.TransferResult(
        req_id=resolved_req_id, success=True, message=finalize_msg
    )

  def close(self) -> None:
    """Clears all registered work-unit metadata and freshness state.

    Example:
      ```python
      handler.close()
      ```
    """
    with self._lock:
      self._units.clear()
      self._fresh_sources.clear()
