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

"""Coordinator-side `WeightSyncHandler` validating staged Orbax checkpoints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
        f"Source work unit(s) did not register a fresh artifact_uri for this"
        f" round: {stale_src!r}",
    )

  artifact_uris: set[str] = {
      uri
      for unit in src_units
      if (uri := units[unit].artifact_uri) is not None
  }
  if len(artifact_uris) != 1:
    return (
        None,
        "Source work units published conflicting artifact_uri values:"
        f" {sorted(artifact_uris)!r}",
    )
  return next(iter(artifact_uris)), ""


def _check_finalized_artifact(artifact_uri: str) -> tuple[bool, str]:
  """Verifies `artifact_uri` exists and passes Orbax finalization checks."""
  try:
    ckpt_path = epath.Path(artifact_uri)
    if not ckpt_path.exists():
      return False, f"Checkpoint artifact does not exist: {artifact_uri}"
    is_finalized = gcs_synchronizer.run_blocking(
        lambda: ocp.path.step.is_path_finalized(ckpt_path)
    )
  except Exception as exc:  # pylint: disable=broad-exception-caught
    return (
        False,
        f"Failed checking checkpoint artifact finalization for"
        f" {artifact_uri}: {exc}",
    )
  if not is_finalized:
    return False, f"Checkpoint artifact is not finalized: {artifact_uri}"
  return True, ""


class GCSHandler(weight_sync.WeightSyncHandler):
  """Validates per-round Orbax checkpoint finalization before destination H2D."""

  def __init__(self) -> None:
    self._lock = threading.Lock()
    self._units: dict[weight_sync.WorkUnitId, weight_sync.WorkUnitMetadata] = {}
    self._fresh_sources: set[weight_sync.WorkUnitId] = set()

  def register_work_unit(self, metadata: weight_sync.WorkUnitMetadata) -> None:
    """Registers or replaces `metadata` for `metadata.unit`."""
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
    """Verifies the source round's staged checkpoint exists and is finalized."""
    del generation
    resolved_req_id = req_id or ""
    with self._lock:
      artifact_uri, err_msg = _resolve_registered_artifact(
          self._units, self._fresh_sources, src_units, dst_units
      )
      if artifact_uri is None:
        return weight_sync.TransferResult(
            req_id=resolved_req_id, success=False, message=err_msg
        )
      self._fresh_sources.difference_update(src_units)

    ok, finalize_msg = _check_finalized_artifact(artifact_uri)
    return weight_sync.TransferResult(
        req_id=resolved_req_id, success=ok, message=finalize_msg
    )

  def close(self) -> None:
    """Clears registered work-unit metadata."""
    with self._lock:
      self._units.clear()
      self._fresh_sources.clear()
