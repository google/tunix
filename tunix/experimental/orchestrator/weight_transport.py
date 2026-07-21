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

"""Weight transport library: how staged weights get from trainer to replicas.

Weights never cross the RPC control plane; a transport stages a version on the
trainer side and lets replicas pull it. `WeightTransport` is the ABC both worker
plans build against; `InProcessTransport` is the same-process rung of the ladder
(a live-handle hand-off, effectively zero-copy), ahead of checkpoint- and
P2P-based transports for the multi-process cases.

Staged versions are retained until explicitly released, so a replica that is
retrying (or catching up after a restart) can always re-fetch its target version.
"""

import abc
from collections.abc import Iterator
from typing import Any

from tunix.experimental.common import datatypes


class WeightTransport(abc.ABC):
  """Stages a weight version and serves pulls of it."""

  @abc.abstractmethod
  def stage(self, state: Any, spec: datatypes.WeightSyncSpec) -> datatypes.WeightSyncMetadata:
    """Stages `state` under `spec.version` and returns fetch metadata."""
    raise NotImplementedError

  @abc.abstractmethod
  def fetch(self, meta: datatypes.WeightSyncMetadata) -> Iterator[Any]:
    """Yields the staged state for `meta.version`, in one or more chunks."""
    raise NotImplementedError

  @abc.abstractmethod
  def release(self, version: int) -> None:
    """Drops a staged version (idempotent)."""
    raise NotImplementedError


class InProcessTransport(WeightTransport):
  """Same-process transport: the staged state is handed over by reference."""

  def __init__(self):
    self._staged: dict[int, Any] = {}

  def stage(
      self, state: Any, spec: datatypes.WeightSyncSpec
  ) -> datatypes.WeightSyncMetadata:
    self._staged[spec.version] = state
    # In-process: the locator is the live handle itself (the one wire carve-out).
    return datatypes.WeightSyncMetadata(
        version=spec.version, method="in_process", locator=state
    )

  def fetch(self, meta: datatypes.WeightSyncMetadata) -> Iterator[Any]:
    if meta.version not in self._staged:
      raise KeyError(f"no staged weights for version {meta.version}")
    yield self._staged[meta.version]

  def release(self, version: int) -> None:
    self._staged.pop(version, None)

  def staged_versions(self) -> list[int]:
    """Currently-staged versions (ascending); for diagnostics and tests."""
    return sorted(self._staged)
