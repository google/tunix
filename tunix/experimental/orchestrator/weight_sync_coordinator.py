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

"""Coordinates a fleet weight sync: stage on the trainer, install on replicas.

The orchestrator mints a new version V, the trainer stages it, and the
coordinator then fans out to the rollout replicas by **explicit iteration** (not
a pool broadcast that could silently update just one replica): each replica is
fenced (`prepare_weight_sync`) and then installs and acks V (`sync_weights`). A
replica that fails to reach V is quarantined -- its fence is released by the
worker contract, so a bounded retry re-fetches -- and V advances for the synced
subset. v1 syncs rollout replicas only; the reference InferenceWorker is frozen.
"""

import dataclasses

from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import worker_registry


@dataclasses.dataclass(kw_only=True)
class SyncOutcome:
  """Result of one fleet weight sync.

  Attributes:
    version: The version that was staged and requested.
    metadata: The trainer's staging metadata (echoes version).
    synced: Ids of rollout replicas that installed and acked `version`.
    quarantined: Ids of rollout replicas that failed to reach `version`.
  """

  version: int
  metadata: datatypes.WeightSyncMetadata
  synced: list[str]
  quarantined: list[str]

  @property
  def all_synced(self) -> bool:
    return not self.quarantined


class WeightSyncCoordinator:
  """Drives trainer-stage -> per-replica fence + install for one version."""

  def __init__(
      self, registry: worker_registry.WorkerRegistry, *, max_retries: int = 1
  ):
    self._registry = registry
    self._max_retries = max_retries

  def sync(self, version: int, *, param_filter: str = "full") -> SyncOutcome:
    """Stages `version` on the trainer and installs it across rollout replicas.

    Args:
      version: The weight version to mint (orchestrator-assigned).
      param_filter: Which params to stage ("full" or "lora").

    Returns:
      A SyncOutcome listing which replicas reached `version`.

    Raises:
      ValueError: If no trainer worker is registered.
    """
    trainers = self._registry.group("trainer").members()
    if not trainers:
      raise ValueError("no trainer worker registered")
    trainer = trainers[0]

    spec = datatypes.WeightSyncSpec(version=version, param_filter=param_filter)
    metadata = trainer.prepare_weight_sync(spec)

    synced: list[str] = []
    quarantined: list[str] = []
    for replica in self._registry.group("rollout").members():
      worker_id = replica.info().worker_id
      if self._sync_one(replica, metadata, version):
        synced.append(worker_id)
      else:
        quarantined.append(worker_id)

    return SyncOutcome(
        version=version,
        metadata=metadata,
        synced=synced,
        quarantined=quarantined,
    )

  def _sync_one(self, replica, metadata, version: int) -> bool:
    """Fences, installs, and verifies one replica acked `version`."""
    for _ in range(self._max_retries + 1):
      try:
        replica.prepare_weight_sync(metadata)
        acked = replica.sync_weights(metadata)
        if acked != version:
          raise ValueError(
              f"replica acked version {acked}, expected {version}"
          )
        return True
      except Exception:  # pylint: disable=broad-except
        # The worker contract releases the fence on any failure path; a bounded
        # retry re-fetches the same version.
        continue
    return False
