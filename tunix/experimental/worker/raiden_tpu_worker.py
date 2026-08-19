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

"""Per-host TPU worker: owns the weight buffers and runs the copies."""

from __future__ import annotations

import dataclasses
from typing import Any

from tunix.experimental.worker import raiden_synchronizer


class RaidenTpuWorker:
  """One host's device-buffer owner and transport executor."""

  def __init__(
      self,
      job_name: str,
      worker_index: int = 0,
      *,
      auto_h2d: bool = False,
      host_stage: bool = False,
  ):
    self.job_name = job_name
    self.worker_index = worker_index
    self._auto_h2d = auto_h2d
    self._host_stage = host_stage
    self._synchronizer: Any = None

  @property
  def bound(self) -> bool:
    return self._synchronizer is not None

  def bind(self, state: Any) -> None:
    """Binds the transport; host_stage pulls proxy arrays to host memory first."""
    if self._host_stage:
      state = raiden_synchronizer.to_host_cpu_state(state)
    if self._synchronizer is None:
      self._synchronizer = raiden_synchronizer.RaidenSynchronizer(
          self.job_name, state, auto_h2d=self._auto_h2d
      )
    else:
      self._synchronizer.rebind(state)

  def work_unit_metadata(self) -> Any:
    """One WorkUnitMetadata per host, per the neutral contract."""
    md = self._synchronizer.work_unit_metadata()
    if self.worker_index:
      md = dataclasses.replace(
          md,
          unit=dataclasses.replace(
              md.unit, job_replica_id=str(self.worker_index)
          ),
      )
    return md

  def d2h(self) -> None:
    self._synchronizer.d2h()

  def h2d(self) -> None:
    self._synchronizer.h2d()

  def metrics(self) -> dict:
    return self._synchronizer.metrics() if self._synchronizer else {}

  def checksums(self) -> dict:
    return self._synchronizer.checksums() if self._synchronizer else {}