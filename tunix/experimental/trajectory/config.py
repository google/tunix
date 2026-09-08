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

"""Configuration gating Trajectory Store construction and usage."""

import dataclasses
import re
from typing import Final


# Mirrors `file_store._TRAJECTORY_ID_PATTERN`: a run_id becomes a path segment
# under root_dir, so it is held to the same character rule as the trajectory
# directory names nested beneath it.
_RUN_ID_REGEX: Final[re.Pattern[str]] = re.compile(r"^[a-zA-Z0-9_\-]+$")


@dataclasses.dataclass
class TrajectoryStoreConfig:
  """Configuration for a single process's Trajectory Store.

  One instance of this config is handed to `factory.build_trajectory_store`
  once per process (e.g. inside the constructor of whatever owns that
  process's lifecycle, such as `RolloutWorker` or `StandardRLProgram`); the
  resulting store is held onto for the life of the process rather than
  rebuilt.

  Attributes:
    enabled: Whether to construct a Trajectory Store at all. When False (the
      default), `factory.build_trajectory_store` returns None and every
      store-guarded call site becomes a no-op.
    backend: "file" for `FileTrajectoryStore` (durable; every process that
      points at the same `root_dir`/`run_id` shares the same underlying
      storage) or "memory" for `InMemoryTrajectoryStore` (process-local only
      — not visible to any other process without extra replication).
    root_dir: Base directory for the file backend (local path or `gs://`
      URI). Required when `enabled` and `backend == "file"`.
    run_id: Identifier scoping paths under `root_dir`. Required when
      `enabled` and `backend == "file"`; unused by the memory backend. Must
      be identical across every process sharing one run (the orchestrator and
      every rollout worker), and must stay the same across a process restart
      for resume to work — so generate it once, at whatever single point
      launches the run, and pass the same value to every process. Never
      default it per-process: each process builds its own config, so a
      generated default would give every process a different run_id and
      silently split one run across N directory trees.
    resume_on_restart: Documents intent to read back prior steps after a
      restart. Rejected together with `backend == "memory"`, since a
      process-local store has nothing to resume from once the process that
      held it is gone.
  """

  enabled: bool = False
  backend: str = "file"
  root_dir: str | None = None
  run_id: str | None = None
  resume_on_restart: bool = False

  def __post_init__(self) -> None:
    if not self.enabled:
      return
    if self.backend not in ("file", "memory"):
      raise ValueError(
          "TrajectoryStoreConfig.backend must be 'file' or 'memory', got"
          f" {self.backend!r}."
      )
    if self.backend == "file":
      if not self.root_dir:
        raise ValueError(
            "TrajectoryStoreConfig.root_dir is required when enabled and"
            " backend == 'file'."
        )
      if not self.run_id:
        raise ValueError(
            "TrajectoryStoreConfig.run_id is required when enabled and"
            " backend == 'file'. Without it every run writes straight into"
            " root_dir, and since trajectory ids restart from the first"
            " prompt each run, a later run silently overwrites an earlier"
            " one's trajectories."
        )
      if not _RUN_ID_REGEX.match(self.run_id):
        raise ValueError(
            f"TrajectoryStoreConfig.run_id {self.run_id!r} contains"
            " unsupported characters; only letters, digits, underscores and"
            " hyphens are allowed. A run_id carrying a path separator would"
            " nest the trajectory directories a level deeper than reads look"
            " for them, which fails silently as an empty store."
        )
    if self.backend == "memory" and self.resume_on_restart:
      raise ValueError(
          "TrajectoryStoreConfig(backend='memory', resume_on_restart=True)"
          " is contradictory: a process-local in-memory store has nothing to"
          " resume from once the process that held it exits."
      )
