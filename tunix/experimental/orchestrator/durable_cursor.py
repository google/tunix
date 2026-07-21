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

"""Durable cursor and checkpoint cadence for resumable runs.

The durable cursor is the small piece of orchestrator state that must survive a
restart: the global step, the weight version, the lineage incarnation, and the
dataset position. v0 persists it as a step-boundary JSON sidecar next to the
trainer checkpoint (full atomic Tier-1 crash-safety comes later). The
`CheckpointCoordinator` owns the cadence: on a save boundary it checkpoints the
trainer and writes the cursor; on resume it restores the trainer and returns the
last cursor so the loop can pick up where it stopped.
"""

import dataclasses
import json
import pathlib


@dataclasses.dataclass(kw_only=True)
class DurableCursor:
  """Minimal restartable orchestrator state (Tier-1, v0).

  Attributes:
    global_step: Optimizer steps completed.
    weight_version: Installed weight/policy version.
    incarnation: Lineage epoch (bumped on rewind).
    dataset_cursor: Position in the dataset (index or opaque state token).
    seed: Seed salt for the shuffle/RNG chain.
  """

  global_step: int = 0
  weight_version: int = 0
  incarnation: int = 0
  dataset_cursor: int = 0
  seed: int = 0

  def to_dict(self) -> dict[str, int]:
    return dataclasses.asdict(self)

  @classmethod
  def from_dict(cls, data: dict[str, int]) -> "DurableCursor":
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


class CheckpointCoordinator:
  """Saves a trainer checkpoint + durable cursor on a step cadence; resumes them."""

  def __init__(self, trainer, cursor_path, *, save_every_n_steps: int = 1):
    self._trainer = trainer
    self._cursor_path = pathlib.Path(cursor_path)
    self._save_every_n_steps = save_every_n_steps

  def should_save(self, global_step: int) -> bool:
    return (
        self._save_every_n_steps > 0
        and global_step % self._save_every_n_steps == 0
    )

  def save(self, cursor: DurableCursor) -> None:
    """Checkpoints the trainer and writes the durable cursor JSON."""
    self._trainer.save_checkpoint(cursor.to_dict())
    self._cursor_path.parent.mkdir(parents=True, exist_ok=True)
    self._cursor_path.write_text(json.dumps(cursor.to_dict()))

  def maybe_save(self, cursor: DurableCursor) -> bool:
    """Saves iff `cursor.global_step` is on the cadence; returns whether it did."""
    if not self.should_save(cursor.global_step):
      return False
    self.save(cursor)
    return True

  def resume(self) -> DurableCursor | None:
    """Restores the trainer and returns the last cursor, or None if none exists."""
    if not self._cursor_path.exists():
      return None
    self._trainer.restore_checkpoint()
    return DurableCursor.from_dict(json.loads(self._cursor_path.read_text()))
