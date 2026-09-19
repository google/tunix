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

"""Testing and mock data utilities for Trajectory Explorer CLI commands."""

from collections.abc import Sequence
import contextlib
import io

from tunix.experimental.trajectory import in_memory_store
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.trajectory import trajectory_testing
from tunix.experimental.trajectory.explorer.commands import base


def create_store(
    prefill: bool = True,
    trajectories: Sequence[trajectory_lib.Trajectory] | None = None,
) -> in_memory_store.InMemoryTrajectoryStore:
  """Creates an in-memory trajectory store optionally pre-populated with test fixtures.

  Args:
    prefill: If True and `trajectories` is None, populates the store with
      `trajectory_testing.TRAJECTORY_1` and `trajectory_testing.TRAJECTORY_2`.
    trajectories: Optional explicit sequence of trajectories to populate into
      the store instead of the default fixtures.

  Returns:
    An initialized InMemoryTrajectoryStore instance.
  """
  mem_store = in_memory_store.InMemoryTrajectoryStore()
  items = trajectories
  if items is None and prefill:
    items = (
        trajectory_testing.TRAJECTORY_1,
        trajectory_testing.TRAJECTORY_2,
    )
  if items:
    for traj in items:
      meta = trajectory_lib.TrajectoryMetadata(
          **traj.model_dump(exclude={"steps", "subagent_trajectories"})
      )
      if not traj.steps:
        mem_store.update_metadata(meta)
      else:
        for step in traj.steps:
          mem_store.add_step(step, meta)
  return mem_store


def execute_command(
    cmd: base.BaseCommand,
    reader: store.TrajectoryReader,
    *,
    output_json: bool = False,
) -> tuple[int, str]:
  """Executes a CLI command against a reader while capturing standard output.

  Args:
    cmd: The CLI command instance to execute.
    reader: The TrajectoryReader to run the command against.
    output_json: Whether to request machine-readable JSON output.

  Returns:
    A tuple of `(exit_code, stdout_output)`.
  """
  buffer = io.StringIO()
  with contextlib.redirect_stdout(buffer):
    exit_code = cmd.execute(reader, output_json=output_json)
  return exit_code, buffer.getvalue()
