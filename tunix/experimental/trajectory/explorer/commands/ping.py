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

"""Minimal placeholder command to verify CLI execution."""

import dataclasses
import json
import termcolor
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory.explorer.commands import base


@dataclasses.dataclass(frozen=True, kw_only=True)
class PingCommand(base.BaseCommand):
  """Verifies trajectory store connectivity and reports trajectory count."""

  def execute(
      self, reader: store.TrajectoryReader, output_json: bool = False
  ) -> int:
    """Executes the ping command against the reader.

    Args:
      reader: The trajectory store reader backend.
      output_json: Whether to format output as machine-readable JSON.

    Returns:
      Process exit code (0 for success).
    """
    metas = reader.get_trajectories_metadata()
    if output_json:
      print(json.dumps({"status": "ok", "trajectories_count": len(metas)}))
      return 0

    print(
        termcolor.colored(
            "Trajectory Explorer initialized successfully."
            f" ({len(metas)} trajectories accessible)",
            "green",
        )
    )
    return 0
