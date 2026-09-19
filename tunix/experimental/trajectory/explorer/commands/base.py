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

"""Base protocol for Trajectory Explorer CLI subcommands."""

import dataclasses
from typing import Any, ClassVar, Protocol
from tunix.experimental.trajectory import store


class BaseCommand(Protocol):
  """Base protocol that every explorer subcommand must inherit from and implement.

  How to implement a new command:
  1. Create `commands/<name>.py`.
  2. Define a `@dataclasses.dataclass(frozen=True, kw_only=True)` command class
     inheriting from `base.BaseCommand`.
  3. Register your command in `COMMAND_REGISTRY` in `commands/__init__.py`.
  4. Write unit tests in `commands/<name>_test.py`.
  """

  __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]

  def execute(
      self, reader: store.TrajectoryReader, output_json: bool = False
  ) -> int:
    """Executes the command against a TrajectoryReader and returns an exit code.

    Args:
      reader: The trajectory store reader backend.
      output_json: Whether to format output as machine-readable JSON.

    Returns:
      Process exit code (0 for success, non-zero for failure).
    """
    ...
