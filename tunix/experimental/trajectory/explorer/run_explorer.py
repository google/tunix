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

"""Entry point for the Trajectory Explorer CLI."""

from collections.abc import Sequence
import dataclasses
import sys
from typing import Any
from absl import app
from absl import flags
from etils import epath
import simple_parsing
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory import trajectory as trajectory_lib
from tunix.experimental.trajectory.explorer import commands
from tunix.experimental.trajectory.explorer.commands import base
from tunix.experimental.trajectory.explorer.commands import ping

_PROGRAM_NAME = "trajectory-explore"


@dataclasses.dataclass(frozen=True, kw_only=True)
class FileTrajectoryStoreConfig:
  """Configuration for the file-backed TrajectoryStore."""

  root_dir: str = simple_parsing.field(
      default="",
      help=(
          "Root directory of a 'file' store (supports local paths and gs://"
          " URLs)."
      ),
  )
  run_id: str = simple_parsing.field(
      default="",
      help="Run to read. Required by the 'file' backend.",
  )

  def build_store_config(self, metadata_type: str) -> dict[str, Any]:
    """Validates the run directory and returns the `from_config` dict.

    Args:
      metadata_type: Registered `TrajectoryMetadata.METADATA_TYPE` the store
        reads metadata back as.

    Returns:
      A `TrajectoryStore.from_config` dict for the file backend.

    Raises:
      ValueError: If the target run directory does not exist.
    """
    if self.root_dir and self.run_id:
      run_path = epath.Path(self.root_dir) / self.run_id
      if not run_path.exists():
        raise ValueError(f"Run directory does not exist: {run_path}")
    return {
        "enabled": True,
        "backend": "file",
        "root_dir": self.root_dir,
        "run_id": self.run_id,
        "metadata_type": metadata_type,
    }


@dataclasses.dataclass(frozen=True, kw_only=True)
class InMemoryTrajectoryStoreConfig:
  """Configuration for the in-memory TrajectoryStore."""

  def build_store_config(self, metadata_type: str) -> dict[str, Any]:
    """Returns the `from_config` dict for the in-memory backend.

    Args:
      metadata_type: Registered `TrajectoryMetadata.METADATA_TYPE` the store
        reads metadata back as.

    Returns:
      A `TrajectoryStore.from_config` dict for the in-memory backend.
    """
    return {
        "enabled": True,
        "backend": "memory",
        "metadata_type": metadata_type,
    }


StoreConfig = FileTrajectoryStoreConfig | InMemoryTrajectoryStoreConfig

# Every registered TrajectoryMetadata subclass; trajectory.py registers both
# base and Tunix metadata on import.
_METADATA_TYPES: tuple[str, ...] = tuple(
    sorted(trajectory_lib.TrajectoryMetadata._REGISTRY)  # pylint: disable=protected-access
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ExplorerConfig:
  """Global flags and subcommand selection."""

  store: StoreConfig = simple_parsing.subgroups(
      {
          "file": FileTrajectoryStoreConfig,
          "memory": InMemoryTrajectoryStoreConfig,
      },
      default="memory",
      help="Trajectory Store backend to read from.",
  )
  metadata_type: str = simple_parsing.choice(
      *_METADATA_TYPES,
      default=trajectory_lib.TrajectoryMetadata.METADATA_TYPE,
      help=(
          "TrajectoryMetadata type the store reads trajectories back as, e.g."
          " 'tunix' to read Tunix rollouts with their first-class reward and"
          " status."
      ),
  )
  json: bool = simple_parsing.field(
      default=False,
      action="store_true",
      help="Output machine-readable JSON.",
  )
  cmd: base.BaseCommand = simple_parsing.subparsers(
      commands.COMMAND_REGISTRY,
      default_factory=ping.PingCommand,
  )


def make_parser() -> simple_parsing.ArgumentParser:
  """Creates the configured simple_parsing ArgumentParser.

  Returns:
    The configured ArgumentParser instance.
  """
  parser = simple_parsing.ArgumentParser(
      prog=_PROGRAM_NAME, description="Trajectory Explorer CLI"
  )
  parser.add_arguments(ExplorerConfig, dest="config")
  return parser


def parse_args(argv: Sequence[str] | None = None) -> ExplorerConfig:
  """Parses command-line arguments into an ExplorerConfig instance.

  Args:
    argv: Optional sequence of command-line arguments.

  Returns:
    An instance of ExplorerConfig containing parsed options and subcommand.
  """
  parser = make_parser()
  args = parser.parse_args(argv)
  return args.config


def get_reader(config: ExplorerConfig) -> store.TrajectoryStore[Any]:
  """Opens the store described by the parsed flags.

  Args:
    config: Parsed CLI configuration.

  Returns:
    The constructed TrajectoryStore instance.

  Raises:
    ValueError: If the selected backend is missing a flag it requires or the
      target run directory does not exist.
  """
  backend_config = config.store.build_store_config(config.metadata_type)
  trajectory_store = store.TrajectoryStore.from_config(backend_config)
  if trajectory_store is None:
    raise ValueError(f"Store config {backend_config} built no store.")
  return trajectory_store


def run_cli(argv: Sequence[str] | None = None) -> int:
  """Parses arguments and executes the selected subcommand.

  Args:
    argv: Optional sequence of command-line arguments.

  Returns:
    Process exit code: 0 on success, 2 for a bad combination of flags, and
    whatever the subcommand returns otherwise.
  """
  config = parse_args(argv)
  try:
    reader = get_reader(config)
  except ValueError as error:
    # argparse exits 2 for a flag it can reject on its own; a flag combination
    # only the backend can reject should look no different to the user.
    print(f"{_PROGRAM_NAME}: error: {error}", file=sys.stderr)
    return 2
  try:
    return config.cmd.execute(reader, output_json=config.json)
  finally:
    reader.close()


def _flags_parser(argv: Sequence[str]) -> Sequence[str]:
  """Pre-parses absl flags without failing on simple_parsing CLI arguments."""
  flags.FLAGS(argv[:1])
  return argv


def main(argv: Sequence[str]) -> None:
  """Main entry point when executed via absl.app.run.

  Args:
    argv: Command-line arguments passed by absl.app.run.
  """
  sys.exit(run_cli(argv[1:]))


def launch_cli() -> None:
  """Launches the CLI via absl.app.run with custom flags parser."""
  app.run(main, flags_parser=_flags_parser)


if __name__ == "__main__":
  launch_cli()
