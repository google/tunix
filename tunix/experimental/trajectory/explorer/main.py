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
from absl import app
from absl import flags
import simple_parsing
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory.explorer import commands
from tunix.experimental.trajectory.explorer.commands import base
from tunix.experimental.trajectory.explorer.commands import ping

_PROGRAM_NAME = "trajectory-explore"


@dataclasses.dataclass(frozen=True, kw_only=True)
class ExplorerConfig:
  """Global flags and subcommand selection."""

  backend: str = simple_parsing.field(
      default="memory",
      choices=("file", "memory"),
      help="Trajectory Store backend to read from.",
  )
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


def get_reader(config: ExplorerConfig) -> store.TrajectoryReader:
  """Opens the store described by the parsed flags.

  The Explorer never writes, so the result is only ever held as a
  `TrajectoryReader`.

  Args:
    config: Parsed CLI configuration.

  Returns:
    A reader over the selected store.

  Raises:
    ValueError: If the selected backend is missing a flag it requires.
  """
  reader = store.TrajectoryStore.from_config({
      "enabled": True,
      "backend": config.backend,
      "root_dir": config.root_dir,
      "run_id": config.run_id,
  })
  if reader is None:
    raise ValueError("Failed to initialize Trajectory Store reader.")
  return reader


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
    close_fn = getattr(reader, "close", None)
    if callable(close_fn):
      close_fn()


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
