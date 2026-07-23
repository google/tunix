# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main entry point for distributed process execution.

Note that the "process" abstraction represents a logical distributed process
and does not map directly to an OS process; when multiple processes are
specified within a single invocation, they execute concurrently as OS threads
sharing the same `--process_executor`.

TODO: choose a dedicated name for "process" to avoid confusion. e.g. "LogicalProcess".

Usage Examples:
  1. Single "process" execution:
    ```shell
    python -m tunix.experimental.distributed.runtime.main \
        --process_main=basics.flag.main \
        --message="hello flag"
    ```

  2. Multi "process" execution separated by `--process`:
    ```shell
    python -m tunix.experimental.distributed.runtime.main \
        --process_executor=tunix.experimental.distributed.runtime.executor.LocalExecutor \
        --process \
            --process_main=basics.door.main \
            --discovery_id=door \
            --discovery_port=12345 \
        --process \
            --process_main=basics.knocker.main \
            --discovery_addrs=door:12345 \
            --say="open the door"
    ```
"""

import argparse
import concurrent.futures
import dataclasses
import importlib
import logging
import os
import sys
from typing import Any, Callable


@dataclasses.dataclass(frozen=True)
class PreparedProcess:
  """Parsed configuration and entrypoint for a distributed process.

  Attributes:
    main_fn: Resolved callable entrypoint for the process.
    argv: Unrecognized command-line arguments forwarded to `main_fn`.
    context_args: Parsed runtime discovery and execution flags.
  """

  main_fn: Callable[..., Any]
  argv: list[str]
  context_args: argparse.Namespace


def import_symbol(fqn: str) -> Any:
  """Imports a symbol (class or function) from its fully qualified name.

  Args:
    fqn: Fully qualified dot-separated path to the symbol (e.g.
      'package.module.ClassName').

    Returns:
      The resolved symbol object.

    Raises:
      ValueError: If `fqn` does not contain a module path and symbol name.
      ModuleNotFoundError: If the module cannot be imported.
      AttributeError: If the symbol does not exist in the module.
  """
  if "." not in fqn:
    raise ValueError(f"invalid symbol path: {fqn}")
  module_path, *symbol_names = fqn.rsplit(".", maxsplit=1)
  symbol = importlib.import_module(module_path)
  for symbol_name in symbol_names:
    symbol = getattr(symbol, symbol_name)
  return symbol


def split_process_argv(argv: list[str]) -> list[list[str]]:
  """Splits command-line arguments into per-process slices separated by `--process`.

  Args:
    argv: List of command-line arguments after stripping the script path.

  Returns:
    A list of argument lists, where each entry corresponds to one process.
  """
  if "--process" not in argv:
    return [argv]

  slices: list[list[str]] = []
  current_slice: list[str] = []
  for token in argv:
    if token == "--process":
      if current_slice:
        slices.append(current_slice)
        current_slice = []
    else:
      current_slice.append(token)
  if current_slice:
    slices.append(current_slice)

  return slices


def prepare_process(argv: list[str]) -> PreparedProcess:
  """Parses discovery flags and imports the target entrypoint for a single process.

  Args:
    argv: Command-line arguments slice belonging to this process.

  Returns:
    A `PreparedProcess` containing the imported main function, application
    arguments, and parsed discovery context arguments.

  Raises:
    ValueError: If `--process_main` cannot be imported or resolved.
  """
  parser = argparse.ArgumentParser(
      description="process main", allow_abbrev=False, add_help=False
  )

  parser.add_argument(
      "--process_main",
      type=str,
      default="",
      help="Fully qualified name of the target main function to execute.",
  )
  parser.add_argument(
      "--discovery_id",
      type=str,
      default="",
      help="Id to identify the process. Id and port form a discovery address.",
  )
  parser.add_argument(
      "--discovery_port",
      type=int,
      default=0,
      help=(
          "Port of this process, that other processes register themselves to."
          " If non-zero, will run a service."
      ),
  )
  parser.add_argument(
      "--discovery_addrs",
      type=str,
      default="",
      help="Addresses of other processes, that this process registers to.",
  )

  context_args, process_argv = parser.parse_known_args(argv)

  try:
    process_main = import_symbol(context_args.process_main)
  except AttributeError as e:
    raise ValueError(
        f"Invalid --process_main={context_args.process_main}: {e}"
    ) from e

  return PreparedProcess(
      main_fn=process_main,
      argv=process_argv,
      context_args=context_args,
  )


def main(argv: list[str]) -> None:
  """Main entry point for distributed process runtime execution.

  Parses global executor settings, splits per-process argument slices delimited
  by `--process`, and executes the configured processes using the executor.

  Args:
    argv: System command-line arguments (`sys.argv`).
  """
  parser = argparse.ArgumentParser(
      description="distributed main", allow_abbrev=False, add_help=False
  )

  # The same executor is used for all processes started in one invocation.
  parser.add_argument(
      "--process_executor",
      type=str,
      default="tunix.experimental.distributed.runtime.executor.LocalExecutor",
      help="Fully qualified class name of the process executor implementation.",
  )

  main_args, processes_argv = parser.parse_known_args(argv)
  # Strip the first argument (program path), which should be hidden from processes.
  processes_argv = processes_argv[1:]

  process_executor = import_symbol(main_args.process_executor)()

  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s",
      force=True,
  )

  prepared_processes = [
      prepare_process(slice_argv)
      for slice_argv in split_process_argv(processes_argv)
  ]

  if len(prepared_processes) == 1:
    prepared = prepared_processes[0]
    process_executor.run(prepared.main_fn, prepared.argv, prepared.context_args)
  else:
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(prepared_processes)
    ) as executor:
      futures = [
          executor.submit(
              process_executor.run,
              prepared.main_fn,
              prepared.argv,
              prepared.context_args,
          )
          for prepared in prepared_processes
      ]
      try:
        for future in concurrent.futures.as_completed(futures):
          future.result()
      except SystemExit as e:
        code = (
            e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        )
        if code != 0:
          logging.exception(
              "Distributed process exited with non-zero code. Forcefully"
              " terminating application."
          )
        os._exit(code)
      except BaseException:
        logging.exception(
            "Distributed process execution failed or interrupted. Forcefully"
            " terminating application."
        )
        os._exit(1)


if __name__ == "__main__":
  main(sys.argv)
