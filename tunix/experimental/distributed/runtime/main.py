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

"""Main entry point for distributed process runtime execution."""

import argparse
import importlib
import logging
import sys
from typing import Any


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


def main(argv: list[str]) -> None:
  """Parses command line arguments and runs the configured distributed process.

  Args:
    argv: List of command-line arguments passed to the process.
  """
  parser = argparse.ArgumentParser(description="distributed main")

  # process flags
  parser.add_argument(
      "--process_executor",
      type=str,
      default="tunix.experimental.distributed.runtime.executor.LocalExecutor",
      help="Fully qualified class name of the process executor implementation.",
  )
  parser.add_argument(
      "--process_main",
      type=str,
      default="",
      help="Fully qualified name of the target main function to execute.",
  )

  # discovery flags
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
  # strip the first argument (i.e. the program path), which should be hidden from the process.
  process_argv = process_argv[1:]

  try:
    process_main = import_symbol(context_args.process_main)
  except AttributeError as e:
    logging.error(e)
    logging.error(
        "Does --process_main point to a valid main function? --process_main=%s",
        context_args.process_main,
    )
    return

  process_executor = import_symbol(context_args.process_executor)()

  logging.basicConfig(level=logging.DEBUG, format="%(message)s", force=True)
  process_executor.run(process_main, process_argv, context_args)


if __name__ == "__main__":
  main(sys.argv)
