import argparse
import concurrent.futures
import importlib
import logging
import sys

from typing import Any, List

def import_symbol(fqn: str) -> Any:
  """Imports a symbol (class or function) from its fully qualified name."""
  if "." not in fqn:
    raise ValueError(f"invalid symbol path: {fqn}")
  module_path, *symbol_names = fqn.rsplit(".", maxsplit=1)
  symbol = importlib.import_module(module_path)
  for symbol_name in symbol_names:
      symbol = getattr(symbol, symbol_name)
  return symbol

def split_argv(argv: List[str]) -> List[List[str]]:
  """Splits command-line arguments, separated by --process."""
  if "--process" not in argv:
    return [argv]

  slices: List[List[str]] = []
  current_slice: List[str] = []
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

def prepare_process(
    argv: list[str],
) -> tuple[Any, list[str], argparse.Namespace]:
  """Prepares a process for execution by parsing arguments and importing the main function.

  Args:
    argv: Command-line arguments.
  """
  parser = argparse.ArgumentParser(description="process main", allow_abbrev=False)

  parser.add_argument("--process_main", type=str, default="", help="Fully qualified name of the target main function to execute.")
  parser.add_argument("--discovery_id", type=str, default="", help="Id to identify the process. Id and port form a discovery address.")
  parser.add_argument("--discovery_port", type=int, default=0, help="Port of this process, that other processes register themselves to. If non-zero, will run a service.")
  parser.add_argument("--discovery_addrs", type=str, default="", help="Addresses of other processes, that this process registers to.")

  context_args, process_argv = parser.parse_known_args(argv)
  process_main = import_symbol(context_args.process_main)

  return (process_main, process_argv, context_args)

def main(argv):
  parser = argparse.ArgumentParser(description="distributed main", allow_abbrev=False)

  parser.add_argument("--process_executor", type=str, default="tunix.experimental.distributed.runtime.executor.LocalExecutor", help="")

  main_args, processes_argv = parser.parse_known_args(argv)

  # strip the first argument (i.e. the program path), which should be hidden from the process.
  processes_argv = processes_argv[1:]

  process_executor = import_symbol(main_args.process_executor)()

  logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s %(message)s", force=True)

  # prepare processes
  processes = []
  for process_argv in split_argv(processes_argv):
    processes.append(prepare_process(process_argv))

  # run processes
  if len(processes) == 1:
    process_main, process_argv, context_args = processes[0]
    process_executor.run(process_main, process_argv, context_args)
  else:
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(processes)
    ) as executor:
      for process in processes:
        executor.submit(
            process_executor.run, process[0], process[1], process[2]
        )

if __name__ == '__main__':
  main(sys.argv)
