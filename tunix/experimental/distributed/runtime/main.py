import argparse
import importlib
import logging
import sys

from typing import Any

def import_symbol(fqn: str) -> Any:
  """Imports a symbol (class or function) from its fully qualified name."""
  if "." not in fqn:
    raise ValueError(f"invalid symbol path: {fqn}")
  module_path, *symbol_names = fqn.rsplit(".", maxsplit=1)
  symbol = importlib.import_module(module_path)
  for symbol_name in symbol_names:
      symbol = getattr(symbol, symbol_name)
  return symbol

def main(argv):
  parser = argparse.ArgumentParser(description="distributed main")

  # process flags
  parser.add_argument("--process_executor", type=str, default="tunix.experimental.distributed.runtime.executor.LocalExecutor", help="")
  parser.add_argument("--process_main", type=str, default="", help="")

  # discovery flags
  parser.add_argument("--discovery_id", type=str, default="", help="Id to identify the process. Id and port form a discovery address.")
  parser.add_argument("--discovery_port", type=int, default=0, help="Port of this process, that other processes register themselves to. If non-zero, will run a service.")
  parser.add_argument("--discovery_addrs", type=str, default="", help="Addresses of other processes, that this process registers to.")

  context_args, process_argv = parser.parse_known_args(argv)
  # strip the first argument (i.e. the program path), which should be hidden from the process.
  process_argv = process_argv[1:]

  try:
    process_main = import_symbol(context_args.process_main)
  except AttributeError as e:
    logging.error(e)
    logging.error(f"is --process_main point to main function ? --process_main={context_args.process_main}")
    return

  process_executor = import_symbol(context_args.process_executor)()

  logging.basicConfig(level=logging.DEBUG, format="%(message)s", force=True)
  process_executor.run(process_main, process_argv, context_args)

if __name__ == '__main__':
  main(sys.argv)
