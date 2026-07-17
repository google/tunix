import argparse
from concurrent import futures
import logging
import pickle
import time

from tunix.experimental.distributed.runtime.context import ProcessContext

def main(argv, context: ProcessContext | None) -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--message", type=str, default="this is door!", help="")
  args = parser.parse_args(argv)

  logging.info(args.message)

  knock_future = futures.Future()
  context.ipc.discovery.on_register(callback=lambda hostname, _, metadata: (
    logging.info(f"{hostname} knocked and said: {pickle.loads(metadata)}"),
    knock_future.set_result(True)
    ))

  knock_future.result()
