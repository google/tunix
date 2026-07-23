import argparse
import logging
import pickle

from tunix.experimental.distributed.runtime.context import ProcessContext


def main(argv, context: ProcessContext | None) -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--message", type=str, default="this is knocker!", help=""
  )
  parser.add_argument("--say", type=str, default="", help="")
  args = parser.parse_args(argv)

  logging.info(args.message)

  context.ipc.discovery.register(metadata=pickle.dumps(args.say))
