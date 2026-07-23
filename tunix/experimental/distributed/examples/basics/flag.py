import argparse

from tunix.experimental.distributed.runtime.context import ProcessContext


def main(argv, context: ProcessContext | None) -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--message", type=str, default="", help="")
  args = parser.parse_args(argv)

  print(args.message)
