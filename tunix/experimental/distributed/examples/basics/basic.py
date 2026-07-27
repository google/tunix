from tunix.experimental.distributed.runtime.context import ProcessContext


def main(argv, context: ProcessContext | None) -> None:
  # use "argv" to access command line flags
  # use "context" to access platform-dependent apis
  #   e.g. context.jax.initialize()

  # ...write your own logic...
  print("hello world")
