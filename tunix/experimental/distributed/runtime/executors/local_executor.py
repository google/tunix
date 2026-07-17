from argparse import Namespace
from typing import Callable, List

from tunix.experimental.distributed.runtime.context import ProcessContext
from tunix.experimental.distributed.runtime.contexts.local_context import LocalProcessContext

class LocalExecutor:
  def run(self, process_main: Callable[[List[str], ProcessContext], None], process_argv: List[str], context_args: Namespace) -> None:
    with LocalProcessContext(context_args) as context:
      process_main(process_argv, context)
