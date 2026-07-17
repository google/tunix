from argparse import Namespace
from typing import Callable, List

from tunix.experimental.distributed.runtime.context import ProcessContext
from tunix.experimental.distributed.runtime.contexts.k8s_context import K8sProcessContext

class K8sExecutor:
  def run(self, process_main: Callable[[List[str], ProcessContext], None], process_argv: List[str], context_args: Namespace) -> None:
    with K8sProcessContext(context_args) as context:
      process_main(process_argv, context)
