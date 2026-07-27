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

"""Local distributed runtime process executor."""

from argparse import Namespace
from typing import Callable, List

from tunix.experimental.distributed.runtime.context import ProcessContext
from tunix.experimental.distributed.runtime.contexts.local_context import LocalProcessContext


class LocalExecutor:
  """Process executor that runs a target main function inside a local single-machine runtime context."""

  def run(
      self,
      process_main: Callable[[List[str], ProcessContext], None],
      process_argv: List[str],
      context_args: Namespace,
  ) -> None:
    """Executes `process_main` inside a `LocalProcessContext`.

    Args:
      process_main: Callable entrypoint accepting (argv, context).
      process_argv: Remaining command-line arguments for the target process.
      context_args: Parsed runtime context configuration arguments.
    """
    with LocalProcessContext(context_args) as context:
      process_main(process_argv, context)
