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

"""Factory function to dynamically construct the appropriate ProcessContext."""

import os
from typing import Any

from tunix.experimental.distributed.runtime.context import ProcessContext
from tunix.experimental.distributed.runtime.contexts.borg_context import BorgProcessContext
from tunix.experimental.distributed.runtime.contexts.k8s_context import K8sProcessContext
from tunix.experimental.distributed.runtime.contexts.local_context import LocalProcessContext


def get_default_process_context(args: Any) -> ProcessContext:
  """Automatically creates the appropriate ProcessContext based on the runtime environment.

  If running under Borg / XManager, returns BorgProcessContext.
  If running under Kubernetes JobSet, returns K8sProcessContext.
  Otherwise, returns LocalProcessContext.

  Args:
    args: Command line or parsed namespace arguments.

  Returns:
    An instance of ProcessContext suitable for the detected platform.
  """
  if (
      os.getenv("BORG_TASK_HANDLE")
      or os.getenv("BORG_JOB_NAME")
      or os.getenv("BORG_ALLOC_DIR")
      or os.getenv("XM_BORG_MODE") == "true"
  ):
    return BorgProcessContext(args)

  if (
      os.getenv("KUBERNETES_SERVICE_HOST")
      or os.getenv("JOBSET_NAME")
      or os.getenv("POD_NAME")
  ):
    return K8sProcessContext(args)

  return LocalProcessContext(args)
