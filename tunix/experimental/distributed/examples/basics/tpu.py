# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example main for running a process with TPUs.

Usage:

  Case 1: Start a process with all TPUs:

    python -m tunix.experimental.distributed.runtime.main \
        --process_main=tunix.experimental.distributed.examples.basics.tpu.main

  Case 2: Start two processes, each with 2 TPUs:

  Process 1 (TPUs 0 and 1 visible):

    TPU_VISIBLE_DEVICES=0,1 \
    TPU_VISIBLE_CHIPS=$TPU_VISIBLE_DEVICES \
    TPU_CHIPS_PER_HOST_BOUNDS=1,2,1 \
    TPU_HOST_BOUNDS=1,1,1 \
    LIBTPU_INIT_ARGS=deepsea_chips_per_host_bounds=$TPU_CHIPS_PER_HOST_BOUNDS,deepsea_host_bounds=$TPU_HOST_BOUNDS \
    python -m tunix.experimental.distributed.runtime.main \
        --process_main=tunix.experimental.distributed.examples.basics.tpu.main

  Process 2 (TPUs 2 and 3 visible):

    TPU_VISIBLE_DEVICES=2,3 \
    TPU_VISIBLE_CHIPS=$TPU_VISIBLE_DEVICES \
    TPU_CHIPS_PER_HOST_BOUNDS=1,2,1 \
    TPU_HOST_BOUNDS=1,1,1 \
    LIBTPU_INIT_ARGS=deepsea_chips_per_host_bounds=$TPU_CHIPS_PER_HOST_BOUNDS,deepsea_host_bounds=$TPU_HOST_BOUNDS \
    python -m tunix.experimental.distributed.runtime.main \
        --process_main=tunix.experimental.distributed.examples.basics.tpu.main

"""

import os
import time

import jax
from tunix.experimental.distributed.runtime.context import ProcessContext


def main(argv, context: ProcessContext | None) -> None:
  for device in jax.devices():
    print(repr(device))

  print("Press Ctrl+C to exit...")
  try:
    while True:
      time.sleep(86400)
  except KeyboardInterrupt:
    pass
