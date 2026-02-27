# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Root pytest configuration."""

import os
import sys


def _suppress_libtpu_on_non_tpu_machines() -> None:
  """Prevents a false vLLM platform conflict on CUDA machines with jax[tpu].

  vLLM detects the TPU platform solely by checking if `libtpu` is importable
  (vllm/platforms/__init__.py::tpu_platform_plugin). The `jax[tpu]` extra
  installs `libtpu` as a dependency, so CUDA-only machines that have the prod
  dependencies installed appear to have both CUDA and TPU, causing vLLM to
  raise RuntimeError during test collection.

  We block libtpu from being imported when no TPU hardware is detected.
  On real Cloud TPU VMs the /dev/accel* device files are present, so the
  suppression is skipped and TPU developers are unaffected.
  """
  # Do nothing if libtpu was already successfully loaded (e.g. JAX TPU backend
  # initialised before conftest ran), or if real TPU devices are present.
  if 'libtpu' in sys.modules:
    return
  tpu_device_paths = ['/dev/accel0', '/dev/accel1']
  if any(os.path.exists(p) for p in tpu_device_paths):
    return

  # Insert a sentinel so that `import libtpu` raises ImportError, preventing
  # vLLM's tpu_platform_plugin from activating on this machine.
  sys.modules['libtpu'] = None  # type: ignore[assignment]


_suppress_libtpu_on_non_tpu_machines()
