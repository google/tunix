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

"""Loads Raiden's native modules ahead of the TPU backend.

Each links its own XLA, and with it its own copy of
`xla/pjrt/proto/execute_options.proto`. Whichever is dlopened first wins the
protobuf descriptor registry, and one arriving after libtpu has claimed it
aborts the process. Importing jax is not the trigger -- the backend comes up
lazily -- so this only has to run before the first call that touches a device.
"""

import importlib
import logging

# The .so files, not the Python wrappers beside them, so a wrapper going lazy
# cannot silently turn the preload into a no-op.
RAIDEN_MODULES = (
    "_tpu_raiden_jax",
    "_weight_synchronizer_ffi",
    "_kv_cache_manager_ffi",
)

# The wheel is distributed as `tpu_raiden_jax` but installs `tpu_sync`.
_PACKAGE = "tpu_sync.frameworks.jax"


def import_raiden() -> tuple[str, ...]:
  """Imports Raiden's native modules, if the wheel is installed.

  Safe to call from any process, and more than once.

  Returns:
    The modules that were imported, in load order. Empty without Raiden.
  """
  loaded = []
  for name in RAIDEN_MODULES:
    try:
      importlib.import_module(f"{_PACKAGE}.{name}")
    except ImportError:
      # Raiden absent, or an older build that does not ship this module.
      continue
    loaded.append(name)

  if loaded:
    logging.info("Preloaded Raiden modules: %s", ", ".join(loaded))
  return tuple(loaded)
