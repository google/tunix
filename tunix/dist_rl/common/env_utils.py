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

"""Environment variable parsing utilities for distributed RL."""

# Intentionally separate from the top-level `tunix/utils/env_utils.py`.
# That module imports `flax` at module scope, and its BUILD target
# (`utils:env_utils`) depends on `//pyglib:gfile`,
# `//third_party/py/flax:core`, and `//third_party/py/pathwaysutils`.
# This module is a dep of `dist_rl/distributed/runtime:context`, whose
# `context_no_ml_deps_test` genquery guard rejects any JAX, Flax, Optax,
# PyTorch, or TPU Raiden target in `deps(:context)`; merging the two would
# pull Flax into that closure and fail the guard. Keep this module
# standard-library only (its BUILD target has no deps).

import os
from typing import Final

_TRUTHY_ENV_VALUES: Final[frozenset[str]] = frozenset({"1", "true", "yes"})
_FALSY_ENV_VALUES: Final[frozenset[str]] = frozenset({"0", "false", "no"})


def parse_env_bool(
    name: str,
    *,
    default: bool = False,
    strict: bool = False,
) -> bool:
  """Returns whether environment variable `name` is set to a truthy boolean."""
  raw = os.environ[name].strip().lower() if name in os.environ else ""
  if not raw:
    return default
  if raw in _TRUTHY_ENV_VALUES:
    return True
  if not strict or raw in _FALSY_ENV_VALUES:
    return False
  raise ValueError(
      f"Unsupported {name}={os.environ[name]!r}; expected"
      " 0/1/true/false/yes/no."
  )
