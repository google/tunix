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

"""Runtime environment detection and boolean variable parsing utilities."""

import importlib.util
import os
from typing import Final

_INTERNAL_PACKAGE_ROOT: Final[str] = "GOOGLE_INTERNAL_PACKAGE_PATH"
_TRUTHY_ENV_VALUES: Final[frozenset[str]] = frozenset(
    {"1", "true", "yes", "on"}
)
_FALSY_ENV_VALUES: Final[frozenset[str]] = frozenset(
    {"0", "false", "no", "off"}
)


def is_internal_env() -> bool:
  """Returns whether the current runtime has the internal package loader."""
  try:
    spec = importlib.util.find_spec(_INTERNAL_PACKAGE_ROOT)
  except (ImportError, ValueError):
    return False
  return spec is not None and spec.loader is not None


def parse_env_bool(
    name: str,
    *,
    default: bool = False,
) -> bool:
  """Parses boolean environment variable `name`, failing fast on bad tokens.

  Args:
    name: Environment variable name to inspect in `os.environ`.
    default: Fallback boolean returned when `name` is unset or whitespace-only.

  Returns:
    Parsed boolean value (`1/true/yes/on` -> `True`, `0/false/no/off` ->
    `False`), or `default` when `name` is unset or empty after stripping.

  Raises:
    ValueError: If `name` is set to a non-empty value outside the recognized
      boolean tokens.
  """
  raw_value = os.environ.get(name)
  if raw_value is None:
    return default
  raw = raw_value.strip().lower()
  if not raw:
    return default
  if raw in _TRUTHY_ENV_VALUES:
    return True
  if raw in _FALSY_ENV_VALUES:
    return False
  raise ValueError(
      f"Unsupported {name}={raw_value!r}; expected"
      " 0/1/true/false/yes/no/on/off."
  )
