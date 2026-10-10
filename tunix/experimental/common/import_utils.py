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

"""Import utilities for resolving dynamic symbols."""

import importlib
from typing import Any


def import_symbol(fqn: str) -> Any:
  """Imports a symbol (class or function) from its fully qualified name.

  Args:
    fqn: Fully qualified dot-separated path to the symbol (e.g.
      'package.module.ClassName').

  Returns:
    The resolved symbol object.

  Raises:
    ValueError: If `fqn` does not contain a module path and symbol name.
    ModuleNotFoundError: If the module cannot be imported.
    AttributeError: If the symbol does not exist in the module.
  """
  if "." not in fqn:
    raise ValueError(f"invalid symbol path: {fqn}")
  module_path, *symbol_names = fqn.rsplit(".", maxsplit=1)
  symbol = importlib.import_module(module_path)
  for symbol_name in symbol_names:
    symbol = getattr(symbol, symbol_name)
  return symbol
