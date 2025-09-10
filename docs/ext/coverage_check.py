# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Asserts all public symbols are covered in the docs."""

from collections.abc import Mapping
import inspect
import types
from typing import Any, Sequence, Tuple

import optax
from sphinx import application
from sphinx import builders
from sphinx import errors


def find_internal_python_modules(
    root_module: types.ModuleType,
) -> Sequence[Tuple[str, types.ModuleType]]:
  """Returns `(name, module)` for all Optax submodules under `root_module`."""
  modules = set([(root_module.__name__, root_module)])
  visited = set()
  to_visit = [root_module]

  while to_visit:
    mod = to_visit.pop()
    visited.add(mod)

    for name in dir(mod):
      obj = getattr(mod, name)
      if inspect.ismodule(obj) and obj not in visited:
        if obj.__name__.startswith("optax"):
          if "_src" not in obj.__name__:
            to_visit.append(obj)
            modules.add((obj.__name__, obj))

  return sorted(modules)


def optax_public_symbols():
  """Collect all optax public symbols."""
  names = set()
  for module_name, module in find_internal_python_modules(optax):
    for name in module.__all__:
      names.add(module_name + "." + name)
  return names


class OptaxCoverageCheck(builders.Builder):
  """Builder that checks all public symbols are included."""

  name = "coverage_check"

  def get_outdated_docs(self) -> str:
    return "coverage_check"

  def write(self, *ignored: Any) -> None:  # pylint: disable=overridden-final-method
    pass

  def finish(self) -> None:
    documented_objects = frozenset(self.env.domaindata["py"]["objects"])  # pytype: disable=attribute-error
    undocumented_objects = set(optax_public_symbols()) - documented_objects
    if undocumented_objects:
      undocumented_objects = tuple(sorted(undocumented_objects))
      raise errors.SphinxError(
          "All public symbols must be included in our documentation, did you "
          "forget to add an entry to `api.rst`?\n"
          f"Undocumented symbols: {undocumented_objects}"
      )

  def get_target_uri(self, docname, typ=None):
    raise NotImplementedError

  def write_doc(self, docname, doctree):
    raise NotImplementedError


def setup(app: application.Sphinx) -> Mapping[str, Any]:
  app.add_builder(OptaxCoverageCheck)
  return {"version": optax.__version__, "parallel_read_safe": True}
