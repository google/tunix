# Copyright 2026 Google LLC
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

"""Standalone component registry supporting decorator pattern for dynamic class resolution."""

import ast
import contextlib
import importlib
import importlib.util
import os
import pkgutil
import sys
from typing import Any, Callable, Dict, Optional, Sequence, Type, TypeVar, Union, overload

T = TypeVar("T")


class Registry:
  """Generic registry mapping string keys to class implementations.

  Supports registration via decorator pattern:

    agent_registry = Registry("AgentRegistry")

    @agent_registry.register("diagnostic")
    class DiagnosticAgent:
      ...

    # Auto-registers using the class's __name__ if string key is omitted:
    @agent_registry.register()
    class CustomAgent:
      ...

    # Or as a bare decorator without parentheses:
    @agent_registry.register
    class BareAgent:
      ...

    agent_cls = agent_registry.get("diagnostic")
  """

  def __init__(self, name: str = "Registry"):
    self.name = name
    self._registry: Dict[str, Type[Any]] = {}

  def register(
      self, name: Optional[Union[str, Type[T]]] = None
  ) -> Union[Type[T], Callable[[Type[T]], Type[T]]]:
    """Decorator to register a class under a specific string key.

    Supports registration as:
      @registry.register("custom_name")
      @registry.register()
      @registry.register

    Args:
      name: Optional string key to register the class under, or the class itself
        when used as a bare decorator.

    Returns:
      The registered class or a decorator function that registers the class.
    """
    if isinstance(name, type):
      cls = name
      reg_name = cls.__name__
      if reg_name in self._registry:
        raise KeyError(
            f"Key '{reg_name}' is already registered in {self.name} "
            f"for class '{self._registry[reg_name].__name__}'."
        )
      self._registry[reg_name] = cls
      return cls

    def decorator(cls: Type[T]) -> Type[T]:
      reg_name = name if name is not None else cls.__name__
      if reg_name in self._registry:
        raise KeyError(
            f"Key '{reg_name}' is already registered in {self.name} "
            f"for class '{self._registry[reg_name].__name__}'."
        )
      self._registry[reg_name] = cls
      return cls

    return decorator

  def get(self, name: str) -> Type[Any]:
    """Resolves a registered string key to a class.

    Args:
      name: Registered string key (e.g., "k8s").

    Returns:
      The resolved class object.

    Raises:
      KeyError: If the key is not registered.
    """
    if name in self._registry:
      return self._registry[name]

    raise KeyError(
        f"'{name}' is not registered in {self.name}. "
        f"Available keys: {list(self._registry.keys())}"
    )

  def contains(self, name: str) -> bool:
    """Checks if a key is registered."""
    return name in self._registry

  def __contains__(self, name: str) -> bool:
    return self.contains(name)

  def keys(self) -> list[str]:
    """Returns list of registered keys."""
    return list(self._registry.keys())


# Pre-instantiated global registries for Agent and Environment classes
AGENT_REGISTRY = Registry("AgentRegistry")
ENV_REGISTRY = Registry("EnvRegistry")

# Convenience decorator aliases
register_agent = AGENT_REGISTRY.register
register_env = ENV_REGISTRY.register


def has_registry_decorator(file_path: str) -> bool:
  """Inspects a Python file AST to check if any class definition uses a registry decorator."""
  try:
    with open(file_path, "r", encoding="utf-8") as f:
      tree = ast.parse(f.read(), filename=file_path)

    names = {"register_agent", "register_env", "AGENT_REGISTRY", "ENV_REGISTRY"}
    for node in ast.walk(tree):
      if isinstance(node, ast.ImportFrom):
        is_agentic = bool(
            node.level
            or not node.module
            or "agentic" in node.module
            or node.module == "registry"
        )
        for a in node.names:
          if a.name in names or (
              is_agentic and a.name in ("register", "registry")
          ):
            names.add(a.asname or a.name)
      elif isinstance(node, ast.Import):
        for a in node.names:
          if "agentic" in a.name and "registry" in a.name:
            names.add(a.asname or a.name)
      elif isinstance(node, ast.Assign):
        func = (
            node.value.func
            if isinstance(node.value, ast.Call)
            else node.value
        )
        val = getattr(func, "id", None) or getattr(func, "attr", None)
        if val in names or val == "Registry":
          names.update(t.id for t in node.targets if isinstance(t, ast.Name))

    for node in ast.walk(tree):
      if isinstance(node, ast.ClassDef):
        for dec in node.decorator_list:
          fn = dec.func if isinstance(dec, ast.Call) else dec
          val_id = getattr(getattr(fn, "value", None), "id", None)
          if (
              getattr(fn, "id", None) in names
              or getattr(fn, "attr", None) in names
              or (val_id in names and getattr(fn, "attr", None) == "register")
          ):
            return True
    return False
  except Exception:  # pylint: disable=broad-exception-caught
    return False


def _file_path_to_module_name(file_path: str) -> Optional[str]:
  """Converts an absolute .py file path to its canonical Python module dot-name using sys.path."""
  abs_path = os.path.abspath(file_path)
  for entry in sys.path:
    if not entry:
      continue
    entry_abs = os.path.abspath(entry)
    if abs_path.startswith(entry_abs + os.sep):
      rel = os.path.relpath(abs_path, entry_abs)
      return os.path.splitext(rel)[0].replace(os.sep, ".")
  return None


@contextlib.contextmanager
def _add_sys_path(path: str):
  """Context manager that temporarily adds a directory to sys.path if not already present."""
  abs_path = os.path.abspath(path)
  inserted = False
  if _file_path_to_module_name(abs_path) is None and abs_path not in sys.path:
    sys.path.insert(0, abs_path)
    inserted = True
  try:
    yield abs_path
  finally:
    if inserted:
      try:
        sys.path.remove(abs_path)
      except ValueError:
        pass


def auto_discover_modules(*package_or_directory_paths: str) -> None:
  """Recursively finds and imports Python modules that contain registry decorators.

  Only imports files whose AST contains a class decorated with a registry
  decorator,
  avoiding unnecessary side-effects or heavy imports from unrelated files.

  Args:
    *package_or_directory_paths: Python import paths (e.g., "my_project.agents")
      or filesystem directory paths (e.g., "/path/to/environments" or
      "./my_envs").
  """
  for target in package_or_directory_paths:
    if os.path.exists(target) and os.path.isdir(target):
      target_abs = os.path.abspath(target)
      with _add_sys_path(target_abs):
        # Scan directory for .py files containing registry decorators
        for root, _, files in os.walk(target_abs):
          for file in files:
            if file.endswith(".py") and not file.startswith("__"):
              file_path = os.path.join(root, file)
              if has_registry_decorator(file_path):
                mod_name = _file_path_to_module_name(file_path)
                if mod_name:
                  importlib.import_module(mod_name)
    else:
      # Treat as Python package/module import path
      mod = importlib.import_module(target)
      if hasattr(mod, "__path__"):
        for _, sub_name, _ in pkgutil.walk_packages(
            mod.__path__, prefix=mod.__name__ + "."
        ):
          try:
            spec = importlib.util.find_spec(sub_name)
            if spec and spec.origin and os.path.exists(spec.origin):
              if has_registry_decorator(spec.origin):
                importlib.import_module(sub_name)
          except Exception:  # pylint: disable=broad-exception-caught
            pass
