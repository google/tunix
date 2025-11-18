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


import threading
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Optional

_POLICY_LOSS_FN_CATEGORY = "policy_loss_fn"
_ADVANTAGE_ESTIMATOR_CATEGORY = "advantage_estimator"


class FunctionRegistry:
  """A thread-safe registry for functions, organized by category."""

  DEFAULT_ALLOWED_CATEGORIES: FrozenSet[str] = frozenset({
      _POLICY_LOSS_FN_CATEGORY,
      _ADVANTAGE_ESTIMATOR_CATEGORY,
  })

  def __init__(self, allowed_categories: Optional[Iterable[str]] = None):
    """Initializes the registry.

    Args:
        allowed_categories: An iterable of strings representing the only
          category names permitted for registration. If None, defaults to
          DEFAULT_ALLOWED_CATEGORIES.
    """
    if not allowed_categories:
      self._allowed_categories: FrozenSet[str] = self.DEFAULT_ALLOWED_CATEGORIES

    else:
      self._allowed_categories: FrozenSet[str] = frozenset(allowed_categories)

    if not self._allowed_categories:
      raise ValueError(
          "FunctionRegistry initialized with no allowed categories."
      )

    self._registry: Dict[str, Dict[str, Callable[..., Any]]] = {
        cat: {} for cat in self._allowed_categories
    }
    self._lock = threading.Lock()

  def _validate_category(self, category: str) -> None:
    """Raises ValueError if the category is not allowed."""
    if category not in self._allowed_categories:
      raise ValueError(
          f"Invalid category: '{category}'. "
          f"Allowed categories are: {sorted(list(self._allowed_categories))}"
      )

  def register(
      self, category: str, name: str
  ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Returns a decorator to register a function under a category and name."""
    self._validate_category(category)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
      with self._lock:
        if category not in self._registry:
          self._registry[category] = {}

        if name in self._registry[category]:
          raise ValueError(
              f"Function '{name}' is already registered in category"
              f" '{category}'."
          )
        self._registry[category][name] = func
      return func

    return decorator

  def get(self, category: str, name: str) -> Callable[..., Any]:
    """Retrieves a registered function by category and name."""
    with self._lock:
      try:
        category_funcs = self._registry[category]
      except KeyError:
        raise LookupError(f"No such category: '{category}'") from None
      try:
        return category_funcs[name]
      except KeyError:
        raise LookupError(
            f"No function named '{name}' in category '{category}'"
        ) from None

  def list_categories(self) -> List[str]:
    """Lists all registered categories."""
    with self._lock:
      return list(self._registry.keys())

  def list_functions(self, category: str) -> List[str]:
    """Lists all function names within a given category."""
    with self._lock:
      try:
        return list(self._registry[category].keys())
      except KeyError:
        raise LookupError(f"No such category: '{category}'") from None


# module-level registry instance.
default_registry = FunctionRegistry()


def get_policy_loss_fn(name: str) -> Callable[..., Any]:
  """Returns the policy loss function by name."""
  return default_registry.get(_POLICY_LOSS_FN_CATEGORY, name)


def register_policy_loss_fn(
    name: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
  """Returns a decorator to register a policy loss function by name."""
  return default_registry.register(_POLICY_LOSS_FN_CATEGORY, name)


def get_advantage_estimator(name: str) -> Callable[..., Any]:
  """Returns the advantage estimator function by name."""
  return default_registry.get(_ADVANTAGE_ESTIMATOR_CATEGORY, name)


def register_advantage_estimator(
    name: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
  """Returns a decorator to register an advantage estimator function by name."""
  return default_registry.register(_ADVANTAGE_ESTIMATOR_CATEGORY, name)
