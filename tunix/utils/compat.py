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

"""Compatibility utils."""

import functools
from typing import Any, Callable
from absl import logging
from flax import nnx
import jax


# Flax version compatibility.
ModuleList = list

# To accomodate github requirements. nnx.List is available in flax 0.12.0 and
# later.
if hasattr(nnx, "List"):
  ModuleList = nnx.List  # noqa: N816 (public alias)


# JAX version compatibility
if hasattr(jax.sharding, "use_mesh"):
  set_mesh = jax.sharding.use_mesh
else:
  set_mesh = jax.set_mesh


def alias_init_param(old_name: str, new_name: str):
  """Decorator to support deprecated keyword arguments on __init__."""

  def decorator(init_fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(init_fn)
    def wrapper(self, *args, **kwargs):
      if old_name in kwargs:
        if new_name in kwargs:
          raise ValueError(
              f"Cannot specify both '{new_name}' and '{old_name}'. Please use"
              f" '{new_name}' only."
          )
        logging.warning(
            "The '%s' keyword argument is deprecated; please use '%s' instead.",
            old_name,
            new_name,
        )
        kwargs[new_name] = kwargs.pop(old_name)
      return init_fn(self, *args, **kwargs)

    return wrapper

  return decorator
