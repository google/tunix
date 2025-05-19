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

"""Utilities for wrapping nnx.Modules to capture their outputs."""

import logging

from flax import nnx

logger = logging.getLogger(__name__)


class SowedModule(nnx.Module):
  """Wraps an nnx.Module to capture its final output using `sow`.

  The captured output is stored in this wrapper module's state
  under the tag 'intermediate_output' using `nnx.Intermediates`.
  Compatible with `nnx.jit`.
  """

  _SOW_TAG = "intermediate_output"  # Tag to store the output under

  def __init__(self, model_to_wrap: nnx.Module):
    self.wrapped_model = model_to_wrap

  def __call__(self, *args, **kwargs):
    # Execute the wrapped model's forward pass
    output = self.wrapped_model(*args, **kwargs)
    # Sow the output into this wrapper's state
    logger.debug(
        "SowedModule sowing output for %s under tag '%s'",
        type(self.wrapped_model).__name__,
        self._SOW_TAG,
    )
    self.sow(nnx.Intermediate, self._SOW_TAG, output)
    return output


def pop_sowed_intermediate_outputs(
    model: nnx.Module,
) -> nnx.graph.GraphState:
  """Returns the intermediate output from a model."""
  return nnx.pop(model, nnx.Intermediate)


def wrap_model_with_sowed_modules(
    model: nnx.Module,
    modules_to_capture: list[type[nnx.Module]],
):
  """Wrap the layers of a model with a SowedModule to capture their outputs.

  Recursively finds instances of `modules_to_capture` within the `model`
  and replaces them with a `SowedModule` instance wrapping the
  original layer.
  Modifies the `model` in-place. But does not wrap the model itself.

  Args:
      model: The root nnx.Module to traverse and modify.
      modules_to_capture: List of nnx.Module subclass to find and wrap.

  Returns:
      The number of layers that were wrapped.
  """
  replacements = []

  for path, module in model.iter_modules():
    # Only wrap if it's the target module_to_wrap and not already wrapped
    if isinstance(module, tuple(modules_to_capture)) and not isinstance(
        module, SowedModule
    ):
      wrapped_instance = SowedModule(module)
      replacements.append((path, wrapped_instance))

  # Apply replacements
  for path, wrapped_instance in replacements:
    if not path:
      raise ValueError(
          "Attempted to wrap the root module"
          f" ({type(model).__name__}). This utility is designed for"
          " wrapping submodules in-place. The root module itself will not be"
          " replaced by this function's modification logic. If you need to"
          " wrap the root module, instantiate the wrapper directly around it."
      )

    # Traverse the path to find the direct parent of the module to be replaced
    current_parent_obj = model
    for part in path[:-1]:
      if isinstance(part, str):
        current_parent_obj = getattr(current_parent_obj, part)
      elif isinstance(part, int):
        current_parent_obj = current_parent_obj[part]
      else:
        raise TypeError(
            f"Unsupported path part type: {type(part)}. Path: {path}"
        )

    if isinstance(current_parent_obj, SowedModule):
      # Do not wrap if the parent is already a SowedModule
      continue

    last_key = path[-1]

    if isinstance(last_key, str):
      setattr(current_parent_obj, last_key, wrapped_instance)
    elif isinstance(last_key, int):
      # If the parent is a sequence, try to modify in-place
      current_parent_obj[last_key] = wrapped_instance
    else:
      raise TypeError(
          f"Unsupported key type for replacement: {type(last_key)}. Path:"
          f" {path}"
      )


def unwrap_sowed_modules(model: nnx.Module):
  """Unwraps all instances of `SowedModule` from a model.

  Recursively finds instances of `SowedModule` within the `model`
  and replaces them with the original wrapped layer instance.
  Modifies the `model` in-place.

  Args:
      model: The root nnx.Module to traverse and modify.
  """
  replacements = []

  # Collect all wrapper instances and their original wrapped modules
  for path, module in model.iter_modules():
    if isinstance(module, SowedModule):
      if not hasattr(module, "wrapped_model"):
        raise ValueError(
            f"Wrapper module at path {path} does not have a 'wrapped_model'"
            " attribute."
        )
      replacements.append((path, module.wrapped_model))

  # Apply replacements
  for path, original_module in replacements:
    if not path:
      raise ValueError(
          "Attempted to unwrap the root module"
          f" ({type(model).__name__}). This utility is designed for"
          " unwrapping submodules in-place. The root module itself will not be"
          " replaced by this function's modification logic. If you need to"
          " unwrap the root module, you should typically re-assign the result"
          " of this function: `model = unwrap_module_instances(model)`."
      )

    # Traverse the path to find the direct parent of the module to be unwrapped
    current_parent_obj = model
    for part in path[:-1]:
      if isinstance(part, str):
        current_parent_obj = getattr(current_parent_obj, part)
      elif isinstance(part, int):
        current_parent_obj = current_parent_obj[part]
      else:
        raise TypeError(
            f"Unsupported path part type: {type(part)}. Path: {path}"
        )

    last_key = path[-1]

    if isinstance(last_key, str):
      setattr(current_parent_obj, last_key, original_module)
    elif isinstance(last_key, int):
      current_parent_obj[last_key] = original_module
    else:
      raise TypeError(
          f"Unsupported key type for replacement: {type(last_key)}. Path:"
          f" {path}"
      )
