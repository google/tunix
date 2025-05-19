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
from typing import Type

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
    modules_to_capture: list[Type[nnx.Module]],
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
  for path, child_module in model.iter_children():
    # Check recursively for any modules that needs to be wrapped.
    wrap_model_with_sowed_modules(child_module, modules_to_capture)
    # Check if the child is the target types and is not already wrapped.
    if any(
        [isinstance(child_module, module) for module in modules_to_capture]
    ) and not isinstance(model, SowedModule):
      logger.debug(
          "Wrapping %s instance found at %s.%s",
          child_module.__class__,
          model.__class__,
          path,
      )
      # Create the wrapper
      wrapper = SowedModule(child_module)
      # Replace the attribute on the parent model
      setattr(model, path, wrapper)


def unwrap_sowed_modules(model: nnx.Module):
  """Unwraps all instances of `SowedModule` from a model.

  Recursively finds instances of `SowedModule` within the `model`
  and replaces them with the original wrapped layer instance.
  Modifies the `model` in-place.

  Args:
      model: The root nnx.Module to traverse and modify.
  """
  for path, child_module in model.iter_children():
    # Check recursively for any modules that needs to be unwrapped.
    unwrap_sowed_modules(child_module)
    # Check if the child is a SowedModule.
    if isinstance(child_module, SowedModule):
      logger.debug(
          "Unwrapping %s instance found at %s.%s",
          child_module.wrapped_model.__class__,
          model.__class__,
          path,
      )
      # Replace the attribute on the parent model
      setattr(model, path, child_module.wrapped_model)
