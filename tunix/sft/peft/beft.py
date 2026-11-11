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

"""BEFT (Bias-Elevation Fine-Tuning) PEFT method for Tunix.

BEFT (ACL 2026, https://arxiv.org/abs/2509.15974) is a parameter-efficient
fine-tuning technique that trains only bias terms (specifically on value
projections or target linear layers) while keeping all weight matrices frozen.
If the underlying architecture is bias-free, BEFT injects and trains a 1D
bias vector, achieving extreme parameter efficiency (<0.01% trainable parameters).
"""

import dataclasses
import re
from typing import Any, Sequence

from absl import logging
from flax import nnx
import jax
import jax.numpy as jnp


class BEFTParam(nnx.Param):
  """Flax NNX variable type for BEFT trainable parameters."""


class BEFTLinear(nnx.Module):
  """BEFT wrapper around a linear layer or projection module.

  Elevates/adds a trainable bias vector (`BEFTParam`) to the wrapped base layer:
    y = base_layer(x) + bias
  """

  def __init__(
      self,
      base_layer: nnx.Module,
      bias_shape: tuple[int, ...] | None = None,
      bias_init: nnx.Initializer = nnx.initializers.zeros_init(),
      dtype: jnp.dtype = jnp.float32,
      rngs: nnx.Rngs | None = None,
  ):
    """Initializes the BEFTLinear wrapper.

    Args:
      base_layer: The target linear/projection module to wrap.
      bias_shape: The shape of the elevated bias vector. If None, inferred
        from `base_layer.out_features` or `base_layer.kernel.shape[-1]`.
      bias_init: Initializer for the bias parameter. Defaults to zeros.
      dtype: Data type of the elevated bias vector. Defaults to float32.
      rngs: Optional RNGs for parameter initialization.
    """
    self.base_layer = base_layer
    if bias_shape is None:
      if hasattr(base_layer, "out_features"):
        bias_shape = (base_layer.out_features,)
      elif hasattr(base_layer, "kernel") and hasattr(base_layer.kernel, "value"):
        bias_shape = (base_layer.kernel.value.shape[-1],)
      elif hasattr(base_layer, "w") and hasattr(base_layer.w, "value"):
        bias_shape = (base_layer.w.value.shape[-1],)
      else:
        raise ValueError(
            f"Could not automatically infer bias shape for layer {type(base_layer).__name__}. "
            "Please provide explicit `bias_shape`."
        )

    if rngs is None:
      bias_value = bias_init(jax.random.key(0), bias_shape, dtype)
    else:
      bias_value = bias_init(rngs.params(), bias_shape, dtype)

    self.bias = BEFTParam(bias_value)

  def __call__(self, *args, **kwargs) -> Any:
    out = self.base_layer(*args, **kwargs)
    return out + self.bias.value

  def __getattr__(self, name: str) -> Any:
    # Delegate attribute lookups to the inner base layer when not found on wrapper
    base_layer = self.__dict__.get("base_layer")
    if base_layer is not None and hasattr(base_layer, name):
      return getattr(base_layer, name)
    raise AttributeError(
        f"'{type(self).__name__}' object has no attribute '{name}'"
    )

  def merge(self) -> None:
    """Merges elevated BEFT bias into base_layer if base_layer has a bias."""
    if hasattr(self.base_layer, "bias") and self.base_layer.bias is not None:
      self.base_layer.bias.value = (
          self.base_layer.bias.value + self.bias.value.astype(self.base_layer.bias.value.dtype)
      )


@dataclasses.dataclass
class BEFTConfig:
  """Configuration for applying BEFT to a model."""

  module_path: str = r".*v_proj.*|.*qkv_einsum.*"
  bias_init: nnx.Initializer = nnx.initializers.zeros_init()
  dtype: jnp.dtype = jnp.float32


def apply_beft_to_model(
    model: nnx.Module,
    config: BEFTConfig | None = None,
    *,
    module_path: str | None = None,
    bias_init: nnx.Initializer | None = None,
    dtype: jnp.dtype | None = None,
    rngs: nnx.Rngs | None = None,
) -> nnx.Module:
  """Applies BEFT to matching layers of the given model in-place.

  Args:
    model: The root nnx.Module to traverse and wrap.
    config: Optional BEFTConfig instance.
    module_path: Regex pattern to match module paths. Overrides `config.module_path`.
    bias_init: Initializer for bias. Overrides `config.bias_init`.
    dtype: Data type for bias. Overrides `config.dtype`.
    rngs: Optional NNX RNGs for parameter creation.

  Returns:
    The modified model with BEFT applied.
  """
  cfg = config or BEFTConfig()
  target_path_regex = module_path if module_path is not None else cfg.module_path
  target_bias_init = bias_init if bias_init is not None else cfg.bias_init
  target_dtype = dtype if dtype is not None else cfg.dtype

  compiled_pattern = re.compile(target_path_regex)

  replacements: list[tuple[Sequence[str | int], BEFTLinear]] = []

  for path, module in model.iter_modules():
    path_str = ".".join(str(p) for p in path)
    if compiled_pattern.search(path_str) and not isinstance(module, BEFTLinear):
      # Wrap matching module
      beft_wrapper = BEFTLinear(
          base_layer=module,
          bias_init=target_bias_init,
          dtype=target_dtype,
          rngs=rngs,
      )
      replacements.append((path, beft_wrapper))

  logging.info(
      "BEFT: Wrapping %d module(s) matching pattern '%s'",
      len(replacements),
      target_path_regex,
  )

  # Apply module replacements in-place
  for path, wrapped_instance in replacements:
    if not path:
      raise ValueError("Attempted to wrap the root model with BEFT.")

    current_parent = model
    for part in path[:-1]:
      if isinstance(part, str):
        current_parent = getattr(current_parent, part)
      elif isinstance(part, int):
        current_parent = current_parent[part]
      else:
        raise TypeError(f"Unsupported path part type: {type(part)}")

    last_key = path[-1]
    if isinstance(last_key, str):
      setattr(current_parent, last_key, wrapped_instance)
    elif isinstance(last_key, int):
      current_parent[last_key] = wrapped_instance
    else:
      raise TypeError(f"Unsupported key type: {type(last_key)}")

  return model


def unwrap_beft_from_model(model: nnx.Module) -> nnx.Module:
  """Unwraps all `BEFTLinear` wrappers in the model in-place."""
  replacements: list[tuple[Sequence[str | int], nnx.Module]] = []

  for path, module in model.iter_modules():
    if isinstance(module, BEFTLinear):
      replacements.append((path, module.base_layer))

  for path, base_layer in replacements:
    if not path:
      continue
    current_parent = model
    for part in path[:-1]:
      if isinstance(part, str):
        current_parent = getattr(current_parent, part)
      elif isinstance(part, int):
        current_parent = current_parent[part]
      else:
        raise TypeError(f"Unsupported path part type: {type(part)}")

    last_key = path[-1]
    if isinstance(last_key, str):
      setattr(current_parent, last_key, base_layer)
    elif isinstance(last_key, int):
      current_parent[last_key] = base_layer
    else:
      raise TypeError(f"Unsupported key type: {type(last_key)}")

  return model
