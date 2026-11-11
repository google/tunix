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
"""JAX and Optax optimizer and learning-rate schedule backend for Tunix CLI."""

from collections.abc import Callable, Mapping
from typing import Any

import jax.numpy as jnp
import optax
from tunix.cli import config


def get_schedule_fn(
    schedule_type: str, config_path_info: str = ""
) -> Callable[..., Any]:
  """Returns the named schedule factory from `optax.schedules`."""
  try:
    return getattr(optax.schedules, schedule_type)
  except AttributeError as exc:
    raise AttributeError(
        f"Config {config_path_info}: '{schedule_type}' is not a valid"
        " function in optax.schedules."
    ) from exc


def create_learning_rate(
    optimizer_config: Mapping[str, Any], config_path_info: str = ""
) -> Any:
  """Creates an Optax learning-rate schedule or scalar from `optimizer_config`."""
  schedule_type = optimizer_config.get("schedule_type")
  schedule_fn = None
  if schedule_type:
    schedule_fn = get_schedule_fn(schedule_type, config_path_info)
  return config.create_learning_rate(
      optimizer_config, config_path_info, schedule_fn=schedule_fn
  )


def create_optimizer(
    optimizer_config: Mapping[str, Any], config_path_info: str = ""
) -> optax.GradientTransformation:
  """Creates an Optax `GradientTransformation` from `optimizer_config`."""
  opt_type, chain_config = config.parse_optimizer_config(optimizer_config)
  try:
    opt_func = getattr(optax, opt_type.lower())
  except (AttributeError, ValueError) as exc:
    raise ValueError(
        f"Optimizer type '{opt_type}' not supported from {config_path_info}."
        " Available options, see"
        " https://optax.readthedocs.io/en/latest/api/optimizers.html#optimizers"
    ) from exc

  learning_rate_val = create_learning_rate(optimizer_config, config_path_info)
  opt_kwargs = config.extract_kwargs(
      opt_func, optimizer_config, config_path_info, learning_rate_val
  )
  opt_func_to_call = opt_func
  if learning_rate_val is not None:
    opt_func_to_call = optax.inject_hyperparams(
        opt_func,
        hyperparam_dtype=jnp.float32,  # pyrefly: ignore[bad-argument-type]
    )
  try:
    optimizer = opt_func_to_call(**opt_kwargs)
  except TypeError as exc:
    raise TypeError(
        f"Error calling {opt_type} with arguments {opt_kwargs}. "
        f"Check if the arguments match the signature of optax.{opt_type}: {exc}"
    ) from exc

  if chain_config is None:
    return optimizer
  chained = create_optimizer(chain_config, config_path_info)
  return optax.chain(chained, optimizer)
