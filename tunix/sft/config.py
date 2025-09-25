# Copyright 2025 Google LLC
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
"""Config and CLI launched interface."""

import ast
import collections
import copy
import os
import pathlib
from typing import Any

from absl import logging
import jax
import numpy as np
import omegaconf
import optax
import orbax.checkpoint as ocp
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
from tunix.sft import profiler

# Define a prefix for environment variables that can override YAML keys
_TUNIX_PREFIX = "T_"


def yaml_key_to_env_key(s: str) -> str:
  return _TUNIX_PREFIX + s.upper()


def string_to_bool(s: str) -> bool:
  if s.lower() == "true":
    return True
  if s.lower() == "false":
    return False
  raise ValueError(f"Can't convert {s} to bool")


# Map optimizer names to their optax functions
_OPTIMIZER_MAP: dict[
    str, collections.abc.Callable[..., optax.GradientTransformation]
] = {
    "adagrad": optax.adagrad,
    "adam": optax.adam,
    "adamw": optax.adamw,
    "rmsprop": optax.rmsprop,
    "sgd": optax.sgd,
    # Add other optax optimizers here as needed
}


_yaml_types_to_parser = {
    str: str,
    int: int,
    float: float,
    bool: string_to_bool,
    omegaconf.dictconfig.DictConfig: dict,
    omegaconf.listconfig.ListConfig: list,
}


class HyperParameters:
  """This class is responsible for loading, merging, and overriding the configuration."""

  def __init__(self, argv: list[str], **kwargs):
    # Use omegaconf.OmegaConf.from_cli to capture CLI arguments.
    raw_keys = collections.OrderedDict()
    config_name = argv[1]
    raw_data_from_yaml = self._load_config_from_yaml(config_name)
    self._validate_env_variable(raw_data_from_yaml)
    keys_from_env_and_command_line = self._update_from_env_and_command_line(
        raw_keys, raw_data_from_yaml, argv, **kwargs
    )
    logging.info(
        "Updating keys from env and command line: %s",
        keys_from_env_and_command_line,
    )

    self.config = raw_keys
    self._validate_model_source(raw_keys)
    self._create_optimizer(raw_keys)
    self._create_mesh(raw_keys)
    self._validate_training_config_and_assign(raw_keys)

  def _validate_model_source(self, raw_keys: collections.OrderedDict[str, Any]):
    """Validate the checkpoint source and intermediate checkpoint."""
    model_source = raw_keys.get("model_source")
    intermediate_ckpt = raw_keys.get("intermediate_ckpt_dir")

    if model_source not in ["kaggle", "huggingface", "gcs", ""]:
      raise ValueError(
          f"Invalid model_source: {model_source}. Must be 'kaggle',"
          " 'huggingface', 'gcs' or ''."
      )

    if model_source in ["kaggle", "huggingface"] and not intermediate_ckpt:
      raise ValueError(
          "intermediate_ckpt must be specified when model_source is 'kaggle' or"
          " 'huggingface'"
      )

  def _create_optimizer(self, raw_keys: collections.OrderedDict[str, Any]):
    """Create the optimizer from the name and learning rate."""
    optimizer_config = raw_keys["optimizer_config"]
    if optimizer_config is None or not isinstance(
        optimizer_config, omegaconf.dictconfig.DictConfig
    ):
      raise ValueError("optimizer_config must be a dictionary")

    if not optimizer_config.get("learning_rate"):
      raise ValueError("Learning rate is required")
    if not optimizer_config.get("opt_type"):
      raise ValueError("Optimizer name is required")
    optimizer_name = optimizer_config["opt_type"]
    learning_rate = optimizer_config["learning_rate"]
    if optimizer_name not in _OPTIMIZER_MAP:
      raise ValueError(
          f"Optimizer {optimizer_name} not found in {_OPTIMIZER_MAP.keys()}"
      )
    # Schedule is not supported yet.
    if not isinstance(learning_rate, float):
      raise ValueError("Learning rate is not a scalar")
    self.optimizer = _OPTIMIZER_MAP[optimizer_name](learning_rate)

  def _create_mesh(self, raw_keys: collections.OrderedDict[str, Any]):
    """Validate and extract mesh configuration from a dictionary.

    Expects raw_keys to contain a 'mesh' key, which is a dictionary with 'shape'
    and 'axis_names' keys.

    Args:
      raw_keys: A dictionary containing the raw configuration.

    Returns:
      A tuple containing (axis_shapes, axis_names), both as tuples.

    Raises:
      ValueError: If the mesh configuration is missing, malformed, or invalid.
    """

    mesh_config = raw_keys.get("mesh")
    if not mesh_config:
      raise ValueError("Missing 'mesh' configuration in raw_keys.")

    if not isinstance(mesh_config, collections.abc.Mapping):
      raise ValueError(
          "The 'mesh' configuration must be a dictionary-like object, got"
          f" {type(mesh_config)}."
      )

    shape = mesh_config.get("shape")
    if not shape:
      raise ValueError("Missing 'shape' key in 'mesh' configuration.")
    names = mesh_config.get("axis_names")
    if not names:
      raise ValueError("Missing 'axis_names' key in 'mesh' configuration.")

    try:
      axis_shapes = ast.literal_eval(shape)
    except ValueError as e:
      raise ValueError(
          "Invalid 'shape' key in 'mesh' configuration:"
          f" {mesh_config.get('shape')}"
      ) from e
    try:
      axis_names = ast.literal_eval(names)
    except ValueError as e:
      raise ValueError(
          "Invalid 'axis_names' key in 'mesh' configuration:"
          f" {mesh_config.get('axis_names')}"
      ) from e

    # Validate axis_shapes
    if not isinstance(axis_shapes, tuple):
      raise ValueError(
          f"'mesh.shape' must be a list or tuple, got {type(axis_shapes)}."
      )
    if not all(isinstance(x, int) for x in axis_shapes):
      raise ValueError(
          f"All elements in mesh.shape must be integers, got {axis_shapes}."
      )

    # Validate axis_names
    if not isinstance(axis_names, tuple):
      raise ValueError(
          f"'mesh.axis_names' must be a tuple, got {type(axis_names)}."
      )
    if not all(isinstance(x, str) for x in axis_names):
      raise ValueError(
          f"All elements in mesh.axis_names must be strings, got {axis_names}."
      )

    # Validate lengths match
    if len(axis_shapes) != len(axis_names):
      raise ValueError(
          f"mesh.shape {axis_shapes} and mesh.axis_names {axis_names} "
          "must have the same length."
      )

    # Validate mesh shape <= device count
    num_devices = jax.device_count()
    if np.prod(axis_shapes) > num_devices:
      raise ValueError(
          f"Mesh shape {axis_shapes} requires {np.prod(axis_shapes)} devices, "
          f"but found {num_devices}."
      )
    self.mesh = (tuple(axis_shapes), tuple(axis_names))

  def _validate_training_config_and_assign(
      self, raw_keys: collections.OrderedDict[str, Any]
  ):
    """Validate the complex configuration. Raise ValueError if invalid."""
    training_config = raw_keys["training_config"]
    if not isinstance(training_config, collections.abc.MutableMapping):
      raise ValueError(
          "Expected 'training_config' to be a dictionary, but got "
          f"{type(training_config).__name__}"
      )

    constructed_training_config = collections.defaultdict()
    for key, value in training_config.items():
      if key == "checkpointing_options":
        try:
          constructed_training_config[key] = ocp.CheckpointManagerOptions(
              **value
          )
        except ValueError as e:
          raise ValueError(f"Invalid checkpointing options: {value}") from e
      elif key == "metrics_logging_options":
        try:
          constructed_training_config[key] = (
              metrics_logger.MetricsLoggerOptions(**value)
          )
        except ValueError as e:
          raise ValueError(f"Invalid metrics logging options: {value}") from e
      elif key == "profiler_options":
        try:
          constructed_training_config[key] = profiler.ProfilerOptions(**value)
        except ValueError as e:
          raise ValueError(f"Invalid profiler options: {value}") from e
      else:
        constructed_training_config[key] = value

    self.training_config = peft_trainer.TrainingConfig(
        **constructed_training_config
    )

  def _update_from_env_and_command_line(
      self,
      raw_keys: collections.OrderedDict[str, Any],
      raw_data_from_yaml: dict[str, Any],
      argv: list[str],
      **kwargs,
  ):
    """Update the configuration from the environment and command line."""
    # CLI and Kwargs Overrides
    cli_cfg = omegaconf.OmegaConf.from_cli(argv[2:])
    # Also create a configuration from any extra keyword arguments.
    kwargs_cfg = omegaconf.OmegaConf.create(kwargs)
    # Merge command-line and keyword arguments.
    cmdline_cfg = omegaconf.OmegaConf.merge(cli_cfg, kwargs_cfg)

    raw_data_from_cmd_line = omegaconf.OmegaConf.to_container(
        cmdline_cfg, resolve=True
    )
    updated_keys = []
    logging.info("raw_data_from_cmd_line: %s", raw_data_from_cmd_line)

    # Check for conflicts and unknown keys.
    for k in raw_data_from_cmd_line:
      if not k:
        continue
      if k not in raw_data_from_yaml:
        raise ValueError(
            f"Key {k} was passed at the command line but isn't in config."
        )

    for k in raw_data_from_yaml:
      if k in raw_data_from_cmd_line and yaml_key_to_env_key(k) in os.environ:
        raise ValueError(
            f"You are passing overrides by both CLI and ENV for `{k}`. This"
            " isn't allowed."
        )

      if (
          k not in raw_data_from_cmd_line
          and yaml_key_to_env_key(k) not in os.environ
      ):
        # take the config value from the YAML file.
        raw_keys[k] = raw_data_from_yaml[k]
        continue

      updated_keys.append(k)
      if k in raw_data_from_cmd_line:
        new_proposal = raw_data_from_cmd_line[k]
      else:
        new_proposal = os.environ.get(yaml_key_to_env_key(k))

      if (not isinstance(new_proposal, type(raw_data_from_yaml[k]))) and (
          type(raw_data_from_yaml[k]) not in _yaml_types_to_parser
      ):
        raise ValueError(
            f"For key '{k}', type {type(raw_data_from_yaml[k])} not in"
            f" {_yaml_types_to_parser.keys()}, can't pass at the CLI or ENV"
        )
      if new_proposal is None:
        # This allows users to set empty strings via CLI, otherwise parsed as
        # "None"
        raw_keys[k] = None
      elif isinstance(new_proposal, type(raw_data_from_yaml[k])):
        raw_keys[k] = new_proposal  # take the raw data, no type conversion
      else:
        try:
          parsed_new_proposal = _yaml_types_to_parser[
              type(raw_data_from_yaml[k])
          ](
              new_proposal
          )  # take the command line value, but type it like the config value.
          if isinstance(parsed_new_proposal, dict):
            # merge the dict recursively
            raw_keys[k] = self.update_dict(
                schema=raw_data_from_yaml[k], source=parsed_new_proposal
            )
          else:
            raw_keys[k] = parsed_new_proposal
        except ValueError as e:
          raise ValueError(
              f"Couldn't parse value from CLI or ENV '{new_proposal}' for key"
              f" '{k}'"
          ) from e
    return updated_keys

  def update_dict(self, schema: dict[str, Any], source: dict[str, Any]):
    """Recursively updates a dictionary with values from another dictionary.

    Args:
        schema (dict): The base dictionary to be updated.
        source (dict): The dictionary containing updates.

    Returns:
        dict: A new dictionary with updates applied.
    """
    output = copy.deepcopy(schema)
    for key, value in source.items():
      if isinstance(value, collections.abc.Mapping):
        output[key] = self.update_dict(output.get(key, {}), value)
      else:
        output[key] = value
    return output

  def _validate_env_variable(self, raw_data_from_yaml):
    """Validate the environment variables."""
    for environment_var in os.environ:
      if environment_var[: len(_TUNIX_PREFIX)] == _TUNIX_PREFIX:
        proposed_key = environment_var[len(_TUNIX_PREFIX) :].lower()
        if proposed_key not in raw_data_from_yaml:
          raise ValueError(
              f"We received env {environment_var} but it doesn't match a key,"
              " so it is assumed a mistake."
          )
        if not environment_var[len(_TUNIX_PREFIX) :].isupper():
          raise ValueError(
              f"We received env {environment_var} but it isn't all uppercase."
          )

  def _load_config_from_yaml(self, config_name: str):
    """Try Loading and validate the configuration from the YAML file."""

    path = pathlib.Path(__file__).parent / config_name
    try:
      config_oconf = omegaconf.OmegaConf.load(path)
    except FileNotFoundError as e:
      raise ValueError(f"Config {config_name} not found.") from e

    return config_oconf


def initialize(argv, **kwargs):
  return HyperParameters(argv, **kwargs)
