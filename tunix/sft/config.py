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

import collections
import os
import pathlib
from absl import logging
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


def validate_model_name(s: str) -> bool:
  """Validate provided model name."""
  # currently supported models
  valid_model_names = ("llama3.1-8b", "gemma-2b", "gemma-7bgemma2-2b")
  if s not in valid_model_names:
    raise ValueError(
        f"Invalid model name was passed. Got {s}, Valid options"
        f" {valid_model_names}"
    )


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


_yaml_types_to_parser = {str: str, int: int, float: float, bool: string_to_bool}


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
        "Updating keys from env and command line:"
        f" {keys_from_env_and_command_line}"
    )

    self.config = raw_keys
    self._validate_ckpt_source(raw_keys)
    self._create_optimizer(raw_keys)
    self._create_mesh(raw_keys)
    self._validate_training_config_and_assign(raw_keys)

  def _validate_ckpt_source(self, raw_keys):
    """Validate the checkpoint source and intermediate checkpoint."""
    ckpt_source = raw_keys.get("ckpt_source")
    intermediate_ckpt = raw_keys.get("intermediate_ckpt_dir")

    if ckpt_source not in ["kaggle", "huggingface", None]:
      raise ValueError(
          f"Invalid ckpt_source: {ckpt_source}. Must be 'kaggle',"
          " 'huggingface', or None."
      )

    if ckpt_source in ["kaggle", "huggingface"] and not intermediate_ckpt:
      raise ValueError(
          "intermediate_ckpt must be specified when ckpt_source is 'kaggle' or"
          " 'huggingface'."
      )

  def _create_optimizer(self, raw_keys):
    """Create the optimizer from the name and learning rate."""
    optimizer_name = raw_keys["optimizer"]
    learning_rate = raw_keys["learning_rate"]
    if optimizer_name not in _OPTIMIZER_MAP:
      raise ValueError(
          f"Optimizer {optimizer_name} not found in {_OPTIMIZER_MAP.keys()}"
      )
    # Schedule is not supported yet.
    if not isinstance(learning_rate, float):
      raise ValueError("Learning rate is not a scalar")
    self.optimizer = _OPTIMIZER_MAP[optimizer_name](learning_rate)

  def _create_mesh(self, raw_keys):
    """Validate and create the mesh configuration."""
    mesh = raw_keys.get("mesh")

    if len(mesh) != 2:
      raise ValueError(
          "The 'mesh' must be of length 2, containing axis shapes and"
          " axis names."
      )

    axis_shapes, axis_names = mesh
    if not all(isinstance(x, int) for x in axis_shapes):
      raise ValueError("All elements in axis_shapes must be integers.")
    if not all(isinstance(x, str) for x in axis_names):
      raise ValueError("All elements in axis_names must be strings.")

    if len(axis_shapes) != len(axis_names):
      raise ValueError("axis_shapes and axis_names must have the same length.")

    self.mesh = (tuple(axis_shapes), tuple(axis_names))

  def _validate_training_config_and_assign(self, raw_keys):
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
      self, raw_keys, raw_data_from_yaml, argv, **kwargs
  ):
    """Update the configuration from the environment and command line."""
    cli_cfg = omegaconf.OmegaConf.from_cli(argv[2:])
    # Also create a configuration from any extra keyword arguments.
    kwargs_cfg = omegaconf.OmegaConf.create(kwargs)
    # Merge command-line and keyword arguments.
    cmdline_cfg = omegaconf.OmegaConf.merge(cli_cfg, kwargs_cfg)
    raw_data_from_cmd_line = omegaconf.OmegaConf.to_container(
        cmdline_cfg, resolve=True
    )
    updated_keys = []

    for k in raw_data_from_cmd_line:
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
        # This allows users to set empty strings via CLI, otherwise parsed as "None"
        raw_keys[k] = None
      elif isinstance(new_proposal, type(raw_data_from_yaml[k])):
        raw_keys[k] = new_proposal  # take the raw data, no type conversion
      else:
        try:
          raw_keys[k] = _yaml_types_to_parser[type(raw_data_from_yaml[k])](
              new_proposal
          )  # take the command line value, but type it like the config value.
        except ValueError as e:
          raise ValueError(
              f"Couldn't parse value from CLI or ENV '{new_proposal}' for key"
              f" '{k}'"
          ) from e

    return updated_keys

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