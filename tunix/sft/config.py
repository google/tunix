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
import pathlib
import sys
import omegaconf
import orbax.checkpoint as ocp
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
from tunix.sft import profiler


class HyperParameters:
  """This class is responsible for loading, merging, and overriding the configuration."""

  def __init__(self, argv: list[str], **kwargs):
    # Use omegaconf.OmegaConf.from_cli to capture CLI arguments.
    config_name: str = argv[1]
    raw_data_from_yaml = self._load_and_validate_config_from_yaml(config_name)
    self._config = collections.OrderedDict()
    self._update_from_cmd_line(argv, raw_data_from_yaml, **kwargs)
    self._validate_config()
    self._convert_to_training_config()

  def _validate_config(self):
    """Validate the complex configuration. Raise ValueError if invalid."""
    if self._config["checkpointing_options"]:
      try:
        self._config["checkpointing_options"] = ocp.CheckpointManagerOptions(
            **self._config["checkpointing_options"]
        )
      except ValueError as e:
        raise ValueError(
            "Invalid checkpointing options: "
            f"{self._config['checkpointing_options']}"
        ) from e
    if self._config["metrics_logging_options"]:
      try:
        self._config["metrics_logging_options"] = (
            metrics_logger.MetricsLoggerOptions(
                **self._config["metrics_logging_options"]
            )
        )
      except ValueError as e:
        raise ValueError(
            "Invalid metrics logging options: "
            f"{self._config['metrics_logging_options']}"
        ) from e
    if self._config["profiler_options"]:
      try:
        self._config["profiler_options"] = profiler.ProfilerOptions(
            **self._config["profiler_options"]
        )
      except ValueError as e:
        raise ValueError(
            f"Invalid profiler options: {self._config['profiler_options']}"
        ) from e

  def _convert_to_training_config(self):
    """Convert the configuration to a TrainingConfig."""
    self.training_config = peft_trainer.TrainingConfig(**self._config)

  def _update_from_cmd_line(self, argv, raw_data_from_yaml, **kwargs):
    """Update the configuration from the command line and keyword arguments if exist."""
    cli_cfg = omegaconf.OmegaConf.from_cli(argv[2:])
    # Also create a configuration from any extra keyword arguments.
    kwargs_cfg = omegaconf.OmegaConf.create(kwargs)
    # Merge command-line and keyword arguments.
    cmdline_cfg = omegaconf.OmegaConf.merge(cli_cfg, kwargs_cfg)
    raw_data_from_cmd_line = omegaconf.OmegaConf.to_container(
        cmdline_cfg, resolve=True
    )

    for key in raw_data_from_cmd_line:
      if key not in raw_data_from_yaml:
        raise ValueError(
            f"Key {key} was passed at the command line but isn't in config."
        )
    for key in raw_data_from_yaml:
      if key not in raw_data_from_cmd_line:
        self._config[key] = raw_data_from_yaml[key]
      else:
        self._config[key] = raw_data_from_cmd_line[key]

  def _load_and_validate_config_from_yaml(self, config_name: str):
    """Try Loading and validate the configuration from the YAML file."""
    path = pathlib.Path(__file__).parent / config_name
    try:
      training_config_oconf = omegaconf.OmegaConf.load(path)
    except FileNotFoundError as e:
      raise ValueError(f"Config {config_name} not found.") from e

    training_config = omegaconf.OmegaConf.structured(
        peft_trainer.TrainingConfig(**training_config_oconf)
    )
    print(
        "Structured, Type-Checked: \n"
        f" {omegaconf.OmegaConf.to_yaml(training_config)}"
    )
    return training_config


def initialize(argv, **kwargs):
  return HyperParameters(argv, **kwargs)


if __name__ == "__main__":
  hp = initialize(sys.argv)
