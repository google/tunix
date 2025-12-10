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
from collections.abc import Callable
import copy
import importlib
import inspect
import os
import pathlib
from pathlib import Path  # pylint: disable=g-importing-member
import shutil
import stat
from typing import Any, Dict, Sequence
from absl import logging
import dotenv
import jax
import numpy as np
import omegaconf
import optax
import orbax.checkpoint as ocp
from tunix.sft import metrics_logger
from tunix.sft import profiler

# Define a prefix for environment variables that can override YAML keys
_TUNIX_PREFIX = "T_"
_SUPPORTED_MODEL_SOURCES = (
    "kaggle",
    "huggingface",
    "gcs",
    "internal",
    "",
)


def yaml_key_to_env_key(s: str) -> str:
  return _TUNIX_PREFIX + s.upper()


def string_to_bool(s: str) -> bool:
  if s.lower() == "true":
    return True
  if s.lower() == "false":
    return False
  raise ValueError(f"Can't convert {s} to bool")

_yaml_types_to_parser = {
    str: str,
    int: int,
    float: float,
    bool: string_to_bool,
    omegaconf.dictconfig.DictConfig: dict,
    omegaconf.listconfig.ListConfig: list,
}


def get_project_root() -> Path:
  """Returns the project root folder.

  It searches up from the current file until it finds a marker file like '.git',
  'pyproject.toml', or 'setup.py'.
  """
  current_path = Path(__file__).resolve().parent
  # List of files that define the root of your project
  root_markers = [
      "LICENSE",
  ]

  # Iterate up through parent directories
  for parent in [current_path] + list(current_path.parents):
    # Check if any marker exists in this parent directory
    if any((parent / marker).exists() for marker in root_markers):
      return parent

  # Fallback: if no marker is found, return the current working directory
  return Path.cwd()


class HyperParameters:
  """Loads, merges, overrides, validates, and prepares the configuration for pipeline execution."""

  def __init__(self, argv: list[str], **kwargs):
    # Use omegaconf.OmegaConf.from_cli to capture CLI arguments.

    dotenv.load_dotenv()
    raw_keys = collections.OrderedDict()
    config_name = argv[1]
    raw_data_from_yaml = self._load_config_from_yaml(config_name)
    self._validate_env_variable(raw_data_from_yaml)
    self.replace_keys = {
        "lora_config",
        "training_config",
        "optimizer_config",
        "profiler_options",
        "rl_training_config",
    }
    keys_from_env_and_command_line = self._update_from_env_and_command_line(
        raw_keys, raw_data_from_yaml, argv, **kwargs
    )
    logging.info(
        "Updating keys from env and command line: %s",
        keys_from_env_and_command_line,
    )
    self.config = raw_keys
    self._validate_tokenizer()
    self._validate_model_source(raw_keys)
    self.check_supported_workflow()

  def _validate_tokenizer(self):
    """Validate the tokenizer configuration.

    Currently only sentencepiece and huggingface are supported. `HF_TOKEN` must
    be set if huggingface tokenizer is used.
    """
    tokenizer_config = self.config["tokenizer_config"]
    tokenizer_type = tokenizer_config["tokenizer_type"]
    tokenizer_path = tokenizer_config["tokenizer_path"]
    valid_tokenizer_type = {"sentencepiece", "huggingface"}
    if tokenizer_type not in valid_tokenizer_type:
      raise ValueError(
          f"tokenizer_type {tokenizer_type} is not supported, currently only"
          f" {valid_tokenizer_type} is supported"
      )
    if tokenizer_type == "huggingface":
      if "HF_TOKEN" not in os.environ:
        raise ValueError("Missing `HF_TOKEN` to access hf tokenizer")
      if not tokenizer_path:
        raise ValueError("tokenizer_path must be specified.")

  def clear_directory_contents(self):
    """Removes all files, directories, and links within the specified directory."""
    model_download_path = self.config.get("model_download_path")
    if not model_download_path:
      model_download_path = self.config["model_config"].get(
          "model_download_path"
      )
    if not os.path.isdir(model_download_path):
      logging.error(f"Error: '{model_download_path}' is not a valid directory.")
      return

    logging.info(f"Clearing contents of '{model_download_path}'...")
    for item in os.listdir(model_download_path):
      item_path = os.path.join(model_download_path, item)
      try:
        if os.path.isfile(item_path) or os.path.islink(item_path):
          # Attempt to make the file writable before removing,
          # in case of permission issues.
          try:
            os.chmod(item_path, stat.S_IWUSR | stat.S_IRUSR | stat.S_IXUSR)
          except OSError:
            pass  # Continue and let os.remove() raise the error if it fails
          os.remove(item_path)
          logging.info(f"  Removed file/link: {item_path}")
        elif os.path.isdir(item_path):
          # shutil.rmtree can also handle permission issues internally
          # by providing an onerror handler if needed.
          shutil.rmtree(item_path)
          logging.info(f"  Removed directory: {item_path}")
      except (OSError, shutil.Error) as e:
        logging.info(f"  Failed to delete {item_path}. Reason: {e}")
    logging.info(f"Finished clearing '{model_download_path}'.")

  def _validate_model_source(self, raw_keys: collections.OrderedDict[str, Any]):
    """Validate the checkpoint source and intermediate checkpoint."""
    model_config = raw_keys["model_config"]
    model_source = model_config.get("model_source")
    intermediate_ckpt = model_config.get("intermediate_ckpt_dir")

    if model_source not in _SUPPORTED_MODEL_SOURCES:
      raise ValueError(
          f"Invalid model_source: {model_source!r}. Must be one of"
          f" {_SUPPORTED_MODEL_SOURCES}."
      )

    if model_source in ["kaggle", "huggingface"] and not intermediate_ckpt:
      raise ValueError(
          "intermediate_ckpt must be specified when model_source is 'kaggle' or"
          " 'huggingface'"
      )

  def check_supported_workflow(self) -> None:
    """Checks if the model_source is supported for the given model_name.

    Raises:
      ValueError: If the model_source is not supported for the model_name.
    """
    model_config = self.config["model_config"]
    model_name = model_config["model_name"]
    model_source = model_config["model_source"]
    supported_sources = collections.defaultdict(
        lambda: ["huggingface", "internal"]
    )
    # TODO(b/467448875): Add support for other sources, such as kaggle for other
    # models.
    supported_sources["gemma"] = ["kaggle", "internal"]
    supported_sources["gemma2"] = ["kaggle", "internal"]
    supported_sources["gemma3"] = ["gcs", "internal"]

    if model_name.startswith("gemma3"):
      expected_sources = supported_sources["gemma3"]
      if model_source not in expected_sources:
        raise ValueError(
            f"Model '{model_name}' must use source(s) {expected_sources}, but"
            f" got '{model_source}'."
        )
    elif model_name.startswith("gemma2"):
      expected_sources = supported_sources["gemma2"]
      if model_source not in expected_sources:
        raise ValueError(
            f"Model '{model_name}' must use source(s) {expected_sources}, but"
            f" got '{model_source}'."
        )
    elif model_name.startswith("gemma"):
      expected_sources = supported_sources["gemma"]
      if model_source not in expected_sources:
        raise ValueError(
            f"Model '{model_name}' must use source(s) {expected_sources}, but"
            f" got '{model_source}'."
        )
    else:
      # Default case for other models
      expected_sources = supported_sources["other"]
      if model_source not in expected_sources:
        raise ValueError(
            f"Model '{model_name}' must use source(s) {expected_sources}, but"
            f" got '{model_source}'."
        )

  def _get_nested_config(self, keys: Sequence[str]) -> Any:
    """Helper to retrieve a value from a nested dictionary."""
    current_level = self.config
    for i, key in enumerate(keys):
      if not isinstance(current_level, omegaconf.dictconfig.DictConfig | dict):
        raise TypeError(
            f"Attempted to access key '{key}' on a non-dictionary element "
            f"at path: {' -> '.join(keys[:i])}"
        )
      try:
        current_level = current_level[key]
      except KeyError as exc:
        raise KeyError(
            f"Key '{key}' not found in config path: {' -> '.join(keys[:i+1])}"
        ) from exc
    return current_level

  def _extract_kwargs(
      self,
      func: Callable[..., Any],
      config: Dict[str, Any],
      config_path_info: str,
      learning_rate: Any | None = None,
  ) -> Dict[str, Any]:
    """Extracts and validates kwargs for a function from a config dictionary."""
    sig = inspect.signature(func)
    kwargs = {}
    for param in sig.parameters.values():
      param_name = param.name
      if param_name in config:
        kwargs[param_name] = config[param_name]
      elif learning_rate is not None and param_name == "learning_rate":
        kwargs[param_name] = learning_rate
      elif param.default is param.empty:
        # Safely get a name or representation for the callable
        func_name = getattr(func, "__name__", repr(func))
        raise ValueError(
            f"Missing required argument '{param_name}' for {func_name} "
            f"in config at {config_path_info}."
        )
    return kwargs

  def _get_schedule_fn(self, schedule_type: str, config_path_info: str):
    """Dynamically imports a schedule function from optax.schedules.

    Args:
      schedule_type: The type of schedule (e.g., "constant_schedule",
        "warmup_cosine_decay_schedule").
      config_path_info: The path to the config file, used for error reporting.

    Returns:
      The corresponding function from optax.schedules.

    Raises:
      AttributeError: If the schedule type does not correspond to a valid
                      function in optax.schedules.
    """
    try:
      schedule_fn = getattr(optax.schedules, schedule_type)
      return schedule_fn
    except AttributeError as exc:
      raise AttributeError(
          f"Config {config_path_info}: '{schedule_type}' is not a valid"
          " function in optax.schedules."
      ) from exc

  def _create_learning_rate(
      self, optimizer_config: Dict[str, Any], config_path_info: str
  ) -> Any:
    """Creates a learning rate schedule based on the optimizer config."""
    schedule_type = optimizer_config.get("schedule_type")
    if schedule_type:
      schedule_func = self._get_schedule_fn(schedule_type, config_path_info)
      schedule_kwargs = self._extract_kwargs(
          schedule_func, optimizer_config, config_path_info
      )
      logging.info(
          "Creating learning rate with schedule_type: %s, and following"
          " kwargs: %s",
          schedule_type,
          schedule_kwargs,
      )
      return schedule_func(**schedule_kwargs)

    # Default: No schedule, learning_rate should be a scalar
    learning_rate = optimizer_config.get("learning_rate")
    if learning_rate is not None and not isinstance(
        learning_rate, (float, int)
    ):
      raise TypeError(
          "learning_rate must be a scalar when no schedule_type is specified, "
          f"got {type(learning_rate)} in config at {config_path_info}."
      )
    logging.info("Creating learning rate with learning_rate: %d", learning_rate)
    return learning_rate

  def create_optimizer(
      self, *optimizer_keys: str
  ) -> optax.GradientTransformation:
    """Creates the optimizer based on a config path.

    Args:
        *optimizer_keys: One or more strings representing the keys to navigate
          the nested self.config dictionary. For example, ('rl_training_config',
          'actor_optimizer_config') would access
          self.config['rl_training_config']['actor_optimizer_config'].

    Returns:
        An optimizer instance.

    Raises:
        ValueError: If no optimizer_keys are provided.
        KeyError: If a key in the path is not found.
        TypeError: If an intermediate element in the path is not a dictionary.
    """

    if not optimizer_keys:
      raise ValueError("At least one optimizer key must be provided.")
    config_path_info = " -> ".join(optimizer_keys)

    try:
      optimizer_config = self._get_nested_config(optimizer_keys)
    except KeyError as e:
      raise KeyError(f"Could not resolve optimizer config path: {e}") from e

    if not isinstance(optimizer_config, omegaconf.dictconfig.DictConfig | dict):
      raise ValueError("optimizer_config must be a dictionary")

    opt_type = optimizer_config.get("opt_type")
    if not opt_type:
      raise ValueError("Optimizer name is required")

    try:
      opt_func = getattr(optax, opt_type.lower())
    except ValueError as e:
      raise ValueError(
          f"Optimizer type '{opt_type}' not supported from {config_path_info}."
          " Available options, see"
          " https://optax.readthedocs.io/en/latest/api/optimizers.html#optimizers"
      ) from e

    # Handle learning rate, potentially creating a schedule
    learning_rate_val = self._create_learning_rate(
        optimizer_config, config_path_info
    )
    if learning_rate_val is None and (
        "learning_rate" in inspect.signature(opt_func).parameters
        and inspect.signature(opt_func).parameters["learning_rate"].default
        is inspect.Parameter.empty
    ):
      # learning_rate is required by opt_func but not provided and no schedule
      raise ValueError(
          "Missing required argument 'learning_rate' for optimizer"
          f" '{opt_type}' and no schedule defined in config at"
          f" {config_path_info}."
      )

    opt_kwargs = self._extract_kwargs(
        opt_func, optimizer_config, config_path_info, learning_rate_val
    )
    # Call the optimizer function with the extracted kwargs
    try:
      return opt_func(**opt_kwargs)
    except TypeError as e:
      raise TypeError(
          f"Error calling {opt_type} with arguments {opt_kwargs}. "
          f"Check if the arguments match the signature of optax.{opt_type}: {e}"
      ) from e

  def create_mesh(self, model_key: str):
    """Validate and extract mesh configuration from a dictionary.

    Expects raw_keys to contain a 'mesh' key, which is a dictionary with 'shape'
    and 'axis_names' keys.

    Args:
      model_key: A model key that contain raw mesh configuration. For example,
        in rl, there are actor_model, critic_model and reference_model, each of
        them could have different mesh configuration.

    Returns:
      A tuple containing (axis_shapes, axis_names), both as tuples.

    Raises:
      ValueError: If the mesh configuration is missing, malformed, or invalid.
    """

    mesh_config = self.config[model_key].get("mesh")
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
    return jax.make_mesh(
        tuple(axis_shapes),
        tuple(axis_names),
        axis_types=(jax.sharding.AxisType.Auto,) * len(tuple(axis_names)),
    )

  def obtain_training_config_dict(self, key):
    """Obtain training config dictionary from specified key in self.config.

    Check and construct each component in training config and return them in a
    dictionary, which is ready to be used to create the training config object.

    Args:
      key: The key of the training config in the self.config dictionary.

    Returns:
      A dictionary constructed training config.
    """
    training_config = self.config[key]
    if not isinstance(training_config, collections.abc.MutableMapping):
      raise ValueError(
          "Expected 'training_config' to be a dictionary, but got "
          f"{type(training_config).__name__}"
      )

    constructed_training_config = collections.defaultdict()
    for key, value in training_config.items():
      if key == "checkpointing_options" and value:
        try:
          constructed_training_config[key] = ocp.CheckpointManagerOptions(
              **value
          )
        except ValueError as e:
          raise ValueError(f"Invalid checkpointing options: {value}") from e
      elif key == "metrics_logging_options" and value:
        try:
          constructed_training_config[key] = (
              metrics_logger.MetricsLoggerOptions(**value)
          )
        except ValueError as e:
          raise ValueError(f"Invalid metrics logging options: {value}") from e
      elif key == "profiler_options":
        if value:
          try:
            constructed_training_config[key] = profiler.ProfilerOptions(**value)
          except ValueError as e:
            raise ValueError(f"Invalid profiler options: {value}") from e
        else:
          constructed_training_config[key] = None
      elif "optimizer_config" in key:
        continue
      else:
        constructed_training_config[key] = value

    return constructed_training_config

  def _update_from_env_and_command_line(
      self,
      raw_keys: collections.OrderedDict[str, Any],
      raw_data_from_yaml: dict[str, Any],
      argv: list[str],
      **kwargs,
  ):
    """Update the configuration from command line."""

    cli_cfg = omegaconf.OmegaConf.from_cli(argv[2:])

    raw_data_from_cmd_line = omegaconf.OmegaConf.to_container(
        cli_cfg, resolve=True
    )

    updated_keys = []

    # Check for conflicts and unknown keys.
    for k in raw_data_from_cmd_line:
      if not k:
        continue
      if k not in raw_data_from_yaml:
        raise ValueError(
            f"Key {k} was passed at the command line but isn't in config."
        )

    # Iterate over key from base yaml
    for k in raw_data_from_yaml:

      # Error out if same key defined in cmd line and environment
      if k in raw_data_from_cmd_line and yaml_key_to_env_key(k) in os.environ:
        raise ValueError(
            f"You are passing overrides by both CLI and ENV for `{k}`. This"
            " isn't allowed."
        )

      # Take value from base config yaml if key is not specified in command line
      # or environment.
      if (
          k not in raw_data_from_cmd_line
          and yaml_key_to_env_key(k) not in os.environ
      ):
        # take the config value from the YAML file.
        raw_keys[k] = raw_data_from_yaml[k]
        continue

      #  Key is specified on either command line or enviornment
      updated_keys.append(k)

      # take updated value from command line or enviornment
      if k in raw_data_from_cmd_line:
        new_proposal = raw_data_from_cmd_line[k]
      else:
        new_proposal = os.environ.get(yaml_key_to_env_key(k))

      # If specified value is not one of type in base config yaml or is not
      # consumed by to type parser, error out
      if (not isinstance(new_proposal, type(raw_data_from_yaml[k]))) and (
          type(raw_data_from_yaml[k]) not in _yaml_types_to_parser
      ):
        raise ValueError(
            f"For key '{k}', type {type(raw_data_from_yaml[k])} not in"
            f" {_yaml_types_to_parser.keys()}, can't pass at the CLI or ENV"
        )

      # Take the config value
      if new_proposal is None:
        # This allows users to set empty strings via CLI, otherwise parsed as
        # "None"
        raw_keys[k] = None
      elif isinstance(new_proposal, type(raw_data_from_yaml[k])):
        raw_keys[k] = new_proposal  # take the raw data, no type conversion
      else:
        parsed_new_proposal = _yaml_types_to_parser[
            type(raw_data_from_yaml[k])
        ](
            new_proposal
        )  # take the command line value, but type it like the config value.

        if isinstance(parsed_new_proposal, dict):
          if k in self.replace_keys:
            raw_keys[k] = parsed_new_proposal
          else:
            # merge the dict recursively
            raw_keys[k] = self.update_dict(
                schema=raw_data_from_yaml[k], source=parsed_new_proposal
            )
        else:
          raw_keys[k] = parsed_new_proposal

    return updated_keys

  def update_dict(self, schema: dict[str, Any], source: dict[str, Any]):
    """Recursively updates a dictionary with values from another dictionary.

    Uses the `self.replace_keys` set to determine which keys from the source
    should completely overwrite existing values in the schema.

    Args:
        schema (dict): The base dictionary to be updated.
        source (dict): The dictionary containing updates.

    Returns:
        dict: A new dictionary with updates applied.
    """
    output = copy.deepcopy(schema)
    for key, source_val in source.items():

      if key in self.replace_keys:
        # For keys in self.replace_keys, take the value from source entirely.
        output[key] = copy.deepcopy(source_val)
      else:
        output_val = output.get(key)
        # Check if both source and output values are dictionaries for merging.
        if isinstance(
            source_val,
            collections.abc.Mapping | omegaconf.dictconfig.DictConfig,
        ) and isinstance(
            output_val,
            collections.abc.Mapping | omegaconf.dictconfig.DictConfig,
        ):
          # Both are dictionaries, so recurse.
          # The recursive call uses the same self.replace_keys instance.
          output[key] = self.update_dict(output_val, source_val)
        else:
          # Otherwise (not both dictionaries), the source value overwrites.
          output[key] = copy.deepcopy(source_val)

    # Identify keys that are in self.replace_keys and were in the
    # original schema
    # (hence in output now) but are NOT in the source dictionary.
    keys_to_remove = []
    for key in self.replace_keys:
      if key in output and key not in source:
        keys_to_remove.append(key)

    # Remove these keys from the output dictionary.
    for key in keys_to_remove:
      del output[key]

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

  def obtain_reward_fn(self) -> list[Callable[..., Any]]:
    """Obtain reward function from the config."""
    project_root = get_project_root()
    reward_fns = []
    for reward_fn_path in self.config["reward_functions"]:
      full_path = str(project_root / reward_fn_path)
      module_name = os.path.splitext(os.path.basename(full_path))[0]
      # load from source
      loader = importlib.machinery.SourceFileLoader(module_name, full_path)
      spec = importlib.util.spec_from_loader(module_name, loader)

      if spec is None:
        raise ImportError(f"Cannot find spec for module at {full_path}")
      if spec.loader is None:
        raise ImportError(f"Spec for module {module_name} has no loader")

      module = importlib.util.module_from_spec(spec)

      try:
        spec.loader.exec_module(module)
      except Exception as e:
        raise ImportError(
            f"Failed to execute module {module_name} from {full_path}: {e}"
        )

      if self.config["verl_compatible"]:

        def reward_fn(prompts, completions, reward_model, **kwargs):
          del prompts, kwargs
          ground_truths = reward_model["ground_truth"]
          return [
              module.compute_score(c, gt)
              for c, gt in zip(completions, ground_truths)
          ]

        reward_fns.append(reward_fn)

      else:
        # Get all defined functions in the file as reward functions
        defined_functions = []
        for _, member in inspect.getmembers(module):
          if inspect.isfunction(member):
            # Check if the function was defined in this module
            if member.__module__ == module_name:
              defined_functions.append(member)
        reward_fns.extend(defined_functions)
    return reward_fns

  def _get_function_from_path(self, path_str):
    """Dynamically imports a function from a string path.

    Args:
      path_str: The string path to the function, e.g.
        "tunix.rl.reward_fn.check_answer"

    Returns:
      The dynamically imported function, or None if the import fails.
    """
    try:
      # Split the path into module and function name
      module_path, function_name = path_str.rsplit(".", 1)

      # Import the module
      module = importlib.import_module(module_path)

      # Get the function from the module
      function = getattr(module, function_name)
      return function
    except (ImportError, AttributeError, ValueError) as e:
      logging.warning("Error importing '%s': %s", path_str, e)
      return None

def initialize(argv, **kwargs):
  return HyperParameters(argv, **kwargs)
