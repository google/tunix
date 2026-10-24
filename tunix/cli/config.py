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
from collections.abc import Mapping
from collections.abc import MutableMapping
import copy
import importlib
import inspect
import os
import pathlib
import shutil
import stat
from typing import Any, Dict, Iterator, Sequence

from absl import logging
import dotenv
import jax
import omegaconf
import optax
from tunix.perf import metrics as perf_metrics
from tunix.sft import checkpoint_options
from tunix.sft import metrics_logger
from tunix.sft import profiler
from tunix.utils import mesh as mesh_lib

# Define a prefix for environment variables that can override YAML keys
_TUNIX_PREFIX = "T_"
_SUPPORTED_MODEL_SOURCES = (
    "kaggle",
    "huggingface",
    "gcs",
    "internal",
    "maxtext",
    "",
)
_OPEN_DICT_KEYS: frozenset[str] = frozenset({
    "kwargs",
    "chain_kwargs",
    "lora_config",
    "data_config",
    "agent_kwargs",
    "env_kwargs",
    "kubernetes_config",
    "sglang_jax_config",
    "agentic_ppo_config",
    "perf_metrics_options",
})


# Mesh axis names understood by `HyperParameters.mesh_axes`.
_MESH_AXES: tuple[str, ...] = ("fsdp", "tp", "expert", "dp")
# `mesh` sub-keys that are positionally coupled and so are inherited together.
_MESH_PAIR_KEYS: tuple[str, ...] = ("shape", "axis_names")
_ROLE_MODEL_CONFIG_KEYS: tuple[str, ...] = (
    "actor_model_config",
    "critic_model_config",
    "reference_model_config",
    "rollout_model_config",
)


def to_str_dict(cfg: object) -> dict[str, object]:
  """Narrows a config value to `dict[str, object]`.

  Config sections are loosely typed (`OrderedDict[str, Any]` values), so this
  takes any value and fails loud unless it holds a mapping.

  Args:
    cfg: A `DictConfig` (converted with interpolations resolved, which yields
      plain nested containers) or a mapping (copied shallowly).

  Returns:
    A new dictionary with the top-level keys converted to `str`.

  Raises:
    TypeError: If `cfg` does not hold a mapping.
  """
  container = (
      omegaconf.OmegaConf.to_container(cfg, resolve=True)
      if isinstance(cfg, omegaconf.DictConfig)
      else cfg
  )
  if not isinstance(container, Mapping):
    raise TypeError(f"Expected mapping container, got {type(container)!r}.")
  return {str(k): v for k, v in container.items()}


# TODO(noghabi): Let callers supply the open-section allowlist (e.g. a class
# attribute) instead of the hard-coded _OPEN_DICT_KEYS.
def _validate_nested_keys(
    schema: Mapping[str, object],
    candidate: Mapping[str, object],
    prefix: str = "",
    open_dict_keys: frozenset[str] = _OPEN_DICT_KEYS,
) -> None:
  """Validates that every key in `candidate` exists in `schema`, at any depth.

  The schema is expected to be complete. A section is open (it accepts any
  nested keys) when its name is in `open_dict_keys` or its schema value is an
  empty mapping (`{}`). A null schema value also accepts any mapping, since it
  carries no structure to check against.

  Args:
    schema: The base config to validate against.
    candidate: The parsed overrides to validate.
    prefix: The dotted path of `schema` and `candidate`; empty at the root.
    open_dict_keys: Section names that accept any nested keys.

  Raises:
    ValueError: If `candidate` has a key that `schema` lacks, or nests keys
      under a key whose schema value is neither a mapping nor null. The message
      names the dotted path.
  """
  for key, value in candidate.items():
    dotpath = f"{prefix}.{key}" if prefix else str(key)
    if key not in schema:
      raise ValueError(
          f"Unknown configuration key '{dotpath}'. Allowed keys at"
          f" '{prefix or '<root>'}': {sorted(schema.keys())}"
      )
    if key in open_dict_keys or not isinstance(value, Mapping):
      continue
    schema_val = schema[key]
    if schema_val is None:
      continue
    if not isinstance(schema_val, Mapping):
      raise ValueError(
          f"'{dotpath}' is a {type(schema_val).__name__} in the base config;"
          f" nested keys {sorted(value)} are not allowed."
      )
    if not schema_val:
      continue
    _validate_nested_keys(
        to_str_dict(schema_val),
        to_str_dict(value),
        prefix=dotpath,
        open_dict_keys=open_dict_keys,
    )


def _override_paths(
    overrides: Sequence[str],
) -> tuple[frozenset[str], frozenset[str]]:
  """Returns the dotted paths that `overrides` assign and touch.

  Args:
    overrides: Flattened `key=value` overrides (override files, then CLI).

  Returns:
    A tuple `(assigned, touched)`. `assigned` holds every path that was set
    directly: each override's left-hand side (so `x={}` and flow-style `x={...}`
    count as assigning `x`) and every leaf of the parsed override tree, where
    an empty mapping is a leaf. `touched` holds `assigned` plus all prefixes of
    those paths.
  """
  assigned: set[str] = {arg.split("=", 1)[0] for arg in overrides}
  pending: list[tuple[str, Mapping[str, object]]] = [
      ("", to_str_dict(omegaconf.OmegaConf.from_cli(list(overrides))))
  ]
  while pending:
    prefix, tree = pending.pop()
    for key, value in tree.items():
      path = f"{prefix}.{key}" if prefix else key
      if isinstance(value, Mapping) and value:
        pending.append((path, to_str_dict(value)))
      else:
        assigned.add(path)
  touched: set[str] = set()
  for path in assigned:
    parts = path.split(".")
    touched.update(".".join(parts[:i]) for i in range(1, len(parts) + 1))
  return frozenset(assigned), frozenset(touched)


def _inherits(
    role: Mapping[str, object], base: Mapping[str, object], key: str
) -> bool:
  """Returns whether `role[key]` is absent or still equals `base[key]`."""
  if key not in role:
    return True
  return key in base and role[key] == base[key]


def normalize_argv(
    argv: Sequence[str],
    default_base_config: str | pathlib.Path,
) -> list[str]:
  """Normalizes `argv` into `[entry_point, base_yaml_path, *overrides]`.

  Args:
    argv: Command-line arguments. The first element is treated as the entry
      point when it neither ends in `.yaml`/`.yml` nor contains `=`; otherwise
      `"resolve_config"` is used. The next element is the base YAML when it does
      not contain `=`.
    default_base_config: Base YAML used when `argv` does not name one, or names
      a file with the same basename.

  Returns:
    The normalized argument list.
  """
  args = list(argv)
  entry_point = "resolve_config"
  if args and not args[0].endswith((".yaml", ".yml")) and "=" not in args[0]:
    entry_point = args[0] or "resolve_config"
    args = args[1:]

  default_path = pathlib.Path(default_base_config)
  if not args or "=" in args[0]:
    return [entry_point, str(default_path)] + args
  if args[0] == default_path.name:
    return [entry_point, str(default_path)] + args[1:]
  return [entry_point, args[0]] + args[1:]


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


def _normalize_cli_override(schema_value: Any, override_value: Any) -> Any:
  """Restores empty string overrides that OmegaConf parses as None.

  OmegaConf.from_cli interprets CLI values like key="" as None. For string
  fields we want to preserve the user's intent and treat that as an empty
  string, including for nested dictionary overrides.

  Args:
    schema_value: Pre-existing schema value for reference.
    override_value: Proposed override value.

  Returns:
    The normalized override value.
  """
  if override_value is None and isinstance(schema_value, str):
    return ""
  if isinstance(
      schema_value, (collections.abc.Mapping, omegaconf.DictConfig)
  ) and isinstance(
      override_value, (collections.abc.Mapping, omegaconf.DictConfig)
  ):
    normalized = {}
    for key, value in override_value.items():
      normalized[key] = _normalize_cli_override(schema_value.get(key), value)
    return normalized
  return override_value


def _can_override_nullable_schema(override_value: Any) -> bool:
  """Returns whether a null-default schema key can accept the override.

  Nullable schema fields do not provide enough type information to route
  through `_yaml_types_to_parser`. In that case, preserve the CLI-parsed value
  directly, or the raw string from the environment.

  Args:
    override_value: Proposed override value.

  Returns:
    True if the override value is compatible.
  """
  return isinstance(
      override_value,
      (
          str,
          int,
          float,
          bool,
          collections.abc.Mapping,
          list,
          omegaconf.dictconfig.DictConfig,
          omegaconf.listconfig.ListConfig,
      ),
  )


def get_project_root() -> pathlib.Path:
  """Returns the project root folder.

  It searches up from the current file until it finds a marker file like '.git',
  'pyproject.toml', or 'setup.py'.
  """
  current_path = pathlib.Path(__file__).resolve().parent
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
  return pathlib.Path.cwd()


# TODO(noghabi): Merge override files with OmegaConf.load/merge to keep value
# types (quoted strings currently re-parse as YAML); affects the flags-off path,
# so needs its own change.
def _dict_to_cli_args(
    d: collections.abc.Mapping[str, Any], parent_key: str = "", sep: str = "."
) -> Iterator[str]:
  """Converts a dictionary to an Iterator string of CLI arguments."""
  for k, v in d.items():
    new_key = f"{parent_key}{sep}{k}" if parent_key else k
    if isinstance(v, (collections.abc.Mapping, omegaconf.DictConfig)):
      if v:
        yield from _dict_to_cli_args(v, parent_key=new_key, sep=sep)
      else:
        yield f"{new_key}={{}}"
    else:
      if v is None:
        yield f"{new_key}=null"
      else:
        yield f"{new_key}={v}"


def _extract_kwargs(
    func: Callable[..., Any],
    config: Dict[str, Any],
    config_path_info: str,
    learning_rate: Any | None = None,
) -> Dict[str, Any]:
  """Extracts and validates kwargs for a function from a config dictionary.

  Args:
    func: The callable whose signature selects the keys to extract.
    config: The config block to pull the keyword values from.
    config_path_info: The path to the config file, used for error reporting.
    learning_rate: An already resolved learning rate. It takes precedence over
      `config["learning_rate"]`, because it is the value a `schedule_type` was
      resolved into; the raw scalar would otherwise silently shadow it.

  Returns:
    The keyword arguments to call `func` with.

  Raises:
    ValueError: If a parameter without a default is missing from the config.
  """
  sig = inspect.signature(func)
  kwargs = {}
  for param in sig.parameters.values():
    param_name = param.name
    if learning_rate is not None and param_name == "learning_rate":
      kwargs[param_name] = learning_rate
    elif param_name in config:
      kwargs[param_name] = config[param_name]
    elif param.default is param.empty:
      # Safely get a name or representation for the callable
      func_name = getattr(func, "__name__", repr(func))
      raise ValueError(
          f"Missing required argument '{param_name}' for {func_name} "
          f"in config at {config_path_info}."
      )
  return kwargs


def get_schedule_fn(
    schedule_type: str, config_path_info: str = ""
) -> Callable[..., Any]:
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


def create_learning_rate(
    optimizer_config: Dict[str, Any], config_path_info: str = ""
) -> Any:
  """Creates a learning rate schedule based on the optimizer config.

  Args:
    optimizer_config: Optimizer settings. When it holds a `schedule_type` key,
      the remaining keys are passed as kwargs to the matching
      `optax.schedules` function; otherwise `learning_rate` is used as a
      constant.
    config_path_info: The path to the config file, used for error reporting.

  Returns:
    An `optax` schedule callable, or the scalar learning rate when no schedule
    is configured.

  Raises:
    TypeError: If no schedule is configured and `learning_rate` is not a
      scalar.
  """
  schedule_type = optimizer_config.get("schedule_type")
  if schedule_type:
    schedule_func = get_schedule_fn(schedule_type, config_path_info)
    schedule_kwargs = _extract_kwargs(
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
  if learning_rate is not None and not isinstance(learning_rate, (float, int)):
    raise TypeError(
        "learning_rate must be a scalar when no schedule_type is specified, "
        f"got {type(learning_rate)} in config at {config_path_info}."
    )
  logging.info("Creating learning rate with learning_rate: %s", learning_rate)
  return learning_rate


_OPT_CHAIN_TYPE_KEY = "opt_chain_type"
_CHAIN_KWARGS_KEY = "chain_kwargs"


def create_optimizer(
    optimizer_config: Dict[str, Any], config_path_info: str = ""
) -> optax.GradientTransformation:
  """Creates an optimizer from a dictionary of optimizer settings.

  Args:
    optimizer_config: Optimizer settings. `opt_type` names a function in
      `optax`, and the remaining keys are forwarded to it when they match its
      signature; unrelated keys are ignored. Any `optax` factory can be named,
      including plain gradient transformations such as `clip_by_global_norm`.
      The learning rate is resolved by `create_learning_rate`. The optional
      `opt_chain_type` names a further `optax` factory (e.g.
      `clip_by_global_norm`) built from `chain_kwargs` (e.g.
      `{"max_norm": 0.1}`) and chained ahead of the optimizer; it defaults to
      `None`, which leaves the optimizer unchained.
    config_path_info: The path to the config file, used for error reporting.

  Returns:
    An optimizer instance, preceded by `opt_chain_type` when it is set.

  Raises:
    ValueError: If the config is not a dictionary, `opt_type` is missing or
      unsupported, or the optimizer requires a learning rate that the config
      does not provide.
    TypeError: If the extracted kwargs do not match the optimizer signature.
  """
  if not isinstance(optimizer_config, omegaconf.dictconfig.DictConfig | dict):
    raise ValueError("optimizer_config must be a dictionary")

  opt_type = optimizer_config.get("opt_type")
  if not opt_type:
    raise ValueError("Optimizer name is required")

  try:
    opt_func = getattr(optax, opt_type.lower())
  except (AttributeError, ValueError) as e:
    # `getattr` raises AttributeError for an unknown name, and `opt_type.lower`
    # raises AttributeError when `opt_type` is not a string.
    raise ValueError(
        f"Optimizer type '{opt_type}' not supported from {config_path_info}."
        " Available options, see"
        " https://optax.readthedocs.io/en/latest/api/optimizers.html#optimizers"
    ) from e

  # Handle learning rate, potentially creating a schedule
  learning_rate_val = create_learning_rate(optimizer_config, config_path_info)
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

  opt_kwargs = _extract_kwargs(
      opt_func, optimizer_config, config_path_info, learning_rate_val
  )
  # Wrap the optimizer function with inject_hyperparams so that
  # the learning rate can be tracked and logged during training.
  # Plain gradient transformations (such as chained clip_by_global_norm)
  # have no learning rate and must not be wrapped, preserving their argument
  # dtypes (e.g. max_norm) so they do not conflict with bfloat16 updates.
  if learning_rate_val is not None:
    opt_func_to_call = optax.inject_hyperparams(
        opt_func, hyperparam_dtype=jax.numpy.float32  # pyrefly: ignore[bad-argument-type]
    )
  else:
    opt_func_to_call = opt_func
  # Call the optimizer function with the extracted kwargs
  try:
    optimizer = opt_func_to_call(**opt_kwargs)
  except TypeError as e:
    raise TypeError(
        f"Error calling {opt_type} with arguments {opt_kwargs}. "
        f"Check if the arguments match the signature of optax.{opt_type}: {e}"
    ) from e

  # `opt_chain_type` is built by this same helper, since `opt_type` resolves any
  # `optax` factory, and is chained ahead of the optimizer. Gradient clipping
  # reaches the optimizer this way rather than as an optimizer keyword, which
  # `optax` optimizers do not declare.
  opt_chain_type = optimizer_config.get(_OPT_CHAIN_TYPE_KEY)
  if not opt_chain_type:
    return optimizer
  chain_kwargs = optimizer_config.get(_CHAIN_KWARGS_KEY) or {}
  chained = create_optimizer(
      {"opt_type": opt_chain_type, **chain_kwargs}, config_path_info
  )
  return optax.chain(chained, optimizer)


class HyperParameters:
  """Loads, merges, overrides, validates, and prepares the configuration for pipeline execution.

  Configurations are merged from multiple sources. The following order of
  precedence applies, with later sources overriding earlier ones:
  1. Base Config File: The first positional argument, path to a YAML config
  file.
  2. Config File Override: An optional `override_config_file=/path/to/file.yaml`
     argument. Values in this file override values in the base config file.
  3. CLI Arguments: `key=value` pairs provided as arguments override values
     from both the base config and the config file override.

  Environment variables prefixed with `T_` can also be used to set parameters,
  but it is an error to set a parameter via both an environment variable and
  a command-line argument or override file.

  Optional keyword-only flags (all off by default; existing CLI behavior is
  unchanged when they are off):
  * `default_base_config`: Base YAML used when `argv` does not name one.
  * `merge_replace_keys`: Deep-merge instead of replace. Allows multiple
    `override_config_file=` arguments, disables `replace_keys`, and merges
    `model_config` into role configs per nested sub-key while preserving
    explicit role overrides.
    - "Explicit" means any role path set by an override file or the CLI,
      including values that YAML anchors expand to in an override file (anchors
      in the base YAML do not count). A role key set as a whole (`role.k=...`,
      `role.k={}`, flow-style `role.k={...}`, or `k: {}` in an override file)
      is never inherited; a nested sub-key is explicit when it or any of its
      descendants was set.
    - `mesh.shape` and `mesh.axis_names` are inherited as a pair: a role that
      sets either one keeps both of its own. Other `mesh` sub-keys are merged
      one by one.
    - Overrides only add keys, so `x={}` cannot clear a mapping; use
      `x=null` instead.
  * `validate_nested`: Rejects override keys (at any depth) that are not in the
    base YAML, and nested keys under scalar base values. The base YAML must be
    a complete schema; sections listed in `_OPEN_DICT_KEYS`, empty (`{}`)
    sections, and null entries accept any nested keys.
  """

  # TODO(noghabi): Converge the default and merge-mode role-inheritance
  # algorithms.

  config: collections.OrderedDict[str, Any]
  replace_keys: set[str]

  # TODO(noghabi): Reject unknown **kwargs (currently forwarded and discarded);
  # agentic *_main entry points forward kwargs, so this needs its own change.
  def __init__(
      self,
      argv: Sequence[str],
      *,
      default_base_config: str | pathlib.Path | None = None,
      merge_replace_keys: bool = False,
      validate_nested: bool = False,
      **kwargs: object,
  ):
    # Use omegaconf.OmegaConf.from_cli to capture CLI arguments.

    dotenv.load_dotenv()
    raw_keys = collections.OrderedDict()

    if default_base_config is not None:
      argv = normalize_argv(argv, default_base_config)
    elif len(argv) < 2 or "=" in argv[1]:
      raise ValueError(
          "The first argument must be a path to a base config file."
      )

    # Handle relative paths used in example scripts as a special case.
    # TODO(noghabi): Remove this once the example scripts are updated.
    if argv[1] in ("base_config.yaml", "base_agentic_config.yaml"):
      base_config_file = pathlib.Path(__file__).parent / argv[1]
    else:
      base_config_file = argv[1]
    raw_data_from_yaml = self._load_config_from_yaml(str(base_config_file))
    self._validate_env_variable(raw_data_from_yaml)
    base_model_config = raw_data_from_yaml.get("model_config", {})

    config_file_overrides: list[str] = []
    cli_overrides: list[str] = []
    for arg in argv[2:]:
      if arg.startswith("override_config_file="):
        if config_file_overrides and not merge_replace_keys:
          raise ValueError("Only one override_config_file argument is allowed.")
        _, override_path = arg.split("=", 1)
        if not override_path:
          raise ValueError("Empty path in override_config_file= argument.")
        config_file_overrides.append(override_path)
      else:
        if validate_nested and "=" not in arg:
          raise ValueError(
              f"Invalid CLI override argument {arg!r}; expected"
              " 'section.key=value' or 'override_config_file=path.yaml'."
          )
        cli_overrides.append(arg)

    if merge_replace_keys:
      self.replace_keys = set()
    else:
      self.replace_keys = {
          "lora_config",
          "training_config",
          "optimizer_config",
          "profiler_options",
          "rl_training_config",
      }

    all_overrides: list[str] = []
    for config_file_override in config_file_overrides:
      next_conf = self._load_config_from_yaml(config_file_override)
      all_overrides.extend(_dict_to_cli_args(next_conf))

    # First we apply the file overrides, then the overrides from the command
    # line making the command line the highest priority.
    all_overrides.extend(cli_overrides)
    if validate_nested:
      _validate_nested_keys(
          to_str_dict(raw_data_from_yaml),
          to_str_dict(omegaconf.OmegaConf.from_cli(all_overrides)),
      )

    keys_from_env_and_command_line = self._update_from_env_and_command_line(
        raw_keys, raw_data_from_yaml, all_overrides, **kwargs
    )
    logging.info(
        "Updating keys from env and command line: %s",
        keys_from_env_and_command_line,
    )
    self.config = raw_keys
    if merge_replace_keys:
      self._merge_model_config_into_roles(
          to_str_dict(base_model_config), all_overrides
      )
    else:
      # Inherit missing keys from model_config to actor_model_config, etc.
      # Also update keys that were not explicitly overridden
      current_model_config = self.config.get("model_config", {})
      for config_key in [
          "actor_model_config",
          "critic_model_config",
          "reference_model_config",
          "rollout_model_config",
      ]:
        if config_key in self.config and self.config[config_key] is not None:
          for k, v in current_model_config.items():
            if k not in self.config[config_key] or self.config[config_key][
                k
            ] == base_model_config.get(k):
              self.config[config_key][k] = v
    self._validate_tokenizer()
    self._validate_model_source(raw_keys)
    self.check_supported_workflow()
    self._validate_perf_metrics(entry_point=argv[0])

  def _merge_model_config_into_roles(
      self,
      base_model_config: Mapping[str, object],
      all_overrides: Sequence[str],
  ) -> None:
    """Merges `model_config` into role configs (`merge_replace_keys=True`).

    Unlike the default inheritance, nested mappings (e.g. `mesh`,
    `lora_config`, `kwargs`) are merged per sub-key, and any role key that was
    explicitly overridden is preserved even if it equals the base default.

    A role key is inherited from `model_config` only when it was not set
    explicitly and is absent from the role or still equals the base
    `model_config` value. "Explicit" covers every role path assigned by an
    override file or the CLI, including values that YAML anchors expand to in
    an override file:
    * A role key assigned as a whole (`role.k=v`, `role.k=null`, `role.k={}`,
      flow-style `role.k={...}`, or `k: {}` in an override file) is kept as is.
    * A nested sub-key is kept when it or any of its descendants was assigned,
      so depth-4 overrides such as `role.kwargs.rope.factor=2` count.
    * `mesh.shape` and `mesh.axis_names` are one unit: if a role sets either,
      it keeps its own pair and inherits neither.

    Overrides only add keys, so `role.k={}` cannot clear a mapping that
    `role.k` already holds; only `role.k=null` can.

    Args:
      base_model_config: `model_config` from the base YAML before overrides.
      all_overrides: Flattened `key=value` overrides (files, then CLI).

    Raises:
      KeyError: If the config has no `model_config` section.
      TypeError: If `model_config` is null or not a mapping.
    """
    current_model_config = self.config_mapping("model_config")
    assigned, touched = _override_paths(all_overrides)
    for config_key in _ROLE_MODEL_CONFIG_KEYS:
      if config_key not in self.config or self.config[config_key] is None:
        continue
      role_cfg = self._mutable_config_mapping(config_key)
      for k, v in current_model_config.items():
        path = f"{config_key}.{k}"
        if path in assigned:
          continue
        if (
            isinstance(v, Mapping)
            and k in role_cfg
            and isinstance(role_cfg[k], Mapping)
        ):
          base_sub: Mapping[str, object] = {}
          if k in base_model_config:
            base_val = base_model_config[k]
            if isinstance(base_val, Mapping):
              base_sub = to_str_dict(base_val)
          role_cfg[k] = self._merge_role_sub_keys(
              path=path,
              is_mesh=k == "mesh",
              model_sub=to_str_dict(v),
              role_sub=copy.deepcopy(to_str_dict(role_cfg[k])),
              base_sub=base_sub,
              touched=touched,
          )
        elif path not in touched and _inherits(role_cfg, base_model_config, k):
          role_cfg[k] = copy.deepcopy(v)

  @staticmethod
  def _merge_role_sub_keys(
      *,
      path: str,
      is_mesh: bool,
      model_sub: Mapping[str, object],
      role_sub: dict[str, object],
      base_sub: Mapping[str, object],
      touched: frozenset[str],
  ) -> dict[str, object]:
    """Merges one nested `model_config` mapping into a role's mapping.

    Args:
      path: Dotted path of the role's mapping, e.g. `actor_model_config.mesh`.
      is_mesh: Whether the mapping is `mesh`, whose `shape` and `axis_names` are
        inherited as one unit.
      model_sub: The current `model_config` value of the mapping.
      role_sub: A copy of the role's value of the mapping; updated in place.
      base_sub: The base `model_config` value of the mapping.
      touched: Paths assigned by the overrides, plus all their prefixes.

    Returns:
      `role_sub` with the inherited sub-keys applied.
    """
    units: list[tuple[str, ...]] = []
    if is_mesh:
      units.append(_MESH_PAIR_KEYS)
    units.extend(
        (sub_k,)
        for sub_k in model_sub
        if not (is_mesh and sub_k in _MESH_PAIR_KEYS)
    )
    for unit in units:
      if any(f"{path}.{sub_k}" in touched for sub_k in unit):
        continue
      if not all(_inherits(role_sub, base_sub, sub_k) for sub_k in unit):
        continue
      for sub_k in unit:
        if sub_k in model_sub:
          role_sub[sub_k] = copy.deepcopy(model_sub[sub_k])
    return role_sub

  def config_mapping(self, key: str) -> dict[str, Any]:
    """Returns a deep, plain-dictionary copy of a config section.

    Unlike `_config_mapping`, a missing or null section is an error rather than
    `{}`, and mutating the result never changes `self.config`.

    Args:
      key: Top-level config key of the section.

    Returns:
      A deep copy of the section. `DictConfig` sections are converted to plain
      containers with interpolations resolved.

    Raises:
      KeyError: If `key` is not in the config.
      TypeError: If the section is null or not a mapping.
    """
    value = self.config[key]
    if value is None:
      raise TypeError(f"Config section {key!r} is null.")
    if not isinstance(value, Mapping):
      raise TypeError(
          f"Expected config section {key!r} to be a mapping, got"
          f" {type(value).__name__}."
      )
    if isinstance(value, omegaconf.DictConfig):
      return to_str_dict(value)
    return copy.deepcopy(to_str_dict(value))

  def _config_mapping(self, key: str) -> dict[str, Any]:
    """Returns a config section as a plain dictionary.

    This narrows nested config sections that may otherwise be inferred as broad
    unions of scalars and mappings by static type checkers.

    Args:
      key: Key of config section.

    Returns:
      The mapped dictionary config section.
    """
    value = self.config.get(key)
    if value is None:
      return {}
    if not isinstance(value, Mapping):
      raise TypeError(
          f"Expected config section {key!r} to be a mapping, got"
          f" {type(value).__name__}."
      )
    if isinstance(value, omegaconf.DictConfig):
      return omegaconf.OmegaConf.to_container(value, resolve=True)  # pyrefly: ignore[bad-return]
    return dict(value)

  def _mutable_config_mapping(self, key: str) -> MutableMapping[str, Any]:
    """Returns a mutable config section for in-place updates.

    Args:
      key: Key of config section.

    Returns:
      The mutable mapping of the config section.
    """
    value = self.config.get(key)
    if value is None:
      section: dict[str, Any] = {}
      self.config[key] = section
      return section
    if not isinstance(value, MutableMapping):
      raise TypeError(
          f"Expected config section {key!r} to be a mutable mapping, got"
          f" {type(value).__name__}."
      )
    return value

  def _config_string(self, key: str, default: str = "") -> str:
    """Returns a string config value with validation.

    Args:
      key: Key of config value.
      default: Default fallback value if not set.

    Returns:
      The string config value.
    """
    value = self.config.get(key, default)
    if value is None:
      return default
    if not isinstance(value, str):
      raise TypeError(
          f"Expected config value {key!r} to be a string, got"
          f" {type(value).__name__}."
      )
    return value

  def _config_bool(self, key: str, default: bool = False) -> bool:
    """Returns a boolean config value with validation.

    Args:
      key: Key of config value.
      default: Default fallback value if not set.

    Returns:
      The boolean config value.
    """
    value = self.config.get(key, default)
    if value is None:
      return default
    if not isinstance(value, bool):
      raise TypeError(
          f"Expected config value {key!r} to be a bool, got"
          f" {type(value).__name__}."
      )
    return value

  def _validate_perf_metrics(self, entry_point: str):
    """Validates that perf metrics are only enabled for GRPO.

    Args:
      entry_point: The entry point of the pipeline.

    Raises:
      ValueError: If perf metrics are enabled but the entry point is not
        "grpo_main".
    """
    perf_config = self.config.get("training_config", {}).get(
        "perf_metrics_options", {}
    )

    if perf_config:
      if not entry_point.endswith("grpo_main"):
        raise ValueError(
            "Perf metrics are currently only supported for GRPO training"
            " (grpo_main)."
        )
      custom_export_fn_path = perf_config.get("custom_export_fn_path")
      if (
          custom_export_fn_path
          and self._get_function_from_path(custom_export_fn_path) is None
      ):
        raise ValueError(
            "Could not load custom export function from"
            f" {custom_export_fn_path}"
        )
      custom_export_fn_path_v2 = perf_config.get("custom_export_fn_path_v2")
      if (
          custom_export_fn_path_v2
          and self._get_function_from_path(custom_export_fn_path_v2) is None
      ):
        raise ValueError(
            "Could not load custom export function v2 from"
            f" {custom_export_fn_path_v2}"
        )

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
      # Only require HF_TOKEN when loading from HuggingFace Hub (not a local
      # or CNS path).
      is_local_path = tokenizer_path.startswith(("/", "gs://"))
      if not is_local_path and "HF_TOKEN" not in os.environ:
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
      logging.error(
          "Error: '%r' is not a valid directory.", model_download_path
      )
      return

    logging.info("Clearing contents of '%s'...", model_download_path)
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
          logging.info("  Removed file/link: %s", item_path)
        elif os.path.isdir(item_path):
          # shutil.rmtree can also handle permission issues internally
          # by providing an onerror handler if needed.
          shutil.rmtree(item_path)
          logging.info("  Removed directory: %s", item_path)
      except (OSError, shutil.Error):
        logging.warning("  Failed to delete %s.", item_path, exc_info=True)
    logging.info("Finished clearing '%r'.", model_download_path)

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
        lambda: ["huggingface", "internal", "maxtext"]
    )
    # TODO: b/467448875 - Add support for other sources, such as kaggle for other
    # models.
    supported_sources["gemma"] = ["kaggle", "internal", "maxtext"]
    supported_sources["gemma2"] = ["kaggle", "internal", "maxtext"]
    supported_sources["gemma3"] = ["gcs", "internal", "maxtext"]
    # TODO: support maxtext for gemma4.
    supported_sources["gemma4"] = ["kaggle", "huggingface", "gcs", "internal"]

    if model_name.startswith(("gemma4", "gemma-4")):
      model_family = "gemma4"
    elif model_name.startswith(("gemma3", "gemma-3")):
      model_family = "gemma3"
    elif model_name.startswith(("gemma2", "gemma-2")):
      model_family = "gemma2"
    elif model_name.startswith("gemma"):
      model_family = "gemma"
    else:
      model_family = "other"

    expected_sources = supported_sources[model_family]
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

    # Use module-level helper.
    return create_optimizer(optimizer_config, config_path_info)

  def parse_mesh_config(
      self, model_key: str
  ) -> tuple[tuple[int, ...], tuple[str, ...]]:
    """Validates and parses the mesh shape and axis names for one model.

    Args:
      model_key: Config section name such as ``model_config`` or
        ``actor_model_config``.

    Returns:
      A tuple ``(axis_shapes, axis_names)`` ready to pass to
      ``tunix.utils.mesh.create_mesh``.

    Raises:
      ValueError: If the mesh config is missing, malformed, or internally
        inconsistent.
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

    if not isinstance(axis_shapes, tuple):
      raise ValueError(
          f"'mesh.shape' must be a list or tuple, got {type(axis_shapes)}."
      )
    if not all(isinstance(x, int) for x in axis_shapes):
      raise ValueError(
          f"All elements in mesh.shape must be integers, got {axis_shapes}."
      )

    if not isinstance(axis_names, tuple):
      raise ValueError(
          f"'mesh.axis_names' must be a tuple, got {type(axis_names)}."
      )
    if not all(isinstance(x, str) for x in axis_names):
      raise ValueError(
          f"All elements in mesh.axis_names must be strings, got {axis_names}."
      )

    if len(axis_shapes) != len(axis_names):
      raise ValueError(
          f"mesh.shape {axis_shapes} and mesh.axis_names {axis_names} "
          "must have the same length."
      )
    return tuple(axis_shapes), tuple(axis_names)

  def mesh_axes(self, model_key: str) -> dict[str, int]:
    """Returns the size of each named mesh axis for one model.

    Reads only `self.config[model_key]["mesh"]`; `same_mesh_as` is not
    resolved.

    Args:
      model_key: Config section name such as `actor_model_config`.

    Returns:
      A dictionary with exactly the keys `fsdp`, `tp`, `expert`, and `dp`.
      Axes absent from `mesh.axis_names` have size 1.

    Raises:
      ValueError: If the mesh config is invalid (see `parse_mesh_config`), or
        `mesh.axis_names` repeats a name or holds a name other than `fsdp`,
        `tp`, `expert`, or `dp`.
    """
    axis_shapes, axis_names = self.parse_mesh_config(model_key)
    duplicates = sorted({n for n in axis_names if axis_names.count(n) > 1})
    if duplicates:
      raise ValueError(
          f"{model_key}.mesh.axis_names {axis_names} repeats {duplicates}."
      )
    unknown = sorted(set(axis_names) - set(_MESH_AXES))
    if unknown:
      raise ValueError(
          f"{model_key}.mesh.axis_names {axis_names} has unknown axis names"
          f" {unknown}; supported names are {list(_MESH_AXES)}."
      )
    axes = {name: 1 for name in _MESH_AXES}
    axes.update(zip(axis_names, axis_shapes))
    return axes

  def _parse_mesh_allocation_policy(self, model_key: str) -> str:
    """Validates and returns the mesh allocation policy for one model.

    Mesh allocation policy controls how Tunix chooses device subsets when a
    mesh must be carved out of a larger device pool.

    Supported values are:

    * ``COMPACT``: prefer the smallest remaining region that can satisfy the
      request.
    * ``PERFORMANCE``: prefer the most cubical supported extracted shape.

    When ``mesh.allocation_policy`` is omitted, this defaults to ``COMPACT``.

    Args:
      model_key: Config section name such as ``model_config`` or
        ``actor_model_config``.

    Returns:
      The normalized allocation policy string.

    Raises:
      ValueError: If the mesh config is missing or the policy value is not
        supported.
    """
    mesh_config = self.config[model_key].get("mesh")
    if not mesh_config:
      raise ValueError("Missing 'mesh' configuration in raw_keys.")
    if not isinstance(mesh_config, collections.abc.Mapping):
      raise ValueError(
          "The 'mesh' configuration must be a dictionary-like object, got"
          f" {type(mesh_config)}."
      )
    return mesh_lib.normalize_allocation_policy(
        mesh_config.get("allocation_policy")
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
        if not isinstance(value, collections.abc.Mapping):
          raise ValueError(
              "Expected dictionary for checkpointing_options, but got"
              f" {type(value).__name__}"
          )
        try:
          constructed_training_config[key] = (
              checkpoint_options.checkpointing_options_from_dict(value)
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
      elif key == "perf_metrics_options":
        if value:
          try:
            constructed_training_config[key] = perf_metrics.PerfMetricsOptions(
                **value
            )
          except ValueError as e:
            raise ValueError(f"Invalid perf metrics options: {value}") from e
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
      overrides: list[str],
      **_kwargs,
  ):
    """Update the configuration from command line."""

    cli_cfg = omegaconf.OmegaConf.from_cli(overrides)

    raw_data_from_cmd_line = omegaconf.OmegaConf.to_container(
        cli_cfg, resolve=True
    )

    updated_keys = []

    # Check for conflicts and unknown keys.
    for k in raw_data_from_cmd_line:  # pyrefly: ignore[not-iterable]
      if not k:
        continue
      if k not in raw_data_from_yaml:
        raise ValueError(
            f"Key {k} was passed at the command line but isn't in config."
        )

    # Iterate over key from base yaml
    for k in raw_data_from_yaml:

      # Error out if same key defined in cmd line and environment
      if k in raw_data_from_cmd_line and yaml_key_to_env_key(k) in os.environ:  # pyrefly: ignore[not-iterable]
        raise ValueError(
            f"You are passing overrides by both CLI and ENV for `{k}`. This"
            " isn't allowed."
        )

      # Take value from base config yaml if key is not specified in command line
      # or environment.
      if (
          k not in raw_data_from_cmd_line  # pyrefly: ignore[not-iterable]
          and yaml_key_to_env_key(k) not in os.environ
      ):
        # take the config value from the YAML file.
        raw_keys[k] = raw_data_from_yaml[k]
        continue

      #  Key is specified on either command line or enviornment
      updated_keys.append(k)

      # take updated value from command line or enviornment
      if k in raw_data_from_cmd_line:  # pyrefly: ignore[not-iterable]
        new_proposal = raw_data_from_cmd_line[k]  # pyrefly: ignore[bad-index, unsupported-operation]
      else:
        new_proposal = os.environ.get(yaml_key_to_env_key(k))

      new_proposal = _normalize_cli_override(
          raw_data_from_yaml[k], new_proposal
      )

      if raw_data_from_yaml[k] is None:
        if new_proposal is None:
          raw_keys[k] = None
          continue
        if _can_override_nullable_schema(new_proposal):
          raw_keys[k] = copy.deepcopy(new_proposal)
          continue
        raise ValueError(
            f"For key '{k}', nullable schema can't accept value of type"
            f" {type(new_proposal)} from the CLI or ENV"
        )

      # If specified value is not one of type in base config yaml or is not
      # consumed by to type parser, error out
      # TODO(b/477343879): ensure Type checking for values with no defaults such
      # as lora_config
      if (not isinstance(new_proposal, type(raw_data_from_yaml[k]))) and (
          type(raw_data_from_yaml[k]) not in _yaml_types_to_parser
      ):
        raise ValueError(
            f"For key '{k}', type {type(raw_data_from_yaml[k])} not in"
            f" {_yaml_types_to_parser.keys()}, can't pass at the CLI or ENV"
        )

      # Take the config value
      if new_proposal is None:
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
          output[key] = self.update_dict(output_val, source_val)  # pyrefly: ignore[bad-argument-type]
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

  def _load_config_from_yaml(self, config_path: str):
    """Try Loading and validate the configuration from the YAML file."""

    try:
      config_oconf = omegaconf.OmegaConf.load(config_path)
    except FileNotFoundError as e:
      raise ValueError(f"Config {config_path} not found.") from e

    return config_oconf

  def obtain_reward_fn(self) -> list[Callable[..., Any]]:
    """Obtain reward function from the config."""
    project_root = get_project_root()
    reward_fns = []
    for reward_fn_path in self.config["reward_functions"]:

      module = None
      # If the path is relative, try importing the module directly.
      if "/" not in reward_fn_path:
        try:
          module = importlib.import_module(reward_fn_path)
          module_name = module.__name__
        except Exception as e:  # pylint: disable=broad-except
          logging.warning(
              "'%s' import failed: %s", reward_fn_path, e, exc_info=True
          )
          module = None

      # Try importing the module from the project root.
      if module is None:
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
              f"Failed to execute module {module_name} from {full_path}"
          ) from e
      if module is None:
        raise RuntimeError(
            f"Module from path '{reward_fn_path}' failed to load."
        )

      loaded_module = module
      if self.config["verl_compatible"]:

        def reward_fn(
            prompts, completions, reward_model, *, lm=loaded_module, **kwargs
        ):
          del prompts, kwargs
          ground_truths = reward_model["ground_truth"]
          return [
              lm.compute_score(c, gt)
              for c, gt in zip(completions, ground_truths)
          ]

        reward_fns.append(reward_fn)

      else:
        # Get all defined functions in the file as reward functions.
        # We explicitly ignore functions whose names start with an underscore
        # (_), ensuring private helper functions are never mistaken for
        # reward functions.
        defined_functions = []
        for name, member in inspect.getmembers(module):
          if inspect.isfunction(member) and not name.startswith("_"):
            # Check if the function was defined in this module
            if member.__module__ == module_name:  # pyrefly: ignore[unbound-name]
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


def initialize(
    argv: Sequence[str],
    *,
    default_base_config: str | pathlib.Path | None = None,
    merge_replace_keys: bool = False,
    validate_nested: bool = False,
    **kwargs: object,
) -> HyperParameters:
  return HyperParameters(
      argv,
      default_base_config=default_base_config,
      merge_replace_keys=merge_replace_keys,
      validate_nested=validate_nested,
      **kwargs,
  )
