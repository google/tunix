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

"""AutoModel class."""

import enum
import importlib
from typing import Any

from absl import logging
import jax
from tunix.models import naming


_BASE_MODULE_PATH = 'tunix.models'


class ModelModule(enum.Enum):
  """Specifies the type of model module to import."""

  MODEL = 'model'
  PARAMS = 'params'


def get_model_module(model_name: str, module_type: ModelModule) -> Any:
  """Dynamically imports a model module (e.g., 'model' or 'params')."""
  model_config_category = naming.get_model_config_category(model_name)
  module_path = (
      f'{_BASE_MODULE_PATH}.{model_config_category}.{module_type.value}'
  )
  try:
    logging.info('Attempting to import: %s', module_path)
    model_lib_module = importlib.import_module(module_path)
    return model_lib_module
  except ImportError as exc:
    raise ImportError(
        'Could not import module for model config category: '
        f'{model_config_category} at path: {module_path}. Please check '
        'BASE_MODULE_PATH and ensure the module exists and is a dependency.'
    ) from exc


def obtain_model_params(model_name: str) -> Any:
  """Dynamically calls a configuration function based on the model_string.

  The routing to the correct module/class instance is based on the longest
  matching prefix of model_name found in CONFIG_MAP.
  Hyphens and dots in the model_name are converted to underscores
  to form the function name.

  Args:
      model_name: The string indicating which model config function to call
        (e.g., "gemma-2b", "llama3.1-8b", "qwen2.5-0.5b").

  Returns:
      The result from calling the dynamically determined function.

  Raises:
      ValueError: If the model_string doesn't match any known prefix.
      AttributeError: If the derived function name does not exist in the target
      object.
      TypeError: If the attribute found on the target object is not callable.
  """
  config_id = naming.get_model_config_id(model_name)
  model_lib_module = get_model_module(model_name, ModelModule.MODEL)
  target_obj = model_lib_module.ModelConfig

  if not hasattr(target_obj, config_id):
    raise AttributeError(
        f"Error: Function '{config_id}' not found on the target object "
        f"for model '{model_name}'. Target object type: {type(target_obj)}"
    )

  method_to_call = getattr(target_obj, config_id)

  if not callable(method_to_call):
    raise TypeError(
        f"Error: Attribute '{config_id}' on the target object is not callable."
    )

  logging.info(
      'Attempting to call: %s() on object of type %s',
      config_id,
      type(target_obj),
  )
  return method_to_call()


def _create_model_from_safe_tensors_dynamically(
    model_name: str, file_dir: str, model_config: Any, mesh: jax.sharding.Mesh
) -> Any:
  """Dynamically imports the correct module and calls `create_model_from_safe_tensors` based on the model_name.

  Args:
      model_name: The name of the model (e.g., "qwen2.5-0.5b", "llama3.2-3b").
      file_dir: Directory containing the safe tensors.
      model_config: Model configuration object.
      mesh: Mesh object for device layout.

  Returns:
      The result of the create_model_from_safe_tensors call.

  Raises:
      ValueError: If the model_name is invalid.
      ImportError: If the required model module cannot be found.
      AttributeError: If create_model_from_safe_tensors is not in the module.
  """

  model_config_category = naming.get_model_config_category(model_name)
  if model_config_category.startswith('gemma'):
    # TODO(b/444572467): Remove this check once Gemma safetensors works
    raise NotImplementedError(
        'Gemma safetensors loading is not supported in AutoModel. Please use'
        ' the original model module.'
    )
  params_module = get_model_module(model_name, ModelModule.PARAMS)

  try:
    create_fn = getattr(params_module, 'create_model_from_safe_tensors')
  except AttributeError as exc:
    raise AttributeError(
        "'create_model_from_safe_tensors' not found in module "
        f'{params_module.__name__} for model {model_name}'
    ) from exc

  logging.info(
      'Calling %s.create_model_from_safe_tensors', params_module.__name__
  )
  return create_fn(file_dir=file_dir, config=model_config, mesh=mesh)


class AutoModel:
  """A generic model class that will be instantiated as one of the model classes of the library.

  This class provides a way to instantiate a model from a configuration, or load
  a pretrained model from a file directory.
  It relies on dynamic imports based on the model name to load the correct model
  class and parameters.

  Example:
    To load a pretrained model from safe tensors:
    ```
    model = AutoModel.from_pretrained("qwen2.5-0.5b", "/path/to/weights", mesh)
    ```

    To instantiate a model from a config:
    ```
    model_params = obtain_model_params("qwen2.5-0.5b")
    model = AutoModel.from_config("qwen2.5-0.5b", model_params)
    ```
  """

  def __init__(self):
    raise EnvironmentError(
        'AutoModel is designed to be instantiated using site-class methods '
        'like `from_pretrained()` or `from_config()`.'
    )

  @classmethod
  def from_config(cls, model_name: str, *args, **kwargs):
    """Instantiates one of the model classes of the library from a configuration.

    Args:
      model_name: The name of the model (e.g., "qwen2.5-0.5b", "llama3.2-3b").
      *args: Positional arguments to pass to the model constructor.
      **kwargs: Keyword arguments to pass to the model constructor.

    Returns:
      An instance of the model class (e.g., Transformer).
    """
    model_lib_module = get_model_module(model_name, ModelModule.MODEL)
    return model_lib_module.Transformer(*args, **kwargs)

  @classmethod
  def from_pretrained(
      cls, model_name: str, file_dir: str, mesh: jax.sharding.Mesh
  ):
    """Instantiates one of the pretrained model classes of the library from a file directory.

    This method loads model weights from safe tensors found in `file_dir`.

    Args:
      model_name: The name of the model (e.g., "qwen2.5-0.5b", "llama3.2-3b").
      file_dir: Directory containing the safe tensors.
      mesh: Mesh object for device layout.

    Returns:
      An instance of the model class with pretrained weights loaded.
    """
    model_params = obtain_model_params(model_name)
    return _create_model_from_safe_tensors_dynamically(
        model_name, file_dir, model_params, mesh
    )
