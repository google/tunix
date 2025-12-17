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
"""AutoModel class for Tunix."""

import enum
import gc
import importlib
import os
from typing import Any
from absl import logging
from flax import nnx
import jax
import jax.numpy as jnp
from orbax import checkpoint as ocp
from tunix.models import naming


_BASE_MODULE_PATH = 'tunix.models'  # pylint: disable=invalid-name


class ModelModule(enum.Enum):
  """Specifies the type of model module to import."""

  MODEL = 'model'
  PARAMS = 'params'
  PARAMS_SAFETENSORS = 'params_safetensors'


class ModelSource(enum.Enum):
  """Specifies the source of the model."""

  KAGGLE = 'kaggle'  # Download model from Kaggle requires NNX conversion.
  GCS = 'gcs'  # Load model from GCS.
  HUGGINGFACE = 'huggingface'  # Load model from HuggingFace.
  INTERNAL = 'internal'  # Load model from Internal.


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


def call_model_config(model_name: str) -> Any:
  """Dynamically calls a configuration function based on the model_name.

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


def _get_gemma_base_model(
    model_name: str,
    intermediate_ckpt_dir: str,
    rng_seed: int,
    mesh: jax.sharding.Mesh,
) -> tuple[nnx.Module, Any]:
  """Get the base model from the intermediate checkpoint."""
  model_params = call_model_config(model_name)
  model_lib_module = get_model_module(model_name, ModelModule.MODEL)
  abs_model: nnx.Module = nnx.eval_shape(
      lambda: model_lib_module.Gemma(model_params, rngs=nnx.Rngs(rng_seed))
  )
  abs_state = nnx.state(abs_model)
  abs_state = jax.tree.map(
      lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.bfloat16, sharding=s),
      abs_state,
      nnx.get_named_sharding(abs_state, mesh),
  )
  checkpointer = ocp.StandardCheckpointer()
  restored_params = checkpointer.restore(
      os.path.join(intermediate_ckpt_dir, 'state'),
      target=abs_state,
  )

  graph_def, _ = nnx.split(abs_model)
  model = nnx.merge(graph_def, restored_params)
  return model, model_params


def create_gemma_model_with_nnx_conversion(
    model_name: str,
    ckpt_path: str,
    intermediate_ckpt_dir: str,
    rng_seed: int,
    mesh: jax.sharding.Mesh,
) -> tuple[nnx.Module, Any]:
  """Creates a Gemma model with NNX conversion, using a cached checkpoint if available."""

  def _nnx_convert_and_reload() -> tuple[nnx.Module, Any]:
    """Converts the model to an NNX checkpoint and reloads it.

    This is a workaround, as the checkpoints on Kaggle don't work with NNX. This
    takes a long time. Skip if conversion is not needed.
    """
    if model_name.startswith('gemma2'):
      params_path = os.path.join(ckpt_path, model_name)
    else:  # gemma
      suffix = '-'.join(model_name.split('-')[1:])
      params_path = os.path.join(ckpt_path, suffix)

    model, params = create_gemma_model_from_params(params_path, model_name)

    checkpointer = ocp.StandardCheckpointer()
    _, state = nnx.split(model)
    checkpointer.save(
        os.path.join(intermediate_ckpt_dir, 'state'), state, force=True
    )
    checkpointer.wait_until_finished()
    del model, params, state
    gc.collect()

    return _get_gemma_base_model(
        model_name, intermediate_ckpt_dir, rng_seed, mesh
    )

  if os.path.exists(intermediate_ckpt_dir):
    logging.info(
        'Loading from intermediate_ckpt_dir %s.', intermediate_ckpt_dir
    )
    try:
      return _get_gemma_base_model(
          model_name, intermediate_ckpt_dir, rng_seed, mesh
      )
    except (FileNotFoundError, ValueError, RuntimeError):
      logging.warning(
          'Failed to load from intermediate_ckpt_dir %s. '
          'Falling back to NNX conversion.',
          intermediate_ckpt_dir,
          exc_info=True,
      )
  return _nnx_convert_and_reload()


def create_gemma_model_from_params(
    params_path: str, model_name: str
) -> tuple[nnx.Module, Any]:
  """Loads Gemma params and creates a model."""
  params_lib = get_model_module(model_name, ModelModule.PARAMS)
  model_params = params_lib.load_and_format_params(params_path)
  model_module_lib = get_model_module(model_name, ModelModule.MODEL)
  family, version = naming.get_model_family_and_version(model_name)
  # TODO(b/451662153): have gemma2 version handling done better in naming.py
  if family == 'gemma2':
    version = f'2-{version}'
  model = model_module_lib.Gemma.from_params(model_params, version=version)
  return model, model_params


# TODO(b/451662153): make gemma3 and gemma2 loading logic more consistent.
# Currently, gemma2 uses _create_gemma_model_from_params while gemma3 uses
# _create_gemma3_model_from_checkpoint.
def create_gemma3_model_from_checkpoint(
    ckpt_path: str, model_name: str, mesh: jax.sharding.Mesh
) -> tuple[nnx.Module, Any]:
  """Creates a Gemma3 model from a checkpoint.

  Args:
      ckpt_path: The path to the checkpoint.
      model_name: The name of the model (e.g., "qwen2.5-0.5b", "llama3.2-3b").
      mesh: Mesh object for device layout.

  Returns:
      A tuple containing:
          - model: The loaded and potentially LoRA-applied nnx.Module.
          - model_params: The model parameters.
  """
  model_params = call_model_config(model_name)
  params_lib = get_model_module(model_name, ModelModule.PARAMS)
  model = params_lib.create_model_from_checkpoint(ckpt_path, model_params, mesh)
  return model, model_params


def download_model(
    model_id_or_path: str,
    model_download_path: str | None,
    model_source: ModelSource,
) -> str:
  """Downloads a model to a new model path based on the specified source.

  Args:
      model_id_or_path: The full identifier for the model (e.g.,
        "google/gemma-2b" for KAGGLE/HF, or path for GCS/INTERNAL).
      model_download_path: The local directory where the model should be
        downloaded.
      model_source: The source of the model (e.g., KAGGLE, GCS, HUGGINGFACE,
        INTERNAL).

  Returns:
      The path to the downloaded model.

  Raises:
      ValueError: If the model source is not supported for downloading.
  """

  if model_source == ModelSource.KAGGLE:
    from tunix.oss import utils as oss_utils  # pylint: disable=g-import-not-at-top

    return oss_utils.kaggle_pipeline(model_id_or_path, model_download_path)
  elif model_source == ModelSource.HUGGINGFACE:
    from tunix.oss import utils as oss_utils  # pylint: disable=g-import-not-at-top

    return oss_utils.hf_pipeline(model_id_or_path, model_download_path)
  elif model_source == ModelSource.GCS:
    return model_id_or_path
  elif model_source == ModelSource.INTERNAL:
    raise ValueError('INTERNAL model source is not supported in OSS.')
  else:
    raise ValueError(
        f'Unsupported model source: {model_source}. Only KAGGLE, GCS,'
        ' HUGGINGFACE and INTERNAL are supported.'
    )


def create_model_from_safe_tensors(
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
  if model_name.startswith('gemma'):
    params_module = get_model_module(model_name, ModelModule.PARAMS_SAFETENSORS)
  else:
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
  """Factory class for instantiating Tunix models from pretrained checkpoints.

  Similar to the Hugging Face AutoModel API.
  """

  @classmethod
  def from_pretrained(
      cls,
      model_id: str,
      mesh: jax.sharding.Mesh,
      *,
      model_source: ModelSource = ModelSource.HUGGINGFACE,
      model_path: str | None = None,
      model_download_path: str | None = None,
      **kwargs,
  ) -> tuple[nnx.Module, str | None]:
    """Loads a pretrained model from a given identifier.

    This method mimics the Hugging Face `from_pretrained` interface,
    providing a unified way to load models from various sources such as
    Hugging Face Hub, Kaggle, or GCS. The mainstream case it downloads the model
    and creates the model from safe tensors. However, for special cases, such as
    Gemma models from certain sources, different logic is used to create the
    model.

    Args:
        model_id: The full model id, e.g., "meta-llama/Llama-3.1-8B" for
          HuggingFace/Kaggle or a GCS path for GCS sources.
        mesh: The JAX sharding Mesh object.
        model_source: The source of the model (e.g., Kaggle, HuggingFace, GCS).
          Default is HuggingFace.
        model_path: The path to the model. This is particularly used for sources
          with paths that cannot be inferred from the model id, e.g., GCS and
          INTERNAL.
        model_download_path: The local directory where the model should be
          downloaded. The corresponding model_source will handle `None` cases
          differently.
        **kwargs: Additional keyword arguments passed to the underlying model
          creation functions. - For ModelSource.KAGGLE, Gemma models:
          `intermediate_ckpt_dir` , `rng_seed`

    Returns:
        The loaded nnx.Module model.
        The path where the model was downloaded to.
    """

    model: nnx.Module = None
    model_params: Any = None
    model_name = naming.get_model_name_from_model_id(model_id)

    # Download the model
    if model_source in (ModelSource.INTERNAL, ModelSource.GCS):
      if model_path is None:
        raise ValueError(
            'model_path is required for model_source: '
            f'{model_source}. Please provide a valid model_path.'
        )
      model_id_or_path = model_path
    else:
      model_id_or_path = model_id
    resolved_model_path = download_model(
        model_id_or_path, model_download_path, model_source
    )

    # Case 1: Special handling cases for Gemma models
    if model_name.startswith(('gemma3', 'gemma-3')):
      if model_source in (ModelSource.GCS, ModelSource.INTERNAL):
        model, model_params = create_gemma3_model_from_checkpoint(
            ckpt_path=resolved_model_path, model_name=model_name, mesh=mesh
        )
      else:
        raise NotImplementedError(
            'Gemma 3 models are only supported from GCS or INTERNAL.'
            f' Specified model source: {model_source}'
        )
    elif model_name.startswith('gemma'):
      if model_source == ModelSource.KAGGLE:
        # Download model from Kaggle requires NNX conversion and can takes long.
        # It is recommended to save the NNX converted model for later runs.
        # kwargs.get('intermediate_ckpt_dir') is used to save the intermediate
        # checkpoint. If it is not provided, the model will be converted but not
        # saved.
        # TODO(sizhi): Remove gemma conversion logic once load safetensors for
        # gemma is ready.
        intermediate_ckpt_dir = kwargs.get('intermediate_ckpt_dir')
        rng_seed = kwargs.get('rng_seed', 0)
        model, model_params = create_gemma_model_with_nnx_conversion(
            model_name=model_name,
            ckpt_path=resolved_model_path,
            intermediate_ckpt_dir=intermediate_ckpt_dir,
            rng_seed=rng_seed,
            mesh=mesh,
        )
      elif model_source == ModelSource.INTERNAL:
        model, model_params = create_gemma_model_from_params(
            params_path=resolved_model_path, model_name=model_name
        )
      else:
        raise NotImplementedError(
            'Gemma models are only supported from KAGGLE or INTERNAL.'
            f' Specified model source: {model_source}'
        )
    # TODO(b/467448875): Add support for other models from KAGGLE/GCS.
    elif model_source in (ModelSource.KAGGLE, ModelSource.GCS):
      raise NotImplementedError(
          'Only Gemma models are supported from KAGGLE or GCS. Please use'
          ' HUGGINGFACE for other models. Specified model source:'
          f' {model_source} and model name: {model_name}'
      )

    # Case 2: Common path for all models -- create model from safe tensors
    if not model_params:
      # pick corresponding config based on model version
      model_params = call_model_config(model_name)

      with mesh:
        model = create_model_from_safe_tensors(
            model_name,
            resolved_model_path,
            model_params,
            mesh,
        )

    return model, resolved_model_path
