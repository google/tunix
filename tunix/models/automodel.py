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

import dataclasses
import enum
import gc
import importlib
import os
import shutil
from typing import Any

from absl import logging
from flax import nnx
import jax
import jax.numpy as jnp
from orbax import checkpoint as ocp
from tunix.models import maxtext_parallelism
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
  MAXTEXT = 'maxtext'  # Load model from Maxtext.


def get_model_module(model_name: str, module_type: ModelModule) -> Any:
  """Dynamically imports a model module (e.g., 'model' or 'params')."""
  model_config_category = naming.ModelNaming(
      model_name=model_name
  ).model_config_category
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
        (e.g., "gemma-2b", "llama-3.1-8b", "qwen2.5-0.5b").

  Returns:
      The result from calling the dynamically determined function.

  Raises:
      ValueError: If the model_string doesn't match any known prefix.
      AttributeError: If the derived function name does not exist in the target
      object.
      TypeError: If the attribute found on the target object is not callable.
  """
  naming_info = naming.ModelNaming(model_name=model_name)
  config_id = naming_info.model_config_id
  model_lib_module = get_model_module(model_name, ModelModule.MODEL)
  target_obj = model_lib_module.ModelConfig

  if not hasattr(target_obj, config_id):  # pyrefly: ignore[bad-argument-type]
    raise AttributeError(
        f"Error: Function '{config_id}' not found on the target object "
        f"for model '{model_name}'. Target object type: {type(target_obj)}"
    )

  method_to_call = getattr(
      target_obj, config_id
  )  # pyrefly: ignore[bad-argument-type]

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
    model_path: str | None = None,
) -> tuple[nnx.Module, Any]:
  """Creates a Gemma model with NNX conversion, using a cached checkpoint if available.

  Args:
      model_name: The name of the model (e.g., "gemma-2b").
      ckpt_path: The base path to the checkpoints.
      intermediate_ckpt_dir: Directory to save or load the NNX converted
        checkpoint.
      rng_seed: The random seed for model initialization.
      mesh: Mesh object for device layout.
      model_path: Optional. The specific path to the model files. If None,
        the path is inferred from `model_name` and `ckpt_path`.

  Returns:
      A tuple containing:
          - model: The loaded nnx.Module.
          - model_params: The model parameters.
  """

  def _nnx_convert_and_reload() -> tuple[nnx.Module, Any]:
    """Converts the model to an NNX checkpoint and reloads it.

    This is a workaround, as the checkpoints on Kaggle don't work with NNX. This
    takes a long time. Skip if conversion is not needed.
    """
    if model_path:
      dir_name = os.path.basename(model_path)
    else:
      # If model_path is not provided, fall back to inferring from model_name
      logging.warning(
          'model_path is not provided. Inferring from model_name. This may lead'
          ' to incorrect results if the model_name (%s) is not a standard Gemma'
          ' model name.',
          model_name,
      )
      naming_info = naming.ModelNaming(model_name=model_name)
      version_dashed = None
      if naming_info.model_version is not None:
        version_dashed = naming_info.model_version.replace('_', '-')

      if naming_info.model_family == 'gemma2':
        dir_name = f'gemma-2-{version_dashed}'
      elif naming_info.model_family == 'gemma1p1':
        dir_name = f'gemma-1.1-{version_dashed}'
      else:  # gemma
        dir_name = version_dashed

    params_path = os.path.join(
        ckpt_path, dir_name
    )  # pyrefly: ignore[no-matching-overload]

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
    except Exception as e:  # pylint: disable=broad-exception-caught

      logging.warning(
          'Failed to load from intermediate_ckpt_dir %s: %s. '
          'Purging directory and falling back to fresh NNX conversion.',
          intermediate_ckpt_dir,
          e,
          exc_info=True,
      )
      shutil.rmtree(intermediate_ckpt_dir, ignore_errors=True)
  return _nnx_convert_and_reload()


def create_gemma_model_from_params(
    params_path: str, model_name: str
) -> tuple[nnx.Module, Any]:
  """Loads Gemma params and creates a model."""
  params_lib = get_model_module(model_name, ModelModule.PARAMS)
  model_params = params_lib.load_and_format_params(params_path)
  model_module_lib = get_model_module(model_name, ModelModule.MODEL)
  # TODO(b/451662153): have gemma2 version handling done better in naming.py
  naming_info = naming.ModelNaming(model_name=model_name)
  version = naming_info.model_version
  if naming_info.model_family == 'gemma1p1':
    version = f'1.1-{version}'
  elif naming_info.model_family == 'gemma2':
    version = f'2-{version}'
  model = model_module_lib.Gemma.from_params(model_params, version=version)
  return model, model_params


# TODO(b/451662153): make gemma3 and gemma2 loading logic more consistent.
# Currently, gemma2 uses _create_gemma_model_from_params while gemma3 uses
# _create_gemma3_model_from_checkpoint.
def create_gemma3_model_from_checkpoint(
    ckpt_path: str, model_name: str, mesh: jax.sharding.Mesh, **kwargs
) -> tuple[nnx.Module, Any]:
  """Creates a Gemma3/Gemma4 model from a checkpoint.

  Args:
      ckpt_path: The path to the checkpoint.
      model_name: The name of the model (e.g., "qwen2.5-0.5b", "llama-3.2-3b").
      mesh: Mesh object for device layout.
      **kwargs: Additional keyword arguments to override on model_params.

  Returns:
      A tuple containing:
          - model: The loaded and potentially LoRA-applied nnx.Module.
          - model_params: The model parameters.
  """
  model_params = call_model_config(model_name)
  valid_kwargs = {
      k: v
      for k, v in kwargs.items()
      if hasattr(model_params, k) and v is not None
  }
  model_params = dataclasses.replace(model_params, **valid_kwargs)
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
        "google/gemma-2b" for HF, or model_path for KAGGLE/GCS/INTERNAL).
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

    return oss_utils.kaggle_pipeline(
        model_id_or_path, model_download_path
    )  # pyrefly: ignore[bad-argument-type]
  elif model_source == ModelSource.HUGGINGFACE:
    from tunix.oss import utils as oss_utils  # pylint: disable=g-import-not-at-top

    return oss_utils.hf_pipeline(
        model_id_or_path, model_download_path
    )  # pyrefly: ignore[bad-argument-type]
  elif model_source in (ModelSource.GCS, ModelSource.MAXTEXT):
    return model_id_or_path
  elif model_source == ModelSource.INTERNAL:
    raise ValueError('INTERNAL model source is not supported in OSS.')
  else:
    raise ValueError(
        f'Unsupported model source: {model_source}. Only KAGGLE, GCS,'
        ' HUGGINGFACE and INTERNAL are supported.'
    )


def create_model_from_safe_tensors(
    model_name: str,
    file_dir: str,
    model_config: Any,
    mesh: jax.sharding.Mesh,
    dtype: jnp.dtype | None = None,
    mode: str = 'auto',
) -> Any:
  """Dynamically imports the correct module and calls `create_model_from_safe_tensors` based on the model_name.

  Args:
      model_name: The name of the model (e.g., "qwen2.5-0.5b", "llama-3.2-3b").
      file_dir: Directory containing the safe tensors.
      model_config: Model configuration object.
      mesh: Mesh object for device layout.
      dtype: Optional dtype to cast the loaded tensors to.
      mode: The mode to use for loading the model. Options are ('auto',
        'optimized', 'original').

  Returns:
      The result of the create_model_from_safe_tensors call.

  Raises:
      ValueError: If the model_name is invalid.
      ImportError: If the required model module cannot be found.
      AttributeError: If create_model_from_safe_tensors is not in the module.
  """
  naming_info = naming.ModelNaming(model_name=model_name)
  if naming_info.model_family in (
      'gemma',
      'gemma1p1',
      'gemma2',
      'gemma3',
      'gemma4',
  ):
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
  return create_fn(
      file_dir=file_dir,
      config=model_config,
      mesh=mesh,
      dtype=dtype,
      mode=mode,
  )


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
      maxtext_pipeline_config: (
          maxtext_parallelism.MaxTextPipelineConfig | None
      ) = None,
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
        maxtext_pipeline_config: Optional validated single-slice pipeline and
          tensor parallelism layout for ``ModelSource.MAXTEXT``. The supplied
          mesh must match this configuration.
        **kwargs: Additional keyword arguments passed to the underlying model
          creation functions. - For ModelSource.KAGGLE, Gemma models:
          `intermediate_ckpt_dir` , `rng_seed`

    Returns:
        The loaded nnx.Module model.
        The path where the model was downloaded to.
    """
    # TODO(b/477915179): Allow model_id to be config_id or a Kaggle_id
    model: nnx.Module = None  # pyrefly: ignore[bad-assignment]
    model_params: Any = None
    naming_info = naming.ModelNaming(model_id=model_id)

    if (
        maxtext_pipeline_config is not None
        and model_source != ModelSource.MAXTEXT
    ):
      raise ValueError(
          'maxtext_pipeline_config is only supported with '
          'model_source=ModelSource.MAXTEXT.'
      )

    # Download the model
    if model_path:
      model_id_or_path = model_path
    else:
      if model_source in (
          ModelSource.INTERNAL,
          ModelSource.GCS,
          ModelSource.KAGGLE,
      ):
        raise ValueError(
            'model_path is required for model_source: '
            f'{model_source}. Please provide a valid model_path.'
        )
      model_id_or_path = model_id
    resolved_model_path = download_model(
        model_id_or_path, model_download_path, model_source
    )

    # Case 1: MaxText models
    if model_source == ModelSource.MAXTEXT:
      if maxtext_pipeline_config is not None:
        maxtext_pipeline_config.validate_mesh(mesh)
        pipeline_overrides = maxtext_pipeline_config.as_maxtext_kwargs()
        conflicting_overrides = {
            key: {'config': value, 'kwargs': kwargs[key]}
            for key, value in pipeline_overrides.items()
            if key in kwargs and kwargs[key] != value
        }
        if conflicting_overrides:
          raise ValueError(
              'MaxText pipeline overrides conflict with explicit kwargs: '
              f'{conflicting_overrides}.'
          )
        kwargs.update(pipeline_overrides)

      try:
        import maxtext.configs.pyconfig as pyconfig  # pylint: disable=g-import-not-at-top # pytype: disable=import-error
        from maxtext.configs.types import MaxTextConfig  # pylint: disable=g-import-not-at-top # pytype: disable=import-error
        from maxtext.utils import model_creation_utils as maxtext_model_creation_utils  # pylint: disable=g-import-not-at-top # pytype: disable=import-error
      except ImportError:
        from maxtext.src.maxtext.configs import pyconfig  # pylint: disable=g-import-not-at-top
        from maxtext.src.maxtext.configs.types import MaxTextConfig  # pylint: disable=g-import-not-at-top
        from maxtext.src.maxtext.utils import model_creation_utils as maxtext_model_creation_utils  # pylint: disable=g-import-not-at-top

      # We provide load_parameters_path instead of model_path since that's what maxtext expects.
      argv = [
          '',
          'src/maxtext/configs/base.yml',
          f'model_name={naming_info.model_name}',
      ]

      if model_path:
        argv.append(f'load_parameters_path={resolved_model_path}')

      # We handle jax distribution outside or it's not needed by default.
      if 'skip_jax_distributed_system' not in kwargs:
        kwargs['skip_jax_distributed_system'] = True

      if 'hf_access_token' not in kwargs and 'HF_TOKEN' in os.environ:
        kwargs['hf_access_token'] = os.environ['HF_TOKEN']

      valid_keys = set()
      if hasattr(MaxTextConfig, 'model_fields'):
        valid_keys = set(MaxTextConfig.model_fields.keys())
      elif hasattr(MaxTextConfig, '__annotations__'):
        valid_keys = set(MaxTextConfig.__annotations__.keys())

      for k, v in kwargs.items():
        if v is not None and k in valid_keys:
          val_str = str(v).lower() if isinstance(v, bool) else str(v)
          argv.append(f'{k}={val_str}')

      maxtext_config = pyconfig.initialize(argv)
      model = maxtext_model_creation_utils.from_pretrained(
          maxtext_config, mesh=mesh, wrap_with_tunix_adapter=True
      )
      return model, resolved_model_path
    # For other native Tunix models with special handling cases for Gemma3 models
    elif naming_info.model_family == 'gemma3':
      if model_source in (ModelSource.GCS, ModelSource.INTERNAL):
        model, model_params = create_gemma3_model_from_checkpoint(
            ckpt_path=resolved_model_path,
            model_name=naming_info.model_name,  # pyrefly: ignore[bad-argument-type]
            mesh=mesh,
            **kwargs,
        )
      else:
        raise NotImplementedError(
            'Gemma 3 models are only supported from GCS or INTERNAL.'
            f' Specified model source: {model_source}'
        )
    # Gemma 4: Orbax loading for GCS/INTERNAL sources.
    # Other sources (e.g., HuggingFace) fall through to the common
    # SafeTensors path, which resolves to gemma4/params_safetensors.py.
    elif naming_info.model_family == 'gemma4':
      if model_source in (ModelSource.GCS, ModelSource.INTERNAL):
        # Name is legacy — dynamically resolves to gemma4 via ModelNaming.
        model, model_params = create_gemma3_model_from_checkpoint(
            ckpt_path=resolved_model_path,
            model_name=naming_info.model_name,  # pyrefly: ignore[bad-argument-type]
            mesh=mesh,
            **kwargs,
        )
      else:
        logging.info(
            'Gemma 4 source %s is not GCS/INTERNAL, falling through to'
            ' SafeTensors loader.',
            model_source,
        )
    # For other native Tunix models with special handling cases for Gemma2 models
    elif naming_info.model_family in ('gemma', 'gemma1p1', 'gemma2'):
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
            model_name=naming_info.model_name,  # pyrefly: ignore[bad-argument-type]
            ckpt_path=resolved_model_path,
            intermediate_ckpt_dir=intermediate_ckpt_dir,  # pyrefly: ignore[bad-argument-type]
            rng_seed=rng_seed,
            mesh=mesh,
            model_path=model_path,
        )
      elif model_source == ModelSource.INTERNAL:
        model, model_params = create_gemma_model_from_params(
            params_path=resolved_model_path,
            model_name=naming_info.model_name,  # pyrefly: ignore[bad-argument-type]
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
          f' {model_source} and model name: {naming_info.model_name}'
      )

    # Common path for all other native Tunix models -- create model from safe tensors
    if not model_params:
      # pick corresponding config based on model version
      model_params = call_model_config(
          naming_info.model_name
      )  # pyrefly: ignore[bad-argument-type]

      # Get load_dtype explicitly from kwargs
      load_dtype_str = kwargs.get('load_dtype')
      try:
        load_dtype = getattr(jnp, load_dtype_str)
      except AttributeError:
        raise ValueError(
            f'Invalid load_dtype: {load_dtype_str}. Must be a valid'
            ' jax.numpy type.'
        )
      except TypeError:
        load_dtype = load_dtype_str

      # Apply any model config field overrides passed via kwargs (e.g.
      # use_flash_attention, flash_attention_block_size).
      if dataclasses.is_dataclass(model_params):
        valid_fields = {f.name for f in dataclasses.fields(model_params)}
        overrides = {
            k: v
            for k, v in kwargs.items()
            if k in valid_fields and v is not None
        }
        if 'remat_config' in overrides and isinstance(
            overrides['remat_config'], str
        ):
          model_module = get_model_module(
              naming_info.model_name, ModelModule.MODEL
          )
          if hasattr(model_module, 'RematConfig'):
            remat_cfg_str = overrides['remat_config']
            try:
              overrides['remat_config'] = getattr(
                  model_module.RematConfig, remat_cfg_str
              )
            except AttributeError:
              raise ValueError(
                  f'Invalid remat_config: {remat_cfg_str}. Must be a valid'
                  ' RematConfig type.'
              )
        if 'dtype' in overrides:
          dtype_str = overrides['dtype']
          try:
            overrides['dtype'] = getattr(jnp, dtype_str)
          except AttributeError:
            raise ValueError(
                f'Invalid dtype: {dtype_str}. Must be a valid jax.numpy type.'
            )
          except TypeError:
            pass

        if overrides:
          logging.info('Applying model config overrides: %s', overrides)
          model_params = dataclasses.replace(model_params, **overrides)

      with mesh:
        model = create_model_from_safe_tensors(
            naming_info.model_name,  # pyrefly: ignore[bad-argument-type]
            resolved_model_path,
            model_params,
            mesh,
            dtype=load_dtype,
        )

    return model, resolved_model_path
