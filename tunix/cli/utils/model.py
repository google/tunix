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
"""Utilities for creating and managing models in Tunix CLI."""

import os
from typing import Any, Tuple

from absl import logging
from flax import nnx
import jax
import qwix
from tunix.generate import tokenizer_adapter as tokenizer_lib
from tunix.models import automodel
from tunix.models import naming
from tunix.rl import reshard

_DEFAULT_TOKENIZER_PATH = 'meta-llama/Llama-3.1-8B'


def apply_lora_to_model(base_model, mesh, lora_config, rng_seed=0):
  """Apply Lora to the base model if given lora config."""
  logging.info('lora_config %r', lora_config)
  # Basic keyword arguments for LoraProvider
  lora_kwargs = {
      'module_path': lora_config['module_path'],
      'rank': lora_config['rank'],
      'alpha': lora_config['alpha'],
  }
  has_tile_size = 'tile_size' in lora_config
  has_weight_qtype = 'weight_qtype' in lora_config
  if has_tile_size:
    lora_kwargs['tile_size'] = lora_config['tile_size']
  if has_weight_qtype:
    lora_kwargs['weight_qtype'] = lora_config['weight_qtype']
    logging.info('Qlora is applied')
  else:
    logging.info('Lora is applied')

  try:
    lora_provider = qwix.LoraProvider(**lora_kwargs)
  except TypeError as e:
    logging.error(
        'Error initializing qwix.LoraProvider: %s. Kwargs: %s', e, lora_kwargs
    )
    # Depending on desired behavior, you might re-raise or return base_model
    raise

  model_input = base_model.get_model_input()

  # Disable remat during dummy forward pass for LoRA initialization
  original_remat = None
  if hasattr(base_model, 'config'):
    original_remat = getattr(base_model.config, 'remat_config', None)
    if original_remat is not None:
      base_model.config.remat_config = 1  # RematConfig.NONE

  lora_model = qwix.apply_lora_to_model(
      base_model, lora_provider, rngs=nnx.Rngs(rng_seed), **model_input
  )

  if original_remat is not None:
    lora_model.config.remat_config = original_remat

  if mesh is not None:
    lora_model = reshard.reshard_model_to_mesh(lora_model, mesh)
  return lora_model


def resolve_tokenizer_path(
    model_config: dict[str, Any],
    tokenizer_path: str,
) -> str:
  """Resolves the tokenizer path dynamically based on model config."""
  model_source_str = model_config.get('model_source')
  if not model_source_str:
    return tokenizer_path

  model_family = naming.ModelNaming(
      model_name=model_config['model_name']
  ).model_family

  if model_family == 'gemma3' and model_source_str == 'gcs':
    if not tokenizer_path or tokenizer_path == _DEFAULT_TOKENIZER_PATH:
      return 'gs://gemma-data/tokenizers/tokenizer_gemma3.model'
  elif (
      model_family in ('gemma', 'gemma1p1', 'gemma2')
      and model_source_str == 'kaggle'
  ):
    model_source = automodel.ModelSource(model_source_str)
    model_path = model_config.get('model_path')
    if not model_path:
      raise ValueError('model_path is required for kaggle source')
    resolved_model_path = automodel.download_model(
        model_path, model_config.get('model_download_path'), model_source
    )
    if not tokenizer_path or tokenizer_path == _DEFAULT_TOKENIZER_PATH:
      return os.path.join(resolved_model_path, 'tokenizer.model')

  return tokenizer_path


def create_tokenizer(
    tokenizer_config: dict[str, Any],
    tokenizer_path: str | None,
    model_config: dict[str, Any] | None = None,
):
  """Creates a tokenizer from the given configuration.

  Args:
      tokenizer_config: Configuration dictionary for the tokenizer.
      tokenizer_path: Optional path to the tokenizer.
      model_config: Optional model configuration dictionary used to resolve
        paths.

  Returns:
      A configured Tokenizer object.
  """
  if not tokenizer_path:
    tokenizer_path = tokenizer_config['tokenizer_path']

  if model_config is not None:
    tokenizer_path = resolve_tokenizer_path(model_config, tokenizer_path)

  tokenizer_type, add_bos, add_eos = (
      tokenizer_config['tokenizer_type'],
      tokenizer_config['add_bos'],
      tokenizer_config['add_eos'],
  )

  return tokenizer_lib.Tokenizer(
      tokenizer_type,
      tokenizer_path,
      add_bos,
      add_eos,
      os.environ.get('HF_TOKEN'),
  )


def create_model(
    model_config: dict[str, Any],
    tokenizer_config: dict[str, Any],
    mesh: jax.sharding.Mesh,
) -> Tuple[nnx.Module, str]:
  """Creates a model and determines the tokenizer path based on the model config.

  This function handles model loading from various sources (GCS, Kaggle, HF)
  and applies LoRA if specified in the config.

  Args:
      model_config: A dictionary containing model configuration, including
        'model_name', 'model_source', 'model_id', 'model_download_path',
        'intermediate_ckpt_dir', and optionally 'lora_config'.
      tokenizer_config: A dictionary containing tokenizer configuration,
        including 'tokenizer_path'.
      mesh: The JAX sharding Mesh object.

  Returns:
      A tuple containing:
          - model: The loaded and potentially LoRA-applied nnx.Module.
          - tokenizer_path: The determined path to the tokenizer model.
  """
  tokenizer_path: str = tokenizer_config['tokenizer_path']
  tokenizer_path = resolve_tokenizer_path(model_config, tokenizer_path)
  model_source_str = model_config['model_source']

  # Create Model
  try:
    model_source = automodel.ModelSource(model_source_str)
  except ValueError as exc:
    raise ValueError(
        f'Unsupported model source: {model_source_str}. '
        f'Available sources: {[s.value for s in automodel.ModelSource]}'
    ) from exc

  model, _ = automodel.AutoModel.from_pretrained(
      model_id=model_config['model_id'],
      mesh=mesh,
      model_source=model_source,
      model_download_path=model_config.get('model_download_path'),
      intermediate_ckpt_dir=model_config.get('intermediate_ckpt_dir'),
      rng_seed=model_config.get('rng_seed', 0),
      model_path=model_config.get('model_path'),
      use_flash_attention=model_config.get('use_flash_attention', False),
      flash_attention_block_size=model_config.get(
          'flash_attention_block_size', 1024
      ),
      remat_config=model_config.get('remat_config', 1),
      dtype=model_config.get('dtype'),
      load_dtype=model_config.get('load_dtype'),
      use_sliding_window_kv_cache=model_config.get('use_sliding_window_kv_cache'),
  )

  if model_config.get('lora_config'):
    # Apply Lora to model if given lora config
    model = apply_lora_to_model(
        model, mesh, model_config['lora_config'],
        rng_seed=model_config.get('rng_seed', 0)
    )
  else:
    logging.info('Training with Full Weight')

  if model_config['model_display']:
    nnx.display(model)

  return model, tokenizer_path
