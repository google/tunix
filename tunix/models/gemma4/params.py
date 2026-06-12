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

"""Gemma4 model parameters.

This provides a mapping from upstream ORBAX_FLAX NESTED checkpoints [1] to our
Tunix NNX implementation.

[1] https://github.com/google-deepmind/gemma
"""

from collections.abc import Mapping
import itertools
from typing import Any

from absl import logging
from etils import epath
import flax
from flax import nnx
import jax
from jax import numpy as jnp
from orbax import checkpoint as ocp
from tunix.models.gemma4 import model as model_lib

import sentencepiece as spm

# Pretrained
GEMMA4_E2B_PT = 'gs://gemma-data/checkpoints/gemma4-e2b-pt'
GEMMA4_E4B_PT = 'gs://gemma-data/checkpoints/gemma4-e4b-pt'
# Instruction Tuned
GEMMA4_E2B_IT = 'gs://gemma-data/checkpoints/gemma4-e2b-it'
GEMMA4_E4B_IT = 'gs://gemma-data/checkpoints/gemma4-e4b-it'
# Tokenizer
GEMMA4_TOKENIZER = 'gs://gemma-data/tokenizers/tokenizer_gemma4.model'


def create_model_from_checkpoint(
    checkpoint_path: str,
    model_config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype = jnp.bfloat16,
) -> model_lib.Gemma4:
  """Load a Gemma4 model from an Orbax checkpoint.

  Args:
    checkpoint_path: Path to an Orbax checkpoint directory.
    model_config: Gemma4 model configuration.
    mesh: Optional JAX sharding mesh for distributed loading.
    dtype: Parameter dtype (default: bfloat16).

  Returns:
    A Gemma4 model instance with loaded weights.
  """
  abs_model = nnx.eval_shape(
      lambda: model_lib.Gemma4(model_config, rngs=nnx.Rngs(0))
  )
  raw_params = ocp.StandardCheckpointer().restore(checkpoint_path)
  mapped_params = map_from_upstream_checkpoint(raw_params)
  model_state = nnx.state(abs_model)

  # Prune multimodal params (e.g., audio_input_projection) not in the
  # text-only model. Must precede validation to avoid false mismatches.
  pruned_params = _prune_to_model_keys(mapped_params, model_state)
  _validate_param_shapes(pruned_params, model_state)

  # TODO: b/343224716 - Add per-component dtype handling (like Gemma 3's
  # _get_param_dtype) when multimodal Gemma 4 loading is needed.
  if mesh is not None:
    typed_params = jax.tree_util.tree_map_with_path(
        lambda p, x, s: jnp.asarray(x, device=s, dtype=dtype),
        pruned_params,
        nnx.to_pure_dict(nnx.get_named_sharding(model_state, mesh)),
    )
  else:
    typed_params = jax.tree_util.tree_map_with_path(
        lambda p, x: jnp.asarray(x, dtype=dtype),
        pruned_params,
    )
  nnx.update(abs_model, typed_params)
  return abs_model


def _validate_param_shapes(
    mapped_params: Mapping[str, Any],
    model_state: Any,
) -> None:
  """Validate that mapped checkpoint params match expected model shapes.

  Args:
    mapped_params: Flattened parameter dict from the checkpoint.
    model_state: The model's NNX state (defines expected structure).

  Raises:
    ValueError: If the checkpoint is missing keys expected by the model or
      if any parameter shapes do not match.
  """
  flat_mapped = flax.traverse_util.flatten_dict(mapped_params)
  flat_model = flax.traverse_util.flatten_dict(nnx.to_pure_dict(model_state))

  mapped_keys = set(flat_mapped.keys())
  model_keys = set(flat_model.keys())

  missing_keys = model_keys - mapped_keys
  if missing_keys:
    raise ValueError(
        'Checkpoint is missing keys expected by the model:'
        f' {sorted(str(k) for k in missing_keys)}'
    )

  # Should not fire after _prune_to_model_keys; kept as defensive guard.
  extra_keys = mapped_keys - model_keys
  if extra_keys:
    logging.warning(
        'Checkpoint has extra keys not in model (will be ignored): %s',
        sorted(str(k) for k in extra_keys),
    )

  mismatched = []
  for key in mapped_keys & model_keys:
    mapped_val = flat_mapped[key]
    model_val = flat_model[key]
    if not hasattr(mapped_val, 'shape') or not hasattr(model_val, 'shape'):
      logging.warning(
          'Skipping shape check for non-array value at key %r '
          '(types: checkpoint=%s, model=%s)',
          key,
          type(mapped_val).__name__,
          type(model_val).__name__,
      )
      continue
    if mapped_val.shape != model_val.shape:
      mismatched.append((key, mapped_val.shape, model_val.shape))

  if mismatched:
    details = '\n'.join(
        f'  {k}: checkpoint={cs} vs model={ms}' for k, cs, ms in mismatched
    )
    raise ValueError(f'Shape mismatches (May need transpose):\n{details}')


def _prune_to_model_keys(
    params: Mapping[str, Any],
    model_state: Any,
) -> dict[str, Any]:
  """Prune checkpoint params to only include keys present in the model.

  IT checkpoints contain multimodal params (audio_input_projection,
  mm_input_projection, etc.) that don't exist in the text-only Tunix model.
  These must be removed before tree_map_with_path, which requires matching
  pytree structures.

  Args:
    params: Mapped parameter dict (may contain extra keys).
    model_state: The model's NNX state (defines expected structure).

  Returns:
    A filtered copy of params with only model-expected keys.
  """
  model_dict = nnx.to_pure_dict(model_state)
  flat_params = flax.traverse_util.flatten_dict(params)
  flat_model = flax.traverse_util.flatten_dict(model_dict)

  pruned_keys = set(flat_params.keys()) - set(flat_model.keys())
  if pruned_keys:
    logging.info(
        'Pruning %d extra checkpoint keys not in model: %s',
        len(pruned_keys),
        sorted(str(k) for k in pruned_keys),
    )

  filtered = {k: v for k, v in flat_params.items() if k in flat_model}
  return flax.traverse_util.unflatten_dict(filtered)


def create_tokenizer(
    path: str = GEMMA4_TOKENIZER,
) -> spm.SentencePieceProcessor:
  """Load the Gemma 4 SentencePiece tokenizer.

  Args:
    path: Path to the SentencePiece model file.

  Returns:
    A loaded SentencePieceProcessor instance.
  """
  spm_processor = spm.SentencePieceProcessor()
  model_proto = epath.Path(path).read_bytes()
  spm_processor.LoadFromSerializedProto(model_proto)
  return spm_processor


def map_from_upstream_checkpoint(params: Mapping[str, Any]) -> dict[str, Any]:
  """Map from upstream Orbax NESTED checkpoint to Tunix NNX layout.

  Handles both key formats produced by Orbax checkpoints:

  **Semi-flat (Gemma 3 GCS pattern):**
    ('transformer/layer_0/attn/q_einsum', 'w')  → 2-tuple

  **Genuinely nested (ORBAX_FLAX NESTED format):**
    ('transformer', 'layer_0', 'attn', 'q_einsum', 'w')  → N-tuple

  Both are normalized to a uniform list of path components before remapping
  to the nested tuple keys expected by the Tunix NNX Gemma4 model.

  Unlike Gemma 3's mapper, this function does not take a ``text_only``
  parameter.  Multimodal-only keys are mapped through and pruned later
  by ``_prune_to_model_keys``.

  Args:
    params: Raw parameter dict restored from an Orbax checkpoint.

  Returns:
    A nested dict with keys matching the Tunix NNX Gemma4 model tree.
  """
  new_params: dict[tuple[str | int, ...], Any] = {}

  for key_path, value in flax.traverse_util.flatten_dict(params).items():
    # Normalize semi-flat or nested key_path to a flat list of components.
    parts = list(
        itertools.chain.from_iterable(
            segment.split('/') for segment in key_path
        )
    )

    if parts and parts[0] == 'transformer':
      parts = parts[1:]

    if not parts:
      logging.warning('Skipping empty key path: %r', key_path)
      continue
    param_name = parts[-1]
    module_path = parts[:-1]

    # --- Embedder ---
    if module_path and module_path[0] == 'embedder':
      if len(module_path) > 1 and module_path[1] == 'per_layer_embeddings':
        # Rename upstream 'per_layer_embeddings' → Tunix field name.
        new_params[('embedder', 'per_layer_input_embedding')] = value
      elif param_name in ('per_layer_embeddings', 'per_layer_input_embedding'):
        new_params[('embedder', 'per_layer_input_embedding')] = value
      elif len(module_path) > 1:
        # Sub-modules of the embedder (e.g., mm_input_projection).
        new_params[tuple(module_path + [param_name])] = value
      else:
        # Bare embedder leaf (input_embedding).
        new_params[('embedder', param_name)] = value
      continue

    # --- Final norm ---
    if module_path and module_path[0] == 'final_norm':
      new_params[('final_norm', param_name)] = value
      continue

    # --- Layer weights ---
    if not module_path:
      logging.warning('Unexpected bare param after transformer: %r', key_path)
      continue

    # Skip multimodal modules (e.g., audio_encoder).
    if not module_path[0].startswith('layer_'):
      logging.info('Skipping non-layer module: %s', '/'.join(parts))
      continue

    layer_idx = ('layers', int(module_path[0].removeprefix('layer_')))

    # Bare leaf on the layer itself (e.g., skip_scale).
    if len(module_path) == 1:
      new_params[(*layer_idx, param_name)] = value
      continue

    # MLP gating_einsum -> split into gate_proj and up_proj.
    if module_path[1:] == ['mlp', 'gating_einsum']:
      if value.shape[0] != 2:
        raise ValueError(
            f'Expected gating_einsum shape[0]=2, got {value.shape[0]} for'
            f' {"/".join(parts)}'
        )
      new_params[(*layer_idx, 'mlp', 'gate_proj', 'kernel')] = value[0].T
      new_params[(*layer_idx, 'mlp', 'up_proj', 'kernel')] = value[1].T
      continue

    # MLP linear -> down_proj (no transpose).
    if module_path[1:] == ['mlp', 'linear']:
      new_params[(*layer_idx, 'mlp', 'down_proj', 'kernel')] = value
      continue

    # Normalize query/key norm names to underscore-prefixed form.
    if module_path[-1] in ('query_norm', '_query_norm'):
      new_params[
          (*layer_idx, *module_path[1:-1], '_query_norm', param_name)
      ] = value
      continue
    if module_path[-1] in ('key_norm', '_key_norm'):
      new_params[(*layer_idx, *module_path[1:-1], '_key_norm', param_name)] = (
          value
      )
      continue

    # Everything else: direct mapping.
    new_params[(*layer_idx, *module_path[1:], param_name)] = value

  return flax.traverse_util.unflatten_dict(new_params)
