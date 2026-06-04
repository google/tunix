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

"""Safetensors loader for the real Gemma 4 vision stack.

Maps `google/gemma-4-*-it` checkpoint keys (`model.vision_tower.*`,
`model.embed_vision.*`) onto a `vision_real.Gemma4VisionStack`. Audio keys
(`model.audio_tower.*`, `model.embed_audio.*`) and the e2b clip buffers
(`...{proj}.input_min/max`, `...{proj}.output_min/max`) are intentionally
unmapped, so the generic loader skips them (logged, not loaded).

IMPORTANT: the real keys are prefixed `model.`, and the loader matches with
`re.match` (anchored at the start). Every pattern therefore starts with an
optional `(?:model\\.)?` and is `$`-anchored so it matches exactly one tensor.
"""

from __future__ import annotations

import jax
from jax import numpy as jnp
from tunix.models import safetensors_loader
from tunix.models.gemma4 import vision_real


# Per-layer linears that are bias-free and stored as `...{name}.linear.weight`
# (torch [out, in]); nnx holds `...{name}.linear.kernel` [in, out] -> transpose.
_ATTN_LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_LINEARS = ("gate_proj", "up_proj", "down_proj")
# Scaled RMSNorms (q_norm/k_norm + the four sandwich norms). v_norm is unscaled
# (no weight in the checkpoint) and the projector pre-norm is unscaled too.
_ATTN_NORMS = ("q_norm", "k_norm")
_LAYER_NORMS = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
)

_TRANSPOSE = ((1, 0), None)  # (permute, reshape) for a 2D weight.
_IDENTITY = None


def vision_key_mapping(config: vision_real.Gemma4VisionConfig):
  """Returns {torch_key_regex: (nnx_key_repl, transform)} for the vision stack."""
  p = r"(?:model\.)?"  # optional checkpoint `model.` prefix
  mapping = {
      # --- patch embedder ---
      p + r"vision_tower\.patch_embedder\.input_proj\.weight$": (
          "vision_tower.patch_embedder.input_proj.kernel",
          _TRANSPOSE,
      ),
      p + r"vision_tower\.patch_embedder\.position_embedding_table$": (
          "vision_tower.patch_embedder.position_embedding_table",
          _IDENTITY,
      ),
      # --- projector (embed_vision); pre-projection norm is unscaled (no key) ---
      p + r"embed_vision\.embedding_projection\.weight$": (
          "embed_vision.embedding_projection.kernel",
          _TRANSPOSE,
      ),
  }
  # --- encoder layers ---
  for name in _ATTN_LINEARS:
    mapping[
        p + rf"vision_tower\.encoder\.layers\.(\d+)\.self_attn\.{name}\.linear\.weight$"
    ] = (rf"vision_tower.encoder.layers.\1.self_attn.{name}.linear.kernel", _TRANSPOSE)
  for name in _ATTN_NORMS:
    mapping[
        p + rf"vision_tower\.encoder\.layers\.(\d+)\.self_attn\.{name}\.weight$"
    ] = (rf"vision_tower.encoder.layers.\1.self_attn.{name}.scale", _IDENTITY)
  for name in _MLP_LINEARS:
    mapping[
        p + rf"vision_tower\.encoder\.layers\.(\d+)\.mlp\.{name}\.linear\.weight$"
    ] = (rf"vision_tower.encoder.layers.\1.mlp.{name}.linear.kernel", _TRANSPOSE)
  for name in _LAYER_NORMS:
    mapping[
        p + rf"vision_tower\.encoder\.layers\.(\d+)\.{name}\.weight$"
    ] = (rf"vision_tower.encoder.layers.\1.{name}.scale", _IDENTITY)
  return mapping


def create_vision_stack_from_safe_tensors(
    file_dir: str,
    config: vision_real.Gemma4VisionConfig,
    text_hidden_size: int,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
):
  """Builds a `Gemma4VisionStack` and loads vision weights from safetensors."""
  return safetensors_loader.load_and_create_model(
      file_dir=file_dir,
      model_class=lambda cfg, rngs: vision_real.Gemma4VisionStack(
          cfg, text_hidden_size, rngs=rngs
      ),
      config=config,
      key_mapping=vision_key_mapping,
      mesh=mesh,
      dtype=dtype,
  )
