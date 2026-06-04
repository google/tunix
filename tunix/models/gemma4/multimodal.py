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

"""Stage 4: wire the real gemma4_vision stack into the Gemma 4 text model.

`Gemma4Multimodal` composes the existing text `Gemma4` with the ported
`Gemma4VisionStack` (vision tower + embed_vision projector). The forward:

  1. embed text tokens (scaled, as usual);
  2. run the vision stack to get soft tokens already projected into text space;
  3. scatter the soft tokens into the positions where ``tokens == image_token_id``
     (the HF `masked_scatter` equivalent), via ``merge_embeddings``;
  4. run the text transformer on the merged embeddings.

This deliberately wraps `Gemma4` (passing merged embeddings through the new
``input_embeddings`` arg) rather than editing its layer loop, and leaves the
legacy SigLIP scaffolding untouched.

Scope note: the merge currently assumes the number of valid soft tokens equals
the number of ``image_token_id`` placeholders in the sequence (true for a
single, non-padded image, which is what the parity tests cover). Variable-size
images / multiple images per sequence need a gather of valid pooled tokens
first; that is a follow-up.
"""

from __future__ import annotations

import jax
from flax import nnx
import jax.numpy as jnp
import jaxtyping
from tunix.models.gemma3 import merge_embeddings as merge_embeddings_lib
from tunix.models.gemma3 import utils as mm_utils
from tunix.models.gemma4 import model as model_lib
from tunix.models.gemma4 import params_safetensors as text_params
from tunix.models.gemma4 import vision_params_safetensors as vision_params
from tunix.models.gemma4 import vision_real


class Gemma4Multimodal(nnx.Module):
  """Text `Gemma4` + real `Gemma4VisionStack`, merged at image-token positions."""

  def __init__(
      self,
      text_model: model_lib.Gemma4,
      vision_stack: vision_real.Gemma4VisionStack,
      *,
      image_token_id: int,
      pad_token_id: int = 0,
      bidirectional_image_span: bool = False,
  ):
    self.text_model = text_model
    self.vision = vision_stack
    self.image_token_id = image_token_id
    # Per-layer-input embeddings must be computed from pad-substituted
    # tokens/embeddings at image positions (HF Gemma4Model.forward behavior);
    # otherwise PLE leaks vision soft tokens and image_token_id lookups into
    # positions that HF treats as "no language token present".
    self.pad_token_id = pad_token_id
    # Larger Gemma 4 checkpoints (config `use_bidirectional_attention == "vision"`)
    # use the Gemma-3-style bidirectional mask over each image's soft-token span;
    # smaller ones use a plain causal mask. Default matches HF small-model
    # behaviour; flip True for checkpoints that set `use_bidirectional_attention`.
    self.bidirectional_image_span = bidirectional_image_span

  def encode_multimodal_inputs(
      self,
      tokens: jaxtyping.Array,  # (B, L)
      pixel_values: jaxtyping.Array,  # (B, P, 3*patch^2)
      pixel_position_ids: jaxtyping.Array,  # (B, P, 2)
  ) -> jaxtyping.Array:  # (B, L, D)
    """Text embeddings with vision soft tokens scattered into image slots."""
    text_embeddings = self.text_model.embedder.encode(tokens)  # (B, L, D)
    soft_tokens, _mask = self.vision(pixel_values, pixel_position_ids)  # (B, S, D)
    # merge_embeddings expects vision as (B, num_images, tokens_per_image, D).
    return merge_embeddings_lib.merge_embeddings(
        text_embeddings=text_embeddings,
        vision_embeddings=soft_tokens[:, None, :, :],
        mask=(tokens == self.image_token_id),
    )

  def _compute_per_layer_inputs(
      self, tokens: jaxtyping.Array, merged_embeddings: jaxtyping.Array
  ) -> jaxtyping.Array:
    """HF-style PLE: ONLY the token-identity branch is pad-substituted.

    HF `Gemma4Model.forward` calls `get_per_layer_inputs(llm_input_ids, …)`
    which ignores `inputs_embeds` and just looks up `embed_tokens_per_layer`
    on the pad-substituted ids. The context-projection branch (inside
    `project_per_layer_inputs`) then runs on the **merged** `inputs_embeds`
    (vision features at image positions). So we pass merged embeddings to the
    `x` argument and pad-substituted ids to `t`.
    """
    llm_tokens = jnp.where(
        tokens == self.image_token_id, self.pad_token_id, tokens
    )
    return self.text_model.embedder.encode_per_layer_input(
        merged_embeddings, llm_tokens
    )

  def get_attention_mask(
      self,
      tokens: jaxtyping.Array,  # (B, L)
      *,
      inputs_mask: jaxtyping.Array | None = None,
  ):
    """Causal by default; bidirectional over image spans iff requested."""
    placeholder = (
        self.image_token_id if self.bidirectional_image_span else None
    )
    return mm_utils.get_attention_mask(
        tokens, inputs_mask=inputs_mask, token_placeholder_id=placeholder,
    )

  def __call__(
      self,
      tokens: jaxtyping.Array,  # (B, L)
      pixel_values: jaxtyping.Array,  # (B, P, 3*patch^2)
      pixel_position_ids: jaxtyping.Array,  # (B, P, 2)
      positions=None,
      cache=None,
      attention_mask=None,
      decode_only_last_token: bool = False,
  ):
    x = self.encode_multimodal_inputs(tokens, pixel_values, pixel_position_ids)
    if attention_mask is None:
      attention_mask = self.get_attention_mask(tokens)
    ple = None
    if self.text_model.config.per_layer_input_dim > 0:
      ple = self._compute_per_layer_inputs(tokens, x)
    return self.text_model(
        tokens,
        positions=positions,
        cache=cache,
        attention_mask=attention_mask,
        decode_only_last_token=decode_only_last_token,
        input_embeddings=x,
        per_layer_inputs=ple,
    )


def create_multimodal_from_safe_tensors(
    file_dir: str,
    text_config: model_lib.ModelConfig,
    vision_config: vision_real.Gemma4VisionConfig,
    *,
    image_token_id: int,
    pad_token_id: int = 0,
    bidirectional_image_span: bool = False,
    mesh: jax.sharding.Mesh | None = None,
    dtype=None,
) -> Gemma4Multimodal:
  """Loads a real `google/gemma-4-*-it` checkpoint into a `Gemma4Multimodal`.

  The text and vision loaders each read the same directory and skip the other's
  keys (text skips `vision_tower`/`audio_tower`/`embed_*`; vision skips
  `language_model`/`audio_tower`). The vision projector output dim must equal the
  text model's `embed_dim`.
  """
  text_model = text_params.create_model_from_safe_tensors(
      file_dir, text_config, mesh, dtype=dtype
  )
  vision_stack = vision_params.create_vision_stack_from_safe_tensors(
      file_dir, vision_config, text_config.embed_dim, mesh=mesh, dtype=dtype
  )
  return Gemma4Multimodal(
      text_model, vision_stack,
      image_token_id=image_token_id, pad_token_id=pad_token_id,
      bidirectional_image_span=bidirectional_image_span,
  )
