# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Layer-by-layer consistency test: JAX Qwen3-VL vs HuggingFace PyTorch.

Loads the same checkpoint into both frameworks, feeds an identical input, and
compares hidden states after every decoder layer as well as the final logits.

Usage::

    # HuggingFace model ID (downloaded automatically):
    python -m tunix.models.qwen3vl.consistency_test --model_id_or_dir Qwen/Qwen3-VL-4B-Instruct

    # Local checkpoint directory:
    python -m tunix.models.qwen3vl.consistency_test --model_id_or_dir /path/to/checkpoint

The script prints a diff table and exits with code 0 if the top-1 prediction
at the last token position matches across both frameworks.
"""

import argparse
import io
import os

import huggingface_hub
import jax
import jax.numpy as jnp
import numpy as np
import torch
from transformers import AutoProcessor
from transformers import Qwen3VLForConditionalGeneration
from tunix.models.qwen3vl import model as model_lib
from tunix.models.qwen3vl import params as params_lib
from tunix.models.qwen3vl.model import make_causal_mask_from_positions

# ---------------------------------------------------------------------------
# Model ID / directory resolution
# ---------------------------------------------------------------------------


def resolve_model_dir(model_id_or_dir: str) -> str:
  """Return a local directory path for the given model ID or local path.

  If ``model_id_or_dir`` is an existing directory it is returned as-is.
  Otherwise it is treated as a HuggingFace Hub repo ID and the snapshot is
  downloaded (or retrieved from the local cache) via ``huggingface_hub``.
  """
  if os.path.isdir(model_id_or_dir):
    return model_id_or_dir
  print(f'Downloading snapshot for "{model_id_or_dir}" from HuggingFace Hub…')
  return huggingface_hub.snapshot_download(model_id_or_dir)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def to_jax(t: torch.Tensor | None) -> jnp.ndarray | None:
  """Convert a PyTorch tensor to a JAX array, preserving dtype."""
  if t is None:
    return None
  dtype_map = {
      torch.bfloat16: jnp.bfloat16,
      torch.float32: jnp.float32,
      torch.int32: jnp.int32,
      torch.int64: jnp.int64,
      torch.bool: jnp.bool_,
  }
  if t.dtype == torch.bool:
    return jnp.array(t.detach().cpu().numpy())
  return jnp.array(t.detach().cpu().float().numpy()).astype(dtype_map[t.dtype])


def to_torch(
    x: jnp.ndarray | None,
    device: torch.device | str = 'cpu',
) -> torch.Tensor | None:
  """Convert a JAX array to a PyTorch tensor, preserving dtype."""
  if x is None:
    return None
  dtype_map = {
      'bfloat16': torch.bfloat16,
      'float32': torch.float32,
      'int32': torch.int32,
      'int64': torch.int64,
  }
  return (
      torch.tensor(np.array(x.astype(jnp.float32)))
      .to(dtype_map.get(x.dtype.name, torch.float32))
      .to(device)
  )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_models(
    model_id_or_dir: str,
    config: model_lib.ModelConfig,
    pt_device: str = 'cuda',
    pt_dtype: torch.dtype = torch.bfloat16,
    jax_dtype: jnp.dtype = jnp.bfloat16,
) -> tuple[model_lib.Qwen3VL, Qwen3VLForConditionalGeneration]:
  """Load JAX and PyTorch models from the same safetensors checkpoint."""
  model_dir = resolve_model_dir(model_id_or_dir)
  pt_model = Qwen3VLForConditionalGeneration.from_pretrained(
      model_dir,
      torch_dtype=pt_dtype,
      device_map=pt_device,
      attn_implementation='eager',
  )
  pt_model.eval()

  with jax.default_device(jax.devices()[0]):
    jax_model = params_lib.create_model_from_safe_tensors(
        model_dir, config, mesh=None, dtype=jax_dtype
    )
  return jax_model, pt_model


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------


def compare_layerwise(
    model_id_or_dir: str,
    config: model_lib.ModelConfig,
    prompt: str = 'The quick brown fox jumps over the lazy dog.',
    pt_device: str = 'cuda',
    dtype: str = 'bfloat16',
    image_url: str | None = None,
) -> bool:
  """Run a layer-by-layer hidden-state comparison.

  Args:
    model_id_or_dir: HuggingFace repo ID (e.g. ``'Qwen/Qwen3-VL-4B-Instruct'``)
      or path to a local safetensors checkpoint directory.
    config: ``ModelConfig`` for the JAX model (must match the checkpoint).
    prompt: Text prompt to use as input.
    pt_device: PyTorch device string (e.g. ``'cuda'`` or ``'cpu'``).
    dtype: Compute dtype, ``'bfloat16'`` or ``'float32'``.
    image_url: Optional path or URL to an image for multimodal testing.

  Returns:
    ``True`` if the top-1 next-token prediction matches across both frameworks.
  """
  pt_dtype = torch.bfloat16 if dtype == 'bfloat16' else torch.float32
  jax_dtype = jnp.bfloat16 if dtype == 'bfloat16' else jnp.float32

  jax_model, pt_model = load_models(
      model_id_or_dir,
      config,
      pt_device=pt_device,
      pt_dtype=pt_dtype,
      jax_dtype=jax_dtype,
  )
  model_dir = resolve_model_dir(model_id_or_dir)

  # ------------------------------------------------------------------
  # Tokenise (with optional image)
  # ------------------------------------------------------------------
  processor = AutoProcessor.from_pretrained(model_dir)

  if image_url is not None:
    from PIL import Image
    import requests

    if os.path.isfile(image_url):
      image = Image.open(image_url).convert('RGB')
    else:
      image = Image.open(
          io.BytesIO(requests.get(image_url, timeout=30).content)
      ).convert('RGB')
    messages = [{
        'role': 'user',
        'content': [
            {'type': 'image', 'image': image_url},
            {'type': 'text', 'text': prompt},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=[image], return_tensors='pt', padding=True
    )
    pixel_values_pt = inputs['pixel_values'].to(pt_device, pt_dtype)
    image_grid_thw_pt = inputs['image_grid_thw'].to(pt_device)
    image_grid_thw_np = np.array(image_grid_thw_pt.cpu())
    mm_token_type_ids_pt = inputs['mm_token_type_ids'].to(pt_device)
    print(f'Image grid (t,h,w): {image_grid_thw_np}')
  else:
    pixel_values_pt = None
    image_grid_thw_pt = None
    image_grid_thw_np = None
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], return_tensors='pt', padding=True)
    # For text-only inputs mm_token_type_ids is all zeros (all text tokens).
    mm_token_type_ids_pt = torch.zeros_like(inputs['input_ids'])

  input_ids_pt = inputs['input_ids'].to(pt_device)  # [1, L]
  attention_mask_pt = inputs['attention_mask'].to(pt_device)  # [1, L]
  mm_token_type_ids_pt = mm_token_type_ids_pt.to(pt_device)
  seq_len = input_ids_pt.shape[1]

  # ------------------------------------------------------------------
  # Build 3D M-RoPE position ids
  # ------------------------------------------------------------------
  # HF transformers >= 5.3.0 requires mm_token_type_ids as a positional arg.
  position_ids_pt, _ = pt_model.model.get_rope_index(
      input_ids_pt,
      mm_token_type_ids_pt,
      image_grid_thw=image_grid_thw_pt,
      video_grid_thw=None,
      attention_mask=attention_mask_pt,
  )
  position_ids_pt = position_ids_pt.to(pt_device)  # [3, B, L]

  # JAX positions: convert from PT.
  positions_jax = to_jax(position_ids_pt)  # [3, B, L]
  input_tokens_jax = to_jax(input_ids_pt.int())  # [B, L]
  attention_mask_jax = to_jax(attention_mask_pt.bool())  # [B, L]

  # The text sub-model lives at pt_model.model.language_model.
  tf_lm = pt_model.model.language_model

  # ------------------------------------------------------------------
  # Embeddings
  # ------------------------------------------------------------------
  x = jax_model.embedder.encode(input_tokens_jax)  # [B, L, D]
  inputs_embeds_pt = pt_model.get_input_embeddings()(input_ids_pt)  # [B, L, D]

  emb_diff = jnp.abs(
      x.astype(jnp.float32) - to_jax(inputs_embeds_pt).astype(jnp.float32)
  ).max()
  print(f'Embedding max diff (pre-injection): {float(emb_diff):.6f}')

  # ------------------------------------------------------------------
  # Vision injection (if image provided)
  # ------------------------------------------------------------------
  visual_mask = None
  deepstack_visual_embeds = None
  visual_pos_masks = None
  vision_embeds = None

  if image_url is not None:
    pixel_values_jax = to_jax(inputs['pixel_values'].to(torch.float32))
    bsz = x.shape[0]
    image_pad_id = jax_model.config.vision_config.image_pad_id

    # JAX vision encoder
    vision_precomputed = jax_model.visual.compute_grid_data(image_grid_thw_np)
    vision_embeds = jax_model.encode_vision(
        pixel_values_jax, vision_precomputed
    )
    vision_embeds = vision_embeds.cast(
        jax_model.config.param_dtype
    ).with_batch_dim(bsz)
    visual_mask = input_tokens_jax == jnp.int32(image_pad_id)

    # PT vision encoder
    image_outputs = pt_model.get_image_features(
        pixel_values_pt, image_grid_thw_pt, return_dict=True
    )
    image_embeds_pt = torch.cat(image_outputs.pooler_output, dim=0).to(
        pt_device, inputs_embeds_pt.dtype
    )
    deepstack_visual_embeds = image_outputs.deepstack_features

    # Compare raw vision encoder outputs
    jax_vis_tokens = vision_embeds.tokens[0]  # [N_vis, D]
    vis_diff = jnp.abs(
        jax_vis_tokens.astype(jnp.float32)
        - to_jax(image_embeds_pt).astype(jnp.float32)
    )
    print(
        f'Vision tokens  max={float(vis_diff.max()):.4f}'
        f'  mean={float(vis_diff.mean()):.4f}'
    )
    for ds_idx, (jax_ds, pt_ds) in enumerate(
        zip(vision_embeds.deepstack, deepstack_visual_embeds)
    ):
      ds_diff = jnp.abs(
          jax_ds.astype(jnp.float32) - to_jax(pt_ds).astype(jnp.float32)
      )
      print(
          f'Deepstack[{ds_idx}]   max={float(ds_diff.max()):.4f}'
          f'  mean={float(ds_diff.mean()):.4f}'
      )

    # Inject vision tokens into embeddings
    def _inject(h, tok, vis):
      num_vis = vis.shape[0]
      pos = jnp.where(
          tok == jnp.int32(image_pad_id), size=num_vis, fill_value=-1
      )[0]
      valid = pos >= 0
      pos = jnp.where(valid, pos, 0)
      updates = jnp.where(valid[:, None], vis.astype(h.dtype), h[pos])
      return h.at[pos].set(updates)

    x = jax.vmap(_inject)(x, input_tokens_jax, vision_embeds.tokens)

    image_mask_pt, _ = pt_model.model.get_placeholder_mask(
        input_ids_pt,
        inputs_embeds=inputs_embeds_pt,
        image_features=image_embeds_pt,
    )
    inputs_embeds_pt = inputs_embeds_pt.masked_scatter(
        image_mask_pt, image_embeds_pt
    )
    visual_pos_masks = image_mask_pt[..., 0]

    emb_diff_post = jnp.abs(
        x.astype(jnp.float32) - to_jax(inputs_embeds_pt).astype(jnp.float32)
    ).max()
    print(f'Embedding max diff (post-injection): {float(emb_diff_post):.6f}')

  # ------------------------------------------------------------------
  # Causal masks
  # ------------------------------------------------------------------
  # JAX boolean causal mask [B, L, L] built from text/temporal position axis.
  text_positions_jax = positions_jax[0]  # [B, L]
  causal_mask_jax = make_causal_mask_from_positions(
      text_positions_jax, attention_mask_jax
  )

  # Convert JAX boolean mask → PyTorch 4-D additive float mask so that
  # both frameworks use *identical* masking regardless of HF's default
  # is_causal=True path.
  causal_mask_4d_pt = (
      torch.tensor(np.array(causal_mask_jax)).unsqueeze(1).to(pt_device)
  )  # [B, 1, L, L]
  explicit_causal_mask_pt = torch.where(
      causal_mask_4d_pt,
      torch.zeros_like(causal_mask_4d_pt, dtype=inputs_embeds_pt.dtype),
      torch.full_like(
          causal_mask_4d_pt, float('-inf'), dtype=inputs_embeds_pt.dtype
      ),
  )

  # ------------------------------------------------------------------
  # Pre-compute rotary embeddings (shared across all PT layers)
  # ------------------------------------------------------------------
  position_embeddings_pt = tf_lm.rotary_emb(inputs_embeds_pt, position_ids_pt)

  # ------------------------------------------------------------------
  # Build deepstack map keyed by layer index (mirrors JAX model __call__)
  # ------------------------------------------------------------------
  deepstack_indexes = (
      jax_model.config.vision_config.deepstack_visual_indexes
      if jax_model.config.vision_config
      else ()
  )
  if deepstack_visual_embeds is not None:
    pt_deepstack_map = dict(zip(deepstack_indexes, deepstack_visual_embeds))
    jax_deepstack_map = dict(zip(deepstack_indexes, vision_embeds.deepstack))
  else:
    pt_deepstack_map = {}
    jax_deepstack_map = {}

  # ------------------------------------------------------------------
  # Layer-by-layer comparison
  # ------------------------------------------------------------------
  pt_hidden = inputs_embeds_pt
  cache_position = torch.arange(seq_len, device=pt_device)
  text_position_ids_pt = position_ids_pt[0]  # [B, L] — text axis for PT layer

  print(
      f'\n{"Layer":>5}  {"max diff":>10}  {"mean diff":>10}'
      f'  {"worst_seq":>9}  {"worst_dim":>9}  {"is_vision":>9}'
  )
  print('-' * 65)

  for layer_idx, (jax_layer, pt_layer) in enumerate(
      zip(jax_model.layers, tf_lm.layers)
  ):
    # JAX forward (no padding mask — Qwen3-VL uses only causal mask)
    _, x = jax_layer(x, positions_jax, None, causal_mask_jax)

    # PyTorch forward with the JAX-derived causal mask for an exact comparison.
    pt_hidden = pt_layer(
        pt_hidden,
        attention_mask=explicit_causal_mask_pt,
        position_ids=text_position_ids_pt,
        past_key_values=None,
        cache_position=cache_position,
        position_embeddings=position_embeddings_pt,
    )

    # Deepstack injection (same layer indices as JAX model __call__)
    if layer_idx in jax_deepstack_map and visual_mask is not None:
      x = jax_model._apply_deepstack(
          x, visual_mask, jax_deepstack_map[layer_idx]
      )
    if layer_idx in pt_deepstack_map and visual_pos_masks is not None:
      pt_hidden = tf_lm._deepstack_process(
          pt_hidden, visual_pos_masks, pt_deepstack_map[layer_idx]
      )

    abs_diff = jnp.abs(
        x.astype(jnp.float32) - to_jax(pt_hidden).astype(jnp.float32)
    )
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())
    worst_seq = int(jnp.argmax(abs_diff.max(axis=-1)))
    worst_dim = int(jnp.argmax(abs_diff[0, worst_seq]))
    is_vision = bool(
        visual_mask is not None and bool(visual_mask[0, worst_seq])
    )
    print(
        f'{layer_idx:>5}  {max_diff:>10.4f}  {mean_diff:>10.6f}'
        f'  {worst_seq:>9}  {worst_dim:>9}  {str(is_vision):>9}'
    )

  # ------------------------------------------------------------------
  # Final norm
  # ------------------------------------------------------------------
  x_normed = jax_model.final_norm(x)
  pt_hidden_normed = tf_lm.norm(pt_hidden)

  norm_diff = jnp.abs(
      x_normed.astype(jnp.float32)
      - to_jax(pt_hidden_normed).astype(jnp.float32)
  ).max()
  print(f'\nFinal norm max diff: {float(norm_diff):.6f}')

  # ------------------------------------------------------------------
  # Logits at the last token position only (avoids [B, L, vocab] OOM).
  # ------------------------------------------------------------------
  last = seq_len - 1
  x_last = x_normed[:, last : last + 1, :]  # [B, 1, D]
  pt_last = pt_hidden_normed[:, last : last + 1, :]  # [B, 1, D]

  # Qwen3-VL uses tied embeddings: lm_head reuses the embedding matrix.
  jax_logits = jax_model.embedder.decode(x_last)  # [B, 1, V]
  pt_logits = pt_model.lm_head(pt_last)  # [B, 1, V]

  logit_diff = jnp.abs(
      jax_logits.astype(jnp.float32) - to_jax(pt_logits).astype(jnp.float32)
  )
  print(
      f'Logits  max={float(logit_diff.max()):.4f}'
      f'  mean={float(logit_diff.mean()):.4f}'
  )

  jax_top5 = jnp.argsort(jax_logits[0, 0])[-5:][::-1].tolist()
  pt_top5 = jnp.argsort(to_jax(pt_logits)[0, 0])[-5:][::-1].tolist()
  print(f'JAX top-5 tokens: {jax_top5}')
  print(f' PT top-5 tokens: {pt_top5}')
  top1_match = jax_top5[0] == pt_top5[0]
  print(f'Top-1 match: {top1_match}')
  return top1_match


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(
    model_id_or_dir: str = 'Qwen/Qwen3-VL-4B-Instruct',
    prompt: str = 'The quick brown fox jumps over the lazy dog.',
    device: str = 'cpu',
    dtype: str = 'bfloat16',
    image_url: str | None = None,
    config: model_lib.ModelConfig | None = None,
) -> bool:
  """Run the layer-by-layer consistency check.

  Can be called directly from a Python console::

      from tunix.models.qwen3vl.consistency_test import main
      main('Qwen/Qwen3-VL-4B-Instruct')                 # downloads from Hub
      main('/path/to/local/Qwen3-VL-4B-Instruct')        # local directory
      main('Qwen/Qwen3-VL-4B-Instruct', image_url='...')  # multimodal

  Or from the command line::

      python -m tunix.models.qwen3vl.consistency_test --model_id_or_dir Qwen/Qwen3-VL-4B-Instruct

  Args:
    model_id_or_dir: HuggingFace repo ID or local checkpoint directory.
    prompt: Text prompt to tokenise.
    device: PyTorch device string (e.g. ``'cuda'`` or ``'cpu'``).
    dtype: Compute dtype, ``'bfloat16'`` or ``'float32'``.
    image_url: If set, run multimodal test with this image (path or URL).
    config: Explicit ``ModelConfig`` to use.  Defaults to ``qwen3vl_4b()``.

  Returns:
    ``True`` if the top-1 next-token prediction matches across both frameworks.
  """
  if config is None:
    config = model_lib.ModelConfig.qwen3vl_4b()
  return compare_layerwise(
      model_id_or_dir=model_id_or_dir,
      config=config,
      prompt=prompt,
      pt_device=device,
      dtype=dtype,
      image_url=image_url,
  )


if __name__ == '__main__' and '__file__' in globals():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      '--model_id_or_dir',
      default='Qwen/Qwen3-VL-4B-Instruct',
      help='HuggingFace repo ID or local checkpoint directory.',
  )
  parser.add_argument(
      '--prompt',
      default='The quick brown fox jumps over the lazy dog.',
      help='Text prompt to tokenise.',
  )
  parser.add_argument(
      '--device',
      default='cpu',
      help='PyTorch device (e.g. "cuda" or "cpu").',
  )
  parser.add_argument(
      '--dtype',
      choices=['bfloat16', 'float32'],
      default='bfloat16',
      help='Compute dtype.',
  )
  parser.add_argument(
      '--image_url',
      default='https://fastly.picsum.photos/id/541/300/200.jpg?hmac=EU9KBKReX22D8zAU9GY1iRAuNDwf5pJa3hyZA2eHiDQ',
      help='Path or URL of an image for multimodal (vision-language) testing.',
  )
  _args = parser.parse_args()
  raise SystemExit(
      0
      if main(
          model_id_or_dir=_args.model_id_or_dir,
          prompt=_args.prompt,
          device=_args.device,
          dtype=_args.dtype,
          image_url=_args.image_url,
      )
      else 1
  )
