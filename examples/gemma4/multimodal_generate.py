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

"""End-to-end caption demo for Gemma 4 multimodal (JIT + KV cache).

Loads a real `google/gemma-4-*-it` checkpoint into `Gemma4Multimodal`,
processes one image, and greedy-decodes a caption using a prefill/decode split
with a fixed-size KV cache. JIT-compiled: first call compiles (~1-2 min for
2B), subsequent tokens decode in milliseconds.

Usage:
    source ~/tunix-venv/bin/activate
    python examples/gemma4/multimodal_generate.py \\
        --ckpt ~/gemma4-e2b --image cat.jpg --prompt "Describe this image."
"""

from __future__ import annotations

import argparse
import functools
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx


# ---------------------------------------------------------------------------
# JIT-compiled prefill and decode (module-level so compilation is cached)
# ---------------------------------------------------------------------------

@functools.partial(nnx.jit, static_argnames=('max_new_tokens',))
def _prefill(model, tokens, pixel_values, pixel_position_ids, max_new_tokens):
  """Full prefill: encodes image+text, populates KV cache.

  Returns (next_token_ids (B,), kv_cache).
  """
  B, L = tokens.shape
  cache = model.text_model.init_cache(B, L + max_new_tokens, jnp.bfloat16)
  attn_mask = model.get_attention_mask(tokens)  # (B, L, L) causal
  logits, new_cache = model(
      tokens, pixel_values, pixel_position_ids,
      cache=cache, attention_mask=attn_mask,
      decode_only_last_token=True,
  )
  return jnp.argmax(logits[:, 0, :], axis=-1), new_cache  # (B,), cache


@nnx.jit
def _decode_step(model, token, position, cache):
  """Single cached decode step.

  token: (B, 1) int32, position: (B, 1) int32.
  Calls model.text_model directly — vision stack is not re-run on decode.
  Returns (next_token_ids (B,), updated_kv_cache).
  """
  first_cache = next(iter(cache.values()))
  cache_len = first_cache['v'].shape[1]
  # Causal mask: attend to every position up to and including `position`.
  attn_mask = (
      jnp.arange(cache_len)[None, None, :] <= position
  ).astype(jnp.bool_)  # (B, 1, cache_len)
  logits, new_cache = model.text_model(
      token, positions=position, cache=cache,
      attention_mask=attn_mask, decode_only_last_token=True,
  )
  return jnp.argmax(logits[:, 0, :], axis=-1), new_cache  # (B,), cache


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def build_image_prompt_tokens(tokenizer, prompt_text, image_token_id, num_soft,
                              bos_id):
  """[BOS] <image_token>*num_soft <prompt tokens>. Returns (1, L) int32."""
  text_ids = tokenizer(prompt_text, add_special_tokens=False)['input_ids']
  ids = [bos_id] + [image_token_id] * num_soft + list(text_ids)
  return np.asarray([ids], dtype=np.int32)


def _unpadded_patches(proc, image_chw, do_resize=True):
  """Process one image and strip padding so #patches == num_soft * pool^2."""
  px, pos, num_soft = proc([image_chw], do_resize=do_resize)
  n = num_soft[0] * proc.pooling_kernel_size ** 2
  return px[:, :n], pos[:, :n], num_soft[0]


def greedy_generate(model, tokens, pixel_values, pixel_position_ids,
                    max_new_tokens, eos_ids, tokenizer):
  """JIT-compiled greedy decode with prefill/decode KV-cache split.

  First call JIT-compiles both stages (~1-2 min for 2B on A100).
  Subsequent tokens decode in milliseconds.
  """
  B, L = tokens.shape
  tokens_j = jnp.asarray(tokens)
  px_j = jnp.asarray(pixel_values)
  pos_j = jnp.asarray(pixel_position_ids)

  # ---- Prefill ------------------------------------------------
  t0 = time.time()
  print('Prefill (JIT-compiles on first run — patience)...', end=' ',
        flush=True)
  next_id_arr, cache = _prefill(model, tokens_j, px_j, pos_j, max_new_tokens)
  jax.block_until_ready(next_id_arr)
  print(f'done ({time.time() - t0:.1f}s)', flush=True)

  next_id = int(next_id_arr[0])
  generated = [next_id]
  if next_id in eos_ids:
    return generated

  # ---- Autoregressive decode ----------------------------------
  print('Decoding...', flush=True)
  t_decode = time.time()
  for step in range(1, max_new_tokens):
    cur_pos = L + step - 1  # absolute position of the newly emitted token
    tok = jnp.array([[next_id]], dtype=jnp.int32)
    pos_arr = jnp.array([[cur_pos]], dtype=jnp.int32)

    next_id_arr, cache = _decode_step(model, tok, pos_arr, cache)
    jax.block_until_ready(next_id_arr)
    next_id = int(next_id_arr[0])
    generated.append(next_id)

    if step % 10 == 0 or next_id in eos_ids:
      elapsed = time.time() - t_decode
      rate = step / max(elapsed, 1e-6)
      partial = tokenizer.decode(generated, skip_special_tokens=True)
      print(f'  [{step}/{max_new_tokens}] {rate:.1f} tok/s | {partial[:72]}',
            flush=True)

    if next_id in eos_ids:
      break

  elapsed = time.time() - t_decode
  print(f'Generated {len(generated)} tokens in {elapsed:.1f}s '
        f'({len(generated)/max(elapsed, 1e-6):.1f} tok/s)', flush=True)
  return generated


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--ckpt', required=True,
                  help='Path to local gemma-4-*-it checkpoint directory')
  ap.add_argument('--image', required=True, help='Image file path')
  ap.add_argument('--prompt', default='Describe this image.')
  ap.add_argument('--max-new-tokens', type=int, default=64)
  args = ap.parse_args()

  try:
    from PIL import Image
    from transformers import AutoTokenizer, Gemma4Config
    from tunix.models.gemma4 import image_processing as ip
    from tunix.models.gemma4 import model as model_lib
    from tunix.models.gemma4 import multimodal as mm_lib
    from tunix.models.gemma4 import vision_real
  except ImportError as e:
    print(f'ERROR: {e}\nSee the module docstring for required packages.',
          file=sys.stderr)
    sys.exit(2)

  print('Loading Gemma4Config...', flush=True)
  hf_cfg = Gemma4Config.from_pretrained(args.ckpt)
  vc = hf_cfg.vision_config
  rope_theta = (vc.rope_parameters or {}).get('rope_theta', 100.0)
  vision_cfg = vision_real.Gemma4VisionConfig(
      hidden_size=vc.hidden_size,
      intermediate_size=vc.intermediate_size,
      num_hidden_layers=vc.num_hidden_layers,
      num_attention_heads=vc.num_attention_heads,
      num_key_value_heads=vc.num_key_value_heads,
      head_dim=vc.head_dim,
      rms_norm_eps=vc.rms_norm_eps,
      patch_size=vc.patch_size,
      position_embedding_size=vc.position_embedding_size,
      pooling_kernel_size=vc.pooling_kernel_size,
      rope_theta=rope_theta,
      use_clipped_linears=vc.use_clipped_linears,
      standardize=vc.standardize,
      param_dtype=jnp.bfloat16,
      dtype=jnp.bfloat16,
  )
  text_cfg = model_lib.ModelConfig.gemma4_e2b()

  print('Loading checkpoint (text + vision)...', flush=True)
  t0 = time.time()
  model = mm_lib.create_multimodal_from_safe_tensors(
      args.ckpt, text_cfg, vision_cfg,
      image_token_id=hf_cfg.image_token_id, dtype=jnp.bfloat16,
  )
  print(f'  checkpoint loaded in {time.time() - t0:.1f}s', flush=True)
  tokenizer = AutoTokenizer.from_pretrained(args.ckpt)

  proc = ip.Gemma4ImageProcessor(
      patch_size=vc.patch_size, pooling_kernel_size=vc.pooling_kernel_size
  )
  img = np.transpose(
      np.asarray(Image.open(args.image).convert('RGB')), (2, 0, 1)
  )
  px, pos_ids, num_soft = _unpadded_patches(proc, img)
  print(f'Image: {num_soft} soft tokens ({px.shape[1]} raw patches)',
        flush=True)

  bos_id = tokenizer.bos_token_id or 2
  tokens = build_image_prompt_tokens(
      tokenizer, args.prompt, hf_cfg.image_token_id, num_soft, bos_id
  )
  eos_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id else {1}
  print(f'Prompt: {args.prompt!r}  (total prompt length: {tokens.shape[1]} tokens)',
        flush=True)

  out_ids = greedy_generate(
      model, tokens, px, pos_ids,
      max_new_tokens=args.max_new_tokens,
      eos_ids=eos_ids,
      tokenizer=tokenizer,
  )

  print('\n=== Caption ===')
  print(tokenizer.decode(out_ids, skip_special_tokens=True))


if __name__ == '__main__':
  main()
