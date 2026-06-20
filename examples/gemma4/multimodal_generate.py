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
  cache_dtype = model.text_model.config.dtype  # match computation dtype
  cache = model.text_model.init_cache(B, L + max_new_tokens, cache_dtype)
  attn_mask = model.get_attention_mask(tokens)  # (B, L, L) causal
  logits, new_cache = model(
      tokens, pixel_values, pixel_position_ids,
      cache=cache, attention_mask=attn_mask,
      decode_only_last_token=True,
  )
  return jnp.argmax(logits[:, 0, :], axis=-1), new_cache  # (B,), cache


@functools.partial(nnx.jit, static_argnames=('n_steps',))
def _decode_n_tokens(text_model, initial_token, initial_pos, initial_cache,
                     n_steps):
  """Decode n_steps tokens entirely on-device via lax.scan.

  Calling nnx.jit once per token costs ~55ms Python overhead per step (NNX
  module-tree extraction). lax.scan amortises that cost over all n_steps,
  yielding ~10x higher throughput on A100.

  Args:
    text_model: Gemma4TextModel NNX module (weights are frozen during decode).
    initial_token: (B, 1) int32 — first token to feed (from prefill).
    initial_pos:   (B, 1) int32 — absolute position of initial_token.
    initial_cache: KV-cache dict from _prefill.
    n_steps: (static) number of tokens to generate.

  Returns:
    all_tokens: (n_steps, B) int32 — generated token ids.
    final_cache: updated KV-cache (useful if you want to extend later).
  """
  def _step(carry, _):
    token, pos, cache = carry
    cache_len = next(iter(cache.values()))['v'].shape[1]
    attn_mask = (
        jnp.arange(cache_len)[None, None, :] <= pos
    ).astype(jnp.bool_)  # (B, 1, cache_len)
    logits, new_cache = text_model(
        token, positions=pos, cache=cache,
        attention_mask=attn_mask, decode_only_last_token=True,
    )
    next_token = jnp.argmax(logits[:, 0, :], axis=-1)  # (B,)
    return (next_token[:, None], pos + 1, new_cache), next_token

  (_, _, final_cache), all_tokens = jax.lax.scan(
      _step,
      (initial_token, initial_pos, initial_cache),
      xs=None,
      length=n_steps,
  )
  return all_tokens, final_cache  # all_tokens: (n_steps, B)


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

  first_token = int(next_id_arr[0])
  if first_token in eos_ids:
    return [first_token]

  # ---- Autoregressive decode (lax.scan — no per-step Python overhead) ------
  # All n_decode steps run on-device in a single JIT dispatch, avoiding the
  # ~55ms Python overhead that nnx.jit incurs per call on a 2B model.
  n_decode = max_new_tokens - 1  # first token already came from prefill
  t_decode = time.time()
  print(f'Decoding {n_decode} tokens via lax.scan '
        '(JIT-compiles on first run)...', end=' ', flush=True)

  all_tokens, _ = _decode_n_tokens(
      model.text_model,
      next_id_arr[:, None],              # (B, 1)
      jnp.full((B, 1), L, jnp.int32),   # absolute pos of first decode token
      cache,
      n_decode,
  )
  jax.block_until_ready(all_tokens)

  elapsed = time.time() - t_decode
  tokens_list = all_tokens[:, 0].tolist()  # (n_decode,)
  rate = n_decode / max(elapsed, 1e-6)
  print(f'done ({elapsed:.1f}s, {rate:.1f} tok/s)', flush=True)

  # Assemble output, stopping at first EOS
  generated = [first_token]
  for tok in tokens_list:
    generated.append(tok)
    if tok in eos_ids:
      break

  return generated


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument('--ckpt', required=True,
                  help='Path to local gemma-4-*-it checkpoint directory')
  ap.add_argument('--image', required=True, help='Image file path')
  ap.add_argument('--prompt', default='Describe this image.')
  ap.add_argument('--max-new-tokens', type=int, default=64)
  ap.add_argument('--tp-size', type=int, default=1,
                  help='Tensor-parallel degree (default 1 = single device). '
                       'TP shards each weight across GPUs, but for a model that '
                       'fits on one GPU at batch=1 it is SLOWER than single-GPU '
                       '(per-token all-reduces dominate): measured 46 tok/s at '
                       'TP=4 vs 136 tok/s at TP=1 for Gemma4 2B on A100. Only '
                       'raise this for models too large for one device.')
  args = ap.parse_args()

  # ---- Tensor-parallel mesh ------------------------------------------------
  # Weights carry per-array sharding metadata (ShardingConfig); the safetensors
  # loader reads it via nnx.get_named_sharding(mesh) and device_puts each tensor
  # onto its target shards as it loads (no single-GPU bottleneck). XLA's SPMD
  # partitioner then inserts the cross-device collectives automatically.
  tp = args.tp_size if args.tp_size > 0 else jax.device_count()
  mesh = jax.make_mesh(
      (1, tp), ('fsdp', 'tp'),
      axis_types=(jax.sharding.AxisType.Auto,) * 2,
  )
  print(f'Mesh: {tp}-way tensor parallel over {jax.device_count()} device(s)',
        flush=True)

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
  # Run computation in bfloat16 to match weights and KV cache dtype.
  text_cfg.dtype = jnp.bfloat16
  text_cfg.param_dtype = jnp.bfloat16
  # Inference (Megatron-style) sharding: residual stream replicated, inner
  # attention/FFN dims sharded along 'tp'. is_sampling=True drops the FSDP
  # axis used in training.
  text_cfg.shd_config = model_lib.ShardingConfig.get_default_sharding(
      is_sampling=True
  )

  print('Loading checkpoint (text + vision)...', flush=True)
  t0 = time.time()
  model = mm_lib.create_multimodal_from_safe_tensors(
      args.ckpt, text_cfg, vision_cfg,
      image_token_id=hf_cfg.image_token_id, mesh=mesh, dtype=jnp.bfloat16,
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
  # Collect all EOS token IDs (Gemma4-IT uses both token 1 and token 106 as stop signals)
  raw_eos = tokenizer.eos_token_id
  if isinstance(raw_eos, (list, tuple)):
    eos_ids = set(raw_eos)
  elif raw_eos is not None:
    eos_ids = {raw_eos}
  else:
    eos_ids = {1}
  eot = tokenizer.convert_tokens_to_ids('<end_of_turn>')
  if eot and eot != tokenizer.unk_token_id:
    eos_ids.add(eot)
  print(f'Prompt: {args.prompt!r}  (total prompt length: {tokens.shape[1]} tokens)',
        flush=True)

  # Enter the mesh context so the activation-level shard() constraints inside
  # the model forward resolve against our tensor-parallel mesh.
  with mesh:
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
