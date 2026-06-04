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

"""Stage 4 end-to-end: caption a real image with Gemma 4 multimodal.

Loads a real `google/gemma-4-*-it` checkpoint into `Gemma4Multimodal`, processes
one image into (unpadded) patches, builds a prompt with exactly `num_soft`
image-token placeholders, and greedy-decodes a caption.

Single image, no padding (so #valid-soft-tokens == #placeholders), eager (no
jit, recomputes the full forward per step) -- this is a correctness/caption
demo, not an efficient sampler. Run on a machine with the checkpoint:

    pip install torch 'transformers==5.9.0' safetensors pillow
    python examples/gemma4/multimodal_generate.py \
        --ckpt ~/gemma4-e2b --image cat.jpg --prompt "Describe this image."

The greedy-loop and prompt-building helpers are import-safe and unit-tested in
tests/models/gemma4/multimodal_test.py-adjacent sandbox runs without weights.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def build_image_prompt_tokens(tokenizer, prompt_text, image_token_id, num_soft,
                              bos_id):
  """[BOS] <image_token>*num_soft <prompt tokens>. Returns a (1, L) int array."""
  text_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
  ids = [bos_id] + [image_token_id] * num_soft + list(text_ids)
  return np.asarray([ids], dtype=np.int32)


def greedy_generate(model, tokens, pixel_values, pixel_position_ids,
                    max_new_tokens, eos_ids):
  """Eager greedy decode (full re-forward each step). Returns the new token ids."""
  import jax.numpy as jnp

  tokens = np.asarray(tokens)
  generated = []
  for _ in range(max_new_tokens):
    logits, _ = model(
        jnp.asarray(tokens),
        jnp.asarray(pixel_values),
        jnp.asarray(pixel_position_ids),
    )
    next_id = int(np.asarray(logits[0, -1]).argmax())
    generated.append(next_id)
    if next_id in eos_ids:
      break
    tokens = np.concatenate([tokens, [[next_id]]], axis=1)
  return generated


def _unpadded_patches(proc, image_chw, do_resize=True):
  """Process one image and strip padding so #patches == num_soft * pool^2."""
  px, pos, num_soft = proc([image_chw], do_resize=do_resize)
  n = num_soft[0] * proc.pooling_kernel_size**2
  return px[:, :n], pos[:, :n], num_soft[0]


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--image", required=True)
  ap.add_argument("--prompt", default="Describe this image.")
  ap.add_argument("--max-new-tokens", type=int, default=64)
  args = ap.parse_args()

  try:
    import jax.numpy as jnp
    from PIL import Image
    from transformers import AutoTokenizer, Gemma4Config
    from tunix.models.gemma4 import image_processing as ip
    from tunix.models.gemma4 import model as model_lib
    from tunix.models.gemma4 import multimodal as mm_lib
    from tunix.models.gemma4 import vision_real
  except ImportError as e:
    print(f"ERROR: {e}\nSee the module docstring for required packages.",
          file=sys.stderr)
    sys.exit(2)

  hf_cfg = Gemma4Config.from_pretrained(args.ckpt)
  vc = hf_cfg.vision_config
  rope_theta = (vc.rope_parameters or {}).get("rope_theta", 100.0)
  vision_cfg = vision_real.Gemma4VisionConfig(
      hidden_size=vc.hidden_size, intermediate_size=vc.intermediate_size,
      num_hidden_layers=vc.num_hidden_layers,
      num_attention_heads=vc.num_attention_heads,
      num_key_value_heads=vc.num_key_value_heads, head_dim=vc.head_dim,
      rms_norm_eps=vc.rms_norm_eps, patch_size=vc.patch_size,
      position_embedding_size=vc.position_embedding_size,
      pooling_kernel_size=vc.pooling_kernel_size, rope_theta=rope_theta,
      use_clipped_linears=vc.use_clipped_linears, standardize=vc.standardize,
      param_dtype=jnp.bfloat16, dtype=jnp.bfloat16,
  )
  # Text config: e2b text defaults (adjust if using a different variant).
  text_cfg = model_lib.ModelConfig.gemma4_e2b()

  print("Loading checkpoint (text + vision)...", flush=True)
  model = mm_lib.create_multimodal_from_safe_tensors(
      args.ckpt, text_cfg, vision_cfg,
      image_token_id=hf_cfg.image_token_id, dtype=jnp.bfloat16,
  )
  tokenizer = AutoTokenizer.from_pretrained(args.ckpt)

  proc = ip.Gemma4ImageProcessor(
      patch_size=vc.patch_size, pooling_kernel_size=vc.pooling_kernel_size
  )
  img = np.transpose(np.asarray(Image.open(args.image).convert("RGB")), (2, 0, 1))
  px, pos, num_soft = _unpadded_patches(proc, img)
  print(f"image -> {num_soft} soft tokens")

  bos_id = tokenizer.bos_token_id or 2
  tokens = build_image_prompt_tokens(
      tokenizer, args.prompt, hf_cfg.image_token_id, num_soft, bos_id
  )
  eos_ids = {tokenizer.eos_token_id} if tokenizer.eos_token_id else {1}

  print("Generating...", flush=True)
  out_ids = greedy_generate(
      model, tokens, px, pos, args.max_new_tokens, eos_ids
  )
  print("\n=== caption ===")
  print(tokenizer.decode(out_ids, skip_special_tokens=True))


if __name__ == "__main__":
  main()
