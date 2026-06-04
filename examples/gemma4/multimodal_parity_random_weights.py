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

"""Checkpoint-free full-model parity: HF Gemma4ForConditionalGeneration vs
Tunix `Gemma4Multimodal`.

Builds a tiny random HF model (vision + text + projector + LM head), saves its
state dict as safetensors, loads into Tunix via
`create_multimodal_from_safe_tensors`, feeds identical (tokens, image) inputs,
and diffs per-position logits in fp32. Also captures the PLE values handed to
the text model on each side and compares them.

How to read the output (see docs/gemma4_vision_port.md for the long version):

  * BOS position SHOULD be exact (~1e-7). It validates the embedding +
    first-position attention + MLP path.
  * If image positions diverge but the PLE-y-embed diff is zero, the divergence
    is in the merge / per-layer-projection-of-merged-embeds path.
  * If text positions diverge AND a pure-text parity (no image) also diverges,
    you've hit the pre-existing text-model divergence in `tunix.models.gemma4`
    -- out of scope for the vision PR.

Run on any machine with torch + transformers (no weights, no GPU needed):

    pip install torch 'transformers>=5.9' safetensors
    python examples/gemma4/multimodal_parity_random_weights.py
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  try:
    import numpy as np
    import torch
    from safetensors.numpy import save_file
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4Config, Gemma4TextConfig, Gemma4VisionConfig)
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4ForConditionalGeneration)
  except ImportError as e:
    print(f"ERROR: {e}\nRun: pip install torch 'transformers>=5.9' safetensors",
          file=sys.stderr)
    sys.exit(2)

  import jax.numpy as jnp
  from tunix.models.gemma4 import model as tml
  from tunix.models.gemma4 import multimodal as mm
  from tunix.models.gemma4 import vision_real

  PAD, BOS, IMG_TOK, PER = 0, 2, 5, 8
  torch.manual_seed(args.seed)
  tc = Gemma4TextConfig(
      vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=1,
      global_head_dim=16, attention_k_eq_v=False,
      num_attention_heads=4, num_key_value_heads=2, head_dim=16,
      hidden_size_per_layer_input=PER, vocab_size_per_layer_input=64,
      num_kv_shared_layers=0, final_logit_softcapping=None,
      pad_token_id=PAD, bos_token_id=BOS,
  )
  vc = Gemma4VisionConfig(
      hidden_size=32, intermediate_size=64, num_hidden_layers=1,
      num_attention_heads=4, num_key_value_heads=4, head_dim=8,
      patch_size=4, position_embedding_size=64, pooling_kernel_size=2,
  )
  cfg = Gemma4Config(text_config=tc, vision_config=vc, audio_config=None,
                     image_token_id=IMG_TOK)
  hf = Gemma4ForConditionalGeneration(cfg).eval()

  # Save HF state dict as safetensors with real key names.
  d = tempfile.mkdtemp()
  save_file({k: v.detach().to(torch.float32).numpy()
             for k, v in hf.state_dict().items()},
            os.path.join(d, "model.safetensors"))

  # Load into Tunix.
  text_cfg = tml.ModelConfig.gemma4_e2b()
  text_cfg.num_layers = 1; text_cfg.num_embed = 64; text_cfg.embed_dim = 64
  text_cfg.hidden_dim = 128; text_cfg.num_heads = 4; text_cfg.num_kv_heads = 2
  text_cfg.head_dim = 16; text_cfg.per_layer_input_dim = PER
  text_cfg.frac_shared_layers = 0.0; text_cfg.final_logit_softcap = None
  text_cfg.param_dtype = jnp.float32; text_cfg.dtype = jnp.float32

  vis_cfg = vision_real.Gemma4VisionConfig(
      hidden_size=32, intermediate_size=64, num_hidden_layers=1,
      num_attention_heads=4, num_key_value_heads=4, head_dim=8, patch_size=4,
      position_embedding_size=64, pooling_kernel_size=2,
      param_dtype=jnp.float32, dtype=jnp.float32,
  )
  model = mm.create_multimodal_from_safe_tensors(
      d, text_cfg, vis_cfg,
      image_token_id=IMG_TOK, pad_token_id=PAD, dtype=jnp.float32,
  )

  # Inputs: BOS + 4 image placeholders + 3 text tokens; 4x4 patch grid -> 4 soft tokens.
  tokens = np.array([[BOS, IMG_TOK, IMG_TOK, IMG_TOK, IMG_TOK, 10, 11, 12]],
                    dtype=np.int64)
  side = 4
  xs, ys = np.meshgrid(np.arange(side), np.arange(side), indexing="xy")
  pos = np.stack([xs.reshape(-1), ys.reshape(-1)], -1)[None].astype(np.int64)
  px = np.random.default_rng(args.seed + 1).random(
      (1, side * side, 3 * vc.patch_size**2), dtype=np.float32)

  # Capture HF's per_layer_inputs (the y_embed handed to language_model).
  ple_taps = {}
  def hook(module, args_, kwargs_):
    ple_taps["per_layer_inputs"] = kwargs_.get("per_layer_inputs")
  hf.model.language_model.register_forward_pre_hook(hook, with_kwargs=True)

  with torch.no_grad():
    hf_out = hf(input_ids=torch.from_numpy(tokens),
                pixel_values=torch.from_numpy(px),
                image_position_ids=torch.from_numpy(pos))
    hf_logits = hf_out.logits.float().numpy()
  hf_ple = ple_taps["per_layer_inputs"].float().numpy()

  # Tunix forward + PLE-y-embed for direct apples-to-apples comparison.
  tk_j = jnp.asarray(tokens.astype(np.int32))
  llm_tokens = jnp.where(tk_j == IMG_TOK, PAD, tk_j)
  emb_tab = model.text_model.embedder.per_layer_input_embedding.value
  tunix_y = np.asarray(
      (emb_tab[llm_tokens]
       * jnp.sqrt(text_cfg.per_layer_input_dim).astype(emb_tab.dtype)
      ).astype(jnp.float32))

  tunix_logits = np.asarray(
      model(tk_j, jnp.asarray(px), jnp.asarray(pos.astype(np.int32)))[0]
      .astype(jnp.float32))

  print("== PLE y_embed (HF.per_layer_inputs at language_model boundary) ==")
  d_ple = np.abs(hf_ple - tunix_y)
  print(f"  max={d_ple.max():.3e}  mean={d_ple.mean():.3e}    "
        f"-> {'OK (bit-exact)' if d_ple.max() < 1e-5 else 'DIVERGENT'}")

  print("\n== logits per position ==")
  for i in range(tokens.shape[1]):
    di = np.abs(hf_logits[0, i] - tunix_logits[0, i])
    tag = "IMG" if tokens[0, i] == IMG_TOK else ("TXT" if i > 4 else "BOS")
    print(f"  pos {i} ({tag}, tok={tokens[0,i]:3d})  max={di.max():.3e}  "
          f"mean={di.mean():.3e}")

  diff = np.abs(hf_logits - tunix_logits)
  print(f"\noverall logit diff:  max={diff.max():.3e}  mean={diff.mean():.3e}")
  print(
      "\nNote: BOS (pos 0) parity isolates the text-model basics; non-zero diff\n"
      "at later positions that also appears in pure-text parity is a\n"
      "pre-existing Tunix-vs-HF text-model issue (out of scope for the vision\n"
      "port).")


if __name__ == "__main__":
  main()
