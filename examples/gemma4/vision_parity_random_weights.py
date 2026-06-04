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

"""Checkpoint-free numeric parity for the gemma4_vision JAX port.

Unlike `vision_parity_check.py` (which needs a real checkpoint), this builds a
small HF `Gemma4VisionModel` + `Gemma4MultimodalEmbedder` with RANDOM weights,
serializes them to a temporary safetensors file with real checkpoint key names,
loads them into the JAX `Gemma4VisionStack` via the production loader, and
compares per-layer activations. It therefore validates the PORT MATH + the key
mapping + the loader in one shot, on any machine with torch + a transformers
that ships Gemma 4 -- no model download required.

Everything runs in fp32 so the tolerance is tight (default 1e-3).

Run:
    pip install torch 'transformers>=5.9'        # must include models/gemma4
    python examples/gemma4/vision_parity_random_weights.py

NOTE: if `transformers` fails to import gemma4 with a `torch._dynamo`
"Duplicate dispatch rule" error, your transformers/torch combo trips a torch
dynamo packaging bug. `pip install 'transformers==5.9.0'` is a known-good combo.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--layers", type=int, default=2)
  ap.add_argument("--tol", type=float, default=1e-3)
  ap.add_argument("--seed", type=int, default=0)
  args = ap.parse_args()

  try:
    import numpy as np
    import torch
    from safetensors.numpy import save_file
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4TextConfig,
        Gemma4VisionConfig,
    )
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4MultimodalEmbedder,
        Gemma4VisionModel,
    )
  except ImportError as e:
    print(
        f"ERROR: {e}\nRun: pip install torch 'transformers>=5.9' safetensors\n"
        "(If gemma4 import hits a torch._dynamo 'Duplicate dispatch rule' error, "
        "use transformers==5.9.0.)",
        file=sys.stderr,
    )
    sys.exit(2)

  import jax.numpy as jnp
  from tunix.models.gemma4 import vision_params_safetensors as vp
  from tunix.models.gemma4 import vision_real

  torch.manual_seed(args.seed)
  # Small but structurally identical to the real tower.
  patch = 4
  vc = Gemma4VisionConfig(
      hidden_size=64, intermediate_size=128, num_hidden_layers=args.layers,
      num_attention_heads=4, num_key_value_heads=4, head_dim=16, patch_size=patch,
      position_embedding_size=256, pooling_kernel_size=2, rms_norm_eps=1e-6,
  )
  tc = Gemma4TextConfig(hidden_size=48)
  vm = Gemma4VisionModel(vc).eval()
  ev = Gemma4MultimodalEmbedder(vc, tc).eval()

  # Serialize random weights under real checkpoint key names.
  ckpt = {}
  for k, v in vm.state_dict().items():
    ckpt[f"model.vision_tower.{k}"] = v.detach().to(torch.float32).numpy()
  for k, v in ev.state_dict().items():
    ckpt[f"model.embed_vision.{k}"] = v.detach().to(torch.float32).numpy()
  d = tempfile.mkdtemp()
  save_file(ckpt, os.path.join(d, "model.safetensors"))

  # No-padding square input -> HF gather is identity.
  side = 4
  xs, ys = np.meshgrid(np.arange(side), np.arange(side), indexing="xy")
  pos = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=-1)[None].astype(np.int64)
  px = np.random.default_rng(args.seed + 1).random(
      (1, side * side, 3 * patch**2), dtype=np.float32
  )

  # --- HF reference with per-layer hooks ---
  hf = {}
  vm.patch_embedder.register_forward_hook(
      lambda m, i, o: hf.update(after_patch_embed=o.detach().float().numpy())
  )
  for idx, layer in enumerate(vm.encoder.layers):
    layer.register_forward_hook(
        (lambda j: (lambda m, i, o: hf.update(
            **{f"after_layer_{j:02d}": (o[0] if isinstance(o, tuple) else o).detach().float().numpy()}
        )))(idx)
    )
  with torch.no_grad():
    out = vm(torch.from_numpy(px), torch.from_numpy(pos))
    hf["tower"] = out.last_hidden_state.float().numpy()
    hf["proj"] = ev(out.last_hidden_state).float().numpy()

  # --- JAX port through the production loader ---
  cfg = vision_real.Gemma4VisionConfig(
      hidden_size=64, intermediate_size=128, num_hidden_layers=args.layers,
      num_attention_heads=4, num_key_value_heads=4, head_dim=16, patch_size=patch,
      position_embedding_size=256, pooling_kernel_size=2, rms_norm_eps=1e-6,
      param_dtype=jnp.float32, dtype=jnp.float32,
  )
  stack = vp.create_vision_stack_from_safe_tensors(d, cfg, text_hidden_size=48, dtype=jnp.float32)
  pos_j, px_j = jnp.asarray(pos, jnp.int32), jnp.asarray(px, jnp.float32)
  pad = jnp.all(pos_j == -1, axis=-1)
  valid = jnp.logical_not(pad)

  jx = {}
  pe = stack.vision_tower.patch_embedder(px_j, pos_j, pad)
  jx["after_patch_embed"] = np.asarray(pe.astype(jnp.float32))
  bias = jnp.where(valid[:, None, None, :], 0.0, jnp.finfo(jnp.float32).min)
  cos, sin = stack.vision_tower.encoder.rotary_emb(pos_j)
  x = pe
  for idx, layer in enumerate(stack.vision_tower.encoder.layers):
    x = layer(x, cos, sin, pos_j, bias)
    jx[f"after_layer_{idx:02d}"] = np.asarray(x.astype(jnp.float32))
  output_length = px_j.shape[-2] // (cfg.pooling_kernel_size**2)
  pooled, mask = stack.vision_tower.pooler(x, pos_j, pad, output_length)
  flat = pooled.reshape(-1, pooled.shape[-1])[mask.reshape(-1)]
  jx["tower"] = np.asarray(flat.astype(jnp.float32))
  jx["proj"] = np.asarray(stack.embed_vision(flat).astype(jnp.float32))

  # --- report ---
  print(f"== checkpoint-free parity (fp32, tol={args.tol:.0e}) ==")
  order = (
      ["after_patch_embed"]
      + [f"after_layer_{i:02d}" for i in range(args.layers)]
      + ["tower", "proj"]
  )
  ok = True
  for k in order:
    a, b = hf[k], jx[k]
    m = float(np.abs(a - b).max())
    passed = a.shape == b.shape and m < args.tol
    ok &= passed
    print(f"  {k:20s} {'OK ' if passed else 'FAIL'} max_abs={m:.3e}  shapes hf{a.shape} jx{b.shape}")
  print("\nPARITY PASSED" if ok else "\nPARITY FAILED")
  sys.exit(0 if ok else 1)


if __name__ == "__main__":
  main()
