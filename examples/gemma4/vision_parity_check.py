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

"""Stage 3 — numeric parity harness for the gemma4_vision JAX port.

Compares per-layer activations between:
  * HF torch reference: ``transformers.models.gemma4.modeling_gemma4.Gemma4VisionModel``
    + ``Gemma4MultimodalEmbedder``, loaded from a real ``google/gemma-4-*-it``
    safetensors checkpoint.
  * This port: ``tunix.models.gemma4.vision_real.Gemma4VisionStack``, loaded
    from the same weights via
    ``vision_params_safetensors.create_vision_stack_from_safe_tensors``.

Both sides receive identical synthetic ``pixel_values`` + ``pixel_position_ids``
with NO PADDING, so any divergence is attributable to the port (and not, e.g.,
torchvision-vs-PIL resize). All comparisons are cast to fp32 for clean diffs.

Run:
    pip install torch transformers safetensors
    python examples/gemma4/vision_parity_check.py --ckpt ~/gemma4-e2b

Exit code 0 iff every checkpoint passes (max-abs diff < tolerance).
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np


# bf16 has ~3 decimal digits; after 16 sandwich layers compounded error around
# 1e-2 is normal. Set the per-checkpoint failure threshold accordingly.
_DEFAULT_TOL = 5e-2


def _require_torch() -> bool:
  try:
    import torch  # noqa: F401
    from safetensors.torch import load_file  # noqa: F401
    from transformers import Gemma4Config  # noqa: F401
    from transformers.models.gemma4.modeling_gemma4 import (  # noqa: F401
        Gemma4MultimodalEmbedder,
        Gemma4VisionModel,
    )
    return True
  except ImportError as e:  # pragma: no cover - environment check
    print(
        f"ERROR: missing dependency for the torch reference: {e}\n"
        "Run: pip install torch transformers safetensors",
        file=sys.stderr,
    )
    return False


def make_inputs(num_soft: int, pool_k: int, patch_size: int, seed: int):
  """Build a square, all-valid input grid (no padding)."""
  side_pool = int(round(num_soft**0.5))
  if side_pool * side_pool != num_soft:
    raise ValueError(f"--num-soft must be a perfect square (got {num_soft})")
  side_patch = side_pool * pool_k  # patches per side
  num_patches = side_patch * side_patch
  in_channels = 3 * patch_size * patch_size

  rng = np.random.default_rng(seed)
  pixel_values = rng.random((1, num_patches, in_channels), dtype=np.float32)
  xs, ys = np.meshgrid(
      np.arange(side_patch), np.arange(side_patch), indexing="xy"
  )
  positions = np.stack([xs.reshape(-1), ys.reshape(-1)], axis=-1)[None]
  return pixel_values, positions.astype(np.int64)


# ---------------------------------------------------------------------------
# Torch reference
# ---------------------------------------------------------------------------
def torch_reference(ckpt_dir: str, pixel_values, pixel_position_ids):
  """Loads vision + embed_vision from safetensors and runs a hooked forward."""
  import torch
  from safetensors.torch import load_file
  from transformers import Gemma4Config
  from transformers.models.gemma4.modeling_gemma4 import (
      Gemma4MultimodalEmbedder,
      Gemma4VisionModel,
  )

  ckpt = os.path.expanduser(ckpt_dir)
  cfg = Gemma4Config.from_pretrained(ckpt)
  vision_model = Gemma4VisionModel(cfg.vision_config).eval()
  embed_vision = Gemma4MultimodalEmbedder(cfg.vision_config, cfg.text_config).eval()

  # Selective state-dict load: only model.vision_tower.* and model.embed_vision.*.
  combined = {}
  for fn in sorted(glob.glob(os.path.join(ckpt, "*.safetensors"))):
    combined.update(load_file(fn))
  v_prefix = "model.vision_tower."
  e_prefix = "model.embed_vision."
  vision_state = {
      k[len(v_prefix):]: v for k, v in combined.items() if k.startswith(v_prefix)
  }
  embed_state = {
      k[len(e_prefix):]: v for k, v in combined.items() if k.startswith(e_prefix)
  }
  missing_v, unexpected_v = vision_model.load_state_dict(vision_state, strict=False)
  missing_e, unexpected_e = embed_vision.load_state_dict(embed_state, strict=False)
  if missing_v or unexpected_v:
    # `_extra_state` / clip-buffer ±inf inits and config flags can show up here;
    # surface counts but don't bail unless real weights are missing.
    print(
        f"  [hf] vision_tower: missing={len(missing_v)} unexpected={len(unexpected_v)}",
        file=sys.stderr,
    )
  if missing_e or unexpected_e:
    print(
        f"  [hf] embed_vision: missing={len(missing_e)} unexpected={len(unexpected_e)}",
        file=sys.stderr,
    )

  param_dtype = next(vision_model.parameters()).dtype
  taps: dict[str, np.ndarray] = {}

  def to_np(t):
    return t.detach().to(torch.float32).cpu().numpy()

  def hook(name):
    def fn(module, inp, out):
      t = out
      if isinstance(out, tuple):
        t = out[0]
      if hasattr(t, "last_hidden_state"):
        t = t.last_hidden_state
      taps[name] = to_np(t)
    return fn

  vision_model.patch_embedder.register_forward_hook(hook("after_patch_embed"))
  for i, layer in enumerate(vision_model.encoder.layers):
    layer.register_forward_hook(hook(f"after_layer_{i:02d}"))

  def pool_hook(module, inp, out):
    pooled, mask = out
    taps["after_pool"] = to_np(pooled)
    taps["after_pool_mask"] = mask.detach().cpu().numpy()
  vision_model.pooler.register_forward_hook(pool_hook)

  with torch.no_grad():
    px_t = torch.from_numpy(pixel_values).to(param_dtype)
    pos_t = torch.from_numpy(pixel_position_ids).long()
    vis_out = vision_model(px_t, pos_t)
    taps["after_vision_tower_gathered"] = to_np(vis_out.last_hidden_state)
    proj = embed_vision(vis_out.last_hidden_state)
    taps["after_projector"] = to_np(proj)
  return taps


# ---------------------------------------------------------------------------
# JAX port path
# ---------------------------------------------------------------------------
def jax_port(ckpt_dir: str, pixel_values, pixel_position_ids):
  """Instrumented forward through the ported Gemma4VisionStack."""
  import jax
  import jax.numpy as jnp
  from transformers import Gemma4Config
  from tunix.models.gemma4 import vision_params_safetensors as vp
  from tunix.models.gemma4 import vision_real

  ckpt = os.path.expanduser(ckpt_dir)
  hf_cfg = Gemma4Config.from_pretrained(ckpt)
  vc = hf_cfg.vision_config
  rope_theta = (vc.rope_parameters or {}).get("rope_theta", 100.0)

  cfg = vision_real.Gemma4VisionConfig(
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
  stack = vp.create_vision_stack_from_safe_tensors(
      file_dir=ckpt,
      config=cfg,
      text_hidden_size=hf_cfg.text_config.hidden_size,
      dtype=jnp.bfloat16,
  )

  taps: dict[str, np.ndarray] = {}
  px = jnp.asarray(pixel_values, dtype=jnp.bfloat16)
  pos = jnp.asarray(pixel_position_ids, dtype=jnp.int32)
  padding_positions = jnp.all(pos == -1, axis=-1)
  valid_mask = jnp.logical_not(padding_positions)

  def cap(name, arr):
    taps[name] = np.asarray(arr.astype(jnp.float32))

  # 1. patch embedder
  pe = stack.vision_tower.patch_embedder(px, pos, padding_positions)
  cap("after_patch_embed", pe)

  # 2. encoder, layer-by-layer
  attn_bias = jnp.where(
      valid_mask[:, None, None, :], 0.0, jnp.finfo(jnp.float32).min
  )
  cos, sin = stack.vision_tower.encoder.rotary_emb(pos)
  x = pe
  for i, layer in enumerate(stack.vision_tower.encoder.layers):
    x = layer(x, cos, sin, pos, attn_bias)
    cap(f"after_layer_{i:02d}", x)

  # 3. pooler
  pk = cfg.pooling_kernel_size
  output_length = px.shape[-2] // (pk * pk)
  pooled, mask = stack.vision_tower.pooler(
      x, pos, padding_positions, output_length
  )
  cap("after_pool", pooled)
  taps["after_pool_mask"] = np.asarray(mask)

  # 4. gather valid rows (mirrors HF `hidden_states[pooler_mask]`).
  flat_pool = pooled.reshape(-1, pooled.shape[-1])
  flat_mask = mask.reshape(-1)
  cap("after_vision_tower_gathered", flat_pool[flat_mask])

  # 5. projector (embed_vision) on the gathered tensor, to match HF call site.
  proj = stack.embed_vision(flat_pool[flat_mask])
  cap("after_projector", proj)
  return taps


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _diff_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
  d = np.abs(a.astype(np.float32) - b.astype(np.float32))
  denom = np.maximum(np.abs(a.astype(np.float32)), 1e-6)
  return float(d.max()), float(d.mean()), float(np.median(d / denom))


def report(name: str, hf: np.ndarray, jx: np.ndarray, tol: float) -> bool:
  if hf.shape != jx.shape:
    print(f"  {name:36s}  SHAPE MISMATCH  hf={hf.shape}  jx={jx.shape}")
    return False
  max_abs, mean_abs, median_rel = _diff_stats(hf, jx)
  ok = max_abs < tol
  print(
      f"  {name:36s} {'OK ' if ok else 'FAIL'}"
      f"  max_abs={max_abs:.3e}  mean_abs={mean_abs:.3e}"
      f"  median_rel={median_rel:.3e}"
  )
  return ok


def main():
  ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  ap.add_argument("--ckpt", required=True, help="Gemma-4 checkpoint directory")
  ap.add_argument(
      "--num-soft",
      type=int,
      default=4,
      help="Soft tokens per image (perfect square; 4 means 6x6 patches).",
  )
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument(
      "--tol", type=float, default=_DEFAULT_TOL, help="Per-checkpoint max-abs threshold."
  )
  args = ap.parse_args()

  if not _require_torch():
    sys.exit(2)

  from transformers import Gemma4Config

  hf_cfg = Gemma4Config.from_pretrained(os.path.expanduser(args.ckpt))
  pool_k = hf_cfg.vision_config.pooling_kernel_size
  patch = hf_cfg.vision_config.patch_size
  print(
      f"Inputs: num_soft={args.num_soft}, pool_k={pool_k}, patch={patch}, "
      f"layers={hf_cfg.vision_config.num_hidden_layers}, seed={args.seed}"
  )
  px, pos = make_inputs(args.num_soft, pool_k, patch, args.seed)
  print(f"  pixel_values={px.shape}  pixel_position_ids={pos.shape}\n")

  print("== HF torch reference ==", flush=True)
  hf = torch_reference(args.ckpt, px, pos)
  print(f"   captured {len(hf)} taps\n")

  print("== JAX port ==", flush=True)
  jx = jax_port(args.ckpt, px, pos)
  print(f"   captured {len(jx)} taps\n")

  print(f"== Parity report (tol={args.tol:.0e}, fp32 cast) ==")
  passed = True
  ordered = (
      ["after_patch_embed"]
      + [f"after_layer_{i:02d}" for i in range(hf_cfg.vision_config.num_hidden_layers)]
      + ["after_pool", "after_vision_tower_gathered", "after_projector"]
  )
  for k in ordered:
    if k not in hf or k not in jx:
      print(f"  {k:36s}  MISSING from {'hf' if k not in hf else 'jx'}")
      passed = False
      continue
    passed &= report(k, hf[k], jx[k], args.tol)

  # Mask sanity (must be identical):
  if "after_pool_mask" in hf and "after_pool_mask" in jx:
    same = np.array_equal(hf["after_pool_mask"].reshape(-1), jx["after_pool_mask"].reshape(-1))
    print(f"  {'pool_mask':36s} {'OK ' if same else 'FAIL'}  identical={same}")
    passed &= same

  print()
  if passed:
    print("PARITY PASSED  -- the JAX port matches HF within tolerance.")
    sys.exit(0)
  else:
    print("PARITY FAILED  -- see the first FAIL row to localize the bug.")
    sys.exit(1)


if __name__ == "__main__":
  main()
