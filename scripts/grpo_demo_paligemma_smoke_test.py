from __future__ import annotations
import argparse
import os
import sys
sys.path.insert(0, "/home/grads/tianjiao/tunix")
os.environ["JAX_PLATFORMS"] = "cpu"


import json
from typing import Tuple, Optional

# --- JAX / NNX ---
import jax
import jax.numpy as jnp
from flax import nnx

# --- Local imports (tunix) ---
# Assumes you run this from repo root or have `pip install -e .`'d it.
from tunix.models.siglip import model as siglip_model
from tunix.models.siglip import params as siglip_params

# Gemma3 is optional in this script; only used if --gemma3_ckpt is passed.
from tunix.models.gemma3 import model as gemma3_model
from tunix.models.gemma3 import params as gemma3_params
from orbax import checkpoint as ocp

from PIL import Image

# --------------------------
# Utils
# --------------------------

def make_mesh_or_cpu(mesh_spec: Optional[Tuple[Tuple[int, ...], Tuple[str, ...]]]) -> Optional[jax.sharding.Mesh]:
  """Create a JAX mesh if feasible; otherwise return None (single-device/CPU)."""
  devices = jax.devices()
  if not devices:
    print("No JAX devices found; running on CPU.", flush=True)
    return None

  if not mesh_spec:
    # Single device or whatever is available — no named sharding.
    print(f"Using {len(devices)} device(s) without a named mesh.", flush=True)
    return None

  shape, names = mesh_spec
  want = 1
  for s in shape:
    want *= s
  if want > len(devices):
    print(f"Requested mesh {shape} requires {want} devices; only {len(devices)} present. "
          "Proceeding without a mesh.", flush=True)
    return None

  mesh = jax.make_mesh(shape, names)
  print(f"Created mesh: shape={shape} names={names}", flush=True)
  return mesh


def load_siglip(siglip_dir: str,
                cfg: Optional[siglip_model.SigLIPConfig],
                mesh: Optional[jax.sharding.Mesh]):
  """Load SigLIP from a local folder. If `cfg` is None, infer from HF folder."""
  if cfg is None:
    # Let params helper infer the config (it checks HF folder config.json).
    cfg = None
    print("Inferring SigLIP config from folder...", flush=True)

  enc = siglip_params.create_model_from_safe_tensors(siglip_dir, cfg, mesh)
  # Build a dummy and sanity forward once we have it:
  enc_input = enc.get_model_input()
  dummy = jax.tree.map(lambda x: jnp.zeros_like(x), enc_input)
  _ = nnx.jit(enc)(**dummy)
  # Report some key bits:
  if cfg is None:
    # Params loader returns a fully constructed model; pull cfg off it.
    # (We stored cfg inside SigLIPEngine as `cfg`.)
    cfg = enc.cfg  # type: ignore[attr-defined]
  print(f"SigLIP loaded. image_size={cfg.image_size}, patch={cfg.patch_size}, "
        f"embed_dim={cfg.embed_dim}, depth={cfg.depth}, heads={cfg.num_heads}", flush=True)
  return enc, cfg


def preprocess_image(path: str, image_size: int) -> jnp.ndarray:
  """Load RGB image -> resize (bilinear) -> float32 [0,1] -> normalize -> NHWC."""
  img = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
  x = jnp.array(img, dtype=jnp.float32) / 255.0  # [H,W,3]
  # SigLIP typically uses CLIP-like normalization; if you have exact stats, plug here.
  mean = jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32)
  std = jnp.array([0.5, 0.5, 0.5], dtype=jnp.float32)
  x = (x - mean) / std
  x = x[None, ...]  # [1,H,W,3]
  return x


def maybe_load_gemma3(ckpt_path: Optional[str],
                      mesh: Optional[jax.sharding.Mesh]) -> Tuple[Optional[nnx.Module], Optional[gemma3_model.Gemma3Config]]:
  """Load Gemma3 from Orbax checkpoint (local path). Returns (model, config) or (None, None)."""
  if ckpt_path is None:
    print("No Gemma3 checkpoint provided; skipping text model.", flush=True)
    return None, None
  if ckpt_path.startswith("gs://"):
    print("Gemma3 checkpoint is on GCS; please copy locally or install gcsfs/fsspec. Skipping.", flush=True)
    return None, None
  if gemma3_model is None or gemma3_params is None or ocp is None:
    print("Gemma3 or Orbax not importable; skipping text model.", flush=True)
    return None, None

  # Pick a small-ish config for a quick debug (change if needed).
  cfg = gemma3_model.Gemma3Config.gemma3_4b()
  abs_model = nnx.eval_shape(lambda: gemma3_model.Gemma3(cfg, rngs=nnx.Rngs(params=0)))

  # Shape+sharding target
  target = nnx.state(abs_model)
  if mesh is not None:
    target = jax.tree.map(
      lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.float32, sharding=s),
      target,
      nnx.get_named_sharding(target, mesh),
    )
  else:
    target = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, jnp.float32), target)

  # Restore
  ckptr = ocp.StandardCheckpointer()
  restored = ckptr.restore(ckpt_path, target=target)

  # Merge graph + params
  model = nnx.merge(*nnx.split(abs_model)[:1], restored)
  print("Gemma3 loaded.", flush=True)
  return model, cfg


def meanpool_and_project(vision_tokens: jnp.ndarray,
                         out_dim: int,
                         rngs: nnx.Rngs) -> Tuple[nnx.Module, jnp.ndarray]:
  """A tiny projection: mean-pool over patches -> Linear to `out_dim`."""
  class _Proj(nnx.Module):
    def __init__(self, in_dim: int, out_dim: int, *, rngs: nnx.Rngs):
      self.proj = nnx.Linear(in_features=in_dim, out_features=out_dim, use_bias=True, rngs=rngs)
    def __call__(self, x):
      return self.proj(x)
  b, n, d = vision_tokens.shape
  pooled = jnp.mean(vision_tokens, axis=1)  # [B,D]
  proj = _Proj(d, out_dim, rngs=rngs)
  out = proj(pooled)
  return proj, out


def main():
  parser = argparse.ArgumentParser(description="Debug VLM pipeline")
  parser.add_argument("--siglip_dir", type=str, default="/home/grads/tianjiao/checkpoints/siglip-so400m-patch14-384",
                      help="Path to SigLIP *.safetensors folder or HF export dir (with model.safetensors).")
  parser.add_argument("--image", type=str, default="/home/grads/tianjiao/sa2pali-jax/images/truck.jpg", help="Path to an RGB image.")
  parser.add_argument("--gemma3_ckpt", type=str, default=None,
                      help="(Optional) Local Orbax checkpoint path for Gemma3 (directory).")
  parser.add_argument("--mesh", type=str, default="", 
                      help='Optional mesh spec JSON, e.g. \'{"shape":[1,2],"names":["fsdp","tp"]}\'')
  parser.add_argument("--siglip_config_json", type=str, default="",
                      help="Optional path to a SigLIP config.json to force config (instead of infer).")
  args = parser.parse_args()

  mesh_spec = None
  if args.mesh:
    try:
      spec = json.loads(args.mesh)
      mesh_spec = (tuple(spec["shape"]), tuple(spec["names"]))
    except Exception as e:
      print(f"Invalid --mesh spec; ignoring. ({e})", flush=True)

  mesh = make_mesh_or_cpu(mesh_spec)

  # If user provided a config.json, load it; else let params infer from HF folder.
  forced_cfg = None
  if args.siglip_config_json:
    with open(args.siglip_config_json, "r") as f:
      cfg_json = json.load(f)
    # Build a config that matches your model.py dataclass fields.
    forced_cfg = siglip_model.SigLIPConfig(
      image_size=cfg_json.get("vision_config", {}).get("image_size", 384),
      patch_size=cfg_json.get("vision_config", {}).get("patch_size", 14),
      embed_dim=cfg_json.get("vision_config", {}).get("hidden_size", 1152),
      depth=cfg_json.get("vision_config", {}).get("num_hidden_layers", 27),
      num_heads=cfg_json.get("vision_config", {}).get("num_attention_heads", 16),
      mlp_ratio=float(cfg_json.get("vision_config", {}).get("mlp_ratio", 4.0)),
      use_cls_token=False,
      use_abs_pos_emb=True,
    )

  # 1) Load SigLIP
  siglip, sigcfg = load_siglip(args.siglip_dir, forced_cfg, mesh)

  # 2) Preprocess image
  image = preprocess_image(args.image, sigcfg.image_size)

  # 3) Run SigLIP forward
  print("Running SigLIP forward...", flush=True)
  vtoks = nnx.jit(siglip)(images=image)  # [B, N, D]
  print(f"SigLIP tokens shape: {tuple(vtoks.shape)}", flush=True)

  # 4) Optional: load Gemma3 and do a tiny bridge
  text_model, text_cfg = maybe_load_gemma3(args.gemma3_ckpt, mesh)

  if text_model is not None and text_cfg is not None:
    print("Projecting vision tokens to text embed dim and running a tiny text forward...", flush=True)
    # Build a simple PaLI-style image prefix embedding:
    proj, img_prefix = meanpool_and_project(vtoks, text_cfg.embed_dim, rngs=nnx.Rngs(params=0))
    print(f"Image prefix shape (B, D_text): {tuple(img_prefix.shape)}", flush=True)

    # Make dummy prompt tokens to test a forward pass
    # NOTE: This doesn’t use a tokenizer to avoid GCS deps; it’s just to confirm
    # the text model forward works with shapes.
    B = vtoks.shape[0]
    T = 16  # tiny dummy prompt length
    dummy_tokens = jnp.ones((B, T), dtype=jnp.int32)
    dummy_pos = jnp.arange(T)[None, :].repeat(B, axis=0)
    attn_mask = jnp.ones((B, T, T), dtype=bool)

    # Run text forward
    logits, _ = nnx.jit(text_model)(
      last_tokens=dummy_tokens,
      positions=dummy_pos,
      cache=None,
      attention_mask=attn_mask,
      output_hidden_states=False,
    )
    print(f"Gemma3 logits shape: {tuple(logits.shape)}  (B, T, vocab)", flush=True)

  print("Done. Everything wired up.", flush=True)


if __name__ == "__main__":
  main()