# Copyright ...
# Licensed under the Apache License, Version 2.0

"""Checkpoint loader for SigLIP encoder."""

from __future__ import annotations
import re
from typing import Tuple, Dict, Any

from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
import safetensors.numpy as stnp  # or safetensors.flax if you prefer

from tunix.models.siglip import model as model_lib

import json


def _get_key_and_transform_mapping(cfg: model_lib.SigLIPConfig):
  D = cfg.embed_dim
  F = int(cfg.embed_dim * cfg.mlp_ratio)

  return {
      # Patch projection (Conv2D as linear projection for patches)
      r"vision_model\.embeddings\.patch_embedding\.projection\.weight": (
          "patch.proj.kernel",
          ((2, 3, 1, 0), None),  # [Co,Ci,Kh,Kw] -> [Kh,Kw,Ci,Co]
      ),
      r"vision_model\.embeddings\.patch_embedding\.projection\.bias": (
          "patch.proj.bias",
          (None, None),
      ),

      # Encoder layer norms
      r"vision_model\.encoder\.layers\.([0-9]+)\.layernorm_before\.weight": (
          r"blocks.\1.ln1.scale",
          (None, None),
      ),
      r"vision_model\.encoder\.layers\.([0-9]+)\.layernorm_before\.bias": (
          r"blocks.\1.ln1.bias",
          (None, None),
      ),
      r"vision_model\.encoder\.layers\.([0-9]+)\.layernorm_after\.weight": (
          r"blocks.\1.ln2.scale",
          (None, None),
      ),
      r"vision_model\.encoder\.layers\.([0-9]+)\.layernorm_after\.bias": (
          r"blocks.\1.ln2.bias",
          (None, None),
      ),

      # Attention proj (separate q/k/v/out)
      r"vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
          r"blocks.\1.attn.q.kernel",
          ((1, 0), None),
      ),
      r"vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.q_proj\.bias": (
          r"blocks.\1.attn.q.bias",
          (None, None),
      ),
      r"vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
          r"blocks.\1.attn.k.kernel",
          ((1, 0), None),
      ),
      r"vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.k_proj\.bias": (
          r"blocks.\1.attn.k.bias",
          (None, None),
      ),
      r"vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
          r"blocks.\1.attn.v.kernel",
          ((1, 0), None),
      ),
      r"vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.v_proj\.bias": (
          r"blocks.\1.attn.v.bias",
          (None, None),
      ),
      r"vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.weight": (
          r"blocks.\1.attn.o.kernel",
          ((1, 0), None),
      ),
      r"vision_model\.encoder\.layers\.([0-9]+)\.self_attn\.out_proj\.bias": (
          r"blocks.\1.attn.o.bias",
          (None, None),
      ),

      # MLP (GELU)
      r"vision_model\.encoder\.layers\.([0-9]+)\.mlp\.fc1\.weight": (
          r"blocks.\1.mlp.fc1.kernel",
          ((1, 0), None),
      ),
      r"vision_model\.encoder\.layers\.([0-9]+)\.mlp\.fc1\.bias": (
          r"blocks.\1.mlp.fc1.bias",
          (None, None),
      ),
      r"vision_model\.encoder\.layers\.([0-9]+)\.mlp\.fc2\.weight": (
          r"blocks.\1.mlp.fc2.kernel",
          ((1, 0), None),
      ),
      r"vision_model\.encoder\.layers\.([0-9]+)\.mlp\.fc2\.bias": (
          r"blocks.\1.mlp.fc2.bias",
          (None, None),
      ),

      # Final norm (Transformers often: `vision_model.post_layernorm.*` or `vision_model.layernorm.*`)
      r"vision_model\.(post_layernorm|layernorm)\.weight": ("norm.scale", (None, None)),
      r"vision_model\.(post_layernorm|layernorm)\.bias": ("norm.bias", (None, None)),

      # (Some SigLIP variants may include absolute pos embed; if present and shapes match, map here)
      r"vision_model\.embeddings\.position_embedding(?:\.weight)?": ("pos_embed", (None, None)),
      # If CLS token exists in the checkpoint (many SigLIP variants don’t use it):
      r"vision_model\.embeddings\.cls_token": ("cls_token", (None, None)),
  }


def _torch_key_to_jax_key(mapping, source_key):
  subs = [
      (re.sub(pat, repl, source_key), transform)
      for pat, (repl, transform) in mapping.items()
      if re.match(pat, source_key)
  ]
  if len(subs) != 1:
    raise KeyError(f"Ambiguous or missing mapping for: {source_key} -> {subs}")
  return subs[0]


def _transpose_and_reshape(x, transform):
  if transform is None:
    return x
  permute, reshape = transform
  if permute:
    x = x.transpose(permute)
  if reshape:
    x = x.reshape(reshape)
  return x


def _siglip_cfg_from_hf(dir_path: str) -> model_lib.SigLIPConfig:
  """Read HF config.json (vision_config) and build a SigLIPConfig."""
  cfg_path = epath.Path(dir_path).expanduser() / "config.json"
  data = json.loads(cfg_path.read_text())
  vc = data.get("vision_config", {})  # transformers puts vision params here

  image_size = int(vc.get("image_size", 384))
  patch_size = int(vc.get("patch_size", 14))
  hidden_size = int(vc.get("hidden_size", 768))
  intermediate_size = int(vc.get("intermediate_size", 3072))
  num_layers = int(vc.get("num_hidden_layers", 12))
  num_heads = int(vc.get("num_attention_heads", 12))
  mlp_ratio = float(intermediate_size) / float(hidden_size)

  return model_lib.SigLIPConfig(
      image_size=image_size,
      patch_size=patch_size,
      embed_dim=hidden_size,
      depth=num_layers,
      num_heads=num_heads,
      mlp_ratio=mlp_ratio,
      # keep defaults: abs pos = True, no cls token, default sharding
  )

def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.SigLIPConfig | None = None,
    mesh: jax.sharding.Mesh | None = None,
) -> model_lib.SigLIPEngine:
  """Load SigLIP encoder from a folder of safetensors (HF-style)."""

  files = list(epath.Path(file_dir).expanduser().glob("*.safetensors"))
  if not files:
    raise ValueError(f"No safetensors in {file_dir}")

  # 1) Infer config if not provided
  if config is None:
    config = _siglip_cfg_from_hf(file_dir)

  # 2) Build a CONCRETE model/state (not eval_shape) so params are real arrays
  enc_concrete = model_lib.SigLIPEngine(config, rngs=nnx.Rngs(params=0))
  graph_def, state = nnx.split(enc_concrete)
  state_dict = state.to_pure_dict()

  # Optional: get sharding PartitionSpecs for later
  pspecs = nnx.get_partition_spec(state) if mesh is not None else None

  key_map = _get_key_and_transform_mapping(config)

  def path_to_key(path):
    parts = []
    for p in path:
      parts.append(str(p.key if hasattr(p, "key") else p))
    return ".".join(parts)

  # Helpful diagnostics
  loaded_keys = set()
  missing_param_keys = set()

  for f in files:
    current_file_tensors: Dict[str, jnp.ndarray] = {}
    with stnp.safe_open(f, framework="numpy") as sf:
      for torch_key in sf.keys():
        arr = sf.get_tensor(torch_key)
        try:
          jax_key_mapped, transform = _torch_key_to_jax_key(key_map, torch_key)
        except KeyError:
          # Skip unknown keys (e.g., text tower, optimizer states)
          continue
        arr = _transpose_and_reshape(arr, transform)
        current_file_tensors[jax_key_mapped] = jax.numpy.array(arr)

    def update_tensor(path, param):
      k = path_to_key(path)
      if k in current_file_tensors:
        v = current_file_tensors[k]

        # ---- shape fixups (HF -> NNX) ----
        # pos_embed: [N,D] -> [1,N,D]
        if (
            v.ndim + 1 == param.shape.__len__() and
            getattr(param, "shape", None) is not None and
            param.shape[0] == 1 and tuple(v.shape) == tuple(param.shape[1:])
        ):
          v = v[None, ...]
        # cls_token: [1,D] -> [1,1,D]
        if k.endswith("cls_token") and v.ndim == 2 and len(param.shape) == 3:
          if param.shape[0] == 1 and param.shape[1] == 1 and v.shape[-1] == param.shape[-1]:
            v = v[:, None, :]

        if v.shape != param.shape:
          raise ValueError(f"Shape mismatch for {k}: got {v.shape}, expected {param.shape}")

        loaded_keys.add(k)
        return v
      # Not found in safetensors — keep initialized param
      missing_param_keys.add(k)
      return param

    state_dict = jax.tree.map_with_path(update_tensor, state_dict)

  # Re-merge concrete state
  enc = nnx.merge(graph_def, state_dict)

  # If you want sharding, apply after merge
  if mesh is not None:
    with mesh:
      st = nnx.state(enc)
      st = jax.lax.with_sharding_constraint(st, pspecs)
      nnx.update(enc, st)

  # (Optional) print a brief summary of missing/loaded
  print(f"Loaded {len(loaded_keys)} params; left {len(missing_param_keys)} as initialized.")

  return enc
