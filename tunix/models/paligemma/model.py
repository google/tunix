# PaLI-Gemma wrapper: SigLIP (vision) + projector + Gemma (text)
from __future__ import annotations
import dataclasses
from typing import Tuple, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax.interpreters import pxla
import jax.sharding as shd
import jaxtyping
from typing import Callable
from tunix.models.siglip import model as siglip_lib
from tunix.models.gemma3 import model as gemma_lib  # or tunix.models.gemma.gemma if you prefer Gemma2

def shard(x: jnp.ndarray, s: Tuple[str | None, ...]):
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh.empty or jax.devices()[0].platform == "cpu":
    return x
  return jax.lax.with_sharding_constraint(
      x, shd.NamedSharding(mesh, shd.PartitionSpec(*s))
  )

@dataclasses.dataclass(slots=True, frozen=True)
class ShardingConfig:
  # reuse Gemma’s sharding for dense params/acts; projector acts like FFW (D->D)
  proj_df: Tuple[str | None, ...]
  proj_fd: Tuple[str | None, ...]
  act_btd: Tuple[str | None, ...]  # [B,T,D]

  @staticmethod
  def from_gemma_sharding(gcfg: gemma_lib.ShardingConfig):
    return ShardingConfig(
        proj_df=gcfg.ffw_weight_df,
        proj_fd=gcfg.ffw_weight_fd,
        act_btd=gcfg.act_btd,
    )

# ---- Add these classmethods on PaLIGemmaConfig ----
@dataclasses.dataclass(frozen=True)
class PaLIGemmaConfig:
  vision: siglip_lib.SigLIPConfig
  text: gemma_lib.Gemma3Config
  max_vision_tokens: int = 256
  shd: ShardingConfig = None

  def with_sharding(self):
    return dataclasses.replace(self, shd=ShardingConfig.from_gemma_sharding(self.text.shd_config))

  # NEW: build from explicit component configs you already have
  @classmethod
  def from_components(cls, vision_cfg, text_cfg, max_vision_tokens=256):
    return cls(vision=vision_cfg, text=text_cfg, max_vision_tokens=max_vision_tokens).with_sharding()

  # NEW: a simple preset factory so your script can call a single line
  @classmethod
  def from_presets(
      cls,
      vision_factory: Callable[[], siglip_lib.SigLIPConfig] = siglip_lib.SigLIPConfig.so400m_patch14_384,
      text_factory: Callable[[], gemma_lib.Gemma3Config] = None,
      max_vision_tokens: int = 256,
  ) -> "PaLIGemmaConfig":
    """
    vision_factory: a zero-arg callable returning SigLIPConfig (defaults to so400m_patch14_384)
    text_factory: a zero-arg callable returning Gemma3Config (must be provided)
    """
    if text_factory is None:
      raise ValueError("You must pass a Gemma3 text_factory, e.g., gemma_lib.Gemma3Config.gemma3_4b_it")
    vcfg = vision_factory()
    tcfg = text_factory()
    return cls.from_components(vcfg, tcfg, max_vision_tokens=max_vision_tokens)

class VisionProjector(nnx.Module):
  """Project SigLIP tokens D_v -> D_text."""
  def __init__(self, d_in: int, d_out: int, *, rngs: nnx.Rngs, shd: ShardingConfig):
    self.proj = nnx.Linear(
        in_features=d_in, out_features=d_out, use_bias=True, rngs=rngs,
        kernel_init=nnx.with_partitioning(nnx.initializers.xavier_uniform(), shd.proj_df),
        bias_init=nnx.initializers.zeros_init(),
    )
    self._shd = shd

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    y = self.proj(x)
    return shard(y, self._shd.act_btd)

class PaLIGemma(nnx.Module):
  """PaLI-Gemma: concatenate projected vision tokens with Gemma text embeddings and run Gemma blocks."""
  def __init__(self, cfg: PaLIGemmaConfig, *, rngs: nnx.Rngs):
    cfg = cfg.with_sharding()
    self.cfg = cfg
    self.vision = siglip_lib.SigLIPEngine(cfg.vision, rngs=rngs)
    self.projector = VisionProjector(
        d_in=cfg.vision.embed_dim, d_out=cfg.text.embed_dim, rngs=rngs, shd=cfg.shd
    )
    # build a text model; we will use its embedder + blocks + final_norm + decode
    self.text = gemma_lib.Gemma3(cfg.text, rngs=rngs)

  def get_model_input(self):
    b = 2
    t = 16
    img = self.cfg.vision.image_size
    return {
        "images": jnp.ones((b, img, img, 3), jnp.float32),
        "input_tokens": jnp.ones((b, t), jnp.int32),
        "positions": jnp.arange(t, dtype=jnp.int32)[None, :].repeat(b, 0),
        "cache": None,
        "attention_mask": jnp.ones((b, 1, self.cfg.max_vision_tokens + t), jnp.bool_),
    }

  def _build_causal_mask(self, b:int, L:int) -> jnp.ndarray:
    # [B, L, L] lower-triangular True
    i = jnp.arange(L)[:, None]
    j = jnp.arange(L)[None, :]
    m = (i >= j)[None, :, :].repeat(b, 0)
    return m

  def __call__(
      self,
      images: jnp.ndarray,               # [B,H,W,3] float32 preprocessed
      input_tokens: jnp.ndarray,         # [B,T]
      positions: jnp.ndarray | None,     # [B,T] (text positions)
      cache: gemma_lib.Cache | None,     # usually None for training
      attention_mask: jnp.ndarray | None # [B, Lq, Lk] if supplied
  ) -> tuple[jnp.ndarray, Optional[gemma_lib.Cache]]:
    # 1) vision tokens
    v = self.vision(images)                               # [B, N_v, D_v]
    v = v[:, : self.cfg.max_vision_tokens]                # clip
    v = self.projector(v)                                 # [B, N_v, D_text]

    # 2) text embeddings
    x_txt = self.text.embedder.encode(input_tokens)       # [B, T, D_text]

    # 3) concat sequence
    x = jnp.concatenate([v, x_txt], axis=1)               # [B, N_v+T, D]
    B, L, _ = x.shape

    # 4) build positions for combined seq (Gemma3 uses RoPE per layer via positions)
    if positions is None:
      positions = jnp.arange(L - input_tokens.shape[1], L, dtype=jnp.int32)[None, :].repeat(B, 0)
    pos_vis = jnp.arange(0, L - input_tokens.shape[1], dtype=jnp.int32)[None, :].repeat(B, 0)
    pos_all = jnp.concatenate([pos_vis, positions], axis=1)  # [B, L]

    # 5) attention mask (causal over combined sequence) if not provided
    if attention_mask is None:
      attn = self._build_causal_mask(B, L)
    else:
      # ensure shape [B, L, L]
      attn = attention_mask
    # Gemma forward expects [B, L, L’] but without cache L’==L; expand head/broadcast happens in-layer
    # Run blocks manually to inject our combined hidden states
    new_cache = None if cache is None else {}
    h = x
    for i, layer in enumerate(self.text.layers):
      layer_name = f"layer_{i}"
      layer_cache = cache[layer_name] if cache else None
      layer_cache, h = layer(h, pos_all, layer_cache, attn)
      if cache is not None:
        new_cache[layer_name] = layer_cache

    h = self.text.final_norm(h)
    logits = self.text.embedder.decode(h)  # [B, L, V]
    return logits, new_cache