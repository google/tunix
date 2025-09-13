"""SigLIP vision encoder (ViT-style) implemented with Flax NNX."""

from __future__ import annotations
import dataclasses
from typing import Tuple

from flax import nnx
import jax
import jax.numpy as jnp
from jax.interpreters import pxla
import jax.sharding as shd
import jaxtyping


def shard(x: jnp.ndarray, s: Tuple[str | None, ...]):
  """Apply named sharding if a mesh is present; no-op on CPU."""
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh.empty or jax.devices()[0].platform == "cpu":
    return x
  return jax.lax.with_sharding_constraint(
      x, shd.NamedSharding(mesh, shd.PartitionSpec(*s))
  )


@dataclasses.dataclass(slots=True, frozen=True)
class ShardingConfig:
  """Sharding configuration for SigLIP encoder."""

  # weight shardings
  patch_kernel_hwci: Tuple[str | None, ...]  # Conv: [H, W, C, D]
  attn_qkvo_dd: Tuple[str | None, ...]  # Linear: [D, D]
  mlp_df: Tuple[str | None, ...]  # Linear: [D, F]
  mlp_fd: Tuple[str | None, ...]  # Linear: [F, D]
  ln_weight: Tuple[str | None, ...]  # LayerNorm scale/bias

  # activations
  act_bnd: Tuple[str | None, ...]  # [B, N, D]
  act_bnf: Tuple[str | None, ...]  # [B, N, F]
  act_bnhd: Tuple[str | None, ...]  # [B, N, H, Dh]

  @staticmethod
  def get_default_sharding(is_sampling: bool = False):
    fsdp = "fsdp" if not is_sampling else None
    return ShardingConfig(
        patch_kernel_hwci=(None, None, None, "tp"),
        attn_qkvo_dd=("tp", fsdp),
        mlp_df=("tp", fsdp),
        mlp_fd=("tp", fsdp),
        ln_weight=("tp",),
        act_bnd=("fsdp", None, None if is_sampling else "tp"),
        act_bnf=("fsdp", None, "tp"),
        act_bnhd=("fsdp", None, "tp", None),
    )


@dataclasses.dataclass(frozen=True)
class SigLIPConfig:
  image_size: int = 224
  patch_size: int = 16
  embed_dim: int = 768
  depth: int = 12
  num_heads: int = 12
  mlp_ratio: float = 4.0
  # NEW: explicit hidden size if provided
  mlp_hidden_dim: int | None = None
  drop_rate: float = 0.0
  attn_drop_rate: float = 0.0
  use_cls_token: bool = False
  use_abs_pos_emb: bool = True
  shd_config: ShardingConfig = ShardingConfig.get_default_sharding()

  @property
  def head_dim(self) -> int:
    if self.embed_dim % self.num_heads != 0:
      raise ValueError("embed_dim must be divisible by num_heads")
    return self.embed_dim // self.num_heads

  @property
  def num_patches(self) -> int:
    g = self.image_size // self.patch_size
    return g * g

  @classmethod
  def so400m_patch14_384(cls):
    return cls(
        image_size=384,
        patch_size=14,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=3.0,  # keep whatever; it’ll be ignored
        mlp_hidden_dim=4304,  # THIS drives the shapes
        use_cls_token=False,
        use_abs_pos_emb=True,
        shd_config=ShardingConfig.get_default_sharding(),
    )


class PatchEmbed(nnx.Module):
  """Patchify with a Conv2D (stride=patch_size), then flatten to tokens."""

  def __init__(self, cfg: SigLIPConfig, *, rngs: nnx.Rngs):
    self.cfg = cfg
    # NNX Conv uses [H,W,C_in,C_out] kernel layout by default.
    self.proj = nnx.Conv(
        in_features=3,
        out_features=cfg.embed_dim,
        kernel_size=(cfg.patch_size, cfg.patch_size),
        strides=(cfg.patch_size, cfg.patch_size),
        padding="VALID",
        kernel_init=nnx.with_partitioning(
            nnx.initializers.lecun_normal(), cfg.shd_config.patch_kernel_hwci
        ),
        bias_init=nnx.initializers.zeros_init(),
        rngs=rngs,
    )

  @jax.named_scope("patch_embed")
  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    # x: [B,H,W,3] -> conv -> [B,H/P,W/P,D] -> [B,N,D]
    x = self.proj(x)
    b, h, w, d = x.shape
    x = x.reshape(b, h * w, d)
    x = shard(x, self.cfg.shd_config.act_bnd)
    return x


class MLP(nnx.Module):
  """Standard ViT MLP block with GELU."""

  def __init__(self, cfg: SigLIPConfig, *, rngs: nnx.Rngs):
    self.cfg = cfg
    hidden = cfg.mlp_hidden_dim or int(cfg.embed_dim * cfg.mlp_ratio)
    self.fc1 = nnx.Linear(
        in_features=cfg.embed_dim,
        out_features=hidden,
        use_bias=True,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(), cfg.shd_config.mlp_df
        ),
        bias_init=nnx.initializers.zeros_init(),
        rngs=rngs,
    )
    self.fc2 = nnx.Linear(
        in_features=hidden,
        out_features=cfg.embed_dim,
        use_bias=True,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(), cfg.shd_config.mlp_fd
        ),
        bias_init=nnx.initializers.zeros_init(),
        rngs=rngs,
    )

  @jax.named_scope("mlp")
  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    h = jax.nn.gelu(self.fc1(x))
    h = shard(h, self.cfg.shd_config.act_bnf)
    return self.fc2(h)


class MultiHeadSelfAttention(nnx.Module):
  """MHA with separate Q/K/V projections and output projection."""

  def __init__(self, cfg: SigLIPConfig, *, rngs: nnx.Rngs):
    self.cfg = cfg
    d = cfg.embed_dim
    self.q = nnx.Linear(
        in_features=d,
        out_features=d,
        use_bias=True,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(), cfg.shd_config.attn_qkvo_dd
        ),
        rngs=rngs,
    )
    self.k = nnx.Linear(
        in_features=d,
        out_features=d,
        use_bias=True,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(), cfg.shd_config.attn_qkvo_dd
        ),
        rngs=rngs,
    )
    self.v = nnx.Linear(
        in_features=d,
        out_features=d,
        use_bias=True,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(), cfg.shd_config.attn_qkvo_dd
        ),
        rngs=rngs,
    )
    self.o = nnx.Linear(
        in_features=d,
        out_features=d,
        use_bias=True,
        kernel_init=nnx.with_partitioning(
            nnx.initializers.xavier_uniform(), cfg.shd_config.attn_qkvo_dd
        ),
        rngs=rngs,
    )
    self.scale = (cfg.head_dim) ** -0.5

  @jax.named_scope("mhsa")
  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    b, n, d = x.shape
    h = self.cfg.num_heads
    dh = self.cfg.head_dim

    q = self.q(x).reshape(b, n, h, dh)
    k = self.k(x).reshape(b, n, h, dh)
    v = self.v(x).reshape(b, n, h, dh)

    q = shard(q, self.cfg.shd_config.act_bnhd)
    k = shard(k, self.cfg.shd_config.act_bnhd)
    v = shard(v, self.cfg.shd_config.act_bnhd)

    attn = jnp.einsum("bnhd,bmhd->bhnm", q * self.scale, k)  # [B,H,N,N]
    attn = jax.nn.softmax(attn, axis=-1)
    out = jnp.einsum("bhnm,bmhd->bnhd", attn, v).reshape(b, n, d)
    out = self.o(out)
    out = shard(out, self.cfg.shd_config.act_bnd)
    return out


class EncoderBlock(nnx.Module):
  """(LN -> MHA -> residual) + (LN -> MLP -> residual)."""

  def __init__(self, cfg: SigLIPConfig, *, rngs: nnx.Rngs):
    self.cfg = cfg
    self.ln1 = nnx.LayerNorm(
        cfg.embed_dim, use_bias=True, param_dtype=jnp.float32, rngs=rngs
    )
    self.attn = MultiHeadSelfAttention(cfg, rngs=rngs)
    self.ln2 = nnx.LayerNorm(
        cfg.embed_dim, use_bias=True, param_dtype=jnp.float32, rngs=rngs
    )
    self.mlp = MLP(cfg, rngs=rngs)

  @jax.named_scope("encoder_block")
  def __call__(self, x: jaxtyping.Array) -> jaxtyping.Array:
    x = x + self.attn(self.ln1(x))
    x = x + self.mlp(self.ln2(x))
    return x


class SigLIPEngine(nnx.Module):

  def __init__(self, cfg: SigLIPConfig, *, rngs: nnx.Rngs):
    self.cfg = cfg
    self.patch = PatchEmbed(cfg, rngs=rngs)
    self.blocks = nnx.List(
        [EncoderBlock(cfg, rngs=rngs) for _ in range(cfg.depth)]
    )
    self.norm = nnx.LayerNorm(
        cfg.embed_dim, use_bias=True, param_dtype=jnp.float32, rngs=rngs
    )

    # Create params only if enabled; do NOT pre-assign None.
    if cfg.use_abs_pos_emb:
      pe_shape = (
          1,
          cfg.num_patches + (1 if cfg.use_cls_token else 0),
          cfg.embed_dim,
      )
      self.pos_embed = nnx.Param(
          jax.random.normal(rngs.params(), pe_shape) * 0.02
      )

    if cfg.use_cls_token:
      self.cls_token = nnx.Param(
          jax.random.normal(rngs.params(), (1, 1, cfg.embed_dim)) * 0.02
      )

  def get_model_input(self):
    """Dummy input (compatible with sharding) — used by Qwix/rollout."""
    b = 2
    return {
        "images": jnp.ones(
            (b, self.cfg.image_size, self.cfg.image_size, 3), jnp.float32
        )
    }

  @jax.named_scope("siglip_encoder")
  def __call__(self, images):
    x = self.patch(images)  # [B, N, D]
    b, n, d = x.shape

    if hasattr(self, "cls_token"):
      cls = jnp.tile(self.cls_token.value, (b, 1, 1))
      x = jnp.concatenate([cls, x], axis=1)

    if hasattr(self, "pos_embed"):
      x = x + self.pos_embed.value[:, : x.shape[1], :]

    for blk in self.blocks:
      x = blk(x)

    x = self.norm(x)

    if hasattr(self, "cls_token"):
      return x[:, 1:, :]
    return x
