"""Numpy toy Qwen3 — faithfully mirrors tunix/models/qwen3/model.py structure.

Purpose (Phase 2, CPU): a small, dependency-light stand-in used ONLY to validate
the logp-diff / per-op attribution HARNESS logic. NOT the real model; real
Qwen3-32B numerics run remotely (Phase 3).

Mirrors (origin/yuxzhang/deepswe-quality-fix:tunix/models/qwen3/model.py):
  - RMSNorm (:392) : x_f32; rms=sqrt(mean(x^2,-1)+eps); w_f32*(x/rms)
  - Attention (:421): QK-norm (q_norm/k_norm), GQA, scale=head_dim^-0.5, RoPE, causal softmax
  - MLP (:983)     : silu(gate(x)) * up(x) -> down   (SwiGLU)
  - block          : pre-norm residual (norm->attn->+res ; norm->mlp->+res); final norm; lm_head
Everything float32 (toy validates LOGIC; bf16/kernel batch-variance is the remote concern).

Activation hooks: forward() returns an ORDERED list of (name, array) at every op
boundary, so the harness can diff two forwards op-by-op and find the first divergence.
A per-op "perturbation" can be injected by name to build the known-answer attribution test.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ToyConfig:
  n_layers: int = 2
  hidden: int = 64
  n_heads: int = 4
  n_kv_heads: int = 2          # GQA: groups = n_heads // n_kv_heads
  head_dim: int = 16
  ffn: int = 128
  vocab: int = 100
  rope_theta: float = 1_000_000.0
  norm_eps: float = 1e-6


def init_weights(cfg: ToyConfig, seed: int = 0) -> dict:
  """Deterministic random weights (numpy) for the toy model."""
  rng = np.random.default_rng(seed)
  def r(*shape, s=0.02):
    return (rng.standard_normal(shape) * s).astype(np.float32)
  H, Hd, Kv = cfg.hidden, cfg.head_dim, cfg.n_kv_heads
  qd, kd = cfg.n_heads * Hd, cfg.n_kv_heads * Hd
  w = {"embed": r(cfg.vocab, H), "final_norm": np.ones(H, np.float32),
       "lm_head": r(H, cfg.vocab), "layers": []}
  for _ in range(cfg.n_layers):
    w["layers"].append({
        "attn_norm": np.ones(H, np.float32),
        "q_proj": r(H, qd), "k_proj": r(H, kd), "v_proj": r(H, kd), "o_proj": r(qd, H),
        "q_norm": np.ones(Hd, np.float32), "k_norm": np.ones(Hd, np.float32),
        "mlp_norm": np.ones(H, np.float32),
        "gate": r(H, cfg.ffn), "up": r(H, cfg.ffn), "down": r(cfg.ffn, H),
    })
  return w


# ---- ops (mirror the tunix formulas) ----
def rms_norm(x, w, eps):
  x = x.astype(np.float32)
  rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
  return (w.astype(np.float32) * (x / rms)).astype(np.float32)


def apply_rope(x, positions, rope_theta):
  # x: [T, n, head_dim]; standard RoPE (rotate-half), fraction over head_dim/2.
  head_dim = x.shape[-1]
  half = head_dim // 2
  fraction = np.arange(0, head_dim, 2, dtype=np.float32) / head_dim
  timescale = rope_theta ** fraction                       # [half]
  angle = positions[:, None].astype(np.float32) / timescale[None, :]  # [T, half]
  sin, cos = np.sin(angle)[:, None, :], np.cos(angle)[:, None, :]     # [T,1,half]
  x1, x2 = x[..., :half], x[..., half:]
  return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(np.float32)


def _softmax(z):
  z = z - z.max(axis=-1, keepdims=True)
  e = np.exp(z.astype(np.float32))
  return e / e.sum(axis=-1, keepdims=True)


def attention(x, lw, cfg, positions, perturb=None):
  T, H = x.shape
  Hd, nH, nKv = cfg.head_dim, cfg.n_heads, cfg.n_kv_heads
  G = nH // nKv
  q = (x @ lw["q_proj"]).reshape(T, nH, Hd)
  k = (x @ lw["k_proj"]).reshape(T, nKv, Hd)
  v = (x @ lw["v_proj"]).reshape(T, nKv, Hd)
  # QK-norm (per-head RMSNorm over head_dim)
  q = rms_norm(q, lw["q_norm"], cfg.norm_eps)
  k = rms_norm(k, lw["k_norm"], cfg.norm_eps)
  q = apply_rope(q, positions, cfg.rope_theta)
  k = apply_rope(k, positions, cfg.rope_theta)
  scale = Hd ** -0.5
  qg = q.reshape(T, nKv, G, Hd)                                   # [T,Kv,G,Hd]
  # attn logits [Kv,G,T,S]
  logits = np.einsum('tkgd,skd->kgts', qg, k) * scale
  mask = np.triu(np.ones((T, T), bool), k=1)
  logits = np.where(mask[None, None], -1e30, logits)
  attn = _softmax(logits)
  out = np.einsum('kgts,skd->tkgd', attn, v).reshape(T, nH * Hd)
  if perturb == "attention":
    # perturb the attention OUTPUT (a genuine value change; a real kernel diff is
    # NOT shift-invariant). Perturbing pre-softmax logits by a constant would be
    # shift-invariant and thus a no-op — bad stand-in.
    out = out + 1e-2
  return (out @ lw["o_proj"]).astype(np.float32)


def mlp(x, lw, perturb=None):
  def silu(z): return z / (1.0 + np.exp(-z.astype(np.float32)))
  act = silu(x @ lw["gate"]) * (x @ lw["up"])
  if perturb == "mlp":
    act = act + 1e-2
  return (act @ lw["down"]).astype(np.float32)


def forward(w, cfg: ToyConfig, tokens, perturb_op=None, perturb_layer=None):
  """Returns (logits [T,vocab], activations: ordered list of (name, array)).

  perturb_op in {"rms_norm","attention","mlp"} injected at layer perturb_layer to
  simulate a mis-aligned kernel (known-answer test for the attribution logic).
  """
  T = len(tokens)
  positions = np.arange(T)
  acts = []
  def cap(name, arr):
    acts.append((name, np.asarray(arr, np.float32).copy())); return arr

  x = cap("embed", w["embed"][tokens])
  for li, lw in enumerate(w["layers"]):
    p = perturb_op if perturb_layer == li else None
    # --- attention sub-block (pre-norm residual) ---
    n1 = rms_norm(x, lw["attn_norm"], cfg.norm_eps)
    if p == "rms_norm": n1 = n1 + 1e-2
    n1 = cap(f"L{li}.attn_norm", n1)
    a = cap(f"L{li}.attn", attention(n1, lw, cfg, positions, perturb=p))
    x = cap(f"L{li}.attn_resid", x + a)
    # --- mlp sub-block ---
    n2 = rms_norm(x, lw["mlp_norm"], cfg.norm_eps)
    if p == "rms_norm": n2 = n2 + 1e-2
    n2 = cap(f"L{li}.mlp_norm", n2)
    m = cap(f"L{li}.mlp", mlp(n2, lw, perturb=p))
    x = cap(f"L{li}.mlp_resid", x + m)
  x = cap("final_norm", rms_norm(x, w["final_norm"], cfg.norm_eps))
  logits = cap("logits", x @ w["lm_head"])
  return logits, acts
