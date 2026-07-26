"""Framework-agnostic logp-diff / per-op attribution harness.

Operates on plain numpy arrays, so the SAME logic serves both the CPU toy model
(Phase 2) and the real jax models on TPU (Phase 3: jax arrays -> np.asarray -> here).

Provides:
  - per_token_logp(logits, tokens)        : next-token logp (mirrors selective_log_softmax)
  - logp_diff_stats(a, b, mask)           : mean/max/pearson (mirrors agentic_grpo_learner:306)
  - activation_diff(acts_a, acts_b, atol) : per-op-boundary diff + FIRST divergence (attribution)
  - op_isolation(fn_a, fn_b, x)           : same input -> two impls -> max abs diff (standalone)
  - determinism_check(fn, *args)          : run twice -> bitwise identical? (determinism != alignment)
  - build_report(...)                     : structured report reused by the remote probe
"""
from __future__ import annotations
import numpy as np


def per_token_logp(logits, tokens):
  """logp of the NEXT token at each position. logits [T,V], tokens [T] -> [T-1].

  Mirrors tunix/rl/common.py get_per_token_logps: for position t, take
  log_softmax(logits[t]) at tokens[t+1].
  """
  logits = np.asarray(logits, np.float32)
  z = logits[:-1]
  z = z - z.max(axis=-1, keepdims=True)
  logsm = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
  nxt = np.asarray(tokens)[1:]
  return logsm[np.arange(len(nxt)), nxt].astype(np.float32)


def logp_diff_stats(a, b, mask=None):
  """mean/max abs diff + pearson between two per-token logp arrays."""
  a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
  if mask is not None:
    m = np.asarray(mask, bool)
    a, b = a[m], b[m]
  d = np.abs(a - b)
  if a.size >= 2 and a.std() > 0 and b.std() > 0:
    pear = float(np.corrcoef(a, b)[0, 1])
  else:
    pear = 1.0 if np.allclose(a, b) else 0.0
  return {"mean": float(d.mean()) if d.size else 0.0,
          "max": float(d.max()) if d.size else 0.0,
          "pearson": pear, "n": int(a.size)}


def activation_diff(acts_a, acts_b, atol=0.0):
  """Per-op-boundary diff between two ordered activation lists.

  Returns {per_op: [(name, max_abs_diff)], first_divergence: name|None}.
  first_divergence = first op-boundary whose max abs diff > atol = the culprit op.
  """
  names_a = [n for n, _ in acts_a]
  names_b = [n for n, _ in acts_b]
  if names_a != names_b:
    raise ValueError(f"activation name/order mismatch:\n{names_a}\n{names_b}")
  per_op, first = [], None
  for (name, xa), (_, xb) in zip(acts_a, acts_b):
    md = float(np.abs(np.asarray(xa, np.float64) - np.asarray(xb, np.float64)).max())
    per_op.append((name, md))
    if first is None and md > atol:
      first = name
  return {"per_op": per_op, "first_divergence": first}


def op_isolation(fn_a, fn_b, *inputs):
  """Standalone op-type isolation: same input -> two impls -> max abs diff."""
  ya = np.asarray(fn_a(*inputs), np.float64)
  yb = np.asarray(fn_b(*inputs), np.float64)
  return float(np.abs(ya - yb).max())


def determinism_check(fn, *args):
  """Run fn twice on the same args; return (bitwise_identical, max_abs_diff).

  NOTE: determinism (same kernel twice == bitwise) is necessary-not-sufficient;
  the real target is cross-kernel ALIGNMENT (op_isolation == 0).
  """
  y1 = np.asarray(fn(*args))
  y2 = np.asarray(fn(*args))
  return bool(np.array_equal(y1, y2)), float(np.abs(y1.astype(np.float64) - y2.astype(np.float64)).max())


def build_report(logp_stats, attribution, op_iso=None, determinism=None):
  return {"logp_diff": logp_stats, "attribution": attribution,
          "op_isolation": op_iso or {}, "determinism": determinism or {}}
