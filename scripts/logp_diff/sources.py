"""Three-source abstraction for the logp-diff probe.

A Source names one forward path and exposes:
  - get_logp(tokens) -> per-token logp   (ALWAYS)
  - get_acts(tokens) -> ordered [(name, array)] activations  (OPTIONAL; None if unavailable)

The three real sources (Phase 3, remote):
  A = vLLM-decode  : rollout's decode-time logprobs (logp only, no per-layer acts)
  B = vLLM-forward : full-sequence prefill via prompt_logprobs (logp; acts only if tpu_inference hookable)
  C = tunix-forward: tunix qwen3 full forward (logp + per-layer acts via hooks)

Phase 2 (CPU) uses toy sources built from toy_qwen3 to validate compare().
compare(B,C) is the main isolation (both full-sequence forward); A vs B measures decode effect.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import harness as H


@dataclass
class Source:
  name: str
  get_logp: Callable            # tokens -> per-token logp (np array)
  get_acts: Optional[Callable] = None  # tokens -> [(name, array)] or None


def compare(src_a: Source, src_b: Source, tokens, atol: float = 0.0) -> dict:
  """Full report between two sources: logp diff (always) + attribution (if both have acts)."""
  lp_a, lp_b = src_a.get_logp(tokens), src_b.get_logp(tokens)
  logp_stats = H.logp_diff_stats(lp_a, lp_b)

  attribution = {"first_divergence": None, "per_op": [], "note": "acts unavailable for one/both sources"}
  if src_a.get_acts is not None and src_b.get_acts is not None:
    attribution = H.activation_diff(src_a.get_acts(tokens), src_b.get_acts(tokens), atol=atol)

  determinism = {}
  for s in (src_a, src_b):
    ok, d = H.determinism_check(s.get_logp, tokens)
    determinism[s.name] = {"bitwise_identical": ok, "max_diff": d}

  return {"pair": f"{src_a.name} vs {src_b.name}",
          "logp_diff": logp_stats, "attribution": attribution, "determinism": determinism}


# ---- Phase 2 CPU toy sources (validate compare) ----
def toy_source(name, weights, cfg, perturb_op=None, perturb_layer=None) -> Source:
  import toy_qwen3 as tq
  def fwd(tokens):
    return tq.forward(weights, cfg, tokens, perturb_op=perturb_op, perturb_layer=perturb_layer)
  return Source(
      name=name,
      get_logp=lambda t: H.per_token_logp(fwd(t)[0], t),
      get_acts=lambda t: fwd(t)[1],
  )
