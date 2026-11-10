# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Framework-neutral Score Centering coefficient and diagnostic math."""

from __future__ import annotations

import types
from typing import Any

import numpy as np


def compute_score_centering_coefficients(
    trainer_topk_logps: Any,
    sampler_topk_logps: Any,
    sampler_is: str | None,
    sampler_is_threshold: float | None,
    eps: float,
    has_sampler_is_weights: bool = True,
    completion_ids: Any | None = None,
    topk_token_ids: Any | None = None,
    *,
    xp: types.ModuleType = np,
) -> tuple[Any, dict[str, Any], Any]:
  """Computes detached Score Centering coefficients and unreduced diagnostics.

  Operates on already-detached arrays via `xp` (`numpy` by default, or
  `jax.numpy` inside JIT) without importing JAX or Flax.

  With `q` / `p` the sampler / trainer probabilities of the sampler's top-k
  tokens and `rho = max(1 - sum q, eps) / max(1 - sum p, eps)`, the per-slot
  coefficient is `q_k * w_k - alpha * p_k` on valid slots (`0` on `-inf`
  padding slots). Without IS, `w = 1` and `alpha = rho`; with token IS clipped
  at `c`, `w = min(p / q, c)` and `alpha = min(1, c * rho)`.

  Args:
    trainer_topk_logps: Detached `[B, L, K]` trainer log probs of top-k tokens.
    sampler_topk_logps: Detached `[B, L, K]` sampler log probs (`-inf` padded).
    sampler_is: `None` or `"token"`.
    sampler_is_threshold: The token IS clip `c`.
    eps: Positive floor for the tail probability masses.
    has_sampler_is_weights: Whether token IS weights are active on this batch.
    completion_ids: Optional `[B, L]` sampled token IDs for `sampled_in_topk`.
    topk_token_ids: Optional `[B, L, K]` top-k token IDs for `sampled_in_topk`.
    xp: Array module (`numpy` or `jax.numpy`).

  Returns:
    `(coeff, token_stats, has_valid_topk)` where `coeff` has shape `[B, L, K]`,
    `token_stats` maps unreduced diagnostic names to `[B, L]` float32 arrays,
    and `has_valid_topk` is a `[B, L]` boolean mask marking positions with at
    least one finite sampler top-k candidate.
  """
  valid = xp.isfinite(sampler_topk_logps)
  has_valid_topk = xp.any(valid, axis=-1)
  safe_trainer_logps = xp.where(
      valid, trainer_topk_logps.astype(xp.float32), -1e9
  )
  safe_sampler_logps = xp.where(
      valid, sampler_topk_logps.astype(xp.float32), -1e9
  )
  q = xp.where(valid, xp.exp(safe_sampler_logps), 0.0)
  p = xp.where(valid, xp.exp(safe_trainer_logps), 0.0)
  head_mass_q = xp.sum(q, axis=-1)
  head_mass_p = xp.sum(p, axis=-1)
  tail_q = 1.0 - head_mass_q
  tail_p = 1.0 - head_mass_p
  rho = xp.maximum(tail_q, eps) / xp.maximum(tail_p, eps)

  tail_is_clipped = None
  if sampler_is == "token" and has_sampler_is_weights:
    c = xp.inf if sampler_is_threshold is None else sampler_is_threshold
    log_ratio = xp.clip(safe_trainer_logps - safe_sampler_logps, -20.0, 20.0)
    w = xp.minimum(xp.exp(log_ratio), c)
    alpha = xp.minimum(1.0, c * rho)
    tail_is_clipped = (c * rho > 1.0).astype(xp.float32)
  else:
    w, alpha = 1.0, rho

  coeff = xp.where(valid, q * w - alpha[..., None] * p, 0.0)
  token_stats: dict[str, Any] = {
      "head_mass_q": head_mass_q,
      "head_mass_p": head_mass_p,
      "tail_ratio_rho": rho,
      "abs_coeff_sum": xp.sum(xp.abs(coeff), axis=-1),
      "tail_eps_clamped": ((tail_q <= eps) | (tail_p <= eps)).astype(
          xp.float32
      ),
      "tv_distance": (
          0.5 * xp.sum(xp.abs(q - p), axis=-1)
          + 0.5 * xp.abs(head_mass_p - head_mass_q)
      ),
  }
  if tail_is_clipped is not None:
    token_stats["tail_is_clipped"] = tail_is_clipped
  if completion_ids is not None and topk_token_ids is not None:
    token_stats["sampled_in_topk"] = xp.any(
        valid & (topk_token_ids == completion_ids[..., None]), axis=-1
    ).astype(xp.float32)
  return coeff, token_stats, has_valid_topk
