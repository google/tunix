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
"""Automatic zero-padding wrapper around fixed-tile `pallas_matmul.matmul`.

Zero-pads `M` to a multiple of `BM = 128`, `K` to a multiple of `block_k`, and
`N` to a multiple of `block_n` (or to a 256-aligned wide-`N` tile when within
10% padding overhead), runs `pallas_matmul.matmul`, and slices `[:M, :N]`.

Why zero-padding preserves Zero-TIM bitwise invariance:
- `M` rows are independent; decode (`M=1..8`) runs the exact same 128-row
  program as a 128-row prefill slice.
- Zero `K`-padding appends exact `+0.0` blocks at the end of the `f32`
  contraction loop (`acc + 0.0 == acc` under IEEE-754 RNE).
- Zero `N`-padding appends output columns that are sliced off.
"""

from __future__ import annotations

import jax.numpy as jnp
from tunix.zero_tim_kernel import model_contracts
from tunix.zero_tim_kernel import pallas_matmul

BM = pallas_matmul.BM
WIDE_N_CANDIDATES = (1280, 640, 512)
WIDE_N_MAX_PADDING_NUM, WIDE_N_MAX_PADDING_DEN = 11, 10  # np <= 1.1 * n


def wide_n_tiles(
    m: int, k: int, n: int, tiles: tuple[int, int, int]
) -> tuple[int, int] | None:
  """Returns `(np, block_n)` when 256-aligning `n` admits a wide output tile."""
  del m, k
  if tiles not in pallas_matmul.CONTRACT_TILES:
    return None
  if any(n % candidate == 0 for candidate in pallas_matmul.BLOCK_N_CANDIDATES):
    return None
  padded = model_contracts.round_up(int(n), model_contracts.SWIGLU_FEATURE_TILE)
  if padded <= n:
    return None
  if padded * WIDE_N_MAX_PADDING_DEN > n * WIDE_N_MAX_PADDING_NUM:
    return None
  for candidate in WIDE_N_CANDIDATES:
    if padded % candidate == 0:
      return padded, candidate
  return None


def padded_matmul_extents(
    k: int,
    n: int,
    *,
    contract: model_contracts.ModelContract | None = None,
    block_k: int | None = None,
    block_n: int | None = None,
) -> tuple[int, int]:
  """Returns zero-padded `(kp, np)` extents executed by `padded_matmul.matmul`."""
  default_bm = BM if contract is None else contract.block_m
  default_bk = pallas_matmul.BK if contract is None else contract.block_k
  default_bn = pallas_matmul.BN if contract is None else contract.block_n
  bk = int(default_bk if block_k is None else block_k)
  bn = int(default_bn if block_n is None else block_n)
  kp = model_contracts.round_up(int(k), bk)
  np = model_contracts.round_up(int(n), bn)
  wide = wide_n_tiles(default_bm, kp, int(n), (default_bm, bn, bk))
  if wide is not None:
    np, _ = wide
  return kp, np


def matmul(
    x,
    y,
    *,
    contract: model_contracts.ModelContract | None = None,
    interpret: bool = False,
    shape_invariant_numerics: bool = True,
    block_m: int = BM,
    block_n: int | None = None,
    block_k: int | None = None,
):
  """Fixed-tile matmul of arbitrary positive `(M, K) @ (K, N)` with zero-padding."""
  if x.ndim != 2 or y.ndim != 2:
    raise ValueError(
        f"padded_matmul expects rank-2 inputs, got {x.shape}, {y.shape} "
    )
  m, k = map(int, x.shape)
  ky, n = map(int, y.shape)
  if k != ky:
    raise ValueError(f"padded_matmul contracted dimensions differ: {k} vs {ky}")
  if m <= 0:
    raise ValueError(f"padded_matmul requires positive M, got {m}")
  default_bk = pallas_matmul.BK if contract is None else contract.block_k
  default_bn = pallas_matmul.BN if contract is None else contract.block_n
  block_m = int(block_m)
  block_k = int(default_bk if block_k is None else block_k)
  block_n = int(default_bn if block_n is None else block_n)
  kp, np = padded_matmul_extents(
      k, n, contract=contract, block_k=block_k, block_n=block_n
  )
  mp = model_contracts.round_up(m, BM)

  wide = wide_n_tiles(mp, kp, n, (block_m, block_n, block_k))
  if wide is not None:
    np, block_n = wide
    block_m = 256 if mp % 256 == 0 else block_m

  if mp == m and kp == k and np == n:
    return pallas_matmul.matmul(
        x,
        y,
        interpret=interpret,
        shape_invariant_numerics=shape_invariant_numerics,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
    )
  x_padded = jnp.pad(x, ((0, mp - m), (0, kp - k)), constant_values=0)
  y_padded = jnp.pad(y, ((0, kp - k), (0, np - n)), constant_values=0)
  out = pallas_matmul.matmul(
      x_padded,
      y_padded,
      interpret=interpret,
      shape_invariant_numerics=shape_invariant_numerics,
      block_m=block_m,
      block_n=block_n,
      block_k=block_k,
  )
  return out[:m, :n]
