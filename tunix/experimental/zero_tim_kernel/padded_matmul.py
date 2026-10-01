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
"""Model-pinned shape padding around the unchanged fixed-tile matmul ("P22.XI").

Padding is what makes the kernel usable for every M: M is zero-padded up to a
multiple of 128 (decode M=1..8 runs the same 128-row program as a 128-row
prefill slice, and a padded row can never change a real row because rows are
independent), and K/N are zero-padded only to the widths the model contract
admits.  Zero K-padding adds exact ``+0.0`` terms at the end of the f32
accumulation and N-padding adds columns that are sliced off, so every real
output element keeps its contraction order bit for bit.
"""

from __future__ import annotations

from absl import logging
import jax.numpy as jnp
from tunix.experimental.zero_tim_kernel import model_contracts
from tunix.experimental.zero_tim_kernel import pallas_matmul

BM = pallas_matmul.BM

# A TP-local width that no policy candidate
# divides (Qwen3-4B MLP: 2432 at TP4, 1216 at TP8) used to run at the
# contract's 128-wide output tile.  When the model contract already admits a
# zero-padded width for it, that padded width can carry a wide tile: padding
# columns are sliced off and every real column's contraction order is
# unchanged, so the result is bitwise identical (gate/up
# 98.1 -> 37.7 us at TP4, 57.1 -> 28.2 us at TP8, M=256).  Only contract
# tiles trigger this; explicit tiles are honored as written.
WIDE_N_CANDIDATES = (1280, 640, 512)
WIDE_N_MAX_PADDING_NUM, WIDE_N_MAX_PADDING_DEN = 11, 10  # np <= 1.1 * n
_WIDE_N_RECEIPTS: set[tuple[int, int, int]] = set()


def wide_n_tiles(m: int, k: int, n: int, tiles, mapping):
  """Return (np, block_n) when a contract-padded width admits a wide tile, else None."""
  del m, k  # Only the output width decides; kept for signature parity.
  block_m, block_n, block_k = tiles
  if (block_m, block_n, block_k) not in pallas_matmul.CONTRACT_TILES:
    return None
  if any(n % candidate == 0 for candidate in pallas_matmul.BLOCK_N_CANDIDATES):
    return None
  if not isinstance(mapping, dict):
    return None
  padded = mapping.get(int(n))
  if not isinstance(padded, int) or isinstance(padded, bool) or padded <= n:
    return None
  if padded * WIDE_N_MAX_PADDING_DEN > n * WIDE_N_MAX_PADDING_NUM:
    return None
  for candidate in WIDE_N_CANDIDATES:
    if padded % candidate == 0:
      return padded, candidate
  return None


def _padded_extent(width: int, tile: int, mapping, *, axis: str) -> int:
  """Returns `width` or its contract-admitted zero-padded multiple of `tile`."""
  width = int(width)
  tile = int(tile)
  if width <= 0 or tile <= 0:
    raise ValueError(
        f"P22.XI requires positive {axis}/tile, got {width}/{tile}"
    )
  if width % tile == 0:
    return width
  if not isinstance(mapping, dict):
    raise ValueError(f"P22.XI model {axis} padding contract must be a dict")
  padded = mapping.get(width)
  if (
      not isinstance(padded, int)
      or isinstance(padded, bool)
      or padded <= width
      or padded % tile
  ):
    raise ValueError(
        f"P22.XI {axis}={width} is not admitted by the model-pinned "
        f"tile={tile} padding contract: {mapping!r}"
    )
  return padded


def padded_matmul_extents(
    k: int,
    n: int,
    *,
    contract: model_contracts.ModelContract,
    block_k: int | None = None,
    block_n: int | None = None,
) -> tuple[int, int]:
  """Returns admitted TP-local K/N extents for the given model contract."""
  kp = _padded_extent(
      k,
      contract.block_k if block_k is None else block_k,
      contract.matmul_k_padding,
      axis="K",
  )
  np = _padded_extent(
      n,
      contract.block_n if block_n is None else block_n,
      contract.matmul_n_padding,
      axis="N",
  )
  return kp, np


def matmul(
    x,
    y,
    *,
    contract: model_contracts.ModelContract,
    interpret: bool = False,
    shape_invariant_numerics: bool = True,
    block_m: int = BM,
    block_n: int | None = None,
    block_k: int | None = None,
):
  """Fixed-tile matmul of arbitrary M with contract-admitted K/N padding."""
  if x.ndim != 2 or y.ndim != 2:
    raise ValueError(f"P22.XI expects rank-2 inputs, got {x.shape}, {y.shape}")
  m, k = map(int, x.shape)
  ky, n = map(int, y.shape)
  if k != ky:
    raise ValueError(f"P22.XI contracted dimensions differ: {k} vs {ky}")
  if m <= 0:
    raise ValueError(f"P22.XI requires positive M, got {m}")
  block_m = int(block_m)
  block_k = int(contract.block_k if block_k is None else block_k)
  block_n = int(contract.block_n if block_n is None else block_n)
  kp, np = padded_matmul_extents(
      k, n, contract=contract, block_k=block_k, block_n=block_n
  )
  mp = ((m + BM - 1) // BM) * BM

  wide = wide_n_tiles(
      mp, kp, n, (block_m, block_n, block_k), contract.matmul_n_padding
  )
  if wide is not None:
    np, wide_bn = wide
    wide_bm = 256 if mp % 256 == 0 else block_m
    key = (m, k, n)
    if key not in _WIDE_N_RECEIPTS:
      _WIDE_N_RECEIPTS.add(key)
      logging.info(
          "[PATHTRACE] matmul_wide_n_padding=v2 M=%d K=%d N=%d->%d bm=%d "
          "bn=%d bk=%d",
          m,
          k,
          n,
          np,
          wide_bm,
          wide_bn,
          block_k,
      )
    x_padded = jnp.pad(x, ((0, mp - m), (0, kp - k)), constant_values=0)
    y_padded = jnp.pad(y, ((0, kp - k), (0, np - n)), constant_values=0)
    out = pallas_matmul.matmul(
        x_padded,
        y_padded,
        interpret=interpret,
        shape_invariant_numerics=shape_invariant_numerics,
        block_m=wide_bm,
        block_n=wide_bn,
        block_k=block_k,
    )
    return out[:m, :n]
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
