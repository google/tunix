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
"""Model-pinned padding around the unchanged SwiGLU kernel ("P22.XJ").

SwiGLU is elementwise, so zero-padding rows/features never changes a real
element (``silu(0) * 0 == 0`` and the padding is sliced off).
"""

from __future__ import annotations

import jax.numpy as jnp
from tunix.experimental.zero_tim_kernel import model_contracts
from tunix.experimental.zero_tim_kernel import pallas_swiglu

BM = pallas_swiglu.BM
BF = pallas_swiglu.BF


def padded_feature_extent(
    feature: int, *, contract: model_contracts.ModelContract
) -> int:
  """Returns an admitted BF-aligned extent for one TP-local model width."""
  feature = int(feature)
  if feature <= 0:
    raise ValueError(f"P22.XJ requires positive feature width, got {feature}")
  if feature % BF == 0:
    return feature

  padding = contract.swiglu_feature_padding
  if not isinstance(padding, dict):
    raise ValueError("P22.XJ model SWIGLU_FEATURE_PADDING must be a dict")
  padded = padding.get(feature)
  if (
      not isinstance(padded, int)
      or isinstance(padded, bool)
      or padded <= feature
      or padded % BF
  ):
    raise ValueError(
        f"P22.XJ feature width F={feature} is not admitted by the "
        f"model-pinned BF={BF} padding contract: {padding!r}"
    )
  return padded


def swiglu(
    gate,
    up,
    *,
    contract: model_contracts.ModelContract,
    interpret: bool = False,
    shape_invariant_numerics: bool = True,
):
  """``silu(gate) * up`` for any M with contract-admitted feature padding."""
  if tuple(gate.shape) != tuple(up.shape):
    raise ValueError(
        f"P22.XJ gate/up shapes differ: {gate.shape} vs {up.shape}"
    )
  if gate.ndim != 2:
    raise ValueError(f"P22.XJ expects rank-2 inputs, got {gate.shape}")
  m, f = map(int, gate.shape)
  if m <= 0:
    raise ValueError(f"P22.XJ requires positive M, got {(m, f)}")
  fp = padded_feature_extent(f, contract=contract)
  if gate.dtype != jnp.bfloat16 or up.dtype != jnp.bfloat16:
    raise ValueError(
        f"P22.XJ requires bf16 inputs, got {gate.dtype}, {up.dtype}"
    )
  mp = ((m + BM - 1) // BM) * BM
  if mp == m and fp == f:
    return pallas_swiglu.swiglu(
        gate,
        up,
        interpret=interpret,
        shape_invariant_numerics=shape_invariant_numerics,
    )
  pad = ((0, mp - m), (0, fp - f))
  gate_padded = jnp.pad(gate, pad, constant_values=0)
  up_padded = jnp.pad(up, pad, constant_values=0)
  out = pallas_swiglu.swiglu(
      gate_padded,
      up_padded,
      interpret=interpret,
      shape_invariant_numerics=shape_invariant_numerics,
  )
  return out[:m, :f]
