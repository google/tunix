# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tile policies (`ModelContract`) and `round_up` helper for Zero-TIM kernels."""

from __future__ import annotations

import dataclasses

VOCAB_SIZE = 151936
LM_HEAD_TILE = 256
SWIGLU_FEATURE_TILE = 256


def round_up(value: int, multiple: int) -> int:
  """Rounds a positive integer `value` up to a positive `multiple`."""
  if value <= 0:
    raise ValueError(f"value must be positive, got {value}")
  if multiple <= 0:
    raise ValueError(f"multiple must be positive, got {multiple}")
  return ((value + multiple - 1) // multiple) * multiple


@dataclasses.dataclass(frozen=True)
class ModelContract:
  """Tile policy and optional model metadata for fixed-order Pallas matmuls."""

  name: str = "default"
  block_m: int = 128
  block_n: int = 256
  block_k: int = 256
  hidden_size: int = 0
  intermediate_size: int = 0
  num_attention_heads: int = 0
  num_kv_heads: int = 0
  head_dim: int = 0
  tp_size: int = 1
  vocab_size: int = 0

  def __post_init__(self) -> None:
    if self.block_m <= 0 or self.block_n <= 0 or self.block_k <= 0:
      raise ValueError(
          f"ModelContract({self.name!r}): tiles must be positive, got"
          f" {(self.block_m, self.block_n, self.block_k)}"
      )
    if self.tp_size <= 0:
      raise ValueError(
          f"ModelContract({self.name!r}): tp_size must be positive, got"
          f" {self.tp_size}"
      )
    if (
        self.hidden_size < 0
        or self.intermediate_size < 0
        or self.num_attention_heads < 0
        or self.num_kv_heads < 0
        or self.head_dim < 0
        or self.vocab_size < 0
    ):
      raise ValueError(
          f"ModelContract({self.name!r}): model dimensions must be non-negative"
      )
    if self.vocab_size > 0 and self.vocab_size % self.tp_size != 0:
      raise ValueError(
          f"ModelContract({self.name!r}): vocab_size={self.vocab_size} is not"
          f" divisible by tp_size={self.tp_size}"
      )
    if (
        self.num_attention_heads > 0
        and self.num_attention_heads % self.tp_size != 0
    ):
      raise ValueError(
          f"ModelContract({self.name!r}):"
          f" num_attention_heads={self.num_attention_heads} is not divisible"
          f" by tp_size={self.tp_size}"
      )
    if self.num_kv_heads > 0 and self.num_kv_heads % self.tp_size != 0:
      raise ValueError(
          f"ModelContract({self.name!r}): num_kv_heads={self.num_kv_heads} is"
          f" not divisible by tp_size={self.tp_size}"
      )

  @property
  def tiles(self) -> tuple[int, int, int]:
    return (self.block_m, self.block_n, self.block_k)


def build_contract(
    name: str,
    *,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_kv_heads: int,
    head_dim: int,
    tp_size: int,
    tiles: tuple[int, int, int],
    vocab_size: int = VOCAB_SIZE,
) -> ModelContract:
  """Constructs a `ModelContract` for one `(model, tp_size)` pair."""
  if tp_size <= 0:
    raise ValueError(f"build_contract({name!r}): tp_size={tp_size} must be > 0")
  if (
      hidden_size <= 0
      or intermediate_size <= 0
      or head_dim <= 0
      or vocab_size <= 0
      or num_attention_heads <= 0
      or num_kv_heads <= 0
  ):
    raise ValueError(
        f"build_contract({name!r}): model dimensions must be positive"
    )
  bm, bn, bk = (int(t) for t in tiles)
  return ModelContract(
      name=name,
      block_m=bm,
      block_n=bn,
      block_k=bk,
      hidden_size=hidden_size,
      intermediate_size=intermediate_size,
      num_attention_heads=num_attention_heads,
      num_kv_heads=num_kv_heads,
      head_dim=head_dim,
      tp_size=tp_size,
      vocab_size=vocab_size,
  )


def _qwen3(
    name: str,
    *,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    tp_size: int,
    tiles: tuple[int, int, int],
) -> ModelContract:
  return build_contract(
      name,
      hidden_size=hidden_size,
      intermediate_size=intermediate_size,
      num_attention_heads=num_attention_heads,
      num_kv_heads=8,
      head_dim=128,
      tp_size=tp_size,
      tiles=tiles,
      vocab_size=VOCAB_SIZE,
  )


CONTRACTS: dict[str, ModelContract] = {
    "qwen1p7b": _qwen3(
        "qwen1p7b",
        hidden_size=2048,
        intermediate_size=6144,
        num_attention_heads=16,
        tp_size=4,
        tiles=(128, 256, 256),
    ),
    "qwen1p7b_tp1": _qwen3(
        "qwen1p7b_tp1",
        hidden_size=2048,
        intermediate_size=6144,
        num_attention_heads=16,
        tp_size=1,
        tiles=(128, 256, 256),
    ),
    "qwen1p7b_tp2": _qwen3(
        "qwen1p7b_tp2",
        hidden_size=2048,
        intermediate_size=6144,
        num_attention_heads=16,
        tp_size=2,
        tiles=(128, 256, 256),
    ),
    "qwen4b": _qwen3(
        "qwen4b",
        hidden_size=2560,
        intermediate_size=9728,
        num_attention_heads=32,
        tp_size=8,
        tiles=(128, 128, 128),
    ),
    "qwen4b_tp4": _qwen3(
        "qwen4b_tp4",
        hidden_size=2560,
        intermediate_size=9728,
        num_attention_heads=32,
        tp_size=4,
        tiles=(128, 128, 128),
    ),
    "qwen8b": _qwen3(
        "qwen8b",
        hidden_size=4096,
        intermediate_size=12288,
        num_attention_heads=32,
        tp_size=4,
        tiles=(128, 256, 256),
    ),
    "qwen8b_tp1": _qwen3(
        "qwen8b_tp1",
        hidden_size=4096,
        intermediate_size=12288,
        num_attention_heads=32,
        tp_size=1,
        tiles=(128, 256, 256),
    ),
    "qwen8b_tp2": _qwen3(
        "qwen8b_tp2",
        hidden_size=4096,
        intermediate_size=12288,
        num_attention_heads=32,
        tp_size=2,
        tiles=(128, 256, 256),
    ),
    "qwen8b_tp8": _qwen3(
        "qwen8b_tp8",
        hidden_size=4096,
        intermediate_size=12288,
        num_attention_heads=32,
        tp_size=8,
        tiles=(128, 128, 128),
    ),
    "qwen32b": _qwen3(
        "qwen32b",
        hidden_size=5120,
        intermediate_size=25600,
        num_attention_heads=64,
        tp_size=8,
        tiles=(128, 128, 128),
    ),
}


def get_contract(name: str) -> ModelContract:
  """Returns the registered `ModelContract` for `name`."""
  if name not in CONTRACTS:
    raise KeyError(
        f"Unknown model contract {name!r}; valid: {sorted(CONTRACTS)}"
    )
  return CONTRACTS[name]


def find_contract(*, hidden_size: int, tp_size: int) -> ModelContract:
  """Looks up the unique registered `ModelContract` for `(hidden_size, tp_size)`."""
  matches = [
      c
      for c in CONTRACTS.values()
      if c.hidden_size == hidden_size and c.tp_size == tp_size
  ]
  if not matches:
    raise KeyError(
        f"No model contract for hidden_size={hidden_size}, tp_size={tp_size};"
        " registered:"
        f" {[(c.name, c.hidden_size, c.tp_size) for c in CONTRACTS.values()]}"
    )
  if len(matches) > 1:
    raise KeyError(
        f"Ambiguous model contract for hidden_size={hidden_size}, "
        f"tp_size={tp_size}: {[c.name for c in matches]}"
    )
  return matches[0]
