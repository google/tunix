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

"""Synthetic bf16 weight updates for the delta weight sync benchmarks."""

from concurrent import futures

import numpy as np
from tunix.experimental.delta_sync import _delta_encoder_ext


def routed_expert_layout(
    layers: int, experts: int
) -> tuple[list[tuple[int, int]], list[int]]:
  """Returns the shapes and rows per block of an MoE shard's routed experts.

  Each layer has two projections, wi [experts * 2048, 1024] and
  wo [experts * 512, 2048], both cut into 32K-element blocks. With 40 layers
  of 64 experts this is 80 tensors holding 16.1 GB of bf16.

  Args:
    layers: number of MoE layers.
    experts: routed experts per layer.
  """
  shapes = [(experts * 2048, 1024), (experts * 512, 2048)] * layers
  block_rows = [32, 16] * layers
  return shapes, block_rows


def generate(
    seed: int,
    shape: tuple[int, int],
    block_rows: int,
    change_rate: float,
    block_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Returns bf16 bits of N(0, 1) weights and a copy with some of them nudged.

  Args:
    seed: seed of the random generator.
    shape: [rows, cols] of the tensor.
    block_rows: rows per block.
    change_rate: mean fraction of elements that change.
    block_sigma: sigma of the lognormal multiplier, with mean 1, that scales the
      change rate of each block. Zero gives every block the same rate.
  """
  rng = np.random.default_rng(seed)
  block_rates = change_rate * rng.lognormal(
      -0.5 * block_sigma**2, block_sigma, size=shape[0] // block_rows
  )
  row_rates = np.repeat(block_rates, block_rows).astype(np.float32)[:, None]
  weights = rng.standard_normal(shape, dtype=np.float32)
  old = (weights.view(np.uint32) >> 16).astype(np.uint16)
  del weights
  # Flipping low mantissa bits always changes a value and keeps it finite.
  nudges = rng.integers(1, 0x7F, size=shape, dtype=np.uint16, endpoint=True)
  changed = rng.random(shape, dtype=np.float32) < row_rates
  return old, np.where(changed, old ^ nudges, old)


def generate_all(
    shapes: list[tuple[int, int]],
    block_rows: list[int],
    *,
    change_rate: float,
    block_sigma: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
  """Generates every tensor of a layout; returns (old_bits, new_bits)."""
  # Few workers: each holds over 1 GB of temporaries per tensor.
  with futures.ThreadPoolExecutor(max_workers=4) as pool:
    pairs = list(
        pool.map(
            lambda index: generate(
                index,
                shapes[index],
                block_rows[index],
                change_rate,
                block_sigma,
            ),
            range(len(shapes)),
        )
    )
  return [old for old, _ in pairs], [new for _, new in pairs]


def capacities_by_shape(
    shapes: list[tuple[int, int]], fullest: list[int]
) -> list[int]:
  """Returns a capacity per tensor that fits the fullest block of its shape.

  Tensors of one shape share a capacity, so a kernel compiled for a shape
  serves all of them.

  Args:
    shapes: [rows, cols] of each tensor.
    fullest: changed-word count of each tensor's fullest block.
  """
  shape_capacity: dict[tuple[int, int], int] = {}
  for shape, count in zip(shapes, fullest):
    # Rounded up to the payload alignment, and never empty.
    align = _delta_encoder_ext.CAPACITY_ALIGN
    capacity = max(1, -(-count // align)) * align
    shape_capacity[shape] = max(shape_capacity.get(shape, 0), capacity)
  return [shape_capacity[shape] for shape in shapes]
