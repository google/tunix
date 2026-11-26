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

"""Benchmarks the host-side delta encoder across thread counts.

The workload is the routed-expert weights of one MoE model shard: by default
40 layers x 2 projections = 80 tensors, 16.1 GB of bf16, which needs about
37 GB of host memory. No accelerator is used.
"""

from collections.abc import Sequence
import json
import statistics
import time

from absl import app
from absl import flags
import numpy as np
from tunix.experimental.delta_sync import _delta_encoder_ext
from tunix.experimental.delta_sync import synthetic_deltas

_LAYERS = flags.DEFINE_integer(
    "layers", 40, "MoE layers in the model shard.", lower_bound=1
)
_EXPERTS = flags.DEFINE_integer(
    "experts", 64, "Routed experts per layer.", lower_bound=1
)
_CHANGE_RATE = flags.DEFINE_float(
    "change_rate",
    0.0173,
    "Mean fraction of weights that change per step.",
    lower_bound=0.0,
    upper_bound=1.0,
)
_BLOCK_SIGMA = flags.DEFINE_float(
    "block_sigma",
    0.16,
    "Sigma of the lognormal per-block multiplier on the change rate.",
    lower_bound=0.0,
)
_REPEATS = flags.DEFINE_integer(
    "repeats", 5, "Timed iterations.", lower_bound=1
)
_THREADS = flags.DEFINE_multi_integer(
    "threads", [1], "Thread counts to measure.", lower_bound=1
)


def _median_ms(repeats: int, fn) -> float:
  """Runs `fn` once to warm up, then returns its median time in ms."""
  fn()
  seconds = []
  for _ in range(repeats):
    start = time.perf_counter()
    fn()
    seconds.append(time.perf_counter() - start)
  return round(statistics.median(seconds) * 1e3, 1)


def run_benchmark(
    *,
    layers: int,
    experts: int,
    change_rate: float,
    block_sigma: float,
    repeats: int,
    threads: Sequence[int],
) -> dict[str, object]:
  """Times the capacity scan and the encode at each thread count."""
  shapes, block_rows = synthetic_deltas.routed_expert_layout(layers, experts)
  old_bits, new_bits = synthetic_deltas.generate_all(
      shapes, block_rows, change_rate=change_rate, block_sigma=block_sigma
  )
  weight_bytes = sum(bits.nbytes for bits in old_bits)

  fullest = _delta_encoder_ext.fullest_block_changes(
      old_bits, new_bits, block_rows=block_rows, num_threads=max(threads)
  )
  capacities = synthetic_deltas.capacities_by_shape(shapes, fullest)
  offsets = [
      np.empty((shape[0] // block, capacity), np.int32)
      for shape, block, capacity in zip(shapes, block_rows, capacities)
  ]
  values = [np.empty_like(o) for o in offsets]

  def encode(num_threads: int, force_scalar: bool = False) -> int:
    return _delta_encoder_ext.encode_block_patches(
        old_bits,
        new_bits,
        block_rows=block_rows,
        offsets=offsets,
        values=values,
        num_threads=num_threads,
        force_scalar=force_scalar,
    )

  rows = []
  for num_threads in threads:
    count_ms = _median_ms(
        repeats,
        lambda n=num_threads: _delta_encoder_ext.fullest_block_changes(
            old_bits, new_bits, block_rows=block_rows, num_threads=n
        ),
    )
    encode_ms = _median_ms(repeats, lambda n=num_threads: encode(n))
    rows.append({
        "threads": num_threads,
        "count_ms": count_ms,
        "encode_ms": encode_ms,
        "encode_gb_per_s": round(weight_bytes / 1e6 / encode_ms, 1),
    })
  scalar_ms = _median_ms(repeats, lambda: encode(max(threads), True))

  # The scalar payload is in the buffers now; the default kernel must
  # reproduce it exactly.
  scalar_offsets = [o.copy() for o in offsets]
  scalar_values = [v.copy() for v in values]
  changed_words = encode(max(threads))
  kernels_agree = all(
      np.array_equal(a, b)
      for a, b in zip(offsets + values, scalar_offsets + scalar_values)
  )

  payload_bytes = sum(o.nbytes + v.nbytes for o, v in zip(offsets, values))
  return {
      "simd_kernel": _delta_encoder_ext.simd_kernel(),
      "tensors": len(shapes),
      "weights_gb": round(weight_bytes / 1e9, 2),
      "payload_gb": round(payload_bytes / 1e9, 3),
      "payload_gb_without_padding": round(changed_words * 8 / 1e9, 3),
      "changed_words": changed_words,
      "capacity_by_shape": {
          str(list(shape)): capacity
          for shape, capacity in zip(shapes, capacities)
      },
      "by_threads": rows,
      "encode_scalar_ms": scalar_ms,
      "encode_scalar_threads": max(threads),
      "kernels_agree": kernels_agree,
  }


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  result = run_benchmark(
      layers=_LAYERS.value,
      experts=_EXPERTS.value,
      change_rate=_CHANGE_RATE.value,
      block_sigma=_BLOCK_SIGMA.value,
      repeats=_REPEATS.value,
      threads=_THREADS.value,
  )
  print(json.dumps(result, indent=2), flush=True)
  if not result["kernels_agree"]:
    raise RuntimeError("the default and scalar kernels disagree")


if __name__ == "__main__":
  app.run(main)
