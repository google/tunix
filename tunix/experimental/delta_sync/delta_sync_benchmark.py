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

"""Benchmarks delta weight sync end to end on one machine with a TPU.

Stands in for both ends of a sync, without the network hop between them:

  1. encode: the trainer host diffs the old and new bf16 weights and writes the
     per-block patches (C++, multi-threaded).
  2. h2d: the rollout host copies the patches to the TPU.
  3. apply: the SparseCore kernel patches the rollout weights in place.

Afterwards the weights on the TPU are read back and compared, bit for bit, with
the new weights.

Not included in the stage times: the one-off scan that sizes the patch buffers
(reported as count_ms, a single cold sample) and refreshing the trainer's copy
of the last synced weights.

The workload is the routed-expert weights of one MoE model shard: by default
40 layers x 2 projections = 80 tensors, 16.1 GB of bf16, with the same change
model as delta_encoder_benchmark. The host needs about 2.3x that in memory,
and the TPU chip about 1.3x.
"""

from collections.abc import Sequence
import json
import os
import statistics
import time

from absl import app
from absl import flags
import jax
import jax.numpy as jnp
import numpy as np
from tunix.experimental.delta_sync import _delta_encoder_ext
from tunix.experimental.delta_sync import sparsecore_delta_apply as sda
from tunix.experimental.delta_sync import synthetic_deltas


def _available_cpus() -> int:
  """Returns the CPUs this process may run on."""
  if hasattr(os, "sched_getaffinity"):
    return len(os.sched_getaffinity(0))
  # Not every platform reports the affinity mask.
  return os.cpu_count() or 1


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
_ENCODE_THREADS = flags.DEFINE_integer(
    "encode_threads",
    _available_cpus(),
    "Host threads the encoder uses.",
    lower_bound=1,
)


def _seconds(fn) -> float:
  start = time.perf_counter()
  fn()
  return time.perf_counter() - start


def run_benchmark(
    *,
    layers: int,
    experts: int,
    change_rate: float,
    block_sigma: float,
    repeats: int,
    encode_threads: int,
) -> dict[str, object]:
  """Times encode, h2d and apply, and checks the result bit for bit."""
  device = jax.devices()[0]
  if device.platform != "tpu":
    raise RuntimeError(f"needs a TPU with SparseCores, got {device.platform}")

  shapes, block_rows = synthetic_deltas.routed_expert_layout(layers, experts)
  old_bits, new_bits = synthetic_deltas.generate_all(
      shapes, block_rows, change_rate=change_rate, block_sigma=block_sigma
  )
  weight_bytes = sum(bits.nbytes for bits in old_bits)

  # Size every tensor of a shape for the fullest block of that shape, as a
  # trainer would when it binds the delta buffers.
  fullest: list[int] = []
  count_s = _seconds(
      lambda: fullest.extend(
          _delta_encoder_ext.fullest_block_changes(
              old_bits,
              new_bits,
              block_rows=block_rows,
              num_threads=encode_threads,
          )
      )
  )
  capacities = synthetic_deltas.capacities_by_shape(shapes, fullest)
  # Reused across rounds, like the trainer's persistent staging buffers.
  offsets = [
      np.empty((shape[0] // block, capacity), np.int32)
      for shape, block, capacity in zip(shapes, block_rows, capacities)
  ]
  values = [np.empty_like(o) for o in offsets]
  payload_bytes = sum(o.nbytes + v.nbytes for o, v in zip(offsets, values))

  changed_words = 0
  weights: list[jax.Array] = []
  device_offsets: list[jax.Array] = []
  device_values: list[jax.Array] = []
  times: dict[str, list[float]] = {
      "encode": [],
      "h2d": [],
      "apply": [],
      "total": [],
  }
  # The first iteration compiles the kernels and faults in the staging
  # buffers; it is not timed.
  for iteration in range(repeats + 1):
    del weights, device_offsets, device_values
    weights = [
        jax.device_put(bits.view(jnp.bfloat16), device) for bits in old_bits
    ]
    jax.block_until_ready(weights)

    start = time.perf_counter()
    changed_words = _delta_encoder_ext.encode_block_patches(
        old_bits,
        new_bits,
        block_rows=block_rows,
        offsets=offsets,
        values=values,
        num_threads=encode_threads,
    )
    encoded = time.perf_counter()
    device_offsets = [jax.device_put(o, device) for o in offsets]
    device_values = [jax.device_put(v, device) for v in values]
    jax.block_until_ready((device_offsets, device_values))
    on_device = time.perf_counter()
    weights = [
        sda.apply_block_patches(w, offsets=o, values=v)
        for w, o, v in zip(weights, device_offsets, device_values)
    ]
    jax.block_until_ready(weights)
    applied = time.perf_counter()
    if iteration > 0:
      times["encode"].append(encoded - start)
      times["h2d"].append(on_device - encoded)
      times["apply"].append(applied - on_device)
      times["total"].append(applied - start)

  bit_exact = all(
      np.array_equal(np.asarray(w).view(np.uint16), new)
      for w, new in zip(weights, new_bits)
  )

  result: dict[str, object] = {
      "device": device.device_kind,
      "host_cpus": _available_cpus(),
      "encode_threads": encode_threads,
      "tensors": len(shapes),
      "weights_gb": round(weight_bytes / 1e9, 2),
      "payload_gb": round(payload_bytes / 1e9, 3),
      "payload_gb_without_padding": round(changed_words * 8 / 1e9, 3),
      "changed_words": changed_words,
      "capacity_by_shape": {
          str(list(shape)): capacity
          for shape, capacity in zip(shapes, capacities)
      },
      "count_ms": round(count_s * 1e3, 1),
      "bit_exact": bit_exact,
  }
  for stage, seconds in times.items():
    result[f"{stage}_median_ms"] = round(statistics.median(seconds) * 1e3, 1)
    result[f"{stage}_min_ms"] = round(min(seconds) * 1e3, 1)
  return result


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  result = run_benchmark(
      layers=_LAYERS.value,
      experts=_EXPERTS.value,
      change_rate=_CHANGE_RATE.value,
      block_sigma=_BLOCK_SIGMA.value,
      repeats=_REPEATS.value,
      encode_threads=_ENCODE_THREADS.value,
  )
  print(json.dumps(result, indent=2), flush=True)
  if not result["bit_exact"]:
    raise RuntimeError("the patched weights differ from the new weights")


if __name__ == "__main__":
  app.run(main)
