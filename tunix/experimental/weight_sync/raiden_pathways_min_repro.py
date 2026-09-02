#!/usr/bin/env python3

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

"""Minimal Pathways Raiden src->dst repro.

This script isolates the exact weight-sync path used by the rollout stack:

1. Build a tiny nested source state and a deliberately different destination
   state on JAX meshes.
2. Bind the source with Raiden FFI D2H and the destination with Raiden FFI H2D.
3. Register both work units with a local RaidenHandler controller.
4. Execute one transfer and install the result on device.
5. Compare source/destination checksums and tensor values.

Run this inside the Pathways proxy runtime where `JAX_PLATFORMS` includes
`proxy` and the TPU-sync FFI wheel is installed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any

from absl import logging
import jax
from jax import numpy as jnp
import numpy as np

from tunix.experimental.weight_sync.raiden_handler import RaidenHandler
from tunix.experimental.weight_sync.raiden_synchronizer import RaidenSynchronizer
from tunix.experimental.weight_sync import raiden_synchronizer


def _parse_args(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--req_id", default="raiden-pathways-min-repro")
  parser.add_argument("--parallelism", type=int, default=2)
  parser.add_argument("--log_level", default="INFO")
  return parser.parse_args(argv)


def _require_pathways_ffi() -> None:
  if raiden_synchronizer._raiden_ffi is None:
    raise RuntimeError(
        "TPU-sync FFI is unavailable. Install the Pathways Raiden wheel first."
    )
  if "proxy" not in os.environ.get("JAX_PLATFORMS", ""):
    raise RuntimeError(
        "This repro targets the Pathways proxy runtime. Set JAX_PLATFORMS to"
        " include 'proxy' and run it inside the TPU worker environment."
    )
  if jax.device_count() < 2:
    raise RuntimeError(
        "This repro needs at least 2 visible JAX devices to exercise Raiden."
    )


def _initialize_jax_runtime() -> None:
  if "proxy" in os.environ.get("JAX_PLATFORMS", "") and os.environ.get(
      "JAX_BACKEND_TARGET"
  ):
    import pathwaysutils

    logging.info(
        "Initializing Pathways runtime via pathwaysutils for %s",
        os.environ["JAX_BACKEND_TARGET"],
    )
    pathwaysutils.initialize()


def _make_meshes() -> tuple[jax.sharding.Mesh, jax.sharding.Mesh, int]:
  devices = np.array(jax.devices())
  device_count = len(devices)
  src_mesh = jax.sharding.Mesh(devices, ("data",))

  if device_count % 4 == 0:
    dst_cols = 4
  elif device_count % 2 == 0:
    dst_cols = 2
  else:
    dst_cols = 1

  if dst_cols == 1:
    return src_mesh, src_mesh, dst_cols

  dst_rows = device_count // dst_cols
  dst_mesh = jax.sharding.Mesh(
      devices.reshape((dst_rows, dst_cols)), ("x", "y")
  )
  return src_mesh, dst_mesh, dst_cols


def _put_source_state(mesh: jax.sharding.Mesh) -> dict[str, Any]:
  device_count = jax.device_count()
  sharded = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec("data", None)
  )
  sharded_vec = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec("data")
  )
  return {
      "block": {
          "w": jax.device_put(
              jnp.arange(device_count * 8, dtype=jnp.float32).reshape(
                  device_count, 8
              )
              + 1.0,
              sharded,
          ),
          "v": jax.device_put(
              jnp.linspace(1.0, 2.0, device_count * 4, dtype=jnp.float32)
              .reshape(device_count, 4),
              sharded,
          ),
      },
      "norm": jax.device_put(
          jnp.linspace(3.0, 4.0, device_count, dtype=jnp.float32),
          sharded_vec,
      ),
  }


def _put_destination_state(
    mesh: jax.sharding.Mesh, dst_cols: int
) -> dict[str, Any]:
  device_count = jax.device_count()
  if dst_cols == 1:
    sharded = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("data", None)
    )
    sharded_vec = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("data")
    )
  else:
    sharded = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("x", "y")
    )
    sharded_vec = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x"))

  return {
      "block": {
          "w": jax.device_put(jnp.zeros((device_count, 8), jnp.float32), sharded),
          "v": jax.device_put(
              -jnp.ones((device_count, 4), jnp.float32), sharded
          ),
      },
      "norm": jax.device_put(
          -jnp.ones((device_count,), jnp.float32),
          sharded_vec,
      ),
  }


def _checksums_by_name(sync: RaidenSynchronizer) -> dict[str, float]:
  sums: dict[str, float] = {}
  for name, arr in zip(sync.names, sync.arrays):
    sums[name] = float(jnp.sum(jnp.abs(arr).astype(jnp.float32)))
  return sums


def _assert_initial_difference(
    src_sync: RaidenSynchronizer, dst_sync: RaidenSynchronizer
) -> None:
  diffs = []
  for name, src_arr, dst_arr in zip(src_sync.names, src_sync.arrays, dst_sync.arrays):
    equal = bool(jnp.allclose(src_arr, dst_arr))
    diffs.append((name, equal))
  if all(equal for _, equal in diffs):
    raise AssertionError(
        "Destination already matches source before transfer; repro would be"
        " inconclusive."
    )


def _assert_synced(
    src_sync: RaidenSynchronizer, dst_sync: RaidenSynchronizer
) -> None:
  mismatches = []
  for name, src_arr, dst_arr in zip(src_sync.names, src_sync.arrays, dst_sync.arrays):
    if not bool(jnp.allclose(src_arr, dst_arr)):
      max_abs = float(jnp.max(jnp.abs(src_arr - dst_arr)))
      mismatches.append((name, max_abs))
  if mismatches:
    details = ", ".join(f"{name}: max_abs={max_abs}" for name, max_abs in mismatches)
    raise AssertionError(f"Raiden src->dst mismatch after h2d: {details}")


async def _run_once(args: argparse.Namespace) -> None:
  src_mesh, dst_mesh, dst_cols = _make_meshes()
  src_state = _put_source_state(src_mesh)
  dst_state = _put_destination_state(dst_mesh, dst_cols)

  handler = RaidenHandler(port=0, transfer_parallelism=args.parallelism)
  try:
    src_sync = RaidenSynchronizer("trainer", parallelism=args.parallelism)
    dst_sync = RaidenSynchronizer(
        "rollout", auto_h2d=True, parallelism=args.parallelism
    )
    src_sync.bind(src_state)
    dst_sync.bind(dst_state)

    logging.info("Source metadata: %s", src_sync.work_unit_metadata())
    logging.info("Destination metadata: %s", dst_sync.work_unit_metadata())
    logging.info("Source checksums before transfer: %s", src_sync.checksums())
    logging.info("Destination checksums before transfer: %s", dst_sync.checksums())
    _assert_initial_difference(src_sync, dst_sync)

    src_sync.d2h()
    handler.register_work_unit(src_sync.work_unit_metadata())
    handler.register_work_unit(dst_sync.work_unit_metadata())

    result = await asyncio.to_thread(
        handler.transfer,
        req_id=args.req_id,
        src_units=[src_sync.work_unit_metadata().unit],
        dst_units=[dst_sync.work_unit_metadata().unit],
    )
    logging.info("Transfer result: %s", result)

    dst_sync.h2d()
    logging.info("Destination checksums after transfer: %s", dst_sync.checksums())
    logging.info("Per-tensor checksum diff: %s", {
        name: (_checksums_by_name(src_sync)[name], _checksums_by_name(dst_sync)[name])
        for name in src_sync.names
    })
    _assert_synced(src_sync, dst_sync)
    logging.info("Raiden minimal repro succeeded: destination matches source.")
  finally:
    handler.close()


def main(argv: list[str] | None = None) -> None:
  args = _parse_args(argv or [])
  logging.set_verbosity(args.log_level)
  logging.use_absl_handler()
  logging.info("JAX_PLATFORMS=%s", os.environ.get("JAX_PLATFORMS", ""))
  _initialize_jax_runtime()
  logging.info("Visible JAX devices=%d", jax.device_count())
  _require_pathways_ffi()
  asyncio.run(_run_once(args))


if __name__ == "__main__":
  import sys
  main(sys.argv[1:])