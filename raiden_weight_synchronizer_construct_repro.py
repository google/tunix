#!/usr/bin/env python3

"""Minimal McJax repro for WeightSynchronizer construction."""

from __future__ import annotations

import argparse
import sys
import time

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
from tpu_sync.api.jax import weight_synchronizer


P = jax.sharding.PartitionSpec


def _parse_args(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--parallelism", type=int, default=4)
  parser.add_argument("--unsafe_skip_buffer_lock", action="store_true")
  parser.add_argument("--auto_h2d", action="store_true")
  parser.add_argument("--sleep_s", type=float, default=1.0)
  return parser.parse_args(argv)


def _specs() -> list[tuple[tuple[int, ...], jax.sharding.PartitionSpec, str]]:
  return [
      ((8, 8), P("fsdp", "tp"), "block.w_in"),
      ((8, 8), P("tp", "fsdp"), "block.w_out"),
      ((8,), P("fsdp"), "block.norm"),
  ]


def _build_mesh() -> jax.sharding.Mesh:
  devices = np.array(jax.devices())
  if len(devices) < 16:
    raise ValueError(f"Expected at least 16 visible devices, got {len(devices)}")
  return jax.sharding.Mesh(devices[:16].reshape((4, 4)), ("fsdp", "tp"))


def _build_arrays(mesh: jax.sharding.Mesh) -> list[jax.Array]:
  arrays: list[jax.Array] = []
  for shape, pspec, _ in _specs():
    sharding = jax.sharding.NamedSharding(mesh, pspec)
    arrays.append(
      jax.jit(
          lambda s=shape: jnp.zeros(s, dtype=jnp.float32),
          out_shardings=sharding,
      )()
    )
  for arr in arrays:
    arr.block_until_ready()
  return arrays


def main(argv: list[str] | None = None) -> None:
  jax.distributed.initialize()

  args = _parse_args(argv or sys.argv[1:])
  logging.set_verbosity("INFO")
  logging.use_absl_handler()

  logging.info("Starting minimal WeightSynchronizer construction repro")
  logging.info(
      "JAX runtime: process_index=%d process_count=%d local_device_count=%d device_count=%d",
      jax.process_index(),
      jax.process_count(),
      jax.local_device_count(),
      jax.device_count(),
  )

  mesh = _build_mesh()
  arrays = _build_arrays(mesh)
  logging.info("Mesh shape=%s", dict(mesh.shape))
  logging.info("Constructing WeightSynchronizer for %d arrays", len(arrays))

  ws = weight_synchronizer.WeightSynchronizer(
      arrays,
      local_port=0,
      parallelism=args.parallelism,
      unsafe_skip_buffer_lock=args.unsafe_skip_buffer_lock,
      listener_port=0,
      bind_ip=None,
      auto_h2d=args.auto_h2d,
  )

  logging.info(
      "WeightSynchronizer constructed: local_port=%s listener_port=%s num_shards=%s num_layers=%s slice_byte_size=%s",
      ws.local_port,
      ws.listener_port,
      ws.num_shards,
      ws.num_layers,
      ws.slice_byte_size,
  )
  logging.info("Local endpoints: %s", ws.get_local_endpoints())
  time.sleep(args.sleep_s)


if __name__ == "__main__":
  main(sys.argv[1:])
