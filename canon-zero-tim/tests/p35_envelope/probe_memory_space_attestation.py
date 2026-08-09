#!/usr/bin/env python3
"""Exercises exact weight attestation across TPU host and device memory."""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
import numpy as np

from tunix.rl import canonical_qwen3_adapter


def main() -> None:
  mesh = Mesh(np.asarray(jax.devices()), ("tp",))
  device_sharding = NamedSharding(mesh, P())
  host_sharding = device_sharding.with_memory_kind("pinned_host")

  base = jnp.asarray([0.0, 1.0], jnp.float32)
  host_value = jax.device_put(base, host_sharding)
  device_value = jax.device_put(base, device_sharding)
  try:
    canonical_qwen3_adapter._exact_leaf_bits_equal(  # pylint: disable=protected-access
        host_value, device_value
    )
  except ValueError as exc:
    if "memory_space of all inputs" not in str(exc):
      raise
    print("P35_MEMORY direct_mixed_memory_rejected=1", flush=True)
  else:
    raise AssertionError("the direct mixed-memory negative control did not fail")

  if not canonical_qwen3_adapter._bitwise_arrays_equal(  # pylint: disable=protected-access
      host_value, device_value
  ):
    raise AssertionError("equal mixed-memory values did not compare exactly")
  if not canonical_qwen3_adapter._bitwise_arrays_equal(  # pylint: disable=protected-access
      device_value, host_value
  ):
    raise AssertionError(
        "reversed equal mixed-memory values did not compare exactly"
    )

  negative_zero = jax.device_put(
      jnp.asarray([-0.0, 1.0], jnp.float32), device_sharding
  )
  if canonical_qwen3_adapter._bitwise_arrays_equal(  # pylint: disable=protected-access
      host_value, negative_zero
  ):
    raise AssertionError("signed-zero negative control compared equal")

  one_bit = np.asarray([0.0, 1.0], np.float32)
  one_bit.view(np.uint32)[1] ^= np.uint32(1)
  one_bit_device = jax.device_put(jnp.asarray(one_bit), device_sharding)
  if canonical_qwen3_adapter._bitwise_arrays_equal(  # pylint: disable=protected-access
      host_value, one_bit_device
  ):
    raise AssertionError("one-bit negative control compared equal")

  print(
      "P35_MEMORY result=PASS devices="
      f"{len(jax.devices())} before=pinned_host,device after=device,device "
      "equal=1 reversed_equal=1 signed_zero_equal=0 one_bit_equal=0",
      flush=True,
  )


if __name__ == "__main__":
  main()
