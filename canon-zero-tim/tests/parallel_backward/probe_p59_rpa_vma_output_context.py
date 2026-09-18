#!/usr/bin/env python3
"""Pin RPA Pallas output VMA metadata in exact P59 manual meshes."""

from __future__ import annotations

import os


os.environ["CANON_P66_P59_CHECK_VMA"] = "1"
os.environ["CANON_P67_P66_VMA_P59_ONLY"] = "1"
os.environ["CANON_P59_RANK_PARALLEL_BACKWARD"] = "1"
os.environ.pop("CANON_P32_WORKLOAD", None)
os.environ.pop("CANON_P66_BACKWARD_ARM", None)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from tpu_inference.kernels.ragged_paged_attention.v3 import (  # noqa: E402
    kernel as rpa_kernel,
)


def _mesh(dp_size: int, tp_size: int):
  devices = np.asarray(jax.devices()[: dp_size * tp_size])
  return jax.sharding.Mesh(
      devices.reshape(dp_size, tp_size), ("data", "model")
  )


def _run_manual_context(
    dp_size: int, tp_size: int, *, enabled: bool
) -> None:
  mesh = _mesh(dp_size, tp_size)
  sharding = jax.sharding.NamedSharding(
      mesh, jax.sharding.PartitionSpec("data", "model")
  )
  value = jax.device_put(
      jnp.arange(dp_size * tp_size, dtype=jnp.float32).reshape(
          dp_size, tp_size
      ),
      sharding,
  )
  observed = []

  def local_fn(local_value):
    manual_axis_type = rpa_kernel.p66_vma_output_manual_axis_type(
        jax, local_value
    )
    observed.append(manual_axis_type)
    if enabled:
      if manual_axis_type is None:
        raise AssertionError("RPA Pallas output lost manual_axis_type")
      input_type = jax.typeof(local_value).mat
      if manual_axis_type != input_type:
        raise AssertionError(
            "RPA Pallas output MAT differs from its input: "
            f"output={manual_axis_type} input={input_type}"
        )
    elif manual_axis_type is not None:
      raise AssertionError("flag-off RPA Pallas output retained VMA metadata")
    return local_value

  mapped = jax.shard_map(
      local_fn,
      mesh=mesh,
      in_specs=jax.sharding.PartitionSpec("data", "model"),
      out_specs=jax.sharding.PartitionSpec("data", "model"),
      check_vma=True,
  )
  with jax.transfer_guard("disallow"):
    result = jax.block_until_ready(mapped(value))
  if not observed:
    raise AssertionError("RPA Pallas output probe did not trace the helper")
  result_host = np.asarray(result)
  value_host = np.asarray(value)
  if (
      result_host.shape != value_host.shape
      or result_host.dtype != value_host.dtype
      or result_host.tobytes() != value_host.tobytes()
  ):
    raise AssertionError("RPA Pallas output context probe changed array bytes")


def main() -> None:
  if len(jax.devices()) < 4:
    raise RuntimeError("RPA VMA output probe requires four devices")
  if not hasattr(rpa_kernel, "p66_vma_output_manual_axis_type"):
    raise RuntimeError("installed RPA kernel lacks shared VMA output helper")

  _run_manual_context(1, 4, enabled=True)
  _run_manual_context(2, 2, enabled=True)

  if rpa_kernel.p66_vma_output_manual_axis_type(
      jax, jnp.ones((1,), dtype=jnp.float32)
  ) is not None:
    raise AssertionError("ordinary serving retained RPA VMA metadata")

  os.environ["CANON_P66_P59_CHECK_VMA"] = "0"
  _run_manual_context(1, 4, enabled=False)

  print(
      "P59_RPA_VMA_OUTPUT_CONTEXT_PASS "
      "topologies=DP1xTP4,DP2xTP2 checked_vma=1 p66_arm=empty "
      "host_transfers=0 serving_negative=1 flag_off_negative=1",
      flush=True,
  )


if __name__ == "__main__":
  main()
