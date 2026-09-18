#!/usr/bin/env python3
"""Pin the installed attention shim's exact P59 manual-mesh predicate."""

from __future__ import annotations

import importlib.util
import os


os.environ["CANON_P59_RANK_PARALLEL_BACKWARD"] = "1"
os.environ.pop("CANON_P32_WORKLOAD", None)
os.environ.pop("CANON_P66_BACKWARD_ARM", None)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import canon_shim_root  # noqa: E402


ENGINE_AXES = (
    "data",
    "attn_dp",
    "attn_dp_expert",
    "expert",
    "model",
    "dcp",
)


def _load_installed_attention():
  path = canon_shim_root.resolve("attn_iface_patched.py")
  spec = importlib.util.spec_from_file_location(
      "_p59_attention_context_predicate", path
  )
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load installed attention interface from {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  if not hasattr(module, "_p59_local_attention_context"):
    raise RuntimeError("installed attention shim lacks P59 context predicate")
  return module


def _meshes(dp_size: int, tp_size: int):
  devices = np.asarray(jax.devices()[: dp_size * tp_size])
  manual = jax.sharding.Mesh(
      devices.reshape(dp_size, tp_size), ("data", "model")
  )
  engine = jax.sharding.Mesh(
      devices.reshape(dp_size, 1, 1, 1, tp_size, 1), ENGINE_AXES
  )
  return manual, engine


def _run_manual_context(
    attention, dp_size: int, tp_size: int, *, expected: bool
):
  manual, engine = _meshes(dp_size, tp_size)
  value = jax.device_put(
      jnp.arange(dp_size * tp_size, dtype=jnp.float32).reshape(
          dp_size, tp_size
      ),
      jax.sharding.NamedSharding(
          manual, jax.sharding.PartitionSpec("data", "model")
      ),
  )
  observed = []

  def local_fn(local_value):
    observed.append(attention._p59_local_attention_context(engine))
    return local_value

  mapped = jax.shard_map(
      local_fn,
      mesh=manual,
      in_specs=jax.sharding.PartitionSpec("data", "model"),
      out_specs=jax.sharding.PartitionSpec("data", "model"),
      check_vma=True,
  )
  with jax.transfer_guard("disallow"):
    result = jax.block_until_ready(mapped(value))
  if not observed or any(item is not expected for item in observed):
    raise AssertionError(
        f"P59 attention context changed for DP{dp_size}xTP{tp_size}: "
        f"observed={observed} expected={expected}"
    )
  result_host = np.asarray(result)
  value_host = np.asarray(value)
  if (
      result_host.shape != value_host.shape
      or result_host.dtype != value_host.dtype
      or result_host.tobytes() != value_host.tobytes()
  ):
    raise AssertionError("P59 attention context probe changed array bytes")


def main() -> None:
  if len(jax.devices()) < 4:
    raise RuntimeError("P59 attention context probe requires four devices")
  attention = _load_installed_attention()

  # The production singleton carrier has no P66/GSM8K diagnostic selectors.
  # Its checked outer map and live attention engine mesh are nevertheless exact.
  _run_manual_context(attention, 1, 4, expected=True)
  # The established non-singleton geometry remains admitted.
  _run_manual_context(attention, 2, 2, expected=True)

  # Flag presence alone must not select the local path in serving context.
  _, serving_engine = _meshes(1, 4)
  if attention._p59_local_attention_context(serving_engine):
    raise AssertionError("ordinary serving selected P59 attention context")

  os.environ["CANON_P59_RANK_PARALLEL_BACKWARD"] = "0"
  _run_manual_context(attention, 1, 4, expected=False)
  os.environ["CANON_P59_RANK_PARALLEL_BACKWARD"] = "1"

  # Exact manual axes are insufficient when they disagree with the live
  # engine topology. Preserve fail-closed behavior.
  manual, _ = _meshes(1, 4)
  _, mismatched_engine = _meshes(2, 2)
  value = jax.device_put(
      jnp.arange(4, dtype=jnp.float32).reshape(1, 4),
      jax.sharding.NamedSharding(
          manual, jax.sharding.PartitionSpec("data", "model")
      ),
  )

  def mismatched(local_value):
    attention._p59_local_attention_context(mismatched_engine)
    return local_value

  mapped = jax.shard_map(
      mismatched,
      mesh=manual,
      in_specs=jax.sharding.PartitionSpec("data", "model"),
      out_specs=jax.sharding.PartitionSpec("data", "model"),
      check_vma=True,
  )
  try:
    mapped(value)
  except RuntimeError as error:
    if "context and engine topology differ" not in str(error):
      raise
  else:
    raise AssertionError(
        "mismatched attention engine topology negative did not fire"
    )

  print(
      "P59_LOCAL_ATTENTION_CONTEXT_PASS "
      "topologies=DP1xTP4,DP2xTP2 checked_vma=1 p66_arm=empty "
      "host_transfers=0 serving_negative=1 flag_off_negative=1 "
      "mesh_mismatch_negative=1",
      flush=True,
  )


if __name__ == "__main__":
  main()
