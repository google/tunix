#!/usr/bin/env python3
"""Pin the installed projection shim's exact P59 manual-mesh predicate."""

from __future__ import annotations

import os

for name in (
    "CANON_PALLAS_ALL_PROJ",
    "CANON_PALLAS_ALL_RMSNORM",
    "CANON_PALLAS_SWIGLU",
    "CANON_PALLAS_MPAD",
    "CANON_PALLAS_SWIGLU_MPAD",
    "CANON_PALLAS_CANONICAL_VJP",
    "CANON_FIXED_AR",
    "CANON_FIXED_AR_EMBED",
    "CANON_FIXED_AR_GATHER",
    "CANON_P59_RANK_PARALLEL_BACKWARD",
):
  os.environ[name] = "1"
os.environ.update({
    "CANON_QWEN3_HIDDEN_SIZE": "2048",
    "CANON_QWEN3_INTERMEDIATE_SIZE": "6144",
    "CANON_QWEN3_NUM_ATTENTION_HEADS": "16",
    "CANON_QWEN3_NUM_KV_HEADS": "8",
    "CANON_QWEN3_HEAD_DIM": "128",
    "CANON_QWEN3_TP_SIZE": "4",
})
os.environ.pop("CANON_P66_BACKWARD_ARM", None)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import linear_p22xk as linear  # noqa: E402
import numpy as np  # noqa: E402


ENGINE_AXES = (
    "data",
    "attn_dp",
    "attn_dp_expert",
    "expert",
    "model",
    "dcp",
)


def _projection_shim():
  shim = linear.P22XK_LINEAR_BASE.P22XI_XF_MODULE
  if not hasattr(shim, "_p59_local_tp_context"):
    raise RuntimeError("installed projection shim lacks P59 context predicate")
  return shim


def _meshes(dp_size: int, tp_size: int):
  devices = np.asarray(jax.devices()[: dp_size * tp_size])
  manual = jax.sharding.Mesh(
      devices.reshape(dp_size, tp_size), ("data", "model")
  )
  engine = jax.sharding.Mesh(
      devices.reshape(dp_size, 1, 1, 1, tp_size, 1), ENGINE_AXES
  )
  return manual, engine


def _run_manual_context(dp_size: int, tp_size: int, *, expected: bool):
  shim = _projection_shim()
  manual, engine = _meshes(dp_size, tp_size)
  shim.base._CANON_MESH = engine
  shim.base._CANON_TP_AXIS = "model"
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
    observed.append(shim._p59_local_tp_context())
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
        f"P59 projection context changed for DP{dp_size}xTP{tp_size}: "
        f"observed={observed} expected={expected}"
    )
  result_host = np.asarray(result)
  value_host = np.asarray(value)
  if (
      result_host.shape != value_host.shape
      or result_host.dtype != value_host.dtype
      or result_host.tobytes() != value_host.tobytes()
  ):
    raise AssertionError("P59 projection context probe changed array bytes")


def main() -> None:
  if len(jax.devices()) < 4:
    raise RuntimeError("P59 projection context probe requires four devices")
  shim = _projection_shim()

  # The production singleton carrier has no P66 diagnostic selector.  Its
  # checked outer P59 map and live engine mesh are nevertheless exact.
  _run_manual_context(1, 4, expected=True)
  # The established non-singleton geometry remains admitted.
  _run_manual_context(2, 2, expected=True)

  # Flag presence alone must not select the local path in serving context.
  if shim._p59_local_tp_context():
    raise AssertionError("ordinary serving selected P59 projection context")

  os.environ["CANON_P59_RANK_PARALLEL_BACKWARD"] = "0"
  _run_manual_context(1, 4, expected=False)
  os.environ["CANON_P59_RANK_PARALLEL_BACKWARD"] = "1"

  # Exact manual axes are insufficient when they disagree with the live
  # engine topology.  Preserve fail-closed behavior.
  manual, _ = _meshes(1, 4)
  _, mismatched_engine = _meshes(2, 2)
  shim.base._CANON_MESH = mismatched_engine
  shim.base._CANON_TP_AXIS = "model"
  value = jax.device_put(
      jnp.arange(4, dtype=jnp.float32).reshape(1, 4),
      jax.sharding.NamedSharding(
          manual, jax.sharding.PartitionSpec("data", "model")
      ),
  )

  def mismatched(local_value):
    shim._p59_local_tp_context()
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
    raise AssertionError("mismatched engine topology negative did not fire")

  print(
      "P59_LOCAL_PROJECTION_CONTEXT_PASS "
      "topologies=DP1xTP4,DP2xTP2 checked_vma=1 p66_arm=empty "
      "host_transfers=0 serving_negative=1 flag_off_negative=1 "
      "mesh_mismatch_negative=1",
      flush=True,
  )


if __name__ == "__main__":
  main()
