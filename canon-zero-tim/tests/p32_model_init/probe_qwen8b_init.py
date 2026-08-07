#!/usr/bin/env python3
"""Materialize the Qwen3-8B DP16xTP4 training-state admission boundary.

This probe allocates the exact actor, AdamW, and gradient-accumulator state
shapes without loading a checkpoint and without running a forward, backward,
optimizer update, or training step.  It is a capacity and sharding gate, not a
numerical-training gate.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


_T1 = Path(__file__).resolve().parents[1] / "t1_tpu"
sys.path.insert(0, str(_T1))
from pathways_bootstrap import initialize_pathways  # pylint: disable=g-import-not-at-top


initialize_pathways()

from flax import nnx  # pylint: disable=g-import-not-at-top
import jax  # pylint: disable=g-import-not-at-top
import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
from jax.experimental import mesh_utils  # pylint: disable=g-import-not-at-top
from jax.sharding import Mesh  # pylint: disable=g-import-not-at-top
from jax.sharding import NamedSharding  # pylint: disable=g-import-not-at-top
from jax.sharding import PartitionSpec as P  # pylint: disable=g-import-not-at-top
import numpy as np  # pylint: disable=g-import-not-at-top
import optax  # pylint: disable=g-import-not-at-top

from tunix.models.qwen3 import model as model_lib  # pylint: disable=g-import-not-at-top
from tunix.rl import dp_training  # pylint: disable=g-import-not-at-top


_EXPECTED_LOGICAL_BYTES = {
    "model": 32_762_941_440,
    "optimizer": 65_525_882_884,
    "accumulator": 32_762_941_440,
}
_EXPECTED_LEAVES = {"model": 399, "optimizer": 799, "accumulator": 399}


def _required_int(environ: Mapping[str, str], name: str) -> int:
  value = environ.get(name, "")
  if not value:
    raise ValueError(f"{name} is required")
  try:
    return int(value)
  except ValueError as exc:
    raise ValueError(f"{name} must be an integer") from exc


def _topology_mesh(devices: list[jax.Device], dp: int, tp: int) -> Mesh:
  """Builds a full-slice topology-aware DP-by-TP mesh."""
  if len(devices) != dp * tp:
    raise RuntimeError(
        f"model init requires exactly dp*tp={dp * tp} devices, got "
        f"{len(devices)}"
    )
  try:
    arranged = mesh_utils.create_device_mesh(
        (dp, tp), devices, allow_split_physical_axes=True
    )
  except TypeError as exc:
    raise RuntimeError(
        "this JAX build lacks allow_split_physical_axes; refusing a logical "
        "reshape for the DP16xTP4 materialization gate"
    ) from exc
  array = np.asarray(arranged, dtype=object)
  source_ids = [int(device.id) for device in devices]
  arranged_ids = [int(device.id) for device in array.flat]
  if array.shape != (dp, tp):
    raise RuntimeError(f"topology-aware mesh shape changed: {array.shape}")
  if len(set(arranged_ids)) != len(arranged_ids):
    raise RuntimeError("topology-aware mesh repeats at least one device")
  if set(arranged_ids) != set(source_ids):
    raise RuntimeError("topology-aware mesh does not cover the visible slice")
  return Mesh(array, ("dp", "tp"))


def _index_shape(
    index: tuple[slice | int, ...], global_shape: tuple[int, ...]
) -> tuple[int, ...]:
  shape = []
  if len(index) != len(global_shape):
    raise ValueError(
        f"shard index rank changed: {len(index)} != {len(global_shape)}"
    )
  for item, dimension in zip(index, global_shape):
    if isinstance(item, slice):
      start = 0 if item.start is None else item.start
      stop = dimension if item.stop is None else item.stop
      step = 1 if item.step is None else item.step
      shape.append(len(range(start, stop, step)))
    else:
      if item not in range(dimension):
        raise ValueError(f"shard index {item} is outside dimension {dimension}")
  return tuple(shape)


def _with_memory_kind(sharding: NamedSharding, memory_kind: str) -> NamedSharding:
  if memory_kind not in ("device", "pinned_host"):
    raise ValueError(
        "state memory kind must be device or pinned_host, got "
        f"{memory_kind!r}"
    )
  if sharding.memory_kind == memory_kind:
    return sharding
  return sharding.with_memory_kind(memory_kind)


def materialize_zero_state(
    abstract_state: Any, *, memory_kind: str
) -> Any:
  """Materializes a ShapeDtypeStruct tree directly into its admitted shards."""

  def allocate(value):
    if not isinstance(value, jax.ShapeDtypeStruct):
      return value
    sharding = value.sharding
    if not isinstance(sharding, NamedSharding):
      raise ValueError("every abstract state leaf must have NamedSharding")
    sharding = _with_memory_kind(sharding, memory_kind)
    return jax.make_array_from_callback(
        value.shape,
        sharding,
        lambda index, dtype=value.dtype, shape=value.shape: np.zeros(
            _index_shape(index, shape), dtype=dtype
        ),
    )

  return jax.tree.map(allocate, abstract_state)


def _physical_bytes_by_device(state: Any) -> dict[int, int]:
  totals: dict[int, int] = {}
  arrays = [
      value for value in jax.tree.leaves(state) if isinstance(value, jax.Array)
  ]
  if not arrays:
    raise ValueError("materialized state has no JAX arrays")
  for value in arrays:
    for shard in value.addressable_shards:
      device_id = int(shard.device.id)
      totals[device_id] = totals.get(device_id, 0) + int(
          shard.data.size * shard.data.dtype.itemsize
      )
  return totals


def _uniform_physical_bytes(state: Any, expected_devices: int) -> int:
  totals = _physical_bytes_by_device(state)
  if len(totals) != expected_devices:
    raise ValueError(
        "state does not expose all addressable devices: "
        f"{len(totals)} != {expected_devices}"
    )
  unique = set(totals.values())
  if len(unique) != 1:
    raise ValueError(
        "DP replicas have unequal physical state bytes: "
        f"{sorted(unique)[:8]}"
    )
  return unique.pop()


def _abstract_training_state(
    mesh: Mesh, config: model_lib.ModelConfig | None = None
):
  config = model_lib.ModelConfig.qwen3_8b() if config is None else config
  config.dtype = jnp.bfloat16
  config.param_dtype = jnp.float32
  config.remat_config = model_lib.RematConfig.DECODER
  config.use_flash_attention = True
  config.flash_attention_block_size = 256
  config.shd_config = model_lib.ShardingConfig.get_data_parallel_sharding()

  with jax.set_mesh(mesh):
    abstract_model = nnx.eval_shape(
        lambda: model_lib.Qwen3(config, rngs=nnx.Rngs(params=0))
    )
  graphdef, raw_abstract_state = nnx.split(abstract_model)
  model_shardings = nnx.get_named_sharding(raw_abstract_state, mesh)
  abstract_params = jax.tree.map(
      lambda value, sharding: jax.ShapeDtypeStruct(
          value.shape, value.dtype, sharding=sharding
      ),
      raw_abstract_state,
      model_shardings,
  )

  optimizer = optax.adamw(
      learning_rate=1.0e-6,
      b1=0.9,
      b2=0.95,
      weight_decay=0.0,
  )
  abstract_optimizer = jax.eval_shape(optimizer.init, abstract_params)
  abstract_optimizer = dp_training.attach_adam_state_shardings(
      abstract_optimizer, params=abstract_params, mesh=mesh
  )
  abstract_accumulator = jax.tree.map(
      lambda value: jax.ShapeDtypeStruct(
          value.shape, jnp.float32, sharding=value.sharding
      ),
      abstract_params,
  )
  return config, graphdef, abstract_params, abstract_optimizer, abstract_accumulator


def materialize_training_state(
    mesh: Mesh,
    *,
    optimizer_memory_kind: str,
    config: model_lib.ModelConfig | None = None,
) -> tuple[Any, Any, Any, Any, dict[str, dict[str, Any]]]:
  """Materializes exact Qwen3-8B actor, AdamW, and accumulator states."""
  (
      config,
      graphdef,
      abstract_params,
      abstract_optimizer,
      abstract_accumulator,
  ) = _abstract_training_state(mesh, config)
  model_state = materialize_zero_state(abstract_params, memory_kind="device")
  optimizer_state = materialize_zero_state(
      abstract_optimizer, memory_kind=optimizer_memory_kind
  )
  accumulator_state = materialize_zero_state(
      abstract_accumulator, memory_kind="device"
  )
  model = nnx.merge(graphdef, model_state)
  states = (nnx.state(model, nnx.Param), optimizer_state, accumulator_state)
  jax.block_until_ready(states)
  inventory = dp_training.inspect_training_state_inventories(
      model=states[0], optimizer=states[1], accumulator=states[2]
  )
  return config, states[0], states[1], states[2], inventory


def _validate_inventory(
    inventory: Mapping[str, Mapping[str, Any]], *, optimizer_memory_kind: str
) -> None:
  if set(inventory) != set(_EXPECTED_LEAVES):
    raise ValueError(f"training state classes changed: {sorted(inventory)}")
  expected_memory = {
      "model": ("device",),
      "optimizer": (optimizer_memory_kind,),
      "accumulator": ("device",),
  }
  for label, summary in inventory.items():
    if summary["leaves"] != _EXPECTED_LEAVES[label]:
      raise ValueError(
          f"{label} leaf count changed: {summary['leaves']} != "
          f"{_EXPECTED_LEAVES[label]}"
      )
    if summary["logical_bytes"] != _EXPECTED_LOGICAL_BYTES[label]:
      raise ValueError(
          f"{label} logical bytes changed: {summary['logical_bytes']} != "
          f"{_EXPECTED_LOGICAL_BYTES[label]}"
      )
    if summary["dp_partitioned_leaves"] != 0:
      raise ValueError(f"{label} is sharded over DP")
    if summary["tp_partitioned_leaves"] <= 0:
      raise ValueError(f"{label} has no TP-sharded leaves")
    if tuple(summary["memory_kinds"]) != expected_memory[label]:
      raise ValueError(
          f"{label} memory kinds changed: {summary['memory_kinds']} != "
          f"{expected_memory[label]}"
      )


def main() -> int:
  env = os.environ
  if env.get("CANON_MODE") != "model-init-only":
    raise RuntimeError("probe requires CANON_MODE=model-init-only")
  if env.get("CANON_P32_MODEL_INIT_ONLY") != "1":
    raise RuntimeError("CANON_P32_MODEL_INIT_ONLY=1 is required")
  if env.get("JOBSET_RESTART_ATTEMPT") != "0":
    raise RuntimeError("model-init admission requires JOBSET_RESTART_ATTEMPT=0")
  if env.get("CANON_P32_MODEL_STATE_KIND") != "zero-structural":
    raise RuntimeError("model-init state kind must remain zero-structural")

  dp = _required_int(env, "CANON_DP_SIZE")
  tp = _required_int(env, "CANON_TP_SIZE")
  total = _required_int(env, "CANON_TOTAL_DEVICES")
  if (dp, tp, total) != (16, 4, 64):
    raise RuntimeError(
        f"model-init topology must remain DP16xTP4 on 64 devices, got "
        f"DP{dp}xTP{tp} on {total}"
    )
  optimizer_memory_kind = env.get(
      "CANON_P32_OPTIMIZER_MEMORY_KIND", ""
  )
  if optimizer_memory_kind != "pinned_host":
    raise RuntimeError("model-init requires pinned-host optimizer state")
  wandb_identity = {
      "project": env.get("CANON_WANDB_PROJECT", ""),
      "group": env.get("CANON_WANDB_GROUP", ""),
      "run_name": env.get("CANON_WANDB_RUN_NAME", ""),
  }
  if not all(wandb_identity.values()):
    raise RuntimeError("non-secret W&B identity is incomplete")

  devices = list(jax.devices())
  mesh = _topology_mesh(devices, dp, tp)
  mesh_ids = tuple(int(device.id) for device in mesh.devices.flat)
  print(
      f"[P32.INIT] START dp={dp} tp={tp} devices={len(devices)} "
      "checkpoint_loaded=0 forward=0 backward=0 update=0",
      flush=True,
  )
  print(
      f"[P32.INIT] MESH shape={mesh.devices.shape} unique={len(set(mesh_ids))} "
      f"full_slice={int(set(mesh_ids) == {int(d.id) for d in devices})}",
      flush=True,
  )

  config, model, optimizer, accumulator, inventory = materialize_training_state(
      mesh, optimizer_memory_kind=optimizer_memory_kind
  )
  _validate_inventory(
      inventory, optimizer_memory_kind=optimizer_memory_kind
  )
  physical_bytes = {
      "model": _uniform_physical_bytes(model, total),
      "optimizer": _uniform_physical_bytes(optimizer, total),
      "accumulator": _uniform_physical_bytes(accumulator, total),
  }
  record = {
      "attempt": 0,
      "topology": {
          "dp": dp,
          "tp": tp,
          "devices": total,
          "mesh_shape": list(mesh.devices.shape),
          "unique_devices": len(set(mesh_ids)),
          "full_slice": set(mesh_ids) == {int(device.id) for device in devices},
      },
      "model": {
          "name": "qwen3-8b",
          "layers": config.num_layers,
          "vocab": config.vocab_size,
          "embed": config.embed_dim,
          "hidden": config.hidden_dim,
          "heads": config.num_heads,
          "kv_heads": config.num_kv_heads,
          "head_dim": config.head_dim,
          "compute_dtype": str(config.dtype),
          "param_dtype": str(config.param_dtype),
          "checkpoint_loaded": False,
          "state_kind": "zero-structural",
      },
      "inventory": inventory,
      "physical_bytes_per_device": physical_bytes,
      "optimizer": {
          "name": "adamw",
          "learning_rate": 1.0e-6,
          "b1": 0.9,
          "b2": 0.95,
          "weight_decay": 0.0,
          "memory_kind": optimizer_memory_kind,
          "commits": 0,
      },
      "execution": {
          "forward": 0,
          "backward": 0,
          "optimizer_updates": 0,
          "training_steps": 0,
      },
      "wandb": {
          **wandb_identity,
          "network_initialized": False,
      },
  }
  print(f"[P32.INIT] JSON {json.dumps(record, sort_keys=True)}", flush=True)
  print("[P32.INIT] VERDICT PASS", flush=True)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
