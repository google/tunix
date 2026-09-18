#!/usr/bin/env python3
"""Real-v5p save/restore gate for the P45 checkpoint machinery.

This is deliberately a small DP1xTP4 device test.  It proves that Tunix can
save and restore sharded model parameters, device-resident Adam state, global
step metadata, the exact resume contract, the interval policy, and LatestN(1)
on the pinned one-host runtime.  It does not claim that a DP8xTP8 Pathways GCS
checkpoint has been restored; that remains a target-run gate.
"""

from __future__ import annotations

import pathlib
import tempfile

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from orbax.checkpoint import v1 as ocp

from tunix.rl import frozenlake_checkpoint
from tunix.sft import checkpoint_manager
from tunix.sft import checkpoint_options


class _ToyActor(nnx.Module):

  def __init__(self, rngs: nnx.Rngs):
    init = nnx.initializers.lecun_normal()
    self.up = nnx.Linear(
        8,
        16,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(init, ("fsdp", "tp")),
    )
    self.down = nnx.Linear(
        16,
        8,
        rngs=rngs,
        kernel_init=nnx.with_partitioning(init, ("tp", "fsdp")),
    )

  def __call__(self, x):
    return self.down(nnx.gelu(self.up(x)))


def _create_actor(mesh: jax.sharding.Mesh) -> _ToyActor:
  @nnx.jit
  def create():
    model = _ToyActor(nnx.Rngs(0))
    state = nnx.state(model)
    specs = nnx.get_partition_spec(state)
    nnx.update(model, jax.lax.with_sharding_constraint(state, specs))
    return model

  with jax.set_mesh(mesh):
    return create()


def _host_snapshot(state):
  return jax.tree.map(lambda value: np.asarray(jax.device_get(value)).copy(), state)


def _assert_exact(label: str, expected, actual) -> None:
  expected_leaves, expected_tree = jax.tree.flatten(expected)
  actual_leaves, actual_tree = jax.tree.flatten(actual)
  if expected_tree != actual_tree or len(expected_leaves) != len(actual_leaves):
    raise AssertionError(f"{label} tree structure changed across restore")
  for index, (left, right) in enumerate(
      zip(expected_leaves, actual_leaves, strict=True)
  ):
    right_host = np.asarray(jax.device_get(right))
    if not np.array_equal(left, right_host):
      raise AssertionError(f"{label} leaf {index} differs after restore")


def _add_one(state):
  return jax.tree.map(
      lambda value: value + jnp.asarray(1, dtype=value.dtype), state
  )


def main() -> None:
  devices = jax.devices()
  if jax.default_backend() != "tpu" or len(devices) != 4:
    raise SystemExit(
        "P45 one-host checkpoint gate requires exactly four TPU devices; "
        f"backend={jax.default_backend()} devices={len(devices)}"
    )
  mesh = jax.sharding.Mesh(
      np.asarray(devices).reshape(1, 4), axis_names=("fsdp", "tp")
  )
  actor = _create_actor(mesh)
  optimizer = nnx.Optimizer(
      actor,
      optax.inject_hyperparams(optax.adamw)(learning_rate=1e-3),
      wrt=nnx.Param,
  )
  model_expected = _host_snapshot(nnx.state(actor))
  optimizer_expected = _host_snapshot(
      nnx.state(optimizer, nnx.optimizer.OptState)
  )

  config = frozenlake_checkpoint.Config(
      mode="resume",
      root=frozenlake_checkpoint.GCS_ROOT,
      tag="onehost-v5p-gate",
      interval=10,
      max_to_keep=1,
  )
  contract = frozenlake_checkpoint.build_contract(
      config,
      {
          "source_commit": "onehost-v5p-gate",
          "workload": "synthetic-checkpoint-roundtrip",
          "mesh_dp": 1,
          "mesh_tp": 4,
          "optimizer_placement": "device-resident",
          "claim_scope": "onehost-mechanism-only",
      },
  )
  metadata10 = {
      "global_step": 10,
      "role": "actor",
      "canon_resume_contract": contract,
  }

  options = checkpoint_options.TunixCheckpointingOptions(
      save_decision_policy=ocp.training.save_decision_policies.FixedIntervalPolicy(
          10
      ),
      preservation_policy=ocp.training.preservation_policies.LatestN(1),
      enable_async_checkpointing=False,
      save_on_close=False,
  )
  with tempfile.TemporaryDirectory(prefix="p45-v5p-checkpoint-") as root:
    manager = checkpoint_manager.CheckpointManager(root, options=options)
    if manager.save(9, actor, optimizer, custom_metadata=metadata10):
      raise AssertionError("FixedIntervalPolicy(10) unexpectedly saved step 9")
    if not manager.save(10, actor, optimizer, custom_metadata=metadata10):
      raise AssertionError("FixedIntervalPolicy(10) skipped step 10")
    if manager.latest_step() != 10:
      raise AssertionError(f"latest checkpoint is {manager.latest_step()}, not 10")

    nnx.update(actor, _add_one(nnx.state(actor)))
    nnx.update(
        optimizer,
        _add_one(nnx.state(optimizer, nnx.optimizer.OptState)),
    )
    restored_step, restored_metadata = manager.maybe_restore(actor, optimizer)
    frozenlake_checkpoint.validate_restored(
        config,
        restored_step=restored_step,
        optimizer_restored=manager.last_restore_had_optimizer,
        metadata=restored_metadata,
        expected_contract=contract,
    )
    _assert_exact("model", model_expected, nnx.state(actor))
    _assert_exact(
        "optimizer",
        optimizer_expected,
        nnx.state(optimizer, nnx.optimizer.OptState),
    )

    metadata20 = {**metadata10, "global_step": 20}
    if not manager.save(20, actor, optimizer, custom_metadata=metadata20):
      raise AssertionError("FixedIntervalPolicy(10) skipped step 20")
    manager.close()
    step_dirs = sorted(
        path.name
        for path in pathlib.Path(root).iterdir()
        if path.is_dir() and path.name.isdigit()
    )
    if step_dirs != ["20"]:
      raise AssertionError(f"LatestN(1) retained unexpected steps: {step_dirs}")

  model_kinds = sorted(
      {
          getattr(leaf.sharding, "memory_kind", None)
          for leaf in jax.tree.leaves(nnx.state(actor))
          if hasattr(leaf, "sharding")
      }
  )
  print(
      "P45_ONEHOST_CHECKPOINT_PASS "
      "backend=tpu devices=4 topology=DP1xTP4 step=10 "
      "model_exact=1 optimizer_exact=1 metadata_exact=1 "
      "interval=10 latest_n=1 optimizer_restored=1 "
      f"memory_kinds={model_kinds} scope=mechanism-only",
      flush=True,
  )


if __name__ == "__main__":
  main()
