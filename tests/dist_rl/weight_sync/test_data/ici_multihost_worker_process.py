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

"""One worker of a 4-process CPU JAX world for `ici_multihost_checkpoint_test`.

Runs two multi-host FSDP trainer processes (even on non-zero-primary ranks like
{1, 2}, mirroring GKE TPU torus ordering) and two rollout processes ({0, 3})
in one shared `jax.distributed` world. Exercises:
  - Real `PeftTrainer` multi-host `fwd_bwd` + `update` with global batches.
  - Real `IciSourceSynchronizer` -> `IciWeightSyncDelegate` ICI weight sync.
  - Sub-world Orbax v1 async checkpoint SAVE + RESTORE on trainer processes
    while rollout processes run independent local JAX compute (no deadlock).
  - Sub-world Orbax v0 `Checkpointer` and `CheckpointManager` save + restore
    with trainer-scoped public `MultiprocessingOptions` (the MaxText engine
    and `load_parameters_path` paths), with and without OCDBT + Zarr3.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import json
import os
from typing import Any

from absl import app
import numpy as np
from tunix.dist_rl.testing import ici_testing

def _run_worker(cfg: dict[str, Any]) -> None:
  """Initializes CPU Gloo JAX and serves worker commands over IPC."""
  if cfg["role"] == "trainer":
    os.environ["DIST_RL_TRAINER_REPLICA"] = str(cfg["replica"])

  jax = ici_testing.init_cpu_gloo_world(
      width=int(cfg["width"]),
      jax_port=int(cfg["jax_port"]),
      processes=int(cfg["processes"]),
      rank=int(cfg["rank"]),
  )

  # Lazy imports ensure XLA_FLAGS and JAX_PLATFORMS take effect before JAX init.
  # pylint: disable=g-import-not-at-top
  from flax import nnx
  import jax.numpy as jnp
  import optax
  import orbax.checkpoint as ocp_v0
  from tunix.dist_rl.train import peft_trainer_v2
  from tunix.dist_rl.weight_sync import ici_transport
  from tunix.dist_rl.weight_sync import ici_weight_sync_delegate
  from tunix.sft import checkpoint_options
  from tunix.utils import subworld_checkpoint
  # pylint: enable=g-import-not-at-top

  # Every process, rollouts included, joins as `ici_slice_worker` does.
  subworld_checkpoint.enable_distributed_process_ids()

  class _TinyLinear(nnx.Module):
    """Tiny 2-parameter NNX module sharded across the trainer's FSDP mesh."""

    def __init__(self, mesh: jax.sharding.Mesh):
      w_init = np.arange(32, dtype=np.float32).reshape(8, 4) * 0.05 + 0.1
      b_init = np.arange(4, dtype=np.float32) * 0.1
      sh_w = jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec("fsdp", None)
      )
      sh_b = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
      # `out_sharding` gives the optimizer state the same partition spec, so
      # `_shard_optimizer` keeps it in place instead of re-sharding it.
      self.w = nnx.Param(
          jax.make_array_from_callback(
              w_init.shape, sh_w, lambda idx: w_init[idx]
          ),
          out_sharding=("fsdp", None),
          mesh=mesh,
      )
      self.b = nnx.Param(
          jax.make_array_from_callback(
              b_init.shape, sh_b, lambda idx: b_init[idx]
          )
      )

    def __call__(self, x: jax.Array) -> jax.Array:
      return x @ self.w[...] + self.b[...]

  def _loss_fn(
      model: Any, x: jax.Array, y: jax.Array
  ) -> tuple[jax.Array, dict[str, jax.Array]]:
    pred = model(x)
    loss = jnp.mean((pred - y) ** 2)
    return loss, {"mse": loss}

  def _gen_input_fn(payload: dict[str, np.ndarray]):
    return {"x": payload["x"], "y": payload["y"]}

  class _FakeRolloutSampler:
    """Rollout sampler holding TP-sharded weights matching `_TinyLinear`."""

    model_runner = None
    config = None

    def __init__(self):
      devs = ici_transport.local_devices()
      self.mesh = jax.sharding.Mesh(np.asarray(devs), ("tp",))
      sh_w = jax.sharding.NamedSharding(
          self.mesh, jax.sharding.PartitionSpec(None, "tp")
      )
      sh_b = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())
      self._state = {
          "w": jax.device_put(jnp.zeros((8, 4), jnp.float32), sh_w),
          "b": jax.device_put(jnp.zeros((4,), jnp.float32), sh_b),
      }

    @property
    def transformer_state(self):
      return self._state

    def delete_cache(self):
      pass

    def reset_prefix_cache(self):
      pass

    def reinitialize_cache(self):
      pass

    def refresh_state_leaves(self):
      pass

  class _RolloutWorkerProc:

    def __init__(self):
      self.sampler = _FakeRolloutSampler()
      self.delegate = ici_weight_sync_delegate.IciWeightSyncDelegate(
          server_id=f"rollout-{cfg['replica']}",
      )

    def run_local_compute(self, iters: int):
      w = self.sampler.transformer_state["w"]
      x = jax.device_put(
          jnp.ones((2, 8), jnp.float32),
          jax.sharding.NamedSharding(
              self.sampler.mesh, jax.sharding.PartitionSpec()
          ),
      )
      fn = jax.jit(lambda a, b: jnp.sum(a @ b))
      total = 0.0
      for _ in range(iters):
        total = float(fn(x, w))
      return {"rank": int(jax.process_index()), "sum": total}

    def bind(self, _):
      return asyncio.run(self.delegate.bind_weight_sync(sampler=self.sampler))

    def metadata(self, _):
      return asyncio.run(self.delegate.get_weight_sync_metadata())

    def pre(self, fields):
      return asyncio.run(
          self.delegate.pre_weight_sync(ici_testing.FakeSyncRequest(fields))
      )

    def ici_check(self, request):
      return self.delegate.check(request)

    def ici_transfer(self, request):
      return self.delegate.transfer(request)

    def weight_sync(self, fields):
      return asyncio.run(
          self.delegate.weight_sync(ici_testing.FakeSyncRequest(fields))
      )

    def post(self, fields):
      return asyncio.run(
          self.delegate.post_weight_sync(ici_testing.FakeSyncRequest(fields))
      )

    def snapshot(self, _):
      return {
          k: np.asarray(v) for k, v in self.sampler.transformer_state.items()
      }

  class _TrainerWorkerProc:

    def __init__(self):
      # Trainers sit on non-zero-primary ranks, unlike `ici_slice_worker`'s
      # `0..trainer_replicas-1`, so the Orbax primary host is not process 0.
      self.trainer_processes = sorted(int(r) for r in cfg["trainer_ranks"])
      trainer_devices = [
          d
          for p in self.trainer_processes
          for d in ici_transport.local_devices(p)
      ]
      self.mesh = jax.sharding.Mesh(np.asarray(trainer_devices), ("fsdp",))
      self.model = _TinyLinear(self.mesh)
      self.worker_id = f"trainer-{cfg['replica']}"
      save_policies = checkpoint_options.ocp.training.save_decision_policies
      pres_policies = checkpoint_options.ocp.training.preservation_policies
      ckpt_opts = checkpoint_options.TunixCheckpointingOptions(
          save_decision_policy=save_policies.FixedIntervalPolicy(1),
          preservation_policy=pres_policies.LatestN(2),
          enable_async_checkpointing=True,
          # Memory-budgeted saving maps devices to distributed process ids.
          device_host_concurrent_gb=1,
      )
      training_cfg = peft_trainer_v2.TrainingConfig(
          eval_every_n_steps=1000,
          gradient_accumulation_steps=1,
          data_sharding_axis=("fsdp",),
          checkpoint_root_directory=cfg["ckpt_dir"],
          checkpointing_options=ckpt_opts,
          resume_from_checkpoint_on_init=False,
      )
      with self.mesh:
        self.trainer = peft_trainer_v2.PeftTrainer(
            self.model,
            optax.adamw(learning_rate=0.1),
            training_cfg,
            sampler_type="vanilla",
            weight_sync_worker_factory=lambda: (
                ici_weight_sync_delegate.IciSourceSynchronizer(self.worker_id)
            ),
        )
        self.trainer.with_loss_fn(_loss_fn, has_aux=True)
        self.trainer.with_gen_model_input_fn(_gen_input_fn)

    def info(self, _):
      return {
          "rank": int(jax.process_index()),
          "worker_id": self.worker_id,
          "trainer_processes": self.trainer_processes,
      }

    def fwd_bwd(self, payload):
      with self.mesh:
        self.trainer.fwd_bwd(payload)
      return True

    def update(self, _):
      with self.mesh:
        step = self.trainer.update()
        metrics = self.trainer.get_metrics()
      return {"step": int(step), "has_metrics": metrics is not None}

    def save_checkpoint(self, args):
      with self.mesh:
        self.trainer.save_checkpoint(
            metadata=args["metadata"], step=args["step"], force=args["force"]
        )
        self.trainer.checkpoint_manager.wait()
      return True

    def restore_checkpoint(self, step):
      with self.mesh:
        meta = self.trainer.restore_checkpoint(step=step)
      # pylint: disable=protected-access
      train_steps = int(self.trainer._train_steps)
      # pylint: enable=protected-access
      return {"train_steps": train_steps, "metadata": meta}

    def _v0_handler(self, use_ocdbt: bool, prefix: str):
      """Builds a trainer-scoped v0 handler the way MaxText's engine does."""
      procs = subworld_checkpoint.mesh_process_indices(self.mesh)
      mp = subworld_checkpoint.multiprocessing_options(
          procs, barrier_sync_key_prefix=prefix
      ).v0()
      type_handlers = ocp_v0.type_handlers
      registry = type_handlers.create_type_handler_registry(*(
          (ty, type_handlers.get_type_handler(ty))
          for ty in type_handlers.supported_types()
      ))
      # `PyTreeCheckpointHandler` does not forward `primary_host` to the
      # registered `ArrayHandler`, which writes array metadata.
      registry.add(
          jax.Array,
          type_handlers.ArrayHandler(primary_host=mp.primary_host),
          override=True,
      )
      handler = ocp_v0.PyTreeCheckpointHandler(
          use_ocdbt=use_ocdbt,
          use_zarr3=use_ocdbt,
          save_device_host_concurrent_gb=1,
          multiprocessing_options=mp,
          type_handler_registry=registry,
      )
      return handler, mp

    def _params_match(self, params, restored) -> bool:
      return all(
          np.array_equal(np.asarray(a.data), np.asarray(b.data))
          for k in ("w", "b")
          for a, b in zip(
              params[k].addressable_shards,
              restored[k].addressable_shards,
              strict=True,
          )
      )

    def test_v0_subworld_restore(self, args):
      """Saves and restores with a v0 `Checkpointer` (load_parameters_path)."""
      v0_dir, use_ocdbt = args
      params = {"w": self.model.w[...], "b": self.model.b[...]}
      handler, mp = self._v0_handler(use_ocdbt, f"v0_ocdbt{use_ocdbt}")
      ckptr = ocp_v0.Checkpointer(handler, multiprocessing_options=mp)
      ckptr.save(v0_dir, params)
      restored = ckptr.restore(
          v0_dir,
          args=ocp_v0.args.PyTreeRestore(
              item=params,
              restore_args=ocp_v0.checkpoint_utils.construct_restore_args(
                  params
              ),
          ),
      )
      return self._params_match(params, restored)

    def test_v0_manager_save_restore(self, args):
      """Saves and restores with a v0 `CheckpointManager` (MaxText engine)."""
      mgr_dir, use_ocdbt = args
      params = {"w": self.model.w[...], "b": self.model.b[...]}
      handler, mp = self._v0_handler(use_ocdbt, f"v0mgr_ocdbt{use_ocdbt}")
      # Orbax rejects `create=True` with `active_processes`; create it here.
      if ocp_v0.multihost.is_primary_host(mp.primary_host):
        os.makedirs(mgr_dir, exist_ok=True)
      ocp_v0.multihost.sync_global_processes(
          f"create_{mgr_dir}", processes=mp.active_processes
      )
      mgr = ocp_v0.CheckpointManager(
          mgr_dir,
          options=ocp_v0.CheckpointManagerOptions(
              enable_async_checkpointing=True,
              create=False,
              multiprocessing_options=mp,
          ),
          item_handlers={"model_params": handler},
      )
      mgr.save(
          1,
          args=ocp_v0.args.Composite(
              model_params=ocp_v0.args.PyTreeSave(item=params)
          ),
          force=True,
      )
      mgr.wait_until_finished()
      restored = mgr.restore(
          1,
          args=ocp_v0.args.Composite(
              model_params=ocp_v0.args.PyTreeRestore(
                  item=params,
                  restore_args=ocp_v0.checkpoint_utils.construct_restore_args(
                      params
                  ),
              )
          ),
      )["model_params"]
      mgr.close()
      return self._params_match(params, restored)

    def prepare_weight_sync(self, _):
      with self.mesh:
        return self.trainer.prepare_weight_sync()

    def ici_check(self, request):
      return self.trainer.ici_check(request)

    def ici_transfer(self, request):
      return self.trainer.ici_transfer(request)

    def release_weight_sync(self, _):
      return self.trainer.release_weight_sync()

    def snapshot_local_shards(self, _):
      w_shards: dict[tuple[int, int], np.ndarray] = {}
      for shard in self.model.w[...].addressable_shards:
        row_slice = shard.index[0]
        start, stop, _ = row_slice.indices(8)
        w_shards[(start, stop)] = np.asarray(shard.data).copy()
      b_val = np.asarray(self.model.b[...].addressable_shards[0].data).copy()
      opt_state = nnx.state(self.trainer.optimizer, nnx.optimizer.OptState)
      opt_leaves = [
          np.asarray(leaf.addressable_shards[0].data).copy()
          for leaf in jax.tree_util.tree_leaves(opt_state)
          if hasattr(leaf, "addressable_shards")
      ]
      return {"w_shards": w_shards, "b": b_val, "opt_leaves": opt_leaves}

    def close(self, _):
      self.trainer.close()
      return True

  worker: Any = (
      _TrainerWorkerProc()
      if cfg["role"] == "trainer"
      else _RolloutWorkerProc()
  )
  ici_testing.serve_worker_ipc(
      int(cfg["control_port"]), int(cfg["rank"]), worker
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) < 2:
    raise app.UsageError("Expected JSON worker config as positional argument.")
  _run_worker(json.loads(argv[1]))


if __name__ == "__main__":
  app.run(main)
