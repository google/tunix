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

"""One worker of a 4-process CPU JAX world for `ici_multihost_checkpoint_test`.

Runs two multi-host FSDP trainer processes (even on non-zero-primary ranks like
{1, 2}, mirroring GKE TPU torus ordering!) and two rollout processes ({0, 3})
in ONE shared `jax.distributed` world. Exercises:
  - Dynamic trainer process discovery via JAX distributed coordinator KV store.
  - Real `PeftTrainer` multi-host `fwd_bwd` + `update` with global batches.
  - Real `IciSourceSynchronizer` -> `IciWeightSyncDelegate` ICI weight sync.
  - Sub-world Orbax v1 async checkpoint SAVE + RESTORE on trainer processes
    while rollout processes run independent local JAX compute (no deadlock).
  - Sub-world Orbax v0 `Checkpointer.restore` via `patch_orbax_for_subworld`
    (the MaxText `load_parameters_path` path).
"""

import asyncio
import inspect
import json
from multiprocessing import connection
import os
import sys
import traceback

_CFG = json.loads(sys.argv[1])
os.environ["XLA_FLAGS"] = (
    f"--xla_force_host_platform_device_count={_CFG['width']}"
)
os.environ["JAX_PLATFORMS"] = "cpu"
if _CFG["role"] == "trainer":
  os.environ["DIST_RL_TRAINER_REPLICA"] = str(_CFG["replica"])

# pylint: disable=g-import-not-at-top,wrong-import-position
import jax

jax.config.update("jax_cpu_collectives_implementation", "gloo")
jax.distributed.initialize(
    coordinator_address=f"127.0.0.1:{_CFG['jax_port']}",
    num_processes=_CFG["processes"],
    process_id=_CFG["rank"],
)
jax.devices()

from flax import nnx
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp_v0
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import trainer as trainer_node
from tunix.dist_rl.train import peft_trainer_v2
from tunix.dist_rl.weight_sync import ici_transport
from tunix.dist_rl.weight_sync import ici_weight_sync_delegate
from tunix.sft import checkpoint_options
from tunix.utils import subworld_checkpoint


class _TinyLinear(nnx.Module):
  """Tiny 2-parameter NNX module sharded across the trainer's FSDP mesh."""

  def __init__(self, mesh: jax.sharding.Mesh):
    w_init = np.arange(32, dtype=np.float32).reshape(8, 4) * 0.05 + 0.1
    b_init = np.arange(4, dtype=np.float32) * 0.1
    sh_w = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("fsdp", None)
    )
    sh_b = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    self.w = nnx.Param(
        jax.make_array_from_callback(
            w_init.shape, sh_w, lambda idx: w_init[idx]
        )
    )
    self.b = nnx.Param(
        jax.make_array_from_callback(
            b_init.shape, sh_b, lambda idx: b_init[idx]
        )
    )

  def __call__(self, x: jax.Array) -> jax.Array:
    return x @ self.w[...] + self.b[...]


def _loss_fn(model: _TinyLinear, x: jax.Array, y: jax.Array):
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


class _Request:

  def __init__(self, fields):
    self.policy_version = fields["policy_version"]
    self.extra_config = {"req_id": fields["req_id"], "uuid": fields["uuid"]}


class _RolloutWorkerProc:

  def __init__(self):
    self.sampler = _FakeRolloutSampler()
    self.delegate = ici_weight_sync_delegate.IciWeightSyncDelegate(
        server_id=f"rollout-{_CFG['replica']}",
        drop_old_weights=True,
    )

  def run_local_compute(self, iters: int):
    # Runs local JAX ops on rollout devices while trainer checkpoints.
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
    return {"rank": jax.process_index(), "sum": total}

  def bind(self, _):
    return asyncio.run(self.delegate.bind_weight_sync(sampler=self.sampler))

  def metadata(self, _):
    return asyncio.run(self.delegate.get_weight_sync_metadata())

  def pre(self, fields):
    return asyncio.run(self.delegate.pre_weight_sync(_Request(fields)))

  def ici_check(self, request):
    return self.delegate.check(request)

  def ici_transfer(self, request):
    return self.delegate.transfer(request)

  def weight_sync(self, fields):
    return asyncio.run(self.delegate.weight_sync(_Request(fields)))

  def post(self, fields):
    return asyncio.run(self.delegate.post_weight_sync(_Request(fields)))

  def snapshot(self, _):
    return {
        k: np.asarray(v) for k, v in self.sampler.transformer_state.items()
    }


class _TrainerWorkerProc:

  def __init__(self):
    hp = resolve_config.initialize([
        "cluster_config.target=external",
        f"cluster_config.trainer_replicas={_CFG['trainer_replicas']}",
        f"cluster_config.rollout_replicas={_CFG['rollout_replicas']}",
        f"cluster_config.run_id={_CFG['run_id']}",
        "cluster_config.artifact_root=/tmp/ici_multihost_test",
        "weight_sync_config.mode=ici",
        "rollout_engine=inprocess_vllm",
    ])
    # Exercises dynamic coordinator KV discovery (independent of POD_INDEX!).
    trainer_devices = trainer_node._ici_trainer_devices(hp)  # pylint: disable=protected-access
    assert trainer_devices is not None
    self.discovered_processes = sorted(
        {int(d.process_index) for d in trainer_devices}
    )
    self.mesh = jax.sharding.Mesh(np.asarray(trainer_devices), ("fsdp",))
    self.model = _TinyLinear(self.mesh)
    self.worker_id = f"trainer-{_CFG['replica']}"
    ckpt_opts = checkpoint_options.TunixCheckpointingOptions(
        save_decision_policy=(
            checkpoint_options.ocp.training.save_decision_policies.FixedIntervalPolicy(
                1
            )
        ),
        preservation_policy=(
            checkpoint_options.ocp.training.preservation_policies.LatestN(2)
        ),
        enable_async_checkpointing=True,
    )
    training_cfg = peft_trainer_v2.TrainingConfig(
        eval_every_n_steps=1000,
        gradient_accumulation_steps=1,
        data_sharding_axis=("fsdp",),
        checkpoint_root_directory=_CFG["ckpt_dir"],
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
        "discovered_processes": self.discovered_processes,
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
    return {
        "train_steps": int(self.trainer._train_steps),  # pylint: disable=protected-access
        "metadata": meta,
    }

  def test_v0_subworld_restore(self, v0_dir: str):
    """Tests Orbax v0 Checkpointer save+restore via `patch_orbax_for_subworld`."""
    procs = subworld_checkpoint.mesh_process_indices(self.mesh)
    params = {"w": self.model.w[...], "b": self.model.b[...]}
    with subworld_checkpoint.patch_orbax_for_subworld(procs):
      ckptr = ocp_v0.Checkpointer(
          ocp_v0.PyTreeCheckpointHandler(use_ocdbt=False)
      )
      ckptr.save(v0_dir, params)
      restore_args = jax.tree.map(
          lambda a: ocp_v0.ArrayRestoreArgs(sharding=a.sharding), params
      )
      restored = ckptr.restore(
          v0_dir,
          args=ocp_v0.args.PyTreeRestore(
              item=params, restore_args=restore_args
          ),
      )
    ok = True
    for k in ("w", "b"):
      for s_orig, s_rest in zip(
          params[k].addressable_shards, restored[k].addressable_shards
      ):
        if not np.array_equal(np.asarray(s_orig.data), np.asarray(s_rest.data)):
          ok = False
    return ok

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
    w_shards = {
        str(s.index): np.asarray(s.data).copy()
        for s in self.model.w[...].addressable_shards
    }
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


def main() -> None:
  worker = (
      _TrainerWorkerProc()
      if _CFG["role"] == "trainer"
      else _RolloutWorkerProc()
  )
  conn = connection.Client(("127.0.0.1", _CFG["control_port"]))
  conn.send(_CFG["rank"])
  while True:
    try:
      method, argument = conn.recv()
    except EOFError:
      return
    if method == "exit":
      return
    try:
      result = getattr(worker, method)(argument)
      if inspect.isawaitable(result):
        result = asyncio.run(result)
      conn.send(("ok", result))
    except Exception as e:  # pylint: disable=broad-exception-caught
      traceback.print_exc()
      conn.send(("error", f"{type(e).__name__}: {e}"))


if __name__ == "__main__":
  main()
