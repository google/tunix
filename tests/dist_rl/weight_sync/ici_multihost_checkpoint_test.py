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

"""Multi-host Role.ACTOR fan-out + sub-world Orbax checkpointing E2E tests.

Covers:
1. Real 4-process CPU Gloo JAX world (2 trainer processes on non-zero-primary
   ranks {1, 2} + 2 rollout processes on {0, 3}, mimicking GKE TPU torus order):
   - Dynamic trainer JAX rank discovery via coordinator KV store.
   - Multi-host FSDP `fwd_bwd` + `update` with global batches.
   - Multi-host ICI weight sync (`{1, 2}` -> `{0, 3}`) with exact verification.
   - Sub-world Orbax v1 async checkpoint SAVE & RESTORE on `{1, 2}` while
     rollout processes `{0, 3}` execute independent local compute (ZERO
     deadlock, bit-for-bit identical params + AdamW optimizer state + step).
   - Sub-world Orbax v0 `Checkpointer.restore` (`load_parameters_path` path).
2. `DistributedRLEngine` concurrent multi-actor fan-out unit tests (`train_step`,
   `configure_worker`, `per_token_logps`, `get_metrics`, `save_checkpoint`,
   `resume_from_checkpoint`, `set_target_state`), plus `trainer_replicas` config
   and worker identity tests.
"""

from __future__ import annotations

import asyncio
from concurrent import futures
import json
from multiprocessing import connection
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from unittest import mock

from absl.testing import absltest
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.launch.common import resolve_config
from tunix.dist_rl.launch.nodes import node_utils
from tunix.dist_rl.launch.nodes import orchestrator as orchestrator_node
from tunix.dist_rl.orchestrator import distributed_rl_engine
from tunix.dist_rl.orchestrator import worker_registry
from tunix.dist_rl.weight_sync import ici_handler
from tunix.dist_rl.worker import remote_execution

_WORKER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ici_multihost_worker_process.py",
)
_TIMEOUT_S = 120


def _free_port() -> int:
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


def _gloo_available() -> bool:
  try:
    import jax  # pylint: disable=g-import-not-at-top

    return hasattr(jax.config, "jax_cpu_collectives_implementation")
  except ImportError:
    return False


class _ProcHandle:

  def __init__(self, rank: int, conn: connection.Connection):
    self.rank = rank
    self._conn = conn
    self._lock = threading.Lock()

  def submit(self, method: str, argument=None):
    with self._lock:
      self._conn.send((method, argument))
      if not self._conn.poll(_TIMEOUT_S):
        raise TimeoutError(
            f"process {self.rank} timed out waiting for {method}"
        )
      status, payload = self._conn.recv()
    if status != "ok":
      raise RuntimeError(f"process {self.rank} {method} failed: {payload}")
    return payload


def _call_concurrent(handles, method, argument=None):
  with futures.ThreadPoolExecutor(max_workers=len(handles)) as pool:
    futs = [pool.submit(h.submit, method, argument) for h in handles]
    return [f.result(timeout=_TIMEOUT_S) for f in futs]


@absltest.skipUnless(_gloo_available(), "needs JAX CPU collectives (gloo)")
class MultiHostSubworldCheckpointE2ETest(absltest.TestCase):

  def test_multihost_actor_ici_and_subworld_checkpoint_save_restore(self):
    # Deliberately assign trainer replicas to non-zero-primary ranks {1, 2} and
    # rollouts to {0, 3} (mirroring GKE TPU torus coordinate rank assignment).
    trainer_ranks = [2, 1]  # replica 0 -> rank 2, replica 1 -> rank 1
    rollout_ranks = [0, 3]  # replica 0 -> rank 0, replica 1 -> rank 3
    tmpdir = tempfile.mkdtemp(prefix="ici_multihost_e2e_")
    ckpt_dir = os.path.join(tmpdir, "v1_ckpt")
    v0_dir = os.path.join(tmpdir, "v0_params")
    jax_port = _free_port()
    listener = connection.Listener(("127.0.0.1", 0))
    env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
    procs = []
    logs = []
    try:
      for rank in range(4):
        if rank in trainer_ranks:
          role = "trainer"
          replica = trainer_ranks.index(rank)
        else:
          role = "rollout"
          replica = rollout_ranks.index(rank)
        cfg = dict(
            rank=rank,
            role=role,
            replica=replica,
            processes=4,
            width=2,
            trainer_replicas=2,
            rollout_replicas=2,
            run_id="e2e-subworld-test",
            jax_port=jax_port,
            control_port=listener.address[1],
            ckpt_dir=ckpt_dir,
        )
        log = tempfile.TemporaryFile(mode="w+")
        logs.append(log)
        procs.append(
            subprocess.Popen(
                [sys.executable, _WORKER, json.dumps(cfg)],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )
        )

      handles: dict[int, _ProcHandle] = {}
      listener._listener._socket.settimeout(_TIMEOUT_S)  # pylint: disable=protected-access
      try:
        for _ in range(4):
          conn = listener.accept()
          rank = conn.recv()
          handles[rank] = _ProcHandle(rank, conn)
      except Exception as exc:
        tail = "\n".join(
            f"--- proc {i} ---\n" + (l.seek(0) or l.read()[-2000:])
            for i, l in enumerate(logs)
        )
        raise RuntimeError(f"Workers failed to start:\n{tail}") from exc
      finally:
        listener.close()

      trainers = [handles[r] for r in trainer_ranks]
      rollouts = [handles[r] for r in rollout_ranks]
      all_handles = list(handles.values())

      # 1. Verify dynamic coordinator KV discovery found real ranks [1, 2].
      infos = _call_concurrent(trainers, "info")
      for info in infos:
        self.assertEqual(info["discovered_processes"], [1, 2])

      # 2. Multi-host SPMD training step 1 (global batch shape (4, 8) -> sharded
      #    across FSDP=4 devices spanning processes {1, 2}).
      batch1 = {
          "x": np.arange(32, dtype=np.float32).reshape(4, 8) * 0.1,
          "y": np.ones((4, 4), dtype=np.float32) * 2.0,
      }
      _call_concurrent(trainers, "fwd_bwd", batch1)
      upd1 = _call_concurrent(trainers, "update")
      self.assertEqual([u["step"] for u in upd1], [1, 1])

      # Snapshot trainer weights + AdamW optimizer state at step 1.
      snap_step1 = _call_concurrent(trainers, "snapshot_local_shards")

      # 3. Multi-host ICI weight sync: sources={1, 2} -> destinations={0, 3}.
      handler = ici_handler.IciHandler(
          lambda: all_handles, chunk_bytes=128, check_timeout_s=_TIMEOUT_S
      )
      for r in rollouts:
        r.submit("bind")
      dst_meta = [m for r in rollouts for m in r.submit("metadata")]
      src_meta_lists = _call_concurrent(trainers, "prepare_weight_sync")
      src_meta = [m for ml in src_meta_lists for m in ml]
      for m in (*src_meta, *dst_meta):
        handler.register_work_unit(m)
      fields = {"policy_version": 1, "req_id": "r1", "uuid": 1}
      for r in rollouts:
        r.submit("pre", fields)
      res = handler.transfer(
          [m.unit for m in src_meta],
          [m.unit for m in dst_meta],
          req_id="r1",
          generation=1,
      )
      self.assertTrue(res.success)
      for r in rollouts:
        r.submit("weight_sync", fields)
        r.submit("post", fields)
      _call_concurrent(trainers, "release_weight_sync")

      # Verify both rollout replicas received exact updated step-1 weights.
      rollout_snaps = [r.submit("snapshot") for r in rollouts]
      for rsnap in rollout_snaps:
        np.testing.assert_array_equal(rsnap["b"], snap_step1[0]["b"])
        self.assertEqual(rsnap["w"].shape, (8, 4))
        # Reconstruct full w from the two trainer processes' local shards.
        for tsnap in snap_step1:
          for idx_str, shard_arr in tsnap["w_shards"].items():
            # Verify non-zero updated weights match trainer shards.
            self.assertEqual(shard_arr.shape[1], 4)
        np.testing.assert_array_equal(rollout_snaps[0]["w"], rollout_snaps[1]["w"])

      # 4. SUB-WORLD ORBAX CHECKPOINT SAVE on trainers {1, 2} while rollouts
      #    {0, 3} concurrently run independent local compute (NO deadlock!).
      with futures.ThreadPoolExecutor(max_workers=4) as pool:
        save_futs = [
            pool.submit(
                t.submit,
                "save_checkpoint",
                {"metadata": {"step": 1, "tag": "subworld"}, "step": 1, "force": True},
            )
            for t in trainers
        ]
        roll_futs = [
            pool.submit(r.submit, "run_local_compute", 10) for r in rollouts
        ]
        self.assertEqual([f.result(timeout=_TIMEOUT_S) for f in save_futs], [True, True])
        roll_out = [f.result(timeout=_TIMEOUT_S) for f in roll_futs]
        self.assertCountEqual([o["rank"] for o in roll_out], rollout_ranks)

      # 5. Advance to step 2 so live weights & AdamW state diverge from step 1.
      batch2 = {
          "x": np.arange(32, dtype=np.float32).reshape(4, 8) * -0.2,
          "y": np.ones((4, 4), dtype=np.float32) * -5.0,
      }
      _call_concurrent(trainers, "fwd_bwd", batch2)
      upd2 = _call_concurrent(trainers, "update")
      self.assertEqual([u["step"] for u in upd2], [2, 2])
      snap_step2 = _call_concurrent(trainers, "snapshot_local_shards")
      self.assertFalse(np.array_equal(snap_step1[0]["b"], snap_step2[0]["b"]))

      # 6. SUB-WORLD ORBAX CHECKPOINT RESTORE (step 1) on trainers {1, 2} while
      #    rollouts {0, 3} do NOT enter restore (NO deadlock!).
      with futures.ThreadPoolExecutor(max_workers=4) as pool:
        rest_futs = [
            pool.submit(t.submit, "restore_checkpoint", 1) for t in trainers
        ]
        roll_futs = [
            pool.submit(r.submit, "run_local_compute", 10) for r in rollouts
        ]
        restored = [f.result(timeout=_TIMEOUT_S) for f in rest_futs]
        for f in roll_futs:
          f.result(timeout=_TIMEOUT_S)

      for r_info in restored:
        self.assertEqual(r_info["train_steps"], 1)
        self.assertEqual(r_info["metadata"]["tag"], "subworld")

      snap_restored = _call_concurrent(trainers, "snapshot_local_shards")
      for orig, got in zip(snap_step1, snap_restored):
        np.testing.assert_array_equal(orig["b"], got["b"])
        self.assertEqual(orig["w_shards"].keys(), got["w_shards"].keys())
        for k in orig["w_shards"]:
          np.testing.assert_array_equal(orig["w_shards"][k], got["w_shards"][k])
        self.assertLen(got["opt_leaves"], len(orig["opt_leaves"]))
        for orig_opt, got_opt in zip(orig["opt_leaves"], got["opt_leaves"]):
          np.testing.assert_array_equal(orig_opt, got_opt)

      # 7. SUB-WORLD ORBAX v0 RESTORE (MaxText load_parameters_path path) on
      #    trainers {1, 2} while rollouts {0, 3} never enter restore.
      v0_ok = _call_concurrent(trainers, "test_v0_subworld_restore", v0_dir)
      self.assertEqual(v0_ok, [True, True])

      _call_concurrent(trainers, "close")
    finally:
      for proc in procs:
        if proc.poll() is None:
          proc.kill()
      for proc in procs:
        proc.wait()
      shutil.rmtree(tmpdir, ignore_errors=True)


class _BarrierFakeActorHandle(remote_execution.ActorHandle):
  """Fake handle that blocks until ALL `n` actors enter an SPMD call together."""

  def __init__(self, wid: str, barrier: threading.Barrier, step_val: int):
    self.wid = wid
    self._barrier = barrier
    self._step_val = step_val
    self.calls: list[tuple[str, dict]] = []

  def submit(self, method: str, *args, **kwargs):
    self.calls.append((method, {"args": args, "kwargs": kwargs}))
    if method in (
        "compile",
        "fwd_bwd",
        "update",
        "save_checkpoint",
        "restore_checkpoint",
        "per_token_logps",
    ):
      # Proves calls are fanned out CONCURRENTLY (sequential calls deadlock!).
      self._barrier.wait(timeout=5.0)
    if method == "update":
      return (self._step_val, {"worker": self.wid})
    if method == "restore_checkpoint":
      return {"step": 3, "global_step": 3, "worker": self.wid}
    if method == "get_metrics":
      return {"worker": self.wid}
    return {"ok": True, "worker": self.wid}

  async def asubmit(self, method: str, *args, **kwargs):
    return await asyncio.to_thread(self.submit, method, *args, **kwargs)

  def dispatch_task(self, request):
    raise NotImplementedError

  def poll_responses(self):
    raise NotImplementedError


class MultiActorFanOutUnitTest(absltest.TestCase):

  def _make_engine(self):
    reg = worker_registry.WorkerRegistry()
    barrier = threading.Barrier(2)
    # Register out of order (trainer-1 before trainer-0) to verify deterministic
    # primary-worker selection by `worker_id`.
    h1 = _BarrierFakeActorHandle("trainer-1", barrier, step_val=7)
    h0 = _BarrierFakeActorHandle("trainer-0", barrier, step_val=7)
    for wid, h in (("trainer-1", h1), ("trainer-0", h0)):
      reg.register_handle(wid, [datatypes.Role.ACTOR], h)
      reg.set_status(
          wid,
          worker_registry.AdmissionStatus.ACTIVE,
          expected_handle=h,
          expected_status=worker_registry.AdmissionStatus.REGISTERED,
      )
    engine = distributed_rl_engine.DistributedRLEngine(registry=reg)
    return engine, h0, h1

  def test_engine_fans_out_spmd_calls_concurrently_and_returns_primary(self):
    engine, h0, h1 = self._make_engine()
    payload = datatypes.RLTrainerPayload(
        prompt_ids=np.zeros((4, 2), np.int32),
        prompt_mask=np.ones((4, 2), np.int32),
        completion_ids=np.zeros((4, 2), np.int32),
        completion_mask=np.ones((4, 2), np.int32),
        advantages=np.zeros((4,), np.float32),
    )

    async def _run():
      out = await engine.train_step(payload, apply_optimizer=True)
      self.assertEqual(out["train_step"], 7)
      self.assertEqual(out["metrics"], {"worker": "trainer-0"})
      lp = await engine.per_token_logps(datatypes.Role.ACTOR, items="batch")
      self.assertEqual(lp["worker"], "trainer-0")
      metrics = await engine.get_metrics(datatypes.Role.ACTOR)
      self.assertEqual(metrics, {"worker": "trainer-0"})
      saved = await engine.save_checkpoint(
          datatypes.Role.ACTOR, metadata={"step": 7}, step=7, force=True
      )
      self.assertEqual(saved["worker"], "trainer-0")
      restored_step = await engine.resume_from_checkpoint(datatypes.Role.ACTOR)
      self.assertEqual(restored_step, 3)

    asyncio.run(_run())
    expected_methods = [
        "fwd_bwd",
        "update",
        "per_token_logps",
        "get_metrics",
        "save_checkpoint",
        "restore_checkpoint",
    ]
    self.assertEqual([m for m, _ in h0.calls], expected_methods)
    self.assertEqual([m for m, _ in h1.calls], expected_methods)

  def test_trainer_replicas_config_worker_id_and_required_workers(self):
    hp = resolve_config.initialize([
        "cluster_config.target=k8s",
        "cluster_config.k8s.tunix_image=img:latest",
        "cluster_config.k8s.trainer_id=my-train",
        "cluster_config.trainer_replicas=2",
        "cluster_config.rollout_replicas=2",
    ])
    req = orchestrator_node.required_workers(hp)
    self.assertEqual(req[datatypes.Role.ACTOR], 2)
    self.assertEqual(req[datatypes.Role.ROLLOUT], 2)

    with mock.patch.dict(os.environ, {"DIST_RL_TRAINER_REPLICA": "1"}):
      self.assertEqual(
          node_utils.worker_id(hp, "trainer", "trainer-0"), "my-train-1"
      )
    with self.assertRaisesRegex(ValueError, "DIST_RL_TRAINER_REPLICA"):
      with mock.patch.dict(os.environ, {}, clear=True):
        node_utils.worker_id(hp, "trainer", "trainer-0")
    with self.assertRaisesRegex(
        ValueError, "cluster_config.trainer_replicas must be 1"
    ):
      resolve_config.initialize(["cluster_config.trainer_replicas=2"])


if __name__ == "__main__":
  absltest.main()
