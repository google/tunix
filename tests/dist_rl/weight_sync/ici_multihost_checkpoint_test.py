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

"""Multi-host Role.ACTOR and sub-world Orbax checkpointing E2E test.

Runs a real 4-process CPU Gloo JAX world (2 trainer processes on the
non-zero-primary ranks {1, 2} + 2 rollout processes on {0, 3}, mimicking GKE TPU
torus order):
  - Multi-host FSDP `fwd_bwd` + `update` with global batches.
  - Multi-host ICI weight sync (`{1, 2}` -> `{0, 3}`) with exact verification.
  - Sub-world Orbax v1 async checkpoint SAVE & RESTORE on `{1, 2}` while
    rollout processes `{0, 3}` execute independent local compute (zero
    deadlock, bit-for-bit identical params + AdamW optimizer state + step).
  - Sub-world Orbax v0 `Checkpointer` and `CheckpointManager` save + restore
    (MaxText engine and `load_parameters_path` paths), with and without OCDBT.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent import futures
import os
from typing import Any

from absl.testing import absltest
import numpy as np
from tunix.dist_rl.weight_sync import ici_handler
from tunix.dist_rl.testing import ici_testing

_TIMEOUT_S = 120


def _reconstruct_w(snaps: Sequence[dict[str, Any]]) -> np.ndarray:
  """Reconstructs full `(8, 4)` `w` from distributed trainer row shards."""
  full_w = np.empty((8, 4), dtype=np.float32)
  covered = np.zeros(8, dtype=bool)
  for snap in snaps:
    for (start, stop), shard_arr in snap["w_shards"].items():
      full_w[start:stop, :] = shard_arr
      covered[start:stop] = True
  if not np.all(covered):
    raise AssertionError(f"Trainer w shards missed rows: {covered.tolist()}")
  return full_w


def _run_ici_sync(
    trainers: Sequence[ici_testing.ProcHandle],
    rollouts: Sequence[ici_testing.ProcHandle],
    all_handles: Sequence[ici_testing.ProcHandle],
) -> None:
  """Runs one full multi-host trainer -> rollout ICI weight sync round."""
  handler = ici_handler.IciHandler(
      lambda: list(all_handles), chunk_bytes=128, check_timeout_s=_TIMEOUT_S
  )
  for r in rollouts:
    r.submit("bind")
  dst_meta = [m for r in rollouts for m in r.submit("metadata")]
  src_meta = [
      m
      for ml in ici_testing.call_concurrent(trainers, "prepare_weight_sync")
      for m in ml
  ]
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
  if not res.success:
    raise AssertionError(f"ICI transfer failed: {res}")
  for r in rollouts:
    r.submit("weight_sync", fields)
    r.submit("post", fields)
  ici_testing.call_concurrent(trainers, "release_weight_sync")


def _assert_shards_equal(
    orig_snaps: Sequence[dict[str, Any]],
    got_snaps: Sequence[dict[str, Any]],
) -> None:
  """Asserts params and AdamW optimizer state shards match bit-for-bit."""
  for orig, got in zip(orig_snaps, got_snaps, strict=True):
    np.testing.assert_array_equal(orig["b"], got["b"])
    assert orig["w_shards"].keys() == got["w_shards"].keys()
    for k in orig["w_shards"]:
      np.testing.assert_array_equal(orig["w_shards"][k], got["w_shards"][k])
    assert len(got["opt_leaves"]) == len(orig["opt_leaves"])
    for orig_opt, got_opt in zip(
        orig["opt_leaves"], got["opt_leaves"], strict=True
    ):
      np.testing.assert_array_equal(orig_opt, got_opt)


@absltest.skipUnless(
    ici_testing.gloo_available(), "needs JAX CPU collectives (gloo)"
)
class MultiHostSubworldCheckpointE2ETest(absltest.TestCase):

  def test_multihost_actor_ici_and_subworld_checkpoint_save_restore(self):
    # Deliberately assign trainer replicas to non-zero-primary ranks {1, 2} and
    # rollouts to {0, 3} (mirroring GKE TPU torus coordinate rank assignment).
    trainer_ranks = [2, 1]  # replica 0 -> rank 2, replica 1 -> rank 1
    rollout_ranks = [0, 3]  # replica 0 -> rank 0, replica 1 -> rank 3
    tmpdir = self.create_tempdir().full_path
    ckpt_dir = os.path.join(tmpdir, "v1_ckpt")
    v0_dir = os.path.join(tmpdir, "v0_params")

    def _configs(jax_port: int, control_port: int) -> list[dict[str, Any]]:
      cfgs = []
      for rank in range(4):
        is_trainer = rank in trainer_ranks
        cfgs.append({
            "rank": rank,
            "role": "trainer" if is_trainer else "rollout",
            "replica": (
                trainer_ranks.index(rank)
                if is_trainer
                else rollout_ranks.index(rank)
            ),
            "processes": 4,
            "width": 2,
            "trainer_replicas": 2,
            "rollout_replicas": 2,
            "jax_port": jax_port,
            "control_port": control_port,
            "ckpt_dir": ckpt_dir,
            "trainer_ranks": trainer_ranks,
        })
      return cfgs

    with ici_testing.WorkerCluster(
        "ici_multihost_worker_process",
        __file__,
        _configs,
        startup_timeout_s=_TIMEOUT_S,
        call_timeout_s=_TIMEOUT_S,
    ) as cluster:
      trainers = [cluster.handles[r] for r in trainer_ranks]
      rollouts = [cluster.handles[r] for r in rollout_ranks]
      all_handles = list(cluster.handles.values())

      # 1. Trainers run on ranks [1, 2].
      infos = ici_testing.call_concurrent(trainers, "info")
      for info in infos:
        self.assertEqual(info["trainer_processes"], [1, 2])

      # 2. Multi-host SPMD training step 1 (global batch shape (4, 8) -> sharded
      #    across FSDP=4 devices spanning processes {1, 2}).
      batch1 = {
          "x": np.arange(32, dtype=np.float32).reshape(4, 8) * 0.1,
          "y": np.ones((4, 4), dtype=np.float32) * 2.0,
      }
      ici_testing.call_concurrent(trainers, "fwd_bwd", batch1)
      upd1 = ici_testing.call_concurrent(trainers, "update")
      self.assertEqual([u["step"] for u in upd1], [1, 1])

      # Snapshot trainer weights + AdamW optimizer state at step 1.
      snap_step1 = ici_testing.call_concurrent(
          trainers, "snapshot_local_shards"
      )
      expected_w = _reconstruct_w(snap_step1)
      initial_w = np.arange(32, dtype=np.float32).reshape(8, 4) * 0.05 + 0.1
      self.assertFalse(np.array_equal(expected_w, initial_w))

      # 3. Multi-host ICI weight sync: sources={1, 2} -> destinations={0, 3}.
      _run_ici_sync(trainers, rollouts, all_handles)
      for rsnap in (r.submit("snapshot") for r in rollouts):
        np.testing.assert_array_equal(rsnap["b"], snap_step1[0]["b"])
        np.testing.assert_array_equal(rsnap["w"], expected_w)

      # 4. Sub-world Orbax checkpoint save on trainers {1, 2} while rollouts
      #    {0, 3} concurrently run independent local compute (no deadlock).
      with futures.ThreadPoolExecutor(max_workers=4) as pool:
        save_args = {
            "metadata": {"step": 1, "tag": "subworld"},
            "step": 1,
            "force": True,
        }
        save_futs = [
            pool.submit(t.submit, "save_checkpoint", save_args)
            for t in trainers
        ]
        roll_futs = [
            pool.submit(r.submit, "run_local_compute", 10) for r in rollouts
        ]
        self.assertEqual(
            [f.result(timeout=_TIMEOUT_S) for f in save_futs], [True, True]
        )
        roll_out = [f.result(timeout=_TIMEOUT_S) for f in roll_futs]
        self.assertCountEqual([o["rank"] for o in roll_out], rollout_ranks)

      # 5. Advance to step 2 so live weights & AdamW state diverge from step 1.
      batch2 = {
          "x": np.arange(32, dtype=np.float32).reshape(4, 8) * -0.2,
          "y": np.ones((4, 4), dtype=np.float32) * -5.0,
      }
      ici_testing.call_concurrent(trainers, "fwd_bwd", batch2)
      upd2 = ici_testing.call_concurrent(trainers, "update")
      self.assertEqual([u["step"] for u in upd2], [2, 2])
      snap_step2 = ici_testing.call_concurrent(
          trainers, "snapshot_local_shards"
      )
      self.assertFalse(np.array_equal(snap_step1[0]["b"], snap_step2[0]["b"]))

      # 6. Sub-world Orbax checkpoint restore (step 1) on trainers {1, 2} while
      #    rollouts {0, 3} do not enter restore (no deadlock).
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

      snap_restored = ici_testing.call_concurrent(
          trainers, "snapshot_local_shards"
      )
      _assert_shards_equal(snap_step1, snap_restored)

      # 7. Sub-world Orbax v0 save + restore (MaxText engine and
      #    load_parameters_path paths), with and without OCDBT + Zarr3, on
      #    trainers {1, 2} while rollouts {0, 3} never enter Orbax.
      for use_ocdbt in (False, True):
        v0_ok = ici_testing.call_concurrent(
            trainers,
            "test_v0_subworld_restore",
            (f"{v0_dir}_ocdbt{use_ocdbt}", use_ocdbt),
        )
        self.assertEqual(v0_ok, [True, True])
        mgr_ok = ici_testing.call_concurrent(
            trainers,
            "test_v0_manager_save_restore",
            (os.path.join(tmpdir, f"v0_mgr_ocdbt{use_ocdbt}"), use_ocdbt),
        )
        self.assertEqual(mgr_ok, [True, True])

      ici_testing.call_concurrent(trainers, "close")


if __name__ == "__main__":
  absltest.main()
