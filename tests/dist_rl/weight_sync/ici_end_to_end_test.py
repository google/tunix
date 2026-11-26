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

"""End-to-end test of ICI weight sync: handler, worker endpoints, transport.

A trainer process and several rollout processes join one multi-process CPU
JAX world (Gloo collectives stand in for ICI). The test plays the
orchestrator: it drives the same call sequence `WeightSyncCoordinator` does,
with the real `IciHandler`, against the real `IciSourceSynchronizer` and
`IciWeightSyncDelegate` living in those processes.
"""

from __future__ import annotations

import dataclasses
import os
import threading
import time
from typing import Any
from unittest import mock

from absl.testing import absltest
import ml_dtypes
import numpy as np
from tunix.dist_rl.weight_sync import ici_handler
from tunix.dist_rl.testing import ici_testing
from tunix.dist_rl.weight_sync import weight_sync

_STARTUP_TIMEOUT_S = 90.0
_CALL_TIMEOUT_S = 120.0

_WEIGHTS = {
    "embed": ((8, 6), np.float32),
    "block.kernel": ((4, 4), ml_dtypes.bfloat16),
    "block.bias": ((3,), np.float32),
    "head.kernel": ((6, 2), np.float32),
}


def _value(name: str, version: int) -> np.ndarray:
  shape, dtype = _WEIGHTS[name]
  base = sum(map(ord, name)) % 97 + 100 * version
  exact = (np.arange(int(np.prod(shape))) + base).reshape(shape)
  return exact.astype(dtype).astype(np.float64)


class _World:
  """A trainer (rank 0) and rollouts (ranks 1..) sharing one JAX world."""

  def __init__(self, rollouts: int, *, width: int = 2):
    processes = rollouts + 1
    self._cluster = ici_testing.WorkerCluster(
        "ici_worker_process",
        __file__,
        lambda jax_port, control_port: [
            {
                "rank": rank,
                "role": "trainer" if rank == 0 else "rollout",
                "processes": processes,
                "width": width,
                "jax_port": jax_port,
                "control_port": control_port,
            }
            for rank in range(processes)
        ],
        startup_timeout_s=_STARTUP_TIMEOUT_S,
        call_timeout_s=_CALL_TIMEOUT_S,
    )
    self.handles = self._cluster.handles
    self.trainer = self.handles[0]
    self.rollouts = [self.handles[r] for r in range(1, processes)]
    self._uuid = 0

  def close(self) -> None:
    self._cluster.close()

  def __enter__(self) -> _World:
    return self

  def __exit__(self, *exc: Any) -> None:
    self.close()

  def new_handler(self, handles=None) -> ici_handler.IciHandler:
    handles = list(self.handles.values()) if handles is None else handles
    # Tiny chunks so a sync runs several collectives, as a real model does.
    return ici_handler.IciHandler(
        lambda: handles, chunk_bytes=128, check_timeout_s=_CALL_TIMEOUT_S
    )

  def begin_round(self, handler, version: int, destinations=None):
    """Runs the coordinator's steps up to and including `pre_weight_sync`."""
    destinations = self.rollouts if destinations is None else destinations
    self._uuid += 1
    fields = dict(
        policy_version=version, req_id=f"r{self._uuid}", uuid=self._uuid
    )
    for dst in destinations:
      dst.submit("bind")
    dst_meta = []
    for dst in destinations:
      dst_meta.extend(dst.submit("metadata"))
    src_meta = self.trainer.submit("prepare", version)
    fields["source_metadata"] = tuple(src_meta)
    for meta in (*src_meta, *dst_meta):
      handler.register_work_unit(meta)
    for dst in destinations:
      dst.submit("pre", fields)
    return fields, [m.unit for m in src_meta], [m.unit for m in dst_meta]

  def transfer(self, handler, fields, src_units, dst_units):
    return handler.transfer(
        src_units, dst_units, req_id=fields["req_id"], generation=fields["uuid"]
    )

  def finish_round(self, fields, destinations=None):
    destinations = self.rollouts if destinations is None else destinations
    for dst in destinations:
      dst.submit("weight_sync", fields)
    for dst in destinations:
      dst.submit("post", fields)
    self.trainer.submit("release")


@absltest.skipUnless(
    ici_testing.gloo_available(), "needs JAX CPU collectives (gloo)"
)
class IciEndToEndTest(absltest.TestCase):

  def assert_serving(self, handle: ici_testing.ProcHandle, version: int):
    weights = handle.submit("snapshot")["weights"]
    self.assertCountEqual(weights, _WEIGHTS)
    for name, got in weights.items():
      self.assertIsNotNone(got, f"{name} was freed on worker {handle.rank}")
      np.testing.assert_array_equal(
          got, _value(name, version), err_msg=f"{name} on worker {handle.rank}"
      )

  def test_rounds_install_in_place_and_rebuild_kv_cache(self):
    with _World(rollouts=3) as world:
      handler = world.new_handler()
      for version in (1, 2):
        fields, src, dst = world.begin_round(handler, version)
        result = world.transfer(handler, fields, src, dst)
        self.assertTrue(result.success)
        self.assertIn("3 replica(s)", result.message)
        for rollout in world.rollouts:
          self.assert_serving(rollout, version)
          rollout.submit("weight_sync", fields)
          rollout.submit("post", fields)
          self.assert_serving(rollout, version)
          self.assertEqual(rollout.submit("status")["phase"], "committed")
        world.trainer.submit("release")
      calls = world.rollouts[0].submit("snapshot")["calls"]
      self.assertEqual(calls, ["delete_cache", "reinitialize_cache"] * 2)

  def test_verify_weights_compares_every_tensor_with_the_source(self):
    with mock.patch.dict(os.environ, {"VERIFY_WEIGHTS": "true"}):
      world = _World(rollouts=2)
    with world:
      handler = world.new_handler()
      fields, src, dst = world.begin_round(handler, 1)
      checksums = fields["source_metadata"][0].checksums
      self.assertEqual(checksums["__tensor_count__"], len(_WEIGHTS))
      self.assertIn("block.kernel@pw", checksums)
      world.transfer(handler, fields, src, dst)
      world.finish_round(fields)
      for rollout in world.rollouts:
        self.assert_serving(rollout, 1)

      # A destination checked against checksums of other weights fails.
      fields, src, dst = world.begin_round(handler, 2)
      world.transfer(handler, fields, src, dst)
      fields["source_metadata"] = tuple(
          dataclasses.replace(meta, checksums=checksums)
          for meta in fields["source_metadata"]
      )
      with self.assertRaisesRegex(RuntimeError, "checksum"):
        world.rollouts[0].submit("weight_sync", fields)

  def test_abort_before_transfer_keeps_old_weights_and_rebuilds_kv(self):
    with _World(rollouts=2) as world:
      handler = world.new_handler()
      fields, _, _ = world.begin_round(handler, 1)
      for rollout in world.rollouts:
        self.assertTrue(rollout.submit("abort", fields))
        self.assert_serving(rollout, 0)
        self.assertEqual(rollout.submit("status")["phase"], "aborted")
        self.assertEqual(
            rollout.submit("snapshot")["calls"],
            ["delete_cache", "reinitialize_cache"],
        )
      world.trainer.submit("release")
      # The fleet stays healthy: the next round goes through.
      fields, src, dst = world.begin_round(handler, 2)
      world.transfer(handler, fields, src, dst)
      world.finish_round(fields)
      for rollout in world.rollouts:
        self.assert_serving(rollout, 2)

  def test_abort_after_transfer_fails_fast_until_commit(self):
    with _World(rollouts=2) as world:
      handler = world.new_handler()
      fields, src, dst = world.begin_round(handler, 1)
      world.transfer(handler, fields, src, dst)
      # In place: new weights are live as soon as the transfer returns.
      for rollout in world.rollouts:
        self.assert_serving(rollout, 1)
      world.finish_round(fields)
      for rollout in world.rollouts:
        self.assert_serving(rollout, 1)

      fields, src, dst = world.begin_round(handler, 2)
      world.transfer(handler, fields, src, dst)
      # Even after weight_sync, aborting before post must still fail.
      world.rollouts[0].submit("weight_sync", fields)
      with self.assertRaisesRegex(RuntimeError, "cannot be rolled back"):
        world.rollouts[0].submit("abort", fields)
      # A subsequent check must also fail while corrupted.
      with self.assertRaisesRegex(RuntimeError, "cannot be rolled back"):
        world.rollouts[0].submit("ici_check", {"plan": None})

  def test_replica_outside_the_round_joins_but_keeps_its_weights(self):
    with _World(rollouts=3) as world:
      handler = world.new_handler()
      active, joining = world.rollouts[:2], world.rollouts[2]
      fields, src, dst = world.begin_round(handler, 1, destinations=active)
      result = world.transfer(handler, fields, src, dst)
      self.assertIn("2 replica(s)", result.message)
      world.finish_round(fields, destinations=active)
      for rollout in active:
        self.assert_serving(rollout, 1)
      self.assert_serving(joining, 0)

  def test_world_process_without_a_worker_fails_before_any_collective(self):
    with _World(rollouts=2) as world:
      handler = world.new_handler(handles=[world.trainer, world.rollouts[0]])
      fields, src, dst = world.begin_round(
          handler, 1, destinations=world.rollouts[:1]
      )
      with self.assertRaisesRegex(RuntimeError, "one live worker per process"):
        world.transfer(handler, fields, src, dst)
      self.assert_serving(world.rollouts[0], 0)


class _FakeHandle:

  def __init__(self, process, world, *, fail_on=None, block_on=None):
    self.process = process
    self.world = world
    self.fail_on = fail_on
    self.block_on = block_on
    self.calls: list[str] = []
    self.release = threading.Event()

  def submit(self, method: str, request: dict[str, Any]) -> dict[str, Any]:
    self.calls.append(method)
    if method == self.block_on and request.get("plan") is not None:
      self.release.wait(10)
    if method == self.fail_on and request.get("plan") is not None:
      raise ValueError(f"process {self.process} cannot {method}")
    if method == "ici_check":
      return {"process": self.process, "world": list(self.world)}
    return {"bytes": 8.0}


def _metadata(job, processes, *, dtype="float32", shape=(4, 2), width=2):
  return weight_sync.WorkUnitMetadata(
      unit=weight_sync.WorkUnitId(job_name=job),
      variables=(
          weight_sync.TensorMetadata(
              name="w",
              shape=shape,
              mesh_shape=(1,) * len(shape),
              layout=tuple(reversed(range(len(shape)))),
              item_size=4,
              dtype=dtype,
          ),
      ),
      transport_mode="ici",
      process_indices=tuple(processes),
      devices_per_process=width,
  )


class IciHandlerTest(absltest.TestCase):

  def _handler(self, handles, **kwargs):
    handler = ici_handler.IciHandler(lambda: handles, **kwargs)
    self.src = _metadata("trainer", [0])
    self.dst = [_metadata("rollout-1", [1]), _metadata("rollout-2", [2])]
    for meta in (self.src, *self.dst):
      handler.register_work_unit(meta)
    return handler

  def _transfer(self, handler):
    return handler.transfer(
        [self.src.unit], [m.unit for m in self.dst], req_id="r", generation=1
    )

  def test_every_worker_checks_then_transfers(self):
    handles = [_FakeHandle(p, range(3)) for p in range(3)]
    result = self._transfer(self._handler(handles))
    self.assertTrue(result.success)
    for handle in handles:
      self.assertEqual(handle.calls, ["ici_check", "ici_check", "ici_transfer"])

  def test_failed_check_fails_fast_even_when_earlier_worker_blocks(self):
    handles = [_FakeHandle(p, range(3)) for p in range(3)]
    handles[0].block_on = "ici_check"
    handles[2].fail_on = "ici_check"
    started = time.monotonic()
    try:
      with self.assertRaisesRegex(ValueError, "cannot ici_check"):
        self._transfer(self._handler(handles, check_timeout_s=5.0))
      self.assertLess(time.monotonic() - started, 2.0)
    finally:
      handles[0].release.set()
    for handle in handles:
      self.assertNotIn("ici_transfer", handle.calls)

  def test_failure_inside_the_transfer_is_outcome_unknown(self):
    handles = [_FakeHandle(p, range(3)) for p in range(3)]
    handles[1].fail_on = "ici_transfer"
    handles[2].block_on = "ici_transfer"
    with self.assertRaisesRegex(
        weight_sync.TransferOutcomeUnknownError, "failed on a participant"
    ):
      self._transfer(self._handler(handles))
    handles[2].release.set()

  def test_check_timeout_uses_shared_deadline_and_not_bare_timeout_error(self):
    handles = [_FakeHandle(p, range(3)) for p in range(3)]
    handles[0].block_on = "ici_check"
    handles[1].block_on = "ici_check"
    started = time.monotonic()
    try:
      with self.assertRaisesRegex(RuntimeError, "timed out"):
        self._transfer(self._handler(handles, check_timeout_s=0.2))
      self.assertLess(time.monotonic() - started, 1.0)
    finally:
      handles[0].release.set()
      handles[1].release.set()

  def test_workers_in_different_worlds_are_rejected(self):
    handles = [_FakeHandle(0, [0]) for _ in range(3)]
    with self.assertRaisesRegex(RuntimeError, "one live worker per process"):
      self._transfer(self._handler(handles))

  def test_dtype_mismatch_is_rejected_when_planning(self):
    handler = ici_handler.IciHandler(lambda: [])
    src = _metadata("trainer", [0], dtype="float32")
    dst = _metadata("rollout", [1], dtype="int32")
    handler.register_work_unit(src)
    handler.register_work_unit(dst)
    with self.assertRaisesRegex(ValueError, "does not match the source"):
      handler.transfer([src.unit], [dst.unit])

  def test_unequal_device_counts_are_rejected_when_planning(self):
    handler = ici_handler.IciHandler(lambda: [])
    src = _metadata("trainer", [0], width=8)
    dst = _metadata("rollout", [1], width=4)
    handler.register_work_unit(src)
    handler.register_work_unit(dst)
    with self.assertRaisesRegex(ValueError, "same number of devices"):
      handler.transfer([src.unit], [dst.unit])

  def test_units_of_another_transport_are_rejected_at_registration(self):
    handler = ici_handler.IciHandler(lambda: [])
    tcp = weight_sync.WorkUnitMetadata(
        unit=weight_sync.WorkUnitId(job_name="trainer"), transport_mode="tcp"
    )
    with self.assertRaisesRegex(ValueError, "needs 'ici'"):
      handler.register_work_unit(tcp)


if __name__ == "__main__":
  absltest.main()
