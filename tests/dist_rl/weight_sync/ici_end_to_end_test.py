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
JAX world (gloo collectives stand in for ICI). The test plays the
orchestrator: it drives the same call sequence `WeightSyncCoordinator` does,
with the real `IciHandler`, against the real `IciSourceSynchronizer` and
`IciWeightSyncDelegate` living in those processes.
"""

import json
from multiprocessing import connection
import os
import socket
import subprocess
import sys
import tempfile
import threading

from absl.testing import absltest
import ml_dtypes
import numpy as np
from tunix.dist_rl.weight_sync import ici_handler
from tunix.dist_rl.weight_sync import weight_sync

_WORKER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "ici_worker_process.py"
)
_STARTUP_TIMEOUT_S = 90
_CALL_TIMEOUT_S = 120

# Mirrors `ici_worker_process.WEIGHTS` / `value`; that module cannot be
# imported here because importing it joins a JAX world.
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
  # Through the weight's own dtype: bfloat16 cannot hold every integer here.
  return exact.astype(dtype).astype(np.float64)


def _free_port() -> int:
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


class _Handle:
  """Stand-in for a worker `ActorHandle`: blocking `submit(method, arg)`."""

  def __init__(self, rank: int, conn: connection.Connection):
    self.rank = rank
    self._conn = conn
    self._lock = threading.Lock()

  def submit(self, method: str, argument=None):
    with self._lock:
      self._conn.send((method, argument))
      if not self._conn.poll(_CALL_TIMEOUT_S):
        raise TimeoutError(f"worker {self.rank} did not answer {method}")
      status, payload = self._conn.recv()
    if status != "ok":
      raise RuntimeError(f"worker {self.rank} {method} failed: {payload}")
    return payload


class _World:
  """A trainer (rank 0) and rollouts (ranks 1..) sharing one JAX world."""

  def __init__(self, rollouts: int, *, width: int = 2, drop_old: bool = False):
    processes = rollouts + 1
    jax_port = _free_port()
    listener = connection.Listener(("127.0.0.1", 0))
    env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower()}
    self._procs = []
    self._logs = []
    for rank in range(processes):
      cfg = dict(
          rank=rank,
          role="trainer" if rank == 0 else "rollout",
          processes=processes,
          width=width,
          jax_port=jax_port,
          control_port=listener.address[1],
          drop_old=drop_old,
      )
      log = tempfile.TemporaryFile(mode="w+")
      self._logs.append(log)
      self._procs.append(
          subprocess.Popen(
              [sys.executable, _WORKER, json.dumps(cfg)],
              stdout=log,
              stderr=subprocess.STDOUT,
              env=env,
          )
      )
    self.handles: dict[int, _Handle] = {}
    listener._listener._socket.settimeout(_STARTUP_TIMEOUT_S)  # pylint: disable=protected-access
    try:
      for _ in range(processes):
        conn = listener.accept()
        rank = conn.recv()
        self.handles[rank] = _Handle(rank, conn)
    except Exception as e:
      self.close()
      raise RuntimeError(
          "ICI test workers did not start:\n" + self.worker_logs()
      ) from e
    finally:
      listener.close()
    self.trainer = self.handles[0]
    self.rollouts = [self.handles[r] for r in range(1, processes)]
    self._uuid = 0

  def worker_logs(self, tail: int = 1500) -> str:
    out = []
    for rank, log in enumerate(self._logs):
      log.seek(0)
      out.append(f"--- worker {rank} ---\n{log.read()[-tail:]}")
    return "\n".join(out)

  def close(self) -> None:
    for proc in self._procs:
      if proc.poll() is None:
        proc.kill()
    for proc in self._procs:
      proc.wait()

  def __enter__(self):
    return self

  def __exit__(self, *exc):
    self.close()

  def new_handler(self, handles=None) -> ici_handler.IciHandler:
    handles = list(self.handles.values()) if handles is None else handles
    # Tiny chunks so a sync is several collectives, as a real model's is.
    return ici_handler.IciHandler(
        lambda: handles, chunk_bytes=128, check_timeout_s=_CALL_TIMEOUT_S
    )

  def begin_round(self, handler, version: int, destinations=None):
    """The coordinator's steps up to and including `pre_weight_sync`."""
    destinations = self.rollouts if destinations is None else destinations
    self._uuid += 1
    fields = dict(
        policy_version=version, req_id=f"r{self._uuid}", uuid=self._uuid
    )
    for d in destinations:
      d.submit("bind")
    dst_meta = [m for d in destinations for m in d.submit("metadata")]
    src_meta = self.trainer.submit("prepare", version)
    for m in (*src_meta, *dst_meta):
      handler.register_work_unit(m)
    for d in destinations:
      d.submit("pre", fields)
    return fields, [m.unit for m in src_meta], [m.unit for m in dst_meta]

  def transfer(self, handler, fields, src_units, dst_units):
    return handler.transfer(
        src_units, dst_units, req_id=fields["req_id"], generation=fields["uuid"]
    )

  def finish_round(self, fields, destinations=None):
    destinations = self.rollouts if destinations is None else destinations
    for d in destinations:
      d.submit("weight_sync", fields)
    for d in destinations:
      d.submit("post", fields)
    self.trainer.submit("release")


def _gloo_available() -> bool:
  try:
    import jax  # pylint: disable=g-import-not-at-top

    return hasattr(jax.config, "jax_cpu_collectives_implementation")
  except ImportError:
    return False


@absltest.skipUnless(_gloo_available(), "needs JAX CPU collectives (gloo)")
class IciEndToEndTest(absltest.TestCase):

  def assert_serving(self, handle: _Handle, version: int):
    weights = handle.submit("snapshot")["weights"]
    self.assertCountEqual(weights, _WEIGHTS)
    for name, got in weights.items():
      self.assertIsNotNone(got, f"{name} was freed on worker {handle.rank}")
      np.testing.assert_array_equal(
          got, _value(name, version), err_msg=f"{name} on worker {handle.rank}"
      )

  def test_staged_rounds_publish_only_in_weight_sync(self):
    with _World(rollouts=3) as world:
      handler = world.new_handler()
      for version in (1, 2):
        fields, src, dst = world.begin_round(handler, version)
        result = world.transfer(handler, fields, src, dst)
        self.assertTrue(result.success)
        self.assertIn("3 replica(s)", result.message)
        # Received but not published: the serving weights are untouched.
        for rollout in world.rollouts:
          self.assert_serving(rollout, version - 1)
        world.finish_round(fields)
        for rollout in world.rollouts:
          self.assert_serving(rollout, version)
          self.assertEqual(rollout.submit("status")["phase"], "committed")
      calls = world.rollouts[0].submit("snapshot")["calls"]
      self.assertEqual(
          calls, ["delete_cache", "reinitialize_cache"] * 2
      )

  def test_staged_abort_keeps_the_old_weights(self):
    with _World(rollouts=2) as world:
      handler = world.new_handler()
      fields, src, dst = world.begin_round(handler, 1)
      world.transfer(handler, fields, src, dst)
      for rollout in world.rollouts:
        self.assertTrue(rollout.submit("abort", fields))
        self.assert_serving(rollout, 0)
        self.assertEqual(rollout.submit("status")["phase"], "aborted")
      world.trainer.submit("release")
      # The fleet is still usable: the next round goes through.
      fields, src, dst = world.begin_round(handler, 2)
      world.transfer(handler, fields, src, dst)
      world.finish_round(fields)
      for rollout in world.rollouts:
        self.assert_serving(rollout, 2)

  def test_drop_old_installs_in_place_and_cannot_roll_back(self):
    with _World(rollouts=2, drop_old=True) as world:
      handler = world.new_handler()
      fields, src, dst = world.begin_round(handler, 1)
      world.transfer(handler, fields, src, dst)
      # In place: the new weights are live as soon as the transfer returns.
      for rollout in world.rollouts:
        self.assert_serving(rollout, 1)
      world.finish_round(fields)
      for rollout in world.rollouts:
        self.assert_serving(rollout, 1)

      fields, src, dst = world.begin_round(handler, 2)
      world.transfer(handler, fields, src, dst)
      with self.assertRaisesRegex(RuntimeError, "cannot be rolled back"):
        world.rollouts[0].submit("abort", fields)

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
      # The orchestrator lost track of one rollout that is still in the JAX
      # world. A collective without it would hang, so the handler must refuse.
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
    self.calls = []
    self.release = threading.Event()

  def submit(self, method, request):
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
    for m in (self.src, *self.dst):
      handler.register_work_unit(m)
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

  def test_failed_check_is_an_ordinary_error_and_nothing_transfers(self):
    handles = [_FakeHandle(p, range(3)) for p in range(3)]
    handles[2].fail_on = "ici_check"
    with self.assertRaisesRegex(ValueError, "cannot ici_check"):
      self._transfer(self._handler(handles))
    for handle in handles:
      self.assertNotIn("ici_transfer", handle.calls)

  def test_failure_inside_the_transfer_is_outcome_unknown(self):
    handles = [_FakeHandle(p, range(3)) for p in range(3)]
    handles[1].fail_on = "ici_transfer"
    # A healthy peer would still be blocked in the collective.
    handles[2].block_on = "ici_transfer"
    with self.assertRaises(weight_sync.TransferOutcomeUnknownError):
      self._transfer(self._handler(handles))
    handles[2].release.set()

  def test_check_timeout_is_not_reported_as_a_running_transfer(self):
    handles = [_FakeHandle(p, range(3)) for p in range(3)]
    handles[0].block_on = "ici_check"
    try:
      self._transfer(self._handler(handles, check_timeout_s=0.2))
    except TimeoutError:
      self.fail("a bare TimeoutError poisons the coordinator")
    except RuntimeError as e:
      self.assertIn("timed out", str(e))
    else:
      self.fail("expected the check to time out")
    finally:
      handles[0].release.set()

  def test_workers_in_different_worlds_are_rejected(self):
    # Each worker initialised its own single-process JAX runtime.
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
      handler.build_plan([src.unit], [dst.unit])

  def test_unequal_device_counts_are_rejected_when_planning(self):
    handler = ici_handler.IciHandler(lambda: [])
    src = _metadata("trainer", [0], width=8)
    dst = _metadata("rollout", [1], width=4)
    handler.register_work_unit(src)
    handler.register_work_unit(dst)
    with self.assertRaisesRegex(ValueError, "same number of devices"):
      handler.build_plan([src.unit], [dst.unit])

  def test_units_of_another_transport_are_rejected_at_registration(self):
    handler = ici_handler.IciHandler(lambda: [])
    tcp = weight_sync.WorkUnitMetadata(
        unit=weight_sync.WorkUnitId(job_name="trainer"), transport_mode="tcp"
    )
    with self.assertRaisesRegex(ValueError, "needs 'ici'"):
      handler.register_work_unit(tcp)


if __name__ == "__main__":
  absltest.main()
