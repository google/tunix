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

"""Integration test: real ClusterOrchestrator + real gRPC workers + cleanup.

Verifies the orchestrator-exit sequence end to end over loopback gRPC:

  cluster.shutdown()  ->  gRPC ``stop`` reaches the worker and blocks while
                          it "drains" (simulated checkpoint flush)
  wait_until_workers_stopped  ->  polls ``heartbeat`` over the same channel
                          and returns only once the worker reports STOPPED
  delete_sibling_jobsets  ->  called afterwards with the run's worker names

Requires jax/flax (pulled in by ``tunix.experimental.common.datatypes``) and
grpcio, so it is skipped where those are unavailable.
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest import mock

from absl.testing import absltest

try:
  from tunix.experimental.common import datatypes
  from tunix.experimental.examples.common import orch_k8s_cleanup
  from tunix.experimental.orchestrator import orchestrator as orchestrator_lib
  from tunix.experimental.worker import remote_execution

  _IMPORT_ERROR = None
except Exception as e:  # pylint: disable=broad-exception-caught
  _IMPORT_ERROR = e


class _DrainingWorker:
  """gRPC-served worker whose stop() takes ``drain_s`` to finish.

  Mirrors the shape the orchestrator relies on: ``stop`` is synchronous and
  returns only after draining; ``heartbeat`` reports the lifecycle state.
  """

  def __init__(self, drain_s: float):
    self._drain_s = drain_s
    self.state = datatypes.WorkerState.READY
    self.stop_calls = 0
    self.heartbeat_calls = 0
    self.stop_started = threading.Event()

  def stop(self):
    self.stop_calls += 1
    self.stop_started.set()
    self.state = datatypes.WorkerState.DRAINING
    time.sleep(self._drain_s)
    self.state = datatypes.WorkerState.STOPPED
    return datatypes.Response()

  def heartbeat(self):
    self.heartbeat_calls += 1
    return datatypes.HealthReport(state=self.state)


class _GrpcWorkerProcess:
  """Runs a GrpcRemoteExecutionServer for ``worker`` on a background loop."""

  def __init__(self, worker, port: int):
    self._worker = worker
    self._port = port
    self._loop = asyncio.new_event_loop()
    self._server = None
    self._ready = threading.Event()
    self._thread = threading.Thread(target=self._run, daemon=True)

  def _run(self):
    asyncio.set_event_loop(self._loop)

    async def _main():
      self._server = remote_execution.GrpcRemoteExecutionServer(self._worker)
      await self._server.start_serving_async(self._port)
      self._ready.set()
      await self._server._server.wait_for_termination()  # pylint: disable=protected-access

    try:
      self._loop.run_until_complete(_main())
    except (asyncio.CancelledError, RuntimeError):
      pass  # cancelled from stop(); nothing to report
    finally:
      self._loop.close()

  def start(self):
    self._thread.start()
    assert self._ready.wait(30), "gRPC worker server did not start"

  def stop(self):
    # Best effort, bounded: initiating ``stop`` is enough for the process to
    # exit cleanly, and grpc.aio sometimes keeps the stop coroutine pending on
    # its internal serving loop for the long-poll timeout even after the
    # client handle is closed. Do not let that stall the suite.
    if self._loop.is_closed() or self._server is None:
      return
    grpc_server = self._server._server  # pylint: disable=protected-access
    if grpc_server is not None:
      fut = asyncio.run_coroutine_threadsafe(
          grpc_server.stop(grace=None), self._loop
      )
      try:
        fut.result(timeout=1.0)
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    self._thread.join(timeout=1.0)


def _free_port() -> int:
  import socket  # pylint: disable=g-import-not-at-top

  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


@absltest.skipIf(
    _IMPORT_ERROR is not None, f"deps unavailable: {_IMPORT_ERROR}"
)
class OrchestratorCleanupIntegrationTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._procs = []
    self._handles = []

  def tearDown(self):
    # Close client channels before stopping servers; an open long-poll
    # stream would otherwise make Server.stop() wait for the poll timeout.
    for h in self._handles:
      try:
        asyncio.run(h.close())
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    for p in self._procs:
      try:
        p.stop()
      except Exception:  # pylint: disable=broad-exception-caught
        pass
    super().tearDown()

  def _cluster_with_workers(self, workers):
    cluster = orchestrator_lib.ClusterOrchestrator()
    for worker_id, worker in workers.items():
      port = _free_port()
      proc = _GrpcWorkerProcess(worker, port)
      proc.start()
      self._procs.append(proc)
      handle = remote_execution.ActorHandle.from_address(
          f"grpc://127.0.0.1:{port}", rpc_timeout_s=30
      )
      self._handles.append(handle)
      cluster.register_worker_handle(
          worker_id=worker_id, roles=[datatypes.Role.ACTOR], handle=handle
      )
    return cluster

  def test_shutdown_waits_for_drain_then_deletes_jobsets(self):
    # Trainer drains for 2s on stop(); rollout stops instantly. The poll
    # interval is 0.25s so several heartbeats land while the trainer drains.
    trainer = _DrainingWorker(drain_s=2.0)
    rollout = _DrainingWorker(drain_s=0.0)
    cluster = self._cluster_with_workers(
        {"foo-train": trainer, "foo-roll-0": rollout}
    )

    api = mock.Mock()
    api.list_namespaced_custom_object.return_value = {
        "items": [
            {"metadata": {"name": n}}
            for n in ("foo-orch", "foo-train", "foo-roll-0", "bar-train")
        ]
    }
    timeline = []
    real_delete = orch_k8s_cleanup.delete_sibling_jobsets

    def _delete(orch_id, **kw):
      timeline.append(("delete", time.monotonic(), trainer.state))
      return real_delete(orch_id, namespace="ns1", api=api)

    t0 = time.monotonic()
    with mock.patch.object(
        orch_k8s_cleanup, "DEFAULT_WORKER_STOP_POLL_S", 0.25
    ), mock.patch.object(
        orch_k8s_cleanup, "delete_sibling_jobsets", side_effect=_delete
    ):
      orch_k8s_cleanup.shutdown_cluster_and_workers(
          cluster, orchestrator_id="foo-orch", stop_deadline_s=30
      )

    # 1. The gRPC stop reached both workers exactly once.
    self.assertEqual(trainer.stop_calls, 1)
    self.assertEqual(rollout.stop_calls, 1)
    # 2. Deletion happened only after the trainer finished draining.
    self.assertLen(timeline, 1)
    _, t_delete, state_at_delete = timeline[0]
    self.assertEqual(state_at_delete, datatypes.WorkerState.STOPPED)
    self.assertGreaterEqual(t_delete - t0, 2.0)
    # 3. Exactly this run's worker JobSets were deleted, in ns1.
    deleted = sorted(
        c.kwargs["name"]
        for c in api.delete_namespaced_custom_object.call_args_list
    )
    self.assertEqual(deleted, ["foo-roll-0", "foo-train"])
    for c in api.delete_namespaced_custom_object.call_args_list:
      self.assertEqual(c.kwargs["namespace"], "ns1")
      self.assertEqual(c.kwargs["group"], "jobset.x-k8s.io")

  def test_heartbeat_poll_sees_stopped_over_real_grpc(self):
    """wait_until_workers_stopped works against a live handle, not a fake."""
    worker = _DrainingWorker(drain_s=0.0)
    cluster = self._cluster_with_workers({"foo-train": worker})
    handles = cluster.remote_worker_handles()
    self.assertEqual(list(handles), ["foo-train"])

    # Not stopped yet -> deadline expires with the worker still pending.
    pending = orch_k8s_cleanup.wait_until_workers_stopped(
        handles, deadline_s=0.6, poll_s=0.2
    )
    self.assertEqual(pending, ["foo-train"])
    self.assertGreaterEqual(worker.heartbeat_calls, 2)

    # Stop it (directly, as the orchestrator's shutdown() would over gRPC).
    handles["foo-train"].submit("stop")
    pending = orch_k8s_cleanup.wait_until_workers_stopped(
        handles, deadline_s=5, poll_s=0.2
    )
    self.assertEqual(pending, [])

  def test_dead_worker_is_treated_as_stopped(self):
    """A worker whose server is gone must not stall the deletion."""
    cluster = orchestrator_lib.ClusterOrchestrator()
    # Nothing ever listens on this port: the channel refuses immediately,
    # which is what a crashed / already-terminated worker pod looks like.
    handle = remote_execution.ActorHandle.from_address(
        f"grpc://127.0.0.1:{_free_port()}", rpc_timeout_s=30
    )
    self._handles.append(handle)
    cluster.register_worker_handle(
        worker_id="foo-train", roles=[datatypes.Role.ACTOR], handle=handle
    )
    handles = cluster.remote_worker_handles()

    t0 = time.monotonic()
    pending = orch_k8s_cleanup.wait_until_workers_stopped(
        handles, deadline_s=30, poll_s=0.2, heartbeat_timeout_s=5
    )
    self.assertEqual(pending, [])
    self.assertLess(time.monotonic() - t0, 15)

  def test_remote_worker_handles_is_a_copy(self):
    worker = _DrainingWorker(drain_s=0.0)
    cluster = self._cluster_with_workers({"foo-train": worker})
    snapshot = cluster.remote_worker_handles()
    cluster.unregister_worker("foo-train")
    self.assertEqual(list(snapshot), ["foo-train"])
    self.assertEqual(cluster.remote_worker_handles(), {})


if __name__ == "__main__":
  absltest.main()
