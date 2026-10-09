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

"""Unit tests for the orchestrator-side worker JobSet teardown."""

from __future__ import annotations

import enum
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized

try:
  from tunix.experimental.examples.deepswe_dist import orch_k8s_cleanup
except (ImportError, ModuleNotFoundError):
  _module_path = (
      Path(__file__).resolve().parents[4]
      / "tunix"
      / "experimental"
      / "examples"
      / "deepswe_dist"
      / "orch_k8s_cleanup.py"
  )
  _spec = importlib.util.spec_from_file_location(
      "orch_k8s_cleanup", _module_path
  )
  orch_k8s_cleanup = importlib.util.module_from_spec(_spec)
  _spec.loader.exec_module(orch_k8s_cleanup)


class _State(str, enum.Enum):
  """Mirrors ``datatypes.WorkerState`` (a ``str`` enum) without importing it."""

  READY = "READY"
  DRAINING = "DRAINING"
  STOPPED = "STOPPED"


class _Report:

  def __init__(self, state):
    self.state = state


class _FakeHandle:
  """ActorHandle stand-in whose ``submit("heartbeat")`` replays a script."""

  def __init__(self, script):
    # Each entry is a state, an exception instance to raise, or the string
    # "hang" to block past the heartbeat timeout.
    self._script = list(script)
    self.calls = 0

  def submit(self, method, *args, **kwargs):
    assert method == "heartbeat", method
    self.calls += 1
    item = self._script.pop(0) if len(self._script) > 1 else self._script[0]
    if isinstance(item, BaseException):
      raise item
    if item == "hang":
      import time  # pylint: disable=g-import-not-at-top

      time.sleep(10)
    return _Report(item)


class _FakeJobSetApi:
  """Minimal ``CustomObjectsApi`` double for list/delete of JobSets."""

  def __init__(self, names, fail_delete=()):
    self._names = list(names)
    self._fail_delete = set(fail_delete)
    self.deleted = []
    self.list_calls = []

  def list_namespaced_custom_object(self, **kwargs):
    self.list_calls.append(kwargs)
    return {"items": [{"metadata": {"name": n}} for n in self._names]}

  def delete_namespaced_custom_object(self, **kwargs):
    name = kwargs["name"]
    if name in self._fail_delete:
      raise RuntimeError(f"forbidden: {name}")
    self.deleted.append((kwargs["namespace"], name))


class RunPrefixTest(parameterized.TestCase):

  @parameterized.parameters(
      ("atwigg-orch", "atwigg"),
      ("mohit-q397b-dsw-0922f-orch", "mohit-q397b-dsw-0922f"),
      ("a-orch", "a"),
  )
  def test_valid(self, orch_id, expected):
    self.assertEqual(orch_k8s_cleanup.run_prefix(orch_id), expected)

  @parameterized.parameters(
      "orchestrator",  # local launcher default
      "atwigg-train",
      "atwigg-orch-extra",
      "-orch",
      "",
      None,
  )
  def test_invalid(self, orch_id):
    self.assertIsNone(orch_k8s_cleanup.run_prefix(orch_id))


class SiblingJobSetNamesTest(absltest.TestCase):

  def test_matches_only_this_runs_workers(self):
    names = [
        "foo-orch",
        "foo-train",
        "foo-roll",
        "foo-roll-0",
        "foo-roll-15",
        "foo-eval",
        "foo-eval-0",
        "foo-v2-train",  # different run sharing a textual prefix
        "foo-v2-roll-0",
        "foobar-train",
        "foo-roll-x",
        "bar-train",
        "cluster-reaper",
    ]
    self.assertEqual(
        orch_k8s_cleanup.sibling_jobset_names("foo", names),
        ["foo-roll", "foo-roll-0", "foo-roll-15", "foo-train"],
    )

  def test_prefix_is_regex_escaped(self):
    self.assertEqual(
        orch_k8s_cleanup.sibling_jobset_names(
            "a.b", ["a.b-train", "axb-train"]
        ),
        ["a.b-train"],
    )

  def test_empty(self):
    self.assertEqual(orch_k8s_cleanup.sibling_jobset_names("foo", []), [])


class PodNamespaceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.tmpdir = self.enter_context(tempfile.TemporaryDirectory())

  def test_reads_downward_file(self):
    path = os.path.join(self.tmpdir, "namespace")
    with open(path, "w", encoding="utf-8") as f:
      f.write("trellis\n")
    with mock.patch.object(orch_k8s_cleanup, "_SA_NAMESPACE_FILE", path):
      self.assertEqual(orch_k8s_cleanup.pod_namespace("fallback"), "trellis")

  def test_falls_back_when_missing(self):
    with mock.patch.object(
        orch_k8s_cleanup, "_SA_NAMESPACE_FILE", "/nonexistent/namespace"
    ):
      self.assertEqual(orch_k8s_cleanup.pod_namespace("fallback"), "fallback")
      self.assertIsNone(orch_k8s_cleanup.pod_namespace())

  def test_falls_back_when_empty(self):
    path = os.path.join(self.tmpdir, "namespace")
    open(path, "w", encoding="utf-8").close()
    with mock.patch.object(orch_k8s_cleanup, "_SA_NAMESPACE_FILE", path):
      self.assertEqual(orch_k8s_cleanup.pod_namespace("fallback"), "fallback")


class WaitUntilWorkersStoppedTest(absltest.TestCase):

  def _run(self, handles, deadline_s=100.0, poll_s=1.0):
    clock = [0.0]
    sleeps = []

    def monotonic():
      return clock[0]

    def sleep(s):
      sleeps.append(s)
      clock[0] += s

    pending = orch_k8s_cleanup.wait_until_workers_stopped(
        handles,
        deadline_s=deadline_s,
        poll_s=poll_s,
        heartbeat_timeout_s=0.5,
        monotonic=monotonic,
        sleep=sleep,
    )
    return pending, sleeps

  def test_no_workers_returns_immediately(self):
    pending, sleeps = self._run({})
    self.assertEqual(pending, [])
    self.assertEqual(sleeps, [])

  def test_all_stopped_on_first_poll(self):
    handles = {
        "t": _FakeHandle([_State.STOPPED]),
        "r": _FakeHandle(["STOPPED"]),  # plain string state also accepted
    }
    pending, sleeps = self._run(handles)
    self.assertEqual(pending, [])
    self.assertEqual(sleeps, [])
    self.assertEqual(handles["t"].calls, 1)
    self.assertEqual(handles["r"].calls, 1)

  def test_waits_for_draining_worker(self):
    handles = {
        "t": _FakeHandle([_State.DRAINING, _State.DRAINING, _State.STOPPED]),
        "r": _FakeHandle([_State.STOPPED]),
    }
    pending, sleeps = self._run(handles)
    self.assertEqual(pending, [])
    self.assertEqual(sleeps, [1.0, 1.0])
    self.assertEqual(handles["t"].calls, 3)
    # Once a worker reports STOPPED it is not polled again.
    self.assertEqual(handles["r"].calls, 1)

  def test_unreachable_worker_counts_as_stopped(self):
    handles = {"t": _FakeHandle([ConnectionError("channel closed")])}
    pending, sleeps = self._run(handles)
    self.assertEqual(pending, [])
    self.assertEqual(sleeps, [])

  def test_deadline_returns_pending_workers(self):
    handles = {
        "t": _FakeHandle([_State.DRAINING]),
        "r": _FakeHandle([_State.STOPPED]),
    }
    pending, sleeps = self._run(handles, deadline_s=3.5, poll_s=1.0)
    self.assertEqual(pending, ["t"])
    # Polls at t=0,1,2,3 then the deadline (3.5) is reached at t=4.
    self.assertEqual(sleeps, [1.0, 1.0, 1.0, 1.0])

  def test_hung_heartbeat_does_not_block_forever(self):
    handles = {"t": _FakeHandle(["hang"])}
    pending, _ = self._run(handles, deadline_s=1.0, poll_s=1.0)
    # The RPC is bounded by heartbeat_timeout_s (0.5s); the worker is still
    # pending at the deadline rather than the call hanging for 10s.
    self.assertEqual(pending, ["t"])


class DeleteSiblingJobSetsTest(absltest.TestCase):

  def test_deletes_only_this_runs_workers(self):
    api = _FakeJobSetApi([
        "foo-orch",
        "foo-train",
        "foo-roll-0",
        "foo-roll-1",
        "foo-v2-train",
        "bar-train",
    ])
    deleted = orch_k8s_cleanup.delete_sibling_jobsets(
        "foo-orch", namespace="ns1", api=api
    )
    self.assertEqual(deleted, ["foo-roll-0", "foo-roll-1", "foo-train"])
    self.assertEqual(
        api.deleted,
        [("ns1", "foo-roll-0"), ("ns1", "foo-roll-1"), ("ns1", "foo-train")],
    )
    self.assertEqual(api.list_calls[0]["namespace"], "ns1")
    self.assertEqual(api.list_calls[0]["group"], "jobset.x-k8s.io")
    self.assertEqual(api.list_calls[0]["plural"], "jobsets")

  def test_never_deletes_the_orchestrator_itself(self):
    api = _FakeJobSetApi(["foo-orch"])
    self.assertEqual(
        orch_k8s_cleanup.delete_sibling_jobsets(
            "foo-orch", namespace="ns1", api=api
        ),
        [],
    )
    self.assertEqual(api.deleted, [])

  def test_non_orch_id_is_noop(self):
    api = _FakeJobSetApi(["orchestrator-train"])
    for orch_id in ("orchestrator", "", None):
      self.assertEqual(
          orch_k8s_cleanup.delete_sibling_jobsets(
              orch_id, namespace="ns1", api=api
          ),
          [],
      )
    self.assertEqual(api.list_calls, [])
    self.assertEqual(api.deleted, [])

  def test_one_failed_delete_does_not_stop_the_others(self):
    api = _FakeJobSetApi(
        ["foo-train", "foo-roll-0", "foo-roll-1"], fail_delete={"foo-roll-0"}
    )
    deleted = orch_k8s_cleanup.delete_sibling_jobsets(
        "foo-orch", namespace="ns1", api=api
    )
    self.assertEqual(deleted, ["foo-roll-1", "foo-train"])

  def test_list_failure_is_swallowed(self):
    api = mock.Mock()
    api.list_namespaced_custom_object.side_effect = RuntimeError("403")
    self.assertEqual(
        orch_k8s_cleanup.delete_sibling_jobsets(
            "foo-orch", namespace="ns1", api=api
        ),
        [],
    )
    api.delete_namespaced_custom_object.assert_not_called()

  def test_namespace_defaults_to_pod_namespace_then_env(self):
    api = _FakeJobSetApi(["foo-train"])
    with mock.patch.object(
        orch_k8s_cleanup, "_SA_NAMESPACE_FILE", "/nonexistent/namespace"
    ), mock.patch.dict(os.environ, {"NAMESPACE": "from-env"}):
      orch_k8s_cleanup.delete_sibling_jobsets("foo-orch", api=api)
    self.assertEqual(api.deleted, [("from-env", "foo-train")])

  def test_no_namespace_is_noop(self):
    api = _FakeJobSetApi(["foo-train"])
    env = {k: v for k, v in os.environ.items() if k != "NAMESPACE"}
    with mock.patch.object(
        orch_k8s_cleanup, "_SA_NAMESPACE_FILE", "/nonexistent/namespace"
    ), mock.patch.dict(os.environ, env, clear=True):
      self.assertEqual(
          orch_k8s_cleanup.delete_sibling_jobsets("foo-orch", api=api), []
      )
    self.assertEqual(api.list_calls, [])

  def test_in_cluster_client_is_built_lazily(self):
    """Without an injected api the kubernetes client is loaded in-cluster."""
    fake_api = _FakeJobSetApi(["foo-train"])
    fake_client = types.SimpleNamespace(CustomObjectsApi=lambda: fake_api)
    fake_config = types.SimpleNamespace(load_incluster_config=mock.Mock())
    fake_pkg = types.ModuleType("kubernetes")
    fake_pkg.client = fake_client
    fake_pkg.config = fake_config
    with mock.patch.dict(
        sys.modules,
        {
            "kubernetes": fake_pkg,
            "kubernetes.client": fake_client,
            "kubernetes.config": fake_config,
        },
    ):
      deleted = orch_k8s_cleanup.delete_sibling_jobsets(
          "foo-orch", namespace="ns1"
      )
    self.assertEqual(deleted, ["foo-train"])
    fake_config.load_incluster_config.assert_called_once()

  def test_missing_kubernetes_package_is_swallowed(self):
    with mock.patch.dict(sys.modules, {"kubernetes": None}):
      self.assertEqual(
          orch_k8s_cleanup.delete_sibling_jobsets("foo-orch", namespace="ns1"),
          [],
      )


class ShutdownClusterAndWorkersTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.events = []
    self.handles = {"t": _FakeHandle([_State.STOPPED])}
    self.cluster = mock.Mock()
    self.cluster.remote_worker_handles.return_value = self.handles
    self.cluster.shutdown.side_effect = lambda: self.events.append("shutdown")
    self.wait = self.enter_context(
        mock.patch.object(
            orch_k8s_cleanup,
            "wait_until_workers_stopped",
            side_effect=lambda *a, **k: (self.events.append("wait"), [])[1],
        )
    )
    self.delete = self.enter_context(
        mock.patch.object(
            orch_k8s_cleanup,
            "delete_sibling_jobsets",
            side_effect=lambda *a, **k: (self.events.append("delete"), [])[1],
        )
    )

  def test_order_is_shutdown_then_wait_then_delete(self):
    orch_k8s_cleanup.shutdown_cluster_and_workers(
        self.cluster, orchestrator_id="foo-orch"
    )
    self.assertEqual(self.events, ["shutdown", "wait", "delete"])
    self.wait.assert_called_once()
    self.assertIs(self.wait.call_args.args[0]["t"], self.handles["t"])
    self.delete.assert_called_once_with("foo-orch")

  def test_handles_are_snapshotted_before_shutdown(self):
    """shutdown() may unregister workers; the wait must still cover them."""

    def _shutdown():
      self.events.append("shutdown")
      self.handles.clear()

    self.cluster.shutdown.side_effect = _shutdown
    orch_k8s_cleanup.shutdown_cluster_and_workers(
        self.cluster, orchestrator_id="foo-orch"
    )
    self.assertEqual(list(self.wait.call_args.args[0]), ["t"])

  def test_shutdown_exception_still_deletes_and_propagates(self):
    self.cluster.shutdown.side_effect = RuntimeError("stop rpc timed out")
    with self.assertRaisesRegex(RuntimeError, "stop rpc timed out"):
      orch_k8s_cleanup.shutdown_cluster_and_workers(
          self.cluster, orchestrator_id="foo-orch"
      )
    self.assertEqual(self.events, ["wait", "delete"])

  def test_wait_exception_still_deletes(self):
    self.wait.side_effect = RuntimeError("boom")
    orch_k8s_cleanup.shutdown_cluster_and_workers(
        self.cluster, orchestrator_id="foo-orch"
    )
    self.assertEqual(self.events, ["shutdown", "delete"])

  def test_delete_jobsets_false_only_shuts_down(self):
    orch_k8s_cleanup.shutdown_cluster_and_workers(
        self.cluster, orchestrator_id="foo-orch", delete_jobsets=False
    )
    self.assertEqual(self.events, ["shutdown"])

  def test_stop_deadline_is_forwarded(self):
    orch_k8s_cleanup.shutdown_cluster_and_workers(
        self.cluster, orchestrator_id="foo-orch", stop_deadline_s=42.0
    )
    self.assertEqual(self.wait.call_args.kwargs["deadline_s"], 42.0)


if __name__ == "__main__":
  absltest.main()
