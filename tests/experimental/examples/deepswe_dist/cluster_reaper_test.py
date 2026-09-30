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

"""Unit tests for Cluster Reaper daemon."""

from __future__ import annotations

import datetime
import types
from unittest import mock
from absl.testing import absltest

import importlib.util
from pathlib import Path

try:
  from tunix.experimental.examples.deepswe_dist import cluster_reaper
except (ImportError, ModuleNotFoundError):
  _module_path = (
      Path(__file__).resolve().parents[4]
      / "tunix"
      / "experimental"
      / "examples"
      / "deepswe_dist"
      / "cluster_reaper.py"
  )
  _spec = importlib.util.spec_from_file_location("cluster_reaper", _module_path)
  cluster_reaper = importlib.util.module_from_spec(_spec)
  _spec.loader.exec_module(cluster_reaper)


class ClusterReaperTest(absltest.TestCase):

  def test_extract_run_prefix_standard_jobsets(self):
    test_cases = [
        ("atwigg-train", "atwigg"),
        ("atwigg-openhands-orch", "atwigg-openhands"),
        ("atwigg-openhands-train", "atwigg-openhands"),
        ("atwigg-openhands-roll-0", "atwigg-openhands"),
        ("atwigg-openhands-roll-15", "atwigg-openhands"),
        ("atwigg-oh-rr-orch", "atwigg-oh-rr"),
        ("atwigg-oh-rr-train", "atwigg-oh-rr"),
        ("atwigg-oh-rr-roll-5", "atwigg-oh-rr"),
        ("atwigg-oh-rr-sp-orch", "atwigg-oh-rr-sp"),
        ("atwigg-oh-rr-sp-train", "atwigg-oh-rr-sp"),
        ("atwigg-oh-rr-sp-roll-10", "atwigg-oh-rr-sp"),
        ("atwigg-fl-roll", "atwigg-fl"),
        ("jfacevedo-train", "jfacevedo"),
        ("mohit-q397b-dsw-0922f-roll-2", "mohit-q397b-dsw-0922f"),
        ("ohatwigg-orch", "ohatwigg"),
    ]
    for name, expected in test_cases:
      with self.subTest(name=name):
        self.assertEqual(cluster_reaper.extract_run_prefix(name), expected)

  def test_extract_run_prefix_standalone_jobsets(self):
    standalone = ["ctl-35b", "ab2-35b", "jt-r3-35b-off", "standalone-job"]
    for name in standalone:
      with self.subTest(name=name):
        self.assertEqual(cluster_reaper.extract_run_prefix(name), name)

  def test_reap_failed_jobsets_isolates_sibling_runs(self):
    # Simulates atwigg-train failing, ensuring atwigg-openhands is NOT touched.
    custom_api = mock.MagicMock()
    batch_api = mock.MagicMock()
    core_api = mock.MagicMock()

    now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
    custom_api.list_namespaced_custom_object.return_value = {
        "items": [
            {
                "metadata": {"name": "atwigg-train", "creationTimestamp": now_str},
                "status": {"terminalState": "Failed", "restarts": 0},
            },
            {
                "metadata": {"name": "atwigg-openhands-orch", "creationTimestamp": now_str},
                "status": {"terminalState": None, "restarts": 0},
            },
            {
                "metadata": {"name": "atwigg-openhands-train", "creationTimestamp": now_str},
                "status": {"terminalState": None, "restarts": 0},
            },
            {
                "metadata": {"name": "atwigg-openhands-roll-0", "creationTimestamp": now_str},
                "status": {"terminalState": None, "restarts": 0},
            },
        ]
    }
    core_api.list_namespaced_pod.return_value.items = []
    batch_api.list_namespaced_job.return_value.items = []

    count = cluster_reaper.reap_failed_crashed_hung_jobs(custom_api, batch_api, core_api)
    self.assertEqual(count, 1)

    # Verify ONLY atwigg-train was deleted, not atwigg-openhands-*
    deleted_names = [
        call.kwargs.get("name")
        for call in custom_api.delete_namespaced_custom_object.call_args_list
    ]
    self.assertIn("atwigg-train", deleted_names)
    self.assertNotIn("atwigg-openhands-orch", deleted_names)
    self.assertNotIn("atwigg-openhands-train", deleted_names)
    self.assertNotIn("atwigg-openhands-roll-0", deleted_names)

  def test_reap_orphaned_warmpools_and_templates(self):
    custom_api = mock.MagicMock()
    core_api = mock.MagicMock()

    old_time = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=30)
    ).isoformat()

    # Active runs: only atwigg-oh-rr exists
    custom_api.list_namespaced_custom_object.side_effect = [
        # 1. JobSets
        {"items": [{"metadata": {"name": "atwigg-oh-rr-orch"}}]},
        # 2. WarmPools
        {
            "items": [
                {
                    "metadata": {
                        "name": "pool-oh-atwigg-oh-rr-1234",
                        "creationTimestamp": old_time,
                    }
                },
                {
                    "metadata": {
                        "name": "pool-oh-dead-run-5678",
                        "creationTimestamp": old_time,
                    }
                },
            ]
        },
        # 3. Templates
        {
            "items": [
                {
                    "metadata": {
                        "name": "oh-atwigg-oh-rr-1234",
                        "creationTimestamp": old_time,
                    }
                },
                {
                    "metadata": {
                        "name": "oh-dead-run-5678",
                        "creationTimestamp": old_time,
                    }
                },
            ]
        },
    ]
    core_api.list_namespaced_pod.return_value.items = []

    count = cluster_reaper.reap_orphaned_warmpools_and_templates(custom_api, core_api)
    self.assertEqual(count, 2)  # 1 dead warmpool + 1 dead template

    deleted_names = [
        call.kwargs.get("name")
        for call in custom_api.delete_namespaced_custom_object.call_args_list
    ]
    self.assertIn("pool-oh-dead-run-5678", deleted_names)
    self.assertIn("oh-dead-run-5678", deleted_names)
    self.assertNotIn("pool-oh-atwigg-oh-rr-1234", deleted_names)
    self.assertNotIn("oh-atwigg-oh-rr-1234", deleted_names)

  def _run_zombie_detector(
      self,
      container_statuses: list[types.SimpleNamespace],
      init_container_statuses: list[types.SimpleNamespace] | None = None,
      phase: str = "Failed",
  ) -> list[str]:
    """Runs the reaper on one rollout pod; returns deleted JobSet names."""
    custom_api = mock.MagicMock()
    batch_api = mock.MagicMock()
    core_api = mock.MagicMock()
    now = datetime.datetime.now(datetime.timezone.utc)
    custom_api.list_namespaced_custom_object.return_value = {
        "items": [
            {
                "metadata": {"name": name, "creationTimestamp": now.isoformat()},
                "status": {"terminalState": None, "restarts": 0},
            }
            for name in ("run-orch", "run-train", "run-roll-0")
        ]
    }
    pod = types.SimpleNamespace(
        metadata=types.SimpleNamespace(
            name="run-roll-0-proc-0-0-abcde",
            labels={"jobset.sigs.k8s.io/jobset-name": "run-roll-0"},
            deletion_timestamp=None,
            creation_timestamp=now - datetime.timedelta(minutes=30),
        ),
        status=types.SimpleNamespace(
            phase=phase,
            container_statuses=container_statuses,
            init_container_statuses=init_container_statuses,
        ),
    )
    core_api.list_namespaced_pod.return_value.items = [pod]
    batch_api.list_namespaced_job.return_value.items = []

    cluster_reaper.reap_failed_crashed_hung_jobs(custom_api, batch_api, core_api)
    return [
        call.kwargs["name"]
        for call in custom_api.delete_namespaced_custom_object.call_args_list
    ]

  def test_startup_retry_exit_code_is_not_a_crash(self):
    self.assertEmpty(
        self._run_zombie_detector(
            [_terminated("main", cluster_reaper.STARTUP_RETRY_EXIT_CODE)]
        )
    )

  def test_nonzero_exit_code_tears_down_run(self):
    self.assertCountEqual(
        self._run_zombie_detector([_terminated("main", 1)]),
        ["run-orch", "run-train", "run-roll-0"],
    )

  def test_sidecar_crash_loop_before_main_starts_tears_down_run(self):
    # Init:CrashLoopBackOff: main never started, pod stays Pending.
    self.assertCountEqual(
        self._run_zombie_detector(
            [_waiting("main", "PodInitializing")],
            init_container_statuses=[
                _waiting("pathways-rm", "CrashLoopBackOff"),
                _running("pathways-proxy"),
            ],
            phase="Pending",
        ),
        ["run-orch", "run-train", "run-roll-0"],
    )

  def test_sidecar_terminated_by_sigterm_is_not_a_crash(self):
    # Sidecars get SIGTERM (143) whenever main exits; main's own exit code
    # decides. Here main exited 75 (startup retry), so nothing is torn down.
    self.assertEmpty(
        self._run_zombie_detector(
            [_terminated("main", cluster_reaper.STARTUP_RETRY_EXIT_CODE)],
            init_container_statuses=[
                _terminated("pathways-rm", 143),
                _terminated("pathways-proxy", 143),
            ],
        )
    )

  def test_healthy_sidecars_are_not_a_crash(self):
    self.assertEmpty(
        self._run_zombie_detector(
            [_running("main")],
            init_container_statuses=[
                _running("pathways-rm"),
                _running("pathways-proxy"),
            ],
            phase="Running",
        )
    )


def _state(
    waiting: types.SimpleNamespace | None = None,
    terminated: types.SimpleNamespace | None = None,
) -> types.SimpleNamespace:
  return types.SimpleNamespace(waiting=waiting, terminated=terminated)


def _terminated(name: str, exit_code: int) -> types.SimpleNamespace:
  return types.SimpleNamespace(
      name=name,
      state=_state(terminated=types.SimpleNamespace(exit_code=exit_code)),
  )


def _waiting(name: str, reason: str) -> types.SimpleNamespace:
  return types.SimpleNamespace(
      name=name, state=_state(waiting=types.SimpleNamespace(reason=reason))
  )


def _running(name: str) -> types.SimpleNamespace:
  return types.SimpleNamespace(name=name, state=_state())


def _sandbox_obj(
    name: str, created_by: str | None, run_id: str | None, age_min: float
) -> dict:
  labels = {}
  if created_by is not None:
    labels[cluster_reaper.CREATED_BY_LABEL] = created_by
  if run_id is not None:
    labels[cluster_reaper.SANDBOX_RUN_ID_LABEL] = run_id
  created = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
      minutes=age_min
  )
  return {
      "metadata": {
          "name": name,
          "creationTimestamp": created.isoformat(),
          "labels": labels,
      },
      "status": {"terminalState": None, "restarts": 0},
  }


def _fake_custom_api(objects_by_plural: dict[str, list[dict]]) -> mock.MagicMock:
  """CustomObjectsApi whose list call honours plural and a k=v label_selector."""

  def _list(group, version, namespace, plural, label_selector=None):
    del group, version, namespace
    items = objects_by_plural.get(plural, [])
    if label_selector is not None:
      key, value = label_selector.split("=")
      items = [o for o in items if o["metadata"]["labels"].get(key) == value]
    return {"items": items}

  api = mock.MagicMock()
  api.list_namespaced_custom_object.side_effect = _list
  return api


def _workload_pod(jobset: str, age_min: float) -> types.SimpleNamespace:
  now = datetime.datetime.now(datetime.timezone.utc)
  return types.SimpleNamespace(
      metadata=types.SimpleNamespace(
          name=f"{jobset}-proc-0-0-abcde",
          labels={"jobset.sigs.k8s.io/jobset-name": jobset},
          deletion_timestamp=None,
          creation_timestamp=now - datetime.timedelta(minutes=age_min),
      ),
      status=types.SimpleNamespace(
          phase="Running", container_statuses=[], init_container_statuses=None
      ),
  )


class SandboxRunReapTest(absltest.TestCase):

  def test_failed_run_reaps_only_its_sandbox_run_ids(self):
    failed = _sandbox_obj("dead-train", None, None, 1)
    failed["status"]["terminalState"] = "Failed"
    custom_api = _fake_custom_api({
        "jobsets": [
            failed,
            _sandbox_obj("dead-orch", None, None, 1),
            _sandbox_obj("dead-35b-orch", None, None, 1),
        ],
        # r1: orchestrator fleet, r2: a rollout worker fleet, r9: another run
        # whose prefix merely starts with "dead".
        "sandboxclaims": [
            _sandbox_obj("c1", "dead-orch", "r2", 1),
            _sandbox_obj("c2", "dead-35b-orch", "r9", 1),
        ],
        "sandboxwarmpools": [_sandbox_obj("p1", "dead-orch", "r1", 1)],
        "sandboxtemplates": [_sandbox_obj("t1", "dead-orch", "r1", 1)],
    })
    core_api = mock.MagicMock()
    core_api.list_namespaced_pod.return_value.items = []
    batch_api = mock.MagicMock()
    batch_api.list_namespaced_job.return_value.items = []

    with mock.patch.object(cluster_reaper, "sandbox_reap") as sandbox_reap:
      cluster_reaper.reap_failed_crashed_hung_jobs(custom_api, batch_api, core_api)

    self.assertCountEqual(
        sandbox_reap.call_args_list,
        [
            mock.call(
                run_id=run_id,
                namespace=cluster_reaper.NAMESPACE,
                in_cluster=True,
                delete_pods=False,
            )
            for run_id in ("r1", "r2")
        ],
    )

  def test_missing_sdk_is_logged_not_raised(self):
    with mock.patch.object(cluster_reaper, "sandbox_reap", None):
      with self.assertLogs(cluster_reaper.logger, level="ERROR"):
        self.assertEqual(
            cluster_reaper.reap_run_sandboxes(mock.MagicMock(), "dead"), 0
        )

  def test_labelled_pools_and_templates_use_exact_owner_match(self):
    custom_api = _fake_custom_api({
        "jobsets": [_sandbox_obj("atwigg-orch", None, None, 1)],
        "sandboxwarmpools": [
            _sandbox_obj("pool-oh-atwigg-35b-1", "atwigg-35b-orch", "r1", 30),
            _sandbox_obj("pool-oh-atwigg-2", "atwigg-orch", "r2", 30),
        ],
        "sandboxtemplates": [
            _sandbox_obj("oh-atwigg-35b-1", "atwigg-35b-orch", "r1", 30),
            _sandbox_obj("oh-atwigg-2", "atwigg-orch", "r2", 30),
        ],
    })
    core_api = mock.MagicMock()
    core_api.list_namespaced_pod.return_value.items = []

    cluster_reaper.reap_orphaned_warmpools_and_templates(custom_api, core_api)

    deleted = [
        c.kwargs["name"]
        for c in custom_api.delete_namespaced_custom_object.call_args_list
    ]
    self.assertCountEqual(deleted, ["pool-oh-atwigg-35b-1", "oh-atwigg-35b-1"])

  def test_labelled_claim_of_dead_run_is_reaped_while_other_runs_live(self):
    custom_api = _fake_custom_api({
        "jobsets": [_sandbox_obj("live-orch", None, None, 60)],
        "sandboxclaims": [
            # Created after the live run's pods, so the time-based rule alone
            # would keep it forever.
            _sandbox_obj("dead-claim", "dead-orch", "r1", 10),
            _sandbox_obj("live-claim", "live-orch", "r2", 10),
        ],
    })
    core_api = mock.MagicMock()
    core_api.list_namespaced_pod.return_value.items = [
        _workload_pod("live-orch", 60)
    ]

    cluster_reaper.reap_orphaned_claims(custom_api, core_api)

    deleted = [
        c.kwargs["name"]
        for c in custom_api.delete_namespaced_custom_object.call_args_list
    ]
    self.assertEqual(deleted, ["dead-claim"])


if __name__ == "__main__":
  absltest.main()
