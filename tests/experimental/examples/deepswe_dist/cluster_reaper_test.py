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


if __name__ == "__main__":
  absltest.main()
