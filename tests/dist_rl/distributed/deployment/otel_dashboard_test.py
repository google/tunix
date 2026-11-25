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

import json
import os
import subprocess
import sys
from unittest import mock

from absl.testing import absltest
from tunix.dist_rl.common import otel_registry
from tunix.dist_rl.distributed.deployment import otel_dashboard

_EXPECTED_ROLLOUT_METRICS = frozenset({
    "tunix.worker.trajectories.completed",
    "tunix.worker.turns.total",
    "tunix.worker.prompt_tokens",
    "tunix.worker.completion_tokens",
    "tunix.worker.execution_sec",
    "tunix.worker.sampler_time_s",
    "tunix.worker.env_time_sec",
    "tunix.worker.reward_time_sec",
    "tunix.worker.weight_sync.duration",
    "tunix.worker.requests_inflight",
})
_EXPECTED_TRAINER_METRICS = frozenset({
    "tunix.trainer.step.duration",
    "tunix.trainer.tokens.processed",
    "tunix.trainer.weight_sync.duration",
    "tunix.trainer.starvation.duration",
})


def _xy_widgets(dashboard: dict[str, object]) -> list[dict[str, object]]:
  tiles = dashboard["mosaicLayout"]["tiles"]
  return [tile["widget"] for tile in tiles if "xyChart" in tile["widget"]]


def _queries(widget: dict[str, object]) -> list[dict[str, object]]:
  return [
      ds["timeSeriesQuery"]["timeSeriesFilter"]
      for ds in widget["xyChart"]["dataSets"]
  ]


class OtelDashboardTest(absltest.TestCase):

  def test_auto_discovers_all_registered_worker_metrics_by_role(self):
    sections = dict(otel_dashboard.charts_by_role())
    rollout_title = otel_registry.role_title("rollout")
    trainer_title = otel_registry.role_title("trainer")
    self.assertIn(rollout_title, sections)
    self.assertIn(trainer_title, sections)

    rollout_metrics: set[str] = set()
    for chart in sections[rollout_title]:
      rollout_metrics.update(chart.metrics)
    trainer_metrics: set[str] = set()
    for chart in sections[trainer_title]:
      trainer_metrics.update(chart.metrics)
    self.assertEqual(rollout_metrics, _EXPECTED_ROLLOUT_METRICS)
    self.assertEqual(trainer_metrics, _EXPECTED_TRAINER_METRICS)
    self.assertLen(sections[rollout_title], 11)
    self.assertLen(sections[trainer_title], 4)

    all_metrics = rollout_metrics | trainer_metrics
    for metric in all_metrics:
      self.assertFalse(metric.startswith("tunix.actor."))
      self.assertFalse(metric.startswith("tunix.rewards."))

    for charts in sections.values():
      for chart in charts:
        if chart.kind == "histogram":
          self.assertEqual(chart.plot_type, "LINE", msg=chart.title)

    self.assertNotIn("tunix.dist_rl.worker", sys.modules)
    self.assertNotIn("jax", sys.modules)

  def test_new_spec_automatically_appears_under_its_role_section(self):
    custom_spec = otel_registry.InstrumentSpec(
        name="tunix.env.sandbox_acquire_sec",
        kind="histogram",
        role="env",
        unit="s",
        description="Sandbox acquire latency.",
        group_by=("backend",),
    )
    dashboard = otel_dashboard.render_dashboard(specs=(custom_spec,))
    tiles = dashboard["mosaicLayout"]["tiles"]
    headers = [
        tile["widget"]["text"]["content"]
        for tile in tiles
        if "text" in tile["widget"]
    ]
    self.assertIn(f"### {otel_registry.role_title('env')}", headers)
    widgets = _xy_widgets(dashboard)
    self.assertLen(widgets, 1)
    query = _queries(widgets[0])[0]
    self.assertIn(
        'metric.type="workload.googleapis.com/tunix.env.sandbox_acquire_sec"',
        query["filter"],
    )
    self.assertEqual(
        query["aggregation"]["groupByFields"],
        [otel_dashboard.RUN_ID_LABEL, "metric.label.backend"],
    )
    log_titles = [
        tile["widget"]["title"]
        for tile in tiles
        if "logsPanel" in tile["widget"]
    ]
    self.assertEqual(
        log_titles,
        [
            'Trajectory lineage events (all runs; filter labels."tunix.run_id"'
            " in Logs Explorer)"
        ],
    )

  def test_chart_rejects_mismatched_kind_and_aggregation(self):
    with self.assertRaises(ValueError):
      otel_dashboard.Chart(
          title="bad",
          kind="counter",
          metrics=("tunix.worker.turns.total",),
          aligner="ALIGN_MEAN",
          reducer="REDUCE_MEAN",
          y_label="turns/s",
      )

  def test_mosaic_layout_has_no_overlapping_tiles_and_stamps_rev(self):
    dashboard = otel_dashboard.render_dashboard(display_name="d")
    self.assertRegex(
        dashboard["labels"][otel_dashboard.DASHBOARD_REV_LABEL],
        r"^[0-9a-f]{12}$",
    )
    occupied: set[tuple[int, int]] = set()
    for tile in dashboard["mosaicLayout"]["tiles"]:
      x0, y0 = tile["xPos"], tile["yPos"]
      w, h = tile["width"], tile["height"]
      self.assertLessEqual(x0 + w, 12)
      for x in range(x0, x0 + w):
        for y in range(y0, y0 + h):
          self.assertNotIn((x, y), occupied)
          occupied.add((x, y))

  def test_run_id_filters_every_chart_and_logs_panel(self):
    dashboard = otel_dashboard.render_dashboard(
        display_name="d", run_id="Run.42"
    )
    self.assertEqual(dashboard["labels"]["tunix_run_id"], "run_42")
    self.assertEqual(dashboard["dashboardFilters"], [])
    for widget in _xy_widgets(dashboard):
      for query in _queries(widget):
        self.assertIn(
            f'{otel_dashboard.RUN_ID_LABEL}="Run.42"', query["filter"]
        )
        self.assertNotIn(
            otel_dashboard.RUN_ID_LABEL,
            query["aggregation"].get("groupByFields", []),
        )
    panels = [
        tile["widget"]["logsPanel"]
        for tile in dashboard["mosaicLayout"]["tiles"]
        if "logsPanel" in tile["widget"]
    ]
    self.assertEqual(
        panels,
        [{
            "filter": (
                'log_id("tunix-dist-rl-events") AND'
                ' labels."tunix.run_id"="Run.42"'
            )
        }],
    )

  def test_dashboard_url_formats_filtered_and_unfiltered_links(self):
    self.assertEqual(
        otel_dashboard.dashboard_url(dashboard_id="d/1", project="p 1"),
        "https://console.cloud.google.com/monitoring/dashboards/builder/d%2F1"
        "?project=p%201",
    )
    self.assertEqual(
        otel_dashboard.dashboard_url(
            dashboard_id="dash-1", project="proj-1", run_id="r,a+b"
        ),
        "https://console.cloud.google.com/monitoring/dashboards/builder/dash-1"
        ";filters=type:mlabel,key:tunix_run_id,val:r%5C%2Ca%5C%2Bb"
        "?project=proj-1",
    )

  def test_ensure_dashboard_reuses_up_to_date_template(self):
    expected_rev = otel_dashboard.render_dashboard()["labels"][
        otel_dashboard.DASHBOARD_REV_LABEL
    ]
    calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
      calls.append(list(cmd))
      return subprocess.CompletedProcess(
          cmd,
          0,
          stdout=json.dumps([
              {
                  "name": "projects/1/dashboards/unrelated-id",
                  "displayName": "Other dashboard",
                  "labels": {"tunix": "dist_rl"},
              },
              {
                  "name": "projects/1/dashboards/manual-same-name",
                  "displayName": otel_dashboard.DEFAULT_DISPLAY_NAME,
                  "labels": {},
              },
              {
                  "name": "projects/1/dashboards/shared-id",
                  "displayName": otel_dashboard.DEFAULT_DISPLAY_NAME,
                  "labels": {
                      "tunix": "dist_rl",
                      otel_dashboard.DASHBOARD_REV_LABEL: expected_rev,
                  },
              },
          ]),
          stderr="",
      )

    dash_id, project, action = otel_dashboard.ensure_dashboard(
        project="proj-a", run_cmd=fake_run
    )
    self.assertEqual(
        (dash_id, project, action), ("shared-id", "proj-a", "reused")
    )
    self.assertLen(calls, 1)
    self.assertIn("--quiet", calls[0])

  def test_ensure_dashboard_updates_in_place_when_revision_changes(self):
    captured_update: dict[str, object] = {}

    def fake_run(cmd, **_kwargs):
      if cmd[1:4] == ["monitoring", "dashboards", "list"]:
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps([
                {
                    "name": "projects/1/dashboards/manual-same-name",
                    "displayName": otel_dashboard.DEFAULT_DISPLAY_NAME,
                },
                {
                    "name": "projects/1/dashboards/per-run-static",
                    "displayName": otel_dashboard.DEFAULT_DISPLAY_NAME,
                    "labels": {
                        "tunix": "dist_rl",
                        "tunix_run_id": "run_old",
                        otel_dashboard.DASHBOARD_REV_LABEL: "111111111111",
                    },
                },
                {
                    "name": "projects/1/dashboards/shared-id",
                    "displayName": otel_dashboard.DEFAULT_DISPLAY_NAME,
                    "etag": "etag-v1",
                    "labels": {
                        "tunix": "dist_rl",
                        otel_dashboard.DASHBOARD_REV_LABEL: "000000000000",
                    },
                },
            ]),
            stderr="",
        )
      if cmd[1:5] == [
          "monitoring",
          "dashboards",
          "update",
          "projects/1/dashboards/shared-id",
      ]:
        cfg_arg = next(a for a in cmd if a.startswith("--config-from-file="))
        with open(cfg_arg.split("=", 1)[1]) as f:
          captured_update.update(json.load(f))
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps({"name": "projects/1/dashboards/shared-id"}),
            stderr="",
        )
      raise AssertionError(f"Unexpected command: {cmd}")

    dash_id, project, action = otel_dashboard.ensure_dashboard(
        project="proj-a", run_cmd=fake_run
    )
    self.assertEqual(
        (dash_id, project, action), ("shared-id", "proj-a", "updated")
    )
    self.assertEqual(captured_update["etag"], "etag-v1")

  def test_ensure_dashboard_recovers_from_concurrent_etag_update_race(self):
    expected_rev = otel_dashboard.render_dashboard()["labels"][
        otel_dashboard.DASHBOARD_REV_LABEL
    ]
    list_count = 0

    def fake_run(cmd, **_kwargs):
      nonlocal list_count
      if cmd[1:4] == ["monitoring", "dashboards", "list"]:
        list_count += 1
        rev = "000000000000" if list_count == 1 else expected_rev
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps([{
                "name": "projects/1/dashboards/shared-id",
                "displayName": otel_dashboard.DEFAULT_DISPLAY_NAME,
                "etag": f"etag-v{list_count}",
                "labels": {
                    "tunix": "dist_rl",
                    otel_dashboard.DASHBOARD_REV_LABEL: rev,
                },
            }]),
            stderr="",
        )
      if cmd[1:5] == [
          "monitoring",
          "dashboards",
          "update",
          "projects/1/dashboards/shared-id",
      ]:
        return subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="412 Precondition Failed: etag mismatch"
        )
      raise AssertionError(f"Unexpected command: {cmd}")

    dash_id, project, action = otel_dashboard.ensure_dashboard(
        project="proj-a", run_cmd=fake_run
    )
    self.assertEqual(
        (dash_id, project, action), ("shared-id", "proj-a", "reused")
    )
    self.assertEqual(list_count, 2)

  def test_ensure_dashboard_preserves_primary_update_error(self):
    stale_entry = json.dumps([{
        "name": "projects/1/dashboards/shared-id",
        "displayName": otel_dashboard.DEFAULT_DISPLAY_NAME,
        "etag": "etag-v1",
        "labels": {
            "tunix": "dist_rl",
            otel_dashboard.DASHBOARD_REV_LABEL: "000000000000",
        },
    }])

    def stale_relist(cmd, **_kwargs):
      if cmd[1:4] == ["monitoring", "dashboards", "list"]:
        return subprocess.CompletedProcess(
            cmd, 0, stdout=stale_entry, stderr=""
        )
      return subprocess.CompletedProcess(
          cmd, 1, stdout="", stderr="403 Permission Denied"
      )

    with self.assertRaisesRegex(RuntimeError, "403 Permission Denied"):
      otel_dashboard.ensure_dashboard(project="proj-a", run_cmd=stale_relist)

    list_calls = 0

    def broken_relist(cmd, **_kwargs):
      nonlocal list_calls
      if cmd[1:4] == ["monitoring", "dashboards", "list"]:
        list_calls += 1
        if list_calls == 1:
          return subprocess.CompletedProcess(
              cmd, 0, stdout=stale_entry, stderr=""
          )
        return subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="secondary list failure"
        )
      return subprocess.CompletedProcess(
          cmd, 1, stdout="", stderr="primary update failure"
      )

    with self.assertRaisesRegex(RuntimeError, "primary update failure"):
      otel_dashboard.ensure_dashboard(project="proj-a", run_cmd=broken_relist)

    timeout_lists = 0

    def timeout_update(cmd, **_kwargs):
      nonlocal timeout_lists
      if cmd[1:4] == ["monitoring", "dashboards", "list"]:
        timeout_lists += 1
        return subprocess.CompletedProcess(
            cmd, 0, stdout=stale_entry, stderr=""
        )
      raise subprocess.TimeoutExpired(cmd, 30.0)

    with self.assertRaisesRegex(RuntimeError, "failed to invoke gcloud"):
      otel_dashboard.ensure_dashboard(project="proj-a", run_cmd=timeout_update)
    self.assertEqual(timeout_lists, 1)

  def test_ensure_dashboard_creates_shared_template_when_missing(self):
    captured_payload: dict[str, object] = {}

    def fake_run(cmd, **_kwargs):
      if cmd[1:4] == ["config", "get-value", "project"]:
        return subprocess.CompletedProcess(
            cmd, 0, stdout="proj-cfg\n", stderr=""
        )
      if cmd[1:4] == ["monitoring", "dashboards", "list"]:
        return subprocess.CompletedProcess(cmd, 0, stdout="[]", stderr="")
      if cmd[1:4] == ["monitoring", "dashboards", "create"]:
        cfg_arg = next(a for a in cmd if a.startswith("--config-from-file="))
        with open(cfg_arg.split("=", 1)[1]) as f:
          captured_payload.update(json.load(f))
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps({"name": "projects/1/dashboards/new-id"}),
            stderr="",
        )
      raise AssertionError(f"Unexpected command: {cmd}")

    with mock.patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": ""}, clear=False):
      dash_id, project, action = otel_dashboard.ensure_dashboard(
          run_cmd=fake_run
      )
    self.assertEqual(
        (dash_id, project, action), ("new-id", "proj-cfg", "created")
    )
    self.assertEqual(
        captured_payload["dashboardFilters"],
        [{"filterType": "METRIC_LABEL", "labelKey": "tunix_run_id"}],
    )

  def test_ensure_dashboard_fails_clearly_when_project_or_gcloud_missing(self):
    def unset_project(cmd, **_kwargs):
      return subprocess.CompletedProcess(cmd, 0, stdout="(unset)\n", stderr="")

    with mock.patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": ""}, clear=False):
      with self.assertRaisesRegex(RuntimeError, "No GCP project configured"):
        otel_dashboard.ensure_dashboard(run_cmd=unset_project)

    def missing_gcloud(cmd, **_kwargs):
      del cmd
      raise FileNotFoundError("gcloud not found")

    with self.assertRaisesRegex(RuntimeError, "failed to invoke gcloud"):
      otel_dashboard.ensure_dashboard(project="proj-a", run_cmd=missing_gcloud)

    def timed_out_gcloud(cmd, **_kwargs):
      raise subprocess.TimeoutExpired(cmd, 30.0)

    with self.assertRaisesRegex(RuntimeError, "failed to invoke gcloud"):
      otel_dashboard.ensure_dashboard(
          project="proj-a", run_cmd=timed_out_gcloud
      )

  def test_main_ensure_prints_url_without_gcloud_when_dashboard_id_set(self):
    out = mock.MagicMock()
    with (
        mock.patch.object(
            otel_dashboard, "ensure_dashboard", autospec=True
        ) as ensure_mock,
        mock.patch.object(
            otel_dashboard.subprocess, "run", autospec=True
        ) as run_mock,
        mock.patch.object(otel_dashboard.sys, "stdout", out),
    ):
      otel_dashboard.main([
          "--ensure",
          "--dashboard_id=dash-9",
          "--project=proj-9",
          "--run_id=run-9",
      ])
    ensure_mock.assert_not_called()
    run_mock.assert_not_called()
    out.write.assert_called_once_with(
        "https://console.cloud.google.com/monitoring/dashboards/builder/dash-9"
        ";filters=type:mlabel,key:tunix_run_id,val:run-9?project=proj-9\n"
    )

  def test_main_ensure_honors_dashboard_id_with_env_project(self):
    out = mock.MagicMock()
    with (
        mock.patch.dict(
            os.environ, {"GOOGLE_CLOUD_PROJECT": "proj-env"}, clear=False
        ),
        mock.patch.object(
            otel_dashboard, "ensure_dashboard", autospec=True
        ) as ensure_mock,
        mock.patch.object(
            otel_dashboard.subprocess, "run", autospec=True
        ) as run_mock,
        mock.patch.object(otel_dashboard.sys, "stdout", out),
    ):
      otel_dashboard.main([
          "--ensure",
          "--dashboard_id=dash-9",
          "--run_id=run-9",
      ])
    ensure_mock.assert_not_called()
    run_mock.assert_not_called()
    out.write.assert_called_once_with(
        "https://console.cloud.google.com/monitoring/dashboards/builder/dash-9"
        ";filters=type:mlabel,key:tunix_run_id,val:run-9?project=proj-env\n"
    )


if __name__ == "__main__":
  absltest.main()
