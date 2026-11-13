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

r"""Renders a Cloud Monitoring dashboard from registered dist_rl OTel metrics.

All streaming instruments are declared once in `common/otel_registry.py`
(`deps = []`, pure Python stdlib so OSS and Blaze launchers never import JAX
or compiled worker extensions). This module reads that registry, groups charts
automatically by `role`, validates per-kind aggregations, and appends a Cloud
Logging `logsPanel` for `tunix.lineage.*` event logs. Adding a new spec to
`otel_registry.py` automatically adds it to the dashboard and updates
`labels["tunix_dashboard_rev"]`, so `--ensure` refreshes the shared project
dashboard in place without editing this file.

Usage:

  # Automatically create or update the shared project template dashboard:
  python -m tunix.dist_rl.distributed.deployment.otel_dashboard \
      --ensure --run_id=run_20260101_000000

  # Or render static JSON to a file:
  python -m tunix.dist_rl.distributed.deployment.otel_dashboard \
      --run_id=run_20260101_000000 --output=/tmp/dashboard.json
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import contextlib
import dataclasses
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Literal, cast
import urllib.parse

from tunix.dist_rl.common import otel_registry

DEFAULT_DISPLAY_NAME = "Tunix dist_rl: streaming OTel metrics"
DEFAULT_METRIC_PREFIX = "workload.googleapis.com"
RUN_ID_LABEL = "metric.label.tunix_run_id"
DASHBOARD_REV_LABEL = "tunix_dashboard_rev"
DEFAULT_EVENT_LOG_NAME = "tunix-dist-rl-events"
EVENT_LOG_RUN_ID_LABEL = "tunix.run_id"

Kind = otel_registry.Kind
Aligner = otel_registry.Aligner
Reducer = otel_registry.Reducer
PlotType = otel_registry.PlotType
VALID_AGGREGATIONS = otel_registry.VALID_AGGREGATIONS

_REDUCER_SUFFIX: Mapping[Reducer, str] = {
    "REDUCE_PERCENTILE_50": "p50",
    "REDUCE_PERCENTILE_90": "p90",
    "REDUCE_PERCENTILE_99": "p99",
    "REDUCE_SUM": "sum",
    "REDUCE_MEAN": "mean",
    "REDUCE_MAX": "max",
}

_LABEL_VALUE_INVALID_CHARS = re.compile(r"[^a-z0-9_-]")
_LABEL_VALUE_MAX_LEN = 63

_COLUMNS = 12
_TILE_WIDTH = 6
_TILE_HEIGHT = 4
_LOGS_PANEL_HEIGHT = 6

JsonObject = dict[str, object]
EnsureAction = Literal["created", "reused", "updated"]


@dataclasses.dataclass(frozen=True, kw_only=True)
class Chart:
  """One aggregated Cloud Monitoring chart."""

  title: str
  kind: Kind
  metrics: tuple[str, ...]
  aligner: Aligner
  reducer: Reducer
  group_by: tuple[str, ...] = ()
  y_label: str
  plot_type: PlotType = "LINE"

  def __post_init__(self) -> None:
    if not self.metrics:
      raise ValueError(f"{self.title!r}: metrics must be non-empty.")
    valid = VALID_AGGREGATIONS[self.kind]
    if (self.aligner, self.reducer) not in valid:
      raise ValueError(
          f"{self.title!r}: ({self.aligner}, {self.reducer}) is not valid for"
          f" {self.kind!r}; expected one of {sorted(valid)}."
      )


def _metric_label(field: str) -> str:
  return field if field.startswith("metric.label.") else f"metric.label.{field}"


def charts_from_spec(spec: otel_registry.InstrumentSpec) -> tuple[Chart, ...]:
  """Builds one or more `Chart` definitions automatically from `spec`."""
  aggs = spec.effective_aggregations
  group_by = tuple(_metric_label(field) for field in spec.group_by)
  charts: list[Chart] = []
  for aligner, reducer in aggs:
    title = spec.effective_title
    if len(aggs) > 1:
      title = f"{title} {_REDUCER_SUFFIX[reducer]}"
    charts.append(
        Chart(
            title=title,
            kind=spec.kind,
            metrics=(spec.name,),
            aligner=aligner,
            reducer=reducer,
            group_by=group_by,
            y_label=spec.effective_y_label,
            plot_type=spec.plot_type,
        )
    )
  return tuple(charts)


def charts_by_role(
    specs: Sequence[otel_registry.InstrumentSpec] | None = None,
) -> tuple[tuple[str, tuple[Chart, ...]], ...]:
  """Builds `(section_title, charts)` sections automatically by role."""
  sections: list[tuple[str, tuple[Chart, ...]]] = []
  for role, role_specs in otel_registry.specs_by_role(specs).items():
    role_charts: list[Chart] = []
    for spec in role_specs:
      role_charts.extend(charts_from_spec(spec))
    sections.append((otel_registry.role_title(role), tuple(role_charts)))
  return tuple(sections)


def _quote(value: str) -> str:
  return json.dumps(value)


def label_value(raw: str) -> str:
  """Sanitizes a string into a Cloud Monitoring dashboard label value."""
  cleaned = _LABEL_VALUE_INVALID_CHARS.sub("_", raw.lower())
  return cleaned[:_LABEL_VALUE_MAX_LEN] or "unknown"


def _dataset(
    chart: Chart,
    metric: str,
    *,
    metric_prefix: str,
    run_id: str | None,
    alignment_period: str,
) -> JsonObject:
  """Builds one `xyChart.dataSets` entry for `metric`."""
  metric_type = f"{metric_prefix.rstrip('/')}/{metric}"
  clauses = [f"metric.type={_quote(metric_type)}"]
  if run_id:
    clauses.append(f"{RUN_ID_LABEL}={_quote(run_id)}")
  group_by = list(chart.group_by)
  if not run_id and RUN_ID_LABEL not in group_by:
    group_by = [RUN_ID_LABEL, *group_by]
  aggregation: JsonObject = {
      "alignmentPeriod": alignment_period,
      "perSeriesAligner": chart.aligner,
      "crossSeriesReducer": chart.reducer,
  }
  if group_by:
    aggregation["groupByFields"] = group_by
  return {
      "plotType": chart.plot_type,
      "timeSeriesQuery": {
          "timeSeriesFilter": {
              "filter": " AND ".join(clauses),
              "aggregation": aggregation,
          }
      },
  }


def xy_chart_widget(
    chart: Chart,
    *,
    metric_prefix: str,
    run_id: str | None,
    alignment_period: str,
) -> JsonObject:
  """Renders `chart` as a Cloud Monitoring `Widget` dict."""
  return {
      "title": chart.title,
      "xyChart": {
          "dataSets": [
              _dataset(
                  chart,
                  metric,
                  metric_prefix=metric_prefix,
                  run_id=run_id,
                  alignment_period=alignment_period,
              )
              for metric in chart.metrics
          ],
          "yAxis": {"label": chart.y_label, "scale": "LINEAR"},
      },
  }


def section_header(title: str) -> JsonObject:
  """Renders a full-width section banner widget."""
  return {
      "text": {
          "content": f"### {title}",
          "format": "MARKDOWN",
      }
  }


def logs_panel_widget(
    title: str,
    *,
    event_log_name: str,
    run_id: str | None,
) -> JsonObject:
  """Renders a Cloud Logging `logsPanel` widget for lineage event logs.

  When `run_id` is None (the shared `--ensure` template), Cloud Monitoring's
  `METRIC_LABEL` dashboard filter applies only to metric charts, so the panel
  shows all runs in the project (filter by `labels."tunix.run_id"` in Logs
  Explorer to isolate one run).

  Args:
    title: Base widget title.
    event_log_name: Cloud Logging log ID for lineage events.
    run_id: Optional `tunix_run_id` to filter logs by; None for shared template.

  Returns:
    A Cloud Monitoring `logsPanel` widget dictionary.
  """
  clauses = [f"log_id({_quote(event_log_name)})"]
  panel_title = title
  if run_id:
    clauses.append(f"labels.{_quote(EVENT_LOG_RUN_ID_LABEL)}={_quote(run_id)}")
  else:
    panel_title = (
        f'{title} (all runs; filter labels."{EVENT_LOG_RUN_ID_LABEL}" in Logs'
        " Explorer)"
    )
  return {
      "title": panel_title,
      "logsPanel": {
          "filter": " AND ".join(clauses),
      },
  }


def _append_section_tiles(
    tiles: list[JsonObject],
    *,
    y: int,
    title: str,
    charts: Sequence[Chart],
    metric_prefix: str,
    run_id: str | None,
    alignment_period: str,
) -> int:
  """Appends a section header and two-column chart tiles, returning next `y`."""
  tiles.append({
      "xPos": 0,
      "yPos": y,
      "width": _COLUMNS,
      "height": 1,
      "widget": section_header(title),
  })
  charts_y = y + 1
  num_rows = (len(charts) + 1) // 2
  for idx, chart in enumerate(charts):
    row, col = divmod(idx, 2)
    tiles.append({
        "xPos": col * _TILE_WIDTH,
        "yPos": charts_y + row * _TILE_HEIGHT,
        "width": _TILE_WIDTH,
        "height": _TILE_HEIGHT,
        "widget": xy_chart_widget(
            chart,
            metric_prefix=metric_prefix,
            run_id=run_id,
            alignment_period=alignment_period,
        ),
    })
  return charts_y + num_rows * _TILE_HEIGHT


def _dashboard_revision(
    *,
    dashboard_filters: Sequence[JsonObject],
    mosaic_layout: JsonObject,
) -> str:
  """Returns a 12-char content hash of the rendered dashboard structure."""
  payload = json.dumps(
      {"dashboardFilters": dashboard_filters, "mosaicLayout": mosaic_layout},
      sort_keys=True,
      separators=(",", ":"),
  )
  return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def render_dashboard(
    *,
    display_name: str = DEFAULT_DISPLAY_NAME,
    metric_prefix: str = DEFAULT_METRIC_PREFIX,
    run_id: str | None = None,
    alignment_period: str = "60s",
    event_log_name: str = DEFAULT_EVENT_LOG_NAME,
    specs: Sequence[otel_registry.InstrumentSpec] | None = None,
) -> JsonObject:
  """Renders the dashboard as a Cloud Monitoring `Dashboard` JSON object."""
  tiles: list[JsonObject] = []
  y = 0
  for section_title, charts in charts_by_role(specs):
    y = _append_section_tiles(
        tiles,
        y=y,
        title=section_title,
        charts=charts,
        metric_prefix=metric_prefix,
        run_id=run_id,
        alignment_period=alignment_period,
    )
  tiles.append({
      "xPos": 0,
      "yPos": y,
      "width": _COLUMNS,
      "height": 1,
      "widget": section_header("Event logs (tunix.lineage.*)"),
  })
  tiles.append({
      "xPos": 0,
      "yPos": y + 1,
      "width": _COLUMNS,
      "height": _LOGS_PANEL_HEIGHT,
      "widget": logs_panel_widget(
          "Trajectory lineage events",
          event_log_name=event_log_name,
          run_id=run_id,
      ),
  })
  dashboard_filters: list[JsonObject] = []
  if not run_id:
    dashboard_filters.append({
        "filterType": "METRIC_LABEL",
        "labelKey": "tunix_run_id",
    })
  mosaic_layout: JsonObject = {"columns": _COLUMNS, "tiles": tiles}
  labels = {
      "tunix": "dist_rl",
      DASHBOARD_REV_LABEL: _dashboard_revision(
          dashboard_filters=dashboard_filters,
          mosaic_layout=mosaic_layout,
      ),
  }
  if run_id:
    labels["tunix_run_id"] = label_value(run_id)
  return {
      "displayName": display_name,
      "dashboardFilters": dashboard_filters,
      "labels": labels,
      "mosaicLayout": mosaic_layout,
  }


def dashboard_url(
    *,
    dashboard_id: str,
    project: str,
    run_id: str | None = None,
) -> str:
  """Builds a Cloud Monitoring dashboard URL, optionally filtered to `run_id`."""
  base = (
      "https://console.cloud.google.com/monitoring/dashboards/builder/"
      + urllib.parse.quote(dashboard_id, safe="")
  )
  if run_id:
    escaped = (
        run_id.replace("\\", "\\\\").replace(",", "\\,").replace("+", "\\+")
    )
    encoded_run = urllib.parse.quote(escaped, safe="-._~")
    base += f";filters=type:mlabel,key:tunix_run_id,val:{encoded_run}"
  return base + "?project=" + urllib.parse.quote(project, safe="")


_GCLOUD_TIMEOUT_S = 30.0


def _run_gcloud(
    args: Sequence[str],
    *,
    run_cmd: Callable[..., subprocess.CompletedProcess[str]],
) -> str:
  """Runs a non-interactive `gcloud` command and returns its stripped stdout."""
  try:
    result = run_cmd(
        ["gcloud", *args, "--quiet"],
        check=False,
        capture_output=True,
        stdin=subprocess.DEVNULL,
        text=True,
        timeout=_GCLOUD_TIMEOUT_S,
    )
  except (OSError, subprocess.TimeoutExpired) as exc:
    raise RuntimeError(f"failed to invoke gcloud: {exc}") from exc
  if result.returncode != 0:
    detail = result.stderr.strip() or result.stdout.strip()
    raise RuntimeError(
        f"gcloud {' '.join(args[:3])} exited {result.returncode}: {detail}"
    )
  return result.stdout.strip()


def _resolve_project(
    project: str | None,
    *,
    run_cmd: Callable[..., subprocess.CompletedProcess[str]],
) -> str:
  """Resolves the target GCP project from argument, env, or gcloud config."""
  candidate = (project or "").strip() or os.environ.get(
      "GOOGLE_CLOUD_PROJECT", ""
  ).strip()
  if not candidate:
    raw = _run_gcloud(["config", "get-value", "project"], run_cmd=run_cmd)
    candidate = "" if raw == "(unset)" else raw
  if not candidate:
    raise RuntimeError(
        "No GCP project configured (set GOOGLE_CLOUD_PROJECT, --project, or"
        " `gcloud config set project`)."
    )
  return candidate


def _dashboard_id_from_name(name: str) -> str:
  """Extracts the trailing dashboard id from `projects/<p>/dashboards/<id>`."""
  dashboard_id = name.strip().rsplit("/", 1)[-1]
  if not dashboard_id:
    raise RuntimeError(f"Unexpected dashboard resource name: {name!r}")
  return dashboard_id


def _apply_dashboard_config(
    gcloud_subcmd: Sequence[str],
    payload: Mapping[str, object],
    *,
    project: str,
    run_cmd: Callable[..., subprocess.CompletedProcess[str]],
) -> JsonObject:
  """Writes `payload` to a temp JSON file and runs `gcloud` create/update."""
  tmp_path: str | None = None
  try:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp:
      json.dump(payload, tmp, indent=2)
      tmp_path = tmp.name
    raw = _run_gcloud(
        [
            *gcloud_subcmd,
            f"--project={project}",
            f"--config-from-file={tmp_path}",
            "--format=json",
        ],
        run_cmd=run_cmd,
    )
    return json.loads(raw)
  finally:
    if tmp_path is not None and os.path.exists(tmp_path):
      os.unlink(tmp_path)


def _find_owned_dashboard(
    listed: object, display_name: str
) -> tuple[str, str | None, str | None] | None:
  """Finds `(resource_name, existing_rev, etag)` for the shared template."""
  if not isinstance(listed, Sequence):
    return None
  for entry in listed:
    if (
        not isinstance(entry, Mapping)
        or entry.get("displayName") != display_name
    ):
      continue
    labels = entry.get("labels")
    if (
        not isinstance(labels, Mapping)
        or labels.get("tunix") != "dist_rl"
        or "tunix_run_id" in labels
    ):
      continue
    rev = labels.get(DASHBOARD_REV_LABEL)
    etag = entry.get("etag")
    return (
        str(entry["name"]),
        str(rev) if isinstance(rev, str) else None,
        str(etag) if isinstance(etag, str) else None,
    )
  return None


def ensure_dashboard(
    *,
    display_name: str = DEFAULT_DISPLAY_NAME,
    project: str | None = None,
    metric_prefix: str = DEFAULT_METRIC_PREFIX,
    alignment_period: str = "60s",
    event_log_name: str = DEFAULT_EVENT_LOG_NAME,
    specs: Sequence[otel_registry.InstrumentSpec] | None = None,
    run_cmd: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> tuple[str, str, EnsureAction]:
  """Finds, updates in place, or creates one shared template dashboard.

  The template dashboard always leaves `run_id=None` so every run in `project`
  shares one template and isolates its metric charts via URL filters (the
  `logsPanel` shows all runs in `project`; filter by `labels."tunix.run_id"` in
  Logs Explorer). Only shared template dashboards carrying
  `labels["tunix"] == "dist_rl"` without `labels["tunix_run_id"]` are matched.
  Whenever registered instruments change (`labels["tunix_dashboard_rev"]`
  differs), the existing dashboard is updated in place so the `dashboard_id` and
  URL stay stable.

  Args:
    display_name: Shared template dashboard display name.
    project: Target GCP project ID, or None to resolve from env / gcloud.
    metric_prefix: Metric type prefix of the googlecloud exporter.
    alignment_period: Alignment window of every aggregation.
    event_log_name: Cloud Logging log ID for lineage events.
    specs: Optional explicit instrument specs (defaults to global registry).
    run_cmd: Subprocess runner (injectable for tests).

  Returns:
    `(dashboard_id, resolved_project, action)` where `action` is `"created"`,
    `"reused"`, or `"updated"`.
  """
  resolved_project = _resolve_project(project, run_cmd=run_cmd)
  desired = render_dashboard(
      display_name=display_name,
      metric_prefix=metric_prefix,
      run_id=None,
      alignment_period=alignment_period,
      event_log_name=event_log_name,
      specs=specs,
  )
  desired_rev = cast(Mapping[str, str], desired["labels"])[DASHBOARD_REV_LABEL]
  list_args = [
      "monitoring",
      "dashboards",
      "list",
      f"--project={resolved_project}",
      "--format=json",
  ]
  listed = json.loads(_run_gcloud(list_args, run_cmd=run_cmd) or "[]")
  found = _find_owned_dashboard(listed, display_name)
  if found is None:
    created = _apply_dashboard_config(
        ["monitoring", "dashboards", "create"],
        desired,
        project=resolved_project,
        run_cmd=run_cmd,
    )
    return (
        _dashboard_id_from_name(str(created["name"])),
        resolved_project,
        "created",
    )
  resource_name, existing_rev, etag = found
  dash_id = _dashboard_id_from_name(resource_name)
  if existing_rev == desired_rev:
    return dash_id, resolved_project, "reused"
  update_payload = {**desired, **({"etag": etag} if etag else {})}
  try:
    _apply_dashboard_config(
        ["monitoring", "dashboards", "update", resource_name],
        update_payload,
        project=resolved_project,
        run_cmd=run_cmd,
    )
    return dash_id, resolved_project, "updated"
  except RuntimeError as exc:
    if isinstance(exc.__cause__, (OSError, subprocess.TimeoutExpired)):
      raise
    with contextlib.suppress(RuntimeError, ValueError):
      relisted = json.loads(_run_gcloud(list_args, run_cmd=run_cmd) or "[]")
      refreshed = _find_owned_dashboard(relisted, display_name)
      if refreshed is not None and refreshed[1] == desired_rev:
        return _dashboard_id_from_name(refreshed[0]), resolved_project, "reused"
    raise


_CLI_FLAGS: tuple[tuple[str, str, str], ...] = (
    (
        "--display_name",
        DEFAULT_DISPLAY_NAME,
        "Dashboard display name.",
    ),
    (
        "--metric_prefix",
        DEFAULT_METRIC_PREFIX,
        "Metric type prefix of the googlecloud exporter.",
    ),
    (
        "--run_id",
        "",
        "Restrict every chart to this tunix_run_id (empty: all runs).",
    ),
    (
        "--alignment_period",
        "60s",
        "Alignment window of every aggregation.",
    ),
    (
        "--event_log_name",
        DEFAULT_EVENT_LOG_NAME,
        "Cloud Logging log ID of the googlecloud exporter's event logs.",
    ),
    (
        "--project",
        "",
        (
            "GCP project ID for --ensure (defaults to GOOGLE_CLOUD_PROJECT or"
            " gcloud config)."
        ),
    ),
    (
        "--dashboard_id",
        "",
        (
            "Existing Cloud Monitoring dashboard ID (skips gcloud lookup under"
            " --ensure)."
        ),
    ),
    ("--output", "-", "Output JSON path (`-` for stdout)."),
)


def main(argv: Sequence[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  for flag, default, help_text in _CLI_FLAGS:
    parser.add_argument(flag, default=default, help=help_text)
  parser.add_argument(
      "--ensure",
      action="store_true",
      help=(
          "Ensure the shared template dashboard exists and is up to date in"
          " Cloud Monitoring, then print its URL."
      ),
  )
  args = parser.parse_args(sys.argv[1:] if argv is None else argv)
  if args.ensure:
    if args.dashboard_id:
      dash_id = args.dashboard_id
      project = _resolve_project(args.project or None, run_cmd=subprocess.run)
    else:
      dash_id, project, _ = ensure_dashboard(
          display_name=args.display_name,
          project=args.project or None,
          metric_prefix=args.metric_prefix,
          alignment_period=args.alignment_period,
          event_log_name=args.event_log_name,
      )
    sys.stdout.write(
        dashboard_url(
            dashboard_id=dash_id,
            project=project,
            run_id=args.run_id or None,
        )
        + "\n"
    )
    return
  dashboard = render_dashboard(
      display_name=args.display_name,
      metric_prefix=args.metric_prefix,
      run_id=args.run_id or None,
      alignment_period=args.alignment_period,
      event_log_name=args.event_log_name,
  )
  text = json.dumps(dashboard, indent=2) + "\n"
  if args.output == "-":
    sys.stdout.write(text)
  else:
    with open(args.output, "w") as f:
      f.write(text)


if __name__ == "__main__":
  main()
