#!/usr/bin/env python3
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
"""Monitors Tunix GitHub Actions TPU CI queue wait times and job durations.

Queries the GitHub Actions REST API for Tunix TPU workflows (`Tunix Package
Tests` and `Tunix Nightly Regression Tests`) and extracts job-level timing
telemetry for self-hosted TPU jobs (`run_core`, `run_dev`, `run_prod`,
`run_latest`) running on `linux-x86-ct6e-180-8tpu`:

  * `queue_wait_secs`: Time spent waiting for a TPU runner before execution
    starts (`started_at - created_at` for completed/in-progress jobs, or
    `now - created_at` for jobs still in `queued` state).
  * `duration_mins`: Active execution time on the TPU runner
    (`(completed_at - started_at) / 60` for completed jobs, or
    `(now - started_at) / 60` for `in_progress` jobs).

Evaluates two SLA thresholds (`evaluate_queue_sla`):
  * 24-hour queue wait p90 <= 600s (10 minutes)
  * Recent 6-hour / active single-job maximum queue wait <= 1200s (20 minutes)
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import dataclasses
import datetime
import json
import math
import os
import subprocess
import sys
from typing import Any
import urllib.request


# ==============================================================================
# 1. Configuration & Monitored Workflow Constants
# ==============================================================================

TPU_RUNNER_LABEL = "linux-x86-ct6e-180-8tpu"
TPU_JOB_SUFFIXES = ("run_core", "run_dev", "run_prod", "run_latest")

DEFAULT_WORKFLOWS = (
    "Tunix Package Tests",
    "Tunix Nightly Regression Tests",
)

DEFAULT_WINDOW_HOURS = 24.0
DEFAULT_MAX_WAIT_WINDOW_HOURS = 6.0
DEFAULT_SLA_P90_WAIT_SECS = 600.0
DEFAULT_SLA_MAX_WAIT_SECS = 1200.0


# ==============================================================================
# 2. Telemetry Data Models
# ==============================================================================


@dataclasses.dataclass(frozen=True)
class TpuJobRecord:
  """Telemetry snapshot for a single TPU CI job across its execution lifecycle.

  Attributes:
    name: Full GitHub Actions job name (e.g. `tunix_tpu_unit_tests / run_core`).
    conclusion: Final GitHub job conclusion (`"success"`, `"failure"`,
      `"cancelled"`, or `None` while `status` is `"queued"` or `"in_progress"`).
    created_at: UTC timestamp when the job was created and entered the queue.
    started_at: UTC timestamp when the job started running on a TPU runner (or
      `now` as a placeholder if still `"queued"`).
    completed_at: UTC timestamp when the job finished executing (or `now` if
      still `"queued"` or `"in_progress"`).
    queue_wait_secs: Seconds spent waiting in the runner queue prior to
      execution (`started_at - created_at`, or `now - created_at` if queued).
    duration_mins: Minutes spent actively executing on the TPU runner
      (`(completed_at - started_at) / 60`, or `0.0` if still queued).
    job_id: Unique numeric GitHub Actions job ID.
    run_id: Numeric ID of the parent GitHub Actions workflow run.
    status: Current lifecycle state (`"queued"`, `"in_progress"`, or
      `"completed"`).
  """

  name: str
  conclusion: str | None
  created_at: datetime.datetime
  started_at: datetime.datetime
  completed_at: datetime.datetime
  queue_wait_secs: float
  duration_mins: float
  job_id: int = 0
  run_id: int = 0
  status: str = "completed"


@dataclasses.dataclass(frozen=True)
class WorkflowRunRecord:
  """Telemetry container for a single GitHub Actions workflow run.

  Attributes:
    run_id: Unique numeric GitHub Actions workflow run ID.
    event: Trigger event name (e.g., `"pull_request"`, `"push"`, `"schedule"`).
    conclusion: Overall workflow run conclusion (`"success"`, `"failure"`, etc.,
      or `None` if the run is still active).
    created_at: UTC timestamp when the workflow run was triggered.
    tpu_jobs: Tuple of child `TpuJobRecord` instances belonging to this run.
  """

  run_id: int
  event: str
  conclusion: str | None
  created_at: datetime.datetime
  tpu_jobs: tuple[TpuJobRecord, ...]


# ==============================================================================
# 3. GitHub API Client & Job Telemetry Parser
# ==============================================================================

_GH_CLI_AVAILABLE = True


def gh_api(endpoint: str) -> Any:
  """Queries the GitHub REST API via `gh api` with an HTTP `urllib` fallback.

  Args:
    endpoint: Relative GitHub REST API path (e.g.
      `"repos/google/tunix/actions/workflows"`).

  Returns:
    Decoded JSON response payload (typically a `dict` or `list`).
  """
  global _GH_CLI_AVAILABLE
  if _GH_CLI_AVAILABLE:
    try:
      output = subprocess.check_output(
          ["gh", "api", endpoint],
          text=True,
          stderr=subprocess.PIPE,
          timeout=15,
      )
      return json.loads(output)
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
    ):
      _GH_CLI_AVAILABLE = False

  url = f"https://api.github.com/{endpoint.lstrip('/')}"
  headers = {"Accept": "application/vnd.github+json"}
  token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
  if token:
    headers["Authorization"] = f"Bearer {token}"
  request = urllib.request.Request(url, headers=headers)
  with urllib.request.urlopen(request, timeout=30) as response:
    return json.loads(response.read().decode("utf-8"))


def parse_iso(timestamp: str | None) -> datetime.datetime | None:
  """Parses an ISO-8601 timestamp string into a timezone-aware UTC datetime.

  Args:
    timestamp: ISO-8601 formatted string (e.g. `"2026-09-28T10:00:00Z"`) or
      `None`.

  Returns:
    A UTC-normalized `datetime.datetime` object, or `None` if `timestamp` is
    empty or `None`.
  """
  if not timestamp:
    return None
  dt = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
  if dt.tzinfo is None:
    return dt.replace(tzinfo=datetime.timezone.utc)
  return dt.astimezone(datetime.timezone.utc)


def resolve_workflow_id(
    repo: str, workflow_name: str, *, required: bool = True
) -> int | None:
  """Resolves a human-readable GitHub Actions workflow name to its numeric ID.

  Args:
    repo: GitHub repository in `"owner/name"` format (e.g. `"google/tunix"`).
    workflow_name: Display name of the workflow (e.g. `"Tunix Package Tests"`).
    required: If `True`, terminates with `sys.exit` when the workflow is not
      found; if `False`, returns `None`.

  Returns:
    The numeric workflow ID, or `None` if not found and `required=False`.
  """
  workflows = gh_api(f"repos/{repo}/actions/workflows").get("workflows", [])
  workflow_id = next(
      (
          w["id"]
          for w in workflows
          if workflow_name.lower() in str(w.get("name", "")).lower()
      ),
      None,
  )
  if workflow_id is None and required:
    sys.exit(f"ERROR: Could not find workflow '{workflow_name}' in {repo}.")
  return int(workflow_id) if workflow_id is not None else None


def _is_tpu_job(job_payload: Mapping[str, Any]) -> bool:
  """Returns `True` if `job_payload` targets the self-hosted TPU runner pool."""
  labels = job_payload.get("labels") or []
  if TPU_RUNNER_LABEL in labels:
    return True
  name = str(job_payload.get("name") or "")
  return "tpu" in name.lower() and any(
      tier in name for tier in TPU_JOB_SUFFIXES
  )


def parse_tpu_job_record(
    job_payload: Mapping[str, Any],
    *,
    run_id: int = 0,
    now: datetime.datetime | None = None,
) -> TpuJobRecord | None:
  """Parses a raw GitHub Actions job dictionary into a `TpuJobRecord`.

  Handles `completed`, `in_progress`, and `queued` jobs while ignoring non-TPU
  or `skipped` jobs.

  Args:
    job_payload: Raw job dictionary from the GitHub Actions REST API.
    run_id: Parent workflow run ID to attach to the returned record.
    now: Reference UTC timestamp used to compute live wait/elapsed times for
      `queued` and `in_progress` jobs. Defaults to current UTC time.

  Returns:
    A populated `TpuJobRecord`, or `None` if the job is not an active/completed
    TPU job.
  """
  if not _is_tpu_job(job_payload) or job_payload.get("conclusion") == "skipped":
    return None

  created_at = parse_iso(job_payload.get("created_at"))
  if created_at is None:
    return None

  now = now or datetime.datetime.now(datetime.timezone.utc)
  started_at = parse_iso(job_payload.get("started_at"))
  completed_at = parse_iso(job_payload.get("completed_at"))
  raw_status = str(job_payload.get("status") or "completed").lower()
  job_id = int(job_payload.get("id") or 0)
  name = str(job_payload.get("name") or "")

  if started_at and completed_at and completed_at >= started_at:
    duration_secs = (completed_at - started_at).total_seconds()
    return TpuJobRecord(
        name=name,
        conclusion=job_payload.get("conclusion"),
        created_at=created_at,
        started_at=started_at,
        completed_at=completed_at,
        queue_wait_secs=max(0.0, (started_at - created_at).total_seconds()),
        duration_mins=max(0.0, duration_secs / 60.0),
        job_id=job_id,
        run_id=run_id,
        status="completed",
    )

  if raw_status == "in_progress" or (
      started_at is not None and completed_at is None
  ):
    effective_start = started_at or created_at
    wait_secs = (effective_start - created_at).total_seconds()
    elapsed_secs = (now - effective_start).total_seconds()
    return TpuJobRecord(
        name=name,
        conclusion=None,
        created_at=created_at,
        started_at=effective_start,
        completed_at=now,
        queue_wait_secs=max(0.0, wait_secs),
        duration_mins=max(0.0, elapsed_secs / 60.0),
        job_id=job_id,
        run_id=run_id,
        status="in_progress",
    )

  if raw_status in ("queued", "waiting", "pending", "requested") or (
      started_at is None
      and completed_at is None
      and job_payload.get("conclusion") is None
  ):
    return TpuJobRecord(
        name=name,
        conclusion=None,
        created_at=created_at,
        started_at=now,
        completed_at=now,
        queue_wait_secs=max(0.0, (now - created_at).total_seconds()),
        duration_mins=0.0,
        job_id=job_id,
        run_id=run_id,
        status="queued",
    )

  return None


def fetch_workflow_runs(
    repo: str,
    workflow_id: int,
    num_runs: int,
    *,
    since: datetime.datetime | None = None,
    now: datetime.datetime | None = None,
) -> list[WorkflowRunRecord]:
  """Fetches recent workflow runs and parses their child TPU job records.

  Args:
    repo: GitHub repository in `"owner/name"` format.
    workflow_id: Numeric GitHub Actions workflow ID.
    num_runs: Maximum number of workflow runs to inspect.
    since: Optional UTC cutoff timestamp; completed runs created before `since`
      are skipped and stop further pagination.
    now: Reference UTC timestamp passed to `parse_tpu_job_record`.

  Returns:
    A list of `WorkflowRunRecord` objects containing parsed `TpuJobRecord`s.
  """
  now = now or datetime.datetime.now(datetime.timezone.utc)
  raw_runs: list[dict[str, Any]] = []
  page = 1

  while len(raw_runs) < num_runs:
    per_page = min(100, num_runs - len(raw_runs))
    endpoint = (
        f"repos/{repo}/actions/workflows/{workflow_id}/runs"
        f"?per_page={per_page}&page={page}"
    )
    batch = gh_api(endpoint).get("workflow_runs", [])
    if not batch:
      break

    reached_cutoff = False
    for run in batch:
      created_at = parse_iso(run.get("created_at"))
      if (
          since is not None
          and created_at is not None
          and created_at < since
          and run.get("status") == "completed"
      ):
        reached_cutoff = True
        continue
      raw_runs.append(run)

    if reached_cutoff or len(batch) < per_page:
      break
    page += 1

  records: list[WorkflowRunRecord] = []
  for run in raw_runs[:num_runs]:
    run_id = int(run["id"])
    created_at = parse_iso(run.get("created_at"))
    if created_at is None:
      continue
    jobs_payload = gh_api(
        f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100"
    ).get("jobs", [])
    tpu_jobs = [
        record
        for job in jobs_payload
        if (record := parse_tpu_job_record(job, run_id=run_id, now=now))
        is not None
    ]
    records.append(
        WorkflowRunRecord(
            run_id=run_id,
            event=str(run.get("event") or ""),
            conclusion=run.get("conclusion"),
            created_at=created_at,
            tpu_jobs=tuple(tpu_jobs),
        )
    )
  return records


# ==============================================================================
# 4. SLA Evaluation, Markdown Report Formatting & CLI Entrypoint
# ==============================================================================


def percentile(values: Sequence[float], pct: float) -> float:
  """Computes the `pct`-th percentile (`0..100`) using linear interpolation.

  Args:
    values: Sequence of numeric observations.
    pct: Target percentile in the range `[0.0, 100.0]`.

  Returns:
    The interpolated percentile value, or `0.0` if `values` is empty.
  """
  if not values:
    return 0.0
  sorted_vals = sorted(values)
  index = (len(sorted_vals) - 1) * (pct / 100.0)
  lower, upper = math.floor(index), math.ceil(index)
  if lower == upper:
    return sorted_vals[int(index)]
  weight = index - lower
  return sorted_vals[lower] * (1.0 - weight) + sorted_vals[upper] * weight


def _median_tier_duration(jobs: Sequence[TpuJobRecord], tier: str) -> float:
  """Computes the median execution duration (minutes) for finished jobs in `tier`."""
  durations = [
      j.duration_mins
      for j in jobs
      if tier in j.name and j.conclusion in ("success", "failure")
  ]
  return percentile(durations, 50)


def render_markdown_report(
    *,
    window_jobs: Sequence[TpuJobRecord],
    history_jobs: Sequence[TpuJobRecord],
    reasons: Sequence[str],
    window_hours: float,
    sla_p90_secs: float,
    sla_max_secs: float,
    recent_max_wait_secs: float | None = None,
    max_wait_window_hours: float = DEFAULT_MAX_WAIT_WINDOW_HOURS,
) -> str:
  """Renders a GitHub-Flavored Markdown summary and daily trend table.

  Args:
    window_jobs: TPU jobs observed within the active evaluation window (`24h`).
    history_jobs: TPU jobs across the fetched multi-day lookback window.
    reasons: Human-readable SLA breach descriptions (empty if healthy).
    window_hours: Duration of the SLA observation window in hours.
    sla_p90_secs: Configured 90th-percentile queue wait SLA threshold (seconds).
    sla_max_secs: Configured single-job max queue wait threshold (seconds).
    recent_max_wait_secs: Optional max queue wait in seconds across the recent
      `max_wait_window_hours` interval (and any currently active jobs).
    max_wait_window_hours: Duration of the recent single-job max wait window in
      hours (default `6.0`).

  Returns:
    Formatted Markdown string suitable for `$GITHUB_STEP_SUMMARY` and GitHub
    alert issue bodies.
  """
  waits = [j.queue_wait_secs for j in window_jobs]
  p50_wait = percentile(waits, 50)
  p90_wait = percentile(waits, 90)
  max_wait = max(waits, default=0.0)
  recent_max = (
      max_wait if recent_max_wait_secs is None else recent_max_wait_secs
  )
  queued_count = sum(1 for j in window_jobs if j.status == "queued")
  running_count = sum(1 for j in window_jobs if j.status == "in_progress")

  status_banner = "ALERT: SLA BREACH" if reasons else "HEALTHY"
  lines = [
      "## Tunix TPU CI Queue & Runtime Report",
      (
          f"- **Status**: **{status_banner}** (`{len(window_jobs)}` TPU jobs in"
          f" last `{window_hours:.0f}h`: `{queued_count}` queued,"
          f" `{running_count}` in_progress)"
      ),
  ]
  for reason in reasons:
    lines.append(f"- **Alert**: {reason}")

  tier_summary = ", ".join(
      f"`{tier}` = `{_median_tier_duration(window_jobs, tier):.1f}m`"
      for tier in TPU_JOB_SUFFIXES
  )
  lines.extend([
      "",
      "| Metric | Observed (Last 24h) | SLA Target |",
      "| :--- | :--- | :--- |",
      (
          "| **Queue Wait (`started_at - created_at`)** |"
          f" **p50 = `{p50_wait:.1f}s`**, **p90 = `{p90_wait:.1f}s`**,"
          f" **Max (24h) = `{max_wait:.1f}s`**"
          f" (**Last `{max_wait_window_hours:.0f}h` = `{recent_max:.1f}s`**) |"
          f" `p90 <= {sla_p90_secs:.0f}s` (24h),"
          f" `Max <= {sla_max_secs:.0f}s`"
          f" (last {max_wait_window_hours:.0f}h / active) |"
      ),
      f"| **Median Job Duration (`p50`)** | {tier_summary} | Informational |",
      "",
      "### Recent Daily Queue Trend",
      "",
      (
          "| Date (UTC) | Jobs | Queue p50 (`s`) | Queue p90 (`s`) |"
          " Queue Max (`s`) | `run_core` p50 (`m`) | `run_dev` p50 (`m`) |"
      ),
      "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
  ])

  jobs_by_day: dict[str, list[TpuJobRecord]] = {}
  for job in history_jobs or window_jobs:
    utc_dt = job.created_at.astimezone(datetime.timezone.utc)
    day_str = utc_dt.strftime("%Y-%m-%d")
    jobs_by_day.setdefault(day_str, []).append(job)

  for day_str in sorted(jobs_by_day.keys()):
    day_jobs = jobs_by_day[day_str]
    day_waits = [j.queue_wait_secs for j in day_jobs]
    day_p50 = percentile(day_waits, 50)
    day_p90 = percentile(day_waits, 90)
    day_max = max(day_waits, default=0.0)
    core_p50 = _median_tier_duration(day_jobs, "run_core")
    dev_p50 = _median_tier_duration(day_jobs, "run_dev")
    lines.append(
        f"| `{day_str}` | `{len(day_jobs)}` | `{day_p50:.1f}s` |"
        f" `{day_p90:.1f}s` | `{day_max:.1f}s` |"
        f" `{core_p50:.1f}m` | `{dev_p50:.1f}m` |"
    )

  return "\n".join(lines)


def evaluate_queue_sla(
    window_jobs: Sequence[TpuJobRecord],
    history_jobs: Sequence[TpuJobRecord] = (),
    *,
    window_hours: float = DEFAULT_WINDOW_HOURS,
    max_wait_window_hours: float = DEFAULT_MAX_WAIT_WINDOW_HOURS,
    sla_p90_secs: float = DEFAULT_SLA_P90_WAIT_SECS,
    sla_max_secs: float = DEFAULT_SLA_MAX_WAIT_SECS,
    now: datetime.datetime | None = None,
) -> dict[str, Any]:
  """Evaluates queue wait SLA thresholds and builds the alert payload.

  A breach is triggered if either:
    * `p90(queue_wait_secs) > sla_p90_secs` across the full `window_hours`
      (`24h`) window (default `600s` / 10 minutes), or
    * `max(queue_wait_secs) > sla_max_secs` across jobs created in the latest
      `max_wait_window_hours` (`6h` cron interval) or still actively queued /
      running (default `1200s` / 20 minutes). Scoping `sla_max_secs` to the
      latest 6h interval ensures that a single transient outlier job does not
      keep a P1 ticket open for 24 hours after the queue has already recovered.

  Args:
    window_jobs: TPU jobs in the current evaluation window (including any
      currently `queued` or `in_progress` jobs).
    history_jobs: Multi-day sequence of TPU jobs for trend rendering.
    window_hours: Observation window length in hours for `p90` (default `24.0`).
    max_wait_window_hours: Recent window length in hours for single-job `max`
      evaluation (default `6.0`, matching the 6-hour cron schedule).
    sla_p90_secs: Maximum allowed 90th-percentile queue wait in seconds.
    sla_max_secs: Maximum allowed single-job queue wait in seconds.
    now: Optional UTC reference timestamp for computing the recent
      `max_wait_window_hours` cutoff.

  Returns:
    A JSON-serializable dictionary with `"breached"`, `"reasons"`, `"metrics"`,
    and `"report_markdown"`.
  """
  waits = [j.queue_wait_secs for j in window_jobs]
  p50_wait = percentile(waits, 50)
  p90_wait = percentile(waits, 90)
  max_wait = max(waits, default=0.0)
  queued_count = sum(1 for j in window_jobs if j.status == "queued")
  running_count = sum(1 for j in window_jobs if j.status == "in_progress")

  ref_now = now or max(
      (j.created_at for j in window_jobs),
      default=datetime.datetime.now(datetime.timezone.utc),
  )
  recent_cutoff = ref_now - datetime.timedelta(hours=max_wait_window_hours)
  recent_waits = [
      j.queue_wait_secs
      for j in window_jobs
      if j.created_at >= recent_cutoff or j.status != "completed"
  ]
  recent_max_wait = max(recent_waits, default=0.0)

  reasons: list[str] = []
  if p90_wait > sla_p90_secs:
    reasons.append(
        f"24h queue wait p90 (`{p90_wait:.1f}s`) exceeded SLA"
        f" (`{sla_p90_secs:.0f}s`)."
    )
  if recent_max_wait > sla_max_secs:
    reasons.append(
        f"Recent ({max_wait_window_hours:.0f}h / active) single-job max queue"
        f" wait (`{recent_max_wait:.1f}s`) exceeded threshold"
        f" (`{sla_max_secs:.0f}s`)."
    )

  report_markdown = render_markdown_report(
      window_jobs=window_jobs,
      history_jobs=history_jobs or window_jobs,
      reasons=reasons,
      window_hours=window_hours,
      sla_p90_secs=sla_p90_secs,
      sla_max_secs=sla_max_secs,
      recent_max_wait_secs=recent_max_wait,
      max_wait_window_hours=max_wait_window_hours,
  )
  return {
      "breached": bool(reasons),
      "reasons": reasons,
      "metrics": {
          "total_tpu_jobs": len(window_jobs),
          "queued_jobs_count": queued_count,
          "in_progress_jobs_count": running_count,
          "p50_wait_secs": round(p50_wait, 1),
          "p90_wait_secs": round(p90_wait, 1),
          "max_wait_secs": round(max_wait, 1),
          "recent_max_wait_secs": round(recent_max_wait, 1),
      },
      "report_markdown": report_markdown,
  }


def main(argv: Sequence[str] | None = None) -> None:
  """CLI entrypoint that collects TPU CI telemetry and writes SLA outputs."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--repo", default="google/tunix")
  parser.add_argument("--workflows", nargs="+", default=list(DEFAULT_WORKFLOWS))
  parser.add_argument(
      "--window-hours", type=float, default=DEFAULT_WINDOW_HOURS
  )
  parser.add_argument("--max-runs", type=int, default=80)
  parser.add_argument(
      "--sla-p90-seconds", type=float, default=DEFAULT_SLA_P90_WAIT_SECS
  )
  parser.add_argument(
      "--sla-max-seconds", type=float, default=DEFAULT_SLA_MAX_WAIT_SECS
  )
  parser.add_argument("--markdown-output", default="")
  parser.add_argument("--json-output", default="")
  args = parser.parse_args(argv)

  now = datetime.datetime.now(datetime.timezone.utc)
  window_start = now - datetime.timedelta(hours=args.window_hours)
  fetch_since = now - datetime.timedelta(days=7)

  all_fetched_jobs: list[TpuJobRecord] = []
  for workflow_name in args.workflows:
    workflow_id = resolve_workflow_id(args.repo, workflow_name, required=False)
    if workflow_id is not None:
      runs = fetch_workflow_runs(
          args.repo, workflow_id, args.max_runs, since=fetch_since, now=now
      )
      for run in runs:
        all_fetched_jobs.extend(run.tpu_jobs)

  window_jobs = [
      job
      for job in all_fetched_jobs
      if job.created_at >= window_start or job.status != "completed"
  ]
  payload = evaluate_queue_sla(
      window_jobs,
      all_fetched_jobs,
      window_hours=args.window_hours,
      sla_p90_secs=args.sla_p90_seconds,
      sla_max_secs=args.sla_max_seconds,
      now=now,
  )

  print(payload["report_markdown"])
  if args.markdown_output:
    with open(args.markdown_output, "w", encoding="utf-8") as f:
      f.write(payload["report_markdown"] + "\n")
  if args.json_output:
    with open(args.json_output, "w", encoding="utf-8") as f:
      json.dump(payload, f, indent=2, sort_keys=True)


if __name__ == "__main__":
  main()
