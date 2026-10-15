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

Queries the GitHub Actions REST API for `Tunix Package Tests` and `Tunix Nightly
Regression Tests` and extracts timing metrics for self-hosted TPU jobs
(`run_core`, `run_dev`, `run_prod`, `run_latest`) on `linux-x86-ct6e-180-8tpu`:
  * `queue_wait_secs`: `started_at - created_at` (or `now - created_at` if still
    in `queued` state).
  * `duration_mins`: `(completed_at - started_at) / 60` (or
    `(now - started_at) / 60` if `in_progress`).

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
  """Telemetry snapshot for a single TPU CI job (`queued`, `in_progress`, or `completed`)."""

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
  """Telemetry container for a single GitHub Actions workflow run and its TPU jobs."""

  run_id: int
  event: str
  conclusion: str | None
  created_at: datetime.datetime
  tpu_jobs: tuple[TpuJobRecord, ...]


# ==============================================================================
# 3. GitHub API Client & Job Telemetry Parser
# ==============================================================================


def gh_api(endpoint: str) -> Any:
  """Queries the GitHub REST API via `gh api` with a `urllib.request` fallback."""
  try:
    out = subprocess.check_output(
        ["gh", "api", endpoint], text=True, stderr=subprocess.PIPE, timeout=15
    )
    return json.loads(out)
  except (subprocess.SubprocessError, FileNotFoundError):
    url = f"https://api.github.com/{endpoint.lstrip('/')}"
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
      headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
      return json.loads(resp.read().decode("utf-8"))


def parse_iso(timestamp: str | None) -> datetime.datetime | None:
  """Parses an ISO-8601 timestamp string into a UTC-normalized datetime."""
  if not timestamp:
    return None
  dt = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
  return (
      dt.replace(tzinfo=datetime.timezone.utc)
      if dt.tzinfo is None
      else dt.astimezone(datetime.timezone.utc)
  )


def resolve_workflow_id(
    repo: str, workflow_name: str, *, required: bool = True
) -> int | None:
  """Resolves a human-readable GitHub Actions workflow name to its numeric ID."""
  workflows = gh_api(f"repos/{repo}/actions/workflows").get("workflows", [])
  wf_id = next(
      (
          w["id"]
          for w in workflows
          if workflow_name.lower() in str(w.get("name", "")).lower()
      ),
      None,
  )
  if wf_id is None and required:
    sys.exit(f"ERROR: Could not find workflow '{workflow_name}' in {repo}.")
  return int(wf_id) if wf_id is not None else None


def parse_tpu_job_record(
    job_payload: Mapping[str, Any],
    *,
    run_id: int = 0,
    now: datetime.datetime | None = None,
) -> TpuJobRecord | None:
  """Parses a raw GitHub Actions job dict into a `TpuJobRecord` (or `None` if non-TPU/skipped)."""
  labels = job_payload.get("labels") or []
  name = str(job_payload.get("name") or "")
  is_tpu = TPU_RUNNER_LABEL in labels or (
      "tpu" in name.lower() and any(t in name for t in TPU_JOB_SUFFIXES)
  )
  if not is_tpu or job_payload.get("conclusion") == "skipped":
    return None

  created_at = parse_iso(job_payload.get("created_at"))
  if created_at is None:
    return None

  now = now or datetime.datetime.now(datetime.timezone.utc)
  started_at = parse_iso(job_payload.get("started_at"))
  completed_at = parse_iso(job_payload.get("completed_at"))
  raw_status = str(job_payload.get("status") or "completed").lower()
  conclusion = job_payload.get("conclusion")

  if started_at and completed_at and completed_at >= started_at:
    status = "completed"
    wait_secs = (started_at - created_at).total_seconds()
    duration_mins = (completed_at - started_at).total_seconds() / 60.0
  elif raw_status == "in_progress" or (started_at and not completed_at):
    status, conclusion = "in_progress", None
    started_at = started_at or created_at
    completed_at = now
    wait_secs = (started_at - created_at).total_seconds()
    duration_mins = (now - started_at).total_seconds() / 60.0
  elif raw_status in ("queued", "waiting", "pending", "requested") or (
      not started_at and not completed_at and conclusion is None
  ):
    status, conclusion = "queued", None
    started_at = completed_at = now
    wait_secs = (now - created_at).total_seconds()
    duration_mins = 0.0
  else:
    return None

  return TpuJobRecord(
      name=name,
      conclusion=conclusion,
      created_at=created_at,
      started_at=started_at,
      completed_at=completed_at,
      queue_wait_secs=max(0.0, wait_secs),
      duration_mins=max(0.0, duration_mins),
      job_id=int(job_payload.get("id") or 0),
      run_id=run_id,
      status=status,
  )


def fetch_workflow_runs(
    repo: str,
    workflow_id: int,
    num_runs: int,
    *,
    since: datetime.datetime | None = None,
    now: datetime.datetime | None = None,
) -> list[WorkflowRunRecord]:
  """Fetches up to `num_runs` recent workflow runs and their child `TpuJobRecord`s."""
  now = now or datetime.datetime.now(datetime.timezone.utc)
  endpoint = (
      f"repos/{repo}/actions/workflows/{workflow_id}/runs"
      f"?per_page={min(100, num_runs)}"
  )
  raw_runs = gh_api(endpoint).get("workflow_runs", [])[:num_runs]

  records: list[WorkflowRunRecord] = []
  for run in raw_runs:
    created_at = parse_iso(run.get("created_at"))
    if created_at is None or (
        since and created_at < since and run.get("status") == "completed"
    ):
      continue
    run_id = int(run["id"])
    jobs = gh_api(f"repos/{repo}/actions/runs/{run_id}/jobs?per_page=100").get(
        "jobs", []
    )
    tpu_jobs = [
        rec
        for job in jobs
        if (rec := parse_tpu_job_record(job, run_id=run_id, now=now))
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
  """Computes the `pct`-th percentile (`0..100`) using linear interpolation."""
  if not values:
    return 0.0
  vals = sorted(values)
  idx = (len(vals) - 1) * (pct / 100.0)
  lo, hi = math.floor(idx), math.ceil(idx)
  return vals[lo] + (idx - lo) * (vals[hi] - vals[lo])


def _median_tier_duration(jobs: Sequence[TpuJobRecord], tier: str) -> float:
  """Returns median duration (minutes) for finished jobs matching `tier`."""
  return percentile(
      [
          j.duration_mins
          for j in jobs
          if tier in j.name and j.conclusion in ("success", "failure")
      ],
      50,
  )


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
  """Renders a GitHub-Flavored Markdown summary and daily queue trend table."""
  waits = [j.queue_wait_secs for j in window_jobs]
  p50_wait, p90_wait = percentile(waits, 50), percentile(waits, 90)
  max_wait = max(waits, default=0.0)
  recent_max = (
      max_wait if recent_max_wait_secs is None else recent_max_wait_secs
  )
  queued = sum(1 for j in window_jobs if j.status == "queued")
  running = sum(1 for j in window_jobs if j.status == "in_progress")
  status_banner = "ALERT: SLA BREACH" if reasons else "HEALTHY"

  tier_summary = ", ".join(
      f"`{t}` = `{_median_tier_duration(window_jobs, t):.1f}m`"
      for t in TPU_JOB_SUFFIXES
  )
  lines = [
      "## Tunix TPU CI Queue & Runtime Report",
      (
          f"- **Status**: **{status_banner}** (`{len(window_jobs)}` TPU jobs in"
          f" last `{window_hours:.0f}h`: `{queued}` queued,"
          f" `{running}` in_progress)"
      ),
      *[f"- **Alert**: {r}" for r in reasons],
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
  ]

  by_day: dict[str, list[TpuJobRecord]] = {}
  for job in history_jobs or window_jobs:
    day = job.created_at.astimezone(datetime.timezone.utc).strftime("%Y-%m-%d")
    by_day.setdefault(day, []).append(job)

  for day, day_jobs in sorted(by_day.items()):
    d_waits = [j.queue_wait_secs for j in day_jobs]
    d_max = max(d_waits, default=0.0)
    lines.append(
        f"| `{day}` | `{len(day_jobs)}` | `{percentile(d_waits, 50):.1f}s` |"
        f" `{percentile(d_waits, 90):.1f}s` | `{d_max:.1f}s` |"
        f" `{_median_tier_duration(day_jobs, 'run_core'):.1f}m` |"
        f" `{_median_tier_duration(day_jobs, 'run_dev'):.1f}m` |"
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
  """Evaluates 24h `p90` and recent 6h / active single-job `max` queue wait SLAs."""
  waits = [j.queue_wait_secs for j in window_jobs]
  p50_wait, p90_wait = percentile(waits, 50), percentile(waits, 90)
  max_wait = max(waits, default=0.0)

  ref_now = now or max(
      (j.created_at for j in window_jobs),
      default=datetime.datetime.now(datetime.timezone.utc),
  )
  recent_cutoff = ref_now - datetime.timedelta(hours=max_wait_window_hours)
  recent_max_wait = max(
      (
          j.queue_wait_secs
          for j in window_jobs
          if j.created_at >= recent_cutoff or j.status != "completed"
      ),
      default=0.0,
  )

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

  report_md = render_markdown_report(
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
          "queued_jobs_count": sum(
              1 for j in window_jobs if j.status == "queued"
          ),
          "in_progress_jobs_count": sum(
              1 for j in window_jobs if j.status == "in_progress"
          ),
          "p50_wait_secs": round(p50_wait, 1),
          "p90_wait_secs": round(p90_wait, 1),
          "max_wait_secs": round(max_wait, 1),
          "recent_max_wait_secs": round(recent_max_wait, 1),
      },
      "report_markdown": report_md,
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

  all_jobs: list[TpuJobRecord] = []
  for wf_name in args.workflows:
    wf_id = resolve_workflow_id(args.repo, wf_name, required=False)
    if wf_id is not None:
      for run in fetch_workflow_runs(
          args.repo, wf_id, args.max_runs, since=fetch_since, now=now
      ):
        all_jobs.extend(run.tpu_jobs)

  window_jobs = [
      j
      for j in all_jobs
      if j.created_at >= window_start or j.status != "completed"
  ]
  payload = evaluate_queue_sla(
      window_jobs,
      all_jobs,
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
