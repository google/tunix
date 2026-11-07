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
"""Monitors Tunix GitHub Actions TPU CI queue wait times and job durations."""

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

TPU_RUNNER_LABEL = "linux-x86-ct6e-180-8tpu"
TPU_JOB_SUFFIXES = ("run_core", "run_dev", "run_prod")
DEFAULT_WORKFLOWS = ("Tunix Package Tests", "Tunix Nightly Regression Tests")
DEFAULT_WINDOW_HOURS = 24.0
DEFAULT_RETENTION_DAYS = 35
DEFAULT_SLA_P90_WAIT_SECS = 600.0
DEFAULT_SLA_MAX_WAIT_SECS = 1200.0
DEFAULT_RUN_CORE_DURATION_MINS = 9.5
DEFAULT_RUN_DEV_DURATION_MINS = 13.5


@dataclasses.dataclass(frozen=True)
class TpuJobRecord:
  """Telemetry record for a single TPU CI job execution or queued job."""

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

  def to_dict(self) -> dict[str, Any]:
    return {
        "job_id": self.job_id,
        "run_id": self.run_id,
        "name": self.name,
        "status": self.status,
        "conclusion": self.conclusion,
        "created_at": self.created_at.isoformat(),
        "started_at": self.started_at.isoformat(),
        "completed_at": self.completed_at.isoformat(),
        "queue_wait_secs": round(self.queue_wait_secs, 2),
        "duration_mins": round(self.duration_mins, 2),
    }

  @classmethod
  def from_dict(cls, d: Mapping[str, Any]) -> TpuJobRecord:
    c_at = parse_iso(d.get("created_at"))
    if c_at is None:
      raise ValueError(f"Missing created_at in {d}")
    s_at = parse_iso(d.get("started_at")) or c_at
    e_at = parse_iso(d.get("completed_at")) or s_at
    return cls(
        job_id=int(d.get("job_id") or 0),
        run_id=int(d.get("run_id") or 0),
        name=str(d.get("name") or ""),
        status=str(d.get("status") or "completed"),
        conclusion=d.get("conclusion"),
        created_at=c_at,
        started_at=s_at,
        completed_at=e_at,
        queue_wait_secs=float(d.get("queue_wait_secs") or 0.0),
        duration_mins=float(d.get("duration_mins") or 0.0),
    )


@dataclasses.dataclass(frozen=True)
class WorkflowRunRecord:
  """Telemetry record for a single GitHub Actions workflow run."""

  run_id: int
  event: str
  conclusion: str | None
  created_at: datetime.datetime
  tpu_jobs: tuple[TpuJobRecord, ...]


_GH_CLI_AVAILABLE = True


def gh_api(endpoint: str) -> Any:
  """Executes `gh api <endpoint>` and returns parsed JSON."""
  global _GH_CLI_AVAILABLE
  if _GH_CLI_AVAILABLE:
    try:
      out = subprocess.check_output(
          ["gh", "api", endpoint],
          text=True,
          stderr=subprocess.PIPE,
          timeout=10,
      )
      return json.loads(out)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
      _GH_CLI_AVAILABLE = False
  url = f"https://api.github.com/{endpoint.lstrip('/')}"
  req = urllib.request.Request(
      url, headers={"Accept": "application/vnd.github+json"}
  )
  with urllib.request.urlopen(req, timeout=30) as resp:
    return json.loads(resp.read().decode("utf-8"))


def parse_iso(ts: str | None) -> datetime.datetime | None:
  """Parses an ISO-8601 timestamp string into a UTC datetime."""
  if not ts:
    return None
  dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
  return (
      dt.replace(tzinfo=datetime.timezone.utc)
      if dt.tzinfo is None
      else dt.astimezone(datetime.timezone.utc)
  )


def percentile(data: Sequence[float], p: float) -> float:
  """Computes the p-th percentile (0..100) using linear interpolation."""
  if not data:
    return 0.0
  s = sorted(data)
  k = (len(s) - 1) * (p / 100.0)
  f, c = math.floor(k), math.ceil(k)
  return s[int(k)] if f == c else s[f] * (c - k) + s[c] * (k - f)


def resolve_workflow_id(
    repo: str, workflow_name: str, *, required: bool = True
) -> int | None:
  """Finds the GitHub Actions workflow ID matching `workflow_name`."""
  workflows = gh_api(f"repos/{repo}/actions/workflows").get("workflows", [])
  wf_id = next(
      (
          w["id"]
          for w in workflows
          if workflow_name.lower() in w["name"].lower()
      ),
      None,
  )
  if wf_id is None and required:
    sys.exit(f"ERROR: Could not find workflow '{workflow_name}' in {repo}.")
  return int(wf_id) if wf_id is not None else None


def parse_tpu_job_record(
    j: Mapping[str, Any],
    *,
    run_id: int = 0,
    now: datetime.datetime | None = None,
) -> TpuJobRecord | None:
  """Parses a GitHub Actions job dict into a `TpuJobRecord` (completed, running, or queued)."""
  now = now or datetime.datetime.now(datetime.timezone.utc)
  name = str(j.get("name") or "")
  labels = j.get("labels") or []
  is_tpu = TPU_RUNNER_LABEL in labels or (
      "tpu" in name.lower() and any(s in name for s in TPU_JOB_SUFFIXES)
  )
  if not is_tpu or j.get("conclusion") == "skipped":
    return None

  c_at = parse_iso(j.get("created_at"))
  if c_at is None:
    return None
  s_at = parse_iso(j.get("started_at"))
  e_at = parse_iso(j.get("completed_at"))
  status = str(j.get("status") or "completed").lower()
  job_id = int(j.get("id") or 0)

  if s_at and e_at and e_at >= s_at:
    return TpuJobRecord(
        name=name,
        conclusion=j.get("conclusion"),
        created_at=c_at,
        started_at=s_at,
        completed_at=e_at,
        queue_wait_secs=max(0.0, (s_at - c_at).total_seconds()),
        duration_mins=max(0.0, (e_at - s_at).total_seconds() / 60.0),
        job_id=job_id,
        run_id=run_id,
        status="completed",
    )
  if status == "in_progress" or (s_at is not None and e_at is None):
    eff_s = s_at or c_at
    return TpuJobRecord(
        name=name,
        conclusion=None,
        created_at=c_at,
        started_at=eff_s,
        completed_at=now,
        queue_wait_secs=max(0.0, (eff_s - c_at).total_seconds()),
        duration_mins=max(0.0, (now - eff_s).total_seconds() / 60.0),
        job_id=job_id,
        run_id=run_id,
        status="in_progress",
    )
  if status in ("queued", "waiting", "pending", "requested") or (
      s_at is None and e_at is None and j.get("conclusion") is None
  ):
    return TpuJobRecord(
        name=name,
        conclusion=None,
        created_at=c_at,
        started_at=now,
        completed_at=now,
        queue_wait_secs=max(0.0, (now - c_at).total_seconds()),
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
  """Fetches recent workflow runs and their TPU child jobs."""
  now = now or datetime.datetime.now(datetime.timezone.utc)
  runs: list[dict[str, Any]] = []
  page = 1
  while len(runs) < num_runs:
    per_page = min(100, num_runs - len(runs))
    batch = gh_api(
        f"repos/{repo}/actions/workflows/{workflow_id}/runs?per_page={per_page}&page={page}"
    ).get("workflow_runs", [])
    if not batch:
      break
    stop_early = False
    for r in batch:
      c_at = parse_iso(r.get("created_at"))
      if (
          since
          and c_at
          and c_at < since
          and r.get("status") == "completed"
      ):
        stop_early = True
        continue
      runs.append(r)
    if stop_early or len(batch) < per_page:
      break
    page += 1

  records: list[WorkflowRunRecord] = []
  for r in runs[:num_runs]:
    rid = int(r["id"])
    created_at = parse_iso(r.get("created_at"))
    if created_at is None:
      continue
    jobs = gh_api(f"repos/{repo}/actions/runs/{rid}/jobs?per_page=100").get(
        "jobs", []
    )
    tpu_jobs = [
        rec
        for j in jobs
        if (rec := parse_tpu_job_record(j, run_id=rid, now=now)) is not None
    ]
    records.append(
        WorkflowRunRecord(
            run_id=rid,
            event=r.get("event", ""),
            conclusion=r.get("conclusion"),
            created_at=created_at,
            tpu_jobs=tuple(tpu_jobs),
        )
    )
  return records


def update_history_jsonl(
    history_path: str,
    new_jobs: Sequence[TpuJobRecord],
    *,
    now: datetime.datetime | None = None,
    retention_days: int = DEFAULT_RETENTION_DAYS,
) -> list[TpuJobRecord]:
  """Loads, merges, deduplicates, prunes (>retention_days), and saves JSONL history."""
  now = now or datetime.datetime.now(datetime.timezone.utc)
  cutoff = now - datetime.timedelta(days=retention_days)
  existing: list[TpuJobRecord] = []
  if history_path and os.path.isfile(history_path):
    with open(history_path, "r", encoding="utf-8") as f:
      for line in f:
        if line.strip():
          try:
            existing.append(TpuJobRecord.from_dict(json.loads(line)))
          except (json.JSONDecodeError, ValueError, TypeError):
            continue

  rank = {"queued": 0, "in_progress": 1, "completed": 2}
  merged: dict[tuple[Any, ...], TpuJobRecord] = {}
  for j in existing + list(new_jobs):
    if j.created_at < cutoff:
      continue
    key = (
        ("id", j.job_id)
        if j.job_id > 0
        else ("run", j.run_id, j.name, j.created_at.isoformat())
    )
    prev = merged.get(key)
    if prev is None or (
        rank.get(j.status, 1),
        j.queue_wait_secs,
        j.duration_mins,
    ) >= (
        rank.get(prev.status, 1),
        prev.queue_wait_secs,
        prev.duration_mins,
    ):
      merged[key] = j

  result = sorted(merged.values(), key=lambda j: (j.created_at, j.job_id))
  if history_path:
    parent = os.path.dirname(os.path.abspath(history_path))
    if parent:
      os.makedirs(parent, exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
      for j in result:
        f.write(json.dumps(j.to_dict(), sort_keys=True) + "\n")
  return result


def evaluate_queue_sla(
    window_jobs: Sequence[TpuJobRecord],
    history_jobs: Sequence[TpuJobRecord],
    *,
    window_hours: float = DEFAULT_WINDOW_HOURS,
    sla_p90_secs: float = DEFAULT_SLA_P90_WAIT_SECS,
    sla_max_secs: float = DEFAULT_SLA_MAX_WAIT_SECS,
) -> dict[str, Any]:
  """Evaluates queue wait SLA and builds a Markdown report with 30-day daily trends."""
  waits = [j.queue_wait_secs for j in window_jobs]
  p50_w = percentile(waits, 50)
  p90_w = percentile(waits, 90)
  max_w = max(waits, default=0.0)
  queued_n = sum(1 for j in window_jobs if j.status == "queued")
  running_n = sum(1 for j in window_jobs if j.status == "in_progress")

  def _tier_p50(tier: str) -> float:
    return percentile(
        [
            j.duration_mins
            for j in window_jobs
            if tier in j.name and j.conclusion in ("success", "failure")
        ],
        50,
    )

  reasons: list[str] = []
  if p90_w > sla_p90_secs:
    reasons.append(
        f"24h queue wait p90 (`{p90_w:.1f}s`) exceeded SLA (`{sla_p90_secs:.0f}s`)."
    )
  if max_w > sla_max_secs:
    reasons.append(
        f"Single-job max queue wait (`{max_w:.1f}s`) exceeded threshold (`{sla_max_secs:.0f}s`)."
    )

  by_day: dict[str, list[TpuJobRecord]] = {}
  for j in history_jobs or window_jobs:
    day = j.created_at.astimezone(datetime.timezone.utc).strftime("%Y-%m-%d")
    by_day.setdefault(day, []).append(j)

  status_text = "ALERT: SLA BREACH" if reasons else "HEALTHY"
  md = [
      "## Tunix TPU CI Queue & Runtime Report",
      f"- **Status**: **{status_text}** (`{len(window_jobs)}` TPU jobs in last `{window_hours:.0f}h`: `{queued_n}` queued, `{running_n}` in_progress)",
  ]
  for r in reasons:
    md.append(f"- **Alert**: {r}")
  md.extend([
      "",
      "| Metric | Observed (Last 24h) | SLA Target |",
      "| :--- | :--- | :--- |",
      f"| **Queue Wait (`started_at - created_at`)** | **p50 = `{p50_w:.1f}s`**, **p90 = `{p90_w:.1f}s`**, **Max = `{max_w:.1f}s`** | `p90 <= {sla_p90_secs:.0f}s`, `Max <= {sla_max_secs:.0f}s` |",
      f"| **Median Job Duration (`p50`)** | `run_core` = `{_tier_p50('run_core'):.1f}m`, `run_dev` = `{_tier_p50('run_dev'):.1f}m`, `run_prod` = `{_tier_p50('run_prod'):.1f}m` | Informational |",
      "",
      "### Daily Historical Queue Trend (Last 30+ Days)",
      "",
      "| Date (UTC) | Jobs | Queue p50 (`s`) | Queue p90 (`s`) | Queue Max (`s`) | `run_core` p50 (`m`) | `run_dev` p50 (`m`) |",
      "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
  ])
  for day in sorted(by_day.keys())[-35:]:
    dj = by_day[day]
    dw = [x.queue_wait_secs for x in dj]
    c_p50 = percentile(
        [x.duration_mins for x in dj if "run_core" in x.name and x.conclusion],
        50,
    )
    d_p50 = percentile(
        [x.duration_mins for x in dj if "run_dev" in x.name and x.conclusion],
        50,
    )
    md.append(
        f"| `{day}` | `{len(dj)}` | `{percentile(dw, 50):.1f}s` | `{percentile(dw, 90):.1f}s` | `{max(dw, default=0.0):.1f}s` | `{c_p50:.1f}m` | `{d_p50:.1f}m` |"
    )

  return {
      "breached": bool(reasons),
      "reasons": reasons,
      "metrics": {
          "total_tpu_jobs": len(window_jobs),
          "queued_jobs_count": queued_n,
          "in_progress_jobs_count": running_n,
          "p50_wait_secs": round(p50_w, 1),
          "p90_wait_secs": round(p90_w, 1),
          "max_wait_secs": round(max_w, 1),
      },
      "report_markdown": "\n".join(md),
  }


def main(argv: Sequence[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--repo", default="google/tunix")
  parser.add_argument("--workflows", nargs="+", default=list(DEFAULT_WORKFLOWS))
  parser.add_argument("--window-hours", type=float, default=DEFAULT_WINDOW_HOURS)
  parser.add_argument("--max-runs", type=int, default=80)
  parser.add_argument("--history-file", default="")
  parser.add_argument("--retention-days", type=int, default=DEFAULT_RETENTION_DAYS)
  parser.add_argument("--sla-p90-seconds", type=float, default=DEFAULT_SLA_P90_WAIT_SECS)
  parser.add_argument("--sla-max-seconds", type=float, default=DEFAULT_SLA_MAX_WAIT_SECS)
  parser.add_argument("--markdown-output", default="")
  parser.add_argument("--json-output", default="")
  args = parser.parse_args(argv)

  now = datetime.datetime.now(datetime.timezone.utc)
  window_start = now - datetime.timedelta(hours=args.window_hours)
  has_history = bool(args.history_file and os.path.isfile(args.history_file))
  fetch_since = window_start if has_history else now - datetime.timedelta(days=7)

  runs: list[WorkflowRunRecord] = []
  for wf_name in args.workflows:
    wf_id = resolve_workflow_id(args.repo, wf_name, required=False)
    if wf_id is not None:
      runs.extend(
          fetch_workflow_runs(
              args.repo, wf_id, args.max_runs, since=fetch_since, now=now
          )
      )

  all_fetched_jobs = [j for r in runs for j in r.tpu_jobs]
  history_jobs = update_history_jsonl(
      args.history_file,
      all_fetched_jobs,
      now=now,
      retention_days=args.retention_days,
  )
  window_jobs = [
      j
      for j in all_fetched_jobs
      if j.created_at >= window_start or j.status != "completed"
  ]
  payload = evaluate_queue_sla(
      window_jobs,
      history_jobs,
      window_hours=args.window_hours,
      sla_p90_secs=args.sla_p90_seconds,
      sla_max_secs=args.sla_max_seconds,
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
