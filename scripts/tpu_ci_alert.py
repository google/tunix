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
"""Synchronizes P1 GitHub Alert Issues for Tunix TPU CI using the `gh` CLI.

Manages two independent P1 GitHub Issue lifecycles:
  1. `tpu-ci-monitor-error`: Opened or updated in-place (title and body) when
     `scripts/tpu_ci_monitor.py` fails (`--monitor-outcome != "success"`) or
     produces no valid JSON payload, and auto-closed with a recovery comment on
     the next successful monitor run.
  2. `tpu-ci-alert`: Evaluated only when the monitor script succeeds; opened or
     updated in-place (title and body) when `payload.breached` is `True`, and
     auto-closed with a recovery comment when `False`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import json
import os
import subprocess
from typing import Any

SLA_BREACH_LABEL = "tpu-ci-alert"
MONITOR_ERROR_LABEL = "tpu-ci-monitor-error"

REQUIRED_LABELS: tuple[tuple[str, str], ...] = (
    (SLA_BREACH_LABEL, "d93025"),
    (MONITOR_ERROR_LABEL, "e37400"),
    ("bug", "d73a4a"),
    ("p1", "b60205"),
)

ALERT_ASSIGNEE = "dingying-ai"
ALERT_CC_USERS: tuple[str, ...] = (
    "fineguy",
    "shz-google",
    "s-noghabi",
    "tianshub",
)

GhRunner = Callable[[Sequence[str]], str]


@dataclasses.dataclass(frozen=True)
class AlertMetrics:
  """Queue wait metrics extracted from `tpu_ci_alert.json`.

  Attributes:
    p90_wait_secs: 24-hour 90th percentile queue wait time in seconds.
    max_wait_secs: 24-hour maximum single-job queue wait time in seconds.
    recent_max_wait_secs: Recent window / active maximum queue wait time in
      seconds.
  """

  p90_wait_secs: float = 0.0
  max_wait_secs: float = 0.0
  recent_max_wait_secs: float = 0.0

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> AlertMetrics:
    """Constructs `AlertMetrics` from a raw JSON metrics mapping.

    Args:
      data: Mapping of raw metric fields from `tpu_ci_alert.json`.

    Returns:
      A validated `AlertMetrics` instance.
    """
    max_wait = float(data.get("max_wait_secs", 0.0))
    return cls(
        p90_wait_secs=float(data.get("p90_wait_secs", 0.0)),
        max_wait_secs=max_wait,
        recent_max_wait_secs=float(data.get("recent_max_wait_secs", max_wait)),
    )


@dataclasses.dataclass(frozen=True)
class AlertPayload:
  """Validated SLA evaluation payload loaded from `tpu_ci_alert.json`.

  Attributes:
    breached: Whether the TPU CI queue wait SLA was breached.
    metrics: Validated `AlertMetrics` snapshot.
    report_markdown: Rendered Markdown summary table.
  """

  breached: bool
  metrics: AlertMetrics
  report_markdown: str

  @classmethod
  def from_dict(cls, data: Mapping[str, Any]) -> AlertPayload:
    """Constructs `AlertPayload` from a raw JSON dictionary.

    Args:
      data: Parsed JSON dictionary from `tpu_ci_alert.json`.

    Returns:
      A validated `AlertPayload` instance.

    Raises:
      ValueError: If the required `breached` field is missing.
    """
    if "breached" not in data:
      raise ValueError("Missing required 'breached' field.")
    raw_metrics = data.get("metrics")
    metrics_map = raw_metrics if isinstance(raw_metrics, Mapping) else {}
    return cls(
        breached=bool(data["breached"]),
        metrics=AlertMetrics.from_dict(metrics_map),
        report_markdown=str(data.get("report_markdown") or ""),
    )


@dataclasses.dataclass(frozen=True)
class IssueSyncResult:
  """Outcome of synchronizing a single tracked GitHub alert issue label.

  Attributes:
    action: Synchronization action taken (`created`, `updated`, `closed`,
      `noop`, or `skipped_due_to_monitor_error`).
    issue_number: GitHub issue number acted upon, or `None` if no issue was
      touched.
  """

  action: str
  issue_number: int | None

  def to_dict(self) -> dict[str, str | int | None]:
    """Serializes the issue sync result to a dictionary."""
    return {"action": self.action, "issue_number": self.issue_number}

  def __getitem__(self, key: str) -> str | int | None:
    """Supports dictionary-style key lookup for backward compatibility."""
    return self.to_dict()[key]


@dataclasses.dataclass(frozen=True)
class AlertSyncReport:
  """Combined synchronization report for monitor-error and SLA-breach issues.

  Attributes:
    monitor_error_sync: Sync outcome for `tpu-ci-monitor-error`.
    sla_alert_sync: Sync outcome for `tpu-ci-alert`.
  """

  monitor_error_sync: IssueSyncResult
  sla_alert_sync: IssueSyncResult

  def to_dict(self) -> dict[str, dict[str, str | int | None]]:
    """Serializes the combined alert sync report to a dictionary."""
    return {
        "monitor_error_sync": self.monitor_error_sync.to_dict(),
        "sla_alert_sync": self.sla_alert_sync.to_dict(),
    }

  def __getitem__(self, key: str) -> IssueSyncResult:
    """Supports dictionary-style key lookup for backward compatibility."""
    if key == "monitor_error_sync":
      return self.monitor_error_sync
    if key == "sla_alert_sync":
      return self.sla_alert_sync
    raise KeyError(key)


def run_gh(args: Sequence[str]) -> str:
  """Executes a `gh` CLI command and returns its stdout.

  Args:
    args: Sequence of CLI arguments passed to `gh`.

  Returns:
    Stripped standard output from the `gh` command.
  """
  return subprocess.check_output(
      ["gh", *args], text=True, stderr=subprocess.PIPE, timeout=30
  ).strip()


def _format_routing_footer(
    *,
    run_id: str,
    run_url: str,
    assignee: str = ALERT_ASSIGNEE,
    cc_users: Sequence[str] = ALERT_CC_USERS,
    include_mentions: bool = True,
) -> str:
  """Formats the Markdown footer (`@`-mentions in issue body, backticked on close)."""
  assignee_fmt = f"@{assignee}" if include_mentions else f"`{assignee}`"
  cc_fmt = (
      " ".join(f"@{u}" for u in cc_users)
      if include_mentions
      else ", ".join(f"`{u}`" for u in cc_users)
  )
  run_link = f"[Run #{run_id}]({run_url})" if run_url else f"Run #{run_id}"
  return "\n".join([
      "---",
      f"- **Workflow Run**: {run_link}",
      f"- **Assignee**: {assignee_fmt}",
      f"- **CC**: {cc_fmt}",
  ])


def sync_issue_by_label(
    *,
    repo: str,
    tracking_label: str,
    is_active: bool,
    title: str,
    summary_markdown: str,
    recovery_heading: str,
    run_id: str,
    run_url: str,
    assignee: str = ALERT_ASSIGNEE,
    cc_users: Sequence[str] = ALERT_CC_USERS,
    run_fn: GhRunner = run_gh,
) -> IssueSyncResult:
  """Synchronizes at most one open GitHub issue for `tracking_label` via `gh`.

  When `is_active` is `True` and an open issue already exists, updates `--title`
  and `--body` in-place via `gh issue edit` (posting zero comments to avoid
  6-hourly notification noise). When `is_active` is `False` and an open issue
  exists, closes it via `gh issue close --reason completed --comment ...`.

  Args:
    repo: GitHub repository in `owner/name` format.
    tracking_label: Issue label identifying the alert category.
    is_active: Whether the alert condition is currently active.
    title: Desired issue title when active.
    summary_markdown: Markdown body summarizing telemetry or recovery state.
    recovery_heading: Heading used in the closing comment upon recovery.
    run_id: GitHub Actions workflow run ID.
    run_url: URL to the GitHub Actions workflow run.
    assignee: GitHub username to assign new alert issues to.
    cc_users: GitHub usernames to CC in the issue routing footer.
    run_fn: Callable executing `gh` CLI subcommands.

  Returns:
    An `IssueSyncResult` describing the action taken and affected issue number.
  """
  create_footer = _format_routing_footer(
      run_id=run_id,
      run_url=run_url,
      assignee=assignee,
      cc_users=cc_users,
      include_mentions=True,
  )
  comment_footer = _format_routing_footer(
      run_id=run_id,
      run_url=run_url,
      assignee=assignee,
      cc_users=cc_users,
      include_mentions=False,
  )
  create_body = (
      f"{summary_markdown}\n\n{create_footer}"
      if summary_markdown
      else create_footer
  )
  followup_body = (
      f"{summary_markdown}\n\n{comment_footer}"
      if summary_markdown
      else comment_footer
  )

  raw_list = run_fn([
      "issue",
      "list",
      "--repo",
      repo,
      "--state",
      "open",
      "--label",
      tracking_label,
      "--limit",
      "1",
      "--json",
      "number",
  ])
  open_issues = json.loads(raw_list) if raw_list else []
  existing_num = int(open_issues[0]["number"]) if open_issues else None

  if is_active:
    if existing_num is not None:
      run_fn([
          "issue",
          "edit",
          str(existing_num),
          "--repo",
          repo,
          "--title",
          title,
          "--body",
          create_body,
      ])
      return IssueSyncResult(action="updated", issue_number=existing_num)

    base_cmd = [
        "issue",
        "create",
        "--repo",
        repo,
        "--title",
        title,
        "--body",
        create_body,
        "--label",
        f"{tracking_label},bug,p1",
    ]
    try:
      out = run_fn([*base_cmd, "--assignee", assignee])
    except Exception:  # pylint: disable=broad-exception-caught
      out = run_fn(base_cmd)
    tail = out.rstrip("/").rsplit("/", 1)[-1] if out else ""
    return IssueSyncResult(
        action="created",
        issue_number=int(tail) if tail.isdigit() else None,
    )

  if existing_num is not None:
    run_fn([
        "issue",
        "close",
        str(existing_num),
        "--repo",
        repo,
        "--reason",
        "completed",
        "--comment",
        f"### {recovery_heading}\n\nAuto-closing alert.\n\n{followup_body}",
    ])
    return IssueSyncResult(action="closed", issue_number=existing_num)

  return IssueSyncResult(action="noop", issue_number=None)


def _load_alert_payload(alert_json_path: str) -> AlertPayload | None:
  """Loads and validates `tpu_ci_alert.json` if present."""
  if not alert_json_path or not os.path.isfile(alert_json_path):
    return None
  try:
    with open(alert_json_path, "r", encoding="utf-8") as f:
      data = json.load(f)
    return AlertPayload.from_dict(data) if isinstance(data, Mapping) else None
  except (OSError, json.JSONDecodeError, ValueError, TypeError):
    return None


def sync_github_alerts(
    *,
    repo: str,
    run_id: str,
    run_url: str,
    monitor_outcome: str,
    alert_json_path: str,
    assignee: str = ALERT_ASSIGNEE,
    cc_users: Sequence[str] = ALERT_CC_USERS,
    run_fn: GhRunner = run_gh,
) -> AlertSyncReport:
  """Synchronizes both monitor-execution and queue-SLA P1 GitHub alert issues.

  Args:
    repo: GitHub repository in `owner/name` format.
    run_id: GitHub Actions workflow run ID.
    run_url: URL to the GitHub Actions workflow run.
    monitor_outcome: Execution outcome of the monitor step (e.g. `success`).
    alert_json_path: Path to the JSON payload emitted by `tpu_ci_monitor.py`.
    assignee: GitHub username to assign new P1 alert issues to.
    cc_users: GitHub usernames to CC in the issue routing footer.
    run_fn: Callable executing `gh` CLI subcommands.

  Returns:
    An `AlertSyncReport` containing sync outcomes for both alert categories.
  """
  for name, color in REQUIRED_LABELS:
    run_fn([
        "label",
        "create",
        name,
        "--repo",
        repo,
        "--color",
        color,
        "--force",
    ])

  payload = _load_alert_payload(alert_json_path)
  if monitor_outcome != "success" or payload is None:
    error_summary = "\n".join([
        "## Tunix TPU CI Monitor Execution Failure",
        "",
        (
            "The TPU CI monitoring script (`scripts/tpu_ci_monitor.py`) failed"
            " or did not produce a valid `tpu_ci_alert.json` payload."
        ),
        f"- **Monitor Outcome**: `{monitor_outcome}`",
        f"- **Valid Alert Payload Found**: `{payload is not None}`",
    ])
    monitor_sync = sync_issue_by_label(
        repo=repo,
        tracking_label=MONITOR_ERROR_LABEL,
        is_active=True,
        title=(
            "[TPU CI Monitor Error] Monitoring workflow execution failed"
            f" (outcome: {monitor_outcome})"
        ),
        summary_markdown=error_summary,
        recovery_heading="TPU CI Monitor Execution Recovered",
        run_id=run_id,
        run_url=run_url,
        assignee=assignee,
        cc_users=cc_users,
        run_fn=run_fn,
    )
    return AlertSyncReport(
        monitor_error_sync=monitor_sync,
        sla_alert_sync=IssueSyncResult(
            action="skipped_due_to_monitor_error",
            issue_number=None,
        ),
    )

  monitor_sync = sync_issue_by_label(
      repo=repo,
      tracking_label=MONITOR_ERROR_LABEL,
      is_active=False,
      title="[TPU CI Monitor Error] Monitoring workflow execution failed",
      summary_markdown="`scripts/tpu_ci_monitor.py` completed successfully.",
      recovery_heading="TPU CI Monitor Execution Recovered",
      run_id=run_id,
      run_url=run_url,
      assignee=assignee,
      cc_users=cc_users,
      run_fn=run_fn,
  )

  p90_wait = payload.metrics.p90_wait_secs
  max_wait = payload.metrics.recent_max_wait_secs
  sla_sync = sync_issue_by_label(
      repo=repo,
      tracking_label=SLA_BREACH_LABEL,
      is_active=payload.breached,
      title=(
          "[TPU CI Alert] Queue wait SLA breach"
          f" (p90={p90_wait}s, max={max_wait}s)"
      ),
      summary_markdown=payload.report_markdown.strip(),
      recovery_heading="TPU CI Queue SLA Recovered",
      run_id=run_id,
      run_url=run_url,
      assignee=assignee,
      cc_users=cc_users,
      run_fn=run_fn,
  )
  return AlertSyncReport(
      monitor_error_sync=monitor_sync, sla_alert_sync=sla_sync
  )


def main(argv: Sequence[str] | None = None) -> None:
  """CLI entrypoint for synchronizing P1 GitHub Alert Issues.

  Args:
    argv: Optional sequence of command-line arguments (defaults to `sys.argv`).
  """
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--repo", required=True)
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--run-url", required=True)
  parser.add_argument("--monitor-outcome", required=True)
  parser.add_argument("--alert-json", default="tpu_ci_alert.json")
  args = parser.parse_args(argv)

  result = sync_github_alerts(
      repo=args.repo,
      run_id=args.run_id,
      run_url=args.run_url,
      monitor_outcome=args.monitor_outcome,
      alert_json_path=args.alert_json,
  )
  print(json.dumps(result.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
