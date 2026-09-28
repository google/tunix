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
  1. `tpu-ci-monitor-error`: Opened/updated when `scripts/tpu_ci_monitor.py`
     fails (`--monitor-outcome != "success"`) or produces no valid JSON payload,
     and auto-closed on the next successful monitor run.
  2. `tpu-ci-alert`: Evaluated only when the monitor script succeeds; opened or
     updated when `payload["breached"]` is `True`, and auto-closed when `False`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
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


def run_gh(args: Sequence[str]) -> str:
  """Executes a `gh` CLI command and returns its stdout."""
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
  """Formats the Markdown footer (with `@`-mentions only on initial creation)."""
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
) -> dict[str, Any]:
  """Synchronizes at most one open GitHub issue for `tracking_label` via `gh`."""
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
      run_fn(
          ["issue", "edit", str(existing_num), "--repo", repo, "--title", title]
      )
      run_fn([
          "issue",
          "comment",
          str(existing_num),
          "--repo",
          repo,
          "--body",
          followup_body,
      ])
      return {"action": "updated", "issue_number": existing_num}

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
    return {
        "action": "created",
        "issue_number": int(tail) if tail.isdigit() else None,
    }

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
    return {"action": "closed", "issue_number": existing_num}

  return {"action": "noop", "issue_number": None}


def _load_alert_payload(alert_json_path: str) -> dict[str, Any] | None:
  """Loads and validates `tpu_ci_alert.json` if present."""
  if not alert_json_path or not os.path.isfile(alert_json_path):
    return None
  try:
    with open(alert_json_path, "r", encoding="utf-8") as f:
      data = json.load(f)
    return data if isinstance(data, dict) and "breached" in data else None
  except (OSError, json.JSONDecodeError, ValueError):
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
) -> dict[str, Any]:
  """Synchronizes both monitor-execution and queue-SLA P1 GitHub alert issues."""
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
    return {
        "monitor_error_sync": monitor_sync,
        "sla_alert_sync": {
            "action": "skipped_due_to_monitor_error",
            "issue_number": None,
        },
    }

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

  metrics = payload.get("metrics") or {}
  p90_wait = metrics.get("p90_wait_secs", 0.0)
  max_wait = metrics.get(
      "recent_max_wait_secs", metrics.get("max_wait_secs", 0.0)
  )
  sla_sync = sync_issue_by_label(
      repo=repo,
      tracking_label=SLA_BREACH_LABEL,
      is_active=bool(payload.get("breached")),
      title=(
          "[TPU CI Alert] Queue wait SLA breach"
          f" (p90={p90_wait}s, max={max_wait}s)"
      ),
      summary_markdown=str(payload.get("report_markdown") or "").strip(),
      recovery_heading="TPU CI Queue SLA Recovered",
      run_id=run_id,
      run_url=run_url,
      assignee=assignee,
      cc_users=cc_users,
      run_fn=run_fn,
  )
  return {"monitor_error_sync": monitor_sync, "sla_alert_sync": sla_sync}


def main(argv: Sequence[str] | None = None) -> None:
  """CLI entrypoint for synchronizing P1 GitHub Alert Issues."""
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
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
