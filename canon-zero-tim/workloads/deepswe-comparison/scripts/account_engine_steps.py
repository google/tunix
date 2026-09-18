#!/usr/bin/env python3
"""Account a DeepSWE run from its [ENGINE_STEP] / [PERF] receipts (perf receipt v1 + v2).

usage: account_engine_steps.py <raw.log or engine_step_lines.log> [...] [--edges 32,256,1024,4096]

Prints Markdown: overview, time share by scheduled-token bucket and path, the synced
samples (device time vs dispatch vs sample wall, v2 only), per-program launch walls
(v2 only) and [PERF] stage totals.  Stdlib only; tolerant of v1 lines (fields absent).
Reading rule (bd07 ANALYSIS.md section 3): exec_ms is host dispatch, sample_ms absorbs
device execution and the tail launches; only dev_model_ms (synced rows) is device time.
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys

_TOK = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^ ]+)")


def parse_engine_line(line: str):
  if "[ENGINE_STEP] n=" not in line:
    return None
  row = {}
  for key, value in _TOK.findall(line[line.index("[ENGINE_STEP]"):]):
    if key == "launch_ms":
      launches = {}
      if value != "-":
        for part in value.split("/"):
          name, _, ms = part.partition(":")
          try:
            launches[name] = launches.get(name, 0.0) + float(ms)
          except ValueError:
            pass
      row["launch"] = launches
      continue
    try:
      row[key] = float(value) if "." in value else int(value)
    except ValueError:
      row[key] = value
  return row if "n" in row else None


def parse_perf_line(line: str):
  match = re.search(r"\[PERF\] (?:step=(\d+) )?stage=([A-Za-z0-9_]+) seconds=([0-9.]+)", line)
  if not match:
    return None
  return {"step": int(match.group(1)) if match.group(1) else None,
          "stage": match.group(2), "seconds": float(match.group(3))}


def p50(values):
  return statistics.median(values) if values else 0.0


def bucket(tokens, edges):
  low = 1
  for edge in edges:
    if tokens <= edge:
      return f"{low}-{edge}"
    low = edge + 1
  return f">{edges[-1]}"


def account(rows, perf, edges):
  out = []
  wall = lambda r: r.get("gap_ms", 0) + r.get("exec_ms", 0) + r.get("sample_ms", 0) + r.get("sync_ms", 0)
  paths = {}
  for r in rows:
    paths[r.get("path", "?")] = paths.get(r.get("path", "?"), 0) + 1
  out.append("## 1. Overview")
  out.append(f"- receipts: {len(rows)}; paths: " + ", ".join(f"{k}={v}" for k, v in sorted(paths.items())))
  if rows:
    out.append(f"- wall/step p50: {p50([wall(r) for r in rows]):.1f} ms = gap {p50([r.get('gap_ms', 0) for r in rows]):.1f} + exec {p50([r.get('exec_ms', 0) for r in rows]):.1f} + sample {p50([r.get('sample_ms', 0) for r in rows]):.1f} + sync {p50([r.get('sync_ms', 0) for r in rows]):.1f}")
    fills = [r["fill"] for r in rows if "fill" in r]
    kv = [r["kv_tok"] for r in rows if "kv_tok" in r]
    if fills:
      out.append(f"- fill (sched_tok/pad_tok) p50: {p50(fills):.3f}; kv_tok p50: {p50(kv):.0f} (v2 rows: {len(fills)})")
    else:
      out.append("- fill / kv_tok: not present (v1 receipts)")
  out.append("")
  out.append("## 2. Time share by scheduled-token bucket and path (exec_ms + sample_ms)")
  out.append("| path | sched_tok | n | reqs p50 | prefill_reqs p50 | exec p50 | sample p50 | share |")
  out.append("|---|---|---:|---:|---:|---:|---:|---:|")
  groups = {}
  for r in rows:
    key = (r.get("path", "?"), bucket(int(r.get("sched_tok", 0)), edges))
    groups.setdefault(key, []).append(r)
  total = sum(r.get("exec_ms", 0) + r.get("sample_ms", 0) for r in rows) or 1.0
  for key, group in sorted(groups.items(), key=lambda kv: -sum(r.get("exec_ms", 0) + r.get("sample_ms", 0) for r in kv[1])):
    share = sum(r.get("exec_ms", 0) + r.get("sample_ms", 0) for r in group) / total * 100
    out.append(f"| {key[0]} | {key[1]} | {len(group)} | {p50([r.get('reqs', 0) for r in group]):.0f} | {p50([r.get('prefill_reqs', 0) for r in group]):.0f} | {p50([r.get('exec_ms', 0) for r in group]):.1f} | {p50([r.get('sample_ms', 0) for r in group]):.1f} | {share:.1f}% |")
  out.append("")
  synced = [r for r in rows if r.get("synced") == 1]
  out.append("## 3. Synced samples: device time vs host (v2, CANON_ENGINE_STEP_LOG_SYNC_EVERY)")
  if synced:
    out.append("| sched_tok | n | dev_model_ms p50 | exec (dispatch) p50 | sample p50 | gap p50 |")
    out.append("|---|---:|---:|---:|---:|---:|")
    by = {}
    for r in synced:
      by.setdefault(bucket(int(r.get("sched_tok", 0)), edges), []).append(r)
    for key, group in sorted(by.items()):
      out.append(f"| {key} | {len(group)} | {p50([r.get('dev_model_ms', 0) for r in group]):.1f} | {p50([r.get('exec_ms', 0) for r in group]):.1f} | {p50([r.get('sample_ms', 0) for r in group]):.1f} | {p50([r.get('gap_ms', 0) for r in group]):.1f} |")
    out.append(f"- all synced rows: dev_model_ms p50 {p50([r.get('dev_model_ms', 0) for r in synced]):.1f} ms of wall p50 {p50([wall(r) for r in synced]):.1f} ms")
  else:
    out.append("- no synced rows (v1 receipts or CANON_ENGINE_STEP_LOG_SYNC_EVERY=0)")
  out.append("")
  out.append("## 4. Per-program launch wall (dispatch only, v2)")
  launched = [r for r in rows if r.get("launch")]
  if launched:
    names = {}
    for r in launched:
      for name, ms in r["launch"].items():
        names.setdefault(name, []).append(ms)
    out.append("| program | rows | p50 ms | p90 ms |")
    out.append("|---|---:|---:|---:|")
    for name, values in sorted(names.items(), key=lambda kv: -p50(kv[1])):
      ordered = sorted(values)
      out.append(f"| {name} | {len(values)} | {p50(values):.2f} | {ordered[min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))]:.2f} |")
    out.append(f"- sum of launches p50 {p50([sum(r['launch'].values()) for r in launched]):.2f} ms vs exec_ms p50 {p50([r.get('exec_ms', 0) for r in launched]):.2f} ms")
  else:
    out.append("- no launch_ms fields (v1 receipts)")
  out.append("")
  out.append("## 5. [PERF] stage totals")
  if perf:
    stages = {}
    for p in perf:
      stages.setdefault(p["stage"], []).append(p["seconds"])
    out.append("| stage | calls | total s | p50 s |")
    out.append("|---|---:|---:|---:|")
    for stage, values in sorted(stages.items(), key=lambda kv: -sum(kv[1])):
      out.append(f"| {stage} | {len(values)} | {sum(values):.1f} | {p50(values):.2f} |")
  else:
    out.append("- no [PERF] stage lines")
  return "\n".join(out)


def main(argv=None):
  parser = argparse.ArgumentParser()
  parser.add_argument("logs", nargs="+")
  parser.add_argument("--edges", default="32,256,1024,4096")
  args = parser.parse_args(argv)
  edges = [int(v) for v in args.edges.split(",")]
  rows, perf = [], []
  for path in args.logs:
    with open(path, "rb") as handle:
      for raw in handle:
        line = raw.decode("utf-8", "replace")
        row = parse_engine_line(line)
        if row:
          rows.append(row)
          continue
        stage = parse_perf_line(line)
        if stage:
          perf.append(stage)
  print(account(rows, perf, edges))
  return 0


if __name__ == "__main__":
  sys.exit(main())
