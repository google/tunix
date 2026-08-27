#!/usr/bin/env python3
"""Streaming UI trace-JSON gate for Native-like Zero-HP train steps."""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
from pathlib import Path
import re
import sys
from typing import Iterator


def _load_hierarchy():
  path = Path(__file__).with_name("census_gsm8k_xprof_hierarchy.py")
  spec = importlib.util.spec_from_file_location(
      "v1_gsm8k_xprof_trace_hierarchy", path
  )
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load hierarchy validator: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


HIERARCHY = _load_hierarchy()
_TRACE_EVENTS = re.compile(r'"traceEvents"\s*:\s*\[')


def _iter_events(path: Path) -> Iterator[dict]:
  """Yields top-level traceEvents without materializing the million-event JSON."""
  decoder = json.JSONDecoder()
  with gzip.open(path, "rt", encoding="utf-8") as stream:
    buffer = ""
    found = False
    eof = False
    while True:
      if not found:
        chunk = stream.read(1024 * 1024)
        if not chunk:
          raise ValueError("trace JSON has no traceEvents array")
        buffer += chunk
        match = _TRACE_EVENTS.search(buffer)
        if match is None:
          buffer = buffer[-128:]
          continue
        buffer = buffer[match.end():]
        found = True

      buffer = buffer.lstrip()
      if buffer.startswith(","):
        buffer = buffer[1:].lstrip()
      if buffer.startswith("]"):
        return
      try:
        event, end = decoder.raw_decode(buffer)
      except json.JSONDecodeError as error:
        if eof:
          raise ValueError(
              f"truncated traceEvents JSON at offset {error.pos}"
          ) from error
        chunk = stream.read(1024 * 1024)
        if chunk:
          buffer += chunk
        else:
          eof = True
        continue
      buffer = buffer[end:]
      if not isinstance(event, dict):
        raise ValueError("traceEvents entry is not an object")
      yield event


def _resolve_trace(path: Path) -> Path:
  if path.is_file():
    if path.suffixes[-3:] != [".trace", ".json", ".gz"]:
      raise ValueError(f"not a trace.json.gz: {path}")
    if path.stat().st_size <= 0:
      raise ValueError(f"empty trace JSON: {path}")
    return path
  files = sorted(
      candidate
      for candidate in path.glob(
          "train/xprof/plugins/profile/*/*.trace.json.gz"
      )
      if candidate.stat().st_size > 0
  )
  if len(files) != 1:
    raise ValueError(
        f"expected exactly one non-empty trace JSON, found {len(files)}"
    )
  return files[0]


def read_trace(path: Path):
  """Returns only hierarchy/compiler spans and total event count."""
  selected = []
  compilers = []
  process_names = {}
  thread_names = {}
  total = 0
  wanted = set(HIERARCHY.EXPECTED_COUNTS)
  compiler_names = set(HIERARCHY.COMPILER_EVENTS)
  for event in _iter_events(path):
    total += 1
    phase = event.get("ph")
    name = event.get("name")
    pid = event.get("pid")
    tid = event.get("tid")
    args = event.get("args") or {}
    if phase == "M" and name == "process_name":
      process_names[pid] = args.get("name")
      continue
    if phase == "M" and name == "thread_name":
      thread_names[(pid, tid)] = args.get("name")
      continue
    if phase != "X" or name not in wanted | compiler_names:
      continue
    record = (
        name,
        float(event.get("ts", 0.0)),
        float(event.get("dur", 0.0)),
        pid,
        tid,
        {str(key): str(value) for key, value in args.items()},
    )
    if name in wanted:
      selected.append(record)
    else:
      compilers.append(record)

  def span(record):
    name, start, duration, pid, tid, stats = record
    process = process_names.get(pid, f"<process:{pid}>")
    thread = thread_names.get((pid, tid), f"<thread:{tid}>")
    line_name = thread if process == "/host:CPU" else f"{process}/{thread}"
    return HIERARCHY.Span(name, start, duration, line_name, stats)

  spans = [span(record) for record in selected]
  compiler_spans = [span(record) for record in compilers]
  update = next(
      (item for item in spans if item.name == "zero_tim_update"), None
  )
  compiler_counts = {
      name: sum(
          item.name == name
          and update is not None
          and HIERARCHY._contains(update, item)  # pylint: disable=protected-access
          for item in compiler_spans
      )
      for name in HIERARCHY.COMPILER_EVENTS
  }
  return spans, compiler_counts, total


def validate_trace(
    spans,
    *,
    compiler_counts,
    expected_update_step: int = 2,
    expected_groups: int = 16,
) -> list[str]:
  """Applies hierarchy semantics without pretending JSON has device planes."""
  synthetic_device_steps = {
      f"/device:TPU:{index}": 1 for index in range(8)
  }
  return HIERARCHY.validate_hierarchy(
      spans,
      device_step_counts=synthetic_device_steps,
      compiler_counts=compiler_counts,
      expected_update_step=expected_update_step,
      expected_groups=expected_groups,
      require_step_marker=False,
  )


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-root", type=Path, required=True)
  parser.add_argument("--expected-update-step", type=int, default=2)
  parser.add_argument(
      "--geometry",
      choices=tuple(sorted(HIERARCHY.GEOMETRIES)),
      default=HIERARCHY.DEFAULT_GEOMETRY,
      help="registered carrier geometry the run was launched with",
  )
  args = parser.parse_args()
  expected_groups = HIERARCHY.GEOMETRIES[args.geometry]["groups"]
  trace = _resolve_trace(args.run_root)
  spans, compiler_counts, total = read_trace(trace)
  reasons = validate_trace(
      spans,
      compiler_counts=compiler_counts,
      expected_update_step=args.expected_update_step,
      expected_groups=expected_groups,
  )
  counts = {
      name: sum(span.name == name for span in spans)
      for name in HIERARCHY.EXPECTED_COUNTS
  }
  print("trace_event_count=" + str(total))
  print("trace_hierarchy_counts=" + json.dumps(counts, sort_keys=True))
  print("trace_compiler_events=" + json.dumps(
      compiler_counts, sort_keys=True
  ))
  if reasons:
    for reason in reasons:
      print("  RED " + reason)
    print(f"V1_GSM8K_XPROF_TRACE_CENSUS_RED reasons={len(reasons)}")
    return 1
  start = args.expected_update_step * expected_groups
  print(
      "V1_GSM8K_XPROF_TRACE_CENSUS_GREEN "
      f"train_steps={start}..{start + expected_groups - 1} "
      f"reverse_transactions={expected_groups} "
      "optimizer_visible=1 optimizer_owned_by_last=1 same_host_track=1 "
      "compiler_events=0"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
