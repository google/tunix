#!/usr/bin/env python3
"""Validates the complete P45 warm R1/R2 transaction in its full XPlane."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
from typing import Mapping, Sequence


SCHEMA = "canon.v2-frozenlake-onehost.xplane-census.v1"
XPROF_VERSION = "2.23.1"
TPU_PLANE = re.compile(r"/device:TPU:\d+")
HOST_LINE = "python3"
GROUPS = 8
COMMON_COUNTS = {
    "train": 1,
    "zero_tim_update": 1,
    "reverse_groups": 1,
    "forward_group": GROUPS,
    "group_loss_pullback": GROUPS,
    "reverse_group": GROUPS,
    "model_backward": GROUPS,
    "report_adjoint": GROUPS,
    "loss_pullback": 1,
    "deferred_finite_receipts": 1,
}
ARM_COUNTS = {
    "r1": {
        "staged_accumulate": 0,
        "fixed_dp_reduce": GROUPS,
        "staged_receipt_fetch": 0,
        "gradient_accumulate": GROUPS,
    },
    "r2": {
        "staged_accumulate": GROUPS,
        "fixed_dp_reduce": 1,
        "staged_receipt_fetch": 1,
        "gradient_accumulate": 1,
    },
}
EVENT_NAMES = frozenset(COMMON_COUNTS) | frozenset(
    name for counts in ARM_COUNTS.values() for name in counts
)
GROUPED_COMMON = (
    "forward_group",
    "group_loss_pullback",
    "reverse_group",
    "model_backward",
    "report_adjoint",
)


@dataclass(frozen=True)
class Span:
  name: str
  start_ns: float
  duration_ns: float
  line_name: str
  stats: Mapping[str, str]

  @property
  def end_ns(self) -> float:
    return self.start_ns + self.duration_ns


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    while chunk := source.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def _contains(parent: Span, child: Span) -> bool:
  return parent.start_ns <= child.start_ns and child.end_ns <= parent.end_ns


def _one(by_name: Mapping[str, list[Span]], name: str) -> Span | None:
  spans = by_name.get(name, [])
  return spans[0] if len(spans) == 1 else None


def _indexed(
    by_name: Mapping[str, list[Span]], name: str, reasons: list[str]
) -> dict[int, Span]:
  result = {}
  for span in by_name.get(name, []):
    raw = span.stats.get("group_index")
    try:
      index = int(raw) if raw is not None else None
    except ValueError:
      index = None
    if index is None:
      reasons.append(f"{name}:missing_or_invalid_group_index")
    elif index in result:
      reasons.append(f"{name}:duplicate_group_index={index}")
    else:
      result[index] = span
  return result


def validate(
    arm: str,
    spans: Sequence[Span],
    *,
    device_modules: Mapping[str, Mapping[str, float | int]],
    trace_buffer_drops: int,
) -> list[str]:
  """Pure fail-closed validator for synthetic tests and a parsed XPlane."""
  if arm not in ARM_COUNTS:
    raise ValueError(f"unsupported arm: {arm!r}")
  reasons = []
  by_name = {
      name: [span for span in spans if span.name == name]
      for name in EVENT_NAMES
  }
  expected_counts = {**COMMON_COUNTS, **ARM_COUNTS[arm]}
  for name, expected in expected_counts.items():
    actual = len(by_name[name])
    if actual != expected:
      reasons.append(f"{name}:count={actual} expected={expected}")
  for span in spans:
    if span.line_name != HOST_LINE:
      reasons.append(
          f"{span.name}:host_line={span.line_name} expected={HOST_LINE}"
      )

  update = _one(by_name, "zero_tim_update")
  reverse_parent = _one(by_name, "reverse_groups")
  train = _one(by_name, "train")
  loss = _one(by_name, "loss_pullback")
  if update is not None and update.stats.get("update_step") != "0":
    reasons.append(
        f"zero_tim_update:update_step={update.stats.get('update_step')} expected=0"
    )
  if train is not None:
    if train.stats.get("step_num") != "0":
      reasons.append(f"train:step_num={train.stats.get('step_num')} expected=0")
    if train.stats.get("_r") != "1":
      reasons.append("train:not_step_trace_annotation")
    if update is not None and not _contains(train, update):
      reasons.append("zero_tim_update:outside_train")
  if update is not None:
    for span in spans:
      if span.name != "train" and not _contains(update, span):
        reasons.append(f"{span.name}:outside_zero_tim_update")

  grouped = {}
  for name in GROUPED_COMMON:
    grouped[name] = _indexed(by_name, name, reasons)
    if set(grouped[name]) != set(range(GROUPS)):
      reasons.append(
          f"{name}:group_indexes={sorted(grouped[name])} expected=0..7"
      )
  arm_grouped = (
      ("fixed_dp_reduce", "gradient_accumulate")
      if arm == "r1"
      else ("staged_accumulate",)
  )
  for name in arm_grouped:
    grouped[name] = _indexed(by_name, name, reasons)
    if set(grouped[name]) != set(range(GROUPS)):
      reasons.append(
          f"{name}:group_indexes={sorted(grouped[name])} expected=0..7"
      )
  if arm == "r2":
    for name in ("fixed_dp_reduce", "gradient_accumulate"):
      span = _one(by_name, name)
      if span is not None and span.stats.get("group_index") != "7":
        reasons.append(
            f"{name}:group_index={span.stats.get('group_index')} expected=7"
        )

  for index in range(GROUPS):
    reverse = grouped["reverse_group"].get(index)
    backward = grouped["model_backward"].get(index)
    adjoint = grouped["report_adjoint"].get(index)
    if reverse is not None:
      for name, child in (("model_backward", backward), ("report_adjoint", adjoint)):
        if child is not None and not _contains(reverse, child):
          reasons.append(f"{name}[{index}]:outside_reverse_group")
      following = grouped["forward_group"].get(index + 1)
      following_loss = grouped["group_loss_pullback"].get(index + 1)
      for name, child in (
          ("forward_group", following),
          ("group_loss_pullback", following_loss),
      ):
        if child is not None and not _contains(reverse, child):
          reasons.append(f"{name}[{index + 1}]:outside_reverse_group[{index}]")
      if (
          backward is not None
          and adjoint is not None
          and backward.end_ns > adjoint.start_ns
      ):
        reasons.append(f"reverse_group[{index}]:model_backward_overlaps_adjoint")
    if reverse_parent is not None and reverse is not None and not _contains(
        reverse_parent, reverse
    ):
      reasons.append(f"reverse_group[{index}]:outside_reverse_groups")
  last_reverse = grouped["reverse_group"].get(GROUPS - 1)
  if (
      loss is not None
      and last_reverse is not None
      and loss.start_ns < last_reverse.end_ns
  ):
    reasons.append("loss_pullback:before_last_reverse_group")

  final_reduce = by_name["fixed_dp_reduce"][-1] if by_name["fixed_dp_reduce"] else None
  final_accumulate = (
      by_name["gradient_accumulate"][-1]
      if by_name["gradient_accumulate"]
      else None
  )
  finite = _one(by_name, "deferred_finite_receipts")
  if arm == "r2":
    staged_fetch = _one(by_name, "staged_receipt_fetch")
    ordered = (final_reduce, staged_fetch, final_accumulate, finite)
  else:
    ordered = (final_reduce, final_accumulate, finite)
  for left, right in zip(ordered, ordered[1:]):
    if left is not None and right is not None and left.end_ns > right.start_ns:
      reasons.append(f"terminal_order:{left.name}>{right.name}")

  if trace_buffer_drops:
    reasons.append(f"trace_buffer_drops={trace_buffer_drops}")
  if set(device_modules) != {f"/device:TPU:{index}" for index in range(8)}:
    reasons.append(
        f"device_module_planes={sorted(device_modules)} expected=TPU:0..7"
    )
  update_seconds = update.duration_ns / 1e9 if update is not None else 0.0
  for plane, summary in sorted(device_modules.items()):
    count = int(summary.get("events", 0))
    span_seconds = float(summary.get("span_seconds", 0.0))
    if count <= 0:
      reasons.append(f"{plane}:xla_modules_empty")
    ratio = span_seconds / update_seconds if update_seconds > 0 else 0.0
    if ratio < 0.90:
      reasons.append(f"{plane}:coverage_ratio={ratio:.6f} expected>=0.90")
  return reasons


def _resolve_xplane(root: Path) -> Path:
  paths = sorted((root / "xprof-update").glob("plugins/profile/*/*.xplane.pb"))
  if len(paths) != 1 or paths[0].stat().st_size <= 0:
    raise ValueError(f"expected one nonempty XPlane, found {len(paths)}")
  return paths[0]


def census(root: Path, arm: str) -> dict:
  """Parses the full XPlane and returns a signed, self-contained receipt."""
  if arm not in ARM_COUNTS:
    raise ValueError(f"unsupported arm: {arm!r}")
  version = importlib.metadata.version("xprof")
  xplane = _resolve_xplane(root)
  from xprof.profile_data import ProfileData  # pylint: disable=g-import-not-at-top

  profile = ProfileData.from_file(str(xplane))
  try:
    host_planes = [plane for plane in profile.planes if plane.name == "/host:CPU"]
    spans = []
    if len(host_planes) == 1:
      for line in host_planes[0].lines:
        for event in line.events:
          if event.name in EVENT_NAMES:
            spans.append(Span(
                name=event.name,
                start_ns=float(event.start_ns),
                duration_ns=float(event.duration_ns),
                line_name=line.name,
                stats=dict(event.stats),
            ))
    drops = sum(
        event.name == "Trace Buffers Dropped"
        for plane in profile.planes
        for line in plane.lines
        for event in line.events
    )
    device_modules = {}
    for plane in profile.planes:
      if not TPU_PLANE.fullmatch(plane.name):
        continue
      events = [
          event
          for line in plane.lines
          if line.name == "XLA Modules"
          for event in line.events
      ]
      if events:
        first = min(float(event.start_ns) for event in events)
        last = max(float(event.start_ns + event.duration_ns) for event in events)
        distinct = len({event.name for event in events})
      else:
        first = last = 0.0
        distinct = 0
      device_modules[plane.name] = {
          "events": len(events),
          "distinct_names": distinct,
          "span_seconds": (last - first) / 1e9,
      }
  finally:
    profile.close()

  reasons = []
  if version != XPROF_VERSION:
    reasons.append(f"xprof_version={version} expected={XPROF_VERSION}")
  if len(host_planes) != 1:
    reasons.append(f"host_planes={len(host_planes)} expected=1")
  reasons.extend(validate(
      arm,
      spans,
      device_modules=device_modules,
      trace_buffer_drops=drops,
  ))
  counts = Counter(span.name for span in spans)
  relative = xplane.relative_to(root)
  return {
      "schema": SCHEMA,
      "verdict": "PASS" if not reasons else "FAIL",
      "reasons": reasons,
      "arm": arm,
      "xprof_version": version,
      "xplane": {
          "path": str(relative),
          "bytes": xplane.stat().st_size,
          "sha256": _sha256(xplane),
      },
      "host": {
          "plane": "/host:CPU" if len(host_planes) == 1 else None,
          "line": HOST_LINE,
          "counts": {name: counts[name] for name in sorted(EVENT_NAMES)},
      },
      "device_modules": device_modules,
      "trace_buffer_drops": drops,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--arm", choices=tuple(ARM_COUNTS), required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite census: {args.output}")
  record = census(args.root, args.arm)
  args.output.write_text(
      json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(
      f"V2_FL_XPLANE_CENSUS verdict={record['verdict']} "
      f"arm={args.arm} reasons={record['reasons']}"
  )
  return 0 if record["verdict"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
