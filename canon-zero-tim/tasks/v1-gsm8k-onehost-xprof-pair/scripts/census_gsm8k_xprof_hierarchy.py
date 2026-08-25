#!/usr/bin/env python3
"""Validates the Zero-HP host hierarchy in one complete XPlane."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Mapping, Sequence


EXPECTED_COUNTS = {
    "train": 1,
    "zero_tim_update": 1,
    "forward_groups": 1,
    "forward_group": 16,
    "loss_pullback": 1,
    "reverse_groups": 1,
    "reverse_group": 16,
    "replay_forward": 16,
    "model_backward": 16,
    "report_adjoint": 16,
    "fixed_dp_reduce": 16,
    "gradient_accumulate": 16,
    "optimizer_commit": 1,
}
GROUP_NAMES = (
    "forward_group",
    "reverse_group",
    "replay_forward",
    "model_backward",
    "report_adjoint",
    "fixed_dp_reduce",
    "gradient_accumulate",
)
REVERSE_STAGES = (
    "replay_forward",
    "model_backward",
    "report_adjoint",
    "fixed_dp_reduce",
    "gradient_accumulate",
)
TPU_PLANE = re.compile(r"/device:TPU:\d+")
HOST_LINE_NAME = "python3"


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


def _contains(parent: Span, child: Span) -> bool:
  return parent.start_ns <= child.start_ns and child.end_ns <= parent.end_ns


def _one(by_name: Mapping[str, list[Span]], name: str) -> Span | None:
  values = by_name.get(name, [])
  return values[0] if len(values) == 1 else None


def _grouped(
    by_name: Mapping[str, list[Span]],
    name: str,
    *,
    expected_groups: int,
    reasons: list[str],
) -> dict[int, Span]:
  result = {}
  for span in by_name.get(name, []):
    raw_index = span.stats.get("group_index")
    try:
      index = int(raw_index) if raw_index is not None else None
    except ValueError:
      index = None
    if index is None:
      reasons.append(f"{name}:missing_or_invalid_group_index")
      continue
    if index in result:
      reasons.append(f"{name}:duplicate_group_index={index}")
      continue
    result[index] = span
  expected = set(range(expected_groups))
  if set(result) != expected:
    reasons.append(
        f"{name}:group_indexes={sorted(result)} "
        f"expected=0..{expected_groups - 1}"
    )
  return result


def validate_hierarchy(
    spans: Sequence[Span],
    *,
    device_step_counts: Mapping[str, int],
    expected_step: int = 1,
    expected_groups: int = 16,
) -> list[str]:
  """Pure interval/count validator used by real and synthetic censuses."""
  reasons = []
  by_name = {
      name: [span for span in spans if span.name == name]
      for name in EXPECTED_COUNTS
  }
  for name, expected in EXPECTED_COUNTS.items():
    actual = len(by_name[name])
    adjusted_expected = expected_groups if expected == 16 else expected
    if actual != adjusted_expected:
      reasons.append(f"{name}:count={actual} expected={adjusted_expected}")

  for span in spans:
    if span.name not in EXPECTED_COUNTS or span.line_name == HOST_LINE_NAME:
      continue
    group_index = span.stats.get("group_index")
    suffix = f"[{group_index}]" if group_index is not None else ""
    reasons.append(
        f"{span.name}{suffix}:host_line={span.line_name} "
        f"expected={HOST_LINE_NAME}"
    )

  if len(device_step_counts) != 8:
    reasons.append(
        f"device_steps:planes={len(device_step_counts)} expected=8"
    )
  for plane, count in sorted(device_step_counts.items()):
    if not TPU_PLANE.fullmatch(plane):
      reasons.append(f"device_steps:invalid_plane={plane}")
    if count <= 0:
      reasons.append(f"device_steps:{plane}=empty")

  train = _one(by_name, "train")
  update = _one(by_name, "zero_tim_update")
  forward_parent = _one(by_name, "forward_groups")
  loss = _one(by_name, "loss_pullback")
  reverse_parent = _one(by_name, "reverse_groups")
  optimizer = _one(by_name, "optimizer_commit")
  if train is not None:
    try:
      step_num = int(train.stats.get("step_num", ""))
    except ValueError:
      step_num = None
    if step_num != expected_step:
      reasons.append(f"train:step_num={step_num} expected={expected_step}")
    if train.stats.get("_r") != "1":
      reasons.append("train:not_step_trace_annotation")
  if train is not None and update is not None and not _contains(train, update):
    reasons.append("zero_tim_update:outside_train")

  for child_name, child in (
      ("forward_groups", forward_parent),
      ("loss_pullback", loss),
      ("reverse_groups", reverse_parent),
      ("optimizer_commit", optimizer),
  ):
    if (
        update is not None
        and child is not None
        and not _contains(update, child)
    ):
      reasons.append(f"{child_name}:outside_zero_tim_update")

  grouped = {
      name: _grouped(
          by_name,
          name,
          expected_groups=expected_groups,
          reasons=reasons,
      )
      for name in GROUP_NAMES
  }
  for index, accumulator in grouped["gradient_accumulate"].items():
    try:
      micro_step = int(accumulator.stats.get("micro_step", ""))
    except ValueError:
      micro_step = None
    if micro_step != index:
      reasons.append(
          f"gradient_accumulate[{index}]:micro_step={micro_step} "
          f"expected={index}"
      )
    try:
      is_last = int(accumulator.stats.get("is_last_accumulate", ""))
    except ValueError:
      is_last = None
    expected_last = int(index == expected_groups - 1)
    if is_last != expected_last:
      reasons.append(
          f"gradient_accumulate[{index}]:is_last_accumulate={is_last} "
          f"expected={expected_last}"
      )
  if optimizer is not None:
    try:
      update_step = int(optimizer.stats.get("update_step", ""))
    except ValueError:
      update_step = None
    if update_step != expected_step:
      reasons.append(
          f"optimizer_commit:update_step={update_step} "
          f"expected={expected_step}"
      )
  for index, span in grouped["forward_group"].items():
    if forward_parent is not None and not _contains(forward_parent, span):
      reasons.append(f"forward_group[{index}]:outside_forward_groups")
  for index, reverse in grouped["reverse_group"].items():
    if reverse_parent is not None and not _contains(reverse_parent, reverse):
      reasons.append(f"reverse_group[{index}]:outside_reverse_groups")
    stage_spans = []
    for stage in REVERSE_STAGES:
      child = grouped[stage].get(index)
      if child is None:
        continue
      if not _contains(reverse, child):
        reasons.append(f"{stage}[{index}]:outside_reverse_group")
      stage_spans.append((stage, child))
    for (left_name, left), (right_name, right) in zip(
        stage_spans, stage_spans[1:]
    ):
      if left.end_ns > right.start_ns:
        reasons.append(
            f"reverse_group[{index}]:order={left_name}>{right_name}"
        )

  if all(
      value is not None
      for value in (forward_parent, loss, reverse_parent, optimizer)
  ):
    ordered = (
        ("forward_groups", forward_parent),
        ("loss_pullback", loss),
        ("reverse_groups", reverse_parent),
        ("optimizer_commit", optimizer),
    )
    for (left_name, left), (right_name, right) in zip(ordered, ordered[1:]):
      if left.end_ns > right.start_ns:
        reasons.append(f"update:order={left_name}>{right_name}")
  return reasons


def _resolve_xplane(path: Path) -> Path:
  if path.is_file():
    if path.suffixes[-2:] != [".xplane", ".pb"] or path.stat().st_size <= 0:
      raise ValueError(f"not a non-empty xplane: {path}")
    return path
  files = sorted(
      candidate
      for candidate in path.glob(
          "train/xprof/plugins/profile/*/*.xplane.pb"
      )
      if candidate.stat().st_size > 0
  )
  if len(files) != 1:
    raise ValueError(
        f"expected exactly one non-empty xplane, found {len(files)}"
    )
  return files[0]


def read_xplane(path: Path) -> tuple[list[Span], dict[str, int]]:
  """Reads only the host hierarchy and device Steps rows from a full XPlane."""
  from xprof import profile_data  # pylint: disable=g-import-not-at-top

  profile = profile_data.ProfileData.from_file(str(path))
  try:
    host_planes = [
        plane for plane in profile.planes if plane.name == "/host:CPU"
    ]
    if len(host_planes) != 1:
      raise ValueError(
          f"expected one /host:CPU plane, found {len(host_planes)}"
      )
    spans = []
    for line in host_planes[0].lines:
      for event in line.events:
        if event.name not in EXPECTED_COUNTS:
          continue
        spans.append(Span(
            name=event.name,
            start_ns=float(event.start_ns),
            duration_ns=float(event.duration_ns),
            line_name=line.name,
            stats=dict(event.stats),
        ))
    device_step_counts = {}
    for plane in profile.planes:
      if not TPU_PLANE.fullmatch(plane.name):
        continue
      device_step_counts[plane.name] = sum(
          len(line.events) for line in plane.lines if line.name == "Steps"
      )
    return spans, device_step_counts
  finally:
    profile.close()


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-root", type=Path, required=True)
  parser.add_argument("--expected-step", type=int, default=1)
  args = parser.parse_args()
  xplane = _resolve_xplane(args.run_root)
  spans, device_step_counts = read_xplane(xplane)
  reasons = validate_hierarchy(
      spans,
      device_step_counts=device_step_counts,
      expected_step=args.expected_step,
  )
  counts = {
      name: sum(span.name == name for span in spans)
      for name in EXPECTED_COUNTS
  }
  print("hierarchy_counts=" + json.dumps(counts, sort_keys=True))
  print("device_steps=" + json.dumps(device_step_counts, sort_keys=True))
  if reasons:
    for reason in reasons:
      print("  RED " + reason)
    print(
        f"V1_GSM8K_XPROF_HIERARCHY_CENSUS_RED reasons={len(reasons)}"
    )
    return 1
  print(
      "V1_GSM8K_XPROF_HIERARCHY_CENSUS_GREEN "
      f"train_step={args.expected_step} host_plane=/host:CPU "
      f"host_line={HOST_LINE_NAME} steps_planes=8 "
      "forward_groups=16 reverse_groups=16 transactions=16 "
      "micro_steps=0..15 last_accumulate=15 optimizer_update=1"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
