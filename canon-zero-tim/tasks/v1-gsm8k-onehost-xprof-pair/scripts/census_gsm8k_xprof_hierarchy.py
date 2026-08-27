#!/usr/bin/env python3
"""Validates the Zero-HP host hierarchy in one complete XPlane."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Mapping, Sequence


# The registered carrier geometries: one committed update owns one
# transaction per gradient group (64 global trajectories / dp_size).
GEOMETRIES = {
    "dp4-tp1": {"groups": 16},
    "dp2-tp2": {"groups": 32},
}
DEFAULT_GEOMETRY = "dp4-tp1"
EXPECTED_COUNTS = {
    "train": 16,
    "zero_tim_update": 1,
    "forward_groups": 1,
    "forward_group": 16,
    "loss_pullback": 1,
    "reverse_group": 16,
    "replay_forward": 16,
    "model_backward": 16,
    "report_adjoint": 16,
    "fixed_dp_reduce": 16,
    "gradient_accumulate": 16,
    "optimizer_commit": 1,
}
COMPILER_EVENTS = (
    "backend_compile_and_load",
    "PJRT_Client_Compile",
    "TpuCompiler::Compile",
)
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
    compiler_counts: Mapping[str, int],
    expected_update_step: int = 2,
    expected_groups: int = 16,
    require_step_marker: bool = True,
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

  update = _one(by_name, "zero_tim_update")
  forward_parent = _one(by_name, "forward_groups")
  loss = _one(by_name, "loss_pullback")
  optimizer = _one(by_name, "optimizer_commit")
  trains = {}
  first_train_step = expected_update_step * expected_groups
  for train in by_name["train"]:
    try:
      step_num = int(train.stats.get("step_num", ""))
    except ValueError:
      step_num = None
    micro_step = (
        step_num - first_train_step if step_num is not None else None
    )
    if micro_step is None or not 0 <= micro_step < expected_groups:
      reasons.append(
          f"train:step_num={step_num} "
          f"expected={first_train_step}.."
          f"{first_train_step + expected_groups - 1}"
      )
    elif micro_step in trains:
      reasons.append(f"train:duplicate_step_num={step_num}")
    else:
      trains[micro_step] = train
    if require_step_marker and train.stats.get("_r") != "1":
      reasons.append(f"train[{step_num}]:not_step_trace_annotation")
  if set(trains) != set(range(expected_groups)):
    reasons.append(
        f"train:microsteps={sorted(trains)} expected=0..{expected_groups - 1}"
    )
  if update is not None:
    try:
      update_step = int(update.stats.get("update_step", ""))
    except ValueError:
      update_step = None
    if update_step != expected_update_step:
      reasons.append(
          f"zero_tim_update:update_step={update_step} "
          f"expected={expected_update_step}"
      )

  for child_name, child in (
      ("forward_groups", forward_parent),
      ("loss_pullback", loss),
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
    if update_step != expected_update_step:
      reasons.append(
          f"optimizer_commit:update_step={update_step} "
          f"expected={expected_update_step}"
      )
  for index, span in grouped["forward_group"].items():
    if forward_parent is not None and not _contains(forward_parent, span):
      reasons.append(f"forward_group[{index}]:outside_forward_groups")
  for index, reverse in grouped["reverse_group"].items():
    train = trains.get(index)
    if train is not None and not _contains(train, reverse):
      reasons.append(f"reverse_group[{index}]:outside_train")
    if update is not None and not _contains(update, reverse):
      reasons.append(f"reverse_group[{index}]:outside_zero_tim_update")
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

  if update is not None:
    for index, train in trains.items():
      if not _contains(update, train):
        reasons.append(f"train[{index}]:outside_zero_tim_update")
  last_train = trains.get(expected_groups - 1)
  if (
      last_train is not None
      and optimizer is not None
      and not _contains(last_train, optimizer)
  ):
    reasons.append("optimizer_commit:outside_last_train")
  last_accumulator = grouped["gradient_accumulate"].get(expected_groups - 1)
  if (
      last_accumulator is not None
      and optimizer is not None
      and last_accumulator.end_ns > optimizer.start_ns
  ):
    reasons.append("last_train:gradient_accumulate_overlaps_optimizer")

  ordered = []
  if forward_parent is not None:
    ordered.append(("forward_groups", forward_parent))
  if loss is not None:
    ordered.append(("loss_pullback", loss))
  ordered.extend(
      (f"train[{index}]", trains[index])
      for index in range(expected_groups)
      if index in trains
  )
  if len(ordered) == expected_groups + 2:
    for (left_name, left), (right_name, right) in zip(ordered, ordered[1:]):
      if left.end_ns > right.start_ns:
        reasons.append(f"update:order={left_name}>{right_name}")
  for name in COMPILER_EVENTS:
    count = compiler_counts.get(name, 0)
    if count:
      reasons.append(f"captured_compile:{name}={count} expected=0")
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


def read_xplane(
    path: Path,
) -> tuple[list[Span], dict[str, int], dict[str, int]]:
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
    compiler_spans = []
    for line in host_planes[0].lines:
      for event in line.events:
        span = Span(
            name=event.name,
            start_ns=float(event.start_ns),
            duration_ns=float(event.duration_ns),
            line_name=line.name,
            stats=dict(event.stats),
        )
        if event.name in EXPECTED_COUNTS:
          spans.append(span)
        if event.name in COMPILER_EVENTS:
          compiler_spans.append(span)
    device_step_counts = {}
    for plane in profile.planes:
      if not TPU_PLANE.fullmatch(plane.name):
        continue
      device_step_counts[plane.name] = sum(
          len(line.events) for line in plane.lines if line.name == "Steps"
      )
    update = next(
        (span for span in spans if span.name == "zero_tim_update"), None
    )
    compiler_counts = {
        name: sum(
            span.name == name
            and update is not None
            and _contains(update, span)
            for span in compiler_spans
        )
        for name in COMPILER_EVENTS
    }
    return spans, device_step_counts, compiler_counts
  finally:
    profile.close()


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-root", type=Path, required=True)
  parser.add_argument(
      "--expected-update-step", "--expected-step",
      dest="expected_update_step", type=int, default=2
  )
  parser.add_argument(
      "--geometry",
      choices=tuple(sorted(GEOMETRIES)),
      default=DEFAULT_GEOMETRY,
      help="registered carrier geometry the run was launched with",
  )
  args = parser.parse_args()
  expected_groups = GEOMETRIES[args.geometry]["groups"]
  xplane = _resolve_xplane(args.run_root)
  spans, device_step_counts, compiler_counts = read_xplane(xplane)
  reasons = validate_hierarchy(
      spans,
      device_step_counts=device_step_counts,
      compiler_counts=compiler_counts,
      expected_update_step=args.expected_update_step,
      expected_groups=expected_groups,
  )
  counts = {
      name: sum(span.name == name for span in spans)
      for name in EXPECTED_COUNTS
  }
  print("hierarchy_counts=" + json.dumps(counts, sort_keys=True))
  print("device_steps=" + json.dumps(device_step_counts, sort_keys=True))
  print("captured_compiler_events=" + json.dumps(
      compiler_counts, sort_keys=True
  ))
  if reasons:
    for reason in reasons:
      print("  RED " + reason)
    print(
        f"V1_GSM8K_XPROF_HIERARCHY_CENSUS_RED reasons={len(reasons)}"
    )
    return 1
  first_step = args.expected_update_step * expected_groups
  print(
      "V1_GSM8K_XPROF_HIERARCHY_CENSUS_GREEN "
      f"update_step={args.expected_update_step} "
      f"train_steps={first_step}.."
      f"{first_step + expected_groups - 1} host_plane=/host:CPU "
      f"host_line={HOST_LINE_NAME} steps_planes=8 "
      f"forward_groups={expected_groups} "
      f"reverse_transactions={expected_groups} "
      f"micro_steps=0..{expected_groups - 1} "
      f"last_accumulate={expected_groups - 1} optimizer_owned_by_last=1 "
      "compiler_events=0"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
