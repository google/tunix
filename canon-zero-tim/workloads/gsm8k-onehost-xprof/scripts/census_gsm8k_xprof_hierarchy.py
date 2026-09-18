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
    "dp2-tp2-long": {"groups": 8},
    "dp2-tp2-long8k": {"groups": 8},
    "dp2-tp2-p45": {"groups": 8},
}
DEFAULT_GEOMETRY = "dp4-tp1"
EXPECTED_COUNTS = {
    "train": 16,
    "zero_tim_update": 1,
    "forward_groups": 1,
    "forward_group": 16,
    "loss_pullback": 1,
    "group_loss_pullback": 0,
    "staged_accumulate": 0,
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
    "group_loss_pullback",
    "staged_accumulate",
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
    keep_tape: bool = False,
    stream_tape: bool = False,
    reduce_once: bool = False,
) -> list[str]:
  """Pure interval/count validator used by real and synthetic censuses.

  With ``keep_tape`` (CANON_P32_KEEP_TAPE=1) the forward phase keeps each
  group's tape and the reverse never replays it, so the ``replay_forward``
  span family must be ABSENT; seeing one means the kept tape was not
  consumed and the run silently fell back to the replay.

  With ``stream_tape`` (CANON_P32_KEEP_TAPE=stream, implies ``keep_tape``)
  the loss cotangent is built per group and the forward of group g+1 is
  issued right after the reverse of group g was dispatched: there is no
  ``forward_groups`` parent; ``forward_group[0]`` and
  ``group_loss_pullback[0]`` close before ``reverse_group[0]`` opens; for
  every later group, ``forward_group[g+1]`` and ``group_loss_pullback[g+1]``
  lie inside ``reverse_group[g]``, after ``model_backward[g]`` closed and
  before ``report_adjoint[g]`` opens (the pipelined two-tape window); and
  the single batch ``loss_pullback`` (the end-of-update bitwise self-check)
  runs after the last reverse and before the optimizer commit.

  With ``reduce_once`` (CANON_DP_REDUCE_ONCE=1) every group owns one
  ``staged_accumulate`` inside its ``reverse_group`` instead of a
  ``fixed_dp_reduce`` and a ``gradient_accumulate``; exactly one
  ``fixed_dp_reduce`` and one ``gradient_accumulate`` (the last group's
  index, ``is_last_accumulate=1``) run after the last reverse and before the
  optimizer commit.
  """
  keep_tape = keep_tape or stream_tape
  reasons = []
  by_name = {
      name: [span for span in spans if span.name == name]
      for name in EXPECTED_COUNTS
  }
  for name, expected in EXPECTED_COUNTS.items():
    actual = len(by_name[name])
    adjusted_expected = expected_groups if expected == 16 else expected
    if keep_tape and name == "replay_forward":
      if actual:
        reasons.append(f"keep_tape_unexpected_replay_forward={actual}")
      continue
    if stream_tape and name == "forward_groups":
      adjusted_expected = 0
    if stream_tape and name == "group_loss_pullback":
      adjusted_expected = expected_groups
    if reduce_once and name in ("fixed_dp_reduce", "gradient_accumulate"):
      adjusted_expected = 1
    if reduce_once and name == "staged_accumulate":
      adjusted_expected = expected_groups
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
      name: (
          {}
          if (keep_tape and name == "replay_forward")
          or (not stream_tape and name == "group_loss_pullback")
          or (not reduce_once and name == "staged_accumulate")
          or (
              reduce_once
              and name in ("fixed_dp_reduce", "gradient_accumulate")
          )
          else _grouped(
              by_name,
              name,
              expected_groups=expected_groups,
              reasons=reasons,
          )
      )
      for name in GROUP_NAMES
  }
  for index, accumulator in (
      {} if reduce_once else grouped["gradient_accumulate"]
  ).items():
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
  if stream_tape:
    for index, reverse in grouped["reverse_group"].items():
      issued = grouped["forward_group"].get(index)
      cotangent = grouped["group_loss_pullback"].get(index)
      following = grouped["forward_group"].get(index + 1)
      if update is not None:
        for name, span in (
            (f"forward_group[{index}]", issued),
            (f"group_loss_pullback[{index}]", cotangent),
        ):
          if span is not None and not _contains(update, span):
            reasons.append(f"{name}:outside_zero_tim_update")
      if (
          issued is not None
          and cotangent is not None
          and issued.end_ns > cotangent.start_ns
      ):
        reasons.append(
            f"forward_group[{index}]:after_group_loss_pullback[{index}]"
        )
      if cotangent is not None and cotangent.end_ns > reverse.start_ns:
        reasons.append(
            f"group_loss_pullback[{index}]:after_reverse_group[{index}]"
        )
      following_cotangent = grouped["group_loss_pullback"].get(index + 1)
      backward = grouped["model_backward"].get(index)
      adjoint = grouped["report_adjoint"].get(index)
      for name, span in (
          (f"forward_group[{index + 1}]", following),
          (f"group_loss_pullback[{index + 1}]", following_cotangent),
      ):
        if span is None:
          continue
        if not _contains(reverse, span):
          reasons.append(f"{name}:outside_reverse_group[{index}]")
        if backward is not None and span.start_ns < backward.end_ns:
          reasons.append(f"{name}:before_model_backward[{index}]_closed")
        if adjoint is not None and span.end_ns > adjoint.start_ns:
          reasons.append(f"{name}:after_report_adjoint[{index}]")
    last_reverse = grouped["reverse_group"].get(expected_groups - 1)
    if (
        loss is not None
        and last_reverse is not None
        and loss.start_ns < last_reverse.end_ns
    ):
      reasons.append("loss_pullback:before_last_reverse_group")
    if (
        loss is not None
        and optimizer is not None
        and loss.end_ns > optimizer.start_ns
    ):
      reasons.append("loss_pullback:after_optimizer_commit")
  if reduce_once:
    # The update-level reduce and accumulate carry the last group's index.
    single = {}
    for name in ("fixed_dp_reduce", "gradient_accumulate"):
      spans_of = by_name.get(name, [])
      span = spans_of[0] if len(spans_of) == 1 else None
      single[name] = span
      if span is not None:
        raw_index = span.stats.get("group_index")
        if raw_index is None or int(raw_index) != expected_groups - 1:
          reasons.append(
              f"{name}:group_index={raw_index} expected={expected_groups - 1}"
          )
    grouped["fixed_dp_reduce"] = (
        {expected_groups - 1: single["fixed_dp_reduce"]}
        if single["fixed_dp_reduce"] is not None
        else {}
    )
    grouped["gradient_accumulate"] = (
        {expected_groups - 1: single["gradient_accumulate"]}
        if single["gradient_accumulate"] is not None
        else {}
    )
    last_reverse = grouped["reverse_group"].get(expected_groups - 1)
    reduce = single["fixed_dp_reduce"]
    accumulate = single["gradient_accumulate"]
    if (
        reduce is not None
        and last_reverse is not None
        and reduce.start_ns < last_reverse.end_ns
    ):
      reasons.append("fixed_dp_reduce:before_last_reverse_group")
    if (
        reduce is not None
        and accumulate is not None
        and reduce.end_ns > accumulate.start_ns
    ):
      reasons.append("update:order=fixed_dp_reduce>gradient_accumulate")
    if (
        accumulate is not None
        and optimizer is not None
        and accumulate.end_ns > optimizer.start_ns
    ):
      reasons.append("gradient_accumulate:after_optimizer_commit")
  for index, reverse in grouped["reverse_group"].items():
    train = trains.get(index)
    if train is not None and not _contains(train, reverse):
      reasons.append(f"reverse_group[{index}]:outside_train")
    if update is not None and not _contains(update, reverse):
      reasons.append(f"reverse_group[{index}]:outside_zero_tim_update")
    if reduce_once:
      accumulated = grouped["staged_accumulate"].get(index)
      backward = grouped["model_backward"].get(index)
      if accumulated is not None and not _contains(reverse, accumulated):
        reasons.append(f"staged_accumulate[{index}]:outside_reverse_group")
      if (
          accumulated is not None
          and backward is not None
          and accumulated.start_ns < backward.end_ns
      ):
        reasons.append(
            f"staged_accumulate[{index}]:before_model_backward[{index}]_closed"
        )
    stage_spans = []
    for stage in REVERSE_STAGES:
      if reduce_once and stage in ("fixed_dp_reduce", "gradient_accumulate"):
        continue
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
  if not stream_tape:
    if forward_parent is not None:
      ordered.append(("forward_groups", forward_parent))
    if loss is not None:
      ordered.append(("loss_pullback", loss))
  ordered.extend(
      (f"train[{index}]", trains[index])
      for index in range(expected_groups)
      if index in trains
  )
  if len(ordered) == expected_groups + (0 if stream_tape else 2):
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
  parser.add_argument(
      "--p32-keep-tape",
      default="",
      help=(
          "the CANON_P32_KEEP_TAPE value the run was launched with; 1 "
          "requires the replay_forward span family to be absent; stream "
          "additionally requires the per-group group_loss_pullback spans, "
          "no forward_groups parent and the two-tape window order"
      ),
  )
  parser.add_argument(
      "--dp-reduce-once",
      default="",
      help=(
          "the CANON_DP_REDUCE_ONCE value the run was launched with; 1 "
          "requires one staged_accumulate per group and exactly one "
          "fixed_dp_reduce and gradient_accumulate after the last reverse"
      ),
  )
  args = parser.parse_args()
  if args.dp_reduce_once not in ("", "0", "1"):
    raise ValueError(
        f"--dp-reduce-once must be empty, 0 or 1: {args.dp_reduce_once!r}"
    )
  if args.p32_keep_tape not in ("", "0", "1", "stream"):
    raise ValueError(
        "--p32-keep-tape must be empty, 0, 1 or stream: "
        f"{args.p32_keep_tape!r}"
    )
  expected_groups = GEOMETRIES[args.geometry]["groups"]
  xplane = _resolve_xplane(args.run_root)
  spans, device_step_counts, compiler_counts = read_xplane(xplane)
  reasons = validate_hierarchy(
      spans,
      device_step_counts=device_step_counts,
      compiler_counts=compiler_counts,
      expected_update_step=args.expected_update_step,
      expected_groups=expected_groups,
      keep_tape=args.p32_keep_tape in ("1", "stream"),
      stream_tape=args.p32_keep_tape == "stream",
      reduce_once=args.dp_reduce_once == "1",
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
