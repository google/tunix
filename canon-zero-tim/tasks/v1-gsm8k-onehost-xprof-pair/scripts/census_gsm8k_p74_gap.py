#!/usr/bin/env python3
"""Fail-closed XPlane receipt for the P74 checked-VMA chunk boundary."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import glob
import json
from pathlib import Path
import re
import statistics
from typing import Mapping, Sequence


SCHEMA = "canon.v1.gsm8k-onehost-xprof.p74-gap.v1"
GEOMETRY = "dp2-tp2"
EXPECTED_WINDOWS = 64
# The long geometry's windows are data dependent (one per chunk pass): the
# expectation is read from the captured update's own forward_group_issued
# lines instead of a constant.
LONG_GEOMETRIES = {
    "dp2-tp2-long": {"groups": 8, "sequence_bucket": 256},
    "dp2-tp2-long8k": {"groups": 8, "sequence_bucket": 256},
    "dp2-tp2-p45": {"groups": 8, "sequence_bucket": 256},
}
_ISSUED_RE = re.compile(r"forward_group_issued .*?n_real=\(([0-9, ]+)\)")


def expected_windows_from_raw_log(raw_log, *, groups, sequence_bucket):
  """Sum of the last ``groups`` groups' chunk counts (the captured update)."""
  text = raw_log.read_text(encoding="utf-8", errors="replace")
  issued = [
      max(int(value) for value in match.group(1).split(","))
      for match in _ISSUED_RE.finditer(text)
  ]
  if len(issued) < groups:
    raise ValueError(
        f"forward_group_issued lines={len(issued)} < groups={groups}"
    )
  return sum(
      (longest + sequence_bucket - 1) // sequence_bucket
      for longest in issued[-groups:]
  )
MAX_MEAN_GAP_MS = 70.0
SEED_MODULE = "jit_convert_element_type"
# The window opens when the head cotangent is ready.  Before the rows glue
# was folded into the rows programs that was the eager cast of dlogits
# (jit_convert_element_type); since then the zt_tr_bwd_logprob program
# emits dlogits in the head dtype itself.  The nearest preceding seed of
# either spelling opens the window; the identity contract (only the P74
# partition program between seed and head) is unchanged.
SEED_MODULES = ("jit_zt_tr_bwd_logprob", SEED_MODULE)
HEAD_MODULE = "jit_zt_tr_dp_parallel_bwd_head"
# tasks/v2_dispatch Phase 7: with the reverse chunk program the seed, the
# partition and the head pullback are ops of one zt_tr_bwd_chunk program
# per chunk pass -- no host boundary exists between them, so the identity
# contract holds by construction and the census counts chunk programs
# instead of seed-to-head windows (and reds a standalone head launch).
CHUNK_MODULE = "jit_zt_tr_bwd_chunk"
PARTITION_MODULE = "jit__p74_identity_head_cotangent_partition"
VICTIM_KINDS = (
    "slow_np.asarray(jax.Array)",
    "slow_shard_args",
    "D2H Dispatch",
    "XlaDelinearize",
    "d2h_77791232",
    "h2d_buffer_bf16_256x75968",
    "h2d_38895616",
)
_REVERSE_RE = re.compile(
    r"\[PERF\]\s+stage=p32_vag_reverse\s+"
    r"seconds=(?P<seconds>[0-9.]+)\s+groups=(?P<groups>\d+)\s+"
    r"mean=(?P<mean>[0-9.]+)\s+max=(?P<max>[0-9.]+)"
)


@dataclass(frozen=True)
class Event:
  name: str
  start_ns: int
  duration_ns: int
  stats: Mapping[str, str]

  @property
  def end_ns(self) -> int:
    return self.start_ns + self.duration_ns


@dataclass(frozen=True)
class Window:
  start_ns: int
  end_ns: int
  intervening_modules: tuple[str, ...]

  @property
  def gap_ms(self) -> float:
    return (self.end_ns - self.start_ns) / 1e6


def _base(name: str) -> str:
  """Drops the XLA fingerprint suffix without weakening the stem match."""
  return re.sub(r"[(.].*", "", name).strip()


def find_windows(modules: Sequence[Event]) -> list[Window]:
  """Finds each seed-program end to first rank-parallel-head start."""
  ordered = sorted(modules, key=lambda event: event.start_ns)
  windows = []
  for head_index, head in enumerate(ordered):
    if HEAD_MODULE not in _base(head.name):
      continue
    seed_index = None
    for candidate in range(head_index - 1, max(-1, head_index - 13), -1):
      candidate_name = _base(ordered[candidate].name)
      if candidate_name in SEED_MODULES:
        seed_index = candidate
        break
      if "dp_parallel_bwd" in candidate_name:
        break
    if seed_index is None:
      raise ValueError(
          f"head at module index {head_index} has no seed in prior 12 modules"
      )
    seed = ordered[seed_index]
    windows.append(Window(
        start_ns=seed.end_ns,
        end_ns=head.start_ns,
        intervening_modules=tuple(
            _base(event.name) for event in ordered[seed_index + 1:head_index]
        ),
    ))
  return windows


def victim_kind(event: Event) -> str | None:
  size = str(event.stats.get("size", ""))
  if event.name == "tpu::System::TransferFromDevice" and size == "77791232":
    return "d2h_77791232"
  if "BufferFromHostBuffer(bf16[256,75968])" in event.name:
    return "h2d_buffer_bf16_256x75968"
  if event.name == "tpu::System::TransferToDevice" and size == "38895616":
    return "h2d_38895616"
  if event.name in ("np.asarray(jax.Array)", "shard_args"):
    if event.duration_ns > 1_000_000:
      return f"slow_{event.name}"
  if event.name in ("D2H Dispatch", "XlaDelinearize"):
    return event.name
  return None


def _overlaps(event: Event, window: Window) -> bool:
  return event.start_ns < window.end_ns and event.end_ns > window.start_ns


def analyze(
    modules: Sequence[Event],
    host_events: Sequence[Event],
    *,
    expected_windows: int = EXPECTED_WINDOWS,
    max_mean_gap_ms: float = MAX_MEAN_GAP_MS,
    reverse_chunk: bool = False,
) -> dict:
  """Builds the quantitative P74 receipt from synthetic or real events."""
  if reverse_chunk:
    return _analyze_reverse_chunk(
        modules, host_events, expected_windows=expected_windows
    )
  windows = find_windows(modules)
  gaps = [window.gap_ms for window in windows]
  reasons = []
  if len(windows) != expected_windows:
    reasons.append(f"windows={len(windows)} expected={expected_windows}")

  identity_windows = sum(
      window.intervening_modules == (PARTITION_MODULE,) for window in windows
  )
  if identity_windows != len(windows):
    reasons.append(
        f"identity_windows={identity_windows} expected={len(windows)}"
    )

  overlap = Counter()
  global_victim = Counter()
  victim_windows = set()
  for event in host_events:
    kind = victim_kind(event)
    if kind is None:
      continue
    global_victim[kind] += 1
    for index, window in enumerate(windows):
      if _overlaps(event, window):
        overlap[kind] += 1
        victim_windows.add(index)
        break
  for kind in VICTIM_KINDS:
    if overlap[kind]:
      reasons.append(f"victim_overlap.{kind}={overlap[kind]}")

  mean_ms = statistics.fmean(gaps) if gaps else None
  if mean_ms is None or mean_ms > max_mean_gap_ms:
    reasons.append(
        f"mean_gap_ms={mean_ms!r} max_allowed={max_mean_gap_ms:.3f}"
    )
  intervening = Counter(
      name for window in windows for name in window.intervening_modules
  )
  return {
      "schema": SCHEMA,
      "status": "PASS" if not reasons else "FAIL",
      "geometry": GEOMETRY,
      "acceptance": {
          "expected_windows": expected_windows,
          "max_mean_gap_ms": max_mean_gap_ms,
          "exact_victim_overlap_events": 0,
          "partition_module_per_window": PARTITION_MODULE,
      },
      "gap": {
          "windows": len(windows),
          "total_ms": sum(gaps),
          "mean_ms": mean_ms,
          "max_ms": max(gaps) if gaps else None,
          "min_ms": min(gaps) if gaps else None,
      },
      "identity_windows": identity_windows,
      "intervening_modules": dict(sorted(intervening.items())),
      "victim_overlap": {kind: overlap[kind] for kind in VICTIM_KINDS},
      "victim_global": {kind: global_victim[kind] for kind in VICTIM_KINDS},
      "windows_with_any_victim": len(victim_windows),
      "reasons": reasons,
  }


def _analyze_reverse_chunk(modules, host_events, *, expected_windows):
  chunks = sum(_base(event.name) == CHUNK_MODULE for event in modules)
  heads = sum(HEAD_MODULE in _base(event.name) for event in modules)
  reasons = []
  if chunks != expected_windows:
    reasons.append(f"reverse_chunk_programs={chunks} expected={expected_windows}")
  if heads:
    reasons.append(f"reverse_chunk_unexpected_head_module={heads}")
  global_victim = Counter()
  for event in host_events:
    kind = victim_kind(event)
    if kind is not None:
      global_victim[kind] += 1
  return {
      "schema": SCHEMA,
      "status": "PASS" if not reasons else "FAIL",
      "geometry": GEOMETRY,
      "acceptance": {
          "expected_windows": expected_windows,
          "max_mean_gap_ms": 0.0,
          "exact_victim_overlap_events": 0,
          "partition_module_per_window": "in-graph (zt_tr_bwd_chunk)",
      },
      "gap": {
          "windows": chunks,
          "total_ms": 0.0,
          "mean_ms": 0.0,
          "max_ms": 0.0,
          "min_ms": 0.0,
      },
      "identity_windows": chunks,
      "intervening_modules": {},
      "victim_overlap": {kind: 0 for kind in VICTIM_KINDS},
      "victim_global": {kind: global_victim[kind] for kind in VICTIM_KINDS},
      "windows_with_any_victim": 0,
      "reverse_chunk": True,
      "reasons": reasons,
  }


def parse_reverse_wall(raw_path: Path) -> dict:
  text = raw_path.read_text(encoding="utf-8", errors="replace")
  rows = [
      {
          "seconds": float(match.group("seconds")),
          "groups": int(match.group("groups")),
          "mean_seconds": float(match.group("mean")),
          "max_seconds": float(match.group("max")),
      }
      for match in _REVERSE_RE.finditer(text)
  ]
  return {"rows": rows, "captured_update": rows[-1] if rows else None}


def _resolve_xplane(run_root: Path) -> Path:
  files = sorted(
      Path(path) for path in glob.glob(
          str(run_root / "train/xprof/plugins/profile/*/*.xplane.pb")
      )
      if Path(path).stat().st_size > 0
  )
  if len(files) != 1:
    raise ValueError(
        f"expected exactly one non-empty XPlane, found {len(files)}"
    )
  return files[0]


def _read_xplane(path: Path) -> tuple[list[Event], list[Event]]:
  # Import lazily so synthetic host tests do not require the xprof wheel.
  from xprof import profile_data  # pylint: disable=import-outside-toplevel

  profile = profile_data.ProfileData.from_file(str(path))
  device = next(
      (
          plane for plane in profile.planes
          if plane.name.endswith("TPU:0")
          or re.search(r"/device:TPU:0$", plane.name)
      ),
      None,
  )
  if device is None:
    raise ValueError("XPlane has no TPU:0 device plane")
  modules = []
  for line in device.lines:
    if line.name == "XLA Modules":
      modules.extend(
          Event(
              event.name,
              int(event.start_ns),
              int(event.duration_ns),
              {str(key): str(value) for key, value in event.stats},
          )
          for event in line.events
      )
  host_events = []
  for plane in profile.planes:
    if re.search(r"/device:TPU:\d+$", plane.name):
      continue
    for line in plane.lines:
      host_events.extend(
          Event(
              event.name,
              int(event.start_ns),
              int(event.duration_ns),
              {str(key): str(value) for key, value in event.stats},
          )
          for event in line.events
      )
  return modules, host_events


def main() -> int:
  global GEOMETRY
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-root", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
      "--geometry", default=GEOMETRY,
      choices=(GEOMETRY,) + tuple(sorted(LONG_GEOMETRIES)),
  )
  parser.add_argument(
      "--p32-reverse-chunk", default="", choices=("", "0", "1"),
      help="1 when the adapter under test runs each reverse chunk as one program",
  )
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(args.output)
  GEOMETRY = args.geometry

  try:
    expected_windows = EXPECTED_WINDOWS
    if args.geometry in LONG_GEOMETRIES:
      expected_windows = expected_windows_from_raw_log(
          args.run_root / "train/raw.log", **LONG_GEOMETRIES[args.geometry]
      )
    xplane = _resolve_xplane(args.run_root)
    modules, host_events = _read_xplane(xplane)
    receipt = analyze(
        modules, host_events, expected_windows=expected_windows,
        reverse_chunk=args.p32_reverse_chunk == "1",
    )
    receipt["xplane"] = str(xplane.relative_to(args.run_root))
    receipt["reverse_wall"] = parse_reverse_wall(
        args.run_root / "train/raw.log"
    )
  except Exception as exc:  # The receipt must explain every fail-closed exit.
    receipt = {
        "schema": SCHEMA,
        "status": "FAIL",
        "geometry": GEOMETRY,
        "reasons": [f"{type(exc).__name__}:{exc}"],
    }
  args.output.write_text(
      json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )

  marker = (
      "V1_GSM8K_P74_GAP_CENSUS_GREEN"
      if receipt["status"] == "PASS"
      else "V1_GSM8K_P74_GAP_CENSUS_RED"
  )
  gap = receipt.get("gap", {})
  reverse = receipt.get("reverse_wall", {}).get("captured_update") or {}
  print(
      f"{marker} windows={gap.get('windows')} "
      f"mean_ms={gap.get('mean_ms')} max_ms={gap.get('max_ms')} "
      f"identity_windows={receipt.get('identity_windows')} "
      f"windows_with_any_victim={receipt.get('windows_with_any_victim')} "
      f"reverse_seconds={reverse.get('seconds')} "
      f"reverse_group_mean_ms="
      f"{reverse.get('mean_seconds', 0.0) * 1000.0 if reverse else None} "
      f"reasons={receipt.get('reasons')}"
  )
  return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
