#!/usr/bin/env python3
"""Arm-aware device XLA-module census for the GSM8K XProf pair."""

from __future__ import annotations

import argparse
import glob
import re
import sys

from xprof.profile_data import ProfileData


DECODE = re.compile(r"run_model|jit_sample|compute_and_gather")
ZERO_REQUIRED = (
    re.compile(r"zt_tr_dp_parallel_bwd_layer"),
    re.compile(r"zt_tr_dp_parallel_bwd_head"),
    re.compile(r"zt_tr_dp_parallel_bwd_norm"),
    re.compile(r"zt_tr_dp_parallel_bwd_embed"),
    re.compile(r"zt_tr_dp_parallel_bwd_adjoint"),
)


def _base(name: str) -> str:
  return re.sub(r"[(.].*", "", name)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--arm", choices=("native", "zero-hp"), required=True)
  parser.add_argument("--run-root", required=True)
  args = parser.parse_args()

  files = glob.glob(
      f"{args.run_root.rstrip('/')}/train/xprof/plugins/profile/*/*.xplane.pb"
  )
  if len(files) != 1:
    raise SystemExit(f"expected exactly one xplane, found {len(files)}")
  profile = ProfileData.from_file(files[0])

  checked = 0
  failures: list[str] = []
  detail: tuple[str, dict[str, int]] | None = None
  for plane in profile.planes:
    if "TPU" not in plane.name or "SparseCore" in plane.name:
      continue
    names: dict[str, int] = {}
    tmin = None
    tmax = None
    for line in plane.lines:
      if line.name != "XLA Modules":
        continue
      for event in line.events:
        name = _base(event.name)
        names[name] = names.get(name, 0) + 1
        start = event.start_ns
        end = start + event.duration_ns
        tmin = start if tmin is None else min(tmin, start)
        tmax = end if tmax is None else max(tmax, end)
    span = 0.0 if tmin is None or tmax is None else (tmax - tmin) / 1e9
    has_decode = any(DECODE.search(name) for name in names)
    if args.arm == "native":
      # The stock learner runs one monolithic forward/backward train_step for
      # each of the 16 trajectory groups.  It does not expose pullback-named
      # modules, so a P55 segmented-backward census is the wrong contract.
      count = names.get("jit__train_step", 0)
      missing = [] if count == 16 else [f"jit__train_step={count}!=16"]
      summary = f"train_step={count}/16"
    else:
      missing = [
          pattern.pattern
          for pattern in ZERO_REQUIRED
          if not any(pattern.search(name) for name in names)
      ]
      summary = "required=" + ("present" if not missing else "MISSING")
    print(
        f"plane={plane.name} distinct_modules={len(names)} span={span:.3f}s "
        f"arm={args.arm} {summary} "
        f"decode={'PRESENT' if has_decode else 'absent'}"
    )
    checked += 1
    if missing or has_decode or span <= 0.0:
      failures.append(
          f"{plane.name}(missing={missing},decode={has_decode},span={span:.3f})"
      )
    if detail is None:
      detail = (plane.name, names)

  if checked == 0:
    raise SystemExit("no TensorCore TPU planes in xplane")
  if detail is not None:
    print(f"module detail for {detail[0]}:")
    for name, count in sorted(detail[1].items(), key=lambda item: -item[1]):
      print(f"  {count:7d}  {name}")
  if failures:
    print("V1_GSM8K_XPROF_CENSUS_RED " + ";".join(failures))
    raise SystemExit(1)
  print(
      f"V1_GSM8K_XPROF_CENSUS_GREEN arm={args.arm} "
      f"planes={checked} backward=present decode=absent"
  )


if __name__ == "__main__":
  main()
