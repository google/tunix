#!/usr/bin/env python3
"""Arm-aware device XLA-module census for the GSM8K XProf pair."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import glob
import re


DECODE = re.compile(r"run_model|jit_sample|compute_and_gather")
# Boundary pullbacks emitted by every zero-HP backward, whatever rung of
# the P71 scan/block ladder is selected.  The layer-depth family is mode
# dependent and is checked by validate_backward_family instead.
ZERO_REQUIRED = (
    re.compile(r"zt_tr_dp_parallel_bwd_head"),
    re.compile(r"zt_tr_dp_parallel_bwd_norm"),
    re.compile(r"zt_tr_dp_parallel_bwd_embed"),
    re.compile(r"zt_tr_dp_parallel_bwd_adjoint"),
)
ZERO_TAIL_EXACT = {
    "jit__precomputed_gradient_scaled_step": 16,
    "jit__precomputed_gradient_commit": 1,
}
# The two registered carrier geometries share the global work (64
# trajectories), so one committed update owns 64 / dp_size gradient groups:
# the scaled-step count, the native train_step count, and the per-group
# backward executions all derive from it.
GEOMETRIES = {
    "dp4-tp1": {"groups": 16},
    "dp2-tp2": {"groups": 32},
}
DEFAULT_GEOMETRY = "dp4-tp1"
# The two mutually exclusive layer-depth backward families.  `_base` has
# already stripped the XLA fingerprint suffix, so these anchor the whole
# module name and expose the family index.
BWD_LAYER = re.compile(r"^jit_zt_tr_dp_parallel_bwd_layer_(\d+)$")
BWD_BLOCK = re.compile(r"^jit_zt_tr_dp_parallel_bwd_block_(\d+)$")
FWD_TAPE_SCAN = "jit_zt_tr_fwd_scan"
# Frozen carrier geometry, measured on the update:2->3 capture window of
# /mnt/disks/tunix-data/gsm8k-onehost-xprof/v1_zero-hp_p70bc_final_20260827
# and .../v1_zero-hp_p71e2p_gate3_20260827: Qwen3-1.7B has 28 decoder
# layers and the grouped reverse pass runs one backward program per
# (gradient group, chunk) = 16 x 2 = 32 executions per captured update.
# The capture window is one update wide, so these counts do not move with
# the P33 run stage.
ZERO_LAYER_COUNT = 28
ZERO_BACKWARD_EXECS = 32
# Chunk policy: one backward program runs per (gradient group, chunk) where
# a group's chunk count is ceil(max-per-rank real tokens / local M=256).  On
# the frozen dp4 captures every group measured exactly 2 chunks (all rows
# 349..460 real tokens), pinning 16 x 2 = 32.  A dp2 group draws only 2
# rows from the same length distribution, so its chunk count is data
# dependent (1..5 at the 1024+256 envelope); until a first dp2 capture pins
# the constant, the census requires internal consistency (one shared count
# across the block family / a layer total divisible by the layer count /
# a matching forward-tape count) inside the derived floor and ceiling
# instead of guessing one number.
PER_GROUP_CHUNK_BOUNDS = (1, 5)


def expected_backward_execs(geometry: str) -> int | None:
  """Returns the pinned per-update executions, or None when unpinned."""
  if geometry not in GEOMETRIES:
    raise ValueError(f"unknown geometry: {geometry!r}")
  if geometry == DEFAULT_GEOMETRY:
    return ZERO_BACKWARD_EXECS
  return None


def backward_exec_bounds(geometry: str) -> tuple[int, int]:
  """Returns the admissible per-update execution range for one geometry."""
  groups = GEOMETRIES[geometry]["groups"]
  low, high = PER_GROUP_CHUNK_BOUNDS
  return (groups * low, groups * high)
# Mirrors _P71_BWD_BLOCK_LAYERS in tunix/rl/canonical_qwen3_adapter.py.
# That constant has a documented fallback ladder 7 -> 4 -> 2, so the
# census exposes it as a flag rather than welding 4 blocks into the code.
P71_BWD_BLOCK_LAYERS = 7
P71_SCAN_MODES = ("off", "fwd", "bwd")
EXPECTED_TPU_PLANES = {
    f"/device:TPU:{index}" for index in range(8)
}


def _base(name: str) -> str:
  return re.sub(r"[(.].*", "", name)


def p71_scan_mode(value: str | None) -> str:
  """Normalizes a CANON_P71_SCAN value with the adapter's exact enum.

  Mirrors `_p71_scan_mode` in tunix/rl/canonical_qwen3_adapter.py.  The
  adapter returns '' for the historical per-layer rung; the census spells
  it 'off' so every census line names the inventory it asserted.
  """
  if value is None or value in ("", "0", "off"):
    return "off"
  if value in ("fwd", "bwd"):
    return value
  if value == "full":
    raise ValueError(
        "CANON_P71_SCAN='full' is reserved for the unimplemented E3 "
        "segment; only off/fwd/bwd exist in E2'"
    )
  raise ValueError(
      "CANON_P71_SCAN must be unset/0/off/fwd/bwd (full reserved), "
      f"got {value!r}"
  )


def expected_block_indices(
    layer_count: int = ZERO_LAYER_COUNT,
    block_layers: int = P71_BWD_BLOCK_LAYERS,
) -> tuple[int, ...]:
  """Returns the block indices `_p71_bwd_block_spans` would produce."""
  if layer_count < 1:
    raise ValueError(f"layer count must be >= 1, got {layer_count}")
  if block_layers < 1:
    raise ValueError(f"block size must be >= 1, got {block_layers}")
  return tuple(range(len(range(0, layer_count, block_layers))))


def _family(
    names: Mapping[str, int], pattern: re.Pattern[str]
) -> dict[int, int]:
  found = {}
  for name, count in names.items():
    match = pattern.match(name)
    if match is not None:
      found[int(match.group(1))] = count
  return found


def _indices(values) -> str:
  return ",".join(f"{index:02d}" for index in sorted(values)) or "none"


def validate_backward_family(
    p71_scan: str,
    names: Mapping[str, int],
    *,
    layer_count: int = ZERO_LAYER_COUNT,
    block_layers: int = P71_BWD_BLOCK_LAYERS,
    geometry: str = DEFAULT_GEOMETRY,
) -> list[str]:
  """Returns fail-closed reasons for one mode's backward program inventory.

  Exactly one layer-depth family may exist.  On the dp4 geometry its size
  is pinned to the measured `ZERO_BACKWARD_EXECS`; on a geometry without a
  frozen capture the per-update execution count E must instead be
  internally consistent (one shared block count / a layer total divisible
  by the layer count / a matching forward-tape count) inside
  `backward_exec_bounds`:

    off / fwd  per-layer pullbacks, `layer_count * E` executions in
               total.  XLA folds the identically shaped layers onto a
               single module name, so the census pins the family's total
               execution count and index range rather than its name
               cardinality.
    bwd        the P71-E2' unrolled blocks: exactly
               `ceil(layer_count / block_layers)` programs with
               contiguous indices from 00, each executed exactly E times.

  Seeing the other family is a red in both directions, so a silent
  fallback (or a silent fusion) cannot pass as the requested mode.
  """
  if p71_scan not in P71_SCAN_MODES:
    raise ValueError(f"unknown P71 scan mode: {p71_scan!r}")
  pinned_execs = expected_backward_execs(geometry)
  low, high = backward_exec_bounds(geometry)
  layers = _family(names, BWD_LAYER)
  blocks = _family(names, BWD_BLOCK)
  reasons = []
  observed_execs = None
  if p71_scan == "bwd":
    expected = expected_block_indices(layer_count, block_layers)
    if layers:
      reasons.append(f"p71=bwd_unexpected_bwd_layer={_indices(layers)}")
    if not blocks:
      reasons.append("missing_backward=zt_tr_dp_parallel_bwd_block")
    elif tuple(sorted(blocks)) != expected:
      reasons.append(
          f"bwd_block_indices={_indices(blocks)} "
          f"expected={_indices(expected)}"
      )
    if pinned_execs is not None:
      reasons.extend(
          f"bwd_block_{index:02d}={blocks[index]}!={pinned_execs}"
          for index in sorted(set(blocks) & set(expected))
          if blocks[index] != pinned_execs
      )
    elif blocks:
      counts = sorted(set(blocks.values()))
      if len(counts) != 1:
        reasons.append(
            "bwd_block_execs_inconsistent="
            + ",".join(str(count) for count in counts)
        )
      else:
        observed_execs = counts[0]
        if not low <= observed_execs <= high:
          reasons.append(
              f"bwd_block_execs={observed_execs} outside={low}..{high}"
          )
  else:
    if blocks:
      reasons.append(
          f"p71={p71_scan}_unexpected_bwd_block={_indices(blocks)}"
      )
    if not layers:
      reasons.append("missing_backward=zt_tr_dp_parallel_bwd_layer")
    else:
      overflow = [index for index in layers if index >= layer_count]
      if overflow:
        reasons.append(
            f"bwd_layer_index_overflow={_indices(overflow)} "
            f"layers={layer_count}"
        )
      total = sum(layers.values())
      if pinned_execs is not None:
        expected_total = layer_count * pinned_execs
        if total != expected_total:
          reasons.append(f"bwd_layer_execs={total}!={expected_total}")
      elif total % layer_count:
        reasons.append(
            f"bwd_layer_execs={total} not_a_multiple_of={layer_count}"
        )
      else:
        observed_execs = total // layer_count
        if not low <= observed_execs <= high:
          reasons.append(
              f"bwd_layer_execs_per_layer={observed_execs} "
              f"outside={low}..{high}"
          )
  # E1 rebuilds the per-chunk forward tape as one scanned program and E2'
  # inherits it, so its absence means the requested rung did not run.
  if p71_scan in ("fwd", "bwd"):
    if FWD_TAPE_SCAN not in names:
      reasons.append(f"missing_forward_tape_scan={FWD_TAPE_SCAN}")
    else:
      tape_expected = (
          pinned_execs if pinned_execs is not None else observed_execs
      )
      if (
          tape_expected is not None
          and names[FWD_TAPE_SCAN] != tape_expected
      ):
        reasons.append(
            f"forward_tape_scan={names[FWD_TAPE_SCAN]}!={tape_expected}"
        )
  return reasons


def backward_family_text(names: Mapping[str, int]) -> str:
  """Returns the observed layer-depth family without naming modules.

  The arm classifier reds a native run whose census text mentions any
  `zt_tr_dp_parallel_bwd_` module, so this stays a compact shape.
  """
  layers = _family(names, BWD_LAYER)
  blocks = _family(names, BWD_BLOCK)
  parts = []
  if layers:
    parts.append(f"layer[{_indices(layers)}]x{sum(layers.values())}")
  if blocks:
    parts.append(
        f"block[{_indices(blocks)}]x"
        + ",".join(str(blocks[index]) for index in sorted(blocks))
    )
  return "+".join(parts) if parts else "absent"


def validate_module_counts(
    arm: str,
    names: Mapping[str, int],
    *,
    p71_scan: str = "off",
    layer_count: int = ZERO_LAYER_COUNT,
    block_layers: int = P71_BWD_BLOCK_LAYERS,
    geometry: str = DEFAULT_GEOMETRY,
) -> list[str]:
  """Returns fail-closed reasons for one TensorCore TPU plane."""
  if geometry not in GEOMETRIES:
    raise ValueError(f"unknown geometry: {geometry!r}")
  groups = GEOMETRIES[geometry]["groups"]
  reasons = []
  if any(DECODE.search(name) for name in names):
    reasons.append("decode=present")
  if arm == "native":
    # The stock learner runs one monolithic train_step per gradient
    # accumulation microbatch: 64 global trajectories / dp_size = groups.
    count = names.get("jit__train_step", 0)
    if count != groups:
      reasons.append(f"jit__train_step={count}!={groups}")
    return reasons
  if arm != "zero-hp":
    raise ValueError(f"unknown arm: {arm}")
  reasons.extend(
      f"missing_backward={pattern.pattern}"
      for pattern in ZERO_REQUIRED
      if not any(pattern.search(name) for name in names)
  )
  reasons.extend(
      validate_backward_family(
          p71_scan,
          names,
          layer_count=layer_count,
          block_layers=block_layers,
          geometry=geometry,
      )
  )
  tail_exact = {
      "jit__precomputed_gradient_scaled_step": groups,
      "jit__precomputed_gradient_commit": 1,
  }
  reasons.extend(
      f"{name}={names.get(name, 0)}!={expected}"
      for name, expected in tail_exact.items()
      if names.get(name, 0) != expected
  )
  return reasons


def validate_plane_names(names: list[str]) -> list[str]:
  actual = set(names)
  if len(names) == 8 and actual == EXPECTED_TPU_PLANES:
    return []
  return [
      "TensorCore_planes="
      + ",".join(sorted(actual))
      + " expected="
      + ",".join(sorted(EXPECTED_TPU_PLANES))
  ]


def main() -> None:
  from xprof.profile_data import ProfileData

  parser = argparse.ArgumentParser()
  parser.add_argument("--arm", choices=("native", "zero-hp"), required=True)
  parser.add_argument("--run-root", required=True)
  parser.add_argument(
      "--geometry",
      choices=tuple(sorted(GEOMETRIES)),
      default=DEFAULT_GEOMETRY,
      help="registered carrier geometry the run was launched with",
  )
  parser.add_argument(
      "--p71-scan",
      default="",
      help=(
          "the CANON_P71_SCAN value the run was launched with; selects "
          "the expected backward program inventory (native ignores it)"
      ),
  )
  parser.add_argument("--p71-layers", type=int, default=ZERO_LAYER_COUNT)
  parser.add_argument(
      "--p71-block-layers", type=int, default=P71_BWD_BLOCK_LAYERS
  )
  args = parser.parse_args()
  p71_scan = p71_scan_mode(args.p71_scan)
  # Reject an impossible geometry before reading a 1 GB xplane.
  expected_block_indices(args.p71_layers, args.p71_block_layers)

  files = glob.glob(
      f"{args.run_root.rstrip('/')}/train/xprof/plugins/profile/*/*.xplane.pb"
  )
  if len(files) != 1:
    raise SystemExit(f"expected exactly one xplane, found {len(files)}")
  profile = ProfileData.from_file(files[0])

  checked = 0
  plane_names = []
  failures: list[str] = []
  detail: tuple[str, dict[str, int]] | None = None
  for plane in profile.planes:
    if "TPU" not in plane.name or "SparseCore" in plane.name:
      continue
    plane_names.append(plane.name)
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
    groups = GEOMETRIES[args.geometry]["groups"]
    reasons = validate_module_counts(
        args.arm,
        names,
        p71_scan=p71_scan,
        layer_count=args.p71_layers,
        block_layers=args.p71_block_layers,
        geometry=args.geometry,
    )
    if args.arm == "native":
      # The stock learner runs one monolithic forward/backward train_step
      # per trajectory group.  It does not expose pullback-named modules,
      # so a P55 segmented-backward census is the wrong contract.
      count = names.get("jit__train_step", 0)
      summary = f"train_step={count}/{groups}"
    else:
      backward_missing = any(
          reason.startswith("missing_backward=") for reason in reasons
      )
      tail_exact = {
          "jit__precomputed_gradient_scaled_step": groups,
          "jit__precomputed_gradient_commit": 1,
      }
      summary = (
          f"p71_scan={p71_scan} required="
          + ("MISSING" if backward_missing else "present")
          + " backward=" + backward_family_text(names)
          + " optimizer_tail="
          + ",".join(
              f"{name.removeprefix('jit__precomputed_gradient_')}="
              f"{names.get(name, 0)}/{expected}"
              for name, expected in tail_exact.items()
          )
      )
    print(
        f"plane={plane.name} distinct_modules={len(names)} span={span:.3f}s "
        f"arm={args.arm} {summary} "
        f"decode={'PRESENT' if has_decode else 'absent'}"
    )
    checked += 1
    if reasons or span <= 0.0:
      failures.append(
          f"{plane.name}(reasons={reasons},span={span:.3f})"
      )
    if detail is None:
      detail = (plane.name, names)

  if checked == 0:
    raise SystemExit("no TensorCore TPU planes in xplane")
  failures.extend(validate_plane_names(plane_names))
  if detail is not None:
    print(f"module detail for {detail[0]}:")
    for name, count in sorted(detail[1].items(), key=lambda item: -item[1]):
      print(f"  {count:7d}  {name}")
  if failures:
    print("V1_GSM8K_XPROF_CENSUS_RED " + ";".join(failures))
    raise SystemExit(1)
  tail = (
      f" p71_scan={p71_scan}"
      f" optimizer_tail=scaled_step:{GEOMETRIES[args.geometry]['groups']}"
      ",commit:1"
      if args.arm == "zero-hp" else ""
  )
  print(
      f"V1_GSM8K_XPROF_CENSUS_GREEN arm={args.arm} "
      f"planes={checked} backward=present decode=absent{tail}"
  )


if __name__ == "__main__":
  main()
