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
) -> list[str]:
  """Returns fail-closed reasons for one mode's backward program inventory.

  Exactly one layer-depth family may exist, and its size is pinned:

    off / fwd  per-layer pullbacks, `layer_count * ZERO_BACKWARD_EXECS`
               executions in total.  XLA folds the identically shaped
               layers onto a single module name, so the census pins the
               family's total execution count and index range rather than
               its name cardinality.
    bwd        the P71-E2' unrolled blocks: exactly
               `ceil(layer_count / block_layers)` programs with
               contiguous indices from 00, each executed exactly
               `ZERO_BACKWARD_EXECS` times.

  Seeing the other family is a red in both directions, so a silent
  fallback (or a silent fusion) cannot pass as the requested mode.
  """
  if p71_scan not in P71_SCAN_MODES:
    raise ValueError(f"unknown P71 scan mode: {p71_scan!r}")
  layers = _family(names, BWD_LAYER)
  blocks = _family(names, BWD_BLOCK)
  reasons = []
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
    reasons.extend(
        f"bwd_block_{index:02d}={blocks[index]}!={ZERO_BACKWARD_EXECS}"
        for index in sorted(set(blocks) & set(expected))
        if blocks[index] != ZERO_BACKWARD_EXECS
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
      expected_total = layer_count * ZERO_BACKWARD_EXECS
      if total != expected_total:
        reasons.append(f"bwd_layer_execs={total}!={expected_total}")
  # E1 rebuilds the per-chunk forward tape as one scanned program and E2'
  # inherits it, so its absence means the requested rung did not run.
  if p71_scan in ("fwd", "bwd") and FWD_TAPE_SCAN not in names:
    reasons.append(f"missing_forward_tape_scan={FWD_TAPE_SCAN}")
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
) -> list[str]:
  """Returns fail-closed reasons for one TensorCore TPU plane."""
  reasons = []
  if any(DECODE.search(name) for name in names):
    reasons.append("decode=present")
  if arm == "native":
    count = names.get("jit__train_step", 0)
    if count != 16:
      reasons.append(f"jit__train_step={count}!=16")
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
      )
  )
  reasons.extend(
      f"{name}={names.get(name, 0)}!={expected}"
      for name, expected in ZERO_TAIL_EXACT.items()
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
    reasons = validate_module_counts(
        args.arm,
        names,
        p71_scan=p71_scan,
        layer_count=args.p71_layers,
        block_layers=args.p71_block_layers,
    )
    if args.arm == "native":
      # The stock learner runs one monolithic forward/backward train_step for
      # each of the 16 trajectory groups.  It does not expose pullback-named
      # modules, so a P55 segmented-backward census is the wrong contract.
      count = names.get("jit__train_step", 0)
      summary = f"train_step={count}/16"
    else:
      backward_missing = any(
          reason.startswith("missing_backward=") for reason in reasons
      )
      summary = (
          f"p71_scan={p71_scan} required="
          + ("MISSING" if backward_missing else "present")
          + " backward=" + backward_family_text(names)
          + " optimizer_tail="
          + ",".join(
              f"{name.removeprefix('jit__precomputed_gradient_')}="
              f"{names.get(name, 0)}/{expected}"
              for name, expected in ZERO_TAIL_EXACT.items()
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
      " optimizer_tail=scaled_step:16,commit:1"
      if args.arm == "zero-hp" else ""
  )
  print(
      f"V1_GSM8K_XPROF_CENSUS_GREEN arm={args.arm} "
      f"planes={checked} backward=present decode=absent{tail}"
  )


if __name__ == "__main__":
  main()
