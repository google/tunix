#!/usr/bin/env python3
"""Counts program dispatches per captured update, by program family.

Reads one run's XPlane (one captured update, possibly a truncated prefix),
counts every "XLA Modules" event on each TensorCore plane, groups the
program names into families, and normalises by the update's chunk passes
(sum over the captured update's groups of ceil(max n_real / bucket), read
from raw.log ``forward_group_issued`` lines).  Pure host-side analysis; no
TPU involved.

usage: census_dispatch_count.py --run-root RUN --geometry GEO [--json OUT]
"""

from __future__ import annotations

import argparse
from collections import Counter
import glob
import json
from pathlib import Path
import re

GEOMETRIES = {
    "dp4-tp1": {"groups": 16},
    "dp2-tp2": {"groups": 32},
    "dp1-tp4": {"groups": 64},
    "dp2-tp2-long": {"groups": 8},
    "dp2-tp2-long8k": {"groups": 8},
    "dp2-tp2-p45": {"groups": 8},
}
_ISSUED_RE = re.compile(r"forward_group_issued .*?n_real=\(([0-9, ]+)\)")

# Program-name stem -> family.  Order matters: first match wins.
FAMILIES = (
    ("bwd_chunk", re.compile(r"zt_tr_bwd_chunk")),
    ("rows", re.compile(r"zt_tr_fwd_logprob|zt_tr_bwd_logprob")),
    ("group_glue", re.compile(r"jit_group_pack|jit_group_lengths|jit_fwd_glue_zeros|jit_bwd_flat_cotangents|jit_pairs_equal|jit_fused_chunk_metadata")),
    ("fwd_layer", re.compile(r"zt_tr_fwd_layer\b")),
    ("fwd_scan", re.compile(r"zt_tr_fwd_scan|fwd_tape_scan")),
    ("bwd_layer", re.compile(r"zt_tr_dp_parallel_bwd_layer|bwd_layer_block|zt_tr_dp_parallel_bwd_block")),
    ("bwd_head", re.compile(r"zt_tr_dp_parallel_bwd_head")),
    ("bwd_norm", re.compile(r"zt_tr_dp_parallel_bwd_norm")),
    ("grad_add", re.compile(r"zt_tr_grad_tree_add|grad_tree_start|grad_tree")),
    ("finalize", re.compile(r"finalize_staged|scaled_step|precomputed_gradient")),
    ("entry_cache", re.compile(r"rebuild|entry_cache|zero_trees|fresh_caches")),
    ("fwd_embed", re.compile(r"embed")),
    ("fwd_norm", re.compile(r"zt_tr_fwd_norm|norm_forward|final_norm")),
    ("fwd_head", re.compile(r"zt_tr_fwd_head|head_forward|lm_head")),
    ("processed_rows", re.compile(r"processed_rows")),
    ("glue_eager", re.compile(r"dynamic_update_slice|dynamic_slice|convert_element_type|^jit_slice|^jit_stack|concatenate|jit_where|_p74_identity|gather_rows|jit_clip|jit_add|jit_sub|jit_mul|jit_true_divide|jit_reshape|jit_broadcast|jit_zeros|jit_ones|jit_arange|jit_remainder|jit_floor_divide|jit__lambda|jit_fn|jit_asarray|copy|jit_squeeze|jit_expand_dims|jit_transpose|jit_select_n|jit_lt|jit_ge|jit_eq|jit_max|jit_min|jit_sum|jit_cumsum|jit_take|jit_scatter|jit_pad")),
    ("sampler", re.compile(r"vllm|prefill|decode|sample|paged")),
)


def _base(name: str) -> str:
  return re.sub(r"[(.].*", "", name).strip()


def family_of(stem: str) -> str:
  for family, pattern in FAMILIES:
    if pattern.search(stem):
      return family
  return "other"


def chunk_passes_from_raw_log(raw_log: Path, *, groups: int, bucket: int) -> tuple[int, list[int]]:
  text = raw_log.read_text(encoding="utf-8", errors="replace")
  issued = [max(int(v) for v in m.group(1).split(",")) for m in _ISSUED_RE.finditer(text)]
  if len(issued) < groups:
    raise ValueError(f"forward_group_issued lines={len(issued)} < groups={groups}")
  per_group = [(longest + bucket - 1) // bucket for longest in issued[-groups:]]
  return sum(per_group), per_group


def main() -> None:
  from xprof.profile_data import ProfileData

  parser = argparse.ArgumentParser()
  parser.add_argument("--run-root", required=True)
  parser.add_argument("--geometry", choices=tuple(sorted(GEOMETRIES)), required=True)
  parser.add_argument("--sequence-bucket", type=int, default=256)
  parser.add_argument("--top", type=int, default=40)
  parser.add_argument("--json", default="")
  args = parser.parse_args()

  root = args.run_root.rstrip("/")
  files = glob.glob(f"{root}/train/xprof/plugins/profile/*/*.xplane.pb")
  if len(files) != 1:
    raise SystemExit(f"expected exactly one xplane, found {len(files)}")
  groups = GEOMETRIES[args.geometry]["groups"]
  passes, per_group = chunk_passes_from_raw_log(
      Path(root) / "train" / "raw.log", groups=groups, bucket=args.sequence_bucket)

  profile = ProfileData.from_file(files[0])
  planes = {}
  for plane in profile.planes:
    if "TPU" not in plane.name or "SparseCore" in plane.name:
      continue
    names: Counter[str] = Counter()
    busy_ns = 0
    tmin = tmax = None
    for line in plane.lines:
      if line.name != "XLA Modules":
        continue
      for event in line.events:
        names[_base(event.name)] += 1
        busy_ns += event.duration_ns
        start = event.start_ns
        end = start + event.duration_ns
        tmin = start if tmin is None else min(tmin, start)
        tmax = end if tmax is None else max(tmax, end)
    span_ns = 0 if tmin is None else tmax - tmin
    planes[plane.name] = {
        "total": sum(names.values()),
        "span_s": span_ns / 1e9,
        "busy_s": busy_ns / 1e9,
        "idle_s": (span_ns - busy_ns) / 1e9,
        "names": names,
    }
  if not planes:
    raise SystemExit("no TensorCore TPU planes in xplane")

  first_name = sorted(planes)[0]
  first = planes[first_name]
  families: Counter[str] = Counter()
  for stem, count in first["names"].items():
    families[family_of(stem)] += count
  total = first["total"]
  print(f"run={root}")
  print(f"geometry={args.geometry} groups={groups} chunk_passes={passes} per_group={per_group}")
  print("planes: " + ", ".join(
      f"{name}:total={p['total']} span={p['span_s']:.2f}s busy={p['busy_s']:.2f}s idle={p['idle_s']:.2f}s"
      for name, p in sorted(planes.items())))
  print(f"[{first_name}] total_executions={total} per_chunk_pass={total / passes:.2f} "
        f"idle_per_pass_ms={first['idle_s'] * 1e3 / passes:.1f} "
        f"mean_gap_per_dispatch_ms={first['idle_s'] * 1e3 / max(total, 1):.3f}")
  print("families (count, per chunk pass):")
  for family, count in families.most_common():
    print(f"  {family:16s} {count:8d} {count / passes:9.2f}")
  print(f"top {args.top} program stems:")
  for stem, count in first["names"].most_common(args.top):
    print(f"  {count:8d}  {family_of(stem):14s} {stem}")
  if args.json:
    Path(args.json).write_text(json.dumps({
        "run": root, "geometry": args.geometry, "groups": groups,
        "chunk_passes": passes, "per_group_chunks": per_group,
        "planes": {name: {k: v for k, v in p.items() if k != "names"} for name, p in planes.items()},
        "families": dict(families),
        "stems": dict(first["names"].most_common()),
    }, indent=1, sort_keys=True))


if __name__ == "__main__":
  main()
