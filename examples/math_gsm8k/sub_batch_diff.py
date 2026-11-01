# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Weight-level diff of two actor checkpoints.

Compares two actor checkpoints item by item and leaf by leaf
  - bit-identical leaf count (np.array_equal),
  - global / per-leaf max and mean absolute difference,
  - relative scale (max |a-b| / max |a|),
  - the top-N most-different leaves.

Usage (on the VM, venv active):

  python3 examples/math_gsm8k/qwen3_grpo_sub_batch_ckpt_diff.py \
      artifacts/qwen3_grpo_gsm8k_vtc/checkpoints/validate-control-<t>/actor/6 \
      artifacts/qwen3_grpo_gsm8k_vtc/checkpoints/validate-preempt-<t>/actor/6

Interpretation guide: Small uniform differences (max abs diff ~1e-6..1e-3
relative to weight scale) mean the resume introduced a bounded numeric 
deviation -- the expected outcome under the design's statistical-equivalence
contract; large or structured differences (whole layers diverging) would
indicate a real training-path difference worth investigating.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import pathlib
import sys

import jax
import numpy as np
from orbax.checkpoint import v1 as ocp


def _flat(tree) -> dict[str, np.ndarray]:
  return {
      jax.tree_util.keystr(path): np.asarray(leaf)
      for path, leaf in jax.tree_util.tree_leaves_with_path(tree)
  }


def _diff_item(name: str, a: dict, b: dict, top: int) -> list[str]:
  lines = [f"## {name}", ""]
  only_a = sorted(set(a) - set(b))
  only_b = sorted(set(b) - set(a))
  if only_a or only_b:
    lines.append(
        f"STRUCTURE MISMATCH: {len(only_a)} leaves only in control,"
        f" {len(only_b)} only in preempt -- comparison limited to the"
        " intersection."
    )
    lines.append("")
  rows = []
  bit_identical = 0
  total_elems = 0
  sum_abs = 0.0
  global_max = 0.0
  for key in sorted(set(a) & set(b)):
    x, y = a[key], b[key]
    if x.shape != y.shape or x.dtype != y.dtype:
      rows.append((float("inf"), key, f"shape/dtype mismatch: {x.shape}"
                   f"/{x.dtype} vs {y.shape}/{y.dtype}", None, None))
      continue
    if not np.issubdtype(x.dtype, np.number):
      continue
    xf = x.astype(np.float64)
    yf = y.astype(np.float64)
    d = np.abs(xf - yf)
    mx = float(d.max()) if d.size else 0.0
    if mx == 0.0 and np.array_equal(x, y):
      bit_identical += 1
    total_elems += d.size
    sum_abs += float(d.sum())
    global_max = max(global_max, mx)
    scale = float(np.abs(xf).max()) if xf.size else 0.0
    rel = mx / scale if scale > 0 else (0.0 if mx == 0.0 else float("inf"))
    n_mismatch = int((d > 0).sum())
    rows.append((mx, key, None, rel, n_mismatch))

  n_leaves = len([r for r in rows if r[2] is None])
  lines.append(
      f"- leaves compared: {n_leaves}; bit-identical:"
      f" {bit_identical}/{n_leaves}"
  )
  lines.append(
      f"- global max |diff| = {global_max:.3e}; mean |diff| over all"
      f" {total_elems} elements = {sum_abs / max(1, total_elems):.3e}"
  )
  lines.append("")
  lines.append(f"Top {top} most-different leaves:")
  lines.append("")
  lines.append("| max |diff| | rel (vs max|control|) | mismatched elems | leaf |")
  lines.append("|---|---|---|---|")
  for mx, key, err, rel, n_mismatch in sorted(
      rows, key=lambda r: r[0], reverse=True
  )[:top]:
    if err is not None:
      lines.append(f"| - | - | - | {key}: {err} |")
    else:
      lines.append(f"| {mx:.3e} | {rel:.3e} | {n_mismatch} | `{key}` |")
  lines.append("")
  return lines


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("control_dir", help="control arm actor step dir")
  parser.add_argument("preempt_dir", help="preempt arm actor step dir")
  parser.add_argument("--top", type=int, default=10)
  parser.add_argument(
      "--out", default="ckpt_diff_report.md",
      help="Where to write the markdown report.",
  )
  args = parser.parse_args()

  control_dir = pathlib.Path(args.control_dir).resolve()
  preempt_dir = pathlib.Path(args.preempt_dir).resolve()

  lines = ["# Checkpoint weight diff: control vs preempted+resumed", ""]
  lines.append(f"- control: `{control_dir}`")
  lines.append(f"- preempt: `{preempt_dir}`")
  lines.append("")

  def _load_item_as_numpy(step_dir: pathlib.Path, item: str):
    """Restores one item as plain host numpy arrays."""
    meta = ocp.checkpointables_metadata(step_dir).metadata[item]
    abstract = jax.tree_util.tree_map(
        lambda m: np.empty(m.shape, dtype=m.dtype), meta
    )
    return ocp.load_checkpointables(step_dir, {item: abstract})[item]

  items = sorted(
      set(ocp.checkpointables_metadata(control_dir).metadata)
      & set(ocp.checkpointables_metadata(preempt_dir).metadata)
  )

  for item in items:
    print(f"[ckpt-diff] loading + diffing {item}...", flush=True)
    a = _flat(_load_item_as_numpy(control_dir, item))
    b = _flat(_load_item_as_numpy(preempt_dir, item))
    lines.extend(_diff_item(item, a, b, args.top))
    del a, b

  report = "\n".join(lines) + "\n"
  print("\n" + report)
  pathlib.Path(args.out).write_text(report)
  print(f"[ckpt-diff] report written to {args.out}", flush=True)
  return 0


if __name__ == "__main__":
  sys.exit(main())
