#!/usr/bin/env python3
"""Builds Harbor JSONL datasets for `obiwan96/endless-terminals` (`arXiv:2601.16443`).

Produces:
  1. `endless_terminals_all_2492.jsonl` — all 2,492 tasks, round-robin interleaved
     across the 5 o3 difficulty tiers (`L1_o3_16of16` .. `L5_o3_1to4of16`) so any
     prefix slice is balanced across difficulty levels.
  2. `endless_terminals_val_100.jsonl` — the official 100-task held-out dev/val
     split matching `kanishkg/endless-terminals` `train/prepare_endless.py`
     (`sorted(task_dirs)` shuffled with `random.seed(42)`, last 100 tasks).
  3. `endless_terminals_train_2392.jsonl` — the official 2,392-task train split
     matching `train/prepare_endless.py` (`random.seed(42)`, first 2,392 tasks).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import pathlib
import random
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from examples.deepswe import harbor_utils  # pylint: disable=g-import-not-at-top


TIER_ORDER = (
    "L1_o3_16of16",
    "L2_o3_13to15of16",
    "L3_o3_9to12of16",
    "L4_o3_5to8of16",
    "L5_o3_1to4of16",
)


def _write_jsonl(path: pathlib.Path, entries: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", encoding="utf-8") as f:
    for entry in entries:
      f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _interleave_by_difficulty(
    entries: list[dict[str, Any]], seed: int = 42
) -> list[dict[str, Any]]:
  """Round-robin interleaves entries across difficulty tiers."""
  rng = random.Random(seed)
  buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
  for entry in entries:
    buckets[str(entry.get("difficulty") or "unknown")].append(entry)
  ordered_tiers = [t for t in TIER_ORDER if t in buckets] + [
      t for t in sorted(buckets.keys()) if t not in TIER_ORDER
  ]
  for tier in ordered_tiers:
    buckets[tier].sort(key=lambda e: str(e["instance_id"]))
    rng.shuffle(buckets[tier])

  interleaved: list[dict[str, Any]] = []
  max_len = max((len(buckets[t]) for t in ordered_tiers), default=0)
  for idx in range(max_len):
    for tier in ordered_tiers:
      if idx < len(buckets[tier]):
        interleaved.append(buckets[tier][idx])
  return interleaved


def build_endless_terminals_datasets(
    source_dir: pathlib.Path,
    output_dir: pathlib.Path,
    seed: int = 42,
    val_size: int = 100,
) -> dict[str, Any]:
  """Loads all Endless Terminals tasks and writes the 3 JSONL splits."""
  task_dirs = sorted(
      d
      for d in source_dir.iterdir()
      if d.is_dir() and d.name.startswith("task_") and harbor_utils.is_harbor_task_dir(d)
  )
  if not task_dirs:
    raise ValueError(f"No Harbor task_* directories found under {source_dir}")

  entries_by_id: dict[str, dict[str, Any]] = {}
  for task_dir in task_dirs:
    entry = harbor_utils.load_harbor_task_dir(task_dir)
    entries_by_id[entry["instance_id"]] = entry

  # Official split from kanishkg/endless-terminals/train/prepare_endless.py:
  # sort directory names, shuffle with random.Random(42), take last 100 for val.
  sorted_ids = sorted(entries_by_id.keys())
  rng = random.Random(seed)
  shuffled_ids = list(sorted_ids)
  rng.shuffle(shuffled_ids)
  train_ids = shuffled_ids[:-val_size]
  val_ids = shuffled_ids[-val_size:]

  train_entries = [entries_by_id[i] for i in train_ids]
  val_entries = [entries_by_id[i] for i in val_ids]
  all_interleaved = _interleave_by_difficulty(
      [entries_by_id[i] for i in sorted_ids], seed=seed
  )

  all_path = output_dir / f"endless_terminals_all_{len(all_interleaved)}.jsonl"
  val_path = output_dir / f"endless_terminals_val_{len(val_entries)}.jsonl"
  train_path = output_dir / f"endless_terminals_train_{len(train_entries)}.jsonl"

  _write_jsonl(all_path, all_interleaved)
  _write_jsonl(val_path, val_entries)
  _write_jsonl(train_path, train_entries)

  manifest = {
      "source_dir": str(source_dir),
      "total_tasks": len(all_interleaved),
      "seed": seed,
      "files": {
          "all": str(all_path),
          "val": str(val_path),
          "train": str(train_path),
      },
      "difficulty_counts": {
          "all": dict(Counter(e["difficulty"] for e in all_interleaved)),
          "val": dict(Counter(e["difficulty"] for e in val_entries)),
          "train": dict(Counter(e["difficulty"] for e in train_entries)),
      },
  }
  manifest_path = output_dir / "manifest.json"
  manifest_path.write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  return manifest


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--source_dir",
      type=pathlib.Path,
      default=pathlib.Path("/usr/local/google/home/wuhao/datasets/endless-terminals"),
  )
  parser.add_argument(
      "--output_dir",
      type=pathlib.Path,
      default=pathlib.Path(
          "/usr/local/google/home/wuhao/datasets/endless_terminals_jsonl"
      ),
  )
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--val_size", type=int, default=100)
  args = parser.parse_args()

  manifest = build_endless_terminals_datasets(
      source_dir=args.source_dir,
      output_dir=args.output_dir,
      seed=args.seed,
      val_size=args.val_size,
  )
  print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
  main()
