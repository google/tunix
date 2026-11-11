# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare the reference training corpus with a disjoint development holdout.

This intentionally does not use MATH-500/AIME test questions for development.
Deduplication and split membership are deterministic and recorded in a manifest.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

DATASET = "aaabiao/dapo_filter"
REVISION = "0b10fea35812902fe4f6985f2064424ec460f9c0"
_SUFFIX = re.compile(
    r"(\r?\n|\\n)*\s*Let'?s\s+think\s+step\s+by\s+step\s+and\s+output\s+the\s+final\s+answer\s+within\s*\\boxed\{\}\.\s*$",
    re.IGNORECASE,
)


def convert_and_split(rows, *, dev_size=256, seed=42, stats=None):
  unique = {}
  conflicting = set()
  raw_count = 0
  for row in rows:
    raw_count += 1
    messages = row["prompt"]
    if len(messages) != 1 or not isinstance(messages[0]["content"], str):
      raise ValueError("Expected one text prompt per reference training row.")
    problem = _SUFFIX.sub("", messages[0]["content"]).strip()
    answer = str(row["reward_model"]["ground_truth"])
    if not problem or not answer:
      raise ValueError("Empty problem or ground truth.")
    digest = hashlib.sha256(problem.encode()).hexdigest()
    item = {"task_id": f"dapo/{digest}", "problem": problem, "answer": answer}
    if digest in unique and unique[digest]["answer"] != answer:
      conflicting.add(digest)
    unique[digest] = item
  for digest in conflicting:
    del unique[digest]
  if stats is not None:
    stats.update({
        "raw_rows": raw_count,
        "retained_unique_problems": len(unique),
        "conflicting_problem_count": len(conflicting),
        "excluded_conflicting_problem_sha256": sorted(conflicting),
    })
  ordered = sorted(
      unique.values(),
      key=lambda x: hashlib.sha256(
          f'{seed}/{x["task_id"]}'.encode()
      ).hexdigest(),
  )
  if not 0 < dev_size < len(ordered):
    raise ValueError("dev_size must leave nonempty train and dev splits.")
  return ordered[dev_size:], ordered[:dev_size]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--dev-size", type=int, default=256)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()
  from datasets import load_dataset  # pylint: disable=g-import-not-at-top

  args.output.mkdir(parents=True, exist_ok=True)
  if any(args.output.iterdir()):
    raise ValueError("Use an empty data output directory.")
  rows = load_dataset(DATASET, revision=REVISION, split="train")
  stats = {}
  train, dev = convert_and_split(
      rows, dev_size=args.dev_size, seed=args.seed, stats=stats
  )
  manifest = {
      "dataset": DATASET,
      "revision": REVISION,
      "seed": args.seed,
      **stats,
      "splits": {},
  }
  for name, items in (("train", train), ("dev", dev)):
    content = "".join(json.dumps(row) + "\n" for row in items)
    (args.output / f"{name}.jsonl").write_text(content)
    manifest["splits"][name] = {
        "count": len(items),
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
    }
  (args.output / "manifest.json").write_text(
      json.dumps(manifest, indent=2) + "\n"
  )
  print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
  main()
