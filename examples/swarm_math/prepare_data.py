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

"""Prepare the reference training corpus and the reference math test suite.

Training questions keep a disjoint development holdout. Benchmark questions are
written as separate evaluation files and never enter the train/dev split.
Deduplication, split membership and source revisions are recorded in a manifest.
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
# Upstream examples/data_preprocess/drmas_math.py test sources, in upstream
# test.parquet order. Revisions are the Hub commits holding that unchanged data.
TEST_SETS = (
    (
        "math500",
        "HuggingFaceH4/MATH-500",
        "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
        "test",
    ),
    (
        "aime24",
        "HuggingFaceH4/aime_2024",
        "2fe88a2f1091d5048c0f36abc874fb997b3dd99a",
        "train",
    ),
    (
        "aime25",
        "MathArena/aime_2025",
        "c94da77eb22bbd6439e62a323bec18493a421302",
        "train",
    ),
    (
        "olympiadbench",
        "realtreetune/olympiadbench",
        "ba3660e5eb252709639dfff5b6f16d2c482fce7b",
        "test",
    ),
    (
        "amc23",
        "knoveleng/AMC-23",
        "55c4747921a0971a18408979dcc2bc85af98f136",
        "train",
    ),
    (
        "minerva",
        "zwhe99/simplerl-minerva-math",
        "647bd705ccd4bde066950b8c4afcaae5cfe73620",
        "test",
    ),
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


def build_test_suites(sources, training_problems):
  """Upstream test.parquet, its 110-question training-time sample and overlap.

  Like upstream `make_map_fn_test`, problem text is unchanged and answers are
  stringified. Upstream does not decontaminate; exact matches with the training
  corpus are listed so reports can also exclude them.

  Args:
    sources: Mapping from upstream test-set name to its rows.
    training_problems: Problem texts of the reference training corpus.

  Returns:
    `(suites, overlap)`: the `test` and `test_sampled` task lists, and sorted
    task IDs whose problem text also occurs in the training corpus.
  """
  converted = {
      name: [
          {
              "task_id": f"{name}/{index}",
              "problem": row["problem"],
              "answer": str(row["answer"]),
          }
          for index, row in enumerate(sources[name])
      ]
      for name, *_ in TEST_SETS
  }
  suites = {
      "test": [row for name, *_ in TEST_SETS for row in converted[name]],
      "test_sampled": (
          converted["math500"][:50] + converted["aime24"] + converted["aime25"]
      ),
  }
  overlap = sorted(
      row["task_id"]
      for row in suites["test"]
      if row["problem"].strip() in training_problems
  )
  return suites, overlap


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
  suites, overlap = build_test_suites(
      {
          name: load_dataset(dataset, revision=revision, split=split)
          for name, dataset, revision, split in TEST_SETS
      },
      {row["problem"] for row in train + dev},
  )
  manifest = {
      "dataset": DATASET,
      "revision": REVISION,
      "seed": args.seed,
      **stats,
      "test_sources": [
          {"name": name, "dataset": dataset, "revision": revision, "split": s}
          for name, dataset, revision, s in TEST_SETS
      ],
      "test_training_overlap": overlap,
      "splits": {},
  }
  for name, items in {"train": train, "dev": dev, **suites}.items():
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
