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

"""Import the pinned original splits without downloading or executing source code."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

SOURCE_COMMIT = "12ceb7563aa50439687f21ba5da60ad74a75eb89"
SOURCE_URL = "https://github.com/beita6969/R2Orch"
SPLITS = {
    "train": ("train/train_pool.jsonl", 947),
    "dev": ("heldout/aime.jsonl", 30),
    "test": ("test/iid/aime.jsonl", 30),
    "ood": ("test/ood/math_hard.jsonl", 128),
}


def prepare(source, output):
  """Preserve source IDs, prompts, targets and curriculum metadata exactly."""
  source, output = Path(source), Path(output)
  revision = subprocess.check_output(
      ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
  ).strip()
  if revision != SOURCE_COMMIT:
    raise ValueError(
        f"Expected upstream revision {SOURCE_COMMIT}, got {revision}."
    )
  datasets, manifest = {}, {
      "source": SOURCE_URL,
      "commit": revision,
      "splits": {},
  }
  for name, (relative, count) in SPLITS.items():
    relative = "data/r2orch/" + relative
    # Read committed bytes: local edits must not silently change provenance.
    raw = subprocess.check_output(
        ["git", "-C", str(source), "show", f"{revision}:{relative}"]
    )
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if name == "train":
      records = [r for r in records if r["dataset"] == "aime"]
    converted = []
    for row in records:
      answers = row.get("target_answers", [row.get("answer")])
      if len(answers) != 1 or answers[0] is None:
        raise ValueError(
            "The mathematical recipe requires one reference answer."
        )
      converted.append({
          "task_id": row.get("id", row.get("task_id")),
          "problem": row["prompt"],
          "answer": str(answers[0]),
          "benchmark": "math_hard" if name == "ood" else "aime",
          "source": row,
      })
    if len(converted) != count:
      raise ValueError(
          f"Unexpected {name} cardinality: {len(converted)} != {count}."
      )
    datasets[name] = converted
    manifest["splits"][name] = {
        "count": count,
        "source_path": relative,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
    }
  seen_ids, seen_questions = set(), set()
  for name, rows in datasets.items():
    ids = [row["task_id"] for row in rows]
    questions = [" ".join(row["problem"].split()) for row in rows]
    if len(set(ids)) != len(ids) or seen_ids.intersection(ids):
      raise ValueError("Duplicate task IDs across source splits.")
    if len(set(questions)) != len(questions) or seen_questions.intersection(
        questions
    ):
      raise ValueError(
          "Exact normalized question overlap across source splits."
      )
    seen_ids.update(ids)
    seen_questions.update(questions)
  output.mkdir(parents=True, exist_ok=True)
  if any(output.iterdir()):
    raise ValueError("Use an empty output directory.")
  for name, rows in datasets.items():
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    (output / f"{name}.jsonl").write_text(data)
    manifest["splits"][name]["sha256"] = hashlib.sha256(
        data.encode()
    ).hexdigest()
  (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
  return manifest


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", required=True)
  parser.add_argument("--output", required=True)
  args = parser.parse_args()
  print(json.dumps(prepare(args.source, args.output), indent=2))


if __name__ == "__main__":
  main()
