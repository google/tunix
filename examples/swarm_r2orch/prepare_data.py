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

"""Prepare the 4-domain self-contained R2Orch training, heldout, and test splits.

Fetches or reads pinned upstream JSONL manifests from `beita6969/R2Orch` across
the four zero-external-server reasoning and coding domains (`aime`, `hotpotqa`,
`medqa`, `mbppplus`) plus two out-of-distribution generalization suites
(`math_hard`, `musique`), verifies SHA-256 digests, checks train/eval prompt
disjointness, and writes uniform task JSONL files with a provenance manifest.
"""

import argparse
import collections
import hashlib
import json
from pathlib import Path
import urllib.request

UPSTREAM_REPO = "beita6969/R2Orch"
UPSTREAM_REVISION = "12ceb7563aa50439687f21ba5da60ad74a75eb89"
ENABLED_DOMAINS = ("aime", "hotpotqa", "medqa", "mbppplus")

SOURCE_FILES = {
    "train/train_pool.jsonl": (
        "4ffcfc9d7c5abcc54db89b1abcfa0746e29bc8fab6f3f77d75986c4ee1533a53"
    ),
    "heldout/aime.jsonl": (
        "65f17996a788f68f57b3688111cd50bb79e49e99611a8b5400279ab732b99030"
    ),
    "heldout/hotpotqa.jsonl": (
        "d871c6c751cfc39d6de8b074cddb278e07aaff3df4a8ffcc1df2bdc19c5beda7"
    ),
    "heldout/medqa.jsonl": (
        "bb6537346746c58bf6a0489542bfcefac60f4891687fb612b73d80d35acccfda"
    ),
    "heldout/mbppplus.jsonl": (
        "9042a4d3bb0ac40d07d51bb23dc54c3c51fae261638379edf5d461b11010c738"
    ),
    "test/iid/aime.jsonl": (
        "817b6a92a7ed21abee3c06d17849aefb89076c29d5519c09ef0d32ee4b9e9f77"
    ),
    "test/iid/hotpotqa.jsonl": (
        "ee3b1333780412577c02e923dc339ab2e3bcc778f8a92ff03283821158aece9e"
    ),
    "test/iid/medqa.jsonl": (
        "13e84c66d27f08ead5ec610bb538161b9ed61bd19ca3c470982f8e126cae81de"
    ),
    "test/iid/mbppplus.jsonl": (
        "4e6cb5cfd74d16eadbbe4263f5e2a0cfcdea3a0101d090870a2959cf0d064ad7"
    ),
    "test/ood/math_hard.jsonl": (
        "35dc246ccca837f6245e6fb5d0a930fb52933b8cb5e870ed0200caf52ace28b1"
    ),
    "test/ood/musique.jsonl": (
        "40dfa1a25f900144add66699de82369903e97bcfe02e3160b75fd13cdc1be4bb"
    ),
}
SEED_CARDS_REL = "configs/skills/seed_cards.json"
SEED_CARDS_SHA256 = (
    "7d6b3c324d75ab7a8d539f6991134a1e596ff9abbc2299f7fc3264139acfbf06"
)


def _read_verified_jsonl(rel_path: str, source_dir: Path | None) -> list[dict]:
  """Loads one upstream JSONL file and verifies its pinned SHA-256 digest."""
  expected_sha = SOURCE_FILES[rel_path]
  if source_dir is not None and (source_dir / rel_path).is_file():
    raw = (source_dir / rel_path).read_bytes()
  else:
    url = (
        f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/"
        f"{UPSTREAM_REVISION}/data/r2orch/{rel_path}"
    )
    with urllib.request.urlopen(url, timeout=120) as resp:
      raw = resp.read()
  found_sha = hashlib.sha256(raw).hexdigest()
  if found_sha != expected_sha:
    raise ValueError(
        f"SHA-256 mismatch for {rel_path}: {found_sha} != {expected_sha}"
    )
  return [
      json.loads(line)
      for line in raw.decode("utf-8").splitlines()
      if line.strip()
  ]


def normalize_iid_row(row: dict) -> dict:
  """Converts an upstream IID/train/heldout row into a uniform R2OrchTask dict."""
  dataset = str(row["dataset"])
  verifier = str(row["verifier"])
  metadata = row.get("metadata") or {}
  options = [str(x) for x in metadata.get("options", ())]
  answers = [str(x) for x in (row.get("target_answers") or ())]
  code_spec = dict(row["reference"]) if "reference" in row else {}
  curriculum = {
      key: row[key]
      for key in (
          "ads_sample_id",
          "cluster_id",
          "difficulty_score",
          "rank_in_cluster",
          "cluster_size",
          "task_type",
      )
      if key in row
  }
  curriculum["source_id"] = str(row.get("source_id") or row["id"])
  if "embedding" in row:
    curriculum["embedding"] = [round(float(x), 6) for x in row["embedding"]]
  return {
      "task_id": str(row["id"]),
      "dataset": dataset,
      "verifier": verifier,
      "prompt": str(row["prompt"]).strip(),
      "answers": answers,
      "options": options,
      "code_spec": code_spec,
      "curriculum": curriculum,
  }


def normalize_ood_row(row: dict) -> dict:
  """Converts an upstream OOD row (`math_hard` or `musique`) into a task dict."""
  kind = str(row["kind"])
  verifier = "math_symbolic" if kind == "math_hard" else "qa_exact_match"
  raw_answers = row.get("answers") or [row["answer"]]
  return {
      "task_id": str(row["task_id"]),
      "dataset": kind,
      "verifier": verifier,
      "prompt": str(row["prompt"]).strip(),
      "answers": [str(x) for x in raw_answers],
      "options": [],
      "code_spec": {},
  }


def build_splits(source_dir: Path | None = None) -> tuple[dict[str, list[dict]], list[str]]:
  """Builds `train`, `heldout`, `test_iid`, and `test_ood` task lists."""
  pool = _read_verified_jsonl("train/train_pool.jsonl", source_dir)
  train_rows = [
      normalize_iid_row(r) for r in pool if r.get("dataset") in ENABLED_DOMAINS
  ]
  heldout_rows = []
  test_iid_rows = []
  for domain in ENABLED_DOMAINS:
    for row in _read_verified_jsonl(f"heldout/{domain}.jsonl", source_dir):
      heldout_rows.append(normalize_iid_row(row))
    for row in _read_verified_jsonl(f"test/iid/{domain}.jsonl", source_dir):
      test_iid_rows.append(normalize_iid_row(row))
  test_ood_rows = [
      normalize_ood_row(row)
      for name in ("math_hard", "musique")
      for row in _read_verified_jsonl(f"test/ood/{name}.jsonl", source_dir)
  ]
  splits = {
      "train": train_rows,
      "heldout": heldout_rows,
      "test_iid": test_iid_rows,
      "test_ood": test_ood_rows,
  }
  train_prompts = {r["prompt"] for r in train_rows}
  overlap = sorted(
      r["task_id"]
      for key in ("heldout", "test_iid", "test_ood")
      for r in splits[key]
      if r["prompt"] in train_prompts
  )
  return splits, overlap


def _read_seed_cards(source_dir: Path | None) -> bytes:
  """Loads and SHA-256-verifies the 18 official R2Orch seed skill cards."""
  local_candidates = (
      [source_dir / SEED_CARDS_REL, source_dir.parent.parent / SEED_CARDS_REL]
      if source_dir is not None
      else []
  )
  raw = next((p.read_bytes() for p in local_candidates if p.is_file()), None)
  if raw is None:
    url = (
        f"https://raw.githubusercontent.com/{UPSTREAM_REPO}/"
        f"{UPSTREAM_REVISION}/{SEED_CARDS_REL}"
    )
    with urllib.request.urlopen(url, timeout=60) as resp:
      raw = resp.read()
  if hashlib.sha256(raw).hexdigest() != SEED_CARDS_SHA256:
    raise ValueError("SHA-256 mismatch for seed_cards.json.")
  return raw


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--source-dir", type=Path, default=None)
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)
  if any(args.output.iterdir()):
    raise ValueError("Use an empty data output directory.")
  splits, overlap = build_splits(args.source_dir)
  if overlap:
    raise ValueError(f"Detected train/eval prompt overlap: {overlap[:5]}")
  seed_cards = _read_seed_cards(args.source_dir)
  (args.output / "seed_cards.json").write_bytes(seed_cards)
  manifest = {
      "upstream_repo": UPSTREAM_REPO,
      "upstream_revision": UPSTREAM_REVISION,
      "enabled_domains": list(ENABLED_DOMAINS),
      "train_eval_prompt_overlap": overlap,
      "seed_cards_sha256": SEED_CARDS_SHA256,
      "splits": {},
  }
  for name, rows in splits.items():
    content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    (args.output / f"{name}.jsonl").write_text(content, encoding="utf-8")
    by_domain = dict(collections.Counter(r["dataset"] for r in rows))
    manifest["splits"][name] = {
        "count": len(rows),
        "by_dataset": by_domain,
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }
  (args.output / "manifest.json").write_text(
      json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
  )
  print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
  main()
