#!/usr/bin/env python3
"""Build a signed B2xG2 replay by duplicating one strict-exact real group."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys


SOURCES = (
    {
        "name": "scrapy-a",
        "root": Path(
            "/mnt/disks/tunix-data/deepswe-onehost-xprof/"
            "p58_zero-hp_p58s22lr3_20260829t2256z"
        ),
        "journal_sha256": (
            "bffb324f097f959ee16593bc741b8c83e940cc556665c1d051d3f480a8657fc0"
        ),
        "manifest_sha256": (
            "96f1ff1e9db641e7d0735c593176d4dbc9ab8799cfe1a7a010bcf8634502201e"
        ),
        "run_id": "p58-onehost-xprof-zero-hp-p58s22lr3_20260829t2256z",
        "image": (
            "namanjain12/scrapy_final:"
            "439a3e59b8e858441f8d97dbc32f398db392330d"
        ),
        "prompt_length": 1745,
        "prompt_width": 1792,
    },
    {
        "name": "scrapy-b",
        "root": Path(
            "/mnt/disks/tunix-data/deepswe-onehost-xprof/"
            "p58_zero-hp_p58s22lr3_20260829t2256z"
        ),
        "journal_sha256": (
            "bffb324f097f959ee16593bc741b8c83e940cc556665c1d051d3f480a8657fc0"
        ),
        "manifest_sha256": (
            "96f1ff1e9db641e7d0735c593176d4dbc9ab8799cfe1a7a010bcf8634502201e"
        ),
        "run_id": "p58-onehost-xprof-zero-hp-p58s22lr3_20260829t2256z",
        "image": (
            "namanjain12/scrapy_final:"
            "439a3e59b8e858441f8d97dbc32f398db392330d"
        ),
        "prompt_length": 1745,
        "prompt_width": 1792,
    },
)
SOURCE_COMMIT = "16c224aa80eb6b3a544be19f693c0542ab4b0dcb"
SAMPLING = {"source": "explicit-cli", "temperature": 1.0, "top_k": 0, "top_p": 1.0}


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_source(spec: dict[str, object]) -> list[dict[str, object]]:
  root = spec["root"]
  assert isinstance(root, Path)
  journal = root / "batch-000000.trajectories.jsonl.gz"
  manifest_path = root / "run_manifest.json"
  if _sha256(journal) != spec["journal_sha256"]:
    raise ValueError(f"{spec['name']} journal SHA-256 changed")
  if _sha256(manifest_path) != spec["manifest_sha256"]:
    raise ValueError(f"{spec['name']} manifest SHA-256 changed")
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  expected_manifest = {
      "source_commit": SOURCE_COMMIT,
      "run_id": spec["run_id"],
      "model_id": "Qwen/Qwen3-4B-Instruct-2507",
      "generations": 2,
      "global_trajectories": 2,
      "sampling_contract": SAMPLING,
  }
  changed = {
      key: manifest.get(key)
      for key, expected in expected_manifest.items()
      if manifest.get(key) != expected
  }
  if changed:
    raise ValueError(f"{spec['name']} manifest identity changed: {changed}")
  with gzip.open(journal, "rt", encoding="utf-8") as source:
    rows = [json.loads(line) for line in source if line.strip()]
  if len(rows) != 2:
    raise ValueError(f"{spec['name']} must contain exactly two rows")
  for pair_index, (row, reward) in enumerate(zip(rows, (1.0, 0.0))):
    trajectory = row.get("trajectory", {})
    identity = row.get("task_identity", {})
    exact = {
        "group_id": "0",
        "pair_index": pair_index,
        "status": "SUCCEEDED",
        "complete": True,
        "compact_filtered": False,
        "raw_final_reward": reward,
        "training_reward": reward,
    }
    drift = {
        key: row.get(key)
        for key, expected in exact.items()
        if row.get(key) != expected
    }
    if drift:
      raise ValueError(f"{spec['name']} row {pair_index} changed: {drift}")
    if identity.get("docker_image") != spec["image"]:
      raise ValueError(f"{spec['name']} task image changed")
    if (
        trajectory.get("prompt_length") != spec["prompt_length"]
        or len(trajectory.get("prompt_tokens", [])) != spec["prompt_width"]
    ):
      raise ValueError(f"{spec['name']} prompt geometry changed")
  return rows


def build(output_dir: Path) -> tuple[Path, Path]:
  if not output_dir.is_absolute():
    raise ValueError("output directory must be absolute")
  if output_dir.exists() and any(output_dir.iterdir()):
    raise ValueError(f"output directory must be absent or empty: {output_dir}")
  output_dir.mkdir(parents=True, exist_ok=True)
  merged_rows = []
  source_receipts = []
  for group_id, spec in enumerate(SOURCES):
    rows = _load_source(spec)
    source_receipts.append({
        "name": spec["name"],
        "run_id": spec["run_id"],
        "journal": str(spec["root"] / "batch-000000.trajectories.jsonl.gz"),
        "journal_sha256": spec["journal_sha256"],
        "manifest": str(spec["root"] / "run_manifest.json"),
        "manifest_sha256": spec["manifest_sha256"],
        "task_image": spec["image"],
    })
    for pair_index, row in enumerate(rows):
      normalized = dict(row)
      normalized["group_id"] = str(group_id)
      normalized["pair_index"] = pair_index
      normalized["replay_source"] = {
          "name": spec["name"],
          "run_id": spec["run_id"],
          "source_group_id": 0,
          "source_pair_index": pair_index,
      }
      trajectory = dict(normalized["trajectory"])
      trajectory["group_id"] = group_id
      normalized["trajectory"] = trajectory
      merged_rows.append(normalized)

  manifest = {
      "schema": "canon.p58.b2g2-replay-source.v1",
      "source_commit": SOURCE_COMMIT,
      "model_id": "Qwen/Qwen3-4B-Instruct-2507",
      "prompt_groups": 2,
      "generations": 2,
      "global_trajectories": 4,
      "prompt_identity": "same-strict-exact-real-prompt-repeated-twice",
      "sampling_contract": SAMPLING,
      "sources": source_receipts,
  }
  manifest_path = output_dir / "run_manifest.json"
  manifest_path.write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  journal_path = output_dir / "batch-000000.trajectories.jsonl.gz"
  with journal_path.open("wb") as raw:
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
      with io.TextIOWrapper(compressed, encoding="utf-8") as output:
        for row in merged_rows:
          output.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
          output.write("\n")
  return manifest_path, journal_path


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--output-dir", type=Path, required=True)
  args = parser.parse_args()
  try:
    manifest, journal = build(args.output_dir)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    print(f"P58.23 B2xG2 replay preparation failed: {exc}", file=sys.stderr)
    return 1
  print(
      "[P58.23.REPLAY_SOURCE] PASS "
      f"groups=2 generations=2 trajectories=4 manifest_sha256={_sha256(manifest)} "
      f"journal_sha256={_sha256(journal)} output={args.output_dir}"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
