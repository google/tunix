#!/usr/bin/env python3
"""Stage one self-contained M15 E0 KV discriminator round."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


class E0RoundError(RuntimeError):
  """Raised when one E0 round is incomplete or crosses round boundaries."""


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise E0RoundError(message)


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
  _require(path.is_file(), f"{label} is absent: {path}")
  records = []
  for line_number, line in enumerate(
      path.read_text(encoding="utf-8").splitlines(), start=1
  ):
    if not line.strip():
      continue
    try:
      record = json.loads(line)
    except json.JSONDecodeError as exc:
      raise E0RoundError(
          f"invalid {label} JSON at line {line_number}"
      ) from exc
    _require(isinstance(record, dict), f"{label} row is not an object")
    records.append(record)
  _require(records, f"{label} is empty")
  return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
  path.write_text(
      "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
      encoding="utf-8",
  )


def _boundary(record: dict[str, Any], name: str) -> tuple[int, int]:
  boundary = record.get("boundaries", {}).get(name, {})
  differing_bytes = boundary.get("differing_bytes")
  differing_elements = boundary.get("differing_elements")
  _require(
      isinstance(differing_bytes, int)
      and not isinstance(differing_bytes, bool)
      and differing_bytes >= 0
      and isinstance(differing_elements, int)
      and not isinstance(differing_elements, bool)
      and differing_elements >= 0,
      f"{name} boundary is invalid",
  )
  return differing_bytes, differing_elements


def _copy_observer_records(
    source: Path, output: Path, round_index: int
) -> tuple[list[dict[str, Any]], list[int]]:
  selected = []
  for json_path in sorted(source.glob("p38_kv_observer_*.json")):
    record = json.loads(json_path.read_text(encoding="utf-8"))
    if int(record.get("diagnostic_round", -1)) != round_index:
      continue
    _require(
        record.get("schema") == "p38-live-kv-prefix-table-v1",
        f"observer schema drifted: {json_path.name}",
    )
    index = int(record.get("record_index", -1))
    arm = record.get("arm")
    _require(
        index >= 0 and arm in ("A", "B"),
        f"observer identity drifted: {json_path.name}",
    )
    expected = f"p38_kv_observer_{index:04d}_{arm.lower()}.json"
    _require(json_path.name == expected, f"observer filename drifted: {json_path.name}")
    npz_path = json_path.with_suffix(".npz")
    _require(npz_path.is_file(), f"observer NPZ is absent: {npz_path.name}")
    _require(
        record.get("npz_sha256") == _sha256(npz_path),
        f"observer NPZ SHA failed: {npz_path.name}",
    )
    shutil.copyfile(json_path, output / json_path.name)
    shutil.copyfile(npz_path, output / npz_path.name)
    selected.append(record)

  _require(len(selected) == 16, f"round {round_index} requires 16 KV records")
  indices = sorted(int(record["record_index"]) for record in selected)
  _require(
      indices == list(range(indices[0], indices[0] + 16)),
      f"round {round_index} KV record indices are not one contiguous window",
  )
  a_records = [record for record in selected if record["arm"] == "A"]
  b_records = [record for record in selected if record["arm"] == "B"]
  _require(
      len(a_records) == len(b_records) == 8,
      f"round {round_index} requires eight A and eight B records",
  )
  a_indices = {int(record["record_index"]) for record in a_records}
  b_sources = {int(record.get("source_a_record_index", -1)) for record in b_records}
  _require(
      a_indices == b_sources,
      f"round {round_index} A/B source indices are not bijective",
  )
  return selected, indices


def stage(
    *,
    directory: Path,
    alignment_report: Path,
    capsule_base: Path,
    replay_ledger: Path,
    classifier: Path,
    output: Path,
    round_index: int,
    arm: str,
    expected_source: str,
    runtime_source: str,
) -> dict[str, Any]:
  _require(arm in ("off", "on"), "E0 arm must be off or on")
  _require(0 <= round_index < 3, "E0 diagnostic round must be in [0, 3)")
  _require(
      len(expected_source) == 40
      and all(value in "0123456789abcdef" for value in expected_source),
      "expected source is not one full lowercase SHA",
  )
  _require(expected_source == runtime_source, "runtime source does not match render")
  _require(not output.exists(), f"E0 round output already exists: {output}")
  _require(directory.is_dir(), f"KV observer directory is absent: {directory}")
  _require(classifier.is_file(), f"KV classifier is absent: {classifier}")
  output.mkdir(parents=True, mode=0o700)

  alignments = [
      record for record in _read_jsonl(alignment_report, "alignment report")
      if int(record.get("diagnostic_round", -1)) == round_index
  ]
  _require(
      len(alignments) == 1,
      f"round {round_index} requires exactly one alignment record",
  )
  alignment = alignments[0]
  a_b_bytes, a_b_elements = _boundary(
      alignment, "S_decode_vs_S_prefill"
  )
  b_c_bytes, b_c_elements = _boundary(
      alignment, "S_prefill_vs_T_old"
  )
  _require(
      b_c_bytes == 0 and b_c_elements == 0,
      f"round {round_index} B-C is red",
  )
  if arm == "off":
    _require(
        a_b_bytes == 0 and a_b_elements == 0,
        f"round {round_index} APC-off control A-B is red",
    )
  elif (a_b_bytes == 0) != (a_b_elements == 0):
    raise E0RoundError(f"round {round_index} A-B counters disagree")

  selected, indices = _copy_observer_records(directory, output, round_index)
  _write_jsonl(output / "pre-alignment.jsonl", [alignment])

  replay_records = [
      record for record in _read_jsonl(replay_ledger, "M15 replay ledger")
      if int(record.get("diagnostic_round", -1)) == round_index
  ]
  _require(replay_records, f"round {round_index} has no replay-ledger rows")
  _require(
      all(record.get("schema") == "m15-apc-serving-envelope-v1"
          for record in replay_records),
      f"round {round_index} replay-ledger schema drifted",
  )
  _write_jsonl(output / "m15-replay-envelope.jsonl", replay_records)

  capsule = Path(
      f"{capsule_base.with_suffix('')}.round-{round_index:06d}.npz"
  )
  capsule_present = capsule.is_file() and capsule.stat().st_size > 0
  red = a_b_bytes > 0
  _require(
      capsule_present == red,
      f"round {round_index} mismatch capsule presence disagrees with A-B",
  )
  if capsule_present:
    shutil.copyfile(capsule, output / "mismatch-capsule.npz")

  shutil.copyfile(classifier, output / "classify_p38_kv_observer.py")
  runtime = {
      "path": classifier.name,
      "runtime_source_commit": runtime_source,
      "schema": "m15-e0-kv-classifier-runtime-v2",
      "sha256": _sha256(classifier),
      "status": "source-bound",
  }
  (output / "CLASSIFIER_RUNTIME.json").write_text(
      json.dumps(runtime, sort_keys=True, indent=2) + "\n", encoding="utf-8"
  )
  record = {
      "a_b_differing_bytes": a_b_bytes,
      "a_b_differing_elements": a_b_elements,
      "arm": arm,
      "b_c_differing_bytes": b_c_bytes,
      "b_c_differing_elements": b_c_elements,
      "capsule_present": capsule_present,
      "diagnostic_round": round_index,
      "expected_source_commit": expected_source,
      "kv_pairs": 8,
      "kv_record_index_end": indices[-1],
      "kv_record_index_start": indices[0],
      "kv_records": len(selected),
      "replay_records": len(replay_records),
      "runtime_source_commit": runtime_source,
      "schema": "m15-e0-kv-round-input-v1",
      "status": "STAGED_FOR_CLASSIFIER_CHECKPOINT",
  }
  (output / "ROUND_INPUT.json").write_text(
      json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8"
  )
  return record


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--directory", required=True, type=Path)
  parser.add_argument("--alignment-report", required=True, type=Path)
  parser.add_argument("--capsule-base", required=True, type=Path)
  parser.add_argument("--replay-ledger", required=True, type=Path)
  parser.add_argument("--classifier", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--round", required=True, type=int)
  parser.add_argument("--arm", required=True, choices=("off", "on"))
  parser.add_argument("--expected-source", required=True)
  parser.add_argument("--runtime-source", required=True)
  args = parser.parse_args()
  result = stage(
      directory=args.directory,
      alignment_report=args.alignment_report,
      capsule_base=args.capsule_base,
      replay_ledger=args.replay_ledger,
      classifier=args.classifier,
      output=args.output,
      round_index=args.round,
      arm=args.arm,
      expected_source=args.expected_source,
      runtime_source=args.runtime_source,
  )
  print(
      "[M15.E0.KV.ROUND] STAGED "
      f"arm={result['arm']} round={result['diagnostic_round']} "
      f"records={result['kv_records']} pairs={result['kv_pairs']} "
      f"a_b={result['a_b_differing_bytes']} b_c=0"
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
