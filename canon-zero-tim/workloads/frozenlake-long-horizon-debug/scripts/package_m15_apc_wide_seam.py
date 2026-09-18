#!/usr/bin/env python3
"""Package the minimal independently replayable M15 seam evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
from typing import Iterable


class M15WideSeamPackageError(RuntimeError):
  pass


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise M15WideSeamPackageError(message)


def _sha256(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def _read(path: Path, label: str) -> bytes:
  _require(path.is_file(), f"{label} is absent: {path}")
  data = path.read_bytes()
  _require(data, f"{label} is empty: {path}")
  return data


def _record_members(
    directory: Path, prefix: str, indices: Iterable[int]
) -> dict[str, bytes]:
  members = {}
  for index in sorted(set(int(value) for value in indices)):
    _require(index >= 0, f"negative {prefix} record index")
    for suffix in ("json", "npz"):
      name = f"{prefix}_{index:06d}.{suffix}"
      members[f"records/{name}"] = _read(directory / name, name)
  return members


def _first_record_per_arm(directory: Path, prefix: str) -> list[int]:
  selected: dict[str, int] = {}
  for path in sorted(directory.glob(f"{prefix}_*.json")):
    record = json.loads(path.read_text(encoding="utf-8"))
    arm = str(record.get("arm"))
    index = int(record.get("record_index", -1))
    if arm in ("A", "B") and arm not in selected:
      selected[arm] = index
  _require(set(selected) == {"A", "B"}, f"{prefix} lacks an A/B record pair")
  return list(selected.values())


def _tar_bytes(members: dict[str, bytes]) -> bytes:
  output = io.BytesIO()
  with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
    for name in sorted(members):
      info = tarfile.TarInfo(name)
      info.size = len(members[name])
      info.mtime = 0
      info.uid = info.gid = 0
      info.uname = info.gname = ""
      info.mode = 0o644
      archive.addfile(info, io.BytesIO(members[name]))
  return output.getvalue()


def package(
    *,
    directory: Path,
    classification_path: Path,
    alignment_report: Path,
    capsules: list[Path],
    replay_ledger: Path | None,
    output: Path,
) -> dict:
  classification_bytes = _read(classification_path, "seam classification")
  classification = json.loads(classification_bytes)
  _require(
      classification.get("schema") == "m15-apc-wide-seam-classification-v1"
      and classification.get("status") == "PASS",
      "M15 seam classification is not a PASS receipt",
  )
  mode = str(classification.get("observer_mode"))
  _require(mode in ("layer", "full"), "M15 observer mode drifted")
  members = {
      "classification.json": classification_bytes,
      "pre-alignment.jsonl": _read(alignment_report, "pre-alignment report"),
  }
  anchors = classification.get("anchors", [])
  seam_indices = []
  tail_indices = []
  if anchors:
    for anchor in anchors:
      for arm in ("a", "b"):
        seam_indices.append(int(anchor[arm]["record_index"]))
      for key in ("a_observation_candidates", "b_observation_candidates"):
        for candidate in anchor.get(key, ()):
          seam_indices.append(int(candidate["record_index"]))
      for key in ("a_tail_record_index", "b_tail_record_index"):
        value = anchor.get(key)
        if value is not None:
          tail_indices.append(int(value))
  else:
    seam_indices = _first_record_per_arm(directory, "p38_seam")
    if mode == "layer":
      tail_indices = _first_record_per_arm(directory, "p38_tail")
  members.update(_record_members(directory, "p38_seam", seam_indices))
  if tail_indices:
    members.update(_record_members(directory, "p38_tail", tail_indices))
  for offset, capsule in enumerate(capsules):
    members[f"capsules/capsule-{offset:02d}.npz"] = _read(
        capsule, "mismatch capsule"
    )
  if replay_ledger is not None and replay_ledger.is_file():
    members["m15-replay-envelope.jsonl"] = _read(
        replay_ledger, "M15 replay ledger"
    )
  receipt = {
      "schema": "m15-apc-wide-seam-bundle-v1",
      "status": "PASS",
      "classification": classification["classification"],
      "observer_mode": mode,
      "arm": classification["arm"],
      "selected_seam_records": sorted(set(seam_indices)),
      "selected_tail_records": sorted(set(tail_indices)),
      "capsules": len(capsules),
      "claim_ceiling": (
          "This bundle preserves the classifier-selected standard-path rows; "
          "it does not claim coverage of continue-decode-only red actions."
      ),
  }
  members["RECEIPT.json"] = (
      json.dumps(receipt, sort_keys=True, indent=2) + "\n"
  ).encode("utf-8")
  manifest_names = sorted(members)
  members["SHA256SUMS"] = "".join(
      f"{_sha256(members[name])}  {name}\n" for name in manifest_names
  ).encode("ascii")
  payload = _tar_bytes(members)
  _require(not output.exists(), f"refusing to overwrite M15 seam bundle: {output}")
  output.parent.mkdir(parents=True, exist_ok=True)
  partial = output.with_name(output.name + ".partial")
  partial.write_bytes(payload)
  partial.replace(output)
  return {
      **receipt,
      "output": str(output),
      "bytes": len(payload),
      "sha256": _sha256(payload),
      "logical_files": len(members),
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--directory", type=Path, required=True)
  parser.add_argument("--classification", type=Path, required=True)
  parser.add_argument("--alignment-report", type=Path, required=True)
  parser.add_argument("--capsule", type=Path, action="append", default=[])
  parser.add_argument("--replay-ledger", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  print(json.dumps(package(
      directory=args.directory,
      classification_path=args.classification,
      alignment_report=args.alignment_report,
      capsules=args.capsule,
      replay_ledger=args.replay_ledger,
      output=args.output,
  ), sort_keys=True))


if __name__ == "__main__":
  main()
