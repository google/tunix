#!/usr/bin/env python3
"""Classify a bounded DeepSWE decode-vs-prefill evidence package.

This tool never re-scores or fabricates log probabilities.  It joins the
bounded mismatch records in ``pre_alignment.jsonl`` back to the durable
trajectory journal and proves that reported token ids and S_decode values are
the values actually stored for the trajectory.  A hardware carrier can return
either an exact boundary or a reproduced finite RED; both are useful probe
outcomes.  Missing, malformed, non-finite, or unjoinable evidence fails closed.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable


_TRAJECTORY_GLOB = "batch-*.trajectories.jsonl.gz"
_VALID_SCHEMAS = {
    "canon.p58.deepswe.trajectory.v1",
    "canon.local.deepswe.trajectory.v1",
}


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
  with path.open(encoding="utf-8") as source:
    value = json.load(source)
  if not isinstance(value, dict):
    raise ValueError(f"{path.name} must contain one JSON object")
  return value


def _load_last_jsonl(path: Path) -> dict[str, Any]:
  records = []
  with path.open(encoding="utf-8") as source:
    for line_number, line in enumerate(source, 1):
      if not line.strip():
        continue
      value = json.loads(line)
      if not isinstance(value, dict):
        raise ValueError(f"{path.name}:{line_number} is not a JSON object")
      records.append(value)
  if not records:
    raise ValueError(f"{path.name} contains no records")
  return records[-1]


def _load_trajectories(path: Path) -> list[dict[str, Any]]:
  records = []
  with gzip.open(path, "rt", encoding="utf-8") as source:
    for line_number, line in enumerate(source, 1):
      value = json.loads(line)
      if not isinstance(value, dict):
        raise ValueError(f"{path.name}:{line_number} is not a JSON object")
      if value.get("schema") not in _VALID_SCHEMAS:
        raise ValueError(
            f"{path.name}:{line_number} has unsupported schema "
            f"{value.get('schema')!r}"
        )
      records.append(value)
  if not records:
    raise ValueError(f"{path.name} contains no trajectory records")
  return records


def _trajectory_arrays(
    record: dict[str, Any], row_index: int
) -> tuple[list[int], list[int], list[float] | None]:
  trajectory = record.get("trajectory")
  if not isinstance(trajectory, dict):
    raise ValueError(f"trajectory row {row_index} has no trajectory object")
  tokens = trajectory.get("conversation_tokens")
  masks = trajectory.get("conversation_masks")
  logps = trajectory.get("old_logprobs")
  if not isinstance(tokens, list) or not isinstance(masks, list):
    raise ValueError(f"trajectory row {row_index} token/mask arrays are absent")
  if len(tokens) != len(masks):
    raise ValueError(f"trajectory row {row_index} token/mask lengths differ")
  if logps is not None:
    if not isinstance(logps, list) or len(logps) != len(tokens):
      raise ValueError(f"trajectory row {row_index} logprob length differs")
    if not all(math.isfinite(float(value)) for value in logps):
      raise ValueError(f"trajectory row {row_index} has non-finite logprobs")
  if any(value not in (0, 1) for value in masks):
    raise ValueError(f"trajectory row {row_index} has a non-binary action mask")
  return tokens, masks, logps


def _percentile(values: list[float], fraction: float) -> float | None:
  if not values:
    return None
  ordered = sorted(values)
  return float(ordered[int(fraction * (len(ordered) - 1))])


def _shift_stats(
    mismatches: Iterable[dict[str, Any]], shift: int
) -> dict[str, Any]:
  by_row: dict[int, dict[int, dict[str, Any]]] = collections.defaultdict(dict)
  for mismatch in mismatches:
    row, position = mismatch["coordinate"]
    by_row[int(row)][int(position)] = mismatch
  deltas = []
  for positions in by_row.values():
    for position, mismatch in positions.items():
      peer = positions.get(position + shift)
      if peer is not None:
        deltas.append(abs(float(mismatch["a"]) - float(peer["b"])))
  return {
      "shift": shift,
      "pairs": len(deltas),
      "mean_abs": (sum(deltas) / len(deltas)) if deltas else None,
      "median_abs": statistics.median(deltas) if deltas else None,
      "p90_abs": _percentile(deltas, 0.9),
  }


def _join_mismatch(
    mismatch: dict[str, Any],
    records: list[dict[str, Any]],
    candidates: dict[tuple[int, int], list[int]],
) -> int:
  required = {
      "coordinate",
      "completion_position",
      "completion_valid_length",
      "prompt_length",
      "token_id",
      "a",
      "b",
  }
  missing = sorted(required - mismatch.keys())
  if missing:
    raise ValueError(f"mismatch record is missing fields: {missing}")
  coordinate = mismatch["coordinate"]
  if (
      not isinstance(coordinate, list)
      or len(coordinate) != 2
      or int(coordinate[1]) != int(mismatch["completion_position"])
  ):
    raise ValueError("mismatch coordinate is malformed")
  position = int(mismatch["completion_position"])
  key = (
      int(mismatch["prompt_length"]),
      int(mismatch["completion_valid_length"]),
  )
  matches = []
  for row_index in candidates.get(key, []):
    tokens, masks, logps = _trajectory_arrays(records[row_index], row_index)
    if (
        logps is not None
        and 0 <= position < len(tokens)
        and int(tokens[position]) == int(mismatch["token_id"])
        and int(masks[position]) == 1
        and float(logps[position]) == float(mismatch["a"])
    ):
      matches.append(row_index)
  if len(matches) != 1:
    raise ValueError(
        "mismatch does not join exactly one durable trajectory row: "
        f"coordinate={coordinate} key={key} matches={matches}"
    )
  return matches[0]


def classify(
    root: Path,
    *,
    source_sha: str | None = None,
    expected_hostname: str | None = None,
    prealignment_path: Path | None = None,
) -> dict[str, Any]:
  root = root.resolve()
  manifest_path = root / "run_manifest.json"
  prealignment_path = (
      prealignment_path.resolve()
      if prealignment_path is not None
      else root / "pre_alignment.jsonl"
  )
  metrics_path = root / "batch_metrics.jsonl"
  trajectory_paths = sorted(root.glob(_TRAJECTORY_GLOB))
  if len(trajectory_paths) != 1:
    raise ValueError(
        f"probe requires exactly one trajectory journal, got {trajectory_paths}"
    )
  for path in (manifest_path, prealignment_path, metrics_path):
    if not path.is_file() or path.stat().st_size <= 0:
      raise ValueError(f"required probe artifact is absent or empty: {path}")

  manifest = _load_json(manifest_path)
  whitelist_sha256 = manifest.get("whitelist_sha256")
  if (
      not isinstance(whitelist_sha256, str)
      or len(whitelist_sha256) != 64
      or any(character not in "0123456789abcdef" for character in whitelist_sha256)
  ):
    raise ValueError("probe manifest has no lowercase SHA-256 whitelist identity")
  if manifest.get("contract_name") == "local-qwen4b-dp1-tp4-seam-probe":
    if (
        manifest.get("onehost_seam_probe") is not True
        or manifest.get("onehost_xprof_arm") != "zero-hp"
        or manifest.get("stage") != "backward-no-commit"
    ):
      raise ValueError("one-host seam manifest selectors are inconsistent")
  if source_sha is not None and manifest.get("source_commit") != source_sha:
    raise ValueError(
        "probe source SHA differs from the requested source: "
        f"{manifest.get('source_commit')!r} != {source_sha!r}"
    )
  if expected_hostname is not None:
    observed = manifest.get("expected_hostname")
    if observed != expected_hostname:
      raise ValueError(
          f"probe hostname provenance differs: {observed!r} != "
          f"{expected_hostname!r}"
      )

  records = _load_trajectories(trajectory_paths[0])
  metrics = _load_last_jsonl(metrics_path)
  trajectory_sha256 = _sha256(trajectory_paths[0])
  if metrics.get("trajectory_sha256") not in (None, trajectory_sha256):
    raise ValueError("batch metrics trajectory SHA-256 differs from journal")
  metric_path = metrics.get("trajectory_path")
  if (
      metric_path is not None
      and Path(str(metric_path)).name != trajectory_paths[0].name
  ):
    raise ValueError("batch metrics trajectory path differs from journal")
  if metrics.get("trajectories") not in (None, len(records)):
    raise ValueError("batch metrics trajectory count differs from journal")
  if manifest.get("global_trajectories") not in (None, len(records)):
    raise ValueError("manifest trajectory count differs from journal")
  candidates: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
  admitted_action_tokens = 0
  statuses = collections.Counter()
  compact_rows = 0
  for row_index, record in enumerate(records):
    tokens, masks, _ = _trajectory_arrays(record, row_index)
    trajectory = record["trajectory"]
    statuses[str(record.get("status"))] += 1
    if record.get("compact_filtered") is True:
      compact_rows += 1
      # Timeout/overlong rows may be deliberately compacted to empty arrays.
      # They are durable audit records, but they are not admitted to alignment
      # and therefore need neither a prompt length nor a mismatch join key.
      continue
    prompt_length = trajectory.get("prompt_length")
    if not isinstance(prompt_length, int) or prompt_length <= 0:
      raise ValueError(f"trajectory row {row_index} has invalid prompt_length")
    candidates[(prompt_length, len(tokens))].append(row_index)
    admitted_action_tokens += sum(int(value) for value in masks)

  prealignment = _load_last_jsonl(prealignment_path)
  boundaries = prealignment.get("boundaries")
  if not isinstance(boundaries, dict):
    raise ValueError("pre-alignment record has no boundaries")
  boundary = boundaries.get("S_decode_vs_S_prefill")
  if not isinstance(boundary, dict):
    raise ValueError("S_decode_vs_S_prefill boundary is absent")
  if boundary.get("valid") is not True or boundary.get("finite") is not True:
    raise ValueError("S_decode_vs_S_prefill is invalid or non-finite")
  n_action = int(prealignment.get("N_action", -1))
  if n_action != admitted_action_tokens:
    raise ValueError(
        "pre-alignment action count differs from durable trajectories: "
        f"{n_action} != {admitted_action_tokens}"
    )

  differing_elements = int(boundary.get("differing_elements", -1))
  differing_bytes = int(boundary.get("differing_bytes", -1))
  total_elements = int(boundary.get("total_elements", -1))
  if min(differing_elements, differing_bytes, total_elements) < 0:
    raise ValueError("pre-alignment boundary counters are invalid")
  if total_elements != n_action or differing_elements > total_elements:
    raise ValueError(
        "pre-alignment element counters differ from admitted action tokens"
    )
  mismatches = boundary.get("mismatches")
  if not isinstance(mismatches, list):
    raise ValueError("pre-alignment mismatches must be a list")
  if differing_elements == 0 and mismatches:
    raise ValueError("exact boundary contains mismatch records")
  if differing_elements > 0 and not mismatches:
    raise ValueError("red boundary contains no bounded mismatch records")
  if mismatches and boundary.get("first_mismatch") != mismatches[0]:
    raise ValueError("first mismatch differs from bounded mismatch record 0")

  joined_rows = collections.Counter()
  action_starts = 0
  previous_environment = 0
  for mismatch in mismatches:
    if not isinstance(mismatch, dict):
      raise ValueError("mismatch entry is not an object")
    row_index = _join_mismatch(mismatch, records, candidates)
    joined_rows[row_index] += 1
    action_starts += int(mismatch.get("action_run_start") is True)
    previous_environment += int(
        mismatch.get("previous_token_is_environment") is True
    )

  if differing_elements:
    outcome = "FINITE_RED_REPRODUCED"
  elif n_action:
    outcome = "EXACT_ON_THIS_CARRIER"
  else:
    outcome = "INCONCLUSIVE_NO_ACTION_TOKENS"
  verdict = "PASS" if outcome != "INCONCLUSIVE_NO_ACTION_TOKENS" else "INCONCLUSIVE"
  process_status = None
  process_status_path = root / "probe_process_status.json"
  if process_status_path.is_file():
    process_status = _load_json(process_status_path)
    if process_status.get("profile") != "seam":
      raise ValueError("probe process status has the wrong profile")
    if not isinstance(process_status.get("training_process_status"), int):
      raise ValueError("probe process status has no integer exit status")

  return {
      "schema": "canon.p58.decode-prefill-probe.classification.v1",
      "verdict": verdict,
      "outcome": outcome,
      "source_commit": manifest.get("source_commit"),
      "model_id": manifest.get("model_id"),
      "contract_name": manifest.get("contract_name"),
      "role_topology": manifest.get("role_topology"),
      "carrier_provenance": {
          key: manifest.get(key)
          for key in (
              "run_id",
              "expected_hostname",
              "model_snapshot",
              "r2egym_commit",
              "task_image",
              "task_image_id",
              "runner_sha256",
              "whitelist_sha256",
              "stage",
              "checked_vma_diagnostic",
              "onehost_xprof_arm",
              "onehost_seam_probe",
              "max_response_length",
              "max_turns",
          )
      },
      "process_status": process_status,
      "trajectory_rows": len(records),
      "compact_filtered_rows": compact_rows,
      "status_histogram": dict(sorted(statuses.items())),
      "N_action": n_action,
      "S_decode_vs_S_prefill": {
          "differing_elements": differing_elements,
          "differing_bytes": differing_bytes,
          "total_elements": total_elements,
          "element_fraction": boundary.get("element_fraction"),
          "byte_fraction": boundary.get("byte_fraction"),
          "max_abs": boundary.get("max_abs"),
          "first_mismatch": boundary.get("first_mismatch"),
          "reported_mismatches": len(mismatches),
          "mismatches_truncated": boundary.get("mismatches_truncated"),
          "joined_artifact_rows": {
              str(key): value for key, value in sorted(joined_rows.items())
          },
          "reported_action_run_starts": action_starts,
          "reported_previous_environment": previous_environment,
          "shift_discriminator": [
              _shift_stats(mismatches, shift) for shift in (-1, 0, 1)
          ],
      },
      "artifacts": {
          "manifest": manifest_path.name,
          "manifest_sha256": _sha256(manifest_path),
          "pre_alignment": str(prealignment_path),
          "pre_alignment_sha256": _sha256(prealignment_path),
          "batch_metrics": metrics_path.name,
          "batch_metrics_sha256": _sha256(metrics_path),
          "trajectory": trajectory_paths[0].name,
          "trajectory_sha256": trajectory_sha256,
      },
      "claim": (
          "A finite RED reproduces decode-vs-prefill divergence on this exact "
          "carrier only. An exact result does not certify other TP/DP geometry."
      ),
  }


def _write_return_note(root: Path, report: dict[str, Any]) -> None:
  boundary = report["S_decode_vs_S_prefill"]
  text = f"""# P58 decode-vs-prefill probe return

Verdict: `{report['verdict']}`
Outcome: `{report['outcome']}`
Source: `{report.get('source_commit')}`
Topology: `{json.dumps(report.get('role_topology'), sort_keys=True)}`

- trajectory rows: {report['trajectory_rows']}
- action tokens: {report['N_action']}
- differing elements: {boundary['differing_elements']}
- differing bytes: {boundary['differing_bytes']}
- max abs: {boundary['max_abs']}
- reported mismatches joined to durable rows: {sum(boundary['joined_artifact_rows'].values())}
- training process status: {json.dumps(report.get('process_status'), sort_keys=True)}

Return only `P58_SEAM_PROBE_RETURN.tar.gz` and its adjacent `.sha256` file.
After extraction, `SHA256SUMS` must verify from inside the directory. This
carrier does not certify a different TP/DP geometry. The runner is configured
backward-no-commit; this classifier does not authorize or certify any
optimizer commit.
"""
  (root / "RETURN_TO_AGENT.md").write_text(text, encoding="utf-8")


def _write_checksums(root: Path, names: Iterable[str]) -> None:
  lines = []
  for name in sorted(set(names)):
    path = root / name
    if path.is_file():
      lines.append(f"{_sha256(path)}  {name}\n")
  (root / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--artifact-dir", type=Path, required=True)
  parser.add_argument("--source-sha")
  parser.add_argument("--expected-hostname")
  parser.add_argument("--output", type=Path)
  parser.add_argument("--package", action="store_true")
  args = parser.parse_args()

  root = args.artifact_dir.resolve()
  output = (args.output or (root / "decode_prefill_probe.classification.json")).resolve()
  if output.parent != root:
    raise SystemExit("classification output must live inside --artifact-dir")
  try:
    report = classify(
        root,
        source_sha=args.source_sha,
        expected_hostname=args.expected_hostname,
    )
  except (OSError, ValueError, json.JSONDecodeError) as exc:
    report = {
        "schema": "canon.p58.decode-prefill-probe.classification.v1",
        "verdict": "FAIL",
        "outcome": "MALFORMED_OR_INCOMPLETE_EVIDENCE",
        "error": str(exc),
    }
  output.write_text(
      json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  if args.package and report["verdict"] in ("PASS", "INCONCLUSIVE"):
    _write_return_note(root, report)
    names = [
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    ]
    _write_checksums(root, names)
  print(json.dumps(report, sort_keys=True, separators=(",", ":")))
  raise SystemExit(0 if report["verdict"] == "PASS" else 3 if report["verdict"] == "INCONCLUSIVE" else 1)


if __name__ == "__main__":
  main()
