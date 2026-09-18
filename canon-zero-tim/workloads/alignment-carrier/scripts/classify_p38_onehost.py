#!/usr/bin/env python3
"""Classify one P38 direct-attached pre-backward record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class ClassificationError(ValueError):
  pass


def _difference(record: dict[str, Any], name: str) -> int:
  boundary = record.get("boundaries", {}).get(name)
  if not isinstance(boundary, dict):
    raise ClassificationError(f"missing boundary {name}")
  if boundary.get("valid") is not True:
    raise ClassificationError(f"invalid boundary {name}")
  value = boundary.get("differing_bytes")
  if not isinstance(value, int) or value < 0:
    raise ClassificationError(f"invalid differing_bytes for {name}: {value!r}")
  return value


def classify(records: list[dict[str, Any]], log_text: str) -> dict[str, Any]:
  if len(records) != 1:
    raise ClassificationError(
        f"expected exactly one pre-alignment record, got {len(records)}"
    )
  record = records[0]
  if not isinstance(record.get("N_action"), int) or record["N_action"] <= 0:
    raise ClassificationError("pre-alignment record has no action tokens")
  a_b = _difference(record, "S_decode_vs_S_prefill")
  b_c = _difference(record, "S_prefill_vs_T_old")
  stop_count = log_text.count("[CANON_P38] PRECHECK_COMPLETE STOP_BEFORE_BACKWARD")
  backward_count = log_text.count("CANON_PROCESSED_LOGPROB_VJP backward")
  optimizer_markers = sum(
      log_text.count(marker)
      for marker in (
          "optimizer_commit",
          "OPTIMIZER_COMMIT",
          "TRAINING_DONE",
      )
  )
  path_counts = {
      "fixed_ar": log_text.count("CANON_FIXED_AR=1 fixed-order tree"),
      "fixed_embed": log_text.count(
          "CANON_FIXED_AR_EMBED=1 fixed-order embed gather"
      ),
      "logprob_m": log_text.count("CANON_LOGPROB_M on"),
      "shared_tail": log_text.count(
          "runner_sampling_adapter_same_object=True"
      ),
      "four_tpu_devices": log_text.count("[P38.ONEHOST] devices=4 "),
      "tpu_backend": log_text.count(" platform=tpu"),
      "overlay_identity": log_text.count(
          "[P38.ONEHOST] OVERLAY_BYTE_IDENTITY PASS files=6"
      ),
  }
  if not all(value > 0 for value in path_counts.values()):
    raise ClassificationError(f"missing canonical path evidence: {path_counts}")
  if backward_count or optimizer_markers:
    raise ClassificationError(
        "one-host precheck crossed the mutation boundary: "
        f"backward={backward_count} optimizer_markers={optimizer_markers}"
    )
  contract_red = log_text.count("C7/C8 violation [post-import]")
  if contract_red:
    raise ClassificationError(
        f"canonical environment contract failed {contract_red} time(s)"
    )

  if b_c:
    verdict = "VOID_REGRESSION"
  elif a_b:
    verdict = "LOCAL_REPRODUCED"
  elif stop_count == 1 and record.get("verdict") == "PASS":
    verdict = "LOCAL_NOT_REPRODUCED"
  else:
    raise ClassificationError(
        "exact boundaries lack one clean precheck-only completion marker"
    )
  return {
      "schema_version": 1,
      "verdict": verdict,
      "N_action": record["N_action"],
      "differing_bytes": {
          "S_decode_vs_S_prefill": a_b,
          "S_prefill_vs_T_old": b_c,
      },
      "path_counts": path_counts,
      "stop_before_backward": stop_count == 1,
      "backward_count": backward_count,
      "optimizer_markers": optimizer_markers,
      "contract_red": contract_red,
      "claim_scope": "direct-attached-one-host-only",
  }


def _records(path: Path) -> list[dict[str, Any]]:
  return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--report", type=Path, required=True)
  parser.add_argument("--raw", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  try:
    result = classify(_records(args.report), args.raw.read_text(errors="replace"))
  except ClassificationError as exc:
    result = {
        "schema_version": 1,
        "verdict": "INCONCLUSIVE",
        "error": str(exc),
        "claim_scope": "direct-attached-one-host-only",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(1) from exc
  args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, sort_keys=True))
  if result["verdict"] == "VOID_REGRESSION":
    raise SystemExit(1)


if __name__ == "__main__":
  main()
