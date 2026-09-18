#!/usr/bin/env python3
"""Proves matched profiled work across GSM8K Native and Zero-HP arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def classify(native: dict, zero: dict) -> dict:
  reasons = []
  if native.get("arm") != "native" or zero.get("arm") != "zero-hp":
    reasons.append("arm_identity")
  for field in (
      "source_sha",
      "source_diff_sha256",
      "runtime_manifest_sha256",
      "model_snapshot",
      "image_id",
      "topology",
      "capture",
  ):
    if native.get(field) != zero.get(field):
      reasons.append(field)
  arm_verdicts = (native.get("verdict"), zero.get("verdict"))
  native_work = native.get("profiled_work") or {}
  zero_work = zero.get("profiled_work") or {}
  work_fields = sorted(set(native_work) | set(zero_work))
  mismatched_work_fields = [
      field for field in work_fields if native_work.get(field) != zero_work.get(field)
  ]
  native_arrays = (native_work.get("fields") or {})
  zero_arrays = (zero_work.get("fields") or {})
  mismatched_work_arrays = sorted(
      field
      for field in set(native_arrays) | set(zero_arrays)
      if native_arrays.get(field) != zero_arrays.get(field)
  )
  work_match = not mismatched_work_fields
  if "FAIL" in arm_verdicts:
    verdict = "FAIL"
  elif any(value != "PASS" for value in arm_verdicts):
    verdict = "INCONCLUSIVE_CAPTURE"
  elif reasons or not work_match:
    if not work_match:
      reasons.append("profiled_work")
    verdict = "INCONCLUSIVE_INPUT_MISMATCH"
  else:
    verdict = "PASS"
  return {
      "schema": "canon.v1.gsm8k-onehost-xprof.pair.v1",
      "verdict": verdict,
      "arm_verdicts": {"native": arm_verdicts[0], "zero-hp": arm_verdicts[1]},
      "matched_source": not any(
          field in reasons
          for field in (
              "source_sha",
              "source_diff_sha256",
              "runtime_manifest_sha256",
              "model_snapshot",
              "image_id",
          )
      ),
      "matched_profiled_work": work_match,
      "mismatched_profiled_work_fields": mismatched_work_fields,
      "mismatched_profiled_work_arrays": mismatched_work_arrays,
      "topology": native.get("topology"),
      "capture": native.get("capture"),
      "reasons": reasons,
      "claim": (
          "matched one-host operation-attribution pair; use unprofiled timing for speed"
          if verdict == "PASS"
          else "no causal performance comparison"
      ),
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--native", type=Path, required=True)
  parser.add_argument("--zero-hp", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(args.output)
  native = json.loads(args.native.read_text(encoding="utf-8"))
  zero = json.loads(args.zero_hp.read_text(encoding="utf-8"))
  result = classify(native, zero)
  args.output.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps(result, sort_keys=True))
  if result["verdict"] == "PASS":
    return 0
  return 3 if result["verdict"].startswith("INCONCLUSIVE") else 1


if __name__ == "__main__":
  raise SystemExit(main())
