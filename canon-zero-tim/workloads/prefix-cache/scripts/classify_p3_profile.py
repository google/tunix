#!/usr/bin/env python3
"""Validate Phase3 diagnostic-round XProf and semantic Perfetto artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


class ProfileError(RuntimeError):
  pass


def _require(condition: bool, message: str) -> None:
  if not condition:
    raise ProfileError(message)


def _artifact(path: Path, state: Path) -> dict:
  _require(path.is_file() and path.stat().st_size > 0, f"empty artifact: {path}")
  return {
      "path": str(path.relative_to(state)),
      "bytes": path.stat().st_size,
      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
  }


def classify(raw_path: Path, state: Path, expect_apc: bool) -> dict:
  _require(raw_path.is_file(), f"missing raw log: {raw_path}")
  _require(state.is_dir(), f"missing state directory: {state}")
  raw = raw_path.read_text(encoding="utf-8", errors="replace")
  classification_path = state / "alignment.classification.json"
  _require(classification_path.is_file(), "alignment classification is absent")
  alignment = json.loads(classification_path.read_text(encoding="utf-8"))
  expected_status = (
      "GB_GC_CERTIFICATION_GREEN" if expect_apc else "CONTROL_GREEN"
  )
  _require(alignment.get("status") == expected_status,
           f"alignment status is not {expected_status}")
  _require(alignment.get("expect_apc") is expect_apc,
           "alignment APC arm identity drifted")
  _require(
      raw.count(
          "[P3.XPROF] phase=diagnostic started "
          "completed_rounds=1 capture_round=1"
      ) == 1,
      "diagnostic XProf start marker drifted",
  )
  _require(
      raw.count(
          "[P3.XPROF] phase=diagnostic stopped "
          "completed_rounds=2 captured_round=1"
      ) == 1,
      "diagnostic XProf stop marker drifted",
  )
  _require(
      raw.count("[P3.XPROF] semantic_perfetto_exported completed_rounds=2") == 1,
      "semantic Perfetto export marker drifted",
  )
  _require(len(re.findall(r"\[CANON_ALIGN_PRE\].*verdict=PASS", raw)) == 3,
           "profile run did not retain three passing byte gates")
  _require(not re.search(r"\[CANON_ALIGN(?:_PRE)?\].*verdict=FAIL", raw),
           "profile run contains an alignment FAIL")

  xplanes = sorted((state / "xprof").rglob("*.xplane.pb"))
  trace_json = sorted((state / "xprof").rglob("*.trace.json.gz"))
  perfetto = sorted((state / "perf").rglob("perfetto_trace_v2_*.pb"))
  _require(xplanes, "XProf emitted no device xplane.pb")
  _require(trace_json, "XProf emitted no trace.json.gz")
  _require(len(perfetto) == 1,
           f"expected one semantic Perfetto trace, found {len(perfetto)}")
  artifacts = {
      "xplane": [_artifact(path, state) for path in xplanes],
      "trace_json_gz": [_artifact(path, state) for path in trace_json],
      "semantic_perfetto": [_artifact(path, state) for path in perfetto],
  }
  return {
      "schema": "phase3-apc-diagnostic-profile-v1",
      "status": "PROFILE_GREEN",
      "expect_apc": expect_apc,
      "captured_round": 1,
      "xprof_phase": "diagnostic",
      "host_tracer_level": 1,
      "python_tracer_level": 0,
      "artifacts": artifacts,
      "raw_sha256_before_profile_classification": hashlib.sha256(
          raw_path.read_bytes()
      ).hexdigest(),
      "claim": (
          "shape attribution only; timing decision comes from the matched non-profile pair"
      ),
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--raw", type=Path, required=True)
  parser.add_argument("--state", type=Path, required=True)
  parser.add_argument("--expect-apc", choices=("0", "1"), required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise SystemExit(f"refusing to overwrite profile classification: {args.output}")
  try:
    result = classify(args.raw, args.state, args.expect_apc == "1")
  except (ProfileError, json.JSONDecodeError, OSError) as exc:
    result = {
        "schema": "phase3-apc-diagnostic-profile-v1",
        "status": "INCONCLUSIVE",
        "error": str(exc),
    }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(json.dumps(result, sort_keys=True))
  if result["status"] != "PROFILE_GREEN":
    raise SystemExit(1)


if __name__ == "__main__":
  main()
