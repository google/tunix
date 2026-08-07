#!/usr/bin/env python3
"""Classify one archived 64-chip Pathways admission log.

The classifier promotes only the bounded claims that the log actually measures:
single-slice platform admission, the canonical Qwen operator chain, and the
same-session toy DP update. It never promotes full-model initialization,
segmented backward, an optimizer commit, or training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


EXPECTED_DEPTHS = {1, 2, 4, 8}
EXPECTED_WAYCOUNT_KEYS = {
    (width, depth, arm)
    for width in (2, 4, 8)
    for depth in (8, 15)
    for arm in ("replicated", "stock-ar", "f4-fixed")
}
EXPECTED_T2_CHECKS = {
    "auto_repeat_exact",
    "fault_rejected",
    "fixed_mesh_order_exact",
    "fixed_repeat_exact",
    "fixed_replicas_exact",
    "stock_repeat_exact",
    "stock_replicas_exact",
}
EXPECTED_T2_DECISION = (
    "FIXED_TOPOLOGY_ONLY_DEVICE_ORDER_SENSITIVE+BATCH_PLACEMENT_SENSITIVE"
)
FATAL_PATTERNS = {
    "python_traceback": r"Traceback \(most recent call last\):",
    "ignored_exception": r"Exception ignored in:",
    "fatal_runtime": r"(?m)^(?:FATAL(?:\s|:)|F\d{4}\s)",
    "runtime_error": r"(?m)^(?:RuntimeError|InternalError|XlaRuntimeError):",
    "tainted_session": r"SKIP_TAINTED|SESSION_TAINTED",
    "t1_failure": r"(?m)^===== T1 FAIL",
    "hard_gate_failure": (
        r"(?m)^\[(?:mosaic\.compat|canonical-op|P32\.DP)\] VERDICT:? FAIL$"
    ),
}

WAYCOUNT_RE = re.compile(
    r"(?m)^\[waycount\] width=\s*(?P<width>\d+) "
    r"replicas=\s*(?P<replicas>\d+) depth=\s*(?P<depth>\d+) "
    r"arm=(?P<arm>replicated|stock-ar\s*|f4-fixed\s*) "
    r"differing_bytes=\s*(?P<different>\d+)/(?P<total>\d+) .* "
    r"(?P<verdict>SAME|DIFFERS)$"
)
CANONICAL_RE = re.compile(
    r"(?m)^\[canonical-op\] depth=\s*(?P<depth>\d+) "
    r"differing_bytes=(?P<different>\d+)/(?P<total>\d+) "
    r"gradient_finite=(?P<finite>[01]) "
    r"gradient_nonzero=(?P<nonzero>\d+) (?P<verdict>SAME|DIFFERS)$"
)


def _sha256(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def _exact_count(
    text: str, pattern: str, expected: int, label: str, reasons: list[str]
) -> list[str]:
  matches = re.findall(pattern, text, flags=re.MULTILINE)
  if len(matches) != expected:
    reasons.append(f"{label}: expected {expected}, found {len(matches)}")
  return matches


def classify_text(
    text: str,
    *,
    artifact_sha256: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
  """Return a machine-readable, fail-closed classification."""
  reasons: list[str] = []

  if expected_sha256 and artifact_sha256 != expected_sha256:
    reasons.append(
        "artifact_sha256: expected "
        f"{expected_sha256}, found {artifact_sha256}"
    )

  fatal_hits = {
      name: len(re.findall(pattern, text))
      for name, pattern in FATAL_PATTERNS.items()
  }
  for name, count in fatal_hits.items():
    if count:
      reasons.append(f"{name}: found {count} fatal marker(s)")

  attempt_values = _exact_count(
      text,
      r"^\[entrypoint\] JOBSET_ATTEMPT (\d+) \(first attempt\)",
      1,
      "jobset_attempt",
      reasons,
  )
  if attempt_values and attempt_values[0] != "0":
    reasons.append(f"jobset_attempt: expected 0, found {attempt_values[0]}")

  _exact_count(
      text, r"^\[sync\] tracked_dirty=0$", 1, "tracked_provenance", reasons
  )
  _exact_count(
      text,
      r"^\[sync\] package_untracked=0$",
      1,
      "package_provenance",
      reasons,
  )
  _exact_count(
      text,
      r"^\[probe\] SUMMARY same=6 drift=0 missing=0$",
      1,
      "image_anchor",
      reasons,
  )
  _exact_count(
      text,
      r"^\[T1\.PATHWAYS\] required=1 initialized=1 status=ok$",
      1,
      "pathways_initialization",
      reasons,
  )
  _exact_count(
      text,
      r"^\[t1\.devices\] count=64 kind=TPU v5p platform=tpu$",
      1,
      "device_inventory",
      reasons,
  )
  _exact_count(
      text,
      r"^\[mesh\] slice_count=1 slices=\{0: 64\}$",
      1,
      "single_slice",
      reasons,
  )
  _exact_count(
      text, r"^\[mesh\] VERDICT: MATCH$", 1, "model_mesh", reasons
  )
  _exact_count(
      text, r"^\[bucket\] VERDICT: OK$", 1, "bucket_contract", reasons
  )
  _exact_count(
      text,
      r"^\[mosaic\.compat\] VERDICT: PASS$",
      1,
      "mosaic_compatibility",
      reasons,
  )

  waycount_rows = []
  waycount_keys = set()
  for match in WAYCOUNT_RE.finditer(text):
    row = {
        "width": int(match.group("width")),
        "replicas": int(match.group("replicas")),
        "depth": int(match.group("depth")),
        "arm": match.group("arm").strip(),
        "differing_bytes": int(match.group("different")),
        "total_bytes": int(match.group("total")),
        "verdict": match.group("verdict"),
    }
    key = (row["width"], row["depth"], row["arm"])
    waycount_keys.add(key)
    waycount_rows.append(row)
  if len(waycount_rows) != 18:
    reasons.append(
        f"generic_waycount: expected 18 rows, found {len(waycount_rows)}"
    )
  if waycount_keys != EXPECTED_WAYCOUNT_KEYS:
    reasons.append("generic_waycount: width/depth/arm coverage mismatch")
  _exact_count(
      text,
      r"^\[waycount\] measurements=18 expected=18$",
      1,
      "generic_waycount_count",
      reasons,
  )
  _exact_count(
      text,
      r"^\[waycount\] VERDICT: COMPLETE$",
      1,
      "generic_waycount_completion",
      reasons,
  )

  canonical_rows = []
  for match in CANONICAL_RE.finditer(text):
    canonical_rows.append(
        {
            "depth": int(match.group("depth")),
            "differing_bytes": int(match.group("different")),
            "total_bytes": int(match.group("total")),
            "gradient_finite": int(match.group("finite")),
            "gradient_nonzero": int(match.group("nonzero")),
            "verdict": match.group("verdict"),
        }
    )
  if len(canonical_rows) != 4:
    reasons.append(
        f"canonical_operator: expected 4 rows, found {len(canonical_rows)}"
    )
  if {row["depth"] for row in canonical_rows} != EXPECTED_DEPTHS:
    reasons.append("canonical_operator: depth coverage mismatch")
  for row in canonical_rows:
    if (
        row["differing_bytes"] != 0
        or row["total_bytes"] != 2_097_152
        or row["gradient_finite"] != 1
        or row["gradient_nonzero"] <= 0
        or row["verdict"] != "SAME"
    ):
      reasons.append(
          f"canonical_operator: depth {row['depth']} failed its numeric gate"
      )
  _exact_count(
      text,
      r"^\[canonical-op\] measurements=4 expected=4$",
      1,
      "canonical_operator_count",
      reasons,
  )
  _exact_count(
      text,
      r"^\[canonical-op\] VERDICT: PASS$",
      1,
      "canonical_operator_verdict",
      reasons,
  )

  t2_json_lines = _exact_count(
      text, r"^\[P32\.DP\] JSON (\{.*\})$", 1, "t2_json", reasons
  )
  t2: dict[str, Any] = {}
  if t2_json_lines:
    try:
      t2 = json.loads(t2_json_lines[0])
    except json.JSONDecodeError as exc:
      reasons.append(f"t2_json: invalid JSON: {exc}")
  if t2:
    checks = t2.get("checks", {})
    if set(checks) != EXPECTED_T2_CHECKS or not all(checks.values()):
      reasons.append("t2_json: expected all seven registered checks to be true")
    expected_scalars = {
        "dp": 16,
        "tp": 4,
        "local_samples": 16,
        "global_samples": 256,
        "decision": EXPECTED_T2_DECISION,
    }
    for key, expected in expected_scalars.items():
      if t2.get(key) != expected:
        reasons.append(
            f"t2_json: {key} expected {expected!r}, found {t2.get(key)!r}"
        )
    mesh_ids = t2.get("mesh_ids", [])
    if len(mesh_ids) != 64 or len(set(mesh_ids)) != 64:
      reasons.append("t2_json: mesh_ids must contain 64 unique devices")
    update = t2.get("update", {})
    expected_update_keys = {
        "gradient_sha256",
        "parameter_sha256",
        "moment_sha256",
        "variance_sha256",
    }
    if set(update) != expected_update_keys or not all(
        re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in update.values()
    ):
      reasons.append("t2_json: update fingerprints are incomplete or malformed")
  _exact_count(
      text, r"^\[P32\.DP\] VERDICT PASS$", 1, "t2_verdict", reasons
  )
  _exact_count(
      text,
      r"^\[entrypoint\] <-- 70_run_t1\.sh ok$",
      1,
      "t1_stage_exit",
      reasons,
  )
  _exact_count(
      text,
      r"^\[entrypoint\] <-- 75_run_dp\.sh ok$",
      1,
      "t2_stage_exit",
      reasons,
  )
  _exact_count(
      text,
      r"^\[entrypoint\] mode=gate-only -- topology admission probes complete\.  No training was run\.$",
      1,
      "no_training_boundary",
      reasons,
  )

  ordered_markers = (
      "[mosaic.compat] VERDICT: PASS",
      "[waycount] VERDICT: COMPLETE",
      "[canonical-op] VERDICT: PASS",
      "[P32.DP] JSON ",
      "[P32.DP] VERDICT PASS",
      "[entrypoint] <-- 70_run_t1.sh ok",
      "[entrypoint] <-- 75_run_dp.sh ok",
  )
  marker_positions = [text.find(marker) for marker in ordered_markers]
  if any(position < 0 for position in marker_positions) or marker_positions != sorted(
      marker_positions
  ):
    reasons.append("execution_order: registered hard gates are missing or out of order")

  passed = not reasons
  return {
      "status": "TARGET PASS" if passed else "INCONCLUSIVE",
      "artifact_sha256": artifact_sha256,
      "checks": {
          "jobset_attempt_zero": bool(attempt_values and attempt_values[0] == "0"),
          "clean_package_provenance": (
              "tracked_provenance" not in " ".join(reasons)
              and "package_provenance" not in " ".join(reasons)
          ),
          "single_slice_64_v5p": not any(
              key in " ".join(reasons)
              for key in ("device_inventory", "single_slice")
          ),
          "generic_waycount_complete": (
              len(waycount_rows) == 18
              and waycount_keys == EXPECTED_WAYCOUNT_KEYS
          ),
          "canonical_operator_bitwise": (
              len(canonical_rows) == 4
              and all(row["differing_bytes"] == 0 for row in canonical_rows)
          ),
          "same_session_toy_dp_update": bool(t2) and not any(
              reason.startswith("t2_") for reason in reasons
          ),
          "no_fatal_or_taint_markers": not any(fatal_hits.values()),
      },
      "measurements": {
          "generic_waycount": {
              "rows": len(waycount_rows),
              "same_rows": sum(row["verdict"] == "SAME" for row in waycount_rows),
              "dirty_rows": sum(
                  row["verdict"] == "DIFFERS" for row in waycount_rows
              ),
              "claim_role": "advisory platform diagnostic",
          },
          "canonical_operator": canonical_rows,
          "t2": t2,
      },
      "claim_scope": {
          "single_slice_pathways_platform": (
              "TARGET PASS" if passed else "INCONCLUSIVE"
          ),
          "bounded_canonical_qwen_operator": (
              "TARGET PASS" if passed else "INCONCLUSIVE"
          ),
          "same_session_toy_dp_update": (
              "TARGET PASS" if passed else "INCONCLUSIVE"
          ),
          "qwen3_8b_model_initialization": "TARGET NOT RUN",
          "segmented_backward": "TARGET NOT RUN",
          "optimizer_commit": "TARGET NOT RUN",
          "training": "TARGET NOT RUN",
      },
      "reasons": reasons,
  }


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("log", type=Path)
  parser.add_argument("--expected-sha256")
  parser.add_argument("--json-out", type=Path)
  args = parser.parse_args(argv)

  data = args.log.read_bytes()
  result = classify_text(
      data.decode("utf-8", errors="replace"),
      artifact_sha256=_sha256(data),
      expected_sha256=args.expected_sha256,
  )
  rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
  if args.json_out:
    args.json_out.write_text(rendered, encoding="utf-8")
  sys.stdout.write(rendered)
  return 0 if result["status"] == "TARGET PASS" else 1


if __name__ == "__main__":
  sys.exit(main())
