#!/usr/bin/env python3
"""Classify the exact-geometry P58 checked-VMA-off Step-0 diagnostic."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
from typing import Any


_SCRIPT_DIR = Path(__file__).resolve().parent
_PROBE_PATH = _SCRIPT_DIR / "classify_decode_prefill_probe.py"
_SPEC = importlib.util.spec_from_file_location(
    "p58_decode_prefill_probe", _PROBE_PATH
)
if _SPEC is None or _SPEC.loader is None:
  raise RuntimeError("cannot import P58 decode/prefill classifier")
_PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE)

_SOURCE_SHA = re.compile(r"[0-9a-f]{40}")
_WHITELIST_SHA256 = (
    "ec297c9cbc39cd67db15b0b9db6a229b15671b848df5ec3101de9ef8df7c9973"
)
_PROFILE_MARKER = (
    "[P58.VMA.DIAGNOSTIC] profile_resolved selector=off dp=8 tp=8 "
    "checked_vma=0 compatibility_alias=0 vma_p59_only=0 "
    "fixed_ar_gather=1 continue_decode=8 prefix_cache=0 backward=0 "
    "optimizer_commits=0"
)
_CONTROLLED_EXIT = (
    "[CANON_P38] CONTROLLED_EXIT code=42 backward=0 optimizer_commits=0"
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
  records = []
  for number, line in enumerate(
      path.read_text(encoding="utf-8").splitlines(), 1
  ):
    if not line.strip():
      continue
    value = json.loads(line)
    if not isinstance(value, dict):
      raise ValueError(f"expected object at {path}:{number}")
    records.append(value)
  if not records:
    raise ValueError(f"empty diagnostic evidence: {path}")
  return records


def classify(
    *, run_log: Path, pre_alignment: Path, debug_dir: Path, update_report: Path
) -> dict[str, Any]:
  reasons: list[str] = []
  text = run_log.read_text(encoding="utf-8", errors="replace")
  records = _jsonl(pre_alignment)
  if len(records) != 1:
    reasons.append(f"pre_alignment_records={len(records)}")
  record = records[-1]

  try:
    probe = _PROBE.classify(
        debug_dir,
        prealignment_path=pre_alignment,
    )
  except (OSError, ValueError, json.JSONDecodeError) as exc:
    probe = {}
    reasons.append(f"durable_probe={type(exc).__name__}:{exc}")

  provenance = probe.get("carrier_provenance", {})
  if not _SOURCE_SHA.fullmatch(str(probe.get("source_commit", ""))):
    reasons.append("source_commit")
  expected_provenance = {
      "stage": "full",
      "checked_vma_diagnostic": "off",
      "whitelist_sha256": _WHITELIST_SHA256,
  }
  wrong_provenance = {
      key: provenance.get(key)
      for key, expected in expected_provenance.items()
      if provenance.get(key) != expected
  }
  if wrong_provenance:
    reasons.append(f"provenance={wrong_provenance}")
  if probe.get("model_id") != "Qwen/Qwen3-4B-Instruct-2507":
    reasons.append("model_id")
  if probe.get("contract_name") != "p58-qwen4b-tim-128":
    reasons.append("contract_name")
  if probe.get("role_topology") != {"dp": 8, "tp": 8, "devices": 64}:
    reasons.append("role_topology")
  if probe.get("trajectory_rows") != 128:
    reasons.append(f"trajectory_rows={probe.get('trajectory_rows')}")

  boundaries = record.get("boundaries", {})
  a_b = boundaries.get("S_decode_vs_S_prefill", {})
  b_c = boundaries.get("S_prefill_vs_T_old", {})
  for name, boundary in (("A-B", a_b), ("B-C", b_c)):
    if boundary.get("valid") is not True or boundary.get("finite") is not True:
      reasons.append(f"{name}_invalid_or_nonfinite")
    if not isinstance(boundary.get("differing_bytes"), int):
      reasons.append(f"{name}_missing_byte_count")
  if b_c.get("differing_bytes") != 0:
    reasons.append(f"B-C_bytes={b_c.get('differing_bytes')}")
  n_action = record.get("N_action")
  if not isinstance(n_action, int) or n_action <= 0:
    reasons.append(f"N_action={n_action}")

  marker_counts = {
      "profile": text.count(_PROFILE_MARKER),
      "precheck_round": text.count("[CANON_P38] PRECHECK_ROUND_COMPLETE "),
      "controlled_exit": text.count(_CONTROLLED_EXIT),
  }
  wrong_markers = {
      key: count for key, count in marker_counts.items() if count != 1
  }
  if wrong_markers:
    reasons.append(f"marker_counts={wrong_markers}")
  forbidden = {
      "checked_vma_enabled": "[P59.CHECKED_VMA] enabled=1",
      "p59_backward": "[P59.BACKWARD]",
      "p66_backward": "[P66.BACKWARD]",
      "fixed_lm_head_vjp": "CANON_P38_FIXED_LM_HEAD_VJP=1",
      "optimizer_commit": "optimizer_commits=1",
      "global_step_1": "Global step 1 completed",
  }
  present_forbidden = [
      name for name, marker in forbidden.items() if marker in text
  ]
  if present_forbidden:
    reasons.append(f"forbidden_runtime={present_forbidden}")
  if update_report.exists() and update_report.stat().st_size:
    reasons.append("update_report_nonempty")

  a_b_bytes = a_b.get("differing_bytes")
  outcome = (
      "A_B_EXACT_WITH_CHECKED_VMA_OFF"
      if a_b_bytes == 0
      else "A_B_RED_WITH_CHECKED_VMA_OFF"
      if isinstance(a_b_bytes, int) and a_b_bytes > 0
      else "INVALID"
  )
  verdict = "PASS" if not reasons and outcome != "INVALID" else "FAIL"
  return {
      "schema": "canon.p58.checked-vma-off-diagnostic.v1",
      "verdict": verdict,
      "outcome": outcome,
      "source_commit": probe.get("source_commit"),
      "model_id": probe.get("model_id"),
      "role_topology": probe.get("role_topology"),
      "trajectory_rows": probe.get("trajectory_rows"),
      "compact_filtered_rows": probe.get("compact_filtered_rows"),
      "N_action": n_action,
      "A_B_differing_elements": a_b.get("differing_elements"),
      "A_B_differing_bytes": a_b_bytes,
      "A_B_max_abs": a_b.get("max_abs"),
      "B_C_differing_bytes": b_c.get("differing_bytes"),
      "backward": 0,
      "optimizer_commits": 0,
      "marker_counts": marker_counts,
      "durable_probe": probe,
      "reasons": reasons,
      "claim": (
          "This exact DP8xTP8 Step-0 selector discriminates whether disabling "
          "the process-wide checked-VMA family removes the p58z07 A-B seam. "
          "It does not certify backward, optimizer, or full training."
      ),
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--run-log", required=True, type=Path)
  parser.add_argument("--pre-alignment", required=True, type=Path)
  parser.add_argument("--debug-dir", required=True, type=Path)
  parser.add_argument("--update-report", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  result = classify(
      run_log=args.run_log,
      pre_alignment=args.pre_alignment,
      debug_dir=args.debug_dir,
      update_report=args.update_report,
  )
  if args.output.exists():
    raise FileExistsError(
        f"refusing to overwrite P58 diagnostic classification: {args.output}"
    )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(
      "P58_CHECKED_VMA_DIAGNOSTIC_CLASSIFICATION "
      f"verdict={result['verdict']} outcome={result['outcome']} "
      f"a_b_bytes={result['A_B_differing_bytes']} "
      f"b_c_bytes={result['B_C_differing_bytes']} "
      "backward=0 optimizer_commits=0",
      flush=True,
  )
  return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
