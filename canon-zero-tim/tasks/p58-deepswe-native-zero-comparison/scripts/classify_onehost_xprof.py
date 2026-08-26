#!/usr/bin/env python3
"""Classify and seal one P58 one-host DeepSWE XProf arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


_SHA = re.compile(r"[0-9a-f]{40}\Z")


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"expected JSON object: {path}")
  return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
  if not path.is_file():
    raise FileNotFoundError(path)
  rows = []
  for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
      continue
    value = json.loads(line)
    if not isinstance(value, dict):
      raise ValueError(f"expected object at {path}:{number}")
    rows.append(value)
  if not rows:
    raise ValueError(f"empty JSONL: {path}")
  return rows


def _boundaries_exact(rows: list[dict[str, Any]]) -> bool:
  return all(
      boundary.get("valid") is not False
      and boundary.get("finite") is not False
      and boundary.get("differing_bytes") == 0
      for row in rows
      for boundary in row.get("boundaries", {}).values()
  )


def _boundaries_finite(rows: list[dict[str, Any]]) -> bool:
  return all(
      row.get("blocking_reds", []) == []
      and boundary.get("valid") is not False
      and boundary.get("finite") is not False
      and isinstance(boundary.get("differing_bytes"), int)
      and boundary["differing_bytes"] >= 0
      for row in rows
      for boundary in row.get("boundaries", {}).values()
  )


def classify(
    *,
    arm: str,
    root: Path,
    source_sha: str,
    expected_hostname: str,
) -> dict[str, Any]:
  reasons: list[str] = []
  raw = root / "raw.log"
  install_log_path = root / "install.log"
  manifest_path = root / "run_manifest.json"
  report_path = root / "backward_no_commit.json"
  pre_path = root / "pre_alignment.jsonl"
  alignment_path = root / "alignment.jsonl"
  required = (
      raw, install_log_path, manifest_path, report_path, pre_path, alignment_path
  )
  for path in required:
    if not path.is_file() or path.stat().st_size == 0:
      reasons.append(f"missing_or_empty:{path.name}")
  if reasons:
    return {
        "schema": "canon.p58.onehost-xprof.arm.v1",
        "arm": arm,
        "verdict": "INCONCLUSIVE_CAPTURE",
        "reasons": reasons,
    }

  text = raw.read_text(encoding="utf-8", errors="replace")
  install_text = install_log_path.read_text(encoding="utf-8", errors="replace")
  manifest = _json(manifest_path)
  report = _json(report_path)
  pre = _jsonl(pre_path)
  alignment = _jsonl(alignment_path)
  all_alignment = pre + alignment
  xplanes = sorted(root.joinpath("xprof-update").rglob("*.xplane.pb"))
  traces = sorted(root.joinpath("xprof-update").rglob("*.trace.json.gz"))
  perfetto = sorted(root.joinpath("perfetto").rglob("perfetto_trace_v2_*.pb"))

  hard_failures = []
  capture_failures = []
  if arm not in ("native", "zero-hp"):
    hard_failures.append("invalid_arm")
  expected_manifest = {
      "schema": "canon.local.deepswe.run-manifest.v1",
      "source_commit": source_sha,
      "stage": "backward-no-commit",
      "model_id": "Qwen/Qwen3-4B-Instruct-2507",
      "contract_name": "local-qwen4b-dp1-tp4",
      "onehost_xprof_arm": arm,
      "role_topology": {"dp": 1, "tp": 4, "devices": 4},
      "global_prompts": 1,
      "generations": 2,
      "global_trajectories": 2,
      "max_turns": 2,
      "max_response_length": 512,
      "dataset_seed": 42,
      "rollout_seed": 42,
      "seed_scope": "engine-global; async completion order not claimed",
      "expected_hostname": expected_hostname,
  }
  wrong_manifest = {
      key: manifest.get(key)
      for key, expected in expected_manifest.items()
      if manifest.get(key) != expected
  }
  if wrong_manifest:
    hard_failures.append(f"manifest={wrong_manifest}")
  if not _SHA.fullmatch(source_sha):
    hard_failures.append("source_sha")
  source_diff_sha256 = manifest.get("source_diff_sha256")
  if not isinstance(source_diff_sha256, str) or not re.fullmatch(
      r"[0-9a-f]{64}", source_diff_sha256
  ):
    hard_failures.append("source_diff_sha256")
  provenance_patterns = {
      "model_snapshot": r".*/cdbee75f17c01a7cc42f958dc650907174af0554\Z",
      "r2egym_commit": r"0d94c4eb9431cd195c55a7ea3abd54006c9a1735\Z",
      "task_image_id": r"sha256:[0-9a-f]{64}\Z",
      "runner_sha256": r"[0-9a-f]{64}\Z",
  }
  for key, pattern in provenance_patterns.items():
    value = manifest.get(key)
    if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
      hard_failures.append(f"manifest.{key}")
  if text.count(
      f"[P58.ONEHOST.XPROF] ARM_PASS arm={arm} topology=dp1-tp4"
  ) != 1:
    hard_failures.append("arm_marker")
  if text.count(
      f"[P58.ONEHOST.XPROF] warmup_complete arm={arm} "
  ) != 1:
    hard_failures.append("warmup_marker")
  if text.count(
      f"[P58.ONEHOST.XPROF] semantic_warmup_discarded arm={arm} "
      "next_export=profiled-repeat-only"
  ) != 1:
    capture_failures.append("semantic_warmup_discard_marker")
  if text.count(
      "[P58.ONEHOST.XPROF] diagnostic_advantages "
  ) != 1:
    hard_failures.append("diagnostic_advantages_marker")
  if text.count("[DEEPSWE.ONEHOST] optimizer_boundary_skipped commits=0") != 2:
    hard_failures.append("no_commit_count")
  for marker in (
      f"[P51.XPROF] phase=update armed step=0 arm={arm}",
      "[P51.XPROF] phase=update started step=0",
      f"[P51.XPROF] phase=update stopped step=0 arm={arm}",
      "[V1.PERFETTO] captured training_step=0",
  ):
    if text.count(marker) != 1:
      capture_failures.append(f"marker:{marker}")
  if "tpu_trace_mode=TRACE_COMPUTE" not in text:
    capture_failures.append("trace_compute_marker")
  if "verdict=FAIL" in text or "[CANON_ALIGN] FAIL" in text:
    hard_failures.append("alignment_fail_marker")
  if report.get("verdict") != "PASS":
    hard_failures.append(f"backward_verdict={report.get('verdict')}")
  for key, expected in {
      "commits": 0,
      "gradient_finite": True,
      "gradient_nonzero": True,
      "gradient_repeat_exact": True,
      "repeat_count": 2,
      "xprof_arm": arm,
  }.items():
    if report.get(key) != expected:
      hard_failures.append(f"report.{key}={report.get(key)!r}")
  for key in (
      "model_changed_paths",
      "optimizer_changed_paths",
      "accumulator_changed_paths",
      "reference_changed_paths",
  ):
    if report.get(key) != []:
      hard_failures.append(f"report.{key}")
  if report.get("train_steps_before") != report.get("train_steps_after"):
    hard_failures.append("train_step_changed")
  work_hashes = report.get("work_hashes")
  if not isinstance(work_hashes, dict) or work_hashes.get("actor_update_calls") != 2:
    hard_failures.append("work_hashes")

  if arm == "zero-hp":
    if re.search(r"all [0-9]+ files match \(qwen4b\)", install_text) is None:
      hard_failures.append("canonical_install_receipt")
    if text.count(
        "[CANON_" "ADAPTER] differentiable engine adapter registered"
    ) != 1:
      hard_failures.append("canonical_adapter_marker")
    if not _boundaries_exact(all_alignment):
      hard_failures.append("zero_boundaries_not_exact")
  else:
    if install_text.count(
        "[P58.STOCK_OBSERVER] OVERLAY_PASS files=2 "
        "stock_runner_verified=1 canonical_bundle=off "
        "treatment=observer-only onehost=1"
    ) != 1:
      hard_failures.append("stock_observer_install_receipt")
    if (
        "[CANON_" "ADAPTER] differentiable engine adapter registered"
        in text
    ):
      hard_failures.append("native_canonical_adapter_leak")
    if not _boundaries_finite(all_alignment):
      hard_failures.append("native_boundaries_not_finite")

  if not xplanes or any(path.stat().st_size == 0 for path in xplanes):
    capture_failures.append("device_xplane")
  if not traces or any(path.stat().st_size == 0 for path in traces):
    capture_failures.append("trace_json_gz")
  if len(perfetto) != 1 or perfetto[0].stat().st_size == 0:
    capture_failures.append(f"perfetto_count={len(perfetto)}")

  if hard_failures:
    verdict = "FAIL"
  elif capture_failures:
    verdict = "INCONCLUSIVE_CAPTURE"
  else:
    verdict = "PASS"
  return {
      "schema": "canon.p58.onehost-xprof.arm.v1",
      "arm": arm,
      "verdict": verdict,
      "claim_level": "direct-onehost-dp1-tp4-profile-carrier",
      "source_sha": source_sha,
      "source_diff_sha256": source_diff_sha256,
      "expected_hostname": expected_hostname,
      "hard_failures": hard_failures,
      "capture_failures": capture_failures,
      "alignment_records": {
          "pre": len(pre),
          "post_backward": len(alignment),
      },
      "work_hashes": work_hashes,
      "captures": {
          "xplane": [str(path.relative_to(root)) for path in xplanes],
          "trace_json_gz": [str(path.relative_to(root)) for path in traces],
          "perfetto": [str(path.relative_to(root)) for path in perfetto],
      },
      "target_not_run": [
          "dp8-tp8",
          "pathways",
          "p59-rank-parallel-backward",
          "qwen4b-tp8-fixed-head",
          "prefix-cache-apc",
      ],
  }


def _seal(root: Path) -> None:
  target = root / "SHA256SUMS"
  if target.exists():
    raise FileExistsError(f"refusing to overwrite {target}")
  paths = sorted(
      path for path in root.rglob("*")
      if path.is_file() and path != target
  )
  target.write_text(
      "".join(f"{_sha256(path)}  {path.relative_to(root)}\n" for path in paths),
      encoding="utf-8",
  )


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--arm", choices=("native", "zero-hp"), required=True)
  parser.add_argument("--artifact-dir", type=Path, required=True)
  parser.add_argument("--source-sha", required=True)
  parser.add_argument("--expected-hostname", required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  root = args.artifact_dir.resolve()
  output = args.output.resolve()
  if output.parent != root or output.exists():
    raise ValueError("classification output must be a new file in artifact-dir")
  result = classify(
      arm=args.arm,
      root=root,
      source_sha=args.source_sha,
      expected_hostname=args.expected_hostname,
  )
  output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  _seal(root)
  print(json.dumps(result, sort_keys=True))
  return 0 if result["verdict"] == "PASS" else 3 if result["verdict"].startswith("INCONCLUSIVE") else 1


if __name__ == "__main__":
  sys.exit(main())
