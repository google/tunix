#!/usr/bin/env python3
"""Fail-closed classifier for one arm of the GSM8K one-host XProf pair."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path
import re


_WORK_PREFIX = "[V1.GSM8K.XPROF.WORK] "
_ALIGN_RE = re.compile(r"^\[CANON_ALIGN(?:_PRE)?\].*\bverdict=(PASS|FAIL)\b")
_CANON_ADAPTER_MARKER = "[CANON_" "ADAPTER]"
_SIZE_SCHEMA = "canon.v1.gsm8k-onehost-xprof.size.v1"
_SIZE_SOFT_WARNING_BYTES = 1_200_000_000
_SIZE_HARD_MAX_BYTES = 1_500_000_000
# Every committed update emits one pre-alignment verdict plus one per
# gradient group, and the DP4xTP1 carrier has 16 groups.  Measured on
# .../v1_zero-hp_p70bc_final_20260827: 3 [CANON_ALIGN_PRE] + 48
# [CANON_ALIGN] = 51 verdicts over three updates.
_ALIGN_VERDICTS_PER_UPDATE = 17
_DEFAULT_EXPECTED_UPDATES = 3


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(8 * 1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def _work_receipts(text: str, reasons: list[str]) -> list[dict]:
  rows = []
  for number, line in enumerate(text.splitlines(), 1):
    if not line.startswith(_WORK_PREFIX):
      continue
    try:
      row = json.loads(line.removeprefix(_WORK_PREFIX))
    except json.JSONDecodeError as exc:
      reasons.append(f"work_json_line_{number}:{exc.msg}")
      continue
    if not isinstance(row, dict):
      reasons.append(f"work_json_line_{number}:not_object")
      continue
    rows.append(row)
  return rows


def _size_receipt(
    path: Path, xprof_root: Path, reasons: list[str]
) -> dict | None:
  """Validates that the immutable budget receipt matches current files."""
  try:
    receipt = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as exc:
    reasons.append(f"xprof_size_receipt:{type(exc).__name__}")
    return None
  if not isinstance(receipt, dict):
    reasons.append("xprof_size_receipt:not_object")
    return None
  expected = {
      "schema": _SIZE_SCHEMA,
      "xprof_root": "train/xprof",
      "byte_basis": "sum_of_logical_bytes_for_regular_files",
      "soft_warning_bytes": _SIZE_SOFT_WARNING_BYTES,
      "hard_max_bytes": _SIZE_HARD_MAX_BYTES,
  }
  for key, value in expected.items():
    if receipt.get(key) != value:
      reasons.append(f"xprof_size_receipt.{key}={receipt.get(key)!r}")

  actual_files = {}
  if xprof_root.is_dir():
    for candidate in xprof_root.rglob("*"):
      if candidate.is_symlink():
        reasons.append(
            "xprof_size_receipt.symlink="
            + candidate.relative_to(xprof_root).as_posix()
        )
      elif candidate.is_file():
        actual_files[candidate.relative_to(xprof_root).as_posix()] = (
            candidate.stat().st_size
        )
  recorded_files = {}
  rows = receipt.get("files")
  if not isinstance(rows, list):
    reasons.append("xprof_size_receipt.files:not_list")
    rows = []
  for row in rows:
    if not isinstance(row, dict):
      reasons.append("xprof_size_receipt.files:not_object")
      continue
    relative = row.get("path")
    size = row.get("bytes")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
    ):
      reasons.append(f"xprof_size_receipt.file={row!r}")
      continue
    if relative in recorded_files:
      reasons.append(f"xprof_size_receipt.duplicate={relative}")
    recorded_files[relative] = size
  if recorded_files != actual_files:
    reasons.append("xprof_size_receipt.files_mismatch")

  total_bytes = sum(actual_files.values())
  actual_counts = {
      "xplane": sum(name.endswith(".xplane.pb") for name in actual_files),
      "trace_json_gz": sum(
          name.endswith(".trace.json.gz") for name in actual_files
      ),
      "other": sum(
          not name.endswith((".xplane.pb", ".trace.json.gz"))
          for name in actual_files
      ),
  }
  if receipt.get("counts") != actual_counts:
    reasons.append(
        f"xprof_size_receipt.counts={receipt.get('counts')!r} "
        f"actual={actual_counts!r}"
    )
  if receipt.get("total_bytes") != total_bytes:
    reasons.append(
        "xprof_size_receipt.total_bytes="
        f"{receipt.get('total_bytes')!r} actual={total_bytes}"
    )
  if receipt.get("file_count") != len(actual_files):
    reasons.append(
        "xprof_size_receipt.file_count="
        f"{receipt.get('file_count')!r} actual={len(actual_files)}"
    )
  if total_bytes > _SIZE_HARD_MAX_BYTES:
    reasons.append(
        f"xprof_bytes={total_bytes} exceeds_hard_max={_SIZE_HARD_MAX_BYTES}"
    )
  expected_status = (
      "FAIL" if total_bytes > _SIZE_HARD_MAX_BYTES
      else "WARN" if total_bytes > _SIZE_SOFT_WARNING_BYTES
      else "PASS"
  )
  if receipt.get("status") != expected_status:
    reasons.append(
        f"xprof_size_receipt.status={receipt.get('status')!r} "
        f"expected={expected_status}"
    )
  if receipt.get("reasons") != []:
    reasons.append("xprof_size_receipt.reasons_nonempty")
  return receipt


def classify(
    *,
    arm: str,
    run_root: Path,
    source_sha: str,
    source_diff_sha256: str,
    runtime_manifest_sha256: str,
    model_snapshot: str,
    image_id: str,
    xprof_census_rc: int,
    semantic_census_rc: int,
    size_census_rc: int = 0,
    expected_updates: int = _DEFAULT_EXPECTED_UPDATES,
    require_hierarchy: bool = False,
    hierarchy_census_rc: int | None = None,
    trace_census_rc: int | None = None,
) -> dict:
  if arm not in ("native", "zero-hp"):
    raise ValueError(f"invalid arm: {arm!r}")
  if expected_updates < 1:
    raise ValueError(f"invalid expected updates: {expected_updates!r}")
  state = run_root / "train"
  raw_path = state / "raw.log"
  driver_path = run_root / "driver.log"
  xprof_census_path = state / "xprof_census.txt"
  semantic_census_path = state / "semantic_census.txt"
  hierarchy_census_path = state / "hierarchy_census.txt"
  trace_census_path = state / "trace_census.txt"
  size_census_path = state / "xprof_size_census.txt"
  size_receipt_path = state / "xprof_size_receipt.json"
  reasons = []
  required = [
      raw_path,
      driver_path,
      xprof_census_path,
      semantic_census_path,
      size_census_path,
      size_receipt_path,
  ]
  if require_hierarchy:
    required.extend((hierarchy_census_path, trace_census_path))
  for path in required:
    if not path.is_file() or path.stat().st_size == 0:
      reasons.append(f"missing_or_empty:{path.name}")
  text = raw_path.read_text(encoding="utf-8", errors="replace") if raw_path.is_file() else ""
  xprof_text = (
      xprof_census_path.read_text(encoding="utf-8", errors="replace")
      if xprof_census_path.is_file()
      else ""
  )
  semantic_text = (
      semantic_census_path.read_text(encoding="utf-8", errors="replace")
      if semantic_census_path.is_file()
      else ""
  )
  hierarchy_text = (
      hierarchy_census_path.read_text(encoding="utf-8", errors="replace")
      if hierarchy_census_path.is_file()
      else ""
  )
  trace_text = (
      trace_census_path.read_text(encoding="utf-8", errors="replace")
      if trace_census_path.is_file()
      else ""
  )
  size_text = (
      size_census_path.read_text(encoding="utf-8", errors="replace")
      if size_census_path.is_file()
      else ""
  )
  size_receipt = (
      _size_receipt(size_receipt_path, state / "xprof", reasons)
      if size_receipt_path.is_file()
      else None
  )

  steps = len(re.findall(r"Global step \d+ completed in", text))
  if steps != expected_updates:
    reasons.append(f"global_steps={steps} expected={expected_updates}")
  if text.count(
      f"[V1.GSM8K.XPROF] PREFLIGHT_PASS arm={arm} topology=DP4xTP1"
  ) != 1:
    reasons.append("preflight_marker")
  if text.count("[P51.XPROF] phase=update started step=2 ") != 1:
    reasons.append("xprof_start_step")
  if text.count(
      "[P51.XPROF] phase=update stopped step=3 anchor=step_completed"
  ) != 1:
    reasons.append("xprof_stop_step")
  if text.count(
      f"[V1.GSM8K.XPROF] RUN_END arm={arm} docker_exit=0"
  ) != 1:
    reasons.append("run_end")

  work = _work_receipts(text, reasons)
  if len(work) != expected_updates:
    reasons.append(f"work_receipts={len(work)} expected={expected_updates}")
  work_by_step = {row.get("train_step"): row for row in work}
  if set(work_by_step) != set(range(expected_updates)):
    reasons.append(f"work_steps={sorted(str(value) for value in work_by_step)}")
  for step, row in work_by_step.items():
    if row.get("arm") != arm:
      reasons.append(f"work[{step}].arm={row.get('arm')!r}")
    fields = row.get("fields")
    if not isinstance(fields, dict) or not {
        "prompt_ids", "completion_ids", "advantages"
    }.issubset(fields):
      reasons.append(f"work[{step}].fields")

  xplanes = sorted(glob.glob(
      str(state / "xprof/plugins/profile/*/*.xplane.pb")
  ))
  traces = sorted(glob.glob(
      str(state / "xprof/plugins/profile/*/*.trace.json.gz")
  ))
  semantic = sorted(glob.glob(str(state / "perf/perfetto_trace_v2_*.pb")))
  if len(xplanes) != 1 or Path(xplanes[0]).stat().st_size <= 0:
    reasons.append(f"xplane_files={len(xplanes)} expected=1_nonempty")
  if len(traces) != 1 or Path(traces[0]).stat().st_size <= 0:
    reasons.append(f"trace_files={len(traces)} expected=1_nonempty")
  if len(semantic) != 1 or Path(semantic[0]).stat().st_size <= 0:
    reasons.append(f"semantic_files={len(semantic)} expected=1_nonempty")
  if xprof_census_rc != 0 or "CENSUS_GREEN" not in xprof_text:
    reasons.append(f"xprof_census_rc={xprof_census_rc}")
  if semantic_census_rc != 0 or "CENSUS_GREEN" not in semantic_text:
    reasons.append(f"semantic_census_rc={semantic_census_rc}")
  if size_census_rc != 0 or (
      "V1_GSM8K_XPROF_SIZE_CENSUS_GREEN" not in size_text
  ):
    reasons.append(f"size_census_rc={size_census_rc}")
  if require_hierarchy:
    if arm != "zero-hp":
      reasons.append("hierarchy_requirement_is_zero_hp_only")
    if hierarchy_census_rc != 0 or (
        "V1_GSM8K_XPROF_HIERARCHY_CENSUS_GREEN" not in hierarchy_text
    ):
      reasons.append(f"hierarchy_census_rc={hierarchy_census_rc}")
    if trace_census_rc != 0 or (
        "V1_GSM8K_XPROF_TRACE_CENSUS_GREEN" not in trace_text
    ):
      reasons.append(f"trace_census_rc={trace_census_rc}")

  align_verdicts = [
      match.group(1)
      for line in text.splitlines()
      if (match := _ALIGN_RE.match(line.strip()))
  ]
  if arm == "native":
    if text.count("[P56.VANILLA] stock arm:") != 1 or text.count(
        "[P56.VANILLA] engine contract attestation bypassed (stock arm)"
    ) != 1:
      reasons.append("native_vanilla_markers")
    if _CANON_ADAPTER_MARKER in text or "zt_tr_dp_parallel_bwd_" in xprof_text:
      reasons.append("native_canonical_program_present")
    if align_verdicts:
      reasons.append(f"native_alignment_verdicts={len(align_verdicts)} expected=0")
  else:
    if "[P56.VANILLA]" in text:
      reasons.append("zero_inherited_vanilla")
    expected_verdicts = _ALIGN_VERDICTS_PER_UPDATE * expected_updates
    if (
        align_verdicts.count("PASS") != expected_verdicts
        or align_verdicts.count("FAIL") != 0
    ):
      reasons.append(
          "zero_alignment="
          f"{align_verdicts.count('PASS')}/{expected_verdicts} "
          f"fail={align_verdicts.count('FAIL')}"
      )
    if "zt_tr_dp_parallel_bwd_" not in xprof_text:
      reasons.append("zero_semantic_parallel_backward_absent")

  profiled_work = work_by_step.get(2)
  comparable_work = None
  if isinstance(profiled_work, dict):
    comparable_work = {
        "train_step": profiled_work.get("train_step"),
        "global_step": profiled_work.get("global_step"),
        "fields": profiled_work.get("fields"),
        "shape_signature": profiled_work.get("shape_signature"),
    }
  artifacts = {}
  for name, files in (
      ("xplane", xplanes), ("trace_json_gz", traces), ("semantic_perfetto", semantic)
  ):
    if len(files) == 1 and Path(files[0]).is_file():
      path = Path(files[0])
      artifacts[name] = {
          "path": str(path),
          "bytes": path.stat().st_size,
          "sha256": _sha256(path),
      }
  return {
      "schema": "canon.v1.gsm8k-onehost-xprof.arm.v1",
      "verdict": "PASS" if not reasons else "FAIL",
      "arm": arm,
      "source_sha": source_sha,
      "source_diff_sha256": source_diff_sha256,
      "runtime_manifest_sha256": runtime_manifest_sha256,
      "model_snapshot": model_snapshot,
      "image_id": image_id,
      "topology": {"dp": 4, "tp": 1, "devices": 4},
      "capture": {
          "phase": "update",
          "start_step": 2,
          "stop_step": 3,
          "updates": expected_updates,
          "hierarchy_required": require_hierarchy,
      },
      "xprof_budget": (
          {
              key: size_receipt.get(key)
              for key in (
                  "status",
                  "byte_basis",
                  "soft_warning_bytes",
                  "hard_max_bytes",
                  "total_bytes",
                  "file_count",
                  "counts",
              )
          }
          if isinstance(size_receipt, dict)
          else None
      ),
      "profiled_work": comparable_work,
      "artifacts": artifacts,
      "reasons": reasons,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--arm", choices=("native", "zero-hp"), required=True)
  parser.add_argument("--run-root", type=Path, required=True)
  parser.add_argument("--source-sha", required=True)
  parser.add_argument("--source-diff-sha256", required=True)
  parser.add_argument("--runtime-manifest-sha256", required=True)
  parser.add_argument("--model-snapshot", required=True)
  parser.add_argument("--image-id", required=True)
  parser.add_argument(
      "--expected-updates",
      type=int,
      default=_DEFAULT_EXPECTED_UPDATES,
      help="committed updates implied by CANON_P33_RUN_STAGE",
  )
  parser.add_argument("--xprof-census-rc", type=int, required=True)
  parser.add_argument("--semantic-census-rc", type=int, required=True)
  parser.add_argument("--size-census-rc", type=int, required=True)
  parser.add_argument("--require-hierarchy", action="store_true")
  parser.add_argument("--hierarchy-census-rc", type=int)
  parser.add_argument("--trace-census-rc", type=int)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(args.output)
  record = classify(
      arm=args.arm,
      run_root=args.run_root,
      source_sha=args.source_sha,
      source_diff_sha256=args.source_diff_sha256,
      runtime_manifest_sha256=args.runtime_manifest_sha256,
      model_snapshot=args.model_snapshot,
      image_id=args.image_id,
      expected_updates=args.expected_updates,
      xprof_census_rc=args.xprof_census_rc,
      semantic_census_rc=args.semantic_census_rc,
      size_census_rc=args.size_census_rc,
      require_hierarchy=args.require_hierarchy,
      hierarchy_census_rc=args.hierarchy_census_rc,
      trace_census_rc=args.trace_census_rc,
  )
  args.output.write_text(
      json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(
      "V1_GSM8K_XPROF_ARM "
      f"verdict={record['verdict']} arm={args.arm} reasons={record['reasons']}"
  )
  return 0 if record["verdict"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
