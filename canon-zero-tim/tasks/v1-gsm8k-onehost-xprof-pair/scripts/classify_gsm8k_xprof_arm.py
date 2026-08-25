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
    require_hierarchy: bool = False,
    hierarchy_census_rc: int | None = None,
) -> dict:
  if arm not in ("native", "zero-hp"):
    raise ValueError(f"invalid arm: {arm!r}")
  state = run_root / "train"
  raw_path = state / "raw.log"
  driver_path = run_root / "driver.log"
  xprof_census_path = state / "xprof_census.txt"
  semantic_census_path = state / "semantic_census.txt"
  hierarchy_census_path = state / "hierarchy_census.txt"
  reasons = []
  required = [raw_path, driver_path, xprof_census_path, semantic_census_path]
  if require_hierarchy:
    required.append(hierarchy_census_path)
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

  steps = len(re.findall(r"Global step \d+ completed in", text))
  if steps != 3:
    reasons.append(f"global_steps={steps} expected=3")
  if text.count(
      f"[V1.GSM8K.XPROF] PREFLIGHT_PASS arm={arm} topology=DP4xTP1"
  ) != 1:
    reasons.append("preflight_marker")
  if text.count("[P51.XPROF] phase=update started step=1 ") != 1:
    reasons.append("xprof_start_step")
  if text.count(
      "[P51.XPROF] phase=update stopped step=2 anchor=step_completed"
  ) != 1:
    reasons.append("xprof_stop_step")
  if text.count(
      f"[V1.GSM8K.XPROF] RUN_END arm={arm} docker_exit=0"
  ) != 1:
    reasons.append("run_end")

  work = _work_receipts(text, reasons)
  if len(work) != 3:
    reasons.append(f"work_receipts={len(work)} expected=3")
  work_by_step = {row.get("train_step"): row for row in work}
  if set(work_by_step) != {0, 1, 2}:
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
  if require_hierarchy:
    if arm != "zero-hp":
      reasons.append("hierarchy_requirement_is_zero_hp_only")
    if hierarchy_census_rc != 0 or (
        "V1_GSM8K_XPROF_HIERARCHY_CENSUS_GREEN" not in hierarchy_text
    ):
      reasons.append(f"hierarchy_census_rc={hierarchy_census_rc}")

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
    if align_verdicts.count("PASS") != 51 or align_verdicts.count("FAIL") != 0:
      reasons.append(
          "zero_alignment="
          f"{align_verdicts.count('PASS')}/51 fail={align_verdicts.count('FAIL')}"
      )
    if "zt_tr_dp_parallel_bwd_" not in xprof_text:
      reasons.append("zero_semantic_parallel_backward_absent")

  profiled_work = work_by_step.get(1)
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
          "start_step": 1,
          "stop_step": 2,
          "hierarchy_required": require_hierarchy,
      },
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
  parser.add_argument("--xprof-census-rc", type=int, required=True)
  parser.add_argument("--semantic-census-rc", type=int, required=True)
  parser.add_argument("--require-hierarchy", action="store_true")
  parser.add_argument("--hierarchy-census-rc", type=int)
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
      xprof_census_rc=args.xprof_census_rc,
      semantic_census_rc=args.semantic_census_rc,
      require_hierarchy=args.require_hierarchy,
      hierarchy_census_rc=args.hierarchy_census_rc,
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
