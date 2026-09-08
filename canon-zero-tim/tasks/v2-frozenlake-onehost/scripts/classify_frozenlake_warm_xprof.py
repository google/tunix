#!/usr/bin/env python3
"""Classifies the P45 R1/R2 same-process warm XProf diagnostic."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Any


_SHA256 = re.compile(r"[0-9a-f]{64}")
_PERF = re.compile(
    r"\[PERF\] stage=(p32_vag_reverse|segmented_value_and_grad) "
    r"seconds=([0-9.]+)"
)
_GRAD = re.compile(
    r"\[PERF\] stage=grad_accumulate seconds=([0-9.]+).*"
    r"variant=adopt-scaled"
)
_XPLANE_CENSUS_SCHEMA = "canon.v2-frozenlake-onehost.xplane-census.v1"
_XPROF_VERSION = "2.23.1"


def _read_json(path: Path) -> Any:
  return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
  return [
      json.loads(line)
      for line in path.read_text(encoding="utf-8").splitlines()
      if line.strip()
  ]


def _strict_exact(records: list[dict[str, Any]]) -> bool:
  if not records:
    return False
  for record in records:
    if record.get("verdict") != "PASS":
      return False
    boundaries = record.get("boundaries")
    if not isinstance(boundaries, dict) or not boundaries:
      return False
    for boundary in boundaries.values():
      if not isinstance(boundary, dict):
        return False
      if boundary.get("valid") is not True:
        return False
      if boundary.get("finite") is not True:
        return False
      if boundary.get("differing_bytes") != 0:
        return False
      if boundary.get("differing_elements") != 0:
        return False
      if boundary.get("max_abs") != 0.0:
        return False
  return True


def _trace_needles(path: Path, needles: tuple[bytes, ...]) -> dict[str, int]:
  counts = {needle.decode(): 0 for needle in needles}
  overlap = max(map(len, needles)) - 1
  tail = b""
  with gzip.open(path, "rb") as source:
    while True:
      chunk = source.read(1024 * 1024)
      if not chunk:
        break
      data = tail + chunk
      for needle in needles:
        counts[needle.decode()] += data.count(needle)
      tail = data[-overlap:] if overlap else b""
  return counts


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    while chunk := source.read(1024 * 1024):
      digest.update(chunk)
  return digest.hexdigest()


def classify(
    root: Path, arm: str, docker_exit: int, anchor_registry: Path
) -> dict[str, Any]:
  if arm not in ("r1", "r2"):
    raise ValueError(f"warm XProf admits only r1/r2, got {arm!r}")
  paths = {
      name: root / name
      for name in (
          "raw.log",
          "run_manifest.json",
          "runtime.json",
          "pre_alignment.jsonl",
          "alignment.jsonl",
          "updates.json",
          "warm_xprof_repeat.json",
          "xplane_census.json",
      )
  }
  missing = [
      name
      for name, path in paths.items()
      if not path.is_file() or path.stat().st_size == 0
  ]
  reasons = [f"missing_or_empty:{name}" for name in missing]
  if missing:
    return {
        "schema": "canon.v2-frozenlake-onehost.warm-xprof-classification.v1",
        "verdict": "INCONCLUSIVE",
        "arm": arm,
        "reasons": reasons,
    }

  raw = paths["raw.log"].read_text(encoding="utf-8", errors="replace")
  manifest = _read_json(paths["run_manifest.json"])
  runtime = _read_json(paths["runtime.json"])
  pre = _read_jsonl(paths["pre_alignment.jsonl"])
  alignment = _read_jsonl(paths["alignment.jsonl"])
  update = _read_json(paths["updates.json"])
  repeat = _read_json(paths["warm_xprof_repeat.json"])
  xplane_census = _read_json(paths["xplane_census.json"])

  def require(condition: bool, reason: str) -> None:
    if not condition:
      reasons.append(reason)

  selectors = {
      "keep_tape": "stream",
      "reduce_once": "0" if arm == "r1" else "1",
      "length_sort": "0",
      "report_adjoint_buckets": "0",
      "chunk_dependency_ticket": "0",
      "chunk_backpressure": "0",
  }
  require(docker_exit == 0, f"docker_exit={docker_exit}")
  require(runtime.get("verdict") == "PASS", "runtime")
  require(
      manifest.get("schema") == "canon.v2-frozenlake-onehost.run.v1",
      "manifest.schema",
  )
  require(manifest.get("workload") == "p45", "manifest.workload")
  require(
      manifest.get("workload_name")
      == "frozenlake-p45-onehost-dp2-tp2",
      "manifest.workload_name",
  )
  require(manifest.get("arm") == arm, "manifest.arm")
  require(manifest.get("classification_mode") == "profile", "manifest.mode")
  require(
      manifest.get("xprof")
      == {
          "phase": "update",
          "skip_steps": 0,
          "steps": 1,
          "host_tracer": 1,
          "python_tracer": 0,
          "tpu_trace_mode": "TRACE_ONLY_XLA",
          "labels": 1,
      },
      "manifest.xprof",
  )
  require(manifest.get("stage") == "backward-no-commit", "manifest.stage")
  require(
      manifest.get("topology") == {"dp": 2, "tp": 2, "devices": 4},
      "manifest.topology",
  )
  require(manifest.get("selectors") == selectors, "manifest.selectors")
  require(manifest.get("checked_vma") is True, "manifest.checked_vma")
  capsule = manifest.get("training_capsule", {})
  require(capsule.get("mode") == "replay", "capsule.mode")
  capsule_sha = capsule.get("sha256")
  require(_SHA256.fullmatch(str(capsule_sha)) is not None, "capsule.sha256")

  require(len(pre) == 1, f"pre_alignment_records={len(pre)}")
  require(len(alignment) == 16, f"alignment_records={len(alignment)}")
  require(_strict_exact(pre + alignment), "strict_alignment")
  require(
      repeat.get("schema") == "canon.v2-frozenlake-onehost.warm-xprof.v1",
      "repeat.schema",
  )
  require(repeat.get("verdict") == "PASS", f"repeat.verdict={repeat.get('verdict')}")
  require(repeat.get("reasons") == [], f"repeat.reasons={repeat.get('reasons')}")
  require(repeat.get("arm") == arm, "repeat.arm")
  require(repeat.get("repeat_count") == 2, "repeat.count")
  for field in (
      "gradient_repeat_exact",
      "update_gradient_norm_repeat_exact",
      "alignment_repeat_exact",
      "state_repeat_exact",
  ):
    require(repeat.get(field) is True, f"repeat.{field}")

  registry = _read_json(anchor_registry)
  anchor = registry.get("anchors", {}).get(f"p45:dp2-tp2:{arm}")
  require(isinstance(anchor, dict), "gradient_anchor_unregistered")
  profiled_norms = repeat.get("gradient_profiled_norms")
  if isinstance(anchor, dict):
    require(
        profiled_norms == anchor.get("micro_gradient_norms"),
        "gradient_anchor_bitwise",
    )
    require(
        capsule_sha == anchor.get("training_capsule_sha256"),
        "gradient_anchor_capsule_sha",
    )
    require(
        repeat.get("update_gradient_profiled_norm")
        == anchor.get("update_gradient_norm"),
        "update_gradient_anchor_bitwise",
    )
  require(update.get("verdict") == "PASS", "update.verdict")
  require(update.get("commits") == 0, "update.commits")
  require(update.get("micro_gradient_norms") == profiled_norms, "update.norms")

  p66_count = raw.count("[P66.VMA] outer_check_enabled")
  require(p66_count == 39, f"p66_count={p66_count}")
  require(raw.count("[V2.FL.XPROF] warmup_complete") == 1, "warmup_marker")
  require(raw.count("[V2.FL.XPROF] phase=update armed") == 1, "armed_marker")
  require(raw.count("[V2.FL.XPROF] phase=update stopped") == 1, "stopped_marker")
  require(raw.count("[V2.FL.XPROF] repeat_complete") == 1, "repeat_marker")
  expected_loan = 2 if arm == "r2" else 0
  require(
      raw.count("[V2.REDUCE_ONCE.ACCUMULATOR_LOAN]") == expected_loan,
      "loan_count",
  )
  require(
      raw.count("[V2.REDUCE_ONCE.ACCUMULATOR_RESET]") == expected_loan,
      "reset_count",
  )

  perf: dict[str, list[float]] = {
      "p32_vag_reverse": [],
      "segmented_value_and_grad": [],
  }
  for stage, seconds in _PERF.findall(raw):
    perf[stage].append(float(seconds))
  require(all(len(values) == 2 for values in perf.values()), f"perf_counts={perf}")
  adoption = [float(value) for value in _GRAD.findall(raw)]
  require(len(adoption) == expected_loan, f"adoption_count={len(adoption)}")

  xplanes = sorted((root / "xprof-update").glob("plugins/profile/*/*.xplane.pb"))
  traces = sorted((root / "xprof-update").glob("plugins/profile/*/*.trace.json.gz"))
  require(len(xplanes) == 1, f"xplane_files={len(xplanes)}")
  require(len(traces) == 1, f"trace_files={len(traces)}")
  require(len(xplanes) != 1 or xplanes[0].stat().st_size > 0, "xplane_empty")
  require(len(traces) != 1 or traces[0].stat().st_size > 0, "trace_empty")
  require(xplane_census.get("schema") == _XPLANE_CENSUS_SCHEMA, "xplane_census.schema")
  require(xplane_census.get("verdict") == "PASS", "xplane_census.verdict")
  require(xplane_census.get("reasons") == [], "xplane_census.reasons")
  require(xplane_census.get("arm") == arm, "xplane_census.arm")
  require(
      xplane_census.get("xprof_version") == _XPROF_VERSION,
      "xplane_census.xprof_version",
  )
  require(
      xplane_census.get("trace_buffer_drops") == 0,
      "xplane_census.trace_buffer_drops",
  )
  census_artifact = xplane_census.get("xplane", {})
  if len(xplanes) == 1:
    require(
        census_artifact.get("path") == str(xplanes[0].relative_to(root)),
        "xplane_census.path",
    )
    require(
        census_artifact.get("bytes") == xplanes[0].stat().st_size,
        "xplane_census.bytes",
    )
    require(
        census_artifact.get("sha256") == _sha256(xplanes[0]),
        "xplane_census.sha256",
    )

  # The compressed trace is a bounded UI/navigation export. Keep it present
  # and report its visible labels, but use the full XPlane census above for
  # completeness: long P45 updates can exceed the trace-JSON event ceiling.
  trace_view_counts = {}
  if len(traces) == 1 and traces[0].stat().st_size > 0:
    needles = (
        b"zero_tim_update",
        b"reverse_groups",
        b"fixed_dp_reduce",
        b"gradient_accumulate",
        b"deferred_finite_receipts",
    ) + ((b"staged_receipt_fetch",) if arm == "r2" else ())
    trace_view_counts = _trace_needles(traces[0], needles)

  peaks = []
  for field in ("profiled_hbm_before", "profiled_hbm_after"):
    for device in repeat.get(field) or ():
      peak = device.get("peak_bytes_in_use")
      limit = device.get("bytes_limit")
      if isinstance(peak, int) and isinstance(limit, int) and limit > 0:
        peaks.append((peak, limit))
  require(bool(peaks), "hbm_receipts")
  peak_ratio = max((peak / limit for peak, limit in peaks), default=None)
  require(peak_ratio is not None and peak_ratio <= 0.90, f"hbm_ratio={peak_ratio}")

  second_reverse = (
      perf["p32_vag_reverse"][-1]
      if len(perf["p32_vag_reverse"]) == 2
      else None
  )
  second_segmented = (
      perf["segmented_value_and_grad"][-1]
      if len(perf["segmented_value_and_grad"]) == 2
      else None
  )
  return {
      "schema": "canon.v2-frozenlake-onehost.warm-xprof-classification.v1",
      "verdict": "PASS" if not reasons else "FAIL",
      "arm": arm,
      "claim_level": "diagnostic-only-onehost-qwen8b-dp2-tp2-p45",
      "performance_eligible": False,
      "reasons": reasons,
      "strict": {
          "pre_alignment_records": len(pre),
          "post_alignment_records": len(alignment),
          "p66_outer_receipts": p66_count,
      },
      "gradient": {
          "micro_gradient_norms": profiled_norms,
          "update_gradient_norm": repeat.get("update_gradient_profiled_norm"),
          "anchor_exact": "gradient_anchor_bitwise" not in reasons,
      },
      "timing": {
          "warmup_reverse_seconds": (
              perf["p32_vag_reverse"][0]
              if len(perf["p32_vag_reverse"]) == 2
              else None
          ),
          "profiled_reverse_seconds": second_reverse,
          "warmup_segmented_seconds": (
              perf["segmented_value_and_grad"][0]
              if len(perf["segmented_value_and_grad"]) == 2
              else None
          ),
          "profiled_segmented_seconds": second_segmented,
          "profiled_post_reverse_seconds": (
              second_segmented - second_reverse
              if second_segmented is not None and second_reverse is not None
              else None
          ),
          "adoption_seconds": adoption,
      },
      "hbm_peak_to_limit": peak_ratio,
      "trace": {
          "xplane": str(xplanes[0]) if len(xplanes) == 1 else None,
          "trace_json": str(traces[0]) if len(traces) == 1 else None,
          "bounded_view_needles": trace_view_counts,
          "xplane_census": xplane_census,
      },
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--arm", choices=("r1", "r2"), required=True)
  parser.add_argument("--docker-exit", type=int, required=True)
  parser.add_argument("--anchor-registry", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite classification: {args.output}")
  record = classify(args.root, args.arm, args.docker_exit, args.anchor_registry)
  args.output.write_text(
      json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(
      f"V2_FL_WARM_XPROF verdict={record['verdict']} "
      f"arm={args.arm} reasons={record['reasons']}"
  )
  return 0 if record["verdict"] == "PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
