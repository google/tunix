#!/usr/bin/env python3
"""Compare the Phase-0 DP2 reduce-once admission pair fail closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


CHUNK_ELEMENTS = 1_048_576
EXPECTED_LEAVES = 310
EXPECTED_UPDATES = 3
EXPECTED_GROUPS = 32
EXPECTED_VMA_MODULES = 31
TRACE_EVENT_TRUNCATION_FLOOR = 1_000_000
COSINE_MIN = 0.999999
REL_L2_MAX = 1.0e-5
WORK_PREFIX = "[V1.GSM8K.XPROF.WORK] "
VMA_PREFIX = "[P66.VMA] outer_check_enabled "
GROUP_RE = re.compile(
    r"^\[P33\.DP2\] reverse_group_done group=(\d+)/32 .*"
    r"rank_contributions=2 .*unique_rank_fingerprints=2/2(?: |$)"
)
STAGED_RE = re.compile(
    r"^\[P59\.DP2\] staged_rank_contributions_checked "
    r"groups=32/32 unique_min=2/2 unique_max=2/2$"
)
REDUCE_RE = re.compile(
    r"^\[P59\.DP2\] update_reduce_done check_vma=1 "
    r"collectives=2 replicas_exact=1$"
)
TRACE_EVENT_COUNT_RE = re.compile(r"^trace_event_count=(\d+)$")
TRACE_RED_MARKER = "V1_GSM8K_XPROF_TRACE_CENSUS_RED"


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _data_sha256(array: np.ndarray) -> str:
  digest = hashlib.sha256()
  flattened = array.reshape(-1)
  for start in range(0, flattened.size, CHUNK_ELEMENTS):
    digest.update(
        np.ascontiguousarray(
            flattened[start : start + CHUNK_ELEMENTS]
        ).tobytes(order="C")
    )
  return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected a JSON object")
  return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
  rows = [
      json.loads(line)
      for line in path.read_text(encoding="utf-8").splitlines()
      if line.strip()
  ]
  if not all(isinstance(row, dict) for row in rows):
    raise ValueError(f"{path}: every JSONL row must be an object")
  return rows


def _raw_lines(path: Path) -> list[str]:
  return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _load_capture(root: Path) -> list[dict[str, Any]]:
  capture = root / "train/p61_numerical/gradient"
  if capture.is_symlink() or not capture.is_dir():
    raise ValueError(f"invalid gradient capture directory: {capture}")
  manifest_path = capture / "manifest.json"
  manifest = _load_json(manifest_path)
  leaves = manifest.get("leaves")
  if (
      manifest.get("schema") != "canon-p61-full-tree-capture-v1"
      or manifest.get("capture") != "gradient"
      or manifest.get("leaf_count") != EXPECTED_LEAVES
      or not isinstance(leaves, list)
      or len(leaves) != EXPECTED_LEAVES
  ):
    raise ValueError(f"invalid full-gradient manifest: {manifest_path}")
  total_bytes = 0
  for index, leaf in enumerate(leaves):
    if not isinstance(leaf, dict) or leaf.get("index") != index:
      raise ValueError(f"non-contiguous leaf index at {index}")
    filename = f"leaf_{index:05d}.npy"
    if leaf.get("file") != filename:
      raise ValueError(f"unexpected leaf filename at {index}")
    path = capture / filename
    if path.is_symlink() or not path.is_file() or path.parent != capture:
      raise ValueError(f"invalid leaf path: {path}")
    if _sha256(path) != leaf.get("file_sha256"):
      raise ValueError(f"file SHA mismatch: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        list(array.shape) != leaf.get("shape")
        or str(array.dtype) != leaf.get("dtype")
        or int(array.size) != leaf.get("elements")
        or int(array.nbytes) != leaf.get("data_bytes")
        or _data_sha256(array) != leaf.get("data_sha256")
    ):
      raise ValueError(f"leaf metadata/data mismatch: {path}")
    total_bytes += int(array.nbytes)
  if total_bytes != manifest.get("total_data_bytes"):
    raise ValueError(f"manifest byte total mismatch: {manifest_path}")
  return leaves


def _schema(leaves: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
  return [
      (leaf["index"], leaf["path"], leaf["shape"], leaf["dtype"])
      for leaf in leaves
  ]


def _gradient_path(root: Path, leaf: dict[str, Any]) -> Path:
  return root / "train/p61_numerical/gradient" / leaf["file"]


def _leaf_metrics(
    control_path: Path, candidate_path: Path
) -> dict[str, Any]:
  control = np.load(control_path, mmap_mode="r", allow_pickle=False).reshape(-1)
  candidate = np.load(
      candidate_path, mmap_mode="r", allow_pickle=False
  ).reshape(-1)
  if control.shape != candidate.shape:
    raise ValueError(f"gradient shape mismatch: {control_path}")
  control_sq = 0.0
  candidate_sq = 0.0
  difference_sq = 0.0
  dot = 0.0
  finite = True
  for start in range(0, control.size, CHUNK_ELEMENTS):
    stop = min(start + CHUNK_ELEMENTS, control.size)
    left = np.asarray(control[start:stop], dtype=np.float64)
    right = np.asarray(candidate[start:stop], dtype=np.float64)
    finite = finite and bool(np.all(np.isfinite(left)))
    finite = finite and bool(np.all(np.isfinite(right)))
    difference = right - left
    control_sq += float(np.dot(left, left))
    candidate_sq += float(np.dot(right, right))
    difference_sq += float(np.dot(difference, difference))
    dot += float(np.dot(left, right))
  control_norm = math.sqrt(control_sq)
  candidate_norm = math.sqrt(candidate_sq)
  if control_norm == 0.0:
    cosine = 1.0 if candidate_norm == 0.0 else 0.0
    rel_l2 = 0.0 if candidate_norm == 0.0 else math.inf
  elif candidate_norm == 0.0:
    cosine = 0.0
    rel_l2 = 1.0
  else:
    cosine = max(
        -1.0, min(1.0, dot / (control_norm * candidate_norm))
    )
    rel_l2 = math.sqrt(difference_sq) / control_norm
  norm_ratio = (
      candidate_norm / control_norm if control_norm != 0.0 else 1.0
  )
  nearest_integer = round(norm_ratio)
  if (
      nearest_integer >= 2
      and cosine >= COSINE_MIN
      and abs(norm_ratio / nearest_integer - 1.0) <= 1.0e-6
  ):
    factor_class = "INTEGER_FACTOR"
  elif cosine >= COSINE_MIN and abs(norm_ratio - 1.0) <= 1.0e-5:
    factor_class = "REASSOC_NOISE"
  else:
    factor_class = "MISMATCH"
  return {
      "finite": finite,
      "cosine": cosine,
      "rel_l2": rel_l2,
      "control_norm": control_norm,
      "candidate_norm": candidate_norm,
      "norm_ratio": norm_ratio,
      "factor_class": factor_class,
  }


def _work_receipts(lines: list[str]) -> list[dict[str, Any]]:
  receipts = []
  for line in lines:
    if line.startswith(WORK_PREFIX):
      receipt = json.loads(line[len(WORK_PREFIX) :])
      if not isinstance(receipt, dict):
        raise ValueError("work receipt must be an object")
      receipts.append(receipt)
  return receipts


def _validate_contract(
    control_root: Path, candidate_root: Path
) -> tuple[list[str], dict[str, Any]]:
  reasons = []
  classifications = {
      "control": _load_json(control_root / "train/classification.json"),
      "candidate": _load_json(candidate_root / "train/classification.json"),
  }
  classifier_reasons = {}
  for arm, classification in classifications.items():
    arm_classifier_reasons = classification.get("reasons", [])
    if not isinstance(arm_classifier_reasons, list) or not all(
        isinstance(reason, str) for reason in arm_classifier_reasons
    ):
      reasons.append(f"{arm}.zero_tim_classification_reasons_schema")
      arm_classifier_reasons = []
    classifier_reasons[arm] = arm_classifier_reasons
    verdict = classification.get("verdict")
    classification_acceptable = (
        verdict == "PASS" and not arm_classifier_reasons
    )
    if (
        verdict == "FAIL"
        and arm_classifier_reasons == ["trace_census_rc=1"]
    ):
      trace_lines = _raw_lines(
          (control_root if arm == "control" else candidate_root)
          / "train/trace_census.txt"
      )
      event_counts = [
          int(match.group(1))
          for line in trace_lines
          if (match := TRACE_EVENT_COUNT_RE.match(line))
      ]
      known_capture_truncation = (
          len(event_counts) == 1
          and event_counts[0] >= TRACE_EVENT_TRUNCATION_FLOOR
          and any(line.startswith(TRACE_RED_MARKER) for line in trace_lines)
      )
      classification_acceptable = known_capture_truncation
    if not classification_acceptable:
      reasons.append(f"{arm}.zero_tim_classification")
    if classification.get("arm") != "zero-hp":
      reasons.append(f"{arm}.not_zero_hp")
    if classification.get("topology") != {"devices": 4, "dp": 2, "tp": 2}:
      reasons.append(f"{arm}.topology")
    capture = classification.get("capture", {})
    if capture.get("updates") != EXPECTED_UPDATES:
      reasons.append(f"{arm}.updates")
  identity_keys = (
      "source_sha",
      "source_diff_sha256",
      "runtime_manifest_sha256",
      "model_snapshot",
      "image_id",
  )
  identity = {
      key: classifications["control"].get(key) for key in identity_keys
  }
  for key in identity_keys:
    if (
        not identity[key]
        or classifications["candidate"].get(key) != identity[key]
    ):
      reasons.append(f"pair_identity.{key}")

  raw = {
      "control": _raw_lines(control_root / "train/raw.log"),
      "candidate": _raw_lines(candidate_root / "train/raw.log"),
  }
  work = {arm: _work_receipts(lines) for arm, lines in raw.items()}
  for arm in ("control", "candidate"):
    if len(work[arm]) != EXPECTED_UPDATES:
      reasons.append(f"{arm}.work_receipts={len(work[arm])}")
  if (
      len(work["control"]) == EXPECTED_UPDATES
      and len(work["candidate"]) == EXPECTED_UPDATES
      and work["control"][0] != work["candidate"][0]
  ):
    reasons.append("update0.frozen_input_mismatch")

  updates = {
      "control": _load_jsonl(control_root / "train/updates.jsonl"),
      "candidate": _load_jsonl(candidate_root / "train/updates.jsonl"),
  }
  for arm, expected_transactions in (("control", 32), ("candidate", 1)):
    if len(updates[arm]) != EXPECTED_UPDATES:
      reasons.append(f"{arm}.update_rows={len(updates[arm])}")
      continue
    for index, update in enumerate(updates[arm]):
      if (
          update.get("verdict") != "PASS"
          or update.get("dp_size") != 2
          or update.get("tp_size") != 2
          or update.get("microsteps") != EXPECTED_GROUPS
          or update.get("dp_reduction_transactions")
          != expected_transactions
          or update.get("gradient_finite") is not True
          or update.get("train_steps_before") != index
          or update.get("train_steps_after") != index + 1
      ):
        reasons.append(f"{arm}.update_contract[{index}]")

  control_groups = [line for line in raw["control"] if GROUP_RE.match(line)]
  if len(control_groups) != EXPECTED_UPDATES * EXPECTED_GROUPS:
    reasons.append(f"g5.control_groups={len(control_groups)}")
  candidate_staged = [line for line in raw["candidate"] if STAGED_RE.match(line)]
  if len(candidate_staged) != EXPECTED_UPDATES:
    reasons.append(f"g5.candidate_staged={len(candidate_staged)}")

  vma_counts = {
      arm: sum(line.startswith(VMA_PREFIX) for line in lines)
      for arm, lines in raw.items()
  }
  for arm, count in vma_counts.items():
    if count != EXPECTED_VMA_MODULES:
      reasons.append(f"g6.{arm}.outer_vma={count}")
  checked_reductions = sum(
      bool(REDUCE_RE.match(line)) for line in raw["candidate"]
  )
  if checked_reductions != EXPECTED_UPDATES:
    reasons.append(f"g6.candidate_checked_reductions={checked_reductions}")

  return reasons, {
      "identity": identity,
      "classifier_reasons": classifier_reasons,
      "work_receipts": {arm: len(rows) for arm, rows in work.items()},
      "update_transactions": {
          arm: [row.get("dp_reduction_transactions") for row in rows]
          for arm, rows in updates.items()
      },
      "g5": {
          "control_distinct_groups": len(control_groups),
          "candidate_distinct_updates": len(candidate_staged),
      },
      "g6": {
          "outer_vma_modules": vma_counts,
          "candidate_checked_reductions": checked_reductions,
      },
  }


def compare(control_root: Path, candidate_root: Path) -> dict[str, Any]:
  contract_reasons, evidence = _validate_contract(
      control_root, candidate_root
  )
  if contract_reasons:
    zero_tim_red = any("zero_tim" in reason for reason in contract_reasons)
    return {
        "schema": "canon.v2.phase0.dp2-reduce-once-admission.v1",
        "verdict": (
            "ZERO_TIM_REJECT" if zero_tim_red else "INCONCLUSIVE_CARRIER"
        ),
        "reasons": contract_reasons,
        "evidence": evidence,
    }

  control_leaves = _load_capture(control_root)
  candidate_leaves = _load_capture(candidate_root)
  if _schema(control_leaves) != _schema(candidate_leaves):
    raise ValueError("control/candidate gradient schemas differ")

  failures = []
  factor_histogram: dict[str, int] = {}
  worst_cosine = 1.0
  worst_cosine_path = None
  worst_rel_l2 = 0.0
  worst_rel_l2_path = None
  live_leaves = 0
  for left, right in zip(control_leaves, candidate_leaves, strict=True):
    metrics = _leaf_metrics(
        _gradient_path(control_root, left),
        _gradient_path(candidate_root, right),
    )
    factor = metrics["factor_class"]
    factor_histogram[factor] = factor_histogram.get(factor, 0) + 1
    if metrics["control_norm"] > 0.0:
      live_leaves += 1
      if metrics["cosine"] < worst_cosine:
        worst_cosine = metrics["cosine"]
        worst_cosine_path = left["path"]
      if metrics["rel_l2"] > worst_rel_l2:
        worst_rel_l2 = metrics["rel_l2"]
        worst_rel_l2_path = left["path"]
    if (
        not metrics["finite"]
        or metrics["cosine"] < COSINE_MIN
        or metrics["rel_l2"] > REL_L2_MAX
    ):
      failures.append({"path": left["path"], **metrics})

  evidence["g4"] = {
      "thresholds": {"cosine_min": COSINE_MIN, "rel_l2_max": REL_L2_MAX},
      "leaves": len(control_leaves),
      "live_leaves": live_leaves,
      "worst_cosine": worst_cosine,
      "worst_cosine_path": worst_cosine_path,
      "worst_rel_l2": worst_rel_l2,
      "worst_rel_l2_path": worst_rel_l2_path,
      "factor_histogram": factor_histogram,
      "failures": failures[:20],
  }
  return {
      "schema": "canon.v2.phase0.dp2-reduce-once-admission.v1",
      "verdict": "PASS" if not failures else "NUMERICAL_REJECT",
      "reasons": (
          [] if not failures else [f"g4.leaf_failures={len(failures)}"]
      ),
      "evidence": evidence,
  }


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--control-run", type=Path, required=True)
  parser.add_argument("--candidate-run", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  try:
    result = compare(args.control_run.resolve(), args.candidate_run.resolve())
  except Exception as exc:  # fail closed and preserve a machine-readable result
    result = {
        "schema": "canon.v2.phase0.dp2-reduce-once-admission.v1",
        "verdict": "INCONCLUSIVE_CARRIER",
        "reasons": [f"{type(exc).__name__}: {exc}"],
    }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n",
      encoding="utf-8",
  )
  print(
      "[BWDHEALTH] "
      f"{'PASS' if result['verdict'] == 'PASS' else 'FAIL'} "
      "check=dp2_reduce_once_admission "
      f"verdict={result['verdict']} output={args.output}"
  )
  if result["verdict"] == "PASS":
    return 0
  if result["verdict"] == "NUMERICAL_REJECT":
    return 1
  return 2


if __name__ == "__main__":
  sys.exit(main())
