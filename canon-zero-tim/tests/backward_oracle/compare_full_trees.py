#!/usr/bin/env python3
"""Fail-closed P61 full-gradient and real-update numerical comparator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np


HASH_KEYS = (
    "S_decode",
    "S_prefill",
    "T_current",
    "T_old",
    "action_mask",
    "policy_version",
    "tokens",
)
METRIC_KEYS = (
    "rel_l2",
    "one_minus_cos",
    "norm_ratio_error",
    "sign_mismatch_rate",
)
ABSOLUTE_CAPS = {
    "rel_l2": 4.0e-2,
    "one_minus_cos": 3.2e-4,
    "norm_ratio_error": 4.0e-2,
    "sign_mismatch_rate": 2.0e-2,
}
CHUNK_ELEMENTS = 1_048_576


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
    stop = min(start + CHUNK_ELEMENTS, flattened.size)
    digest.update(
        np.ascontiguousarray(flattened[start:stop]).tobytes(order="C")
    )
  return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise ValueError(f"{path}: expected JSON object")
  return value


def _load_one_jsonl(path: Path) -> dict[str, Any]:
  rows = [
      json.loads(line)
      for line in path.read_text(encoding="utf-8").splitlines()
      if line.strip()
  ]
  if len(rows) != 1 or not isinstance(rows[0], dict):
    raise ValueError(f"{path}: expected exactly one JSON object")
  return rows[0]


def _load_capture(root: Path, name: str) -> dict[str, Any]:
  capture_dir = root / name
  manifest_path = capture_dir / "manifest.json"
  if capture_dir.is_symlink() or not capture_dir.is_dir():
    raise ValueError(f"invalid capture directory: {capture_dir}")
  manifest = _load_json(manifest_path)
  if (
      manifest.get("schema") != "canon-p61-full-tree-capture-v1"
      or manifest.get("capture") != name
  ):
    raise ValueError(f"invalid P61 manifest contract: {manifest_path}")
  leaves = manifest.get("leaves")
  if (
      not isinstance(leaves, list)
      or not leaves
      or manifest.get("leaf_count") != len(leaves)
  ):
    raise ValueError(f"invalid P61 manifest leaves: {manifest_path}")
  total_bytes = 0
  for index, leaf in enumerate(leaves):
    if not isinstance(leaf, dict) or leaf.get("index") != index:
      raise ValueError(f"non-contiguous P61 manifest index at {index}")
    filename = leaf.get("file")
    if filename != f"leaf_{index:05d}.npy":
      raise ValueError(f"unexpected P61 leaf filename: {filename!r}")
    path = capture_dir / filename
    if path.is_symlink() or not path.is_file() or path.parent != capture_dir:
      raise ValueError(f"invalid P61 leaf path: {path}")
    if _sha256(path) != leaf.get("file_sha256"):
      raise ValueError(f"P61 leaf SHA mismatch: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        list(array.shape) != leaf.get("shape")
        or str(array.dtype) != leaf.get("dtype")
        or int(array.size) != leaf.get("elements")
        or int(array.nbytes) != leaf.get("data_bytes")
    ):
      raise ValueError(f"P61 leaf metadata mismatch: {path}")
    if _data_sha256(array) != leaf.get("data_sha256"):
      raise ValueError(f"P61 leaf data SHA mismatch: {path}")
    total_bytes += int(array.nbytes)
  if total_bytes != manifest.get("total_data_bytes"):
    raise ValueError(f"P61 manifest byte total mismatch: {manifest_path}")
  return manifest


def _compatible(
    manifests: tuple[dict[str, Any], ...], *, label: str
) -> list[dict[str, Any]]:
  reference = manifests[0]["leaves"]
  identity = [
      (leaf["index"], leaf["path"], leaf["shape"], leaf["dtype"])
      for leaf in reference
  ]
  for manifest in manifests[1:]:
    observed = [
        (leaf["index"], leaf["path"], leaf["shape"], leaf["dtype"])
        for leaf in manifest["leaves"]
    ]
    if observed != identity:
      raise ValueError(f"P61 {label} tree schemas differ")
  return reference


def _array(root: Path, name: str, leaf: dict[str, Any]) -> np.ndarray:
  return np.load(
      root / name / leaf["file"], mmap_mode="r", allow_pickle=False
  ).reshape(-1)


def _tree_metrics(
    leaves: list[dict[str, Any]],
    reference_value: Callable[[dict[str, Any]], np.ndarray],
    candidate_value: Callable[[dict[str, Any]], np.ndarray],
) -> dict[str, Any]:
  ref_squared = 0.0
  got_squared = 0.0
  diff_squared = 0.0
  dot = 0.0
  sign_mismatch = 0
  reference_nonzero = 0
  finite = True
  live_reference_leaves = 0
  dead_candidate_leaves = []
  worst_leaf_scaled_max_error = 0.0
  worst_leaf_path = None
  elements = 0
  for leaf in leaves:
    reference = reference_value(leaf)
    candidate = candidate_value(leaf)
    if reference.shape != candidate.shape:
      raise ValueError(f"P61 value shape mismatch at {leaf['path']}")
    leaf_ref_squared = 0.0
    leaf_got_squared = 0.0
    leaf_max_ref = 0.0
    leaf_max_diff = 0.0
    for start in range(0, reference.size, CHUNK_ELEMENTS):
      stop = min(start + CHUNK_ELEMENTS, reference.size)
      ref = np.asarray(reference[start:stop], dtype=np.float64)
      got = np.asarray(candidate[start:stop], dtype=np.float64)
      finite = finite and bool(np.all(np.isfinite(ref)))
      finite = finite and bool(np.all(np.isfinite(got)))
      diff = got - ref
      ref_squared_chunk = float(np.dot(ref, ref))
      got_squared_chunk = float(np.dot(got, got))
      leaf_ref_squared += ref_squared_chunk
      leaf_got_squared += got_squared_chunk
      ref_squared += ref_squared_chunk
      got_squared += got_squared_chunk
      diff_squared += float(np.dot(diff, diff))
      dot += float(np.dot(ref, got))
      nonzero = ref != 0.0
      reference_nonzero += int(np.count_nonzero(nonzero))
      sign_mismatch += int(np.count_nonzero(
          nonzero & (np.sign(ref) != np.sign(got))
      ))
      leaf_max_ref = max(leaf_max_ref, float(np.max(np.abs(ref), initial=0.0)))
      leaf_max_diff = max(
          leaf_max_diff, float(np.max(np.abs(diff), initial=0.0))
      )
      elements += int(ref.size)
    if leaf_ref_squared > 0.0:
      live_reference_leaves += 1
      if leaf_got_squared == 0.0:
        dead_candidate_leaves.append(leaf["path"])
    leaf_scaled = leaf_max_diff / max(
        leaf_max_ref, np.finfo(np.float64).tiny
    )
    if leaf_scaled > worst_leaf_scaled_max_error:
      worst_leaf_scaled_max_error = leaf_scaled
      worst_leaf_path = leaf["path"]
  ref_norm = math.sqrt(ref_squared)
  got_norm = math.sqrt(got_squared)
  diff_norm = math.sqrt(diff_squared)
  if ref_norm == 0.0:
    rel_l2 = 0.0 if got_norm == 0.0 else math.inf
    one_minus_cos = 0.0 if got_norm == 0.0 else math.inf
    norm_ratio_error = 0.0 if got_norm == 0.0 else math.inf
  elif got_norm == 0.0:
    rel_l2 = 1.0
    one_minus_cos = 1.0
    norm_ratio_error = 1.0
  else:
    rel_l2 = diff_norm / ref_norm
    cosine = max(-1.0, min(1.0, dot / (ref_norm * got_norm)))
    one_minus_cos = max(0.0, 1.0 - cosine)
    norm_ratio_error = abs(got_norm / ref_norm - 1.0)
  return {
      "rel_l2": rel_l2,
      "one_minus_cos": one_minus_cos,
      "norm_ratio_error": norm_ratio_error,
      "sign_mismatch_rate": sign_mismatch / max(1, reference_nonzero),
      "sign_mismatch_elements": sign_mismatch,
      "reference_nonzero_elements": reference_nonzero,
      "finite": finite,
      "elements": elements,
      "live_reference_leaves": live_reference_leaves,
      "dead_candidate_leaves": dead_candidate_leaves,
      "worst_leaf_scaled_max_error": worst_leaf_scaled_max_error,
      "worst_leaf_path": worst_leaf_path,
      "reference_norm": ref_norm,
      "candidate_norm": got_norm,
      "difference_norm": diff_norm,
  }


def compare(
    *,
    control_root: Path,
    candidate_root: Path,
    control_update: Path,
    candidate_update: Path,
    control_classification: Path,
    candidate_classification: Path,
    tier1_baseline: Path,
) -> dict[str, Any]:
  contract_reasons = []
  zero_tim_failure = False
  classifications = {
      "control": _load_json(control_classification),
      "candidate": _load_json(candidate_classification),
  }
  for arm, classification in classifications.items():
    zero_tim = classification.get("zero_tim", {})
    if zero_tim.get("observed_fail", 0) != 0:
      zero_tim_failure = True
      contract_reasons.append(f"{arm}.zero_tim_fail")
    if (
        classification.get("verdict") != "PASS"
        or zero_tim.get("expected_pass") != 17
        or zero_tim.get("observed_pass") != 17
    ):
      contract_reasons.append(f"{arm}.classification")

  updates = {
      "control": _load_one_jsonl(control_update),
      "candidate": _load_one_jsonl(candidate_update),
  }
  for arm, update in updates.items():
    expected_invocations = 4 if arm == "control" else 1
    evidence = update.get("commit_evidence", {})
    if (
        update.get("verdict") != "PASS"
        or update.get("dp_size") != 4
        or update.get("tp_size") != 1
        or update.get("dp_pullback_invocations_per_transaction")
        != expected_invocations
        or not isinstance(evidence, dict)
        or not isinstance(evidence.get("effective_learning_rate"), (int, float))
        or evidence["effective_learning_rate"] <= 0.0
        or not isinstance(evidence.get("parameter_changed_elements"), int)
        or evidence["parameter_changed_elements"] <= 0
    ):
      contract_reasons.append(f"{arm}.update_contract")
  control_hashes = updates["control"].get("alignment_hashes")
  candidate_hashes = updates["candidate"].get("alignment_hashes")
  hash_schema_ok = (
      isinstance(control_hashes, list)
      and isinstance(candidate_hashes, list)
      and len(control_hashes) == len(candidate_hashes) == 16
      and all(tuple(sorted(row)) == tuple(sorted(HASH_KEYS)) for row in control_hashes)
      and all(tuple(sorted(row)) == tuple(sorted(HASH_KEYS)) for row in candidate_hashes)
  )
  if not hash_schema_ok or control_hashes != candidate_hashes:
    contract_reasons.append("same_input_seven_hashes")
  if (
      updates["control"].get("state_fingerprints_before")
      != updates["candidate"].get("state_fingerprints_before")
  ):
    contract_reasons.append("sampled_prestate_fingerprints")

  captures = {}
  for arm, root in (("control", control_root), ("candidate", candidate_root)):
    captures[arm] = {
        name: _load_capture(root, name)
        for name in ("model_before", "gradient", "model_after")
    }
  gradient_leaves = _compatible(
      (captures["control"]["gradient"], captures["candidate"]["gradient"]),
      label="gradient",
  )
  model_leaves = _compatible(
      (
          captures["control"]["model_before"],
          captures["candidate"]["model_before"],
          captures["control"]["model_after"],
          captures["candidate"]["model_after"],
      ),
      label="model",
  )
  model_before_exact = all(
      left["data_sha256"] == right["data_sha256"]
      for left, right in zip(
          captures["control"]["model_before"]["leaves"],
          captures["candidate"]["model_before"]["leaves"],
          strict=True,
      )
  )
  if not model_before_exact:
    contract_reasons.append("full_model_prestate")

  gradient = _tree_metrics(
      gradient_leaves,
      lambda leaf: _array(control_root, "gradient", leaf),
      lambda leaf: _array(candidate_root, "gradient", leaf),
  )

  def parameter_delta(root: Path, leaf: dict[str, Any]) -> np.ndarray:
    before = _array(root, "model_before", leaf)
    after = _array(root, "model_after", leaf)
    # Subtraction is intentionally deferred to the metric loop, where slices
    # are promoted to FP64. Returning a tiny proxy object keeps the full model
    # memory-mapped rather than materializing a second model-sized array.
    class _Delta:

      def __init__(self):
        self.shape = before.shape
        self.size = before.size

      def __getitem__(self, item):
        return (
            np.asarray(after[item], dtype=np.float64)
            - np.asarray(before[item], dtype=np.float64)
        )

    return _Delta()  # type: ignore[return-value]

  parameter_update = _tree_metrics(
      model_leaves,
      lambda leaf: parameter_delta(control_root, leaf),
      lambda leaf: parameter_delta(candidate_root, leaf),
  )

  baseline = _load_json(tier1_baseline)
  if baseline.get("schema") != "canon-p61-tier1-baseline-v1":
    raise ValueError("invalid P61 Tier-1 baseline schema")
  thresholds = {}
  numerical_reasons = []
  for tree_name, metrics in (
      ("gradient", gradient),
      ("parameter_update", parameter_update),
  ):
    baseline_metrics = baseline.get(tree_name)
    if not isinstance(baseline_metrics, dict):
      raise ValueError(f"missing Tier-1 baseline metrics for {tree_name}")
    thresholds[tree_name] = {}
    for metric_name in METRIC_KEYS:
      baseline_value = baseline_metrics.get(metric_name)
      if (
          not isinstance(baseline_value, (int, float))
          or not math.isfinite(baseline_value)
          or baseline_value < 0.0
      ):
        raise ValueError(
            f"invalid Tier-1 baseline {tree_name}.{metric_name}"
        )
      threshold = min(2.0 * baseline_value, ABSOLUTE_CAPS[metric_name])
      thresholds[tree_name][metric_name] = threshold
      if not math.isfinite(metrics[metric_name]) or metrics[metric_name] > threshold:
        numerical_reasons.append(
            f"{tree_name}.{metric_name}={metrics[metric_name]:.9e}>"
            f"{threshold:.9e}"
        )
    if not metrics["finite"]:
      numerical_reasons.append(f"{tree_name}.nonfinite")
    if metrics["dead_candidate_leaves"]:
      numerical_reasons.append(f"{tree_name}.dead_candidate_leaves")

  if zero_tim_failure:
    verdict = "REJECT_ZERO_TIM"
  elif contract_reasons:
    verdict = "INCONCLUSIVE_CARRIER"
  elif numerical_reasons:
    verdict = "NUMERICAL_REJECT"
  else:
    verdict = "NUMERICAL_KEEP_DP4_PROXY"
  return {
      "schema": "canon-p61-full-tree-ab-v1",
      "verdict": verdict,
      "scope": "Qwen3-1.7B DP4xTP1 one update",
      "model_before_array_exact": model_before_exact,
      "same_input_seven_hashes": hash_schema_ok and control_hashes == candidate_hashes,
      "gradient": gradient,
      "parameter_update": parameter_update,
      "thresholds": thresholds,
      "tier1_baseline_sha256": _sha256(tier1_baseline),
      "evidence_sha256": {
          "control_update": _sha256(control_update),
          "candidate_update": _sha256(candidate_update),
          "control_classification": _sha256(control_classification),
          "candidate_classification": _sha256(candidate_classification),
      },
      "contract_reasons": contract_reasons,
      "numerical_reasons": numerical_reasons,
  }


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--control-root", required=True, type=Path)
  parser.add_argument("--candidate-root", required=True, type=Path)
  parser.add_argument("--control-update", required=True, type=Path)
  parser.add_argument("--candidate-update", required=True, type=Path)
  parser.add_argument("--control-classification", required=True, type=Path)
  parser.add_argument("--candidate-classification", required=True, type=Path)
  parser.add_argument("--tier1-baseline", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  if args.output.exists():
    raise FileExistsError(f"refusing to overwrite {args.output}")
  result = compare(
      control_root=args.control_root,
      candidate_root=args.candidate_root,
      control_update=args.control_update,
      candidate_update=args.candidate_update,
      control_classification=args.control_classification,
      candidate_classification=args.candidate_classification,
      tier1_baseline=args.tier1_baseline,
  )
  args.output.write_text(
      json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
  print(
      "P61_FULL_TREE_AB "
      f"verdict={result['verdict']} "
      f"gradient_rel_l2={result['gradient']['rel_l2']:.9e} "
      f"update_rel_l2={result['parameter_update']['rel_l2']:.9e}"
  )
  return 0 if result["verdict"] == "NUMERICAL_KEEP_DP4_PROXY" else 1


if __name__ == "__main__":
  raise SystemExit(main())
