"""Tests for the Phase-0 DP2 reduce-once numerical carrier."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
from unittest import mock

import numpy as np


_SCRIPT = Path(__file__).with_name("compare_dp2_reduce_once.py")
_SPEC = importlib.util.spec_from_file_location("compare_dp2_reduce_once", _SCRIPT)
comparator = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(comparator)


def _write_json(path: Path, value):
  path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_capture(root: Path, arrays):
  capture = root / "train/p61_numerical/gradient"
  capture.mkdir(parents=True)
  leaves = []
  for index, value in enumerate(arrays):
    array = np.asarray(value, dtype=np.float32)
    filename = f"leaf_{index:05d}.npy"
    path = capture / filename
    np.save(path, array, allow_pickle=False)
    leaves.append({
        "index": index,
        "path": f"['leaf_{index}']",
        "file": filename,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "elements": int(array.size),
        "data_bytes": int(array.nbytes),
        "data_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    })
  _write_json(
      capture / "manifest.json",
      {
          "schema": "canon-p61-full-tree-capture-v1",
          "capture": "gradient",
          "leaf_count": len(leaves),
          "total_data_bytes": sum(leaf["data_bytes"] for leaf in leaves),
          "leaves": leaves,
      },
  )


def _make_run(
    root: Path,
    *,
    candidate: bool,
    arrays,
    classifier_reasons=(),
):
  identity = {
      "source_sha": "a" * 40,
      "source_diff_sha256": "b" * 64,
      "runtime_manifest_sha256": "c" * 64,
      "model_snapshot": "d" * 40,
      "image_id": "sha256:" + "e" * 64,
  }
  _write_json(
      root / "train/classification.json",
      {
          "verdict": "FAIL" if classifier_reasons else "PASS",
          "reasons": list(classifier_reasons),
          "arm": "zero-hp",
          "topology": {"devices": 4, "dp": 2, "tp": 2},
          "capture": {"updates": 3},
          **identity,
      },
  )
  updates = []
  for index in range(3):
    updates.append(json.dumps({
        "verdict": "PASS",
        "dp_size": 2,
        "tp_size": 2,
        "microsteps": 32,
        "dp_reduction_transactions": 1 if candidate else 32,
        "gradient_finite": True,
        "train_steps_before": index,
        "train_steps_after": index + 1,
    }))
  (root / "train/updates.jsonl").write_text(
      "\n".join(updates) + "\n", encoding="utf-8"
  )
  raw = []
  for index in range(3):
    raw.append(
        comparator.WORK_PREFIX
        + json.dumps({"schema": "work.v1", "train_step": index})
    )
  raw.extend(
      comparator.VMA_PREFIX + f"module=module_{index} manual_axes=['data']"
      for index in range(31)
  )
  if candidate:
    for _ in range(3):
      raw.append(
          "[P59.DP2] staged_rank_contributions_checked "
          "groups=32/32 unique_min=2/2 unique_max=2/2"
      )
      raw.append(
          "[P59.DP2] update_reduce_done check_vma=1 "
          "collectives=2 replicas_exact=1"
      )
  else:
    for _ in range(3):
      for group in range(1, 33):
        raw.append(
            f"[P33.DP2] reverse_group_done group={group}/32 rows=(0, 1) "
            "rank_contributions=2 pullback_invocations=1 "
            "unique_rank_fingerprints=2/2 reduction_rounds=2"
        )
  (root / "train/raw.log").write_text(
      "\n".join(raw) + "\n", encoding="utf-8"
  )
  if classifier_reasons == ("trace_census_rc=1",):
    (root / "train/trace_census.txt").write_text(
        "trace_event_count=1000000\n"
        "V1_GSM8K_XPROF_TRACE_CENSUS_RED reasons=1\n",
        encoding="utf-8",
    )
  _write_capture(root, arrays)


def test_compare_accepts_leafwise_g4_and_g5_g6_receipts(tmp_path):
  control = tmp_path / "control"
  candidate = tmp_path / "candidate"
  _make_run(
      control,
      candidate=False,
      arrays=(np.asarray([1.0, 2.0]), np.zeros(3)),
  )
  _make_run(
      candidate,
      candidate=True,
      arrays=(np.asarray([1.0, 2.0 + 1.0e-7]), np.zeros(3)),
  )
  with mock.patch.object(comparator, "EXPECTED_LEAVES", 2):
    result = comparator.compare(control, candidate)
  assert result["verdict"] == "PASS"
  assert result["evidence"]["g4"]["live_leaves"] == 1
  assert result["evidence"]["g5"] == {
      "control_distinct_groups": 96,
      "candidate_distinct_updates": 3,
  }
  assert result["evidence"]["g6"]["candidate_checked_reductions"] == 3


def test_compare_accepts_only_the_canonical_trace_census_reason(tmp_path):
  control = tmp_path / "control"
  candidate = tmp_path / "candidate"
  _make_run(
      control,
      candidate=False,
      arrays=(np.asarray([1.0, 2.0]),),
      classifier_reasons=("trace_census_rc=1",),
  )
  _make_run(
      candidate,
      candidate=True,
      arrays=(np.asarray([1.0, 2.0]),),
      classifier_reasons=("trace_census_rc=1",),
  )
  with mock.patch.object(comparator, "EXPECTED_LEAVES", 1):
    result = comparator.compare(control, candidate)
  assert result["verdict"] == "PASS"
  assert result["evidence"]["classifier_reasons"] == {
      "control": ["trace_census_rc=1"],
      "candidate": ["trace_census_rc=1"],
  }


def test_compare_rejects_any_other_classifier_reason(tmp_path):
  control = tmp_path / "control"
  candidate = tmp_path / "candidate"
  _make_run(
      control,
      candidate=False,
      arrays=(np.asarray([1.0, 2.0]),),
      classifier_reasons=("trace_census_rc=1",),
  )
  _make_run(
      candidate,
      candidate=True,
      arrays=(np.asarray([1.0, 2.0]),),
      classifier_reasons=("alignment_red",),
  )
  with mock.patch.object(comparator, "EXPECTED_LEAVES", 1):
    result = comparator.compare(control, candidate)
  assert result["verdict"] == "ZERO_TIM_REJECT"
  assert result["reasons"] == ["candidate.zero_tim_classification"]


def test_compare_rejects_trace_failure_without_capture_truncation(tmp_path):
  control = tmp_path / "control"
  candidate = tmp_path / "candidate"
  _make_run(
      control,
      candidate=False,
      arrays=(np.asarray([1.0, 2.0]),),
      classifier_reasons=("trace_census_rc=1",),
  )
  _make_run(
      candidate,
      candidate=True,
      arrays=(np.asarray([1.0, 2.0]),),
      classifier_reasons=("trace_census_rc=1",),
  )
  (candidate / "train/trace_census.txt").write_text(
      "trace_event_count=999999\n"
      "V1_GSM8K_XPROF_TRACE_CENSUS_RED reasons=1\n",
      encoding="utf-8",
  )
  with mock.patch.object(comparator, "EXPECTED_LEAVES", 1):
    result = comparator.compare(control, candidate)
  assert result["verdict"] == "ZERO_TIM_REJECT"
  assert result["reasons"] == ["candidate.zero_tim_classification"]


def test_compare_rejects_an_integer_factor_gradient(tmp_path):
  control = tmp_path / "control"
  candidate = tmp_path / "candidate"
  _make_run(control, candidate=False, arrays=(np.asarray([1.0, 2.0]),))
  _make_run(candidate, candidate=True, arrays=(np.asarray([2.0, 4.0]),))
  with mock.patch.object(comparator, "EXPECTED_LEAVES", 1):
    result = comparator.compare(control, candidate)
  assert result["verdict"] == "NUMERICAL_REJECT"
  assert result["evidence"]["g4"]["factor_histogram"] == {
      "INTEGER_FACTOR": 1
  }


def test_runner_scopes_full_tree_capture_without_changing_default_bundle():
  runner = (
      Path(__file__).parents[2]
      / "tasks/v1-gsm8k-onehost-xprof-pair/scripts"
      / "run_onehost_gsm8k_xprof_common.sh"
  ).read_text(encoding="utf-8")
  normalized_runner = re.sub(r"\\\n\s*", "", runner)
  normalized_runner = re.sub(r"\s+", " ", normalized_runner)
  assert 'v2_p0_capture_full_tree="${V2_P0_CAPTURE_FULL_TREE:-0}"' in runner
  assert (
      'if [ "$arm" != zero-hp ] || [ "$geometry" != dp2-tp2 ] || '
      '[ "$run_stage" != three-update ]; then' in normalized_runner
  )
  assert (
      '-e CANON_P61_BACKWARD_NUMERICAL_DIR='
      '"$state/p61_numerical"' in runner
  )
  assert '"$repo/tunix/rl/dp_training.py"' in runner
  assert 'v2_p0_negative_control="${V2_P0_NEGATIVE_CONTROL:-}"' in runner
  assert "length-sort-no-inverse requires CANON_P32_LENGTH_SORT=1" in runner
  assert (
      "reduce-once-reassociate-tail requires CANON_DP_REDUCE_ONCE=1"
      in runner
  )
  assert '-e V2_P0_NEGATIVE_CONTROL="$v2_p0_negative_control"' in runner
  assert "v2_p0_performance_eligible=1" in runner
  assert "v2_p0_performance_eligible=0" in runner
  assert "performance_eligible=$v2_p0_performance_eligible" in runner
