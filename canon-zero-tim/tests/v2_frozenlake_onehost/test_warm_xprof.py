"""Contracts for the P45 R1/R2 same-process warm XProf vehicle."""

from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from tunix.rl.agentic import agentic_rl_learner


ROOT = Path(__file__).resolve().parents[3]
CLASSIFIER_PATH = (
    ROOT
    / "canon-zero-tim/tasks/v2-frozenlake-onehost/scripts/"
    "classify_frozenlake_warm_xprof.py"
)
_SPEC = importlib.util.spec_from_file_location("v2_fl_warm_xprof", CLASSIFIER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
CLASSIFIER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(CLASSIFIER)
CENSUS_PATH = (
    ROOT
    / "canon-zero-tim/tasks/v2-frozenlake-onehost/scripts/"
    "census_frozenlake_warm_xprof.py"
)
_CENSUS_SPEC = importlib.util.spec_from_file_location(
    "v2_fl_warm_xprof_census", CENSUS_PATH
)
assert _CENSUS_SPEC is not None and _CENSUS_SPEC.loader is not None
CENSUS = importlib.util.module_from_spec(_CENSUS_SPEC)
sys.modules[_CENSUS_SPEC.name] = CENSUS
_CENSUS_SPEC.loader.exec_module(CENSUS)


def _profile_env(arm: str = "r2") -> dict[str, str]:
  return {
      "V2_FL_MODE": "profile",
      "V2_FL_ARM": arm,
      "V2_FL_WORKLOAD": "p45",
      "V2_FL_CAPSULE_MODE": "replay",
      "CANON_P32_WORKLOAD": "frozenlake-p45-onehost-dp2-tp2",
      "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
      "CANON_P33_NO_COMMIT": "1",
      "CANON_P33_RUN_STAGE": "backward-no-commit",
      "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
      "CANON_P66_P59_CHECK_VMA": "1",
      "CANON_P32_KEEP_TAPE": "stream",
      "CANON_DP_REDUCE_ONCE": "0" if arm == "r1" else "1",
      "CANON_P32_LENGTH_SORT": "0",
      "CANON_P75_REPORT_ADJOINT_BUCKETS": "0",
      "CANON_P76_CHUNK_DEPENDENCY_TICKET": "0",
      "CANON_P77_CHUNK_BACKPRESSURE": "0",
      "CANON_XPROF_SKIP_STEPS": "0",
      "CANON_XPROF_STEPS": "1",
      "CANON_XPROF_PHASE": "update",
      "CANON_XPROF_HOST_TRACER": "1",
      "CANON_XPROF_PYTHON_TRACER": "0",
      "CANON_XPROF_TPU_TRACE_MODE": "TRACE_ONLY_XLA",
      "CANON_XPROF_LABELS": "1",
      "CANON_XPROF_DIR": "/tmp/v2-fl-xprof",
      "CANON_PERF_TRACE_DIR": "/tmp/v2-fl-perfetto",
      "V2_FL_XPROF_REPORT": "/tmp/v2-fl-repeat.json",
  }


def _result(arm: str = "r2") -> dict:
  state = {"tree": {"sha256": "a" * 64}}
  result = {
      "verdict": "PASS",
      "commits": 0,
      "train_steps_before": 0,
      "train_steps_after": 0,
      "model_changed_paths": [],
      "optimizer_changed_paths": [],
      "accumulator_changed_paths": [],
      "reference_changed_paths": [],
      "state_fingerprints_before": state,
      "state_fingerprints_after": copy.deepcopy(state),
      "micro_gradient_norms": [1.0, 2.0, 0.0, 0.0],
      "alignment_hashes": [{"A": "b" * 64}, {"A": "c" * 64}],
      "hbm_before": [{
          "bytes_in_use": 10,
          "peak_bytes_in_use": 70,
          "bytes_limit": 100,
      }],
      "hbm_after_reverse": [{
          "bytes_in_use": 80,
          "peak_bytes_in_use": 80,
          "bytes_limit": 100,
      }],
  }
  if arm == "r2":
    result["update_gradient_norm"] = 3.0
  return result


def _alignment() -> dict:
  boundary = {
      "valid": True,
      "finite": True,
      "differing_bytes": 0,
      "differing_elements": 0,
      "max_abs": 0.0,
  }
  return {"verdict": "PASS", "boundaries": {"A": boundary}}


def _write_classifier_fixture(root: Path, arm: str) -> Path:
  result = _result(arm)
  repeat = agentic_rl_learner._canon_v2_frozenlake_profile_report(
      arm, result, copy.deepcopy(result)
  )
  capsule_sha = "9" * 64
  manifest = {
      "schema": "canon.v2-frozenlake-onehost.run.v1",
      "workload": "p45",
      "workload_name": "frozenlake-p45-onehost-dp2-tp2",
      "arm": arm,
      "classification_mode": "profile",
      "xprof": {
          "phase": "update",
          "skip_steps": 0,
          "steps": 1,
          "host_tracer": 1,
          "python_tracer": 0,
          "tpu_trace_mode": "TRACE_ONLY_XLA",
          "labels": 1,
      },
      "stage": "backward-no-commit",
      "topology": {"dp": 2, "tp": 2, "devices": 4},
      "selectors": {
          "keep_tape": "stream",
          "reduce_once": "0" if arm == "r1" else "1",
          "length_sort": "0",
          "report_adjoint_buckets": "0",
          "chunk_dependency_ticket": "0",
          "chunk_backpressure": "0",
      },
      "checked_vma": True,
      "training_capsule": {"mode": "replay", "sha256": capsule_sha},
  }
  (root / "run_manifest.json").write_text(json.dumps(manifest))
  (root / "runtime.json").write_text(json.dumps({"verdict": "PASS"}))
  (root / "pre_alignment.jsonl").write_text(json.dumps(_alignment()) + "\n")
  (root / "alignment.jsonl").write_text(
      "".join(json.dumps(_alignment()) + "\n" for _ in range(16))
  )
  (root / "updates.json").write_text(json.dumps(result))
  (root / "warm_xprof_repeat.json").write_text(json.dumps(repeat))
  raw = ["[P66.VMA] outer_check_enabled"] * 39
  raw.extend((
      "[V2.FL.XPROF] warmup_complete",
      "[V2.FL.XPROF] phase=update armed",
      "[V2.FL.XPROF] phase=update stopped",
      "[V2.FL.XPROF] repeat_complete",
      "[PERF] stage=p32_vag_reverse seconds=100.000",
      "[PERF] stage=segmented_value_and_grad seconds=160.000",
      "[PERF] stage=p32_vag_reverse seconds=20.000",
      "[PERF] stage=segmented_value_and_grad seconds=30.000",
  ))
  if arm == "r2":
    raw.extend((
        "[V2.REDUCE_ONCE.ACCUMULATOR_LOAN]",
        "[V2.REDUCE_ONCE.ACCUMULATOR_RESET]",
        "[V2.REDUCE_ONCE.ACCUMULATOR_LOAN]",
        "[V2.REDUCE_ONCE.ACCUMULATOR_RESET]",
        "[PERF] stage=grad_accumulate seconds=10.000 variant=adopt-scaled",
        "[PERF] stage=grad_accumulate seconds=1.000 variant=adopt-scaled",
    ))
  (root / "raw.log").write_text("\n".join(raw) + "\n")
  profile = root / "xprof-update/plugins/profile/run"
  profile.mkdir(parents=True)
  xplane = profile / "device.xplane.pb"
  xplane.write_bytes(b"xplane")
  with gzip.open(profile / "device.trace.json.gz", "wb") as output:
    # The bounded UI export need not retain terminal events. The full-XPlane
    # census below is the fail-closed completeness receipt.
    output.write(
        json.dumps({"traceEvents": ["zero_tim_update", "reverse_groups"]}).encode()
    )
  expected_counts = {**CENSUS.COMMON_COUNTS, **CENSUS.ARM_COUNTS[arm]}
  census = {
      "schema": CENSUS.SCHEMA,
      "verdict": "PASS",
      "reasons": [],
      "arm": arm,
      "xprof_version": CENSUS.XPROF_VERSION,
      "xplane": {
          "path": str(xplane.relative_to(root)),
          "bytes": xplane.stat().st_size,
          "sha256": hashlib.sha256(xplane.read_bytes()).hexdigest(),
      },
      "host": {"counts": expected_counts},
      "device_modules": {
          f"/device:TPU:{index}": {
              "events": 1,
              "distinct_names": 1,
              "span_seconds": 1.0,
          }
          for index in range(8)
      },
      "trace_buffer_drops": 0,
  }
  (root / "xplane_census.json").write_text(json.dumps(census))
  registry = root / "anchors.json"
  anchor = {
      "micro_gradient_norms": result["micro_gradient_norms"],
      "training_capsule_sha256": capsule_sha,
  }
  if arm == "r2":
    anchor["update_gradient_norm"] = result["update_gradient_norm"]
  registry.write_text(
      json.dumps({"anchors": {f"p45:dp2-tp2:{arm}": anchor}})
  )
  return registry


def _span(
    name: str,
    start: float,
    duration: float,
    **stats: str,
) -> CENSUS.Span:
  return CENSUS.Span(
      name, start * 1e6, duration * 1e6, CENSUS.HOST_LINE, stats
  )


def _synthetic_census_spans(arm: str) -> list[CENSUS.Span]:
  spans = [
      _span("train", 0, 900, step_num="0", _r="1"),
      _span("zero_tim_update", 0, 900, update_step="0"),
      _span("reverse_groups", 90, 720),
      _span("forward_group", 20, 5, group_index="0"),
      _span("group_loss_pullback", 30, 5, group_index="0"),
  ]
  for index in range(CENSUS.GROUPS):
    reverse_start = 100 + index * 90
    spans.extend((
        _span("reverse_group", reverse_start, 80, group_index=str(index)),
        _span(
            "model_backward", reverse_start + 1, 10, group_index=str(index)
        ),
        _span(
            "report_adjoint", reverse_start + 40, 10, group_index=str(index)
        ),
    ))
    if index + 1 < CENSUS.GROUPS:
      spans.extend((
          _span(
              "forward_group",
              reverse_start + 20,
              5,
              group_index=str(index + 1),
          ),
          _span(
              "group_loss_pullback",
              reverse_start + 27,
              5,
              group_index=str(index + 1),
          ),
      ))
    if arm == "r1":
      spans.extend((
          _span(
              "fixed_dp_reduce", reverse_start + 52, 5, group_index=str(index)
          ),
          _span(
              "gradient_accumulate",
              reverse_start + 60,
              5,
              group_index=str(index),
              micro_step=str(index),
              is_last_accumulate=str(int(index == CENSUS.GROUPS - 1)),
          ),
      ))
    else:
      spans.append(_span(
          "staged_accumulate", reverse_start + 52, 5, group_index=str(index)
      ))
  spans.append(_span("loss_pullback", 815, 5))
  if arm == "r2":
    spans.extend((
        _span("fixed_dp_reduce", 825, 5, group_index="7"),
        _span("staged_receipt_fetch", 832, 2),
        _span(
            "gradient_accumulate",
            836,
            5,
            group_index="7",
            micro_step="7",
            is_last_accumulate="1",
        ),
        _span("deferred_finite_receipts", 845, 5),
    ))
  else:
    spans.append(_span("deferred_finite_receipts", 825, 5))
  return spans


def _synthetic_device_modules() -> dict[str, dict[str, float | int]]:
  return {
      f"/device:TPU:{index}": {
          "events": 100,
          "distinct_names": 10,
          "span_seconds": 0.85,
      }
      for index in range(8)
  }


class WarmXprofContractTest(unittest.TestCase):

  def test_selector_is_default_off_and_exact_for_both_arms(self):
    self.assertEqual(
        agentic_rl_learner._canon_v2_frozenlake_profile_arm({}), ""
    )
    self.assertEqual(
        agentic_rl_learner._canon_v2_frozenlake_profile_arm(
            _profile_env("r1")
        ),
        "r1",
    )
    self.assertEqual(
        agentic_rl_learner._canon_v2_frozenlake_profile_arm(
            _profile_env("r2")
        ),
        "r2",
    )

  def test_selector_rejects_every_partial_or_mixed_contract(self):
    base = _profile_env("r2")
    changes = {
        "arm": {"V2_FL_ARM": "r3"},
        "workload": {"V2_FL_WORKLOAD": "m15"},
        "capsule": {"V2_FL_CAPSULE_MODE": "none"},
        "reduce": {"CANON_DP_REDUCE_ONCE": "0"},
        "vma": {"CANON_P66_P59_CHECK_VMA": "0"},
        "commit": {"CANON_P33_NO_COMMIT": "0"},
        "trace_mode": {"CANON_XPROF_TPU_TRACE_MODE": "TRACE_COMPUTE"},
        "trace_dir": {"CANON_XPROF_DIR": "relative"},
    }
    for name, change in changes.items():
      with self.subTest(name=name), self.assertRaises(ValueError):
        agentic_rl_learner._canon_v2_frozenlake_profile_arm(
            {**base, **change}
        )

  def test_repeat_report_requires_exact_norm_hash_and_state_parity(self):
    for arm in ("r1", "r2"):
      with self.subTest(arm=arm):
        warmup = _result(arm)
        profiled = copy.deepcopy(warmup)
        report = agentic_rl_learner._canon_v2_frozenlake_profile_report(
            arm, warmup, profiled
        )
        self.assertEqual(report["verdict"], "PASS")
        self.assertEqual(report["reasons"], [])
        self.assertTrue(report["gradient_repeat_exact"])
        self.assertTrue(report["alignment_repeat_exact"])
        self.assertTrue(report["state_repeat_exact"])

  def test_repeat_report_negatives_fail_closed(self):
    changes = {
        "norm": lambda value: value["micro_gradient_norms"].__setitem__(0, 4.0),
        "update_norm": lambda value: value.__setitem__(
            "update_gradient_norm", 4.0
        ),
        "alignment": lambda value: value["alignment_hashes"].__setitem__(
            0, {"A": "d" * 64}
        ),
        "state": lambda value: value["state_fingerprints_before"].__setitem__(
            "tree", {"sha256": "e" * 64}
        ),
        "changed_path": lambda value: value["model_changed_paths"].append(
            "model.layers.0"
        ),
        "commit": lambda value: value.__setitem__("commits", 1),
    }
    for name, mutate in changes.items():
      with self.subTest(name=name):
        warmup = _result("r2")
        profiled = copy.deepcopy(warmup)
        mutate(profiled)
        report = agentic_rl_learner._canon_v2_frozenlake_profile_report(
            "r2", warmup, profiled
        )
        self.assertEqual(report["verdict"], "FAIL")
        self.assertTrue(report["reasons"])

  def test_source_has_profile_only_repeat_and_staged_fetch_label(self):
    learner = (ROOT / "tunix/rl/agentic/agentic_rl_learner.py").read_text()
    adapter = (ROOT / "tunix/rl/canonical_qwen3_adapter.py").read_text()
    runner = (
        ROOT
        / "canon-zero-tim/tasks/v2-frozenlake-onehost/scripts/"
        "run_frozenlake_dp2tp2_onehost.sh"
    ).read_text()
    self.assertIn("v2_frozenlake_profiled_repeat", learner)
    self.assertIn("_canon_xprof_v2_frozenlake_update_entry", learner)
    self.assertIn("_canon_xprof_v2_frozenlake_update_complete", learner)
    no_commit = learner[learner.index("if segmented_no_commit:"):]
    self.assertIn('"train_steps": actor_trainer.train_steps', no_commit)
    self.assertIn('trace_annotation("staged_receipt_fetch")', adapter)
    self.assertIn('replay:certify|replay:profile', runner)
    self.assertIn("classify_frozenlake_warm_xprof.py", runner)
    self.assertIn("census_frozenlake_warm_xprof.py", runner)

  def test_classifier_accepts_exact_r1_and_r2_fixtures(self):
    for arm in ("r1", "r2"):
      with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        registry = _write_classifier_fixture(root, arm)
        record = CLASSIFIER.classify(root, arm, 0, registry)
        self.assertEqual(record["verdict"], "PASS", record["reasons"])
        self.assertEqual(record["timing"]["profiled_reverse_seconds"], 20.0)
        self.assertEqual(
            record["timing"]["profiled_post_reverse_seconds"], 10.0
        )

  def test_classifier_accepts_bounded_trace_without_terminal_labels(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      registry = _write_classifier_fixture(root, "r2")
      record = CLASSIFIER.classify(root, "r2", 0, registry)
      self.assertEqual(record["verdict"], "PASS", record["reasons"])
      self.assertEqual(
          record["trace"]["bounded_view_needles"]["staged_receipt_fetch"], 0
      )

  def test_classifier_rejects_failed_full_xplane_census(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      registry = _write_classifier_fixture(root, "r2")
      path = root / "xplane_census.json"
      census = json.loads(path.read_text())
      census["verdict"] = "FAIL"
      census["reasons"] = ["staged_receipt_fetch:count=0 expected=1"]
      path.write_text(json.dumps(census))
      record = CLASSIFIER.classify(root, "r2", 0, registry)
      self.assertEqual(record["verdict"], "FAIL")
      self.assertIn("xplane_census.verdict", record["reasons"])
      self.assertIn("xplane_census.reasons", record["reasons"])

  def test_classifier_rejects_stale_xplane_census_identity(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      registry = _write_classifier_fixture(root, "r1")
      xplane = next((root / "xprof-update").glob("plugins/profile/*/*.xplane.pb"))
      xplane.write_bytes(b"Xplane")
      record = CLASSIFIER.classify(root, "r1", 0, registry)
      self.assertEqual(record["verdict"], "FAIL")
      self.assertIn("xplane_census.sha256", record["reasons"])

  def test_classifier_rejects_unpinned_xprof_census_version(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      registry = _write_classifier_fixture(root, "r1")
      path = root / "xplane_census.json"
      census = json.loads(path.read_text())
      census["xprof_version"] = "2.24.0"
      path.write_text(json.dumps(census))
      record = CLASSIFIER.classify(root, "r1", 0, registry)
      self.assertEqual(record["verdict"], "FAIL")
      self.assertIn("xplane_census.xprof_version", record["reasons"])

  def test_xplane_census_validates_complete_r1_and_r2_hierarchies(self):
    for arm in ("r1", "r2"):
      with self.subTest(arm=arm):
        spans = _synthetic_census_spans(arm)
        reasons = CENSUS.validate(
            arm,
            spans,
            device_modules=_synthetic_device_modules(),
            trace_buffer_drops=0,
        )
        self.assertEqual(reasons, [])

  def test_xplane_census_rejects_terminal_drop_and_short_device_window(self):
    spans = [
        span
        for span in _synthetic_census_spans("r2")
        if span.name != "staged_receipt_fetch"
    ]
    devices = _synthetic_device_modules()
    devices["/device:TPU:0"] = {
        "events": 1,
        "distinct_names": 1,
        "span_seconds": 0.5,
    }
    reasons = CENSUS.validate(
        "r2", spans, device_modules=devices, trace_buffer_drops=1
    )
    self.assertIn("staged_receipt_fetch:count=0 expected=1", reasons)
    self.assertIn("trace_buffer_drops=1", reasons)
    self.assertTrue(
        any(reason.startswith("/device:TPU:0:coverage_ratio=") for reason in reasons)
    )

  def test_classifier_rejects_missing_or_nonzero_strict_max_abs(self):
    for value in (None, 1.0):
      with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        registry = _write_classifier_fixture(root, "r1")
        record = json.loads((root / "pre_alignment.jsonl").read_text())
        boundary = record["boundaries"]["A"]
        if value is None:
          del boundary["max_abs"]
        else:
          boundary["max_abs"] = value
        (root / "pre_alignment.jsonl").write_text(json.dumps(record) + "\n")
        classified = CLASSIFIER.classify(root, "r1", 0, registry)
        self.assertEqual(classified["verdict"], "FAIL")
        self.assertIn("strict_alignment", classified["reasons"])

  def test_classifier_requires_exact_compile_time_vma_receipt_count(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      registry = _write_classifier_fixture(root, "r1")
      raw = (root / "raw.log").read_text()
      (root / "raw.log").write_text(
          raw.replace("[P66.VMA] outer_check_enabled\n", "", 1)
      )
      classified = CLASSIFIER.classify(root, "r1", 0, registry)
      self.assertEqual(classified["verdict"], "FAIL")
      self.assertIn("p66_count=38", classified["reasons"])


if __name__ == "__main__":
  unittest.main()
