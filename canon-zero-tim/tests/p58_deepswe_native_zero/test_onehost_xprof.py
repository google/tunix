#!/usr/bin/env python3
"""Host contracts for the P58 matched one-host XProf carrier."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "canon-zero-tim/tasks/p58-deepswe-native-zero-comparison"
SCRIPTS = TASK / "scripts"
SOURCE_SHA = "1" * 40
HOST = "t1v-profile-host-w-0"


def _load(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


CLASSIFIER = _load(
    "p58_onehost_xprof_classifier", SCRIPTS / "classify_onehost_xprof.py"
)
deepswe_debug = _load(
    "p58_onehost_deepswe_debug", ROOT / "tunix/rl/deepswe_debug.py"
)


def _write_fixture(root: Path, arm: str, *, exact: bool = True) -> None:
  manifest = {
      "schema": "canon.local.deepswe.run-manifest.v1",
      "source_commit": SOURCE_SHA,
      "source_diff_sha256": "2" * 64,
      "model_snapshot": (
          "/models/cdbee75f17c01a7cc42f958dc650907174af0554"
      ),
      "r2egym_commit": "0d94c4eb9431cd195c55a7ea3abd54006c9a1735",
      "task_image_id": "sha256:" + "3" * 64,
      "runner_sha256": "4" * 64,
      "stage": "backward-no-commit",
      "model_id": "Qwen/Qwen3-4B-Instruct-2507",
      "contract_name": "local-qwen4b-dp1-tp4",
      "onehost_xprof_arm": arm,
      "expected_hostname": HOST,
      "role_topology": {"dp": 1, "tp": 4, "devices": 4},
      "global_prompts": 1,
      "generations": 2,
      "global_trajectories": 2,
      "max_turns": 2,
      "max_response_length": 512,
      "dataset_seed": 42,
  }
  work_hashes = {
      "prompt_ids": "a" * 64,
      "completion_ids": "b" * 64,
      "advantages": "c" * 64,
      "shape_signature": "d" * 64,
      "actor_update_calls": 2,
  }
  report = {
      "verdict": "PASS",
      "commits": 0,
      "gradient_finite": True,
      "gradient_nonzero": True,
      "gradient_repeat_exact": True,
      "repeat_count": 2,
      "xprof_arm": arm,
      "work_hashes": work_hashes,
      "model_changed_paths": [],
      "optimizer_changed_paths": [],
      "accumulator_changed_paths": [],
      "reference_changed_paths": [],
      "train_steps_before": 0,
      "train_steps_after": 0,
  }
  boundary = {
      "valid": True,
      "finite": True,
      "differing_bytes": 0 if exact else 4,
  }
  alignment = {
      "blocking_reds": [],
      "boundaries": {
          "A_decode_vs_B_prefill": boundary,
          "B_prefill_vs_C_trainer": boundary,
          "S_prefill_vs_T_old": boundary,
      },
  }
  raw = "\n".join((
      f"[P58.ONEHOST.XPROF] ARM_PASS arm={arm} topology=dp1-tp4 fixed_head=off p59=off apc=off",
      "[P58.ONEHOST.XPROF] diagnostic_advantages original=[0.0, 0.0] injected=[-1.0, 1.0] purpose=backward-shape-only",
      f"[P58.ONEHOST.XPROF] warmup_complete arm={arm} commits=0 state_unchanged=1",
      f"[P58.ONEHOST.XPROF] semantic_warmup_discarded arm={arm} next_export=profiled-repeat-only",
      "[DEEPSWE.ONEHOST] optimizer_boundary_skipped commits=0",
      "[DEEPSWE.ONEHOST] optimizer_boundary_skipped commits=0",
      f"[P51.XPROF] phase=update armed step=0 arm={arm}",
      "[P51.XPROF] phase=update started step=0 tpu_trace_mode=TRACE_COMPUTE",
      f"[P51.XPROF] phase=update stopped step=0 arm={arm}",
      "[V1.PERFETTO] captured training_step=0 timelines=3",
      (
          "[CANON_" "ADAPTER] differentiable engine adapter registered"
          if arm == "zero-hp" else "[P58.STOCK_OBSERVER] active"
      ),
  )) + "\n"
  (root / "raw.log").write_text(raw)
  install = (
      "      all 17 files match (qwen4b)\n"
      if arm == "zero-hp"
      else (
          "[P58.STOCK_OBSERVER] OVERLAY_PASS files=2 "
          "stock_runner_verified=1 canonical_bundle=off "
          "treatment=observer-only onehost=1\n"
      )
  )
  (root / "install.log").write_text(install)
  (root / "run_manifest.json").write_text(json.dumps(manifest))
  (root / "backward_no_commit.json").write_text(json.dumps(report))
  (root / "pre_alignment.jsonl").write_text(json.dumps(alignment) + "\n")
  (root / "alignment.jsonl").write_text(json.dumps(alignment) + "\n")
  xprof = root / "xprof-update/plugins/profile/run"
  xprof.mkdir(parents=True)
  (xprof / "device.xplane.pb").write_bytes(b"xplane")
  (xprof / "device.trace.json.gz").write_bytes(b"trace")
  perfetto = root / "perfetto"
  perfetto.mkdir()
  (perfetto / "perfetto_trace_v2_1.pb").write_bytes(b"perfetto")


class OnehostXprofTest(unittest.TestCase):

  def _env(self, arm: str) -> dict[str, str]:
    return {
        "CANON_P58_ONEHOST_XPROF_ARM": arm,
        "CANON_DEEPSWE_ONEHOST_SMOKE": "1",
        "CANON_DEEPSWE_ONEHOST_STAGE": "backward-no-commit",
        "CANON_DEEPSWE_ONEHOST_NO_COMMIT": "1",
        "CANON_DEEPSWE_ONEHOST_ROLLOUT_ONLY": "0",
        "CANON_P58_DEEPSWE_TIM": "0",
    }

  def test_selector_is_default_off_and_fail_closed(self):
    self.assertEqual(deepswe_debug.onehost_xprof_arm({}), "")
    self.assertEqual(
        deepswe_debug.onehost_xprof_arm(self._env("native")), "native"
    )
    self.assertEqual(
        deepswe_debug.onehost_xprof_arm(self._env("zero-hp")), "zero-hp"
    )
    for changed in (
        {"CANON_DEEPSWE_ONEHOST_NO_COMMIT": "0"},
        {"CANON_DEEPSWE_ONEHOST_STAGE": "one-update"},
        {"CANON_P58_DEEPSWE_TIM": "1"},
    ):
      values = {**self._env("native"), **changed}
      with self.assertRaises(ValueError):
        deepswe_debug.onehost_xprof_arm(values)
    with self.assertRaises(ValueError):
      deepswe_debug.onehost_xprof_arm(
          {**self._env("native"), "CANON_P58_ONEHOST_XPROF_ARM": "other"}
      )

  def test_arm_classifier_accepts_complete_native_and_zero_packages(self):
    for arm in ("native", "zero-hp"):
      with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        _write_fixture(root, arm, exact=(arm == "zero-hp"))
        result = CLASSIFIER.classify(
            arm=arm,
            root=root,
            source_sha=SOURCE_SHA,
            expected_hostname=HOST,
        )
        self.assertEqual(result["verdict"], "PASS", result)

  def test_zero_alignment_mismatch_is_a_hard_failure(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _write_fixture(root, "zero-hp", exact=False)
      result = CLASSIFIER.classify(
          arm="zero-hp",
          root=root,
          source_sha=SOURCE_SHA,
          expected_hostname=HOST,
      )
      self.assertEqual(result["verdict"], "FAIL")
      self.assertIn("zero_boundaries_not_exact", result["hard_failures"])

  def test_runners_pin_scope_and_launch_without_a_pipeline(self):
    common = (SCRIPTS / "run_onehost_deepswe_xprof_common.sh").read_text()
    for marker in (
        "P58_ONEHOST_EXPECT_HOSTNAME",
        'status --porcelain)',
        "ls-files --others --exclude-standard",
        "CANON_XPROF_PHASE=update",
        "CANON_XPROF_TPU_TRACE_MODE=TRACE_COMPUTE",
        "CANON_PERF_TRACE_EXPORT_STEP=0",
        "CANON_P38_FIXED_LM_HEAD=0",
        "CANON_P59_RANK_PARALLEL_BACKWARD=0",
        "--max_concurrency 1",
        "--from-path \"$tpu_inference_path\" --model qwen4b",
    ):
      self.assertIn(marker, common)
    launch = common.split("timeout --signal=TERM", 1)[1].split(
        "run_status=$?", 1
    )[0]
    self.assertNotIn("|", launch)
    self.assertIn('>> "$raw_log" 2>&1', launch)
    native = (SCRIPTS / "run_onehost_deepswe_xprof_native.sh").read_text()
    zero = (SCRIPTS / "run_onehost_deepswe_xprof_zero_hp.sh").read_text()
    self.assertIn("common.sh\" native", native)
    self.assertIn("common.sh\" zero-hp", zero)

  def test_stock_observer_overlay_admits_only_exact_onehost_native(self):
    installer = (
        ROOT / "canon-zero-tim/cluster/steps/p58_install_stock_prompt_observer.sh"
    ).read_text()
    self.assertIn("onehost_native=", installer)
    self.assertIn('CANON_P58_ONEHOST_XPROF_ARM:-}\" = \"native', installer)
    self.assertIn('CANON_DEEPSWE_ONEHOST_NO_COMMIT:-0}\" = \"1', installer)


if __name__ == "__main__":
  unittest.main()
