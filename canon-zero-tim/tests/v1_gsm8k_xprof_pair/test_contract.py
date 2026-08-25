#!/usr/bin/env python3
"""Host contracts for the matched GSM8K Native/Zero-HP XProf pair."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import types
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "canon-zero-tim/tasks/v1-gsm8k-onehost-xprof-pair"
SCRIPTS = TASK / "scripts"


def _load(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


ARM_CLASSIFIER = _load(
    "v1_gsm8k_xprof_arm_classifier",
    SCRIPTS / "classify_gsm8k_xprof_arm.py",
)
PAIR_CLASSIFIER = _load(
    "v1_gsm8k_xprof_pair_classifier",
    SCRIPTS / "classify_gsm8k_xprof_pair.py",
)
MODULE_CENSUS = _load(
    "v1_gsm8k_xprof_module_census",
    SCRIPTS / "census_gsm8k_xprof_modules.py",
)
GSM8K_XPROF = _load("gsm8k_xprof", ROOT / "tunix/rl/gsm8k_xprof.py")


def _common_env(arm: str) -> dict[str, str]:
  values = {
      "CANON_V1_GSM8K_XPROF_ARM": arm,
      "CANON_GSM8K_TRAIN": "1",
      "CANON_OPT_STATE_RESIDENT": "1",
      "CANON_P30_OPT_STATE_OFFLOAD": "0",
      "CANON_VLLM_ENABLE_PREFIX_CACHING": "0",
      "CANON_P60_DETERMINISTIC_AB": "1",
      "CANON_XPROF_PHASE": "update",
      "CANON_XPROF_SKIP_STEPS": "1",
      "CANON_XPROF_STEPS": "1",
      "CANON_XPROF_HOST_TRACER": "1",
      "CANON_XPROF_PYTHON_TRACER": "0",
      "CANON_XPROF_TPU_TRACE_MODE": "TRACE_COMPUTE",
      "CANON_XPROF_DIR": "/tmp/xprof",
      "CANON_PERF_TRACE_DIR": "/tmp/perf",
  }
  if arm == "native":
    values["CANON_GSM8K_VANILLA"] = "1"
    values["CANON_P59_RANK_PARALLEL_BACKWARD"] = "0"
    values["CANON_P28_G6_UPDATE"] = "0"
  else:
    values.update({
        "CANON_P32_WORKLOAD": "gsm8k-p59-dp4-tp1",
        "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
        "CANON_P28_G6_UPDATE": "1",
        "CANON_GSM8K_ALIGNMENT_WARN_ONLY": "0",
    })
  return values


def _work(arm: str, step: int) -> dict:
  fields = {
      name: {"dtype": "int32", "shape": [64, 8], "sha256": name * 4}
      for name in ("prompt_ids", "completion_ids", "advantages")
  }
  return {
      "schema": "canon.v1.gsm8k-onehost-xprof.work.v1",
      "arm": arm,
      "train_step": step,
      "global_step": step,
      "fields": fields,
      "shape_signature": "a" * 64,
  }


def _fixture(root: Path, arm: str) -> None:
  state = root / "train"
  xprof = state / "xprof/plugins/profile/run"
  perf = state / "perf"
  xprof.mkdir(parents=True)
  perf.mkdir(parents=True)
  (xprof / "device.xplane.pb").write_bytes(b"xplane")
  (xprof / "device.trace.json.gz").write_bytes(b"trace")
  (perf / "perfetto_trace_v2_1.pb").write_bytes(b"perfetto")
  lines = [
      "[V1.GSM8K.XPROF] RUN_BEGIN arm=" + arm,
      f"[V1.GSM8K.XPROF] PREFLIGHT_PASS arm={arm} topology=DP4xTP1 mesh_ids=[0, 2, 1, 3] prompts=8 generations=8 trajectories=64 groups=16 capture=update:1->2",
      "[P51.XPROF] phase=update started step=1 anchor=update_entry tpu_trace_mode=TRACE_COMPUTE",
      "[P51.XPROF] phase=update stopped step=2 anchor=step_completed",
  ]
  for step in range(3):
    lines.append(
        "[V1.GSM8K.XPROF.WORK] "
        + json.dumps(_work(arm, step), sort_keys=True, separators=(",", ":"))
    )
    lines.append(f"Global step {step} completed in 1.0 seconds.")
  if arm == "native":
    lines.extend((
        "[P56.VANILLA] stock arm: canonical numeric admission bypassed; yardstick only",
        "[P56.VANILLA] engine contract attestation bypassed (stock arm)",
    ))
  else:
    lines.extend(
        f"[CANON_ALIGN] index={index} verdict=PASS" for index in range(51)
    )
  lines.append(f"[V1.GSM8K.XPROF] RUN_END arm={arm} docker_exit=0 elapsed_seconds=10")
  (state / "raw.log").write_text("\n".join(lines) + "\n")
  (root / "driver.log").write_text("driver\n")
  xprof_detail = "CENSUS_GREEN all 8 planes: backward present, decode absent\n"
  if arm == "zero-hp":
    xprof_detail += (
        "zt_tr_dp_parallel_bwd_layer_00\n"
        "optimizer_tail=scaled_step:16,commit:1\n"
    )
  (state / "xprof_census.txt").write_text(xprof_detail)
  (state / "semantic_census.txt").write_text(
      "CENSUS_GREEN peft_train placed like weight_sync, no custom spans\n"
  )


class ContractTest(unittest.TestCase):

  def test_evidence_ledger_green_red_and_post_manifest_tamper(self):
    helper = SCRIPTS / "finalize_gsm8k_xprof_evidence.sh"
    harness = r'''
source "$1"
root="$2"
docker_rc="$3"
classifier_rc="$4"
tamper="$5"
driver="$root/driver.log"
raw="$root/raw.log"
manifest="$root/SHA256SUMS"
gsm8k_xprof_choose_terminal zero-hp "$root" "$docker_rc" "$classifier_rc"
gsm8k_xprof_write_terminal_manifest \
  "$GSM8K_XPROF_TERMINAL_MARKER" "$driver" "$manifest" "$driver" "$raw"
if [ "$tamper" = 1 ]; then
  printf 'post-manifest tamper\n' >>"$raw"
fi
if ! gsm8k_xprof_verify_manifest "$manifest"; then
  printf '[V1.GSM8K.XPROF] SHA_LEDGER_RED stage=verify root=%s\n' "$root" >&2
  exit 98
fi
printf '[V1.GSM8K.XPROF] SHA_LEDGER_PASS entries=2 root=%s\n' "$root"
printf '%s\n' "$GSM8K_XPROF_TERMINAL_MARKER"
exit "$GSM8K_XPROF_TERMINAL_RC"
'''
    cases = (
        ("green", "0", "0", "0", 0, "GREEN", True),
        ("red", "7", "0", "0", 1, "RED", True),
        ("tamper", "0", "0", "1", 98, "GREEN", False),
    )
    with tempfile.TemporaryDirectory() as directory:
      base = Path(directory)
      for name, docker_rc, classifier_rc, tamper, expected_rc, marker, valid in cases:
        with self.subTest(name=name):
          root = base / name
          root.mkdir()
          (root / "driver.log").write_text("driver preamble\n")
          (root / "raw.log").write_text("raw evidence\n")
          result = subprocess.run(
              [
                  "bash", "-euo", "pipefail", "-c", harness, "ledger-test",
                  str(helper), str(root), docker_rc, classifier_rc, tamper,
              ],
              capture_output=True,
              text=True,
              check=False,
          )
          self.assertEqual(result.returncode, expected_rc, result)
          terminal_lines = [
              line for line in (root / "driver.log").read_text().splitlines()
              if line.startswith("[V1.GSM8K.XPROF] GREEN ")
              or line.startswith("[V1.GSM8K.XPROF] RED ")
          ]
          self.assertEqual(len(terminal_lines), 1)
          self.assertIn(marker, terminal_lines[0])
          verify = subprocess.run(
              ["sha256sum", "-c", str(root / "SHA256SUMS")],
              capture_output=True,
              text=True,
              check=False,
          )
          self.assertEqual(verify.returncode == 0, valid, verify)
          if valid:
            self.assertIn("SHA_LEDGER_PASS", result.stdout)
            self.assertNotIn("SHA_LEDGER_RED", result.stderr)
          else:
            self.assertIn("SHA_LEDGER_RED", result.stderr)
            self.assertNotIn("SHA_LEDGER_PASS", result.stdout)

      green_root = base / "green"
      duplicate = subprocess.run(
          [
              "bash", "-euo", "pipefail", "-c",
              r'''
source "$1"
if gsm8k_xprof_write_terminal_manifest \
    "[V1.GSM8K.XPROF] RED arm=zero-hp docker_rc=1 classifier_rc=1 root=$2" \
    "$2/driver.log" "$2/duplicate-SHA256SUMS" "$2/driver.log"; then
  exit 1
fi
''',
              "duplicate-test", str(helper), str(green_root),
          ],
          capture_output=True,
          text=True,
          check=False,
      )
      self.assertEqual(duplicate.returncode, 0, duplicate)
      terminal_lines = [
          line for line in (green_root / "driver.log").read_text().splitlines()
          if line.startswith("[V1.GSM8K.XPROF] GREEN ")
          or line.startswith("[V1.GSM8K.XPROF] RED ")
      ]
      self.assertEqual(len(terminal_lines), 1)
      self.assertFalse((green_root / "duplicate-SHA256SUMS").exists())

  def test_zero_hp_module_census_requires_complete_optimizer_tail(self):
    counts = {
        pattern.pattern: 1 for pattern in MODULE_CENSUS.ZERO_REQUIRED
    }
    counts.update(MODULE_CENSUS.ZERO_TAIL_EXACT)
    self.assertEqual(
        MODULE_CENSUS.validate_module_counts("zero-hp", counts), []
    )

    without_commit = dict(counts)
    del without_commit["jit__precomputed_gradient_commit"]
    self.assertIn(
        "jit__precomputed_gradient_commit=0!=1",
        MODULE_CENSUS.validate_module_counts("zero-hp", without_commit),
    )

    short_scaled_step = dict(counts)
    short_scaled_step["jit__precomputed_gradient_scaled_step"] = 15
    self.assertIn(
        "jit__precomputed_gradient_scaled_step=15!=16",
        MODULE_CENSUS.validate_module_counts("zero-hp", short_scaled_step),
    )
    self.assertEqual(
        MODULE_CENSUS.validate_plane_names(
            [f"/device:TPU:{index}" for index in range(8)]
        ),
        [],
    )
    self.assertTrue(
        MODULE_CENSUS.validate_plane_names(
            [f"/device:TPU:{index}" for index in range(7)]
        )
    )

  def test_arm_selector_is_default_off_and_treatment_exact(self):
    self.assertEqual(GSM8K_XPROF.arm({}), "")
    self.assertEqual(GSM8K_XPROF.arm(_common_env("native")), "native")
    self.assertEqual(GSM8K_XPROF.arm(_common_env("zero-hp")), "zero-hp")
    for changed in (
        {"CANON_XPROF_PHASE": "step"},
        {"CANON_VLLM_ENABLE_PREFIX_CACHING": "1"},
        {"CANON_P60_DETERMINISTIC_AB": "0"},
        {"CANON_P59_RANK_PARALLEL_BACKWARD": "1"},
    ):
      with self.assertRaises(ValueError):
        GSM8K_XPROF.arm({**_common_env("native"), **changed})
    with self.assertRaises(ValueError):
      GSM8K_XPROF.arm(
          {**_common_env("zero-hp"), "CANON_GSM8K_VANILLA": "1"}
      )

  def test_work_receipt_hashes_required_arrays(self):
    train_example = types.SimpleNamespace(
        prompt_ids=np.arange(8, dtype=np.int32).reshape(2, 4),
        completion_ids=np.arange(12, dtype=np.int32).reshape(2, 6),
        advantages=np.asarray([1.0, -1.0], dtype=np.float32),
        prompt_mask=None,
        completion_mask=None,
        completion_valid_mask=None,
        policy_version=np.asarray([0, 0], dtype=np.int32),
    )
    receipt = GSM8K_XPROF.work_receipt(
        train_example, selected_arm="native", train_step=1, global_step=1
    )
    self.assertEqual(receipt["train_step"], 1)
    self.assertEqual(receipt["fields"]["completion_ids"]["shape"], [2, 6])
    self.assertRegex(receipt["fields"]["advantages"]["sha256"], r"^[0-9a-f]{64}$")

  def test_arm_and_pair_classifiers_require_matched_backward_captures(self):
    records = {}
    with tempfile.TemporaryDirectory() as directory:
      base = Path(directory)
      for arm in ("native", "zero-hp"):
        root = base / arm
        _fixture(root, arm)
        record = ARM_CLASSIFIER.classify(
            arm=arm,
            run_root=root,
            source_sha="1" * 40,
            source_diff_sha256="2" * 64,
            runtime_manifest_sha256="5" * 64,
            model_snapshot="3" * 40,
            image_id="sha256:" + "4" * 64,
            xprof_census_rc=0,
            semantic_census_rc=0,
        )
        self.assertEqual(record["verdict"], "PASS", record)
        records[arm] = record
      pair = PAIR_CLASSIFIER.classify(records["native"], records["zero-hp"])
      self.assertEqual(pair["verdict"], "PASS", pair)
      missing_hierarchy = ARM_CLASSIFIER.classify(
          arm="zero-hp",
          run_root=base / "zero-hp",
          source_sha="1" * 40,
          source_diff_sha256="2" * 64,
          runtime_manifest_sha256="5" * 64,
          model_snapshot="3" * 40,
          image_id="sha256:" + "4" * 64,
          xprof_census_rc=0,
          semantic_census_rc=0,
          require_hierarchy=True,
          hierarchy_census_rc=1,
      )
      self.assertEqual(missing_hierarchy["verdict"], "FAIL")
      hierarchy = base / "zero-hp/train/hierarchy_census.txt"
      hierarchy.write_text(
          "V1_GSM8K_XPROF_HIERARCHY_CENSUS_GREEN "
          "train_step=1 host_plane=/host:CPU host_line=python3 "
          "steps_planes=8 forward_groups=16 reverse_groups=16 "
          "transactions=16 micro_steps=0..15 last_accumulate=15 "
          "optimizer_update=1\n"
      )
      revised_zero = ARM_CLASSIFIER.classify(
          arm="zero-hp",
          run_root=base / "zero-hp",
          source_sha="1" * 40,
          source_diff_sha256="2" * 64,
          runtime_manifest_sha256="5" * 64,
          model_snapshot="3" * 40,
          image_id="sha256:" + "4" * 64,
          xprof_census_rc=0,
          semantic_census_rc=0,
          require_hierarchy=True,
          hierarchy_census_rc=0,
      )
      self.assertEqual(revised_zero["verdict"], "PASS", revised_zero)
      forbidden_native = ARM_CLASSIFIER.classify(
          arm="native",
          run_root=base / "native",
          source_sha="1" * 40,
          source_diff_sha256="2" * 64,
          runtime_manifest_sha256="5" * 64,
          model_snapshot="3" * 40,
          image_id="sha256:" + "4" * 64,
          xprof_census_rc=0,
          semantic_census_rc=0,
          require_hierarchy=True,
          hierarchy_census_rc=0,
      )
      self.assertEqual(forbidden_native["verdict"], "FAIL")
      self.assertIn(
          "hierarchy_requirement_is_zero_hp_only",
          forbidden_native["reasons"],
      )
      native_raw = base / "native/train/raw.log"
      native_raw.write_text(
          native_raw.read_text() + "[CANON_" + "ADAPTER] unexpected\n"
      )
      contaminated = ARM_CLASSIFIER.classify(
          arm="native",
          run_root=base / "native",
          source_sha="1" * 40,
          source_diff_sha256="2" * 64,
          runtime_manifest_sha256="5" * 64,
          model_snapshot="3" * 40,
          image_id="sha256:" + "4" * 64,
          xprof_census_rc=0,
          semantic_census_rc=0,
      )
      self.assertEqual(contaminated["verdict"], "FAIL", contaminated)
      self.assertIn("native_canonical_program_present", contaminated["reasons"])
      changed = json.loads(json.dumps(records["zero-hp"]))
      changed["profiled_work"]["shape_signature"] = "b" * 64
      changed["profiled_work"]["fields"]["completion_ids"]["sha256"] = "c" * 64
      mismatch = PAIR_CLASSIFIER.classify(records["native"], changed)
      self.assertEqual(mismatch["verdict"], "INCONCLUSIVE_INPUT_MISMATCH")
      self.assertEqual(
          mismatch["mismatched_profiled_work_fields"],
          ["fields", "shape_signature"],
      )
      self.assertEqual(
          mismatch["mismatched_profiled_work_arrays"], ["completion_ids"]
      )

  def test_static_runner_is_gsm8k_and_wrappers_select_one_arm(self):
    common = (SCRIPTS / "run_onehost_gsm8k_xprof_common.sh").read_text()
    inner = (SCRIPTS / "run_onehost_gsm8k_xprof_inner.sh").read_text()
    self.assertIn("models--Qwen--Qwen3-1.7B", common)
    self.assertIn("--model qwen1p7b_tp1", common)
    self.assertIn("examples/math_gsm8k/qwen3_grpo_demo.py", inner)
    self.assertIn("--mesh_dp=4 --mesh_tp=1", inner)
    self.assertIn("--max_steps=3", inner)
    self.assertIn("-e CANON_P60_DETERMINISTIC_AB=1", common)
    self.assertIn("census_gsm8k_xprof_modules.py", common)
    self.assertIn("census_gsm8k_semantic_trace.py", common)
    self.assertIn("census_gsm8k_xprof_hierarchy.py", common)
    self.assertIn("--require-hierarchy", common)
    self.assertIn("finalize_gsm8k_xprof_evidence.sh", common)
    self.assertIn('sha_inputs=("$raw" "$driver")', common)
    self.assertIn('"$xprof_census" "$semantic_census" "$classification"', common)
    choose = common.index("gsm8k_xprof_choose_terminal")
    write = common.index("gsm8k_xprof_write_terminal_manifest")
    verify = common.index("gsm8k_xprof_verify_manifest")
    ledger_pass = common.index("SHA_LEDGER_PASS")
    self.assertLess(choose, write)
    self.assertLess(write, verify)
    self.assertLess(verify, ledger_pass)
    self.assertNotIn('tee -a "$driver"', common[verify:])
    self.assertNotIn('>>"$driver"', common[verify:])
    self.assertNotIn('sha256sum "${sha_inputs[@]}"', common)
    analyze = (SCRIPTS / "analyze_gsm8k_xprof_pair.sh").read_text()
    self.assertIn("classify_gsm8k_xprof_pair.py", analyze)
    self.assertIn("xprof_trace_summary.py", analyze)
    self.assertIn("expected exactly one non-empty trace per arm", analyze)
    self.assertNotIn("<run>/*.trace.json.gz", analyze)
    self.assertNotIn("train_deepswe", common + inner)
    self.assertNotIn("R2E", common + inner)
    native = (SCRIPTS / "run_onehost_gsm8k_xprof_native.sh").read_text()
    zero = (SCRIPTS / "run_onehost_gsm8k_xprof_zero_hp.sh").read_text()
    self.assertIn('common.sh" native', native)
    self.assertIn('common.sh" zero-hp', zero)


if __name__ == "__main__":
  unittest.main()
