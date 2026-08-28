#!/usr/bin/env python3
"""Host contracts for the matched GSM8K Native/Zero-HP XProf pair."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
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
  sys.modules[spec.name] = module
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
SIZE_CENSUS = _load(
    "v1_gsm8k_xprof_size_census",
    SCRIPTS / "census_gsm8k_xprof_size.py",
)
P74_GAP_CENSUS = _load(
    "v1_gsm8k_p74_gap_census",
    SCRIPTS / "census_gsm8k_p74_gap.py",
)
GSM8K_XPROF = _load("gsm8k_xprof", ROOT / "tunix/rl/gsm8k_xprof.py")


def _common_env(arm: str, geometry: str = "dp4-tp1") -> dict[str, str]:
  values = {
      "CANON_V1_GSM8K_XPROF_ARM": arm,
      "CANON_GSM8K_TRAIN": "1",
      "CANON_OPT_STATE_RESIDENT": "1",
      "CANON_P30_OPT_STATE_OFFLOAD": "0",
      "CANON_VLLM_ENABLE_PREFIX_CACHING": "0",
      "CANON_P60_DETERMINISTIC_AB": "1",
      "CANON_XPROF_PHASE": "update",
      "CANON_XPROF_SKIP_STEPS": "2",
      "CANON_XPROF_STEPS": "1",
      "CANON_XPROF_HOST_TRACER": "1",
      "CANON_XPROF_PYTHON_TRACER": "0",
      "CANON_XPROF_TPU_TRACE_MODE": "TRACE_ONLY_XLA",
      "CANON_XPROF_LABELS": "1",
      "CANON_XPROF_DIR": "/tmp/xprof",
      "CANON_PERF_TRACE_DIR": "/tmp/perf",
  }
  if geometry != "dp4-tp1":
    # The default geometry stays byte-identical: an unchanged launcher
    # sends no geometry env at all, so the tests only set it off-default.
    values["CANON_V1_GSM8K_XPROF_GEOMETRY"] = geometry
  workload = {
      "dp4-tp1": "gsm8k-p59-dp4-tp1",
      "dp2-tp2": "gsm8k-p59-dp2-tp2",
  }.get(geometry, geometry)
  if arm == "native":
    values["CANON_GSM8K_VANILLA"] = "1"
    values["CANON_P59_RANK_PARALLEL_BACKWARD"] = "0"
    values["CANON_P28_G6_UPDATE"] = "0"
  else:
    values.update({
        "CANON_P32_WORKLOAD": workload,
        "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
        "CANON_P28_G6_UPDATE": "1",
        "CANON_GSM8K_ALIGNMENT_WARN_ONLY": "0",
    })
  return values


def _zero_module_counts(p71_scan: str) -> dict[str, int]:
  """A minimal green zero-HP plane inventory for one P71 scan rung."""
  counts = {pattern.pattern: 1 for pattern in MODULE_CENSUS.ZERO_REQUIRED}
  counts.update(MODULE_CENSUS.ZERO_TAIL_EXACT)
  if p71_scan == "bwd":
    counts.update({
        f"jit_zt_tr_dp_parallel_bwd_block_{index:02d}":
            MODULE_CENSUS.ZERO_BACKWARD_EXECS
        for index in MODULE_CENSUS.expected_block_indices()
    })
  else:
    # XLA folds the identically shaped per-layer pullbacks onto a single
    # module name; both landed captures show one name at 28 x 32 events.
    counts["jit_zt_tr_dp_parallel_bwd_layer_27"] = (
        MODULE_CENSUS.ZERO_LAYER_COUNT * MODULE_CENSUS.ZERO_BACKWARD_EXECS
    )
  if p71_scan in ("fwd", "bwd"):
    counts[MODULE_CENSUS.FWD_TAPE_SCAN] = MODULE_CENSUS.ZERO_BACKWARD_EXECS
  return counts


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


def _fixture(
    root: Path, arm: str, updates: int = 3, geometry: str = "dp4-tp1"
) -> None:
  shape = ARM_CLASSIFIER._GEOMETRIES[geometry]
  groups = shape["groups"]
  state = root / "train"
  xprof = state / "xprof/plugins/profile/run"
  perf = state / "perf"
  xprof.mkdir(parents=True)
  perf.mkdir(parents=True)
  (xprof / "device.xplane.pb").write_bytes(b"xplane")
  (xprof / "device.trace.json.gz").write_bytes(b"trace")
  (perf / "perfetto_trace_v2_1.pb").write_bytes(b"perfetto")
  mesh_ids = "[0, 2, 1, 3]" if geometry == "dp4-tp1" else "[0, 1, 2, 3]"
  lines = [
      "[V1.GSM8K.XPROF] RUN_BEGIN arm=" + arm,
      f"[V1.GSM8K.XPROF] PREFLIGHT_PASS arm={arm} topology={shape['topology']} mesh_ids={mesh_ids} prompts=8 generations=8 trajectories=64 groups={groups} capture=update:2->3",
      "[P51.XPROF] phase=update started step=2 anchor=update_entry tpu_trace_mode=TRACE_ONLY_XLA",
      "[P51.XPROF] phase=update stopped step=3 anchor=step_completed",
  ]
  for step in range(updates):
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
        f"[CANON_ALIGN] index={index} verdict=PASS"
        for index in range((1 + groups) * updates)
    )
  lines.append(f"[V1.GSM8K.XPROF] RUN_END arm={arm} docker_exit=0 elapsed_seconds=10")
  (state / "raw.log").write_text("\n".join(lines) + "\n")
  (root / "driver.log").write_text("driver\n")
  xprof_detail = "CENSUS_GREEN all 8 planes: backward present, decode absent\n"
  if arm == "zero-hp":
    xprof_detail += (
        "zt_tr_dp_parallel_bwd_layer_00\n"
        f"optimizer_tail=scaled_step:{groups},commit:1\n"
    )
    (state / "trace_census.txt").write_text(
        "V1_GSM8K_XPROF_TRACE_CENSUS_GREEN "
        f"train_steps={2 * groups}..{3 * groups - 1} "
        f"reverse_transactions={groups} "
        "optimizer_visible=1 optimizer_owned_by_last=1 "
        "same_host_track=1 compiler_events=0\n"
    )
  (state / "xprof_census.txt").write_text(xprof_detail)
  (state / "semantic_census.txt").write_text(
      "CENSUS_GREEN peft_train placed like weight_sync, no custom spans\n"
  )
  size_receipt = SIZE_CENSUS.build_receipt(root)
  (state / "xprof_size_receipt.json").write_text(
      json.dumps(size_receipt, indent=2, sort_keys=True) + "\n"
  )
  (state / "xprof_size_census.txt").write_text(
      "V1_GSM8K_XPROF_SIZE_CENSUS_GREEN status=PASS "
      f"xprof_bytes={size_receipt['total_bytes']} "
      f"soft_warning_bytes={SIZE_CENSUS.SOFT_WARNING_BYTES} "
      f"hard_max_bytes={SIZE_CENSUS.HARD_MAX_BYTES}\n"
  )


def _p74_fixture(root: Path) -> None:
  state = root / "train"
  victim = {kind: 0 for kind in P74_GAP_CENSUS.VICTIM_KINDS}
  receipt = {
      "schema": P74_GAP_CENSUS.SCHEMA,
      "status": "PASS",
      "geometry": "dp2-tp2",
      "acceptance": {
          "expected_windows": 64,
          "max_mean_gap_ms": 70.0,
          "exact_victim_overlap_events": 0,
          "partition_module_per_window": P74_GAP_CENSUS.PARTITION_MODULE,
      },
      "gap": {
          "windows": 64,
          "total_ms": 4.032,
          "mean_ms": 0.063,
          "max_ms": 0.064,
          "min_ms": 0.062,
      },
      "identity_windows": 64,
      "intervening_modules": {P74_GAP_CENSUS.PARTITION_MODULE: 64},
      "victim_overlap": victim,
      "victim_global": victim,
      "windows_with_any_victim": 0,
      "reverse_wall": {
          "rows": [],
          "captured_update": {
              "seconds": 15.844,
              "groups": 32,
              "mean_seconds": 0.459,
              "max_seconds": 0.542,
          },
      },
      "reasons": [],
  }
  (state / "p74_gap_receipt.json").write_text(
      json.dumps(receipt, indent=2, sort_keys=True) + "\n"
  )
  (state / "p74_gap_census.txt").write_text(
      "V1_GSM8K_P74_GAP_CENSUS_GREEN windows=64 mean_ms=0.063 "
      "max_ms=0.064 identity_windows=64 windows_with_any_victim=0\n"
  )


class ContractTest(unittest.TestCase):

  def test_p74_gap_census_accepts_device_bridge_and_rejects_old_roundtrip(self):
    fast_modules = []
    slow_modules = []
    for index in range(P74_GAP_CENSUS.EXPECTED_WINDOWS):
      base = index * 1_000_000_000
      seed = P74_GAP_CENSUS.Event(
          P74_GAP_CENSUS.SEED_MODULE, base, 1_000, {}
      )
      identity = P74_GAP_CENSUS.Event(
          P74_GAP_CENSUS.PARTITION_MODULE, base + 20_000, 20_000, {}
      )
      fast_modules.extend((
          seed,
          identity,
          P74_GAP_CENSUS.Event(
              P74_GAP_CENSUS.HEAD_MODULE, base + 64_000, 1_000, {}
          ),
      ))
      slow_modules.extend((
          seed,
          P74_GAP_CENSUS.Event(
              P74_GAP_CENSUS.HEAD_MODULE,
              base + 150_747_000,
              1_000,
              {},
          ),
      ))

    green = P74_GAP_CENSUS.analyze(fast_modules, [])
    self.assertEqual(green["status"], "PASS", green)
    self.assertEqual(green["gap"]["windows"], 64)
    self.assertAlmostEqual(green["gap"]["mean_ms"], 0.063)
    self.assertEqual(green["identity_windows"], 64)
    self.assertEqual(set(green["victim_overlap"].values()), {0})

    old_roundtrip = P74_GAP_CENSUS.Event(
        "tpu::System::TransferFromDevice",
        2_000,
        120_000_000,
        {"size": "77791232"},
    )
    red = P74_GAP_CENSUS.analyze(slow_modules, [old_roundtrip])
    self.assertEqual(red["status"], "FAIL", red)
    self.assertEqual(red["gap"]["mean_ms"], 150.746)
    self.assertEqual(red["identity_windows"], 0)
    self.assertEqual(red["victim_overlap"]["d2h_77791232"], 1)
    self.assertTrue(
        any(reason.startswith("mean_gap_ms=") for reason in red["reasons"]),
        red,
    )

  def test_p74_gap_receipt_is_required_only_for_zero_dp2(self):
    keys = dict(
        source_sha="1" * 40,
        source_diff_sha256="2" * 64,
        runtime_manifest_sha256="5" * 64,
        model_snapshot="3" * 40,
        image_id="sha256:" + "4" * 64,
        xprof_census_rc=0,
        semantic_census_rc=0,
    )
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / "zero-dp2"
      _fixture(root, "zero-hp", geometry="dp2-tp2")
      _p74_fixture(root)
      green = ARM_CLASSIFIER.classify(
          arm="zero-hp",
          run_root=root,
          geometry="dp2-tp2",
          require_p74_gap=True,
          p74_gap_census_rc=0,
          **keys,
      )
      self.assertEqual(green["verdict"], "PASS", green)
      self.assertEqual(green["p74_gap"]["gap"]["mean_ms"], 0.063)

      receipt_path = root / "train/p74_gap_receipt.json"
      receipt = json.loads(receipt_path.read_text())
      receipt["gap"]["mean_ms"] = 70.001
      receipt_path.write_text(json.dumps(receipt) + "\n")
      red = ARM_CLASSIFIER.classify(
          arm="zero-hp",
          run_root=root,
          geometry="dp2-tp2",
          require_p74_gap=True,
          p74_gap_census_rc=0,
          **keys,
      )
      self.assertEqual(red["verdict"], "FAIL", red)
      self.assertIn(
          "p74_gap_receipt.gap.mean_ms=70.001", red["reasons"]
      )

      wrong_arm = Path(directory) / "native-dp2"
      _fixture(wrong_arm, "native", geometry="dp2-tp2")
      _p74_fixture(wrong_arm)
      red = ARM_CLASSIFIER.classify(
          arm="native",
          run_root=wrong_arm,
          geometry="dp2-tp2",
          require_p74_gap=True,
          p74_gap_census_rc=0,
          **keys,
      )
      self.assertEqual(red["verdict"], "FAIL", red)
      self.assertIn(
          "p74_gap_requirement_is_zero_hp_dp2_tp2_only", red["reasons"]
      )

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
xplane="$root/device.xplane.pb"
gsm8k_xprof_choose_terminal zero-hp "$root" "$docker_rc" "$classifier_rc"
gsm8k_xprof_write_terminal_manifest \
  "$GSM8K_XPROF_TERMINAL_MARKER" "$driver" "$manifest" \
  "$driver" "$raw" "$xplane"
if [ "$tamper" = raw ]; then
  printf 'post-manifest tamper\n' >>"$raw"
elif [ "$tamper" = xprof ]; then
  printf 'post-manifest xprof tamper\n' >>"$xplane"
fi
if ! gsm8k_xprof_verify_manifest "$manifest"; then
  printf '[V1.GSM8K.XPROF] SHA_LEDGER_RED stage=verify root=%s\n' "$root" >&2
  exit 98
fi
printf '[V1.GSM8K.XPROF] SHA_LEDGER_PASS entries=3 root=%s\n' "$root"
printf '%s\n' "$GSM8K_XPROF_TERMINAL_MARKER"
exit "$GSM8K_XPROF_TERMINAL_RC"
'''
    cases = (
        ("green", "0", "0", "none", 0, "GREEN", True),
        ("red", "7", "0", "none", 1, "RED", True),
        ("tamper", "0", "0", "raw", 98, "GREEN", False),
        ("xprof_tamper", "0", "0", "xprof", 98, "GREEN", False),
    )
    with tempfile.TemporaryDirectory() as directory:
      base = Path(directory)
      for name, docker_rc, classifier_rc, tamper, expected_rc, marker, valid in cases:
        with self.subTest(name=name):
          root = base / name
          root.mkdir()
          (root / "driver.log").write_text("driver preamble\n")
          (root / "raw.log").write_text("raw evidence\n")
          (root / "device.xplane.pb").write_bytes(b"xplane evidence\n")
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

  def test_xprof_size_budget_pass_warn_and_hard_red(self):
    cases = (
        ("pass", 1024, 0, "PASS", "GREEN"),
        (
            "warn",
            SIZE_CENSUS.SOFT_WARNING_BYTES,
            0,
            "WARN",
            "GREEN",
        ),
        (
            "hard_red",
            SIZE_CENSUS.HARD_MAX_BYTES,
            1,
            "FAIL",
            "RED",
        ),
    )
    with tempfile.TemporaryDirectory() as directory:
      base = Path(directory)
      for name, xplane_bytes, expected_rc, status, marker in cases:
        with self.subTest(name=name):
          root = base / name
          profile = root / "train/xprof/plugins/profile/run"
          profile.mkdir(parents=True)
          with (profile / "device.xplane.pb").open("wb") as stream:
            stream.truncate(xplane_bytes)
          (profile / "device.trace.json.gz").write_bytes(b"trace")
          output = root / "train/xprof_size_receipt.json"
          result = subprocess.run(
              [
                  "python3",
                  str(SCRIPTS / "census_gsm8k_xprof_size.py"),
                  "--run-root",
                  str(root),
                  "--output",
                  str(output),
              ],
              capture_output=True,
              text=True,
              check=False,
          )
          self.assertEqual(result.returncode, expected_rc, result)
          self.assertIn(
              f"V1_GSM8K_XPROF_SIZE_CENSUS_{marker} status={status}",
              result.stdout,
          )
          receipt = json.loads(output.read_text())
          self.assertEqual(receipt["status"], status)
          self.assertEqual(
              receipt["total_bytes"], xplane_bytes + len(b"trace")
          )
          self.assertEqual(receipt["counts"]["xplane"], 1)
          self.assertEqual(receipt["counts"]["trace_json_gz"], 1)
          if status == "FAIL":
            self.assertTrue(
                any("exceeds_hard_max" in reason for reason in receipt["reasons"])
            )
          else:
            self.assertEqual(receipt["reasons"], [])

  def test_zero_hp_module_census_requires_complete_optimizer_tail(self):
    counts = _zero_module_counts("off")
    self.assertEqual(
        MODULE_CENSUS.validate_module_counts(
            "zero-hp", counts, p71_scan="off"
        ),
        [],
    )

    without_commit = dict(counts)
    del without_commit["jit__precomputed_gradient_commit"]
    self.assertIn(
        "jit__precomputed_gradient_commit=0!=1",
        MODULE_CENSUS.validate_module_counts(
            "zero-hp", without_commit, p71_scan="off"
        ),
    )

    short_scaled_step = dict(counts)
    short_scaled_step["jit__precomputed_gradient_scaled_step"] = 15
    self.assertIn(
        "jit__precomputed_gradient_scaled_step=15!=16",
        MODULE_CENSUS.validate_module_counts(
            "zero-hp", short_scaled_step, p71_scan="off"
        ),
    )
    missing_boundary = dict(counts)
    del missing_boundary["zt_tr_dp_parallel_bwd_adjoint"]
    self.assertIn(
        "missing_backward=zt_tr_dp_parallel_bwd_adjoint",
        MODULE_CENSUS.validate_module_counts(
            "zero-hp", missing_boundary, p71_scan="off"
        ),
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

  def test_module_census_backward_inventory_is_p71_mode_aware(self):
    """The census must red on the wrong program family in BOTH directions."""
    self.assertEqual(MODULE_CENSUS.p71_scan_mode(None), "off")
    for spelling in ("", "0", "off"):
      self.assertEqual(MODULE_CENSUS.p71_scan_mode(spelling), "off")
    self.assertEqual(MODULE_CENSUS.p71_scan_mode("fwd"), "fwd")
    self.assertEqual(MODULE_CENSUS.p71_scan_mode("bwd"), "bwd")
    with self.assertRaisesRegex(ValueError, "reserved"):
      MODULE_CENSUS.p71_scan_mode("full")
    with self.assertRaisesRegex(ValueError, "unset/0/off/fwd/bwd"):
      MODULE_CENSUS.p71_scan_mode("yes")

    # The block partition must track _P71_BWD_BLOCK_LAYERS' 7 -> 4 -> 2
    # fallback ladder, ceil(L / B) with a smaller remainder block.
    for block_layers, expected in ((7, 4), (4, 7), (2, 14), (1, 28)):
      self.assertEqual(
          MODULE_CENSUS.expected_block_indices(28, block_layers),
          tuple(range(expected)),
      )
    self.assertEqual(MODULE_CENSUS.expected_block_indices(29, 7), tuple(range(5)))
    with self.assertRaises(ValueError):
      MODULE_CENSUS.expected_block_indices(28, 0)

    for mode in MODULE_CENSUS.P71_SCAN_MODES:
      with self.subTest(mode=mode):
        self.assertEqual(
            MODULE_CENSUS.validate_module_counts(
                "zero-hp", _zero_module_counts(mode), p71_scan=mode
            ),
            [],
        )

    block_counts = _zero_module_counts("bwd")
    layer_counts = _zero_module_counts("off")
    for mode in ("off", "fwd"):
      with self.subTest(claimed=mode, ran="bwd"):
        reasons = MODULE_CENSUS.validate_module_counts(
            "zero-hp", block_counts, p71_scan=mode
        )
        self.assertIn(f"p71={mode}_unexpected_bwd_block=00,01,02,03", reasons)
        self.assertIn("missing_backward=zt_tr_dp_parallel_bwd_layer", reasons)
    reasons = MODULE_CENSUS.validate_module_counts(
        "zero-hp", layer_counts, p71_scan="bwd"
    )
    self.assertIn("p71=bwd_unexpected_bwd_layer=27", reasons)
    self.assertIn("missing_backward=zt_tr_dp_parallel_bwd_block", reasons)

    short_block_set = dict(block_counts)
    del short_block_set["jit_zt_tr_dp_parallel_bwd_block_03"]
    self.assertIn(
        "bwd_block_indices=00,01,02 expected=00,01,02,03",
        MODULE_CENSUS.validate_module_counts(
            "zero-hp", short_block_set, p71_scan="bwd"
        ),
    )
    short_block_execs = dict(block_counts)
    short_block_execs["jit_zt_tr_dp_parallel_bwd_block_01"] = 31
    self.assertIn(
        "bwd_block_01=31!=32",
        MODULE_CENSUS.validate_module_counts(
            "zero-hp", short_block_execs, p71_scan="bwd"
        ),
    )
    short_layer_execs = dict(layer_counts)
    short_layer_execs["jit_zt_tr_dp_parallel_bwd_layer_27"] = 864
    self.assertIn(
        "bwd_layer_execs=864!=896",
        MODULE_CENSUS.validate_module_counts(
            "zero-hp", short_layer_execs, p71_scan="off"
        ),
    )
    overflowing_layer = dict(layer_counts)
    del overflowing_layer["jit_zt_tr_dp_parallel_bwd_layer_27"]
    overflowing_layer["jit_zt_tr_dp_parallel_bwd_layer_28"] = 896
    self.assertIn(
        "bwd_layer_index_overflow=28 layers=28",
        MODULE_CENSUS.validate_module_counts(
            "zero-hp", overflowing_layer, p71_scan="off"
        ),
    )
    for mode in ("fwd", "bwd"):
      without_scan = dict(_zero_module_counts(mode))
      del without_scan[MODULE_CENSUS.FWD_TAPE_SCAN]
      self.assertIn(
          f"missing_forward_tape_scan={MODULE_CENSUS.FWD_TAPE_SCAN}",
          MODULE_CENSUS.validate_module_counts(
              "zero-hp", without_scan, p71_scan=mode
          ),
      )

    # CANON_P71_SCAN steers only the canonical adapter, so the stock arm
    # keeps its monolithic train_step contract under every rung.
    for mode in MODULE_CENSUS.P71_SCAN_MODES:
      self.assertEqual(
          MODULE_CENSUS.validate_module_counts(
              "native", {"jit__train_step": 16}, p71_scan=mode
          ),
          [],
      )

    # The carrier must hand the census the value it launched with, or the
    # census would assert an inventory nobody asked for.
    common = (SCRIPTS / "run_onehost_gsm8k_xprof_common.sh").read_text()
    self.assertIn('--p71-scan "${CANON_P71_SCAN:-}"', common)
    self.assertIn('-e CANON_P71_SCAN="${CANON_P71_SCAN:-}"', common)

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

  def test_arm_contract_is_a_signed_two_geometry_contract(self):
    """dp4 binds the dp4 workload, dp2 the dp2 workload; the rest refuses."""
    # Positive controls: the dp4 default parses exactly as before (no
    # geometry env at all), and both arms parse under the registered
    # second geometry.
    dp4_zero = _common_env("zero-hp")
    self.assertNotIn("CANON_V1_GSM8K_XPROF_GEOMETRY", dp4_zero)
    self.assertEqual(GSM8K_XPROF.arm(dp4_zero), "zero-hp")
    self.assertEqual(
        GSM8K_XPROF.arm(_common_env("zero-hp", geometry="dp2-tp2")),
        "zero-hp",
    )
    self.assertEqual(
        GSM8K_XPROF.arm(_common_env("native", geometry="dp2-tp2")),
        "native",
    )
    self.assertEqual(GSM8K_XPROF.geometry({}), "dp4-tp1")
    self.assertEqual(
        GSM8K_XPROF.geometry({"CANON_V1_GSM8K_XPROF_GEOMETRY": "dp2-tp2"}),
        "dp2-tp2",
    )
    self.assertEqual(GSM8K_XPROF.geometry_groups({}), 16)
    self.assertEqual(
        GSM8K_XPROF.geometry_groups(
            {"CANON_V1_GSM8K_XPROF_GEOMETRY": "dp2-tp2"}
        ),
        32,
    )

    # Junk geometry refuses for both arms and for the bare accessor.
    for arm in ("native", "zero-hp"):
      junk = _common_env(arm)
      junk["CANON_V1_GSM8K_XPROF_GEOMETRY"] = "dp8-tp8"
      with self.assertRaisesRegex(ValueError, "CANON_V1_GSM8K_XPROF_GEOMETRY"):
        GSM8K_XPROF.arm(junk)
    with self.assertRaisesRegex(ValueError, "CANON_V1_GSM8K_XPROF_GEOMETRY"):
      GSM8K_XPROF.geometry({"CANON_V1_GSM8K_XPROF_GEOMETRY": "dp1-tp4"})

    # Mismatched geometry/workload pairings refuse in both directions,
    # and an absent geometry with the dp2 workload refuses (absent means
    # the dp4 default).
    dp2_workload_no_geometry = _common_env("zero-hp")
    dp2_workload_no_geometry["CANON_P32_WORKLOAD"] = "gsm8k-p59-dp2-tp2"
    with self.assertRaisesRegex(ValueError, "DP4xTP1"):
      GSM8K_XPROF.arm(dp2_workload_no_geometry)
    dp2_geometry_dp4_workload = _common_env("zero-hp", geometry="dp2-tp2")
    dp2_geometry_dp4_workload["CANON_P32_WORKLOAD"] = "gsm8k-p59-dp4-tp1"
    with self.assertRaisesRegex(ValueError, "DP2xTP2"):
      GSM8K_XPROF.arm(dp2_geometry_dp4_workload)
    dp4_geometry_dp2_workload = _common_env("zero-hp")
    dp4_geometry_dp2_workload["CANON_V1_GSM8K_XPROF_GEOMETRY"] = "dp4-tp1"
    dp4_geometry_dp2_workload["CANON_P32_WORKLOAD"] = "gsm8k-p59-dp2-tp2"
    with self.assertRaisesRegex(ValueError, "DP4xTP1"):
      GSM8K_XPROF.arm(dp4_geometry_dp2_workload)

  def test_microstep_schedule_counts_are_geometry_registered(self):
    """16 and 32 are the two registered counts; the factory binds them."""
    for count in (16, 32):
      schedule = GSM8K_XPROF.ZeroHpTrainMicrostepSchedule(
          update_step=2, microsteps=count
      )
      self.assertEqual(schedule.microsteps, count)
    for count in (8, 24, 48, 0):
      with self.assertRaisesRegex(ValueError, "registered per-geometry"):
        GSM8K_XPROF.ZeroHpTrainMicrostepSchedule(
            update_step=2, microsteps=count
        )
    import unittest.mock as mock

    with mock.patch.dict(
        os.environ, _common_env("zero-hp"), clear=True
    ):
      self.assertEqual(
          GSM8K_XPROF.zero_hp_train_microsteps(
              update_step=2, microsteps=16
          ).microsteps,
          16,
      )
      with self.assertRaisesRegex(RuntimeError, "signed geometry"):
        GSM8K_XPROF.zero_hp_train_microsteps(update_step=2, microsteps=32)
    with mock.patch.dict(
        os.environ, _common_env("zero-hp", geometry="dp2-tp2"), clear=True
    ):
      self.assertEqual(
          GSM8K_XPROF.zero_hp_train_microsteps(
              update_step=2, microsteps=32
          ).microsteps,
          32,
      )
      with self.assertRaisesRegex(RuntimeError, "signed geometry"):
        GSM8K_XPROF.zero_hp_train_microsteps(update_step=2, microsteps=16)

  def test_run_stage_selects_the_update_horizon_fail_closed(self):
    """CANON_P33_RUN_STAGE, not a literal, sets the carrier's horizon."""
    common = (SCRIPTS / "run_onehost_gsm8k_xprof_common.sh").read_text()
    inner = (SCRIPTS / "run_onehost_gsm8k_xprof_inner.sh").read_text()
    # The registered stage flag is reused as the selector; no second flag.
    self.assertIn('run_stage="${CANON_P33_RUN_STAGE:-three-update}"', common)
    self.assertIn('-e CANON_P33_RUN_STAGE="$run_stage"', common)
    self.assertIn('-e V1_GSM8K_XPROF_RUN_STAGE="$run_stage"', common)
    self.assertIn('--expected-updates "$max_steps"', common)
    self.assertNotIn("CANON_P33_RUN_STAGE=three-update", common)
    self.assertIn('run_stage="${V1_GSM8K_XPROF_RUN_STAGE:-three-update}"', inner)
    for script in (common, inner):
      self.assertIn("  three-update) max_steps=3 ;;", script)
      self.assertIn("  six-update) max_steps=6 ;;", script)

    # An unknown stage is rejected before the carrier reads an asset, runs
    # git, checks the hostname, or touches docker and the TPU lane.  The
    # hostname and artifact-root guards below keep an admitted stage from
    # ever reaching a launch, whatever host runs this test.
    env = dict(os.environ)
    env["V1_GSM8K_XPROF_EXPECT_HOSTNAME"] = "__no_such_host__"
    env["V1_GSM8K_XPROF_ARTIFACT_DIR"] = str(SCRIPTS)
    for stage, rejected in (
        (None, False),
        ("three-update", False),
        ("six-update", False),
        ("full", True),
        ("backward-no-commit", True),
        ("p59-eight-update", True),
        ("seven-update", True),
    ):
      with self.subTest(stage=stage):
        case_env = dict(env)
        if stage is None:
          case_env.pop("CANON_P33_RUN_STAGE", None)
        else:
          case_env["CANON_P33_RUN_STAGE"] = stage
        result = subprocess.run(
            [
                "bash",
                str(SCRIPTS / "run_onehost_gsm8k_xprof_common.sh"),
                "zero-hp",
                "stagecontract",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=case_env,
        )
        self.assertNotEqual(result.returncode, 0, result)
        if rejected:
          self.assertEqual(result.returncode, 2, result)
          self.assertIn(
              f"unsupported CANON_P33_RUN_STAGE: {stage}", result.stderr
          )
        else:
          self.assertNotIn(
              "unsupported CANON_P33_RUN_STAGE", result.stderr, result
          )

  def test_arm_classifier_horizon_follows_the_stage(self):
    """The evidence gate counts the stage's updates, not a frozen 3."""
    keys = dict(
        source_sha="1" * 40,
        source_diff_sha256="2" * 64,
        runtime_manifest_sha256="5" * 64,
        model_snapshot="3" * 40,
        image_id="sha256:" + "4" * 64,
        xprof_census_rc=0,
        semantic_census_rc=0,
    )
    with tempfile.TemporaryDirectory() as directory:
      base = Path(directory)
      for updates in (3, 6):
        with self.subTest(updates=updates):
          root = base / f"zero-hp-{updates}"
          _fixture(root, "zero-hp", updates=updates)
          matched = ARM_CLASSIFIER.classify(
              arm="zero-hp",
              run_root=root,
              expected_updates=updates,
              **keys,
          )
          self.assertEqual(matched["verdict"], "PASS", matched)
          self.assertEqual(matched["capture"]["updates"], updates)
          other = 6 if updates == 3 else 3
          mismatched = ARM_CLASSIFIER.classify(
              arm="zero-hp",
              run_root=root,
              expected_updates=other,
              **keys,
          )
          self.assertEqual(mismatched["verdict"], "FAIL", mismatched)
          self.assertIn(
              f"global_steps={updates} expected={other}",
              mismatched["reasons"],
          )
          self.assertIn(
              f"work_receipts={updates} expected={other}",
              mismatched["reasons"],
          )
          self.assertTrue(
              any(
                  reason.startswith("zero_alignment=")
                  for reason in mismatched["reasons"]
              ),
              mismatched,
          )
      # The default stays three-update so an unchanged launcher is unmoved.
      default_root = base / "zero-hp-3"
      self.assertEqual(
          ARM_CLASSIFIER.classify(
              arm="zero-hp", run_root=default_root, **keys
          )["verdict"],
          "PASS",
      )
      with self.assertRaises(ValueError):
        ARM_CLASSIFIER.classify(
            arm="zero-hp", run_root=default_root, expected_updates=0, **keys
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
          "update_step=2 train_steps=32..47 host_plane=/host:CPU "
          "host_line=python3 steps_planes=8 forward_groups=16 "
          "reverse_transactions=16 "
          "transactions=16 micro_steps=0..15 last_accumulate=15 "
          "optimizer_owned_by_last=1 compiler_events=0\n"
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
          trace_census_rc=0,
      )
      self.assertEqual(revised_zero["verdict"], "PASS", revised_zero)
      size_receipt_path = base / "zero-hp/train/xprof_size_receipt.json"
      original_size_receipt = size_receipt_path.read_text()
      stale_size_receipt = json.loads(original_size_receipt)
      stale_size_receipt["total_bytes"] += 1
      size_receipt_path.write_text(json.dumps(stale_size_receipt) + "\n")
      stale_size = ARM_CLASSIFIER.classify(
          arm="zero-hp",
          run_root=base / "zero-hp",
          source_sha="1" * 40,
          source_diff_sha256="2" * 64,
          runtime_manifest_sha256="5" * 64,
          model_snapshot="3" * 40,
          image_id="sha256:" + "4" * 64,
          xprof_census_rc=0,
          semantic_census_rc=0,
          size_census_rc=0,
          require_hierarchy=True,
          hierarchy_census_rc=0,
          trace_census_rc=0,
      )
      self.assertEqual(stale_size["verdict"], "FAIL", stale_size)
      self.assertTrue(
          any("xprof_size_receipt.total_bytes=" in reason
              for reason in stale_size["reasons"]),
          stale_size,
      )
      size_receipt_path.write_text(original_size_receipt)
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
          trace_census_rc=0,
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

  def test_arm_classifier_derives_the_geometry_expectations(self):
    """dp2 runs classify against dp2 counts and never against dp4 counts."""
    keys = dict(
        source_sha="1" * 40,
        source_diff_sha256="2" * 64,
        runtime_manifest_sha256="5" * 64,
        model_snapshot="3" * 40,
        image_id="sha256:" + "4" * 64,
        xprof_census_rc=0,
        semantic_census_rc=0,
    )
    with tempfile.TemporaryDirectory() as directory:
      base = Path(directory)
      records = {}
      for arm in ("native", "zero-hp"):
        root = base / arm
        _fixture(root, arm, geometry="dp2-tp2")
        record = ARM_CLASSIFIER.classify(
            arm=arm, run_root=root, geometry="dp2-tp2", **keys
        )
        self.assertEqual(record["verdict"], "PASS", record)
        self.assertEqual(
            record["topology"], {"dp": 2, "tp": 2, "devices": 4}
        )
        records[arm] = record
      pair = PAIR_CLASSIFIER.classify(records["native"], records["zero-hp"])
      self.assertEqual(pair["verdict"], "PASS", pair)
      self.assertEqual(pair["topology"], {"dp": 2, "tp": 2, "devices": 4})

      # The same dp2 capture judged as dp4 rings on the preflight marker
      # and (zero arm) on the derived verdict count; a dp4 capture judged
      # as dp2 rings the same way.  Nothing silently passes across.
      cross = ARM_CLASSIFIER.classify(
          arm="zero-hp", run_root=base / "zero-hp", **keys
      )
      self.assertEqual(cross["verdict"], "FAIL")
      self.assertIn("preflight_marker", cross["reasons"])
      self.assertTrue(
          any(
              reason.startswith("zero_alignment=99/51")
              for reason in cross["reasons"]
          ),
          cross["reasons"],
      )
      dp4_root = base / "dp4-zero"
      _fixture(dp4_root, "zero-hp")
      cross = ARM_CLASSIFIER.classify(
          arm="zero-hp", run_root=dp4_root, geometry="dp2-tp2", **keys
      )
      self.assertEqual(cross["verdict"], "FAIL")
      self.assertIn("preflight_marker", cross["reasons"])
      self.assertTrue(
          any(
              reason.startswith("zero_alignment=51/99")
              for reason in cross["reasons"]
          ),
          cross["reasons"],
      )
      # A dp2-native x dp4-zero pair can never claim a matched capture.
      mixed = PAIR_CLASSIFIER.classify(
          records["native"],
          ARM_CLASSIFIER.classify(arm="zero-hp", run_root=dp4_root, **keys),
      )
      self.assertNotEqual(mixed["verdict"], "PASS")
      self.assertIn("topology", mixed["reasons"])
      with self.assertRaisesRegex(ValueError, "invalid geometry"):
        ARM_CLASSIFIER.classify(
            arm="zero-hp",
            run_root=base / "zero-hp",
            geometry="dp8-tp8",
            **keys,
        )

  def test_module_census_derives_geometry_counts(self):
    """Native train_steps, the optimizer tail, and the dp2 backward
    execution envelope all follow the geometry."""
    self.assertEqual(
        MODULE_CENSUS.validate_module_counts(
            "native", {"jit__train_step": 32}, geometry="dp2-tp2"
        ),
        [],
    )
    self.assertIn(
        "jit__train_step=16!=32",
        MODULE_CENSUS.validate_module_counts(
            "native", {"jit__train_step": 16}, geometry="dp2-tp2"
        ),
    )
    self.assertIn(
        "jit__train_step=32!=16",
        MODULE_CENSUS.validate_module_counts(
            "native", {"jit__train_step": 32}
        ),
    )
    with self.assertRaisesRegex(ValueError, "unknown geometry"):
      MODULE_CENSUS.validate_module_counts(
          "native", {"jit__train_step": 32}, geometry="dp8-tp8"
      )

    self.assertEqual(MODULE_CENSUS.expected_backward_execs("dp4-tp1"), 32)
    self.assertIsNone(MODULE_CENSUS.expected_backward_execs("dp2-tp2"))
    self.assertEqual(MODULE_CENSUS.backward_exec_bounds("dp2-tp2"), (32, 160))

    def dp2_counts(execs, p71_scan="off"):
      counts = {
          pattern.pattern: 1 for pattern in MODULE_CENSUS.ZERO_REQUIRED
      }
      counts["jit__precomputed_gradient_scaled_step"] = 32
      counts["jit__precomputed_gradient_commit"] = 1
      if p71_scan == "bwd":
        counts.update({
            f"jit_zt_tr_dp_parallel_bwd_block_{index:02d}": execs
            for index in MODULE_CENSUS.expected_block_indices()
        })
      else:
        counts["jit_zt_tr_dp_parallel_bwd_layer_27"] = (
            MODULE_CENSUS.ZERO_LAYER_COUNT * execs
        )
      if p71_scan in ("fwd", "bwd"):
        counts[MODULE_CENSUS.FWD_TAPE_SCAN] = execs
      return counts

    # The dp2 execution count is calibration-pending: any internally
    # consistent count inside groups x [1, 5] chunks passes, and the
    # first green capture pins the constant.
    for execs in (32, 64, 160):
      for mode in MODULE_CENSUS.P71_SCAN_MODES:
        with self.subTest(execs=execs, mode=mode):
          self.assertEqual(
              MODULE_CENSUS.validate_module_counts(
                  "zero-hp",
                  dp2_counts(execs, mode),
                  p71_scan=mode,
                  geometry="dp2-tp2",
              ),
              [],
          )
    self.assertIn(
        "bwd_layer_execs_per_layer=31 outside=32..160",
        MODULE_CENSUS.validate_module_counts(
            "zero-hp", dp2_counts(31), geometry="dp2-tp2"
        ),
    )
    self.assertIn(
        "bwd_layer_execs=897 not_a_multiple_of=28",
        MODULE_CENSUS.validate_module_counts(
            "zero-hp",
            {**dp2_counts(32), "jit_zt_tr_dp_parallel_bwd_layer_27": 897},
            geometry="dp2-tp2",
        ),
    )
    inconsistent = dp2_counts(64, "bwd")
    inconsistent["jit_zt_tr_dp_parallel_bwd_block_01"] = 63
    self.assertIn(
        "bwd_block_execs_inconsistent=63,64",
        MODULE_CENSUS.validate_module_counts(
            "zero-hp", inconsistent, p71_scan="bwd", geometry="dp2-tp2"
        ),
    )
    wrong_tape = dp2_counts(64, "fwd")
    wrong_tape[MODULE_CENSUS.FWD_TAPE_SCAN] = 32
    self.assertIn(
        "forward_tape_scan=32!=64",
        MODULE_CENSUS.validate_module_counts(
            "zero-hp", wrong_tape, p71_scan="fwd", geometry="dp2-tp2"
        ),
    )
    # dp2 never inherits the dp4 tail: 16 scaled steps red under dp2.
    dp4_tail = dp2_counts(64)
    dp4_tail["jit__precomputed_gradient_scaled_step"] = 16
    self.assertIn(
        "jit__precomputed_gradient_scaled_step=16!=32",
        MODULE_CENSUS.validate_module_counts(
            "zero-hp", dp4_tail, geometry="dp2-tp2"
        ),
    )
    # And the dp4 contract is untouched: the frozen 16 x 2 = 32 pin holds,
    # including the newly pinned forward-tape count measured on every
    # landed fwd/bwd capture.
    self.assertEqual(
        MODULE_CENSUS.validate_module_counts(
            "zero-hp", _zero_module_counts("off"), p71_scan="off"
        ),
        [],
    )
    short_tape = _zero_module_counts("fwd")
    short_tape[MODULE_CENSUS.FWD_TAPE_SCAN] = 31
    self.assertIn(
        "forward_tape_scan=31!=32",
        MODULE_CENSUS.validate_module_counts(
            "zero-hp", short_tape, p71_scan="fwd"
        ),
    )

  def test_semantic_census_peft_expectation_follows_geometry(self):
    SEMANTIC_CENSUS = _load(
        "v1_gsm8k_semantic_census",
        SCRIPTS / "census_gsm8k_semantic_trace.py",
    )
    self.assertEqual(SEMANTIC_CENSUS.expected_peft_train("native", "dp4-tp1"), 32)
    self.assertEqual(SEMANTIC_CENSUS.expected_peft_train("native", "dp2-tp2"), 64)
    self.assertEqual(SEMANTIC_CENSUS.expected_peft_train("zero-hp", "dp4-tp1"), 2)
    self.assertEqual(SEMANTIC_CENSUS.expected_peft_train("zero-hp", "dp2-tp2"), 2)
    with self.assertRaisesRegex(ValueError, "unknown geometry"):
      SEMANTIC_CENSUS.expected_peft_train("native", "dp8-tp8")

  def test_launcher_carries_the_geometry_and_refuses_junk(self):
    """The selector, its container passthrough, the label marking, and the
    per-geometry tables are all present; junk refuses before any launch."""
    common = (SCRIPTS / "run_onehost_gsm8k_xprof_common.sh").read_text()
    inner = (SCRIPTS / "run_onehost_gsm8k_xprof_inner.sh").read_text()
    self.assertIn('geometry="${V1_GSM8K_XPROF_GEOMETRY:-dp4-tp1}"', common)
    self.assertIn('geometry="${V1_GSM8K_XPROF_GEOMETRY:-dp4-tp1}"', inner)
    # The container passthrough exists for the non-default geometry (the
    # dp4 docker env block stays byte-identical), and both spellings ride
    # along: the bash entrypoint reads V1_*, the Python contract CANON_*.
    self.assertIn('-e V1_GSM8K_XPROF_GEOMETRY="$geometry"', common)
    self.assertIn('-e CANON_V1_GSM8K_XPROF_GEOMETRY="$geometry"', common)
    self.assertIn('if [ "$geometry" != dp4-tp1 ]; then', common)
    # A dp2 run is never confusable with a dp4 run.
    self.assertIn('label="dp2tp2-${label}"', common)
    # Per-geometry tables: profile, engine overlay, mesh-id expectation,
    # serial-bridge selection.
    self.assertIn("zero_profile=qwen3-1p7b-dp4-tp1-gsm8k-v1-hp.env", common)
    self.assertIn("zero_profile=qwen3-1p7b-dp2-tp2-gsm8k-v1-hp.env", common)
    self.assertIn("zero_model_overlay=qwen1p7b_tp1", common)
    self.assertIn("zero_model_overlay=qwen1p7b_tp2", common)
    self.assertIn('--model "$zero_model_overlay"', common)
    self.assertIn("expected_train_mesh_ids=0,2,1,3", common)
    self.assertIn("expected_train_mesh_ids=0,1,2,3", common)
    self.assertIn(
        '-e CANON_EXPECT_TRAIN_MESH_IDS="$expected_train_mesh_ids"', common
    )
    self.assertIn("serial_mesh_bridge=1", common)
    self.assertIn("serial_mesh_bridge=0", common)
    self.assertIn(
        '-e CANON_P59_DP4_SERIAL_MESH_BRIDGE="$serial_mesh_bridge"', common
    )
    self.assertIn('topology=$topology', common)
    # Every geometry-sensitive judge receives the launched geometry.
    self.assertIn('--geometry "$geometry"', common)
    for tool in (
        "census_gsm8k_xprof_modules.py",
        "census_gsm8k_semantic_trace.py",
        "census_gsm8k_xprof_hierarchy.py",
        "census_gsm8k_xprof_trace.py",
        "classify_gsm8k_xprof_arm.py",
    ):
      # Each tool is named twice (runtime manifest, then its invocation);
      # the invocation is the last occurrence.
      block = common[common.rindex(tool):]
      self.assertIn('--geometry "$geometry"', block[:600], tool)
    # The inner entrypoint derives mesh, microbatch, and rollout budgets
    # from the same selector and sources the per-geometry profile.
    self.assertIn("mesh_dp=4; mesh_tp=1; trajectory_micro=4; vllm_max_seqs=16", inner)
    self.assertIn("mesh_dp=2; mesh_tp=2; trajectory_micro=2; vllm_max_seqs=32", inner)
    self.assertIn('--mesh_dp="$mesh_dp" --mesh_tp="$mesh_tp"', inner)
    self.assertIn('--train_trajectory_micro_batch_size="$trajectory_micro"', inner)
    self.assertIn('--rollout_vllm_max_num_seqs="$vllm_max_seqs"', inner)
    self.assertIn("profiles/$zero_profile", inner)
    self.assertIn("unsupported V1_GSM8K_XPROF_GEOMETRY", inner)

    # Junk refuses in the launcher before assets, git, hostname, or docker.
    env = dict(os.environ)
    env["V1_GSM8K_XPROF_EXPECT_HOSTNAME"] = "__no_such_host__"
    env["V1_GSM8K_XPROF_ARTIFACT_DIR"] = str(SCRIPTS)
    for geometry, rejected in (
        (None, False),
        ("dp4-tp1", False),
        ("dp2-tp2", False),
        ("dp8-tp8", True),
        ("junk", True),
    ):
      with self.subTest(geometry=geometry):
        case_env = dict(env)
        if geometry is None:
          case_env.pop("V1_GSM8K_XPROF_GEOMETRY", None)
        else:
          case_env["V1_GSM8K_XPROF_GEOMETRY"] = geometry
        result = subprocess.run(
            [
                "bash",
                str(SCRIPTS / "run_onehost_gsm8k_xprof_common.sh"),
                "zero-hp",
                "geometrycontract",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=case_env,
        )
        self.assertNotEqual(result.returncode, 0, result)
        if rejected:
          self.assertEqual(result.returncode, 2, result)
          self.assertIn(
              f"unsupported V1_GSM8K_XPROF_GEOMETRY: {geometry}",
              result.stderr,
          )
        else:
          self.assertNotIn(
              "unsupported V1_GSM8K_XPROF_GEOMETRY", result.stderr, result
          )

  def test_static_runner_is_gsm8k_and_wrappers_select_one_arm(self):
    common = (SCRIPTS / "run_onehost_gsm8k_xprof_common.sh").read_text()
    inner = (SCRIPTS / "run_onehost_gsm8k_xprof_inner.sh").read_text()
    self.assertIn("models--Qwen--Qwen3-1.7B", common)
    self.assertIn('--model "$zero_model_overlay"', common)
    self.assertIn("examples/math_gsm8k/qwen3_grpo_demo.py", inner)
    self.assertIn('--mesh_dp="$mesh_dp" --mesh_tp="$mesh_tp"', inner)
    self.assertIn('--max_steps="$max_steps"', inner)
    self.assertNotIn("--max_steps=3", inner)
    self.assertIn("-e CANON_P60_DETERMINISTIC_AB=1", common)
    self.assertIn("census_gsm8k_xprof_modules.py", common)
    self.assertIn("census_gsm8k_semantic_trace.py", common)
    self.assertIn("census_gsm8k_xprof_hierarchy.py", common)
    self.assertIn("census_gsm8k_xprof_trace.py", common)
    self.assertIn("census_gsm8k_xprof_size.py", common)
    self.assertIn("census_gsm8k_p74_gap.py", common)
    self.assertIn("--require-hierarchy", common)
    self.assertIn('--trace-census-rc "$trace_census_rc"', common)
    self.assertIn('--size-census-rc "$size_census_rc"', common)
    self.assertIn('--require-p74-gap --p74-gap-census-rc', common)
    self.assertIn("-e CANON_XPROF_SKIP_STEPS=2", common)
    # The TPU trace mode is passed through with the update-phase default so
    # the rollout wrappers can clear it; the arm contract enforces the
    # phase/trace-mode pairing.
    self.assertIn(
        'CANON_XPROF_TPU_TRACE_MODE="${CANON_XPROF_TPU_TRACE_MODE'
        '-TRACE_ONLY_XLA}"',
        common,
    )
    self.assertIn("-e CANON_PERF_TRACE_EXPORT_STEP=2", common)
    self.assertIn("finalize_gsm8k_xprof_evidence.sh", common)
    self.assertIn('sha_inputs=("$raw" "$driver")', common)
    self.assertIn('"$xprof_census" "$semantic_census" "$classification"', common)
    self.assertIn('"$size_census" "$size_receipt"', common)
    self.assertIn('"$p74_gap_census" "$p74_gap_receipt"', common)
    self.assertIn('find "$xprof_dir" -type f', common)
    self.assertIn("hard:1500000000", common)
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
    p74 = (
        SCRIPTS / "run_onehost_xprof_backward_p74_dp2tp2.sh"
    ).read_text()
    self.assertIn("V1_GSM8K_XPROF_GEOMETRY=dp2-tp2", p74)
    self.assertIn("CANON_DP_COMPARE_MODE=fingerprint-hybrid", p74)
    self.assertIn("CANON_DP_DISTINCT_SCHEDULE=first-group-warmup", p74)
    self.assertIn("CANON_DP_FINITE_FETCH=batched-commit", p74)
    self.assertIn("CANON_P71_SCAN=fwd", p74)
    self.assertNotIn("CANON_P66_P59_CHECK_VMA", p74)


if __name__ == "__main__":
  unittest.main()
