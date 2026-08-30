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
TRAIN_SCRIPT = ROOT / "examples/deepswe/train_deepswe_nb.py"
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
      "rollout_seed": 42,
      "seed_scope": "engine-global; async completion order not claimed",
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

  def _zero_admission_env(self) -> dict[str, str]:
    return {
        **self._env("zero-hp"),
        "CANON_P58_ONEHOST_SEAM_PROBE": "1",
        "CANON_P58_Q4_TP4_ZERO_ADMISSION": "1",
        "CANON_P58_TIM_ADMITTED": "0",
        "CANON_P58_NATIVE_STOCK_PROMPT_OBSERVER": "0",
        "CANON_PROFILE": "qwen3-4b-dp1-tp4-deepswe-zero",
        "CANON_MODEL_DIR_NAME": "qwen4b_tp4",
        "CANON_QWEN3_HIDDEN_SIZE": "2560",
        "CANON_QWEN3_TP_SIZE": "4",
        "CANON_P38_FIXED_LM_HEAD": "1",
        "CANON_P59_RANK_PARALLEL_BACKWARD": "0",
        "CANON_DEEPSWE_ALIGNMENT_WARN_ONLY": "0",
        "CANON_CONTINUE_DECODE": "8",
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
    self.assertFalse(deepswe_debug.onehost_seam_probe(self._env("zero-hp")))
    self.assertTrue(deepswe_debug.onehost_seam_probe({
        **self._env("zero-hp"),
        "CANON_P58_ONEHOST_SEAM_PROBE": "1",
    }))
    seam_manifest = deepswe_debug._manifest(
        {
            **self._env("zero-hp"),
            "CANON_P58_ONEHOST_SEAM_PROBE": "1",
        },
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        output_dir=Path("/tmp/p58-seam-probe"),
    )
    self.assertEqual(
        seam_manifest["contract_name"],
        "local-qwen4b-dp1-tp4-seam-probe",
    )
    self.assertEqual(seam_manifest["global_trajectories"], 2)
    self.assertEqual(seam_manifest["max_response_length"], 4096)
    self.assertEqual(seam_manifest["max_turns"], 16)
    with self.assertRaises(ValueError):
      deepswe_debug.onehost_seam_probe({
          **self._env("native"),
          "CANON_P58_ONEHOST_SEAM_PROBE": "1",
      })
    admission_env = self._zero_admission_env()
    self.assertTrue(deepswe_debug.q4_tp4_zero_admission(admission_env))
    admission_manifest = deepswe_debug._manifest(
        admission_env,
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        output_dir=Path("/tmp/p58-q4-tp4-zero-admission"),
    )
    self.assertEqual(
        admission_manifest["contract_name"],
        "local-qwen4b-dp1-tp4-zero-admission",
    )
    self.assertEqual(admission_manifest["sampling_contract"], {
        "source": "explicit-cli",
        "temperature": 0.7,
        "top_k": 0,
        "top_p": 1.0,
    })
    self.assertEqual(admission_manifest["q4_tp4_seam_diagnostic"], "")
    self.assertFalse(admission_manifest["q4_tp4_continue_kv_diagnostic"])
    self.assertFalse(admission_manifest["q4_tp4_short_backward"])
    self.assertEqual(admission_manifest["continue_decode_steps"], "8")
    standard_decode_env = {
        **admission_env,
        "CANON_P58_Q4_TP4_SEAM_DIAGNOSTIC": "standard-decode",
        "CANON_CONTINUE_DECODE": "",
    }
    self.assertEqual(
        deepswe_debug.q4_tp4_seam_diagnostic(standard_decode_env),
        "standard-decode",
    )
    standard_manifest = deepswe_debug._manifest(
        standard_decode_env,
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        output_dir=Path("/tmp/p58-q4-tp4-standard-decode"),
    )
    self.assertEqual(
        standard_manifest["q4_tp4_seam_diagnostic"], "standard-decode"
    )
    self.assertEqual(standard_manifest["continue_decode_steps"], "")
    continue_kv_env = {
        **admission_env,
        "CANON_P58_Q4_TP4_CONTINUE_KV_DIAGNOSTIC": "1",
        "CANON_P38_PRECHECK_ONLY": "1",
        "CANON_P38_CONTROLLED_EXIT": "1",
        "CANON_P38_DIAGNOSTIC_ROUNDS": "1",
        "CANON_P38_KV_OBSERVER_DIR": "/tmp/p58-continue-kv",
        "CANON_P38_KV_OBSERVER_MAX_CANDIDATES": "1",
        "CANON_P38_KV_OBSERVER_MAX_PAGES": "192",
        "CANON_P38_KV_OBSERVER_MAX_BYTES": "134217728",
        "CANON_P38_KV_OBSERVER_MAX_READ_BYTES": "671088640",
        "CANON_P38_SERVING_CAPTURE_EXPECTED_PATH": "standard",
        "CANON_P58_Q4_TP4_CONTINUE_KV_MIN_PREFIX": "2280",
        "CANON_P58_Q4_TP4_CONTINUE_KV_MAX_PREFIX": "3072",
    }
    self.assertTrue(
        deepswe_debug.q4_tp4_continue_kv_diagnostic(continue_kv_env)
    )
    continue_kv_manifest = deepswe_debug._manifest(
        continue_kv_env,
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        output_dir=Path("/tmp/p58-q4-tp4-continue-kv"),
    )
    self.assertTrue(continue_kv_manifest["q4_tp4_continue_kv_diagnostic"])
    self.assertTrue(continue_kv_manifest["alignment_precheck_only"])
    self.assertTrue(continue_kv_manifest["alignment_controlled_exit"])
    self.assertEqual(continue_kv_manifest["continue_decode_steps"], "8")
    short_backward_env = {
        **admission_env,
        "CANON_P58_Q4_TP4_SHORT_BACKWARD": "1",
    }
    self.assertTrue(
        deepswe_debug.q4_tp4_short_backward(short_backward_env)
    )
    short_manifest = deepswe_debug._manifest(
        short_backward_env,
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        output_dir=Path("/tmp/p58-q4-tp4-short-backward"),
    )
    self.assertTrue(short_manifest["q4_tp4_short_backward"])
    self.assertEqual(short_manifest["max_prompt_length"], 1792)
    self.assertEqual(short_manifest["max_response_length"], 2880)
    self.assertEqual(short_manifest["max_turns"], 16)
    carrier_screen_env = {
        **short_backward_env,
        "CANON_P58_Q4_TP4_CARRIER_SCREEN": "1",
        "CANON_DEEPSWE_ONEHOST_STAGE": "rollout-only",
        "CANON_DEEPSWE_ONEHOST_NO_COMMIT": "0",
        "CANON_DEEPSWE_ONEHOST_ROLLOUT_ONLY": "1",
    }
    self.assertEqual(
        deepswe_debug.onehost_xprof_arm(carrier_screen_env), "zero-hp"
    )
    self.assertTrue(
        deepswe_debug.q4_tp4_carrier_screen(carrier_screen_env)
    )
    carrier_manifest = deepswe_debug._manifest(
        carrier_screen_env,
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        output_dir=Path("/tmp/p58-q4-tp4-carrier-screen"),
    )
    self.assertTrue(carrier_manifest["q4_tp4_carrier_screen"])
    self.assertEqual(carrier_manifest["stage"], "rollout-only")
    self.assertEqual(carrier_manifest["generations"], 16)
    self.assertEqual(carrier_manifest["global_trajectories"], 16)
    self.assertEqual(carrier_manifest["sampling_contract"], {
        "source": "explicit-cli",
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
    })
    with self.assertRaises(ValueError):
      deepswe_debug.q4_tp4_carrier_screen({
          **carrier_screen_env,
          "CANON_P58_Q4_TP4_SHORT_BACKWARD": "0",
      })
    with self.assertRaises(ValueError):
      deepswe_debug.q4_tp4_carrier_screen({
          **carrier_screen_env,
          "CANON_DEEPSWE_ONEHOST_STAGE": "backward-no-commit",
      })
    replay_source = Path(
        "/mnt/disks/tunix-data/deepswe-replay-sources/"
        "p58-q4-b2g2-k2560-v2/"
        "batch-000000.trajectories.jsonl.gz"
    )
    with tempfile.TemporaryDirectory() as directory:
      replay_env = {
          **short_backward_env,
          "CANON_P58_Q4_TP4_TRAJECTORY_REPLAY": "1",
          "CANON_P58_REPLAY_JOURNAL": str(replay_source),
          "CANON_P58_REPLAY_JOURNAL_SHA256": (
              "091a9273c2067876fbee1996ee853e3c8"
              "e861352e307cd5fb94fea2563aec456"
          ),
          "CANON_P59_RANK_PARALLEL_BACKWARD": "0",
          "CANON_P28_SEGMENTED_FORWARD": "1",
          "CANON_P28_SEGMENTED_VJP": "0",
          "CANON_P28_SEGMENTED_TRAIN": "1",
          "CANON_P28_G6_UPDATE": "1",
          "CANON_P29_FULL_TRAIN": "1",
          "CANON_P30_SPARSE_GRAD_ASSEMBLY": "1",
          "CANON_P30_FUSED_PAIR_ACCUMULATION": "0",
          "CANON_P30_REUSE_SEGMENTED_ENGINE": "1",
          "CANON_P30_RELEASE_CAPTURED_STATE": "1",
          "CANON_P30_RESHARD_ACCUMULATOR": "1",
          "CANON_P28_BATCHED_REPORT": "1",
          "CANON_P28_BATCHED_REVERSE": "0",
          "CANON_BATCHED_EVIDENCE": "0",
          "CANON_P71_SCAN": "fwd",
          "CANON_DEEPSWE_ONEHOST_DEBUG_DIR": directory,
      }
      self.assertTrue(
          deepswe_debug.q4_tp4_trajectory_replay(replay_env)
      )
      self.assertEqual(
          deepswe_debug.q4_tp4_trajectory_replay_update_geometry(replay_env),
          (4, 2),
      )
      replay_manifest = deepswe_debug._manifest(
          replay_env,
          model_id="Qwen/Qwen3-4B-Instruct-2507",
          output_dir=Path(directory),
      )
      self.assertTrue(replay_manifest["q4_tp4_trajectory_replay"])
      self.assertEqual(replay_manifest["sampling_contract"], {
          "source": "explicit-cli",
          "temperature": 1.0,
          "top_k": 0,
          "top_p": 1.0,
      })
      self.assertEqual(replay_manifest["global_prompts"], 2)
      self.assertEqual(replay_manifest["global_trajectories"], 4)
      self.assertEqual(replay_manifest["max_prompt_length"], 2048)
      self.assertEqual(replay_manifest["max_response_length"], 512)
      self.assertEqual(
          deepswe_debug.q4_tp4_replay_sampling_contract(),
          {
              "temperature": 1.0,
              "top_p": 1.0,
              "top_k": 0,
              "source_identity": (
                  "p58s22lr3_20260829t2256z@"
                  "16c224aa80eb6b3a544be19f693c0542ab4b0dcb:"
                  "rows7,0x2:B2G2"
              ),
          },
      )
      with self.assertRaises(ValueError):
        deepswe_debug.q4_tp4_trajectory_replay({
            **replay_env,
            "CANON_P58_REPLAY_JOURNAL_SHA256": "0" * 64,
        })
    with self.assertRaises(ValueError):
      deepswe_debug.q4_tp4_seam_diagnostic({
          **admission_env,
          "CANON_P58_Q4_TP4_SEAM_DIAGNOSTIC": "unknown",
      })
    with self.assertRaises(ValueError):
      deepswe_debug.q4_tp4_zero_admission({
          **admission_env,
          "CANON_P58_Q4_TP4_SEAM_DIAGNOSTIC": "standard-decode",
      })
    with self.assertRaises(ValueError):
      deepswe_debug.q4_tp4_continue_kv_diagnostic({
          **continue_kv_env,
          "CANON_P58_Q4_TP4_SEAM_DIAGNOSTIC": "standard-decode",
          "CANON_CONTINUE_DECODE": "",
      })
    with self.assertRaises(ValueError):
      deepswe_debug.q4_tp4_continue_kv_diagnostic({
          **continue_kv_env,
          "CANON_P38_KV_OBSERVER_MAX_CANDIDATES": "2",
      })
    with self.assertRaises(ValueError):
      deepswe_debug.q4_tp4_short_backward({
          **short_backward_env,
          "CANON_P58_Q4_TP4_CONTINUE_KV_DIAGNOSTIC": "1",
      })
    with self.assertRaises(ValueError):
      deepswe_debug.q4_tp4_short_backward({
          **short_backward_env,
          "CANON_P58_Q4_TP4_SEAM_DIAGNOSTIC": "standard-decode",
          "CANON_CONTINUE_DECODE": "",
      })
    with self.assertRaises(ValueError):
      deepswe_debug.q4_tp4_short_backward({
          **short_backward_env,
          "CANON_P38_PRECHECK_ONLY": "1",
      })
    for key, value in (
        ("CANON_P38_PRECHECK_ONLY", "0"),
        ("CANON_P38_CONTROLLED_EXIT", "0"),
        ("CANON_P38_DIAGNOSTIC_ROUNDS", "2"),
        ("CANON_P58_Q4_TP4_CONTINUE_KV_MIN_PREFIX", "2279"),
        ("CANON_P58_Q4_TP4_CONTINUE_KV_MAX_PREFIX", "3073"),
        ("CANON_P38_SERVING_CAPTURE_DIR", "/tmp/foreign-serving-capture"),
    ):
      with self.subTest(key=key), self.assertRaises(ValueError):
        deepswe_debug.q4_tp4_continue_kv_diagnostic({
            **continue_kv_env,
            key: value,
        })
    with self.assertRaises(ValueError):
      deepswe_debug.q4_tp4_zero_admission({
          **admission_env,
          "CANON_P38_FIXED_LM_HEAD": "0",
      })

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
    train_script = TRAIN_SCRIPT.read_text()
    self.assertIn(
        "expected_prompt_length = (\n"
        "      2048\n"
        "      if P58_Q4_TP4_TRAJECTORY_REPLAY\n"
        "      else 1792\n"
        "      if P58_Q4_TP4_SHORT_BACKWARD",
        train_script,
    )
    self.assertIn(
        "expected_response_length = (\n"
        "      8192\n"
        "      if P58_Q4_TP4_CARRIER_SCREEN\n"
        "      else 512\n"
        "      if P58_Q4_TP4_TRAJECTORY_REPLAY\n"
        "      else 2880\n"
        "      if P58_Q4_TP4_SHORT_BACKWARD",
        train_script,
    )
    for marker in (
        "P58_ONEHOST_EXPECT_HOSTNAME",
        "P58_ONEHOST_COMPILATION_CACHE_DIR",
        'status --porcelain)',
        "ls-files --others --exclude-standard",
        "CANON_XPROF_PHASE=update",
        "CANON_XPROF_TPU_TRACE_MODE=TRACE_COMPUTE",
        "CANON_PERF_TRACE_EXPORT_STEP=0",
        "CANON_P38_FIXED_LM_HEAD=0",
        "CANON_P58_Q4_TP4_ZERO_ADMISSION",
        "CANON_P58_Q4_TP4_SEAM_DIAGNOSTIC",
        "CANON_P58_Q4_TP4_CONTINUE_KV_DIAGNOSTIC",
        "CANON_P58_Q4_TP4_SHORT_BACKWARD",
        "CANON_P58_Q4_TP4_CARRIER_SCREEN",
        "CANON_P58_Q4_TP4_TRAJECTORY_REPLAY",
        "CANON_P59_RANK_PARALLEL_BACKWARD=0",
        "CANON_P71_SCAN",
        'carrier_prompts=2',
        '--batch_size "$carrier_prompts"',
        '--mini_batch_size "$carrier_prompts"',
        '--max_concurrency "$carrier_max_concurrency"',
        "carrier_generations=16",
        "carrier_max_concurrency=8",
        "carrier_max_num_seqs=16",
        "P58_ONEHOST_PROBE_PROFILE",
        "CANON_P58_ONEHOST_SEAM_PROBE",
        "classify_decode_prefill_probe.py",
        "classify_continue_kv_probe.py",
        "classify_short_carrier_screen.py",
        "classify_trajectory_replay.py",
        "CANON_P38_KV_OBSERVER_MAX_CANDIDATES=1",
        "CANON_P38_PRECHECK_ONLY=1",
        "CANON_P38_CONTROLLED_EXIT=1",
        "P58_SEAM_PROBE_RETURN.tar.gz",
        "max_response_length=4096",
        "An 8,192-token train width",
        "max_prompt_length=1792",
        "max_prompt_length=2048",
        "max_response_length=512",
        "max_response_length=2880",
        "1,737-token prompt",
        "7294da90559ebace771b7bd3fd8be01de87e0ae9bcb7ae1e317dbe5a6ed0db9f",
        "expected_filtered_rows=1",
        '--expected_filtered_rows "$expected_filtered_rows"',
        "26e06ab7469987b4bc0c66d683e8468c2f10ae7d6842b0e138e563adcf87e257",
        "prompt_identity=repeated-strict-exact",
        '--from-path "$tpu_inference_path" --model "$shim_model"',
        "shim_model=qwen4b_tp4",
        'importlib.util.find_spec("tpu_inference")',
        '"$r2egym_root/src/r2egym/__init__.py"',
        'PYTHONPATH="$repo:$r2egym_root/src',
        "P58_ONEHOST_R2EGYM_HOST_SITE",
        'git -C "$r2egym_runtime_root" apply "$r2egym_patch"',
        "r2egym_patch_sha256=",
        "python_dep_overlay=",
        "never expose the Python 3.11 venv's binary",
        'python_overlay/swebench/__init__.py',
        'gold_whitelist_sha256=',
        'export CANON_P34_WHITELIST_SHA256="$gold_whitelist_sha256"',
        "7294da90559ebace771b7bd3fd8be01de87e0ae9bcb7ae1e317dbe5a6ed0db9f",
        '--metric_logger_dir "$artifact_dir/metrics"',
        'zero_overlay_root="$artifact_dir/install/zero-overlay"',
        'cp -a "$tpu_inference_path/." "$zero_package/"',
        "zero_overlay_sources=(",
        "zero_overlay_targets=(",
        (
            "[P58.ONEHOST.ZERO_OVERLAY] PASS files=1 scope=runner-only "
            "qwen4b_tp8_model_shims=excluded"
        ),
        "[P58.20] ZERO_OVERLAY_PASS files=7",
        'importlib.util.find_spec("tpu_inference")',
        'root / "runner/tpu_runner.py"',
        'export PYTHONPATH="$zero_overlay_root:$shim_root',
    ):
      self.assertIn(marker, common)
    self.assertIn("P58_REPLAY_UPDATE_GEOMETRY", train_script)
    self.assertIn("tpu_runner_p21_l30.py", common)
    for required in (
        "attn_iface_patched.py",
        "linear_p22xk.py",
        "embed_patched.py",
        "qwen3_p22xk.py",
        "qwen2_p22xk.py",
        "rpa_kernel_p66.py",
    ):
      self.assertIn(required, common)
    train_source = (ROOT / "examples/deepswe/train_deepswe_nb.py").read_text()
    self.assertIn("from jax.experimental import mesh_utils", train_source)
    self.assertIn(
        "mesh_utils.create_device_mesh(\n"
        "      (1, 4), devices, allow_split_physical_axes=True\n"
        "  )",
        train_source,
    )
    self.assertIn("device_ids={shared_device_ids}", train_source)
    launch = common.split("timeout --signal=TERM", 1)[1].split(
        "run_status=$?", 1
    )[0]
    self.assertNotIn("|", launch)
    self.assertIn('>> "$raw_log" 2>&1', launch)
    native = (SCRIPTS / "run_onehost_deepswe_xprof_native.sh").read_text()
    zero = (SCRIPTS / "run_onehost_deepswe_xprof_zero_hp.sh").read_text()
    seam = (SCRIPTS / "run_onehost_deepswe_seam_probe.sh").read_text()
    docker = (
        SCRIPTS / "run_onehost_deepswe_seam_probe_docker.sh"
    ).read_text()
    admission = (
        SCRIPTS / "run_onehost_deepswe_zero_admission.sh"
    ).read_text()
    admission_docker = (
        SCRIPTS / "run_onehost_deepswe_zero_admission_docker.sh"
    ).read_text()
    standard_decode = (
        SCRIPTS / "run_onehost_deepswe_zero_standard_decode.sh"
    ).read_text()
    standard_decode_docker = (
        SCRIPTS / "run_onehost_deepswe_zero_standard_decode_docker.sh"
    ).read_text()
    continue_kv = (
        SCRIPTS / "run_onehost_deepswe_zero_continue_kv.sh"
    ).read_text()
    continue_kv_docker = (
        SCRIPTS / "run_onehost_deepswe_zero_continue_kv_docker.sh"
    ).read_text()
    short_backward = (
        SCRIPTS / "run_onehost_deepswe_zero_short_backward.sh"
    ).read_text()
    short_backward_docker = (
        SCRIPTS / "run_onehost_deepswe_zero_short_backward_docker.sh"
    ).read_text()
    carrier_screen = (
        SCRIPTS / "run_onehost_deepswe_zero_carrier_screen.sh"
    ).read_text()
    carrier_screen_docker = (
        SCRIPTS / "run_onehost_deepswe_zero_carrier_screen_docker.sh"
    ).read_text()
    trajectory_replay = (
        SCRIPTS / "run_onehost_deepswe_zero_trajectory_replay.sh"
    ).read_text()
    trajectory_replay_docker = (
        SCRIPTS / "run_onehost_deepswe_zero_trajectory_replay_docker.sh"
    ).read_text()
    self.assertIn("common.sh\" native", native)
    self.assertIn("common.sh\" zero-hp", zero)
    self.assertIn("P58_ONEHOST_PROBE_PROFILE=seam", seam)
    self.assertIn("p58z07_group3_pillow.jsonl", seam)
    self.assertIn("p58z07_group3_pillow.jsonl", seam)
    self.assertIn("--privileged --net=host --ipc=host --uts=host", docker)
    self.assertIn("/var/run/docker.sock:/var/run/docker.sock", docker)
    self.assertIn("P58_ONEHOST_DOCKER_SDK_PATH", docker)
    self.assertIn("-e PYTHONPATH=/opt/p58-deps", docker)
    self.assertIn("P58_ONEHOST_ALLOW_DIRTY", docker)
    self.assertIn('safe.directory "$2/.git"', docker)
    self.assertIn("CANON_P58_Q4_TP4_ZERO_ADMISSION=1", admission)
    self.assertIn("run_onehost_deepswe_xprof_common.sh\" zero-hp", admission)
    self.assertIn("CANON_P58_Q4_TP4_ZERO_ADMISSION=1", admission_docker)
    self.assertIn(
        "CANON_P58_Q4_TP4_SEAM_DIAGNOSTIC=standard-decode",
        standard_decode,
    )
    self.assertIn(
        "CANON_P58_Q4_TP4_SEAM_DIAGNOSTIC=standard-decode",
        standard_decode_docker,
    )
    self.assertIn(
        "CANON_P58_Q4_TP4_CONTINUE_KV_DIAGNOSTIC=1", continue_kv
    )
    self.assertIn(
        "CANON_P58_Q4_TP4_CONTINUE_KV_DIAGNOSTIC=1", continue_kv_docker
    )
    self.assertIn(
        "CANON_P58_Q4_TP4_SHORT_BACKWARD=1", short_backward
    )
    self.assertIn(
        "CANON_P58_Q4_TP4_SHORT_BACKWARD=1", short_backward_docker
    )
    self.assertIn("CANON_P58_Q4_TP4_CARRIER_SCREEN=1", carrier_screen)
    self.assertIn(
        "CANON_P58_Q4_TP4_CARRIER_SCREEN=1", carrier_screen_docker
    )
    self.assertIn(
        "CANON_P58_Q4_TP4_TRAJECTORY_REPLAY=1", trajectory_replay
    )
    self.assertIn(
        "CANON_P58_Q4_TP4_TRAJECTORY_REPLAY=1",
        trajectory_replay_docker,
    )
    self.assertIn("1800", trajectory_replay)
    self.assertIn("1800", trajectory_replay_docker)
    self.assertIn(
        "p58-q4-tp4-systemopt-b2g2-k2560", trajectory_replay
    )
    self.assertIn(
        "p58-q4-tp4-systemopt-b2g2-k2560", trajectory_replay_docker
    )
    replay_launch = launch.split(
        '--batch_size "$carrier_prompts"', 1
    )
    self.assertEqual(len(replay_launch), 2)
    self.assertNotIn("--batch_size 1", launch)
    self.assertIn("sampling_temperature=1.0", common)
    self.assertIn("P58_ONEHOST_TIMEOUT_SECONDS", short_backward)
    self.assertIn("21600", short_backward)
    self.assertIn("p58-q4-tp4-short-backward", short_backward)
    self.assertIn("run_onehost_deepswe_seam_probe.sh", short_backward)
    self.assertIn("run_onehost_deepswe_seam_probe_docker.sh", short_backward_docker)
    self.assertIn("CANON_P58_Q4_TP4_SHORT_BACKWARD", docker)
    self.assertIn("CANON_P58_Q4_TP4_CARRIER_SCREEN", docker)
    self.assertIn("CANON_P58_Q4_TP4_TRAJECTORY_REPLAY", docker)

  def test_stock_observer_overlay_admits_only_exact_onehost_native(self):
    installer = (
        ROOT / "canon-zero-tim/cluster/steps/p58_install_stock_prompt_observer.sh"
    ).read_text()
    self.assertIn("onehost_native=", installer)
    self.assertIn('CANON_P58_ONEHOST_XPROF_ARM:-}\" = \"native', installer)
    self.assertIn('CANON_DEEPSWE_ONEHOST_NO_COMMIT:-0}\" = \"1', installer)


if __name__ == "__main__":
  unittest.main()
