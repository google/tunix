#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / (
    "canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts/"
    "classify_full_recipe.py"
)
SPEC = importlib.util.spec_from_file_location("v1_full_classifier", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
classifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = classifier
SPEC.loader.exec_module(classifier)


class FullClassifierTest(unittest.TestCase):

  def setUp(self):
    self.original_updates = classifier._RECIPES["gsm8k"]["updates"]
    classifier._RECIPES["gsm8k"]["updates"] = 4

  def tearDown(self):
    classifier._RECIPES["gsm8k"]["updates"] = self.original_updates

  def _evidence(self, root: Path, *, alignment_fail: bool = False):
    state = root / "state"
    state.mkdir()
    env = {
        "CANON_PROFILE_FILE": (
            "cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k-v1-hp.env"
        ),
        "CANON_V1_HP_FULL": "1",
        "CANON_P33_RUN_STAGE": "full",
        "CANON_P33_NO_COMMIT": "0",
        "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
        "CANON_CONTINUE_DECODE": "8",
        "CANON_FIXED_AR_GATHER": "1",
        "CANON_PALLAS_GATHERED_LOGPROBS": "1",
        "CANON_LOGPROB_STEP_FUSION": "1",
        "CANON_P28_BATCHED_REPORT": "1",
        "CANON_P28_BATCHED_REVERSE": "0",
        "CANON_FUSED_TREE_OPS": "0",
        "CANON_XPROF_PHASE": "update",
        "CANON_XPROF_SKIP_STEPS": "2",
        "CANON_XPROF_STEPS": "1",
        "CANON_XPROF_PYTHON_TRACER": "0",
        "CANON_XPROF_HOST_TRACER": "1",
        "CANON_XPROF_TPU_TRACE_MODE": "TRACE_COMPUTE",
        "CANON_XPROF_LABELS": "1",
        "CANON_PERF_TRACE_EXPORT_STEP": "2",
        "CANON_VLLM_ENABLE_PREFIX_CACHING": "0",
        "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
        "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES": "all",
        "CANON_GCS_CACHE_BUCKET": (
            "gs://yuxzhang-tunix-models/cache/p33_compilation_cache"
        ),
        "CANON_GSM8K_ALIGNMENT_WARN_ONLY": "0",
        "CANON_BATCHED_EVIDENCE": "1",
    }
    (state / "env.sh").write_text(
        "".join(f"export {name}={value}\n" for name, value in env.items()),
        encoding="utf-8",
    )
    cache_profile = "qwen3-1p7b-dp16-tp4-gsm8k-v1-hp"
    cache_bucket = (
        "gs://yuxzhang-tunix-models/cache/p33_compilation_cache/"
        f"{cache_profile}"
    )
    for phase, status in (("restore", "hit"), ("save", "saved")):
      (state / f"jax_cache_{phase}.receipt").write_text(
          f"[JAX_CACHE_SYNC] phase={phase} status={status} tool=gcloud "
          f"rc=0 entries=3 profile={cache_profile} bucket={cache_bucket} "
          "local=/tmp/jax_compilation_cache\n",
          encoding="utf-8",
      )
    updates = []
    log = [
        "[P57.CONTINUE_DECODE] on-device decode loop enabled max_decode_steps=8",
        "[P56.GATHERED_LOGPROBS] installed data=16 local_m=256 continue_decode=8",
        "[P56.LOGPROB_STEP_FUSION] active target_rows=4096 max_logprobs=1",
        "[PATHTRACE] CANON_FIXED_AR=1 gather-ordered-sum at x",
        "[CANON_XPROF_LABELS] continue-decode stage callables cached",
        "[P59.DP16] head_cotangent_partition_ready "
        "global_shape=(4096, 151936) local_shape=(256,37984) "
        "placement=data,model",
        "[PATHTRACE] CANON_P38_FIXED_LM_HEAD=1 "
        "semantic_M=256 fixed_M=256 K=2048 TP=4 local_N=37984 "
        "fixed_N=38144 BM=128 BN=256 BK=256 chunks=1 "
        "endpoint=tied_embed p59_local=1 global_M=4096 dp=16",
        "[PATHTRACE] CANON_" "P38_FIXED_LM_HEAD_VJP=1 "
        "semantic_M=4096 local_M=256 fixed_M=256 chunks=1 "
        "accumulation=lax.scan order=ascending "
        "tp_input_reduction=all_gather_rank_order_f32_barrier "
        "K=2048 TP=4 local_N=37984 fixed_N=38144 endpoint=tied_embed",
        "[PATHTRACE] P59_RPA_LOCAL_KV_READY tp=4 local_q_heads=4 "
        "local_kv_heads=2 cache_heads=2 packing=2",
        "[PATHTRACE] P59_LOCAL_FUSED_LINEAR_READY tp=4 site=gate_proj "
        "local_width=1536 declared_width=6144 layout_shards=1 pieces=1",
        "[PATHTRACE] P59_LOCAL_FUSED_LINEAR_READY tp=4 site=up_proj "
        "local_width=1536 declared_width=6144 layout_shards=1 pieces=1",
        "[P51.XPROF] phase=update armed step=2",
        "[P51.XPROF] phase=update started step=2 anchor=update_entry tpu_trace_mode=TRACE_COMPUTE",
        "[P51.XPROF] phase=update stopped step=3 anchor=step_completed",
        "[V1.PERFETTO] captured training_step=2 timelines=3",
    ]
    for step in range(4):
      updates.append({
          "elapsed_seconds": 6.0,
          "dp_rank_pullbacks_per_transaction": 16,
          "dp_pullback_invocations_per_transaction": 1,
          "dp_replicas_exact": True,
      })
      log.extend([
          "[P59.DP16] gradient_reducer_ready dp_axis=data dp_size=16 staging=parallel_table",
          f"[PERF] step={step} stage=p32_vag_forward seconds=1.0",
          f"[PERF] step={step} stage=p32_vag_reverse seconds=3.0",
          f"[PERF] step={step} stage=segmented_value_and_grad seconds=4.0",
          f"[PERF] step={step} stage=optimizer_transaction seconds=1.0",
          f"[PERF] step={step} stage=weight_sync seconds=1.0",
          f"Global step {step} completed in {80.0 if step == 2 else 8.0} seconds.",
      ])
    for index in range(68):
      verdict = "FAIL" if alignment_fail and index == 0 else "PASS"
      prefix = "CANON_ALIGN_PRE" if index % 17 == 0 else "CANON_ALIGN"
      log.append(f"[{prefix}] step={index} verdict={verdict}")
    run_log = state / "run.log"
    run_log.write_text("\n".join(log) + "\n", encoding="utf-8")
    update_report = state / "updates.jsonl"
    update_report.write_text(
        "".join(json.dumps(row) + "\n" for row in updates), encoding="utf-8"
    )
    base = {
        "verdict": "PASS",
        "claim_level": "strict-zero-tim",
        "expected_updates": 4,
        "observed_updates": 4,
        "observed_pre_alignments": 4,
        "observed_alignments": 64,
        "alignment_warning_records": 0,
        "pre_alignment_warning_records": 0,
    }
    base_path = state / "base.json"
    base_path.write_text(json.dumps(base), encoding="utf-8")
    xprof = state / "xprof-update/plugins/profile/1"
    xprof.mkdir(parents=True)
    (xprof / "device.xplane.pb").write_bytes(b"xplane")
    (xprof / "device.trace.json.gz").write_bytes(b"trace")
    perfetto = state / "perfetto"
    perfetto.mkdir()
    (perfetto / "perfetto_trace_v2_1.pb").write_bytes(b"perfetto")
    return state, run_log, update_report, base_path

  def test_green_full_contract_and_excludes_profiled_step(self):
    with tempfile.TemporaryDirectory() as tmp:
      state, run_log, updates, base = self._evidence(Path(tmp))
      record = classifier.classify(
          recipe="gsm8k",
          state=state,
          run_log=run_log,
          update_report=updates,
          base_classification=base,
      )
      self.assertEqual(record["verdict"], "PASS", record["reasons"])
      self.assertEqual(record["zero_tim"]["observed_pass"], 68)
      self.assertEqual(
          record["timing"]["steady_steps2_plus_excluding_profile_count"], 1
      )
      self.assertEqual(
          record["timing"]["steady_steps2_plus_excluding_profile_mean"][
              "wall_seconds"
          ],
          8.0,
      )

  def test_any_real_alignment_fail_is_fatal(self):
    with tempfile.TemporaryDirectory() as tmp:
      state, run_log, updates, base = self._evidence(
          Path(tmp), alignment_fail=True
      )
      record = classifier.classify(
          recipe="gsm8k",
          state=state,
          run_log=run_log,
          update_report=updates,
          base_classification=base,
      )
      self.assertEqual(record["verdict"], "FAIL")
      self.assertIn("canon_align_fail=1 expected=0", record["reasons"])

  def test_missing_profile_artifact_is_fatal(self):
    with tempfile.TemporaryDirectory() as tmp:
      state, run_log, updates, base = self._evidence(Path(tmp))
      (state / "xprof-update/plugins/profile/1/device.xplane.pb").unlink()
      record = classifier.classify(
          recipe="gsm8k",
          state=state,
          run_log=run_log,
          update_report=updates,
          base_classification=base,
      )
      self.assertEqual(record["verdict"], "FAIL")
      self.assertIn("missing_xplane", record["reasons"])

  def test_missing_p59_head_partition_receipt_is_fatal(self):
    with tempfile.TemporaryDirectory() as tmp:
      state, run_log, updates, base = self._evidence(Path(tmp))
      text = run_log.read_text(encoding="utf-8")
      run_log.write_text(
          "\n".join(
              line
              for line in text.splitlines()
              if "head_cotangent_partition_ready" not in line
          )
          + "\n",
          encoding="utf-8",
      )
      record = classifier.classify(
          recipe="gsm8k",
          state=state,
          run_log=run_log,
          update_report=updates,
          base_classification=base,
      )
      self.assertEqual(record["verdict"], "FAIL")
      self.assertIn("marker.p59_head_partition=0", record["reasons"])

  def test_wrong_p59_head_shape_is_fatal(self):
    replacements = {
        "global": (
            "global_shape=(4096, 151936)",
            "global_shape=(2048, 151936)",
        ),
        "local": (
            "local_shape=(256,37984)",
            "local_shape=(128,37984)",
        ),
    }
    for label, (before, after) in replacements.items():
      with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
        state, run_log, updates, base = self._evidence(Path(tmp))
        run_log.write_text(
            run_log.read_text(encoding="utf-8").replace(before, after),
            encoding="utf-8",
        )
        record = classifier.classify(
            recipe="gsm8k",
            state=state,
            run_log=run_log,
            update_report=updates,
            base_classification=base,
        )
        self.assertEqual(record["verdict"], "FAIL")
        self.assertIn(
            "p59_head_partition_shape_or_placement", record["reasons"]
        )

  def test_recipe_shape_contracts_cover_dp16_and_dp8(self):
    expected = {
        "gsm8k": (16, 4, 4096, 256, 6144, 37984),
        "p45": (8, 8, 2048, 256, 12288, 18992),
        "m15": (8, 8, 2048, 256, 12288, 18992),
    }
    for recipe, values in expected.items():
      contract = classifier._RECIPES[recipe]
      self.assertEqual(
          (
              contract["dp"],
              contract["tp"],
              contract["global_m"],
              contract["local_m"],
              contract["intermediate"],
              contract["local_vocab"],
          ),
          values,
      )
    self.assertFalse(classifier._RECIPES["p45"]["apc"])
    self.assertFalse(classifier._RECIPES["m15"]["apc"])

  def test_wrong_jax_cache_bucket_is_fatal(self):
    with tempfile.TemporaryDirectory() as tmp:
      state, run_log, updates, base = self._evidence(Path(tmp))
      env = state / "env.sh"
      env.write_text(
          env.read_text(encoding="utf-8").replace(
              "gs://yuxzhang-tunix-models/cache/p33_compilation_cache",
              "gs://wrong/cache",
          ),
          encoding="utf-8",
      )
      record = classifier.classify(
          recipe="gsm8k",
          state=state,
          run_log=run_log,
          update_report=updates,
          base_classification=base,
      )
      self.assertEqual(record["verdict"], "FAIL")
      self.assertTrue(
          any(reason.startswith("resolved_env=") for reason in record["reasons"])
      )

  def test_missing_jax_cache_receipt_is_fatal(self):
    with tempfile.TemporaryDirectory() as tmp:
      state, run_log, updates, base = self._evidence(Path(tmp))
      (state / "jax_cache_restore.receipt").unlink()
      record = classifier.classify(
          recipe="gsm8k",
          state=state,
          run_log=run_log,
          update_report=updates,
          base_classification=base,
      )
      self.assertEqual(record["verdict"], "FAIL")
      self.assertIn("missing_jax_cache_restore_receipt", record["reasons"])

  def test_incoherent_jax_cache_receipt_is_fatal(self):
    with tempfile.TemporaryDirectory() as tmp:
      state, run_log, updates, base = self._evidence(Path(tmp))
      receipt = state / "jax_cache_restore.receipt"
      receipt.write_text(
          receipt.read_text(encoding="utf-8").replace(
              "status=hit tool=gcloud rc=0 entries=3",
              "status=hit tool=none rc=23 entries=0",
          ),
          encoding="utf-8",
      )
      record = classifier.classify(
          recipe="gsm8k",
          state=state,
          run_log=run_log,
          update_report=updates,
          base_classification=base,
      )
      self.assertEqual(record["verdict"], "FAIL")
      self.assertIn("jax_cache_restore.status_contract", record["reasons"])

  def test_wrong_profile_is_fatal(self):
    with tempfile.TemporaryDirectory() as tmp:
      state, run_log, updates, base = self._evidence(Path(tmp))
      env = state / "env.sh"
      env.write_text(
          env.read_text(encoding="utf-8").replace(
              "qwen3-1p7b-dp16-tp4-gsm8k-v1-hp.env",
              "qwen3-8b-dp8-tp8-frozenlake-v1-hp.env",
          ),
          encoding="utf-8",
      )
      record = classifier.classify(
          recipe="gsm8k",
          state=state,
          run_log=run_log,
          update_report=updates,
          base_classification=base,
      )
      self.assertEqual(record["verdict"], "FAIL")
      self.assertTrue(
          any(
              reason.startswith("resolved_env=")
              for reason in record["reasons"]
          )
      )

  def test_apc_on_is_fatal_for_an_apc_off_recipe(self):
    with tempfile.TemporaryDirectory() as tmp:
      state, run_log, updates, base = self._evidence(Path(tmp))
      run_log.write_text(
          "[P3_APC_CONFIG] enabled=1 workload=frozenlake "
          "reader=train_frozenlake_qwen3\n"
          + run_log.read_text(encoding="utf-8"),
          encoding="utf-8",
      )
      record = classifier.classify(
          recipe="gsm8k",
          state=state,
          run_log=run_log,
          update_report=updates,
          base_classification=base,
      )
      self.assertEqual(record["verdict"], "FAIL")
      self.assertIn("unexpected_apc_on", record["reasons"])

  def test_frozenlake_apc_off_requires_exact_runtime_marker(self):
    contract = classifier._RECIPES["gsm8k"]
    original_workload = contract["workload"]
    contract["workload"] = "frozenlake-dp8-tp8"
    try:
      with tempfile.TemporaryDirectory() as tmp:
        state, run_log, updates, base = self._evidence(Path(tmp))
        record = classifier.classify(
            recipe="gsm8k",
            state=state,
            run_log=run_log,
            update_report=updates,
            base_classification=base,
        )
        self.assertEqual(record["verdict"], "FAIL")
        self.assertIn("apc_runtime_marker", record["reasons"])

        marker = (
            "[P3_APC_CONFIG] enabled=0 workload=frozenlake "
            "reader=train_frozenlake_qwen3\n"
        )
        run_log.write_text(
            marker + run_log.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        record = classifier.classify(
            recipe="gsm8k",
            state=state,
            run_log=run_log,
            update_report=updates,
            base_classification=base,
        )
        self.assertEqual(record["verdict"], "PASS", record["reasons"])

        run_log.write_text(
            marker + run_log.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        record = classifier.classify(
            recipe="gsm8k",
            state=state,
            run_log=run_log,
            update_report=updates,
            base_classification=base,
        )
        self.assertEqual(record["verdict"], "FAIL")
        self.assertIn("apc_runtime_marker", record["reasons"])
    finally:
      contract["workload"] = original_workload

  def test_wrong_p59_local_chunks_is_fatal(self):
    with tempfile.TemporaryDirectory() as tmp:
      state, run_log, updates, base = self._evidence(Path(tmp))
      run_log.write_text(
          run_log.read_text(encoding="utf-8").replace(
              "fixed_M=256 chunks=1 accumulation=lax.scan",
              "fixed_M=256 chunks=16 accumulation=lax.scan",
          ),
          encoding="utf-8",
      )
      record = classifier.classify(
          recipe="gsm8k",
          state=state,
          run_log=run_log,
          update_report=updates,
          base_classification=base,
      )
      self.assertEqual(record["verdict"], "FAIL")
      self.assertIn(
          "p59_fixed_head_vjp_global_local_shape_chunks_or_reduction",
          record["reasons"],
      )

  def test_missing_p59_rpa_local_kv_receipt_is_fatal(self):
    with tempfile.TemporaryDirectory() as tmp:
      state, run_log, updates, base = self._evidence(Path(tmp))
      run_log.write_text(
          "\n".join(
              line
              for line in run_log.read_text(encoding="utf-8").splitlines()
              if "P59_RPA_LOCAL_KV_READY" not in line
          )
          + "\n",
          encoding="utf-8",
      )
      record = classifier.classify(
          recipe="gsm8k",
          state=state,
          run_log=run_log,
          update_report=updates,
          base_classification=base,
      )
      self.assertEqual(record["verdict"], "FAIL")
      self.assertIn("p59_rpa_local_kv_receipt_missing", record["reasons"])

  def test_wrong_p59_rpa_local_kv_shape_is_fatal(self):
    with tempfile.TemporaryDirectory() as tmp:
      state, run_log, updates, base = self._evidence(Path(tmp))
      run_log.write_text(
          run_log.read_text(encoding="utf-8").replace(
              "local_kv_heads=2 cache_heads=2",
              "local_kv_heads=4 cache_heads=2",
          ),
          encoding="utf-8",
      )
      record = classifier.classify(
          recipe="gsm8k",
          state=state,
          run_log=run_log,
          update_report=updates,
          base_classification=base,
      )
      self.assertEqual(record["verdict"], "FAIL")
      self.assertIn("p59_rpa_local_kv_shape_or_topology", record["reasons"])

  def test_missing_p59_local_fused_linear_receipt_is_fatal(self):
    with tempfile.TemporaryDirectory() as tmp:
      state, run_log, updates, base = self._evidence(Path(tmp))
      text = run_log.read_text(encoding="utf-8")
      run_log.write_text(
          "\n".join(
              line
              for line in text.splitlines()
              if "P59_LOCAL_FUSED_LINEAR_READY" not in line
          )
          + "\n",
          encoding="utf-8",
      )
      record = classifier.classify(
          recipe="gsm8k",
          state=state,
          run_log=run_log,
          update_report=updates,
          base_classification=base,
      )
      self.assertEqual(record["verdict"], "FAIL")
      self.assertIn(
          "p59_local_fused_linear_receipt_missing", record["reasons"]
      )

  def test_wrong_p59_local_fused_linear_shape_is_fatal(self):
    with tempfile.TemporaryDirectory() as tmp:
      state, run_log, updates, base = self._evidence(Path(tmp))
      text = run_log.read_text(encoding="utf-8")
      run_log.write_text(
          text.replace("local_width=1536", "local_width=6144"),
          encoding="utf-8",
      )
      record = classifier.classify(
          recipe="gsm8k",
          state=state,
          run_log=run_log,
          update_report=updates,
          base_classification=base,
      )
      self.assertEqual(record["verdict"], "FAIL")
      self.assertIn(
          "p59_local_fused_linear_shape_or_topology", record["reasons"]
      )

  def test_direct_eval_cycle_timing_uses_explicit_enclosing_step(self):
    rows = [
        {"global_step": float(step), "wall_seconds": float(step)}
        for step in range(300)
    ]
    steady, eval_steps, direct_eval_cycle_excluded = (
        classifier._steady_timing_rows(
        rows,
        expected_updates=300,
        p57_eval={
            "steps": [0, 50, 100, 150, 200, 250, 300],
            "cycle_receipts": [
                {
                    "policy_step": step,
                    "enclosing_global_step": (
                        None if step == 300 else step + 1
                    ),
                }
                for step in range(0, 301, 50)
            ],
        },
        )
    )
    self.assertNotIn(2, {int(row["global_step"]) for row in steady})
    self.assertEqual(eval_steps, {1, 51, 101, 151, 201, 251})
    training_steps = {
        int(row["global_step"]) for row in direct_eval_cycle_excluded
    }
    self.assertTrue(eval_steps.isdisjoint(training_steps))
    self.assertIn(50, training_steps)
    self.assertIn(299, training_steps)


if __name__ == "__main__":
  unittest.main()
