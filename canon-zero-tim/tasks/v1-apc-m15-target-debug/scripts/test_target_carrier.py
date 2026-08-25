#!/usr/bin/env python3
"""Host/static gates for the bounded M15 APC target carrier."""

from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[4]
CANON = ROOT / "canon-zero-tim"
RENDERER_PATH = ROOT / "canon-zero-tim/cluster/render_v1_apc_m15_target_debug.py"
LEARNER_PATH = ROOT / "tunix/rl/agentic/agentic_rl_learner.py"
TRAIN_PATH = ROOT / "examples/frozenlake/train_frozenlake_qwen3.py"
BASE = ROOT / "canon-zero-tim/cluster/jobset-64chip.yaml"
SOURCE = "6" * 40
SPEC = importlib.util.spec_from_file_location("render_v1_apc_m15_target_debug", RENDERER_PATH)
assert SPEC and SPEC.loader
renderer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = renderer
SPEC.loader.exec_module(renderer)


def _load_consumer_contract():
  """Load the pure geometry helper without importing the TPU runtime stack."""
  tree = ast.parse(LEARNER_PATH.read_text(encoding="utf-8"))
  function = next(
      node
      for node in tree.body
      if isinstance(node, ast.FunctionDef)
      and node.name == "_p38_diagnostic_consumer_contract"
  )
  module = ast.Module(body=[function], type_ignores=[])
  namespace = {}
  exec(compile(ast.fix_missing_locations(module), str(LEARNER_PATH), "exec"), namespace)
  return namespace["_p38_diagnostic_consumer_contract"]


consumer_contract = _load_consumer_contract()


def _load_train_geometry_contracts():
  """Load pure entrypoint contracts without importing JAX/TPU dependencies."""
  names = {
      "_canonical_frozenlake_admission_geometry",
      "_canonical_frozenlake_p38_batch_contract",
  }
  tree = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"))
  functions = [
      node
      for node in tree.body
      if isinstance(node, ast.FunctionDef) and node.name in names
  ]
  if {node.name for node in functions} != names:
    raise AssertionError("FrozenLake entrypoint geometry helpers are incomplete")
  module = ast.Module(body=functions, type_ignores=[])
  namespace = {}
  exec(compile(ast.fix_missing_locations(module), str(TRAIN_PATH), "exec"), namespace)
  return tuple(namespace[name] for name in sorted(names))


admission_geometry, p38_batch_contract = _load_train_geometry_contracts()


class TargetCarrierTest(unittest.TestCase):

  def test_m15_capture_preserves_production_continue_decode_and_scopes_tail(self):
    profile = (CANON / "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-apc-debug.env").read_text(
        encoding="utf-8"
    )
    patch = (CANON / "patches/tpu_inference/27-tpu-runner-m15-mixed-program-tail.patch").read_text(
        encoding="utf-8"
    )
    installer = (CANON / "install.sh").read_text(encoding="utf-8")
    self.assertIn('export CANON_CONTINUE_DECODE=8', profile)
    self.assertIn('_P38_M15_TARGET_DEBUG in ("off", "on")', patch)
    self.assertIn('program_path == "continue_decode"', patch)
    self.assertIn('_P38_SERVING_CAPTURE_SEQ["n"] >=', patch)
    self.assertIn('len(_P38_SERVING_CAPTURED_STRATA) >=', patch)
    self.assertIn('journal_prefixes = [] if m15_continue_tail', patch)
    self.assertIn('if not m15_continue_tail:', patch)
    self.assertIn("27-tpu-runner-m15-mixed-program-tail.patch", installer)

  def test_renders_exact_off_on_pair(self):
    with tempfile.TemporaryDirectory() as directory:
      paths = renderer.render_all(
          base_path=BASE,
          output_dir=Path(directory),
          source_commit=SOURCE,
          run_id="pair-a",
      )
      self.assertEqual([path.name for path in paths], [
          "jobset-v1-apc-m15-off.yaml",
          "jobset-v1-apc-m15-on.yaml",
      ])
      documents = [yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths]
      envs = [renderer.p33._env_values(document) for document in documents]
      self.assertEqual(
          [env["CANON_APC_M15_TARGET_DEBUG"] for env in envs],
          ["off", "on"],
      )
      self.assertEqual(
          [env["CANON_VLLM_ENABLE_PREFIX_CACHING"] for env in envs],
          ["0", "1"],
      )
      for document, env in zip(documents, envs, strict=True):
        self.assertEqual(env["CANON_PROFILE_FILE"], renderer._PROFILE)
        self.assertEqual(env["CANON_P38_DIAGNOSTIC_ROUNDS"], "1")
        self.assertEqual(env["CANON_P38_SERVING_CAPTURE_PREFIX_BOUNDS"], "1152,1216,1280,1408,1696")
        self.assertEqual(env["CANON_P38_INCIDENT_MIN_PREFIX"], "1152")
        self.assertEqual(env["CANON_P38_INCIDENT_MAX_PREFIX"], "7168")
        self.assertEqual(env["CANON_P38_INCIDENT_MAX_BYTES"], "2147483648")
        self.assertEqual(
            env["CANON_APC_M15_REPLAY_LEDGER"],
            f"{env['CANON_P38_SERVING_CAPTURE_DIR']}/m15_replay_envelope.jsonl",
        )
        self.assertEqual(env["CANON_P38_FIXED_LM_HEAD"], "1")
        self.assertEqual(env["CANON_P57_WORKLOAD_CANDIDATE"], "m15")
        self.assertEqual(env["CANON_P57_DATA_SPLIT"], "main")
        self.assertEqual(env["CANON_P33_NO_COMMIT"], "1")
        self.assertEqual(env["CANON_P33_ENABLE_EVAL"], "0")
        self.assertEqual(document["spec"]["failurePolicy"]["maxRestarts"], 0)
        self.assertNotIn("CANON_P38_KV_OBSERVER_DIR", env)
        self.assertNotIn("CANON_P38_SEAM_OBSERVER", env)
        self.assertEqual(
            env["CANON_RUN_CMD"].split()[:4],
            [
                "python3",
                "-u",
                "-m",
                "examples.frozenlake.train_frozenlake_qwen3",
            ],
        )

      # The two target documents must be structurally identical after
      # normalizing the arm name and the two intended arm values.  This catches
      # a renderer change that accidentally couples topology, request order, or
      # capture geometry to the treatment.
      def normalize_strings(value, arm):
        if isinstance(value, str):
          return value.replace(f"-m15-{arm}-", "-m15-<ARM>-")
        if isinstance(value, list):
          return [normalize_strings(item, arm) for item in value]
        if isinstance(value, dict):
          return {
              key: normalize_strings(item, arm)
              for key, item in value.items()
          }
        return value

      normalized = []
      for arm, document in zip(("off", "on"), documents, strict=True):
        candidate = copy.deepcopy(document)
        env_items = renderer._container(candidate)["env"]  # pylint: disable=protected-access
        for item in env_items:
          if item["name"] == "CANON_APC_M15_TARGET_DEBUG":
            item["value"] = "<ARM>"
          elif item["name"] == "CANON_VLLM_ENABLE_PREFIX_CACHING":
            item["value"] = "<APC>"
        candidate["metadata"]["labels"]["canon.zero-tim/apc-m15-arm"] = "<ARM>"
        normalized.append(normalize_strings(candidate, arm))
      self.assertEqual(normalized[0], normalized[1])

  def test_rejects_short_source_sha(self):
    with tempfile.TemporaryDirectory() as directory:
      with self.assertRaisesRegex(ValueError, "full lowercase SHA"):
        renderer.render_all(
            base_path=BASE,
            output_dir=Path(directory),
            source_commit="abc",
            run_id="bad-source",
        )

  def test_m15_coverage_contract_uses_one_full_producer_unit(self):
    self.assertEqual(
        consumer_contract(
            enabled=True,
            full_batch_size=32,
            mini_batch_size=32,
            train_micro_batch_size=8,
            num_generations=8,
            process_in_consumer=True,
            m15_target_debug=True,
        ),
        (32, True, 1),
    )

  def test_m15_target_cannot_masquerade_as_onehost_rehearsal(self):
    with self.assertRaisesRegex(ValueError, "not a one-host rehearsal"):
      consumer_contract(
          enabled=True,
          full_batch_size=2,
          mini_batch_size=2,
          train_micro_batch_size=2,
          num_generations=2,
          process_in_consumer=True,
          onehost_rehearsal=True,
          m15_target_debug=True,
      )

  def test_entrypoint_admits_exact_m15_target_geometry(self):
    for arm in ("off", "on"):
      self.assertEqual(
          admission_geometry(
              p38_precheck_only=True,
              apc_m15_target_arm=arm,
              p57_tim_arm="",
              p57_run_kind="",
          ),
          (32, "none"),
      )
      self.assertEqual(
          p38_batch_contract(
              p38_precheck_only=True,
              apc_m15_target_arm=arm,
              workload_name="frozenlake-dp8-tp8",
              dp_size=8,
              batch_size=32,
              mini_batch_size=32,
              num_generations=8,
          ),
          (256, 1, 256),
      )

  def test_entrypoint_preserves_legacy_p38_geometry(self):
    self.assertEqual(
        admission_geometry(
            p38_precheck_only=True,
            apc_m15_target_arm="",
            p57_tim_arm="",
            p57_run_kind="",
        ),
        (4, "token"),
    )
    self.assertEqual(
        p38_batch_contract(
            p38_precheck_only=True,
            apc_m15_target_arm="",
            workload_name="frozenlake",
            dp_size=16,
            batch_size=32,
            mini_batch_size=4,
            num_generations=8,
        ),
        (32, 8, 256),
    )

  def test_entrypoint_preserves_p57_training_sampler_contracts(self):
    self.assertEqual(
        admission_geometry(
            p38_precheck_only=False,
            apc_m15_target_arm="",
            p57_tim_arm="zero",
            p57_run_kind="train",
        ),
        (32, "none"),
    )
    self.assertEqual(
        admission_geometry(
            p38_precheck_only=False,
            apc_m15_target_arm="",
            p57_tim_arm="is",
            p57_run_kind="train",
        ),
        (32, "token"),
    )

  def test_entrypoint_rejects_m15_target_contract_leaks(self):
    with self.assertRaisesRegex(ValueError, "requires P38 precheck-only"):
      admission_geometry(
          p38_precheck_only=False,
          apc_m15_target_arm="off",
          p57_tim_arm="",
          p57_run_kind="",
      )
    with self.assertRaisesRegex(ValueError, "cannot overlap a P57 TIM run"):
      admission_geometry(
          p38_precheck_only=True,
          apc_m15_target_arm="on",
          p57_tim_arm="zero",
          p57_run_kind="train",
      )
    with self.assertRaisesRegex(ValueError, "diagnostic geometry changed"):
      p38_batch_contract(
          p38_precheck_only=True,
          apc_m15_target_arm="off",
          workload_name="frozenlake-dp8-tp8",
          dp_size=8,
          batch_size=32,
          mini_batch_size=4,
          num_generations=8,
      )

  def test_runtime_markers_are_fail_closed_and_debug_scoped(self):
    sampler = (ROOT / "tunix/generate/vllm_sampler.py").read_text(encoding="utf-8")
    rollout = (ROOT / "tunix/rl/rollout/vllm_rollout.py").read_text(encoding="utf-8")
    run = (ROOT / "canon-zero-tim/cluster/steps/90_run.sh").read_text(encoding="utf-8")
    install = (ROOT / "canon-zero-tim/install.sh").read_text(encoding="utf-8")
    runner_patch = (
        ROOT / "canon-zero-tim/patches/tpu_inference/26-tpu-runner-m15-replay-envelope.patch"
    ).read_text(encoding="utf-8")
    self.assertIn("CAN" "ON_APC_M15_A_CONTRACT", sampler.replace('" "', ""))
    self.assertIn("sampling_params.skip_reading_prefix_cache is not False", sampler)
    self.assertIn("CAN" "ON_APC_M15_B_CONTRACT", rollout.replace('" "', ""))
    self.assertIn("if not reset_prefix_cache or any(cached_tokens)", rollout)
    self.assertIn("classify_m15_apc_target_run.py", run)
    self.assertIn("m15_apc_target.classification.json", run)
    self.assertIn("package_first_red_replay.py", run)
    self.assertIn("package_full_replay_carrier.py", run)
    self.assertIn("M15 APC target classification failed", run)
    self.assertIn("M15 first-red replay bundle failed", run)
    self.assertIn("M15 full replay carrier failed", run)
    self.assertIn("encoding=gcs-only", run)
    self.assertIn("26-tpu-runner-m15-replay-envelope.patch", install)
    self.assertIn("m15-apc-serving-envelope-v1", runner_patch)
    self.assertIn('"serving_arm"', runner_patch)


if __name__ == "__main__":
  unittest.main()
