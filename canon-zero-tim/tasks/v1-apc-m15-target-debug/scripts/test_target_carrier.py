#!/usr/bin/env python3
"""Host/static gates for the bounded M15 APC target carrier."""

from __future__ import annotations

import ast
import copy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[4]
CANON = ROOT / "canon-zero-tim"
RENDERER_PATH = ROOT / "canon-zero-tim/cluster/render_v1_apc_m15_target_debug.py"
LEARNER_PATH = ROOT / "tunix/rl/agentic/agentic_rl_learner.py"
GRPO_LEARNER_PATH = ROOT / "tunix/rl/agentic/agentic_grpo_learner.py"
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

  def test_m15_round_budget_resets_bytes_without_reusing_record_indices(self):
    patch = (
        CANON / "patches/tpu_inference/31-tpu-runner-m15-multiround-budget.patch"
    ).read_text(encoding="utf-8").splitlines()
    start = next(
        index for index, line in enumerate(patch)
        if line.startswith("+def _p38_begin_observer_round")
    )
    body = []
    for line in patch[start:]:
      if line.startswith(" _P38_TAIL_GATHER_FNS"):
        break
      if line.startswith("+"):
        body.append(line[1:])
    namespace = {"os": os}
    exec("\n".join(body), namespace)  # pylint: disable=exec-used
    begin = namespace["_p38_begin_observer_round"]
    previous = os.environ.get("CANON_P38_DURABILITY_PROFILE")
    os.environ["CANON_P38_DURABILITY_PROFILE"] = "m15-wide-v1"
    try:
      state = {"records": 17, "bytes": 1234}
      begin(state, 0, "seam")
      self.assertEqual(state, {
          "records": 17, "bytes": 1234, "diagnostic_round": 0
      })
      begin(state, 1, "seam")
      self.assertEqual(state, {
          "records": 17, "bytes": 0, "diagnostic_round": 1
      })
      with self.assertRaisesRegex(ValueError, "increase by one"):
        begin(state, 3, "seam")
    finally:
      if previous is None:
        os.environ.pop("CANON_P38_DURABILITY_PROFILE", None)
      else:
        os.environ["CANON_P38_DURABILITY_PROFILE"] = previous

  def test_runner_uses_sealed_m15_shards_without_replacing_legacy(self):
    runner = (CANON / "cluster/steps/90_run.sh").read_text(encoding="utf-8")
    env_step = (CANON / "cluster/steps/00_env.sh").read_text(encoding="utf-8")
    persist = (
        CANON / "tasks/p38-pathways-decode-prefill-carrier/scripts/"
        "persist_p38_gcs.sh"
    ).read_text(encoding="utf-8")
    worker = (
        CANON / "tasks/p38-pathways-decode-prefill-carrier/scripts/"
        "p38_live_snapshot_worker.sh"
    ).read_text(encoding="utf-8")
    self.assertIn("classify_m15_apc_wide_seam.py", runner)
    self.assertIn("package_m15_apc_wide_seam.py", runner)
    self.assertIn("verify_m15_wide_round.py", runner)
    self.assertIn("archive=bounded-shards", runner)
    self.assertIn("--require-first-action", runner)
    self.assertIn("classify_p38_seam.py", runner)
    self.assertIn("M15 compact seam bundle failed", runner)
    self.assertIn("M15 fixed lm-head seam runs", env_step)
    self.assertIn("expected_p38_seam_min=960", env_step)
    self.assertIn("m15-shard", persist)
    self.assertIn("m15-round", persist)
    self.assertIn("WIDE_ROUND_COMPLETE.json", persist)
    self.assertIn("flush_m15_shards", worker)
    self.assertNotIn(
        'tar --sort=name --mtime=@0 --owner=0 --group=0 \\\n+      -C "$CANON_P38_SERVING_CAPTURE_DIR"',
        runner.split("archive=bounded-shards", maxsplit=1)[0],
    )

  def test_m15_capture_preserves_production_continue_decode_from_first_call(self):
    profile = (CANON / "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-apc-debug.env").read_text(
        encoding="utf-8"
    )
    tail_patch = (CANON / "patches/tpu_inference/27-tpu-runner-m15-mixed-program-tail.patch").read_text(
        encoding="utf-8"
    )
    path_patch = (CANON / "patches/tpu_inference/28-tpu-runner-m15-mixed-program-path.patch").read_text(
        encoding="utf-8"
    )
    durability_patch = (CANON / "patches/tpu_inference/30-tpu-runner-m15-wide-incident-bypass.patch").read_text(
        encoding="utf-8"
    )
    installer = (CANON / "install.sh").read_text(encoding="utf-8")
    self.assertIn('export CANON_CONTINUE_DECODE=8', profile)
    self.assertIn('_P38_M15_TARGET_DEBUG in ("off", "on")', tail_patch)
    self.assertIn('program_path == "continue_decode"', tail_patch)
    self.assertIn('+def _p38_m15_continue_program_path', path_patch)
    self.assertIn('+    m15_continue_path = _p38_m15_continue_program_path', path_patch)
    self.assertIn('+    journal_prefixes = [] if m15_continue_path', path_patch)
    self.assertIn('+    if not m15_continue_path:', path_patch)
    self.assertIn("27-tpu-runner-m15-mixed-program-tail.patch", installer)
    self.assertIn("28-tpu-runner-m15-mixed-program-path.patch", installer)
    self.assertIn("CANON_P38_INCIDENT_LEDGER_BYPASS", durability_patch)
    self.assertIn("30-tpu-runner-m15-wide-incident-bypass.patch", installer)
    budget_patch = (
        CANON / "patches/tpu_inference/31-tpu-runner-m15-multiround-budget.patch"
    ).read_text(encoding="utf-8")
    self.assertIn("31-tpu-runner-m15-multiround-budget.patch", installer)
    self.assertIn('state["bytes"] = 0', budget_patch)
    self.assertNotIn('state["records"] = 0', budget_patch)
    self.assertIn("diagnostic_round != int(previous) + 1", budget_patch)

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
        self.assertEqual(env["CANON_P38_DURABILITY_PROFILE"], "round-alignment-v1")
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
        self.assertIn("--sampler_is=none", env["CANON_RUN_CMD"].split())
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

  def test_observer_pair_uses_m15_wide_durability(self):
    with tempfile.TemporaryDirectory() as directory:
      paths = renderer.render_all(
          base_path=BASE,
          output_dir=Path(directory),
          source_commit=SOURCE,
          run_id="wide-a",
          observer="layer",
      )
      for path in paths:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        env = renderer.p33._env_values(document)
        self.assertEqual(env["CANON_P38_DURABILITY_PROFILE"], "m15-wide-v1")
        self.assertEqual(env["CANON_P38_DIAGNOSTIC_ROUNDS"], "3")
        self.assertEqual(
            document["metadata"]["labels"]["canon.zero-tim/durability-profile"],
            "m15-wide-v1",
        )
        self.assertEqual(
            document["metadata"]["labels"]["canon.zero-tim/diagnostic-rounds"],
            "3",
        )

  def test_rejects_short_source_sha(self):
    with tempfile.TemporaryDirectory() as directory:
      with self.assertRaisesRegex(ValueError, "full lowercase SHA"):
        renderer.render_all(
            base_path=BASE,
            output_dir=Path(directory),
            source_commit="abc",
          run_id="bad-source",
        )

  def test_layer_observer_renders_bounded_off_on_pair(self):
    with tempfile.TemporaryDirectory() as directory:
      paths = renderer.render_all(
          base_path=BASE,
          output_dir=Path(directory),
          source_commit=SOURCE,
          run_id="layer-a",
          observer="layer",
      )
      self.assertEqual([path.name for path in paths], [
          "jobset-v1-apc-m15-off-layer.yaml",
          "jobset-v1-apc-m15-on-layer.yaml",
      ])
      for path in paths:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        env = renderer.p33._env_values(document)
        self.assertEqual(env["CANON_P38_SEAM_OBSERVER"], "layer")
        self.assertEqual(env["CANON_P38_SEAM_MIN_POSITION"], "960")
        self.assertEqual(env["CANON_P38_SEAM_MAX_POSITION"], "4096")
        self.assertEqual(env["CANON_P38_SEAM_MAX_BYTES"], "8589934592")
        self.assertEqual(env["CANON_P38_TAIL_OBSERVER"], "1")
        self.assertEqual(env["CANON_P38_TAIL_MAX_BYTES"], "268435456")
        self.assertEqual(env["CANON_P38_DIAGNOSTIC_ROUNDS"], "3")
        self.assertNotIn("CANON_P38_SEAM_LAYER", env)

  def test_full_observer_requires_and_pins_one_layer(self):
    with tempfile.TemporaryDirectory() as directory:
      paths = renderer.render_all(
          base_path=BASE,
          output_dir=Path(directory),
          source_commit=SOURCE,
          run_id="full-a",
          observer="full",
          seam_layer=17,
      )
      for path in paths:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        env = renderer.p33._env_values(document)
        self.assertEqual(env["CANON_P38_SEAM_OBSERVER"], "full")
        self.assertEqual(env["CANON_P38_SEAM_LAYER"], "17")
        self.assertEqual(env["CANON_P38_DIAGNOSTIC_ROUNDS"], "3")
        self.assertNotIn("CANON_P38_TAIL_OBSERVER", env)
      with self.assertRaisesRegex(ValueError, "requires --seam-layer"):
        renderer.render_all(
            base_path=BASE,
            output_dir=Path(directory) / "bad",
            source_commit=SOURCE,
            run_id="full-bad",
            observer="full",
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
    grpo_learner = GRPO_LEARNER_PATH.read_text(encoding="utf-8").replace(
        '" "', ""
    )
    self.assertIn("CAN" "ON_APC_M15_SAMPLER_CONTRACT", grpo_learner)
    self.assertIn("encoding=gcs-only", run)
    self.assertIn("26-tpu-runner-m15-replay-envelope.patch", install)
    self.assertIn("m15-apc-serving-envelope-v1", runner_patch)
    self.assertIn('"serving_arm"', runner_patch)


if __name__ == "__main__":
  unittest.main()
