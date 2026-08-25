#!/usr/bin/env python3
"""Contracts for the P64 P45 first-red numerical carrier."""

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
import yaml

ROOT = Path(__file__).resolve().parents[3]
TASK_SCRIPTS = ROOT / (
    "canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts"
)


def _load(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


if "tunix" not in sys.modules:
  tunix_package = types.ModuleType("tunix")
  tunix_package.__path__ = [str(ROOT / "tunix")]
  sys.modules["tunix"] = tunix_package
if "tunix.rl" not in sys.modules:
  rl_package = types.ModuleType("tunix.rl")
  rl_package.__path__ = [str(ROOT / "tunix/rl")]
  sys.modules["tunix.rl"] = rl_package
dp_training = _load("tunix.rl.dp_training", ROOT / "tunix/rl/dp_training.py")
sys.modules["tunix.rl"].dp_training = dp_training
dp_workloads = _load("tunix.rl.dp_workloads", ROOT / "tunix/rl/dp_workloads.py")


classifier = _load(
    "p64_classifier", TASK_SCRIPTS / "classify_p64_p45_numeric_debug.py"
)
renderer = _load(
    "p64_renderer", TASK_SCRIPTS / "render_p64_p45_numeric_debug.py"
)
p64_capsule = _load(
    "tunix.rl.p64_training_capsule",
    ROOT / "tunix/rl/p64_training_capsule.py",
)


def _fixture(
    *,
    nonfinite_stage: str | None = "engine_vjp",
    capsule_mode: str = "capture",
) -> str:
  pre = {
      "verdict": "PASS",
      "N_action": 100,
      "context": {"mesh": "8,8", "run_stage": "backward-no-commit"},
  }
  loss = {
      "schema": "canon-p64-loss-scale-v1",
      "stage": "loss_scale",
      "dp": 8,
      "tp": 8,
      "global_trajectories": 256,
      "local_trajectories": 32,
      "gradient_groups": 32,
      "global_M": 2048,
      "local_M": 256,
      "expected_accumulator_denominator": 32,
      "expected_streamed_multiplier": 0.125,
      "loss_denominator": 256.0,
      "loss_scale": 0.00390625,
  }

  def tree(stage: str, *, ranked: bool = False):
    finite = stage != nonfinite_stage
    return {
        "schema": "canon-p64-tree-numeric-v1",
        "stage": stage,
        "group": -1 if stage == "loss_cotangent" else 0,
        "groups": 32,
        "all_finite": finite,
        "naive_norm_finite": finite,
        "first_nonfinite": None if finite else {"leaf": 0, "index": [1, 0]},
        "first_nonfinite_rank": (
            None if finite or not ranked else {"rank": 1, "leaf": 0}
        ),
        "max_abs": 1.0,
        "stable_norm": 2.0,
        **(
            {"rank_count": 8, "rank_max_abs": [1.0] * 8}
            if ranked else {}
        ),
    }

  records = [tree("loss_cotangent")]
  if nonfinite_stage != "loss_cotangent":
    records.append(tree("group_input_cotangent", ranked=True))
  if nonfinite_stage not in {"loss_cotangent", "group_input_cotangent"}:
    records.append(tree("engine_vjp", ranked=True))
  if nonfinite_stage is None and capsule_mode == "capture":
    records.extend([
        tree("trainer_rank_local", ranked=True),
        tree("fixed_dp_reduced"),
        tree("scaled_microgradient"),
    ])
    for stage, ranked in (
        ("engine_vjp", True),
        ("trainer_rank_local", True),
        ("fixed_dp_reduced", False),
        ("scaled_microgradient", False),
        ("final_accumulator", False),
    ):
      record = tree(stage, ranked=ranked)
      record["group"] = 31
      records.append(record)
  elif nonfinite_stage is None:
    records.extend([
        tree("trainer_rank_local", ranked=True),
        tree("fixed_dp_reduced"),
        tree("scaled_microgradient"),
        tree("final_accumulator"),
    ])
  capsule_lines = (
      [
          "[P64.CAPSULE] capture_ready path=/tmp/capsule.npz sha256="
          + "a" * 64,
          "[P64.CAPSULE] model_bound mode=capture capsule_sha256="
          + "a" * 64,
          "[P64.CAPSULE] transport_ready mode=capture tool=gcloud "
          "capsule_sha256=" + "a" * 64,
      ]
      if capsule_mode == "capture"
      else [
          "[P64.CAPSULE] transport_ready mode=replay tool=gcloud "
          "capsule_sha256=" + "a" * 64,
          "[P64.CAPSULE] diagnostic_replay_ready path=/tmp/capsule.npz "
          "sha256=" + "a" * 64,
          "[P64.CAPSULE] producer_bypass verdict=PASS environment=0 "
          "rollout=0 rescore_b=0",
          "[P64.CAPSULE] model_verified mode=replay capsule_sha256="
          + "a" * 64,
          "[P64.CAPSULE] backward_scope mode=replay groups=1/32 "
          "selected=group0 optimizer_commits=0",
      ]
  )
  lines = [
      "[P64.NUMERIC] profile_resolved workload=frozenlake-dp8-tp8 "
      "dp=8 tp=8 stage=backward-no-commit optimizer_commits=0 "
      f"capsule_mode={capsule_mode}",
      *capsule_lines,
      "[" "CANON" "_ALIGN_PRE_JSON] " + json.dumps(pre),
      "[P64.NUMERIC] admission workload=frozenlake-dp8-tp8 dp=8 tp=8 "
      "global_trajectories=256 local_trajectories=32 global_M=2048 "
      "local_M=256 optimizer_commits=0",
      "[P64.NUMERIC] " + json.dumps(loss),
      *("[P64.NUMERIC] " + json.dumps(record) for record in records),
  ]
  if nonfinite_stage is None:
    lines.append(
        "[P64.NUMERIC] discard_complete optimizer_commits=0 "
        f"microsteps={1 if capsule_mode == 'replay' else 32} "
        f"denominator={1.0 if capsule_mode == 'replay' else 32.0} "
        f"diagnostic_replay={int(capsule_mode == 'replay')}"
    )
  return "\n".join(lines) + "\n"


def _env(document: dict) -> dict[str, str]:
  pod = document["spec"]["replicatedJobs"][0]["template"]["spec"][
      "template"
  ]["spec"]
  main = next(item for item in pod["containers"] if item["name"] == "jax-tpu")
  return {
      item["name"]: item["value"]
      for item in main["env"]
      if "value" in item
  }


class P64NumericDebugTest(unittest.TestCase):

  def test_classifier_localizes_rank1_engine_vjp(self):
    result = classifier.classify(_fixture())
    self.assertEqual(result["verdict"], "ROOT_LOCALIZED_NONFINITE", result)
    self.assertEqual(result["first_red"]["stage"], "engine_vjp")
    self.assertEqual(result["first_red"]["first_nonfinite_rank"]["rank"], 1)
    self.assertEqual(result["optimizer_commits"], 0)

  def test_classifier_accepts_fail_closed_at_loss_boundary(self):
    result = classifier.classify(_fixture(nonfinite_stage="loss_cotangent"))
    self.assertEqual(result["verdict"], "ROOT_LOCALIZED_NONFINITE", result)
    self.assertEqual(result["first_red"]["stage"], "loss_cotangent")

  def test_classifier_accepts_complete_finite_group0_and_group31(self):
    result = classifier.classify(_fixture(nonfinite_stage=None))
    self.assertEqual(result["verdict"], "ALL_BOUNDARIES_FINITE_NO_COMMIT", result)

  def test_classifier_accepts_group0_only_diagnostic_replay(self):
    result = classifier.classify(
        _fixture(nonfinite_stage=None, capsule_mode="replay")
    )
    self.assertEqual(result["verdict"], "ALL_BOUNDARIES_FINITE_NO_COMMIT", result)
    self.assertEqual(result["capsule_mode"], "replay")
    self.assertEqual(
        result["evidence_kind"], "diagnostic-replay-not-certification"
    )

  def test_classifier_rejects_receipt_after_first_nonfinite(self):
    log = _fixture() + (
        "[P64.NUMERIC] "
        + json.dumps({
            "schema": "canon-p64-tree-numeric-v1",
            "stage": "trainer_rank_local",
            "group": 0,
            "groups": 32,
            "all_finite": True,
            "rank_count": 8,
        })
        + "\n"
    )
    result = classifier.classify(log)
    self.assertEqual(result["verdict"], "FATAL_CONTRACT", result)
    self.assertIn("receipt_after_first_nonfinite", result["failures"])

  def test_classifier_rejects_missing_input_boundary_and_optimizer_activity(self):
    missing = "\n".join(
        line for line in _fixture().splitlines()
        if '"stage": "group_input_cotangent"' not in line
    ) + "\n"
    result = classifier.classify(missing)
    self.assertEqual(result["verdict"], "FATAL_CONTRACT")
    self.assertIn("group_input_cotangent=0/1", result["failures"])
    committed = classifier.classify(
        _fixture() + "[" "CANON" "_UPDATE_JSON] {}\n"
    )
    self.assertEqual(committed["verdict"], "FATAL_CONTRACT")
    self.assertIn("optimizer_update_receipt_present", committed["failures"])

  def test_renderer_and_profile_lock_exact_zero_commit_geometry(self):
    with tempfile.TemporaryDirectory() as tmp:
      output = renderer.render(
          source_commit="a" * 40,
          run_id="p64a",
          output_dir=Path(tmp) / "rendered",
          base_path=ROOT / "canon-zero-tim/cluster/jobset-64chip.yaml",
      )
      document = yaml.safe_load(output.read_text(encoding="utf-8"))
      values = _env(document)
      expected = {
          "CANON_PROFILE_FILE": renderer._PROFILE,
          "CANON_P33_RUN_STAGE": "backward-no-commit",
          "CANON_P33_NO_COMMIT": "1",
          "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
          "CANON_P38_FIXED_LM_HEAD": "1",
          "CANON_P64_P45_NUMERIC_DEBUG": "1",
          "CANON_P64_TRAINING_CAPSULE_MODE": "capture",
          "CANON_V1_HP_FULL": "0",
          "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": "0",
      }
      self.assertFalse({
          name: values.get(name)
          for name, value in expected.items()
          if values.get(name) != value
      })
      profile = ROOT / "canon-zero-tim" / renderer._PROFILE
      resolved = subprocess.run(
          ["bash", "-euo", "pipefail", "-c", f"source {profile}"],
          cwd=ROOT,
          env={**os.environ, **values},
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertEqual(resolved.returncode, 0, resolved.stderr)
      wrong = dict(values)
      wrong["CANON_P33_NO_COMMIT"] = "0"
      rejected = subprocess.run(
          ["bash", "-euo", "pipefail", "-c", f"source {profile}"],
          cwd=ROOT,
          env={**os.environ, **wrong},
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertNotEqual(rejected.returncode, 0)

      replay_output = renderer.render(
          source_commit="b" * 40,
          run_id="p64-replay-a",
          output_dir=Path(tmp) / "replayed",
          base_path=ROOT / "canon-zero-tim/cluster/jobset-64chip.yaml",
          capsule_mode="replay",
          capsule_gcs_uri=(
              "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/"
              "p64/p64a/training-capsule.npz"
          ),
          capsule_sha256="c" * 64,
          model_binding_sha256="d" * 64,
      )
      replay_values = _env(yaml.safe_load(replay_output.read_text()))
      self.assertEqual(
          replay_values["CANON_P64_TRAINING_CAPSULE_MODE"], "replay"
      )
      self.assertEqual(
          replay_values["CANON_P64_TRAINING_CAPSULE_SHA256"], "c" * 64
      )

  def test_dp_workload_rejects_neighboring_geometry(self):
    workload = dp_workloads.get_workload("frozenlake-dp8-tp8")
    env = {
        "CANON_P33_RUN_STAGE": "backward-no-commit",
        "CANON_P33_NO_COMMIT": "1",
        "CANON_P59_DP4_TAIL8": "0",
        "CANON_P60_DETERMINISTIC_AB": "0",
        "CANON_P61_BACKWARD_NUMERICAL_DIR": "",
        "CANON_P62_BACKWARD_NUMERIC_DEBUG": "0",
        "CANON_P64_P45_NUMERIC_DEBUG": "1",
        "CANON_P64_TRAINING_CAPSULE_MODE": "capture",
        "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
        "CANON_P38_FIXED_LM_HEAD": "1",
        "CANON_V1_HP_FULL": "0",
    }
    self.assertEqual(dp_workloads.requested_max_steps(workload, env), 1)
    for name, value in (
        ("CANON_P33_NO_COMMIT", "0"),
        ("CANON_P59_RANK_PARALLEL_BACKWARD", "0"),
        ("CANON_P38_FIXED_LM_HEAD", "0"),
        ("CANON_V1_HP_FULL", "1"),
    ):
      with self.subTest(name=name):
        wrong = dict(env)
        wrong[name] = value
        with self.assertRaisesRegex(ValueError, "P64"):
          dp_workloads.requested_max_steps(workload, wrong)

  def test_runner_persists_and_classifies_p64_full_log(self):
    runner = (
        ROOT / "canon-zero-tim/cluster/steps/90_run.sh"
    ).read_text(encoding="utf-8")
    self.assertIn("classify_p64_p45_numeric_debug.py", runner)
    self.assertIn("[P64.NUMERIC.POSTFLIGHT] ROOT_LOCALIZED", runner)
    self.assertIn("P64 full-log seed requires its exact profile", runner)
    self.assertIn("canon_p64_capsule_sync capture", runner)
    self.assertIn("canon_p64_capsule_sync replay", runner)

  def test_capsule_gcs_transport_round_trip_and_hash_negative(self):
    sync_lib = ROOT / (
        "canon-zero-tim/cluster/steps/p64_capsule_gcs_sync_lib.sh"
    )
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      state = root / "state"
      state.mkdir()
      capsule = state / "p64_training_capsule.npz"
      binding = Path(f"{capsule}.model.json")
      capsule.write_bytes(b"capsule")
      binding.write_bytes(b"binding")
      fake_gcs = root / "gcs"
      fake_gcs.mkdir()
      fake_bin = root / "bin"
      fake_bin.mkdir()
      fake_tool = fake_bin / "gcloud"
      fake_tool.write_text(
          "#!/usr/bin/env bash\n"
          "set -euo pipefail\n"
          "[ \"$1:$2:$3\" = storage:cp:--no-clobber ] && shift 3 || shift 2\n"
          "src=$1\n"
          "dst=$2\n"
          "if [[ $src = gs://* ]]; then\n"
          "  cp \"$FAKE_GCS_DIR/$(basename \"$src\")\" \"$dst\"\n"
          "else\n"
          "  target=$FAKE_GCS_DIR/$(basename \"$dst\")\n"
          "  [ ! -e \"$target\" ]\n"
          "  cp \"$src\" \"$target\"\n"
          "fi\n",
          encoding="utf-8",
      )
      fake_tool.chmod(0o755)
      uri = (
          "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/"
          "p64/p64-sync/training-capsule.npz"
      )
      base_env = {
          **os.environ,
          "PATH": f"{fake_bin}:{os.environ['PATH']}",
          "FAKE_GCS_DIR": str(fake_gcs),
          "CANON_STATE": str(state),
          "CANON_P64_TRAINING_CAPSULE": str(capsule),
          "CANON_P64_TRAINING_CAPSULE_GCS_URI": uri,
      }
      captured = subprocess.run(
          ["bash", "-c", f"source {sync_lib}; canon_p64_capsule_sync capture"],
          env={**base_env, "CANON_P64_TRAINING_CAPSULE_MODE": "capture"},
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertEqual(captured.returncode, 0, captured.stderr)
      capsule_sha = p64_capsule.file_sha256(capsule)
      binding_sha = p64_capsule.file_sha256(binding)
      capsule.unlink()
      binding.unlink()
      replayed = subprocess.run(
          ["bash", "-c", f"source {sync_lib}; canon_p64_capsule_sync replay"],
          env={
              **base_env,
              "CANON_P64_TRAINING_CAPSULE_MODE": "replay",
              "CANON_P64_TRAINING_CAPSULE_SHA256": capsule_sha,
              "CANON_P64_MODEL_BINDING_SHA256": binding_sha,
          },
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertEqual(replayed.returncode, 0, replayed.stderr)
      self.assertEqual(capsule.read_bytes(), b"capsule")
      self.assertEqual(binding.read_bytes(), b"binding")
      wrong_state = root / "wrong-state"
      wrong_state.mkdir()
      wrong_capsule = wrong_state / "p64_training_capsule.npz"
      wrong = subprocess.run(
          ["bash", "-c", f"source {sync_lib}; canon_p64_capsule_sync replay"],
          env={
              **base_env,
              "CANON_STATE": str(wrong_state),
              "CANON_P64_TRAINING_CAPSULE": str(wrong_capsule),
              "CANON_P64_TRAINING_CAPSULE_MODE": "replay",
              "CANON_P64_TRAINING_CAPSULE_SHA256": "f" * 64,
              "CANON_P64_MODEL_BINDING_SHA256": binding_sha,
          },
          text=True,
          capture_output=True,
          check=False,
      )
      self.assertNotEqual(wrong.returncode, 0)

  def test_capsule_round_trip_and_model_binding_are_fail_closed(self):
    with tempfile.TemporaryDirectory() as tmp:
      capsule_path = Path(tmp) / "p64_training_capsule.npz"
      capture_env = {
          "CANON_P64_P45_NUMERIC_DEBUG": "1",
          "CANON_PROFILE_FILE": renderer._PROFILE,
          "CANON_P32_WORKLOAD": "frozenlake-dp8-tp8",
          "CANON_DP_SIZE": "8",
          "CANON_TP_SIZE": "8",
          "CANON_GLOBAL_TRAJECTORIES": "256",
          "CANON_LOCAL_TRAJECTORIES": "32",
          "CANON_LOGPROB_M": "256",
          "MIN_TOKEN_BUCKET": "2048",
          "FL_SHARED_MESH": "8,8",
          "CANON_P33_RUN_STAGE": "backward-no-commit",
          "CANON_P33_NO_COMMIT": "1",
          "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
          "CANON_P38_FIXED_LM_HEAD": "1",
          "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": "0",
          "CANON_V1_HP_FULL": "0",
          "CANON_P64_TRAINING_CAPSULE_MODE": "capture",
          "CANON_P64_TRAINING_CAPSULE": str(capsule_path),
          "CANON_P64_TRAINING_CAPSULE_GCS_URI": (
              "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/"
              "p64/p64-unit/training-capsule.npz"
          ),
          "CANON_P64_TRAINING_CAPSULE_SHA256": "",
          "CANON_P64_MODEL_BINDING_SHA256": "",
          "CANON_EXPECT_COMMIT": "a" * 40,
          "CANON_RUN_ID": "p64-unit",
          "CANON_MODEL_DIR_NAME": "qwen8b_tp8",
      }
      prompt_ids = np.zeros((256, 4096), dtype=np.int32)
      prompt_mask = np.ones_like(prompt_ids, dtype=np.bool_)
      completion_ids = np.zeros((256, 2048), dtype=np.int32)
      completion_mask = np.ones_like(completion_ids, dtype=np.bool_)
      logps = np.zeros_like(completion_ids, dtype=np.float32)
      policy_version = np.zeros((256,), dtype=np.int32)
      train = types.SimpleNamespace(
          prompt_ids=prompt_ids,
          prompt_mask=prompt_mask,
          completion_ids=completion_ids,
          completion_mask=completion_mask,
          advantages=np.ones((256,), dtype=np.float32),
          ref_per_token_logps=None,
          old_per_token_logps=logps,
          segment_ids=None,
          segment_positions=None,
          is_update_step=None,
          sampler_is_weights=None,
          policy_version=policy_version,
          completion_valid_mask=completion_mask,
      )
      observed = types.SimpleNamespace(
          train_example=train,
          s_decode=logps.copy(),
          s_prefill=logps.copy(),
          t_old=logps.copy(),
          action_mask=completion_mask,
          completion_valid_mask=completion_mask,
          prompt_mask=prompt_mask,
          tokens=completion_ids,
          policy_version=policy_version,
          sampling_values=np.zeros((256, 3), dtype=np.float32),
          source_name="unit-rescore",
          all_compact_filtered=False,
      )
      capture = p64_capsule.persist(observed, capture_env)
      fingerprint = {"leaves": {"x": {"sha256": "1" * 64}}}
      binding = p64_capsule.bind_or_verify_model(fingerprint, capture_env)
      replay_env = {
          **capture_env,
          "CANON_P64_TRAINING_CAPSULE_MODE": "replay",
          "CANON_P64_TRAINING_CAPSULE_SHA256": capture["sha256"],
          "CANON_P64_MODEL_BINDING_SHA256": binding["binding_sha256"],
          "CANON_EXPECT_COMMIT": "b" * 40,
          "CANON_RUN_ID": "p64-replay-unit",
      }
      verified = p64_capsule.load_verified(replay_env)
      rebuilt = verified.build(types.SimpleNamespace, types.SimpleNamespace)
      self.assertTrue(
          np.array_equal(rebuilt.train_example.completion_ids, completion_ids)
      )
      p64_capsule.bind_or_verify_model(fingerprint, replay_env)
      with self.assertRaisesRegex(
          p64_capsule.P64TrainingCapsuleError, "live-model"
      ):
        p64_capsule.bind_or_verify_model({"leaves": {}}, replay_env)
      wrong_hash = dict(replay_env)
      wrong_hash["CANON_P64_TRAINING_CAPSULE_SHA256"] = "f" * 64
      with self.assertRaisesRegex(
          p64_capsule.P64TrainingCapsuleError, "file hash"
      ):
        p64_capsule.load_verified(wrong_hash)
      self.assertEqual(p64_capsule.reverse_group_limit(32, replay_env), 1)
      self.assertEqual(p64_capsule.reverse_group_limit(32, capture_env), 32)
      with self.assertRaisesRegex(
          p64_capsule.P64TrainingCapsuleError, "32 registered groups"
      ):
        p64_capsule.reverse_group_limit(31, replay_env)


if __name__ == "__main__":
  unittest.main()
