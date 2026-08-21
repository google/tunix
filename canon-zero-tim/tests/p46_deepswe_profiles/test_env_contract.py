"""End-to-end CPU preflight for the P46 JobSet families."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "canon-zero-tim"
CLUSTER = PKG / "cluster"
sys.path.insert(0, str(CLUSTER))

import render_p34_jobset as p34  # pylint: disable=wrong-import-position
import render_p46_deepswe_profiles as renderer  # pylint: disable=wrong-import-position


class P46EnvironmentContractTest(unittest.TestCase):

  def _render(self, workload: str, topology: str, **overrides):
    base_name = (
        "jobset-64chip.yaml"
        if topology == "64"
        else "jobset-256cluster-64chip.yaml"
    )
    if overrides.get("full_campaign") and "resume_tag" not in overrides:
      overrides["resume_tag"] = "envtest"
    return renderer.render(
        yaml.safe_load((CLUSTER / base_name).read_text(encoding="utf-8")),
        workload=workload,
        topology=topology,
        source_commit="6" * 40,
        source_branch=p34.DEFAULT_SOURCE_BRANCH,
        client_image="example.invalid/tunix@sha256:" + "7" * 64,
        run_id="envtest",
        cpu_nodepool="cpu-pool",
        worker_nodepool="tpu-pool",
        model_pvc="models-pvc",
        whitelist=p34.P34_CLEAN_WHITELIST,
        whitelist_sha256=p34.P34_CLEAN_WHITELIST_SHA256,
        logical_shard_index=0,
        physical_shard_index=0,
        **overrides,
    )

  def _run(
      self,
      workload: str,
      topology: str,
      override: str = "",
      input_override: dict[str, str] | None = None,
      render_overrides: dict[str, object] | None = None,
  ):
    with tempfile.TemporaryDirectory() as root_text:
      root = Path(root_text)
      document = self._render(workload, topology, **(render_overrides or {}))
      environ = os.environ.copy()
      environ.update(p34._env(document))
      state = root / "state"
      state.mkdir()
      environ.update({
          "CANON_PKG": str(PKG),
          "CANON_STATE": str(state),
          "INJECTED_WANDB_API_KEY": "test-only",
      })
      environ.update(input_override or {})
      if override:
        original = Path(environ["CANON_PROFILE_FILE"])
        if not original.is_absolute():
          original = PKG / original
        wrapper = root / "profile.env"
        wrapper.write_text(
            f"source {original}\n{override}\n", encoding="utf-8"
        )
        environ["CANON_PROFILE_FILE"] = str(wrapper)
      return subprocess.run(
          ["bash", str(CLUSTER / "steps/00_env.sh")],
          cwd=ROOT,
          env=environ,
          text=True,
          stdout=subprocess.PIPE,
          stderr=subprocess.STDOUT,
          check=False,
      )

  def test_q32_training_preflight_passes_on_both_topologies(self):
    for topology, dp, global_m in (
        ("64", 4, 1024),
        ("256", 16, 4096),
    ):
      with self.subTest(topology=topology):
        result = self._run("q32-train", topology)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(
            f"P34 contract OK: DP{dp}xTP8 per role, local M256, "
            f"global M{global_m}",
            result.stdout,
        )

  def test_fixed_lm_head_training_preflight_passes_for_q4_and_q32(self):
    for workload, topology in (("q4-debug", "64"), ("q32-train", "64")):
      with self.subTest(workload=workload):
        result = self._run(
            workload,
            topology,
            render_overrides={"fixed_lm_head": True},
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(
            "P38.2y2 fixed lm-head DeepSWE training enabled",
            result.stdout,
        )

  def test_clean_evaluation_preflight_passes_without_trainer(self):
    for topology in ("64", "128"):
      with self.subTest(topology=topology):
        result = self._run("q4-clean-eval", topology)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn(
            f"P46 evaluation contract OK: topology={topology}", result.stdout
        )
        self.assertIn(
            "mode=reward_only parity=0 campaign=0 census=0 resume_tag=envtest "
            "sampled_by=stock@",
            result.stdout,
        )

  def test_full_campaign_preflight_is_explicit_and_shard_owning(self):
    result = self._run(
        "q4-clean-eval", "128", render_overrides={"full_campaign": True}
    )
    self.assertEqual(result.returncode, 0, result.stdout)
    self.assertIn("parity=0 campaign=1", result.stdout)
    census = self._run(
        "q4-clean-eval",
        "128",
        render_overrides={
            "full_campaign": True,
            "first_pass_census": True,
        },
    )
    self.assertEqual(census.returncode, 0, census.stdout)
    self.assertIn("parity=0 campaign=1 census=1", census.stdout)
    result = self._run(
        "q4-clean-eval",
        "128",
        override="export CANON_P46_PHYSICAL_SHARD_INDEX=1",
        render_overrides={"full_campaign": True},
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("owns all shards", result.stdout)
    result = self._run(
        "q4-clean-eval",
        "128",
        override="export CANON_P46_RESUME_TAG=../escape",
        render_overrides={"full_campaign": True},
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("must be lowercase and Kubernetes-safe", result.stdout)
    result = self._run(
        "q4-clean-eval",
        "128",
        override="export CANON_P46_CENSUS_FIRST_PASS=1",
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("requires a full reward-only campaign", result.stdout)

  def test_64chip_observer_canary_preflight_is_isolated(self):
    result = self._run(
        "q4-clean-eval",
        "64",
        render_overrides={
            "evaluation_mode": "logprob_observer",
            "parity_canary": True,
        },
    )
    self.assertEqual(result.returncode, 0, result.stdout)
    self.assertIn(
        "mode=logprob_observer parity=1 campaign=0 census=0 resume_tag=envtest "
        "sampled_by=stock@",
        result.stdout,
    )

  def test_legacy_import_fails_before_tpu_without_frozen_snapshot(self):
    result = self._run(
        "q4-clean-eval",
        "128",
        render_overrides={
            "full_campaign": True,
            "sampling_source_commit": "5" * 40,
            "legacy_import_id": "old-run",
        },
    )
    self.assertNotEqual(result.returncode, 0, result.stdout)
    self.assertIn("frozen legacy snapshot is missing SHA256SUMS", result.stdout)
    self.assertIn(
        "frozen legacy snapshot is missing legacy_source_contract.json",
        result.stdout,
    )

    result = self._run(
        "q4-clean-eval",
        "128",
        render_overrides={
            "full_campaign": True,
            "sampling_source_commit": "5" * 40,
            "frozen_v6_import_id": "old-v6-run",
        },
    )
    self.assertNotEqual(result.returncode, 0, result.stdout)
    self.assertIn("frozen v6 snapshot is missing", result.stdout)

  def test_q32_topology_drift_and_eval_trainer_overlap_fail_closed(self):
    result = self._run(
        "q32-train", "64", "export CANON_DP_SIZE=16"
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("role topology does not match", result.stdout)
    result = self._run(
        "q4-clean-eval", "64", "export CANON_P32_TRAIN_ADMITTED=1"
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("must not admit a trainer", result.stdout)
    result = self._run(
        "q4-clean-eval", "64", "export CANON_ALIGNMENT_GATE=1"
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("evaluation contradicts CANON_ALIGNMENT_GATE=1", result.stdout)
    result = self._run(
        "q4-clean-eval", "64", "export CANON_P46_EVALUATION_MODE=training"
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("unsupported P46 evaluation_mode", result.stdout)
    result = self._run(
        "q4-clean-eval",
        "64",
        input_override={"CANON_ALIGNMENT_TRAIN": "1"},
    )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn(
        "evaluation caller contradictions: CANON_ALIGNMENT_TRAIN=1",
        result.stdout,
    )


if __name__ == "__main__":
  unittest.main()
