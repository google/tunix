"""Renderer contracts for the workload-specific P46 topology matrix."""

from __future__ import annotations

from pathlib import Path
import shlex
import sys
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
CLUSTER = ROOT / "cluster"
sys.path.insert(0, str(CLUSTER))

import render_p34_jobset as p34  # pylint: disable=wrong-import-position
import render_p46_deepswe_profiles as renderer  # pylint: disable=wrong-import-position


SOURCE = "6" * 40
IMAGE = "example.invalid/tunix@sha256:" + "7" * 64


class P46RendererTest(unittest.TestCase):

  def _base(self, topology: str):
    name = "jobset-64chip.yaml" if topology == "64" else "jobset-256cluster-64chip.yaml"
    return yaml.safe_load((CLUSTER / name).read_text(encoding="utf-8"))

  def _render(self, workload: str, topology: str, **overrides):
    logical_shard_index = overrides.pop("logical_shard_index", 0)
    physical_shard_index = overrides.pop("physical_shard_index", 0)
    run_id = overrides.pop("run_id", f"t{topology}")
    if overrides.get("full_campaign") and "resume_tag" not in overrides:
      overrides["resume_tag"] = f"t{topology}"
    return renderer.render(
        self._base(topology),
        workload=workload,
        topology=topology,
        source_commit=SOURCE,
        source_branch=p34.DEFAULT_SOURCE_BRANCH,
        client_image=IMAGE,
        run_id=run_id,
        cpu_nodepool="deepswe-cpu-pool",
        worker_nodepool=f"v5p-{topology}",
        model_pvc="models-pvc",
        whitelist=p34.P34_CLEAN_WHITELIST,
        whitelist_sha256=p34.P34_CLEAN_WHITELIST_SHA256,
        logical_shard_index=logical_shard_index,
        physical_shard_index=physical_shard_index,
        **overrides,
    )

  def test_all_three_families_render_on_their_signed_topologies(self):
    for workload, topologies in renderer.WORKLOAD_TOPOLOGIES.items():
      for topology in topologies:
        with self.subTest(workload=workload, topology=topology):
          document = self._render(workload, topology)
          self.assertEqual(
              document["metadata"]["labels"]["canon.zero-tim/profile-family"],
              workload,
          )
          worker = p34._worker(document)
          self.assertEqual(
              worker["completions"], renderer.TOPOLOGIES[topology]["workers"]
          )

  def test_q4_debug_is_instruct_16k_three_update_one_hour(self):
    for topology in renderer.WORKLOAD_TOPOLOGIES["q4-debug"]:
      env = p34._env(self._render("q4-debug", topology))
      args = shlex.split(env["CANON_RUN_CMD"])
      for expected in (
          "--model_version=Qwen3-4B-Instruct-2507",
          "--max_response_length=16384",
          "--batch_size=4",
          "--num_generations=4",
          "--max_steps=3",
          "--rollout_batch_timeout_secs=3600",
          "--no-optimizer-offload",
      ):
        self.assertIn(expected, args)

  def test_q32_is_16k_b8_g8_1000_update_ninety_minutes(self):
    for topology, dp in (("64", "4"), ("256", "16")):
      env = p34._env(self._render("q32-train", topology))
      args = shlex.split(env["CANON_RUN_CMD"])
      for expected in (
          "--model_version=Qwen3-32B",
          "--max_response_length=16384",
          "--batch_size=8",
          "--num_generations=8",
          "--max_steps=1000",
          "--rollout_batch_timeout_secs=5400",
          f"--rollout_mesh_dp={dp}",
          f"--train_mesh_dp={dp}",
          "--no-optimizer-offload",
      ):
        self.assertIn(expected, args)

  def test_eval_is_not_a_training_job(self):
    for topology in renderer.WORKLOAD_TOPOLOGIES["q4-clean-eval"]:
      env = p34._env(self._render("q4-clean-eval", topology))
      self.assertEqual(
          env["CANON_RUN_CMD"],
          "python3 -u examples/deepswe/eval_deepswe.py",
      )
      self.assertEqual(env["CANON_P46_EVALUATION"], "1")
      self.assertEqual(env["CANON_P46_EVALUATION_MODE"], "reward_only")
      self.assertEqual(env["CANON_P46_SAMPLING_SOURCE_COMMIT"], SOURCE)
      self.assertEqual(env["CANON_P46_LEGACY_IMPORT_ID"], "")
      self.assertEqual(env["CANON_P46_DEEPSWE_TRAIN"], "0")
      self.assertEqual(env["CANON_P32_TRAIN_ADMITTED"], "0")
      self.assertEqual(env["CANON_P33_WORKLOAD_LAUNCH_ADMITTED"], "0")
      self.assertEqual(env["CANON_P46_LOGICAL_SHARD_INDEX"], "0")
      self.assertEqual(env["CANON_P46_PHYSICAL_SHARD_INDEX"], "0")
      self.assertEqual(env["CANON_P46_FULL_CAMPAIGN"], "0")
      for key in (
          "CANON_PROMPT_PROCESSED_LOGPROBS",
          "CANON_PALLAS_LOGSOFTMAX",
          "CANON_ENGINE_MODULE_C",
          "CANON_RPA_VJP2",
          "CANON_ALIGNMENT_GATE",
          "CANON_ALIGNMENT_TRAIN",
          "CANON_PRE_ALIGN_GATE",
          "CANON_OPT_STATE_RESIDENT",
      ):
        self.assertEqual(env[key], "0", key)

  def test_launch_checks_out_pinned_sha_after_branch_fetch(self):
    document = self._render("q4-clean-eval", "128", full_campaign=True)
    main = p34._container(p34._head(document)["containers"], "jax-tpu")
    command = main["command"][2]
    self.assertIn(
        'git merge-base --is-ancestor "$CANON_EXPECT_COMMIT" FETCH_HEAD',
        command,
    )
    self.assertIn('git reset -q --hard "$CANON_EXPECT_COMMIT"', command)
    self.assertNotIn("git reset -q --hard FETCH_HEAD", command)

  def test_full_eval_campaign_is_one_resumable_runtime(self):
    for topology in renderer.WORKLOAD_TOPOLOGIES["q4-clean-eval"]:
      document = self._render(
          "q4-clean-eval", topology, full_campaign=True
      )
      env = p34._env(document)
      self.assertEqual(env["CANON_P46_FULL_CAMPAIGN"], "1")
      self.assertEqual(env["CANON_P46_EVALUATION_MODE"], "reward_only")
      self.assertEqual(env["CANON_P46_PARITY_CANARY"], "0")
      self.assertTrue(
          env["CANON_STATE"].endswith(f"/state-launches/t{topology}")
      )
      self.assertTrue(env["CANON_RUN_LOG"].endswith("/logs/campaign.log"))
      self.assertEqual(
          document["metadata"]["labels"]["canon.zero-tim/full-campaign"],
          "1",
      )
      self.assertEqual(env["CANON_P46_RESUME_TAG"], f"t{topology}")
    with self.assertRaisesRegex(ValueError, "cannot be a parity"):
      self._render(
          "q4-clean-eval",
          "64",
          evaluation_mode="logprob_observer",
          parity_canary=True,
          full_campaign=True,
      )
    with self.assertRaisesRegex(ValueError, "owns all shard"):
      self._render(
          "q4-clean-eval",
          "128",
          full_campaign=True,
          physical_shard_index=1,
      )

  def test_resume_tag_is_stable_across_distinct_launch_ids(self):
    first = self._render(
        "q4-clean-eval",
        "128",
        full_campaign=True,
        run_id="launch-a",
        resume_tag="wash-q4-001",
    )
    second = self._render(
        "q4-clean-eval",
        "128",
        full_campaign=True,
        run_id="launch-b",
        resume_tag="wash-q4-001",
    )
    first_env = p34._env(first)
    second_env = p34._env(second)
    self.assertNotEqual(first["metadata"]["name"], second["metadata"]["name"])
    self.assertEqual(first_env["CANON_RUN_ID"], "launch-a")
    self.assertEqual(second_env["CANON_RUN_ID"], "launch-b")
    self.assertEqual(
        first_env["CANON_P46_OUTPUT_DIR"], second_env["CANON_P46_OUTPUT_DIR"]
    )
    self.assertIn("/wash-q4-001/outputs", first_env["CANON_P46_OUTPUT_DIR"])
    for document in (first, second):
      self.assertEqual(
          document["metadata"]["labels"]["canon.zero-tim/resume-tag"],
          "wash-q4-001",
      )

  def test_frozen_legacy_snapshot_is_explicit_and_full_campaign_only(self):
    document = self._render(
        "q4-clean-eval",
        "128",
        full_campaign=True,
        run_id="impa",
        resume_tag="wash-q4-import",
        sampling_source_commit="5" * 40,
        legacy_import_id="old-run",
    )
    env = p34._env(document)
    self.assertEqual(env["CANON_EXPECT_COMMIT"], SOURCE)
    self.assertEqual(env["CANON_P46_SAMPLING_SOURCE_COMMIT"], "5" * 40)
    self.assertEqual(env["CANON_P46_LEGACY_IMPORT_ID"], "old-run")
    self.assertEqual(
        document["metadata"]["labels"]["canon.zero-tim/legacy-import-id"],
        "old-run",
    )
    self.assertIn("/wash-q4-import/outputs", env["CANON_P46_OUTPUT_DIR"])
    with self.assertRaisesRegex(ValueError, "requires a full campaign"):
      self._render(
          "q4-clean-eval", "128", legacy_import_id="old-run"
      )
    with self.assertRaisesRegex(ValueError, "legacy import id"):
      self._render(
          "q4-clean-eval",
          "128",
          full_campaign=True,
          legacy_import_id="../old-run",
      )

  def test_resume_tag_rejects_unsafe_paths(self):
    for value in ("../escape", "UPPER", "a" * 64, "dash-"):
      with self.subTest(value=value):
        with self.assertRaisesRegex(ValueError, "resume tag"):
          self._render(
              "q4-clean-eval", "128", full_campaign=True, resume_tag=value
          )
    common = dict(
        source_commit=SOURCE,
        source_branch=p34.DEFAULT_SOURCE_BRANCH,
        client_image=IMAGE,
        run_id="missing-tag",
        cpu_nodepool="cpu",
        worker_nodepool="worker",
        model_pvc="pvc",
        whitelist=p34.P34_CLEAN_WHITELIST,
        whitelist_sha256=p34.P34_CLEAN_WHITELIST_SHA256,
    )
    with self.assertRaisesRegex(ValueError, "explicit resume tag"):
      renderer.render(
          self._base("128"),
          workload="q4-clean-eval",
          topology="128",
          full_campaign=True,
          **common,
      )

  def test_64chip_parity_canary_renders_exact_observer_and_reward_arms(self):
    for mode in ("logprob_observer", "reward_only"):
      document = self._render(
          "q4-clean-eval",
          "64",
          evaluation_mode=mode,
          parity_canary=True,
      )
      env = p34._env(document)
      self.assertEqual(env["CANON_P46_EVALUATION_MODE"], mode)
      self.assertEqual(env["CANON_P46_PARITY_CANARY"], "1")
      self.assertIn(f"/parity/{mode}/", env["CANON_P46_OUTPUT_DIR"])
      self.assertEqual(
          document["metadata"]["labels"]["canon.zero-tim/parity-canary"],
          "1",
      )
    with self.assertRaisesRegex(ValueError, "restricted"):
      self._render(
          "q4-clean-eval", "64", evaluation_mode="logprob_observer"
      )
    with self.assertRaisesRegex(ValueError, "topology 64"):
      self._render("q4-clean-eval", "128", parity_canary=True)

  def test_bad_workload_topology_and_eval_shard_fail_closed(self):
    common = dict(
        source_commit=SOURCE,
        source_branch=p34.DEFAULT_SOURCE_BRANCH,
        client_image=IMAGE,
        run_id="bad",
        cpu_nodepool="cpu",
        worker_nodepool="worker",
        model_pvc="pvc",
        whitelist=p34.P34_CLEAN_WHITELIST,
        whitelist_sha256=p34.P34_CLEAN_WHITELIST_SHA256,
    )
    with self.assertRaisesRegex(ValueError, "unknown P46 workload"):
      renderer.render(self._base("64"), workload="bad", topology="64", **common)
    with self.assertRaisesRegex(ValueError, "topology"):
      renderer.render(self._base("64"), workload="q4-debug", topology="4", **common)
    with self.assertRaisesRegex(ValueError, "q4-debug topology"):
      renderer.render(self._base("256"), workload="q4-debug", topology="256", **common)
    with self.assertRaisesRegex(ValueError, "q4-clean-eval topology"):
      renderer.render(
          self._base("256"), workload="q4-clean-eval", topology="256", **common
      )
    with self.assertRaisesRegex(ValueError, "q32-train topology"):
      renderer.render(self._base("128"), workload="q32-train", topology="128", **common)
    with self.assertRaisesRegex(ValueError, "evaluation-only controls"):
      renderer.render(
          self._base("64"),
          workload="q32-train",
          topology="64",
          resume_tag="eval-only",
          **common,
      )
    with self.assertRaisesRegex(ValueError, "shard"):
      renderer.render(
          self._base("64"),
          workload="q4-clean-eval",
          topology="64",
          physical_shard_index=8,
          **common,
      )
    with self.assertRaisesRegex(ValueError, "logical shard"):
      renderer.render(
          self._base("64"),
          workload="q4-clean-eval",
          topology="64",
          logical_shard_index=58,
          **common,
      )
    with self.assertRaisesRegex(ValueError, "shard"):
      renderer.render(
          self._base("64"),
          workload="q4-clean-eval",
          topology="64",
          logical_shard_index=57,
          physical_shard_index=7,
          **common,
      )


if __name__ == "__main__":
  unittest.main()
