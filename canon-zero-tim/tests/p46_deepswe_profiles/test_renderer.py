"""Renderer contracts for all three P46 64/256 JobSet families."""

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

  def _render(self, workload: str, topology: str):
    return renderer.render(
        self._base(topology),
        workload=workload,
        topology=topology,
        source_commit=SOURCE,
        source_branch=p34.DEFAULT_SOURCE_BRANCH,
        client_image=IMAGE,
        run_id=f"t{topology}",
        cpu_nodepool="deepswe-cpu-pool",
        worker_nodepool=f"v5p-{topology}",
        model_pvc="models-pvc",
        whitelist=p34.P34_CLEAN_WHITELIST,
        whitelist_sha256=p34.P34_CLEAN_WHITELIST_SHA256,
        logical_shard_index=0,
        physical_shard_index=0,
    )

  def test_all_three_families_render_on_both_topologies(self):
    for workload in renderer.WORKLOADS:
      for topology in renderer.TOPOLOGIES:
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
    for topology in renderer.TOPOLOGIES:
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
    for topology in renderer.TOPOLOGIES:
      env = p34._env(self._render("q4-clean-eval", topology))
      self.assertEqual(
          env["CANON_RUN_CMD"],
          "python3 -u examples/deepswe/eval_deepswe.py",
      )
      self.assertEqual(env["CANON_P46_EVALUATION"], "1")
      self.assertEqual(env["CANON_P46_DEEPSWE_TRAIN"], "0")
      self.assertEqual(env["CANON_P32_TRAIN_ADMITTED"], "0")
      self.assertEqual(env["CANON_P33_WORKLOAD_LAUNCH_ADMITTED"], "0")
      self.assertEqual(env["CANON_P46_LOGICAL_SHARD_INDEX"], "0")
      self.assertEqual(env["CANON_P46_PHYSICAL_SHARD_INDEX"], "0")

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
