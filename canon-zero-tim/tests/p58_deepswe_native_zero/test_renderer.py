#!/usr/bin/env python3
"""Fail-closed contracts for the paired P58 JobSet renderer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shlex
import sys
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "canon-zero-tim"
sys.path.insert(0, str(PKG / "cluster"))
SPEC = importlib.util.spec_from_file_location(
    "p58_renderer", PKG / "cluster/render_p58_deepswe_tim.py"
)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import P58 renderer")
renderer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = renderer
SPEC.loader.exec_module(renderer)


class P58RendererTest(unittest.TestCase):

  def _render(self, arm: str, stage: str = "three-update", **overrides):
    base = yaml.safe_load((PKG / "cluster/jobset-64chip.yaml").read_text())
    kwargs = dict(
        source_commit="1" * 40,
        source_branch="yuxzhang/canon-zero-tim",
        client_image="registry.example/tunix@sha256:" + "2" * 64,
        run_id="pair-test",
        stage=stage,
        arm=arm,
        cpu_nodepool="cpu-pool",
        worker_nodepool="tpu-pool",
        model_pvc="model-pvc",
    )
    kwargs.update(overrides)
    return renderer.render(base, **kwargs)

  def test_both_arms_and_horizons_render_on_128_chips(self):
    for arm in ("native", "zero"):
      for stage, steps in (("three-update", 3), ("full", 1000)):
        with self.subTest(arm=arm, stage=stage):
          document = self._render(arm, stage)
          env = renderer.p34._env(document)
          worker = renderer.p34._worker(document)
          self.assertEqual(worker["completions"], 32)
          self.assertEqual(worker["parallelism"], 32)
          self.assertEqual(env["CANON_P58_TIM_ARM"], arm)
          self.assertEqual(env["CANON_P58_EXPECTED_UPDATES"], str(steps))
          self.assertIn(f"--max_steps={steps}", env["CANON_RUN_CMD"])
          self.assertIn("--num_generations=16", env["CANON_RUN_CMD"])
          self.assertIn("--max_concurrency=64", env["CANON_RUN_CMD"])
          self.assertNotIn("--max_concurrency=128", env["CANON_RUN_CMD"])
          self.assertIn("--loss_scale_factor=16384", env["CANON_RUN_CMD"])
          self.assertIn(
              "--loss_denominator_weighted_accumulation",
              env["CANON_RUN_CMD"],
          )
          self.assertEqual(env["CANON_P34_CLEAN_ROWS"], "1012")

  def test_pair_diff_is_registered_treatment_only(self):
    native = self._render("native")
    zero = self._render("zero")
    self.assertEqual(
        renderer.recipe_signature(native), renderer.recipe_signature(zero)
    )
    self.assertEqual(renderer.treatment_signature(native), {
        "arm": "native",
        "alignment_warning_only": "1",
        "proxy_xla": [],
    })
    self.assertEqual(renderer.treatment_signature(zero), {
        "arm": "zero",
        "alignment_warning_only": "0",
        "proxy_xla": [renderer.p34.PROXY_XLA_FLAG],
    })

  def test_optional_algorithm_interventions_are_absent(self):
    for arm in ("native", "zero"):
      args = shlex.split(
          renderer.p34._env(self._render(arm))["CANON_RUN_CMD"]
      )
      self.assertIn("--use_rollout_logps", args)
      self.assertNotIn("--sampler_is", args)
      self.assertFalse(
          any(item.startswith("--group_clip_filter_threshold") for item in args)
      )
      self.assertNotIn("--optimizer-offload", args)
      index = args.index("--filter_statuses")
      self.assertEqual(
          tuple(args[index + 1:index + 7]), renderer._FILTER_STATUSES
      )

  def test_kueue_managed_worker_pool_does_not_become_literal_affinity(self):
    for worker_nodepool in ("auto", "none", "tpu-v5p-slice", "any"):
      with self.subTest(worker_nodepool=worker_nodepool):
        document = self._render(
            "native", "full", worker_nodepool=worker_nodepool
        )
        worker_pod = renderer.p34._worker(document)["template"]["spec"]
        self.assertNotIn(
            "cloud.google.com/gke-nodepool", worker_pod["nodeSelector"]
        )
        self.assertEqual(
            worker_pod["nodeSelector"]["cloud.google.com/gke-tpu-topology"],
            "4x4x8",
        )

  def test_explicit_worker_pool_remains_exact(self):
    document = self._render(
        "native", "full", worker_nodepool="mlperf-v5p-128-np-0"
    )
    worker_pod = renderer.p34._worker(document)["template"]["spec"]
    self.assertEqual(
        worker_pod["nodeSelector"]["cloud.google.com/gke-nodepool"],
        "mlperf-v5p-128-np-0",
    )

  def test_unregistered_data_arm_and_stage_are_rejected(self):
    with self.assertRaisesRegex(ValueError, "native or zero"):
      self._render("mystery")
    with self.assertRaisesRegex(ValueError, "three-update or full"):
      self._render("native", "one-update")
    with self.assertRaisesRegex(ValueError, "1012-task"):
      self._render(
          "native", whitelist="unreviewed.jsonl", whitelist_sha256="3" * 64
      )


if __name__ == "__main__":
  unittest.main()
