#!/usr/bin/env python3
"""Contracts for the P45/M15 exact-TiTO diagnostic pair renderer."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "canon-zero-tim"
SCRIPT = (
    PACKAGE
    / "tasks/multiturn-tito-cross-workload/scripts/render_tito_diagnostic_pair.py"
)
SPEC = importlib.util.spec_from_file_location("p57_tito_diagnostic_renderer", SCRIPT)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import P57 TiTO diagnostic renderer")
renderer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = renderer
SPEC.loader.exec_module(renderer)
BASE = PACKAGE / "cluster/jobset-64chip.yaml"


def _env(document: dict) -> dict[str, str]:
  pod = document["spec"]["replicatedJobs"][0]["template"]["spec"][
      "template"
  ]["spec"]
  container = next(item for item in pod["containers"] if item["name"] == "jax-tpu")
  return {
      item["name"]: item["value"]
      for item in container["env"]
      if "value" in item
  }


class TitoDiagnosticRendererTest(unittest.TestCase):

  def test_pair_renders_rollout_only_and_passes_profile_preflight(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      p45, m15, index = renderer.render_pair(
          source_commit="a" * 40,
          output_dir=root / "rendered",
          p45_run_id="titop45",
          m15_run_id="titom15",
          campaign_root="p57-tito-diagnostic",
          base_path=BASE,
          cpu_nodepool="canon-cpu-pool",
      )
      self.assertTrue(index.is_file())
      expected_scheduling = None
      for workload, path in (("p45", p45), ("m15", m15)):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        scheduling = renderer._scheduling_contract(document)
        if expected_scheduling is None:
          expected_scheduling = scheduling
        else:
          self.assertEqual(scheduling, expected_scheduling)
        worker = document["spec"]["replicatedJobs"][1]
        self.assertEqual(worker["template"]["spec"]["parallelism"], 16)
        self.assertEqual(worker["template"]["spec"]["completions"], 16)
        worker_template = worker["template"]["spec"]["template"]
        self.assertEqual(
            worker_template["metadata"]["annotations"][
                "alpha.jobset.sigs.k8s.io/exclusive-topology"
            ],
            "cloud.google.com/gke-nodepool",
        )
        self.assertEqual(
            worker_template["spec"]["nodeSelector"][
                "cloud.google.com/gke-tpu-topology"
            ],
            "4x4x4",
        )
        env = _env(document)
        command = env["CANON_RUN_CMD"].split()
        self.assertEqual(env["CANON_P33_RUN_STAGE"], "rollout-only")
        self.assertEqual(env["CANON_P33_NO_COMMIT"], "1")
        self.assertEqual(env["CANON_P57_TOKEN_CONTINUITY"], "exact")
        self.assertIn("--evaluation_only", command)
        self.assertIn(
            "--env_max_steps=15" if workload == "m15" else "--env_max_steps=5",
            command,
        )
        state = root / f"state-{workload}"
        state.mkdir()
        preflight = subprocess.run(
            ["bash", "cluster/steps/00_env.sh"],
            cwd=PACKAGE,
            env={
                **os.environ,
                **env,
                "CANON_PKG": str(PACKAGE),
                "CANON_STATE": str(state),
                "INJECTED_HF_TOKEN": "test-token",
                "INJECTED_WANDB_API_KEY": "test-key",
            },
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(preflight.returncode, 0, preflight.stderr)
        self.assertIn("collect-64 diagnostic enabled", preflight.stdout)
        resolved = (state / "env.sh").read_text(encoding="utf-8")
        self.assertIn("export CANON_P57_TITO_GCS_INTERVAL_SECONDS=30", resolved)
        self.assertIn("export CANON_P59_RANK_PARALLEL_BACKWARD=0", resolved)

  def test_pair_rejects_duplicate_ids_and_overwrite(self):
    with tempfile.TemporaryDirectory() as tmp:
      with self.assertRaisesRegex(ValueError, "must differ"):
        renderer.render_pair(
            source_commit="a" * 40,
            output_dir=Path(tmp) / "bad",
            p45_run_id="same",
            m15_run_id="same",
            campaign_root="p57-tito-diagnostic",
            base_path=BASE,
            cpu_nodepool="canon-cpu-pool",
        )

  def test_profile_cannot_launder_missing_selector_or_raw_gcs_override(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      p45, _, _ = renderer.render_pair(
          source_commit="a" * 40,
          output_dir=root / "rendered",
          p45_run_id="titop45",
          m15_run_id="titom15",
          campaign_root="p57-tito-diagnostic",
          base_path=BASE,
          cpu_nodepool="canon-cpu-pool",
      )
      base_env = _env(yaml.safe_load(p45.read_text(encoding="utf-8")))
      for mutation in (
          "missing-selector",
          "missing-workload",
          "missing-dp",
          "missing-tp",
          "raw-gcs",
      ):
        with self.subTest(mutation=mutation):
          env = dict(base_env)
          if mutation == "missing-selector":
            env.pop("CANON_P57_TOKEN_CONTINUITY")
          elif mutation == "missing-workload":
            env.pop("CANON_P32_WORKLOAD")
          elif mutation == "missing-dp":
            env.pop("CANON_DP_SIZE")
          elif mutation == "missing-tp":
            env.pop("CANON_TP_SIZE")
          else:
            env["CANON_P57_TITO_GCS_PREFIX"] = (
                "gs://yuxzhang-tunix-models/canon-zero-tim/evidence/"
                "p57-tito/foreign/attempt-direct"
            )
          state = root / f"state-{mutation}"
          state.mkdir()
          preflight = subprocess.run(
              ["bash", "cluster/steps/00_env.sh"],
              cwd=PACKAGE,
              env={
                  **os.environ,
                  **env,
                  "CANON_PKG": str(PACKAGE),
                  "CANON_STATE": str(state),
                  "INJECTED_HF_TOKEN": "test-token",
                  "INJECTED_WANDB_API_KEY": "test-key",
              },
              text=True,
              capture_output=True,
              check=False,
          )
          self.assertNotEqual(preflight.returncode, 0)
          self.assertIn("raw P57 exact TITO identity drifted", preflight.stderr)
      output = Path(tmp) / "existing"
      output.mkdir()
      with self.assertRaises(FileExistsError):
        renderer.render_pair(
            source_commit="a" * 40,
            output_dir=output,
            p45_run_id="titop45",
            m15_run_id="titom15",
            campaign_root="p57-tito-diagnostic",
            base_path=BASE,
            cpu_nodepool="canon-cpu-pool",
        )


if __name__ == "__main__":
  unittest.main()
