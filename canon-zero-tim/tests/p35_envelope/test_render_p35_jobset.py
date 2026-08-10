"""Fail-closed tests for the single P35 envelope-short JobSet."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
RENDERER = ROOT / "canon-zero-tim/cluster/render_p35_jobset.py"
SPEC = importlib.util.spec_from_file_location("p35_renderer", RENDERER)
assert SPEC is not None and SPEC.loader is not None
renderer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = renderer
SPEC.loader.exec_module(renderer)


class RenderP35JobSetTest(unittest.TestCase):

  def test_proxy_inherits_excess_precision_env_from_p33(self):
    document = renderer.render(
        base_path=ROOT / "canon-zero-tim/cluster/jobset-64chip.yaml",
        source_commit="1" * 40,
        run_id="r20",
    )
    pod = document["spec"]["replicatedJobs"][0]["template"]["spec"][
        "template"
    ]["spec"]
    proxy = next(
        item
        for item in pod["initContainers"]
        if item["name"] == "pathways-proxy"
    )
    entries = [e for e in proxy["env"] if e["name"] == "XLA_FLAGS"]
    self.assertEqual(
        entries,
        [{
            "name": "XLA_FLAGS",
            "value": "--xla_allow_excess_precision=false",
        }],
    )
    self.assertFalse([a for a in proxy["args"] if "excess_precision" in a])


  def test_renders_one_attempt_zero_pre_backward_job(self):
    document = renderer.render(
        base_path=ROOT / "canon-zero-tim/cluster/jobset-64chip.yaml",
        source_commit="1" * 40,
        run_id="r20",
    )
    env = renderer.p33._env_values(document)
    self.assertEqual(document["spec"]["failurePolicy"]["maxRestarts"], 0)
    self.assertEqual(env["CANON_P35_ENVELOPE"], "1")
    self.assertEqual(env["CANON_P35_EXACT_REPLAY"], "1")
    self.assertTrue(env["CANON_P35_PRE_REPLAY_REPORT"].endswith(
        "/p35_envelope.pre_replay.json"
    ))
    self.assertTrue(env["CANON_P35_EXACT_REPLAY_REPORT"].endswith(
        "/p35_exact_replay.json"
    ))
    self.assertEqual(env["CANON_P33_RUN_STAGE"], "envelope-short")
    self.assertEqual(env["CANON_P33_NO_COMMIT"], "1")
    self.assertIn("--max_response_length=256", env["CANON_RUN_CMD"])
    self.assertIn("--max_steps=1", env["CANON_RUN_CMD"])

  def test_negative_control_rejects_training_stage(self):
    document = renderer.render(
        base_path=ROOT / "canon-zero-tim/cluster/jobset-64chip.yaml",
        source_commit="1" * 40,
        run_id="r20",
    )
    env = renderer._main_env(document)
    next(item for item in env if item["name"] == "CANON_P33_RUN_STAGE")["value"] = "full"
    with self.assertRaisesRegex(ValueError, "drifted"):
      renderer.validate(document, source_commit="1" * 40, run_id="r20")

  def test_negative_control_rejects_disabled_exact_replay(self):
    document = renderer.render(
        base_path=ROOT / "canon-zero-tim/cluster/jobset-64chip.yaml",
        source_commit="1" * 40,
        run_id="r20",
    )
    env = renderer._main_env(document)
    next(
        item for item in env if item["name"] == "CANON_P35_EXACT_REPLAY"
    )["value"] = "0"
    with self.assertRaisesRegex(ValueError, "drifted"):
      renderer.validate(document, source_commit="1" * 40, run_id="r20")

  def test_renders_fail_closed_stage_probe(self):
    document = renderer.render(
        base_path=ROOT / "canon-zero-tim/cluster/jobset-64chip.yaml",
        source_commit="1" * 40,
        run_id="r31",
        stage_probe=True,
    )
    env = renderer.p33._env_values(document)
    self.assertEqual(env["CANON_P35_REPLAY_STAGE_PROBE"], "1")
    self.assertTrue(
        env["CANON_P35_REPLAY_STAGE_REPORT"].endswith(
            "/p35_replay_stages.jsonl"
        )
    )
    self.assertTrue(
        env["CANON_P35_REPLAY_STAGE_CLASSIFICATION"].endswith(
            "/p35_replay_stages.classification.json"
        )
    )

  def test_negative_control_rejects_unattested_stage_probe(self):
    document = renderer.render(
        base_path=ROOT / "canon-zero-tim/cluster/jobset-64chip.yaml",
        source_commit="1" * 40,
        run_id="r31",
        stage_probe=True,
    )
    env = renderer._main_env(document)
    next(
        item
        for item in env
        if item["name"] == "CANON_P35_REPLAY_STAGE_REPORT"
    )["value"] = ""
    with self.assertRaisesRegex(ValueError, "drifted"):
      renderer.validate(
          document,
          source_commit="1" * 40,
          run_id="r31",
          stage_probe=True,
      )

  def test_stage_probe_passes_preflight_and_requires_exact_replay(self):
    document = renderer.render(
        base_path=ROOT / "canon-zero-tim/cluster/jobset-64chip.yaml",
        source_commit="1" * 40,
        run_id="r31",
        stage_probe=True,
    )
    rendered_env = renderer.p33._env_values(document)
    with tempfile.TemporaryDirectory() as state:
      runtime_env = os.environ.copy()
      runtime_env.update(rendered_env)
      runtime_env.update({
          "CANON_PKG": str(ROOT / "canon-zero-tim"),
          "CANON_STATE": state,
          "CANON_P35_ENVELOPE_REPORT": f"{state}/p35.json",
          "CANON_P35_PRE_REPLAY_REPORT": f"{state}/p35.pre.json",
          "CANON_P35_METADATA_DIR": f"{state}/p35_metadata",
          "CANON_P35_CLASSIFICATION": f"{state}/p35.classification.json",
          "CANON_P35_EXACT_REPLAY_REPORT": f"{state}/replay.json",
          "CANON_P35_EXACT_REPLAY_CLASSIFICATION": (
              f"{state}/replay.classification.json"
          ),
          "CANON_P35_REPLAY_STAGE_REPORT": f"{state}/stages.jsonl",
          "CANON_P35_REPLAY_STAGE_CLASSIFICATION": (
              f"{state}/stages.classification.json"
          ),
          "CANON_RUN_LOG": f"{state}/run.log",
          "CANON_PRE_ALIGN_REPORT": f"{state}/pre_alignment.jsonl",
          "CANON_ALIGN_REPORT": f"{state}/alignment.jsonl",
          "CANON_UPDATE_REPORT": f"{state}/updates.jsonl",
          "INJECTED_WANDB_API_KEY": "test-key-not-a-credential",
      })
      preflight = ROOT / "canon-zero-tim/cluster/steps/00_env.sh"
      accepted = subprocess.run(
          ["bash", str(preflight)],
          cwd=ROOT,
          env=runtime_env,
          check=False,
          capture_output=True,
          text=True,
      )
      self.assertEqual(accepted.returncode, 0, accepted.stderr)
      self.assertIn("first-record stage probe enabled", accepted.stdout)

      runtime_env["CANON_P35_EXACT_REPLAY"] = "0"
      rejected = subprocess.run(
          ["bash", str(preflight)],
          cwd=ROOT,
          env=runtime_env,
          check=False,
          capture_output=True,
          text=True,
      )
      self.assertNotEqual(rejected.returncode, 0)
      self.assertIn("requires exact replay", rejected.stderr)

  def test_rendered_command_passes_cluster_preflight_and_64_fails(self):
    document = renderer.render(
        base_path=ROOT / "canon-zero-tim/cluster/jobset-64chip.yaml",
        source_commit="1" * 40,
        run_id="bridge",
    )
    rendered_env = renderer.p33._env_values(document)
    with tempfile.TemporaryDirectory() as state:
      runtime_env = os.environ.copy()
      runtime_env.update(rendered_env)
      runtime_env.update({
          "CANON_PKG": str(ROOT / "canon-zero-tim"),
          "CANON_STATE": state,
          "CANON_P35_ENVELOPE_REPORT": f"{state}/p35.json",
          "CANON_P35_METADATA_DIR": f"{state}/p35_metadata",
          "CANON_P35_CLASSIFICATION": f"{state}/p35.classification.json",
          "CANON_RUN_LOG": f"{state}/run.log",
          "CANON_PRE_ALIGN_REPORT": f"{state}/pre_alignment.jsonl",
          "CANON_ALIGN_REPORT": f"{state}/alignment.jsonl",
          "CANON_UPDATE_REPORT": f"{state}/updates.jsonl",
          "INJECTED_WANDB_API_KEY": "test-key-not-a-credential",
      })
      preflight = ROOT / "canon-zero-tim/cluster/steps/00_env.sh"
      accepted = subprocess.run(
          ["bash", str(preflight)],
          cwd=ROOT,
          env=runtime_env,
          check=False,
          capture_output=True,
          text=True,
      )
      self.assertEqual(accepted.returncode, 0, accepted.stderr)
      self.assertIn("response-256", accepted.stdout)

      runtime_env["CANON_RUN_CMD"] = rendered_env["CANON_RUN_CMD"].replace(
          "--max_response_length=256", "--max_response_length=64"
      )
      rejected = subprocess.run(
          ["bash", str(preflight)],
          cwd=ROOT,
          env=runtime_env,
          check=False,
          capture_output=True,
          text=True,
      )
      self.assertNotEqual(rejected.returncode, 0)
      self.assertNotIn("P35 envelope contract OK", rejected.stdout)
      self.assertIn("must pin max_response_length=256", rejected.stderr)


if __name__ == "__main__":
  unittest.main()
