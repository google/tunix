"""Tests for the strict three-JobSet P33 queue renderer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shlex
import sys
import tempfile
import unittest

import yaml

from tunix.rl import dp_workloads


_ROOT = Path(__file__).resolve().parents[3]
_RENDERER_PATH = _ROOT / "canon-zero-tim/cluster/render_p33_jobsets.py"
_BASE_PATH = _ROOT / "canon-zero-tim/cluster/jobset-64chip.yaml"
_SOURCE_COMMIT = "1" * 40
_RUN_ID = "queue-a"

_MODULE_SPEC = importlib.util.spec_from_file_location(
    "render_p33_jobsets", _RENDERER_PATH
)
assert _MODULE_SPEC is not None and _MODULE_SPEC.loader is not None
renderer = importlib.util.module_from_spec(_MODULE_SPEC)
sys.modules[_MODULE_SPEC.name] = renderer
_MODULE_SPEC.loader.exec_module(renderer)


def _main_env(document):
  pod = document["spec"]["replicatedJobs"][0]["template"]["spec"][
      "template"
  ]["spec"]
  container = next(
      item for item in pod["containers"] if item["name"] == "jax-tpu"
  )
  return {
      entry["name"]: entry.get("value") for entry in container["env"]
  }


class RenderP33JobSetsTest(unittest.TestCase):

  def _render(self, output_dir: Path):
    return renderer.render_all(
        base_path=_BASE_PATH,
        output_dir=output_dir,
        source_commit=_SOURCE_COMMIT,
        run_id=_RUN_ID,
    )

  def test_renders_three_isolated_strict_jobsets(self):
    with tempfile.TemporaryDirectory() as tmp:
      outputs = self._render(Path(tmp))
      self.assertEqual(len(outputs), 3)
      documents = [yaml.safe_load(path.read_text()) for path in outputs]
      names = [document["metadata"]["name"] for document in documents]
      self.assertEqual(len(set(names)), 3)
      scratches = []
      states = []
      wandb_names = []
      for document in documents:
        self.assertEqual(document["spec"]["failurePolicy"]["maxRestarts"], 0)
        self.assertEqual(
            document["spec"]["replicatedJobs"][0]["template"]["spec"][
                "backoffLimit"
            ],
            0,
        )
        self.assertEqual(
            document["spec"]["replicatedJobs"][1]["template"]["spec"][
                "backoffLimit"
            ],
            0,
        )
        env = _main_env(document)
        self.assertEqual(env["CANON_EXPECT_COMMIT"], _SOURCE_COMMIT)
        self.assertEqual(env["CANON_MODE"], "run")
        self.assertEqual(env["CANON_P32_TRAIN_ADMITTED"], "1")
        self.assertEqual(env["CANON_P32_DP_REDUCTION_ADMITTED"], "1")
        self.assertEqual(env["CANON_P33_WORKLOAD_LAUNCH_ADMITTED"], "1")
        self.assertEqual(env["CANON_P33_SHARED_MESH"], "16,4")
        self.assertEqual(env["CANON_P32_EXPECT_MODEL_MESH_IDS"], "")
        self.assertNotIn("CANON_P32_RC_STAGE", env)
        states.append(env["CANON_STATE"])
        wandb_names.append(env["CANON_WANDB_RUN_NAME"])
        head = document["spec"]["replicatedJobs"][0]["template"]["spec"][
            "template"
        ]["spec"]
        scratches.append(tuple(
            arg
            for container in head["initContainers"]
            for arg in container["args"]
            if arg.startswith("--gcs_scratch_location=")
        ))
      self.assertEqual(len(set(states)), 3)
      self.assertEqual(len(set(wandb_names)), 3)
      self.assertEqual(len(set(scratches)), 3)
      self.assertTrue(all(len(values) == 2 for values in scratches))

  def test_rendered_commands_equal_frozen_workload_commands(self):
    with tempfile.TemporaryDirectory() as tmp:
      outputs = self._render(Path(tmp))
      by_stage = {}
      for path in outputs:
        document = yaml.safe_load(path.read_text())
        env = _main_env(document)
        profile = env["CANON_PROFILE_FILE"]
        workload_name = "gsm8k" if "gsm8k" in profile else "frozenlake"
        by_stage[(workload_name, env["CANON_P33_RUN_STAGE"])] = env
      for workload_name, stage in (
          ("frozenlake", "backward-no-commit"),
          ("gsm8k", "full"),
          ("frozenlake", "full"),
      ):
        expected = shlex.join(
            dp_workloads.get_workload(workload_name).command(run_stage=stage)
        )
        self.assertEqual(by_stage[(workload_name, stage)]["CANON_RUN_CMD"], expected)

      frozenlake_command = by_stage[("frozenlake", "full")]["CANON_RUN_CMD"]
      self.assertIn("--vllm_max_num_seqs=16", frozenlake_command)
      self.assertIn("--vllm_max_num_batched_tokens=256", frozenlake_command)
      self.assertNotIn("--vllm_max_num_seqs=256", frozenlake_command)
      self.assertNotIn("--vllm_max_num_batched_tokens=4096", frozenlake_command)

  def test_frozenlake_jobs_disable_periodic_evaluation(self):
    with tempfile.TemporaryDirectory() as tmp:
      for path in self._render(Path(tmp)):
        env = _main_env(yaml.safe_load(path.read_text()))
        if "frozenlake" in env["CANON_PROFILE_FILE"]:
          self.assertEqual(env["CANON_P33_DISABLE_EVAL"], "1")

  def test_rejects_invalid_source_commit_and_run_id(self):
    base = renderer.load_base(_BASE_PATH)
    spec = renderer._SPECS[0]
    with self.assertRaisesRegex(ValueError, "source commit"):
      renderer.render_jobset(base, spec, "abc", _RUN_ID)
    with self.assertRaisesRegex(ValueError, "run id"):
      renderer.render_jobset(base, spec, _SOURCE_COMMIT, "Bad_ID")

  def test_refuses_to_overwrite_existing_render(self):
    with tempfile.TemporaryDirectory() as tmp:
      output_dir = Path(tmp)
      self._render(output_dir)
      with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
        self._render(output_dir)

  def test_negative_control_rejects_shared_scratch(self):
    base = renderer.load_base(_BASE_PATH)
    spec = renderer._SPECS[0]
    document = renderer.render_jobset(base, spec, _SOURCE_COMMIT, _RUN_ID)
    head = document["spec"]["replicatedJobs"][0]["template"]["spec"][
        "template"
    ]["spec"]
    proxy = next(
        item for item in head["initContainers"] if item["name"] == "pathways-proxy"
    )
    scratch_index = next(
        index
        for index, value in enumerate(proxy["args"])
        if value.startswith("--gcs_scratch_location=")
    )
    proxy["args"][scratch_index] = "--gcs_scratch_location=gs://shared"
    with self.assertRaisesRegex(ValueError, "isolated scratch"):
      renderer.validate_jobset(
          document, spec, _SOURCE_COMMIT, _RUN_ID
      )


if __name__ == "__main__":
  unittest.main()
