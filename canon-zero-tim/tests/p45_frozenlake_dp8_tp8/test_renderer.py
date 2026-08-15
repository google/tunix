"""Tests for the isolated P45 DP8xTP8 resident FrozenLake renderer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


_ROOT = Path(__file__).resolve().parents[3]
_RENDERER_PATH = _ROOT / "canon-zero-tim/cluster/render_p45_frozenlake.py"
_BASE_PATH = _ROOT / "canon-zero-tim/cluster/jobset-64chip.yaml"
_SOURCE_COMMIT = "4" * 40
_CHECKPOINT_TAG = "fl-prod-001"

_SPEC = importlib.util.spec_from_file_location(
    "render_p45_frozenlake", _RENDERER_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
renderer = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = renderer
_SPEC.loader.exec_module(renderer)


def _main_env(document):
  pod = document["spec"]["replicatedJobs"][0]["template"]["spec"][
      "template"
  ]["spec"]
  container = next(
      item for item in pod["containers"] if item["name"] == "jax-tpu"
  )
  return {entry["name"]: entry.get("value") for entry in container["env"]}


class RenderP45FrozenLakeTest(unittest.TestCase):

  def test_renders_isolated_full_and_eval_resident_jobsets(self):
    with tempfile.TemporaryDirectory() as tmp:
      outputs = renderer.render_all(
          base_path=_BASE_PATH,
          output_dir=Path(tmp),
          source_commit=_SOURCE_COMMIT,
          run_id="p45-local",
          checkpoint_tag=_CHECKPOINT_TAG,
          checkpoint_mode="new",
      )
      self.assertEqual(len(outputs), 2)
      documents = [yaml.safe_load(path.read_text()) for path in outputs]
      self.assertEqual(len({doc["metadata"]["name"] for doc in documents}), 2)
      for document in documents:
        env = _main_env(document)
        self.assertEqual(
            env["CANON_PROFILE_FILE"],
            "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-resident.env",
        )
        self.assertEqual(env["CANON_P33_SHARED_MESH"], "8,8")
        self.assertEqual(env["MIN_TOKEN_BUCKET"], "2048")
        self.assertEqual(env["CANON_OPT_STATE_RESIDENT"], "1")
        self.assertEqual(env["CANON_P30_OPT_STATE_OFFLOAD"], "0")
        self.assertEqual(env["CANON_P33_RUN_STAGE"], "full")
        self.assertEqual(env["CANON_P33_NO_COMMIT"], "0")
        self.assertEqual(env["CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY"], "1")
        self.assertEqual(env["CANON_FROZENLAKE_CKPT_MODE"], "new")
        self.assertEqual(
            env["CANON_FROZENLAKE_CKPT_ROOT"],
            "gs://yuxzhang-tunix-models/canon-zero-tim/checkpoints/frozenlake",
        )
        self.assertEqual(env["CANON_FROZENLAKE_CKPT_TAG"], _CHECKPOINT_TAG)
        self.assertEqual(env["CANON_FROZENLAKE_CKPT_INTERVAL"], "10")
        self.assertEqual(env["CANON_FROZENLAKE_CKPT_MAX_TO_KEEP"], "1")
        self.assertEqual(document["spec"]["failurePolicy"]["maxRestarts"], 0)
        command = env["CANON_RUN_CMD"]
        for argument in (
            "--mesh_dp=8",
            "--mesh_tp=8",
            "--batch_size=32",
            "--mini_batch_size=32",
            "--train_trajectory_micro_batch_size=8",
            "--num_generations=8",
            "--vllm_max_num_seqs=32",
            "--vllm_max_num_batched_tokens=256",
            "--learning_rate=1e-6",
            "--max_steps=450",
        ):
          self.assertIn(argument, command)

      by_eval = {
          _main_env(document)["CANON_P33_ENABLE_EVAL"]: _main_env(document)
          for document in documents
      }
      self.assertEqual(set(by_eval), {"0", "1"})
      self.assertNotIn("--num_test_batches=4", by_eval["0"]["CANON_RUN_CMD"])
      self.assertIn("--num_test_batches=4", by_eval["1"]["CANON_RUN_CMD"])
      self.assertIn("--eval_every_n_steps=10", by_eval["1"]["CANON_RUN_CMD"])

  def test_refuses_overwrite(self):
    with tempfile.TemporaryDirectory() as tmp:
      output_dir = Path(tmp)
      renderer.render_all(
          base_path=_BASE_PATH,
          output_dir=output_dir,
          source_commit=_SOURCE_COMMIT,
          run_id="p45-once",
          checkpoint_tag=_CHECKPOINT_TAG,
          checkpoint_mode="new",
      )
      with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
        renderer.render_all(
            base_path=_BASE_PATH,
            output_dir=output_dir,
            source_commit=_SOURCE_COMMIT,
            run_id="p45-once",
            checkpoint_tag=_CHECKPOINT_TAG,
            checkpoint_mode="new",
        )

  def test_renders_explicit_resume_mode(self):
    with tempfile.TemporaryDirectory() as tmp:
      outputs = renderer.render_all(
          base_path=_BASE_PATH,
          output_dir=Path(tmp),
          source_commit=_SOURCE_COMMIT,
          run_id="p45-resume",
          checkpoint_tag=_CHECKPOINT_TAG,
          checkpoint_mode="resume",
      )
      for path in outputs:
        env = _main_env(yaml.safe_load(path.read_text()))
        self.assertEqual(env["CANON_FROZENLAKE_CKPT_MODE"], "resume")
        self.assertEqual(env["CANON_FROZENLAKE_CKPT_TAG"], _CHECKPOINT_TAG)

  def test_rejects_bad_checkpoint_identity(self):
    for tag in ("Uppercase", "bad/slash", "-leading", "trailing-"):
      with self.subTest(tag=tag), tempfile.TemporaryDirectory() as tmp:
        with self.assertRaisesRegex(ValueError, "checkpoint tag"):
          renderer.render_all(
              base_path=_BASE_PATH,
              output_dir=Path(tmp),
              source_commit=_SOURCE_COMMIT,
              run_id="p45-invalid",
              checkpoint_tag=tag,
              checkpoint_mode="new",
          )
    with tempfile.TemporaryDirectory() as tmp:
      with self.assertRaisesRegex(ValueError, "checkpoint mode"):
        renderer.render_all(
            base_path=_BASE_PATH,
            output_dir=Path(tmp),
            source_commit=_SOURCE_COMMIT,
            run_id="p45-invalid",
            checkpoint_tag=_CHECKPOINT_TAG,
            checkpoint_mode="automatic",
        )


if __name__ == "__main__":
  unittest.main()
