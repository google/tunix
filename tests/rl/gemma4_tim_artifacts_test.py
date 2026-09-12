"""Host negative controls for immutable Gemma evidence and input contracts."""

import json
from pathlib import Path
import tempfile
import unittest

from examples.frozenlake.gemma4_tim import artifacts, recipe


class ArtifactTest(unittest.TestCase):

  def test_green_red_and_post_manifest_tamper(self):
    for status in ("GREEN", "RED", "INCONCLUSIVE"):
      with self.subTest(status=status), tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "driver.log").write_text("started\n")
        (root / "worker.log").write_text("worker exited\n")
        artifacts.seal_run(root, {"status": status})
        artifacts.verify_run(root)
        self.assertEqual((root / "driver.log").read_text().count("GEMMA4_E2B_TERMINAL"), 1)
        (root / "worker.log").write_text("tampered after manifest\n")
        with self.assertRaises(recipe.RecipeError):
          artifacts.verify_run(root)

  def test_unlisted_file_and_reseal_refused(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      artifacts.seal_run(root, {"status": "GREEN"})
      with self.assertRaises(FileExistsError):
        artifacts.seal_run(root, {"status": "GREEN"})
      (root / "unlisted.txt").write_text("not sealed")
      with self.assertRaises(recipe.RecipeError):
        artifacts.verify_run(root)

  def test_manifest_self_entry_and_path_escape_refused(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      for name in ("../escape", "/etc/passwd"):
        with self.assertRaises(recipe.RecipeError):
          artifacts._inside(root, name)
      (root / "SHA256SUMS").write_text("0" * 64 + "  SHA256SUMS\n")
      with self.assertRaises(recipe.RecipeError):
        artifacts.verify_run(root)

  def test_e4b_geometry_rejected(self):
    with self.assertRaises(recipe.RecipeError):
      artifacts.validate_model_config({"text_config": {"num_hidden_layers": 42}})

  def test_input_manifest_drift_and_no_clobber(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      snapshot, data = root / "model", root / "data"
      snapshot.mkdir()
      data.mkdir()
      config = {"text_config": {
          "num_hidden_layers": 35, "hidden_size": 1536, "vocab_size": 262144,
          "num_attention_heads": 8, "num_key_value_heads": 1, "head_dim": 256,
          "hidden_size_per_layer_input": 256, "num_kv_shared_layers": 20,
          "intermediate_size": 6144,
          "global_head_dim": 512, "sliding_window": 512,
      }}
      (snapshot / "config.json").write_text(json.dumps(config))
      for name in ("tokenizer.json", "tokenizer_config.json", "model.safetensors"):
        (snapshot / name).write_bytes(b"synthetic input binding fixture only")
      for name in ("train.parquet", "test.parquet"):
        (data / name).write_bytes(b"synthetic parquet binding fixture only")
      manifest = root / "inputs.json"
      artifacts.seal_inputs(snapshot, data, "a" * 40, manifest)
      sha = artifacts.file_sha(manifest)
      artifacts.verify_inputs(manifest, sha, snapshot, data)
      with self.assertRaises(FileExistsError):
        artifacts.seal_inputs(snapshot, data, "a" * 40, manifest)
      (data / "train.parquet").write_bytes(b"changed")
      with self.assertRaises(recipe.RecipeError):
        artifacts.verify_inputs(manifest, sha, snapshot, data)


if __name__ == "__main__":
  unittest.main()
