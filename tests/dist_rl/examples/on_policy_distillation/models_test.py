# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pinned model preparation with real tiny local safetensors and tokenizers."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from safetensors.numpy import save_file
from tokenizers import models as token_models
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from tunix.dist_rl.examples.on_policy_distillation import models


class ModelsTest(unittest.TestCase):

  def setUp(self):
    self.temporary = tempfile.TemporaryDirectory()
    self.addCleanup(self.temporary.cleanup)
    self.directory = Path(self.temporary.name).resolve()
    self.snapshots = {}
    for role, repo, revision in (
        ("student", models.STUDENT_MODEL_ID, models.STUDENT_REVISION),
        ("teacher", models.TEACHER_MODEL_ID, models.TEACHER_REVISION),
    ):
      path = (
          self.directory
          / "hf_cache"
          / ("models--" + repo.replace("/", "--"))
          / "snapshots"
          / revision
      )
      path.mkdir(parents=True)
      backend = Tokenizer(
          token_models.WordLevel(
              {"<pad>": 0, "<eos>": 1, "a": 2}, unk_token="<pad>"
          )
      )
      tokenizer = PreTrainedTokenizerFast(
          tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>"
      )
      tokenizer.save_pretrained(path)
      (path / "config.json").write_text(
          json.dumps({"model_type": "qwen3", "max_position_embeddings": 40960})
      )
      save_file(
          {"model.embed_tokens.weight": np.ones((3, 2), np.float32)},
          str(path / "model-00001-of-00001.safetensors"),
      )
      (path / "model.safetensors.index.json").write_text(
          json.dumps({
              "weight_map": {
                  "model.embed_tokens.weight": (
                      "model-00001-of-00001.safetensors"
                  )
              }
          })
      )
      self.snapshots[role] = path

  def _prepare(self):
    with mock.patch.object(
        models,
        "snapshot_download",
        side_effect=[
            str(self.snapshots["student"]),
            str(self.snapshots["teacher"]),
        ],
    ) as download:
      destination = models.prepare_models(self.directory)
    return destination, download

  def test_prepare_pins_revisions_and_produces_real_loader_paths(self):
    with mock.patch.dict(
        "os.environ",
        {"STUDENT_MODEL_ID": "other/model", "TEACHER_REVISION": "main"},
    ):
      destination, download = self._prepare()
    self.assertEqual(destination, self.directory / "models.json")
    self.assertEqual(download.call_count, 2)
    for call, repo, revision in zip(
        download.call_args_list,
        (models.STUDENT_MODEL_ID, models.TEACHER_MODEL_ID),
        (models.STUDENT_REVISION, models.TEACHER_REVISION),
    ):
      self.assertEqual(call.kwargs["repo_id"], repo)
      self.assertEqual(call.kwargs["revision"], revision)
      self.assertEqual(call.kwargs["endpoint"], "https://huggingface.co")
      self.assertNotIn("*.bin", call.kwargs["allow_patterns"])
      self.assertNotIn("*.py", call.kwargs["allow_patterns"])
    with mock.patch.object(
        models,
        "snapshot_download",
        side_effect=AssertionError("load must not download"),
    ):
      prepared = models.load_manifest(destination)
    for role in ("student", "teacher"):
      self.assertEqual(
          prepared[role]["snapshot_path"], str(self.snapshots[role])
      )
      self.assertEqual(
          prepared[role]["tokenizer_path"], str(self.snapshots[role])
      )
      self.assertTrue(
          prepared[role]["declared_revision"].startswith("hf:Qwen/")
      )
      self.assertTrue(
          list(Path(prepared[role]["snapshot_path"]).glob("*.safetensors"))
      )
    self.assertEqual(len(prepared["tokenizer_fingerprint"]), 64)

  def test_tokenizer_loads_are_local_fast_and_disallow_remote_code(self):
    original = models.AutoTokenizer.from_pretrained
    with mock.patch.object(
        models.AutoTokenizer, "from_pretrained", wraps=original
    ) as load:
      self._prepare()
    self.assertEqual(load.call_count, 2)
    for call in load.call_args_list:
      self.assertEqual(
          call.kwargs,
          dict(use_fast=True, trust_remote_code=False, local_files_only=True),
      )

  def test_manifest_rejects_mutated_identity_and_fingerprint(self):
    destination, _ = self._prepare()
    original = json.loads(destination.read_text())
    for mutate in (
        lambda value: value["teacher"].update(revision="main"),
        lambda value: value["student"].update(repo_id="other/model"),
        lambda value: value.update(tokenizer_fingerprint="0" * 64),
        lambda value: value["student"].update(snapshot_path="relative/path"),
    ):
      value = json.loads(json.dumps(original))
      mutate(value)
      destination.write_text(json.dumps(value))
      with self.assertRaises(ValueError):
        models.load_manifest(destination)

  def test_missing_or_unindexed_shard_content_is_rejected(self):
    destination, _ = self._prepare()
    shard = self.snapshots["student"] / "model-00001-of-00001.safetensors"
    save_file({"wrong_tensor": np.ones((3, 2), np.float32)}, str(shard))
    with self.assertRaisesRegex(ValueError, "indexed tensors"):
      models.load_manifest(destination)
    shard.unlink()
    with self.assertRaisesRegex(ValueError, "Missing safetensors"):
      models.load_manifest(destination)

  def test_prompt_template_changes_and_extra_weight_shards_are_rejected(self):
    destination, _ = self._prepare()
    template = self.snapshots["student"] / "chat_template.jinja"
    template.write_text("Changed template")
    with self.assertRaisesRegex(ValueError, "artifacts differ"):
      models.load_manifest(destination)
    template.unlink()
    save_file(
        {"extra": np.ones(1, np.float32)},
        str(self.snapshots["student"] / "extra.safetensors"),
    )
    with self.assertRaisesRegex(ValueError, "outside its pinned index"):
      models.load_manifest(destination)

  def test_remote_code_and_unsafe_weight_map_fail_before_manifest_write(self):
    config = self.snapshots["teacher"] / "config.json"
    config.write_text(
        json.dumps(
            {"model_type": "qwen3", "auto_map": {"AutoModel": "remote.Code"}}
        )
    )
    with self.assertRaisesRegex(ValueError, "Remote model"):
      self._prepare()
    self.assertFalse((self.directory / "models.json").exists())
    config.write_text(json.dumps({"model_type": "qwen3"}))
    index = self.snapshots["teacher"] / "model.safetensors.index.json"
    index.write_text(
        json.dumps({"weight_map": {"x": "../outside.safetensors"}})
    )
    with self.assertRaisesRegex(ValueError, "shard path"):
      self._prepare()

  def test_incompatible_real_tokenizer_pair_is_rejected(self):
    path = self.snapshots["teacher"]
    tokenizer = models._load_tokenizer(str(path))
    tokenizer.add_tokens(["teacher-only"])
    tokenizer.save_pretrained(path)
    with self.assertRaisesRegex(ValueError, "tokenizer .* differ"):
      self._prepare()
    self.assertFalse((self.directory / "models.json").exists())


if __name__ == "__main__":
  unittest.main()
