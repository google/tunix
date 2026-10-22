# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import tempfile
from unittest import mock

from absl import flags
from absl.testing import absltest
import huggingface_hub
from tunix.oss import utils as oss_utils
from tunix.tests import test_common

flags.FLAGS.mark_as_parsed()


class SafeDownloadTest(absltest.TestCase):

  def test_safe_download_returns_existing_nonempty_file(self):
    local_dir = self.enterContext(tempfile.TemporaryDirectory())
    dst = os.path.join(local_dir, "model.safetensors")
    with open(dst, "w") as f:
      f.write("cached-weights")

    with mock.patch.object(
        huggingface_hub, "hf_hub_download"
    ) as mock_hf_download:
      result = test_common.safe_download(
          "org/repo", "model.safetensors", local_dir
      )

    self.assertEqual(result, dst)
    mock_hf_download.assert_not_called()

  def test_safe_download_links_from_hf_cache(self):
    cache_dir = self.enterContext(tempfile.TemporaryDirectory())
    local_dir = self.enterContext(tempfile.TemporaryDirectory())
    cached_file = os.path.join(cache_dir, "blob_123")
    with open(cached_file, "w") as f:
      f.write("downloaded-weights")

    with mock.patch.object(
        huggingface_hub, "hf_hub_download", return_value=cached_file
    ) as mock_hf_download:
      result = test_common.safe_download(
          "org/repo", "sub/model.safetensors", local_dir
      )

    expected_dst = os.path.join(local_dir, "sub/model.safetensors")
    self.assertEqual(result, expected_dst)
    mock_hf_download.assert_called_once_with(
        repo_id="org/repo", filename="sub/model.safetensors"
    )
    self.assertTrue(os.path.samefile(cached_file, expected_dst))

  def test_safe_download_handles_file_exists_race_without_copying(self):
    cache_dir = self.enterContext(tempfile.TemporaryDirectory())
    local_dir = self.enterContext(tempfile.TemporaryDirectory())
    cached_file = os.path.join(cache_dir, "blob_123")
    with open(cached_file, "w") as f:
      f.write("downloaded-weights")

    with (
        mock.patch.object(
            huggingface_hub, "hf_hub_download", return_value=cached_file
        ),
        mock.patch.object(os, "link", side_effect=FileExistsError),
        mock.patch.object(shutil, "copy2") as mock_copy2,
    ):
      result = test_common.safe_download(
          "org/repo", "model.safetensors", local_dir
      )

    self.assertEqual(result, os.path.join(local_dir, "model.safetensors"))
    mock_copy2.assert_not_called()

  def test_safe_download_falls_back_to_copy_on_oserror(self):
    cache_dir = self.enterContext(tempfile.TemporaryDirectory())
    local_dir = self.enterContext(tempfile.TemporaryDirectory())
    cached_file = os.path.join(cache_dir, "blob_123")
    with open(cached_file, "w") as f:
      f.write("downloaded-weights")

    with (
        mock.patch.object(
            huggingface_hub, "hf_hub_download", return_value=cached_file
        ),
        mock.patch.object(os, "link", side_effect=OSError("EXDEV")),
    ):
      result = test_common.safe_download(
          "org/repo", "model.safetensors", local_dir
      )

    self.assertEqual(result, os.path.join(local_dir, "model.safetensors"))
    with open(result) as f:
      self.assertEqual(f.read(), "downloaded-weights")

  def test_hf_pipeline_links_and_skips_existing_files(self):
    cache_dir = self.enterContext(tempfile.TemporaryDirectory())
    local_dir = self.enterContext(tempfile.TemporaryDirectory())
    cached_file = os.path.join(cache_dir, "blob_456")
    with open(cached_file, "w") as f:
      f.write("config-content")

    existing_file = os.path.join(local_dir, "already_there.json")
    with open(existing_file, "w") as f:
      f.write("existing")

    with (
        mock.patch.dict(os.environ, {"HF_TOKEN": "dummy"}),
        mock.patch.object(
            huggingface_hub,
            "list_repo_files",
            return_value=[
                "already_there.json",
                "config.json",
                "original/ignored.bin",
            ],
        ),
        mock.patch.object(
            huggingface_hub, "hf_hub_download", return_value=cached_file
        ) as mock_hf_download,
    ):
      result = oss_utils.hf_pipeline("org/repo", local_dir)

    self.assertEqual(result, local_dir)
    mock_hf_download.assert_called_once_with(
        repo_id="org/repo", filename="config.json"
    )
    self.assertTrue(
        os.path.samefile(cached_file, os.path.join(local_dir, "config.json"))
    )


if __name__ == "__main__":
  absltest.main()
