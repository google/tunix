# Copyright 2025 Google LLC
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
"""Tests for utils."""

import os
from unittest import mock
from absl.testing import absltest
from tunix.oss import utils


class UtilsTest(absltest.TestCase):

  @mock.patch("time.sleep", return_value=None)
  @mock.patch("huggingface_hub.list_repo_files")
  def test_safe_list_repo_files_retries(self, mock_list, mock_sleep):
    mock_list.side_effect = [Exception("fail"), Exception("fail"), ["file1", "file2"]]
    files = utils.safe_list_repo_files("model_id")
    self.assertEqual(files, ["file1", "file2"])
    self.assertEqual(mock_list.call_count, 3)
    self.assertGreater(mock_sleep.call_count, 0)

  @mock.patch("time.sleep", return_value=None)
  @mock.patch("huggingface_hub.hf_hub_download")
  def test_safe_hf_hub_download_retries(self, mock_download, mock_sleep):
    mock_download.side_effect = [Exception("fail"), "path/to/file"]
    path = utils.safe_hf_hub_download("repo_id", "filename", "local_dir")
    self.assertEqual(path, "path/to/file")
    self.assertEqual(mock_download.call_count, 2)
    self.assertGreater(mock_sleep.call_count, 0)

  @mock.patch("time.sleep", return_value=None)
  @mock.patch("huggingface_hub.snapshot_download")
  def test_safe_snapshot_download_retries(self, mock_snapshot, mock_sleep):
    mock_snapshot.side_effect = [Exception("fail"), Exception("fail"), "path/to/dir"]
    path = utils.safe_snapshot_download("repo_id")
    self.assertEqual(path, "path/to/dir")
    self.assertEqual(mock_snapshot.call_count, 3)
    self.assertGreater(mock_sleep.call_count, 0)

  @mock.patch("tunix.oss.utils.safe_hf_hub_download")
  @mock.patch("tunix.oss.utils.safe_list_repo_files")
  @mock.patch("huggingface_hub.login")
  def test_hf_pipeline(self, mock_login, mock_list, mock_download):
    mock_list.return_value = ["file1", "original/file2"]
    # Mock os.environ to ensure HF_TOKEN is not present
    with mock.patch.dict("os.environ", {}, clear=True):
      res = utils.hf_pipeline("model_id", "download_path")

    mock_login.assert_called_once()
    mock_list.assert_called_once_with("model_id")
    # Should only download "file1", not "original/file2"
    mock_download.assert_called_once_with(
        repo_id="model_id",
        filename="file1",
        local_dir="download_path",
    )
    self.assertEqual(res, "download_path")


if __name__ == "__main__":
  absltest.main()
