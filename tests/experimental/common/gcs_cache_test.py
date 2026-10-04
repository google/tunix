# Copyright 2026 Google LLC
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

"""Tests for GCS JAX compilation cache utilities."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import mock

from absl.testing import absltest
from tunix.experimental.common import datatypes
from tunix.experimental.common import gcs_cache
from tunix.experimental.orchestrator import orchestrator
from tunix.experimental.worker import abstract_worker
from tunix.experimental.worker import remote_execution


class GcsCacheTest(absltest.TestCase):

  def test_parse_gcs_uri(self):
    bucket, prefix = gcs_cache._parse_gcs_uri("gs://my-bucket/path/to/cache")
    self.assertEqual(bucket, "my-bucket")
    self.assertEqual(prefix, "path/to/cache/")

    bucket, prefix = gcs_cache._parse_gcs_uri("gs://my-bucket")
    self.assertEqual(bucket, "my-bucket")
    self.assertEqual(prefix, "")

    with self.assertRaises(ValueError):
      gcs_cache._parse_gcs_uri("https://not-gcs.com")

  def test_ensure_jax_cache_env(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      target_dir = Path(tmpdir) / "test_cache"
      res = gcs_cache.ensure_jax_cache_env(target_dir)
      self.assertEqual(res, target_dir)
      self.assertTrue(target_dir.is_dir())
      self.assertEqual(os.environ.get("JAX_COMPILATION_CACHE_DIR"), str(target_dir))
      self.assertEqual(os.environ.get("VLLM_XLA_CACHE_PATH"), str(target_dir))

  def test_ensure_jax_cache_env_updates_jax_config(self):
    import sys
    with tempfile.TemporaryDirectory() as tmpdir:
      target_dir = Path(tmpdir) / "test_cache"
      mock_jax = mock.MagicMock()
      with mock.patch.dict(sys.modules, {"jax": mock_jax}):
        res = gcs_cache.ensure_jax_cache_env(target_dir)
        self.assertEqual(res, target_dir)
        mock_jax.config.update.assert_called_once_with(
            "jax_compilation_cache_dir", str(target_dir)
        )

  @mock.patch.object(gcs_cache, "download_cache", return_value=True)
  def test_restore_jax_cache_with_explicit_uri(self, mock_download):
    with tempfile.TemporaryDirectory() as tmpdir:
      cache_dir = Path(tmpdir) / "cache"
      success = gcs_cache.restore_jax_cache(
          gcs_uri="gs://bucket/rollout_cache",
          local_dir=cache_dir,
      )
      self.assertTrue(success)
      mock_download.assert_called_once_with(cache_dir, "gs://bucket/rollout_cache")

  @mock.patch.object(gcs_cache, "download_cache", return_value=True)
  def test_restore_jax_cache_from_env_role(self, mock_download):
    with tempfile.TemporaryDirectory() as tmpdir:
      cache_dir = Path(tmpdir) / "cache"
      with mock.patch.dict(os.environ, {"ROLLOUT_JAX_CACHE_GCS_DIR": "gs://bucket/rollout_env"}):
        success = gcs_cache.restore_jax_cache(local_dir=cache_dir, role="rollout")
        self.assertTrue(success)
        mock_download.assert_called_once_with(cache_dir, "gs://bucket/rollout_env")

  def test_save_jax_cache_disabled(self):
    with mock.patch.dict(os.environ, {"SAVE_JAX_CACHE": "false"}):
      success = gcs_cache.save_jax_cache(gcs_uri="gs://bucket/test")
      self.assertFalse(success)

  @mock.patch.object(gcs_cache, "upload_cache", return_value=True)
  def test_save_jax_cache_success(self, mock_upload):
    with tempfile.TemporaryDirectory() as tmpdir:
      cache_dir = Path(tmpdir) / "cache"
      cache_dir.mkdir(parents=True, exist_ok=True)
      with mock.patch.dict(os.environ, {"SAVE_JAX_CACHE": "true"}):
        success = gcs_cache.save_jax_cache(
            gcs_uri="gs://bucket/saved_cache",
            local_dir=cache_dir,
        )
        self.assertTrue(success)
        mock_upload.assert_called_once_with(cache_dir, "gs://bucket/saved_cache")

  def test_upload_cache_skip_if_exists(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cache_dir = Path(tmpdir) / "cache"
      cache_dir.mkdir()
      (cache_dir / "obj1").write_text("dummy")
      (cache_dir / "obj2").write_text("dummy")

      mock_storage = mock.MagicMock()
      mock_tm = mock.MagicMock()
      mock_storage.transfer_manager = mock_tm
      mock_exceptions = mock.MagicMock()

      mock_cloud = mock.MagicMock()
      mock_cloud.storage = mock_storage

      mock_api_core = mock.MagicMock()
      mock_api_core.exceptions = mock_exceptions

      PreconditionFailed = type("PreconditionFailed", (Exception,), {})
      precondition_failed = PreconditionFailed("412 Precondition Failed")
      mock_tm.upload_many_from_filenames.return_value = [
          precondition_failed,
          None,
      ]

      with mock.patch.dict(
          sys.modules,
          {
              "google.cloud": mock_cloud,
              "google.cloud.storage": mock_storage,
              "google.cloud.storage.transfer_manager": mock_tm,
              "google.api_core": mock_api_core,
              "google.api_core.exceptions": mock_exceptions,
          },
      ):
        success = gcs_cache.upload_cache(cache_dir, "gs://test-bucket/prefix")
        self.assertTrue(success)
        mock_tm.upload_many_from_filenames.assert_called_once_with(
            mock.ANY,
            mock.ANY,
            source_directory=str(cache_dir),
            blob_name_prefix="prefix/",
            skip_if_exists=True,
            max_workers=mock.ANY,
            worker_type=mock_tm.THREAD,
        )

  def test_download_cache_uses_thread_worker_type(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cache_dir = Path(tmpdir) / "cache"

      mock_storage = mock.MagicMock()
      mock_tm = mock.MagicMock()
      mock_storage.transfer_manager = mock_tm
      mock_cloud = mock.MagicMock()
      mock_cloud.storage = mock_storage

      mock_blob = mock.MagicMock()
      mock_blob.name = "prefix/obj1"
      mock_storage.Client.return_value.list_blobs.return_value = [mock_blob]
      mock_tm.download_many_to_path.return_value = [None]

      with mock.patch.dict(
          sys.modules,
          {
              "google.cloud": mock_cloud,
              "google.cloud.storage": mock_storage,
              "google.cloud.storage.transfer_manager": mock_tm,
          },
      ):
        success = gcs_cache.download_cache(cache_dir, "gs://test-bucket/prefix")
        self.assertTrue(success)
        mock_tm.download_many_to_path.assert_called_once_with(
            mock.ANY,
            ["obj1"],
            destination_directory=str(cache_dir),
            blob_name_prefix="prefix/",
            max_workers=mock.ANY,
            worker_type=mock_tm.THREAD,
        )

  def test_upload_cache_skips_downloaded_files(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cache_dir = Path(tmpdir) / "cache"
      cache_dir.mkdir()
      (cache_dir / "obj1").write_text("dummy")

      mock_storage = mock.MagicMock()
      mock_tm = mock.MagicMock()
      mock_storage.transfer_manager = mock_tm
      mock_cloud = mock.MagicMock()
      mock_cloud.storage = mock_storage

      mock_blob = mock.MagicMock()
      mock_blob.name = "prefix/obj1"
      mock_storage.Client.return_value.list_blobs.return_value = [mock_blob]
      mock_tm.download_many_to_path.return_value = [None]

      with mock.patch.dict(
          sys.modules,
          {
              "google.cloud": mock_cloud,
              "google.cloud.storage": mock_storage,
              "google.cloud.storage.transfer_manager": mock_tm,
          },
      ):
        self.assertTrue(
            gcs_cache.download_cache(cache_dir, "gs://test-bucket/prefix")
        )
        mock_storage.Client.reset_mock()
        self.assertTrue(
            gcs_cache.upload_cache(cache_dir, "gs://test-bucket/prefix")
        )
        mock_tm.upload_many_from_filenames.assert_not_called()
        mock_storage.Client.assert_not_called()

  def test_upload_cache_uploads_only_new_files_after_download(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cache_dir = Path(tmpdir) / "cache"
      cache_dir.mkdir()
      (cache_dir / "obj1").write_text("cached")

      mock_storage = mock.MagicMock()
      mock_tm = mock.MagicMock()
      mock_storage.transfer_manager = mock_tm
      mock_exceptions = mock.MagicMock()
      mock_cloud = mock.MagicMock()
      mock_cloud.storage = mock_storage
      mock_api_core = mock.MagicMock()
      mock_api_core.exceptions = mock_exceptions

      mock_blob = mock.MagicMock()
      mock_blob.name = "prefix/obj1"
      mock_storage.Client.return_value.list_blobs.return_value = [mock_blob]
      mock_tm.download_many_to_path.return_value = [None]
      mock_tm.upload_many_from_filenames.return_value = [None]

      with mock.patch.dict(
          sys.modules,
          {
              "google.cloud": mock_cloud,
              "google.cloud.storage": mock_storage,
              "google.cloud.storage.transfer_manager": mock_tm,
              "google.api_core": mock_api_core,
              "google.api_core.exceptions": mock_exceptions,
          },
      ):
        self.assertTrue(
            gcs_cache.download_cache(cache_dir, "gs://test-bucket/prefix")
        )
        # Simulate a newly compiled artifact written after cache restore.
        (cache_dir / "obj2").write_text("newly_compiled")
        self.assertTrue(
            gcs_cache.upload_cache(cache_dir, "gs://test-bucket/prefix")
        )
        mock_tm.upload_many_from_filenames.assert_called_once_with(
            mock.ANY,
            ["obj2"],
            source_directory=str(cache_dir),
            blob_name_prefix="prefix/",
            skip_if_exists=True,
            max_workers=mock.ANY,
            worker_type=mock_tm.THREAD,
        )

  def test_upload_cache_skips_files_already_in_gcs(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      cache_dir = Path(tmpdir) / "cache"
      cache_dir.mkdir()
      (cache_dir / "obj1").write_text("dummy")

      mock_storage = mock.MagicMock()
      mock_tm = mock.MagicMock()
      mock_storage.transfer_manager = mock_tm
      mock_exceptions = mock.MagicMock()
      mock_cloud = mock.MagicMock()
      mock_cloud.storage = mock_storage
      mock_api_core = mock.MagicMock()
      mock_api_core.exceptions = mock_exceptions

      mock_blob = mock.MagicMock()
      mock_blob.name = "prefix/obj1"
      mock_storage.Client.return_value.list_blobs.return_value = [mock_blob]

      with mock.patch.dict(
          sys.modules,
          {
              "google.cloud": mock_cloud,
              "google.cloud.storage": mock_storage,
              "google.cloud.storage.transfer_manager": mock_tm,
              "google.api_core": mock_api_core,
              "google.api_core.exceptions": mock_exceptions,
          },
      ):
        self.assertTrue(
            gcs_cache.upload_cache(cache_dir, "gs://test-bucket/prefix")
        )
        mock_tm.upload_many_from_filenames.assert_not_called()

  def test_orchestrator_sync_jax_cache(self):
    orch = orchestrator.ClusterOrchestrator(
        jax_cache_config={
            "save_jax_cache": True,
            "rollout_jax_cache_gcs_dir": "gs://bucket/orch_rollout",
        }
    )
    mock_handle = mock.MagicMock(spec=remote_execution.ActorHandle)
    orch.register_worker_handle(
        "rollout-0",
        roles=[datatypes.Role.ROLLOUT],
        handle=mock_handle,
    )
    orch.sync_jax_cache()
    mock_handle.submit.assert_called_once_with(
        "upload_jax_cache", gcs_uri="gs://bucket/orch_rollout"
    )

  def test_orchestrator_sync_jax_cache_rollout_only(self):
    orch = orchestrator.ClusterOrchestrator(
        jax_cache_config={
            "save_jax_cache": True,
            "rollout_jax_cache_gcs_dir": "gs://bucket/orch_rollout",
        }
    )
    mock_rollout = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_trainer = mock.MagicMock(spec=remote_execution.ActorHandle)
    orch.register_worker_handle(
        "rollout-0",
        roles=[datatypes.Role.ROLLOUT],
        handle=mock_rollout,
    )
    orch.register_worker_handle(
        "trainer-0",
        roles=[datatypes.Role.ACTOR],
        handle=mock_trainer,
    )
    orch.sync_jax_cache()
    mock_rollout.submit.assert_called_once_with(
        "upload_jax_cache", gcs_uri="gs://bucket/orch_rollout"
    )
    mock_trainer.submit.assert_not_called()

  def test_orchestrator_sync_jax_cache_single_worker(self):
    orch = orchestrator.ClusterOrchestrator(
        jax_cache_config={
            "save_jax_cache": True,
            "rollout_jax_cache_gcs_dir": "gs://bucket/orch_rollout",
        }
    )
    mock_rollout_0 = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_rollout_1 = mock.MagicMock(spec=remote_execution.ActorHandle)
    orch.register_worker_handle(
        "rollout-0",
        roles=[datatypes.Role.ROLLOUT],
        handle=mock_rollout_0,
    )
    orch.register_worker_handle(
        "rollout-1",
        roles=[datatypes.Role.ROLLOUT],
        handle=mock_rollout_1,
    )
    orch.sync_jax_cache()
    mock_rollout_0.submit.assert_called_once_with(
        "upload_jax_cache", gcs_uri="gs://bucket/orch_rollout"
    )
    mock_rollout_1.submit.assert_not_called()

  def test_orchestrator_sync_jax_cache_fallback(self):
    orch = orchestrator.ClusterOrchestrator(
        jax_cache_config={
            "save_jax_cache": True,
            "rollout_jax_cache_gcs_dir": "gs://bucket/orch_rollout",
        }
    )
    mock_rollout_0 = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_rollout_0.submit.side_effect = RuntimeError("GCS upload failed")

    mock_rollout_1 = mock.MagicMock(spec=remote_execution.ActorHandle)
    mock_rollout_1.submit.return_value = 5

    orch.register_worker_handle(
        "rollout-0",
        roles=[datatypes.Role.ROLLOUT],
        handle=mock_rollout_0,
    )
    orch.register_worker_handle(
        "rollout-1",
        roles=[datatypes.Role.ROLLOUT],
        handle=mock_rollout_1,
    )
    orch.sync_jax_cache()
    mock_rollout_0.submit.assert_called_once_with(
        "upload_jax_cache", gcs_uri="gs://bucket/orch_rollout"
    )
    mock_rollout_1.submit.assert_called_once_with(
        "upload_jax_cache", gcs_uri="gs://bucket/orch_rollout"
    )

  def test_jax_cache_config_shell_hash(self):
    script = Path(__file__).resolve().parents[3] / "tunix/experimental/examples/common/jax_cache_config.sh"
    cmd = (
        'export BUCKET="gs://test-bucket"; '
        'export ROLLOUT_TPU_SLICE="tpuv5:2x2x1"; '
        'export ROLLOUT_MESH_EXPERT=8; '
        'export ROLLOUT_MESH_TP=1; '
        'export MODEL_NAME="qwen3.5-35b-a3b"; '
        f'source "{script}" && echo "$ROLLOUT_JAX_CACHE_GCS_DIR"'
    )
    res1 = subprocess.check_output(["bash", "-c", cmd], text=True).strip()
    self.assertTrue(res1.startswith("gs://test-bucket/jax_cache/v5p/qwen3.5-35b-a3b/rollout_2x2x1_ep8_tp1_"))

    # Verify that changing quantization changes the fingerprint hash
    cmd_fp8 = f"export ROLLOUT_FP8=true; {cmd}"
    res2 = subprocess.check_output(["bash", "-c", cmd_fp8], text=True).strip()
    self.assertTrue(res2.startswith("gs://test-bucket/jax_cache/v5p/qwen3.5-35b-a3b/rollout_2x2x1_ep8_tp1_"))
    self.assertNotEqual(res1, res2)


if __name__ == "__main__":
  absltest.main()
