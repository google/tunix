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

import asyncio
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
from unittest import mock

from absl.testing import absltest
from google.api_core import exceptions as google_exceptions
from tunix.experimental.common import datatypes
from tunix.experimental.common import gcs_cache
from tunix.experimental.common import test_utils as mocks
from tunix.experimental.worker import rollout_worker


def _blob(name: str) -> mock.Mock:
  blob = mock.Mock()
  blob.name = name
  return blob


def _make_rollout_worker(
    worker_id: str = "rollout-0",
    jax_cache_config: gcs_cache.JaxCacheConfig | None = None,
) -> rollout_worker.RolloutWorker:
  return rollout_worker.RolloutWorker(
      worker_id=worker_id,
      sampler=mocks.MockBaseSamplerImpl(
          sampler_name="test_sampler", default_delay=0.0
      ),
      env_pool=mocks.MockEnvironmentPool(pool_size=4, default_delay=0.0),
      agent_factory=mocks.MockAgent,
      tokenizer=mocks.MockTokenizer(),
      chat_parser=mocks.MockChatParser(),
      jax_cache_config=jax_cache_config,
  )


class GcsCacheTest(absltest.TestCase):

  def setUp(self) -> None:
    super().setUp()
    self.enter_context(mock.patch.dict(os.environ, {}, clear=True))

  def _gcs_modules(
      self, mock_storage: mock.MagicMock, mock_tm: mock.MagicMock
  ) -> dict[str, mock.MagicMock]:
    mock_storage.transfer_manager = mock_tm
    mock_cloud = mock.MagicMock()
    mock_cloud.storage = mock_storage
    return {
        "google.cloud": mock_cloud,
        "google.cloud.storage": mock_storage,
        "google.cloud.storage.transfer_manager": mock_tm,
    }

  def test_parse_gcs_uri(self) -> None:
    bucket, prefix = gcs_cache._parse_gcs_uri("gs://my-bucket/path/to/cache")
    self.assertEqual(bucket, "my-bucket")
    self.assertEqual(prefix, "path/to/cache/")

    bucket, prefix = gcs_cache._parse_gcs_uri("gs://my-bucket")
    self.assertEqual(bucket, "my-bucket")
    self.assertEqual(prefix, "")

    with self.assertRaises(ValueError):
      gcs_cache._parse_gcs_uri("https://not-gcs.com")

  def test_ensure_jax_cache_env(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      target_dir = pathlib.Path(tmpdir) / "test_cache"
      res = gcs_cache.ensure_jax_cache_env(target_dir)
      self.assertEqual(res, target_dir)
      self.assertTrue(target_dir.is_dir())
      self.assertEqual(os.environ["JAX_COMPILATION_CACHE_DIR"], str(target_dir))
      self.assertEqual(os.environ["VLLM_XLA_CACHE_PATH"], str(target_dir))

  def test_ensure_jax_cache_env_updates_jax_config(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      target_dir = pathlib.Path(tmpdir) / "test_cache"
      mock_jax = mock.MagicMock()
      with mock.patch.dict(sys.modules, {"jax": mock_jax}):
        res = gcs_cache.ensure_jax_cache_env(target_dir)
        self.assertEqual(res, target_dir)
        mock_jax.config.update.assert_called_once_with(
            "jax_compilation_cache_dir", str(target_dir)
        )

  @mock.patch.object(gcs_cache, "download_cache", return_value=True)
  def test_restore_jax_cache_with_explicit_uri(
      self, mock_download: mock.MagicMock
  ) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      cache_dir = pathlib.Path(tmpdir) / "cache"
      success = gcs_cache.restore_jax_cache(
          gcs_uri="gs://bucket/rollout_cache",
          local_dir=cache_dir,
      )
      self.assertTrue(success)
      mock_download.assert_called_once_with(
          cache_dir, "gs://bucket/rollout_cache"
      )

  @mock.patch.object(gcs_cache, "download_cache", return_value=True)
  def test_restore_jax_cache_from_env(
      self, mock_download: mock.MagicMock
  ) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      cache_dir = pathlib.Path(tmpdir) / "cache"
      with mock.patch.dict(
          os.environ, {"ROLLOUT_JAX_CACHE_GCS_DIR": "gs://bucket/rollout_env"}
      ):
        success = gcs_cache.restore_jax_cache(local_dir=cache_dir)
        self.assertTrue(success)
        mock_download.assert_called_once_with(
            cache_dir, "gs://bucket/rollout_env"
        )

  def test_restore_jax_cache_disabled_does_not_set_env(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      cache_dir = pathlib.Path(tmpdir) / "cache"
      with mock.patch.dict(os.environ, {"DISABLE_JAX_CACHE": "yes"}):
        success = gcs_cache.restore_jax_cache(
            gcs_uri="gs://bucket/rollout_cache",
            local_dir=cache_dir,
        )
        self.assertFalse(success)
        self.assertFalse(cache_dir.exists())
        self.assertNotIn("JAX_COMPILATION_CACHE_DIR", os.environ)
        self.assertNotIn("VLLM_XLA_CACHE_PATH", os.environ)

  def test_save_jax_cache_disabled(self) -> None:
    with mock.patch.dict(os.environ, {"SAVE_JAX_CACHE": "false"}):
      success = gcs_cache.save_jax_cache(gcs_uri="gs://bucket/test")
      self.assertFalse(success)

  @mock.patch.object(gcs_cache, "upload_cache", return_value=True)
  def test_save_jax_cache_success(self, mock_upload: mock.MagicMock) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      cache_dir = pathlib.Path(tmpdir) / "cache"
      cache_dir.mkdir(parents=True, exist_ok=True)
      with mock.patch.dict(os.environ, {"SAVE_JAX_CACHE": "true"}):
        success = gcs_cache.save_jax_cache(
            gcs_uri="gs://bucket/saved_cache",
            local_dir=cache_dir,
        )
        self.assertTrue(success)
        mock_upload.assert_called_once_with(
            cache_dir, "gs://bucket/saved_cache"
        )

  def test_upload_cache_skip_if_exists(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      cache_dir = pathlib.Path(tmpdir) / "cache"
      cache_dir.mkdir()
      (cache_dir / "obj1").write_text("dummy")
      (cache_dir / "obj2").write_text("dummy")

      mock_storage = mock.MagicMock()
      mock_tm = mock.MagicMock()
      precondition_failed = google_exceptions.PreconditionFailed(
          "412 Precondition Failed"
      )
      mock_tm.upload_many_from_filenames.return_value = [
          precondition_failed,
          None,
      ]

      with mock.patch.dict(
          sys.modules, self._gcs_modules(mock_storage, mock_tm)
      ):
        success = gcs_cache.upload_cache(cache_dir, "gs://test-bucket/prefix")
        self.assertTrue(success)
        mock_tm.upload_many_from_filenames.assert_called_once_with(
            mock.ANY,
            mock.ANY,
            source_directory=str(cache_dir),
            blob_name_prefix="prefix/",
            skip_if_exists=True,
            worker_type=mock_tm.THREAD,
            max_workers=mock.ANY,
            deadline=int(gcs_cache.DEFAULT_SYNC_TIMEOUT_S),
            upload_kwargs={"timeout": gcs_cache.DEFAULT_SYNC_TIMEOUT_S},
        )

  def test_upload_cache_skips_transfer_when_all_in_gcs(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      cache_dir = pathlib.Path(tmpdir) / "cache"
      (cache_dir / "sub").mkdir(parents=True)
      (cache_dir / "obj1").write_text("dummy")
      (cache_dir / "sub" / "obj2").write_text("dummy")

      mock_storage = mock.MagicMock()
      mock_tm = mock.MagicMock()
      client = mock_storage.Client.return_value
      client.list_blobs.return_value = [
          _blob("prefix/obj1"),
          _blob("prefix/sub/obj2"),
      ]
      with mock.patch.dict(
          sys.modules, self._gcs_modules(mock_storage, mock_tm)
      ):
        self.assertTrue(gcs_cache.upload_cache(cache_dir, "gs://b/prefix"))
      mock_tm.upload_many_from_filenames.assert_not_called()

  def test_upload_cache_uploads_only_missing_with_threads(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      cache_dir = pathlib.Path(tmpdir) / "cache"
      cache_dir.mkdir()
      (cache_dir / "obj1").write_text("dummy")
      (cache_dir / "obj2").write_text("dummy")

      mock_storage = mock.MagicMock()
      mock_tm = mock.MagicMock()
      client = mock_storage.Client.return_value
      client.list_blobs.return_value = [_blob("prefix/obj1")]
      mock_tm.upload_many_from_filenames.return_value = [None]
      with mock.patch.dict(
          sys.modules, self._gcs_modules(mock_storage, mock_tm)
      ):
        self.assertTrue(gcs_cache.upload_cache(cache_dir, "gs://b/prefix"))
      args, kwargs = mock_tm.upload_many_from_filenames.call_args
      self.assertEqual(args[1], ["obj2"])
      self.assertIs(kwargs["worker_type"], mock_tm.THREAD)

  def test_download_cache_uses_thread_worker_type(self) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      cache_dir = pathlib.Path(tmpdir) / "cache"
      mock_storage = mock.MagicMock()
      mock_tm = mock.MagicMock()
      client = mock_storage.Client.return_value
      client.list_blobs.return_value = [_blob("prefix/obj1")]
      mock_tm.download_many_to_path.return_value = [None]
      with mock.patch.dict(
          sys.modules, self._gcs_modules(mock_storage, mock_tm)
      ):
        self.assertTrue(
            gcs_cache.download_cache(cache_dir, "gs://test-bucket/prefix")
        )
      mock_tm.download_many_to_path.assert_called_once_with(
          mock.ANY,
          ["obj1"],
          destination_directory=str(cache_dir),
          blob_name_prefix="prefix/",
          worker_type=mock_tm.THREAD,
          max_workers=mock.ANY,
          deadline=int(gcs_cache.DEFAULT_SYNC_TIMEOUT_S),
          download_kwargs={"timeout": gcs_cache.DEFAULT_SYNC_TIMEOUT_S},
      )

  def test_is_primary_rollout_worker(self) -> None:
    self.assertTrue(gcs_cache.is_primary_rollout_worker("rollout"))
    self.assertTrue(gcs_cache.is_primary_rollout_worker("linchai-roll"))
    self.assertTrue(gcs_cache.is_primary_rollout_worker("rollout-0"))
    self.assertTrue(gcs_cache.is_primary_rollout_worker("linchai-roll-0"))
    self.assertTrue(gcs_cache.is_primary_rollout_worker("rollout_worker_42"))
    self.assertFalse(gcs_cache.is_primary_rollout_worker("rollout-1"))
    self.assertFalse(gcs_cache.is_primary_rollout_worker("linchai-roll-2"))

  @mock.patch.object(gcs_cache, "save_jax_cache", return_value=True)
  def test_jax_cache_syncer_primary_only(
      self, mock_save: mock.MagicMock
  ) -> None:
    cfg = gcs_cache.JaxCacheConfig(
        save_jax_cache=True,
        rollout_jax_cache_gcs_dir="gs://bucket/rollout_cache",
        local_dir=pathlib.Path("/tmp/cache"),
    )
    syncer_0 = gcs_cache.JaxCacheSyncer(worker_id="rollout-0", config=cfg)
    syncer_1 = gcs_cache.JaxCacheSyncer(worker_id="rollout-1", config=cfg)

    fut_0 = syncer_0.sync(wait=True)
    fut_1 = syncer_1.sync(wait=True)

    self.assertIsNotNone(fut_0)
    self.assertIsNone(fut_1)
    mock_save.assert_called_once_with(
        gcs_uri="gs://bucket/rollout_cache",
        local_dir=pathlib.Path("/tmp/cache"),
    )

  def test_jax_cache_syncer_bounds_hung_upload(self) -> None:
    cfg = gcs_cache.JaxCacheConfig(
        save_jax_cache=True,
        rollout_jax_cache_gcs_dir="gs://bucket/rollout_cache",
        sync_timeout_s=0.2,
    )
    syncer_0 = gcs_cache.JaxCacheSyncer(worker_id="rollout-0", config=cfg)
    release = threading.Event()
    hung = lambda *a, **k: release.wait()
    with mock.patch.object(
        gcs_cache, "save_jax_cache", side_effect=hung
    ) as mock_save:
      try:
        start = time.monotonic()
        syncer_0.sync(wait=True)
        self.assertLess(time.monotonic() - start, 5.0)
        threads = {t.name: t for t in threading.enumerate()}
        self.assertTrue(threads["jax-cache-upload-rollout-0"].daemon)
      finally:
        release.set()
      mock_save.assert_called_once()

  @mock.patch.object(gcs_cache, "save_jax_cache", return_value=True)
  def test_rollout_worker_start_and_first_generate_trigger_autonomous_sync(
      self, mock_save: mock.MagicMock
  ) -> None:
    cfg = gcs_cache.JaxCacheConfig(
        save_jax_cache=True,
        rollout_jax_cache_gcs_dir="gs://bucket/rollout_cache",
        local_dir=pathlib.Path("/tmp/cache"),
    )
    worker = _make_rollout_worker(worker_id="rollout-0", jax_cache_config=cfg)

    async def _run() -> None:
      await worker.start()
      req1 = datatypes.RolloutRequest(
          request_id="req_1",
          prompt="What is 2+2?",
          prompt_id="p1",
          group_index=0,
          generation_kwargs={"max_generation_steps": 16},
      )
      req2 = datatypes.RolloutRequest(
          request_id="req_2",
          prompt="What is 3+3?",
          prompt_id="p2",
          group_index=0,
          generation_kwargs={"max_generation_steps": 16},
      )
      await worker.generate(requests=req1)
      await worker.generate(requests=req2)
      worker.stop()

    asyncio.run(_run())
    # Upload triggers twice: once on start() and once after the first generate().
    self.assertEqual(mock_save.call_count, 2)
    for call in mock_save.call_args_list:
      self.assertEqual(
          call,
          mock.call(
              gcs_uri="gs://bucket/rollout_cache",
              local_dir=pathlib.Path("/tmp/cache"),
          ),
      )

  def test_classify_tpu_hardware(self) -> None:
    self.assertEqual(gcs_cache.classify_tpu_hardware("tpuv5e:2x4"), "v5e")
    self.assertEqual(
        gcs_cache.classify_tpu_hardware("tpu-v5-lite-podslice:2x4"), "v5e"
    )
    self.assertEqual(gcs_cache.classify_tpu_hardware("tpuv5:2x2x1"), "v5p")
    self.assertEqual(gcs_cache.classify_tpu_hardware("tpuv5p:2x2x2"), "v5p")
    self.assertEqual(gcs_cache.classify_tpu_hardware("tpuv6e:2x4"), "v6e")
    self.assertEqual(gcs_cache.classify_tpu_hardware("tpu7x:2x2x1"), "v7x")

  def test_derive_rollout_cache_uri_and_fingerprint(self) -> None:
    params_base = gcs_cache.RolloutCacheKeyParams(
        model_name="qwen3.5-35b-a3b",
        sampler="vllm",
        max_prompt_length=512,
        max_response_length=1024,
        tpu_slice="tpuv5:2x2x1",
        mesh_tp=1,
        mesh_fsdp=1,
        mesh_expert=8,
    )
    uri1 = gcs_cache.derive_rollout_cache_uri(
        bucket="gs://test-bucket", params=params_base, env={}
    )
    self.assertIsNotNone(uri1)
    assert uri1 is not None
    self.assertTrue(
        uri1.startswith(
            "gs://test-bucket/jax_cache/v5p/qwen3.5-35b-a3b/rollout_2x2x1_ep8_tp1_"
        )
    )

    # Verify that changing quantization or max_response_length changes the hash.
    params_fp8 = gcs_cache.RolloutCacheKeyParams(
        model_name="qwen3.5-35b-a3b",
        sampler="vllm",
        max_prompt_length=512,
        max_response_length=1024,
        tpu_slice="tpuv5:2x2x1",
        mesh_tp=1,
        mesh_fsdp=1,
        mesh_expert=8,
        rollout_fp8="true",
    )
    uri2 = gcs_cache.derive_rollout_cache_uri(
        bucket="gs://test-bucket", params=params_fp8, env={}
    )
    self.assertNotEqual(uri1, uri2)

    params_longer = gcs_cache.RolloutCacheKeyParams(
        model_name="qwen3.5-35b-a3b",
        sampler="vllm",
        max_prompt_length=512,
        max_response_length=2048,
        tpu_slice="tpuv5:2x2x1",
        mesh_tp=1,
        mesh_fsdp=1,
        mesh_expert=8,
    )
    uri3 = gcs_cache.derive_rollout_cache_uri(
        bucket="gs://test-bucket", params=params_longer, env={}
    )
    self.assertNotEqual(uri1, uri3)

    # Verify JAX_CACHE_GCS_DIR acts as a base prefix and appends rollout
    # subdirectory + digest.
    uri_base_dir = gcs_cache.derive_rollout_cache_uri(
        params=params_base,
        env={"JAX_CACHE_GCS_DIR": "gs://test-bucket/custom_root/"},
    )
    self.assertIsNotNone(uri_base_dir)
    assert uri_base_dir is not None
    self.assertTrue(
        uri_base_dir.startswith(
            "gs://test-bucket/custom_root/v5p/qwen3.5-35b-a3b/rollout_2x2x1_ep8_tp1_"
        )
    )

    # Verify MAXTEXT_OUTPUT_DIR preserves tenant subpath prefixes.
    uri_tenant = gcs_cache.derive_rollout_cache_uri(
        params=params_base,
        env={"MAXTEXT_OUTPUT_DIR": "gs://shared-bucket/tenant_a/exp1/"},
    )
    self.assertIsNotNone(uri_tenant)
    assert uri_tenant is not None
    self.assertTrue(
        uri_tenant.startswith(
            "gs://shared-bucket/tenant_a/exp1/jax_cache/v5p/qwen3.5-35b-a3b/rollout_2x2x1_ep8_tp1_"
        )
    )

    # Verify VLLM_SERVER_MODE unset vs explicit "true" produce identical
    # fingerprints.
    params_server_unset = gcs_cache.RolloutCacheKeyParams.from_env({})
    params_server_true = gcs_cache.RolloutCacheKeyParams.from_env(
        {"VLLM_SERVER_MODE": "true"}
    )
    self.assertEqual(
        params_server_unset.cache_fingerprint(),
        params_server_true.cache_fingerprint(),
    )

  def test_jax_cache_syncer_derives_isolated_uri(self) -> None:
    syncer = gcs_cache.JaxCacheSyncer(
        worker_id="rollout-local-0",
        config=gcs_cache.JaxCacheConfig(
            save_jax_cache=True,
            jax_cache_gcs_dir="gs://bucket/base_cache",
        ),
    )
    with mock.patch.object(
        gcs_cache, "save_jax_cache", return_value=True
    ) as mock_upload:
      fut = syncer.sync(wait=True)
      self.assertIsNotNone(fut)
      mock_upload.assert_called_once()
      called_uri = mock_upload.call_args.kwargs["gcs_uri"]
      self.assertTrue(
          called_uri.startswith(
              "gs://bucket/base_cache/v5e/model/rollout_4x4_ep1_tp1_"
          )
      )

  def test_jax_cache_syncer_wait_false_does_not_block_on_prior(
      self,
  ) -> None:
    syncer = gcs_cache.JaxCacheSyncer(
        worker_id="rollout-0",
        config=gcs_cache.JaxCacheConfig(
            save_jax_cache=True,
            rollout_jax_cache_gcs_dir="gs://bucket/rollout_cache",
        ),
    )
    first_entered = threading.Event()
    release_first = threading.Event()
    call_count = 0

    def _save(**kwargs: object) -> bool:
      del kwargs
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        first_entered.set()
        release_first.wait(timeout=5.0)
      return True

    with mock.patch.object(gcs_cache, "save_jax_cache", side_effect=_save):
      f1 = syncer.sync(wait=False)
      self.assertIsNotNone(f1)
      self.assertTrue(first_entered.wait(timeout=2.0))

      start = time.monotonic()
      f2 = syncer.sync(wait=False)
      elapsed = time.monotonic() - start
      self.assertIsNotNone(f2)
      self.assertLess(elapsed, 0.5)

      release_first.set()
      syncer.wait()
      assert f1 is not None and f2 is not None
      self.assertTrue(f1.result(timeout=2.0))
      self.assertTrue(f2.result(timeout=2.0))
      self.assertEqual(call_count, 2)

  @mock.patch.object(subprocess, "run")
  def test_run_cli_rsync_handles_timeout(
      self, mock_run: mock.MagicMock
  ) -> None:
    mock_run.side_effect = subprocess.TimeoutExpired(cmd=["gsutil"], timeout=1)
    self.assertFalse(
        gcs_cache._run_cli_rsync(["gsutil", "-m", "rsync"], timeout_s=1.0)
    )
    mock_run.assert_called_once_with(
        ["gsutil", "-m", "rsync"], check=False, timeout=1.0
    )

  @mock.patch.object(gcs_cache, "download_cache", return_value=True)
  def test_main_download_derives_uri_when_empty(
      self, mock_download: mock.MagicMock
  ) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
      with mock.patch.dict(
          os.environ,
          {
              "JAX_CACHE_BUCKET": "gs://derived-bucket",
              "ROLLOUT_TPU_SLICE": "tpuv5e:2x2",
              "MODEL_NAME": "qwen3-0.6b",
          },
      ):
        with self.assertRaises(SystemExit) as cm:
          gcs_cache.main(["download", tmpdir, ""])
        self.assertEqual(cm.exception.code, 0)
        self.assertEqual(mock_download.call_count, 1)
        called_dir, called_uri = mock_download.call_args.args[:2]
        self.assertEqual(called_dir, pathlib.Path(tmpdir))
        self.assertTrue(
            called_uri.startswith(
                "gs://derived-bucket/jax_cache/v5e/qwen3-0.6b/rollout_2x2_"
            )
        )


if __name__ == "__main__":
  absltest.main()
