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

"""Unit tests for `GCSSynchronizer` and `TPUWorker` hooks."""

from __future__ import annotations

import asyncio
import dataclasses
import os
from typing import Any
from unittest import mock

from absl import logging
from absl.testing import absltest
from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.weight_sync import weight_sync
from tunix.experimental.weight_sync import gcs_synchronizer
from tunix.tests import test_common

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _shard_toy_model(
    model: test_common.ToyTransformer,
    mesh: jax.sharding.Mesh,
    axis_name: str,
    *,
    col_major: bool = False,
) -> test_common.ToyTransformer:
  """Shards `model` across `mesh` on rows (`col_major=False`) or columns."""
  model(jnp.zeros((1, 4), dtype=jnp.int32), jnp.zeros((1, 4), dtype=jnp.int32))
  state = nnx.state(model)
  axis_size = int(mesh.shape[axis_name])

  def _put_leaf(x: Any) -> Any:
    if not isinstance(x, jax.Array):
      return x
    if col_major and x.ndim >= 2 and x.shape[1] % axis_size == 0:
      spec = jax.sharding.PartitionSpec(None, axis_name)
    elif not col_major and x.ndim >= 2 and x.shape[0] % axis_size == 0:
      spec = jax.sharding.PartitionSpec(axis_name, None)
    elif x.ndim >= 1 and x.shape[0] % axis_size == 0:
      spec = jax.sharding.PartitionSpec(axis_name)
    else:
      spec = jax.sharding.PartitionSpec()
    return jax.device_put(x, jax.sharding.NamedSharding(mesh, spec))

  nnx.update(model, jax.tree.map(_put_leaf, state))
  return model


class _NnxRunner:

  def __init__(self, state: nnx.State) -> None:
    self.state: Any = state
    self.state_leaves: Any = tuple(jax.tree_util.tree_leaves(state))

  def refresh_state_leaves(self) -> None:
    self.state_leaves = tuple(jax.tree_util.tree_leaves(self.state))


class _FlatDictRunner:

  def __init__(self, state: dict[str, jax.Array]) -> None:
    self.state: dict[str, jax.Array] = state
    self.state_leaves: dict[str, jax.Array] = state


class _FakeTPUWorker:

  def __init__(self, runner: _NnxRunner, rank: int = 0) -> None:
    self.rank: int = rank
    self.model_runner: _NnxRunner = runner

  def get_weights_state(self) -> Any:
    return self.model_runner.state

  def refresh_model_state_leaves(self) -> None:
    self.model_runner.refresh_state_leaves()


class GCSSynchronizerTest(absltest.TestCase):

  def test_config_validation_env_default_and_cleanup_on_start(self):
    with self.assertRaisesRegex(ValueError, "staging_dir must not be empty"):
      gcs_synchronizer.GCSSynchronizerConfig(staging_dir="   ")
    with self.assertRaisesRegex(ValueError, "max_to_keep"):
      gcs_synchronizer.GCSSynchronizerConfig(max_to_keep=-1)
    with self.assertRaisesRegex(ValueError, "max_to_keep"):
      gcs_synchronizer.GCSSynchronizerConfig(max_to_keep=True)
    with self.assertRaisesRegex(ValueError, "save_concurrent_gb"):
      gcs_synchronizer.GCSSynchronizerConfig(save_concurrent_gb=0)
    with self.assertRaisesRegex(ValueError, "cleanup_on_start=True requires"):
      gcs_synchronizer.GCSSynchronizerConfig(cleanup_on_start=True)

    staging = epath.Path(self.create_tempdir().full_path)
    (staging / "step_0_stale").mkdir()
    (staging / "keep_marker.txt").write_text("ok")
    prev_env = os.environ.get("WEIGHT_SYNC_GCS_DIR")
    os.environ["WEIGHT_SYNC_GCS_DIR"] = str(staging)
    try:
      cfg = gcs_synchronizer.GCSSynchronizerConfig(cleanup_on_start=True)
      self.assertEqual(cfg.staging_dir, str(staging))
      gcs_synchronizer.GCSSynchronizer(job_name="trainer", config=cfg)
    finally:
      if prev_env is None:
        os.environ.pop("WEIGHT_SYNC_GCS_DIR", None)
      else:
        os.environ["WEIGHT_SYNC_GCS_DIR"] = prev_env
    self.assertFalse((staging / "step_0_stale").exists())
    self.assertTrue((staging / "keep_marker.txt").exists())

  def test_extract_source_artifact_validation(self):
    meta = weight_sync.WorkUnitMetadata(
        unit=weight_sync.WorkUnitId("trainer"),
        artifact_uri="gs://bucket/step_1_r1",
        checksums={"w": 2.0, "__grand_total__": 2.0, "__tensor_count__": 1},
    )
    req = datatypes.WeightSyncRequest(
        policy_version=1, source_metadata=(meta,)
    )
    uri, sums = gcs_synchronizer.extract_source_artifact(req)
    self.assertEqual(uri, "gs://bucket/step_1_r1")
    assert sums is not None
    self.assertEqual(sums["w"], 2.0)

    with self.assertRaisesRegex(ValueError, "source_metadata is required"):
      gcs_synchronizer.extract_source_artifact(
          datatypes.WeightSyncRequest(policy_version=1)
      )
    conflict = weight_sync.WorkUnitMetadata(
        unit=weight_sync.WorkUnitId("trainer", job_replica_id="1"),
        artifact_uri="gs://bucket/step_2_r2",
    )
    with self.assertRaisesRegex(ValueError, "conflicting artifact_uri"):
      gcs_synchronizer.extract_source_artifact(
          datatypes.WeightSyncRequest(
              policy_version=2, source_metadata=(meta, conflict)
          )
      )

  def test_asymmetric_sharding_d2h_h2d_runner_apply_and_buffer_deletion(self):
    devices = jax.devices()[:4]
    self.assertLen(devices, 4)
    fsdp_mesh = jax.sharding.Mesh(np.array(devices), ("fsdp",))
    tp_mesh = jax.sharding.Mesh(np.array(devices), ("tp",))

    cfg = test_common.ModelConfig(vocab_size=32, num_layers=2)
    src_model = _shard_toy_model(
        test_common.ToyTransformer(cfg, rngs=nnx.Rngs(0)),
        fsdp_mesh,
        "fsdp",
        col_major=False,
    )
    dst_model = _shard_toy_model(
        test_common.ToyTransformer(cfg, rngs=nnx.Rngs(7)),
        tp_mesh,
        "tp",
        col_major=True,
    )
    src_model.emb.embedding.value = (
        jnp.ones_like(src_model.emb.embedding.value) * 3.5
    )

    staging = self.create_tempdir().full_path
    src_sync = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer",
        config=gcs_synchronizer.GCSSynchronizerConfig(staging_dir=staging),
        verify_weights=True,
    )
    dst_sync = gcs_synchronizer.GCSSynchronizer(
        job_name="rollout", verify_weights=True
    )
    src_sync.bind(nnx.state(src_model))
    dst_runner = _NnxRunner(nnx.state(dst_model))
    old_emb_buf = dst_model.emb.embedding.value
    dst_sync.bind(dst_runner.state)

    req = datatypes.WeightSyncRequest(
        policy_version=1, extra_config={"req_id": "r1", "uuid": 1}
    )
    src_sync.d2h(req)
    src_meta = src_sync.work_unit_metadata()

    dst_sync.h2d(
        datatypes.WeightSyncRequest(
            policy_version=1,
            source_metadata=(src_meta,),
            extra_config={"req_id": "r1", "uuid": 1},
        )
    )
    self.assertFalse(old_emb_buf.is_deleted())
    dst_sync.apply_to_runner(dst_runner)
    self.assertTrue(old_emb_buf.is_deleted())
    self.assertEqual(
        dst_model.emb.embedding.value.sharding.mesh.axis_names, ("tp",)
    )
    np.testing.assert_allclose(
        np.asarray(dst_model.emb.embedding.value),
        np.asarray(src_model.emb.embedding.value),
    )

  def test_flat_dict_runner_checksum_rollback_and_retention(self):
    staging = self.create_tempdir().full_path
    w_src = jax.device_put(np.full((4, 4), 7.0, dtype=np.float32))
    w_dst = jax.device_put(np.full((4, 4), 1.0, dtype=np.float32))

    src_sync = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer",
        config=gcs_synchronizer.GCSSynchronizerConfig(
            staging_dir=staging, max_to_keep=1
        ),
        verify_weights=True,
    )
    dst_sync = gcs_synchronizer.GCSSynchronizer(
        job_name="rollout", verify_weights=True
    )
    src_sync.bind({"layers.0.w": w_src})
    runner = _FlatDictRunner({"layers.0.w": w_dst})
    dst_sync.bind(runner.state)

    req1 = datatypes.WeightSyncRequest(
        policy_version=1, extra_config={"req_id": "r1", "uuid": 1}
    )
    src_sync.d2h(req1)
    src_meta = src_sync.work_unit_metadata()
    assert src_meta.checksums is not None
    tampered = dict(src_meta.checksums)
    tampered["__grand_total__"] = float(tampered["__grand_total__"]) + 100.0
    bad_meta = dataclasses.replace(src_meta, checksums=tampered)

    with self.assertRaisesRegex(RuntimeError, "__grand_total__ mismatch"):
      dst_sync.h2d(
          datatypes.WeightSyncRequest(
              policy_version=1,
              source_metadata=(bad_meta,),
              extra_config={"req_id": "r1", "uuid": 1},
          )
      )
    self.assertFalse(w_dst.is_deleted())
    np.testing.assert_allclose(np.asarray(runner.state["layers.0.w"]), 1.0)

    dst_sync.h2d(
        datatypes.WeightSyncRequest(
            policy_version=1,
            source_metadata=(src_meta,),
            extra_config={"req_id": "r1", "uuid": 1},
        )
    )
    dst_sync.abort()
    assert src_meta.artifact_uri is not None
    self.assertTrue(epath.Path(src_meta.artifact_uri).exists())

    dst_sync.h2d(
        datatypes.WeightSyncRequest(
            policy_version=1,
            source_metadata=(src_meta,),
            extra_config={"req_id": "r1", "uuid": 1},
        )
    )
    dst_sync.apply_to_runner(runner)
    self.assertTrue(w_dst.is_deleted())
    np.testing.assert_allclose(np.asarray(runner.state["layers.0.w"]), 7.0)

    req2 = datatypes.WeightSyncRequest(
        policy_version=2, extra_config={"req_id": "r2", "uuid": 2}
    )
    src_sync.d2h(req2)
    step2_uri = src_sync.work_unit_metadata().artifact_uri
    assert step2_uri is not None
    src_sync.release(req2)
    self.assertFalse(epath.Path(src_meta.artifact_uri).exists())
    self.assertTrue(epath.Path(step2_uri).exists())

  def test_tpu_worker_hooks(self):
    cfg = test_common.ModelConfig(vocab_size=32, num_layers=1)
    src_model = test_common.ToyTransformer(cfg, rngs=nnx.Rngs(0))
    src_model(
        jnp.zeros((1, 4), dtype=jnp.int32), jnp.zeros((1, 4), dtype=jnp.int32)
    )
    src_model.emb.embedding.value = (
        jnp.ones_like(src_model.emb.embedding.value) * 9.0
    )
    staging = self.create_tempdir().full_path
    src_sync = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer",
        config=gcs_synchronizer.GCSSynchronizerConfig(staging_dir=staging),
        verify_weights=True,
    )
    src_sync.bind(nnx.state(src_model))
    src_sync.d2h(
        datatypes.WeightSyncRequest(
            policy_version=1, extra_config={"req_id": "r1", "uuid": 1}
        )
    )
    meta = src_sync.work_unit_metadata()
    self.assertIsInstance(hash(meta), int)
    ckpt_uri = meta.artifact_uri
    assert ckpt_uri is not None

    dst_model = test_common.ToyTransformer(cfg, rngs=nnx.Rngs(2))
    dst_model(
        jnp.zeros((1, 4), dtype=jnp.int32), jnp.zeros((1, 4), dtype=jnp.int32)
    )
    worker = _FakeTPUWorker(_NnxRunner(nnx.state(dst_model)))
    self.assertFalse(hasattr(worker, "_gcs_rl_weight_sync"))
    gcs_synchronizer.tpu_worker_gcs_abort(worker)
    with self.assertRaisesRegex(RuntimeError, "bind_gcs_sync must run before"):
      gcs_synchronizer.tpu_worker_get_gcs_metadata(worker)

    meta_dict = gcs_synchronizer.tpu_worker_bind_gcs_sync(worker)
    self.assertEqual(meta_dict["transport_mode"], "gcs")
    rebound_cfg = gcs_synchronizer.GCSSynchronizerConfig(max_to_keep=3)
    rebound_meta = gcs_synchronizer.tpu_worker_bind_gcs_sync(
        worker, job_name="rollout_rebound", config=rebound_cfg
    )
    self.assertEqual(rebound_meta["unit"]["job_name"], "rollout_rebound")
    self.assertEqual(
        gcs_synchronizer.require_gcs_worker_sync(worker, "test").config,
        rebound_cfg,
    )
    self.assertEqual(
        gcs_synchronizer.tpu_worker_get_gcs_metadata(worker)["transport_mode"],
        "gcs",
    )

    async def _run_h2d() -> None:
      gcs_synchronizer.tpu_worker_gcs_h2d(worker, ckpt_uri)

    asyncio.run(_run_h2d())
    gcs_synchronizer.tpu_worker_gcs_commit(worker)
    np.testing.assert_allclose(
        np.asarray(dst_model.emb.embedding.value), 9.0
    )
    self.assertEqual(
        gcs_synchronizer.tpu_worker_gcs_metrics(worker)["h2d_count"], 1
    )

  def test_release_with_max_to_keep_zero_stops_advertising_artifact(self):
    staging = self.create_tempdir().full_path
    sync = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer",
        config=gcs_synchronizer.GCSSynchronizerConfig(
            staging_dir=staging, max_to_keep=0
        ),
        verify_weights=True,
    )
    sync.bind({"layers.0.w": jax.device_put(np.ones((2, 2), np.float32))})
    req = datatypes.WeightSyncRequest(
        policy_version=1, extra_config={"req_id": "r1", "uuid": 1}
    )
    sync.d2h(req)
    staged_uri = sync.work_unit_metadata().artifact_uri
    assert staged_uri is not None
    sync.release(req)
    self.assertFalse(epath.Path(staged_uri).exists())
    meta = sync.work_unit_metadata()
    self.assertIsNone(meta.artifact_uri)
    self.assertIsNone(meta.checksums)

  def test_async_gcs_checkpoint_pruning_logs_warning_on_failure(self):
    sync = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer",
        config=gcs_synchronizer.GCSSynchronizerConfig(max_to_keep=0),
    )
    sync._staged_checkpoints.append("gs://fake-bucket/step_1_r1")
    fake_path = mock.create_autospec(epath.Path, instance=True)
    fake_path.rmtree.side_effect = OSError("permission denied")
    with (
        mock.patch.object(
            gcs_synchronizer.epath,
            "Path",
            autospec=True,
            return_value=fake_path,
        ),
        mock.patch.object(
            logging, "warning", autospec=True
        ) as mock_warning,
    ):
      sync.release()
      sync.close()
    fake_path.rmtree.assert_called_once_with(missing_ok=True)
    mock_warning.assert_called_once()


if __name__ == "__main__":
  absltest.main()
