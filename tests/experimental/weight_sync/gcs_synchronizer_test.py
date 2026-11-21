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
from collections.abc import Sequence
import dataclasses
import json
import multiprocessing as mp
import os
import socket
from typing import Any
from unittest import mock

from absl import logging
from absl.testing import absltest
from etils import epath
from flax import nnx
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.weight_sync import weight_sync
from tunix.experimental.weight_sync import gcs_synchronizer
from tunix.models import hf_mapping
from tunix.models import safetensors_loader
from tunix.models.qwen3 import model as qwen3_model
from tunix.models.qwen3 import params as qwen3_params
from tunix.tests import test_common

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def _free_port() -> int:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    return int(sock.getsockname()[1])


def _multihost_safetensors_worker(
    proc_id: int,
    port: int,
    staging_dir: str,
    out_q: Any,
) -> None:
  """Runs 2-process `jax.distributed` safetensors `d2h` + `h2d` round-trip."""
  os.environ.pop("CLOUDSDK_CONTEXT_AWARE_USE_ECP_HTTP_PROXY", None)
  os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
  os.environ["JAX_PLATFORMS"] = "cpu"
  jax.distributed.initialize(
      f"127.0.0.1:{port}", num_processes=2, process_id=proc_id
  )
  try:
    fsdp_mesh = jax.sharding.Mesh(np.array(jax.devices()), ("fsdp",))
    tp_mesh = jax.sharding.Mesh(np.array(jax.devices()), ("tp",))
    fsdp_sharding = jax.sharding.NamedSharding(
        fsdp_mesh, jax.sharding.PartitionSpec("fsdp", None)
    )
    tp_sharding = jax.sharding.NamedSharding(
        tp_mesh, jax.sharding.PartitionSpec(None, "tp")
    )
    src_global = np.arange(16, dtype=ml_dtypes.bfloat16).reshape(4, 4)
    dst_global = np.zeros((4, 4), dtype=ml_dtypes.bfloat16)
    src_arr = jax.make_array_from_callback(
        src_global.shape, fsdp_sharding, lambda idx: src_global[idx]
    )
    dst_arr = jax.make_array_from_callback(
        dst_global.shape, tp_sharding, lambda idx: dst_global[idx]
    )

    manifest = {
        "schema_version": hf_mapping.SCHEMA_VERSION,
        "rules": [{
            "source_keys": ["layers.0.w"],
            "target_keys": ["model.layers.0.weight"],
            "ops": [{"transpose": {"axes": [1, 0]}}],
        }],
        "tied_keys": [],
        "dropped_keys": [],
    }
    inv_manifest = hf_mapping.invert_manifest(manifest)

    src_sync = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer",
        config=gcs_synchronizer.GCSSynchronizerConfig(staging_dir=staging_dir),
        verify_weights=True,
        to_hf_manifest=manifest,
    )
    dst_sync = gcs_synchronizer.GCSSynchronizer(
        job_name="rollout",
        verify_weights=True,
        from_hf_manifest=inv_manifest,
    )
    src_sync.bind({"layers.0.w": src_arr})
    runner = _FlatDictRunner({"layers.0.w": dst_arr})
    dst_sync.bind(runner.state)

    req = datatypes.WeightSyncRequest(
        policy_version=3, extra_config={"req_id": "mh", "uuid": 3}
    )
    src_sync.d2h(req)
    dst_sync.h2d(
        datatypes.WeightSyncRequest(
            policy_version=3,
            source_metadata=(src_sync.work_unit_metadata(),),
            extra_config={"req_id": "mh", "uuid": 3},
        )
    )
    dst_sync.apply_to_runner(runner)
    local_shards = [
        (s.index, np.asarray(s.data).astype(np.float32).tolist())
        for s in runner.state["layers.0.w"].addressable_shards
    ]
    out_q.put((proc_id, len(local_shards), local_shards[0][1]))
  finally:
    jax.distributed.shutdown()


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


class _FakeSampler:

  def __init__(self, state: nnx.State) -> None:
    self.model_runner: Any = None
    self.transformer_state: nnx.State = state


class _FakeTPUWorker:

  def __init__(self, runner: _NnxRunner, rank: int = 0) -> None:
    self.rank: int = rank
    self.model_runner: _NnxRunner = runner

  def get_weights_state(self) -> Any:
    return self.model_runner.state

  def refresh_model_state_leaves(self) -> None:
    self.model_runner.refresh_state_leaves()


_WIRE_VOCAB, _WIRE_DIM, _WIRE_FFN = 8, 4, 6
_WIRE_EMBED = "model.embed_tokens.weight"
_WIRE_DOWN = "model.layers.0.mlp.down_proj.weight"
_TRANSPOSE = ({"transpose": {"axes": [1, 0]}},)


def _wire_rule(src: str, dst: str, ops: Sequence[Any]) -> dict[str, Any]:
  return {"source_keys": [src], "target_keys": [dst], "ops": list(ops)}


def _wire_trainer(
    staging_dir: str, tied: bool, *, verify_weights: bool = False
) -> gcs_synchronizer.GCSSynchronizer:
  """Binds a Tunix-native trainer whose `to_hf_manifest` writes HF names."""
  embedding = np.arange(_WIRE_VOCAB * _WIRE_DIM, dtype=np.float32)
  rules = [
      _wire_rule(_WIRE_EMBED, "embedder.input_embedding", ()),
      _wire_rule(_WIRE_DOWN, "layers.0.mlp.down_proj.kernel", _TRANSPOSE),
  ]
  state = {
      "embedder.input_embedding": embedding.reshape(_WIRE_VOCAB, _WIRE_DIM),
      "layers.0.mlp.down_proj.kernel": np.ones(
          (_WIRE_FFN, _WIRE_DIM), np.float32
      ),
  }
  if not tied:
    rules.append(_wire_rule("lm_head.weight", "lm_head.w", _TRANSPOSE))
    state["lm_head.w"] = state["embedder.input_embedding"].T.copy()
  from_hf = {
      "rules": rules,
      "dropped_keys": ["lm_head.weight"] if tied else [],
  }
  sync = gcs_synchronizer.GCSSynchronizer(
      job_name="trainer",
      config=gcs_synchronizer.GCSSynchronizerConfig(staging_dir=staging_dir),
      verify_weights=verify_weights,
      to_hf_manifest=hf_mapping.invert_manifest(from_hf),
  )
  sync.bind({k: jax.device_put(v) for k, v in state.items()})
  return sync


def _wire_rollout(
    tied: bool, *, verify_weights: bool = False
) -> tuple[gcs_synchronizer.GCSSynchronizer, _FlatDictRunner]:
  """Binds a vLLM-like reader; tied readers fan `lm_head` out of the embed."""
  head_src = _WIRE_EMBED if tied else "lm_head.weight"
  sync = gcs_synchronizer.GCSSynchronizer(
      job_name="rollout",
      verify_weights=verify_weights,
      from_hf_manifest={
          "rules": [
              _wire_rule(_WIRE_EMBED, "embed_tokens.weight", ()),
              _wire_rule(head_src, "lm_head.weight", _TRANSPOSE),
              _wire_rule(
                  _WIRE_DOWN, "layers.0.mlp.down_proj.weight", _TRANSPOSE
              ),
          ]
      },
  )
  runner = _FlatDictRunner({
      "embed_tokens.weight": jnp.zeros((_WIRE_VOCAB, _WIRE_DIM)),
      "lm_head.weight": jnp.zeros((_WIRE_DIM, _WIRE_VOCAB)),
      "layers.0.mlp.down_proj.weight": jnp.zeros((_WIRE_FFN, _WIRE_DIM)),
  })
  sync.bind(runner.state)
  return sync, runner


class GCSSynchronizerTest(absltest.TestCase):

  def test_config_validation_env_default_and_cleanup_on_start(self):
    self.assertEqual(
        gcs_synchronizer.GCSSynchronizerConfig().checkpoint_format,
        "safetensors",
    )
    with self.assertRaisesRegex(ValueError, "staging_dir must not be empty"):
      gcs_synchronizer.GCSSynchronizerConfig(staging_dir="   ")
    with self.assertRaisesRegex(ValueError, "max_to_keep"):
      gcs_synchronizer.GCSSynchronizerConfig(max_to_keep=-1)
    with self.assertRaisesRegex(ValueError, "max_to_keep"):
      gcs_synchronizer.GCSSynchronizerConfig(max_to_keep=True)
    with self.assertRaisesRegex(ValueError, "checkpoint_format"):
      gcs_synchronizer.GCSSynchronizerConfig(checkpoint_format="npz")
    with self.assertRaisesRegex(ValueError, "max_shard_bytes"):
      gcs_synchronizer.GCSSynchronizerConfig(max_shard_bytes=0)
    with self.assertRaisesRegex(ValueError, "max_shard_bytes must be <="):
      gcs_synchronizer.GCSSynchronizerConfig(max_shard_bytes=(4 << 30) + 1)
    with self.assertRaisesRegex(ValueError, "save_concurrent_gb"):
      gcs_synchronizer.GCSSynchronizerConfig(save_concurrent_gb=0)
    with self.assertRaisesRegex(ValueError, "cleanup_on_start=True requires"):
      gcs_synchronizer.GCSSynchronizerConfig(cleanup_on_start=True)

    staging = epath.Path(self.create_tempdir().full_path)
    (staging / "step_0_stale").mkdir()
    (staging / "keep_marker.txt").write_text("ok")
    prev_env = (
        os.environ["WEIGHT_SYNC_GCS_DIR"]
        if "WEIGHT_SYNC_GCS_DIR" in os.environ
        else None
    )
    os.environ["WEIGHT_SYNC_GCS_DIR"] = str(staging)
    try:
      cfg = gcs_synchronizer.GCSSynchronizerConfig(cleanup_on_start=True)
      self.assertEqual(cfg.staging_dir, str(staging))
      sync = gcs_synchronizer.GCSSynchronizer(job_name="trainer", config=cfg)
      self.assertFalse(sync.bound)
      self.assertFalse(sync.active)
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
    req = datatypes.WeightSyncRequest(policy_version=1, source_metadata=(meta,))
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
    src_model.emb.embedding.set_value(
        jnp.ones_like(src_model.emb.embedding.get_value()) * 3.5
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
    self.assertTrue(src_sync.bound)
    self.assertTrue(src_sync.active)
    self.assertLen(src_sync.work_unit_metadata_all(), 1)
    dst_runner = _NnxRunner(nnx.state(dst_model))
    old_emb_buf = dst_model.emb.embedding.get_value()
    dst_sync.bind(dst_runner.state)

    req = datatypes.WeightSyncRequest(
        policy_version=1, extra_config={"req_id": "r1", "uuid": 1}
    )
    src_sync.d2h(req)
    src_meta = src_sync.work_unit_metadata()
    assert src_meta.artifact_uri is not None
    self.assertTrue(
        (epath.Path(src_meta.artifact_uri) / "model.safetensors").exists()
    )

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
        dst_model.emb.embedding.get_value().sharding.mesh.axis_names, ("tp",)
    )
    np.testing.assert_allclose(
        np.asarray(dst_model.emb.embedding.get_value()),
        np.asarray(src_model.emb.embedding.get_value()),
    )

  def test_work_unit_metadata_pairs_tied_and_untied_peers_on_wire(self):
    def trainer(tied: bool) -> gcs_synchronizer.GCSSynchronizer:
      return _wire_trainer(self.create_tempdir().full_path, tied)

    def wire(sync: gcs_synchronizer.GCSSynchronizer) -> dict[str, Any]:
      return {
          v.name: (v.shape, v.item_size)
          for v in sync.work_unit_metadata().variables
      }

    for tied in (True, False):
      with self.subTest(tied=tied):
        src_wire = wire(trainer(tied))
        self.assertEqual(src_wire, wire(_wire_rollout(tied)[0]))
        self.assertEqual("lm_head.weight" in src_wire, not tied)
        self.assertEqual(src_wire[_WIRE_EMBED], ((_WIRE_VOCAB, _WIRE_DIM), 4))

    # A tied reader materializes `lm_head` from the one wire embedding and
    # ignores a redundant wire `lm_head` (HF checkpoints ship both forms).
    for src_tied in (True, False):
      with self.subTest(src_tied=src_tied):
        src, (dst, runner) = trainer(src_tied), _wire_rollout(tied=True)
        src.d2h(
            datatypes.WeightSyncRequest(
                policy_version=1, extra_config={"req_id": "tie", "uuid": 1}
            )
        )
        dst.h2d(checkpoint_path=src.work_unit_metadata().artifact_uri)
        dst.apply_to_runner(runner)
        out = runner.state
        np.testing.assert_array_equal(
            out["lm_head.weight"], np.asarray(out["embed_tokens.weight"]).T
        )
        np.testing.assert_array_equal(
            out["embed_tokens.weight"],
            np.arange(_WIRE_VOCAB * _WIRE_DIM, dtype=np.float32).reshape(
                _WIRE_VOCAB, _WIRE_DIM
            ),
        )

  def test_manifest_verify_weights_checksums_wire_namespace(self):
    req = datatypes.WeightSyncRequest(
        policy_version=1, extra_config={"req_id": "wire", "uuid": 1}
    )

    def staged(tied: bool) -> weight_sync.WorkUnitMetadata:
      src = _wire_trainer(
          self.create_tempdir().full_path, tied, verify_weights=True
      )
      src.d2h(req)
      return src.work_unit_metadata()

    # Tunix-native and vLLM-native names (and tied tensor counts) differ, so
    # both peers must checksum exactly the HF tensors that cross the wire.
    for tied in (True, False):
      with self.subTest(tied=tied):
        meta = staged(tied)
        assert meta.checksums is not None
        self.assertEqual(meta.checksums["__tensor_count__"], 2 if tied else 3)
        dst, runner = _wire_rollout(tied, verify_weights=True)
        dst.h2d(
            datatypes.WeightSyncRequest(
                policy_version=1, source_metadata=(meta,)
            )
        )
        dst.apply_to_runner(runner)
        np.testing.assert_array_equal(
            runner.state["lm_head.weight"],
            np.asarray(runner.state["embed_tokens.weight"]).T,
        )

    meta = staged(tied=True)
    assert meta.artifact_uri is not None
    corrupt = np.arange(_WIRE_VOCAB * _WIRE_DIM, dtype=np.float32) + 1.0
    gcs_synchronizer._write_safetensors_shard(  # pylint: disable=protected-access
        epath.Path(meta.artifact_uri) / "model.safetensors",
        [
            (_WIRE_EMBED, corrupt.reshape(_WIRE_VOCAB, _WIRE_DIM)),
            (_WIRE_DOWN, np.ones((_WIRE_DIM, _WIRE_FFN), np.float32)),
        ],
    )
    dst, runner = _wire_rollout(tied=True, verify_weights=True)
    with self.assertRaisesRegex(RuntimeError, "Weight checksum"):
      dst.h2d(
          datatypes.WeightSyncRequest(policy_version=1, source_metadata=(meta,))
      )
    np.testing.assert_array_equal(runner.state["lm_head.weight"], 0.0)

  def test_sharded_safetensors_qwen3_manifest_and_orbax_autodetect(self):
    qcfg = qwen3_model.ModelConfig(
        num_layers=1,
        vocab_size=32,
        embed_dim=16,
        hidden_dim=32,
        num_heads=2,
        head_dim=8,
        num_kv_heads=2,
        norm_eps=1e-6,
        rope_theta=10000,
    )
    src_model = qwen3_model.Qwen3(qcfg, rngs=nnx.Rngs(0))
    dst_model = qwen3_model.Qwen3(qcfg, rngs=nnx.Rngs(9))
    to_hf = qwen3_params.build_to_hf_manifest(qcfg)
    from_hf = qwen3_params.build_from_hf_manifest(qcfg)

    staging = self.create_tempdir().full_path
    src_sync = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer",
        config=gcs_synchronizer.GCSSynchronizerConfig(
            staging_dir=staging,
            max_shard_bytes=512,
        ),
        verify_weights=True,
        to_hf_manifest=to_hf,
    )
    dst_sync = gcs_synchronizer.GCSSynchronizer(
        job_name="rollout",
        verify_weights=True,
        from_hf_manifest=from_hf,
    )
    src_sync.bind(nnx.state(src_model))
    sampler = _FakeSampler(nnx.state(dst_model))
    dst_sync.bind(sampler.transformer_state)

    req = datatypes.WeightSyncRequest(
        policy_version=1, extra_config={"req_id": "qwen", "uuid": 1}
    )
    src_sync.d2h(req)
    meta = src_sync.work_unit_metadata()
    assert meta.artifact_uri is not None
    ckpt_dir = epath.Path(meta.artifact_uri)
    index_data = json.loads(
        (ckpt_dir / "model.safetensors.index.json").read_text()
    )
    self.assertGreater(len(set(index_data["weight_map"].values())), 1)
    self.assertIn("model.embed_tokens.weight", index_data["weight_map"])

    dst_sync.h2d(
        datatypes.WeightSyncRequest(policy_version=1, source_metadata=(meta,))
    )
    dst_sync.apply_to_sampler(sampler)
    src_flat = safetensors_loader.flatten_model_state(src_model)
    dst_flat = safetensors_loader.flatten_model_state(dst_model)
    for k, src_val in src_flat.items():
      np.testing.assert_allclose(np.asarray(dst_flat[k]), np.asarray(src_val))

    # Verify Orbax checkpoint auto-detection even when receiver defaults to ST.
    orbax_dir = self.create_tempdir().full_path
    orbax_src = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer_orbax",
        config=gcs_synchronizer.GCSSynchronizerConfig(
            staging_dir=orbax_dir,
            checkpoint_format="orbax",
        ),
        verify_weights=True,
    )
    orbax_dst = gcs_synchronizer.GCSSynchronizer(
        job_name="rollout_auto", verify_weights=True
    )
    w_src = jax.device_put(np.full((4, 4), 5.0, dtype=np.float32))
    w_dst = jax.device_put(np.zeros((4, 4), dtype=np.float32))
    orbax_src.bind({"layers.0.w": w_src})
    runner = _FlatDictRunner({"layers.0.w": w_dst})
    orbax_dst.bind(runner.state)
    orbax_src.d2h(req)
    orbax_dst.h2d(
        datatypes.WeightSyncRequest(
            policy_version=1,
            source_metadata=(orbax_src.work_unit_metadata(),),
        )
    )
    orbax_dst.apply_to_runner(runner)
    np.testing.assert_allclose(np.asarray(runner.state["layers.0.w"]), 5.0)

  def test_tied_keys_manifest_checksum_conservation_and_multihost(self):
    tied_to_hf = {
        "schema_version": hf_mapping.SCHEMA_VERSION,
        "rules": [],
        "tied_keys": [["embed.w", "lm_head.w"]],
        "dropped_keys": [],
    }
    tied_from_hf = {
        "schema_version": hf_mapping.SCHEMA_VERSION,
        "rules": [],
        "tied_keys": [["embed.w", "lm_head.w"]],
        "dropped_keys": [],
    }
    emb = jax.device_put(np.arange(8, dtype=ml_dtypes.bfloat16).reshape(4, 2))
    src_sync = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer",
        config=gcs_synchronizer.GCSSynchronizerConfig(
            staging_dir=self.create_tempdir().full_path
        ),
        verify_weights=True,
        to_hf_manifest=tied_to_hf,
    )
    dst_sync = gcs_synchronizer.GCSSynchronizer(
        job_name="rollout",
        verify_weights=True,
        from_hf_manifest=tied_from_hf,
    )
    src_sync.bind({"embed.w": emb, "lm_head.w": emb})
    runner = _FlatDictRunner({
        "embed.w": jax.device_put(np.zeros((4, 2), dtype=ml_dtypes.bfloat16)),
        "lm_head.w": jax.device_put(np.zeros((4, 2), dtype=ml_dtypes.bfloat16)),
    })
    dst_sync.bind(runner.state)
    req = datatypes.WeightSyncRequest(
        policy_version=1, extra_config={"req_id": "tied", "uuid": 1}
    )
    src_sync.d2h(req)
    meta = src_sync.work_unit_metadata()
    assert meta.artifact_uri is not None
    self.assertEqual(
        safetensors_loader.read_checkpoint_keys(meta.artifact_uri),
        frozenset({"embed.w"}),
    )
    # Both peers advertise exactly the written tensors; tied targets are
    # derived, never on the wire.
    self.assertEqual([v.name for v in meta.variables], ["embed.w"])
    self.assertEqual(
        [v.name for v in dst_sync.work_unit_metadata().variables], ["embed.w"]
    )
    dst_sync.h2d(
        datatypes.WeightSyncRequest(policy_version=1, source_metadata=(meta,))
    )
    dst_sync.apply_to_runner(runner)
    np.testing.assert_array_equal(
        np.asarray(runner.state["lm_head.w"]), np.asarray(emb)
    )

    # 2-process jax.distributed safetensors d2h (allgather) -> h2d (broadcast).
    ctx = mp.get_context("spawn")
    port = _free_port()
    out_q = ctx.Queue()
    mh_dir = self.create_tempdir().full_path
    procs = [
        ctx.Process(
            target=_multihost_safetensors_worker,
            args=(idx, port, mh_dir, out_q),
        )
        for idx in range(2)
    ]
    for proc in procs:
      proc.start()
    for proc in procs:
      proc.join(timeout=30)
      self.assertEqual(proc.exitcode, 0)
    mh_results = sorted([out_q.get(), out_q.get()])
    self.assertEqual(mh_results[0][0], 0)
    self.assertEqual(mh_results[1][0], 1)
    self.assertEqual(mh_results[0][2], [[0.0], [4.0], [8.0], [12.0]])

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
    src_model.emb.embedding.set_value(
        jnp.ones_like(src_model.emb.embedding.get_value()) * 9.0
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

    identity_manifest = {
        "schema_version": hf_mapping.SCHEMA_VERSION,
        "rules": [],
        "tied_keys": [],
        "dropped_keys": [],
    }
    self.assertEqual(
        gcs_synchronizer.GCSSynchronizer("unbound").checksums()[
            "__tensor_count__"
        ],
        0,
    )
    with mock.patch.dict(os.environ, {"VERIFY_WEIGHTS": "true"}):
      meta_dict = gcs_synchronizer.tpu_worker_bind_gcs_sync(
          worker, from_hf_manifest=identity_manifest
      )
    self.assertEqual(meta_dict["transport_mode"], "gcs")
    rebound_cfg = gcs_synchronizer.GCSSynchronizerConfig(max_to_keep=3)
    rebound_meta = gcs_synchronizer.tpu_worker_bind_gcs_sync(
        worker,
        job_name="rollout_rebound",
        config=rebound_cfg,
        from_hf_manifest=identity_manifest,
    )
    self.assertEqual(rebound_meta["unit"]["job_name"], "rollout_rebound")
    self.assertEqual(
        gcs_synchronizer.require_gcs_worker_sync(worker, "test").config,
        rebound_cfg,
    )
    with self.assertRaisesRegex(ValueError, "Manifest must be a mapping"):
      gcs_synchronizer.tpu_worker_bind_gcs_sync(
          worker, job_name="bad_rebind", from_hf_manifest="{}"
      )
    self.assertEqual(
        gcs_synchronizer.require_gcs_worker_sync(worker, "test").job_name,
        "rollout_rebound",
    )
    self.assertEqual(
        gcs_synchronizer.tpu_worker_get_gcs_metadata(worker)["transport_mode"],
        "gcs",
    )

    async def _run_h2d() -> dict[str, float | int]:
      return gcs_synchronizer.tpu_worker_gcs_h2d(worker, ckpt_uri)

    first_sums = asyncio.run(_run_h2d())
    self.assertGreater(first_sums["__tensor_count__"], 0)
    gcs_synchronizer.tpu_worker_gcs_abort(worker)
    asyncio.run(_run_h2d())
    gcs_synchronizer.tpu_worker_gcs_commit(worker)
    np.testing.assert_allclose(
        np.asarray(dst_model.emb.embedding.get_value()), 9.0
    )
    self.assertEqual(
        gcs_synchronizer.tpu_worker_gcs_metrics(worker)["h2d_count"], 2
    )

  def test_safetensors_h2d_error_diagnostics(self):
    staging = self.create_tempdir().full_path
    src_sync = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer",
        config=gcs_synchronizer.GCSSynchronizerConfig(staging_dir=staging),
    )
    src_sync.bind({"layers.0.w": jax.device_put(np.ones((2, 2), np.float32))})
    src_sync.d2h(
        datatypes.WeightSyncRequest(
            policy_version=1, extra_config={"req_id": "diag", "uuid": 1}
        )
    )
    ckpt_uri = src_sync.work_unit_metadata().artifact_uri
    assert ckpt_uri is not None

    empty_dir = self.create_tempdir().full_path
    dst_missing = gcs_synchronizer.GCSSynchronizer(job_name="rollout")
    dst_missing.bind({
        "layers.0.w": jax.device_put(np.zeros((2, 2), np.float32)),
        "layers.1.w": jax.device_put(np.zeros((2, 2), np.float32)),
    })
    with self.assertRaisesRegex(FileNotFoundError, "No safetensors shards"):
      dst_missing.h2d(checkpoint_path=empty_dir)
    with self.assertRaisesRegex(KeyError, "Missing 1 weight tensor"):
      dst_missing.h2d(checkpoint_path=ckpt_uri)

    dst_bad_shape = gcs_synchronizer.GCSSynchronizer(job_name="rollout")
    dst_bad_shape.bind(
        {"layers.0.w": jax.device_put(np.zeros((4, 2), np.float32))}
    )
    with self.assertRaisesRegex(ValueError, "Shape mismatch"):
      dst_bad_shape.h2d(checkpoint_path=ckpt_uri)

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
        mock.patch.object(logging, "warning", autospec=True) as mock_warning,
    ):
      sync.release()
      sync.close()
    fake_path.rmtree.assert_called_once_with(missing_ok=True)
    mock_warning.assert_called_once()


if __name__ == "__main__":
  absltest.main()
