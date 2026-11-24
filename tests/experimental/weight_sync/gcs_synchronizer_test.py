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
import contextlib
import dataclasses
import gc
import json
import multiprocessing as mp
import os
import socket
import struct
import time
import types
from typing import Any, cast
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
from tunix.models import safetensors_io
from tunix.models import safetensors_loader
from tunix.models import safetensors_saver
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
    with self.assertRaisesRegex(ValueError, "staging_dir must not be empty"):
      gcs_synchronizer.GCSSynchronizerConfig(staging_dir="   ")
    with self.assertRaisesRegex(ValueError, "max_to_keep"):
      gcs_synchronizer.GCSSynchronizerConfig(max_to_keep=-1)
    with self.assertRaisesRegex(ValueError, "max_to_keep"):
      gcs_synchronizer.GCSSynchronizerConfig(max_to_keep=True)
    with self.assertRaisesRegex(ValueError, "save_concurrent_gb"):
      gcs_synchronizer.GCSSynchronizerConfig(save_concurrent_gb=0)
    with self.assertRaisesRegex(ValueError, "checkpoint_format"):
      gcs_synchronizer.GCSSynchronizerConfig(checkpoint_format="invalid")
    with self.assertRaisesRegex(ValueError, "max_bucket_bytes"):
      gcs_synchronizer.GCSSynchronizerConfig(max_bucket_bytes=0)
    with self.assertRaisesRegex(ValueError, "max_shard_bytes"):
      gcs_synchronizer.GCSSynchronizerConfig(max_shard_bytes=0)
    with self.assertRaisesRegex(ValueError, "exceeds 4 GiB ceiling"):
      gcs_synchronizer.GCSSynchronizerConfig(
          max_shard_bytes=safetensors_saver.MAX_SAFETENSORS_SHARD_BYTES + 1
      )
    with self.assertRaisesRegex(ValueError, "exceeds max_shard_bytes"):
      gcs_synchronizer.GCSSynchronizerConfig(
          max_bucket_bytes=1024, max_shard_bytes=512
      )
    with self.assertRaisesRegex(ValueError, "write_procs"):
      gcs_synchronizer.GCSSynchronizerConfig(write_procs=0)
    with self.assertRaisesRegex(ValueError, "read_procs"):
      gcs_synchronizer.GCSSynchronizerConfig(read_procs=0)
    with self.assertRaisesRegex(ValueError, "cleanup_on_start=True requires"):
      gcs_synchronizer.GCSSynchronizerConfig(cleanup_on_start=True)

    staging = epath.Path(self.create_tempdir().full_path)
    (staging / "step_0_stale").mkdir()
    (staging / "keep_marker.txt").write_text("ok")
    cfg = gcs_synchronizer.GCSSynchronizerConfig(
        staging_dir=str(staging), cleanup_on_start=True
    )
    gcs_synchronizer.GCSSynchronizer(job_name="trainer", config=cfg)
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

    # Corrupt the embedding bytes in place in whichever shard holds them.
    meta = staged(tied=True)
    assert meta.artifact_uri is not None
    corrupt = np.arange(_WIRE_VOCAB * _WIRE_DIM, dtype=np.float32) + 1.0
    corrupted = 0
    for shard in epath.Path(meta.artifact_uri).glob("*.safetensors"):
      with open(os.fspath(shard), "r+b") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        entry = json.loads(f.read(hdr_len)).get(_WIRE_EMBED)
        if entry is None:
          continue
        f.seek(8 + hdr_len + entry["data_offsets"][0])
        f.write(corrupt.tobytes())
        corrupted += 1
    self.assertEqual(corrupted, 1)
    dst, runner = _wire_rollout(tied=True, verify_weights=True)
    with self.assertRaisesRegex(RuntimeError, "Weight checksum"):
      dst.h2d(
          datatypes.WeightSyncRequest(policy_version=1, source_metadata=(meta,))
      )
    np.testing.assert_array_equal(runner.state["lm_head.weight"], 0.0)

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
    np.testing.assert_allclose(np.asarray(dst_model.emb.embedding.value), 9.0)
    self.assertEqual(
        gcs_synchronizer.tpu_worker_gcs_metrics(worker)["h2d_count"], 1
    )

  def test_tied_reader_checksums_ignore_redundant_disk_target_key(self):
    src_tied = _wire_trainer(
        self.create_tempdir().full_path, tied=True, verify_weights=True
    )
    req = datatypes.WeightSyncRequest(
        policy_version=1, extra_config={"req_id": "redundant", "uuid": 1}
    )
    src_tied.d2h(req)
    tied_meta = src_tied.work_unit_metadata()
    assert tied_meta.checksums is not None
    self.assertEqual(tied_meta.checksums["__tensor_count__"], 2)

    # Write a checkpoint with the same two wire tensors plus an extra on-disk
    # `lm_head.weight` matching the bound target name.
    extra_dir = self.create_tempdir().full_path
    safetensors_saver.save_safetensors(
        {
            _WIRE_EMBED: (
                np.arange(_WIRE_VOCAB * _WIRE_DIM, dtype=np.float32).reshape(
                    _WIRE_VOCAB, _WIRE_DIM
                )
            ),
            _WIRE_DOWN: np.ones((_WIRE_DIM, _WIRE_FFN), dtype=np.float32),
            "lm_head.weight": np.full(
                (_WIRE_DIM, _WIRE_VOCAB), 99.0, dtype=np.float32
            ),
        },
        extra_dir,
    )
    dst, runner = _wire_rollout(tied=True, verify_weights=True)
    dst.h2d(
        checkpoint_path=extra_dir,
        source_checksums=tied_meta.checksums,
    )
    dst.apply_to_runner(runner)
    np.testing.assert_array_equal(
        runner.state["lm_head.weight"],
        np.asarray(runner.state["embed_tokens.weight"]).T,
    )

    # Active `tied_keys` whose source is produced by a rule must not load a
    # same-named backend target tensor from disk into wire checksums.
    active_tied_manifest = {
        "rules": [{
            "source_keys": ["hf.k_proj.weight"],
            "target_keys": ["k0.kernel"],
            "ops": [],
        }],
        "tied_keys": [["k0.kernel", "k1.kernel"]],
    }
    tied_dir = self.create_tempdir().full_path
    k_wire = np.arange(4, dtype=np.float32).reshape(2, 2)
    safetensors_saver.save_safetensors(
        {
            "hf.k_proj.weight": k_wire,
            "k0.kernel": np.full((2, 2), 99.0, dtype=np.float32),
        },
        tied_dir,
    )
    expected_sums = gcs_synchronizer.weight_utils.compute_checksums(
        ["hf.k_proj.weight"], [k_wire], sample=None, canonical_keys=True
    )
    dst_active = gcs_synchronizer.GCSSynchronizer(
        job_name="rollout",
        verify_weights=True,
        from_hf_manifest=active_tied_manifest,
    )
    runner_active = _FlatDictRunner({
        "k0.kernel": jax.device_put(np.zeros((2, 2), np.float32)),
        "k1.kernel": jax.device_put(np.zeros((2, 2), np.float32)),
    })
    dst_active.bind(runner_active.state)
    dst_active.h2d(checkpoint_path=tied_dir, source_checksums=expected_sums)
    dst_active.apply_to_runner(runner_active)
    np.testing.assert_array_equal(runner_active.state["k0.kernel"], k_wire)
    np.testing.assert_array_equal(runner_active.state["k1.kernel"], k_wire)

  def test_multihost_cleanup_error_broadcast_prevents_deadlock(self):
    staging = epath.Path(self.create_tempdir().full_path)
    (staging / "step_0_boom").mkdir()
    cleanup_status: dict[str, Any] = {}

    def _capture_cleanup(payload: Any) -> Any:
      if payload is not None and not cleanup_status:
        cleanup_status["value"] = payload
      return cleanup_status["value"]

    cfg = gcs_synchronizer.GCSSynchronizerConfig(
        staging_dir=str(staging), cleanup_on_start=True
    )
    with (
        mock.patch.object(jax, "process_count", return_value=2),
        mock.patch.object(jax, "process_index", return_value=0),
        mock.patch.object(
            gcs_synchronizer.safetensors_saver,
            "remove_tree",
            side_effect=OSError("disk full"),
        ),
        mock.patch.object(
            gcs_synchronizer.sharding_utils,
            "broadcast_host0_json",
            side_effect=_capture_cleanup,
        ),
    ):
      with self.assertRaisesRegex(OSError, "disk full"):
        gcs_synchronizer.GCSSynchronizer(job_name="trainer", config=cfg)

    with (
        mock.patch.object(jax, "process_count", return_value=2),
        mock.patch.object(jax, "process_index", return_value=1),
        mock.patch.object(
            gcs_synchronizer.sharding_utils,
            "broadcast_host0_json",
            side_effect=_capture_cleanup,
        ),
    ):
      with self.assertRaisesRegex(OSError, "disk full"):
        gcs_synchronizer.GCSSynchronizer(job_name="trainer", config=cfg)

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
    fake_path.is_dir.return_value = True
    fake_path.iterdir.return_value = []
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

  def test_safetensors_moe_unstacking_and_orbax_roundtrip(self):
    staging = self.create_tempdir().full_path
    moe_gate = np.arange(4 * 8 * 4, dtype=np.float32).reshape(4, 8, 4)
    moe_down = np.arange(4 * 4 * 8, dtype=np.float32).reshape(4, 4, 8)
    src_state = {
        "layers.0.mlp.gate_proj": jax.device_put(moe_gate),
        "layers.0.mlp.down_proj": jax.device_put(moe_down),
    }
    dst_state = {
        "layers.0.mlp.gate_proj": jax.device_put(np.zeros_like(moe_gate)),
        "layers.0.mlp.down_proj": jax.device_put(np.zeros_like(moe_down)),
    }
    src_sync = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer",
        config=gcs_synchronizer.GCSSynchronizerConfig(
            staging_dir=staging,
            checkpoint_format="safetensors",
            max_bucket_bytes=128,
            max_shard_bytes=None,
            per_wave_shards=True,
        ),
        verify_weights=True,
    )
    dst_sync = gcs_synchronizer.GCSSynchronizer(
        job_name="rollout", verify_weights=True
    )
    src_sync.bind(src_state)
    runner = _FlatDictRunner(dst_state)
    dst_sync.bind(runner.state)

    req = datatypes.WeightSyncRequest(
        policy_version=1, extra_config={"req_id": "moe", "uuid": 1}
    )
    src_sync.d2h(req)
    meta = src_sync.work_unit_metadata()
    assert meta.artifact_uri is not None
    index_file = epath.Path(meta.artifact_uri) / "model.safetensors.index.json"
    self.assertTrue(index_file.exists())
    # Pure Identity mode must NOT heuristically transpose 3D expert weights.
    weight_map = json.loads(index_file.read_text())["weight_map"]
    gate0_shard = (
        epath.Path(meta.artifact_uri)
        / weight_map["layers.0.mlp.gate_proj.experts.0"]
    )
    down0_shard = (
        epath.Path(meta.artifact_uri)
        / weight_map["layers.0.mlp.down_proj.experts.0"]
    )
    with gate0_shard.open("rb") as f:
      hdr_len = struct.unpack("<Q", f.read(8))[0]
      gate_hdr = json.loads(f.read(hdr_len))
    with down0_shard.open("rb") as f:
      hdr_len = struct.unpack("<Q", f.read(8))[0]
      down_hdr = json.loads(f.read(hdr_len))
    self.assertEqual(
        gate_hdr["layers.0.mlp.gate_proj.experts.0"]["shape"], [8, 4]
    )
    self.assertEqual(
        down_hdr["layers.0.mlp.down_proj.experts.0"]["shape"], [4, 8]
    )
    files = gcs_synchronizer.safetensors_loader.resolve_range_sources(
        meta.artifact_uri
    )
    self.assertGreaterEqual(len(files), 2)
    dst_sync.h2d(
        datatypes.WeightSyncRequest(
            policy_version=1,
            source_metadata=(meta,),
            extra_config={"req_id": "moe", "uuid": 1},
        )
    )
    dst_sync.apply_to_runner(runner)
    np.testing.assert_allclose(
        np.asarray(runner.state["layers.0.mlp.gate_proj"]), moe_gate
    )
    np.testing.assert_allclose(
        np.asarray(runner.state["layers.0.mlp.down_proj"]), moe_down
    )

    orbax_staging = self.create_tempdir().full_path
    orbax_src = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer_orbax",
        config=gcs_synchronizer.GCSSynchronizerConfig(
            staging_dir=orbax_staging,
            checkpoint_format="orbax",
        ),
        verify_weights=True,
    )
    orbax_src.bind(
        {"layers.0.w": jax.device_put(np.full((2, 2), 5.0, dtype=np.float32))}
    )
    orbax_src.d2h(
        datatypes.WeightSyncRequest(
            policy_version=1, extra_config={"req_id": "orbax1", "uuid": 1}
        )
    )
    orbax_meta = orbax_src.work_unit_metadata()
    assert orbax_meta.artifact_uri is not None
    orbax_dst = gcs_synchronizer.GCSSynchronizer(
        job_name="rollout_orbax",
        config=gcs_synchronizer.GCSSynchronizerConfig(
            checkpoint_format="orbax"
        ),
        verify_weights=True,
    )
    orbax_runner = _FlatDictRunner(
        {"layers.0.w": jax.device_put(np.zeros((2, 2), dtype=np.float32))}
    )
    orbax_dst.bind(orbax_runner.state)
    orbax_dst.h2d(checkpoint_path=orbax_meta.artifact_uri)
    orbax_dst.apply_to_runner(orbax_runner)
    np.testing.assert_allclose(
        np.asarray(orbax_runner.state["layers.0.w"]), 5.0
    )

  def test_lifecycle_guards_sampler_apply_and_checksums(self):
    with self.assertRaisesRegex(ValueError, "job_name must not be empty"):
      gcs_synchronizer.GCSSynchronizer(job_name=" ")
    with self.assertRaisesRegex(
        ValueError, "worker_index must be non-negative"
    ):
      gcs_synchronizer.GCSSynchronizer(job_name="trainer", worker_index=-1)

    sync = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer", verify_weights=True
    )
    self.assertFalse(sync.bound)
    self.assertFalse(sync.active)
    self.assertEqual(sync.checksums()["__tensor_count__"], 0)
    sync.apply_to_runner(None)
    with self.assertRaisesRegex(RuntimeError, "bind\\(\\) must run before d2h"):
      sync.d2h()
    with self.assertRaisesRegex(RuntimeError, "bind\\(\\) must run before h2d"):
      sync.h2d()
    with self.assertRaisesRegex(
        RuntimeError, "bind\\(\\) must run before work_unit_metadata"
    ):
      sync.work_unit_metadata()
    with self.assertRaisesRegex(ValueError, "no bindable weight arrays"):
      sync.bind({})
    with self.assertRaisesRegex(ValueError, "empty or colliding canonical"):
      sync.bind({
          "a.b": jax.device_put(np.ones((2,), np.float32)),
          "a": {"b": jax.device_put(np.ones((2,), np.float32))},
      })

    w = jax.device_put(np.ones((2, 2), dtype=np.float32))
    sync.bind({"layers.0.w": w})
    self.assertTrue(sync.bound)
    self.assertTrue(sync.active)
    self.assertLen(sync.work_unit_metadata_all(), 1)
    self.assertEqual(sync.checksums()["__tensor_count__"], 1)
    with self.assertRaisesRegex(ValueError, "staging_dir is required"):
      sync.d2h()

    staging = self.create_tempdir().full_path
    sync.config = gcs_synchronizer.GCSSynchronizerConfig(staging_dir=staging)
    with self.assertRaisesRegex(ValueError, "sync_request is required"):
      sync.d2h(None)
    with self.assertRaisesRegex(ValueError, "policy_version must be an int"):
      sync.d2h(cast(Any, types.SimpleNamespace(policy_version=True)))
    with self.assertRaisesRegex(ValueError, "req_id.*is required"):
      sync.d2h(datatypes.WeightSyncRequest(policy_version=1))
    with self.assertRaisesRegex(ValueError, "Invalid.*req_id"):
      sync.d2h(
          datatypes.WeightSyncRequest(
              policy_version=1, extra_config={"req_id": "../bad"}
          )
      )
    with self.assertRaisesRegex(
        ValueError, "checkpoint_path must not be empty"
    ):
      sync.h2d(checkpoint_path=" ")

    class _FakeSampler:

      def __init__(self, state: dict[str, jax.Array]) -> None:
        self.transformer_state = state
        self.model_runner = None

    sampler = _FakeSampler(
        {"layers.0.w": jax.device_put(np.zeros((2, 2), dtype=np.float32))}
    )
    sync.apply_to_sampler(sampler)
    np.testing.assert_allclose(
        np.asarray(sampler.transformer_state["layers.0.w"]), 1.0
    )

  def _make_tiny_qwen3_config(
      self, *, num_experts: int | None = None
  ) -> qwen3_model.ModelConfig:
    return qwen3_model.ModelConfig(
        num_layers=1,
        vocab_size=32,
        embed_dim=16,
        hidden_dim=32,
        num_heads=4,
        head_dim=4,
        num_kv_heads=2,
        rope_theta=10000,
        norm_eps=1e-6,
        use_tied_embedding=False,
        num_experts=num_experts,
        num_experts_per_tok=2 if num_experts else None,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )

  def _shard_state_on_mesh(
      self,
      state: nnx.State,
      mesh: jax.sharding.Mesh,
      axis_name: str,
      *,
      col_major: bool = False,
  ) -> nnx.State:
    axis_size = int(mesh.shape[axis_name])

    def _put(x: Any) -> Any:
      if not isinstance(x, jax.Array):
        return x
      if col_major and x.ndim >= 2 and x.shape[1] % axis_size == 0:
        spec = jax.sharding.PartitionSpec(None, axis_name)
      elif x.ndim >= 1 and x.shape[0] % axis_size == 0:
        spec = jax.sharding.PartitionSpec(axis_name)
      else:
        spec = jax.sharding.PartitionSpec()
      return jax.device_put(x, jax.sharding.NamedSharding(mesh, spec))

    return jax.tree.map(_put, state)

  def test_qwen3_manifest_ssot_dense_and_moe_roundtrip(self):
    devices = jax.devices()[:4]
    fsdp_mesh = jax.sharding.Mesh(np.array(devices), ("fsdp",))
    tp_mesh = jax.sharding.Mesh(np.array(devices), ("tp",))

    for num_experts in (None, 4):
      cfg = self._make_tiny_qwen3_config(num_experts=num_experts)
      to_hf = qwen3_params.build_to_hf_manifest(cfg)
      from_hf = qwen3_params.build_from_hf_manifest(cfg)
      with fsdp_mesh:
        src_model = qwen3_model.Qwen3(cfg, rngs=nnx.Rngs(1))
      with tp_mesh:
        dst_model = qwen3_model.Qwen3(cfg, rngs=nnx.Rngs(9))
      nnx.update(
          src_model,
          self._shard_state_on_mesh(
              nnx.state(src_model), fsdp_mesh, "fsdp", col_major=False
          ),
      )
      nnx.update(
          dst_model,
          self._shard_state_on_mesh(
              nnx.state(dst_model), tp_mesh, "tp", col_major=True
          ),
      )

      staging = self.create_tempdir().full_path
      src_sync = gcs_synchronizer.GCSSynchronizer(
          job_name="trainer",
          config=gcs_synchronizer.GCSSynchronizerConfig(
              staging_dir=staging,
              checkpoint_format="safetensors",
              max_bucket_bytes=2048,
              max_shard_bytes=16384,
              aot_compile=True,
          ),
          verify_weights=True,
          to_hf_manifest=to_hf,
      )
      src_sync.bind(nnx.state(src_model))
      req = datatypes.WeightSyncRequest(
          policy_version=1,
          extra_config={"req_id": f"q3_{num_experts or 'dense'}", "uuid": 1},
      )
      src_sync.d2h(req)
      src_meta = src_sync.work_unit_metadata()
      assert src_meta.artifact_uri is not None
      idx_path = epath.Path(src_meta.artifact_uri) / (
          "model.safetensors.index.json"
      )
      weight_map = json.loads(idx_path.read_text())["weight_map"]
      self.assertIn("model.layers.0.self_attn.q_proj.weight", weight_map)
      if num_experts:
        self.assertIn(
            "model.layers.0.mlp.experts.0.gate_proj.weight", weight_map
        )
      else:
        self.assertIn("model.layers.0.mlp.gate_proj.weight", weight_map)

      worker = _FakeTPUWorker(_NnxRunner(nnx.state(dst_model)))
      gcs_synchronizer.tpu_worker_bind_gcs_sync(
          worker,
          job_name="rollout",
          config=gcs_synchronizer.GCSSynchronizerConfig(),
          from_hf_manifest=from_hf,
      )
      # Exercise re-bind updating manifests on existing worker synchronizer.
      gcs_synchronizer.tpu_worker_bind_gcs_sync(
          worker,
          job_name="rollout",
          to_hf_manifest=to_hf,
          from_hf_manifest=from_hf,
      )
      gcs_synchronizer.tpu_worker_gcs_h2d(
          worker, src_meta.artifact_uri, source_checksums=src_meta.checksums
      )
      gcs_synchronizer.tpu_worker_gcs_commit(worker)

      src_leaves = jax.tree_util.tree_leaves(nnx.state(src_model))
      dst_leaves = jax.tree_util.tree_leaves(nnx.state(dst_model))
      for s_arr, d_arr in zip(src_leaves, dst_leaves, strict=True):
        np.testing.assert_allclose(np.asarray(d_arr), np.asarray(s_arr))
      src_sync.close()

  def test_index_based_host_shard_filtering(self):
    # pylint: disable=protected-access
    staging = self.create_tempdir().full_path
    moe_w = np.arange(4 * 64 * 64, dtype=np.float32).reshape(4, 64, 64)
    src_sync = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer",
        config=gcs_synchronizer.GCSSynchronizerConfig(
            staging_dir=staging,
            checkpoint_format="safetensors",
            max_bucket_bytes=16384,
            max_shard_bytes=None,
            per_wave_shards=True,
        ),
    )
    src_sync.bind({"layers.0.mlp.gate_proj": jax.device_put(moe_w)})
    src_sync.d2h(
        datatypes.WeightSyncRequest(
            policy_version=1, extra_config={"req_id": "flt", "uuid": 1}
        )
    )
    ckpt_uri = src_sync.work_unit_metadata().artifact_uri
    assert ckpt_uri is not None
    weight_map = gcs_synchronizer._load_safetensors_weight_map(ckpt_uri)
    self.assertLen(weight_map, 4)
    all_files = safetensors_loader.resolve_range_sources(ckpt_uri)
    self.assertLen(all_files, 4)

    src_map = gcs_synchronizer._build_safetensors_source_map(
        ["layers.0.mlp.gate_proj"], [jax.device_put(moe_w)]
    )
    partial_spec = safetensors_loader._TargetSpec(
        param_index=0,
        global_shape=(4, 64, 64),
        host_shape=(2, 64, 64),
        sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0]),
        host_slices=(slice(0, 2), slice(0, 64), slice(0, 64)),
        device_slices=(),
    )
    target_specs = {"layers.0.mlp.gate_proj": partial_spec}
    filtered = gcs_synchronizer._filter_host_shards(
        all_files, weight_map, src_map, frozenset(), target_specs
    )
    self.assertLen(filtered, 2)
    self.assertTrue(
        gcs_synchronizer._shard_needed_by_host(
            "comp.key", src_map, frozenset({"comp.key"}), target_specs
        )
    )
    self.assertFalse(
        gcs_synchronizer._shard_needed_by_host(
            "unknown.key", src_map, frozenset(), target_specs
        )
    )
    self.assertEqual(
        gcs_synchronizer._filter_host_shards(
            all_files, {}, src_map, frozenset(), target_specs
        ),
        list(all_files),
    )

    # Verify _restore_safetensors only inspects headers for filtered shards.
    inspected_names: list[str] = []
    orig_inspect = safetensors_loader.SpmdHostStager.inspect_file_items

    def _spy_inspect(f: Any, *args: Any, **kwargs: Any) -> Any:
      inspected_names.append(f.name)
      return orig_inspect(f, *args, **kwargs)

    with (
        mock.patch.object(
            safetensors_loader,
            "_build_target_specs",
            return_value=target_specs,
        ),
        mock.patch.object(
            safetensors_loader.SpmdHostStager,
            "inspect_file_items",
            side_effect=_spy_inspect,
        ),
        mock.patch.object(
            safetensors_loader.SpmdHostStager,
            "consume_one_file",
            side_effect=lambda *a, **kw: None,
        ),
    ):
      with self.assertRaises(KeyError):
        gcs_synchronizer._restore_safetensors(
            ckpt_uri, ["layers.0.mlp.gate_proj"], [jax.device_put(moe_w)]
        )
    self.assertLen(inspected_names, 2)
    src_sync.close()
    # pylint: enable=protected-access

  def test_aot_compile_metrics_and_plan_reuse(self):
    devices = jax.devices()[:4]
    mesh = jax.sharding.Mesh(np.array(devices), ("fsdp",))
    w = jax.device_put(
        np.arange(16, dtype=np.float32).reshape(4, 4),
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("fsdp")),
    )
    staging = self.create_tempdir().full_path
    sync = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer",
        config=gcs_synchronizer.GCSSynchronizerConfig(
            staging_dir=staging,
            checkpoint_format="safetensors",
            aot_compile=True,
        ),
        to_hf_manifest={"rules": []},
    )
    with mock.patch.object(
        safetensors_saver,
        "compile_save_plan",
        wraps=safetensors_saver.compile_save_plan,
    ) as spy_compile:
      sync.bind({"layers.0.w": w})
      self.assertEqual(spy_compile.call_count, 1)
      bind_metrics = sync.metrics()
      self.assertGreaterEqual(bind_metrics["aot_num_kernels"], 1)
      self.assertGreaterEqual(bind_metrics["aot_compile_secs"], 0.0)
      self.assertIsNotNone(sync.last_validation_report)

      for step in (1, 2):
        w_updated = jax.device_put(
            np.full((4, 4), float(step), dtype=np.float32),
            jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec("fsdp")
            ),
        )
        sync.bind({"layers.0.w": w_updated})
        sync.d2h(
            datatypes.WeightSyncRequest(
                policy_version=step,
                extra_config={"req_id": f"s{step}", "uuid": step},
            )
        )
      # Re-bind with same layout + second d2h must reuse cached AOT SavePlan.
      self.assertEqual(spy_compile.call_count, 1)
    post_metrics = sync.metrics()
    self.assertEqual(post_metrics["d2h_count"], 2)
    self.assertGreaterEqual(post_metrics["last_export_shard_count"], 1)
    self.assertGreaterEqual(post_metrics["last_export_hbm_secs"], 0.0)
    self.assertGreaterEqual(post_metrics["last_export_d2h_secs"], 0.0)
    self.assertGreaterEqual(post_metrics["last_export_disk_secs"], 0.0)

    sync.close()
    self.assertFalse(sync.bound)
    self.assertEqual(sync.metrics()["aot_num_kernels"], 0)

    # Identity-mode reader carrying staging_dir with aot_compile=True must skip
    # eager AOT compilation during bind().
    id_reader = gcs_synchronizer.GCSSynchronizer(
        job_name="rollout",
        config=gcs_synchronizer.GCSSynchronizerConfig(
            staging_dir=staging, aot_compile=True
        ),
    )
    with mock.patch.object(
        safetensors_saver, "compile_save_plan"
    ) as mock_reader_compile:
      id_reader.bind({"layers.0.w": w})
      mock_reader_compile.assert_not_called()
    id_reader.close()

  def test_auto_stage_pool_and_remote_read_config(self):
    # pylint: disable=protected-access
    self.assertEqual(gcs_synchronizer._resolve_write_procs("/tmp/out", None), 1)
    self.assertEqual(gcs_synchronizer._resolve_write_procs("gs://b/p", 24), 24)
    self.assertEqual(
        gcs_synchronizer._resolve_write_procs("/GOOGLE_INTERNAL_STOAGE_PATH/cell-d/out", None), 16
    )
    with mock.patch.dict(os.environ, {"TUNIX_GCS_WRITE_PROCS": "20"}):
      self.assertEqual(
          gcs_synchronizer._resolve_write_procs("gs://b/p", None), 20
      )
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop("TUNIX_GCS_WRITE_PROCS", None)
      self.assertGreaterEqual(
          gcs_synchronizer._resolve_write_procs("gs://b/p", None), 16
      )

    self.assertEqual(
        gcs_synchronizer._resolve_read_procs("/tmp/out", None, 2 << 30), 0
    )
    self.assertEqual(
        gcs_synchronizer._resolve_read_procs("gs://b/p", 12, 1024), 12
    )
    self.assertEqual(
        gcs_synchronizer._resolve_read_procs("/GOOGLE_INTERNAL_STOAGE_PATH/cell-d/out", 10, 1024), 10
    )
    with mock.patch.dict(os.environ, {"TUNIX_CNS_PROCS": "14"}):
      self.assertEqual(
          gcs_synchronizer._resolve_read_procs(
              "/GOOGLE_INTERNAL_STOAGE_PATH/cell-d/out", None, 2 << 30
          ),
          14,
      )
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop("TUNIX_CNS_PROCS", None)
      self.assertEqual(
          gcs_synchronizer._resolve_read_procs("gs://b/p", None, 1024), 0
      )
      self.assertGreaterEqual(
          gcs_synchronizer._resolve_read_procs("gs://b/p", None, 2 << 30), 16
      )
      with mock.patch("os.cpu_count", return_value=192):
        # 0.6B (~1.2 GiB) scales down to 9 procs; 397B (~786 GiB) scales to 96.
        self.assertEqual(
            gcs_synchronizer._resolve_read_procs("gs://b/p", None, 1200 << 20),
            9,
        )
        self.assertEqual(
            gcs_synchronizer._resolve_read_procs("gs://b/p", None, 786 << 30),
            96,
        )
      self.assertEqual(
          gcs_synchronizer._resolve_read_procs(
              "/GOOGLE_INTERNAL_STOAGE_PATH/cell-d/out", None, 2 << 30
          ),
          16,
      )
      self.assertEqual(
          gcs_synchronizer._resolve_read_procs("/GOOGLE_INTERNAL_STOAGE_PATH/cell-d/out", None, 1024), 0
      )

    self.assertEqual(gcs_synchronizer._compute_read_arena_budget(0), 0)
    self.assertGreaterEqual(
        gcs_synchronizer._compute_read_arena_budget(1 << 20), 64 << 20
    )
    with mock.patch(
        "os.statvfs",
        return_value=mock.MagicMock(f_bavail=900 << 30, f_frsize=1),
    ):
      self.assertEqual(
          gcs_synchronizer._compute_read_arena_budget(1 << 30), 4 << 30
      )
      self.assertEqual(
          gcs_synchronizer._compute_read_arena_budget(370 << 30), 192 << 30
      )

    # Persistent remote read arena: local paths return None; remote paths reuse
    # one bounded ShmRangeReader across syncs and unlink cleanly on close().
    read_sync = gcs_synchronizer.GCSSynchronizer(job_name="rollout")
    self.assertIsNone(read_sync._ensure_read_arena("/tmp/local_ckpt", 1 << 20))
    fake_arena_1 = mock.MagicMock(mm=object())
    fake_arena_2 = mock.MagicMock(mm=object())
    with mock.patch.object(
        safetensors_loader,
        "ShmRangeReader",
        side_effect=[fake_arena_1, fake_arena_2],
    ) as mock_arena_cls:
      a1 = read_sync._ensure_read_arena("gs://bucket/step_1", 1 << 20)
      a1_again = read_sync._ensure_read_arena("gs://bucket/step_2", 1 << 20)
      self.assertIs(a1, fake_arena_1)
      self.assertIs(a1_again, fake_arena_1)
      self.assertEqual(mock_arena_cls.call_count, 1)
      with gcs_synchronizer._open_read_arena(
          1 << 20, lambda hb: read_sync._ensure_read_arena("gs://bucket/s3", hb)
      ) as yielded:
        self.assertIs(yielded, fake_arena_1)
      fake_arena_1.__exit__.assert_not_called()
      # Larger budget replaces old arena cleanly via __exit__.
      a2 = read_sync._ensure_read_arena("gs://bucket/step_4", 64 << 20)
      self.assertIs(a2, fake_arena_2)
      fake_arena_1.__exit__.assert_called_once_with(None, None, None)
      read_sync.close()
      fake_arena_2.__exit__.assert_called_once_with(None, None, None)

    # resolve_safetensors_files / _resolve_safetensors_layout: index, single.
    probe_root = epath.Path(self.create_tempdir().full_path)
    idx_dir = probe_root / "idx_ckpt"
    idx_dir.mkdir()
    (idx_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"w": "model-00001.safetensors"}})
    )
    self.assertEqual(
        gcs_synchronizer._resolve_safetensors_layout(idx_dir),
        (["model-00001.safetensors"], {"w": "model-00001.safetensors"}),
    )
    single_dir = probe_root / "single_ckpt"
    single_dir.mkdir()
    (single_dir / "model.safetensors").write_bytes(b"")
    self.assertEqual(
        gcs_synchronizer.resolve_safetensors_files(single_dir),
        [single_dir / "model.safetensors"],
    )
    empty_dir = probe_root / "empty_ckpt"
    empty_dir.mkdir()
    with self.assertRaises(FileNotFoundError):
      gcs_synchronizer.resolve_safetensors_files(empty_dir)

    # Persistent MultiprocStagePool creation, reuse, replacement, and close.
    sync = gcs_synchronizer.GCSSynchronizer(job_name="trainer")
    self.assertIsNone(sync._ensure_stage_pool("/tmp/local_ckpt"))
    fake_pool_1 = mock.MagicMock()
    fake_pool_1.activate.return_value = contextlib.nullcontext()
    fake_pool_2 = mock.MagicMock()
    with mock.patch.object(
        safetensors_saver,
        "MultiprocStagePool",
        side_effect=[fake_pool_1, fake_pool_2],
    ) as mock_pool_cls:
      sync.config = gcs_synchronizer.GCSSynchronizerConfig(write_procs=8)
      p1 = sync._ensure_stage_pool("gs://bucket/step_1")
      p1_again = sync._ensure_stage_pool("gs://bucket/step_2")
      self.assertIs(p1, fake_pool_1)
      self.assertIs(p1_again, fake_pool_1)
      self.assertEqual(mock_pool_cls.call_count, 1)
      self.assertEqual(mock_pool_cls.call_args.kwargs["shm_budget_bytes"], 0)
      sync.arrays = [np.zeros((1024, 1024), dtype=np.float32)]
      sync.config = gcs_synchronizer.GCSSynchronizerConfig(write_procs=16)
      p2 = sync._ensure_stage_pool("gs://bucket/step_3")
      self.assertIs(p2, fake_pool_2)
      self.assertEqual(
          mock_pool_cls.call_args.kwargs["shm_budget_bytes"], 4 << 30
      )
      fake_pool_1.close.assert_called_once()
      sync.close()
      fake_pool_2.close.assert_called_once()

    # 1-pass _restore_safetensors builds CNS RangeSources from validated names
    # (no glob), spawns CNS workers directly, and forwards arena + read_procs.
    fake_w = jax.device_put(np.ones((4, 4), dtype=np.float32))

    def _fill_params(*args: Any, **kwargs: Any) -> None:
      del kwargs
      params_arg = args[7]
      params_arg[0] = fake_w

    with (
        mock.patch.dict(os.environ, {"TUNIX_CNS_PROCS": " "}),
        mock.patch.object(safetensors_io.CnsPath, "configure_flags"),
        mock.patch.object(safetensors_io.CnsPath, "spawn_procs") as mock_spawn,
        mock.patch.object(
            safetensors_loader, "stream_load_files", side_effect=_fill_params
        ) as mock_stream,
    ):
      restored, total_s, filt_s = gcs_synchronizer._restore_safetensors(
          "/GOOGLE_INTERNAL_STOAGE_PATH/cell-d/ckpt",
          ["w1"],
          [fake_w],
          read_procs=16,
          weight_map={"w1": "model-00001.safetensors"},
      )
      self.assertIn("w1", restored)
      self.assertEqual((total_s, filt_s), (1, 1))
      mock_spawn.assert_called_once_with(16)
      self.assertEqual(mock_stream.call_args.kwargs["read_procs"], 16)
      self.assertIsNotNone(mock_stream.call_args.kwargs["arena"])
    # pylint: enable=protected-access

  def test_validate_weight_sync_runtime_errors_warnings_and_strict(self):
    cfg = gcs_synchronizer.GCSSynchronizerConfig(read_procs=1)
    bad_manifest = {
        "rules": [{"source_pattern": ["a"], "target_pattern": ["b"]}]
    }
    w = jax.device_put(np.ones((4, 4), dtype=np.float32))
    high_hbm_plan = safetensors_saver.CompiledSavePlan(
        executables={},
        peak_temp_hbm_bytes=cfg.max_bucket_bytes * 5,
        compile_time_sec=0.1,
        num_unique_kernels=1,
    )
    unmatched_to = {
        "rules": [{
            "source_keys": ["other.w"],
            "target_keys": ["hf.other.w"],
            "ops": [],
        }]
    }
    unmatched_from = {
        "rules": [{
            "source_keys": ["hf.other.w"],
            "target_keys": ["other.w"],
            "ops": [],
        }]
    }
    report = gcs_synchronizer.validate_weight_sync_runtime(
        cfg,
        names=["layers.0.w"],
        arrays=[w],
        to_hf_manifest=unmatched_to,
        from_hf_manifest=unmatched_from,
        compiled_plan=high_hbm_plan,
        strict=False,
    )
    self.assertFalse(report.ok)
    self.assertTrue(any("to_hf_manifest missing" in e for e in report.errors))
    self.assertTrue(any("from_hf_manifest missing" in e for e in report.errors))
    self.assertTrue(
        any("reader parallelism is pinned <= 1" in w for w in report.warnings)
    )
    self.assertTrue(any("peak_temp_hbm_bytes" in w for w in report.warnings))

    with mock.patch.dict(os.environ, {"TUNIX_CNS_PROCS": "1"}, clear=False):
      GOOGLE_INTERNAL_STOAGE_PATH_rep = gcs_synchronizer.validate_weight_sync_runtime(
          gcs_synchronizer.GCSSynchronizerConfig(staging_dir="/GOOGLE_INTERNAL_STOAGE_PATH/cell-d/out"),
          strict=False,
      )
      self.assertTrue(
          any(
              "reader parallelism is pinned <= 1" in msg
              for msg in GOOGLE_INTERNAL_STOAGE_PATH_rep.warnings
          )
      )
      for non_GOOGLE_INTERNAL_STOAGE_PATH_dir in ("gs://bkt/out", None):
        non_GOOGLE_INTERNAL_STOAGE_PATH_rep = gcs_synchronizer.validate_weight_sync_runtime(
            gcs_synchronizer.GCSSynchronizerConfig(staging_dir=non_GOOGLE_INTERNAL_STOAGE_PATH_dir),
            strict=False,
        )
        self.assertFalse(
            any(
                "reader parallelism is pinned <= 1" in msg
                for msg in non_GOOGLE_INTERNAL_STOAGE_PATH_rep.warnings
            ),
            non_GOOGLE_INTERNAL_STOAGE_PATH_rep.warnings,
        )

    # bind() with AOT compile must not duplicate base warnings.
    tmpdir = self.create_tempdir().full_path
    sync = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer",
        config=gcs_synchronizer.GCSSynchronizerConfig(
            staging_dir=tmpdir,
            read_procs=1,
            aot_compile=True,
        ),
        to_hf_manifest={"rules": []},
    )
    sync.bind({"layers": {"0": {"w": w}}})
    rep = sync.last_validation_report
    self.assertIsNotNone(rep)
    assert rep is not None
    self.assertEqual(
        sum("reader parallelism" in msg for msg in rep.warnings), 1
    )
    sync.close()

    invalid_rep = gcs_synchronizer.validate_weight_sync_runtime(
        gcs_synchronizer.GCSSynchronizerConfig(),
        names=["layers.0.w"],
        arrays=[w],
        to_hf_manifest=bad_manifest,
        from_hf_manifest=bad_manifest,
        strict=False,
    )
    self.assertFalse(invalid_rep.ok)

    with self.assertRaisesRegex(ValueError, "reader parallelism"):
      gcs_synchronizer.validate_weight_sync_runtime(cfg, strict=True)

  def test_hardening_arena_reuse_wave_groups_transactional_bind_and_gc(self):
    # pylint: disable=protected-access
    # 1. Warm read arena must NOT be recreated on Sync 2 when `/dev/shm` free
    #    bytes shrink by the arena's own faulted footprint.
    read_sync = gcs_synchronizer.GCSSynchronizer(job_name="rollout")
    fake_arena = mock.MagicMock(mm=object())
    stat_calls = [
        mock.MagicMock(f_bavail=900 << 30, f_frsize=1),
        mock.MagicMock(f_bavail=300 << 30, f_frsize=1),
    ]
    with (
        mock.patch("os.statvfs", side_effect=stat_calls) as mock_statvfs,
        mock.patch.object(
            safetensors_loader, "ShmRangeReader", return_value=fake_arena
        ) as mock_arena_cls,
    ):
      a1 = read_sync._ensure_read_arena("gs://bucket/step_1", 370 << 30)
      a2 = read_sync._ensure_read_arena("gs://bucket/step_2", 370 << 30)
      self.assertIs(a1, fake_arena)
      self.assertIs(a2, fake_arena)
      self.assertEqual(mock_statvfs.call_count, 1)
      self.assertEqual(mock_arena_cls.call_count, 1)
      # Larger host_bytes hitting the same capped budget also reuses warm arena.
      a3 = read_sync._ensure_read_arena("gs://bucket/step_3", 380 << 30)
      self.assertIs(a3, fake_arena)
      self.assertEqual(mock_arena_cls.call_count, 1)
    read_sync.close()

    # 2. _ensure_stage_pool closes old pool when switching to procs <= 1 or
    #    changing max_bucket_bytes.
    pool_sync = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer",
        config=gcs_synchronizer.GCSSynchronizerConfig(
            write_procs=8, max_bucket_bytes=1 << 20
        ),
    )
    fp1, fp2 = mock.MagicMock(), mock.MagicMock()
    with mock.patch.object(
        safetensors_saver, "MultiprocStagePool", side_effect=[fp1, fp2]
    ):
      self.assertIs(pool_sync._ensure_stage_pool("gs://b/s1"), fp1)
      pool_sync.config = gcs_synchronizer.GCSSynchronizerConfig(
          write_procs=8, max_bucket_bytes=2 << 20
      )
      self.assertIs(pool_sync._ensure_stage_pool("gs://b/s2"), fp2)
      fp1.close.assert_called_once()
      pool_sync.config = gcs_synchronizer.GCSSynchronizerConfig(write_procs=1)
      self.assertIsNone(pool_sync._ensure_stage_pool("gs://b/s3"))
      fp2.close.assert_called_once()

    # 3. Validation respects to_hf dropped_keys, from_hf tied_keys (while
    #    rejecting bound keys listed only in from_hf dropped_keys), and
    #    TUNIX_GCS_WRITE_PROCS.
    w = jax.device_put(np.ones((4, 4), dtype=np.float32))
    to_m = {
        "version": 1,
        "model_type": "test",
        "rules": [{
            "source_keys": ["w1"],
            "target_keys": ["hf.w1"],
            "ops": [],
        }],
        "dropped_keys": ["w_drop", "w_tied"],
        "tied_keys": [],
    }
    from_m = {
        "version": 1,
        "model_type": "test",
        "rules": [{
            "source_keys": ["hf.w1"],
            "target_keys": ["w1"],
            "ops": [],
        }],
        "dropped_keys": ["hf.unused_vision_key"],
        "tied_keys": [["w1", "w_tied"]],
    }
    with mock.patch.dict(os.environ, {"TUNIX_GCS_WRITE_PROCS": "1"}):
      rep = gcs_synchronizer.validate_weight_sync_runtime(
          gcs_synchronizer.GCSSynchronizerConfig(staging_dir="gs://b/p"),
          names=["w1", "w_tied"],
          arrays=[w, w],
          to_hf_manifest=to_m,
          from_hf_manifest=from_m,
          strict=False,
      )
      self.assertTrue(
          any(
              "write_procs" in msg or "single-process" in msg.lower()
              for msg in rep.warnings
          ),
          rep.warnings,
      )
      self.assertFalse(any("w_tied" in e for e in rep.errors), rep.errors)
      # Bound Tunix key only in from_hf dropped_keys must fail validation since
      # dropped_keys are ignored HF sources and would KeyError on h2d().
      bad_from_rep = gcs_synchronizer.validate_weight_sync_runtime(
          gcs_synchronizer.GCSSynchronizerConfig(staging_dir="gs://b/p"),
          names=["w1", "hf.unused_vision_key"],
          arrays=[w, w],
          from_hf_manifest=from_m,
          strict=False,
      )
      self.assertTrue(
          any("hf.unused_vision_key" in e for e in bad_from_rep.errors),
          bad_from_rep.errors,
      )

    # 5. Transactional bind() rolls back on invalid re-bind; read-only rollout
    #    workers skip eager stage-pool, writer validation & AOT compilation.
    tmpdir = self.create_tempdir().full_path
    tx_sync = gcs_synchronizer.GCSSynchronizer(
        job_name="rollout",
        config=gcs_synchronizer.GCSSynchronizerConfig(
            staging_dir=tmpdir, aot_compile=True
        ),
        from_hf_manifest=from_m,
    )
    with mock.patch.object(
        safetensors_saver, "compile_save_plan"
    ) as mock_compile:
      tx_sync.bind({"w1": w, "w_tied": w})
      mock_compile.assert_not_called()
    self.assertTrue(tx_sync.bound)
    prev_names = list(tx_sync.names)
    prev_report = tx_sync.last_validation_report
    with self.assertRaisesRegex(ValueError, "rollout.*from_hf_manifest"):
      tx_sync.bind({"unmapped_key": w})
    self.assertTrue(tx_sync.bound)
    self.assertEqual(tx_sync.names, prev_names)
    self.assertIs(tx_sync.last_validation_report, prev_report)
    tx_sync.close()

    # 6. _prepare_save_plan hashes manifest content (not object id()).
    plan_sync = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer",
        config=gcs_synchronizer.GCSSynchronizerConfig(
            staging_dir=tmpdir, aot_compile=True
        ),
        to_hf_manifest=json.loads(json.dumps(to_m)),
    )
    plan_sync.bind({"w1": w, "w_drop": w})
    with mock.patch.object(
        safetensors_saver, "compile_save_plan"
    ) as mock_compile2:
      plan_sync._to_hf_manifest = json.loads(json.dumps(to_m))
      plan_sync._prepare_save_plan(tmpdir)
      mock_compile2.assert_not_called()
    plan_sync.close()

    # 7. TPUWorker GC finalizer closes bound GCSSynchronizer automatically.
    worker = _FakeTPUWorker(_NnxRunner({"w1": w}))
    gcs_synchronizer.tpu_worker_bind_gcs_sync(worker, job_name="gc_rollout")
    bound_sync = gcs_synchronizer.require_gcs_worker_sync(worker, "test")
    gc_arena = mock.MagicMock(mm=object())
    bound_sync._read_arena = gc_arena
    del bound_sync
    del worker
    gc.collect()
    gc_arena.__exit__.assert_called_once_with(None, None, None)

    # 8. Negative TUNIX_CNS_PROCS raises ValueError; arena budget does not
    #    double-discount available_shm_bytes() headroom.
    with mock.patch.dict(os.environ, {"TUNIX_CNS_PROCS": "-2"}):
      with self.assertRaisesRegex(ValueError, "TUNIX_CNS_PROCS"):
        gcs_synchronizer._resolve_read_procs("/GOOGLE_INTERNAL_STOAGE_PATH/cell-d/out", None, 2 << 30)
    with mock.patch(
        "os.statvfs",
        return_value=mock.MagicMock(f_bavail=100 << 30, f_frsize=1),
    ):
      # 80% of 100 GiB = 80 GiB (single discount, not 64 GiB).
      self.assertEqual(
          gcs_synchronizer._compute_read_arena_budget(50 << 30), 80 << 30
      )

    # 9. Unhealthy stage pool is closed & respawned; monotonic stage_dir avoids
    #    id() reuse collisions; Identity bind() skips eager stage-pool spawn.
    unhealthy_pool = mock.MagicMock()
    unhealthy_pool.is_healthy.return_value = False
    healthy_pool = mock.MagicMock()
    healthy_pool.is_healthy.return_value = True
    respawn_sync = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer",
        config=gcs_synchronizer.GCSSynchronizerConfig(write_procs=4),
    )
    with mock.patch.object(
        safetensors_saver,
        "MultiprocStagePool",
        side_effect=[unhealthy_pool, healthy_pool],
    ) as mock_pool_cls:
      self.assertIs(
          respawn_sync._ensure_stage_pool("gs://b/s1"), unhealthy_pool
      )
      self.assertIs(respawn_sync._ensure_stage_pool("gs://b/s2"), healthy_pool)
      unhealthy_pool.close.assert_called_once()
      dirs = [c.kwargs["stage_dir"] for c in mock_pool_cls.call_args_list]
      self.assertNotEqual(dirs[0], dirs[1])
    respawn_sync.close()

    id_sync = gcs_synchronizer.GCSSynchronizer(
        job_name="rollout",
        config=gcs_synchronizer.GCSSynchronizerConfig(
            staging_dir="gs://b/p", write_procs=4, aot_compile=False
        ),
    )
    with mock.patch.object(
        safetensors_saver, "MultiprocStagePool"
    ) as mock_id_pool:
      id_sync.bind({"w1": w})
      mock_id_pool.assert_not_called()
    id_sync.close()

    # 10. close() isolates teardown failures so prune executor shuts down and
    #     bound state is always cleared.
    err_sync = gcs_synchronizer.GCSSynchronizer(job_name="trainer")
    err_sync.bind({"w1": w})
    bad_pool = mock.MagicMock()
    bad_pool.close.side_effect = OSError("pool close boom")
    mock_prune = mock.MagicMock()
    err_sync._stage_pool = bad_pool
    err_sync._prune_executor = mock_prune
    with self.assertRaisesRegex(OSError, "pool close boom"):
      err_sync.close()
    mock_prune.shutdown.assert_called_once_with(wait=True)
    self.assertFalse(err_sync.bound)
    self.assertIsNone(err_sync._stage_pool)
    self.assertIsNone(err_sync._prune_executor)

    # 11. tpu_worker_bind_gcs_sync rolls back on bind() failure (both initial
    #     and re-bind), and empty proc env vars fall back to defaults.
    bad_worker = _FakeTPUWorker(_NnxRunner({}))
    with self.assertRaisesRegex(ValueError, "no bindable"):
      gcs_synchronizer.tpu_worker_bind_gcs_sync(bad_worker, job_name="rollout")
    self.assertNotIn(bad_worker, gcs_synchronizer._WORKER_SYNCHRONIZERS)

    good_worker = _FakeTPUWorker(_NnxRunner({"w1": w}))
    orig_cfg = gcs_synchronizer.GCSSynchronizerConfig(write_procs=4)
    gcs_synchronizer.tpu_worker_bind_gcs_sync(
        good_worker, job_name="rollout", config=orig_cfg, worker_index=1
    )
    good_sync = gcs_synchronizer.require_gcs_worker_sync(good_worker, "test")
    good_worker.model_runner.state = {}
    new_cfg = gcs_synchronizer.GCSSynchronizerConfig(write_procs=8)
    with self.assertRaisesRegex(ValueError, "no bindable"):
      gcs_synchronizer.tpu_worker_bind_gcs_sync(
          good_worker,
          job_name="mutated",
          config=new_cfg,
          worker_index=9,
          to_hf_manifest={"rules": []},
          from_hf_manifest={"rules": []},
      )
    self.assertEqual(good_sync.job_name, "rollout")
    self.assertEqual(good_sync.worker_index, 1)
    self.assertIs(good_sync.config, orig_cfg)
    self.assertIsNone(good_sync._to_hf_manifest)
    self.assertIsNone(good_sync._from_hf_manifest)
    good_worker.model_runner.state = {"w1": w}
    with self.assertRaisesRegex(ValueError, "Manifest must be a mapping"):
      gcs_synchronizer.tpu_worker_bind_gcs_sync(
          good_worker, job_name="bad_rebind", from_hf_manifest="{}"
      )
    self.assertEqual(good_sync.job_name, "rollout")
    self.assertIsNone(good_sync._from_hf_manifest)
    good_sync.close()
    gcs_synchronizer._WORKER_SYNCHRONIZERS.pop(good_worker, None)

    with mock.patch.dict(
        os.environ, {"TUNIX_GCS_WRITE_PROCS": " ", "TUNIX_CNS_PROCS": ""}
    ):
      self.assertGreaterEqual(
          gcs_synchronizer._resolve_write_procs("gs://b/p", None), 16
      )
      self.assertEqual(
          gcs_synchronizer._resolve_read_procs("/GOOGLE_INTERNAL_STOAGE_PATH/cell-d/p", None, 2 << 30),
          16,
      )
    # pylint: enable=protected-access

  def test_stage_pool_GOOGLE_INTERNAL_STOAGE_PATH_vs_gcs_isolation(self):
    # pylint: disable=protected-access
    sync = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer",
        config=gcs_synchronizer.GCSSynchronizerConfig(write_procs=8),
    )
    gcs_pool = mock.MagicMock()
    gcs_pool.is_healthy.return_value = True
    GOOGLE_INTERNAL_STOAGE_PATH_pool = mock.MagicMock()
    GOOGLE_INTERNAL_STOAGE_PATH_pool.is_healthy.return_value = True
    with mock.patch.object(
        safetensors_saver,
        "MultiprocStagePool",
        side_effect=[gcs_pool, GOOGLE_INTERNAL_STOAGE_PATH_pool],
    ) as mock_pool_cls:
      self.assertIs(sync._ensure_stage_pool("gs://bkt/step_1"), gcs_pool)
      self.assertTrue(
          mock_pool_cls.call_args_list[0].kwargs["use_gcs_http_write"]
      )
      # Switching to /GOOGLE_INTERNAL_STOAGE_PATH/ must close the GCS pool and spawn a CNS-only pool.
      self.assertIs(sync._ensure_stage_pool("/GOOGLE_INTERNAL_STOAGE_PATH/cell-d/step_1"), GOOGLE_INTERNAL_STOAGE_PATH_pool)
      gcs_pool.close.assert_called_once()
      self.assertFalse(
          mock_pool_cls.call_args_list[1].kwargs["use_gcs_http_write"]
      )
      # Subsequent /GOOGLE_INTERNAL_STOAGE_PATH/ target reuses the CNS pool without respawning.
      self.assertIs(sync._ensure_stage_pool("/GOOGLE_INTERNAL_STOAGE_PATH/cell-d/step_2"), GOOGLE_INTERNAL_STOAGE_PATH_pool)
      self.assertEqual(mock_pool_cls.call_count, 2)
    sync.close()
    GOOGLE_INTERNAL_STOAGE_PATH_pool.close.assert_called_once()

    with mock.patch.dict(
        os.environ, {"TUNIX_GCS_WRITE_PROCS": "1"}, clear=False
    ):
      GOOGLE_INTERNAL_STOAGE_PATH_rep = gcs_synchronizer.validate_weight_sync_runtime(
          gcs_synchronizer.GCSSynchronizerConfig(staging_dir="/GOOGLE_INTERNAL_STOAGE_PATH/cell-d/out"),
          strict=False,
      )
      self.assertFalse(
          any("write_procs" in msg for msg in GOOGLE_INTERNAL_STOAGE_PATH_rep.warnings),
          GOOGLE_INTERNAL_STOAGE_PATH_rep.warnings,
      )
    # Review hardening: Orbax `default/` probe parity, word-boundary 404 match,
    # read-only reader skipping writer staging_dir checks, and host_bytes=0.
    probe_dir = epath.Path(self.create_tempdir().full_path)
    (probe_dir / "default").mkdir()
    self.assertIsNone(gcs_synchronizer.resolve_safetensors_files(probe_dir))
    with mock.patch.object(
        epath.Path, "read_text", side_effect=RuntimeError("port 4040 timeout")
    ):
      with self.assertRaisesRegex(RuntimeError, "4040"):
        gcs_synchronizer._load_safetensors_weight_map("gs://b/k")
    with mock.patch.object(
        epath.Path, "read_text", side_effect=RuntimeError("HTTP 404 not found")
    ):
      self.assertEqual(
          gcs_synchronizer._load_safetensors_weight_map("gs://b/k"), {}
      )
    ro_sync = gcs_synchronizer.GCSSynchronizer(
        job_name="ro_rollout",
        config=gcs_synchronizer.GCSSynchronizerConfig(
            staging_dir="gs://b/unused", write_procs=1, strict_validation=True
        ),
        from_hf_manifest={
            "version": 1,
            "model_type": "test",
            "rules": [{"source_keys": ["h"], "target_keys": ["w"], "ops": []}],
            "dropped_keys": [],
            "tied_keys": [],
        },
    )
    self.assertIsNone(ro_sync._ensure_read_arena("gs://b/s1", 0))
    ro_sync.close()
    # pylint: enable=protected-access

  def test_safetensors_h2d_error_diagnostics_and_stray_shard_exclusion(self):
    # pylint: disable=protected-access
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
    with self.assertRaisesRegex(
        FileNotFoundError, "No finalized safetensors or Orbax checkpoint"
    ):
      dst_missing.h2d(checkpoint_path=empty_dir)
    with self.assertRaisesRegex(KeyError, "Missing 1 weight tensor"):
      dst_missing.h2d(checkpoint_path=ckpt_uri)

    # Uncommitted shard without index.json or model.safetensors must fail.
    uncommitted_dir = epath.Path(self.create_tempdir().full_path)
    (uncommitted_dir / "model-00001-of-00002.safetensors").write_bytes(b"")
    with self.assertRaisesRegex(
        FileNotFoundError, "No finalized safetensors or Orbax checkpoint"
    ):
      dst_missing.h2d(checkpoint_path=str(uncommitted_dir))

    # Staging `.tmp` directories are rejected even if a marker file exists.
    tmp_dir = epath.Path(self.create_tempdir().full_path) / "step_1.tmp"
    tmp_dir.mkdir()
    (tmp_dir / "model.safetensors").write_bytes(b"")
    with self.assertRaisesRegex(FileNotFoundError, "not finalized"):
      dst_missing.h2d(checkpoint_path=str(tmp_dir))

    # Index missing `weight_map`, empty `weight_map`, or traversal must fail.
    bad_idx_dir = epath.Path(self.create_tempdir().full_path)
    (bad_idx_dir / "model.safetensors.index.json").write_text('{"metadata":{}}')
    with self.assertRaisesRegex(ValueError, "weight_map"):
      dst_missing.h2d(checkpoint_path=str(bad_idx_dir))
    (bad_idx_dir / "model.safetensors.index.json").write_text(
        '{"weight_map":{}}'
    )
    with self.assertRaisesRegex(FileNotFoundError, "No safetensors shards"):
      dst_missing.h2d(checkpoint_path=str(bad_idx_dir))

    for bad_shard in (
        "../escape.safetensors",
        "/tmp/abs.safetensors",
        "sub/shard.safetensors",
        r"..\escape.safetensors",
        "",
    ):
      with self.subTest(bad_shard=bad_shard):
        (bad_idx_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"layers.0.w": bad_shard}})
        )
        with self.assertRaisesRegex(ValueError, "Invalid shard filename"):
          dst_missing.h2d(checkpoint_path=str(bad_idx_dir))

    dst_bad_shape = gcs_synchronizer.GCSSynchronizer(job_name="rollout")
    dst_bad_shape.bind(
        {"layers.0.w": jax.device_put(np.zeros((4, 2), np.float32))}
    )
    with self.assertRaisesRegex(ValueError, "Shape mismatch"):
      dst_bad_shape.h2d(checkpoint_path=ckpt_uri)

    # `_filter_host_shards` must drop stray unindexed shards.
    indexed = types.SimpleNamespace(name="model-00001-of-00001.safetensors")
    stray = types.SimpleNamespace(name="stray-uncommitted.safetensors")
    src_map = gcs_synchronizer._build_safetensors_source_map(
        ["layers.0.w"], [jax.device_put(np.zeros((2, 2), np.float32))]
    )
    target_specs = safetensors_loader._build_target_specs(
        ["layers.0.w"],
        [jax.ShapeDtypeStruct((2, 2), np.float32)],
        [jax.local_devices()[0]],
    )
    kept = gcs_synchronizer._filter_host_shards(
        [indexed, stray],
        {"layers.0.w": "model-00001-of-00001.safetensors"},
        src_map,
        frozenset(),
        target_specs,
    )
    self.assertEqual([f.name for f in kept], [indexed.name])
    src_sync.close()
    dst_missing.close()
    dst_bad_shape.close()
    # pylint: enable=protected-access

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
    src_sync.close()
    dst_sync.close()

    # Real 2-process jax.distributed safetensors d2h -> h2d round-trip.
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
      proc.join(timeout=60)
      self.assertEqual(proc.exitcode, 0)
    mh_results = sorted([out_q.get(), out_q.get()])
    self.assertEqual(mh_results[0][0], 0)
    self.assertEqual(mh_results[1][0], 1)
    self.assertEqual(mh_results[0][2], [[0.0], [4.0], [8.0], [12.0]])

  def test_multihost_h2d_error_broadcast_and_asymmetric_empty_files(self):
    # pylint: disable=protected-access
    staging = epath.Path(self.create_tempdir().full_path)
    missing_dir = str(staging / "nonexistent_step")
    dst = gcs_synchronizer.GCSSynchronizer(job_name="rollout")
    dst.bind({"layers.0.w": jax.device_put(np.zeros((2, 2), np.float32))})

    shared_status: dict[str, Any] = {}

    def _capture_broadcast(payload: Any) -> Any:
      if payload is not None and not shared_status:
        shared_status["value"] = payload
      return shared_status["value"]

    for rank in (0, 1):
      with (
          mock.patch.object(jax, "process_count", return_value=2),
          mock.patch.object(jax, "process_index", return_value=rank),
          mock.patch.object(
              gcs_synchronizer.sharding_utils,
              "broadcast_host0_json",
              side_effect=_capture_broadcast,
          ),
          mock.patch.object(
              safetensors_loader,
              "stream_load_files",
              side_effect=AssertionError("must not enter stream_load_files"),
          ),
      ):
        with self.assertRaisesRegex(
            FileNotFoundError, "No finalized safetensors or Orbax checkpoint"
        ):
          dst.h2d(checkpoint_path=missing_dir)

    corrupt_dir = staging / "corrupt_index"
    corrupt_dir.mkdir()
    (corrupt_dir / "model.safetensors.index.json").write_text("{bad json")
    corrupt_status: dict[str, Any] = {}

    def _capture_corrupt(payload: Any) -> Any:
      if payload is not None and not corrupt_status:
        corrupt_status["value"] = payload
      return corrupt_status["value"]

    for rank in (0, 1):
      with (
          mock.patch.object(jax, "process_count", return_value=2),
          mock.patch.object(jax, "process_index", return_value=rank),
          mock.patch.object(
              gcs_synchronizer.sharding_utils,
              "broadcast_host0_json",
              side_effect=_capture_corrupt,
          ),
      ):
        with self.assertRaises(ValueError):
          dst.h2d(checkpoint_path=str(corrupt_dir))

    # Asymmetric files=[] on host 0 vs non-empty on host 1 falls back to
    # all_files so all ranks participate symmetrically in stream_load_files.
    all_files = [types.SimpleNamespace(name="shard0.safetensors")]
    with (
        mock.patch.object(jax, "process_count", return_value=2),
        mock.patch.object(
            gcs_synchronizer.multihost_utils,
            "process_allgather",
            return_value=np.asarray([[0, 0], [0, 1]], dtype=np.int32),
        ),
    ):
      resolved = gcs_synchronizer._coordinate_multihost_read_setup(
          all_files, [], None, "/tmp/ckpt"
      )
      self.assertEqual(resolved, all_files)

    # Setup failure on peer rank aborts healthy host before stream_load_files.
    with (
        mock.patch.object(jax, "process_count", return_value=2),
        mock.patch.object(
            gcs_synchronizer.multihost_utils,
            "process_allgather",
            return_value=np.asarray([[1, 0], [0, 1]], dtype=np.int32),
        ),
    ):
      with self.assertRaisesRegex(RuntimeError, "peer process.*\\[0\\]"):
        gcs_synchronizer._coordinate_multihost_read_setup(
            all_files, all_files, None, "/tmp/ckpt"
        )

    # Post-load peer failure aborts healthy host before _staged_checksums.
    with (
        mock.patch.object(jax, "process_count", return_value=2),
        mock.patch.object(
            gcs_synchronizer.multihost_utils,
            "process_allgather",
            return_value=np.asarray([1, 0], dtype=np.int32),
        ),
    ):
      with self.assertRaisesRegex(RuntimeError, "peer process.*\\[0\\]"):
        gcs_synchronizer._sync_multihost_error(None, "Safetensors restore")
    dst.close()
    # pylint: enable=protected-access

  def test_real_stage_pool_unhealthy_close_respawn_and_failed_d2h(self):
    # pylint: disable=protected-access
    def _make_real_pool() -> safetensors_saver.MultiprocStagePool:
      with mock.patch.object(
          safetensors_io, "resolve_gcs_vips", return_value=()
      ):
        return safetensors_saver.MultiprocStagePool(
            stage_dir=self.create_tempdir().full_path,
            num_procs=1,
            shm_budget_bytes=4 << 20,
            max_bucket_bytes=64 << 10,
            num_large_slots=0,
            use_gcs_http_write=True,
        )

    # P1-1: Real unhealthy MultiprocStagePool must close(abort=True) cleanly on
    # both respawn and GCSSynchronizer.close() without raising RuntimeError.
    sync = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer",
        config=gcs_synchronizer.GCSSynchronizerConfig(write_procs=2),
    )
    pool1 = _make_real_pool()
    pool2 = _make_real_pool()
    try:
      pool1._procs[0].kill()
      pool1._procs[0].wait(timeout=5)
      for _ in range(100):
        if not pool1.is_healthy():
          break
        time.sleep(0.02)
      self.assertFalse(pool1.is_healthy())
      self.assertIsNotNone(pool1._worker_err)
      sync._stage_pool = pool1
      sync._stage_pool_sig = (2, sync.config.max_bucket_bytes, 0, True, False)
      with mock.patch.object(
          safetensors_saver, "MultiprocStagePool", return_value=pool2
      ):
        respawned = sync._ensure_stage_pool("gs://bucket/step_2")
      self.assertIs(respawned, pool2)
      self.assertTrue(pool1._closed)

      # Now kill pool2 too and verify GCSSynchronizer.close() does not raise.
      pool2._procs[0].kill()
      pool2._procs[0].wait(timeout=5)
      for _ in range(100):
        if not pool2.is_healthy():
          break
        time.sleep(0.02)
      self.assertFalse(pool2.is_healthy())
      sync.close()
      self.assertTrue(pool2._closed)
      self.assertIsNone(sync._stage_pool)
    finally:
      pool1.close(abort=True)
      pool2.close(abort=True)

    # P1-2: Failed d2h() must abort-close and clear persistent _stage_pool so
    # wedged/desynced workers can never poison the next step's FLUSH.
    staging = self.create_tempdir().full_path
    fail_sync = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer",
        config=gcs_synchronizer.GCSSynchronizerConfig(
            staging_dir=staging, aot_compile=False
        ),
    )
    fail_sync.bind({"w": jax.device_put(np.ones((2, 2), np.float32))})
    pool3 = _make_real_pool()
    try:
      with (
          mock.patch.object(
              fail_sync, "_ensure_stage_pool", return_value=pool3
          ),
          mock.patch.object(
              safetensors_saver,
              "export_safetensors_with_plan",
              side_effect=OSError("simulated upload failure"),
          ),
      ):
        fail_sync._stage_pool = pool3
        fail_sync._stage_pool_sig = (1, 1, 0, True, False)
        with self.assertRaisesRegex(OSError, "simulated upload failure"):
          fail_sync.d2h(
              datatypes.WeightSyncRequest(
                  policy_version=1, extra_config={"req_id": "fail1", "uuid": 1}
              )
          )
      self.assertTrue(pool3._closed)
      self.assertIsNone(fail_sync._stage_pool)
      self.assertIsNone(fail_sync._stage_pool_sig)
    finally:
      pool3.close(abort=True)
      fail_sync.close()
    # pylint: enable=protected-access

  def test_rulespec_validation_wire_l1_cache_shm_credit_and_p2_fixes(self):
    # pylint: disable=protected-access
    w_f32 = jax.device_put(np.full((2, 2), 2.0, dtype=np.float32))
    w_bf16 = jax.device_put(np.zeros((2, 2), dtype=ml_dtypes.bfloat16))

    # P1-3: WeightMappingSpec and JSON str RuleSpec work in __init__ and
    # validation; invalid manifest fails fast in __init__.
    spec_obj = hf_mapping.WeightMappingSpec(
        rules=(
            hf_mapping.ParamRule(
                source_keys=("w1",), target_keys=("hf.w1",), ops=()
            ),
        ),
        tied_keys=(("w1", "w_tied"),),
    )
    sync_spec = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer", to_hf_manifest=spec_obj
    )
    sync_spec.bind({"w1": w_f32, "w_tied": w_f32})
    sync_spec.close()

    from_json = json.dumps({
        "rules": [{
            "source_keys": ["hf.w1"],
            "target_keys": ["w1"],
            "ops": [],
        }],
        "tied_keys": [["w1", "w_tied"]],
        "dropped_keys": [],
    })
    sync_json = gcs_synchronizer.GCSSynchronizer(
        job_name="rollout", from_hf_manifest=from_json
    )
    sync_json.bind({"w1": w_f32, "w_tied": w_f32})
    sync_json.close()

    with self.assertRaisesRegex(ValueError, "Manifest must be a mapping"):
      gcs_synchronizer.GCSSynchronizer(job_name="bad", to_hf_manifest="{}")

    # P1-3 / P2: _validate_from_hf_manifest respects dropped_keys on rules &
    # tied_keys (a rule targeting a dropped key is skipped at runtime).
    dropped_rule_from = {
        "rules": [
            {"source_keys": ["hf.w1"], "target_keys": ["w1"], "ops": []},
            {"source_keys": ["hf.drop"], "target_keys": ["w_drop"], "ops": []},
        ],
        "tied_keys": [["w1", "w_tied_drop"]],
        "dropped_keys": ["hf.drop", "w_tied_drop"],
    }
    rep_drop = gcs_synchronizer.validate_weight_sync_runtime(
        gcs_synchronizer.GCSSynchronizerConfig(),
        names=["w1", "w_drop", "w_tied_drop"],
        arrays=[w_f32, w_f32, w_f32],
        from_hf_manifest=dropped_rule_from,
        strict=False,
    )
    self.assertFalse(rep_drop.ok)
    self.assertTrue(
        any("w_drop" in e for e in rep_drop.errors), rep_drop.errors
    )
    self.assertTrue(
        any("w_tied_drop" in e for e in rep_drop.errors), rep_drop.errors
    )

    # P1-4: _wire_l1_sums JIT closure clears on manifest re-bind & close.
    m1 = {
        "rules": [{"source_keys": ["w1"], "target_keys": ["hf.a"], "ops": []}]
    }
    m2 = {
        "rules": [{"source_keys": ["w1"], "target_keys": ["hf.b"], "ops": []}]
    }
    worker = _FakeTPUWorker(_NnxRunner({"w1": w_f32}))
    gcs_synchronizer.tpu_worker_bind_gcs_sync(
        worker, job_name="trainer", to_hf_manifest=m1
    )
    wsync = gcs_synchronizer.require_gcs_worker_sync(worker, "test")
    sums1 = wsync._staged_checksums(wsync.arrays)
    self.assertIn("hf.a", sums1)
    self.assertNotIn("hf.b", sums1)
    self.assertIsNotNone(wsync._wire_l1_sums)
    gcs_synchronizer.tpu_worker_bind_gcs_sync(
        worker, job_name="trainer", to_hf_manifest=m2
    )
    self.assertIsNone(wsync._wire_l1_sums)
    sums2 = wsync._staged_checksums(wsync.arrays)
    self.assertIn("hf.b", sums2)
    self.assertNotIn("hf.a", sums2)
    wsync.close()
    self.assertIsNone(wsync._wire_l1_sums)
    gcs_synchronizer._WORKER_SYNCHRONIZERS.pop(worker, None)

    # P1-5a: Trainer Step-2 bind() credits live _stage_pool.total_arena_bytes
    # against /dev/shm statvfs free space so re-bind does not falsely fail.
    step2_sync = gcs_synchronizer.GCSSynchronizer(
        job_name="trainer",
        config=gcs_synchronizer.GCSSynchronizerConfig(
            staging_dir="gs://bucket/run",
            max_bucket_bytes=256 << 20,
            aot_compile=False,
        ),
    )
    live_pool = mock.MagicMock(total_arena_bytes=2 << 30)
    live_pool.is_healthy.return_value = True
    step2_sync._stage_pool = live_pool
    # Raw free = 100 MiB (< 512 MiB = 2x bucket), but + 2 GiB live pool passes!
    with (
        mock.patch(
            "os.statvfs",
            return_value=mock.MagicMock(f_bavail=100 << 20, f_frsize=1),
        ),
        mock.patch.object(
            safetensors_io.GsHttpPath, "resolve_token", return_value="tok"
        ),
        mock.patch.object(
            safetensors_io,
            "resolve_gcs_vips",
            return_value=("1.1.1.1", "2.2.2.2"),
        ),
    ):
      step2_sync.bind({"w1": w_f32})
      self.assertTrue(step2_sync.bound)
      # Unhealthy pool is abort-closed BEFORE statvfs so its /dev/shm is freed.
      dead_pool = mock.MagicMock(total_arena_bytes=2 << 30)
      dead_pool.is_healthy.return_value = False
      step2_sync._stage_pool = dead_pool

      def _statvfs_after_reap(_path: str) -> Any:
        # Simulates kernel reclaiming dead pool's /dev/shm once closed.
        free = (2 << 30) if dead_pool.close.called else (100 << 20)
        return mock.MagicMock(f_bavail=free, f_frsize=1)

      with mock.patch("os.statvfs", side_effect=_statvfs_after_reap):
        step2_sync.bind({"w1": w_f32})
      dead_pool.close.assert_called_once_with(abort=True)
      self.assertIsNone(step2_sync._stage_pool)
      # Without live pool credit, low /dev/shm still fails validation.
      step2_sync._stage_pool = None
      with self.assertRaisesRegex(ValueError, "below 2x max_bucket_bytes"):
        step2_sync.bind({"w1": w_f32})
    step2_sync.close()

    # P1-5b: Standalone GCSSynchronizer context manager & GC weakref finalizer.
    gc_arena = mock.MagicMock(mm=object())
    gc_pool = mock.MagicMock()
    with (
        mock.patch.object(
            safetensors_loader, "ShmRangeReader", return_value=gc_arena
        ),
        mock.patch.object(
            safetensors_saver, "MultiprocStagePool", return_value=gc_pool
        ),
    ):
      standalone = gcs_synchronizer.GCSSynchronizer(
          job_name="rollout",
          config=gcs_synchronizer.GCSSynchronizerConfig(write_procs=4),
      )
      standalone._ensure_read_arena("gs://bucket/step_1", 1 << 20)
      standalone._ensure_stage_pool("gs://bucket/step_1")
      del standalone
      gc.collect()
    gc_arena.__exit__.assert_called_once_with(None, None, None)
    gc_pool.close.assert_called_once_with(abort=True)

    ctx_arena = mock.MagicMock(mm=object())
    with mock.patch.object(
        safetensors_loader, "ShmRangeReader", return_value=ctx_arena
    ):
      with gcs_synchronizer.GCSSynchronizer(job_name="rollout") as ctx_sync:
        ctx_sync._ensure_read_arena("gs://bucket/step_1", 1 << 20)
    ctx_arena.__exit__.assert_called_once_with(None, None, None)

    # P2: _build_restore_specs handles non-jax.Array leaves;
    # _restore_safetensors casts to bound dtype & frees partial HBM on error.
    np_leaf = np.ones((2, 3), dtype=np.float32)
    item_specs, restore_args = gcs_synchronizer._build_restore_specs(
        ["np_w"], [np_leaf]
    )
    self.assertEqual(item_specs["np_w"].shape, (2, 3))
    self.assertEqual(restore_args["np_w"].restore_type, np.ndarray)

    dtype_dir = self.create_tempdir().full_path
    safetensors_saver.save_safetensors(
        {"w1": np.full((2, 2), 3.0, dtype=np.float32)}, dtype_dir
    )
    dst_cast = gcs_synchronizer.GCSSynchronizer(job_name="rollout")
    cast_runner = _FlatDictRunner({"w1": w_bf16})
    dst_cast.bind(cast_runner.state)
    dst_cast.h2d(checkpoint_path=dtype_dir)
    dst_cast.apply_to_runner(cast_runner)
    self.assertEqual(cast_runner.state["w1"].dtype, jnp.bfloat16)
    np.testing.assert_allclose(
        np.asarray(cast_runner.state["w1"], dtype=np.float32), 3.0
    )
    dst_cast.close()

    # Partial HBM arrays are freed when a second tensor is missing.
    loaded_refs: list[jax.Array] = []
    orig_stream = safetensors_loader.stream_load_files

    def _spy_stream(*args: Any, **kwargs: Any) -> None:
      orig_stream(*args, **kwargs)
      loaded_refs.extend(p for p in args[7] if isinstance(p, jax.Array))

    with mock.patch.object(
        safetensors_loader, "stream_load_files", side_effect=_spy_stream
    ):
      with self.assertRaisesRegex(KeyError, "Missing 1 weight tensor"):
        gcs_synchronizer._restore_safetensors(
            dtype_dir, ["w1", "missing_w"], [w_f32, w_f32]
        )
    self.assertLen(loaded_refs, 1)
    self.assertTrue(loaded_refs[0].is_deleted())
    # pylint: enable=protected-access

  def test_rapid_zonal_auto_detect_and_sidecar_restore(self):
    # pylint: disable=protected-access
    w0 = jax.device_put(np.arange(16, dtype=np.float32).reshape(4, 4))
    w1 = jax.device_put(np.arange(16, dtype=np.float32).reshape(4, 4) + 20.0)

    def _is_rapid(uri: str) -> bool:
      return uri.startswith("gs://rapid-bkt")

    # 1. Rapid Zonal auto-detect: disables compose/shard_groups, enables
    #    per_wave_shards & gcs_rapid_grpc while keeping max_bucket_bytes.
    with mock.patch.object(
        safetensors_io, "is_rapid_zonal_bucket", side_effect=_is_rapid
    ):
      rapid_cfg = gcs_synchronizer.GCSSynchronizerConfig(
          staging_dir="gs://rapid-bkt/ws",
          max_bucket_bytes=64,
          max_shard_bytes=128,
          write_procs=8,
          aot_compile=False,
      )
      rep = gcs_synchronizer.validate_weight_sync_runtime(
          rapid_cfg,
          names=["w0", "w1"],
          arrays=[w0, w1],
          strict=False,
      )
      self.assertEmpty(rep.errors)

      sync = gcs_synchronizer.GCSSynchronizer(
          job_name="trainer", config=rapid_cfg
      )
      sync.bind({"w0": w0, "w1": w1})
      _, rapid_plan, _, _ = sync._prepare_save_plan("gs://rapid-bkt/ws/step_1")
      self.assertTrue(rapid_plan.per_wave_shards)
      self.assertEmpty(rapid_plan.shard_groups)
      self.assertNotEmpty(rapid_plan.wave_shards)

      _, std_plan, _, _ = sync._prepare_save_plan("gs://std-bkt/ws/step_1")
      self.assertNotEmpty(std_plan.shard_groups)

      rapid_pool, std_pool = mock.MagicMock(), mock.MagicMock()
      rapid_pool.is_healthy.return_value = True
      std_pool.is_healthy.return_value = True
      with mock.patch.object(
          safetensors_saver,
          "MultiprocStagePool",
          side_effect=[rapid_pool, std_pool],
      ) as mock_pool_cls:
        self.assertIs(
            sync._ensure_stage_pool("gs://rapid-bkt/ws/step_1"), rapid_pool
        )
        self.assertTrue(
            mock_pool_cls.call_args_list[0].kwargs["use_gcs_rapid_grpc"]
        )
        self.assertIs(
            sync._ensure_stage_pool("gs://std-bkt/ws/step_1"), std_pool
        )
        rapid_pool.close.assert_called_once()
        self.assertFalse(
            mock_pool_cls.call_args_list[1].kwargs["use_gcs_rapid_grpc"]
        )
      sync.close()

      # Multi-host consensus broadcasts process 0's Rapid Zonal status and works
      # under jax.transfer_guard("disallow").
      with (
          mock.patch.object(jax, "process_count", return_value=2),
          mock.patch.object(jax, "process_index", return_value=1),
          mock.patch.object(
              gcs_synchronizer.multihost_utils,
              "broadcast_one_to_all",
              wraps=lambda x, **_kw: jnp.asarray(x),
          ) as mock_bcast,
          jax.transfer_guard("disallow"),
      ):
        self.assertFalse(
            gcs_synchronizer._is_rapid_zonal_consensus("gs://rapid-bkt/ws/s1")
        )
        mock_bcast.assert_called_once()
      self.assertFalse(
          gcs_synchronizer._is_rapid_zonal_consensus("/local/ckpt")
      )

      # _save_checkpoint resolves Rapid Zonal consensus ONCE (no plan/pool
      # split-brain); _validate_writer_runtime_config avoids uncoordinated
      # probes.
      with (
          mock.patch.object(
              gcs_synchronizer,
              "_is_rapid_zonal_consensus",
              return_value=True,
          ) as mock_cons,
          mock.patch.object(
              safetensors_io,
              "is_rapid_zonal_bucket",
              side_effect=AssertionError("uncoordinated probe forbidden"),
          ),
          mock.patch.object(
              safetensors_saver, "MultiprocStagePool", return_value=rapid_pool
          ),
          mock.patch.object(
              safetensors_saver,
              "export_safetensors_with_plan",
              return_value=None,
          ),
      ):
        sync2 = gcs_synchronizer.GCSSynchronizer(
            job_name="trainer2", config=rapid_cfg
        )
        sync2.bind({"w0": w0, "w1": w1})
        mock_cons.assert_not_called()
        sync2._save_checkpoint(
            epath.Path("gs://rapid-bkt/ws/step_1"), "gs://rapid-bkt/ws/step_1"
        )
        self.assertEqual(mock_cons.call_count, 1)
        sync2.close()

      # Orbax restores skip eager .tunix_headers.json sidecar GETs.
      orbax_dir = self.create_tempdir().full_path
      orbax_cfg = gcs_synchronizer.GCSSynchronizerConfig(
          staging_dir=orbax_dir, checkpoint_format="orbax"
      )
      orbax_sync = gcs_synchronizer.GCSSynchronizer(
          job_name="orbax_reader", config=orbax_cfg
      )
      orbax_sync.bind({"w0": w0})
      orbax_ckpt = epath.Path(orbax_dir) / "step_1"
      orbax_sync._save_checkpoint(orbax_ckpt, str(orbax_ckpt))
      with mock.patch.object(
          safetensors_io,
          "load_sidecar_headers",
          side_effect=AssertionError("Orbax restore must skip sidecar fetch"),
      ):
        restored_orbax = orbax_sync._restore_checkpoint(str(orbax_ckpt))
      np.testing.assert_array_equal(
          np.asarray(restored_orbax["w0"]), np.asarray(w0)
      )
      orbax_sync.close()

    # 2. Parallel .tunix_headers.json + index.json restore skips list_objects
    #    and header range GETs; missing or stale sidecar falls back cleanly.
    stage_dir = self.create_tempdir().full_path
    cfg = gcs_synchronizer.GCSSynchronizerConfig(
        staging_dir=stage_dir,
        max_bucket_bytes=64,
        max_shard_bytes=None,
        per_wave_shards=True,
        aot_compile=False,
    )
    writer = gcs_synchronizer.GCSSynchronizer(job_name="trainer", config=cfg)
    writer.bind({"w0": w0, "w1": w1})
    ckpt_dir = epath.Path(stage_dir) / "step_1"
    writer._save_checkpoint(ckpt_dir, str(ckpt_dir))
    sc_file = ckpt_dir / safetensors_io.SIDECAR_HEADERS_FILENAME
    self.assertTrue(sc_file.exists())

    reader = gcs_synchronizer.GCSSynchronizer(job_name="rollout", config=cfg)
    reader.bind({"w0": jnp.zeros_like(w0), "w1": jnp.zeros_like(w1)})
    with (
        mock.patch.object(
            safetensors_loader,
            "resolve_range_sources",
            side_effect=AssertionError("must skip list_objects on sidecar hit"),
        ),
        mock.patch.object(
            safetensors_loader,
            "_read_safetensors_header",
            side_effect=AssertionError("must skip header GET on sidecar hit"),
        ),
    ):
      restored = reader._restore_checkpoint(str(ckpt_dir))
      np.testing.assert_array_equal(np.asarray(restored["w0"]), np.asarray(w0))
      np.testing.assert_array_equal(np.asarray(restored["w1"]), np.asarray(w1))
      # Direct _restore_safetensors without weight_map also hits the sidecar.
      restored2, _, _ = gcs_synchronizer._restore_safetensors(
          str(ckpt_dir), reader.names, reader.arrays
      )
      np.testing.assert_array_equal(np.asarray(restored2["w0"]), np.asarray(w0))

    # Stale sidecar with extra shard (superset of index.json) falls back.
    orig_sc = json.loads(sc_file.read_text())
    superset_sc = {
        "version": 1,
        "shards": {
            **orig_sc["shards"],
            "extra-stale.safetensors": {
                "header_len": 8,
                "file_size": 16,
                "tensors": {},
            },
        },
    }
    sc_file.write_text(json.dumps(superset_sc))
    restored_superset = reader._restore_checkpoint(str(ckpt_dir))
    np.testing.assert_array_equal(
        np.asarray(restored_superset["w0"]), np.asarray(w0)
    )

    # Stale sidecar missing a shard in index.json falls back silently.
    sc_data = {
        "version": 1,
        "shards": dict(orig_sc["shards"]),
    }
    first_shard = next(iter(sc_data["shards"]))
    sc_data["shards"].pop(first_shard)
    sc_file.write_text(json.dumps(sc_data))
    restored_stale = reader._restore_checkpoint(str(ckpt_dir))
    np.testing.assert_array_equal(
        np.asarray(restored_stale["w1"]), np.asarray(w1)
    )

    # Missing sidecar falls back silently for standard external HF checkpoints.
    sc_file.unlink()
    restored_fb = reader._restore_checkpoint(str(ckpt_dir))
    np.testing.assert_array_equal(np.asarray(restored_fb["w0"]), np.asarray(w0))

    # Single-shard checkpoint (empty weight_map): stale sidecar with wrong
    # shard name or mismatched local file_size falls back cleanly.
    single_dir = self.create_tempdir().full_path
    single_cfg = gcs_synchronizer.GCSSynchronizerConfig(
        staging_dir=single_dir,
        max_bucket_bytes=1024,
        max_shard_bytes=None,
        per_wave_shards=False,
        aot_compile=False,
    )
    s_writer = gcs_synchronizer.GCSSynchronizer(
        job_name="s_trainer", config=single_cfg
    )
    s_writer.bind({"w0": w0})
    s_ckpt = epath.Path(single_dir) / "step_1"
    s_writer._save_checkpoint(s_ckpt, str(s_ckpt))
    s_sc = s_ckpt / safetensors_io.SIDECAR_HEADERS_FILENAME
    s_reader = gcs_synchronizer.GCSSynchronizer(
        job_name="s_rollout", config=single_cfg
    )
    s_reader.bind({"w0": jnp.zeros_like(w0)})
    s_sc.write_text(
        json.dumps({
            "version": 1,
            "shards": {
                "other.safetensors": {
                    "header_len": 8,
                    "file_size": 16,
                    "tensors": {},
                }
            },
        })
    )
    loaded_wrong = safetensors_io.load_sidecar_headers(str(s_ckpt))
    self.assertIsNotNone(loaded_wrong)
    _, _, usable_wrong = gcs_synchronizer._resolve_restore_sources_and_headers(
        str(s_ckpt), ["model.safetensors"], {}, loaded_wrong
    )
    self.assertIsNone(usable_wrong)
    s_sc.write_text(
        json.dumps({
            "version": 1,
            "shards": {
                "model.safetensors": {
                    "header_len": 8,
                    "file_size": 16,
                    "tensors": {},
                }
            },
        })
    )
    _, _, usable_stale_sz = (
        gcs_synchronizer._resolve_restore_sources_and_headers(
            str(s_ckpt),
            ["model.safetensors"],
            {},
            safetensors_io.load_sidecar_headers(str(s_ckpt)),
        )
    )
    self.assertIsNone(usable_stale_sz)
    restored_single = s_reader._restore_checkpoint(str(s_ckpt))
    np.testing.assert_array_equal(
        np.asarray(restored_single["w0"]), np.asarray(w0)
    )
    s_writer.close()
    s_reader.close()
    writer.close()
    reader.close()
    # pylint: enable=protected-access

if __name__ == "__main__":
  absltest.main()
