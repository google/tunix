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

"""Unit tests for Subtree all-to-all safetensors_saver."""

import concurrent.futures
import json
import os
import tempfile
import threading
import time
import types
from unittest import mock

from absl.testing import absltest
from etils import epath
from flax import nnx
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
from safetensors import numpy as safe_np
import safetensors.flax as safe_flax
from tunix.models import safetensors_saver

jax.config.update('jax_num_cpu_devices', 8)


class _ToyModule(nnx.Module):

  def __init__(self, rngs: nnx.Rngs):
    self.w = nnx.Param(jax.random.normal(rngs.params(), (8, 4)))
    self.experts = nnx.Param(jax.random.normal(rngs.params(), (2, 4, 6)))
    self.w_lora_a = nnx.LoRAParam(jnp.ones((8, 2), dtype=jnp.float32))
    self.running_stat = nnx.BatchStat(jnp.zeros((4,), dtype=jnp.float32))


class SafetensorsSaverTest(absltest.TestCase):

  def test_join_and_extract_flat_state(self):
    self.assertEqual(safetensors_saver.join_path(('a', 0, 'b')), 'a.0.b')
    mod = _ToyModule(nnx.Rngs(params=0))
    flat = safetensors_saver.extract_flat_state_dict(mod)
    self.assertIn('w', flat)
    self.assertIn('experts', flat)
    self.assertIn('w_lora_a', flat)
    self.assertNotIn('running_stat', flat)
    self.assertEqual(flat['w'].shape, (8, 4))

  def test_plan_and_wave_bucketing(self):
    self.assertEqual(
        safetensors_saver.shard_filename(0, 1), 'model.safetensors'
    )
    self.assertEqual(
        safetensors_saver.shard_filename(1, 2),
        'model-00002-of-00002.safetensors',
    )
    spec_a = safetensors_saver.TensorExportSpec(
        source_key='w',
        hf_key='model.w.weight',
        transpose_axes=(1, 0),
        nbytes=128,
    )
    spec_b = safetensors_saver.TensorExportSpec(
        source_key='experts',
        hf_key='model.experts.{expert}.weight',
        transpose_axes=(0, 2, 1),
        unstack_axis=0,
        nbytes=192,
    )
    self.assertEqual(
        safetensors_saver.expand_spec_keys(spec_a, {}), ('model.w.weight',)
    )
    self.assertEqual(
        safetensors_saver.expand_spec_keys(spec_b, {'experts': 2}),
        ('model.experts.0.weight', 'model.experts.1.weight'),
    )
    plan = safetensors_saver.build_save_plan(
        [spec_a, spec_b],
        num_ranks=2,
        max_bucket_bytes=150,
        unstack_counts={'experts': 2},
    )
    self.assertEqual(plan.num_ranks, 2)
    self.assertEqual(plan.total_bytes, 320)
    self.assertIn('model.experts.1.weight', plan.weight_map)
    with self.assertRaises(ValueError):
      safetensors_saver.build_save_plan([spec_a], num_ranks=0)

  def test_on_hbm_transforms_and_export_with_plan(self):
    mod = _ToyModule(nnx.Rngs(params=0))
    flat = safetensors_saver.extract_flat_state_dict(mod)
    flat['w_lora_a'] = jnp.ones((8, 2), dtype=jnp.float32)
    flat['w_lora_b'] = jnp.ones((2, 4), dtype=jnp.float32)
    spec_w = safetensors_saver.TensorExportSpec(
        source_key='w',
        hf_key='model.w.weight',
        transpose_axes=(1, 0),
        nbytes=128,
        lora_a_key='w_lora_a',
        lora_b_key='w_lora_b',
        lora_scaling=0.5,
    )
    spec_e = safetensors_saver.TensorExportSpec(
        source_key='experts',
        hf_key='model.experts.{expert}.weight',
        transpose_axes=(0, 2, 1),
        unstack_axis=0,
        nbytes=192,
    )
    jitted = safetensors_saver._jit_transform_on_hbm(
        flat['w'],
        flat['w_lora_a'],
        flat['w_lora_b'],
        0.5,
        None,
        (1, 0),
        jnp.bfloat16,
    )
    self.assertEqual(jitted.shape, (4, 8))
    self.assertEqual(jitted.dtype, jnp.bfloat16)

    transformed = safetensors_saver.transform_tensor_on_hbm(
        flat['w'],
        spec_w,
        lora_a=flat['w_lora_a'],
        lora_b=flat['w_lora_b'],
        target_device=jax.devices()[0],
    )
    self.assertEqual(transformed.shape, (4, 8))

    plan = safetensors_saver.build_save_plan(
        [spec_w, spec_e],
        num_ranks=2,
        max_bucket_bytes=200,
        unstack_counts={'experts': 2},
    )
    wave_arr = safetensors_saver.transform_wave_spec(
        flat, 0, spec_e, jnp.float32, jax.devices()
    )
    unpacked = safetensors_saver.unpack_host_tensor(
        spec_e, np.asarray(wave_arr)
    )
    self.assertEqual(unpacked['model.experts.0.weight'].shape, (6, 4))
    self.assertTrue(unpacked['model.experts.0.weight'].flags.c_contiguous)

    host_wave, _, _ = safetensors_saver.materialize_wave_to_host(
        flat, plan.waves[0], jnp.float32, jax.devices()
    )
    self.assertNotEmpty(host_wave)

    with tempfile.TemporaryDirectory() as tmpdir:
      src_dir = os.path.join(tmpdir, 'src')
      out_dir = os.path.join(tmpdir, 'out')
      async_dir = os.path.join(tmpdir, 'out_async')
      os.makedirs(src_dir)
      with open(os.path.join(src_dir, 'config.json'), 'w') as f:
        json.dump({'arch': 'toy'}, f)
      with open(
          os.path.join(src_dir, 'model.safetensors.index.json'), 'w'
      ) as f:
        json.dump({'stale': True}, f)
      with open(os.path.join(src_dir, 'stale.safetensors'), 'wb') as f:
        f.write(b'stale')
      with open(os.path.join(src_dir, 'scratch.tmp'), 'wb') as f:
        f.write(b'tmp')
      safetensors_saver.copy_non_safetensors_files(src_dir, src_dir)
      hdr = safetensors_saver.build_rank_header_bytes(
          0, plan, flat, jnp.float32
      )
      self.assertGreater(len(hdr), 8)
      manual_dir = os.path.join(tmpdir, 'manual')
      os.makedirs(manual_dir)
      files, paths, n_act = safetensors_saver.open_local_shard_files(
          flat, plan, manual_dir, jnp.float32, jax.devices()
      )
      self.assertEqual(n_act, 2)
      self.assertLen(paths, 2)
      safetensors_saver._write_rank_tensors(
          files[0], {'w': np.asarray(jnp.ones((2, 2), dtype=jnp.bfloat16))}
      )
      safetensors_saver.flush_wave_to_files(
          files,
          {
              0: {'a': np.ones((2, 2), dtype=np.float32)},
              1: {'b': np.ones((2, 2), dtype=np.float32)},
          },
      )
      for fh in files.values():
        fh.close()
      safetensors_saver.finalize_safetensors_metadata(
          plan, manual_dir, src_dir, n_act
      )
      metrics = safetensors_saver.export_safetensors_with_plan(
          flat,
          plan,
          out_dir,
          local_model_path=src_dir,
          output_dtype=jnp.float32,
          devices=jax.devices(),
      )
      metrics_async = safetensors_saver.export_safetensors_with_plan(
          flat,
          plan,
          async_dir,
          local_model_path=src_dir,
          output_dtype=jnp.float32,
          devices=jax.devices(),
          async_write=True,
      )
      wave_plan = safetensors_saver.build_save_plan(
          [spec_w, spec_e],
          num_ranks=1,
          max_bucket_bytes=128,
          unstack_counts={'experts': 2},
          per_wave_shards=True,
      )
      self.assertTrue(wave_plan.per_wave_shards)
      self.assertLen(set(wave_plan.weight_map.values()), 2)
      wave_dir = os.path.join(tmpdir, 'out_wave_shards')
      metrics_wave = safetensors_saver.export_safetensors_with_plan(
          flat,
          wave_plan,
          wave_dir,
          local_model_path=src_dir,
          output_dtype=jnp.float32,
          devices=jax.devices(),
          async_write=True,
          write_buffer_bytes=64,
          max_write_threads=4,
      )
      self.assertLen(metrics_wave.shard_paths, 2)
      self.assertLen(metrics.shard_paths, 2)
      self.assertLen(metrics_async.shard_paths, 2)
      self.assertTrue(os.path.exists(os.path.join(out_dir, 'config.json')))
      self.assertFalse(
          os.path.exists(os.path.join(out_dir, 'stale.safetensors'))
      )
      self.assertFalse(os.path.exists(os.path.join(out_dir, 'scratch.tmp')))
      with open(os.path.join(out_dir, 'model.safetensors.index.json')) as f:
        exported_index = json.load(f)
      self.assertIn('weight_map', exported_index)
      self.assertNotIn('stale', exported_index)
      loaded_0 = safe_flax.load_file(metrics.shard_paths[0])
      loaded_1 = safe_flax.load_file(metrics.shard_paths[1])
      merged = {**loaded_0, **loaded_1}
      expected_w = (flat['w'] + (flat['w_lora_a'] @ flat['w_lora_b']) * 0.5).T
      np.testing.assert_allclose(
          merged['model.w.weight'], expected_w, atol=1e-5
      )
      np.testing.assert_allclose(
          merged['model.experts.1.weight'], flat['experts'][1].T, atol=1e-5
      )
      async_0 = safe_flax.load_file(metrics_async.shard_paths[0])
      async_1 = safe_flax.load_file(metrics_async.shard_paths[1])
      np.testing.assert_allclose(
          merged['model.w.weight'],
          {**async_0, **async_1}['model.w.weight'],
      )
      wave_merged = {}
      for sp in metrics_wave.shard_paths:
        wave_merged.update(safe_flax.load_file(sp))
      np.testing.assert_allclose(
          merged['model.w.weight'], wave_merged['model.w.weight']
      )
      np.testing.assert_allclose(
          merged['model.experts.1.weight'],
          wave_merged['model.experts.1.weight'],
      )
      local_be = safetensors_saver.resolve_storage_backend(out_dir)
      gcs_be = safetensors_saver.resolve_storage_backend('gs://test-bucket/out')
      self.assertEqual(
          (
              local_be.kind,
              local_be.default_per_wave_shards,
              local_be.default_write_threads,
              local_be.default_write_buffer_bytes,
              local_be.requires_bytes_chunks,
          ),
          ('local', True, 8, 0, False),
      )
      self.assertEqual(
          (
              gcs_be.kind,
              gcs_be.default_per_wave_shards,
              gcs_be.default_write_threads,
              gcs_be.default_write_buffer_bytes,
              gcs_be.requires_bytes_chunks,
          ),
          ('gcs', True, 64, 64 << 20, False),
      )

      recorded_buf_sizes: list[int | None] = []

      class _FakeSelector:

        def __init__(self, base_dir: str, **_kwargs):
          self.base_dir = base_dir

      class _FakeGcsFs:

        def open_output_stream(self, key: str, buffer_size: int | None = None):
          recorded_buf_sizes.append(buffer_size)
          full = os.path.join(tmpdir, 'gcs_root', key)
          os.makedirs(os.path.dirname(full), exist_ok=True)
          return open(full, 'wb')

        def get_file_info(self, selector: _FakeSelector):
          full = os.path.join(tmpdir, 'gcs_root', selector.base_dir)
          if not os.path.isdir(full):
            return []
          return [
              types.SimpleNamespace(
                  path=f'{selector.base_dir}/{name}', base_name=name
              )
              for name in os.listdir(full)
          ]

        def delete_file(self, key: str) -> None:
          os.unlink(os.path.join(tmpdir, 'gcs_root', key))

      orig_pa_fs = safetensors_saver._pa_fs
      safetensors_saver._pa_fs = types.SimpleNamespace(
          GcsFileSystem=_FakeGcsFs, FileSelector=_FakeSelector
      )
      try:
        gcs_be.makedirs('gs://test-bucket/wave_out')
        with gcs_be.open_writer('gs://test-bucket/probe.bin') as gf:
          gf.write(b'ok')
        with gcs_be.open_writer(
            'gs://test-bucket/wave_out/model.safetensors'
        ) as stale_gf:
          stale_gf.write(b'stale')
        metrics_gcs = safetensors_saver.export_safetensors_with_plan(
            flat,
            wave_plan,
            'gs://test-bucket/wave_out',
            local_model_path=src_dir,
            output_dtype=jnp.float32,
            devices=jax.devices(),
            async_write=True,
        )
      finally:
        safetensors_saver._pa_fs = orig_pa_fs
      self.assertIn(64 * 1024 * 1024, recorded_buf_sizes)
      self.assertLen(metrics_gcs.shard_paths, 2)
      gcs_local_root = os.path.join(tmpdir, 'gcs_root', 'test-bucket/wave_out')
      self.assertFalse(
          os.path.exists(os.path.join(gcs_local_root, 'model.safetensors'))
      )
      self.assertTrue(
          os.path.exists(os.path.join(gcs_local_root, 'config.json'))
      )
      with open(
          os.path.join(gcs_local_root, 'model.safetensors.index.json')
      ) as f:
        gcs_idx = json.load(f)
      self.assertIn('weight_map', gcs_idx)
      self.assertNotIn('stale', gcs_idx)
      gcs_merged = {}
      for sp in metrics_gcs.shard_paths:
        local_sp = os.path.join(tmpdir, 'gcs_root', sp.removeprefix('gs://'))
        gcs_merged.update(safe_flax.load_file(local_sp))
      np.testing.assert_allclose(
          merged['model.w.weight'], gcs_merged['model.w.weight']
      )
      np.testing.assert_allclose(
          merged['model.experts.0.weight'],
          gcs_merged['model.experts.0.weight'],
      )
      np.testing.assert_allclose(
          merged['model.experts.1.weight'],
          gcs_merged['model.experts.1.weight'],
      )

  def test_storage_backend_write_dispatch_isolation(self):
    hdr = b'12345678'
    small = np.arange(2, dtype=np.float32)  # 8 bytes
    large_local = np.arange(10, dtype=np.float32)  # 40 bytes
    large_gcs = np.arange(10, dtype=np.float32)  # 40 bytes
    large_bytes = np.arange(10, dtype=np.float32)  # 40 bytes
    expected_payload = hdr + small.tobytes() + large_local.tobytes()
    recorded: dict[str, list[tuple[type[object], int, bytes, object]]] = {
        'local': [],
        'gcs': [],
        'bytes_chunks': [],
    }

    def _make_backend(
        kind: str, buf_bytes: int, requires_bytes: bool
    ) -> safetensors_saver.StorageBackend:
      class _Backend(safetensors_saver.StorageBackend):

        def open_writer(self, path: str):
          del path
          return types.SimpleNamespace(
              write=lambda d: recorded[kind].append(
                  (type(d), len(d), bytes(d), getattr(d, 'obj', None))
              ),
              close=lambda: None,
          )

      return _Backend(kind, True, 8, buf_bytes, requires_bytes)

    local_be = _make_backend('local', 0, False)
    gcs_be = _make_backend('gcs', 16, False)
    bytes_be = _make_backend('bytes_chunks', 16, True)
    safetensors_saver._write_rank_tensors(
        (local_be, '/dev/shm/s.safetensors', hdr),
        {'s': small.copy(), 'l': large_local},
    )
    safetensors_saver._write_rank_tensors(
        (gcs_be, 'gs://b/s.safetensors', hdr),
        {'s': small.copy(), 'l': large_gcs},
        write_buffer_bytes=16,
    )
    safetensors_saver._write_rank_tensors(
        (bytes_be, '/tmp/bytes/s.safetensors', hdr),
        {'s': small.copy(), 'l': large_bytes},
        write_buffer_bytes=16,
    )
    self.assertEqual(
        [t for t, _, _, _ in recorded['local']],
        [memoryview, memoryview, memoryview],
    )
    self.assertEqual([n for _, n, _, _ in recorded['local']], [8, 8, 40])
    self.assertTrue(np.shares_memory(recorded['local'][-1][3], large_local))
    self.assertEqual(
        b''.join(b for _, _, b, _ in recorded['local']), expected_payload
    )
    self.assertEqual(
        [t for t, _, _, _ in recorded['gcs']], [memoryview, memoryview]
    )
    self.assertEqual([n for _, n, _, _ in recorded['gcs']], [16, 40])
    self.assertTrue(np.shares_memory(recorded['gcs'][-1][3], large_gcs))
    self.assertEqual(
        b''.join(b for _, _, b, _ in recorded['gcs']), expected_payload
    )
    self.assertEqual(
        [t for t, _, _, _ in recorded['bytes_chunks']],
        [bytes, bytes, bytes, bytes],
    )
    self.assertEqual(
        [n for _, n, _, _ in recorded['bytes_chunks']], [16, 16, 16, 8]
    )
    self.assertEqual(
        b''.join(b for _, _, b, _ in recorded['bytes_chunks']), expected_payload
    )
    # Explicit write_buffer_bytes=0 must not fall back to backend default (16).
    recorded['bytes_chunks'].clear()
    safetensors_saver._write_rank_tensors(
        safetensors_saver._OpenShardFile(
            bytes_be, bytes_be.open_writer('/tmp/bytes/unbuf.safetensors')
        ),
        {'l': large_bytes.copy()},
        write_buffer_bytes=0,
    )
    self.assertEqual([t for t, _, _, _ in recorded['bytes_chunks']], [bytes])
    self.assertEqual([n for _, n, _, _ in recorded['bytes_chunks']], [40])
    self.assertEqual(
        safetensors_saver.resolve_storage_backend(
            '/dev/shm/ttl=bad%rs=6.3'
        ).kind,
        'local',
    )
    self.assertEqual(
        safetensors_saver.resolve_storage_backend(
            'gs://my-bucket/ttl=bad%rs=6.3'
        ).kind,
        'gcs',
    )

  def test_ragged_all_to_all_exchange(self):
    devs = jax.devices()[:1]
    mesh = jax.sharding.Mesh(np.array(devs), ('rank',))
    s0 = jnp.arange(1024, dtype=jnp.float32)
    s1 = jnp.arange(2048, dtype=jnp.float32) + 10.0
    perm_row = safetensors_saver.build_tile_perm_row([1, 2], 2, 4, False)
    np.testing.assert_array_equal(
        perm_row, np.array([0, 4, 1, 2, 5, 6, 3, 7], dtype=np.int32)
    )
    perm_ragged = safetensors_saver.build_tile_perm_row([1, 2], 2, 4, True)
    np.testing.assert_array_equal(
        perm_ragged, np.array([0, 3, 1, 2, 4, 5, 6, 7], dtype=np.int32)
    )
    jitted = safetensors_saver._jit_exchange_shards_all_to_all(
        (s0, s1), mesh, 'rank', (0, 1), (0,), (3072,), 3072, False
    )
    self.assertEqual(jitted.shape, (3072,))
    hlo_text = safetensors_saver._jit_exchange_shards_all_to_all.lower(
        (s0, s1), mesh, 'rank', (0, 1), (0,), (3072,), 3072, True
    ).as_text()
    self.assertIn('ragged_all_to_all', hlo_text)
    self.assertIn('3x8x128xf32', hlo_text)
    out = safetensors_saver.exchange_shards_ragged_all_to_all(
        [s0, s1], [0, 0], mesh, 'rank'
    )
    np.testing.assert_allclose(out[:1024], s0)
    np.testing.assert_allclose(out[1024:3072], s1)
    spec_s0 = safetensors_saver.TensorExportSpec(
        source_key='s0', hf_key='model.s0', nbytes=4096
    )
    spec_s1 = safetensors_saver.TensorExportSpec(
        source_key='s1', hf_key='model.s1', nbytes=8192
    )
    rep_norm = jax.device_put(
        jnp.ones((128,), dtype=jnp.float32),
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
    )
    shd_s0 = jax.device_put(
        s0.reshape(1, 1024),
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('rank')),
    )
    self.assertTrue(safetensors_saver.infer_is_replicated(rep_norm))
    self.assertFalse(safetensors_saver.infer_is_replicated(shd_s0))
    spec_norm = safetensors_saver.TensorExportSpec(
        source_key='norm',
        hf_key='model.norm.weight',
        nbytes=512,
        is_replicated=True,
    )
    batch_routed, _, _ = safetensors_saver.exchange_wave_batch_to_host(
        [(0, spec_s0)], [s0], devs
    )
    np.testing.assert_allclose(batch_routed[0]['model.s0'], s0)
    deferred_batch, _, _ = safetensors_saver.exchange_wave_batch_to_host(
        [(0, spec_s0), (0, spec_s1)], [s0, s1], devs, defer_contiguous=True
    )
    np.testing.assert_allclose(np.asarray(deferred_batch[0]['model.s0']), s0)
    np.testing.assert_allclose(np.asarray(deferred_batch[0]['model.s1']), s1)
    wave_bucket = safetensors_saver.WaveBucket(
        wave_index=0,
        rank_assignments={0: (spec_norm, spec_s0)},
        rank_bytes={0: 4608},
    )
    handle = safetensors_saver.dispatch_wave_on_hbm(
        {'norm': rep_norm, 's0': shd_s0},
        wave_bucket,
        jnp.float32,
        devs,
        defer_contiguous=True,
    )
    completed_wave, _, _ = safetensors_saver.complete_wave_d2h(handle)
    np.testing.assert_allclose(
        np.asarray(completed_wave[0]['model.norm.weight']), rep_norm
    )
    np.testing.assert_allclose(
        np.asarray(completed_wave[0]['model.s0']).reshape(-1), s0
    )
    routed, _, _ = safetensors_saver.route_wave_shards_to_host(
        [(0, spec_norm), (0, spec_s0)],
        [rep_norm, shd_s0],
        devs,
        defer_contiguous=True,
    )
    np.testing.assert_allclose(
        np.asarray(routed[0]['model.norm.weight']), rep_norm
    )
    np.testing.assert_allclose(
        np.asarray(routed[0]['model.s0']).reshape(-1), s0
    )
    with self.assertRaises(ValueError):
      safetensors_saver.exchange_shards_ragged_all_to_all(
          [s0], [0, 0], mesh, 'rank'
      )
    with self.assertRaises(ValueError):
      safetensors_saver.exchange_shards_ragged_all_to_all(
          [jnp.arange(6, dtype=jnp.float32)], [0], mesh, 'rank'
      )
    with self.assertRaises(ValueError):
      safetensors_saver.exchange_shards_ragged_all_to_all(
          [
              jnp.arange(1020, dtype=jnp.float32),
              jnp.arange(4, dtype=jnp.float32),
          ],
          [0, 0],
          mesh,
          'rank',
      )

  def test_save_lora_merged_model_as_safetensors(self):
    class _ToyLoRAModule(nnx.Module):

      def __init__(self):
        self.w = nnx.Param(jnp.ones((4, 2), dtype=jnp.float32))
        self.w_lora_a = nnx.LoRAParam(jnp.ones((4, 1), dtype=jnp.float32))
        self.w_lora_b = nnx.LoRAParam(jnp.ones((1, 2), dtype=jnp.float32))

    with tempfile.TemporaryDirectory() as tmpdir:
      base_dir = os.path.join(tmpdir, 'base')
      out_dir = os.path.join(tmpdir, 'merged')
      os.makedirs(base_dir)
      safe_flax.save_file(
          {'model.w.weight': jnp.ones((2, 4), dtype=jnp.float32)},
          os.path.join(base_dir, 'model.safetensors'),
      )
      safetensors_saver.save_lora_merged_model_as_safetensors(
          local_model_path=base_dir,
          output_dir=out_dir,
          lora_model=_ToyLoRAModule(),
          rank=2,
          alpha=4.0,
          state_key_transform_fn=lambda _: 'model.w.weight',
          transpose_rules={'w': (1, 0)},
      )
      merged = safe_flax.load_file(os.path.join(out_dir, 'model.safetensors'))
      np.testing.assert_allclose(
          merged['model.w.weight'], np.full((2, 4), 3.0, dtype=np.float32)
      )

  def test_async_single_file_preserves_wave_order(self):
    specs = [
        safetensors_saver.TensorExportSpec(
            source_key=f't{i}', hf_key=f'w{i}', nbytes=8192
        )
        for i in range(4)
    ]
    state = {
        f't{i}': jnp.full((32, 64), float(i), dtype=jnp.float32)
        for i in range(4)
    }
    plan = safetensors_saver.build_save_plan(
        specs, num_ranks=1, max_bucket_bytes=8192
    )
    self.assertLen(plan.waves, 4)
    with tempfile.TemporaryDirectory() as tmpdir:
      safetensors_saver.export_safetensors_with_plan(
          state, plan, tmpdir, devices=jax.devices()[:1], async_write=True
      )
      loaded = safe_flax.load_file(os.path.join(tmpdir, 'model.safetensors'))
      for i in range(4):
        np.testing.assert_allclose(loaded[f'w{i}'], np.asarray(state[f't{i}']))

  def test_submit_and_drain_wave_flush_drains_before_submit(self):
    # With one shard file per rank (`limit=1`), a wave must not start writing
    # until the previous wave's flush is done, or tensors land out of order.
    older: concurrent.futures.Future[float] = concurrent.futures.Future()
    older_done_at_write: list[bool] = []
    sink = types.SimpleNamespace(
        write=lambda _: older_done_at_write.append(older.done())
    )
    threading.Timer(0.05, older.set_result, args=(0.0,)).start()
    pending = [older]
    host_wave = {0: {'w': np.zeros(4, dtype=np.float32)}}
    safetensors_saver.submit_and_drain_wave_flush(
        pending, 1, {0: sink}, host_wave, 0, 0, 1
    )
    pending[0].result()
    self.assertEqual(older_done_at_write, [True])
    # `limit=0` writes synchronously and never queues.
    sync_pending: list[concurrent.futures.Future[float]] = []
    safetensors_saver.submit_and_drain_wave_flush(
        sync_pending, 0, {0: sink}, {0: {'w': np.zeros(4)}}, 0, 0, 1
    )
    self.assertEmpty(sync_pending)
    self.assertLen(older_done_at_write, 2)

  def test_multidevice_sharded_1d_norm_mixed_order_dtype_and_2wave_reload(self):
    devs = jax.devices()[:4]
    self.assertLen(devs, 4)
    mesh = jax.sharding.Mesh(np.array(devs).reshape(2, 2), ('fsdp', 'tp'))
    p_tp = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('tp'))
    p_2d = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('fsdp', 'tp')
    )
    p_rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    norm_1d = jax.device_put(jnp.arange(64, dtype=jnp.float32) + 1.0, p_tp)
    self.assertFalse(safetensors_saver.infer_is_replicated(norm_1d))

    # Unaligned replicated tensor larger than sharded tensors so LPT places it
    # BEFORE sharded tensors in the same wave/rank (tests order alignment).
    big_unaligned = jax.device_put(
        jnp.arange(4500, dtype=jnp.float32).reshape(3, 1500), p_rep
    )
    # Aligned sharded tensor explicitly flagged `is_replicated=True`.
    shd_forced_rep = jax.device_put(
        jnp.arange(4096, dtype=jnp.float32).reshape(64, 64) * 0.25, p_2d
    )
    shd_f32 = jax.device_put(
        jnp.arange(4096, dtype=jnp.float32).reshape(64, 64) + 100.0, p_2d
    )
    shd_bf16 = jax.device_put(
        (jnp.arange(4096, dtype=jnp.float32).reshape(64, 64) * 0.5).astype(
            jnp.bfloat16
        ),
        p_2d,
    )
    shd_wave2 = jax.device_put(
        jnp.arange(4096, dtype=jnp.float32).reshape(64, 64) + 500.0, p_2d
    )
    state = {
        'norm': norm_1d,
        'big_unaligned': big_unaligned,
        'forced_rep': shd_forced_rep,
        'shd_f32': shd_f32,
        'shd_bf16': shd_bf16,
        'shd_wave2': shd_wave2,
    }
    specs = [
        safetensors_saver.TensorExportSpec(
            'big_unaligned', 'model.big_unaligned', nbytes=4500 * 4
        ),
        safetensors_saver.TensorExportSpec(
            'shd_f32', 'model.shd_f32', nbytes=4096 * 4
        ),
        safetensors_saver.TensorExportSpec(
            'shd_bf16', 'model.shd_bf16', nbytes=4096 * 2
        ),
        safetensors_saver.TensorExportSpec(
            'forced_rep',
            'model.forced_rep',
            nbytes=4096 * 4,
            is_replicated=True,
        ),
        safetensors_saver.TensorExportSpec(
            'shd_wave2', 'model.shd_wave2', nbytes=4096 * 4
        ),
        safetensors_saver.TensorExportSpec(
            'norm', 'model.norm.weight', nbytes=64 * 4
        ),
    ]
    # Mixed dtypes fail fast in exchange_shards_ragged_all_to_all directly.
    rank_mesh = jax.sharding.Mesh(np.array(devs), ('rank',))
    rank_shd = jax.sharding.NamedSharding(
        rank_mesh, jax.sharding.PartitionSpec('rank')
    )
    with self.assertRaisesRegex(ValueError, 'homogeneous dtypes'):
      safetensors_saver.exchange_shards_ragged_all_to_all(
          [
              jax.device_put(shd_f32.reshape(4, 1024), rank_shd),
              jax.device_put(shd_bf16.reshape(4, 1024), rank_shd),
          ],
          [0, 1],
          rank_mesh,
          'rank',
      )

    # Verify exact key order on mixed replicated + sharded wave in BOTH paths.
    mixed_wave = safetensors_saver.WaveBucket(
        wave_index=0,
        rank_assignments={0: (specs[0], specs[1], specs[2], specs[5])},
        rank_bytes={0: 4500 * 4 + 4096 * 6 + 256},
    )
    handle = safetensors_saver.dispatch_wave_on_hbm(
        state, mixed_wave, None, devs, defer_contiguous=True
    )
    completed, _, _ = safetensors_saver.complete_wave_d2h(handle)
    expected_order = [
        'model.big_unaligned',
        'model.shd_f32',
        'model.shd_bf16',
        'model.norm.weight',
    ]
    self.assertEqual(list(completed[0].keys()), expected_order)
    mat_wave, _, _ = safetensors_saver.materialize_wave_to_host(
        state, mixed_wave, None, devs, defer_contiguous=True
    )
    self.assertEqual(list(mat_wave[0].keys()), expected_order)

    # Full multi-wave 2-wave overlap export (both per_wave_shards modes) and
    # bit-for-bit reload with output_dtype=None (preserving bf16 + f32).
    for per_wave in (False, True):
      plan = safetensors_saver.build_save_plan(
          specs,
          num_ranks=2,
          max_bucket_bytes=36000,
          per_wave_shards=per_wave,
      )
      self.assertGreaterEqual(len(plan.waves), 2)
      with tempfile.TemporaryDirectory() as tmpdir:
        metrics = safetensors_saver.export_safetensors_with_plan(
            state,
            plan,
            tmpdir,
            output_dtype=None,
            devices=devs,
            async_write=True,
        )
        reloaded = {}
        for sp in metrics.shard_paths:
          reloaded.update(safe_flax.load_file(sp))
        self.assertEqual(
            set(reloaded.keys()), {s.hf_key for s in specs}
        )
        for spec in specs:
          got = np.asarray(reloaded[spec.hf_key])
          want = np.asarray(jax.device_get(state[spec.source_key]))
          self.assertEqual(got.dtype, want.dtype, msg=spec.hf_key)
          np.testing.assert_array_equal(got, want, err_msg=spec.hf_key)

  def test_num_ranks_exceeds_devices_and_per_wave_false_bytes_chunks(self):
    devs = jax.devices()[:2]
    self.assertLen(devs, 2)
    mesh = jax.sharding.Mesh(np.array(devs), ('rank',))
    p_rank = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('rank')
    )
    state = {
        f'w{i}': jax.device_put(
            jnp.arange(2048, dtype=jnp.float32).reshape(2, 1024) + i * 100.0,
            p_rank,
        )
        for i in range(4)
    }
    specs = [
        safetensors_saver.TensorExportSpec(
            f'w{i}', f'model.w{i}', nbytes=2048 * 4
        )
        for i in range(4)
    ]
    # P0 #5: plan.num_ranks (4) > len(devices) (2).
    plan = safetensors_saver.build_save_plan(
        specs, num_ranks=4, max_bucket_bytes=1 << 20, per_wave_shards=False
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      metrics = safetensors_saver.export_safetensors_with_plan(
          state, plan, tmpdir, output_dtype=jnp.float32, devices=devs
      )
      self.assertLen(metrics.shard_paths, 4)
      reloaded = {}
      for sp in metrics.shard_paths:
        reloaded.update(safe_flax.load_file(sp))
      for i in range(4):
        np.testing.assert_array_equal(
            np.asarray(reloaded[f'model.w{i}']),
            np.asarray(jax.device_get(state[f'w{i}'])),
        )

      # P0 #4: per_wave_shards=False must use resolved backend (bytes chunks).
      write_types: list[type[object]] = []

      class _BytesFile:

        def __init__(self, inner):
          self._inner = inner

        def write(self, data):
          write_types.append(type(data))
          return self._inner.write(data)

        def close(self):
          self._inner.close()

        def __enter__(self):
          return self

        def __exit__(self, *args):
          self.close()

      class _BytesBackend(safetensors_saver.StorageBackend):

        def open_writer(self, path: str):
          return _BytesFile(open(path, 'wb'))

      bytes_be = _BytesBackend('local', False, 4, 4096, True)
      out_bytes = os.path.join(tmpdir, 'bytes_out')
      with mock.patch.object(
          safetensors_saver, 'resolve_storage_backend', return_value=bytes_be
      ):
        m_bytes = safetensors_saver.export_safetensors_with_plan(
            state, plan, out_bytes, output_dtype=jnp.float32, devices=devs
        )
      self.assertNotEmpty(write_types)
      self.assertTrue(all(t is bytes for t in write_types))
      reloaded_bytes = {}
      for sp in m_bytes.shard_paths:
        reloaded_bytes.update(safe_flax.load_file(sp))
      for i in range(4):
        np.testing.assert_array_equal(
            np.asarray(reloaded_bytes[f'model.w{i}']),
            np.asarray(jax.device_get(state[f'w{i}'])),
        )

  def test_multihost_barriers_devices_none_stale_cleanup_and_guards(self):
    # P2 #15: 0-byte TensorExportSpec builds without KeyError.
    zero_spec = safetensors_saver.TensorExportSpec('z', 'model.z', nbytes=0)
    zero_plan = safetensors_saver.build_save_plan([zero_spec], num_ranks=1)
    self.assertEqual(zero_plan.weight_map, {'model.z': 'model.safetensors'})

    # New P1 #4: flush_wave_to_files fails fast on unmatched rank.
    with self.assertRaises(KeyError):
      safetensors_saver.flush_wave_to_files(
          {}, {0: {'w': np.ones(4, dtype=np.float32)}}
      )

    state = {
        f't{i}': jnp.full((8, 8), float(i), dtype=jnp.float32) for i in range(4)
    }
    specs = [
        safetensors_saver.TensorExportSpec(
            f't{i}', f'model.t{i}', nbytes=64 * 4
        )
        for i in range(4)
    ]
    plan = safetensors_saver.build_save_plan(
        specs, num_ranks=4, max_bucket_bytes=1 << 20, per_wave_shards=False
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      # Stale shards cleaned on reused output_dir + 3 barriers on multihost.
      (epath.Path(tmpdir) / 'model.safetensors').write_bytes(b'stale')
      (epath.Path(tmpdir) / 'model-00009-of-00009.safetensors').write_bytes(
          b'stale'
      )
      (epath.Path(tmpdir) / 'model.safetensors.index.json').write_text('{}')
      barriers: list[str] = []
      fake_devs = [
          types.SimpleNamespace(process_index=0),
          types.SimpleNamespace(process_index=1),
      ]
      with (
          mock.patch.object(jax, 'process_count', return_value=2),
          mock.patch.object(jax, 'process_index', return_value=1),
          mock.patch.object(jax, 'devices', return_value=fake_devs),
      ):
        # P1 #9: devices=None resolves jax.devices() and filters to proc 1.
        p1_files, p1_paths, _ = safetensors_saver.open_local_shard_files(
            state, plan, tmpdir, jnp.float32, devices=None
        )
        try:
          self.assertEqual(sorted(p1_files.keys()), [1, 3])
          self.assertLen(p1_paths, 2)
        finally:
          for fh in p1_files.values():
            fh.close()
      with (
          mock.patch.object(jax, 'process_count', return_value=2),
          mock.patch.object(jax, 'process_index', return_value=0),
          mock.patch.object(
              multihost_utils,
              'sync_global_devices',
              side_effect=barriers.append,
          ),
      ):
        safetensors_saver.export_safetensors_with_plan(
            state,
            plan,
            tmpdir,
            output_dtype=jnp.float32,
            devices=jax.devices()[:1],
        )
      self.assertLen(barriers, 3)
      self.assertIn(':prepare:', barriers[0])
      self.assertIn(':pre_index:', barriers[1])
      self.assertIn(':post_index:', barriers[2])
      self.assertFalse(
          os.path.exists(os.path.join(tmpdir, 'model.safetensors'))
      )
      self.assertFalse(
          os.path.exists(
              os.path.join(tmpdir, 'model-00009-of-00009.safetensors')
          )
      )

      # Pathways (process_count == 1) must fire zero barriers.
      single_barriers: list[str] = []
      with mock.patch.object(
          multihost_utils,
          'sync_global_devices',
          side_effect=single_barriers.append,
      ):
        safetensors_saver.export_safetensors_with_plan(
            state, plan, tmpdir, output_dtype=jnp.float32, devices=jax.devices()
        )
      self.assertEmpty(single_barriers)

      # P1 #11: max_write_threads caps concurrency; error still closes handles.
      active_now = 0
      peak_active = 0
      lock = threading.Lock()
      orig_write = safetensors_saver._write_rank_tensors

      def _tracking_write(tgt, tensors, buf=None, sem=None):
        nonlocal active_now, peak_active
        with sem if sem is not None else types.SimpleNamespace(
            __enter__=lambda s: None, __exit__=lambda s, *a: None
        ):
          with lock:
            active_now += 1
            peak_active = max(peak_active, active_now)
          time.sleep(0.02)
          orig_write(tgt, tensors, buf, None)
          with lock:
            active_now -= 1

      with mock.patch.object(
          safetensors_saver, '_write_rank_tensors', side_effect=_tracking_write
      ):
        safetensors_saver.export_safetensors_with_plan(
            state,
            plan,
            tmpdir,
            output_dtype=jnp.float32,
            devices=jax.devices()[:1],
            max_write_threads=2,
        )
      self.assertLessEqual(peak_active, 2)

      closed_flags: list[bool] = []

      class _TrackingHandle:

        def write(self, data):
          del data

        def close(self):
          closed_flags.append(True)

      err_dir = os.path.join(tmpdir, 'err_out')
      with (
          mock.patch.object(
              safetensors_saver,
              'open_local_shard_files',
              return_value=({0: _TrackingHandle()}, [err_dir], 1),
          ),
          mock.patch.object(
              safetensors_saver,
              'step_wave_to_host',
              side_effect=RuntimeError('boom'),
          ),
      ):
        with self.assertRaisesRegex(RuntimeError, 'boom'):
          safetensors_saver.export_safetensors_with_plan(
              state, plan, err_dir, devices=jax.devices()[:1]
          )
      self.assertEqual(closed_flags, [True])


class SaveSafetensorsTest(absltest.TestCase):

  def test_single_shard_identity_round_trip(self):
    out = self.create_tempdir().full_path
    tensors = {
        'a.w': jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
        'b': np.ones((4,), np.int32),
    }
    result = safetensors_saver.save_safetensors(tensors, out)
    self.assertEqual(result.paths, (f'{out}/model.safetensors',))
    self.assertIsNone(result.checksums)
    written = safe_np.load_file(result.paths[0])
    for key, value in tensors.items():
      np.testing.assert_array_equal(written[key], np.asarray(value))

  def test_multi_shard_index_is_commit_marker(self):
    out = os.path.join(self.create_tempdir().full_path, 'step')
    os.makedirs(out)
    with open(os.path.join(out, 'stale.txt'), 'w') as f:
      f.write('x')
    tensors = {f't{i}': np.full((4,), i, np.float32) for i in range(3)}
    result = safetensors_saver.save_safetensors(
        tensors, out, max_shard_bytes=20
    )
    self.assertEqual(result.paths[-1], f'{out}/model.safetensors.index.json')
    self.assertLen(result.paths, 4)
    self.assertCountEqual(
        os.listdir(out), [os.path.basename(p) for p in result.paths]
    )
    with open(result.paths[-1]) as f:
      index = json.load(f)
    self.assertEqual(index['metadata']['total_size'], 48)
    self.assertEqual(
        index['weight_map']['t2'], 'model-00003-of-00003.safetensors'
    )

  def test_manifest_drops_tied_target_and_checksums_written_tensors(self):
    out = self.create_tempdir().full_path
    manifest = {
        'rules': [
            {'source_keys': ['tunix.w'], 'target_keys': ['hf.w'], 'ops': []},
            {'source_keys': ['embed.w'], 'target_keys': ['embed.w'], 'ops': []},
        ],
        'tied_keys': [['embed.w', 'lm_head.w']],
    }
    seen = []

    def checksum_fn(host):
      seen.append(sorted(host))
      return {k: float(v.sum()) for k, v in host.items()}

    result = safetensors_saver.save_safetensors(
        {
            'tunix.w': np.ones((2,), np.float32),
            'embed.w': np.full((2,), 3.0, np.float32),
        },
        out,
        to_hf_manifest=manifest,
        checksum_fn=checksum_fn,
    )
    self.assertEqual(seen, [['embed.w', 'hf.w']])
    self.assertEqual(result.checksums, {'embed.w': 6.0, 'hf.w': 2.0})
    self.assertEqual(
        sorted(safe_np.load_file(result.paths[0])), ['embed.w', 'hf.w']
    )

  def test_partially_consumed_multi_source_rule_raises(self):
    manifest = {
        'rules': [{
            'source_keys': ['q', 'k'],
            'target_keys': ['qk'],
            'ops': [{'concat': {'axis': 0, 'sizes': [1, 1]}}],
        }]
    }
    with self.assertRaisesRegex(ValueError, r"missing source keys: \['k'\]"):
      safetensors_saver.save_safetensors(
          {'q': np.ones(1, np.float32)},
          self.create_tempdir().full_path,
          to_hf_manifest=manifest,
      )

  def test_rejects_invalid_arguments(self):
    tensors = {'a': np.ones((2,), np.float32)}
    with self.assertRaisesRegex(ValueError, 'must not be empty'):
      safetensors_saver.save_safetensors(tensors, '')
    for bad_max in (0, -4, True, 1.5):
      with self.assertRaisesRegex(ValueError, 'max_shard_bytes'):
        safetensors_saver.save_safetensors(
            tensors, '/tmp/x', max_shard_bytes=bad_max  # type: ignore[arg-type]
        )
    out = self.create_tempdir().full_path
    with self.assertRaisesRegex(ValueError, 'Unsupported safetensors dtype'):
      safetensors_saver.save_safetensors(
          {'c': np.ones((2,), np.complex64)}, out
      )

  def test_oversized_single_tensor_and_large_max_shard_bytes(self):
    tensors = {
        'small0': np.ones((1,), np.float32),  # 4 B
        'huge': np.arange(6, dtype=np.float32),  # 24 B > max_shard_bytes
        'small1': np.full((1,), 2.0, np.float32),  # 4 B
    }
    shards = safetensors_saver.partition_safetensors_shards(
        tensors, max_shard_bytes=8
    )
    self.assertEqual(
        [[k for k, _ in shard] for shard in shards],
        [['small0'], ['huge'], ['small1']],
    )
    out = self.create_tempdir().full_path
    result = safetensors_saver.save_safetensors(
        tensors,
        out,
        max_shard_bytes=1 << 40,
    )
    self.assertEqual(result.paths, (f'{out}/model.safetensors',))
    loaded = safe_np.load_file(result.paths[0])
    for key, val in tensors.items():
      np.testing.assert_array_equal(loaded[key], val)

  def test_round_trip_scalar_bfloat16_and_non_contiguous_view(self):
    out = self.create_tempdir().full_path
    scalar_f32 = np.array(3.5, dtype=np.float32)
    scalar_bf16 = np.asarray(jnp.array(-1.25, dtype=jnp.bfloat16))
    vec_bf16 = np.asarray(jnp.array([1.5, -2.25], dtype=jnp.bfloat16))
    transposed = np.arange(6, dtype=np.float32).reshape(2, 3).T
    self.assertFalse(transposed.flags.c_contiguous)
    tensors = {
        'scalar_f32': scalar_f32,
        'scalar_bf16': scalar_bf16,
        'vec_bf16': vec_bf16,
        'transposed': transposed,
    }
    result = safetensors_saver.save_safetensors(tensors, out)
    raw = epath.Path(result.paths[0]).read_bytes()
    header_len = int.from_bytes(raw[:8], 'little')
    header = json.loads(raw[8 : 8 + header_len].decode('utf-8'))
    self.assertEqual(header['scalar_f32']['shape'], [])
    self.assertEqual(header['scalar_bf16']['shape'], [])
    self.assertEqual(header['vec_bf16']['shape'], [2])
    self.assertEqual(header['transposed']['shape'], [3, 2])
    loaded = safe_np.load_file(result.paths[0])
    for key, expected in tensors.items():
      self.assertEqual(loaded[key].shape, expected.shape)
      self.assertEqual(loaded[key].dtype, expected.dtype)
      np.testing.assert_array_equal(loaded[key], expected)

  def test_multiprocess_barrier_runs_after_commit(self):
    out = self.create_tempdir().full_path
    calls = []
    allgather_calls = []
    sharded_z = mock.Mock(spec=jax.Array)
    sharded_z.is_fully_addressable = False
    sharded_a = mock.Mock(spec=jax.Array)
    sharded_a.is_fully_addressable = False
    local_jax = jnp.arange(3, dtype=jnp.float32)
    host_np = np.array([4.0, 5.0], dtype=np.float32)

    def fake_allgather(x, tiled):
      self.assertIsInstance(x, jax.Array)
      self.assertFalse(x.is_fully_addressable)
      allgather_calls.append((x, tiled))
      return (
          np.ones((2,), np.float32)
          if x is sharded_a
          else np.full((2,), 9.0, np.float32)
      )

    def barrier(name):
      calls.append((name, os.path.exists(f'{out}/model.safetensors')))

    with (
        mock.patch.object(jax, 'process_count', return_value=2),
        mock.patch.object(jax, 'process_index', return_value=0),
        mock.patch.object(
            multihost_utils, 'process_allgather', side_effect=fake_allgather
        ),
        mock.patch.object(
            multihost_utils, 'sync_global_devices', side_effect=barrier
        ),
        mock.patch.object(
            multihost_utils,
            'broadcast_one_to_all',
            side_effect=lambda x, is_source: x,
        ),
    ):
      result = safetensors_saver.save_safetensors(
          {'z': sharded_z, 'm': local_jax, 'a': sharded_a, 'h': host_np}, out
      )
    self.assertEqual(allgather_calls, [(sharded_a, True), (sharded_z, True)])
    self.assertEqual(calls, [(f'tunix:save_safetensors:{out}', True)])
    self.assertEqual(result.paths, (f'{out}/model.safetensors',))
    loaded = safe_np.load_file(result.paths[0])
    np.testing.assert_array_equal(loaded['a'], np.ones((2,), np.float32))
    np.testing.assert_array_equal(loaded['z'], np.full((2,), 9.0, np.float32))
    np.testing.assert_array_equal(loaded['m'], np.asarray(local_jax))
    np.testing.assert_array_equal(loaded['h'], host_np)

  def test_multiprocess_worker_gathers_sharded_only_and_skips_writes(self):
    out = self.create_tempdir().full_path
    allgather_calls = []
    synced = []
    sharded_z = mock.Mock(spec=jax.Array)
    sharded_z.is_fully_addressable = False
    sharded_a = mock.Mock(spec=jax.Array)
    sharded_a.is_fully_addressable = False
    local_jax = jnp.arange(3, dtype=jnp.float32)
    host_np = np.array([4.0, 5.0], dtype=np.float32)
    host0_payload = json.dumps({
        'paths': [f'{out}/model.safetensors'],
        'checksums': {'a': 2.0},
    }).encode('utf-8')
    payload_buf = np.frombuffer(host0_payload, dtype=np.uint8)
    bcast_replies = iter([np.int32(payload_buf.size), payload_buf])

    def fake_allgather(x, tiled):
      allgather_calls.append((x, tiled))
      return np.ones((2,), np.float32)

    with (
        mock.patch.object(jax, 'process_count', return_value=2),
        mock.patch.object(jax, 'process_index', return_value=1),
        mock.patch.object(
            multihost_utils, 'process_allgather', side_effect=fake_allgather
        ),
        mock.patch.object(jax, 'device_get') as mock_dg,
        mock.patch.object(
            multihost_utils, 'sync_global_devices', side_effect=synced.append
        ),
        mock.patch.object(
            multihost_utils,
            'broadcast_one_to_all',
            side_effect=lambda x, is_source: next(bcast_replies),
        ),
    ):
      result = safetensors_saver.save_safetensors(
          {'z': sharded_z, 'm': local_jax, 'a': sharded_a, 'h': host_np}, out
      )
    self.assertEqual(allgather_calls, [(sharded_a, True), (sharded_z, True)])
    mock_dg.assert_not_called()
    self.assertEqual(os.listdir(out), [])
    self.assertEqual(synced, [f'tunix:save_safetensors:{out}'])
    self.assertEqual(
        result,
        safetensors_saver.SafetensorsSaveResult(
            paths=(f'{out}/model.safetensors',), checksums={'a': 2.0}
        ),
    )

  def test_multihost_host0_error_broadcasts_before_raising(self):
    out = self.create_tempdir().full_path
    synced = []
    broadcast_bufs = []

    def fake_broadcast(x, is_source):
      self.assertTrue(is_source)
      if isinstance(x, np.ndarray):
        broadcast_bufs.append(x.tobytes())
      return x

    with (
        mock.patch.object(jax, 'process_count', return_value=2),
        mock.patch.object(jax, 'process_index', return_value=0),
        mock.patch.object(
            multihost_utils, 'sync_global_devices', side_effect=synced.append
        ),
        mock.patch.object(
            multihost_utils, 'broadcast_one_to_all', side_effect=fake_broadcast
        ),
    ):
      with self.assertRaisesRegex(ValueError, 'Unsupported safetensors dtype'):
        safetensors_saver.save_safetensors(
            {'bad': np.ones((2,), np.complex64)}, out
        )
    self.assertEqual(synced, [f'tunix:save_safetensors:{out}'])
    payload = json.loads(broadcast_bufs[-1])
    self.assertIn('error', payload)
    self.assertIn('ValueError', payload['error'])

    synced.clear()
    broadcast_bufs.clear()
    long_msg = 'x' * 5000
    with (
        mock.patch.object(jax, 'process_count', return_value=2),
        mock.patch.object(jax, 'process_index', return_value=0),
        mock.patch.object(
            safetensors_saver,
            'publish_safetensors_shards',
            side_effect=OSError(long_msg),
        ),
        mock.patch.object(
            multihost_utils, 'sync_global_devices', side_effect=synced.append
        ),
        mock.patch.object(
            multihost_utils, 'broadcast_one_to_all', side_effect=fake_broadcast
        ),
    ):
      with self.assertRaisesRegex(OSError, 'x{100}'):
        safetensors_saver.save_safetensors(
            {'ok': np.ones((2,), np.float32)}, out
        )
    self.assertEqual(synced, [f'tunix:save_safetensors:{out}'])
    payload = json.loads(broadcast_bufs[-1])
    self.assertLessEqual(
        len(payload['error']), safetensors_saver._MAX_BROADCAST_ERROR_CHARS
    )
    self.assertStartsWith(payload['error'], 'OSError: ')

  def test_multihost_worker_raises_on_host0_error_broadcast(self):
    out = self.create_tempdir().full_path
    synced = []
    err_bytes = json.dumps({'error': 'OSError: disk full'}).encode('utf-8')
    err_buf = np.frombuffer(err_bytes, dtype=np.uint8)
    bcast_replies = iter([np.int32(err_buf.size), err_buf])

    with (
        mock.patch.object(jax, 'process_count', return_value=2),
        mock.patch.object(jax, 'process_index', return_value=1),
        mock.patch.object(
            multihost_utils, 'sync_global_devices', side_effect=synced.append
        ),
        mock.patch.object(
            multihost_utils,
            'broadcast_one_to_all',
            side_effect=lambda x, is_source: next(bcast_replies),
        ),
    ):
      with self.assertRaisesRegex(
          RuntimeError, 'save_safetensors failed on process 0: OSError: disk'
      ):
        safetensors_saver.save_safetensors(
            {'a': np.ones((2,), np.float32)}, out
        )
    self.assertEqual(synced, [f'tunix:save_safetensors:{out}'])

  def test_invalid_manifest_fails_before_collectives(self):
    out = self.create_tempdir().full_path
    for rank in (0, 1):
      with (
          mock.patch.object(jax, 'process_count', return_value=2),
          mock.patch.object(jax, 'process_index', return_value=rank),
          mock.patch.object(multihost_utils, 'process_allgather') as mock_ag,
          mock.patch.object(
              multihost_utils, 'sync_global_devices'
          ) as mock_sync,
          mock.patch.object(
              multihost_utils, 'broadcast_one_to_all'
          ) as mock_bcast,
      ):
        with self.assertRaisesRegex(ValueError, 'Manifest must be a mapping'):
          safetensors_saver.save_safetensors(
              {'a': np.ones((2,), np.float32)},
              out,
              to_hf_manifest={'not_rules': []},
          )
        mock_ag.assert_not_called()
        mock_sync.assert_not_called()
        mock_bcast.assert_not_called()

  def test_modular_header_shard_index_and_partition_utilities(self):
    out = epath.Path(self.create_tempdir().full_path)
    transposed = np.arange(6, dtype=np.float32).reshape(2, 3).T
    scalar = np.array(3.5, dtype=np.float32)
    header_bytes, payloads, total = safetensors_saver.encode_safetensors_header(
        [('t', transposed), ('s', scalar)]
    )
    header_len = int.from_bytes(header_bytes[:8], 'little')
    self.assertEqual(header_len, len(header_bytes) - 8)
    self.assertEqual(header_len % 8, 0)
    meta = json.loads(header_bytes[8:].decode('utf-8'))
    self.assertEqual(meta['t']['shape'], [3, 2])
    self.assertEqual(meta['s']['shape'], [])
    self.assertEqual(total, 28)
    self.assertEqual(sum(p.nbytes for p in payloads), total)

    shard_path = out / 'custom.safetensors'
    written_bytes = safetensors_saver.write_safetensors_shard(
        shard_path, [('t', transposed), ('s', scalar)]
    )
    self.assertEqual(written_bytes, 28)
    self.assertEqual([p.name for p in out.iterdir()], ['custom.safetensors'])
    loaded = safe_np.load_file(str(shard_path))
    np.testing.assert_array_equal(loaded['t'], transposed)
    np.testing.assert_array_equal(loaded['s'], scalar)

    index_path = safetensors_saver.write_safetensors_index(
        out, {'t': 'custom.safetensors'}, 28
    )
    self.assertEqual(index_path.name, 'model.safetensors.index.json')
    self.assertEqual(
        json.loads(index_path.read_text()),
        {
            'metadata': {'total_size': 28},
            'weight_map': {'t': 'custom.safetensors'},
        },
    )
    self.assertEqual(
        safetensors_saver.partition_safetensors_shards({}, 16), [[]]
    )
    with self.assertRaisesRegex(ValueError, 'max_shard_bytes'):
      safetensors_saver.partition_safetensors_shards({}, 0)

  def test_write_safetensors_shard_chunks_memoryview_writes(self):
    out = epath.Path(self.create_tempdir().full_path)
    shard_path = out / 'chunked.safetensors'
    arr = np.arange(10, dtype=np.float32)  # 40 bytes
    written_chunks: list[tuple[type[object], int]] = []
    real_open = type(shard_path).open

    class _RecordingWriter:

      def __init__(self, inner):
        self._inner = inner

      def write(self, data):
        written_chunks.append((type(data), len(data)))
        return self._inner.write(data)

      def __enter__(self):
        self._inner.__enter__()
        return self

      def __exit__(self, *args):
        return self._inner.__exit__(*args)

    def recording_open(path, mode='r', *args, **kwargs):
      return _RecordingWriter(real_open(path, mode, *args, **kwargs))

    with (
        mock.patch.object(safetensors_saver, '_WRITE_CHUNK_BYTES', 8),
        mock.patch.object(
            type(shard_path), 'open', autospec=True, side_effect=recording_open
        ),
    ):
      total = safetensors_saver.write_safetensors_shard(
          shard_path, [('w', arr)]
      )
    self.assertEqual(total, 40)
    self.assertEqual(written_chunks[0][0], bytes)
    self.assertEqual(written_chunks[1:], [(memoryview, 8)] * 5)
    np.testing.assert_array_equal(safe_np.load_file(str(shard_path))['w'], arr)

  def test_remove_tree_clears_children_when_rmtree_drops_only_the_dir(self):
    root = epath.Path(self.create_tempdir().full_path)
    out = root / 'export'
    (out / 'nested').mkdir(parents=True)
    (out / 'model.safetensors').write_bytes(b'x')
    (out / 'nested' / 'shard.safetensors').write_bytes(b'y')

    def placeholder_only_rmtree(path, missing_ok=False):
      # Object store with a `dir/` placeholder: deletes only an empty "dir" and
      # still reports success when children remain.
      del missing_ok
      if not list(path.iterdir()):
        path.rmdir()

    with mock.patch.object(
        type(out), 'rmtree', autospec=True, side_effect=placeholder_only_rmtree
    ):
      safetensors_saver.remove_tree(out)
      safetensors_saver.remove_tree(root / 'missing')
    self.assertEqual(list(root.iterdir()), [])

  def test_save_replaces_stale_shards_in_existing_output_dir(self):
    out = epath.Path(self.create_tempdir().full_path)
    (out / 'model-00003-of-00003.safetensors').write_bytes(b'stale')
    safetensors_saver.save_safetensors({'a': np.ones((2,), np.float32)}, out)
    self.assertEqual([p.name for p in out.iterdir()], ['model.safetensors'])

  def test_overwrite_unlinks_index_before_shards(self):
    out = epath.Path(self.create_tempdir().full_path)
    (out / 'model-00001-of-00002.safetensors').write_bytes(b's1')
    (out / 'model-00002-of-00002.safetensors').write_bytes(b's2')
    (out / 'model.safetensors.index.json').write_text('{}')
    unlinked: list[str] = []
    real_unlink = type(out).unlink

    def recording_unlink(path, missing_ok=False):
      if path.exists():
        unlinked.append(path.name)
      return real_unlink(path, missing_ok=missing_ok)

    with mock.patch.object(
        type(out), 'unlink', autospec=True, side_effect=recording_unlink
    ):
      safetensors_saver.publish_safetensors_shards(
          out, [[('a', np.ones((2,), np.float32))]]
      )
    self.assertEqual(unlinked[0], 'model.safetensors.index.json')
    self.assertCountEqual(
        unlinked[1:],
        [
            'model-00001-of-00002.safetensors',
            'model-00002-of-00002.safetensors',
        ],
    )

  def test_unsupported_dtype_fails_before_touching_prior_export(self):
    out = epath.Path(self.create_tempdir().full_path)
    prior = {
        'model-00001-of-00002.safetensors': b's1',
        'model-00002-of-00002.safetensors': b's2',
        'model.safetensors.index.json': b'{}',
    }
    for name, data in prior.items():
      (out / name).write_bytes(data)
    shards = [
        [('a', np.ones((2,), np.float32))],
        [('c', np.ones((2,), np.complex64))],
    ]
    with self.assertRaisesRegex(
        ValueError, r"Unsupported safetensors dtype .*complex64.* for 'c'"
    ):
      safetensors_saver.publish_safetensors_shards(out, shards)
    self.assertEqual({p.name: p.read_bytes() for p in out.iterdir()}, prior)

  def test_failed_shard_and_index_writes_clean_up_tmp_files(self):
    out = epath.Path(self.create_tempdir().full_path)
    shard_path = out / 'model.safetensors'
    real_open = type(shard_path).open

    class _FailingWriter:

      def __init__(self, inner):
        self._inner = inner

      def write(self, data):
        del data
        raise OSError('simulated write error')

      def __enter__(self):
        self._inner.__enter__()
        return self

      def __exit__(self, *args):
        return self._inner.__exit__(*args)

    with mock.patch.object(
        type(shard_path),
        'open',
        autospec=True,
        side_effect=lambda p, m='r', *a, **kw: _FailingWriter(
            real_open(p, m, *a, **kw)
        ),
    ):
      with self.assertRaisesRegex(OSError, 'simulated write error'):
        safetensors_saver.write_safetensors_shard(
            shard_path, [('a', np.ones((2,), np.float32))]
        )
    self.assertEqual(list(out.iterdir()), [])

    with mock.patch.object(
        type(shard_path), 'replace', autospec=True, side_effect=OSError('boom')
    ):
      with self.assertRaisesRegex(OSError, 'boom'):
        safetensors_saver.write_safetensors_shard(
            shard_path, [('a', np.ones((2,), np.float32))]
        )
      self.assertEqual(list(out.iterdir()), [])

      with self.assertRaisesRegex(OSError, 'boom'):
        safetensors_saver.write_safetensors_index(
            out, {'a': 'model.safetensors'}, 8
        )
      self.assertEqual(list(out.iterdir()), [])

  def test_non_json_serializable_checksum_fails_before_publish_and_broadcasts(
      self,
  ):
    bad_checksum = mock.Mock(return_value={'a': np.float32(1.0)})
    uncreated = epath.Path(self.create_tempdir().full_path) / 'never_created'
    with self.assertRaises(TypeError):
      safetensors_saver.save_safetensors(
          {'a': np.ones((2,), np.float32)},
          uncreated,
          checksum_fn=bad_checksum,
      )
    bad_checksum.assert_called_once()
    self.assertFalse(uncreated.exists())

    out = epath.Path(self.create_tempdir().full_path)
    sentinel = out / 'keep.txt'
    sentinel.write_text('intact')
    synced = []
    broadcast_bufs = []

    def fake_broadcast(x, is_source):
      self.assertTrue(is_source)
      if isinstance(x, np.ndarray):
        broadcast_bufs.append(x.tobytes())
      return x

    with (
        mock.patch.object(jax, 'process_count', return_value=2),
        mock.patch.object(jax, 'process_index', return_value=0),
        mock.patch.object(
            multihost_utils, 'sync_global_devices', side_effect=synced.append
        ),
        mock.patch.object(
            multihost_utils, 'broadcast_one_to_all', side_effect=fake_broadcast
        ),
    ):
      with self.assertRaises(TypeError):
        safetensors_saver.save_safetensors(
            {'a': np.ones((2,), np.float32)},
            out,
            checksum_fn=bad_checksum,
        )
    self.assertEqual([p.name for p in out.iterdir()], ['keep.txt'])
    self.assertEqual(sentinel.read_text(), 'intact')
    self.assertEqual(synced, [f'tunix:save_safetensors:{out}'])
    payload = json.loads(broadcast_bufs[-1])
    self.assertIn('error', payload)
    self.assertIn('TypeError', payload['error'])

  def test_cleanup_unlink_error_does_not_mask_root_exception(self):
    out = epath.Path(self.create_tempdir().full_path)
    shard_path = out / 'model.safetensors'
    items = [('a', np.ones((2,), np.float32))]

    for exc_cls, msg in (
        (OSError, 'disk full'),
        (KeyboardInterrupt, 'interrupted'),
    ):
      with (
          mock.patch.object(
              type(shard_path),
              'unlink',
              autospec=True,
              side_effect=PermissionError('unlink denied'),
          ) as mock_unlink,
          mock.patch.object(
              type(shard_path),
              'replace',
              autospec=True,
              side_effect=exc_cls(msg),
          ),
      ):
        with self.assertRaisesRegex(exc_cls, msg):
          safetensors_saver.write_safetensors_shard(shard_path, items)
        with self.assertRaisesRegex(exc_cls, msg):
          safetensors_saver.write_safetensors_index(
              out, {'a': 'model.safetensors'}, 8
          )
      self.assertEqual(mock_unlink.call_count, 2)


if __name__ == '__main__':
  absltest.main()
