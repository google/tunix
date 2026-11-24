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
import dataclasses
import fcntl
import io
import json
import os
import queue
import re
import select
import selectors
import sys
import tempfile
import threading
import types
from typing import Any, cast
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
from tunix.models import hf_mapping
from tunix.models import safetensors_io
from tunix.models import safetensors_saver
from tunix.models import safetensors_stage_worker

jax.config.update('jax_num_cpu_devices', 64)


class _ToyModule(nnx.Module):

  def __init__(self, rngs: nnx.Rngs):
    self.w = nnx.Param(jax.random.normal(rngs.params(), (8, 4)))
    self.experts = nnx.Param(jax.random.normal(rngs.params(), (2, 4, 6)))


class _RemoteProcPath(safetensors_io.ProcPoolPath):
  """Reader for the fake `/rfs/` backend registered by each test."""

  proc_env_var = 'TUNIX_REMOTE_PROCS'


def _validate_rfs(path: str) -> None:
  if '%' in path or path.rstrip('/') == '/rfs':
    raise ValueError(f'Invalid remote output_dir {path!r}.')


def _remote_spec(**overrides: Any) -> safetensors_io.StorageBackendSpec:
  """Returns a POSIX-backed spec standing in for a registered remote FS."""
  return safetensors_io.StorageBackendSpec(**{
      'path_cls': _RemoteProcPath,
      'validate': _validate_rfs,
      'write_flags': lambda: (),
      'open_stage_file': lambda path: open(path, 'wb'),
      **overrides,
  })


class SafetensorsSaverTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.enter_context(
        mock.patch.dict(
            safetensors_io.STORAGE_BACKENDS, {'/rfs/': _remote_spec()}
        )
    )

  def test_join_and_extract_flat_state(self):
    self.assertEqual(safetensors_saver.join_path(('a', 0, 'b')), 'a.0.b')
    mod = _ToyModule(nnx.Rngs(params=0))
    flat = safetensors_saver.extract_flat_state_dict(mod)
    self.assertIn('w', flat)
    self.assertIn('experts', flat)
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
      loaded_0 = safe_flax.load_file(metrics.shard_paths[0])
      loaded_1 = safe_flax.load_file(metrics.shard_paths[1])
      merged = {**loaded_0, **loaded_1}
      self.assertIn('model.w.weight', merged)
      self.assertIn('model.experts.1.weight', merged)
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
      remote_be = safetensors_saver.resolve_storage_backend('/rfs/cell-d/out')
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
          ('gcs', True, 64, 8 << 20, True),
      )
      self.assertEqual(
          (
              remote_be.kind,
              remote_be.default_per_wave_shards,
              remote_be.default_write_threads,
              remote_be.default_write_buffer_bytes,
              remote_be.requires_bytes_chunks,
          ),
          ('remote', True, 64, 8 << 20, True),
      )
      remote_sim_dir = os.path.join(tmpdir, 'remote_sim')
      remote_be.makedirs(remote_sim_dir)
      self.assertTrue(os.path.isdir(remote_sim_dir))
      with remote_be.open_writer(
          os.path.join(remote_sim_dir, 'probe.bin')
      ) as cf:
        cf.write(b'remote_ok')

      recorded_buf_sizes: list[int | None] = []

      class _FakeGcsFs:

        def open_output_stream(self, key: str, buffer_size: int | None = None):
          recorded_buf_sizes.append(buffer_size)
          full = os.path.join(tmpdir, 'gcs_root', key)
          os.makedirs(os.path.dirname(full), exist_ok=True)
          return open(full, 'wb')

      orig_pa_fs = safetensors_saver._pa_fs
      safetensors_saver._pa_fs = types.SimpleNamespace(GcsFileSystem=_FakeGcsFs)
      try:
        gcs_be.makedirs('gs://test-bucket/wave_out')
        with gcs_be.open_writer('gs://test-bucket/probe.bin') as gf:
          gf.write(b'ok')
        metrics_gcs = safetensors_saver.export_safetensors_with_plan(
            flat,
            wave_plan,
            'gs://test-bucket/wave_out',
            output_dtype=jnp.float32,
            devices=jax.devices(),
            async_write=True,
        )
      finally:
        safetensors_saver._pa_fs = orig_pa_fs
      self.assertIn(8 * 1024 * 1024, recorded_buf_sizes)
      self.assertLen(metrics_gcs.shard_paths, 2)
      gcs_local_root = os.path.join(tmpdir, 'gcs_root', 'test-bucket/wave_out')
      self.assertTrue(
          os.path.exists(
              os.path.join(gcs_local_root, 'model.safetensors.index.json')
          )
      )
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
    large_remote = np.arange(10, dtype=np.float32)  # 40 bytes
    expected_payload = hdr + small.tobytes() + large_local.tobytes()
    recorded: dict[str, list[tuple[type[object], int, bytes, object]]] = {
        'local': [],
        'gcs': [],
        'remote': [],
    }

    def _make_backend(
        kind: str, buf_bytes: int, requires_bytes: bool
    ) -> safetensors_saver.StorageBackend:
      class _Backend(safetensors_saver.StorageBackend):

        def open_writer(self, path: str):
          del path

          class _Writer:

            def write(self, d):
              recorded[kind].append(
                  (type(d), len(d), bytes(d), getattr(d, 'obj', None))
              )

            def close(self):
              return None

            def __enter__(self):
              return self

            def __exit__(self, *_):
              self.close()

          return _Writer()

      return _Backend(kind, True, 8, buf_bytes, requires_bytes)

    local_be = _make_backend('local', 0, False)
    gcs_be = _make_backend('gcs', 16, False)
    remote_be = _make_backend('remote', 16, True)
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
        (remote_be, '/rfs/c/s.safetensors', hdr),
        {'s': small.copy(), 'l': large_remote},
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
        [t for t, _, _, _ in recorded['remote']], [bytes, bytes, bytes, bytes]
    )
    self.assertEqual([n for _, n, _, _ in recorded['remote']], [16, 16, 16, 8])
    self.assertEqual(
        b''.join(b for _, _, b, _ in recorded['remote']), expected_payload
    )
    self.assertEqual(
        safetensors_saver.resolve_storage_backend('/rfs/cell-d/path').kind,
        'remote',
    )
    for bad_remote in ('/rfs', '/rfs/', '/rfs/cell2-d/home/user/out%rs=6.3'):
      with self.assertRaises(ValueError):
        safetensors_saver.resolve_storage_backend(bad_remote)
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
    flushed: list[str] = []
    staged_be = safetensors_saver.StorageBackend(
        'remote',
        stage_writer_fn=lambda p: types.SimpleNamespace(
            write=lambda d: flushed.append(f'{p}:{len(d)}'),
            close=lambda: None,
        ),
        flush_fn=lambda: flushed.append('flushed'),
    )
    staged_be.open_writer('/rfs/cell-d/s').write(b'abc')
    staged_be.flush()
    self.assertEqual(flushed, ['/rfs/cell-d/s:3', 'flushed'])

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
    local_cat = safetensors_saver._jit_concat_local_rank_shards(
        (s0.reshape(1, 1024), s1.reshape(1, 2048)), mesh, 'rank'
    )
    self.assertEqual(local_cat.shape, (3072,))
    fused_cat = safetensors_saver._jit_transform_and_concat_fusable_wave(
        (s0.reshape(1, 32, 32), s1.reshape(1, 32, 64)),
        (None, None),
        ((0, 2, 1), (0, 2, 1)),
        jnp.float32,
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('rank')),
    )
    self.assertEqual(fused_cat.shape, (3072,))
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
    targs1, ranks1, items1, _ = safetensors_saver._start_batch_d2h_with_layout(
        [(0, spec_s0)], [s0], devs
    )
    batch_routed, _ = safetensors_saver._finish_batch_d2h(
        targs1, ranks1, items1
    )
    np.testing.assert_allclose(batch_routed[0]['model.s0'], s0)
    targs2, ranks2, items2, _ = safetensors_saver._start_batch_d2h_with_layout(
        [(0, spec_s0), (0, spec_s1)], [s0, s1], devs
    )
    deferred_batch, _ = safetensors_saver._finish_batch_d2h(
        targs2,
        ranks2,
        items2,
        defer_contiguous=True,
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
    routed, _, _ = safetensors_saver.materialize_wave_to_host(
        {'norm': rep_norm, 's0': shd_s0},
        wave_bucket,
        jnp.float32,
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
    spec = safetensors_saver.TensorExportSpec(
        source_key='w', hf_key='w', nbytes=16
    )
    wave = safetensors_saver.build_save_plan(
        [spec], num_ranks=1, max_bucket_bytes=16
    ).waves[0]
    # With one shard file per rank (`limit=1`), a wave must not start writing
    # until the previous wave's flush is done, or tensors land out of order.
    older: concurrent.futures.Future[float] = concurrent.futures.Future()
    older_done_at_write: list[bool] = []

    class _Sink:

      def write(self, data: object) -> int:
        del data
        older_done_at_write.append(older.done())
        return 0

      def __enter__(self) -> '_Sink':
        return self

      def __exit__(self, *_: object) -> bool:
        return False

    sink_backend = dataclasses.replace(
        safetensors_saver._LOCAL_BACKEND,
        default_write_buffer_bytes=0,
        stage_writer_fn=lambda _: _Sink(),
    )
    sink_target = (sink_backend, '/tmp/sink.st', b'')
    threading.Timer(0.05, older.set_result, args=(0.0,)).start()
    pending = [(older, 0)]
    host_wave = {0: {'w': np.zeros(4, dtype=np.float32)}}
    safetensors_saver.submit_and_drain_wave_flush(
        pending, 1, {0: sink_target}, host_wave, wave, 0, 1
    )
    pending[0][0].result()
    self.assertEqual(older_done_at_write, [True])
    # `limit=0` writes synchronously and never queues.
    sync_pending: list[tuple[concurrent.futures.Future[float], int]] = []
    safetensors_saver.submit_and_drain_wave_flush(
        sync_pending, 0, {0: sink_target}, {0: {'w': np.zeros(4)}}, wave, 0, 1
    )
    self.assertEmpty(sync_pending)
    self.assertLen(older_done_at_write, 2)

  def test_moe_auto_chunking_and_grouped_routing_helpers(self):
    spec = safetensors_saver.TensorExportSpec(
        source_key='moe',
        hf_key='layers.0.experts.{expert}.w',
        transpose_axes=(0, 2, 1),
        unstack_axis=0,
        nbytes=32768,
    )
    plan = safetensors_saver.build_save_plan(
        [spec], num_ranks=1, max_bucket_bytes=8192, unstack_counts={'moe': 4}
    )
    self.assertLen(plan.waves, 4)
    c0 = plan.waves[0].rank_assignments[0][0]
    self.assertEqual(c0.unstack_range, (0, 1))
    self.assertEqual(
        safetensors_saver._wave_tensor_key(c0), 'layers.0.experts.0.w'
    )
    self.assertEqual(
        safetensors_saver.expand_spec_keys(c0, {}), ('layers.0.experts.0.w',)
    )
    arr = jnp.arange(4 * 32 * 64, dtype=jnp.float32).reshape(4, 32, 64)
    self.assertEqual(safetensors_saver._chunk_count_and_index(arr, c0), (4, 0))
    devs = jax.devices()[:1]
    mesh = jax.sharding.Mesh(np.array(devs), ('rank',))
    rank_shd = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('rank')
    )
    shd, flat_r = safetensors_saver._resolve_transform_sharding(
        arr, c0, rank_shd, 4
    )
    self.assertEqual(flat_r, 0)
    self.assertIsNone(shd)
    meta, max_recv = safetensors_saver.build_grouped_ragged_meta(
        [2048], [(0, 1)], 2
    )
    self.assertEqual(meta.shape, (2, 4, 2))
    self.assertEqual(max_recv, 1024)
    _, _, _, _, rmax, _, _, rcols, rmeta = (
        safetensors_saver._compute_exchange_plan(
            [jnp.zeros((1024,), dtype=jnp.float32)], [0], [(0,)], 1
        )
    )
    self.assertIsNotNone(rmeta)
    self.assertEqual(rmax, 1024)
    self.assertEqual(rcols, [[1024]])
    sharded, owners, groups, items = safetensors_saver._group_batch_shards(
        [(0, c0), (0, dataclasses.replace(c0, unstack_range=(1, 2)))],
        [arr, arr],
        1,
        rank_shd,
    )
    self.assertLen(sharded, 1)
    self.assertEqual(owners, [0])
    self.assertEqual(groups, ((0, 0),))
    self.assertLen(items[0], 2)
    self.assertFalse(
        safetensors_saver._is_fusable_source_group(
            arr, [(0, c0)], plan.waves[0], 1
        )
    )
    self.assertEqual(
        safetensors_saver._fusable_wave_sources({'moe': arr}, plan.waves[0], 1),
        set(),
    )
    self.assertFalse(
        safetensors_saver._can_fuse_identity_wave(plan.waves[0], {'moe'}, 1)
    )
    cache: dict[str, list[jax.Array | None]] = {}
    popped = safetensors_saver._pop_or_transform_chunk(
        {'moe': arr}, c0, jnp.float32, None, cache
    )
    self.assertEqual(popped.shape, (1, 64, 32))
    w_arrs = safetensors_saver._transform_wave_pairs(
        {'moe': arr},
        plan.waves[1],
        [(0, plan.waves[1].rank_assignments[0][0])],
        jnp.float32,
        devs,
        rank_shd,
        cache,
    )
    self.assertLen(w_arrs, 1)
    self.assertIn('moe', cache)
    done: concurrent.futures.Future[float] = concurrent.futures.Future()
    done.set_result(0.0)
    big = safetensors_saver._MAX_INFLIGHT_FLUSH_BYTES_CAP
    for prior_bytes, limit, want_bytes in (
        (big, 4, [8192]),  # Over the byte bound: the older flush is drained.
        (0, 4, [0, 8192]),  # Under both bounds: both flushes stay in flight.
        (0, 1, [8192]),  # At the wave bound: the older flush is drained.
        (0, 0, [0]),  # Synchronous writes never queue.
    ):
      pending = [(done, prior_bytes)]
      safetensors_saver.submit_and_drain_wave_flush(
          pending, limit, {}, {0: {}}, plan.waves[0], 0, 1
      )
      self.assertEqual([b for _, b in pending], want_bytes)
    pw_plan = safetensors_saver.build_save_plan(
        [spec],
        num_ranks=1,
        max_bucket_bytes=8192,
        unstack_counts={'moe': 4},
        per_wave_shards=True,
    )
    self.assertTrue(pw_plan.per_wave_shards)
    mixed_plan = safetensors_saver.build_save_plan(
        [
            spec,
            safetensors_saver.TensorExportSpec(
                source_key='embed', hf_key='embed.w', nbytes=4096
            ),
        ],
        num_ranks=1,
        max_bucket_bytes=8192,
        unstack_counts={'moe': 4},
    )
    self.assertLen(mixed_plan.waves, 5)
    self.assertIsNone(mixed_plan.waves[0].rank_assignments[0][0].unstack_range)
    self.assertEqual(
        mixed_plan.waves[1].rank_assignments[0][0].unstack_range, (0, 1)
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      metrics = safetensors_saver.export_safetensors_with_plan(
          {'moe': arr}, plan, tmpdir, devices=devs, async_write=True
      )
      self.assertEqual(metrics.num_waves, 4)
      loaded = safe_flax.load_file(os.path.join(tmpdir, 'model.safetensors'))
      for e in range(4):
        np.testing.assert_allclose(
            loaded[f'layers.0.experts.{e}.w'], np.asarray(arr[e].T)
        )
    bf16_arr = jnp.arange(256, dtype=jnp.bfloat16)
    packed = safetensors_saver._pack_bf16_to_u32_d2h(bf16_arr)
    self.assertEqual(packed.ndim, 1)
    self.assertIs(
        safetensors_saver._slice_d2h_target(bf16_arr, 64, False), bf16_arr
    )
    self.assertEqual(
        safetensors_saver._slice_d2h_target(
            jnp.zeros((16, 128), dtype=jnp.bfloat16), 256, True
        ).shape,
        (8, 128),
    )
    self.assertEqual(
        safetensors_saver._slice_d2h_target(bf16_arr, 64, True).shape, (64,)
    )
    bf16_2d = bf16_arr.reshape(2, 128)
    host_bf16, _ = safetensors_saver._finish_batch_d2h(
        [bf16_2d],
        [0],
        {0: [(0, c0, (16, 16))]},
        defer_contiguous=True,
    )
    np.testing.assert_array_equal(
        np.asarray(host_bf16[0]['layers.0.experts.0.w']),
        np.asarray(bf16_arr).reshape(16, 16),
    )
    # Packed uint32 (M, 128) zero-copy decodes back to bit-exact bf16 on host.
    u32_2d = jax.lax.bitcast_convert_type(
        bf16_arr.reshape(1, 128, 2), jnp.uint32
    )
    host_u32, _ = safetensors_saver._finish_batch_d2h(
        [u32_2d],
        [0],
        {0: [(0, c0, (16, 16))]},
        defer_contiguous=False,
        is_packed_bf16=True,
    )
    np.testing.assert_array_equal(
        host_u32[0]['layers.0.experts.0.w'],
        np.asarray(bf16_arr).reshape(16, 16),
    )
    buf_1d = np.arange(8, dtype=np.float32)
    s0 = safetensors_saver._DeferredHostSlice(buf_1d, 0, 4, (2, 2))
    s1 = safetensors_saver._DeferredHostSlice(buf_1d, 4, 8, (2, 2))
    views = safetensors_saver._tensor_byte_views({'a': s0, 'b': s1})
    self.assertLen(views, 1)
    self.assertEqual(bytes(views[0]), buf_1d.tobytes())

  def test_shm_ring_and_multiproc_stage_pool(self):
    flags_argv = ('--undefok=f', '--f=1')
    geom = safetensors_saver.compute_shm_ring_geometry(
        shm_budget_bytes=16 << 20,
        max_bucket_bytes=1 << 20,
        num_large_slots=1,
    )
    self.assertLen(geom, 4)
    self.assertGreaterEqual(geom[1], 1)
    auto_geom = safetensors_saver.compute_shm_ring_geometry()
    self.assertLen(auto_geom, 4)
    self.assertEqual(safetensors_saver.slot_offset(2, 4096, 4, 8192), 8192)
    self.assertEqual(safetensors_saver.slot_offset(4, 4096, 4, 8192), 16384)
    cmd = safetensors_io.build_worker_cmd(
        safetensors_stage_worker.__file__,
        'from tunix.models import safetensors_stage_worker as w;'
        ' w.run_stage_worker_stdio()',
        flags_argv,
    )
    self.assertNotEmpty(cmd)
    fast_cmd = safetensors_io.build_worker_cmd(
        safetensors_stage_worker.__file__,
        'from tunix.models import safetensors_stage_worker as w;'
        ' w.run_stage_worker_stdio()',
        (),
    )
    self.assertNotEmpty(fast_cmd)
    self.assertEqual(safetensors_saver._close_parallel_handles(()), [])

    with tempfile.TemporaryDirectory() as tmpdir:
      arena_path = os.path.join(tmpdir, 'arena.bin')
      mm = safetensors_io.create_shm_mmap(arena_path, 32768)
      mm[:6] = b'hello!'
      remote_target = os.path.join(tmpdir, 'remote.bin')

      fake_stdin = io.StringIO(
          f'SLOT\t0\t6\t0\t{remote_target}2\nFLUSH\nQUIT\n'
      )
      fake_stdout = io.StringIO()
      orig_stdin, orig_stdout = sys.stdin, sys.stdout
      os.environ['TUNIX_STAGE_ARENA_PATH'] = arena_path
      os.environ['TUNIX_STAGE_WORKER_IDX'] = '1'
      os.environ['TUNIX_STAGE_NUM_WORKERS'] = '2'
      try:
        sys.stdin, sys.stdout = fake_stdin, fake_stdout
        safetensors_saver.run_stage_worker_stdio()
      finally:
        mm.close()
        sys.stdin, sys.stdout = orig_stdin, orig_stdout
        for k in (
            'TUNIX_STAGE_ARENA_PATH',
            'TUNIX_STAGE_WORKER_IDX',
            'TUNIX_STAGE_NUM_WORKERS',
        ):
          os.environ.pop(k, None)
      self.assertIn('FREE\t0', fake_stdout.getvalue())
      self.assertIn('DONE', fake_stdout.getvalue())
      with open(f'{remote_target}2', 'rb') as rf:
        self.assertEqual(rf.read(), b'hello!')

      recorded_puts: list[tuple[str, str, bytes]] = []

      class _FakeConn:
        timeout = 15.0
        sock = None

        def request(self, method, path, body=None, headers=None):
          del headers
          recorded_puts.append((method, path, bytes(body)))

        def getresponse(self):
          return types.SimpleNamespace(status=200, read=lambda: b'')

        def close(self):
          pass

      orig_get_conn = safetensors_io.get_thread_https_conn
      safetensors_io.get_thread_https_conn = (
          lambda vip, timeout=15.0: _FakeConn()
      )
      try:
        safetensors_saver.upload_gcs_slot_http(
            b'hello!', 'gs://my-bkt/obj.bin', vips=('127.0.0.1',), token='tok'
        )
        self.assertLen(recorded_puts, 1)
        self.assertEqual(
            recorded_puts[0], ('PUT', '/my-bkt/obj.bin', b'hello!')
        )

        stage_dir = os.path.join(tmpdir, 'stage_ring')
        pool_out = os.path.join(tmpdir, 'rs=6.3', 'pool_out')
        os.makedirs(pool_out, exist_ok=True)
        orig_popen = safetensors_saver.subprocess.Popen
        orig_discover = safetensors_io.GsHttpPath.discover_vips
        orig_token = safetensors_io.GsHttpPath.resolve_token
        safetensors_io.GsHttpPath.discover_vips = staticmethod(
            lambda: ('127.0.0.1',)
        )
        safetensors_io.GsHttpPath.resolve_token = staticmethod(lambda: 'tok')
        closed_handles = []

        class _FakeStripes:

          def stripe_bytes(self, path: str) -> int:
            return 16384 if ':rs=6.3/' in path else 0

          def open(self, path: str, num_stripes: int):
            return path, [f'{path}.s{i}' for i in range(num_stripes)]

          def close(self, handle: str) -> None:
            closed_handles.append(handle)

        self.enter_context(
            mock.patch.dict(
                safetensors_io.STORAGE_BACKENDS,
                {f'{pool_out}/': _remote_spec(parallel_stripes=_FakeStripes())},
            )
        )

        def _pipe_worker_popen(*args, **kwargs):
          del args
          env = {**kwargs['env'], 'TUNIX_STAGE_BACKEND': 'gcs_http'}
          in_r, in_w = os.pipe()
          out_r, out_w = os.pipe()
          w_stdin = os.fdopen(in_r, 'r', encoding='utf-8')
          w_stdout = os.fdopen(out_w, 'w', encoding='utf-8', buffering=1)
          p_stdin = os.fdopen(in_w, 'w', encoding='utf-8', buffering=1)
          p_stdout = os.fdopen(out_r, 'r', encoding='utf-8', buffering=1)

          def _worker_thread() -> None:
            orig_in, orig_out = sys.stdin, sys.stdout
            os.environ.update(env)
            sys.stdin, sys.stdout = w_stdin, w_stdout
            try:
              safetensors_saver.run_stage_worker_stdio()
            finally:
              sys.stdin, sys.stdout = orig_in, orig_out
              for k in (
                  'TUNIX_STAGE_ARENA_PATH',
                  'TUNIX_STAGE_NUM_WORKERS',
                  'TUNIX_STAGE_PROCESS_IDX',
                  'TUNIX_STAGE_WORKER_IDX',
                  'TUNIX_STAGE_BACKEND',
                  'TUNIX_GCS_VIPS',
              ):
                os.environ.pop(k, None)
              w_stdin.close()
              w_stdout.close()

          t = threading.Thread(target=_worker_thread, daemon=True)
          t.start()
          return types.SimpleNamespace(
              stdin=p_stdin,
              stdout=p_stdout,
              poll=lambda: None if t.is_alive() else 0,
              wait=lambda timeout=None: t.join(timeout),
          )

        safetensors_saver.subprocess.Popen = _pipe_worker_popen
        try:
          with safetensors_saver.MultiprocStagePool(
              stage_dir=stage_dir,
              num_procs=1,
              shm_budget_bytes=8 << 20,
              max_bucket_bytes=256 << 10,
              num_large_slots=1,
          ) as pool:
            pool.prewarm(pool_out, proc_idx=0)
            small_path = os.path.join(pool_out, 'small.safetensors')
            with pool.open_writer(small_path) as w:
              w.write(b'abcd')
              big_np = np.ones(70000, dtype=np.uint8)
              w.write(memoryview(big_np))
            with pool.open_writer('gs://my-bkt/staged.safetensors') as gw:
              gw.write(b'gcs-payload')
            large_path = os.path.join(pool_out, ':rs=6.3', 'large.safetensors')
            os.makedirs(os.path.dirname(large_path), exist_ok=True)
            with pool.open_writer(large_path) as lw:
              lw.write(b'x' * (pool.slot_bytes + 4096))
              with self.assertRaises(ValueError):
                lw.write(b'y' * (pool.large_slot_bytes + 4096))
            pool.flush()
        finally:
          safetensors_saver.subprocess.Popen = orig_popen
          safetensors_io.GsHttpPath.discover_vips = orig_discover
          safetensors_io.GsHttpPath.resolve_token = orig_token
        with open(small_path, 'rb') as sf:
          self.assertLen(sf.read(), 70004)
        self.assertEqual(closed_handles, [large_path])
        n_stripes = -(-(pool.slot_bytes + 4096) // 16384)
        with open(f'{large_path}.s{n_stripes - 1}', 'rb') as lf:
          self.assertNotEmpty(lf.read())
        self.assertGreaterEqual(len(recorded_puts), 2)
      finally:
        safetensors_io.get_thread_https_conn = orig_get_conn

  def test_shard_groups_and_gcs_parallel_composite(self):
    flat_state = {
        f'l{i}.w': jnp.arange(64, dtype=jnp.float32).reshape(8, 8) + i
        for i in range(6)
    }
    specs = [
        safetensors_saver.TensorExportSpec(
            source_key=k, hf_key=f'model.{k}', nbytes=256
        )
        for k in flat_state
    ]
    with self.assertRaises(ValueError):
      safetensors_saver.build_save_plan(
          specs, num_ranks=1, max_bucket_bytes=256, max_shard_bytes=0
      )
    with self.assertRaises(ValueError):
      safetensors_saver.build_save_plan(
          specs,
          num_ranks=1,
          max_bucket_bytes=256,
          max_shard_bytes=safetensors_saver.MAX_SAFETENSORS_SHARD_BYTES + 1,
      )
    with self.assertRaisesRegex(ValueError, 'exceeds max_shard_bytes'):
      safetensors_saver.build_save_plan(
          specs, num_ranks=1, max_bucket_bytes=512, max_shard_bytes=256
      )
    # Two 256-byte waves plus ~160-byte safetensors header exceed 512 bytes,
    # so `max_shard_bytes=512` splits them into 1-wave shards while `768`
    # groups 2 waves per shard with every file staying <= max_shard_bytes.
    tight_plan = safetensors_saver.build_save_plan(
        specs,
        num_ranks=1,
        max_bucket_bytes=256,
        max_shard_bytes=512,
        per_wave_shards=True,
    )
    self.assertLen(tight_plan.shard_groups, 6)
    plan = safetensors_saver.build_save_plan(
        specs,
        num_ranks=1,
        max_bucket_bytes=256,
        max_shard_bytes=768,
        per_wave_shards=True,
    )
    self.assertLen(plan.waves, 6)
    self.assertLen(plan.shard_groups, 3)
    self.assertEqual(plan.shard_groups[0], (0, (0, 1)))
    self.assertEqual(plan.shard_groups[1], (0, (2, 3)))
    self.assertEqual(plan.shard_groups[2], (0, (4, 5)))
    with tempfile.TemporaryDirectory() as tmpdir:
      metrics = safetensors_saver.export_safetensors_with_plan(
          flat_state, plan, tmpdir, async_write=True
      )
      self.assertLen(metrics.shard_paths, 3)
      for shard_path in metrics.shard_paths:
        self.assertLessEqual(os.path.getsize(shard_path), 768)
      with open(
          os.path.join(tmpdir, 'model.safetensors.index.json'), 'r'
      ) as idx_f:
        wmap = json.load(idx_f)['weight_map']
      self.assertLen(wmap, 6)
      loaded = {}
      for shard_path in metrics.shard_paths:
        loaded.update(safe_flax.load_file(shard_path))
      for k, arr in flat_state.items():
        np.testing.assert_allclose(loaded[f'model.{k}'], np.asarray(arr))

    # Verify GCS HTTP retry, compose, delete, and MultiprocStagePool.activate().
    http_ops: list[tuple[str, str, bytes]] = []
    fail_once = [True]

    class _FakeGcsConn:
      timeout = 15.0
      sock = None

      def request(self, method, path, body=None, headers=None):
        del headers
        self._last = (method, path, bytes(body) if body else b'')
        http_ops.append(self._last)

      def getresponse(self):
        if self._last[0] == 'POST' and fail_once[0]:
          fail_once[0] = False
          return types.SimpleNamespace(status=503, read=lambda: b'retry')
        if self._last[0] == 'DELETE':
          return types.SimpleNamespace(status=204, read=lambda: b'')
        return types.SimpleNamespace(status=200, read=lambda: b'{}')

      def close(self):
        pass

    orig_get_conn = safetensors_io.get_thread_https_conn
    orig_discover = safetensors_io.GsHttpPath.discover_vips
    orig_token = safetensors_io.GsHttpPath.resolve_token
    orig_popen = safetensors_saver.subprocess.Popen
    safetensors_io.get_thread_https_conn = (
        lambda vip, timeout=15.0: _FakeGcsConn()
    )
    safetensors_io.GsHttpPath.discover_vips = staticmethod(
        lambda: ('127.0.0.1', '127.0.0.2')
    )
    safetensors_io.GsHttpPath.resolve_token = staticmethod(lambda: 'tok')
    safetensors_io.GCS_VIP_CACHE.clear()
    try:
      safetensors_saver.compose_gcs_objects_http(
          'gs://bkt/dir/shard.safetensors',
          ('gs://bkt/dir/shard.safetensors.part.0',),
      )
      with self.assertRaises(ValueError):
        safetensors_saver.compose_gcs_objects_http('gs://bkt/a', ())
      with self.assertRaises(ValueError):
        safetensors_saver.compose_gcs_objects_http(
            'gs://bkt/a', ('gs://other/p0',)
        )
      safetensors_saver.delete_gcs_object_http('gs://bkt/dir/part.0')
      safetensors_saver._delete_gcs_parts_direct(())
      safetensors_saver._delete_gcs_parts_direct(
          ('gs://bkt/dir/orphan.part.0',)
      )

      spawned_envs: list[dict[str, str]] = []

      def _thread_popen(*args, **kwargs):
        del args
        spawned_envs.append(dict(kwargs['env']))
        in_r, in_w = os.pipe()
        out_r, out_w = os.pipe()
        w_in = os.fdopen(in_r, 'r', encoding='utf-8')
        w_out = os.fdopen(out_w, 'w', encoding='utf-8', buffering=1)
        p_in = os.fdopen(in_w, 'w', encoding='utf-8', buffering=1)
        p_out = os.fdopen(out_r, 'r', encoding='utf-8', buffering=1)

        def _run():
          orig_in, orig_out = sys.stdin, sys.stdout
          os.environ.update(kwargs['env'])
          sys.stdin, sys.stdout = w_in, w_out
          try:
            safetensors_saver.run_stage_worker_stdio()
          finally:
            sys.stdin, sys.stdout = orig_in, orig_out
            for k in (
                'TUNIX_STAGE_ARENA_PATH',
                'TUNIX_STAGE_NUM_WORKERS',
                'TUNIX_STAGE_PROCESS_IDX',
                'TUNIX_STAGE_WORKER_IDX',
                'TUNIX_STAGE_BACKEND',
                'TUNIX_GCS_VIPS',
            ):
              os.environ.pop(k, None)
            w_in.close()
            w_out.close()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return types.SimpleNamespace(
            stdin=p_in,
            stdout=p_out,
            poll=lambda: None if t.is_alive() else 0,
            wait=lambda timeout=None: t.join(timeout),
        )

      safetensors_saver.subprocess.Popen = _thread_popen
      with tempfile.TemporaryDirectory() as tmpdir:
        pool = safetensors_saver.MultiprocStagePool(
            stage_dir=os.path.join(tmpdir, 'stage'),
            num_procs=1,
            shm_budget_bytes=4 << 20,
            max_bucket_bytes=128 << 10,
            num_large_slots=1,
        )
        try:
          with pool.activate():
            safetensors_saver.export_safetensors_with_plan(
                flat_state, plan, 'gs://bkt/ws_run', async_write=True
            )
            single_plan = safetensors_saver.build_save_plan(
                specs[:1],
                num_ranks=1,
                max_bucket_bytes=256,
                max_shard_bytes=512,
                per_wave_shards=True,
            )
            safetensors_saver.export_safetensors_with_plan(
                flat_state, single_plan, 'gs://bkt/ws_single'
            )
        finally:
          pool.close()
          pool.close()  # Idempotent close check.
      methods = [m for m, _, _ in http_ops]
      paths = [p for _, p, _ in http_ops]
      self.assertIn('PUT', methods)
      self.assertIn('POST', methods)
      self.assertIn('DELETE', methods)
      self.assertTrue(any('part.0' in p for p in paths))
      self.assertTrue(any('compose' in p for p in paths))
      self.assertTrue(any('ws_single' in p and 'part.' not in p for p in paths))
      self.assertEqual(spawned_envs[0]['TUNIX_STAGE_PROCESS_IDX'], '0')
      self.assertEqual(spawned_envs[0]['TUNIX_STAGE_BACKEND'], 'gcs_http')
      self.assertEqual(spawned_envs[0]['TUNIX_GCS_VIPS'], '127.0.0.1,127.0.0.2')

      # Verify process_index propagation and uninitialized-JAX fallback to 0.
      with (
          mock.patch.object(
              jax, 'process_index', side_effect=RuntimeError('uninit')
          ),
          tempfile.TemporaryDirectory() as tmpdir_fb,
      ):
        fb_pool = safetensors_saver.MultiprocStagePool(
            stage_dir=os.path.join(tmpdir_fb, 'stage_fb'),
            num_procs=1,
            shm_budget_bytes=4 << 20,
            max_bucket_bytes=128 << 10,
            num_large_slots=1,
        )
        fb_pool.close()
      self.assertEqual(spawned_envs[-1]['TUNIX_STAGE_PROCESS_IDX'], '0')
      with (
          mock.patch.object(jax, 'process_index', return_value=3),
          tempfile.TemporaryDirectory() as tmpdir_h3,
      ):
        h3_pool = safetensors_saver.MultiprocStagePool(
            stage_dir=os.path.join(tmpdir_h3, 'stage_h3'),
            num_procs=1,
            shm_budget_bytes=4 << 20,
            max_bucket_bytes=128 << 10,
            num_large_slots=1,
        )
        h3_pool.close()
      self.assertEqual(spawned_envs[-1]['TUNIX_STAGE_PROCESS_IDX'], '3')

      # Verify W1 (128 GiB ring cap + 3 GiB large slot), W2 (hash-prefixed
      # parts & >threshold multi-VIP GCS striping), and W4 (128 MiB default).
      self.assertEqual(safetensors_saver.DEFAULT_MAX_BUCKET_BYTES, 128 << 20)
      hashed = safetensors_saver._hashed_part_uri(
          'gs://bkt/dir/model-00001.safetensors', 'part.0'
      )
      self.assertTrue(hashed.startswith('gs://bkt/dir/'))
      self.assertTrue(hashed.endswith('.model-00001.safetensors.part.0'))
      self.assertLen(os.path.basename(hashed).split('.', 1)[0], 4)

      with mock.patch.object(
          safetensors_io,
          'available_shm_bytes',
          return_value=734 << 30,
      ):
        slot_b, n_reg, lg_b, n_lg = safetensors_saver.compute_shm_ring_geometry(
            shm_budget_bytes=0
        )
      self.assertGreaterEqual(n_reg * slot_b + n_lg * lg_b, 120 << 30)
      self.assertGreaterEqual(lg_b, 2560 << 20)

      http_ops.clear()
      with (
          mock.patch.object(
              safetensors_saver, '_GCS_STRIPE_THRESHOLD_BYTES', 128
          ),
          mock.patch.object(safetensors_saver, '_GCS_MIN_STRIPE_BYTES', 64),
          tempfile.TemporaryDirectory() as tmpdir,
      ):
        pool = safetensors_saver.MultiprocStagePool(
            stage_dir=os.path.join(tmpdir, 'stage_stripe'),
            num_procs=1,
            shm_budget_bytes=4 << 20,
            max_bucket_bytes=128 << 10,
            num_large_slots=1,
        )
        try:
          with pool.open_writer('gs://bkt/ws_stripe/vocab.safetensors') as wf:
            wf.write(b'v' * 256)
          pool.flush()
          # Exercise error cleanup branch of _finalize_composite_stage.
          pool._finalize_composite_stage(
              [('gs://bkt/ws_stripe/bad', ('gs://bkt/ws_stripe/p0',))],
              ['prior error'],
          )
        finally:
          pool.close()
      stripe_paths = [p for _, p, _ in http_ops]
      self.assertTrue(any('spart.0' in p for p in stripe_paths))
      self.assertTrue(any('spart.3' in p for p in stripe_paths))
      self.assertTrue(
          any('vocab.safetensors/compose' in p for p in stripe_paths)
      )
    finally:
      safetensors_io.get_thread_https_conn = orig_get_conn
      safetensors_io.GsHttpPath.discover_vips = orig_discover
      safetensors_io.GsHttpPath.resolve_token = orig_token
      safetensors_saver.subprocess.Popen = orig_popen
      safetensors_io.GCS_VIP_CACHE.clear()

  def test_build_export_specs_from_manifest(self):
    flat_state = {
        'embed': jnp.zeros((16, 8), dtype=jnp.bfloat16),
        'layers.0.proj': jnp.zeros((4, 2, 8), dtype=jnp.bfloat16),
        'layers.0.moe.gate': jnp.zeros((4, 8, 6), dtype=jnp.bfloat16),
        'layers.0.moe.raw': jnp.zeros((4, 6, 8), dtype=jnp.bfloat16),
        'unmapped': jnp.zeros((4,), dtype=jnp.bfloat16),
    }
    manifest = {
        'schema_version': hf_mapping.SCHEMA_VERSION,
        'model_type': 'test_model',
        'rules': [
            {
                'source_keys': ['embed'],
                'target_keys': ['model.embed_tokens.weight'],
                'ops': [],
            },
            {
                'source_keys': ['layers.0.proj'],
                'target_keys': ['model.layers.0.proj.weight'],
                'ops': [
                    {'reshape': {'src_shape': [4, 2, 8], 'dst_shape': [8, 8]}},
                    {'transpose': {'axes': [1, 0]}},
                ],
            },
            {
                'source_keys': ['layers.0.moe.gate'],
                'target_keys': [
                    f'model.layers.0.mlp.experts.{i}.gate_proj.weight'
                    for i in range(4)
                ],
                'ops': [
                    {'unstack': {'axis': 0}},
                    {'transpose': {'axes': [1, 0]}},
                ],
            },
            {
                'source_keys': ['layers.0.moe.raw'],
                'target_keys': [
                    f'model.layers.0.mlp.experts.{i}.raw.weight'
                    for i in range(4)
                ],
                'ops': [
                    {'unstack': {'axis': 0}},
                ],
            },
            {
                'source_keys': ['missing.key'],
                'target_keys': ['ignored'],
                'ops': [],
            },
        ],
    }
    specs, unstack_counts = safetensors_saver.build_export_specs_from_manifest(
        flat_state, manifest, output_dtype=jnp.float32
    )
    by_src = {s.source_key: s for s in specs}
    self.assertLen(by_src, 4)
    self.assertEqual(
        unstack_counts, {'layers.0.moe.gate': 4, 'layers.0.moe.raw': 4}
    )
    self.assertEqual(by_src['embed'].hf_key, 'model.embed_tokens.weight')
    self.assertEqual(by_src['embed'].nbytes, 16 * 8 * 4)
    self.assertEqual(by_src['layers.0.proj'].shape_2d, (8, 8))
    self.assertEqual(by_src['layers.0.proj'].transpose_axes, (1, 0))
    self.assertEqual(
        by_src['layers.0.moe.gate'].hf_key,
        'model.layers.0.mlp.experts.{expert}.gate_proj.weight',
    )
    self.assertEqual(by_src['layers.0.moe.gate'].transpose_axes, (0, 2, 1))
    self.assertEqual(by_src['layers.0.moe.gate'].unstack_axis, 0)
    self.assertIsNone(by_src['layers.0.moe.raw'].transpose_axes)
    self.assertEqual(by_src['layers.0.moe.raw'].unstack_axis, 0)

    bad_manifest = {
        'schema_version': hf_mapping.SCHEMA_VERSION,
        'model_type': 'test_model',
        'rules': [{
            'source_keys': ['layers.0.moe.raw'],
            'target_keys': ['a.weight', 'b.weight'],
            'ops': [{'unstack': {'axis': 0}}],
        }],
    }
    with self.assertRaises(ValueError):
      safetensors_saver.build_export_specs_from_manifest(
          flat_state, bad_manifest
      )

  def test_aot_compile_save_plan_and_export(self):
    flat_state = {
        'w': jnp.arange(16, dtype=jnp.float32).reshape(4, 4),
        'moe': jnp.arange(64, dtype=jnp.float32).reshape(4, 4, 4),
        'noop': jnp.arange(8, dtype=jnp.float32),
    }
    specs = [
        safetensors_saver.TensorExportSpec(
            source_key='w',
            hf_key='model.w',
            transpose_axes=(1, 0),
            nbytes=64,
        ),
        safetensors_saver.TensorExportSpec(
            source_key='moe',
            hf_key='model.experts.{expert}.w',
            transpose_axes=(0, 2, 1),
            unstack_axis=0,
            unstack_range=(0, 2),
            nbytes=128,
        ),
        safetensors_saver.TensorExportSpec(
            source_key='moe',
            hf_key='model.experts.{expert}.w',
            transpose_axes=(0, 2, 1),
            unstack_axis=0,
            unstack_range=(2, 4),
            nbytes=128,
        ),
        safetensors_saver.TensorExportSpec(
            source_key='noop', hf_key='model.noop', nbytes=32
        ),
    ]
    devs = jax.devices()[:1]
    plan = safetensors_saver.build_save_plan(
        specs,
        num_ranks=len(devs),
        max_bucket_bytes=128,
        max_shard_bytes=256,
        per_wave_shards=True,
    )
    compiled = safetensors_saver.compile_save_plan(
        flat_state, plan, output_dtype=jnp.float32, devices=devs
    )
    self.assertGreaterEqual(compiled.num_unique_kernels, 2)
    self.assertGreaterEqual(compiled.peak_temp_hbm_bytes, 0)
    self.assertGreaterEqual(compiled.compile_time_sec, 0.0)
    self.assertEqual(
        safetensors_saver._compiled_temp_hbm_bytes(
            types.SimpleNamespace(memory_analysis=lambda: None)
        ),
        0,
    )
    self.assertEqual(
        safetensors_saver._compiled_temp_hbm_bytes(
            types.SimpleNamespace(
                memory_analysis=lambda: (_ for _ in ()).throw(
                    NotImplementedError()
                )
            )
        ),
        0,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      metrics = safetensors_saver.export_safetensors_with_plan(
          flat_state,
          plan,
          tmpdir,
          output_dtype=jnp.float32,
          devices=devs,
          compiled_plan=compiled,
      )
      loaded = {}
      for sp in metrics.shard_paths:
        loaded.update(safe_flax.load_file(sp))
      np.testing.assert_allclose(
          loaded['model.w'], np.asarray(flat_state['w']).T
      )
      np.testing.assert_allclose(
          loaded['model.noop'], np.asarray(flat_state['noop'])
      )
      for e in range(4):
        np.testing.assert_allclose(
            loaded[f'model.experts.{e}.w'], np.asarray(flat_state['moe'][e]).T
        )
    # Exercise AOT local-concat (`_concat_local_key`) and fused-wave
    # (`_fused_wave_key`) compile + cache-hit dispatch parity on 1D rank mesh.
    mesh1 = jax.sharding.Mesh(np.array(devs), ('rank',))
    shd1 = jax.sharding.NamedSharding(mesh1, jax.sharding.PartitionSpec('rank'))
    c0 = jax.device_put(
        jnp.arange(256, dtype=jnp.float32).reshape(1, 16, 16), shd1
    )
    c1 = jax.device_put(
        jnp.arange(256, dtype=jnp.float32).reshape(1, 16, 16) + 10.0, shd1
    )
    cat_pairs = [
        (
            0,
            safetensors_saver.TensorExportSpec(
                'c0', 'model.c0.{expert}', unstack_range=(0, 1), nbytes=512
            ),
        ),
        (
            0,
            safetensors_saver.TensorExportSpec(
                'c1', 'model.c1.{expert}', unstack_range=(0, 1), nbytes=512
            ),
        ),
    ]
    d2h_eager, _, _, _ = safetensors_saver._start_batch_d2h_with_layout(
        cat_pairs, [c0, c1], devs
    )
    cat_key = safetensors_saver._concat_local_key((c0, c1), mesh1, 'rank')
    cat_exec = safetensors_saver._jit_concat_local_rank_shards.lower(
        (c0, c1), mesh1, 'rank'
    ).compile()
    cat_plan = safetensors_saver.CompiledSavePlan(
        executables={cat_key: cat_exec},
        num_unique_kernels=1,
        compile_time_sec=0.0,
        peak_temp_hbm_bytes=0,
        rank_mesh=mesh1,
        rank_shd=shd1,
    )
    d2h_aot, _, _, _ = safetensors_saver._start_batch_d2h_with_layout(
        cat_pairs, [c0, c1], devs, compiled_plan=cat_plan
    )
    np.testing.assert_allclose(np.asarray(d2h_aot[0]), np.asarray(d2h_eager[0]))

    fused_specs = [
        safetensors_saver.TensorExportSpec(
            source_key='c0',
            hf_key='model.e0.{expert}.w',
            transpose_axes=(0, 2, 1),
            unstack_axis=0,
            unstack_range=(0, 1),
            nbytes=64,
        ),
        safetensors_saver.TensorExportSpec(
            source_key='c1',
            hf_key='model.e1.{expert}.w',
            transpose_axes=(0, 2, 1),
            unstack_axis=0,
            unstack_range=(0, 1),
            nbytes=64,
        ),
    ]
    fused_wave = safetensors_saver.WaveBucket(
        wave_index=0,
        rank_assignments={0: tuple(fused_specs)},
        rank_bytes={0: 128},
    )
    fused_plan = safetensors_saver.SavePlan(
        num_ranks=1,
        total_bytes=128,
        max_bucket_bytes=256,
        waves=(fused_wave,),
        rank_total_bytes={0: 128},
        weight_map={},
    )
    orig_can_fuse = safetensors_saver._can_fuse_identity_wave
    safetensors_saver._can_fuse_identity_wave = lambda *_: True
    try:
      fused_compiled = safetensors_saver.compile_save_plan(
          {'c0': c0, 'c1': c1},
          fused_plan,
          output_dtype=jnp.float32,
          devices=devs,
      )
      self.assertEqual(fused_compiled.num_unique_kernels, 1)
      f_pairs = [(0, fused_specs[0]), (0, fused_specs[1])]
      aot_out = safetensors_saver._transform_wave_pairs(
          {'c0': c0, 'c1': c1},
          fused_wave,
          f_pairs,
          jnp.float32,
          devs,
          shd1,
          compiled_plan=fused_compiled,
      )
      eager_out = safetensors_saver._transform_wave_pairs(
          {'c0': c0, 'c1': c1},
          fused_wave,
          f_pairs,
          jnp.float32,
          devs,
          shd1,
      )
      np.testing.assert_allclose(
          np.asarray(aot_out[0]), np.asarray(eager_out[0])
      )
    finally:
      safetensors_saver._can_fuse_identity_wave = orig_can_fuse

    # Also exercise multi-device AOT routing when >= 2 devices exist.
    all_devs = jax.devices()
    if len(all_devs) >= 2:
      mesh = jax.sharding.Mesh(np.array(all_devs[:2]), ('rank',))
      shd = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('rank'))
      shd_arr = jax.device_put(
          jnp.arange(2 * 2048, dtype=jnp.float32).reshape(2, 2048), shd
      )
      shd_specs = [
          safetensors_saver.TensorExportSpec(
              source_key='s', hf_key='model.s', nbytes=2 * 2048 * 4
          )
      ]
      shd_plan = safetensors_saver.build_save_plan(
          shd_specs, num_ranks=2, max_bucket_bytes=1 << 20
      )
      shd_compiled = safetensors_saver.compile_save_plan(
          {'s': shd_arr}, shd_plan, devices=all_devs[:2]
      )
      self.assertGreaterEqual(shd_compiled.num_unique_kernels, 1)

    # Exercise AOT routing (`_compile_dtype_routing`) across mixed dtypes even
    # on single-host CPU by marking sharded arrays as multi-host.
    route_devs = all_devs[:2] if len(all_devs) >= 2 else all_devs[:1]
    n_r = len(route_devs)
    r_mesh = jax.sharding.Mesh(np.array(route_devs), ('rank',))
    r_shd = jax.sharding.NamedSharding(
        r_mesh, jax.sharding.PartitionSpec('rank')
    )
    a_f32 = jax.device_put(
        jnp.arange(n_r * 2048, dtype=jnp.float32).reshape(n_r, 2048), r_shd
    )
    a_bf16 = jax.device_put(
        jnp.arange(n_r * 2048, dtype=jnp.bfloat16).reshape(n_r, 2048), r_shd
    )
    mix_specs = [
        safetensors_saver.TensorExportSpec(
            source_key='sf', hf_key='model.sf', nbytes=n_r * 2048 * 4
        ),
        safetensors_saver.TensorExportSpec(
            source_key='sb', hf_key='model.sb', nbytes=n_r * 2048 * 2
        ),
    ]
    mix_plan = safetensors_saver.build_save_plan(
        mix_specs, num_ranks=n_r, max_bucket_bytes=1 << 20
    )
    mix_compiled = safetensors_saver.compile_save_plan(
        {'sf': a_f32, 'sb': a_bf16}, mix_plan, devices=route_devs
    )
    self.assertGreaterEqual(mix_compiled.num_unique_kernels, 2)
    with tempfile.TemporaryDirectory() as mix_dir:
      safetensors_saver.export_safetensors_with_plan(
          {'sf': a_f32, 'sb': a_bf16},
          mix_plan,
          mix_dir,
          devices=route_devs,
          compiled_plan=mix_compiled,
      )
      loaded_mix = {}
      for fn in os.listdir(mix_dir):
        if fn.endswith('.safetensors'):
          loaded_mix.update(safe_flax.load_file(os.path.join(mix_dir, fn)))
      np.testing.assert_array_equal(loaded_mix['model.sf'], np.asarray(a_f32))
      np.testing.assert_array_equal(loaded_mix['model.sb'], np.asarray(a_bf16))

  def test_validate_safetensors_config(self):
    ok_rep = safetensors_saver.validate_safetensors_config(
        '/tmp/local_out',
        max_bucket_bytes=256 << 20,
        max_shard_bytes=2560 << 20,
        total_bytes=10 << 30,
        num_ranks=8,
    )
    self.assertTrue(ok_rep.ok)
    ok_rep.raise_if_invalid(strict=True)

    bad_rep = safetensors_saver.validate_safetensors_config(
        '/rfs/bad%path',
        max_bucket_bytes=0,
        max_shard_bytes=safetensors_saver.MAX_SAFETENSORS_SHARD_BYTES + 1,
        num_ranks=0,
    )
    self.assertFalse(bad_rep.ok)
    self.assertGreaterEqual(len(bad_rep.errors), 4)
    with self.assertRaises(ValueError):
      bad_rep.raise_if_invalid()

    orig_statvfs = safetensors_saver.os.statvfs
    orig_token = safetensors_io.GsHttpPath.resolve_token
    orig_vips = safetensors_io.GsHttpPath.discover_vips
    safetensors_io.GsHttpPath.resolve_token = staticmethod(lambda: '')
    safetensors_io.GsHttpPath.discover_vips = staticmethod(
        lambda: ('127.0.0.1',)
    )
    safetensors_io.GCS_VIP_CACHE.clear()
    try:
      safetensors_saver.os.statvfs = lambda _: types.SimpleNamespace(
          f_bavail=100, f_frsize=1024 * 1024
      )
      small_shm_rep = safetensors_saver.validate_safetensors_config(
          'gs://bkt/prefix',
          max_bucket_bytes=256 << 20,
          max_shard_bytes=128 << 20,
          total_bytes=200 << 30,
          num_ranks=8,
          write_procs=1,
      )
      self.assertFalse(small_shm_rep.ok)
      self.assertNotEmpty(small_shm_rep.warnings)

      safetensors_saver.os.statvfs = lambda _: types.SimpleNamespace(
          f_bavail=1024, f_frsize=1024 * 1024
      )
      warn_rep = safetensors_saver.validate_safetensors_config(
          'gs://bkt/prefix',
          max_bucket_bytes=256 << 20,
          max_shard_bytes=2560 << 20,
          total_bytes=10 << 30,
          num_ranks=8,
          write_procs=1,
      )
      self.assertTrue(warn_rep.ok)
      self.assertNotEmpty(warn_rep.warnings)
      self.assertTrue(
          any(
              'TUNIX_GCS_TOKEN' in w and ', TOKEN,' not in w
              for w in warn_rep.warnings
          ),
          warn_rep.warnings,
      )
      with self.assertRaises(ValueError):
        warn_rep.raise_if_invalid(strict=True, context='test')
    finally:
      safetensors_saver.os.statvfs = orig_statvfs
      safetensors_io.GsHttpPath.resolve_token = orig_token
      safetensors_io.GsHttpPath.discover_vips = orig_vips
      safetensors_io.GCS_VIP_CACHE.clear()

  def test_hardening_fixes_manifest_validation_abort_and_dtype_partition(self):
    # 1. 11-expert template extraction avoids matching trailing '0' in '10'.
    keys_11 = [f'layers.0.experts.{i}.w' for i in range(11)]
    self.assertEqual(
        safetensors_saver._extract_expert_key_template(keys_11),
        'layers.0.experts.{expert}.w',
    )

    # 2. Manifest export filters dropped_keys and rejects unsupported ops/order.
    arr = jnp.ones((2, 2), dtype=jnp.float32)
    dropped_manifest = {
        'schema_version': hf_mapping.SCHEMA_VERSION,
        'rules': [
            {'source_keys': ['keep'], 'target_keys': ['hf.keep'], 'ops': []},
            {'source_keys': ['drop'], 'target_keys': ['hf.drop'], 'ops': []},
        ],
        'dropped_keys': ['drop'],
    }
    specs, _ = safetensors_saver.build_export_specs_from_manifest(
        {'keep': arr, 'drop': arr}, dropped_manifest
    )
    self.assertEqual([s.source_key for s in specs], ['keep'])

    for bad_rules in (
        [{
            'source_keys': ['a', 'b'],
            'target_keys': ['c'],
            'ops': [{'stack': {'axis': 0}}],
        }],
        [{
            'source_keys': ['a'],
            'target_keys': ['hf.a'],
            'ops': [{'scale': {'factor': 2.0}}],
        }],
        [{
            'source_keys': ['a'],
            'target_keys': ['hf.a'],
            'ops': [
                {'transpose': {'axes': [1, 0]}},
                {'reshape': {'src_shape': [2, 2], 'dst_shape': [4]}},
            ],
        }],
        [{
            'source_keys': ['a'],
            'target_keys': ['hf.a0', 'hf.a1'],
            'ops': [{'unstack': {'axis': 1}}],
        }],
    ):
      with self.assertRaises(ValueError):
        safetensors_saver.build_export_specs_from_manifest(
            {'a': arr, 'b': arr},
            {'schema_version': hf_mapping.SCHEMA_VERSION, 'rules': bad_rules},
        )

    # 3. build_grouped_ragged_meta rejects non-contiguous destination runs.
    with self.assertRaises(ValueError):
      safetensors_saver.build_grouped_ragged_meta(
          weights=[1024, 1024, 1024],
          owner_groups=[(0,), (1,), (0,)],
          num_ranks=2,
      )

    # 4. _partition_by_dtype splits mixed-dtype waves cleanly;
    # _group_batch_shards deduplicates fused identity waves by buffer identity.
    s_f32 = safetensors_saver.TensorExportSpec('f32', 'hf.f32', nbytes=8)
    s_bf16 = safetensors_saver.TensorExportSpec('bf16', 'hf.bf16', nbytes=4)
    groups = safetensors_saver._partition_by_dtype(
        [(0, s_f32), (0, s_bf16)],
        [jnp.ones((2,), dtype=jnp.float32), jnp.ones((2,), dtype=jnp.bfloat16)],
    )
    self.assertLen(groups, 2)

    # 5. StagedShardWriter aborts cleanly on exception instead of enqueueing.
    fake_pool = mock.MagicMock()
    fake_pool.acquire_slot.return_value = 0
    fake_pool.slot_offset.return_value = 0
    fake_pool.slot_bytes = 16
    fake_pool.large_slot_bytes = 16
    fake_pool.num_large_slots = 0
    fake_pool.arena_mmap = bytearray(16)
    with self.assertRaises(ValueError):
      with safetensors_saver.StagedShardWriter('gs://bkt/x', fake_pool) as wf:
        wf.write(b'x' * 32)
    fake_pool.release_slot.assert_called_once_with(0)
    fake_pool.enqueue_slot.assert_not_called()

    # 6. MultiprocStagePool cleans up on failed startup handshake and skips
    # inner striping for parts registered in a multi-wave GCS composite.
    bad_proc = mock.MagicMock()
    bad_proc.stdout = io.StringIO('CRASH\n')
    bad_proc.stdin = io.StringIO()
    with tempfile.TemporaryDirectory() as tmpdir:
      stage_dir = os.path.join(tmpdir, 'bad_stage')
      with mock.patch.object(
          safetensors_saver.subprocess, 'Popen', return_value=bad_proc
      ):
        with self.assertRaises(RuntimeError):
          safetensors_saver.MultiprocStagePool(
              stage_dir=stage_dir,
              num_procs=1,
              shm_budget_bytes=4 << 20,
              max_bucket_bytes=128 << 10,
              num_large_slots=1,
          )
      self.assertFalse(os.path.exists(stage_dir))

    # 7. Pre-unstack 3D transpose preserves axes without double-shifting.
    pre_t_manifest = {
        'schema_version': hf_mapping.SCHEMA_VERSION,
        'rules': [{
            'source_keys': ['moe'],
            'target_keys': ['e.0.w', 'e.1.w'],
            'ops': [
                {'transpose': {'axes': [0, 2, 1]}},
                {'unstack': {'axis': 0}},
            ],
        }],
    }
    pre_specs, _ = safetensors_saver.build_export_specs_from_manifest(
        {'moe': jnp.zeros((2, 4, 8), dtype=jnp.float32)}, pre_t_manifest
    )
    self.assertEqual(pre_specs[0].transpose_axes, (0, 2, 1))

    # 8. Mixed full-rank + partial-rank MoE wave rejects identity-group fusion.
    moe_a = jnp.zeros((2, 32, 64), dtype=jnp.float32)
    sa0 = safetensors_saver.TensorExportSpec(
        'moe_a', 'a.{expert}', unstack_axis=0, unstack_range=(0, 1), nbytes=8192
    )
    sa1 = safetensors_saver.TensorExportSpec(
        'moe_a', 'a.{expert}', unstack_axis=0, unstack_range=(1, 2), nbytes=8192
    )
    sb0 = safetensors_saver.TensorExportSpec(
        'moe_b', 'b.{expert}', unstack_axis=0, unstack_range=(0, 1), nbytes=8192
    )
    mixed_wave = safetensors_saver.WaveBucket(
        wave_index=0,
        rank_assignments={0: (sa0, sb0), 1: (sa1,)},
        rank_bytes={0: 16384, 1: 8192},
    )
    with mock.patch.object(
        safetensors_saver, 'infer_is_replicated', return_value=False
    ):
      self.assertFalse(
          safetensors_saver._is_fusable_source_group(
              moe_a, [(0, sa0), (1, sa1)], mixed_wave, 2
          )
      )

    # 9. _SequentialShardGroup & _write_rank_tensors abort on write failure.
    class _AbortWriter:

      def __init__(self) -> None:
        self.aborted = 0
        self.closed = 0

      def write(self, data: object) -> int:
        del data
        raise OSError('disk full')

      def abort(self) -> None:
        self.aborted += 1

      def close(self) -> None:
        self.closed += 1

      def __enter__(self) -> '_AbortWriter':
        return self

      def __exit__(self, exc_type: object, *_: object) -> bool:
        if exc_type is not None:
          self.abort()
        else:
          self.close()
        return False

    abort_fh = _AbortWriter()
    fake_backend = dataclasses.replace(
        safetensors_saver._LOCAL_BACKEND,
        default_write_buffer_bytes=0,
        stage_writer_fn=lambda _: abort_fh,
    )
    seq_grp = safetensors_saver._SequentialShardGroup(
        fake_backend, '/tmp/x.st', b'hdr', 2
    )
    with self.assertRaises(OSError):
      seq_grp.write_wave({'a': np.ones(4, dtype=np.float32)}, 0)
    self.assertEqual(abort_fh.aborted, 1)
    self.assertEqual(abort_fh.closed, 0)

    abort_fh = _AbortWriter()
    fake_backend = dataclasses.replace(
        safetensors_saver._LOCAL_BACKEND,
        default_write_buffer_bytes=0,
        stage_writer_fn=lambda _: abort_fh,
    )
    with self.assertRaises(OSError):
      safetensors_saver._write_rank_tensors(
          (fake_backend, '/tmp/y.st', b'hdr'),
          {'a': np.ones(4, dtype=np.float32)},
      )
    self.assertEqual(abort_fh.aborted, 1)
    self.assertEqual(abort_fh.closed, 0)

    # 10. MultiprocStagePool: is_healthy, startup TimeoutError, activate abort,
    # and compose-failure orphan cleanup.
    r_fd, w_fd = os.pipe()
    r_fh = os.fdopen(r_fd, 'r', encoding='utf-8')
    try:
      hang_proc = mock.MagicMock()
      hang_proc.stdout = r_fh
      hang_proc.stdin = io.StringIO()
      fake_sel = mock.MagicMock()
      fake_sel.__enter__.return_value = fake_sel
      fake_sel.select.return_value = []
      with (
          tempfile.TemporaryDirectory() as tmpdir,
          mock.patch.object(
              safetensors_saver.subprocess, 'Popen', return_value=hang_proc
          ),
          mock.patch.object(
              safetensors_io.selectors,
              'DefaultSelector',
              return_value=fake_sel,
          ),
      ):
        with self.assertRaisesRegex(RuntimeError, 'startup handshake'):
          safetensors_saver.MultiprocStagePool(
              stage_dir=os.path.join(tmpdir, 'timeout_stage'),
              num_procs=1,
              shm_budget_bytes=4 << 20,
              max_bucket_bytes=128 << 10,
              num_large_slots=1,
          )
    finally:
      r_fh.close()
      os.close(w_fd)

    ok_proc = mock.MagicMock()
    ok_proc.stdout = io.StringIO('READY\n')
    ok_proc.stdin = io.StringIO()
    ok_proc.poll.return_value = None
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        mock.patch.object(
            safetensors_saver.subprocess, 'Popen', return_value=ok_proc
        ),
    ):
      hp = safetensors_saver.MultiprocStagePool(
          stage_dir=os.path.join(tmpdir, 'health_stage'),
          num_procs=1,
          shm_budget_bytes=4 << 20,
          max_bucket_bytes=128 << 10,
          num_large_slots=1,
      )
      try:
        hp._worker_err = None
        self.assertTrue(hp.is_healthy())
        with mock.patch.object(hp, 'flush') as mock_flush:
          with self.assertRaisesRegex(RuntimeError, 'step fail'):
            with hp.activate():
              raise RuntimeError('step fail')
          mock_flush.assert_called_once_with(abort=True)
        # Compose error falls through to direct orphan cleanup.
        with (
            mock.patch.object(
                hp, '_dispatch_worker_composites', return_value=['compose err']
            ),
            mock.patch.object(
                safetensors_saver, '_delete_gcs_parts_direct'
            ) as mock_del_direct,
        ):
          errs: list[str] = []
          hp._finalize_composite_stage(
              [('gs://bkt/a.st', ('gs://bkt/p0',))], errs
          )
          self.assertEqual(errs, ['compose err'])
          mock_del_direct.assert_called_once_with(('gs://bkt/p0',))
        ok_proc.poll.return_value = 1
        self.assertFalse(hp.is_healthy())
      finally:
        hp.close(abort=True)
      self.assertFalse(hp.is_healthy())

  def _count_hlo_collectives(self, hlo: str) -> dict[str, int]:
    return {
        'all-gather': len(
            re.findall(
                r'^\s*%[\w.-]+\s*=.*\ball-gather(?:-start)?\(', hlo, re.M
            )
        ),
        'all-to-all': len(
            re.findall(
                r'^\s*%[\w.-]+\s*=.*\ball-to-all(?:-start)?\(', hlo, re.M
            )
        ),
        'collective-permute': len(
            re.findall(
                r'^\s*%[\w.-]+\s*=.*\bcollective-permute(?:-start)?\(',
                hlo,
                re.M,
            )
        ),
    }

  def _verify_generic_reshard_case(
      self,
      arr_np: np.ndarray,
      mesh_shape: tuple[int, ...],
      axis_names: tuple[str, ...],
      pspec: tuple[object, ...],
      *,
      shape_2d: tuple[int, ...] | None = None,
      transpose_axes: tuple[int, ...] | None = None,
      output_dtype: jnp.dtype | None = None,
      expect_zero_collectives: bool = False,
  ) -> dict[str, int]:
    n_devs = int(np.prod(mesh_shape))
    devs = jax.devices()[:n_devs]
    in_mesh = jax.sharding.Mesh(np.array(devs).reshape(mesh_shape), axis_names)
    in_shd = jax.sharding.NamedSharding(
        in_mesh, jax.sharding.PartitionSpec(*pspec)
    )
    rank_shd = jax.sharding.NamedSharding(
        jax.sharding.Mesh(np.array(devs), ('rank',)),
        jax.sharding.PartitionSpec('rank'),
    )
    arr = jax.device_put(jnp.asarray(arr_np), in_shd)
    compiled = safetensors_saver._jit_transform_on_hbm.lower(
        arr,
        None,
        None,
        1.0,
        shape_2d,
        transpose_axes,
        output_dtype,
        rank_shd,
        n_devs,
        1,
        in_shd,
    ).compile()
    counts = self._count_hlo_collectives(compiled.as_text())
    self.assertEqual(counts['all-gather'], 0, compiled.as_text())
    if expect_zero_collectives:
      self.assertEqual(counts['all-to-all'], 0)
      self.assertEqual(counts['collective-permute'], 0)
    got = np.asarray(compiled(arr, None, None, 1.0))
    ref = arr_np.reshape(shape_2d or arr_np.shape)
    if transpose_axes is not None:
      ref = np.transpose(ref, transpose_axes)
    if output_dtype is not None:
      ref = np.asarray(jnp.asarray(ref, dtype=output_dtype))
    expected_shape = (
        (n_devs, -1, 128) if ref.size % (n_devs * 1024) == 0 else (n_devs, -1)
    )
    np.testing.assert_array_equal(got, ref.reshape(expected_shape))
    return counts

  def test_generic_reshard_w2_meshes_hlo_and_bit_exactness(self):
    rng = np.random.default_rng(7)
    # 1. FSDP (64, 1) and (8, 1) -> 0 collectives (strips size-1 'tp').
    self._verify_generic_reshard_case(
        rng.normal(size=(64, 16, 64)).astype(np.float32),
        (64, 1),
        ('fsdp', 'tp'),
        ('fsdp', None, None),
        transpose_axes=(0, 2, 1),
        expect_zero_collectives=True,
    )
    self._verify_generic_reshard_case(
        rng.normal(size=(64, 128)).astype(np.float32),
        (8, 1),
        ('fsdp', 'tp'),
        ('fsdp', 'tp'),
        expect_zero_collectives=True,
    )
    # 2. TP (32, 2) and (2, 4): gate_proj (post-tr) & down_proj (pre-tr A2A).
    a2a_before, _, _, _, _ = safetensors_saver._plan_axis0_reshard_ops(
        ('fsdp', 'tp', None), (0, 2, 1), ('fsdp', 'tp')
    )
    self.assertTrue(a2a_before)
    c_down = self._verify_generic_reshard_case(
        rng.normal(size=(64, 32, 32)).astype(np.float32),
        (32, 2),
        ('fsdp', 'tp'),
        ('fsdp', 'tp', None),
        transpose_axes=(0, 2, 1),
    )
    self.assertEqual(c_down['all-to-all'], 1)
    self.assertEqual(c_down['collective-permute'], 0)
    self._verify_generic_reshard_case(
        rng.normal(size=(16, 32, 16)).astype(np.float32),
        (2, 4),
        ('fsdp', 'tp'),
        ('fsdp', None, 'tp'),
        transpose_axes=(0, 2, 1),
    )
    # 3. Qwen3.5 P(None, 'fsdp', 'tp') on (32, 2) -> 2 A2As, 0 all-gathers.
    c_q35 = self._verify_generic_reshard_case(
        rng.normal(size=(64, 64, 16)).astype(np.float32),
        (32, 2),
        ('fsdp', 'tp'),
        (None, 'fsdp', 'tp'),
        transpose_axes=(0, 2, 1),
    )
    self.assertEqual(c_q35['all-to-all'], 2)
    self.assertEqual(c_q35['collective-permute'], 0)
    # 4. Non-leading EP P('ep', 'fsdp', 'tp') on (8, 4, 2) -> 0 all-gathers.
    self._verify_generic_reshard_case(
        rng.normal(size=(64, 32, 32)).astype(np.float32),
        (8, 4, 2),
        ('fsdp', 'ep', 'tp'),
        ('ep', 'fsdp', 'tp'),
        transpose_axes=(0, 2, 1),
    )
    # 5. Attn-DP partial-rep on (4, 8, 2) & (4, 2, 2): embedder, q_proj, o_proj.
    self._verify_generic_reshard_case(
        rng.normal(size=(256, 256)).astype(np.float32),
        (4, 8, 2),
        ('dp', 'fsdp', 'tp'),
        ('tp', 'fsdp'),
    )
    self._verify_generic_reshard_case(
        rng.normal(size=(64, 16, 16)).astype(np.float32),
        (4, 2, 2),
        ('dp', 'fsdp', 'tp'),
        ('fsdp', 'tp', None),
        shape_2d=(64, 256),
        transpose_axes=(1, 0),
    )
    self._verify_generic_reshard_case(
        rng.normal(size=(16, 16, 64)).astype(np.float32),
        (4, 2, 2),
        ('dp', 'fsdp', 'tp'),
        ('tp', None, 'fsdp'),
        shape_2d=(256, 64),
        transpose_axes=(1, 0),
    )
    # 6. Tuple inner dims P(None, ('fsdp', 'ep'), 'tp') on (4, 2, 2).
    self._verify_generic_reshard_case(
        rng.normal(size=(16, 32, 32)).astype(np.float32),
        (4, 2, 2),
        ('fsdp', 'ep', 'tp'),
        (None, ('fsdp', 'ep'), 'tp'),
        transpose_axes=(0, 2, 1),
    )
    # 7. Arbitrary custom axis names ('a', 'b', 'c') -> zero hardcoded names.
    self._verify_generic_reshard_case(
        rng.normal(size=(16, 32, 32)).astype(np.float32),
        (4, 2, 2),
        ('a', 'b', 'c'),
        (None, ('a', 'b'), 'c'),
        transpose_axes=(0, 2, 1),
    )
    # 8. Edge cases: target_d==1 reshape, hashable non-str axes, num_chunks > 1
    # zero-all-gather split, non-prefix reshape, tiny axis-0, size-1 mesh,
    # incompatible device order, and unaligned rank flatten.
    devs8 = jax.devices()[:8]
    m24 = jax.sharding.Mesh(np.array(devs8).reshape(2, 4), ('fsdp', 'tp'))
    r8 = jax.sharding.NamedSharding(
        jax.sharding.Mesh(np.array(devs8), ('rank',)),
        jax.sharding.PartitionSpec('rank'),
    )
    self.assertEqual(
        safetensors_saver._merge_reshape_spec(
            (1, 16, 32), jax.sharding.PartitionSpec(None, 'fsdp', None), m24,
            (1, 512),
        ),
        (None, 'fsdp'),
    )
    self.assertEqual(
        safetensors_saver._merge_reshape_spec(
            (1, 2, 32), jax.sharding.PartitionSpec(None, 'fsdp', 'tp'), m24,
            (1, 64),
        ),
        (None, ('fsdp', 'tp')),
    )
    self.assertEqual(
        safetensors_saver._merge_reshape_spec(
            (16, 32, 1), jax.sharding.PartitionSpec('fsdp', 'tp', None), m24,
            (16, 32),
        ),
        ('fsdp', 'tp'),
    )
    self.assertEqual(
        safetensors_saver._merge_reshape_spec(
            (16, 32), jax.sharding.PartitionSpec('fsdp', 'tp'), m24,
            (1, 16, 32),
        ),
        (None, 'fsdp', 'tp'),
    )
    # Hashable non-str mesh axes: never silently dropped by
    # _compatible_mesh_axes; reshape/plan helpers accept arbitrary hashable ids.
    int_m = jax.sharding.Mesh(np.array(devs8).reshape(2, 4), (1, 2))
    mix_m = jax.sharding.Mesh(np.array(devs8).reshape(2, 4), ('fsdp', 1))
    self.assertIsNone(
        safetensors_saver._compatible_mesh_axes(
            jax.sharding.NamedSharding(
                mix_m, jax.sharding.PartitionSpec('fsdp', 1)
            ),
            r8,
        )
    )
    self.assertEqual(
        safetensors_saver._merge_reshape_spec(
            (16, 16), jax.sharding.PartitionSpec(1, 2), int_m, (16, 16)
        ),
        (1, 2),
    )
    _, _, d0_int, rep_int, ops_int = safetensors_saver._plan_axis0_reshard_ops(
        (1, 2), (0, 1), (1, 2)
    )
    self.assertEqual((d0_int, rep_int, ops_int), ((1,), (), ((2, 1),)))
    # num_chunks > 1 on (2, 4) and (2, 2, 2) must emit 0 all-gathers & parity.
    for m_shape, ax_names, pspec in (
        ((2, 4), ('fsdp', 'tp'), ('fsdp', None, 'tp')),
        ((2, 2, 2), ('ep', 'fsdp', 'tp'), ('ep', 'fsdp', 'tp')),
    ):
      c_mesh = jax.sharding.Mesh(np.array(devs8).reshape(m_shape), ax_names)
      c_shd = jax.sharding.NamedSharding(
          c_mesh, jax.sharding.PartitionSpec(*pspec)
      )
      c_np = rng.normal(size=(16, 32, 16)).astype(np.float32)
      c_arr = jax.device_put(jnp.asarray(c_np), c_shd)
      c_comp = safetensors_saver._jit_transform_on_hbm.lower(
          c_arr, None, None, 1.0, None, (0, 2, 1), None, r8, 8, 8, c_shd
      ).compile()
      c_counts = self._count_hlo_collectives(c_comp.as_text())
      self.assertEqual(c_counts['all-gather'], 0, c_comp.as_text())
      got_chunks = c_comp(c_arr, None, None, 1.0)
      ref_chunks = np.split(
          np.transpose(c_np, (0, 2, 1)).reshape(8, 8, -1), 8, axis=0
      )
      for g_c, r_c in zip(got_chunks, ref_chunks):
        np.testing.assert_array_equal(np.asarray(g_c), np.squeeze(r_c, axis=0))
    self.assertIsNone(
        safetensors_saver._merge_reshape_spec(
            (16, 4, 8), jax.sharding.PartitionSpec('fsdp', None, 'tp'), m24,
            (16, 32),
        )
    )
    self.assertIsNone(
        safetensors_saver._merge_reshape_spec(
            (16, 4), jax.sharding.PartitionSpec('fsdp', 'tp'), m24, (16, 3)
        )
    )
    self.assertIsNone(
        safetensors_saver._compatible_mesh_axes(None, r8)
    )
    rev_m = jax.sharding.Mesh(np.array(devs8[::-1]).reshape(2, 4), ('f', 't'))
    self.assertIsNone(
        safetensors_saver._compatible_mesh_axes(
            jax.sharding.NamedSharding(rev_m, jax.sharding.PartitionSpec('f')),
            r8,
        )
    )
    m11 = jax.sharding.Mesh(np.array(devs8[:1]).reshape(1, 1), ('f', 't'))
    r1 = jax.sharding.NamedSharding(
        jax.sharding.Mesh(np.array(devs8[:1]), ('rank',)),
        jax.sharding.PartitionSpec('rank'),
    )
    self.assertIsNone(
        safetensors_saver._compatible_mesh_axes(
            jax.sharding.NamedSharding(m11, jax.sharding.PartitionSpec('f')),
            r1,
        )
    )
    # Non-divisible axis-0 falls back cleanly and stays bit-exact.
    tiny_np = rng.normal(size=(4, 2048)).astype(np.float32)
    tiny_shd = jax.sharding.NamedSharding(
        m24, jax.sharding.PartitionSpec('fsdp', 'tp')
    )
    tiny_arr = jax.device_put(jnp.asarray(tiny_np), tiny_shd)
    tiny_out = safetensors_saver._jit_transform_on_hbm(
        tiny_arr, None, None, 1.0, None, None, None, r8, 8, 1, tiny_shd
    )
    np.testing.assert_array_equal(
        np.asarray(tiny_out), tiny_np.reshape(8, -1, 128)
    )
    # Unaligned total size (< 8 * 1024) directly via _reshard_to_rank_generic.
    small_np = rng.normal(size=(16, 16)).astype(np.float32)
    small_arr = jax.device_put(jnp.asarray(small_np), tiny_shd)
    small_res = safetensors_saver._reshard_to_rank_generic(
        small_arr, tiny_shd, None, None, None, r8
    )
    self.assertIsNotNone(small_res)
    np.testing.assert_array_equal(
        np.asarray(small_res), small_np.reshape(8, -1)
    )

  def test_single_host_multi_rank_routing_w1_and_fused_wave_hlo(self):
    rng = np.random.default_rng(11)
    devs8 = jax.devices()[:8]
    for mesh_shape, pspec, expect_zero in (
        ((8, 1), ('fsdp', None, None), True),
        ((2, 4), ('fsdp', 'tp', None), False),
    ):
      mesh = jax.sharding.Mesh(
          np.array(devs8).reshape(mesh_shape), ('fsdp', 'tp')
      )
      shd = jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec(*pspec)
      )
      rank_shd = jax.sharding.NamedSharding(
          jax.sharding.Mesh(np.array(devs8), ('rank',)),
          jax.sharding.PartitionSpec('rank'),
      )
      w1_np = rng.normal(size=(8, 64, 128)).astype(np.float32)
      w2_np = rng.normal(size=(8, 64, 128)).astype(np.float32)
      flat = {
          'e1': jax.device_put(jnp.asarray(w1_np), shd),
          'e2': jax.device_put(jnp.asarray(w2_np), shd),
      }
      specs = [
          safetensors_saver.TensorExportSpec(
              'e1', 'm.e1.{expert}', transpose_axes=(0, 2, 1),
              unstack_axis=0, nbytes=w1_np.nbytes,
          ),
          safetensors_saver.TensorExportSpec(
              'e2', 'm.e2.{expert}', transpose_axes=(0, 2, 1),
              unstack_axis=0, nbytes=w2_np.nbytes,
          ),
      ]
      self.assertTrue(
          safetensors_saver._uses_multi_rank_routing(specs[0], flat['e1'], 8)
      )
      self.assertFalse(
          safetensors_saver._uses_multi_rank_routing(specs[0], flat['e1'], 1)
      )
      self.assertFalse(
          safetensors_saver._uses_multi_rank_routing(
              dataclasses.replace(specs[0], is_replicated=True), flat['e1'], 8
          )
      )
      plan = safetensors_saver.build_save_plan(
          specs,
          num_ranks=8,
          max_bucket_bytes=96 * 1024,
          unstack_counts={'e1': 8, 'e2': 8},
      )
      # Verify W-1 enables fused identity wave on 1-host 8-device TPU mesh
      # with 0 all-gathers (and 0 collectives total on (8, 1)).
      with mock.patch.object(jax, 'default_backend', return_value='tpu'):
        fusable = safetensors_saver._fusable_wave_sources(
            flat, plan.waves[0], 8
        )
        self.assertEqual(fusable, {'e1', 'e2'})
        self.assertTrue(
            safetensors_saver._can_fuse_identity_wave(plan.waves[0], fusable, 8)
        )
      fused_compiled = (
          safetensors_saver._jit_transform_and_concat_fusable_wave.lower(
              (flat['e1'], flat['e2']),
              (None, None),
              ((0, 2, 1), (0, 2, 1)),
              None,
              rank_shd,
              (shd, shd),
          ).compile()
      )
      f_counts = self._count_hlo_collectives(fused_compiled.as_text())
      self.assertEqual(f_counts['all-gather'], 0)
      if expect_zero:
        self.assertEqual(f_counts['all-to-all'], 0)
        self.assertEqual(f_counts['collective-permute'], 0)
      else:
        self.assertEqual(f_counts['all-to-all'], 2)
      # Full AOT compile + export bit-exact parity on 1H 8-device CPU.
      compiled = safetensors_saver.compile_save_plan(
          flat, plan, devices=devs8
      )
      with tempfile.TemporaryDirectory() as tmpdir:
        safetensors_saver.export_safetensors_with_plan(
            flat, plan, tmpdir, devices=devs8, compiled_plan=compiled
        )
        loaded = {}
        for fn in os.listdir(tmpdir):
          if fn.endswith('.safetensors'):
            loaded.update(safe_flax.load_file(os.path.join(tmpdir, fn)))
        self.assertLen(loaded, 16)
        for i in range(8):
          np.testing.assert_array_equal(loaded[f'm.e1.{i}'], w1_np[i].T)
          np.testing.assert_array_equal(loaded[f'm.e2.{i}'], w2_np[i].T)

  def test_oversized_tensor_sub_chunking_w3_parity_and_hbm_bound(self):
    rng = np.random.default_rng(19)
    devs8 = jax.devices()[:8]
    mesh = jax.sharding.Mesh(np.array(devs8).reshape(2, 4), ('fsdp', 'tp'))
    shd = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('fsdp', 'tp')
    )
    # 256 KiB (8 tiles/rank) + 192 KiB (6 tiles/rank) on rank 0 (same owner!)
    # + 224 KiB (7 tiles/rank, uneven tail) on rank 1 + 64 KiB normal on rank 2.
    a0_np = rng.normal(size=(256, 256)).astype(np.float32)
    a1_np = rng.normal(size=(192, 256)).astype(np.float32)
    b0_np = rng.normal(size=(224, 256)).astype(np.float32)
    sm_np = rng.normal(size=(64, 256)).astype(np.float32)
    flat = {
        'a0': jax.device_put(jnp.asarray(a0_np), shd),
        'a1': jax.device_put(jnp.asarray(a1_np), shd),
        'b0': jax.device_put(jnp.asarray(b0_np), shd),
        'sm': jax.device_put(jnp.asarray(sm_np), shd),
    }
    s_a0 = safetensors_saver.TensorExportSpec(
        'a0', 'hf.a0', nbytes=a0_np.nbytes
    )
    s_a1 = safetensors_saver.TensorExportSpec(
        'a1', 'hf.a1', nbytes=a1_np.nbytes
    )
    s_b0 = safetensors_saver.TensorExportSpec(
        'b0', 'hf.b0', nbytes=b0_np.nbytes
    )
    s_sm = safetensors_saver.TensorExportSpec(
        'sm', 'hf.sm', nbytes=sm_np.nbytes
    )
    # Edge cases of _split_oversized_batch: 0, tiny, None, and unstack_axis.
    _, _, ov_zero, steps_zero = safetensors_saver._split_oversized_batch(
        [(0, s_a0)], [flat['a0']], 8, 0
    )
    self.assertEmpty(ov_zero)
    self.assertEqual(steps_zero, 0)
    _, _, ov_none, steps_none = safetensors_saver._split_oversized_batch(
        [(0, s_a0)], [flat['a0']], 8, None
    )
    self.assertEmpty(ov_none)
    self.assertEqual(steps_none, 0)
    _, _, ov_tiny, _ = safetensors_saver._split_oversized_batch(
        [(0, s_a0)], [flat['a0']], 8, 1024
    )
    self.assertEmpty(ov_tiny)
    s_unstack = dataclasses.replace(s_a0, unstack_axis=0)
    norm_u, _, ov_u, _ = safetensors_saver._split_oversized_batch(
        [(0, s_unstack)], [flat['a0']], 8, 96 * 1024
    )
    self.assertEmpty(ov_u)
    self.assertLen(norm_u, 1)
    # Verify TPU oversized path lowers to 1 dynamic ragged_all_to_all kernel.
    rk_m, rk_s = safetensors_saver._resolve_rank_mesh_shd(devs8, None)
    with mock.patch.object(jax, 'default_backend', return_value='tpu'):
      _, ov_args = safetensors_saver._oversized_exchange_args(
          [(0, s_a0, flat['a0'].reshape(8, -1, 128), 6)], 3, rk_m, rk_s
      )
      ov_hlo = safetensors_saver._jit_exchange_shards_all_to_all.lower(
          *ov_args
      ).as_text()
    self.assertIn('ragged_all_to_all', ov_hlo)
    # Single-wave materialize_wave_to_host with max_bucket_bytes=96 KiB on wave.
    w0 = safetensors_saver.WaveBucket(
        wave_index=0,
        rank_assignments={0: (s_a0, s_a1), 1: (s_b0,), 2: (s_sm,)},
        rank_bytes={
            0: a0_np.nbytes + a1_np.nbytes,
            1: b0_np.nbytes,
            2: sm_np.nbytes,
        },
        max_bucket_bytes=96 * 1024,
    )
    host_w, _, _ = safetensors_saver.materialize_wave_to_host(
        flat, w0, None, devs8
    )
    np.testing.assert_array_equal(
        np.asarray(host_w[0]['hf.a0']).reshape(a0_np.shape), a0_np
    )
    np.testing.assert_array_equal(
        np.asarray(host_w[0]['hf.a1']).reshape(a1_np.shape), a1_np
    )
    np.testing.assert_array_equal(
        np.asarray(host_w[1]['hf.b0']).reshape(b0_np.shape), b0_np
    )
    np.testing.assert_array_equal(
        np.asarray(host_w[2]['hf.sm']).reshape(sm_np.shape), sm_np
    )
    # Multi-wave 2-wave overlap (dispatch_wave_on_hbm + complete_wave_d2h) +
    # AOT: 10 oversized/normal specs across 8 ranks guarantees >= 2 waves.
    mw_flat = dict(flat)
    mw_specs = [s_a0, s_a1, s_b0, s_sm]
    extra_nps: dict[str, np.ndarray] = {}
    for idx in range(6):
      arr_np = rng.normal(size=(160, 256)).astype(np.float32)
      k = f'ov{idx}'
      extra_nps[k] = arr_np
      mw_flat[k] = jax.device_put(jnp.asarray(arr_np), shd)
      mw_specs.append(
          safetensors_saver.TensorExportSpec(k, f'hf.{k}', nbytes=arr_np.nbytes)
      )
    plan = safetensors_saver.build_save_plan(
        mw_specs, num_ranks=8, max_bucket_bytes=96 * 1024
    )
    self.assertGreaterEqual(len(plan.waves), 2)
    compiled = safetensors_saver.compile_save_plan(mw_flat, plan, devices=devs8)
    with tempfile.TemporaryDirectory() as tmpdir:
      safetensors_saver.export_safetensors_with_plan(
          mw_flat, plan, tmpdir, devices=devs8, compiled_plan=compiled
      )
      loaded = {}
      for fn in os.listdir(tmpdir):
        if fn.endswith('.safetensors'):
          loaded.update(safe_flax.load_file(os.path.join(tmpdir, fn)))
      np.testing.assert_array_equal(loaded['hf.a0'], a0_np)
      np.testing.assert_array_equal(loaded['hf.a1'], a1_np)
      np.testing.assert_array_equal(loaded['hf.b0'], b0_np)
      np.testing.assert_array_equal(loaded['hf.sm'], sm_np)
      for k, arr_np in extra_nps.items():
        np.testing.assert_array_equal(loaded[f'hf.{k}'], arr_np)
    # Verify W-3 strictly reduces AOT peak temp HBM on an oversized 1 MiB array.
    big_np = rng.normal(size=(512, 512)).astype(np.float32)
    big_flat = {'big': jax.device_put(jnp.asarray(big_np), shd)}
    big_spec = [
        safetensors_saver.TensorExportSpec(
            'big', 'hf.big', nbytes=big_np.nbytes
        )
    ]
    mono_plan = safetensors_saver.build_save_plan(
        big_spec, num_ranks=8, max_bucket_bytes=16 << 20
    )
    chunk_plan = safetensors_saver.build_save_plan(
        big_spec, num_ranks=8, max_bucket_bytes=128 * 1024
    )
    mono_comp = safetensors_saver.compile_save_plan(
        big_flat, mono_plan, devices=devs8
    )
    chunk_comp = safetensors_saver.compile_save_plan(
        big_flat, chunk_plan, devices=devs8
    )
    self.assertLess(
        chunk_comp.peak_temp_hbm_bytes, mono_comp.peak_temp_hbm_bytes
    )
    # Verify 1 single AOT exchange kernel covers all K=8 oversized steps.
    ex_kernels = [k for k in chunk_comp.executables if k[0] == 'exchange']
    self.assertLen(ex_kernels, 1)
    # Verify step_wave_to_host drains Wave 0 oversized batch BEFORE Wave 1.
    w_ov = safetensors_saver.WaveBucket(
        wave_index=0,
        rank_assignments={0: (s_a0,)},
        rank_bytes={0: a0_np.nbytes},
        max_bucket_bytes=96 * 1024,
    )
    w_norm = safetensors_saver.WaveBucket(
        wave_index=1,
        rank_assignments={2: (s_sm,)},
        rank_bytes={2: sm_np.nbytes},
        max_bucket_bytes=96 * 1024,
    )
    h0 = safetensors_saver.dispatch_wave_on_hbm(flat, w_ov, None, devs8)
    self.assertTrue(h0.oversized_batches)
    events: list[str] = []
    orig_complete = safetensors_saver.complete_wave_d2h
    orig_dispatch = safetensors_saver.dispatch_wave_on_hbm

    def _spy_complete(handle):
      events.append('complete')
      return orig_complete(handle)

    def _spy_dispatch(*args, **kwargs):
      events.append('dispatch')
      return orig_dispatch(*args, **kwargs)

    with (
        mock.patch.object(
            safetensors_saver, 'complete_wave_d2h', side_effect=_spy_complete
        ),
        mock.patch.object(
            safetensors_saver, 'dispatch_wave_on_hbm', side_effect=_spy_dispatch
        ),
    ):
      host_0, _, _, h1 = safetensors_saver.step_wave_to_host(
          flat, [w_ov, w_norm], 0, h0, None, devs8
      )
    self.assertEqual(events, ['complete', 'dispatch'])
    np.testing.assert_array_equal(
        np.asarray(host_0[0]['hf.a0']).reshape(a0_np.shape), a0_np
    )
    self.assertIsNotNone(h1)
    host_1, _, _ = safetensors_saver.complete_wave_d2h(h1)
    np.testing.assert_array_equal(
        np.asarray(host_1[2]['hf.sm']).reshape(sm_np.shape), sm_np
    )

  def test_heterogeneous_ragged_wave_p0_uninterleave_parity(self):
    # Regression test for P0 writer bug: a mixed non-fusable wave where rank 0
    # holds 2 chunked 2D MoE specs while rank 1 holds MoE + attn specs must
    # still uninterleave rank 0's ragged_all_to_all buffer on host.
    rng = np.random.default_rng(29)
    devs8 = jax.devices()[:8]
    mesh = jax.sharding.Mesh(np.array(devs8).reshape(2, 4), ('fsdp', 'tp'))
    shd3 = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('fsdp', None, 'tp')
    )
    shd2 = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec('fsdp', 'tp')
    )
    # Each 2-expert chunk has 2*64*64 = 8192 = 8*1024 elems (1024-aligned/rank).
    e1_np = rng.normal(size=(16, 64, 64)).astype(np.float32)
    e2_np = rng.normal(size=(16, 64, 64)).astype(np.float32)
    attn_np = rng.normal(size=(128, 64)).astype(np.float32)
    flat = {
        'e1': jax.device_put(jnp.asarray(e1_np), shd3),
        'e2': jax.device_put(jnp.asarray(e2_np), shd3),
        'attn': jax.device_put(jnp.asarray(attn_np), shd2),
    }
    chunk_bytes = e1_np.nbytes // 8
    rank_assign: dict[int, tuple[safetensors_saver.TensorExportSpec, ...]] = {}
    rank_bytes: dict[int, int] = {}
    for r in range(8):
      s1 = safetensors_saver.TensorExportSpec(
          'e1', 'm.e1.{expert}', transpose_axes=(0, 2, 1),
          unstack_axis=0, unstack_range=(r * 2, (r + 1) * 2),
          nbytes=chunk_bytes,
      )
      s2 = safetensors_saver.TensorExportSpec(
          'e2', 'm.e2.{expert}', transpose_axes=(0, 2, 1),
          unstack_axis=0, unstack_range=(r * 2, (r + 1) * 2),
          nbytes=chunk_bytes,
      )
      if r == 1:
        sa = safetensors_saver.TensorExportSpec(
            'attn', 'm.attn', nbytes=attn_np.nbytes
        )
        rank_assign[r] = (s1, s2, sa)
        rank_bytes[r] = 2 * chunk_bytes + attn_np.nbytes
      else:
        rank_assign[r] = (s1, s2)
        rank_bytes[r] = 2 * chunk_bytes
    wave = safetensors_saver.WaveBucket(0, rank_assign, rank_bytes)
    self.assertEmpty(safetensors_saver._fusable_wave_sources(flat, wave, 8))
    host_wave, _, _ = safetensors_saver.materialize_wave_to_host(
        flat, wave, None, devs8
    )
    e1_t = np.transpose(e1_np, (0, 2, 1))
    e2_t = np.transpose(e2_np, (0, 2, 1))
    for r in range(8):
      k1 = safetensors_saver._wave_tensor_key(rank_assign[r][0])
      k2 = safetensors_saver._wave_tensor_key(rank_assign[r][1])
      np.testing.assert_array_equal(
          np.asarray(host_wave[r][k1]).reshape(2, 64, 64),
          e1_t[r * 2 : (r + 1) * 2],
      )
      np.testing.assert_array_equal(
          np.asarray(host_wave[r][k2]).reshape(2, 64, 64),
          e2_t[r * 2 : (r + 1) * 2],
      )
    np.testing.assert_array_equal(
        np.asarray(host_wave[1]['m.attn']).reshape(attn_np.shape), attn_np
    )
    # Verify BOTH CPU (`all_to_all` + on-HBM `local_perm`) and simulated TPU
    # (`ragged_all_to_all` + on-HBM `local_perm` + `uint32 (M, 128)` bitcast)
    # produce exact per-rank layouts without host CPU uninterleave.
    e1_bf16 = np.asarray(jnp.asarray(e1_np, jnp.bfloat16))
    e2_bf16 = np.asarray(jnp.asarray(e2_np, jnp.bfloat16))
    attn_bf16 = np.asarray(jnp.asarray(attn_np, jnp.bfloat16))
    flat_bf16 = {
        'e1': jax.device_put(jnp.asarray(e1_bf16), shd3),
        'e2': jax.device_put(jnp.asarray(e2_bf16), shd3),
        'attn': jax.device_put(jnp.asarray(attn_bf16), shd2),
    }
    bf16_wave, _, _ = safetensors_saver.materialize_wave_to_host(
        flat_bf16, wave, jnp.bfloat16, devs8
    )
    e1_t_bf16 = np.transpose(e1_bf16, (0, 2, 1))
    e2_t_bf16 = np.transpose(e2_bf16, (0, 2, 1))
    for r in range(8):
      k1 = safetensors_saver._wave_tensor_key(rank_assign[r][0])
      k2 = safetensors_saver._wave_tensor_key(rank_assign[r][1])
      np.testing.assert_array_equal(
          np.asarray(bf16_wave[r][k1]).reshape(2, 64, 64),
          e1_t_bf16[r * 2 : (r + 1) * 2],
      )
      np.testing.assert_array_equal(
          np.asarray(bf16_wave[r][k2]).reshape(2, 64, 64),
          e2_t_bf16[r * 2 : (r + 1) * 2],
      )
    np.testing.assert_array_equal(
        np.asarray(bf16_wave[1]['m.attn']).reshape(attn_bf16.shape), attn_bf16
    )
    # Directly verify TPU `ragged_all_to_all` -> on-HBM `local_perm` ->
    # `uint32 (M, 128)` bitcast -> `_slice_d2h_target` -> `_finish_batch_d2h`.
    r0_s1, r0_s2 = rank_assign[0]
    r1_s1, r1_s2, r1_sa = rank_assign[1]
    raw_r0 = jnp.asarray(
        np.concatenate(
            [e1_t_bf16[0:2].reshape(8, 1024), e2_t_bf16[0:2].reshape(8, 1024)],
            axis=1,
        )
    ).reshape(16, 8, 128)
    raw_r0 = jnp.pad(raw_r0, ((0, 8), (0, 0), (0, 0)))
    raw_r1 = jnp.asarray(
        np.concatenate(
            [
                e1_t_bf16[2:4].reshape(8, 1024),
                e2_t_bf16[2:4].reshape(8, 1024),
                attn_bf16.reshape(8, 1024),
            ],
            axis=1,
        )
    ).reshape(24, 8, 128)
    perm_r0 = safetensors_saver.build_tile_perm_row([1, 1], 8, 3, True)
    perm_r1 = safetensors_saver.build_tile_perm_row([1, 1, 1], 8, 3, True)
    with mock.patch.object(jax, 'default_backend', return_value='tpu'):
      packed_r0 = safetensors_saver._slice_d2h_target(
          safetensors_saver._pack_bf16_to_u32_d2h(raw_r0[perm_r0]),
          16 * 1024,
          True,
          is_packed_bf16=True,
      )
      packed_r1 = safetensors_saver._slice_d2h_target(
          safetensors_saver._pack_bf16_to_u32_d2h(raw_r1[perm_r1]),
          24 * 1024,
          True,
          is_packed_bf16=True,
      )
    self.assertEqual(packed_r0.dtype, jnp.uint32)
    self.assertEqual(packed_r1.dtype, jnp.uint32)
    tpu_host, _ = safetensors_saver._finish_batch_d2h(
        (packed_r0, packed_r1),
        (0, 1),
        {
            0: [(0, r0_s1, (8, 1024)), (0, r0_s2, (8, 1024))],
            1: [
                (1, r1_s1, (8, 1024)),
                (1, r1_s2, (8, 1024)),
                (1, r1_sa, (8, 1024)),
            ],
        },
        defer_contiguous=False,
        is_packed_bf16=True,
    )
    np.testing.assert_array_equal(
        np.asarray(tpu_host[0][safetensors_saver._wave_tensor_key(r0_s1)])
        .reshape(2, 64, 64),
        e1_t_bf16[0:2],
    )
    np.testing.assert_array_equal(
        np.asarray(tpu_host[0][safetensors_saver._wave_tensor_key(r0_s2)])
        .reshape(2, 64, 64),
        e2_t_bf16[0:2],
    )
    np.testing.assert_array_equal(
        np.asarray(tpu_host[1][safetensors_saver._wave_tensor_key(r1_s1)])
        .reshape(2, 64, 64),
        e1_t_bf16[2:4],
    )
    np.testing.assert_array_equal(
        np.asarray(tpu_host[1][safetensors_saver._wave_tensor_key(r1_s2)])
        .reshape(2, 64, 64),
        e2_t_bf16[2:4],
    )
    np.testing.assert_array_equal(
        np.asarray(tpu_host[1]['m.attn']).reshape(attn_bf16.shape), attn_bf16
    )

  def test_phase4_multihost_replicated_non_id_moe_w3_tpu_and_0d_guards(self):
    # 1. P0-A: Multi-host replicated-only wave must filter by local process and
    # read `addressable_shards[0].data` rather than global `copy_to_host_async`.
    devs2 = jax.devices()[:2]
    local_shard = jnp.arange(16, dtype=jnp.float32)

    class _FakeGlobalReplicated:
      shape = (16,)
      dtype = jnp.dtype(jnp.float32)
      ndim = 1
      size = 16
      sharding = jax.sharding.NamedSharding(
          jax.sharding.Mesh(np.array(devs2), ('x',)),
          jax.sharding.PartitionSpec(),
      )
      addressable_shards = (types.SimpleNamespace(data=local_shard),)

      def copy_to_host_async(self):
        raise RuntimeError('non-addressable global array D2H crash')

    fake_devs = (
        types.SimpleNamespace(process_index=0),
        types.SimpleNamespace(process_index=1),
    )
    rep_s0 = safetensors_saver.TensorExportSpec(
        'n0', 'm.n0', nbytes=64, is_replicated=True
    )
    rep_s1 = safetensors_saver.TensorExportSpec(
        'n1', 'm.n1', nbytes=64, is_replicated=True
    )
    rep_wave = safetensors_saver.WaveBucket(
        0, {0: (rep_s0,), 1: (rep_s1,)}, {0: 64, 1: 64}
    )
    with (
        mock.patch.object(
            safetensors_saver,
            '_resolve_rank_mesh_shd',
            return_value=(None, None),
        ),
        mock.patch.object(
            safetensors_saver,
            '_transform_wave_pairs',
            return_value=[_FakeGlobalReplicated(), _FakeGlobalReplicated()],
        ),
        mock.patch.object(jax, 'process_index', return_value=1),
        mock.patch.object(jax, 'block_until_ready'),
    ):
      handle = safetensors_saver.dispatch_wave_on_hbm(
          {}, rep_wave, None, cast(Any, fake_devs)
      )
    self.assertEqual([r for r, _ in handle.rep_specs], [1])
    self.assertLen(handle.rep_bufs, 1)
    host_rep, _, _ = safetensors_saver.complete_wave_d2h(handle)
    self.assertEqual(set(host_rep), {1})
    np.testing.assert_array_equal(host_rep[1]['m.n1'], np.asarray(local_shard))

    # 2. P0-B: Non-identity 2-source MoE wave (rotated rank order) must NOT
    # be marked fusable under TPU backend (otherwise ragged A2A + uninterleave
    # would shred 7/8 of every tensor).
    devs8 = jax.devices()[:8]
    m8 = jax.sharding.Mesh(np.array(devs8), ('a',))
    shd_e = jax.sharding.NamedSharding(
        m8, jax.sharding.PartitionSpec('a', None, None)
    )
    e_arr = jax.device_put(
        jnp.arange(8 * 32 * 32, dtype=jnp.float32).reshape(8, 32, 32), shd_e
    )
    flat_moe = {'e1': e_arr, 'e2': e_arr + 1.0}
    rot_assign = {}
    for idx in range(8):
      r = (idx + 4) % 8
      rot_assign[r] = (
          safetensors_saver.TensorExportSpec(
              'e1',
              'm.e1.{expert}',
              unstack_axis=0,
              unstack_range=(idx, idx + 1),
              nbytes=4096,
          ),
          safetensors_saver.TensorExportSpec(
              'e2',
              'm.e2.{expert}',
              unstack_axis=0,
              unstack_range=(idx, idx + 1),
              nbytes=4096,
          ),
      )
    rot_wave = safetensors_saver.WaveBucket(
        0, rot_assign, {r: 8192 for r in range(8)}
    )
    with mock.patch.object(jax, 'default_backend', return_value='tpu'):
      self.assertEmpty(
          safetensors_saver._fusable_wave_sources(flat_moe, rot_wave, 8)
      )

    # 3. W-3 TPU scatter uneven tail (`cnt < step_tiles`) with 2D (M, 128) bf16.
    num_r, step_t, tail_cnt = 4, 3, 2
    expected_tail = (
        np.arange(num_r * tail_cnt * 1024, dtype=np.float32)
        .astype(jnp.bfloat16)
        .reshape(num_r, tail_cnt, 1024)
    )
    tpu_packed = safetensors_saver._slice_d2h_target(
        jnp.asarray(expected_tail).reshape(-1, 128),
        num_r * tail_cnt * 1024,
        is_tpu=True,
    )
    assembled = {
        (0, 'ov'): np.empty(
            (num_r, step_t + tail_cnt, 1024), dtype=jnp.bfloat16
        )
    }
    safetensors_saver._scatter_oversized_step(
        active=[(0, tail_cnt, tpu_packed)],
        targets=[tpu_packed],
        owner_items={0: [('ov', step_t + tail_cnt, 0)]},
        assembled=assembled,
        num_ranks=num_r,
        t0=step_t,
        step_tiles=step_t,
        is_tpu=True,
    )
    np.testing.assert_array_equal(
        assembled[(0, 'ov')][:, step_t:, :], expected_tail
    )

    # 4. W-2 0-D scalar and non-1D rank_shd defensive guards.
    m2d = jax.sharding.Mesh(np.array(devs8).reshape(2, 4), ('x', 'y'))
    bad_rk = jax.sharding.NamedSharding(
        m2d, jax.sharding.PartitionSpec('x', 'y')
    )
    rk1d = jax.sharding.NamedSharding(m8, jax.sharding.PartitionSpec('a'))
    self.assertIsNone(
        safetensors_saver._reshard_to_rank_generic(
            e_arr, shd_e, None, None, None, bad_rk
        )
    )
    scalar = jax.device_put(
        jnp.array(1.0, dtype=jnp.float32),
        jax.sharding.NamedSharding(m8, jax.sharding.PartitionSpec()),
    )
    self.assertIsNone(
        safetensors_saver._reshard_to_rank_generic(
            scalar, scalar.sharding, None, None, None, rk1d
        )
    )

    # 5. _is_packed_identity_shard covers 1D, (M, 128), and non-packed shapes.
    self.assertTrue(
        safetensors_saver._is_packed_identity_shard(np.zeros(128))
    )
    self.assertTrue(
        safetensors_saver._is_packed_identity_shard(np.zeros((8, 128)))
    )
    self.assertFalse(
        safetensors_saver._is_packed_identity_shard(np.zeros((8, 64)))
    )
    self.assertFalse(
        safetensors_saver._is_packed_identity_shard(np.zeros((2, 8, 128)))
    )

  def test_write_ici_64dev_u32_bitcast_on_hbm_perm_and_oversized_overlap(self):
    # W-RC1: uint32 (M, 128) bitcast (align=8), exact cut_u32=(cut+1)//2,
    # skip eager slice within 1024 u32 elems, odd bf16, and genuine uint32.
    rng = np.random.default_rng(77)
    raw_bf16 = np.asarray(
        jnp.asarray(rng.standard_normal(4096), jnp.bfloat16)
    )
    with mock.patch.object(jax, 'default_backend', return_value='tpu'):
      u32_aligned = safetensors_saver._pack_bf16_to_u32_d2h(
          jnp.asarray(raw_bf16)
      )
      self.assertEqual(u32_aligned.dtype, jnp.uint32)
      self.assertEqual(u32_aligned.shape, (16, 128))
      np.testing.assert_array_equal(
          safetensors_saver._d2h_buf_to_1d(
              np.asarray(u32_aligned), is_packed_bf16=True
          ),
          raw_bf16,
      )
      # Odd row count (4 rows -> padded only to align=8 = 4 KiB, never 8192).
      u32_pad8 = safetensors_saver._pack_bf16_to_u32_d2h(
          jnp.asarray(raw_bf16[:1024])
      )
      self.assertEqual(u32_pad8.dtype, jnp.uint32)
      self.assertEqual(u32_pad8.shape, (8, 128))
      np.testing.assert_array_equal(
          safetensors_saver._d2h_buf_to_1d(
              np.asarray(u32_pad8), is_packed_bf16=True
          )[:1024],
          raw_bf16[:1024],
      )
      # Unaligned bf16 (< 256) and genuine uint32 stay 1D untouched.
      unaligned = safetensors_saver._pack_bf16_to_u32_d2h(
          jnp.asarray(raw_bf16[:128])
      )
      self.assertEqual(
          (unaligned.dtype, unaligned.shape), (jnp.bfloat16, (128,))
      )
      gen_u32 = safetensors_saver._pack_bf16_to_u32_d2h(
          jnp.arange(1024, dtype=jnp.uint32)
      )
      self.assertEqual((gen_u32.dtype, gen_u32.shape), (jnp.uint32, (1024,)))

    # `_slice_d2h_target` on u32 (M, 128): skip eager slice when slack <= 1024,
    # slice to align=8 with exact odd-bf16 ceiling `(cut_elems + 1) // 2`.
    self.assertIs(
        safetensors_saver._slice_d2h_target(
            u32_pad8, 1024, True, is_packed_bf16=True
        ),
        u32_pad8,
    )
    big_u32 = jax.lax.bitcast_convert_type(
        jnp.asarray(raw_bf16).reshape(16, 128, 2), jnp.uint32
    )
    odd_sliced = safetensors_saver._slice_d2h_target(
        big_u32, 513, True, is_packed_bf16=True
    )
    self.assertEqual(odd_sliced.shape, (8, 128))
    decoded_odd = safetensors_saver._d2h_buf_to_1d(
        np.asarray(odd_sliced), is_packed_bf16=True
    )[:513]
    np.testing.assert_array_equal(decoded_odd, raw_bf16[:513])
    # Genuine 1D and 2D (M, 128) uint32 are never halved or bitcast to bf16.
    self.assertEqual(
        safetensors_saver._slice_d2h_target(gen_u32, 64, True).shape, (64,)
    )
    gen_u32_2d = np.arange(16 * 128, dtype=np.uint32).reshape(16, 128)
    self.assertEqual(
        safetensors_saver._slice_d2h_target(
            jnp.asarray(gen_u32_2d), 1024, True
        ).shape,
        (8, 128),
    )
    np.testing.assert_array_equal(
        safetensors_saver._d2h_buf_to_1d(gen_u32_2d), gen_u32_2d.reshape(-1)
    )
    np.testing.assert_array_equal(
        safetensors_saver._d2h_buf_to_1d(np.arange(16, dtype=np.uint32)),
        np.arange(16, dtype=np.uint32),
    )

    # W-RC2: Multi-target D2H materializes in parallel via `_SHARED_WRITE_POOL`.
    orig_map = safetensors_saver._SHARED_WRITE_POOL.map
    with mock.patch.object(
        safetensors_saver._SHARED_WRITE_POOL, 'map', wraps=orig_map
    ) as spy_map:
      mats = safetensors_saver._materialize_d2h_targets(
          [jnp.ones(8, jnp.float32), jnp.zeros(8, jnp.float32)]
      )
      self.assertLen(mats, 2)
      spy_map.assert_called_once()

    # W-RC3: `/dev/shm` ring auto-sizes up to 128 GiB while Python flush cap
    # stays bounded at 4 waves / 4 GiB.
    self.assertEqual(safetensors_saver._MAX_INFLIGHT_FLUSH_WAVES, 4)
    self.assertEqual(safetensors_saver._MAX_INFLIGHT_FLUSH_BYTES, 4 << 30)
    self.assertEqual(safetensors_saver._SHM_RING_MAX_BYTES, 128 << 30)
    with mock.patch.object(
        safetensors_io, 'available_shm_bytes', return_value=450 << 30
    ):
      sb, n_reg, lg_b, n_lg = safetensors_saver.compute_shm_ring_geometry()
      total_ring = sb * n_reg + lg_b * n_lg
      self.assertGreater(total_ring, 100 << 30)
      self.assertLessEqual(total_ring, 128 << 30)

    # W-RC1 + W-RC4 on 64 devices: [WRITE-W1] fused identity u32 (M, 128) and
    # [WRITE-W2] non-identity ragged A2A + on-HBM local_perm + u32 (M, 128).
    devs64 = jax.devices()[:64]
    mesh64 = jax.sharding.Mesh(np.array(devs64).reshape(32, 2), ('fsdp', 'tp'))
    rank_mesh64 = jax.sharding.Mesh(np.array(devs64), ('rank',))
    rank_shd64 = jax.sharding.NamedSharding(
        rank_mesh64, jax.sharding.PartitionSpec('rank')
    )
    shd64_moe = jax.sharding.NamedSharding(
        mesh64, jax.sharding.PartitionSpec('fsdp', None, 'tp')
    )
    shd64_2d = jax.sharding.NamedSharding(
        mesh64, jax.sharding.PartitionSpec('fsdp', 'tp')
    )
    moe64_np = np.asarray(
        jnp.asarray(rng.standard_normal((64, 32, 32)), jnp.bfloat16)
    )
    moe64 = jax.device_put(jnp.asarray(moe64_np), shd64_moe)
    with mock.patch.object(jax, 'default_backend', return_value='tpu'):
      w1_hlo = (
          safetensors_saver._jit_transform_and_concat_fusable_wave.lower(
              (moe64,),
              (None,),
              ((0, 2, 1),),
              jnp.bfloat16,
              rank_shd64,
              (shd64_moe,),
          )
          .compile()
          .as_text()
      )
      self.assertIn('bitcast', w1_hlo)
      self.assertNotRegex(
          w1_hlo, r'u32\[[^\]]*\]\s*=\s*bitcast-convert\([^)]*bf16'
      )
      w1_counts = self._count_hlo_collectives(w1_hlo)
      self.assertEqual(w1_counts['all-gather'], 0)
      w1_assign = {
          r: (
              safetensors_saver.TensorExportSpec(
                  'm',
                  'm.{expert}',
                  transpose_axes=(0, 2, 1),
                  unstack_axis=0,
                  unstack_range=(r, r + 1),
                  nbytes=32 * 32 * 2,
              ),
          )
          for r in range(64)
      }
      w1_wave = safetensors_saver.WaveBucket(
          0, w1_assign, {r: 32 * 32 * 2 for r in range(64)}
      )
      w1_host, _, _ = safetensors_saver.materialize_wave_to_host(
          {'m': moe64}, w1_wave, jnp.bfloat16, devs64
      )
    moe64_t = np.transpose(moe64_np, (0, 2, 1))
    for r in (0, 1, 31, 63):
      k = safetensors_saver._wave_tensor_key(w1_assign[r][0])
      np.testing.assert_array_equal(
          np.asarray(w1_host[r][k]).reshape(1, 32, 32), moe64_t[r : r + 1]
      )

    # [WRITE-W2]: 64-dev non-identity multi-tensor wave: verify TPU HLO lowers
    # `ragged_all_to_all` + on-HBM `local_perm` + `uint32 (M, 128)` bitcast,
    # and verify 64-dev execution parity on CPU (`all_to_all` + `local_perm`).
    w2a_np = np.asarray(
        jnp.asarray(rng.standard_normal((256, 256)), jnp.bfloat16)
    )
    w2b_np = np.asarray(
        jnp.asarray(rng.standard_normal((256, 256)), jnp.bfloat16)
    )
    w2a = jax.device_put(jnp.asarray(w2a_np), shd64_2d)
    w2b = jax.device_put(jnp.asarray(w2b_np), shd64_2d)
    sa = safetensors_saver.TensorExportSpec(
        'a', 'hf.a', transpose_axes=(1, 0), nbytes=w2a_np.nbytes
    )
    sb_spec = safetensors_saver.TensorExportSpec(
        'b', 'hf.b', transpose_axes=(1, 0), nbytes=w2b_np.nbytes
    )
    w2_wave = safetensors_saver.WaveBucket(
        0, {0: (sa, sb_spec)}, {0: w2a_np.nbytes + w2b_np.nbytes}
    )
    s_a64 = jax.device_put(jnp.asarray(w2a_np.T.reshape(64, -1)), rank_shd64)
    s_b64 = jax.device_put(jnp.asarray(w2b_np.T.reshape(64, -1)), rank_shd64)
    with mock.patch.object(jax, 'default_backend', return_value='tpu'):
      (
          ord_s,
          ord_t,
          in_off,
          s_sz,
          m_recv,
          use_rag,
          need_p,
          r_cols,
          _,
      ) = safetensors_saver._compute_exchange_plan(
          [s_a64, s_b64], [0, 0], None, 64
      )
      self.assertTrue(use_rag)
      self.assertTrue(need_p)
      perm_np = np.stack([
          safetensors_saver.build_tile_perm_row(
              [w // 1024 for w in r_cols[r]], 64, m_recv // 1024, True
          )
          for r in range(64)
      ])
      perm_sds = jax.ShapeDtypeStruct(
          perm_np.shape, jnp.int32, sharding=rank_shd64
      )
      w2_lowered = safetensors_saver._jit_exchange_shards_all_to_all.lower(
          ord_s, rank_mesh64, 'rank', ord_t, in_off, s_sz, m_recv, True,
          perm_sds, None,
      )
      w2_hlo = w2_lowered.as_text()
      self.assertIn('ragged_all_to_all', w2_hlo)
      self.assertIn('gather', w2_hlo)
      self.assertIn('bitcast_convert', w2_hlo)
      self.assertIn('ui32', w2_hlo)
      # Pre-exchange packing: ragged_all_to_all operates on (N, 4, 128) ui32
      # via same-width bf16->ui16 bitcast (never width-upcasting bf16->ui32).
      self.assertRegex(w2_hlo, r'ragged_all_to_all.*4x128xui32')
      self.assertNotRegex(
          w2_hlo, r'bitcast_convert[^\n]*bf16[^\n]*->[^\n]*ui32'
      )
      # Oversized single-tensor wave keeps ragged_all_to_all on bf16 (zero-copy
      # input view; packs only the bounded step_tiles output to ui32).
      ov_meta_sds = jax.ShapeDtypeStruct(
          (64, 4, 64), jnp.int32, sharding=rank_shd64
      )
      ov_s64 = jax.ShapeDtypeStruct(
          (64, 65 * 1024), jnp.bfloat16, sharding=rank_shd64
      )
      ov_hlo = safetensors_saver._jit_exchange_shards_all_to_all.lower(
          (ov_s64,), rank_mesh64, 'rank', (0,), (0,) * 64, (0,) * 64, 1024,
          True, None, ov_meta_sds,
      ).as_text()
      self.assertRegex(ov_hlo, r'ragged_all_to_all.*8x128xbf16')
      self.assertNotRegex(
          ov_hlo, r'bitcast_convert[^\n]*bf16[^\n]*->[^\n]*ui32'
      )
      # Direct public `owner_groups` guard on `_compute_exchange_plan`.
      s4_a = jax.ShapeDtypeStruct((4, 1024), jnp.bfloat16)
      s4_b = jax.ShapeDtypeStruct((4, 1024), jnp.bfloat16)
      for bad_groups, bad_owners in (
          ([(0,)], [0, 1]),  # length mismatch
          ([(), (1,)], [0, 1]),  # empty group
          ([(0, 1, 2), (3,)], [0, 3]),  # multi-owner does not divide num_ranks
          ([(0, 1), (0,)], [0, 0]),  # rank 0 overlaps multi-owner group
          ([(0, 0), (2,)], [0, 2]),  # duplicate rank inside multi-owner group
      ):
        with self.assertRaises(ValueError):
          safetensors_saver._compute_exchange_plan(
              [s4_a, s4_b], bad_owners, bad_groups, 4
          )
      # Disjoint divisible multi-owner group passes validation cleanly.
      _, _, _, _, ok_mrecv, _, _, ok_cols, ok_meta = (
          safetensors_saver._compute_exchange_plan(
              [s4_a, s4_b], [0, 2], [(0, 1), (2,)], 4
          )
      )
      self.assertIsNotNone(ok_meta)
      self.assertEqual(ok_mrecv, 1024)
      self.assertEqual(ok_cols, [[], [], [1024], []])
    w2_host, _, _ = safetensors_saver.materialize_wave_to_host(
        {'a': w2a, 'b': w2b}, w2_wave, jnp.bfloat16, devs64
    )
    np.testing.assert_array_equal(
        np.asarray(w2_host[0]['hf.a']).reshape(256, 256), w2a_np.T
    )
    np.testing.assert_array_equal(
        np.asarray(w2_host[0]['hf.b']).reshape(256, 256), w2b_np.T
    )

    # W-RC5: Double-buffered oversized W-3 loop dispatches step+1 BEFORE
    # scattering step across 3 steps (including uneven tail), and TPU uint32
    # `(M, 128)` oversized scatter decodes bit-exact.
    devs4 = jax.devices()[:4]
    mesh4 = jax.sharding.Mesh(np.array(devs4), ('rank',))
    shd4 = jax.sharding.NamedSharding(mesh4, jax.sharding.PartitionSpec('rank'))
    ov_np = np.asarray(
        jnp.asarray(rng.standard_normal((4, 5, 1024)), jnp.bfloat16)
    )
    ov_arr = jax.device_put(jnp.asarray(ov_np), shd4)
    ov_spec = safetensors_saver.TensorExportSpec(
        'ov', 'hf.ov', nbytes=ov_np.nbytes
    )
    events: list[str] = []
    orig_disp = safetensors_saver._dispatch_oversized_step
    orig_scat = safetensors_saver._scatter_oversized_step

    def _trace_disp(*args, **kwargs):
      events.append(f'dispatch:{args[0]}')
      return orig_disp(*args, **kwargs)

    def _trace_scat(*args, **kwargs):
      events.append(f'scatter:{args[5] // args[6]}')
      return orig_scat(*args, **kwargs)

    with (
        mock.patch.object(
            safetensors_saver, '_dispatch_oversized_step', _trace_disp
        ),
        mock.patch.object(
            safetensors_saver, '_scatter_oversized_step', _trace_scat
        ),
    ):
      ov_host, _, _ = safetensors_saver._stream_oversized_batch_to_host(
          [(0, ov_spec, ov_arr, 5)],
          step_tiles=2,
          devices=devs4,
          compiled_plan=None,
      )
    self.assertEqual(
        events,
        [
            'dispatch:0',
            'dispatch:1',
            'scatter:0',
            'dispatch:2',
            'scatter:1',
            'scatter:2',
        ],
    )
    np.testing.assert_array_equal(ov_host[0]['hf.ov'], ov_np)
    # TPU uint32 (M, 128) oversized step scatter parity.
    with mock.patch.object(jax, 'default_backend', return_value='tpu'):
      ov_u32 = safetensors_saver._slice_d2h_target(
          safetensors_saver._pack_bf16_to_u32_d2h(jnp.asarray(ov_np[:, :2, :])),
          4 * 2 * 1024,
          True,
          is_packed_bf16=True,
      )
    self.assertEqual(ov_u32.dtype, jnp.uint32)
    ov_asm = {(0, 'ov'): np.empty((4, 2, 1024), dtype=jnp.bfloat16)}
    safetensors_saver._scatter_oversized_step(
        active=[(0, 2, ov_u32)],
        targets=[ov_u32],
        owner_items={0: [('ov', 2, 0)]},
        assembled=ov_asm,
        num_ranks=4,
        t0=0,
        step_tiles=2,
        is_tpu=True,
        is_packed_bf16=True,
    )
    np.testing.assert_array_equal(ov_asm[(0, 'ov')], ov_np[:, :2, :])

  def test_stage_pool_backend_isolation_and_high_fd_selector(self):
    # 1. P1 FD_SETSIZE=1024: select module must not be used; DefaultSelector
    # handles fd >= 1024 cleanly where select.select() raises ValueError.
    self.assertFalse(hasattr(safetensors_saver, 'select'))
    self.assertFalse(hasattr(safetensors_io, 'select'))
    self.assertIs(safetensors_io.selectors, selectors)
    r_fd, w_fd = os.pipe()
    hi_fd = fcntl.fcntl(r_fd, fcntl.F_DUPFD, 1024)
    os.close(r_fd)
    hi_fh = os.fdopen(hi_fd, 'r', encoding='utf-8')
    try:
      self.assertGreaterEqual(hi_fd, 1024)
      os.write(w_fd, b'READY\n')
      with self.assertRaises(ValueError):
        select.select([hi_fd], [], [], 0.0)
      hi_proc = mock.MagicMock()
      hi_proc.stdout = hi_fh
      hi_proc.stdin = mock.MagicMock()

      def _close_hi_write(*_args, **_kwargs):
        nonlocal w_fd
        if w_fd >= 0:
          os.close(w_fd)
          w_fd = -1

      hi_proc.stdin.write.side_effect = _close_hi_write
      hi_proc.poll.return_value = None
      with (
          tempfile.TemporaryDirectory() as tmpdir,
          mock.patch.object(
              safetensors_saver.subprocess, 'Popen', return_value=hi_proc
          ),
          mock.patch.object(
              safetensors_io,
              'resolve_gcs_vips',
              return_value=('10.0.0.1',),
          ),
      ):
        hi_pool = safetensors_saver.MultiprocStagePool(
            stage_dir=os.path.join(tmpdir, 'hi_fd_stage'),
            num_procs=1,
            shm_budget_bytes=4 << 20,
            max_bucket_bytes=128 << 10,
            num_large_slots=1,
        )
        hi_pool.close()
    finally:
      if w_fd >= 0:
        os.close(w_fd)
      hi_fh.close()

    # 2. Leaks #1, #2, #4: GCS HTTP pool vs registered-remote pool isolation.
    class _FakeStdout:

      def __init__(self):
        self._q: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._q.put('READY\n')

      def readline(self) -> str:
        return self._q.get()

      def __iter__(self):
        return iter(self._q.get, '')

      def close(self) -> None:
        self._q.put('')

    def _make_ready_popen(recorded_cmds, recorded_envs):
      def _popen(cmd, **kwargs):
        recorded_cmds.append(list(cmd))
        recorded_envs.append(dict(kwargs['env']))
        fake_out = _FakeStdout()
        p = mock.MagicMock()
        p.stdout = fake_out
        p.stdin = mock.MagicMock()
        p.stdin.close.side_effect = fake_out.close
        p.terminate.side_effect = fake_out.close
        p.kill.side_effect = fake_out.close
        p.poll.return_value = None
        return p

      return _popen

    gcs_cmds: list[list[str]] = []
    gcs_envs: list[dict[str, str]] = []
    mock_remote_flags = mock.MagicMock(return_value=('--undefok=f', '--f=1'))
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        mock.patch.dict(
            safetensors_io.STORAGE_BACKENDS,
            {'/rfs/': _remote_spec(write_flags=mock_remote_flags)},
        ),
        mock.patch.object(
            safetensors_io,
            'resolve_gcs_vips',
            return_value=('10.0.0.1', '10.0.0.2'),
        ) as mock_vips,
        mock.patch.object(
            safetensors_saver.subprocess,
            'Popen',
            side_effect=_make_ready_popen(gcs_cmds, gcs_envs),
        ),
    ):
      with safetensors_saver.MultiprocStagePool(
          stage_dir=os.path.join(tmpdir, 'gcs_stage'),
          num_procs=1,
          shm_budget_bytes=4 << 20,
          max_bucket_bytes=128 << 10,
          num_large_slots=1,
          use_gcs_http_write=True,
      ) as gcs_pool:
        mock_remote_flags.assert_not_called()
        mock_vips.assert_called_once()
        self.assertEqual(gcs_envs[0]['TUNIX_STAGE_BACKEND'], 'gcs_http')
        self.assertEqual(gcs_envs[0]['TUNIX_STAGE_NUM_WORKERS'], '1')
        self.assertEqual(gcs_envs[0]['TUNIX_STAGE_WORKER_IDX'], '0')
        self.assertIn('TUNIX_STAGE_ARENA_PATH', gcs_envs[0])
        self.assertIn('TUNIX_STAGE_PROCESS_IDX', gcs_envs[0])
        self.assertEqual(gcs_envs[0]['TUNIX_GCS_VIPS'], '10.0.0.1,10.0.0.2')
        self.assertFalse(
            any(k.startswith('TUNIX_REMOTE_') for k in gcs_envs[0]),
            gcs_envs[0],
        )
        self.assertEqual(
            safetensors_saver.resolve_storage_backend(
                'gs://bkt/x'
            ).stage_writer_fn,
            gcs_pool.open_writer,
        )
        self.assertIsNone(
            safetensors_saver.resolve_storage_backend(
                '/rfs/cell-d/x'
            ).stage_writer_fn
        )
      self.assertIsNone(
          safetensors_saver.resolve_storage_backend(
              'gs://bkt/x'
          ).stage_writer_fn
      )

    remote_cmds: list[list[str]] = []
    remote_envs: list[dict[str, str]] = []
    mock_remote_flags.reset_mock()
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        mock.patch.dict(
            safetensors_io.STORAGE_BACKENDS,
            {'/rfs/': _remote_spec(write_flags=mock_remote_flags)},
        ),
        mock.patch.dict(os.environ, {'TUNIX_GCS_VIPS': '1.2.3.4'}),
        mock.patch.object(
            safetensors_io,
            'resolve_gcs_vips',
            side_effect=AssertionError('Remote pool must not resolve VIPs'),
        ) as mock_vips,
        mock.patch.object(
            safetensors_saver.subprocess,
            'Popen',
            side_effect=_make_ready_popen(remote_cmds, remote_envs),
        ),
    ):
      with safetensors_saver.MultiprocStagePool(
          stage_dir=os.path.join(tmpdir, 'remote_stage'),
          num_procs=1,
          shm_budget_bytes=4 << 20,
          max_bucket_bytes=128 << 10,
          num_large_slots=1,
          use_gcs_http_write=False,
      ) as remote_pool:
        mock_remote_flags.assert_called_once()
        mock_vips.assert_not_called()
        self.assertEqual(remote_envs[0]['TUNIX_STAGE_BACKEND'], 'direct')
        self.assertIn('--f=1', remote_cmds[0])
        self.assertNotIn('TUNIX_GCS_VIPS', remote_envs[0])
        self.assertFalse(
            any(k.startswith('TUNIX_REMOTE_') for k in remote_envs[0]),
            remote_envs[0],
        )
        self.assertEqual(
            safetensors_saver.resolve_storage_backend(
                '/rfs/cell-d/x'
            ).stage_writer_fn,
            remote_pool.open_writer,
        )
        self.assertIsNone(
            safetensors_saver.resolve_storage_backend(
                'gs://bkt/x'
            ).stage_writer_fn
        )
        with remote_pool.activate():
          self.assertEqual(
              safetensors_saver.resolve_storage_backend(
                  '/rfs/cell-d/x'
              ).stage_writer_fn,
              remote_pool.open_writer,
          )
      self.assertIsNone(
          safetensors_saver.resolve_storage_backend(
              '/rfs/cell-d/x'
          ).stage_writer_fn
      )

  def test_review_hardening_u32_2d_exact_divisors_ops_and_flush_abort(self):
    # 1. `_parse_export_rule_ops` rejects duplicate ops and bad op order.
    with self.assertRaisesRegex(ValueError, 'duplicate'):
      safetensors_saver._parse_export_rule_ops(
          'k',
          [{'transpose': {'axes': (1, 0)}}, {'transpose': {'axes': (0, 1)}}],
      )
    with self.assertRaisesRegex(
        ValueError, 'reshape after transpose or unstack'
    ):
      safetensors_saver._parse_export_rule_ops(
          'k', [{'unstack': {'axis': 0}}, {'reshape': {'dst_shape': (2, 2)}}]
      )

    # 2. `_expand_moe_chunk_specs` falls back to largest valid divisor (not 1).
    moe_spec = safetensors_saver.TensorExportSpec(
        source_key='moe',
        hf_key='layers.0.e.{expert}',
        nbytes=4 * 4096,
        unstack_axis=0,
    )
    chunks = safetensors_saver._expand_moe_chunk_specs(
        [moe_spec], {'moe': 4}, limit=4096, num_ranks=1, rank_align_bytes=8192
    )
    self.assertLen(chunks, 2)
    self.assertEqual(chunks[0].unstack_range, (0, 2))
    self.assertEqual(chunks[1].unstack_range, (2, 4))

    # 3. `_group_batch_shards` preserves exact nbytes without rank truncation.
    devs = jax.devices()[:1]
    mesh1 = jax.sharding.Mesh(np.array(devs), ('rank',))
    shd1 = jax.sharding.NamedSharding(mesh1, jax.sharding.PartitionSpec('rank'))
    sp_a = safetensors_saver.TensorExportSpec(
        'moe', 'hf.{expert}', nbytes=20, unstack_range=(0, 1)
    )
    arr_big = jnp.zeros((10,), dtype=jnp.float32)
    _, _, _, items = safetensors_saver._group_batch_shards(
        [(0, sp_a)], [arr_big], num_ranks=3, rank_shd=shd1
    )
    self.assertEqual(items[0][0][2], (5,))

    # 4. `_can_fuse_identity_wave` rejects mixed source dtypes when output_dtype
    # is None and allows when output_dtype casts them to a common dtype.
    s_f32 = safetensors_saver.TensorExportSpec(
        'e0', 'hf.e0.{expert}', nbytes=16, unstack_range=(0, 1)
    )
    s_bf16 = safetensors_saver.TensorExportSpec(
        'e1', 'hf.e1.{expert}', nbytes=16, unstack_range=(0, 1)
    )
    s_f32_1 = dataclasses.replace(s_f32, unstack_range=(1, 2))
    s_bf16_1 = dataclasses.replace(s_bf16, unstack_range=(1, 2))
    mix_wave = safetensors_saver.WaveBucket(
        wave_index=0,
        rank_assignments={0: (s_f32, s_bf16), 1: (s_f32_1, s_bf16_1)},
        rank_bytes={0: 32, 1: 32},
    )
    mix_state = {
        'e0': jnp.zeros((2, 4), dtype=jnp.float32),
        'e1': jnp.zeros((2, 4), dtype=jnp.bfloat16),
    }
    self.assertFalse(
        safetensors_saver._can_fuse_identity_wave(
            mix_wave, {'e0', 'e1'}, 2, mix_state, None
        )
    )
    self.assertTrue(
        safetensors_saver._can_fuse_identity_wave(
            mix_wave, {'e0', 'e1'}, 2, mix_state, jnp.bfloat16
        )
    )

    # 5. `release_slot(count<=0)` is a safe no-op; `_await_worker_flush_reply`
    # returns promptly on dead worker or abort timeout without hanging.
    fake_pool = object.__new__(safetensors_saver.MultiprocStagePool)
    fake_pool._lock = threading.Lock()
    fake_pool._slot_refs = {0: 2}
    fake_pool.num_reg_slots = 2
    fake_pool._free_slots = queue.SimpleQueue()
    fake_pool._free_large_slots = queue.SimpleQueue()
    fake_pool.release_slot(0, count=0)
    fake_pool.release_slot(0, count=-1)
    self.assertEqual(fake_pool._slot_refs[0], 2)
    self.assertTrue(fake_pool._free_slots.empty())

    fake_pool._flush_replies = [queue.SimpleQueue(), queue.SimpleQueue()]
    dead_proc = mock.MagicMock()
    dead_proc.poll.return_value = 1
    wedged_proc = mock.MagicMock()
    wedged_proc.poll.return_value = None
    fake_pool._procs = [dead_proc, wedged_proc]
    self.assertIn(
        'exited',
        fake_pool._await_worker_flush_reply(0, abort=False) or '',
    )
    self.assertIn(
        'timed out',
        fake_pool._await_worker_flush_reply(
            1, abort=True, abort_timeout_sec=0.0
        )
        or '',
    )

  def test_audit_p1_abort_poison_multihost_barriers_moe_dtype_and_p2_limits(
      self,
  ):
    # P1-1: flush(abort=True) timeout & pipe error poison pool, reap dirty
    # workers, skip worker COMPOSE, fall back to direct delete, and fail health.
    # Use 2 wedged dirty workers so shared abort deadline stays bounded (<0.3s)
    # rather than blocking 0.5s per remaining worker.
    wedged_procs = []
    for _ in range(2):
      p = mock.MagicMock()
      p.stdout = io.StringIO('READY\n')
      p.stdin = mock.MagicMock()
      p.poll.return_value = None
      wedged_procs.append(p)
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        mock.patch.object(
            safetensors_saver.subprocess, 'Popen', side_effect=wedged_procs
        ),
        mock.patch.object(safetensors_saver, '_ABORT_FLUSH_TIMEOUT_SEC', 0.05),
        mock.patch.object(
            safetensors_io, 'terminate_proc'
        ) as mock_terminate,
        mock.patch.object(
            safetensors_saver, '_delete_gcs_parts_direct'
        ) as mock_del_direct,
    ):
      pool = safetensors_saver.MultiprocStagePool(
          stage_dir=os.path.join(tmpdir, 'abort_poison'),
          num_procs=2,
          shm_budget_bytes=4 << 20,
          max_bucket_bytes=128 << 10,
          num_large_slots=1,
      )
      try:
        pool._worker_err = None
        for w in range(2):
          pool._flush_replies[w] = queue.SimpleQueue()  # Wedged: never replies.
          pool._dirty_workers.add(w)
        pool.register_gcs_composite(
            'gs://bkt/shard.safetensors', ('gs://bkt/p0', 'gs://bkt/p1')
        )
        t_abort = safetensors_saver.time.monotonic()
        with self.assertRaisesRegex(RuntimeError, 'trigger abort'):
          with pool.activate():
            raise RuntimeError('trigger abort')
        self.assertLess(safetensors_saver.time.monotonic() - t_abort, 0.3)
        self.assertFalse(pool.is_healthy())
        self.assertIsNotNone(pool._worker_err)
        self.assertIn('timed out', pool._worker_err or '')
        self.assertEqual(
            mock_terminate.call_args_list,
            [mock.call(p, timeout=0.1) for p in wedged_procs],
        )
        mock_del_direct.assert_called_once_with(
            ('gs://bkt/p0', 'gs://bkt/p1')
        )
        # Only one FLUSH was written per worker; no COMPOSE/FLUSH after timeout.
        for p in wedged_procs:
          writes = [call.args[0] for call in p.stdin.write.call_args_list]
          self.assertEqual(writes, ['FLUSH\n'])
        with self.assertRaisesRegex(RuntimeError, 'exited unexpectedly'):
          pool.acquire_slot()
      finally:
        pool.close(abort=True)

    # Pipe write error during flush(abort=True) also poisons & reaps worker.
    broken_proc = mock.MagicMock()
    broken_proc.stdout = io.StringIO('READY\n')
    broken_proc.stdin = mock.MagicMock()
    broken_proc.stdin.write.side_effect = BrokenPipeError('epipe')
    broken_proc.poll.return_value = None
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        mock.patch.object(
            safetensors_saver.subprocess, 'Popen', return_value=broken_proc
        ),
        mock.patch.object(
            safetensors_io, 'terminate_proc'
        ) as mock_terminate_pipe,
    ):
      pipe_pool = safetensors_saver.MultiprocStagePool(
          stage_dir=os.path.join(tmpdir, 'pipe_poison'),
          num_procs=1,
          shm_budget_bytes=4 << 20,
          max_bucket_bytes=128 << 10,
          num_large_slots=1,
      )
      try:
        pipe_pool._worker_err = None
        pipe_pool._dirty_workers.add(0)
        pipe_pool.flush(abort=True)
        self.assertFalse(pipe_pool.is_healthy())
        self.assertIn('BrokenPipeError', pipe_pool._worker_err or '')
        mock_terminate_pipe.assert_called_once_with(broken_proc, timeout=0.1)
      finally:
        pipe_pool.close(abort=True)

    # P1-2: Multi-host finalize_safetensors_metadata propagates asymmetric
    # backend.flush() errors before pre_index, enforces post_index barrier,
    # broadcasts host-0 index write errors, and does zero collectives on pc==1.
    dummy_plan = safetensors_saver.SavePlan(
        num_ranks=2,
        total_bytes=16,
        max_bucket_bytes=64,
        waves=(),
        rank_total_bytes={0: 8, 1: 8},
        weight_map={'w': 'model-00001-of-00002.safetensors'},
    )
    with tempfile.TemporaryDirectory() as tmpdir:
      # (a) Host 1 fails backend.flush(): participates in allgather(1) & raises.
      fail_be = dataclasses.replace(
          safetensors_saver._LOCAL_BACKEND,
          flush_fn=lambda: (_ for _ in ()).throw(OSError('gcs 503')),
      )
      ag_vals: list[int] = []
      with (
          mock.patch.object(jax, 'process_count', return_value=2),
          mock.patch.object(jax, 'process_index', return_value=1),
          mock.patch.object(
              safetensors_saver, 'resolve_storage_backend', return_value=fail_be
          ),
          mock.patch.object(
              multihost_utils,
              'process_allgather',
              side_effect=lambda x, tiled=False: (
                  ag_vals.append(int(x)) or np.array([0, int(x)], np.int32)
              ),
          ),
          mock.patch.object(
              multihost_utils, 'sync_global_devices'
          ) as mock_sync,
      ):
        with self.assertRaisesRegex(OSError, 'gcs 503'):
          safetensors_saver.finalize_safetensors_metadata(
              dummy_plan, tmpdir, None, 2
          )
      self.assertEqual(ag_vals, [1])
      mock_sync.assert_not_called()

      # (b) Host 0 succeeds backend.flush() while host 1 failed: raises before
      # pre_index and writes no index.json.
      with (
          mock.patch.object(jax, 'process_count', return_value=2),
          mock.patch.object(jax, 'process_index', return_value=0),
          mock.patch.object(
              multihost_utils,
              'process_allgather',
              return_value=np.array([0, 1], np.int32),
          ),
          mock.patch.object(
              multihost_utils, 'sync_global_devices'
          ) as mock_sync_h0,
      ):
        with self.assertRaisesRegex(RuntimeError, r'process\(es\) \[1\]'):
          safetensors_saver.finalize_safetensors_metadata(
              dummy_plan, tmpdir, None, 2
          )
      mock_sync_h0.assert_not_called()
      self.assertFalse(
          os.path.exists(os.path.join(tmpdir, 'model.safetensors.index.json'))
      )

      # (c) Happy path multi-host: Host 0 runs pre_index -> writes atomic
      # index.json -> post_index -> broadcast; Host 1 waits on both barriers.
      events_h0: list[str] = []
      real_replace = os.replace

      def _spy_replace(src, dst):
        events_h0.append(f'replace:{os.path.basename(src)}')
        return real_replace(src, dst)

      with (
          mock.patch.object(jax, 'process_count', return_value=2),
          mock.patch.object(jax, 'process_index', return_value=0),
          mock.patch.object(os, 'replace', side_effect=_spy_replace),
          mock.patch.object(
              multihost_utils,
              'process_allgather',
              return_value=np.array([0, 0], np.int32),
          ),
          mock.patch.object(
              multihost_utils,
              'sync_global_devices',
              side_effect=events_h0.append,
          ),
          mock.patch.object(
              multihost_utils,
              'broadcast_one_to_all',
              side_effect=lambda x, is_source: x,
          ),
      ):
        safetensors_saver.finalize_safetensors_metadata(
            dummy_plan, tmpdir, None, 2
        )
      self.assertEqual(
          events_h0,
          [
              'safetensors_saver:pre_index',
              'replace:model.safetensors.index.json.tmp',
              'safetensors_saver:post_index',
          ],
      )
      self.assertTrue(
          os.path.exists(os.path.join(tmpdir, 'model.safetensors.index.json'))
      )

      # (d) Host 0 fails writing index.json -> broadcasts error at post_index;
      # Host 1 raises RuntimeError instead of hanging or observing partial dir.
      err_dir = os.path.join(tmpdir, 'h0_err')
      os.makedirs(err_dir)
      h0_barriers: list[str] = []
      bcast_payloads: list[bytes] = []

      def _cap_bcast(x, is_source):
        self.assertTrue(is_source)
        if isinstance(x, np.ndarray):
          bcast_payloads.append(x.tobytes())
        return x

      with (
          mock.patch.object(jax, 'process_count', return_value=2),
          mock.patch.object(jax, 'process_index', return_value=0),
          mock.patch.object(os, 'replace', side_effect=OSError('disk full')),
          mock.patch.object(
              multihost_utils,
              'process_allgather',
              return_value=np.array([0, 0], np.int32),
          ),
          mock.patch.object(
              multihost_utils,
              'sync_global_devices',
              side_effect=h0_barriers.append,
          ),
          mock.patch.object(
              multihost_utils, 'broadcast_one_to_all', side_effect=_cap_bcast
          ),
      ):
        with self.assertRaisesRegex(OSError, 'disk full'):
          safetensors_saver.finalize_safetensors_metadata(
              dummy_plan, err_dir, None, 2
          )
      self.assertEqual(
          h0_barriers,
          ['safetensors_saver:pre_index', 'safetensors_saver:post_index'],
      )
      self.assertIn(
          'OSError: disk full', json.loads(bcast_payloads[-1])['error']
      )
      # Atomic .tmp file is cleaned up on replace failure.
      self.assertEqual(os.listdir(err_dir), [])

      h1_barriers: list[str] = []
      err_bytes = bcast_payloads[-1]
      err_buf = np.frombuffer(err_bytes, dtype=np.uint8)
      bcast_iter = iter([np.int32(err_buf.size), err_buf])
      with (
          mock.patch.object(jax, 'process_count', return_value=2),
          mock.patch.object(jax, 'process_index', return_value=1),
          mock.patch.object(
              multihost_utils,
              'process_allgather',
              return_value=np.array([0, 0], np.int32),
          ),
          mock.patch.object(
              multihost_utils,
              'sync_global_devices',
              side_effect=h1_barriers.append,
          ),
          mock.patch.object(
              multihost_utils,
              'broadcast_one_to_all',
              side_effect=lambda x, is_source: next(bcast_iter),
          ),
      ):
        with self.assertRaisesRegex(
            RuntimeError, 'failed on process 0: OSError: disk full'
        ):
          safetensors_saver.finalize_safetensors_metadata(
              dummy_plan, err_dir, None, 2
          )
      self.assertEqual(
          h1_barriers,
          ['safetensors_saver:pre_index', 'safetensors_saver:post_index'],
      )

      # (e) Pathways (process_count == 1): zero multihost collectives called.
      pw_dir = os.path.join(tmpdir, 'pathways_out')
      os.makedirs(pw_dir)
      with (
          mock.patch.object(jax, 'process_count', return_value=1),
          mock.patch.object(multihost_utils, 'process_allgather') as mock_ag,
          mock.patch.object(multihost_utils, 'sync_global_devices') as mock_sg,
          mock.patch.object(
              multihost_utils, 'broadcast_one_to_all'
          ) as mock_bc,
      ):
        safetensors_saver.finalize_safetensors_metadata(
            dummy_plan, pw_dir, None, 2
        )
      mock_ag.assert_not_called()
      mock_sg.assert_not_called()
      mock_bc.assert_not_called()

    # P1-3: Grouped MoE slice size uses actual array shape & output_dtype
    # rather than stale pre-cast spec.nbytes (float32 spec -> bf16 export).
    devs2 = jax.devices()[:2]
    mesh2 = jax.sharding.Mesh(np.array(devs2), ('rank',))
    shd2 = jax.sharding.NamedSharding(mesh2, jax.sharding.PartitionSpec('rank'))
    moe_f32_np = np.arange(2 * 32 * 64, dtype=np.float32).reshape(2, 32, 64)
    moe_f32 = jax.device_put(jnp.asarray(moe_f32_np), shd2)
    # Deliberately build spec with float32 nbytes (16384), NOT bfloat16 (8192).
    moe_spec_f32 = safetensors_saver.TensorExportSpec(
        source_key='moe',
        hf_key='layers.0.experts.{expert}.w',
        transpose_axes=(0, 2, 1),
        unstack_axis=0,
        nbytes=moe_f32_np.nbytes,
    )
    moe_plan = safetensors_saver.build_save_plan(
        [moe_spec_f32],
        num_ranks=2,
        max_bucket_bytes=moe_f32_np.nbytes // 2,
        unstack_counts={'moe': 2},
    )
    # Direct _group_batch_shards check with bf16 buffer + stale f32 spec.nbytes.
    moe_bf16 = jax.device_put(
        jnp.asarray(moe_f32_np, dtype=jnp.bfloat16).reshape(2, -1), shd2
    )
    c0, c1 = (
        moe_plan.waves[0].rank_assignments[0][0],
        moe_plan.waves[0].rank_assignments[1][0],
    )
    _, _, grp_owners, grp_items = safetensors_saver._group_batch_shards(
        [(0, c0), (1, c1)],
        [moe_bf16, moe_bf16],
        num_ranks=2,
        rank_shd=shd2,
        flat_state={'moe': moe_f32},
    )
    self.assertEqual(grp_owners, ((0, 1),))
    self.assertEqual(grp_items[0][0][2], (32 * 64,))
    self.assertEqual(grp_items[1][0][2], (32 * 64,))
    # E2E grouped export with mocked TPU backend (triggers fusable grouped path)
    # and output_dtype=jnp.bfloat16 produces valid, bit-exact shards & index.
    with (
        tempfile.TemporaryDirectory() as tmpdir,
        mock.patch.object(jax, 'default_backend', return_value='tpu'),
    ):
      compiled_moe = safetensors_saver.compile_save_plan(
          {'moe': moe_f32},
          moe_plan,
          output_dtype=jnp.bfloat16,
          devices=devs2,
      )
      metrics = safetensors_saver.export_safetensors_with_plan(
          {'moe': moe_f32},
          moe_plan,
          tmpdir,
          output_dtype=jnp.bfloat16,
          devices=devs2,
          compiled_plan=compiled_moe,
      )
      expected_bf16_bytes = moe_f32_np.size * 2
      self.assertEqual(metrics.total_bytes, expected_bf16_bytes)
      with open(os.path.join(tmpdir, 'model.safetensors.index.json')) as f:
        idx_json = json.load(f)
      self.assertEqual(idx_json['metadata']['total_size'], expected_bf16_bytes)
      loaded_moe = {}
      for sp in metrics.shard_paths:
        loaded_moe.update(safe_flax.load_file(sp))
      for e in range(2):
        expected_e = np.asarray(
            jnp.asarray(moe_f32_np[e].T, dtype=jnp.bfloat16)
        )
        np.testing.assert_array_equal(
            loaded_moe[f'layers.0.experts.{e}.w'], expected_e
        )
      # Reverse mismatch: bf16-sized spec exported with output_dtype=None on f32
      # still reports exact float32 bytes in SaveMetrics and index.json.
      bf16_plan = safetensors_saver.build_save_plan(
          [dataclasses.replace(moe_spec_f32, nbytes=moe_f32_np.size * 2)],
          num_ranks=2,
          max_bucket_bytes=moe_f32_np.size,
          unstack_counts={'moe': 2},
          rank_align_bytes=1024,
      )
      rev_dir = os.path.join(tmpdir, 'rev_f32')
      rev_metrics = safetensors_saver.export_safetensors_with_plan(
          {'moe': moe_f32},
          bf16_plan,
          rev_dir,
          output_dtype=None,
          devices=devs2,
      )
      self.assertEqual(rev_metrics.total_bytes, moe_f32_np.nbytes)
      with open(os.path.join(rev_dir, 'model.safetensors.index.json')) as f:
        self.assertEqual(
            json.load(f)['metadata']['total_size'], moe_f32_np.nbytes
        )

    # P2-2 & P2-3: Default large_slot_bytes reaches 4 GiB; infer_is_replicated
    # honors is_fully_replicated on abstract ShapeDtypeStruct.
    with mock.patch.object(
        safetensors_io, 'available_shm_bytes', return_value=256 << 30
    ):
      _, _, lg_b, n_lg = safetensors_saver.compute_shm_ring_geometry()
    self.assertEqual(lg_b, safetensors_saver.MAX_SAFETENSORS_SHARD_BYTES)
    self.assertEqual(n_lg, 2)

    class _FakeRepSharding(jax.sharding.Sharding):

      @property
      def device_set(self):
        return set(devs2)

      @property
      def memory_kind(self):
        return None

      @property
      def is_fully_replicated(self) -> bool:
        return True

      def devices_indices_map(self, global_shape):
        del global_shape
        return {}

      def shard_shape(self, global_shape):
        return global_shape

      def is_equivalent_to(self, other, ndim):
        del other, ndim
        return False

      def with_memory_kind(self, kind):
        del kind
        return self

      def check_compatible_aval(self, aval_shape):
        del aval_shape

    sds_rep = jax.ShapeDtypeStruct(
        (8, 8), jnp.float32, sharding=_FakeRepSharding()
    )
    self.assertTrue(safetensors_saver.infer_is_replicated(sds_rep))

  def test_rapid_grpc_pool_lpt_oversized_waves_flush_budget_and_sidecar(self):
    # 1. Oversized waves (> limit) are hoisted to Wave 0.. in descending LPT
    #    order so large dense tensors (embed_tokens / lm_head) start first.
    moe_spec = safetensors_saver.TensorExportSpec(
        'moe.w', 'layers.0.e.{expert}', unstack_axis=0, nbytes=4 * 4096
    )
    big_a = safetensors_saver.TensorExportSpec('embed', 'embed', nbytes=12288)
    big_b = safetensors_saver.TensorExportSpec('lm', 'lm', nbytes=8192)
    small = safetensors_saver.TensorExportSpec('norm', 'norm', nbytes=1024)
    lpt_plan = safetensors_saver.build_save_plan(
        [moe_spec, big_a, big_b, small],
        num_ranks=2,
        max_bucket_bytes=4096,
        unstack_counts={'moe.w': 4},
        per_wave_shards=True,
    )
    def _wave_keys(wave) -> set[str]:
      keys: set[str] = set()
      for specs in wave.rank_assignments.values():
        keys.update(s.source_key for s in specs)
      return keys

    w0_keys = _wave_keys(lpt_plan.waves[0])
    w1_keys = _wave_keys(lpt_plan.waves[1])
    self.assertIn('embed', w0_keys)
    self.assertTrue('lm' in w0_keys or 'lm' in w1_keys)
    self.assertGreater(max(lpt_plan.waves[0].rank_bytes.values()), 4096)
    # Single-rank plan guarantees separate oversized waves in descending LPT.
    lpt_1r = safetensors_saver.build_save_plan(
        [small, big_b, big_a],
        num_ranks=1,
        max_bucket_bytes=4096,
        per_wave_shards=True,
    )
    self.assertIn('embed', _wave_keys(lpt_1r.waves[0]))
    self.assertIn('lm', _wave_keys(lpt_1r.waves[1]))
    self.assertIn('norm', _wave_keys(lpt_1r.waves[2]))
    # Counterfactual: when no wave exceeds limit, original order is preserved.
    small_a = safetensors_saver.TensorExportSpec('sa', 'sa', nbytes=2048)
    small_b = safetensors_saver.TensorExportSpec('sb', 'sb', nbytes=2048)
    no_hoist = safetensors_saver.build_save_plan(
        [small_a, small_b], num_ranks=1, max_bucket_bytes=2048
    )
    self.assertEqual([w.wave_index for w in no_hoist.waves], [0, 1])

    # 2. submit_and_drain_wave_flush preserves 4-wave overlap for large waves
    #    and hoisted oversized Wave 0 + smaller Wave 1, while capping at 16 GiB.
    wave_2g = safetensors_saver.WaveBucket(0, {}, {0: 2 << 30})
    f1: concurrent.futures.Future[float] = concurrent.futures.Future()
    f2: concurrent.futures.Future[float] = concurrent.futures.Future()
    f3: concurrent.futures.Future[float] = concurrent.futures.Future()
    pending_4w = [(f1, 2 << 30), (f2, 2 << 30), (f3, 2 << 30)]
    with mock.patch.object(
        safetensors_saver, 'flush_wave_to_files', return_value=0.0
    ):
      safetensors_saver.submit_and_drain_wave_flush(
          pending_4w, 4, {}, {0: {'w': np.zeros(1)}}, wave_2g, 0, 1
      )
      self.assertLen(pending_4w, 4)
      self.assertIs(pending_4w[0][0], f1)
      # Hoisted 5 GiB Wave 0 + 1 GiB Wave 1 overlaps without blocking on Wave 0.
      f_big: concurrent.futures.Future[float] = concurrent.futures.Future()
      pending_hoist = [(f_big, 5 << 30)]
      wave_1g = safetensors_saver.WaveBucket(1, {}, {0: 1 << 30})
      safetensors_saver.submit_and_drain_wave_flush(
          pending_hoist, 4, {}, {0: {'w': np.zeros(1)}}, wave_1g, 0, 1
      )
      self.assertLen(pending_hoist, 2)
      self.assertIs(pending_hoist[0][0], f_big)
      # Drains when exceeding 16 GiB cap.
      done_cap: concurrent.futures.Future[float] = concurrent.futures.Future()
      done_cap.set_result(0.0)
      pending_cap = [(done_cap, 12 << 30)]
      wave_5g = safetensors_saver.WaveBucket(1, {}, {0: 5 << 30})
      safetensors_saver.submit_and_drain_wave_flush(
          pending_cap, 4, {}, {0: {'w': np.zeros(1)}}, wave_5g, 0, 1
      )
      self.assertLen(pending_cap, 1)
      self.assertIsNot(pending_cap[0][0], done_cap)
      pending_4w[-1][0].result()
      pending_hoist[-1][0].result()
      pending_cap[0][0].result()

    # 3. MultiprocStagePool(use_gcs_rapid_grpc=True) sets gcs_rapid_grpc,
    #    disables compose & >160 MiB striping, and sends direct SLOT commands.
    spawned_envs: list[dict[str, str]] = []
    sent_lines: list[str] = []

    class _FakePipe:

      def __init__(self, q: queue.Queue[str]):
        self._q = q

      def readline(self) -> str:
        return self._q.get()

      def __iter__(self):
        return iter(self._q.get, '')

      def close(self) -> None:
        self._q.put('')

    class _FakeRapidProc:

      def __init__(self, env):
        self.env = env
        self._out: queue.Queue[str] = queue.Queue()
        self._out.put('READY\n')
        self.stdin = mock.MagicMock()
        self.stdin.write.side_effect = self._on_write
        self.stdout = _FakePipe(self._out)
        self._poll = None

      def _on_write(self, line: str) -> int:
        sent_lines.append(line)
        if line.startswith('FLUSH'):
          self._out.put('DONE\n')
        elif line.startswith('QUIT'):
          self._out.put('')
        return len(line)

      def poll(self):
        return self._poll

      def wait(self, timeout=None):
        del timeout
        self._poll = 0
        return 0

      def kill(self):
        self._poll = -9

    def _fake_popen(cmd, **kwargs):
      del cmd
      spawned_envs.append(dict(kwargs['env']))
      return _FakeRapidProc(kwargs['env'])

    with (
        mock.patch.object(
            safetensors_saver.subprocess, 'Popen', side_effect=_fake_popen
        ),
        mock.patch.object(
            safetensors_io, 'resolve_gcs_vips', return_value=('10.1.0.1',)
        ),
    ):
      rpool = safetensors_saver.MultiprocStagePool(
          stage_dir=os.path.join(self.create_tempdir().full_path, 'rstage'),
          num_procs=1,
          shm_budget_bytes=4 << 20,
          max_bucket_bytes=1 << 20,
          use_gcs_http_write=True,
          use_gcs_rapid_grpc=True,
      )
      try:
        self.assertEqual(
            spawned_envs[0]['TUNIX_STAGE_BACKEND'], 'gcs_rapid_grpc'
        )
        self.assertEqual(spawned_envs[0]['TUNIX_GCS_VIPS'], '10.1.0.1')
        with rpool.activate():
          be = safetensors_saver.resolve_storage_backend('gs://rapid/ckpt')
          self.assertIsNone(be.register_composite_fn)
        with mock.patch.object(
            rpool,
            '_enqueue_gcs_stripes',
            side_effect=AssertionError('no stripe'),
        ):
          rpool.enqueue_slot(
              0,
              safetensors_saver._GCS_STRIPE_THRESHOLD_BYTES + 64,
              'gs://rapid/ckpt/big.safetensors',
          )
        self.assertTrue(
            any(
                line.startswith('SLOT\t') and 'big.safetensors' in line
                for line in sent_lines
            )
        )
      finally:
        rpool.close()

    # 4. export_safetensors_with_plan writes .tunix_headers.json matching actual
    #    wire headers & file sizes across all 3 shard layouts, leaves HF
    #    model.safetensors.index.json untouched, and copy_non_safetensors_files
    #    never clobbers .tunix_headers.json.
    flat = {
        'a': jnp.arange(16, dtype=jnp.float32).reshape(4, 4),
        'b': jnp.arange(16, dtype=jnp.float32).reshape(4, 4) + 10.0,
    }
    specs = [
        safetensors_saver.make_tensor_export_spec(flat, k, k)
        for k in flat
    ]
    layouts = [
        dict(
            num_ranks=1,
            max_bucket_bytes=None,
            per_wave_shards=False,
            max_shard_bytes=None,
        ),
        dict(
            num_ranks=2,
            max_bucket_bytes=64,
            per_wave_shards=True,
            max_shard_bytes=None,
        ),
        dict(
            num_ranks=2,
            max_bucket_bytes=64,
            per_wave_shards=False,
            max_shard_bytes=128,
        ),
    ]
    for cfg in layouts:
      out_dir = self.create_tempdir().full_path
      plan = safetensors_saver.build_save_plan(specs, **cfg)
      safetensors_saver.export_safetensors_with_plan(flat, plan, out_dir)
      sc_path = os.path.join(out_dir, safetensors_io.SIDECAR_HEADERS_FILENAME)
      self.assertTrue(os.path.isfile(sc_path))
      with open(sc_path, 'r', encoding='utf-8') as fh:
        sidecar = json.load(fh)
      self.assertEqual(sidecar['version'], 1)
      st_files = sorted(
          f for f in os.listdir(out_dir) if f.endswith('.safetensors')
      )
      self.assertEqual(sorted(sidecar['shards']), st_files)
      srcs = safetensors_io.build_range_sources_from_names(out_dir, st_files)
      for fname, src in zip(st_files, srcs):
        fpath = os.path.join(out_dir, fname)
        actual_len, actual_hdr = safetensors_io.read_safetensors_header(src)
        actual_tensors = {
            k: v for k, v in actual_hdr.items() if k != '__metadata__'
        }
        entry = sidecar['shards'][fname]
        self.assertEqual(entry['header_len'], actual_len)
        self.assertEqual(entry['file_size'], os.path.getsize(fpath))
        self.assertEqual(entry['tensors'], actual_tensors)
      idx_path = os.path.join(out_dir, 'model.safetensors.index.json')
      if len(st_files) > 1:
        with open(idx_path, 'r', encoding='utf-8') as fh:
          idx_json = json.load(fh)
        self.assertEqual(set(idx_json.keys()), {'metadata', 'weight_map'})

    src_meta = self.create_tempdir().full_path
    dst_meta = self.create_tempdir().full_path
    with open(os.path.join(src_meta, 'config.json'), 'w') as fh:
      fh.write('{}')
    with open(
        os.path.join(src_meta, safetensors_io.SIDECAR_HEADERS_FILENAME), 'w'
    ) as fh:
      fh.write('{"stale": true}')
    safetensors_saver.copy_non_safetensors_files(src_meta, dst_meta)
    self.assertTrue(os.path.isfile(os.path.join(dst_meta, 'config.json')))
    self.assertFalse(
        os.path.exists(
            os.path.join(dst_meta, safetensors_io.SIDECAR_HEADERS_FILENAME)
        )
    )

  def test_stage_pool_live_inflight_worker_rebalancing(self):
    # Verify MultiprocStagePool tracks live inflight bytes per worker:
    # _pick_worker routes around busy workers, _watch_stdout decrements on
    # FREE\t<slot_id> (including multi-stripe same-slot dispatches), failed
    # _send_slot_cmd rolls back inflight bytes, and flush() clears state.
    class _FakePipe:

      def __init__(self, q: queue.Queue[str]):
        self._q = q

      def readline(self) -> str:
        return self._q.get()

      def __iter__(self):
        return iter(self._q.get, '')

      def close(self) -> None:
        self._q.put('')

    procs: list[Any] = []

    class _ControllableProc:

      def __init__(self):
        self.out_q: queue.Queue[str] = queue.Queue()
        self.out_q.put('READY\n')
        self.cmds: list[str] = []
        self.fail_next_slot = False
        self.stdin = mock.MagicMock()
        self.stdin.write.side_effect = self._on_write
        self.stdout = _FakePipe(self.out_q)
        self._poll = None

      def _on_write(self, line: str) -> int:
        if line.startswith('SLOT\t') and self.fail_next_slot:
          self.fail_next_slot = False
          raise BrokenPipeError('simulated stdin failure')
        self.cmds.append(line)
        if line.startswith('FLUSH'):
          self.out_q.put('DONE\n')
        elif line.startswith('QUIT'):
          self.out_q.put('')
        return len(line)

      def poll(self):
        return self._poll

      def wait(self, timeout=None):
        del timeout
        self._poll = 0
        return 0

      def kill(self):
        self._poll = -9

    def _fake_popen(cmd, **kwargs):
      del cmd, kwargs
      p = _ControllableProc()
      procs.append(p)
      return p

    def _sync_worker(pool, w_idx: int) -> None:
      procs[w_idx].out_q.put('SYNC\n')
      self.assertEqual(pool._flush_replies[w_idx].get(timeout=5.0), 'SYNC')

    with (
        mock.patch.object(
            safetensors_saver.subprocess, 'Popen', side_effect=_fake_popen
        ),
        mock.patch.object(
            safetensors_io, 'resolve_gcs_vips', return_value=('10.1.0.1',)
        ),
    ):
      pool = safetensors_saver.MultiprocStagePool(
          stage_dir=os.path.join(self.create_tempdir().full_path, 'lb_stage'),
          num_procs=2,
          shm_budget_bytes=8 << 20,
          max_bucket_bytes=512 << 10,
          num_large_slots=0,
          use_gcs_http_write=True,
          use_gcs_rapid_grpc=True,
      )
      try:
        self.assertGreaterEqual(pool.num_reg_slots, 4)
        # 1. Worker 0 gets a heavy 2000B slot; next two 500B slots route to w1.
        s0 = pool.acquire_slot()
        s1 = pool.acquire_slot()
        s2 = pool.acquire_slot()
        pool.enqueue_slot(s0, 2000, 'gs://bkt/heavy')
        pool.enqueue_slot(s1, 500, 'gs://bkt/small1')
        pool.enqueue_slot(s2, 500, 'gs://bkt/small2')
        self.assertEqual(pool._worker_bytes, [2000, 1000])
        self.assertLen(procs[0].cmds, 1)
        self.assertLen(procs[1].cmds, 2)

        # 2. Worker 0 completes s0 via FREE\t<slot_id> (zero wire change).
        #    Live inflight bytes drop to [0, 1000], so next slot routes to w0
        #    instead of starving w0 under cumulative-only accounting.
        procs[0].out_q.put(f'FREE\t{s0}\n')
        _sync_worker(pool, 0)
        self.assertEqual(pool._worker_bytes, [0, 1000])
        self.assertEqual(pool._worker_inflight[0], {})

        s3 = pool.acquire_slot()
        pool.enqueue_slot(s3, 400, 'gs://bkt/rebalanced')
        self.assertEqual(pool._worker_bytes, [400, 1000])
        self.assertIn('rebalanced', procs[0].cmds[-1])

        # 3. Same slot_id dispatched twice to worker 0 (multi-stripe case)
        #    decrements exact FIFO bytes on each FREE without clobbering.
        procs[0].out_q.put(f'FREE\t{s3}\n')
        _sync_worker(pool, 0)
        self.assertEqual(pool._worker_bytes, [0, 1000])
        with pool._lock:
          pool._slot_refs[s3] = 2
        pool._send_slot_cmd(0, 600, s3, 'gs://bkt/st0')
        pool._send_slot_cmd(600, 300, s3, 'gs://bkt/st1')
        self.assertEqual(pool._worker_bytes, [900, 1000])
        procs[0].out_q.put(f'FREE\t{s3}\n')
        _sync_worker(pool, 0)
        self.assertEqual(pool._worker_bytes, [300, 1000])
        procs[0].out_q.put(f'FREE\t{s3}\n')
        _sync_worker(pool, 0)
        self.assertEqual(pool._worker_bytes, [0, 1000])
        self.assertNotIn(s3, pool._worker_inflight[0])

        # 4. _send_slot_cmd write error decrements live inflight bytes cleanly.
        procs[0].fail_next_slot = True
        with self.assertRaises(BrokenPipeError):
          pool._send_slot_cmd(0, 777, s3, 'gs://bkt/broken')
        self.assertEqual(pool._worker_bytes, [0, 1000])
        self.assertNotIn(s3, pool._worker_inflight[0])

        # 5. flush() clears remaining inflight bytes/maps, and a stale FREE
        #    afterward is a safe no-op without underflowing negative.
        pool.flush()
        self.assertEqual(pool._worker_bytes, [0, 0])
        self.assertEqual(pool._worker_inflight, [{}, {}])
        procs[1].out_q.put(f'FREE\t{s1}\n')
        _sync_worker(pool, 1)
        self.assertEqual(pool._worker_bytes, [0, 0])
      finally:
        pool.close()


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

  def test_sidecar_flushed_before_index_json_on_staged_backend(self):
    events: list[tuple[str, str]] = []

    class _RecWriter:

      def __init__(self, path: str):
        self._path = path

      def write(self, data):
        del data
        return 0

      def __enter__(self):
        events.append(('open', os.path.basename(self._path)))
        return self

      def __exit__(self, *_exc):
        return None

    backend = safetensors_saver.StorageBackend(
        kind='gcs',
        default_per_wave_shards=True,
        default_write_buffer_bytes=0,
        requires_bytes_chunks=False,
        stage_writer_fn=_RecWriter,
        flush_fn=lambda: events.append(('flush', '')),
    )
    plan = safetensors_saver.SavePlan(
        waves=(),
        weight_map={'w': 'model-00001-of-00002.safetensors'},
        total_bytes=16,
        num_ranks=2,
        max_bucket_bytes=16,
        rank_total_bytes={0: 16},
    )
    shard_headers = {
        'model-00001-of-00002.safetensors': {
            'header_len': 32,
            'file_size': 56,
            'tensors': {
                'w': {'dtype': 'F32', 'shape': [4], 'data_offsets': [0, 16]}
            },
        }
    }
    safetensors_saver._write_index_and_metadata(
        backend,
        plan,
        'gs://bkt/ckpt',
        local_model_path=None,
        num_active_ranks=2,
        shard_headers=shard_headers,
    )
    self.assertEqual(
        events,
        [
            ('open', safetensors_io.SIDECAR_HEADERS_FILENAME),
            ('flush', ''),
            ('open', 'model.safetensors.index.json'),
            ('flush', ''),
        ],
    )


if __name__ == '__main__':
  absltest.main()
