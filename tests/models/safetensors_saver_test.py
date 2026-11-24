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

"""Round-trip, commit-marker, manifest, and barrier tests for the saver."""

import json
import os
from unittest import mock

from absl.testing import absltest
from etils import epath
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
from safetensors import numpy as safe_np
from tunix.models import safetensors_saver


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
