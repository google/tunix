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

import errno
import fcntl
import io
import json
import mmap
import os
import pathlib
import socket
import struct
import subprocess
import sys
import threading
import time
import types
from unittest import mock

from absl.testing import absltest
from etils import epath
import ml_dtypes
import numpy as np
from tunix.models import safetensors_io


class _FakeProcPath(safetensors_io.ProcPoolPath):
  """`ProcPoolPath` with its own reader env var and client flag argv."""

  proc_env_var = 'TUNIX_FAKE_PROCS'

  @classmethod
  def configure_flags(cls) -> tuple[str, ...]:
    return ('--fake_flag=1',)


def _fake_storage(**overrides) -> safetensors_io.StorageBackendSpec:
  """Returns a POSIX-backed spec standing in for a registered remote FS."""
  return safetensors_io.StorageBackendSpec(**{
      'path_cls': _FakeProcPath,
      'validate': lambda path: None,
      'write_flags': lambda: (),
      'open_stage_file': lambda path: open(path, 'wb'),
      **overrides,
  })


class SafetensorsIoTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir().full_path
    safetensors_io.READ_PROCS.clear()
    safetensors_io.THREAD_CONNS.clear()
    safetensors_io.GCS_TOKEN_CACHE.clear()
    safetensors_io.GCS_RAPID_BUCKET_CACHE.clear()
    safetensors_io.GCS_SOURCE_IPS_CACHE.clear()

  def test_dtype_codec_and_header_wire_format(self):
    self.assertEqual(safetensors_io.to_np_dtype('BF16'), ml_dtypes.bfloat16)
    self.assertEqual(safetensors_io.to_np_dtype('F16'), np.float16)
    self.assertEqual(safetensors_io.to_np_dtype('F32'), np.float32)
    self.assertEqual(safetensors_io.to_np_dtype('F64'), np.float64)
    self.assertEqual(safetensors_io.to_np_dtype('I32'), np.dtype(np.int32))
    self.assertEqual(
        safetensors_io.to_np_dtype(ml_dtypes.bfloat16), ml_dtypes.bfloat16
    )
    self.assertEqual(safetensors_io.to_np_dtype(np.float16), np.float16)
    self.assertEqual(safetensors_io.to_np_dtype(np.float32), np.float32)
    self.assertEqual(safetensors_io.to_np_dtype(np.float64), np.float64)
    self.assertEqual(safetensors_io.to_np_dtype(np.int32), np.int32)
    self.assertEqual(safetensors_io.from_np_dtype(np.float32), 'F32')
    self.assertEqual(safetensors_io.from_np_dtype(ml_dtypes.bfloat16), 'BF16')

    hdr_bytes = safetensors_io.format_safetensors_header(
        {'w': {'dtype': 'F32', 'shape': (2, 2), 'data_offsets': (0, 16)}},
        metadata={'format': 'pt'},
    )
    self.assertEqual(len(hdr_bytes) % 8, 0)
    shard_path = pathlib.Path(self.temp_dir) / 'model.safetensors'
    shard_path.write_bytes(hdr_bytes + b'\x00' * 16)
    size, parsed = safetensors_io.read_safetensors_header(shard_path)
    self.assertEqual(size, len(hdr_bytes) - 8)
    self.assertEqual(parsed['__metadata__']['format'], 'pt')
    self.assertEqual(parsed['w']['data_offsets'], [0, 16])

    bad_short = pathlib.Path(self.temp_dir) / 'short.safetensors'
    bad_short.write_bytes(b'123')
    with self.assertRaisesRegex(OSError, 'Truncated safetensors header'):
      safetensors_io.read_safetensors_header(bad_short)

    bad_body = pathlib.Path(self.temp_dir) / 'bad_body.safetensors'
    bad_body.write_bytes((32).to_bytes(8, 'little') + b'{}')
    with self.assertRaisesRegex(OSError, 'Truncated safetensors metadata'):
      safetensors_io.read_safetensors_header(bad_body)

  def test_cached_range_file_and_range_source_read_write(self):
    raw_file = pathlib.Path(self.temp_dir) / 'data.safetensors'
    payload = b'0123456789abcdef' * 4
    raw_file.write_bytes(payload)
    src = safetensors_io.RangeSource(epath.Path(raw_file))
    self.assertEqual(os.fspath(src), str(raw_file))
    self.assertEqual(str(src), str(raw_file))
    self.assertEqual(src.ensure_procs(128 * 1024 * 1024), 0)

    with src.open('rb') as cf:
      self.assertEqual(cf.read(0), b'')
      self.assertEqual(cf.read(4), b'0123')
      cf.seek(4, io.SEEK_CUR)
      self.assertEqual(cf.fetch_bytes(8, 4), b'89ab')

    src.use_hdr_cache = True
    with src.open('rb') as cf:
      self.assertEqual(cf.read(8), b'01234567')
      self.assertIsNotNone(src.hdr_cache)
      self.assertEqual(cf.read(8), b'89abcdef')
      cf.seek(len(payload) - 4)
      self.assertEqual(cf.read(8), b'cdef')
      cf.done_flags = bytearray([1])
      cf.task_idx = 0
      buf = np.zeros(8, dtype=np.uint8)
      self.assertEqual(cf.readinto(buf), 0)
      self.assertEqual(cf.read_shm_range(0, 8, buf, 0), 0)
      cf.done_flags = bytearray([0])
      self.assertEqual(cf.read_shm_range(0, 8, buf, 0), 8)
      self.assertEqual(bytes(buf), b'01234567')

  def test_proc_pool_path_flags_and_worker_read_protocol(self):
    self.assertEqual(safetensors_io.ProcPoolPath.configure_flags(), ())
    self.assertEqual(_FakeProcPath.configure_flags(), ('--fake_flag=1',))

    raw_file = pathlib.Path(self.temp_dir) / 'pool_shard.safetensors'
    raw_file.write_bytes(b'ABCDEFGHIJKLMNOP')
    with safetensors_io.ShmArena(size_bytes=65536) as arena:
      shm_mmaps = {}
      handles = {}
      lock = threading.Lock()
      out_buf = io.StringIO()
      read_cmd = f'{arena.shm_path}\t{arena.capacity}\t{raw_file}\t4\t8\t128\t7'
      with mock.patch.object(sys, 'stdout', out_buf):
        safetensors_io.ProcPoolPath.run_worker_cmd(
            read_cmd, shm_mmaps, handles, lock
        )
      for fh in handles.values():
        fh.close()
      safetensors_io.ProcPoolPath._close_shm_mmaps(shm_mmaps)
      self.assertEqual(out_buf.getvalue(), '7\t8\n')
      assert arena.shm_uint8 is not None
      self.assertEqual(bytes(arena.shm_uint8[128:136]), b'EFGHIJKL')

  def test_proc_pool_worker_loop_and_proc_multiplexing(self):
    with (
        mock.patch.object(sys, 'stdin', io.StringIO('\n')),
        mock.patch.object(sys, 'stdout', io.StringIO()) as fake_out,
    ):
      safetensors_io.ProcPoolPath.worker_loop()
      self.assertIn('READY', fake_out.getvalue())

    pool_src = _FakeProcPath(epath.Path(self.temp_dir) / 'a.safetensors')
    self.assertEqual(
        safetensors_io.ProcPoolPath(pool_src.raw_path).ensure_procs(1 << 30), 0
    )
    with mock.patch.object(_FakeProcPath, 'spawn_procs') as mock_spawn:
      with mock.patch.dict(os.environ, {'TUNIX_FAKE_PROCS': '4'}):
        pool_src.ensure_procs(1024)
      mock_spawn.assert_called_with(4)
      with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop('TUNIX_FAKE_PROCS', None)
        pool_src.ensure_procs(128 * 1024 * 1024)
        mock_spawn.assert_called_with(2)
        pool_src.ensure_procs(1024)
        mock_spawn.assert_called_with(0)

    fake_proc = mock.MagicMock()
    fake_proc.stdin = io.StringIO()
    fake_proc.stdout.readline.side_effect = [
        'READY\n',
        '1\t8\n',
    ]
    with (
        mock.patch.object(
            subprocess, 'Popen', return_value=fake_proc
        ) as mock_popen,
        mock.patch.object(safetensors_io, '_g3_mp_lib', None),
        mock.patch.object(sys, 'executable', '/app/trainer.par'),
    ):
      _FakeProcPath.spawn_procs(1)
      self.assertLen(safetensors_io.READ_PROCS, 1)
      self.assertIn('--fake_flag=1', mock_popen.call_args.args[0])
      self.assertIn(
          f'from {__name__} import _FakeProcPath as p; p.worker_loop()',
          mock_popen.call_args.args[0],
      )
      cf = pool_src.open('rb', stream_id=0)
      with mock.patch.object(safetensors_io, '_REQ_COUNTER', iter([1])):
        buf = np.zeros(16, dtype=np.uint8)
        self.assertEqual(cf.read_shm_range(0, 8, buf, 0, '/dev/shm/x', 64), 8)
      _FakeProcPath.spawn_procs(0)
      self.assertEmpty(safetensors_io.READ_PROCS)

    bad_proc = mock.MagicMock()
    bad_proc.stdout.readline.return_value = 'FAIL\n'
    with mock.patch.object(subprocess, 'Popen', return_value=bad_proc):
      with self.assertRaisesRegex(OSError, 'failed to start'):
        _FakeProcPath.spawn_procs(1)

  def test_storage_backend_registry_and_range_source_dispatch(self):
    root = os.path.join(self.temp_dir, 'remote_fs')
    os.makedirs(os.path.join(root, 'deep', 'ckpt'))
    for name in ('a.safetensors', 'b.safetensors'):
      pathlib.Path(root, 'deep', 'ckpt', name).write_bytes(b'12345678')
    outer = _fake_storage()
    inner = _fake_storage(path_cls=safetensors_io.ProcPoolPath)
    with mock.patch.dict(safetensors_io.STORAGE_BACKENDS, clear=True):
      with self.assertRaisesRegex(ValueError, 'must end with'):
        safetensors_io.register_storage_backend(root, outer)
      safetensors_io.register_storage_backend(f'{root}/', outer)
      safetensors_io.register_storage_backend(f'{root}/deep/', inner)
      self.assertIs(safetensors_io.find_storage_backend(root), outer)
      self.assertIs(safetensors_io.find_storage_backend(f'{root}/x'), outer)
      self.assertIs(
          safetensors_io.find_storage_backend(f'{root}/deep/ckpt'), inner
      )
      self.assertIsNone(safetensors_io.find_storage_backend(f'{root}x/y'))
      self.assertIsNone(safetensors_io.find_storage_backend('gs://b/k'))
      self.assertCountEqual(
          safetensors_io.STORAGE_BACKENDS.values(), (outer, inner)
      )

      safetensors_io.register_storage_backend(f'{root}/deep/', outer)
      with (
          mock.patch.object(_FakeProcPath, 'spawn_procs') as mock_spawn,
          mock.patch.dict(os.environ, {'TUNIX_FAKE_PROCS': '3'}),
      ):
        sources = safetensors_io.resolve_range_sources(f'{root}/deep/ckpt')
        named = safetensors_io.build_range_sources_from_names(
            f'{root}/deep/ckpt', ['b.safetensors']
        )
      self.assertEqual(mock_spawn.call_args_list, [mock.call(3)] * 2)
      self.assertEqual([type(s) for s in sources], [_FakeProcPath] * 2)
      self.assertEqual([s.name for s in named], ['b.safetensors'])
      with sources[0].open('rb') as cf:
        self.assertEqual(cf.read(8), b'12345678')
    self.assertIsNone(safetensors_io.find_storage_backend(f'{root}/x'))

  def test_gs_http_path_token_vips_and_range_transport(self):
    with mock.patch.dict(os.environ, {'TUNIX_GCS_TOKEN': 'tok-env'}):
      self.assertEqual(safetensors_io.GsHttpPath.resolve_token(), 'tok-env')

    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop('TUNIX_GCS_TOKEN', None)
      os.environ.pop('TOKEN', None)
      fake_conn = mock.MagicMock()
      fake_resp = mock.MagicMock(status=200)
      fake_resp.read.return_value = b'{"access_token": "tok-meta"}'
      fake_conn.getresponse.return_value = fake_resp
      with mock.patch.object(
          safetensors_io.http.client, 'HTTPConnection', return_value=fake_conn
      ):
        self.assertEqual(safetensors_io.GsHttpPath.resolve_token(), 'tok-meta')
        self.assertEqual(safetensors_io.GsHttpPath.resolve_token(), 'tok-meta')

    self.assertIn('storage-upload.googleapis.com', safetensors_io.GCS_HOSTS)
    self.assertIn('private.googleapis.com', safetensors_io.GCS_HOSTS)
    with mock.patch.object(
        safetensors_io.socket,
        'getaddrinfo',
        return_value=[(2, 1, 6, '', ('142.250.1.1', 443))],
    ):
      self.assertEqual(
          safetensors_io.GsHttpPath.discover_vips(), ['142.250.1.1']
      )

    gs_file = safetensors_io.GsHttpPath(
        'gs://my-bucket/dir/shard.safetensors', 'tok', ('142.250.1.1',)
    ).open('rb')
    self.assertIsInstance(gs_file, safetensors_io.GsHttpFile)

    resp_ok = mock.MagicMock(status=206)
    resp_ok.readinto.side_effect = [8, 8, 0]
    fake_https = mock.MagicMock()
    fake_https.getresponse.return_value = resp_ok
    fake_sock = mock.MagicMock()
    with (
        mock.patch.object(
            safetensors_io.http.client,
            'HTTPSConnection',
            return_value=fake_https,
        ),
        mock.patch.object(
            safetensors_io.socket,
            'create_connection',
            return_value=fake_sock,
        ),
        mock.patch.object(safetensors_io.ssl, 'create_default_context'),
    ):
      mv = memoryview(bytearray(8))
      self.assertEqual(gs_file.fetch_range(0, 8, mv), 8)
      self.assertEqual(gs_file.fetch_bytes(0, 8), b'\x00' * 8)
    fake_sock.setsockopt.assert_called_once_with(
        safetensors_io.socket.IPPROTO_TCP,
        safetensors_io.socket.TCP_NODELAY,
        1,
    )

  def test_resolve_range_sources_and_shm_arena(self):
    empty_dir = pathlib.Path(self.temp_dir) / 'empty'
    empty_dir.mkdir()
    with self.assertRaisesRegex(ValueError, 'No safetensors found'):
      safetensors_io.resolve_range_sources(str(empty_dir))

    shard = pathlib.Path(self.temp_dir) / 's.safetensors'
    shard.write_bytes(b'12345678')
    sources = safetensors_io.resolve_range_sources(self.temp_dir)
    self.assertLen(sources, 1)
    self.assertIsInstance(sources[0], safetensors_io.RangeSource)

    with safetensors_io.ShmArena(shm_budget_bytes=8192) as arena:
      arr = arena.alloc(128)
      self.assertEqual(arr.shape, (128,))
      self.assertEqual(arena.alloc(0).shape, (0,))
      fallback = arena.alloc(8192)
      self.assertEqual(fallback.shape, (8192,))
      arena.ring_capacity = 4096
      arena.unit_shm_bytes = [3000, 3000]
      arena.admit_units()
      self.assertEqual(arena.next_admit_unit, 1)
      arena.release_unit(0)
      self.assertEqual(arena.next_admit_unit, 2)
      arena.release_file(0)

  def test_token_refresh_lock_no_empty_cache_and_401_retry(self):
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop('TUNIX_GCS_TOKEN', None)
      os.environ.pop('TOKEN', None)
      safetensors_io.GCS_TOKEN_CACHE.clear()
      with mock.patch.object(
          safetensors_io.http.client,
          'HTTPConnection',
          side_effect=OSError('timeout'),
      ):
        self.assertEqual(safetensors_io.GsHttpPath.resolve_token(), '')
        self.assertEmpty(safetensors_io.GCS_TOKEN_CACHE)

      resp_ok = mock.MagicMock(status=200)
      resp_ok.read.return_value = (
          b'{"access_token": "tok-1", "expires_in": 600}'
      )
      conn_ok = mock.MagicMock()
      conn_ok.getresponse.return_value = resp_ok
      with mock.patch.object(
          safetensors_io.http.client, 'HTTPConnection', return_value=conn_ok
      ):
        self.assertEqual(safetensors_io.GsHttpPath.resolve_token(), 'tok-1')
      with mock.patch.object(
          safetensors_io.http.client,
          'HTTPConnection',
          side_effect=OSError('transient'),
      ):
        self.assertEqual(
            safetensors_io.GsHttpPath.resolve_token(force_refresh=True),
            'tok-1',
        )

    gs_file = safetensors_io.GsHttpPath(
        'gs://bkt/obj.safetensors', 'expired-tok', ('142.250.1.1',)
    ).open('rb')
    resp_401 = mock.MagicMock(status=401)
    resp_206 = mock.MagicMock(status=206)
    resp_206.readinto.return_value = 8
    fake_https = mock.MagicMock()
    fake_https.getresponse.side_effect = [resp_401, resp_206]
    with (
        mock.patch.object(
            safetensors_io.http.client,
            'HTTPSConnection',
            return_value=fake_https,
        ),
        mock.patch.object(safetensors_io.socket, 'create_connection'),
        mock.patch.object(safetensors_io.ssl, 'create_default_context'),
        mock.patch.object(
            safetensors_io.GsHttpPath,
            'resolve_token',
            return_value='fresh-tok',
        ) as mock_refresh,
    ):
      mv = memoryview(bytearray(8))
      self.assertEqual(gs_file.fetch_range(0, 8, mv), 8)
      mock_refresh.assert_called_once_with(force_refresh=True)
      self.assertEqual(gs_file.parent.token, 'fresh-tok')

  def test_worker_unmap_futs_pruning_targeted_demux_and_3p_cmd(self):
    raw_file = pathlib.Path(self.temp_dir) / 'unmap_test.safetensors'
    raw_file.write_bytes(b'0123456789abcdef')
    with safetensors_io.ShmArena(size_bytes=65536) as arena:
      cmds = (
          f'{arena.shm_path}\t{arena.capacity}\t{raw_file}\t0\t8\t0\t1\n'
          'UNMAP\t2\n\n'
      )
      out_buf = io.StringIO()
      with (
          mock.patch.object(sys, 'stdin', io.StringIO(cmds)),
          mock.patch.object(sys, 'stdout', out_buf),
      ):
        safetensors_io.ProcPoolPath.worker_loop()
      lines = out_buf.getvalue().splitlines()
      self.assertEqual(lines[0], 'READY')
      self.assertIn('1\t8', lines)
      self.assertEqual(lines[-1], '2\t0')

    # Verify per-proc targeted reply demux delivers out-of-order replies to
    # concurrent waiters without global notify_all() wake storms.
    both_sent = threading.Event()
    sent_count = 0

    class _CountingStdin(io.StringIO):

      def write(self, s: str) -> int:
        nonlocal sent_count
        sent_count += 1
        if sent_count >= 2:
          both_sent.set()
        return super().write(s)

    fake_proc = mock.MagicMock()
    fake_proc.stdin = _CountingStdin()
    proc_state = safetensors_io._get_proc_state(fake_proc)

    class _OutOfOrderStdout:

      def __init__(self):
        self._emitted: list[str] = []

      def readline(self) -> str:
        both_sent.wait(timeout=5.0)
        if not self._emitted:
          while not proc_state.waiters:
            threading.Event().wait(0.001)
          waiter_id = next(iter(proc_state.waiters))
          other_id = 3 - waiter_id
          self._emitted.append(f'{other_id}\t{other_id * 8}\n')
          return f'{waiter_id}\t{waiter_id * 8}\n'
        return self._emitted.pop(0)

    fake_proc.stdout = _OutOfOrderStdout()
    proc_entry = (fake_proc, threading.Lock())
    results: dict[int, int] = {}

    def _call(idx: int) -> None:
      results[idx] = safetensors_io.ProcPoolPath.proc_read(
          proc_entry, f'CMD_{idx}'
      )

    with mock.patch.object(safetensors_io, '_REQ_COUNTER', iter([1, 2])):
      t1 = threading.Thread(target=_call, args=(1,))
      t2 = threading.Thread(target=_call, args=(2,))
      t1.start()
      t2.start()
      t1.join(timeout=5.0)
      t2.join(timeout=5.0)
    self.assertCountEqual(results.values(), [8, 16])

    # Verify release_proc_mmaps and ShmArena.__exit__ unmap hook.
    unmap_proc = mock.MagicMock()
    unmap_proc.stdin = io.StringIO()
    unmap_proc.stdout.readline.side_effect = ['10\t0\n']
    safetensors_io.READ_PROCS[:] = [(unmap_proc, threading.Lock())]
    with mock.patch.object(safetensors_io, '_REQ_COUNTER', iter([10])):
      with safetensors_io.ShmArena(shm_budget_bytes=8192) as arena:
        arena._used_procs = True
    self.assertIn('UNMAP\t10\n', unmap_proc.stdin.getvalue())
    safetensors_io.READ_PROCS.clear()
    safetensors_io.ProcPoolPath.release_proc_mmaps()

    # Verify Python vs bundled binary worker argv construction.
    with (
        mock.patch.object(safetensors_io, '_g3_mp_lib', None),
        mock.patch.object(sys, 'executable', '/usr/bin/python3'),
    ):
      cmd_py = safetensors_io.build_worker_cmd(
          safetensors_io.__file__,
          'from tunix.models import safetensors_io as s; '
          's.ProcPoolPath.worker_loop()',
          flag_args=('--undefok=x',),
          script_args=('--worker',),
          prefer_script=True,
      )
      self.assertEqual(
          cmd_py,
          [
              '/usr/bin/python3',
              '-u',
              os.path.abspath(safetensors_io.__file__),
              '--undefok=x',
              '--worker',
          ],
      )
      cmd_py_c = safetensors_io.build_worker_cmd(
          safetensors_io.__file__,
          'import sys',
          flag_args=('--undefok=x',),
      )
      self.assertIn('-c', cmd_py_c)
      self.assertIn('--undefok=x', cmd_py_c)
    with (
        mock.patch.object(safetensors_io, '_g3_mp_lib', None),
        mock.patch.object(sys, 'executable', '/app/trainer_bin'),
    ):
      cmd_bin = safetensors_io.build_worker_cmd(
          safetensors_io.__file__,
          'from tunix.models import safetensors_io as s; '
          's.ProcPoolPath.worker_loop()',
          flag_args=('--undefok=x',),
      )
      self.assertIn('--undefok=x', cmd_bin)
      self.assertIn('-c', cmd_bin)

  def test_hardening_fixes_worker_errors_http_200_dedupe_and_reap(self):
    # 1. Worker exception always emits a -1 reply instead of deadlocking.
    with safetensors_io.ShmArena(size_bytes=8192) as arena:
      missing_file = os.path.join(self.temp_dir, 'does_not_exist.bin')
      cmds = (
          f'{arena.shm_path}\t{arena.capacity}\t{missing_file}\t0\t8\t0\t9\n\n'
      )
      out_buf = io.StringIO()
      with (
          mock.patch.object(sys, 'stdin', io.StringIO(cmds)),
          mock.patch.object(sys, 'stdout', out_buf),
      ):
        safetensors_io.ProcPoolPath.worker_loop()
      lines = out_buf.getvalue().splitlines()
      self.assertEqual(lines[0], 'READY')
      self.assertIn('9\t-1', lines)

    # 2. CachedRangeFile.read with use_hdr_cache=False honors prior seek().
    raw_file = pathlib.Path(self.temp_dir) / 'seek_test.bin'
    raw_file.write_bytes(b'abcdefghij')
    src = safetensors_io.RangeSource(raw_file)
    with src.open('rb') as rf:
      rf.seek(4)
      self.assertEqual(rf.read(3), b'efg')
      self.assertEqual(rf.read(2), b'hi')

    # 3. HTTP 200 on start > 0 is rejected (prevents offset-0 corruption),
    # HTTP 403 fails fast, and concurrent 401 refresh deduplicates via gen.
    gs_file = safetensors_io.GsHttpPath(
        'gs://bkt/obj.safetensors', 'tok-a', ('142.250.1.1',)
    ).open('rb')
    resp_200 = mock.MagicMock(status=200)
    resp_403 = mock.MagicMock(status=403)
    fake_https = mock.MagicMock()
    fake_https.getresponse.side_effect = [resp_200, resp_403]
    with (
        mock.patch.object(
            safetensors_io.http.client,
            'HTTPSConnection',
            return_value=fake_https,
        ),
        mock.patch.object(safetensors_io.socket, 'create_connection'),
        mock.patch.object(safetensors_io.ssl, 'create_default_context'),
    ):
      with self.assertRaises(OSError):
        gs_file.fetch_range(16, 8, memoryview(bytearray(8)))
      self.assertEqual(fake_https.getresponse.call_count, 2)

    safetensors_io.GCS_TOKEN_CACHE[:] = [(1e18, 'tok-new')]
    safetensors_io.GCS_TOKEN_GEN[0] = 42
    with mock.patch.object(
        safetensors_io.GsHttpPath, 'resolve_token'
    ) as mock_res:
      self.assertEqual(
          safetensors_io.GsHttpPath.refresh_token_if_stale('tok-old', 41),
          'tok-new',
      )
      mock_res.assert_not_called()

    # 4. _close_shm_mmaps keeps busy mmaps tracked; available_shm_bytes works;
    # spawn_procs(0) reaps live procs and prunes exited procs.
    real_mm = mmap.mmap(-1, 16)
    mmaps = {'/dev/shm/busy': (real_mm, np.frombuffer(real_mm, dtype=np.uint8))}
    with memoryview(real_mm):
      safetensors_io.ProcPoolPath._close_shm_mmaps(mmaps, ignore_busy=True)
      self.assertIn('/dev/shm/busy', mmaps)
    safetensors_io.ProcPoolPath._close_shm_mmaps(mmaps, ignore_busy=False)
    self.assertEmpty(mmaps)
    self.assertGreater(safetensors_io.available_shm_bytes(self.temp_dir), 0)

    dead_proc = mock.MagicMock()
    dead_proc.poll.return_value = 1
    live_proc = mock.MagicMock()
    live_proc.poll.return_value = None
    live_proc.wait.side_effect = [
        safetensors_io.subprocess.TimeoutExpired('p', 5),
        0,
    ]
    safetensors_io.READ_PROCS[:] = [
        (dead_proc, threading.Lock()),
        (live_proc, threading.Lock()),
    ]
    safetensors_io.ProcPoolPath.spawn_procs(1)
    self.assertLen(safetensors_io.READ_PROCS, 1)
    safetensors_io.ProcPoolPath.spawn_procs(0)
    self.assertEmpty(safetensors_io.READ_PROCS)
    live_proc.kill.assert_called_once()

  def test_unified_gcs_primitives_prefault_and_auto_procs(self):
    self.assertEqual(
        safetensors_io.parse_gs_uri('gs://my-bkt/a/b.safetensors'),
        ('my-bkt', 'a/b.safetensors'),
    )
    with self.assertRaises(ValueError):
      safetensors_io.parse_gs_uri('http://not-gcs/a')
    with self.assertRaises(ValueError):
      safetensors_io.parse_gs_uri('gs://only-bucket/')

    safetensors_io.GCS_VIP_CACHE.clear()
    self.assertEqual(
        safetensors_io.resolve_gcs_vips(('10.0.0.1',)), ('10.0.0.1',)
    )
    with mock.patch.object(
        safetensors_io.GsHttpPath, 'discover_vips', side_effect=OSError('dns')
    ):
      self.assertEqual(safetensors_io.resolve_gcs_vips(), ())
      self.assertEmpty(safetensors_io.GCS_VIP_CACHE)
    with mock.patch.object(
        safetensors_io.GsHttpPath,
        'discover_vips',
        return_value=['10.0.0.1', '10.0.0.2'],
    ):
      self.assertEqual(
          safetensors_io.resolve_gcs_vips(), ('10.0.0.1', '10.0.0.2')
      )
      self.assertEqual(
          safetensors_io.resolve_gcs_vips(), ('10.0.0.1', '10.0.0.2')
      )
    self.assertEqual(safetensors_io.pick_thread_vip(()), '')
    vip0 = safetensors_io.pick_thread_vip(('10.0.0.1', '10.0.0.2'), attempt=0)
    vip1 = safetensors_io.pick_thread_vip(('10.0.0.1', '10.0.0.2'), attempt=1)
    self.assertNotEqual(vip0, vip1)
    many_vips = tuple(f'10.0.0.{i}' for i in range(1, 29))
    self.assertEqual(
        safetensors_io.pick_thread_vip(many_vips, attempt=0, vip_offset=5),
        '10.0.0.6',
    )
    self.assertEqual(
        safetensors_io.pick_thread_vip(many_vips, attempt=1, vip_offset=5),
        '10.0.0.7',
    )
    spread = {
        safetensors_io.pick_thread_vip(many_vips, vip_offset=seed)
        for seed in range(2 * 4 * 4)
    }
    self.assertLen(spread, len(many_vips))

    self.assertEqual(safetensors_io.auto_gcs_read_procs(1024), 0)
    self.assertEqual(safetensors_io.auto_gcs_read_procs(1024, read_procs=4), 4)
    self.assertGreaterEqual(
        safetensors_io.auto_gcs_read_procs(1024 * 1024 * 1024), 8
    )
    self.assertLessEqual(
        safetensors_io.auto_gcs_read_procs(500 * 1024**3),
        safetensors_io.GCS_MAX_READ_PROCS,
    )

    shm_path = os.path.join(self.temp_dir, 'prefault_test.shm')
    mm = safetensors_io.create_shm_mmap(shm_path, 32768, prefault=True)
    try:
      safetensors_io.pre_fault_shm(
          mm, 32768, write=False, block_indices=[0, 1], block_bytes=16384
      )
      safetensors_io.pre_fault_shm(mm, 0, write=True)
    finally:
      mm.close()
      os.unlink(shm_path)

    with safetensors_io.ShmArena(shm_budget_bytes=16384) as arena:
      arena.ensure_prefaulted(8192, touch_ptes=False)
      arena.ensure_prefaulted(4096)
      with mock.patch.object(
          os, 'posix_fallocate', side_effect=OSError('unsupported')
      ):
        arena.ensure_prefaulted(16384, touch_ptes=False)
      self.assertEqual(arena._prefaulted_bytes, 16384)
    self.assertFalse(safetensors_io._fallocate_fd(-1, 0, 0))

  def test_gcs_shm_read_pool_worker_e2e_urgent_hedge_401_and_errors(self):
    payloads = {
        '/bkt/f0.st': bytes(range(256)) * 16,
        '/bkt/f1.st': bytes(reversed(range(256))) * 16,
    }
    calls_lock = threading.Lock()
    auth_calls = [0]
    fail_mode = ['none']

    class _FakeResp:

      def __init__(self, status: int, data: bytes):
        self.status = status
        self._data = data
        self._pos = 0

      def read(self) -> bytes:
        return self._data

      def readinto(self, mv) -> int:
        rem = len(self._data) - self._pos
        n = min(len(mv), rem)
        if n > 0:
          mv[:n] = self._data[self._pos : self._pos + n]
          self._pos += n
        return n

    class _FakeConn:

      def __init__(self):
        self.timeout = 4.0
        self.sock = None
        self._req: tuple[str, dict[str, str]] = ('', {})

      def request(self, method: str, path: str, headers: dict[str, str]):
        del method
        self._req = (path, headers)

      def getresponse(self) -> _FakeResp:
        path, hdrs = self._req
        if fail_mode[0] == '403':
          return _FakeResp(403, b'denied')
        if hdrs.get('Authorization') == 'Bearer expired':
          with calls_lock:
            auth_calls[0] += 1
          return _FakeResp(401, b'expired')
        rng = hdrs['Range'].removeprefix('bytes=')
        lo, hi = (int(x) for x in rng.split('-'))
        return _FakeResp(206, payloads[path][lo : hi + 1])

      def close(self) -> None:
        pass

    pool = safetensors_io.GcsShmReadPool(
        1, ('10.0.0.1', '10.0.0.2'), streams_per_proc=2, spawn=False
    )
    worker_err: list[BaseException] = []

    def _run_worker() -> None:
      try:
        safetensors_io.run_gcs_shm_worker_cli([
            '--gcs-shm-worker',
            pool.ctrl_path,
            '0',
            '2',
            '10.0.0.1,10.0.0.2',
        ])
      except BaseException as exc:  # pylint: disable=broad-exception-caught
        worker_err.append(exc)

    with (
        mock.patch.object(
            safetensors_io,
            'get_thread_https_conn',
            side_effect=lambda vip, timeout=15.0, source_address=None: (
                _FakeConn()
            ),
        ),
        mock.patch.object(
            safetensors_io.GsHttpPath,
            'refresh_token_if_stale',
            return_value='fresh-tok',
        ),
    ):
      th = threading.Thread(target=_run_worker, daemon=True)
      th.start()
      try:
        while int(pool.ctrl.hdr.ready_workers) < 1:
          threading.Event().wait(0.005)
        with safetensors_io.ShmArena(shm_budget_bytes=16384) as arena:
          tasks = [
              (0, 0, 0, 1024, 0),
              (0, 0, 1024, 1024, 1024),
              (1, 1, 0, 1024, 2048),
              (1, 1, 1024, 1024, 3072),
          ]
          # Exercise credit-wait (-2), 401 refresh, urgent hedge, batch drain.
          pool.start_batch(
              arena.shm_path,
              arena.capacity,
              ['/bkt/f0.st', '/bkt/f1.st'],
              'expired',
              tasks,
              [2, 2],
              initial_max_allowed=2,
          )
          while int(pool.ctrl.group_rem[0]) > 0:
            pool.check_health()
            threading.Event().wait(0.002)
          pool.trigger_urgent_hedge(4)
          pool.update_max_allowed(4)
          while int(pool.ctrl.group_rem[1]) > 0:
            pool.check_health()
            threading.Event().wait(0.002)
          pool.wait_batch_drained()
          assert arena.shm_uint8 is not None
          self.assertEqual(
              bytes(arena.shm_uint8[:2048]), payloads['/bkt/f0.st'][:2048]
          )
          self.assertEqual(
              bytes(arena.shm_uint8[2048:4096]), payloads['/bkt/f1.st'][:2048]
          )
          self.assertGreaterEqual(auth_calls[0], 1)

          # Exercise fatal HTTP 403 -> check_health raises OSError.
          fail_mode[0] = '403'
          pool.start_batch(
              arena.shm_path,
              arena.capacity,
              ['/bkt/f0.st'],
              'fresh-tok',
              [(0, 0, 0, 64, 0)],
              [1],
              initial_max_allowed=1,
          )
          while not int(pool.ctrl.hdr.error_flag):
            threading.Event().wait(0.002)
          with self.assertRaisesRegex(OSError, 'GCS range read failed'):
            pool.check_health()
          pool.abort_batch()
          pool.wait_batch_drained()
          pool.unmap_data_arena()
      finally:
        pool.shutdown()
        th.join(timeout=5.0)
    self.assertEmpty(worker_err)

    # Exercise singleton lifecycle (ensure/release/shutdown) and spawn failure.
    self.assertIsNone(safetensors_io.ensure_gcs_read_pool(0, ()))
    fake_pool = mock.MagicMock()
    fake_pool.is_alive.return_value = True
    fake_pool.num_procs = 4
    with mock.patch.object(
        safetensors_io, 'GcsShmReadPool', return_value=fake_pool
    ):
      got = safetensors_io.ensure_gcs_read_pool(2, ('10.0.0.1',))
      self.assertIs(got, fake_pool)
      self.assertIs(
          safetensors_io.ensure_gcs_read_pool(2, ('10.0.0.1',)), fake_pool
      )
      safetensors_io.release_gcs_read_pool_mmaps()
      fake_pool.unmap_data_arena.assert_called_once()
      safetensors_io.shutdown_gcs_read_pool()
      fake_pool.shutdown.assert_called_once()

    # Exercise GcsShmReadPool.is_alive() across empty, live, and crashed procs.
    liveness_pool = safetensors_io.GcsShmReadPool(
        1, ('10.0.0.1',), spawn=False
    )
    try:
      self.assertFalse(liveness_pool.is_alive())
      alive_p = mock.MagicMock()
      alive_p.poll.return_value = None
      liveness_pool._procs = [alive_p]  # pylint: disable=protected-access
      self.assertTrue(liveness_pool.is_alive())
      alive_p.poll.return_value = 1
      self.assertFalse(liveness_pool.is_alive())
      with self.assertRaisesRegex(OSError, 'crashed before batch start'):
        liveness_pool.start_batch(
            '/dev/shm/x', 4096, ['/bkt/f'], 'tok', [], [], 0
        )
      liveness_pool._procs = []  # pylint: disable=protected-access
    finally:
      liveness_pool.shutdown()

    dead_p = mock.MagicMock()
    dead_p.poll.return_value = 1
    with mock.patch.object(subprocess, 'Popen', return_value=dead_p):
      with self.assertRaisesRegex(OSError, 'exited during startup'):
        safetensors_io.GcsShmReadPool(1, ('10.0.0.1',), spawn=True)

  def test_hardening_fallocate_enospc_pool_timeout_and_barrier_abort(self):
    # 1. _fallocate_fd falls back on EOPNOTSUPP but propagates ENOSPC.
    with mock.patch.object(
        os, 'posix_fallocate', side_effect=OSError(errno.EOPNOTSUPP, 'unsup')
    ):
      self.assertFalse(safetensors_io._fallocate_fd(1, 0, 4096))
    with mock.patch.object(
        os, 'posix_fallocate', side_effect=OSError(errno.ENOSPC, 'no space')
    ):
      with self.assertRaises(OSError):
        safetensors_io._fallocate_fd(1, 0, 4096)

    # 2. create_shm_mmap cleans up backing file if pre_fault_shm fails.
    bad_shm = os.path.join(self.temp_dir, 'prefault_err.shm')
    with mock.patch.object(
        safetensors_io, 'pre_fault_shm', side_effect=RuntimeError('boom')
    ):
      with self.assertRaisesRegex(RuntimeError, 'boom'):
        safetensors_io.create_shm_mmap(bad_shm, 4096, prefault=True)
    self.assertFalse(os.path.exists(bad_shm))

    # 3. _ShmMutex releases _tlock when flock raises.
    lock_file = os.path.join(self.temp_dir, 'flock_err.lock')
    mu = safetensors_io._ShmMutex(lock_file)
    try:
      with mock.patch.object(fcntl, 'flock', side_effect=OSError('bad fd')):
        with self.assertRaises(OSError):
          with mu:
            pass
      self.assertTrue(mu._tlock.acquire(blocking=False))
      mu._tlock.release()
    finally:
      mu.close()

    # 4. GsHttpPath.open never hijacks READ_PROCS.
    safetensors_io.READ_PROCS.append((mock.MagicMock(), threading.Lock()))
    try:
      gs_fh = safetensors_io.GsHttpPath(
          'gs://bkt/a.safetensors', 'tok', ('10.0.0.1',)
      ).open('rb', stream_id=0)
      self.assertIsNone(gs_fh.proc_entry)
      gs_fh.close()
    finally:
      safetensors_io.READ_PROCS.clear()

    # 5. wait_batch_drained timeout shuts down unhealthy pool and raises.
    pool = safetensors_io.GcsShmReadPool(1, ('10.0.0.1',), spawn=False)
    safetensors_io._GCS_READ_POOL[:] = [pool]
    try:
      pool.start_batch(
          '/dev/shm/x', 4096, ['/bkt/f'], 'tok', [(0, 0, 0, 8, 0)], [1], 1
      )
      with self.assertRaisesRegex(OSError, 'Timed out waiting'):
        pool.wait_batch_drained(timeout_s=0.005)
      self.assertFalse(pool.is_alive())
      self.assertEmpty(safetensors_io._GCS_READ_POOL)
    finally:
      safetensors_io._GCS_READ_POOL.clear()
      pool.shutdown()

    u_pool = safetensors_io.GcsShmReadPool(1, ('10.0.0.1',), spawn=False)
    try:
      u_pool.start_batch(
          '/dev/shm/x', 4096, ['/bkt/f'], 'tok', [(0, 0, 0, 8, 0)], [1], 1
      )
      u_pool.unmap_data_arena(timeout_s=0.005)
      self.assertFalse(u_pool.is_alive())
    finally:
      u_pool.shutdown()

    # 6. _GcsShmWorker._thread_loop aborts barrier on error instead of hanging.
    b_pool = safetensors_io.GcsShmReadPool(
        1, ('10.0.0.1',), streams_per_proc=2, spawn=False
    )
    try:
      worker = safetensors_io._GcsShmWorker(
          b_pool.ctrl_path, 0, 2, ('10.0.0.1',)
      )
      with mock.patch.object(
          worker, '_prepare_batch', side_effect=RuntimeError('prep fail')
      ):
        with mock.patch.object(worker, '_prewarm_vips', return_value=('', '')):
          worker.run()
      self.assertEqual(int(b_pool.ctrl.hdr.error_flag), 1)
      with self.assertRaisesRegex(OSError, 'prep fail'):
        b_pool.check_health()
    finally:
      b_pool.shutdown()

  def test_review_hardening_vips_sockets_ready_and_underflow(self):
    # 1. Partial DNS failure in discover_vips retains VIPs from reachable hosts.
    def _fake_addrinfo(host, *_args, **_kwargs):
      if host == 'private.googleapis.com':
        raise OSError('NXDOMAIN')
      return [(None, None, None, None, (f'10.0.0.{len(host)}', 443))]

    with mock.patch.object(
        safetensors_io.socket, 'getaddrinfo', side_effect=_fake_addrinfo
    ):
      vips = safetensors_io.GsHttpPath.discover_vips()
      self.assertNotEmpty(vips)

    # 2. get_thread_https_conn closes raw_sock on wrap_socket failure and
    #    rebuilds cached connections whose socket was already closed (`None`).
    raw_sock = mock.MagicMock()
    bad_ctx = mock.MagicMock()
    bad_ctx.wrap_socket.side_effect = OSError('tls handshake failed')
    with (
        mock.patch.object(
            safetensors_io.ssl, 'create_default_context', return_value=bad_ctx
        ),
        mock.patch.object(
            safetensors_io.socket, 'create_connection', return_value=raw_sock
        ),
    ):
      with self.assertRaisesRegex(OSError, 'tls handshake failed'):
        safetensors_io.get_thread_https_conn('10.1.2.3')
    raw_sock.close.assert_called_once()

    stale_conn = mock.MagicMock(sock=None)
    good_sock = mock.MagicMock()
    good_ctx = mock.MagicMock()
    good_ctx.wrap_socket.return_value = good_sock
    key = (threading.get_ident(), '10.1.2.4')
    safetensors_io.THREAD_CONNS[key] = stale_conn
    try:
      with (
          mock.patch.object(
              safetensors_io.ssl,
              'create_default_context',
              return_value=good_ctx,
          ),
          mock.patch.object(
              safetensors_io.socket,
              'create_connection',
              return_value=mock.MagicMock(),
          ),
      ):
        rebuilt = safetensors_io.get_thread_https_conn('10.1.2.4')
      self.assertIsNot(rebuilt, stale_conn)
      self.assertIs(rebuilt.sock, good_sock)
      stale_conn.close.assert_called_once()
    finally:
      safetensors_io.evict_thread_https_conn('10.1.2.4')

    # 3. await_worker_ready handles missing stdout, READY, and timeout/EOF.
    with self.assertRaisesRegex(OSError, 'missing stdout'):
      safetensors_io.await_worker_ready(
          types.SimpleNamespace(stdout=None),  # type: ignore[arg-type]
          time.monotonic() + 1.0,
          error_cls=OSError,
      )
    safetensors_io.await_worker_ready(
        types.SimpleNamespace(stdout=io.StringIO('READY\n')),  # type: ignore[arg-type]
        time.monotonic() + 1.0,
    )
    with self.assertRaisesRegex(RuntimeError, 'handshake'):
      safetensors_io.await_worker_ready(
          types.SimpleNamespace(stdout=io.StringIO('')),  # type: ignore[arg-type]
          time.monotonic() + 1.0,
      )

    # 4. _execute_gcs_task alternates both VIPs across retries (`attempt & 1`).
    t_pool = safetensors_io.GcsShmReadPool(1, ('10.0.0.1',), spawn=False)
    try:
      t_pool.start_batch(
          '/dev/shm/x', 4096, ['/bkt/f'], 'tok', [(0, 0, 0, 8, 0)], [1], 1
      )
      tried_vips: list[str] = []

      def _fake_read(*args, **kwargs):
        del kwargs
        tried_vips.append(args[7])
        return 'done' if len(tried_vips) == 3 else 'retry'

      with (
          mock.patch.object(
              safetensors_io, '_gcs_read_http_once', side_effect=_fake_read
          ),
          mock.patch.object(safetensors_io.time, 'sleep'),
      ):
        ok = safetensors_io._execute_gcs_task(
            t_pool.ctrl,
            t_pool.shm_lock,
            ['/bkt/f'],
            np.zeros(64, dtype=np.uint8),
            0,
            False,
            ('10.0.0.1', '10.0.0.2'),
        )
      self.assertTrue(ok)
      self.assertEqual(tried_vips, ['10.0.0.1', '10.0.0.2', '10.0.0.1'])
    finally:
      t_pool.shutdown()

    # 5. ShmArena.release_unit fails fast on ring accounting underflow.
    with safetensors_io.ShmArena(size_bytes=4096) as arena:
      arena.unit_shm_bytes = [1024]
      arena.in_flight_bytes = 512
      with self.assertRaisesRegex(RuntimeError, 'ring underflow'):
        arena.release_unit(0)

  def test_audit_fixes_expires_zero_short_eof_and_dtype_consistency(self):
    # 1. expires_in=0 does not cache an expired token for 300s (or 60s).
    with mock.patch.dict(os.environ, {}, clear=False):
      os.environ.pop('TUNIX_GCS_TOKEN', None)
      safetensors_io.GCS_TOKEN_CACHE.clear()
      resp_zero = mock.MagicMock(status=200)
      resp_zero.read.return_value = (
          b'{"access_token": "tok-exp0", "expires_in": 0}'
      )
      resp_fresh = mock.MagicMock(status=200)
      resp_fresh.read.return_value = (
          b'{"access_token": "tok-fresh", "expires_in": 90}'
      )
      conn = mock.MagicMock()
      conn.getresponse.side_effect = [resp_zero, resp_fresh]
      with mock.patch.object(
          safetensors_io.http.client, 'HTTPConnection', return_value=conn
      ):
        self.assertEqual(
            safetensors_io.GsHttpPath.resolve_token(), 'tok-exp0'
        )
        self.assertEmpty(safetensors_io.GCS_TOKEN_CACHE)
        t_before = time.monotonic()
        self.assertEqual(
            safetensors_io.GsHttpPath.resolve_token(), 'tok-fresh'
        )
        self.assertLen(safetensors_io.GCS_TOKEN_CACHE, 1)
        self.assertLessEqual(
            safetensors_io.GCS_TOKEN_CACHE[0][0] - t_before, 35.0
        )

    # 2. < 256 KiB shard clean 206 EOF succeeds in 1 attempt without evicting
    # TLS, while truncated mid-stream drops fail fast with OSError on attempt 2.
    class _HdrResp:

      def __init__(
          self,
          status: int,
          body: bytes,
          hdrs: dict[str, str],
      ):
        self.status = status
        self._body = body
        self._pos = 0
        self._hdrs = hdrs

      def getheader(self, name: str, default: str | None = None) -> str | None:
        return self._hdrs.get(name, default)

      def read(self) -> bytes:
        return self._body

      def readinto(self, mv) -> int:
        n = min(len(mv), len(self._body) - self._pos)
        if n > 0:
          mv[:n] = self._body[self._pos : self._pos + n]
          self._pos += n
        return n

    small_payload = b'01234567' + b'X' * 92
    gs_src = safetensors_io.GsHttpPath(
        'gs://bkt/small.safetensors', 'tok', ('142.250.9.1',)
    )
    fake_conn = mock.MagicMock(sock=mock.MagicMock())
    n_small = len(small_payload)
    fake_conn.getresponse.return_value = _HdrResp(
        206,
        small_payload,
        {
            'Content-Range': f'bytes 0-{n_small - 1}/{n_small}',
            'Content-Length': str(n_small),
        },
    )
    safetensors_io.THREAD_CONNS.clear()
    with (
        mock.patch.object(
            safetensors_io.http.client,
            'HTTPSConnection',
            return_value=fake_conn,
        ),
        mock.patch.object(safetensors_io.socket, 'create_connection'),
        mock.patch.object(safetensors_io.ssl, 'create_default_context'),
    ):
      with gs_src.open('rb') as gf:
        self.assertEqual(gf.read(8), b'01234567')
      self.assertEqual(fake_conn.getresponse.call_count, 1)
      self.assertEqual(gs_src.hdr_cache, small_payload)
      self.assertIn(
          (threading.get_ident(), '142.250.9.1'), safetensors_io.THREAD_CONNS
      )

      # Truncated mid-stream drop must NOT return partial bytes on final retry.
      trunc_conn = mock.MagicMock(sock=mock.MagicMock())
      trunc_conn.getresponse.side_effect = [
          _HdrResp(
              206,
              b'short',
              {'Content-Range': 'bytes 0-99/100', 'Content-Length': '100'},
          )
          for _ in range(3)
      ]
      safetensors_io.THREAD_CONNS.clear()
      with mock.patch.object(
          safetensors_io.http.client,
          'HTTPSConnection',
          return_value=trunc_conn,
      ):
        with gs_src.open('rb') as gf:
          with self.assertRaisesRegex(OSError, 'Failed HTTP Range read'):
            gf.fetch_range(0, 100, memoryview(bytearray(100)))
        self.assertEqual(trunc_conn.getresponse.call_count, 3)

      # Content-Range start mismatch, oversized span, and mid-file truncation
      # fail fast across retries without corrupting keep-alive framing.
      for bad_hdrs in (
          {'Content-Range': 'bytes 16-23/100', 'Content-Length': '8'},
          {'Content-Range': 'bytes 0-19/20', 'Content-Length': '20'},
          {'Content-Range': 'bytes 0-3/100', 'Content-Length': '4'},
          {'Content-Length': '20'},
      ):
        bad_cr_conn = mock.MagicMock(sock=mock.MagicMock())
        bad_cr_conn.getresponse.side_effect = [
            _HdrResp(206, b'abcdefgh', bad_hdrs) for _ in range(3)
        ]
        safetensors_io.THREAD_CONNS.clear()
        with mock.patch.object(
            safetensors_io.http.client,
            'HTTPSConnection',
            return_value=bad_cr_conn,
        ):
          with gs_src.open('rb') as gf:
            with self.assertRaisesRegex(OSError, 'Failed HTTP Range read'):
              gf.fetch_range(0, 8, memoryview(bytearray(8)))

      # Unsupported seek whence fails fast instead of silently seeking from 0.
      with gs_src.open('rb') as gf:
        with self.assertRaisesRegex(ValueError, 'Unsupported seek whence'):
          gf.seek(0, io.SEEK_END)

    # 3. to_np_dtype returns uniform np.dtype with integer .itemsize.
    for wire in ('BF16', 'F16', 'F32', 'F64', 'I32', 'F8_E4M3'):
      dt = safetensors_io.to_np_dtype(wire)
      self.assertIsInstance(dt, np.dtype)
      self.assertIsInstance(dt.itemsize, int)
    self.assertIsInstance(safetensors_io.to_np_dtype(np.float32), np.dtype)
    self.assertEqual(safetensors_io.to_np_dtype(np.float32).itemsize, 4)

    # 4. pre_fault_shm honors offset when block_indices is set.
    shm_path = os.path.join(self.temp_dir, 'prefault_off.shm')
    mm = safetensors_io.create_shm_mmap(shm_path, 16384, prefault=False)
    try:
      view = np.frombuffer(mm, dtype=np.uint8)
      view[:] = 7
      safetensors_io.pre_fault_shm(
          mm, 8192, offset=4096, write=True, block_indices=[0], block_bytes=4096
      )
      self.assertEqual(int(view[0]), 7)
      self.assertEqual(int(view[4096]), 0)
      self.assertEqual(int(view[8192]), 7)
      del view
    finally:
      mm.close()
      os.unlink(shm_path)

  def test_rapid_zonal_multinic_sidecar_and_high_group_limits(self):
    # 1. Rapid Zonal auto-detect: non-gs://, empty bucket, no token, 200 RAPID,
    #    200 STANDARD, 403, transport error, and cache reuse.
    self.assertFalse(safetensors_io.is_rapid_zonal_bucket('/tmp/local'))
    self.assertFalse(safetensors_io.is_rapid_zonal_bucket('gs:///obj'))
    with mock.patch.object(
        safetensors_io.GsHttpPath, 'resolve_token', return_value=''
    ):
      self.assertFalse(safetensors_io.is_rapid_zonal_bucket('gs://no-tok/p'))
      self.assertNotIn('no-tok', safetensors_io.GCS_RAPID_BUCKET_CACHE)

    def _make_resp_conn(status: int, body: bytes | None = None, exc=None):
      conn = mock.MagicMock()
      if exc is not None:
        conn.request.side_effect = exc
      else:
        conn.getresponse.return_value = types.SimpleNamespace(
            status=status, read=lambda: body or b''
        )
      return conn

    with (
        mock.patch.object(
            safetensors_io.GsHttpPath, 'resolve_token', return_value='tok'
        ),
        mock.patch.object(
            safetensors_io.GsHttpPath,
            'refresh_token_if_stale',
            return_value='fresh-tok',
        ),
        mock.patch.object(
            safetensors_io, 'resolve_gcs_vips', return_value=('10.9.0.1',)
        ),
        mock.patch.object(
            safetensors_io, 'evict_thread_https_conn'
        ) as mock_evict,
        mock.patch.object(
            safetensors_io,
            'get_thread_https_conn',
            side_effect=[
                _make_resp_conn(
                    200, b'{"locationType":"zone","storageClass":"RAPID"}'
                ),
                _make_resp_conn(
                    200, b'{"locationType":"region","storageClass":"STANDARD"}'
                ),
                _make_resp_conn(403, b'forbidden'),
                _make_resp_conn(404, b'not found'),
                _make_resp_conn(401, b'expired'),
                _make_resp_conn(
                    200, b'{"locationType":"zone","storageClass":"RAPID"}'
                ),
                _make_resp_conn(0, exc=OSError('reset')),
                _make_resp_conn(0, exc=OSError('reset')),
                _make_resp_conn(0, exc=OSError('reset')),
                _make_resp_conn(
                    200, b'{"locationType":"zone","storageClass":"RAPID"}'
                ),
            ],
        ) as mock_conn,
    ):
      self.assertTrue(safetensors_io.is_rapid_zonal_bucket('gs://rapid-bkt/x'))
      self.assertTrue(safetensors_io.is_rapid_zonal_bucket('gs://rapid-bkt/y'))
      self.assertFalse(safetensors_io.is_rapid_zonal_bucket('gs://std-bkt/x'))
      self.assertFalse(safetensors_io.is_rapid_zonal_bucket('gs://deny-bkt/x'))
      self.assertFalse(safetensors_io.is_rapid_zonal_bucket('gs://deny-bkt/y'))
      self.assertFalse(safetensors_io.is_rapid_zonal_bucket('gs://miss-bkt/x'))
      self.assertTrue(safetensors_io.is_rapid_zonal_bucket('gs://auth-bkt/x'))
      # Transient transport errors must NOT be cached permanently.
      self.assertFalse(safetensors_io.is_rapid_zonal_bucket('gs://err-bkt/x'))
      self.assertNotIn('err-bkt', safetensors_io.GCS_RAPID_BUCKET_CACHE)
      self.assertTrue(safetensors_io.is_rapid_zonal_bucket('gs://err-bkt/x'))
      self.assertEqual(mock_conn.call_count, 10)
      self.assertEqual(mock_evict.call_count, 6)

    # 2. Multi-NIC discovery via pure-Python Linux AF_NETLINK (zero `ip`
    #    binary): filters loopback/down/virtual/unrouted/failed-TCP, prefers
    #    distinct NUMA nodes (falling back to distinct PCI devices on
    #    single-NUMA VMs), and returns () if <2 NICs or non-Linux.
    with mock.patch.object(safetensors_io, 'resolve_gcs_vips', return_value=()):
      self.assertEqual(safetensors_io.discover_gcs_source_ips(''), ())

    def _nl_msg(msg_type: int, payload: bytes, seq: int = 1, flags: int = 0):
      nl_len = 16 + len(payload)
      pad = ((nl_len + 3) & ~3) - nl_len
      return (
          struct.pack('=IHHII', nl_len, msg_type, flags, seq, 0)
          + payload
          + (b'\x00' * pad)
      )

    def _addr_msg(
        ifindex: int,
        ip: str,
        family=socket.AF_INET,
        scope: int = 0,
        rta_type: int = safetensors_io._IFA_LOCAL,
    ):
      ifa = struct.pack('=BBBBI', family, 24, 0, scope, ifindex)
      rta = struct.pack('=HH', 8, rta_type) + socket.inet_pton(
          socket.AF_INET, ip
      )
      return _nl_msg(safetensors_io._RTM_NEWADDR, ifa + rta, flags=0x02)

    def _route_msg(oif: int):
      rtm = struct.pack('=BBBBBBBBI', socket.AF_INET, 32, 32, 0, 0, 0, 0, 1, 0)
      rta = struct.pack('=HHI', 8, safetensors_io._RTA_OIF, oif)
      return _nl_msg(safetensors_io._RTM_NEWROUTE, rtm + rta)

    idx_to_dev = {
        1: 'lo',
        2: 'eth_down',
        3: 'veth0',
        4: 'eth0',
        5: 'eth1',
        6: 'eth2',
        7: 'eth3',
        8: 'eth4',
    }
    addr_pkt_1 = (
        _nl_msg(safetensors_io._RTM_NEWADDR, b'short', flags=0x02)
        + _nl_msg(safetensors_io._RTM_NEWADDR, b'x' * 8, seq=99, flags=0x02)
        + _addr_msg(1, '127.0.0.1', scope=254)
        + _addr_msg(2, '10.88.0.1')
        + _addr_msg(3, '10.99.0.1')
        + _addr_msg(4, '169.254.1.1')
        + _addr_msg(4, '10.0.0.99')  # unrouted alias before routable IP
        + _addr_msg(4, '10.0.0.1')
        + _addr_msg(5, '10.0.0.2')
    )
    addr_pkt_2 = (
        _addr_msg(6, '10.0.1.1', rta_type=safetensors_io._IFA_ADDRESS)
        + _addr_msg(7, '10.0.2.1')
        + _addr_msg(8, '10.0.3.1')
        + _nl_msg(safetensors_io._NLMSG_DONE, b'\x00\x00\x00\x00', flags=0x02)
    )
    route_oif = {
        '10.0.0.1': 4,
        '10.0.0.2': 5,
        '10.0.1.1': 6,
        '10.0.2.1': 4,  # routes out eth0 instead of eth3 -> rejected
        '10.0.3.1': 8,
    }
    phys_devs = {'eth0', 'eth1', 'eth2', 'eth3', 'eth4'}
    operstates = {d: 'up\n' for d in phys_devs | {'veth0'}}
    operstates['eth_down'] = 'down\n'
    numa_map = {'eth0': '0\n', 'eth1': '0\n', 'eth2': '1\n', 'eth3': '-1\n'}
    active_addr_pkts = [addr_pkt_1, addr_pkt_2]

    class _FakeNetlinkSock:

      def __init__(self):
        self._pkts: list[bytes] = []

      def __enter__(self):
        return self

      def __exit__(self, *_exc):
        return None

      def settimeout(self, timeout: float) -> None:
        del timeout

      def bind(self, addr) -> None:
        del addr

      def send(self, req: bytes) -> int:
        _, mtype, _, _, _ = struct.unpack_from('=IHHII', req, 0)
        if mtype == safetensors_io._RTM_GETADDR:
          self._pkts = list(active_addr_pkts)
        elif mtype == safetensors_io._RTM_GETROUTE:
          attrs = safetensors_io._parse_nl_attrs(req[16:], 12)
          src = socket.inet_ntop(socket.AF_INET, attrs[safetensors_io._RTA_SRC])
          if src in route_oif:
            self._pkts = [_route_msg(route_oif[src])]
          else:
            self._pkts = [
                _nl_msg(safetensors_io._NLMSG_ERROR, struct.pack('=i', -101))
            ]
        return len(req)

      def recv(self, bufsize: int) -> bytes:
        del bufsize
        return self._pkts.pop(0) if self._pkts else b''

    def _fake_if_indextoname(ifidx: int) -> str:
      if ifidx not in idx_to_dev:
        raise OSError('no such interface')
      return idx_to_dev[ifidx]

    def _fake_open(path, *args, **kwargs):
      del args, kwargs
      for dev, val in operstates.items():
        if path == f'/sys/class/net/{dev}/operstate':
          return io.StringIO(val)
      for dev, val in numa_map.items():
        if path == f'/sys/class/net/{dev}/device/numa_node':
          return io.StringIO(val)
      raise FileNotFoundError(path)

    tcp_fail_ips = {'10.0.3.1'}

    def _fake_create_conn(addr, timeout=1.0, source_address=None):
      del addr, timeout
      if source_address and source_address[0] in tcp_fail_ips:
        raise OSError('no egress')
      return mock.MagicMock()

    with (
        mock.patch.object(
            safetensors_io.socket,
            'socket',
            side_effect=lambda *_a, **_kw: _FakeNetlinkSock(),
        ),
        mock.patch.object(
            safetensors_io.socket,
            'if_indextoname',
            side_effect=_fake_if_indextoname,
        ),
        mock.patch.object(
            safetensors_io.os.path,
            'exists',
            side_effect=lambda p: any(
                p == f'/sys/class/net/{d}/device' for d in phys_devs
            ),
        ),
        mock.patch.object(
            safetensors_io.os.path,
            'realpath',
            side_effect=lambda p: f'/sys/devices/pci/{p}',
        ),
        mock.patch('builtins.open', side_effect=_fake_open),
        mock.patch.object(
            safetensors_io.socket,
            'create_connection',
            side_effect=_fake_create_conn,
        ),
    ):
      # Dual-NUMA host: picks 1 per distinct NUMA node, skipping unrouted alias.
      ips = safetensors_io.discover_gcs_source_ips('10.9.0.1')
      self.assertEqual(ips, ('10.0.0.1', '10.0.1.1'))
      # Cache hit avoids re-probing.
      self.assertEqual(safetensors_io.discover_gcs_source_ips('10.9.0.1'), ips)

      # Single-NUMA multi-NIC VM: distinct PCI devices on NUMA 0 both work.
      active_addr_pkts[:] = [
          _addr_msg(4, '10.0.0.1')
          + _addr_msg(5, '10.0.0.2')
          + _nl_msg(safetensors_io._NLMSG_DONE, b'\x00\x00\x00\x00', flags=0x02)
      ]
      self.assertEqual(
          safetensors_io.discover_gcs_source_ips('10.9.0.3'),
          ('10.0.0.1', '10.0.0.2'),
      )

      # Transient TCP failure on >=2 routed NICs does NOT poison cache.
      tcp_fail_ips.add('10.0.0.2')
      self.assertEqual(safetensors_io.discover_gcs_source_ips('10.9.0.4'), ())
      self.assertNotIn('10.9.0.4', safetensors_io.GCS_SOURCE_IPS_CACHE)
      tcp_fail_ips.remove('10.0.0.2')
      self.assertEqual(
          safetensors_io.discover_gcs_source_ips('10.9.0.4'),
          ('10.0.0.1', '10.0.0.2'),
      )

    # Graceful empty fallback on non-Linux / unsupported AF_NETLINK / errors.
    with mock.patch.object(
        safetensors_io.socket, 'socket', side_effect=OSError('EAFNOSUPPORT')
    ):
      self.assertEqual(safetensors_io.discover_gcs_source_ips('10.9.0.2'), ())
    self.assertIsNone(safetensors_io._route_egress_dev('not-an-ip', '10.0.0.1'))
    self.assertEqual(safetensors_io._parse_nl_attrs(b'\x02\x00\x01\x00', 0), {})
    self.assertEqual(
        safetensors_io._parse_nl_packet(b'\x04\x00\x00\x00' + b'\x00' * 12, 1),
        ([], 'error'),
    )

    # 3. source_address plumbing in get_thread_https_conn, evict, and read pool.
    good_ctx = mock.MagicMock()
    good_ctx.wrap_socket.return_value = mock.MagicMock()
    with (
        mock.patch.object(
            safetensors_io.ssl, 'create_default_context', return_value=good_ctx
        ),
        mock.patch.object(
            safetensors_io.socket,
            'create_connection',
            return_value=mock.MagicMock(),
        ) as mock_cc,
    ):
      c1 = safetensors_io.get_thread_https_conn(
          '10.9.0.1', source_address=('10.0.0.1', 0)
      )
      c2 = safetensors_io.get_thread_https_conn(
          '10.9.0.1', source_address=('10.0.0.1', 0)
      )
      self.assertIs(c1, c2)
      self.assertEqual(
          mock_cc.call_args.kwargs['source_address'], ('10.0.0.1', 0)
      )
      safetensors_io.evict_thread_https_conn(
          '10.9.0.1', source_address=('10.0.0.1', 0)
      )
      self.assertNotIn(
          (threading.get_ident(), '10.9.0.1', ('10.0.0.1', 0)),
          safetensors_io.THREAD_CONNS,
      )

    spawned_cmds: list[list[str]] = []

    def _fake_popen(cmd, **_kwargs):
      spawned_cmds.append(list(cmd))
      ctrl_p = cmd[cmd.index('--gcs-shm-worker') + 1]
      fd = os.open(ctrl_p, os.O_RDWR)
      try:
        mm = mmap.mmap(fd, safetensors_io._CTRL_TOTAL_BYTES)
        view = safetensors_io._GcsCtrlView(mm)
        view.hdr.ready_workers += 1
        view.close()
      finally:
        os.close(fd)
      p = mock.MagicMock()
      p.poll.return_value = None
      return p

    with mock.patch.object(
        safetensors_io.subprocess, 'Popen', side_effect=_fake_popen
    ):
      mp_pool = safetensors_io.GcsShmReadPool(
          2,
          ('10.9.0.1',),
          spawn=True,
          source_ips=('10.0.0.1', '10.0.1.1'),
      )
      try:
        self.assertEqual(spawned_cmds[0][-1], '10.0.0.1')
        self.assertEqual(spawned_cmds[1][-1], '10.0.1.1')
      finally:
        mp_pool.shutdown()

    with mock.patch.object(safetensors_io, '_GcsShmWorker') as mock_wcls:
      safetensors_io.run_gcs_shm_worker_cli(
          ['--gcs-shm-worker', '/tmp/c', '0', '3', '10.9.0.1', '10.0.0.1']
      )
      self.assertEqual(mock_wcls.call_args.kwargs['source_ip'], '10.0.0.1')
      mock_wcls.return_value.run.assert_called_once()

    # 4. Sidecar header index (.tunix_headers.json) and fast source building.
    sc_dir = pathlib.Path(self.create_tempdir().full_path)
    self.assertIsNone(safetensors_io.load_sidecar_headers(str(sc_dir)))
    sc_file = sc_dir / safetensors_io.SIDECAR_HEADERS_FILENAME
    sc_file.write_text('{bad json')
    self.assertIsNone(safetensors_io.load_sidecar_headers(str(sc_dir)))
    for bad_payload in (
        {
            'version': 2,
            'shards': {
                'a.safetensors': {
                    'header_len': 8,
                    'file_size': 16,
                    'tensors': {},
                }
            },
        },
        {
            'version': True,
            'shards': {
                'a.safetensors': {
                    'header_len': 8,
                    'file_size': 16,
                    'tensors': {},
                }
            },
        },
        {'version': 1, 'shards': {}},
        {'version': 1, 'shards': {'a.safetensors': 'bad'}},
        {
            'version': 1,
            'shards': {
                'a.safetensors': {
                    'header_len': 0,
                    'file_size': 16,
                    'tensors': {},
                }
            },
        },
        {
            'version': 1,
            'shards': {
                'a.safetensors': {
                    'header_len': -1,
                    'file_size': 16,
                    'tensors': {},
                }
            },
        },
        {
            'version': 1,
            'shards': {
                'a.safetensors': {
                    'header_len': safetensors_io.MAX_HEADER_BYTES + 1,
                    'file_size': safetensors_io.MAX_HEADER_BYTES + 16,
                    'tensors': {},
                }
            },
        },
        {
            'version': 1,
            'shards': {
                'a.safetensors': {
                    'header_len': True,
                    'file_size': 16,
                    'tensors': {},
                }
            },
        },
        {
            'version': 1,
            'shards': {
                'a.safetensors': {'header_len': 8, 'tensors': {}},
            },
        },
        {
            'version': 1,
            'shards': {
                'a.safetensors': {
                    'header_len': 8,
                    'file_size': 12,
                    'tensors': {},
                }
            },
        },
        {
            'version': 1,
            'shards': {
                '../evil.safetensors': {
                    'header_len': 8,
                    'file_size': 16,
                    'tensors': {},
                }
            },
        },
        {
            'version': 1,
            'shards': {
                'sub/a.safetensors': {
                    'header_len': 8,
                    'file_size': 16,
                    'tensors': {},
                }
            },
        },
        {
            'version': 1,
            'shards': {
                'model.safetensors\n': {
                    'header_len': 8,
                    'file_size': 16,
                    'tensors': {},
                }
            },
        },
        {
            'version': 1,
            'shards': {
                'a\x00b.safetensors': {
                    'header_len': 8,
                    'file_size': 16,
                    'tensors': {},
                }
            },
        },
        {
            'version': 1,
            'shards': {
                'a.txt': {'header_len': 8, 'file_size': 16, 'tensors': {}}
            },
        },
        {
            'version': 1,
            'shards': {
                'a.safetensors': {
                    'header_len': 8,
                    'file_size': 64,
                    'tensors': {
                        'w': {
                            'dtype': 'F32',
                            'shape': [4],
                            'data_offsets': [16, 0],
                        }
                    },
                }
            },
        },
        {
            'version': 1,
            'shards': {
                'a.safetensors': {
                    'header_len': 8,
                    'file_size': 64,
                    'tensors': {
                        'w': {
                            'dtype': 'BAD_DTYPE',
                            'shape': [4],
                            'data_offsets': [0, 16],
                        }
                    },
                }
            },
        },
        {
            'version': 1,
            'shards': {
                'a.safetensors': {
                    'header_len': 8,
                    'file_size': 64,
                    'tensors': {
                        'w': {
                            'dtype': 'F32',
                            'shape': [4],
                            'data_offsets': [0, 8],
                        }
                    },
                }
            },
        },
        {
            'version': 1,
            'shards': {
                'a.safetensors': {
                    'header_len': 8,
                    'file_size': 24,
                    'tensors': {
                        'w': {
                            'dtype': 'F32',
                            'shape': [4],
                            'data_offsets': [0, 16],
                        }
                    },
                }
            },
        },
        {
            'version': 1,
            'shards': {
                'a.safetensors': {
                    'header_len': 8,
                    'file_size': 32,
                    'tensors': {
                        'dup': {
                            'dtype': 'F32',
                            'shape': [4],
                            'data_offsets': [0, 16],
                        }
                    },
                },
                'b.safetensors': {
                    'header_len': 8,
                    'file_size': 32,
                    'tensors': {
                        'dup': {
                            'dtype': 'F32',
                            'shape': [4],
                            'data_offsets': [0, 16],
                        }
                    },
                },
            },
        },
        {
            'version': 1,
            'shards': {
                'a.safetensors': {
                    'header_len': 8,
                    'file_size': 16,
                    'tensors': {'w': 'not-a-dict'},
                }
            },
        },
    ):
      sc_file.write_text(json.dumps(bad_payload))
      self.assertIsNone(safetensors_io.load_sidecar_headers(str(sc_dir)))

    valid_sc = {
        'version': 1,
        'shards': {
            'b.safetensors': {
                'header_len': 64,
                'file_size': 128,
                'tensors': {
                    '__metadata__': {'format': 'pt'},
                    'w': {
                        'dtype': 'F32',
                        'shape': [4],
                        'data_offsets': [0, 16],
                    },
                },
            },
            'a.safetensors': {
                'header_len': 32,
                'file_size': 96,
                'tensors': {},
            },
        },
    }
    sc_file.write_text(json.dumps(valid_sc))
    loaded = safetensors_io.load_sidecar_headers(str(sc_dir))
    self.assertIsNotNone(loaded)
    assert loaded is not None
    self.assertEqual(loaded['b.safetensors'][0], 64)
    self.assertIn('w', loaded['b.safetensors'][1])
    self.assertTrue(safetensors_io.sidecar_matches_weight_map(loaded, {}))
    self.assertFalse(
        safetensors_io.sidecar_matches_weight_map(
            loaded, {'w': 'b.safetensors'}
        )
    )
    self.assertFalse(
        safetensors_io.sidecar_matches_weight_map(
            loaded, {'missing': 'b.safetensors', 'x': 'a.safetensors'}
        )
    )
    with mock.patch.object(
        epath.Path, 'glob', side_effect=AssertionError('glob must be skipped')
    ):
      srcs = safetensors_io.build_range_sources_from_names(
          str(sc_dir), list(loaded)
      )
    self.assertEqual([s.name for s in srcs], ['a.safetensors', 'b.safetensors'])
    with self.assertRaisesRegex(ValueError, 'No safetensors'):
      safetensors_io.build_range_sources_from_names(str(sc_dir), [])
    with (
        mock.patch.object(
            safetensors_io.GsHttpPath, 'resolve_token', return_value='tok'
        ),
        mock.patch.object(
            safetensors_io, 'resolve_gcs_vips', return_value=('10.9.0.1',)
        ),
    ):
      gs_srcs = safetensors_io.build_range_sources_from_names(
          'gs://bkt/ckpt', ['model.safetensors']
      )
      self.assertIsInstance(gs_srcs[0], safetensors_io.GsHttpPath)
    with mock.patch.object(
        safetensors_io.GsHttpPath, 'resolve_token', return_value=''
    ) as mock_tok:
      no_tok_srcs = safetensors_io.build_range_sources_from_names(
          'gs://bkt/ckpt', ['model.safetensors']
      )
      self.assertEqual(mock_tok.call_count, 1)
      self.assertIsInstance(no_tok_srcs[0], safetensors_io.RangeSource)

    # 5. GCS_SHM_MAX_GROUPS bumped to 16384 and GCS_SHM_PATHS_CAP to 4 MiB.
    self.assertEqual(safetensors_io.GCS_SHM_MAX_GROUPS, 16384)
    self.assertEqual(safetensors_io.GCS_SHM_PATHS_CAP, 4 * 1024 * 1024)
    self.assertTrue(
        safetensors_io.GcsShmReadPool.can_handle(4096, 4096, 4096 * 40)
    )
    big_pool = safetensors_io.GcsShmReadPool(1, ('10.9.0.1',), spawn=False)
    try:
      n_groups = 3000
      big_pool.start_batch(
          '/dev/shm/big',
          4096,
          ['/bkt/f'],
          'tok',
          [(0, g, 0, 8, 0) for g in range(n_groups)],
          [1] * n_groups,
          n_groups,
      )
      self.assertEqual(int(big_pool.ctrl.hdr.num_groups), n_groups)
    finally:
      big_pool.shutdown()

  def test_sidecar_skew_multinic_fallback_and_proc_env_whitespace(self):
    # 1. sidecar_matches_weight_map rejects extra tensors, total_size skew,
    #    and file_sizes skew.
    headers = {
        'a.safetensors': (
            16,
            {
                '__metadata__': {'format': 'pt'},
                'w1': {'dtype': 'F32', 'shape': [4], 'data_offsets': [0, 16]},
                'extra': {
                    'dtype': 'F32',
                    'shape': [2],
                    'data_offsets': [16, 24],
                },
            },
        )
    }
    self.assertTrue(
        safetensors_io.sidecar_matches_weight_map(
            headers,
            {'w1': 'a.safetensors', 'extra': 'a.safetensors'},
            total_size=24,
            file_sizes={'a.safetensors': 48},
        )
    )
    self.assertFalse(
        safetensors_io.sidecar_matches_weight_map(
            headers, {'w1': 'a.safetensors'}
        )
    )
    self.assertFalse(
        safetensors_io.sidecar_matches_weight_map(
            headers,
            {'w1': 'a.safetensors', 'extra': 'a.safetensors'},
            total_size=16,
        )
    )
    self.assertFalse(
        safetensors_io.sidecar_matches_weight_map(
            headers, {}, file_sizes={'a.safetensors': 40}
        )
    )

    # 2. _execute_gcs_task drops source_address=None on second half of retries
    #    when bound secondary NIC fails connect/egress.
    pool = safetensors_io.GcsShmReadPool(1, ('199.36.153.8',), spawn=False)
    try:
      data_arr = np.zeros(16, dtype=np.uint8)
      pool.start_batch(
          '/dev/shm/fb', 16, ['/bkt/obj'], 'tok', [(0, 0, 0, 8, 0)], [1], 1
      )
      seen_srcs: list[tuple[str, int] | None] = []

      def _fake_read_once(*args, source_address=None):
        del args
        seen_srcs.append(source_address)
        return 'retry' if source_address is not None else 'done'

      with (
          mock.patch.object(
              safetensors_io, '_gcs_read_http_once', side_effect=_fake_read_once
          ),
          mock.patch.object(safetensors_io.time, 'sleep'),
      ):
        ok = safetensors_io._execute_gcs_task(
            pool.ctrl,
            pool.shm_lock,
            ['/bkt/obj'],
            data_arr,
            0,
            False,
            ('199.36.153.8', '142.250.1.1'),
            source_address=('10.0.1.1', 0),
        )
      self.assertTrue(ok)
      self.assertEqual(
          seen_srcs,
          [('10.0.1.1', 0), ('10.0.1.1', 0), ('10.0.1.1', 0), None],
      )
    finally:
      pool.shutdown()

    # Multi-prefix VIP probe intersects NICs across /16 prefixes.
    with (
        mock.patch.object(
            safetensors_io,
            'discover_gcs_source_ips',
            side_effect=lambda vip: (
                ('10.0.0.1', '10.0.1.1')
                if vip.startswith('199.36.')
                else ('10.0.0.1',)
            ),
        ) as mock_disc,
        mock.patch.object(safetensors_io.GcsShmReadPool, '_spawn_workers'),
    ):
      mixed = safetensors_io.GcsShmReadPool(
          2, ('199.36.153.8', '199.36.153.9', '142.250.1.1'), spawn=True
      )
      try:
        self.assertEqual(mixed.source_ips, ())
        self.assertEqual(mock_disc.call_count, 2)
      finally:
        mixed.shutdown()

    # 3. _wrap_range_sources / ProcPoolPath strips/ignores whitespace proc env.
    fake_root = '/fake_proc_ws/'
    fake_path = epath.Path(f'{fake_root}a.st')
    with (
        mock.patch.dict(safetensors_io.STORAGE_BACKENDS),
        mock.patch.object(_FakeProcPath, 'configure_flags'),
        mock.patch.object(_FakeProcPath, 'spawn_procs') as mock_spawn,
    ):
      safetensors_io.register_storage_backend(fake_root, _fake_storage())
      with mock.patch.dict(os.environ, {'TUNIX_FAKE_PROCS': '   '}):
        srcs = safetensors_io._wrap_range_sources([fake_path])
        self.assertLen(srcs, 1)
        mock_spawn.assert_not_called()
        srcs[0].ensure_procs(1024)
        mock_spawn.assert_called_once_with(0)
      mock_spawn.reset_mock()
      with mock.patch.dict(os.environ, {'TUNIX_FAKE_PROCS': ' 4 '}):
        safetensors_io._wrap_range_sources([fake_path])
        mock_spawn.assert_called_once_with(4)


if __name__ == '__main__':
  absltest.main()
