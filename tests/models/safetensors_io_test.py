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
import mmap
import os
import pathlib
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


class SafetensorsIoTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.temp_dir = self.create_tempdir().full_path
    safetensors_io.CNS_PROCS.clear()
    safetensors_io.THREAD_CONNS.clear()
    safetensors_io.GCS_TOKEN_CACHE.clear()

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

  def test_GOOGLE_INTERNAL_STOAGE_PATH_path_flags_and_worker_read_protocol(self):
    self.assertIsInstance(safetensors_io.configure_GOOGLE_INTERNAL_STOAGE_PATH_flags(), tuple)

    raw_file = pathlib.Path(self.temp_dir) / 'GOOGLE_INTERNAL_STOAGE_PATH_shard.safetensors'
    raw_file.write_bytes(b'ABCDEFGHIJKLMNOP')
    with safetensors_io.ShmArena(size_bytes=65536) as arena:
      shm_mmaps = {}
      handles = {}
      lock = threading.Lock()
      out_buf = io.StringIO()
      read_cmd = f'{arena.shm_path}\t{arena.capacity}\t{raw_file}\t4\t8\t128\t7'
      with mock.patch.object(sys, 'stdout', out_buf):
        safetensors_io.CnsPath.run_worker_cmd(
            read_cmd, shm_mmaps, handles, lock
        )
      for fh in handles.values():
        fh.close()
      safetensors_io.CnsPath._close_shm_mmaps(shm_mmaps)
      self.assertEqual(out_buf.getvalue(), '7\t8\n')
      assert arena.shm_uint8 is not None
      self.assertEqual(bytes(arena.shm_uint8[128:136]), b'EFGHIJKL')

  def test_GOOGLE_INTERNAL_STOAGE_PATH_worker_loop_and_proc_multiplexing(self):
    with (
        mock.patch.object(sys, 'stdin', io.StringIO('\n')),
        mock.patch.object(sys, 'stdout', io.StringIO()) as fake_out,
    ):
      safetensors_io.CnsPath.worker_loop()
      self.assertIn('READY', fake_out.getvalue())

    GOOGLE_INTERNAL_STOAGE_PATH_src = safetensors_io.CnsPath(
        epath.Path(self.temp_dir) / 'a.safetensors'
    )
    with mock.patch.object(safetensors_io.CnsPath, 'spawn_procs') as mock_spawn:
      with mock.patch.dict(os.environ, {'TUNIX_CNS_PROCS': '4'}):
        GOOGLE_INTERNAL_STOAGE_PATH_src.ensure_procs(1024)
      mock_spawn.assert_called_with(4)
      with mock.patch.dict(os.environ, {}, clear=False):
        os.environ.pop('TUNIX_CNS_PROCS', None)
        GOOGLE_INTERNAL_STOAGE_PATH_src.ensure_procs(128 * 1024 * 1024)
        mock_spawn.assert_called_with(2)
        GOOGLE_INTERNAL_STOAGE_PATH_src.ensure_procs(1024)
        mock_spawn.assert_called_with(0)

    fake_proc = mock.MagicMock()
    fake_proc.stdin = io.StringIO()
    fake_proc.stdout.readline.side_effect = [
        'READY\n',
        '1\t8\n',
    ]
    with mock.patch.object(subprocess, 'Popen', return_value=fake_proc):
      safetensors_io.CnsPath.spawn_procs(1)
      self.assertLen(safetensors_io.CNS_PROCS, 1)
      cf = GOOGLE_INTERNAL_STOAGE_PATH_src.open('rb', stream_id=0)
      with mock.patch.object(safetensors_io, '_REQ_COUNTER', iter([1])):
        buf = np.zeros(16, dtype=np.uint8)
        self.assertEqual(cf.read_shm_range(0, 8, buf, 0, '/dev/shm/x', 64), 8)
      safetensors_io.CnsPath.spawn_procs(0)
      self.assertEmpty(safetensors_io.CNS_PROCS)

    bad_proc = mock.MagicMock()
    bad_proc.stdout.readline.return_value = 'FAIL\n'
    with mock.patch.object(subprocess, 'Popen', return_value=bad_proc):
      with self.assertRaisesRegex(OSError, 'failed to start'):
        safetensors_io.CnsPath.spawn_procs(1)

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
        safetensors_io.CnsPath.worker_loop()
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
      results[idx] = safetensors_io.CnsPath.proc_read(proc_entry, f'CMD_{idx}')

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
    safetensors_io.CNS_PROCS[:] = [(unmap_proc, threading.Lock())]
    with mock.patch.object(safetensors_io, '_REQ_COUNTER', iter([10])):
      with safetensors_io.ShmArena(shm_budget_bytes=8192) as arena:
        arena._used_procs = True
    self.assertIn('UNMAP\t10\n', unmap_proc.stdin.getvalue())
    safetensors_io.CNS_PROCS.clear()
    safetensors_io.CnsPath.release_proc_mmaps()

    # Verify Python vs bundled binary worker argv construction.
    with (
        mock.patch.object(safetensors_io, '_g3_mp_lib', None),
        mock.patch.object(sys, 'executable', '/usr/bin/python3'),
    ):
      cmd_py = safetensors_io.build_worker_cmd(
          safetensors_io.__file__,
          'from tunix.models import safetensors_io as s; '
          's.CnsPath.worker_loop()',
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
          's.CnsPath.worker_loop()',
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
        safetensors_io.CnsPath.worker_loop()
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
      safetensors_io.CnsPath._close_shm_mmaps(mmaps, ignore_busy=True)
      self.assertIn('/dev/shm/busy', mmaps)
    safetensors_io.CnsPath._close_shm_mmaps(mmaps, ignore_busy=False)
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
    safetensors_io.CNS_PROCS[:] = [
        (dead_proc, threading.Lock()),
        (live_proc, threading.Lock()),
    ]
    safetensors_io.CnsPath.spawn_procs(1)
    self.assertLen(safetensors_io.CNS_PROCS, 1)
    safetensors_io.CnsPath.spawn_procs(0)
    self.assertEmpty(safetensors_io.CNS_PROCS)
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
            side_effect=lambda vip, timeout=15.0: _FakeConn(),
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

    # 4. GsHttpPath.open never hijacks CNS_PROCS.
    safetensors_io.CNS_PROCS.append((mock.MagicMock(), threading.Lock()))
    try:
      gs_fh = safetensors_io.GsHttpPath(
          'gs://bkt/a.safetensors', 'tok', ('10.0.0.1',)
      ).open('rb', stream_id=0)
      self.assertIsNone(gs_fh.proc_entry)
      gs_fh.close()
    finally:
      safetensors_io.CNS_PROCS.clear()

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

      def _fake_read(*args):
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


if __name__ == '__main__':
  absltest.main()
