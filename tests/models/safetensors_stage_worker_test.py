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

"""Unit tests for the zero-JAX `safetensors_stage_worker` leaf module."""

import concurrent.futures
import http.client
import io
import mmap
import os
import sys
import tempfile
import threading
import types
from unittest import mock

from absl.testing import absltest
from tunix.models import safetensors_io
from tunix.models import safetensors_stage_worker as worker


class _FakeConn:

  def __init__(self, responses: list[tuple[int, bytes] | BaseException]):
    self._responses = list(responses)
    self.calls: list[tuple[str, str, bytes, dict[str, str]]] = []
    self.closed = False
    self.timeout = 15.0
    self.sock = mock.MagicMock()

  def request(self, method, path, body=None, headers=None):
    self.calls.append(
        (method, path, bytes(body) if body else b'', dict(headers or {}))
    )
    if self._responses and isinstance(self._responses[0], BaseException):
      raise self._responses.pop(0)

  def getresponse(self):
    outcome = self._responses.pop(0)
    assert isinstance(outcome, tuple)
    status, payload = outcome
    return types.SimpleNamespace(status=status, read=lambda: payload)

  def close(self):
    self.closed = True


class SafetensorsStageWorkerTest(absltest.TestCase):

  def test_sibling_io_path_and_flags(self):
    mod = worker._load_sibling_io()  # pylint: disable=protected-access
    self.assertTrue(hasattr(mod, 'GsHttpPath'))
    self.assertEqual(worker.to_gfile_path('/GOOGLE_INTERNAL_STOAGE_PATH/cell-d/x'), '/GOOGLE_INTERNAL_STOAGE_PATH/cell-d/x')
    flags_argv = worker.configure_stage_write_flags()
    self.assertEqual(safetensors_io.pick_thread_vip(()), '')
    self.assertIn(
        safetensors_io.pick_thread_vip(('10.0.0.1', '10.0.0.2')),
        ('10.0.0.1', '10.0.0.2'),
    )

  def test_gcs_http_upload_compose_delete_and_retry(self):
    conn = _FakeConn([
        BrokenPipeError('EPIPE'),
        (503, b'retry'),
        (200, b''),
        ConnectionResetError('ECONNRESET'),
        (503, b'retry'),
        (200, b'{}'),
        (204, b''),
    ])
    vips_seen: list[str] = []

    def _get_conn(vip, timeout=15.0):
      del timeout
      vips_seen.append(vip)
      return conn

    with (
        mock.patch.object(
            safetensors_io, 'get_thread_https_conn', side_effect=_get_conn
        ) as mock_get_conn,
        mock.patch.object(
            safetensors_io, 'evict_thread_https_conn'
        ) as mock_evict,
        mock.patch.object(worker.time, 'sleep'),
        mock.patch.object(
            safetensors_io.GsHttpPath,
            'discover_vips',
            return_value=('127.0.0.1', '127.0.0.2', '127.0.0.3'),
        ),
        mock.patch.object(
            safetensors_io.GsHttpPath, 'resolve_token', return_value='tok'
        ),
    ):
      safetensors_io.GCS_VIP_CACHE.clear()
      try:
        worker.upload_gcs_slot_http(
            b'payload', 'gs://bkt/dir/a part+0', vip_offset=0
        )
        worker.compose_gcs_objects_http(
            'gs://bkt/dir/a.safetensors',
            ('gs://bkt/dir/a part+0',),
            vip_offset=1,
        )
        worker.delete_gcs_object_http('gs://bkt/dir/a part+0', vip_offset=2)
      finally:
        safetensors_io.GCS_VIP_CACHE.clear()
    self.assertEqual(
        [c[0] for c in conn.calls],
        ['PUT', 'PUT', 'PUT', 'POST', 'POST', 'POST', 'DELETE'],
    )
    self.assertEqual(conn.calls[0][1], '/bkt/dir/a%20part%2B0')
    self.assertEqual(conn.calls[0][3]['Host'], 'storage.googleapis.com')
    self.assertEqual(conn.calls[0][3]['Authorization'], 'Bearer tok')
    self.assertEqual(conn.calls[0][3]['Content-Length'], '7')
    self.assertEqual(conn.calls[0][3]['Connection'], 'keep-alive')
    # EPIPE and 503 each evict the dead connection and rotate to the next VIP.
    self.assertEqual(vips_seen[:3], ['127.0.0.1', '127.0.0.2', '127.0.0.3'])
    self.assertEqual(vips_seen[3:6], ['127.0.0.2', '127.0.0.3', '127.0.0.1'])
    self.assertEqual(mock_evict.call_count, 4)
    mock_get_conn.assert_called_with(mock.ANY, timeout=60.0)

    with self.assertRaises(ValueError):
      worker.compose_gcs_objects_http('gs://bkt/a', ())
    with self.assertRaises(ValueError):
      worker.compose_gcs_objects_http('gs://bkt/a', ('gs://other/p0',))

  def test_arena_stream_drain_and_stdio_loop(self):
    fut_ok: concurrent.futures.Future[None] = concurrent.futures.Future()
    fut_ok.set_result(None)
    fut_err: concurrent.futures.Future[None] = concurrent.futures.Future()
    fut_err.set_exception(RuntimeError('boom'))
    self.assertEqual(worker._drain_worker_futs([fut_ok]), '')  # pylint: disable=protected-access
    self.assertIn('boom', worker._drain_worker_futs([fut_err]))  # pylint: disable=protected-access

    with tempfile.TemporaryDirectory() as tmpdir:
      arena_path = os.path.join(tmpdir, 'arena.bin')
      with open(arena_path, 'wb') as f:
        f.truncate(16384)
      dst_path = os.path.join(tmpdir, 'out.bin')
      self.assertIsNone(
          worker._init_worker_arena('', 0, 1)  # pylint: disable=protected-access
      )
      mm = worker._init_worker_arena(arena_path, 0, 1)  # pylint: disable=protected-access
      assert mm is not None
      mm.close()
      with open(arena_path, 'r+b') as f:
        wmm = mmap.mmap(f.fileno(), 16384, access=mmap.ACCESS_WRITE)
        wmm[:5] = b'hello'
        wmm.flush()
        wmm.close()

      fake_in = io.StringIO(f'SLOT\t0\t5\t3\t{dst_path}\nFLUSH\nQUIT\n')
      fake_out = io.StringIO()
      with (
          mock.patch.object(worker, 'gfile', None),
          mock.patch.dict(
              os.environ,
              {
                  'TUNIX_STAGE_ARENA_PATH': arena_path,
                  'TUNIX_STAGE_WORKER_IDX': '1',
                  'TUNIX_STAGE_NUM_WORKERS': '2',
              },
          ),
          mock.patch.object(sys, 'stdin', fake_in),
          mock.patch.object(sys, 'stdout', fake_out),
      ):
        worker.run_stage_worker_stdio()
      out_lines = fake_out.getvalue().splitlines()
      self.assertIn('READY', out_lines)
      self.assertIn('FREE\t3', out_lines)
      self.assertIn('DONE', out_lines)
      with open(dst_path, 'rb') as rf:
        self.assertEqual(rf.read(), b'hello')

      # Direct evict_thread_https_conn cleanup path.
      dummy_conn = _FakeConn([])
      key = (threading.get_ident(), '127.0.0.1')
      safetensors_io.THREAD_CONNS[key] = dummy_conn  # type: ignore[assignment]
      safetensors_io.evict_thread_https_conn('127.0.0.1')
      self.assertTrue(dummy_conn.closed)

  def test_http_401_refresh_worker_compose_and_buffer_error_free_teardown(self):
    # 1. HTTP 401 forces one metadata refresh and retries cleanly.
    conn_401 = _FakeConn([(401, b'unauth'), (200, b'{}')])
    with (
        mock.patch.object(
            safetensors_io, 'get_thread_https_conn', return_value=conn_401
        ),
        mock.patch.object(
            safetensors_io.GsHttpPath,
            'discover_vips',
            return_value=('127.0.0.1',),
        ),
        mock.patch.object(
            safetensors_io.GsHttpPath,
            'resolve_token',
            side_effect=['stale-tok', 'fresh-tok'],
        ) as mock_tok,
    ):
      safetensors_io.GCS_VIP_CACHE.clear()
      try:
        worker.upload_gcs_slot_http(b'abc', 'gs://bkt/dir/shard.safetensors')
      finally:
        safetensors_io.GCS_VIP_CACHE.clear()
    self.assertEqual(mock_tok.call_count, 2)
    mock_tok.assert_called_with(force_refresh=True)

    # 2. Worker COMPOSE command composes + deletes parts, and failed HTTP SLOT
    # upload still closes arena_mm without raising BufferError.
    with tempfile.TemporaryDirectory() as tmpdir:
      arena_path = os.path.join(tmpdir, 'arena.bin')
      with open(arena_path, 'wb') as f:
        f.truncate(16384)
      fake_in = io.StringIO(
          'COMPOSE\tgs://bkt/dir/out.safetensors\t'
          'gs://bkt/dir/p0\tgs://bkt/dir/p1\n'
          'FLUSH\n'
          'SLOT\t0\t16\t0\tgs://bkt/dir/fail.safetensors\n'
          'FLUSH\n'
          'QUIT\n'
      )
      fake_out = io.StringIO()
      with (
          mock.patch.object(worker, 'gfile', None),
          mock.patch.object(worker, 'compose_gcs_objects_http') as mock_comp,
          mock.patch.object(worker, 'delete_gcs_object_http') as mock_del,
          mock.patch.object(
              worker,
              'upload_gcs_slot_http',
              side_effect=OSError('simulated 401 failure'),
          ),
          mock.patch.dict(
              os.environ,
              {
                  'TUNIX_STAGE_ARENA_PATH': arena_path,
                  'TUNIX_STAGE_WORKER_IDX': '0',
                  'TUNIX_STAGE_NUM_WORKERS': '1',
                  'TUNIX_STAGE_BACKEND': 'gcs_http',
              },
          ),
          mock.patch.object(sys, 'stdin', fake_in),
          mock.patch.object(sys, 'stdout', fake_out),
      ):
        worker.run_stage_worker_stdio()
      mock_comp.assert_called_once_with(
          'gs://bkt/dir/out.safetensors',
          ['gs://bkt/dir/p0', 'gs://bkt/dir/p1'],
          vips=(),
          vip_offset=0,
      )
      self.assertEqual(mock_del.call_count, 2)
      out_lines = fake_out.getvalue().splitlines()
      self.assertIn('DONE', out_lines)
      self.assertTrue(any(l.startswith('ERR\t') for l in out_lines))

  def test_uri_validation_empty_vip_cache_and_missing_arena_slot(self):
    for bad_uri in ('/tmp/not_gcs', 'gs://no_key', 'gs:///missing_bucket'):
      with self.assertRaises(ValueError):
        worker.upload_gcs_slot_http(b'x', bad_uri)
      with self.assertRaises(ValueError):
        worker.delete_gcs_object_http(bad_uri)

    # Empty VIP resolution must not populate the 5-minute cache.
    safetensors_io.GCS_VIP_CACHE.clear()
    try:
      with mock.patch.object(
          safetensors_io.GsHttpPath, 'discover_vips', return_value=()
      ):
        self.assertEqual(safetensors_io.resolve_gcs_vips(), ())
        self.assertEmpty(safetensors_io.GCS_VIP_CACHE)
    finally:
      safetensors_io.GCS_VIP_CACHE.clear()

    # SLOT received without an initialized arena emits FREE and reports ERR.
    fake_in = io.StringIO('SLOT\t0\t8\t5\t/tmp/unused\nFLUSH\nQUIT\n')
    fake_out = io.StringIO()
    with (
        mock.patch.dict(os.environ, {'TUNIX_STAGE_ARENA_PATH': ''}),
        mock.patch.object(sys, 'stdin', fake_in),
        mock.patch.object(sys, 'stdout', fake_out),
    ):
      worker.run_stage_worker_stdio()
    out_lines = fake_out.getvalue().splitlines()
    self.assertIn('FREE\t5', out_lines)
    self.assertTrue(any(l.startswith('ERR\t') for l in out_lines))

    # Error payload snippet is surfaced in OSError on non-retryable HTTP status.
    conn_400 = _FakeConn([(400, b'bad-request-detail')])
    with (
        mock.patch.object(
            safetensors_io, 'get_thread_https_conn', return_value=conn_400
        ),
        mock.patch.object(
            safetensors_io.GsHttpPath, 'resolve_token', return_value='tok'
        ),
    ):
      with self.assertRaisesRegex(OSError, 'bad-request-detail'):
        worker.upload_gcs_slot_http(
            b'x', 'gs://bkt/obj', vips=('127.0.0.1',)
        )

  def test_multi_host_worker_thread_vip_spread(self):
    all_vips = tuple(f'10.1.0.{i}' for i in range(1, 37))
    seen_offsets: list[int] = []
    seen_vips_lists: list[tuple[str, ...]] = []
    rec_lock = threading.Lock()
    with tempfile.TemporaryDirectory() as tmpdir:
      arena_path = os.path.join(tmpdir, 'arena.bin')
      with open(arena_path, 'wb') as f:
        f.truncate(16384)
      for proc_idx in (0, 1):
        for w_idx in (0, 1):
          barrier = threading.Barrier(4)

          def _fake_upload(
              data, dst_path, *, vips=(), token='', vip_offset=None, _b=barrier
          ):
            del data, dst_path, token
            _b.wait(timeout=5.0)
            with rec_lock:
              seen_vips_lists.append(tuple(vips))
              assert vip_offset is not None
              seen_offsets.append(vip_offset)

          cmds = (
              ''.join(f'SLOT\t0\t16\t{i}\tgs://bkt/p{i}\n' for i in range(4))
              + 'FLUSH\nQUIT\n'
          )
          fake_out = io.StringIO()
          with (
              mock.patch.object(worker, 'gfile', None),
              mock.patch.object(
                  worker, 'upload_gcs_slot_http', side_effect=_fake_upload
              ),
              mock.patch.dict(
                  os.environ,
                  {
                      'TUNIX_STAGE_ARENA_PATH': arena_path,
                      'TUNIX_STAGE_PROCESS_IDX': str(proc_idx),
                      'TUNIX_STAGE_WORKER_IDX': str(w_idx),
                      'TUNIX_STAGE_NUM_WORKERS': '16',
                      'TUNIX_STAGE_BACKEND': 'gcs_http',
                      'TUNIX_GCS_VIPS': ','.join(all_vips),
                  },
              ),
              mock.patch.object(sys, 'stdin', io.StringIO(cmds)),
              mock.patch.object(sys, 'stdout', fake_out),
          ):
            worker.run_stage_worker_stdio()
          self.assertIn('DONE', fake_out.getvalue().splitlines())
    self.assertTrue(all(v == all_vips for v in seen_vips_lists))
    expected_offsets = set(range(0, 8)) | set(range(64, 72))
    self.assertEqual(set(seen_offsets), expected_offsets)
    # Every (host, worker, thread) stream across both hosts selects a distinct
    # VIP out of 36 discovered VIPs (zero collision across hosts/workers).
    selected = {
        safetensors_io.pick_thread_vip(all_vips, vip_offset=o)
        for o in seen_offsets
    }
    self.assertLen(selected, 16)

  def test_backend_isolation_gfile_flags_and_1p_gcs_http_pool_workers(self):
    # pylint: disable=protected-access
    with mock.patch.object(worker, 'gfile', object()):
      self.assertEqual(
          worker._resolve_stage_backend({'TUNIX_STAGE_BACKEND': 'gcs_http'}),
          'gcs_http',
      )
      self.assertEqual(
          worker._resolve_stage_backend({'TUNIX_STAGE_BACKEND': 'gfile'}),
          'gfile',
      )
      self.assertEqual(worker._resolve_stage_backend({}), 'gfile')
      with self.assertRaisesRegex(
          ValueError, 'Unsupported TUNIX_STAGE_BACKEND'
      ):
        worker._resolve_stage_backend({'TUNIX_STAGE_BACKEND': 'invalid'})
    with mock.patch.object(worker, 'gfile', None):
      self.assertEqual(worker._resolve_stage_backend({}), 'gcs_http')

    orig_executor = concurrent.futures.ThreadPoolExecutor
    seen_max_workers: list[int] = []

    def _spy_executor(*args, **kwargs):
      seen_max_workers.append(kwargs['max_workers'])
      return orig_executor(*args, **kwargs)

    # 1P (gfile non-None) with gcs_http skips configure_stage_write_flags()
    # and caps pool_workers at 4 for >=16.
    seen_max_workers.clear()
    with (
        mock.patch.object(worker, 'gfile', object()),
        mock.patch.object(worker, 'configure_stage_write_flags') as mock_flags,
        mock.patch.object(
            worker.concurrent.futures,
            'ThreadPoolExecutor',
            side_effect=_spy_executor,
        ),
        mock.patch.dict(
            os.environ,
            {
                'TUNIX_STAGE_ARENA_PATH': '',
                'TUNIX_STAGE_NUM_WORKERS': '48',
                'TUNIX_STAGE_BACKEND': 'gcs_http',
            },
            clear=True,
        ),
        mock.patch.object(sys, 'stdin', io.StringIO('QUIT\n')),
        mock.patch.object(sys, 'stdout', io.StringIO()),
    ):
      worker.run_stage_worker_stdio()
    mock_flags.assert_not_called()
    self.assertEqual(seen_max_workers, [4, 8])

    # 1P gfile backend calls configure_stage_write_flags() with pool_workers=32.
    seen_max_workers.clear()
    with (
        mock.patch.object(worker, 'gfile', object()),
        mock.patch.object(worker, 'configure_stage_write_flags') as mock_flags,
        mock.patch.object(
            worker.concurrent.futures,
            'ThreadPoolExecutor',
            side_effect=_spy_executor,
        ),
        mock.patch.dict(
            os.environ,
            {
                'TUNIX_STAGE_ARENA_PATH': '',
                'TUNIX_STAGE_NUM_WORKERS': '16',
                'TUNIX_STAGE_BACKEND': 'gfile',
            },
            clear=True,
        ),
        mock.patch.object(sys, 'stdin', io.StringIO('QUIT\n')),
        mock.patch.object(sys, 'stdout', io.StringIO()),
    ):
      worker.run_stage_worker_stdio()
    mock_flags.assert_called_once()
    self.assertEqual(seen_max_workers, [32, 8])

    # HTTPException on final retry wraps into OSError; OSError re-raises as-is.
    conn_http_err = _FakeConn([http.client.BadStatusLine('eof')] * 5)
    with (
        mock.patch.object(
            safetensors_io,
            'get_thread_https_conn',
            return_value=conn_http_err,
        ),
        mock.patch.object(safetensors_io, 'evict_thread_https_conn'),
        mock.patch.object(worker.time, 'sleep'),
        mock.patch.object(
            safetensors_io.GsHttpPath, 'resolve_token', return_value='tok'
        ),
    ):
      with self.assertRaisesRegex(OSError, 'GCS HTTP PUT'):
        worker.upload_gcs_slot_http(
            b'x', 'gs://bkt/obj', vips=('127.0.0.1',)
        )
    with self.assertRaises(BrokenPipeError):
      worker._raise_transport_error('PUT', '/b/k', BrokenPipeError('epipe'))

    # VIP TTL refresh refreshes after expiry, evicts retired VIP sockets from
    # THREAD_CONNS while retaining active ones, and falls back on empty DNS.
    retired_conn = _FakeConn([])
    kept_conn = _FakeConn([])
    tid = threading.get_ident()
    safetensors_io.THREAD_CONNS[(tid, '10.1.0.1')] = retired_conn  # type: ignore[assignment]
    safetensors_io.THREAD_CONNS[(tid, '10.2.0.1')] = kept_conn  # type: ignore[assignment]
    try:
      with (
          mock.patch.object(worker.time, 'monotonic', return_value=1000.0),
          mock.patch.object(
              safetensors_io, 'resolve_gcs_vips', return_value=('10.2.0.1',)
          ) as mock_res,
      ):
        # Defer refresh & socket eviction while worker futures are in flight.
        same_vips, same_exp = worker._maybe_refresh_worker_vips(
            ('10.1.0.1', '10.2.0.1'), 500.0, idle=False
        )
        self.assertEqual(same_vips, ('10.1.0.1', '10.2.0.1'))
        self.assertEqual(same_exp, 500.0)
        mock_res.assert_not_called()
        self.assertFalse(retired_conn.closed)
        fresh, exp = worker._maybe_refresh_worker_vips(
            ('10.1.0.1', '10.2.0.1'), 500.0
        )
        self.assertEqual(fresh, ('10.2.0.1',))
        self.assertEqual(exp, 1000.0 + safetensors_io.GCS_VIP_CACHE_TTL_S)
      self.assertTrue(retired_conn.closed)
      self.assertNotIn((tid, '10.1.0.1'), safetensors_io.THREAD_CONNS)
      self.assertFalse(kept_conn.closed)
      self.assertIn((tid, '10.2.0.1'), safetensors_io.THREAD_CONNS)
    finally:
      safetensors_io.THREAD_CONNS.pop((tid, '10.1.0.1'), None)
      safetensors_io.THREAD_CONNS.pop((tid, '10.2.0.1'), None)
    with (
        mock.patch.object(worker.time, 'monotonic', return_value=1000.0),
        mock.patch.object(safetensors_io, 'resolve_gcs_vips', return_value=()),
    ):
      kept, _ = worker._maybe_refresh_worker_vips(('10.1.0.1',), 500.0)
      self.assertEqual(kept, ('10.1.0.1',))

    # Retry backoff applies random jitter to both socket errors and HTTP 503.
    jitter_conn = _FakeConn(
        [BrokenPipeError('epipe'), (503, b'busy'), (200, b'')]
    )
    sleeps: list[float] = []
    with (
        mock.patch.object(
            safetensors_io, 'get_thread_https_conn', return_value=jitter_conn
        ),
        mock.patch.object(safetensors_io, 'evict_thread_https_conn'),
        mock.patch.object(worker.random, 'uniform', side_effect=[0.75, 1.25]),
        mock.patch.object(worker.time, 'sleep', side_effect=sleeps.append),
        mock.patch.object(
            safetensors_io.GsHttpPath, 'resolve_token', return_value='tok'
        ),
    ):
      worker.upload_gcs_slot_http(
          b'x', 'gs://bkt/obj', vips=('127.0.0.1', '127.0.0.2')
      )
    self.assertEqual(sleeps, [0.25 * 0.75, 0.5 * 1.25])
    # pylint: enable=protected-access


if __name__ == '__main__':
  absltest.main()
