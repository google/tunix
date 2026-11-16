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

import asyncio
import concurrent.futures
import http.client
import io
import mmap
import os
import sys
import tempfile
import threading
import time
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

  def test_rapid_grpc_backend_upload_retry_and_fallback(self):
    # pylint: disable=protected-access
    self.assertEqual(
        worker._resolve_stage_backend(
            {'TUNIX_STAGE_BACKEND': 'gcs_rapid_grpc'}
        ),
        'gcs_rapid_grpc',
    )

    class _FakeServiceUnavailableError(Exception):
      pass

    # _load_rapid_grpc_symbols returns tuple when present, None on ImportError.
    fake_mods = {
        'google.cloud.storage.asyncio.async_grpc_client': types.SimpleNamespace(
            AsyncGrpcClient='C'
        ),
        'google.cloud.storage.asyncio.async_appendable_object_writer': (
            types.SimpleNamespace(AsyncAppendableObjectWriter='W')
        ),
        'google.api_core.exceptions': types.SimpleNamespace(
            ServiceUnavailable='E1',
            DeadlineExceeded='E2',
            InternalServerError='E3',
            TooManyRequests='E4',
            ResourceExhausted='E6',
            Aborted='E5',
        ),
    }
    with mock.patch.object(
        worker.importlib, 'import_module', side_effect=fake_mods.__getitem__
    ):
      self.assertEqual(
          worker._load_rapid_grpc_symbols(),
          ('C', 'W', ('E1', 'E2', 'E3', 'E4', 'E6', 'E5')),
      )
    with mock.patch.object(
        worker.importlib, 'import_module', side_effect=ImportError('missing')
    ):
      self.assertIsNone(worker._load_rapid_grpc_symbols())

    # End-to-end mocked AsyncGrpcClient + AsyncAppendableObjectWriter via stdio:
    # verifies stagger, FLUSH_INTERVAL_BYTES, single-append full memoryview
    # (redirect/retry safe), byte-conservation size retry, 0-byte slot,
    # transient retries, bad-URI FREE safety, and transport channel close.
    client_kwargs: list[dict[str, object]] = []
    writer_opts_seen: list[dict[str, object]] = []
    uploaded: dict[str, bytes] = {}
    chunk_lens: dict[str, list[int]] = {}
    abort_closes: list[str] = []
    deleted_objects: list[tuple[str, str]] = []
    transport_closed = [0]
    open_fails = {'retry_open': 1}
    fin_fails = {'retry_fin': 1}
    trunc_fails = {'trunc_once': 1}

    class _FakeTransport:

      async def close(self):
        transport_closed[0] += 1

    class _FakeGrpcClient:

      def __init__(self, **kwargs):
        client_kwargs.append(dict(kwargs))
        self.grpc_client = types.SimpleNamespace(transport=_FakeTransport())

      async def delete_object(self, bucket_name: str, object_name: str):
        deleted_objects.append((bucket_name, object_name))

    class _FakeAppendableWriter:

      def __init__(self, *, client, bucket_name, object_name, writer_options):
        del client
        self.bucket_name = bucket_name
        self.object_name = object_name
        self.is_stream_open = False
        self.persisted_size: int | None = None
        self._buf = bytearray()
        if bucket_name != '_':
          writer_opts_seen.append(dict(writer_options))

      async def open(self):
        if open_fails.get(self.object_name, 0) > 0:
          open_fails[self.object_name] -= 1
          raise _FakeServiceUnavailableError('transient open')
        if self.object_name == 'fatal_obj':
          raise ValueError('bad object name')
        self.is_stream_open = True

      async def append(self, data):
        assert isinstance(data, memoryview)
        chunk_lens.setdefault(self.object_name, []).append(len(data))
        self._buf.extend(data)

      async def close(self, *, finalize_on_close=False):
        if not finalize_on_close:
          self.is_stream_open = False
          abort_closes.append(self.object_name)
          return None
        if fin_fails.get(self.object_name, 0) > 0:
          fin_fails[self.object_name] -= 1
          raise _FakeServiceUnavailableError('transient finalize')
        self.is_stream_open = False
        if trunc_fails.get(self.object_name, 0) > 0:
          trunc_fails[self.object_name] -= 1
          self.persisted_size = len(self._buf) - 2
          return types.SimpleNamespace(size=self.persisted_size)
        self.persisted_size = len(self._buf)
        uploaded[f'gs://{self.bucket_name}/{self.object_name}'] = bytes(
            self._buf
        )
        return types.SimpleNamespace(size=self.persisted_size)

    sleep_calls: list[float] = []

    async def _fake_sleep(delay):
      sleep_calls.append(delay)

    arena_path = os.path.join(self.create_tempdir().full_path, 'arena.bin')
    with open(arena_path, 'wb') as f:
      f.write(b'0123456789ABCDEF' + b'\x00' * 4080)

    local_dst = os.path.join(self.create_tempdir().full_path, 'local_obj')
    cmds = (
        'SLOT\t0\t10\t0\tgs://rapid-bkt/chunked\n'
        'SLOT\t0\t0\t1\tgs://rapid-bkt/empty\n'
        'SLOT\t0\t6\t2\tgs://rapid-bkt/retry_open\n'
        'SLOT\t4\t6\t3\tgs://rapid-bkt/retry_fin\n'
        'SLOT\t0\t6\t4\tgs://rapid-bkt/trunc_once\n'
        'FLUSH\n'
        'SLOT\t0\t4\t5\tgs://rapid-bkt/fatal_obj\n'
        f'SLOT\t0\t4\t6\t{local_dst}\n'
        'FLUSH\n'
        'QUIT\n'
    )
    fake_out = io.StringIO()
    with (
        mock.patch.object(
            worker,
            '_load_rapid_grpc_symbols',
            return_value=(
                _FakeGrpcClient,
                _FakeAppendableWriter,
                (_FakeServiceUnavailableError,),
            ),
        ),
        mock.patch.object(worker, '_RAPID_GRPC_FLUSH_INTERVAL_BYTES', 4),
        mock.patch.object(worker.asyncio, 'sleep', side_effect=_fake_sleep),
        mock.patch.dict(
            os.environ,
            {
                'TUNIX_STAGE_ARENA_PATH': arena_path,
                'TUNIX_STAGE_NUM_WORKERS': '4',
                'TUNIX_STAGE_WORKER_IDX': '5',
                'TUNIX_STAGE_BACKEND': 'gcs_rapid_grpc',
                'TUNIX_GCS_VIPS': '10.0.0.1',
            },
            clear=True,
        ),
        mock.patch.object(sys, 'stdin', io.StringIO(cmds)),
        mock.patch.object(sys, 'stdout', fake_out),
    ):
      worker.run_stage_worker_stdio()

    lines = fake_out.getvalue().splitlines()
    self.assertEqual(lines[0], 'READY')
    self.assertIn('DONE', lines)
    self.assertTrue(
        any(line.startswith('ERR\t') and 'bad object' in line for line in lines)
    )
    for sid in range(7):
      self.assertIn(f'FREE\t{sid}', lines)
    self.assertEqual(client_kwargs, [{'attempt_direct_path': True}])
    self.assertEqual(transport_closed, [1])
    self.assertAlmostEqual(sleep_calls[0], 5 * worker._RAPID_GRPC_STAGGER_SEC)
    self.assertEqual(uploaded['gs://rapid-bkt/chunked'], b'0123456789')
    self.assertEqual(chunk_lens['chunked'], [10])
    self.assertEqual(uploaded['gs://rapid-bkt/empty'], b'')
    self.assertNotIn('empty', chunk_lens)
    self.assertEqual(uploaded['gs://rapid-bkt/retry_open'], b'012345')
    self.assertEqual(uploaded['gs://rapid-bkt/retry_fin'], b'456789')
    self.assertEqual(uploaded['gs://rapid-bkt/trunc_once'], b'012345')
    self.assertIn('retry_fin', abort_closes)
    self.assertEqual(deleted_objects, [('rapid-bkt', 'fatal_obj')])
    self.assertEqual(writer_opts_seen[0], {'FLUSH_INTERVAL_BYTES': 4})
    # Non-`gs://` destinations bypass Rapid and land on the local filesystem.
    with open(local_dst, 'rb') as f:
      self.assertEqual(f.read(), b'0123')

    # Fallback to gcs_http when symbols missing, client init fails, or writer
    # preconditions (e.g. C google_crc32c) fail at startup.
    class _BrokenClient:

      def __init__(self, **_kwargs):
        raise RuntimeError('directpath unavailable')

    class _NoCrcWriter:

      def __init__(self, **_kwargs):
        raise RuntimeError('google_crc32c C extension missing')

    for sym_ret in (
        None,
        (_BrokenClient, _FakeAppendableWriter, (_FakeServiceUnavailableError,)),
        (_FakeGrpcClient, _NoCrcWriter, (_FakeServiceUnavailableError,)),
    ):
      http_uploads: list[tuple[bytes, str]] = []
      fb_out = io.StringIO()
      fb_err = io.StringIO()
      with (
          mock.patch.object(
              worker, '_load_rapid_grpc_symbols', return_value=sym_ret
          ),
          mock.patch.object(
              worker,
              'upload_gcs_slot_http',
              side_effect=lambda data, dst, uploads=http_uploads, **_kw: (
                  uploads.append((bytes(data), dst))
              ),
          ),
          mock.patch.dict(
              os.environ,
              {
                  'TUNIX_STAGE_ARENA_PATH': arena_path,
                  'TUNIX_STAGE_NUM_WORKERS': '2',
                  'TUNIX_STAGE_WORKER_IDX': '0',
                  'TUNIX_STAGE_BACKEND': 'gcs_rapid_grpc',
                  'TUNIX_GCS_VIPS': '10.0.0.1',
              },
              clear=True,
          ),
          mock.patch.object(
              sys,
              'stdin',
              io.StringIO('SLOT\t0\t4\t0\tgs://rapid-bkt/fb\nFLUSH\nQUIT\n'),
          ),
          mock.patch.object(sys, 'stdout', fb_out),
          mock.patch.object(sys, 'stderr', fb_err),
      ):
        worker.run_stage_worker_stdio()
      self.assertEqual(http_uploads, [(b'0123', 'gs://rapid-bkt/fb')])
      self.assertIn('DONE', fb_out.getvalue().splitlines())
      if sym_ret is not None:
        self.assertIn('falling back to gcs_http', fb_err.getvalue())
    # pylint: enable=protected-access

  def test_rapid_grpc_write_hardening_timeouts_resource_exhausted_and_jitter(
      self,
  ):
    # pylint: disable=protected-access
    self.assertEqual(worker._RAPID_GRPC_MAX_ATTEMPTS, 6)
    self.assertEqual(worker._RAPID_GRPC_RPC_TIMEOUT_SEC, 15.0)
    self.assertEqual(worker._RAPID_GRPC_ABORT_CLOSE_TIMEOUT_SEC, 2.0)
    self.assertLess(
        worker._RAPID_GRPC_INIT_TIMEOUT_SEC
        + worker._RAPID_GRPC_ABORT_CLOSE_TIMEOUT_SEC,
        30.0,
    )

    class _FakeResourceExhaustedError(Exception):
      pass

    # Verify GOAWAY & resource_exhausted retries vs non-retryable ValueError.
    token_attempts: list[int] = []

    class _TokenRetryWriter:

      def __init__(self, *, object_name, **_kwargs):
        self.object_name = object_name
        self.is_stream_open = False
        self.persisted_size: int | None = None

      async def open(self):
        idx = len(token_attempts)
        token_attempts.append(idx)
        if self.object_name == 'non_retry':
          raise ValueError('invalid object aborted reset')
        if idx == 0:
          raise RuntimeError('HTTP/2 GOAWAY')
        if idx == 1:
          raise RuntimeError('quota resource exhausted on bucket')
        self.is_stream_open = True

      async def append(self, data):
        self.persisted_size = len(data)

      async def close(self, *, finalize_on_close=False):
        self.is_stream_open = False
        if not finalize_on_close:
          return None
        return types.SimpleNamespace(size=self.persisted_size)

    tok_uploader = worker._RapidGrpcUploader(worker_idx=0)
    tok_uploader._writer_cls = _TokenRetryWriter
    tok_uploader._retryable_excs = (_FakeResourceExhaustedError,)

    async def _noop_sleep(delay: float) -> None:
      del delay

    with mock.patch.object(worker.asyncio, 'sleep', side_effect=_noop_sleep):
      with memoryview(b'abc') as mv:
        asyncio.run(tok_uploader._upload_with_retry('bkt', 'tok_ok', mv))
      self.assertEqual(token_attempts, [0, 1, 2])
      token_attempts.clear()
      with self.assertRaises(ValueError):
        with memoryview(b'abc') as mv:
          asyncio.run(tok_uploader._upload_with_retry('bkt', 'non_retry', mv))
      self.assertEqual(token_attempts, [0])

    # 1. Verify exact per-RPC timeouts: 15s open/close, size-scaled append
    #    (128 MiB -> 17s, 2 GiB -> 47s), and 2.0s error-path abort close.
    seen_timeouts: list[float | None] = []
    orig_wait_for = asyncio.wait_for

    async def _spy_wait_for(aw, timeout):
      seen_timeouts.append(timeout)
      return await orig_wait_for(aw, timeout=timeout)

    class _TimeoutSpyWriter:

      def __init__(self, *, client, bucket_name, object_name, writer_options):
        del client, bucket_name, writer_options
        self.object_name = object_name
        self.is_stream_open = False
        self.persisted_size: int | None = None

      async def open(self):
        self.is_stream_open = True

      async def append(self, data):
        if self.object_name == 'fail_append':
          raise _FakeResourceExhaustedError('throttled')
        self.persisted_size = len(data)

      async def close(self, *, finalize_on_close=False):
        self.is_stream_open = False
        if not finalize_on_close:
          return None
        return types.SimpleNamespace(size=self.persisted_size)

    class _SizedBuf:

      def __init__(self, size: int):
        self._size = size

      def __len__(self) -> int:
        return self._size

    uploader = worker._RapidGrpcUploader(worker_idx=0)
    uploader._writer_cls = _TimeoutSpyWriter
    uploader._retryable_excs = (_FakeResourceExhaustedError,)

    with mock.patch.object(
        worker.asyncio, 'wait_for', side_effect=_spy_wait_for
    ):
      mm_128 = mmap.mmap(-1, 128 << 20)
      try:
        with memoryview(mm_128) as mv_128:
          asyncio.run(uploader._upload_once('bkt', 'obj_128m', mv_128))
      finally:
        mm_128.close()
      self.assertEqual(seen_timeouts, [15.0, 17.0, 15.0])

      seen_timeouts.clear()
      asyncio.run(
          uploader._upload_once('bkt', 'obj_2g', _SizedBuf(2 << 30))  # type: ignore[arg-type]
      )
      self.assertEqual(seen_timeouts, [15.0, 47.0, 15.0])

      seen_timeouts.clear()
      with self.assertRaises(_FakeResourceExhaustedError):
        asyncio.run(
            uploader._upload_once('bkt', 'fail_append', memoryview(b'abcdef'))
        )
      self.assertEqual(seen_timeouts[0], 15.0)
      self.assertEqual(seen_timeouts[-1], 2.0)

    # 2. Real asyncio.wait_for timeouts across open, append (+ wedged abort
    #    close), finalize-close, and ResourceExhausted retries succeed on the
    #    6th attempt (attempt index 5).
    attempts_seen: list[int] = []
    abort_close_calls = [0]
    uploaded_payload: list[bytes] = []

    class _FlakyTimeoutWriter:

      def __init__(self, *, client, bucket_name, object_name, writer_options):
        del client, bucket_name, object_name, writer_options
        self._attempt = len(attempts_seen)
        attempts_seen.append(self._attempt)
        self.is_stream_open = False
        self.persisted_size: int | None = None
        self._buf = bytearray()

      async def open(self):
        if self._attempt == 0:
          await asyncio.Event().wait()
        if self._attempt == 3:
          raise _FakeResourceExhaustedError('open resource exhausted')
        self.is_stream_open = True

      async def append(self, data):
        if self._attempt == 1:
          await asyncio.Event().wait()
        if self._attempt == 4:
          raise RuntimeError('RESOURCE_EXHAUSTED: append quota')
        self._buf.extend(data)

      async def close(self, *, finalize_on_close=False):
        if not finalize_on_close:
          abort_close_calls[0] += 1
          if self._attempt == 1:
            await asyncio.Event().wait()
          self.is_stream_open = False
          return None
        if self._attempt == 2:
          await asyncio.Event().wait()
        self.is_stream_open = False
        self.persisted_size = len(self._buf)
        uploaded_payload.append(bytes(self._buf))
        return types.SimpleNamespace(size=self.persisted_size)

    retry_delays: list[float] = []

    async def _rec_sleep(delay: float) -> None:
      retry_delays.append(delay)

    uploader._writer_cls = _FlakyTimeoutWriter
    with (
        mock.patch.object(worker, '_RAPID_GRPC_RPC_TIMEOUT_SEC', 0.03),
        mock.patch.object(worker, '_RAPID_GRPC_ABORT_CLOSE_TIMEOUT_SEC', 0.03),
        mock.patch.object(worker.asyncio, 'sleep', side_effect=_rec_sleep),
    ):
      with memoryview(b'hardened') as mv:
        asyncio.run(uploader._upload_with_retry('bkt', 'hard', mv))

    self.assertEqual(attempts_seen, [0, 1, 2, 3, 4, 5])
    self.assertGreaterEqual(abort_close_calls[0], 3)
    self.assertEqual(uploaded_payload, [b'hardened'])
    self.assertLen(retry_delays, 5)
    for attempt_idx, delay in enumerate(retry_delays):
      base = min(1.0, 0.05 * (1 << attempt_idx))
      self.assertGreaterEqual(delay, 0.5 * base)
      self.assertLessEqual(delay, 1.5 * base)

    # 3. Exhausting all 6 attempts raises after 5 jittered sleeps, and
    #    attempts >= 5 cap the base backoff at 1.0s (jittered [0.5, 1.5]).
    class _AlwaysExhaustedWriter:

      def __init__(self, **_kwargs):
        self.is_stream_open = False
        self.persisted_size: int | None = None

      async def open(self):
        raise _FakeResourceExhaustedError('still throttled')

    uploader._writer_cls = _AlwaysExhaustedWriter
    a0_samples: list[float] = []
    for _ in range(5):
      retry_delays.clear()
      with mock.patch.object(
          worker.asyncio, 'sleep', side_effect=_rec_sleep
      ):
        with self.assertRaises(_FakeResourceExhaustedError):
          with memoryview(b'x') as mv:
            asyncio.run(uploader._upload_with_retry('bkt', 'exhaust', mv))
      self.assertLen(retry_delays, 5)
      for a_idx, delay in enumerate(retry_delays):
        base = min(1.0, 0.05 * (1 << a_idx))
        self.assertGreaterEqual(delay, 0.5 * base)
        self.assertLessEqual(delay, 1.5 * base)
      a0_samples.append(retry_delays[0])
    self.assertGreater(len(set(a0_samples)), 1)

    retry_delays.clear()
    with (
        mock.patch.object(worker, '_RAPID_GRPC_MAX_ATTEMPTS', 7),
        mock.patch.object(worker.asyncio, 'sleep', side_effect=_rec_sleep),
    ):
      with self.assertRaises(_FakeResourceExhaustedError):
        with memoryview(b'x') as mv:
          asyncio.run(uploader._upload_with_retry('bkt', 'capped', mv))
    self.assertLen(retry_delays, 6)
    self.assertGreaterEqual(retry_delays[5], 0.5)
    self.assertLessEqual(retry_delays[5], 1.5)
    # pylint: enable=protected-access

  def test_rapid_grpc_inner_cancelled_error_retries_but_real_cancel_reraises(
      self,
  ):
    # pylint: disable=protected-access
    uploader = worker._RapidGrpcUploader(worker_idx=0)
    attempts: list[int] = []
    uploaded: list[bytes] = []

    class _InnerCancelWriter:

      def __init__(self, **_kwargs):
        self._attempt = len(attempts)
        attempts.append(self._attempt)
        self.is_stream_open = False
        self.persisted_size: int | None = None
        self._buf = bytearray()

      async def open(self):
        self.is_stream_open = True

      async def append(self, data):
        if self._attempt < 2:
          raise asyncio.CancelledError('bidi stream reset')
        self._buf.extend(data)

      async def close(self, *, finalize_on_close=False):
        self.is_stream_open = False
        if finalize_on_close:
          self.persisted_size = len(self._buf)
          uploaded.append(bytes(self._buf))
          return types.SimpleNamespace(size=self.persisted_size)
        return None

    uploader._writer_cls = _InnerCancelWriter
    retry_delays: list[float] = []

    async def _rec_sleep(delay: float) -> None:
      retry_delays.append(delay)

    with mock.patch.object(worker.asyncio, 'sleep', side_effect=_rec_sleep):
      with memoryview(b'ok') as mv:
        asyncio.run(uploader._upload_with_retry('bkt', 'inner_cancel', mv))

    self.assertEqual(attempts, [0, 1, 2])
    self.assertEqual(uploaded, [b'ok'])
    self.assertLen(retry_delays, 2)

    # Real external task cancellation must propagate immediately without retry.
    class _SlowWriter:

      def __init__(self, **_kwargs):
        self.is_stream_open = True
        self.persisted_size: int | None = None

      async def open(self):
        await asyncio.Event().wait()

      async def close(self, *, finalize_on_close=False):
        del finalize_on_close
        self.is_stream_open = False
        return None

    uploader._writer_cls = _SlowWriter
    retry_delays.clear()

    async def _run_real_cancel() -> None:
      with memoryview(b'cancel') as mv:
        task = asyncio.create_task(
            uploader._upload_with_retry('bkt', 'real_cancel', mv)
        )
        await asyncio.sleep(0)
        task.cancel()
        await task

    with self.assertRaises(asyncio.CancelledError):
      asyncio.run(_run_real_cancel())
    self.assertEmpty(retry_delays)
    # pylint: enable=protected-access

  def test_rapid_grpc_init_timeout_fallback_half_open_stream_and_delete(self):
    # pylint: disable=protected-access
    # 1. Hanging Rapid gRPC init falls back cleanly to gcs_http before READY
    #    timeout (with bounded init + close timeouts).
    unblock_client = threading.Event()

    class _HangingClient:

      def __init__(self, **_kwargs):
        unblock_client.wait(timeout=5.0)

    class _DummyWriter:

      def __init__(self, **_kwargs):
        pass

    arena_path = os.path.join(self.create_tempdir().full_path, 'arena_hang.bin')
    with open(arena_path, 'wb') as f:
      f.write(b'ABCD' + b'\x00' * 4092)
    http_uploads: list[tuple[bytes, str]] = []
    fb_out = io.StringIO()
    fb_err = io.StringIO()
    t0 = time.monotonic()
    try:
      with (
          mock.patch.object(
              worker,
              '_load_rapid_grpc_symbols',
              return_value=(_HangingClient, _DummyWriter, ()),
          ),
          mock.patch.object(worker, '_RAPID_GRPC_INIT_TIMEOUT_SEC', 0.05),
          mock.patch.object(
              worker, '_RAPID_GRPC_ABORT_CLOSE_TIMEOUT_SEC', 0.1
          ),
          mock.patch.object(
              worker,
              'upload_gcs_slot_http',
              side_effect=lambda data, dst, **_kw: http_uploads.append(
                  (bytes(data), dst)
              ),
          ),
          mock.patch.dict(
              os.environ,
              {
                  'TUNIX_STAGE_ARENA_PATH': arena_path,
                  'TUNIX_STAGE_NUM_WORKERS': '1',
                  'TUNIX_STAGE_WORKER_IDX': '0',
                  'TUNIX_STAGE_BACKEND': 'gcs_rapid_grpc',
                  'TUNIX_GCS_VIPS': '10.0.0.1',
              },
              clear=True,
          ),
          mock.patch.object(
              sys,
              'stdin',
              io.StringIO('SLOT\t0\t4\t0\tgs://bkt/hang_fb\nFLUSH\nQUIT\n'),
          ),
          mock.patch.object(sys, 'stdout', fb_out),
          mock.patch.object(sys, 'stderr', fb_err),
      ):
        worker.run_stage_worker_stdio()
    finally:
      unblock_client.set()
    self.assertLess(time.monotonic() - t0, 2.0)
    self.assertEqual(fb_out.getvalue().splitlines()[0], 'READY')
    self.assertEqual(http_uploads, [(b'ABCD', 'gs://bkt/hang_fb')])
    self.assertIn('falling back to gcs_http', fb_err.getvalue())

    # 2. Half-open write_obj_stream (`is_stream_open=False`, matching real
    #    _AsyncWriteObjectStream whose close() raises ValueError) closes its
    #    underlying `socket_like_rpc` on open() timeout, and terminal error
    #    triggers best-effort delete_object (ignoring delete errors).
    rpc_closed = [0]
    raw_closed = [0]
    deleted: list[tuple[str, str]] = []

    class _FakeBidiRpc:

      async def close(self):
        rpc_closed[0] += 1

    class _RawStream:

      def __init__(self, *, is_stream_open: bool = False):
        self.is_stream_open = is_stream_open
        self.socket_like_rpc = _FakeBidiRpc()

      async def close(self):
        if not self.is_stream_open:
          raise ValueError('Stream is not open')
        raw_closed[0] += 1

    class _HalfOpenTimeoutWriter:

      def __init__(self, **_kwargs):
        self.is_stream_open = False
        self.write_obj_stream = None
        self.persisted_size: int | None = None

      async def open(self):
        self.write_obj_stream = _RawStream(is_stream_open=False)
        await asyncio.Event().wait()

    class _DeleteClient:

      async def delete_object(self, bucket: str, key: str):
        deleted.append((bucket, key))
        raise RuntimeError('404 not found')

    uploader = worker._RapidGrpcUploader(worker_idx=0)
    uploader._client = _DeleteClient()
    uploader._writer_cls = _HalfOpenTimeoutWriter
    uploader._retryable_excs = ()

    async def _noop_sleep(delay: float) -> None:
      del delay

    with (
        mock.patch.object(worker, '_RAPID_GRPC_MAX_ATTEMPTS', 2),
        mock.patch.object(worker, '_RAPID_GRPC_RPC_TIMEOUT_SEC', 0.02),
        mock.patch.object(worker, '_RAPID_GRPC_ABORT_CLOSE_TIMEOUT_SEC', 0.1),
        mock.patch.object(worker.asyncio, 'sleep', side_effect=_noop_sleep),
    ):
      with self.assertRaises(TimeoutError):
        with memoryview(b'1234') as mv:
          asyncio.run(uploader._upload_with_retry('bkt', 'half_open', mv))
    self.assertEqual(rpc_closed, [2])
    self.assertEqual(raw_closed, [0])
    self.assertEqual(deleted, [('bkt', 'half_open')])

    # Stalled writer.close(finalize_on_close=False) also cancels socket_like_rpc,
    # and an already-open raw write_obj_stream closes via raw_stream.close().
    class _StalledCloseWriter:

      def __init__(self, **_kwargs):
        self.is_stream_open = True
        self.write_obj_stream = _RawStream(is_stream_open=True)
        self.persisted_size: int | None = None

      async def open(self):
        raise OSError('mid-stream failure')

      async def close(self, *, finalize_on_close=False):
        del finalize_on_close
        await asyncio.Event().wait()

    class _RawOpenWriter:

      def __init__(self, **_kwargs):
        self.is_stream_open = False
        self.write_obj_stream = _RawStream(is_stream_open=True)
        self.persisted_size: int | None = None

      async def open(self):
        raise RuntimeError('failed after raw_stream.open')

    with mock.patch.object(worker, '_RAPID_GRPC_ABORT_CLOSE_TIMEOUT_SEC', 0.02):
      uploader._writer_cls = _StalledCloseWriter
      with self.assertRaises(OSError):
        with memoryview(b'1234') as mv:
          asyncio.run(uploader._upload_once('bkt', 'stalled_close', mv))
      uploader._writer_cls = _RawOpenWriter
      with self.assertRaises(RuntimeError):
        with memoryview(b'1234') as mv:
          asyncio.run(uploader._upload_once('bkt', 'raw_open', mv))
    self.assertEqual(rpc_closed, [3])
    self.assertEqual(raw_closed, [1])

    # 3. persisted is None with total > 0 raises OSError (size mismatch).
    class _NonePersistedWriter:

      def __init__(self, **_kwargs):
        self.is_stream_open = False
        self.persisted_size: int | None = None

      async def open(self):
        self.is_stream_open = True

      async def append(self, data):
        del data

      async def close(self, *, finalize_on_close=False):
        self.is_stream_open = False
        return None if finalize_on_close else None

    uploader._writer_cls = _NonePersistedWriter
    with self.assertRaisesRegex(OSError, 'persisted None != expected 4'):
      with memoryview(b'1234') as mv:
        asyncio.run(uploader._upload_once('bkt', 'none_persisted', mv))
    with memoryview(b'') as mv:
      asyncio.run(uploader._upload_once('bkt', 'empty_ok', mv))
    # pylint: enable=protected-access


if __name__ == '__main__':
  absltest.main()
