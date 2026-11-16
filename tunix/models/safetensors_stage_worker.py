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

"""Zero-JAX Layer-3 GCS/CNS HTTP transport and `/dev/shm` stage worker leaf.

Executing this module directly (`python3 -u
path/to/safetensors_stage_worker.py`)
loads only its sibling `safetensors_io.py` and `numpy` without triggering
`tunix/__init__.py`, `jax`, `flax`, or `orbax`, reducing 48-process pool startup
from ~26s to <0.5s on GCP TPU hosts.
"""

# pylint: disable=g-bare-generic

import asyncio
from collections.abc import Iterator, Mapping, Sequence
import concurrent.futures
import contextlib
import gc
import http.client
import importlib
import importlib.util
import inspect
import itertools
import json
import mmap
import os
import random
import sys
import threading
import time
from typing import Any, NoReturn
import urllib.parse


def _load_sibling_io() -> Any:
  """Loads sibling `safetensors_io.py` without importing `tunix` or `jax`."""
  io_path = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), 'safetensors_io.py'
  )
  spec = importlib.util.spec_from_file_location(
      '_tunix_safetensors_io', io_path
  )
  if spec is None or spec.loader is None:
    raise ImportError(f'Cannot load sibling safetensors_io from {io_path!r}.')
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


if __package__:
  from tunix.models import safetensors_io  # pylint: disable=g-import-not-at-top
else:
  safetensors_io: Any = _load_sibling_io()

gfile: Any = None

# GCS JSON API compose endpoint caps sourceObjects at 32 components per request.
_MAX_GCS_COMPOSE_PARTS = 32
_GS_GFILE_PREFIX = 'gs://'
_CNS_WRITE_FLAGS: tuple[tuple[str, str], ...] = ()

# Transient GCS HTTP statuses safe to retry with exponential backoff.
_GCS_RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
# 128 MiB flush cadence matches default wave bucket size (1 flush per wave).
_RAPID_GRPC_FLUSH_INTERVAL_BYTES = 128 * 1024 * 1024
# 3 concurrent streams/worker x 48 workers saturates NIC without HBM/RSS spike.
_RAPID_GRPC_INFLIGHT = 3
# 15 ms stagger spreads 48-worker DirectPath handshakes over ~0.72 s.
_RAPID_GRPC_STAGGER_SEC = 0.015
# 6 attempts absorb transient DirectPath GOAWAY / RESOURCE_EXHAUSTED bursts.
_RAPID_GRPC_MAX_ATTEMPTS = 6
# 15 s base per-RPC timeout bounds stalled DirectPath open/finalize calls.
_RAPID_GRPC_RPC_TIMEOUT_SEC = 15.0
# 10 s init + 2 s abort close stays strictly below parent's 30 s READY deadline.
_RAPID_GRPC_INIT_TIMEOUT_SEC = 10.0
# 2 s cap on abort/fallback stream and client teardown so retries stay bounded.
_RAPID_GRPC_ABORT_CLOSE_TIMEOUT_SEC = 2.0
_RAPID_RETRY_TOKENS = (
    'unavailable',
    'goaway',
    'deadline',
    'resource_exhausted',
    'resource exhausted',
)


class _WorkerTls(threading.local):
  vip_offset: int | None = None


_WORKER_TLS = _WorkerTls()


def _init_vip_thread(vip_base: int, tid_counter: Iterator[int]) -> None:
  """Assigns each worker pool thread a deterministic VIP offset."""
  _WORKER_TLS.vip_offset = vip_base + next(tid_counter)


def to_gfile_path(path: str) -> str:
  """Normalizes `gs://` URIs for direct 1P `gfile` calls.

  Leaves local filesystem and `/GOOGLE_INTERNAL_STOAGE_PATH/` paths unchanged while rewriting `gs://`
  URIs to the 1P gfile URI prefix expected by the native gfile backend.

  Args:
    path: Filesystem path or `gs://bucket/object` URI.

  Returns:
    Path string suitable for passing to `gfile.GFile` or local `open()`.

  Example:
    ```python
    from tunix.models import safetensors_stage_worker

    gfile_path = safetensors_stage_worker.to_gfile_path(
        'gs://my-bucket/ckpts/model.safetensors'
    )
    ```
  """
  return f'{_GS_GFILE_PREFIX}{path[5:]}' if path.startswith('gs://') else path


def configure_stage_write_flags() -> tuple[str, ...]:
  """Applies shared storage flags plus distributed storage write flags.

  Configures in-process `absl.flags.FLAGS` entries that are registered in the
  current binary and returns `--undefok=...` plus `--<flag>=<val>` argv tokens
  for forwarding to child stage-worker subprocesses.

  Returns:
    Tuple of CLI flag strings ready to append to a worker subprocess command.

  Example:
    ```python
    from tunix.models import safetensors_stage_worker

    flag_argv = safetensors_stage_worker.configure_stage_write_flags()
    cmd = ['/path/to/binary.par', *flag_argv]
    ```
  """
  safetensors_io.configure_GOOGLE_INTERNAL_STOAGE_PATH_flags()
  fl: Mapping[str, Any] = (
      sys.modules['absl.flags'].FLAGS if 'absl.flags' in sys.modules else {}
  )
  for name, val in _CNS_WRITE_FLAGS:
    if name in fl:
      fl[name].parse(val)
  # Deduplicate overlapping keys so write-side QoS/buffer flags override read
  # defaults cleanly without emitting duplicate `--undefok` or CLI flags.
  all_flags = tuple(
      dict((*safetensors_io.CNS_TUNED_FLAGS, *_CNS_WRITE_FLAGS)).items()
  )
  undefok = ','.join(k for k, _ in all_flags)
  return (
      (f'--undefok={undefok}', *(f'--{k}={v}' for k, v in all_flags))
      if all_flags
      else ()
  )


def _maybe_refresh_auth_on_401(
    status: int,
    explicit_token: str,
    refreshed_auth: bool,
    failed_token: str,
    token_gen: int,
    req_hdrs: dict[str, str],
) -> tuple[bool, str]:
  """Forces one deduplicated GCE metadata token refresh on HTTP 401."""
  # Observed in 397B multi-sync runs: when a cached OAuth2 token expired across
  # long training steps, HTTP 401 failed immediately without refreshing.
  if status == 401 and not explicit_token and not refreshed_auth:
    new_tok = safetensors_io.GsHttpPath.refresh_token_if_stale(
        failed_token, token_gen
    )
    if new_tok:
      req_hdrs['Authorization'] = f'Bearer {new_tok}'
      return True, new_tok
  return False, failed_token


def _send_gcs_http_once(
    vip: str,
    method: str,
    path: str,
    body: bytes | memoryview | None,
    req_hdrs: Mapping[str, str],
    timeout: float,
) -> tuple[int, bytes]:
  """Executes a single keep-alive GCS HTTP request on `vip`."""
  conn = safetensors_io.get_thread_https_conn(vip, timeout=timeout)
  conn.request(method, path, body=body, headers=req_hdrs)
  resp = conn.getresponse()
  return resp.status, resp.read()


def _raise_transport_error(
    method: str, path: str, exc: BaseException
) -> NoReturn:
  """Raises `OSError` directly or wraps `http.client.HTTPException`."""
  if isinstance(exc, OSError):
    raise exc
  raise OSError(f'GCS HTTP {method} {path} failed: {exc}') from exc


def _gcs_http_request(
    method: str,
    path: str,
    body: bytes | memoryview | None,
    headers: Mapping[str, str],
    vips: Sequence[str] = (),
    token: str = '',
    *,
    ok_statuses: tuple[int, ...] = (200, 201),
    timeout: float = 60.0,
    vip_offset: int | None = None,
) -> bytes:
  """Executes a multi-VIP keep-alive GCS XML/JSON request with 5x retry."""
  resolved_vips = safetensors_io.resolve_gcs_vips(vips)
  # Snapshot token generation before reading so a concurrent refresh during
  # this request does not cause a redundant refresh on HTTP 401.
  token_gen = safetensors_io.GCS_TOKEN_GEN[0]
  resolved_tok = token or safetensors_io.GsHttpPath.resolve_token()
  req_hdrs = {
      'Host': safetensors_io.GCS_HOSTS[0],
      'Connection': 'keep-alive',
      **headers,
  }
  if resolved_tok:
    req_hdrs['Authorization'] = f'Bearer {resolved_tok}'
  refreshed_auth = False
  attempt = 0
  while attempt < 5:
    vip = safetensors_io.pick_thread_vip(
        resolved_vips, attempt=attempt, vip_offset=vip_offset
    )
    try:
      status, payload = _send_gcs_http_once(
          vip, method, path, body, req_hdrs, timeout
      )
    except (OSError, http.client.HTTPException) as exc:
      safetensors_io.evict_thread_https_conn(vip)
      if attempt == 4:
        _raise_transport_error(method, path, exc)
      time.sleep(0.25 * (1 << min(attempt, 3)) * random.uniform(0.5, 1.5))
      attempt += 1
      continue
    if status in ok_statuses:
      return payload
    safetensors_io.evict_thread_https_conn(vip)
    did_refresh, resolved_tok = _maybe_refresh_auth_on_401(
        status, token, refreshed_auth, resolved_tok, token_gen, req_hdrs
    )
    if did_refresh:
      token_gen = safetensors_io.GCS_TOKEN_GEN[0]
      refreshed_auth = True
      continue
    if status not in _GCS_RETRY_STATUSES or attempt == 4:
      snippet = payload[:256].decode('utf-8', errors='replace')
      raise OSError(
          f'GCS HTTP {method} {path} failed: HTTP {status} ({snippet})'
      )
    time.sleep(0.25 * (1 << min(attempt, 3)) * random.uniform(0.5, 1.5))
    attempt += 1
  raise OSError(f'GCS HTTP {method} {path} exhausted retries.')


def upload_gcs_slot_http(
    data: bytes | memoryview,
    gs_uri: str,
    vips: Sequence[str] = (),
    token: str = '',
    vip_offset: int | None = None,
) -> None:
  """Uploads a shard or composite part to `gs://` via multi-VIP XML PUT.

  Streams `data` using HTTP/1.1 Keep-Alive XML `PUT /{bucket}/{key}` across
  discovered GCS VIPs with up to 5 retries on transient socket or HTTP
  408/429/5xx errors. When `token` is empty, authenticates via GCE metadata or
  the `TUNIX_GCS_TOKEN` environment variable.

  Args:
    data: Byte buffer or zero-copy `memoryview` slice (e.g. from `/dev/shm`).
    gs_uri: Destination `gs://<bucket>/<object>` URI.
    vips: Optional explicit GCS VIP list; defaults to 5-minute cached DNS VIPs.
    token: Optional OAuth bearer token; auto-resolved when empty.
    vip_offset: Optional deterministic VIP index offset for multi-host/worker
      thread striping.

  Raises:
    OSError: If the upload fails with a non-retryable HTTP status or exhausts
      all retry attempts.

  Example:
    ```python
    from tunix.models import safetensors_stage_worker

    safetensors_stage_worker.upload_gcs_slot_http(
        b'shard-bytes',
        'gs://my-bucket/ckpts/step_1/model-00001-of-00004.safetensors.part.0',
    )
    ```
  """
  bucket, key = safetensors_io.parse_gs_uri(gs_uri)
  path = f'/{bucket}/{urllib.parse.quote(key, safe="/")}'
  # Release both parent and cast memoryviews deterministically so an exception
  # unwind never leaves an exported buffer blocking `arena_mm.close()`.
  with memoryview(data) as raw_mv, raw_mv.cast('B') as mv:
    _gcs_http_request(
        'PUT',
        path,
        mv,
        {
            'Content-Type': 'application/octet-stream',
            'Content-Length': str(len(mv)),
        },
        vips=vips,
        token=token,
        vip_offset=vip_offset,
    )


def compose_gcs_objects_http(
    dst_gs_uri: str,
    part_keys: Sequence[str],
    vips: Sequence[str] = (),
    token: str = '',
    vip_offset: int | None = None,
) -> None:
  """Composes up to 32 GCS part objects into `dst_gs_uri` server-side.

  Issues a zero-egress GCS JSON API `compose` request so multi-wave shards are
  concatenated inside GCS without re-reading or re-uploading bytes from host.

  Args:
    dst_gs_uri: Final destination `gs://<bucket>/<object>` URI.
    part_keys: Ordered sequence of 1..32 source object names (or `gs://` URIs
      in the same bucket as `dst_gs_uri`).
    vips: Optional explicit GCS VIP list; defaults to cached multi-VIP list.
    token: Optional OAuth bearer token; auto-resolved when empty.
    vip_offset: Optional deterministic VIP index offset for multi-host/worker
      thread striping.

  Raises:
    ValueError: If `part_keys` is empty, has >32 parts, or references another
      bucket.
    OSError: If the compose RPC fails or exhausts retries.

  Example:
    ```python
    from tunix.models import safetensors_stage_worker

    safetensors_stage_worker.compose_gcs_objects_http(
        'gs://my-bucket/ckpts/model-00001-of-00004.safetensors',
        [
            'gs://my-bucket/ckpts/model-00001-of-00004.safetensors.part.0',
            'gs://my-bucket/ckpts/model-00001-of-00004.safetensors.part.1',
        ],
    )
    ```
  """
  if not part_keys or len(part_keys) > _MAX_GCS_COMPOSE_PARTS:
    raise ValueError(
        f'GCS compose requires 1..{_MAX_GCS_COMPOSE_PARTS} parts, got'
        f' {len(part_keys)}.'
    )
  bucket, dst_key = safetensors_io.parse_gs_uri(dst_gs_uri)
  prefix = f'gs://{bucket}/'
  if any(k.startswith('gs://') and not k.startswith(prefix) for k in part_keys):
    raise ValueError(f'All compose parts must reside in bucket {bucket!r}.')
  norm_keys = [k.removeprefix(prefix) for k in part_keys]
  body = json.dumps({
      'sourceObjects': [{'name': k} for k in norm_keys],
      'destination': {'contentType': 'application/octet-stream'},
  }).encode('utf-8')
  q_dst = urllib.parse.quote(dst_key, safe='')
  _gcs_http_request(
      'POST',
      f'/storage/v1/b/{bucket}/o/{q_dst}/compose',
      body,
      {
          'Content-Type': 'application/json',
          'Content-Length': str(len(body)),
      },
      vips=vips,
      token=token,
      vip_offset=vip_offset,
  )


def delete_gcs_object_http(
    gs_uri: str,
    vips: Sequence[str] = (),
    token: str = '',
    vip_offset: int | None = None,
) -> None:
  """Deletes a GCS object via multi-VIP HTTP (treats HTTP 404 as success).

  Idempotently removes temporary `.part.<k>` objects after GCS `compose` or on
  error cleanup so no partial objects leak in the bucket.

  Args:
    gs_uri: Target `gs://<bucket>/<object>` URI to delete.
    vips: Optional explicit GCS VIP list; defaults to cached multi-VIP list.
    token: Optional OAuth bearer token; auto-resolved when empty.
    vip_offset: Optional deterministic VIP index offset for multi-host/worker
      thread striping.

  Raises:
    OSError: If the delete RPC fails with a non-404 error or exhausts retries.

  Example:
    ```python
    from tunix.models import safetensors_stage_worker

    safetensors_stage_worker.delete_gcs_object_http(
        'gs://my-bucket/ckpts/model-00001-of-00004.safetensors.part.0'
    )
    ```
  """
  bucket, key = safetensors_io.parse_gs_uri(gs_uri)
  q_key = urllib.parse.quote(key, safe='')
  _gcs_http_request(
      'DELETE',
      f'/storage/v1/b/{bucket}/o/{q_key}',
      None,
      {},
      vips=vips,
      token=token,
      ok_statuses=(200, 204, 404),
      vip_offset=vip_offset,
  )


def _resolve_stage_backend(env: Mapping[str, str]) -> str:
  """Returns active stage worker transport backend."""
  raw = (
      env['TUNIX_STAGE_BACKEND'].strip().lower()
      if 'TUNIX_STAGE_BACKEND' in env
      else ''
  )
  if raw:
    if raw not in ('gcs_http', 'gcs_rapid_grpc', 'gfile'):
      raise ValueError(
          f'Unsupported TUNIX_STAGE_BACKEND={raw!r}; expected'
          " 'gcs_http', 'gcs_rapid_grpc', or 'gfile'."
      )
    return raw
  return 'gfile' if gfile is not None else 'gcs_http'


def _load_rapid_grpc_symbols() -> (
    tuple[Any, Any, tuple[type[BaseException], ...]] | None
):
  """Lazily imports Rapid Zonal gRPC writer symbols via `importlib`."""
  try:
    client_mod = importlib.import_module(
        'google.cloud.storage.asyncio.async_grpc_client'
    )
    writer_mod = importlib.import_module(
        'google.cloud.storage.asyncio.async_appendable_object_writer'
    )
    exc_mod = importlib.import_module('google.api_core.exceptions')
    retryable_excs = (
        exc_mod.ServiceUnavailable,
        exc_mod.DeadlineExceeded,
        exc_mod.InternalServerError,
        exc_mod.TooManyRequests,
        exc_mod.ResourceExhausted,
        exc_mod.Aborted,
    )
    return (
        client_mod.AsyncGrpcClient,
        writer_mod.AsyncAppendableObjectWriter,
        retryable_excs,
    )
  except (ImportError, AttributeError):
    return None


class _RapidGrpcUploader:
  """Async DirectPath gRPC `AsyncAppendableObjectWriter` uploader per worker."""

  def __init__(self, worker_idx: int) -> None:
    self._worker_idx = worker_idx
    self._client_cls: Any = None
    self._writer_cls: Any = None
    self._retryable_excs: tuple[type[BaseException], ...] = (OSError,)
    self._loop: asyncio.AbstractEventLoop | None = None
    self._thread: threading.Thread | None = None
    self._client: Any = None
    self._sem: asyncio.Semaphore | None = None

  def _run_loop(self, loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()

  async def _async_init(self) -> None:
    """Staggers startup, initializes gRPC client, and validates writer."""
    # Stagger client init to avoid 48-way DirectPath handshaker GOAWAY bursts.
    stagger = (self._worker_idx % 48) * _RAPID_GRPC_STAGGER_SEC
    if stagger > 0:
      await asyncio.sleep(stagger)
    self._client = self._client_cls(attempt_direct_path=True)
    # Validate writer preconditions (e.g. C google_crc32c) before READY.
    _ = self._writer_cls(
        client=self._client,
        bucket_name='_',
        object_name='_',
        writer_options={
            'FLUSH_INTERVAL_BYTES': _RAPID_GRPC_FLUSH_INTERVAL_BYTES,
        },
    )
    self._sem = asyncio.Semaphore(_RAPID_GRPC_INFLIGHT)

  def start(self) -> bool:
    """Starts background event loop and gRPC client; False on fallback."""
    symbols = _load_rapid_grpc_symbols()
    if symbols is None:
      return False
    (
        self._client_cls,
        self._writer_cls,
        self._retryable_excs,
    ) = symbols
    loop = asyncio.new_event_loop()
    self._loop = loop
    self._thread = threading.Thread(
        target=self._run_loop, args=(loop,), daemon=True
    )
    self._thread.start()
    init_fut = asyncio.run_coroutine_threadsafe(self._async_init(), loop)
    try:
      init_fut.result(timeout=_RAPID_GRPC_INIT_TIMEOUT_SEC)
      return True
    except Exception as exc:  # pylint: disable=broad-exception-caught
      init_fut.cancel()
      sys.stderr.write(
          f'Rapid gRPC init failed, falling back to gcs_http: {exc}\n'
      )
      sys.stderr.flush()
      self.close(timeout=_RAPID_GRPC_ABORT_CLOSE_TIMEOUT_SEC)
      return False

  async def _upload_once(
      self, bucket: str, key: str, slot_mv: memoryview
  ) -> None:
    """Streams `slot_mv` into one finalized Rapid Zonal appendable object."""
    writer = self._writer_cls(
        client=self._client,
        bucket_name=bucket,
        object_name=key,
        writer_options={
            'FLUSH_INTERVAL_BYTES': _RAPID_GRPC_FLUSH_INTERVAL_BYTES,
        },
    )
    try:
      await asyncio.wait_for(
          writer.open(), timeout=_RAPID_GRPC_RPC_TIMEOUT_SEC
      )
      total = len(slot_mv)
      # Pass the full object payload in one append() call: the SDK's retry
      # handler calls `user_buffer.seek(write_state.persisted_size)` with the
      # object-global offset, so sub-chunked append() would corrupt redirects.
      if total:
        append_timeout = max(
            _RAPID_GRPC_RPC_TIMEOUT_SEC,
            _RAPID_GRPC_RPC_TIMEOUT_SEC + total / (64 << 20),
        )
        await asyncio.wait_for(writer.append(slot_mv), timeout=append_timeout)
      res = await asyncio.wait_for(
          writer.close(finalize_on_close=True),
          timeout=_RAPID_GRPC_RPC_TIMEOUT_SEC,
      )
      persisted = res.size if res is not None else writer.persisted_size
      if persisted != total and not (persisted is None and total == 0):
        raise OSError(
            f'Rapid gRPC size mismatch for gs://{bucket}/{key}:'
            f' persisted {persisted} != expected {total}.'
        )
    except BaseException:
      # `open()` can time out after opening `socket_like_rpc` before
      # `write_obj_stream.is_stream_open` flips True (when `close()` raises
      # ValueError), and a stalled `writer.close()` can time out before
      # cancelling `socket_like_rpc`; fall back to closing `socket_like_rpc`.
      raw_stream = getattr(writer, 'write_obj_stream', None)
      target = getattr(raw_stream, 'socket_like_rpc', None)
      if writer.is_stream_open:
        with contextlib.suppress(Exception):
          await asyncio.wait_for(
              writer.close(finalize_on_close=False),
              timeout=_RAPID_GRPC_ABORT_CLOSE_TIMEOUT_SEC,
          )
          target = None
      elif getattr(raw_stream, 'is_stream_open', False):
        target = raw_stream
      if target is not None:
        with contextlib.suppress(Exception):
          await asyncio.wait_for(
              target.close(),
              timeout=_RAPID_GRPC_ABORT_CLOSE_TIMEOUT_SEC,
          )
      raise

  async def _upload_with_retry(
      self, bucket: str, key: str, slot_mv: memoryview
  ) -> None:
    """Retries transient gRPC failures on open, append, or finalize."""
    last_exc: BaseException | None = None
    retry_types = (*self._retryable_excs, OSError, TimeoutError)
    for attempt in range(_RAPID_GRPC_MAX_ATTEMPTS):
      try:
        await self._upload_once(bucket, key, slot_mv)
        return
      except asyncio.CancelledError as exc:
        # SDK-internal cancellation (bidi stream reset / multiplexer recv-loop
        # failure) surfaces as CancelledError; retry unless we are cancelled.
        last_exc = exc
        task = asyncio.current_task()
        assert task is not None
        if task.cancelling():
          break
        sys.stderr.write(f'Rapid gRPC inner CancelledError; retry {attempt}\n')
        delay = min(1.0, 0.05 * (1 << attempt))
      except Exception as exc:  # pylint: disable=broad-exception-caught
        last_exc = exc
        msg = str(exc).lower()
        is_retryable = isinstance(exc, retry_types) or any(
            map(msg.__contains__, _RAPID_RETRY_TOKENS)
        )
        if not is_retryable:
          break
        delay = min(1.0, 0.05 * (1 << attempt)) * random.uniform(0.5, 1.5)
      if attempt == _RAPID_GRPC_MAX_ATTEMPTS - 1:
        break
      await asyncio.sleep(delay)
    # Best-effort delete of any unfinalized appendable object left by open().
    if self._client is not None:
      with contextlib.suppress(Exception):
        await asyncio.wait_for(
            self._client.delete_object(bucket, key),
            timeout=_RAPID_GRPC_ABORT_CLOSE_TIMEOUT_SEC,
        )
    assert last_exc is not None
    raise last_exc

  async def _upload_slot_coro(
      self,
      arena_mm: mmap.mmap,
      base: int,
      nbytes: int,
      slot_id: int,
      dst_path: str,
      stdout_lock: threading.Lock,
  ) -> None:
    r"""Streams one `/dev/shm` slot over gRPC and emits `FREE\t<slot_id>`."""
    try:
      assert self._sem is not None
      bucket, key = safetensors_io.parse_gs_uri(dst_path)
      async with self._sem:
        with (
            memoryview(arena_mm) as arena_mv,
            arena_mv[base : base + nbytes] as slot_mv,
        ):
          await self._upload_with_retry(bucket, key, slot_mv)
    finally:
      with stdout_lock:
        sys.stdout.write(f'FREE\t{slot_id}\n')
        sys.stdout.flush()

  def submit_slot(
      self,
      arena_mm: mmap.mmap,
      base: int,
      nbytes: int,
      slot_id: int,
      dst_path: str,
      stdout_lock: threading.Lock,
  ) -> concurrent.futures.Future[None]:
    """Schedules one `/dev/shm` slot upload on the background asyncio loop."""
    assert self._loop is not None
    return asyncio.run_coroutine_threadsafe(
        self._upload_slot_coro(
            arena_mm, base, nbytes, slot_id, dst_path, stdout_lock
        ),
        self._loop,
    )

  async def _async_close(self, client: Any) -> None:
    """Cancels stray tasks and closes the underlying gRPC transport."""
    current = asyncio.current_task()
    pending = [
        t for t in asyncio.all_tasks() if t is not current and not t.done()
    ]
    for task in pending:
      task.cancel()
    if pending:
      await asyncio.gather(*pending, return_exceptions=True)
    if client is not None:
      with contextlib.suppress(Exception):
        res = client.grpc_client.transport.close()
        if inspect.isawaitable(res):
          await res

  def close(self, timeout: float = 5.0) -> None:
    """Closes gRPC transport, stops background loop, and joins daemon thread."""
    loop, self._loop = self._loop, None
    thread, self._thread = self._thread, None
    client, self._client = self._client, None
    self._sem = None
    if loop is not None:
      deadline = time.monotonic() + timeout
      if loop.is_running():
        with contextlib.suppress(Exception):
          asyncio.run_coroutine_threadsafe(
              self._async_close(client), loop
          ).result(timeout=max(0.05, timeout / 2))
        loop.call_soon_threadsafe(loop.stop)
      if thread is not None:
        thread.join(timeout=max(0.05, deadline - time.monotonic()))
      if not loop.is_closed() and not loop.is_running():
        loop.close()


def _stream_slot_to_remote(
    arena_mm: mmap.mmap,
    base: int,
    nbytes: int,
    slot_id: int,
    dst_path: str,
    stdout_lock: threading.Lock,
    vips: Sequence[str] = (),
) -> None:
  """Streams a pre-faulted `/dev/shm` mmap slot to 1P gfile or 3P GCS."""
  step = 16 << 20
  end = base + nbytes
  use_http = dst_path.startswith('gs://') and _resolve_stage_backend(
      os.environ
  ) in ('gcs_http', 'gcs_rapid_grpc')
  try:
    if use_http:
      # Explicitly scope both the parent mmap view and its slice so failed HTTP
      # uploads release all buffer exports immediately on teardown.
      with memoryview(arena_mm) as arena_mv, arena_mv[base:end] as slot_mv:
        upload_gcs_slot_http(
            slot_mv,
            dst_path,
            vips=vips,
            vip_offset=_WORKER_TLS.vip_offset,
        )
    elif gfile is not None:
      with gfile.GFile(to_gfile_path(dst_path), 'wb') as fdst:
        for off in range(base, end, step):
          fdst.write(arena_mm[off : min(off + step, end)])
    else:
      with open(dst_path, 'wb') as fdst:
        for off in range(base, end, step):
          fdst.write(arena_mm[off : min(off + step, end)])
  finally:
    with stdout_lock:
      sys.stdout.write(f'FREE\t{slot_id}\n')
      sys.stdout.flush()


def _delete_gcs_part_tls(uri: str, vips: Sequence[str]) -> None:
  """Deletes a GCS composite part on the delete thread's pinned warm VIP."""
  delete_gcs_object_http(uri, vips=vips, vip_offset=_WORKER_TLS.vip_offset)


def _compose_and_delete_gcs_shard(
    dst_uri: str,
    part_uris: Sequence[str],
    del_pool: concurrent.futures.ThreadPoolExecutor,
    vips: Sequence[str] = (),
) -> None:
  """Composes `part_uris` into `dst_uri` on warm TLS sockets, then deletes parts.

  Observed in 397B@8H weight sync: serial parent-process compose + delete took
  ~3.3-5.0s over cold parent connections. Running COMPOSE inside persistent
  workers reuses warm multi-VIP TLS sockets and pipelines per-shard part
  deletes immediately after each shard's compose completes.

  Args:
    dst_uri: Target composed GCS object URI (empty string for delete-only).
    part_uris: Ephemeral part object URIs to compose and delete.
    del_pool: Dedicated worker thread pool for concurrent part deletions.
    vips: Pre-discovered GCS VIP list shared across workers.

  Raises:
    Exception: Re-raises the first compose or part-deletion error after
      attempting cleanup of all `part_uris`.
  """
  vip_offset = _WORKER_TLS.vip_offset
  compose_err: Exception | None = None
  try:
    if dst_uri:
      compose_gcs_objects_http(
          dst_uri, part_uris, vips=vips, vip_offset=vip_offset
      )
  except Exception as exc:  # pylint: disable=broad-exception-caught
    compose_err = exc
  finally:
    del_futs = [
        del_pool.submit(_delete_gcs_part_tls, uri, vips) for uri in part_uris
    ]
    del_err: Exception | None = None
    for fut in del_futs:
      try:
        fut.result()
      except Exception as exc:  # pylint: disable=broad-exception-caught
        if del_err is None:
          del_err = exc
  if compose_err is not None:
    raise compose_err
  if del_err is not None:
    raise del_err


def _init_worker_arena(
    arena_path: str,
    worker_idx: int,
    num_workers: int,
) -> mmap.mmap | None:
  """Maps the parent-allocated `/dev/shm` arena read-only and pre-faults PTEs."""
  if not arena_path:
    return None
  total_bytes = os.path.getsize(arena_path)
  blk = min(64 << 20, max(4096, total_bytes))
  n_blks = (total_bytes + blk - 1) // blk
  my_blks = range(worker_idx, n_blks, max(1, num_workers))
  # Parent `create_shm_mmap(..., prefault=True)` already backed and zeroed the
  # arena; open read-only so stage workers never mutate shared pages.
  fd = os.open(arena_path, os.O_RDONLY)
  try:
    mm = mmap.mmap(fd, total_bytes, access=mmap.ACCESS_READ)
  finally:
    os.close(fd)
  # Pre-fault only this worker's strided blocks on >32 GiB arenas (instead of
  # faulting 128 GiB x 48 workers = 6 TiB of PTEs at startup).
  safetensors_io.pre_fault_shm(
      mm,
      total_bytes,
      write=False,
      block_indices=None if total_bytes <= 32 << 30 else list(my_blks),
      block_bytes=blk,
  )
  return mm


def _drain_worker_futs(
    futs: list[concurrent.futures.Future[None]],
) -> str:
  """Waits for all in-flight worker slot futures and returns first error."""
  err = ''
  for fut in futs:
    try:
      fut.result()
    except Exception as exc:  # pylint: disable=broad-exception-caught
      if not err:
        err = repr(exc)
  futs.clear()
  return err


def _fail_missing_arena_slot(
    slot_id: int, stdout_lock: threading.Lock
) -> None:
  """Releases `slot_id` and raises when SLOT arrives without an arena mmap."""
  with stdout_lock:
    sys.stdout.write(f'FREE\t{slot_id}\n')
    sys.stdout.flush()
  raise RuntimeError(
      f'Received SLOT {slot_id} without an initialized /dev/shm arena.'
  )


def _dispatch_worker_cmd(
    cmd: str,
    arena_mm: mmap.mmap | None,
    pool: concurrent.futures.ThreadPoolExecutor,
    del_pool: concurrent.futures.ThreadPoolExecutor,
    stdout_lock: threading.Lock,
    futs: list[concurrent.futures.Future[None]],
    vips: Sequence[str] = (),
    rapid_uploader: _RapidGrpcUploader | None = None,
) -> bool:
  """Handles one worker stdin command; returns False on QUIT."""
  if cmd.startswith('SLOT\t'):
    _, base_s, nbytes_s, slot_s, dst_path = cmd.split('\t', 4)
    if arena_mm is None:
      futs.append(
          pool.submit(_fail_missing_arena_slot, int(slot_s), stdout_lock)
      )
    elif rapid_uploader is not None and dst_path.startswith('gs://'):
      futs.append(
          rapid_uploader.submit_slot(
              arena_mm,
              int(base_s),
              int(nbytes_s),
              int(slot_s),
              dst_path,
              stdout_lock,
          )
      )
    else:
      futs.append(
          pool.submit(
              _stream_slot_to_remote,
              arena_mm,
              int(base_s),
              int(nbytes_s),
              int(slot_s),
              dst_path,
              stdout_lock,
              vips,
          )
      )
  elif cmd.startswith('COMPOSE\t'):
    parts = cmd.split('\t')
    futs.append(
        pool.submit(
            _compose_and_delete_gcs_shard,
            parts[1],
            parts[2:],
            del_pool,
            vips,
        )
    )
  elif cmd == 'FLUSH':
    err = _drain_worker_futs(futs)
    with stdout_lock:
      sys.stdout.write(f'ERR\t{err}\n' if err else 'DONE\n')
      sys.stdout.flush()
  elif cmd == 'QUIT':
    return False
  return True


def _maybe_refresh_worker_vips(
    vips: tuple[str, ...], vip_expiry: float, *, idle: bool = True
) -> tuple[tuple[str, ...], float]:
  """Refreshes inherited worker VIPs after `GCS_VIP_CACHE_TTL_S` expires."""
  now = time.monotonic()
  if not vips or not idle or now < vip_expiry:
    return vips, vip_expiry
  fresh = safetensors_io.resolve_gcs_vips()
  next_vips = fresh or vips
  retired = set(vips) - set(next_vips)
  if retired:
    for tid, vip in list(safetensors_io.THREAD_CONNS):
      if vip in retired:
        safetensors_io.evict_thread_https_conn(vip, tid=tid)
  return next_vips, now + safetensors_io.GCS_VIP_CACHE_TTL_S


def _start_rapid_uploader_if_enabled(
    stage_backend: str, worker_idx: int
) -> _RapidGrpcUploader | None:
  """Starts `_RapidGrpcUploader` when requested, else returns `None`."""
  if stage_backend != 'gcs_rapid_grpc':
    return None
  uploader = _RapidGrpcUploader(worker_idx)
  return uploader if uploader.start() else None


def run_stage_worker_stdio(argv: Sequence[str] = ()) -> None:
  r"""Runs persistent multi-threaded stage writer worker over stdin/stdout.

  Pre-faults this worker's slice of `TUNIX_STAGE_ARENA_PATH` (`/dev/shm`), emits
  `READY` on stdout, then handles tab-delimited `SLOT`, `COMPOSE`, `FLUSH`, and
  `QUIT` commands until EOF. Controlled by environment variables
  `TUNIX_STAGE_ARENA_PATH`, `TUNIX_STAGE_WORKER_IDX`, `TUNIX_STAGE_NUM_WORKERS`,
  `TUNIX_STAGE_PROCESS_IDX`, `TUNIX_STAGE_BACKEND`, and `TUNIX_GCS_VIPS`.

  Args:
    argv: Unused CLI argument sequence (accepted for `app.run` compatibility).

  Example:
    ```python
    # Spawned by MultiprocStagePool via:
    #   python3 -u path/to/safetensors_stage_worker.py
    # Protocol on stdin -> stdout:
    #   SLOT\t<offset>\t<nbytes>\t<slot_id>\t<dst_path>  ->  FREE\t<slot_id>
    #   COMPOSE\t<dst_uri>\t<p0>[\t<p1>...]              ->  (drained by FLUSH)
    #   FLUSH                                            ->  DONE | ERR\t<msg>
    #   QUIT                                             ->  (process exits)
    ```
  """
  del argv
  env = os.environ
  stage_backend = _resolve_stage_backend(env)
  if stage_backend == 'gfile':
    configure_stage_write_flags()
  arena_path = (
      env['TUNIX_STAGE_ARENA_PATH'] if 'TUNIX_STAGE_ARENA_PATH' in env else ''
  )
  w_idx = (
      int(env['TUNIX_STAGE_WORKER_IDX'])
      if 'TUNIX_STAGE_WORKER_IDX' in env
      else 0
  )
  n_workers = (
      int(env['TUNIX_STAGE_NUM_WORKERS'])
      if 'TUNIX_STAGE_NUM_WORKERS' in env
      else 1
  )
  proc_idx = (
      int(env['TUNIX_STAGE_PROCESS_IDX'])
      if 'TUNIX_STAGE_PROCESS_IDX' in env
      else 0
  )
  vips_csv = env['TUNIX_GCS_VIPS'] if 'TUNIX_GCS_VIPS' in env else ''
  vips = tuple(filter(None, vips_csv.split(',')))
  vip_expiry = time.monotonic() + safetensors_io.GCS_VIP_CACHE_TTL_S
  rapid_uploader = _start_rapid_uploader_if_enabled(stage_backend, w_idx)
  pool_workers = (
      4
      if (stage_backend in ('gcs_http', 'gcs_rapid_grpc') and n_workers >= 16)
      else 32
  )
  vip_base = (proc_idx * n_workers + w_idx) * pool_workers
  pool = concurrent.futures.ThreadPoolExecutor(
      max_workers=pool_workers,
      initializer=_init_vip_thread,
      initargs=(vip_base, itertools.count()),
  )
  del_pool = concurrent.futures.ThreadPoolExecutor(
      max_workers=8,
      initializer=_init_vip_thread,
      initargs=(vip_base, itertools.count()),
  )
  arena_mm: mmap.mmap | None = None
  futs: list[concurrent.futures.Future[None]] = []
  try:
    arena_mm = _init_worker_arena(arena_path, w_idx, n_workers)
    stdout_lock = threading.Lock()
    with stdout_lock:
      sys.stdout.write('READY\n')
      sys.stdout.flush()
    while line := sys.stdin.readline():
      vips, vip_expiry = _maybe_refresh_worker_vips(
          vips, vip_expiry, idle=not futs
      )
      if not _dispatch_worker_cmd(
          line.rstrip('\n'),
          arena_mm,
          pool,
          del_pool,
          stdout_lock,
          futs,
          vips,
          rapid_uploader=rapid_uploader,
      ):
        break
  finally:
    _drain_worker_futs(futs)
    if rapid_uploader is not None:
      rapid_uploader.close()
    pool.shutdown(wait=True)
    del_pool.shutdown(wait=True)
    gc.collect()
    if arena_mm is not None:
      with contextlib.suppress(BufferError):
        arena_mm.close()


if __name__ == '__main__':
  run_stage_worker_stdio()
