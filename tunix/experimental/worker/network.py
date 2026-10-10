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

"""Low-level network transport, zero-copy bulk sockets, and chunk serde for `remote_execution`.

Provides two complementary transport paths for Pickle Protocol 5 payloads:
1. **Out-of-Band Zero-Copy Bulk TCP Data Plane (`_BulkTransferServer` /
   `_BulkSocketPool`)**: For payloads >= 512 KiB with out-of-band
   `PickleBuffer`s, sends a lightweight length manifest over gRPC while
   striping large buffers via `sendall(memoryview)` and coalesced headers/small
   buffers via vectored `sendmsg` across one or more NIC endpoints
   (`bulk_hosts`) into pooled TCP sockets with the GIL released.
2. **GIL-Free Chunked gRPC Streaming (`_iter_async_from_chunk_specs` /
   `_ChunkReassembler`)**: For inline or fallback payloads, slices and
   reassembles chunks on a background thread pool using GIL-free
   `ctypes.memmove` and NumPy slice assignment.
"""

import asyncio
import bisect
import collections
import concurrent.futures
import contextlib
import ctypes
import errno
import ipaddress
import os
import pickle
import re
import secrets
import socket
import struct
import threading
import time
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Deque,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from absl import logging
import cloudpickle
import numpy as np


# Default per-call deadline (seconds) applied to remote invocations so a dead or
# wedged worker surfaces an error instead of hanging the caller indefinitely.
RPC_TIMEOUT_S = 60.0

# Server side timeout for handling a poll_responses() call.
# It should be shorter than the RPC_TIMEOUT_S to allow time for a response to
# be sent before the connection is torn down.
LONG_POLL_TIMEOUT_S = RPC_TIMEOUT_S - 10.0

# Cap for a single gRPC message frame. Payloads exceeding this (including >4 GiB
# packed training batches) are streamed in _STREAM_CHUNK_BYTES frames using
# Pickle Protocol 5 out-of-band buffers.
_MAX_MESSAGE_BYTES = 128 * 1024 * 1024

# Default slice size (4 MiB) per frame on streaming gRPC calls. Must remain
# strictly smaller than _MAX_MESSAGE_BYTES. Aligned with the 4 MiB HTTP/2 frame
# cap so each Cython SendMessageOperation / ReceiveMessageOperation completes in
# ~3-4 ms (<10 ms) on the event loop thread.
_STREAM_CHUNK_BYTES = 4 * 1024 * 1024

# Sub-chunk slice size (1 MiB) used dynamically when multiple bulk async
# serialization streams are active concurrently on the event loop, keeping the
# aggregate per-tick Cython memcpy budget bounded across multiplexed streams
# while remaining above _NOGIL_COPY_THRESHOLD_BYTES.
_CONCURRENT_STREAM_CHUNK_BYTES = 1 * 1024 * 1024

# Buffers smaller than this threshold are coalesced via a single b"".join()
# pass; buffers at or above this threshold are emitted directly as standalone
# frames (or chunk_size slices) to avoid intermediate coalescing copies.
_COALESCE_THRESHOLD_BYTES = 256 * 1024

# Payloads at or above this byte threshold offload chunk reassembly and
# unpickling to _SERDE_EXECUTOR so multi-megabyte buffer operations do not
# block the asyncio event loop.
_ASYNC_OFFLOAD_THRESHOLD_BYTES = 512 * 1024

# Minimum slice size (256 KiB) for GIL-free ctypes.memmove into uninitialized
# PyBytesObject. Smaller slices use memoryview.tobytes() directly.
_NOGIL_COPY_THRESHOLD_BYTES = 256 * 1024

# Maximum number of concurrent chunk serialization tasks in flight on
# _SERDE_EXECUTOR per stream.
_MAX_PARALLEL_SERDE_TASKS = 4

_STREAM_COUNTER_LOCK = threading.Lock()
_ACTIVE_ASYNC_SERDE_STREAMS: int = 0
_ACTIVE_POLL_RPCS: int = 0


def _inc_active_serde_streams() -> None:
  global _ACTIVE_ASYNC_SERDE_STREAMS
  with _STREAM_COUNTER_LOCK:
    _ACTIVE_ASYNC_SERDE_STREAMS += 1


def _dec_active_serde_streams() -> None:
  global _ACTIVE_ASYNC_SERDE_STREAMS
  with _STREAM_COUNTER_LOCK:
    _ACTIVE_ASYNC_SERDE_STREAMS = max(0, _ACTIVE_ASYNC_SERDE_STREAMS - 1)


def _inc_active_poll_rpcs() -> None:
  global _ACTIVE_POLL_RPCS
  with _STREAM_COUNTER_LOCK:
    _ACTIVE_POLL_RPCS += 1


def _dec_active_poll_rpcs() -> None:
  global _ACTIVE_POLL_RPCS
  with _STREAM_COUNTER_LOCK:
    _ACTIVE_POLL_RPCS = max(0, _ACTIVE_POLL_RPCS - 1)


def _has_concurrent_streams() -> bool:
  with _STREAM_COUNTER_LOCK:
    return _ACTIVE_ASYNC_SERDE_STREAMS > 1 or _ACTIVE_POLL_RPCS > 1


# Adaptive HTTP/2 and TCP socket parameters:
# - 4 MiB max HTTP/2 frame size reduces framing overhead 256x on large tensors
#   while still sending small messages at their exact byte size.
# - 4 MiB max TCP read chunk size lets gRPC's adaptive read estimator start at
#   8 KiB for small RPCs and scale up to 4 MiB slices under bulk streaming.
_GRPC_HTTP2_MAX_FRAME_BYTES = 4 * 1024 * 1024
_GRPC_TCP_MAX_READ_CHUNK_BYTES = 4 * 1024 * 1024

_SERDE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="tunix-rpc-serde",
)

# Use private ctypes.PYFUNCTYPE prototypes rather than mutating process-global
# attributes on ctypes.pythonapi.* functions.
_PYBYTES_FROM_STRING_AND_SIZE = ctypes.PYFUNCTYPE(
    ctypes.py_object, ctypes.c_char_p, ctypes.c_ssize_t
)(("PyBytes_FromStringAndSize", ctypes.pythonapi))

_PYBYTES_AS_STRING = ctypes.PYFUNCTYPE(ctypes.c_void_p, ctypes.py_object)(
    ("PyBytes_AsString", ctypes.pythonapi)
)


def _copy_view_slice_nogil(
    mv: memoryview, offset: int, length: int  # pylint: disable=g-bare-generic
) -> bytes:
  """Copies `mv[offset : offset + length]` into a `bytes` object without holding the GIL."""
  if length < _NOGIL_COPY_THRESHOLD_BYTES:
    return mv[offset : offset + length].tobytes()
  src_arr = np.frombuffer(mv, dtype=np.uint8, count=length, offset=offset)
  dst_bytes: bytes = _PYBYTES_FROM_STRING_AND_SIZE(None, length)
  dst_ptr = _PYBYTES_AS_STRING(dst_bytes)
  # ctypes.memmove is a non-PythonAPI C function call, which releases the GIL
  # (Py_UNBLOCK_THREADS) during the memory copy and any kernel page faults.
  ctypes.memmove(dst_ptr, src_arr.ctypes.data, length)
  return dst_bytes


def _grpc_options(
    max_message_bytes: int = _MAX_MESSAGE_BYTES,
) -> List[Tuple[str, int]]:
  """Channel/server options lifting the message-size cap and enabling keepalive."""
  return [
      ("grpc.max_send_message_length", max_message_bytes),
      ("grpc.max_receive_message_length", max_message_bytes),
      ("grpc.http2.max_frame_size", _GRPC_HTTP2_MAX_FRAME_BYTES),
      (
          "grpc.experimental.tcp_max_read_chunk_size",
          _GRPC_TCP_MAX_READ_CHUNK_BYTES,
      ),
      ("grpc.keepalive_time_ms", 20000),
      ("grpc.keepalive_timeout_ms", 10000),
      ("grpc.keepalive_permit_without_calls", 1),
      ("grpc.http2.max_pings_without_data", 0),
      ("grpc.http2.max_ping_strikes", 0),
      ("grpc.http2.min_ping_interval_without_data_ms", 5000),
      ("grpc.http2.min_recv_ping_interval_without_data_ms", 5000),
  ]


def _validate_stream_config(
    stream_chunk_bytes: int, max_message_bytes: int
) -> None:
  """Validates streaming chunk size and gRPC max message size bounds."""
  if stream_chunk_bytes <= 0:
    raise ValueError(
        f"stream_chunk_bytes must be positive, got {stream_chunk_bytes}."
    )
  if max_message_bytes <= 0:
    raise ValueError(
        f"max_message_bytes must be positive, got {max_message_bytes}."
    )
  if stream_chunk_bytes > max_message_bytes:
    raise ValueError(
        f"stream_chunk_bytes ({stream_chunk_bytes}) must not exceed "
        f"max_message_bytes ({max_message_bytes})."
    )


def _flush_pending_views(
    pending_views: List[memoryview],  # pylint: disable=g-bare-generic
) -> bytes:
  """Materializes coalesced small views in a single allocation and copy."""
  if len(pending_views) == 1:
    data = pending_views[0].tobytes()
  else:
    data = b"".join(pending_views)
  pending_views.clear()
  return data


# A chunk descriptor is either an already-materialized `bytes` object (such as
# Frame 0 manifest or coalesced small views) or a `(memoryview, offset, length)`
# slice to be materialized via `_copy_view_slice_nogil`.
_ChunkSpec = Union[
    bytes, Tuple[memoryview, int, int]  # pylint: disable=g-bare-generic
]


def _materialize_chunk_spec(spec: _ChunkSpec) -> bytes:
  if isinstance(spec, bytes):
    return spec
  mv, offset, length = spec
  return _copy_view_slice_nogil(mv, offset, length)


def _release_views_and_buffers(
    views: Sequence[memoryview],  # pylint: disable=g-bare-generic
    raw_buffers: Sequence[pickle.PickleBuffer],
) -> None:
  """Releases memoryviews and PickleBuffers in order."""
  for mv in views:
    mv.release()
  for pb in raw_buffers:
    pb.release()


def _drain_and_release_async(
    raw_futs: Sequence[concurrent.futures.Future[Any]],
    release_fn: Callable[[], None],
) -> None:
  """Cancels unstarted futures and runs `release_fn` once running tasks finish without blocking."""
  running: List[concurrent.futures.Future[Any]] = []
  for rf in raw_futs:
    if not rf.cancel() and not rf.done():
      running.append(rf)
  if not running:
    release_fn()
    return
  remaining = len(running)
  lock = threading.Lock()

  def _on_done(_: concurrent.futures.Future[Any]) -> None:
    nonlocal remaining
    with lock:
      remaining -= 1
      should_release = remaining == 0
    if should_release:
      release_fn()

  for rf in running:
    rf.add_done_callback(_on_done)


def _prepare_serialized_chunk_specs(
    obj: Any,
    chunk_size: int = _STREAM_CHUNK_BYTES,
) -> Tuple[
    int,
    bytes,
    List[memoryview],  # pylint: disable=g-bare-generic
    List[pickle.PickleBuffer],
    Iterator[_ChunkSpec],
]:
  """Extracts Pickle Protocol 5 views and returns a lazy `_ChunkSpec` iterator."""
  if chunk_size <= 0:
    raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
  raw_buffers: List[pickle.PickleBuffer] = []
  header_bytes = cloudpickle.dumps(
      obj, protocol=5, buffer_callback=raw_buffers.append
  )
  views: List[memoryview] = [memoryview(header_bytes)]  # pylint: disable=g-bare-generic
  try:
    for pb in raw_buffers:
      try:
        views.append(pb.raw())
      except BufferError:
        with memoryview(pb) as view:
          views.append(memoryview(view.tobytes()))
    buffer_lengths = tuple(len(v) for v in views[1:])
    total_bytes = len(views[0]) + sum(buffer_lengths)
    manifest = pickle.dumps((len(views[0]), buffer_lengths), protocol=5)
  except Exception:
    _release_views_and_buffers(views, raw_buffers)
    raise

  def _spec_iter() -> Iterator[_ChunkSpec]:
    coalesce_limit = min(chunk_size, _COALESCE_THRESHOLD_BYTES)
    pending_views: List[memoryview] = []  # pylint: disable=g-bare-generic
    pending_bytes = 0
    for mv in views:
      mv_len = len(mv)
      if mv_len == 0:
        continue
      if mv_len >= coalesce_limit:
        if pending_views:
          yield _flush_pending_views(pending_views)
          pending_bytes = 0
        for offset in range(0, mv_len, chunk_size):
          take = min(chunk_size, mv_len - offset)
          yield (mv, offset, take)
      else:
        if pending_bytes + mv_len > chunk_size and pending_views:
          yield _flush_pending_views(pending_views)
          pending_bytes = 0
        pending_views.append(mv)
        pending_bytes += mv_len
    if pending_views:
      yield _flush_pending_views(pending_views)

  return total_bytes, manifest, views, raw_buffers, _spec_iter()


def _iter_serialized_chunks_with_size(
    obj: Any,
    chunk_size: int = _STREAM_CHUNK_BYTES,
) -> Tuple[int, Iterator[bytes]]:
  """Serializes `obj` with Pickle Protocol 5 and returns `(total_bytes, chunk_iter)`.

  Args:
    obj: Arbitrary Python object to serialize.
    chunk_size: Maximum byte length of each emitted chunk frame.

  Returns:
    A 2-tuple `(total_bytes, chunk_iter)` where `total_bytes` is the total
    byte length across the pickled header and all out-of-band buffers, and
    `chunk_iter` yields the manifest frame followed by payload chunks.
  """
  total_bytes, manifest, views, raw_buffers, spec_iter = (
      _prepare_serialized_chunk_specs(obj, chunk_size=chunk_size)
  )

  def _gen() -> Iterator[bytes]:
    try:
      yield manifest
      for spec in spec_iter:
        yield _materialize_chunk_spec(spec)
    finally:
      _release_views_and_buffers(views, raw_buffers)

  return total_bytes, _gen()


def _iter_serialized_chunks(
    obj: Any,
    chunk_size: int = _STREAM_CHUNK_BYTES,
) -> Iterator[bytes]:
  """Serializes `obj` with Pickle Protocol 5 and returns a chunk iterator."""
  _, chunk_iter = _iter_serialized_chunks_with_size(obj, chunk_size=chunk_size)
  return chunk_iter


async def _iter_async_from_chunk_specs(
    manifest: bytes,
    views: List[memoryview],  # pylint: disable=g-bare-generic
    raw_buffers: List[pickle.PickleBuffer],
    spec_iter: Iterator[_ChunkSpec],
    *,
    offload: bool = True,
) -> AsyncIterator[bytes]:
  """Streams materialized `bytes` chunks with multi-thread GIL-free prefetching."""
  pending_futs: Deque[
      Tuple[
          Optional[concurrent.futures.Future[bytes]],
          asyncio.Future[bytes],
      ]
  ] = collections.deque()
  counted_stream = False
  try:
    if not offload:
      yield manifest
      for spec in spec_iter:
        if isinstance(spec, bytes):
          yield spec
        else:
          mv, offset, length = spec
          yield mv[offset : offset + length].tobytes()
      return

    _inc_active_serde_streams()
    counted_stream = True
    loop = asyncio.get_running_loop()

    def _adaptive_specs() -> Iterator[_ChunkSpec]:
      for spec in spec_iter:
        if isinstance(spec, bytes):
          yield spec
        else:
          mv, offset, length = spec
          pos = 0
          while pos < length:
            limit = (
                _CONCURRENT_STREAM_CHUNK_BYTES
                if _has_concurrent_streams()
                else length
            )
            take = min(limit, length - pos)
            yield (mv, offset + pos, take)
            pos += take

    adaptive_iter = _adaptive_specs()

    def _enqueue_next() -> bool:
      spec = next(adaptive_iter, None)
      if spec is None:
        return False
      if isinstance(spec, bytes):
        f: asyncio.Future[bytes] = loop.create_future()
        f.set_result(spec)
        pending_futs.append((None, f))
      else:
        raw_fut = _SERDE_EXECUTOR.submit(_materialize_chunk_spec, spec)
        pending_futs.append((raw_fut, asyncio.wrap_future(raw_fut, loop=loop)))
      return True

    initial_prefetch = (
        2 if _has_concurrent_streams() else _MAX_PARALLEL_SERDE_TASKS
    )
    for _ in range(initial_prefetch):
      if not _enqueue_next():
        break

    yield manifest
    while len(pending_futs) < _MAX_PARALLEL_SERDE_TASKS and _enqueue_next():
      pass
    while pending_futs:
      _, aio_fut = pending_futs.popleft()
      was_done = aio_fut.done()
      chunk = await aio_fut
      _enqueue_next()
      if was_done:
        await asyncio.sleep(0)
      yield chunk
  finally:
    if counted_stream:
      _dec_active_serde_streams()
    raw_to_drain = [rf for rf, _ in pending_futs if rf is not None]
    pending_futs.clear()
    _drain_and_release_async(
        raw_to_drain, lambda: _release_views_and_buffers(views, raw_buffers)
    )


_FeedSliceOp = Tuple[np.ndarray, int, bytes, int, int]


def _execute_feed_ops(ops: List[_FeedSliceOp]) -> None:
  """Copies a planned list of chunk slices into target buffers without the GIL."""
  for target_buf, dst_offset, chunk, src_offset, take in ops:
    # NumPy slice assignment releases the GIL (NPY_BEGIN_ALLOW_THREADS).
    target_buf[dst_offset : dst_offset + take] = np.frombuffer(
        chunk, dtype=np.uint8, count=take, offset=src_offset
    )


def _is_exact_int(val: object) -> bool:
  """Returns True if `val` is an `int` and not a `bool`."""
  return type(val) is int  # pylint: disable=unidiomatic-typecheck


def _validate_manifest_meta(meta: object) -> Tuple[int, List[int]]:
  """Validates an unpickled `(header_len, buffer_lengths)` chunk manifest."""
  try:
    header_len, buffer_lengths = meta  # type: ignore[misc]
  except Exception as exc:
    raise ValueError("Invalid chunk stream manifest.") from exc

  if not _is_exact_int(header_len) or header_len <= 0:
    raise ValueError(
        f"Invalid header_len in chunk stream manifest: {header_len!r}."
    )
  if not isinstance(buffer_lengths, (tuple, list)) or any(
      not _is_exact_int(length) or length < 0 for length in buffer_lengths
  ):
    raise ValueError(
        "Invalid buffer_lengths in chunk stream manifest:"
        f" {buffer_lengths!r}."
    )
  return header_len, [header_len, *buffer_lengths]


class _ChunkReassembler:
  """Incremental zero-copy reassembler for Pickle Protocol 5 chunk streams."""

  def __init__(
      self,
      manifest_bytes: bytes = b"",
      *,
      unpickled_meta: object = None,
  ) -> None:
    if unpickled_meta is None:
      try:
        unpickled_meta = pickle.loads(manifest_bytes)  # pylint: disable=g-unsafe-pickle-load
      except Exception as exc:
        raise ValueError("Invalid chunk stream manifest.") from exc

    _, self._target_lengths = _validate_manifest_meta(unpickled_meta)
    self.total_bytes: int = sum(self._target_lengths)
    # Lazily allocate uninitialized uint8 NumPy arrays on first write to avoid
    # upfront memset(0) across multi-gigabyte buffers and release the GIL
    # during memcpy.
    self._targets: List[Optional[np.ndarray]] = [
        np.empty(0, dtype=np.uint8) if length == 0 else None
        for length in self._target_lengths
    ]
    self._target_idx = 0
    self._target_offset = 0
    self._advance_empty_targets()

  def _advance_empty_targets(self) -> None:
    while (
        self._target_idx < len(self._target_lengths)
        and self._target_lengths[self._target_idx] == 0
    ):
      self._target_idx += 1

  def plan_feed(self, chunk: bytes) -> List[_FeedSliceOp]:
    """Plans non-overlapping slice copy operations for `chunk` in O(1) time."""
    if not chunk:
      return []
    chunk_len = len(chunk)
    chunk_pos = 0
    ops: List[_FeedSliceOp] = []
    while chunk_pos < chunk_len:
      if self._target_idx >= len(self._target_lengths):
        raise ValueError(
            "Received more chunk bytes than declared in stream manifest."
        )
      target_len = self._target_lengths[self._target_idx]
      remaining = target_len - self._target_offset
      take = min(chunk_len - chunk_pos, remaining)
      target_buf = self._targets[self._target_idx]
      if target_buf is None:
        target_buf = np.empty(target_len, dtype=np.uint8)
        self._targets[self._target_idx] = target_buf
      ops.append((target_buf, self._target_offset, chunk, chunk_pos, take))
      self._target_offset += take
      chunk_pos += take
      if self._target_offset == target_len:
        self._target_idx += 1
        self._target_offset = 0
        self._advance_empty_targets()
    return ops

  def feed(self, chunk: bytes) -> None:
    """Writes a chunk into lazily allocated uninitialized target buffers."""
    ops = self.plan_feed(chunk)
    if ops:
      _execute_feed_ops(ops)

  def finish(self) -> Any:
    """Validates completion and unpickles the object from reassembled buffers."""
    if self._target_idx < len(self._target_lengths):
      raise ValueError(
          "Stream ended before all declared buffer bytes were received."
      )
    completed_buffers: List[np.ndarray] = []
    for buf in self._targets:
      assert buf is not None
      completed_buffers.append(buf)
    return cloudpickle.loads(  # pylint: disable=g-unsafe-pickle-load
        completed_buffers[0], buffers=completed_buffers[1:]
    )


def _deserialize_from_chunks(chunks: Iterable[bytes]) -> Any:
  """Deserializes an object from a synchronous iterable of chunks."""
  reassembler: Optional[_ChunkReassembler] = None
  for chunk in chunks:
    if reassembler is None:
      reassembler = _ChunkReassembler(chunk)
    else:
      reassembler.feed(chunk)
  if reassembler is None:
    raise ValueError("Cannot deserialize from an empty chunk stream.")
  return reassembler.finish()


# ==============================================================================
# Out-of-Band Zero-Copy Bulk Socket Data Plane
# ==============================================================================

_OP_PULL = 1
_OP_PUSH = 2
_TRANSFER_TOKEN_BYTES = 16
# Wire header: op (1B), transfer_token (16B), total_bytes (8B), buf_idx (8B),
# buf_total_len (8B), offset (8B), length (8B) = 57 bytes.
_BULK_CMD_STRUCT = struct.Struct("<B16sQQQQQ")
_MIN_BULK_STRIPE_BYTES = 32 * 1024 * 1024
_MAX_BULK_STRIPE_BYTES = 1024 * 1024 * 1024
_MAX_BULK_STRIPES = 4
# Caps per-transfer stripe target calculation at the _BULK_IO_EXECUTOR worker
# count (16); multi-buffer or concurrent transfers may still queue.
_MAX_TOTAL_BULK_STRIPES = 16
_MAX_BULK_TRANSFER_BYTES = 64 * 1024 * 1024 * 1024
_MAX_BULK_BUFFERS = 65536
_BULK_SOCKET_TIMEOUT_S = 30.0
_BULK_PROBE_TIMEOUT_S = 2.0
_BULK_PROBE_RETRY_COOLDOWN_S = 60.0
# Client pooled socket idle TTL (15s) MUST remain strictly smaller than the
# server's SO_RCVTIMEO (_BULK_SOCKET_TIMEOUT_S = 30s) so idle sockets are
# retired on the client before the server closes them mid-reuse.
_MAX_IDLE_SOCKET_AGE_S = 15.0
_INCOMING_TTL_S = 120.0
_MAX_ACTIVE_SERVER_CONNS = 128
# Maximum number of iovecs per vectored `sock.sendmsg` call (well below Linux
# IOV_MAX = 1024).
_MAX_SENDMSG_IOV = 512
_BULK_PACK_ALIGN = 64
# Fixed wire protocol threshold (4 MiB) below which out-of-band buffers are
# packed into the 64-byte-aligned trailing _SegmentedBulkView.
_BULK_PACK_MAX_BYTES = 4 * 1024 * 1024
_ZERO_PAD_MV = memoryview(b"\x00" * _BULK_PACK_ALIGN)

# Linux kernel MAX_RW_COUNT (include/linux/fs.h: INT_MAX & PAGE_MASK =
# 0x7ffff000 = 2,147,479,552 bytes). Even with MSG_WAITALL, a single recv_into
# syscall is clamped to MAX_RW_COUNT by the kernel, and can also return short
# reads across non-loopback TCP windows or when a POSIX signal interrupts the
# syscall after partial bytes arrive.
_MAX_SOCKET_RW_CHUNK_BYTES = 0x7FFFF000
_MAX_POOLED_SOCKETS_PER_TARGET = 16

# Total byte cap across concurrent incoming transfers whose gRPC manifests have
# not yet been claimed by `wait_incoming`, bounded by 25% of physical RAM.
_MAX_UNCLAIMED_INCOMING_BYTES = min(
    256 * 1024**3,
    max(
        64 * 1024**2,
        (os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) // 4,
    ),
)
_UNCLAIMED_BUDGET_WAIT_S = 5.0

_BULK_IO_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=16,
    thread_name_prefix="tunix-bulk-io",
)


def _alloc_aligned_u8(length: int) -> np.ndarray:
  """Allocates an uninitialized uint8 array whose data pointer is 64-byte aligned."""
  raw = np.empty(length + _BULK_PACK_ALIGN, dtype=np.uint8)
  rem = raw.ctypes.data % _BULK_PACK_ALIGN
  pad = 0 if rem == 0 else (_BULK_PACK_ALIGN - rem)
  return raw[pad : pad + length]


class _BulkTransportError(ConnectionError):
  """Raised when an out-of-band bulk transfer fails due to port/server mismatch."""


def _allocate_transfer_id() -> bytes:
  """Generates a 128-bit cryptographically random transfer capability token."""
  return secrets.token_bytes(_TRANSFER_TOKEN_BYTES)


def _shutdown_socket_only(sock: socket.socket) -> None:
  """Shuts down both directions to wake blocked threads without closing the fd."""
  with contextlib.suppress(OSError):
    sock.shutdown(socket.SHUT_RDWR)


def _shutdown_and_close_socket(sock: socket.socket) -> None:
  """Shuts down both directions to wake blocked threads before closing `sock`."""
  _shutdown_socket_only(sock)
  with contextlib.suppress(OSError):
    sock.close()


def _configure_bulk_socket(sock: socket.socket) -> None:
  """Configures TCP_NODELAY and kernel-level SO_RCVTIMEO/SO_SNDTIMEO."""
  # Ensure the socket file descriptor is in blocking mode (~O_NONBLOCK) so
  # MSG_WAITALL and MSG_DONTWAIT behave natively without CPython poll wrappers.
  sock.settimeout(None)
  with contextlib.suppress(OSError):
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
  # Use OS-level SO_RCVTIMEO / SO_SNDTIMEO instead of CPython's settimeout()
  # so the file descriptor stays in blocking mode (~O_NONBLOCK) for native
  # MSG_WAITALL / sendmsg syscalls without per-call poll() overhead.
  timeout_s = max(0.001, float(_BULK_SOCKET_TIMEOUT_S))
  sec = int(timeout_s)
  usec = int(round((timeout_s - sec) * 1_000_000))
  timeval = struct.pack("ll", sec, usec)
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, timeval)
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDTIMEO, timeval)


def _recv_into_exact(
    sock: socket.socket,
    mv: memoryview,  # pylint: disable=g-bare-generic
    length: int,
) -> int:
  """Reads `length` bytes into `mv`, looping across short reads."""
  if length <= 0:
    return 0
  try:
    if length <= _MAX_SOCKET_RW_CHUNK_BYTES:
      got = sock.recv_into(mv, length, socket.MSG_WAITALL)
      if got == length or got == 0:
        return got
      offset = got
    else:
      offset = 0
    while offset < length:
      step = min(length - offset, _MAX_SOCKET_RW_CHUNK_BYTES)
      tail_mv = mv[offset : offset + step]
      try:
        n = sock.recv_into(tail_mv, step, socket.MSG_WAITALL)
      finally:
        tail_mv.release()
      if n <= 0:
        break
      offset += n
    return offset
  except BlockingIOError as exc:
    # A blocking socket with SO_RCVTIMEO raises BlockingIOError (EAGAIN /
    # EWOULDBLOCK) when the kernel receive timer expires.
    raise TimeoutError("Bulk socket receive timed out.") from exc


def _sendall_with_timeout(
    sock: socket.socket,
    data: Union[bytes, memoryview],  # pylint: disable=g-bare-generic
) -> None:
  """Calls `sock.sendall(data)`, converting kernel SO_SNDTIMEO EAGAIN to TimeoutError."""
  try:
    sock.sendall(data)
  except BlockingIOError as exc:
    raise TimeoutError("Bulk socket send timed out.") from exc


class _SegmentedBulkView:
  """Virtual contiguous view over one or more `memoryview`s for zero-copy `sendmsg`."""

  def __init__(
      self, segments: Sequence[memoryview]  # pylint: disable=g-bare-generic
  ) -> None:
    self.segments: List[memoryview] = []  # pylint: disable=g-bare-generic
    self.cum_offsets: List[int] = []
    running = 0
    for seg in segments:
      seg_len = len(seg)
      if seg_len > 0:
        self.segments.append(seg)
        self.cum_offsets.append(running)
        running += seg_len
    self.total_len = running

  def __len__(self) -> int:
    return self.total_len

  def slice_iov(
      self, offset: int, length: int
  ) -> Tuple[List[memoryview], List[memoryview]]:  # pylint: disable=g-bare-generic
    """Returns `(iov_slices, temp_views_to_release)` for `[offset : offset + length]`."""
    if offset < 0 or length < 0 or offset + length > self.total_len:
      raise ValueError(
          f"slice_iov out of bounds: offset={offset}, length={length}, "
          f"total_len={self.total_len}."
      )
    if length == 0 or not self.segments:
      return [], []
    idx = bisect.bisect_right(self.cum_offsets, offset) - 1
    rem = length
    cur_off = offset - self.cum_offsets[idx]
    iov: List[memoryview] = []  # pylint: disable=g-bare-generic
    temp_views: List[memoryview] = []  # pylint: disable=g-bare-generic
    num_segs = len(self.segments)
    while rem > 0 and idx < num_segs:
      seg = self.segments[idx]
      avail = len(seg) - cur_off
      take = min(rem, avail)
      # Always create an independent sub-slice so releasing the parent view on
      # cancellation never raises BufferError while sendmsg is in flight.
      sub = seg[cur_off : cur_off + take]
      iov.append(sub)
      temp_views.append(sub)
      rem -= take
      cur_off = 0
      idx += 1
    return iov, temp_views


def _send_bulk_view_slice(
    sock: socket.socket,
    view: _SegmentedBulkView,
    offset: int,
    length: int,
) -> None:
  """Sends `[offset : offset + length]` of `view` via vectored `sock.sendmsg`."""
  iov, temp_views = view.slice_iov(offset, length)
  try:
    idx = 0
    sub_off = 0
    num_iov = len(iov)
    while idx < num_iov:
      batch_end = min(idx + _MAX_SENDMSG_IOV, num_iov)
      batch = iov[idx:batch_end]
      partial_head: Optional[memoryview] = None  # pylint: disable=g-bare-generic
      if sub_off > 0:
        partial_head = batch[0][sub_off:]
        batch = [partial_head, *batch[1:]]
      try:
        sent = sock.sendmsg(batch)
      except BlockingIOError as exc:
        raise TimeoutError("Bulk socket send timed out.") from exc
      finally:
        if partial_head is not None:
          partial_head.release()
      if sent <= 0:
        raise ConnectionError("Bulk socket closed during vectored sendmsg.")
      while idx < num_iov and sent >= (len(iov[idx]) - sub_off):
        sent -= len(iov[idx]) - sub_off
        idx += 1
        sub_off = 0
      if sent > 0:
        sub_off += sent
  finally:
    for tv in temp_views:
      tv.release()


def _extract_host_and_port(target_address: str) -> Tuple[str, Optional[int]]:
  """Parses a `host[:port]` or `grpc://[ipv6][:port]` string into `(host, port)`."""
  hp = target_address.removeprefix("grpc://")
  if not hp or "://" in hp or "/" in hp:
    raise ValueError(f"Invalid target address: {target_address!r}")
  if hp.startswith("["):
    end = hp.find("]")
    if end <= 1:
      raise ValueError(f"Invalid IPv6 target address: {target_address!r}")
    host = hp[1:end]
    rest = hp[end + 1 :]
    if not rest:
      return (host, None)
    if not rest.startswith(":"):
      raise ValueError(f"Invalid port in target address: {target_address!r}")
    port_str = rest[1:]
  elif ":" in hp:
    host, port_str = hp.rsplit(":", 1)
    if not host or ":" in host:
      raise ValueError(
          f"Invalid host:port target address: {target_address!r}"
      )
  else:
    return (hp, None)
  if not (port_str.isascii() and port_str.isdigit()):
    raise ValueError(f"Invalid port in target address: {target_address!r}")
  port = int(port_str)
  if port <= 0 or port > 65535:
    raise ValueError(
        f"Port out of range (1..65535) in target address: {target_address!r}"
    )
  return (host, port)


_HOSTNAME_LABEL_RE = re.compile(
    r"^[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\Z"
)


def _validate_bulk_hosts(
    bulk_hosts: Optional[Sequence[str]],
    *,
    allow_empty: bool = True,
) -> Tuple[str, ...]:
  """Validates, normalizes, and deduplicates bare hostnames or IP literals.

  Brackets around IPv6 literals are stripped and IPv6 hex digits/hostnames are
  normalized to canonical lowercase form (preserving IPv6 `%scope` zone ID case)
  before deduplicating in insertion order.

  Args:
    bulk_hosts: Sequence of bare hostnames, IPv4 literals, or bracketed/bare
      IPv6 literals without ports, schemes, or paths.
    allow_empty: If False, raises `ValueError` when `bulk_hosts` is `None` or
      empty.

  Returns:
    A tuple of canonical, deduplicated host strings.
  """
  if bulk_hosts is None:
    if not allow_empty:
      raise ValueError("bulk_hosts must not be empty.")
    return ()
  if isinstance(bulk_hosts, (str, bytes)):
    raise ValueError(
        "bulk_hosts must be a sequence of host strings, not a single string."
    )
  deduped: List[str] = []
  seen: Set[str] = set()
  for raw_host in bulk_hosts:
    if not isinstance(raw_host, str) or not raw_host:
      raise ValueError(f"Invalid bulk host entry: {raw_host!r}")
    if (
        "://" in raw_host
        or "/" in raw_host
        or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in raw_host)
    ):
      raise ValueError(
          f"Bulk host must be a bare host or IP literal: {raw_host!r}"
      )
    if raw_host.startswith("["):
      if not raw_host.endswith("]"):
        if "]:" in raw_host:
          raise ValueError(f"Bulk host must not include a port: {raw_host!r}")
        raise ValueError(f"Invalid bracketed IPv6 bulk host: {raw_host!r}")
      host_body = raw_host[1:-1]
      if not host_body or ":" not in host_body:
        raise ValueError(f"Invalid bracketed IPv6 bulk host: {raw_host!r}")
    else:
      if "[" in raw_host or "]" in raw_host:
        raise ValueError(f"Invalid brackets in bulk host: {raw_host!r}")
      if raw_host.count(":") == 1:
        raise ValueError(f"Bulk host must not include a port: {raw_host!r}")
      host_body = raw_host

    if ":" in host_body:
      try:
        canonical = str(ipaddress.IPv6Address(host_body))
      except ValueError as exc:
        raise ValueError(
            f"Invalid IPv6 bulk host literal: {raw_host!r}"
        ) from exc
    else:
      try:
        canonical = str(ipaddress.IPv4Address(host_body))
      except ValueError:
        trimmed = host_body[:-1] if host_body.endswith(".") else host_body
        labels = trimmed.split(".")
        if (
            not trimmed
            or len(trimmed) > 253
            or any(not _HOSTNAME_LABEL_RE.fullmatch(lbl) for lbl in labels)
            or labels[-1].isdigit()
        ):
          raise ValueError(
              f"Invalid bulk hostname or IPv4 literal: {raw_host!r}"
          ) from None
        canonical = host_body.lower()

    if canonical not in seen:
      seen.add(canonical)
      deduped.append(canonical)
  if not deduped and not allow_empty:
    raise ValueError("bulk_hosts must not be empty.")
  return tuple(deduped)


def _encode_bulk_port_response(
    port: int, bulk_hosts: Sequence[str] = ()
) -> bytes:
  """Encodes `GetBulkPort` wire bytes (8B LE port + optional newline hosts)."""
  if not _is_exact_int(port) or port < 0 or port > 65535:
    raise ValueError(f"Invalid bulk port: {port!r}")
  header = struct.pack("<Q", port)
  if port == 0 or not bulk_hosts:
    return header
  return header + "\n".join(bulk_hosts).encode("utf-8")


def _decode_bulk_port_response(
    data: bytes,
) -> Tuple[int, Tuple[str, ...]]:
  """Decodes `GetBulkPort` wire bytes into `(port, bulk_hosts)`."""
  if len(data) < 8:
    raise ValueError(
        f"GetBulkPort response too short: {len(data)} bytes (< 8)."
    )
  (raw_port,) = struct.unpack("<Q", data[:8])
  port = int(raw_port)
  if port < 0 or port > 65535:
    raise ValueError(f"GetBulkPort port out of range: {port}.")
  if port == 0 or len(data) == 8:
    return (port, ())
  try:
    hosts_text = data[8:].decode("utf-8")
  except UnicodeDecodeError as exc:
    raise ValueError("Invalid UTF-8 in GetBulkPort bulk_hosts.") from exc
  raw_lines = [line for line in hosts_text.split("\n") if line]
  return (port, _validate_bulk_hosts(raw_lines, allow_empty=False))


class _BulkSocketPool:
  """Thread-safe LIFO pool of persistent TCP sockets for zero-copy buffer transfers."""

  def __init__(self) -> None:
    self._lock = threading.Lock()
    self._pid = os.getpid()
    self._pools: Dict[Tuple[str, int], Deque[Tuple[socket.socket, float]]] = (
        collections.defaultdict(collections.deque)
    )

  @staticmethod
  def _normalize_connect_host(host: str) -> str:
    return (
        "127.0.0.1" if host in ("localhost", "::1", "::", "0.0.0.0") else host
    )

  def _discard_inherited_sockets_locked(self) -> None:
    """Closes inherited socket wrappers after `os.fork()` without `shutdown()`."""
    curr_pid = os.getpid()
    if curr_pid == self._pid:
      return
    for q in self._pools.values():
      for sock, _ in q:
        with contextlib.suppress(OSError):
          sock.close()
    self._pools.clear()
    self._pid = curr_pid

  def _connect(self, host: str, port: int) -> socket.socket:
    """Opens and configures a new blocking TCP connection to `(host, port)`."""
    target_host = self._normalize_connect_host(host)
    sock = socket.create_connection(
        (target_host, port), timeout=float(_BULK_SOCKET_TIMEOUT_S)
    )
    try:
      _configure_bulk_socket(sock)
      return sock
    except BaseException:
      sock.close()
      raise

  @staticmethod
  def _is_socket_alive(sock: socket.socket) -> bool:
    try:
      _ = sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
      # An idle pooled socket must have no unread bytes and not be EOF (b"").
      return False
    except BlockingIOError:
      return True
    except OSError:
      return False

  def acquire(self, host: str, port: int) -> socket.socket:
    """Acquires the most recently used idle TCP socket to `(host, port)` or connects a new one."""
    key = (host, port)
    now = time.monotonic()
    while True:
      entry: Optional[Tuple[socket.socket, float]] = None
      with self._lock:
        self._discard_inherited_sockets_locked()
        q = self._pools.get(key)
        if q:
          # LIFO pop reuses warm sockets first and lets cold sockets at the
          # bottom of the deque age out after _MAX_IDLE_SOCKET_AGE_S.
          entry = q.pop()
      if entry is None:
        return self._connect(host, port)
      sock, released_at = entry
      if (
          now - released_at
      ) <= _MAX_IDLE_SOCKET_AGE_S and self._is_socket_alive(sock):
        return sock
      _shutdown_and_close_socket(sock)

  async def probe_async(self, host: str, port: int) -> None:
    """Probes TCP connectivity on the event loop across IPv4/IPv6 without pooling."""
    target_host = self._normalize_connect_host(host)
    try:
      _, writer = await asyncio.wait_for(
          asyncio.open_connection(target_host, port),
          timeout=_BULK_PROBE_TIMEOUT_S,
      )
      writer.close()
      with contextlib.suppress(OSError):
        await writer.wait_closed()
    except TimeoutError as exc:
      raise TimeoutError(
          f"Bulk TCP probe to {target_host}:{port} timed out after"
          f" {_BULK_PROBE_TIMEOUT_S}s."
      ) from exc

  def release(self, host: str, port: int, sock: socket.socket) -> None:
    key = (host, port)
    now = time.monotonic()
    with self._lock:
      self._discard_inherited_sockets_locked()
      q = self._pools[key]
      if len(q) < _MAX_POOLED_SOCKETS_PER_TARGET:
        q.append((sock, now))
        return
    _shutdown_and_close_socket(sock)

  def close_target(self, host: str, port: int) -> None:
    key = (host, port)
    with self._lock:
      self._discard_inherited_sockets_locked()
      q = self._pools.pop(key, None)
    if q:
      for sock, _ in q:
        _shutdown_and_close_socket(sock)


_DEFAULT_BULK_SOCKET_POOL = _BulkSocketPool()


def _resolve_future_if_pending(fut: asyncio.Future[Any]) -> None:
  if not fut.done():
    fut.set_result(True)


def _fail_future_if_pending(
    fut: asyncio.Future[Any], exc: Exception
) -> None:
  if not fut.done():
    fut.set_exception(exc)


class _OutgoingBulkTransfer:
  """State for an active Server -> Client zero-copy pull transfer."""

  def __init__(
      self,
      views: Sequence[_SegmentedBulkView],
      remaining_bytes: int,
      loop: asyncio.AbstractEventLoop,
      done_fut: asyncio.Future[bool],
  ) -> None:
    self.views: List[_SegmentedBulkView] = list(views)
    self.remaining_bytes = remaining_bytes
    self.loop = loop
    self.done_fut = done_fut
    self.lock = threading.Lock()
    self.created_at = time.monotonic()

  def mark_sent(self, count: int) -> None:
    with self.lock:
      self.remaining_bytes -= count
      done = self.remaining_bytes <= 0
    if done:
      with contextlib.suppress(RuntimeError):
        self.loop.call_soon_threadsafe(
            _resolve_future_if_pending, self.done_fut
        )

  def mark_failed(self, exc: Exception) -> None:
    with contextlib.suppress(RuntimeError):
      self.loop.call_soon_threadsafe(
          _fail_future_if_pending, self.done_fut, exc
      )


class _IncomingBulkTransfer:
  """State for an active Client -> Server (or Pull) zero-copy transfer."""

  def __init__(self, total_bytes: int, *, claimed: bool = False) -> None:
    self.total_bytes = total_bytes
    self.allocated_bytes = 0
    self.remaining_bytes = total_bytes
    self.claimed = claimed
    self.buffer_lengths: Dict[int, int] = {}
    self.received_per_buf: Dict[int, int] = {}
    self.buffers: Dict[int, np.ndarray] = {}
    self.views: Dict[int, memoryview] = {}  # pylint: disable=g-bare-generic
    self.buf_locks: Dict[int, threading.Lock] = {}
    self.loop: Optional[asyncio.AbstractEventLoop] = None
    self.done_fut: Optional[asyncio.Future[bool]] = None
    self.error: Optional[Exception] = None
    self.lock = threading.Lock()
    self.created_at = time.monotonic()

  def get_or_alloc_view(
      self, buf_idx: int, buf_total_len: int
  ) -> memoryview:  # pylint: disable=g-bare-generic
    """Returns or allocates a 64B-aligned target `memoryview` outside `self.lock`."""
    if (
        buf_idx < 0
        or buf_idx >= (_MAX_BULK_BUFFERS + 1)
        or buf_total_len <= 0
        or buf_total_len > self.total_bytes
        or buf_total_len > _MAX_BULK_TRANSFER_BYTES
    ):
      raise ValueError(
          f"Invalid bulk buffer specification: buf_idx={buf_idx}, "
          f"buf_total_len={buf_total_len}, total_bytes={self.total_bytes}."
      )
    with self.lock:
      if self.error is not None:
        raise ConnectionError(
            "Bulk transfer aborted due to peer error."
        ) from self.error
      existing_len = self.buffer_lengths.get(buf_idx)
      if existing_len is not None:
        if existing_len != buf_total_len:
          raise ValueError(
              f"Mismatched buf_total_len for buffer {buf_idx}: "
              f"{buf_total_len} != {existing_len}."
          )
      else:
        if self.allocated_bytes + buf_total_len > self.total_bytes:
          raise ValueError(
              f"Invalid bulk buffer allocation exceeds declared total_bytes: "
              f"{self.allocated_bytes + buf_total_len} > {self.total_bytes}."
          )
        self.buffer_lengths[buf_idx] = buf_total_len
        self.allocated_bytes += buf_total_len
      mv = self.views.get(buf_idx)
      if mv is not None:
        return mv
      buf_lock = self.buf_locks.get(buf_idx)
      if buf_lock is None:
        buf_lock = threading.Lock()
        self.buf_locks[buf_idx] = buf_lock

    # Serialize allocation per `buf_idx` outside `self.lock` so concurrent
    # stripes of a 2+ GiB buffer do not redundantly mmap 4x virtual memory.
    with buf_lock:
      with self.lock:
        mv = self.views.get(buf_idx)
        if mv is not None:
          return mv
      arr = _alloc_aligned_u8(buf_total_len)
      new_mv = memoryview(arr)
      with self.lock:
        if self.error is not None:
          new_mv.release()
          raise ConnectionError(
              "Bulk transfer aborted due to peer error."
          ) from self.error
        self.buffers[buf_idx] = arr
        self.views[buf_idx] = new_mv
        return new_mv

  def mark_received(self, buf_idx: int, count: int) -> None:
    """Records `count` received bytes for `buf_idx` and resolves `done_fut`."""
    with self.lock:
      buf_len = self.buffer_lengths.get(buf_idx, 0)
      new_buf_rec = self.received_per_buf.get(buf_idx, 0) + count
      if new_buf_rec > buf_len:
        raise ValueError(
            f"Received bytes for buffer {buf_idx} exceeded length: "
            f"{new_buf_rec} > {buf_len}."
        )
      self.received_per_buf[buf_idx] = new_buf_rec
      self.remaining_bytes -= count
      done = self.remaining_bytes <= 0
      loop = self.loop
      fut = self.done_fut
    if done and loop is not None and fut is not None:
      with contextlib.suppress(RuntimeError):
        loop.call_soon_threadsafe(_resolve_future_if_pending, fut)

  def mark_failed(self, exc: Exception) -> None:
    with self.lock:
      if self.error is None:
        self.error = exc
      loop = self.loop
      fut = self.done_fut
    if loop is not None and fut is not None:
      with contextlib.suppress(RuntimeError):
        loop.call_soon_threadsafe(_fail_future_if_pending, fut, exc)

  def release_views(self) -> None:
    with self.lock:
      views = list(self.views.values())
      self.views.clear()
      self.buffers.clear()
    for mv in views:
      mv.release()


_MAX_COMPLETED_TIDS = 4096


class _BulkTransferServer:
  """Zero-copy TCP data-plane server for out-of-band PickleBuffer transfers."""

  def __init__(self) -> None:
    self._sock: Optional[socket.socket] = None
    self.port: int = 0
    self._stop_event = threading.Event()
    self._lock = threading.Lock()
    self._cond = threading.Condition(self._lock)
    self._conns: Set[socket.socket] = set()
    self._accept_thread: Optional[threading.Thread] = None
    self._cleanup_thread: Optional[threading.Thread] = None
    self._conn_threads: Set[threading.Thread] = set()
    self._conn_sem = threading.Semaphore(_MAX_ACTIVE_SERVER_CONNS)
    self._outgoing: Dict[bytes, _OutgoingBulkTransfer] = {}
    self._incoming: Dict[bytes, _IncomingBulkTransfer] = {}
    self._unclaimed_bytes: int = 0
    self._completed_tids: Set[bytes] = set()
    self._completed_tids_order: Deque[bytes] = collections.deque()

  def _claim_incoming_entry_locked(self, entry: _IncomingBulkTransfer) -> None:
    if not entry.claimed:
      entry.claimed = True
      assert self._unclaimed_bytes >= entry.total_bytes, (
          f"Unclaimed byte accounting underflow: {self._unclaimed_bytes} < "
          f"{entry.total_bytes}"
      )
      self._unclaimed_bytes -= entry.total_bytes
      self._cond.notify_all()

  def _retire_incoming_locked(
      self, tid: bytes, entry: _IncomingBulkTransfer
  ) -> _IncomingBulkTransfer:
    """Removes `tid` from `_incoming`, reclaims unclaimed budget, and records tombstone."""
    self._incoming.pop(tid, None)
    self._claim_incoming_entry_locked(entry)
    if tid not in self._completed_tids:
      self._completed_tids.add(tid)
      self._completed_tids_order.append(tid)
      if len(self._completed_tids_order) > _MAX_COMPLETED_TIDS:
        oldest = self._completed_tids_order.popleft()
        self._completed_tids.discard(oldest)
    return entry

  def start(self, port: int = 0) -> int:
    """Binds a listening TCP socket on `port` (0 = ephemeral) and starts worker threads."""
    sock: Optional[socket.socket] = None
    try:
      sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
      sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      with contextlib.suppress(OSError):
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
      sock.bind(("::", port))
    except OSError as exc:
      if sock is not None:
        sock.close()
        sock = None
      if exc.errno not in (
          errno.EAFNOSUPPORT,
          errno.EADDRNOTAVAIL,
          errno.EPROTONOSUPPORT,
      ):
        raise
      sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
      except BaseException:
        sock.close()
        raise
    try:
      sock.listen(128)
      self._sock = sock
      self.port = int(sock.getsockname()[1])
    except BaseException:
      sock.close()
      self._sock = None
      raise
    self._accept_thread = threading.Thread(
        target=self._accept_loop,
        name=f"tunix-bulk-accept-{self.port}",
        daemon=True,
    )
    self._cleanup_thread = threading.Thread(
        target=self._cleanup_loop,
        name=f"tunix-bulk-cleanup-{self.port}",
        daemon=True,
    )
    self._accept_thread.start()
    self._cleanup_thread.start()
    return self.port

  def stop(self) -> None:
    """Shuts down the listening socket, active connections, and pending transfers."""
    self._stop_event.set()
    if self._sock is not None:
      _shutdown_and_close_socket(self._sock)
      self._sock = None
    with self._lock:
      conns = list(self._conns)
      self._conns.clear()
      inc_list = list(self._incoming.values())
      self._incoming.clear()
      self._unclaimed_bytes = 0
      self._cond.notify_all()
      out_list = list(self._outgoing.values())
      self._outgoing.clear()
      conn_threads = list(self._conn_threads)
    stop_err = ConnectionError("BulkTransferServer stopped.")
    for out in out_list:
      out.mark_failed(stop_err)
    for inc in inc_list:
      inc.mark_failed(stop_err)
    for conn in conns:
      _shutdown_and_close_socket(conn)
    deadline = time.monotonic() + 1.0
    all_threads = [
        self._accept_thread,
        self._cleanup_thread,
        *conn_threads,
    ]
    curr_thread = threading.current_thread()
    for t in all_threads:
      if t is not None and t.is_alive() and t is not curr_thread:
        rem = max(0.0, deadline - time.monotonic())
        t.join(timeout=rem)
    for inc in inc_list:
      inc.release_views()

  def _cleanup_loop(self) -> None:
    """Periodically evicts unclaimed incoming transfers that exceed `_INCOMING_TTL_S`."""
    while not self._stop_event.wait(timeout=0.5):
      evicted: List[_IncomingBulkTransfer] = []
      now = time.monotonic()
      with self._lock:
        stale_tids = [
            tid
            for tid, entry in self._incoming.items()
            if not entry.claimed and (now - entry.created_at) > _INCOMING_TTL_S
        ]
        for tid in stale_tids:
          entry = self._incoming.get(tid)
          if entry is not None:
            evicted.append(self._retire_incoming_locked(tid, entry))
      for old_entry in evicted:
        old_entry.release_views()

  def register_outgoing(
      self,
      views: Sequence[_SegmentedBulkView],
      total_bytes: int,
      loop: asyncio.AbstractEventLoop,
  ) -> Tuple[bytes, asyncio.Future[bool]]:
    tid = _allocate_transfer_id()
    done_fut: asyncio.Future[bool] = loop.create_future()
    entry = _OutgoingBulkTransfer(views, total_bytes, loop, done_fut)
    with self._lock:
      self._outgoing[tid] = entry
    return tid, done_fut

  def unregister_outgoing(self, tid: bytes) -> None:
    with self._lock:
      self._outgoing.pop(tid, None)

  def _get_or_create_incoming(
      self,
      tid: bytes,
      total_bytes: int,
      *,
      claim: bool = False,
  ) -> Optional[_IncomingBulkTransfer]:
    """Returns or allocates the `_IncomingBulkTransfer` state for `tid`."""
    if total_bytes <= 0 or total_bytes > _MAX_BULK_TRANSFER_BYTES:
      raise ValueError(f"Invalid bulk transfer total_bytes: {total_bytes}.")

    def _over_unclaimed_budget() -> bool:
      return (
          not claim
          and self._unclaimed_bytes > 0
          and self._unclaimed_bytes + total_bytes
          > _MAX_UNCLAIMED_INCOMING_BYTES
      )

    with self._cond:
      self._cond.wait_for(
          lambda: (
              self._stop_event.is_set()
              or tid in self._completed_tids
              or tid in self._incoming
              or not _over_unclaimed_budget()
          ),
          timeout=float(_UNCLAIMED_BUDGET_WAIT_S),
      )
      if self._stop_event.is_set() or tid in self._completed_tids:
        return None
      entry = self._incoming.get(tid)
      if entry is None:
        # Always allow a single transfer up to _MAX_BULK_TRANSFER_BYTES when no
        # other unclaimed incoming transfers are occupying the budget.
        if _over_unclaimed_budget():
          raise MemoryError(
              "Bulk transfer unclaimed incoming byte budget exceeded: "
              f"{self._unclaimed_bytes + total_bytes} > "
              f"{_MAX_UNCLAIMED_INCOMING_BYTES}."
          )
        entry = _IncomingBulkTransfer(total_bytes, claimed=claim)
        self._incoming[tid] = entry
        if not claim:
          self._unclaimed_bytes += total_bytes
        self._cond.notify_all()
      else:
        if entry.total_bytes != total_bytes:
          raise ValueError(
              "Conflicting total_bytes for incoming bulk transfer: "
              f"{total_bytes} != {entry.total_bytes}."
          )
        if claim:
          self._claim_incoming_entry_locked(entry)
      return entry

  async def wait_incoming(
      self,
      tid: bytes,
      total_bytes: int,
      buffer_lengths: Sequence[int],
  ) -> Dict[int, np.ndarray]:
    """Waits for all incoming `_OP_PUSH` stripes for `tid` and returns bulk buffers."""
    if total_bytes == 0:
      return {}
    entry = self._get_or_create_incoming(tid, total_bytes, claim=True)
    if entry is None:
      raise RuntimeError(
          "Bulk transfer was already completed or evicted after TTL."
      )
    loop = asyncio.get_running_loop()
    fut: Optional[asyncio.Future[bool]] = None
    existing_err: Optional[Exception] = None
    with entry.lock:
      if entry.error is not None:
        existing_err = entry.error
      elif entry.remaining_bytes > 0:
        entry.loop = loop
        fut = loop.create_future()
        entry.done_fut = fut
    try:
      if existing_err is not None:
        raise existing_err
      if fut is not None:
        await fut
      result_buffers: Dict[int, np.ndarray] = {}
      for i, length in enumerate(buffer_lengths):
        if length > 0:
          buf = entry.buffers.get(i)
          rec_len = entry.received_per_buf.get(i, 0)
          if buf is None or len(buf) != length or rec_len != length:
            raise RuntimeError(
                f"Bulk push buffer {i} missing or size mismatch: expected "
                f"{length} bytes, received {rec_len}."
            )
          result_buffers[i] = buf
      return result_buffers
    finally:
      with self._lock:
        self._retire_incoming_locked(tid, entry)
      entry.release_views()

  def _accept_loop(self) -> None:
    """Accepts incoming TCP data-plane connections and spawns reader threads."""
    sock = self._sock
    if sock is None:
      return
    while not self._stop_event.is_set():
      try:
        conn, _ = sock.accept()
      except OSError:
        break
      if not self._conn_sem.acquire(timeout=float(_BULK_SOCKET_TIMEOUT_S)):
        _shutdown_and_close_socket(conn)
        continue
      try:
        _configure_bulk_socket(conn)
      except OSError:
        self._conn_sem.release()
        _shutdown_and_close_socket(conn)
        continue
      with self._lock:
        if self._stop_event.is_set():
          self._conn_sem.release()
          _shutdown_and_close_socket(conn)
          break
        self._conns.add(conn)
        t = threading.Thread(
            target=self._serve_conn,
            args=(conn,),
            name=f"tunix-bulk-conn-{self.port}",
            daemon=True,
        )
        self._conn_threads.add(t)
      t.start()

  def _serve_conn(self, conn: socket.socket) -> None:
    """Handles `_OP_PULL` and `_OP_PUSH` stripe commands on a client socket."""
    header_buf = bytearray(_BULK_CMD_STRUCT.size)
    header_mv = memoryview(header_buf)
    active_inc_entry: Optional[_IncomingBulkTransfer] = None
    active_out_entry: Optional[_OutgoingBulkTransfer] = None
    try:
      while not self._stop_event.is_set():
        try:
          n = _recv_into_exact(conn, header_mv, _BULK_CMD_STRUCT.size)
        except (TimeoutError, OSError) as exc:
          logging.debug("Idle bulk connection closed: %s", exc)
          break
        if n != _BULK_CMD_STRUCT.size:
          break
        (
            op,
            tid,
            total_bytes,
            buf_idx,
            buf_total_len,
            offset,
            length,
        ) = _BULK_CMD_STRUCT.unpack(header_buf)
        if op == _OP_PULL:
          with self._lock:
            out_entry = self._outgoing.get(tid)
          if out_entry is None:
            break
          active_out_entry = out_entry
          if (
              buf_idx >= len(out_entry.views)
              or length == 0
              or buf_total_len != len(out_entry.views[buf_idx])
              or offset + length > len(out_entry.views[buf_idx])
          ):
            out_entry.mark_failed(
                ValueError(
                    f"Invalid OP_PULL bounds: buf_idx={buf_idx}, "
                    f"offset={offset}, length={length}."
                )
            )
            active_out_entry = None
            break
          view = out_entry.views[buf_idx]
          _send_bulk_view_slice(conn, view, offset, length)
          out_entry.mark_sent(length)
          active_out_entry = None
        elif op == _OP_PUSH:
          inc_entry = self._get_or_create_incoming(tid, total_bytes)
          if inc_entry is None:
            break
          active_inc_entry = inc_entry
          if length == 0 or offset + length > buf_total_len:
            inc_entry.mark_failed(
                ValueError(
                    f"Invalid OP_PUSH stripe bounds: offset={offset}, "
                    f"length={length}, buf_total_len={buf_total_len}."
                )
            )
            inc_entry.release_views()
            active_inc_entry = None
            break
          target_mv = inc_entry.get_or_alloc_view(buf_idx, buf_total_len)
          sub_mv = target_mv[offset : offset + length]
          try:
            got = _recv_into_exact(conn, sub_mv, length)
            if got != length:
              inc_entry.mark_failed(
                  ConnectionError(
                      f"Bulk push truncated: expected {length} bytes, got"
                      f" {got}."
                  )
              )
              inc_entry.release_views()
              active_inc_entry = None
              break
          finally:
            sub_mv.release()
          inc_entry.mark_received(buf_idx, length)
          _sendall_with_timeout(conn, b"\x01")
          active_inc_entry = None
        else:
          break
    except Exception as exc:  # pylint: disable=broad-exception-caught
      if not self._stop_event.is_set():
        logging.warning("Bulk transfer connection handler error: %s", exc)
      if active_inc_entry is not None:
        active_inc_entry.mark_failed(exc)
        active_inc_entry.release_views()
      if active_out_entry is not None:
        active_out_entry.mark_failed(exc)
    finally:
      header_mv.release()
      with self._lock:
        self._conns.discard(conn)
        self._conn_threads.discard(threading.current_thread())
      self._conn_sem.release()
      _shutdown_and_close_socket(conn)


def _plan_bulk_stripes(
    buffer_lengths: Sequence[int],
    *,
    num_hosts: int = 1,
) -> List[Tuple[int, int, int, int]]:
  """Plans `(buf_idx, buf_total_len, offset, take)` stripes with size bounds."""
  if not _is_exact_int(num_hosts) or num_hosts <= 0:
    raise ValueError(
        f"num_hosts must be a positive integer, got {num_hosts!r}."
    )
  max_stripes = min(_MAX_BULK_STRIPES * num_hosts, _MAX_TOTAL_BULK_STRIPES)
  total_bytes = sum(buffer_lengths)
  stripe_target = min(
      _MAX_BULK_STRIPE_BYTES,
      max(
          _MIN_BULK_STRIPE_BYTES,
          (total_bytes + max_stripes - 1) // max_stripes,
      ),
  )
  stripes: List[Tuple[int, int, int, int]] = []
  for buf_idx, length in enumerate(buffer_lengths):
    if length <= 0:
      continue
    for offset in range(0, length, stripe_target):
      take = min(stripe_target, length - offset)
      stripes.append((buf_idx, length, offset, take))
  return stripes


def _assign_stripe_hosts(
    stripes: Sequence[Tuple[int, int, int, int]],
    hosts: Sequence[str],
) -> List[str]:
  """Assigns each stripe to the host with the fewest cumulative bytes."""
  if not hosts:
    raise ValueError("bulk_hosts must not be empty.")
  num_hosts = len(hosts)
  if num_hosts == 1:
    host = hosts[0]
    return [host] * len(stripes)
  loads = [0] * num_hosts
  assigned: List[str] = []
  for _, _, _, take in stripes:
    best_idx = min(range(num_hosts), key=loads.__getitem__)
    assigned.append(hosts[best_idx])
    loads[best_idx] += take
  return assigned


class _BulkLayout(NamedTuple):
  """Computed buffer layout for out-of-band bulk TCP transfers."""

  bulk_lengths: List[int]
  packed_offsets: Dict[int, int]
  total_bulk_bytes: int


def _plan_bulk_buffer_layout(
    header_len: int,
    buffer_lengths: Sequence[int],
) -> _BulkLayout:
  """Single source of truth for bulk buffer partitioning and 64-byte alignment."""
  bulk_lengths: List[int] = []
  packed_offsets: Dict[int, int] = {}
  packed_tail_len = header_len
  for i, length in enumerate(buffer_lengths):
    if length == 0:
      bulk_lengths.append(0)
    elif length < _BULK_PACK_MAX_BYTES:
      packed_tail_len = (packed_tail_len + (_BULK_PACK_ALIGN - 1)) & ~(
          _BULK_PACK_ALIGN - 1
      )
      packed_offsets[i] = packed_tail_len
      packed_tail_len += length
      bulk_lengths.append(0)
    else:
      bulk_lengths.append(length)
  bulk_lengths.append(packed_tail_len)
  return _BulkLayout(bulk_lengths, packed_offsets, sum(bulk_lengths))


def _partition_bulk_views(
    views: Sequence[memoryview],  # pylint: disable=g-bare-generic
) -> Tuple[
    int,
    Tuple[int, ...],
    _BulkLayout,
    List[_SegmentedBulkView],
]:
  """Partitions `views` without copying into standalone and 64B-aligned trailing views."""
  header_mv = views[0]
  data_views = views[1:]
  header_len = len(header_mv)
  buffer_lengths = tuple(len(v) for v in data_views)
  layout = _plan_bulk_buffer_layout(header_len, buffer_lengths)
  bulk_views: List[_SegmentedBulkView] = []
  small_segments: List[memoryview] = [header_mv]  # pylint: disable=g-bare-generic
  cur_tail_off = header_len
  for i, mv in enumerate(data_views):
    if i in layout.packed_offsets:
      target_off = layout.packed_offsets[i]
      pad = target_off - cur_tail_off
      if pad > 0:
        small_segments.append(_ZERO_PAD_MV[:pad])
      small_segments.append(mv)
      cur_tail_off = target_off + len(mv)
      bulk_views.append(_SegmentedBulkView(()))
    elif len(mv) == 0:
      bulk_views.append(_SegmentedBulkView(()))
    else:
      bulk_views.append(_SegmentedBulkView((mv,)))
  bulk_views.append(_SegmentedBulkView(small_segments))
  return header_len, buffer_lengths, layout, bulk_views


def _reconstruct_target_buffers(
    header_len: int,
    buffer_lengths: Sequence[int],
    raw_bulk_buffers: Mapping[int, np.ndarray],
    layout: _BulkLayout,
) -> Tuple[np.ndarray, List[np.ndarray]]:
  """Reconstructs header and 64B-aligned target buffers via zero-copy views."""
  # Small buffers (< _BULK_PACK_MAX_BYTES) coalesced into the trailing packed
  # buffer are returned as 64-byte-aligned zero-copy views sharing `packed_buf`.
  # Callers retaining only a small subset of arrays long-term across steps
  # should `np.copy()` them if they wish to release the underlying packed buffer
  # early.
  packed_idx = len(buffer_lengths)
  expected_packed_len = layout.bulk_lengths[packed_idx]
  packed_buf = raw_bulk_buffers.get(packed_idx)
  if packed_buf is None or len(packed_buf) != expected_packed_len:
    raise RuntimeError(
        "Bulk packed tail buffer missing or size mismatch: expected "
        f"{expected_packed_len} bytes."
    )
  header_view = packed_buf[:header_len]
  target_buffers: List[np.ndarray] = []
  empty_u8 = np.empty(0, dtype=np.uint8)
  for i, length in enumerate(buffer_lengths):
    if length == 0:
      target_buffers.append(empty_u8)
    elif i in layout.packed_offsets:
      off = layout.packed_offsets[i]
      target_buffers.append(packed_buf[off : off + length])
    else:
      buf = raw_bulk_buffers.get(i)
      if buf is None or len(buf) != length:
        raise RuntimeError(
            f"Bulk buffer {i} missing or size mismatch: expected {length}"
            " bytes."
        )
      target_buffers.append(buf)
  return header_view, target_buffers


def _reconstruct_and_unpickle(
    header_len: int,
    buffer_lengths: Sequence[int],
    raw_bulk_buffers: Mapping[int, np.ndarray],
    layout: _BulkLayout,
) -> Any:
  """Reconstructs zero-copy buffer views and unpickles on `_SERDE_EXECUTOR`."""
  header_view, target_buffers = _reconstruct_target_buffers(
      header_len, buffer_lengths, raw_bulk_buffers, layout
  )
  header_mv = memoryview(header_view)
  try:
    return cloudpickle.loads(  # pylint: disable=g-unsafe-pickle-load
        header_mv, buffers=target_buffers
    )
  finally:
    header_mv.release()
    target_buffers.clear()


class _ActiveStripeSockets:
  """Tracks sockets in use by active stripe worker threads so cancellation can abort them."""

  def __init__(self) -> None:
    self._lock = threading.Lock()
    self._aborted = False
    self._socks: Set[socket.socket] = set()

  def register(self, sock: socket.socket) -> None:
    with self._lock:
      if self._aborted:
        _shutdown_socket_only(sock)
        raise ConnectionError("Bulk transfer aborted.")
      self._socks.add(sock)

  def unregister(self, sock: socket.socket) -> bool:
    """Removes `sock` and returns True if the socket may be returned to the pool."""
    with self._lock:
      self._socks.discard(sock)
      return not self._aborted

  def abort_all(self) -> None:
    """Shuts down active sockets under lock so owning threads cannot close the fd first."""
    with self._lock:
      self._aborted = True
      for sock in self._socks:
        _shutdown_socket_only(sock)
      self._socks.clear()


def _push_stripe_sync(
    host: str,
    port: int,
    tid: bytes,
    total_bytes: int,
    buf_idx: int,
    buf_total_len: int,
    view: _SegmentedBulkView,
    offset: int,
    length: int,
    active_socks: _ActiveStripeSockets,
) -> None:
  """Sends a buffer slice directly from `view` over a pooled TCP socket without GIL."""
  sock = _DEFAULT_BULK_SOCKET_POOL.acquire(host, port)
  try:
    active_socks.register(sock)
    _sendall_with_timeout(
        sock,
        _BULK_CMD_STRUCT.pack(
            _OP_PUSH, tid, total_bytes, buf_idx, buf_total_len, offset, length
        ),
    )
    _send_bulk_view_slice(sock, view, offset, length)
    try:
      ack = sock.recv(1, socket.MSG_WAITALL)
    except BlockingIOError as exc:
      raise TimeoutError("Bulk push ACK timed out.") from exc
    if ack != b"\x01":
      raise ConnectionError("Bulk push did not receive completion ACK.")
    if active_socks.unregister(sock):
      _DEFAULT_BULK_SOCKET_POOL.release(host, port, sock)
    else:
      _shutdown_and_close_socket(sock)
  except Exception:
    active_socks.unregister(sock)
    _shutdown_and_close_socket(sock)
    raise


def _pull_stripe_sync(
    host: str,
    port: int,
    tid: bytes,
    buf_idx: int,
    buf_total_len: int,
    inc_state: _IncomingBulkTransfer,
    offset: int,
    length: int,
    active_socks: _ActiveStripeSockets,
) -> None:
  """Reads a buffer slice directly into an uninitialized NumPy array via `recv_into` without GIL."""
  target_mv = inc_state.get_or_alloc_view(buf_idx, buf_total_len)
  sock = _DEFAULT_BULK_SOCKET_POOL.acquire(host, port)
  try:
    active_socks.register(sock)
    _sendall_with_timeout(
        sock,
        _BULK_CMD_STRUCT.pack(
            _OP_PULL, tid, 0, buf_idx, buf_total_len, offset, length
        ),
    )
    sub_mv = target_mv[offset : offset + length]
    try:
      got = _recv_into_exact(sock, sub_mv, length)
      if got != length:
        raise ConnectionError(
            f"Bulk pull truncated: expected {length} bytes, got {got}."
        )
    finally:
      sub_mv.release()
    if active_socks.unregister(sock):
      _DEFAULT_BULK_SOCKET_POOL.release(host, port, sock)
    else:
      _shutdown_and_close_socket(sock)
  except Exception:
    active_socks.unregister(sock)
    _shutdown_and_close_socket(sock)
    raise


class _BulkPushTracker:
  """Captures any exception raised inside `_stream_bulk_push_chunks` before gRPC swallows it."""

  def __init__(self) -> None:
    self.error: Optional[Exception] = None


class _BulkManifest(NamedTuple):
  """Wire manifest exchanged over gRPC for out-of-band bulk TCP transfers."""

  op: int
  tid: bytes
  port: int
  total_bulk_bytes: int
  header_len: int
  buffer_lengths: Tuple[int, ...]


async def _stream_bulk_pull_chunks(
    views: List[memoryview],  # pylint: disable=g-bare-generic
    raw_buffers: List[pickle.PickleBuffer],
    bulk_server: _BulkTransferServer,
) -> AsyncIterator[bytes]:
  """Streams a bulk manifest over gRPC while serving buffers via `_OP_PULL`."""
  tid: Optional[bytes] = None
  try:
    header_len, buffer_lengths, layout, bulk_views = _partition_bulk_views(
        views
    )
    total_bulk_bytes = layout.total_bulk_bytes
    loop = asyncio.get_running_loop()
    tid, done_fut = bulk_server.register_outgoing(
        bulk_views, total_bulk_bytes, loop
    )
    manifest_chunk = pickle.dumps(
        _BulkManifest(
            _OP_PULL,
            tid,
            bulk_server.port,
            total_bulk_bytes,
            header_len,
            buffer_lengths,
        ),
        protocol=5,
    )
    yield manifest_chunk
    if total_bulk_bytes > 0:
      await done_fut
  finally:
    if tid is not None:
      bulk_server.unregister_outgoing(tid)
    _release_views_and_buffers(views, raw_buffers)


async def _stream_bulk_push_chunks(
    views: List[memoryview],  # pylint: disable=g-bare-generic
    raw_buffers: List[pickle.PickleBuffer],
    bulk_hosts: Sequence[str],
    bulk_port: int,
    *,
    push_tracker: Optional[_BulkPushTracker] = None,
    seen_bulk_targets: Optional[Set[Tuple[str, int]]] = None,
) -> AsyncIterator[bytes]:
  """Pushes buffers via `_OP_PUSH` sockets and yields a lightweight manifest."""
  raw_futs: List[concurrent.futures.Future[None]] = []
  push_futs: List[asyncio.Future[None]] = []
  active_socks = _ActiveStripeSockets()
  completed_all = False
  try:
    if not bulk_hosts:
      raise ValueError("bulk_hosts must not be empty.")
    header_len, buffer_lengths, layout, bulk_views = _partition_bulk_views(
        views
    )
    total_bulk_bytes = layout.total_bulk_bytes
    tid = _allocate_transfer_id()
    loop = asyncio.get_running_loop()
    stripes = _plan_bulk_stripes(layout.bulk_lengths, num_hosts=len(bulk_hosts))
    stripe_hosts = _assign_stripe_hosts(stripes, bulk_hosts)
    for target_host, (buf_idx, buf_total_len, offset, take) in zip(
        stripe_hosts, stripes
    ):
      if seen_bulk_targets is not None:
        seen_bulk_targets.add((target_host, bulk_port))
      raw_fut = _BULK_IO_EXECUTOR.submit(
          _push_stripe_sync,
          target_host,
          bulk_port,
          tid,
          total_bulk_bytes,
          buf_idx,
          buf_total_len,
          bulk_views[buf_idx],
          offset,
          take,
          active_socks,
      )
      raw_futs.append(raw_fut)
      push_futs.append(asyncio.wrap_future(raw_fut, loop=loop))
    manifest_chunk = pickle.dumps(
        _BulkManifest(
            _OP_PUSH,
            tid,
            bulk_port,
            total_bulk_bytes,
            header_len,
            buffer_lengths,
        ),
        protocol=5,
    )
    yield manifest_chunk
    if push_futs:
      try:
        await asyncio.gather(*push_futs)
        completed_all = True
      except Exception as exc:
        if push_tracker is not None and push_tracker.error is None:
          push_tracker.error = exc
        raise
    else:
      completed_all = True
  finally:
    if not completed_all:
      if push_tracker is not None and push_tracker.error is None:
        for rf in raw_futs:
          if rf.done() and not rf.cancelled():
            rf_exc = rf.exception()
            if isinstance(rf_exc, Exception):
              push_tracker.error = rf_exc
              break
      active_socks.abort_all()
    _drain_and_release_async(
        raw_futs, lambda: _release_views_and_buffers(views, raw_buffers)
    )


def _can_use_bulk_transfer(
    total_bytes: int,
    views: Sequence[memoryview],  # pylint: disable=g-bare-generic
) -> bool:
  """Returns True if `views` qualify for out-of-band bulk TCP transfer."""
  num_buffers = len(views) - 1
  if num_buffers <= 0 or num_buffers > _MAX_BULK_BUFFERS:
    return False
  max_padding = (num_buffers + 1) * (_BULK_PACK_ALIGN - 1)
  return (
      _ASYNC_OFFLOAD_THRESHOLD_BYTES
      <= total_bytes
      <= (_MAX_BULK_TRANSFER_BYTES - max_padding)
  )


def _validate_bulk_manifest(meta: _BulkManifest) -> _BulkLayout:
  """Validates a `_BulkManifest` and returns its computed `_BulkLayout`."""
  if (
      not _is_exact_int(meta.op)
      or meta.op not in (_OP_PULL, _OP_PUSH)
      or not isinstance(meta.tid, bytes)
      or len(meta.tid) != _TRANSFER_TOKEN_BYTES
      or not _is_exact_int(meta.port)
      or meta.port <= 0
      or meta.port > 65535
      or not _is_exact_int(meta.total_bulk_bytes)
  ):
    raise ValueError("Invalid bulk manifest header fields.")
  if not _is_exact_int(meta.header_len) or meta.header_len <= 0:
    raise ValueError(f"Invalid bulk manifest header_len: {meta.header_len!r}.")
  if (
      not isinstance(meta.buffer_lengths, tuple)
      or not meta.buffer_lengths
      or len(meta.buffer_lengths) > _MAX_BULK_BUFFERS
      or any(
          not _is_exact_int(length) or length < 0
          for length in meta.buffer_lengths
      )
  ):
    raise ValueError(
        f"Invalid bulk manifest buffer_lengths: {meta.buffer_lengths!r}."
    )
  layout = _plan_bulk_buffer_layout(meta.header_len, meta.buffer_lengths)
  if (
      layout.total_bulk_bytes <= 0
      or layout.total_bulk_bytes > _MAX_BULK_TRANSFER_BYTES
  ):
    raise ValueError(
        "Invalid bulk transfer byte total in manifest:"
        f" {layout.total_bulk_bytes}."
    )
  if meta.total_bulk_bytes != layout.total_bulk_bytes:
    raise ValueError(
        f"Mismatched total_bulk_bytes in bulk manifest: {meta.total_bulk_bytes}"
        f" != {layout.total_bulk_bytes}."
    )
  return layout


async def _deserialize_from_async_chunks(
    chunks: AsyncIterable[bytes],
    *,
    allow_empty: bool = False,
    bulk_hosts: Sequence[str] = ("localhost",),
    bulk_server: Optional[_BulkTransferServer] = None,
    seen_bulk_targets: Optional[Set[Tuple[str, int]]] = None,
) -> Any:
  """Deserializes an object from an async iterable of chunks or zero-copy bulk sockets."""
  reassembler: Optional[_ChunkReassembler] = None
  offload = False
  loop: Optional[asyncio.AbstractEventLoop] = None
  pending_feeds: Deque[asyncio.Future[None]] = collections.deque()
  chunk_iter = chunks.__aiter__()
  try:
    try:
      first_chunk = await chunk_iter.__anext__()
    except StopAsyncIteration:
      if allow_empty:
        return None
      raise ValueError(
          "Cannot deserialize from an empty chunk stream."
      ) from None

    if not first_chunk and allow_empty:
      return None

    try:
      first_meta = pickle.loads(first_chunk)  # pylint: disable=g-unsafe-pickle-load
    except Exception as exc:
      raise ValueError("Invalid stream manifest header.") from exc

    if isinstance(first_meta, _BulkManifest):
      layout = _validate_bulk_manifest(first_meta)
      loop = asyncio.get_running_loop()
      if first_meta.op == _OP_PULL:
        if not bulk_hosts:
          raise ValueError("bulk_hosts must not be empty.")
        inc_state = _IncomingBulkTransfer(first_meta.total_bulk_bytes)
        raw_futs: List[concurrent.futures.Future[None]] = []
        pull_futs: List[asyncio.Future[None]] = []
        active_socks = _ActiveStripeSockets()
        completed_pull = False
        raw_bulk_buffers: Dict[int, np.ndarray] = {}
        try:
          stripes = _plan_bulk_stripes(
              layout.bulk_lengths, num_hosts=len(bulk_hosts)
          )
          stripe_hosts = _assign_stripe_hosts(stripes, bulk_hosts)
          for target_host, (
              buf_idx,
              buf_total_len,
              offset,
              take,
          ) in zip(stripe_hosts, stripes):
            if seen_bulk_targets is not None:
              seen_bulk_targets.add((target_host, first_meta.port))
            raw_fut = _BULK_IO_EXECUTOR.submit(
                _pull_stripe_sync,
                target_host,
                first_meta.port,
                first_meta.tid,
                buf_idx,
                buf_total_len,
                inc_state,
                offset,
                take,
                active_socks,
            )
            raw_futs.append(raw_fut)
            pull_futs.append(asyncio.wrap_future(raw_fut, loop=loop))
          if pull_futs:
            await asyncio.gather(*pull_futs)
          raw_bulk_buffers = dict(inc_state.buffers)
          completed_pull = True
        except OSError as exc:
          raise _BulkTransportError(str(exc)) from exc
        finally:
          if not completed_pull:
            active_socks.abort_all()
          _drain_and_release_async(raw_futs, inc_state.release_views)
        del inc_state
      else:
        if bulk_server is None:
          raise _BulkTransportError(
              "Bulk push manifest received, but server bulk transport is"
              " disabled."
          )
        if first_meta.port != bulk_server.port:
          raise _BulkTransportError(
              "Bulk push manifest port mismatch: client targeted port"
              f" {first_meta.port}, server listening on {bulk_server.port}."
          )
        raw_bulk_buffers = await bulk_server.wait_incoming(
            first_meta.tid, first_meta.total_bulk_bytes, layout.bulk_lengths
        )
      unpickle_fut = loop.run_in_executor(
          _SERDE_EXECUTOR,
          _reconstruct_and_unpickle,
          first_meta.header_len,
          first_meta.buffer_lengths,
          raw_bulk_buffers,
          layout,
      )
      del raw_bulk_buffers
      async for _ in chunk_iter:
        pass
      return await unpickle_fut

    reassembler = _ChunkReassembler(unpickled_meta=first_meta)
    if reassembler.total_bytes >= _ASYNC_OFFLOAD_THRESHOLD_BYTES:
      offload = True
      loop = asyncio.get_running_loop()

    async for chunk in chunk_iter:
      if offload and loop is not None:
        ops = reassembler.plan_feed(chunk)
        if ops:
          while pending_feeds and pending_feeds[0].done():
            pending_feeds.popleft().result()
          if len(pending_feeds) >= 16:
            await pending_feeds.popleft()
          pending_feeds.append(
              loop.run_in_executor(_SERDE_EXECUTOR, _execute_feed_ops, ops)
          )
          await asyncio.sleep(0)
      else:
        reassembler.feed(chunk)
    while pending_feeds:
      await pending_feeds.popleft()
  finally:
    if pending_feeds:
      await asyncio.gather(*pending_feeds, return_exceptions=True)
      pending_feeds.clear()
  if offload and loop is not None:
    return await loop.run_in_executor(_SERDE_EXECUTOR, reassembler.finish)
  return reassembler.finish()
