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

"""Universal Actor Model abstraction layer (`RemoteExecutionServer`, `ActorHandle`, `ActorPool`).

Eliminates RPC boilerplate (`rpc_generate`, `rpc_sync_weights`, etc.) across
worker types by serializing arbitrary method invocations (`submit`, `asubmit`)
over a universal execution protocol (`ExecutionRequest`).

Security Notes / Trust Boundaries:
  This module uses `cloudpickle` to serialize and deserialize dynamic execution
  requests and responses (`ExecutionRequest`, `ExecutionResponse`). Because
  `cloudpickle.loads()` executes arbitrary Python code via `__reduce__` gadgets
  during unpickling, this protocol must NEVER be exposed to unauthenticated or
  untrusted network traffic.
  For production deployment across trust boundaries (e.g. multi-tenant Borg jobs
  or external networks), ensure payloads are authenticated and encrypted via
  ALTS / mTLS channels (`secure_channel` / `secure_server_credentials`) or
  signed via shared HMAC-SHA256 signatures before unpickling.
  Where dynamic function shipping is not required, use a custom
  `pickle.Unpickler` (`find_class`) to whitelist only trusted domain data types
  (`int`, `str`, `dict`, `list`, `numpy.ndarray`, `data_types.*`).
"""

import abc
import asyncio
import collections
import concurrent.futures
import contextlib
import dataclasses
import hashlib
import inspect
import pickle
import threading
import time
import traceback as traceback_lib
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from absl import logging
import cloudpickle
import numpy as np

try:
  import grpc as _grpc_lib
  import grpc.aio as _grpc_aio_lib

  _GRPC_AVAILABLE = True
except ImportError:
  _grpc_lib = None
  _grpc_aio_lib = None
  _GRPC_AVAILABLE = False


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

# Default slice size (16 MiB) per frame on streaming gRPC calls. Must remain
# strictly smaller than _MAX_MESSAGE_BYTES.
_STREAM_CHUNK_BYTES = 16 * 1024 * 1024

# Buffers smaller than this threshold are coalesced via a single b"".join()
# pass; buffers at or above this threshold are emitted directly as standalone
# frames (or chunk_size slices) to avoid intermediate coalescing copies.
_COALESCE_THRESHOLD_BYTES = 256 * 1024

# Payloads at or above this byte threshold offload chunk reassembly and
# unpickling to _SERDE_EXECUTOR so multi-megabyte buffer operations do not
# block the asyncio event loop.
_ASYNC_OFFLOAD_THRESHOLD_BYTES = 512 * 1024

_SERDE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=8, thread_name_prefix="tunix-rpc-serde"
)


def _grpc_options(
    max_message_bytes: int = _MAX_MESSAGE_BYTES,
) -> List[Tuple[str, int]]:
  """Channel/server options lifting the message-size cap and enabling keepalive."""
  return [
      ("grpc.max_send_message_length", max_message_bytes),
      ("grpc.max_receive_message_length", max_message_bytes),
      ("grpc.keepalive_time_ms", 20000),
      ("grpc.keepalive_timeout_ms", 10000),
      ("grpc.keepalive_permit_without_calls", 1),
      ("grpc.http2.max_pings_without_data", 0),
      ("grpc.http2.max_ping_strikes", 0),
      ("grpc.http2.min_ping_interval_without_data_ms", 5000),
      ("grpc.http2.min_recv_ping_interval_without_data_ms", 5000),
      ("grpc.use_local_subchannel_pool", 1),
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


def _flush_pending_views(pending_views: List[memoryview]) -> bytes:
  """Materializes coalesced small views in a single allocation and copy."""
  if len(pending_views) == 1:
    data = pending_views[0].tobytes()
  else:
    data = b"".join(pending_views)
  pending_views.clear()
  return data


def _iter_serialized_chunks(
    obj: Any,
    chunk_size: int = _STREAM_CHUNK_BYTES,
) -> Iterator[bytes]:
  """Serializes `obj` with Pickle Protocol 5 and returns a chunk iterator.

  Pickling and out-of-band buffer extraction (`buffer_callback`) run eagerly
  when this function is called so any serialization error (`TypeError`,
  `pickle.PicklingError`) is raised immediately in the caller before opening a
  gRPC stream.

  Args:
    obj: Arbitrary Python object to serialize with cloudpickle.
    chunk_size: Maximum byte length of each yielded data chunk.

  Returns:
    An iterator yielding Frame 0 (pickled `(header_len, buffer_lengths)`
    manifest), followed by `chunk_size` slices of the pickle header and each
    out-of-band buffer.
  """
  if chunk_size <= 0:
    raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
  raw_buffers: List[pickle.PickleBuffer] = []
  views: List[memoryview] = []  # pylint: disable=g-bare-generic
  try:
    header_bytes = cloudpickle.dumps(
        obj, protocol=5, buffer_callback=raw_buffers.append
    )
    views.append(memoryview(header_bytes))
    for pb in raw_buffers:
      try:
        views.append(pb.raw())
      except BufferError:
        with memoryview(pb) as view:
          views.append(memoryview(view.tobytes()))
    manifest = cloudpickle.dumps(
        (len(views[0]), tuple(len(v) for v in views[1:]))
    )
  except Exception:
    for mv in views:
      mv.release()
    for pb in raw_buffers:
      pb.release()
    raise

  def _gen() -> Iterator[bytes]:
    try:
      yield manifest
      coalesce_limit = min(chunk_size, _COALESCE_THRESHOLD_BYTES)
      pending_views: List[memoryview] = []
      pending_bytes = 0
      for mv in views:
        mv_len = len(mv)
        if mv_len == 0:
          continue
        if mv_len >= coalesce_limit:
          if pending_views:
            yield _flush_pending_views(pending_views)
            pending_bytes = 0
          if mv_len <= chunk_size:
            yield mv.tobytes()
          else:
            for offset in range(0, mv_len, chunk_size):
              yield mv[offset : offset + chunk_size].tobytes()
        else:
          if pending_bytes + mv_len > chunk_size and pending_views:
            yield _flush_pending_views(pending_views)
            pending_bytes = 0
          pending_views.append(mv)
          pending_bytes += mv_len
      if pending_views:
        yield _flush_pending_views(pending_views)
    finally:
      for mv in views:
        mv.release()
      for pb in raw_buffers:
        pb.release()

  return _gen()


def _next_chunk_or_end(it: Iterator[bytes]) -> Optional[bytes]:
  return next(it, None)


async def _iter_async_from_sync_chunks(
    sync_iter: Iterator[bytes],
) -> AsyncIterator[bytes]:
  """Adapts a synchronous chunk iterator into a pipelined async iterator."""
  loop = asyncio.get_running_loop()
  first = next(sync_iter, None)
  if first is None:
    return
  next_fut: Optional[asyncio.Future[Optional[bytes]]] = loop.run_in_executor(
      _SERDE_EXECUTOR, _next_chunk_or_end, sync_iter
  )
  try:
    yield first
    while next_fut is not None:
      chunk = await next_fut
      if chunk is None:
        next_fut = None
        break
      next_fut = loop.run_in_executor(
          _SERDE_EXECUTOR, _next_chunk_or_end, sync_iter
      )
      yield chunk
  finally:
    try:
      if next_fut is not None:
        await next_fut
    except Exception:  # pylint: disable=broad-exception-caught
      pass
    finally:
      if hasattr(sync_iter, "close"):
        sync_iter.close()


class _ChunkReassembler:
  """Incremental zero-copy reassembler for Pickle Protocol 5 chunk streams."""

  def __init__(self, manifest_bytes: bytes):
    try:
      header_len, buffer_lengths = cloudpickle.loads(  # pylint: disable=g-unsafe-pickle-load
          manifest_bytes
      )
    except Exception as exc:
      raise ValueError("Invalid chunk stream manifest.") from exc

    if not isinstance(header_len, int) or header_len <= 0:
      raise ValueError(
          f"Invalid header_len in chunk stream manifest: {header_len!r}."
      )
    if not isinstance(buffer_lengths, (tuple, list)) or any(
        not isinstance(length, int) or length < 0 for length in buffer_lengths
    ):
      raise ValueError(
          "Invalid buffer_lengths in chunk stream manifest:"
          f" {buffer_lengths!r}."
      )

    self._target_lengths: List[int] = [header_len, *buffer_lengths]
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

  def feed(self, chunk: bytes) -> None:
    """Writes a chunk into lazily allocated uninitialized target buffers."""
    if not chunk:
      return
    chunk_len = len(chunk)
    chunk_pos = 0
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
      # NumPy slice assignment releases the GIL (NPY_BEGIN_ALLOW_THREADS).
      target_buf[self._target_offset : self._target_offset + take] = (
          np.frombuffer(chunk, dtype=np.uint8, count=take, offset=chunk_pos)
      )
      self._target_offset += take
      chunk_pos += take
      if self._target_offset == target_len:
        self._target_idx += 1
        self._target_offset = 0
        self._advance_empty_targets()

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


async def _deserialize_from_async_chunks(
    chunks: AsyncIterable[bytes],
    *,
    allow_empty: bool = False,
) -> Any:
  """Deserializes an object from an async iterable of chunks."""
  reassembler: Optional[_ChunkReassembler] = None
  offload = False
  loop: Optional[asyncio.AbstractEventLoop] = None
  pending_feed: Optional[asyncio.Future[None]] = None
  try:
    async for chunk in chunks:
      if reassembler is None:
        if not chunk and allow_empty:
          return None
        reassembler = _ChunkReassembler(chunk)
        if reassembler.total_bytes >= _ASYNC_OFFLOAD_THRESHOLD_BYTES:
          offload = True
          loop = asyncio.get_running_loop()
      else:
        if offload and loop is not None:
          if pending_feed is not None:
            await pending_feed
          pending_feed = loop.run_in_executor(
              _SERDE_EXECUTOR, reassembler.feed, chunk
          )
        else:
          reassembler.feed(chunk)
    if pending_feed is not None:
      await pending_feed
      pending_feed = None
  finally:
    if pending_feed is not None:
      try:
        await pending_feed
      except Exception:  # pylint: disable=broad-exception-caught
        pass
  if reassembler is None:
    if allow_empty:
      return None
    raise ValueError("Cannot deserialize from an empty chunk stream.")
  if offload and loop is not None:
    return await loop.run_in_executor(_SERDE_EXECUTOR, reassembler.finish)
  return reassembler.finish()


def _running_loop() -> Optional["asyncio.AbstractEventLoop"]:
  """Returns the currently running event loop, or None if there is none."""
  try:
    return asyncio.get_running_loop()
  except RuntimeError:
    return None


class ExecutionRequest:
  """Universal execution request payload wrapping request_id, method name, args, and kwargs."""

  def __init__(
      self,
      request_id: Optional[str] = None,
      method_name: Optional[str] = None,
      args: Optional[Sequence[Any]] = None,
      kwargs: Optional[Dict[str, Any]] = None,
  ):
    self.request_id = request_id
    self.method_name = method_name or "__call__"
    self.args: Tuple[Any, ...] = tuple(args or ())
    self.kwargs: Dict[str, Any] = dict(kwargs or {})
    if "request_id" in self.kwargs:
      raise ValueError(
          "'request_id' is a reserved framework parameter for remote execution "
          "and cannot be passed in method kwargs."
      )

  def serialize_chunks(
      self, chunk_size: int = _STREAM_CHUNK_BYTES
  ) -> Iterator[bytes]:
    """Serializes request into Pickle Protocol 5 out-of-band buffer chunks."""
    return _iter_serialized_chunks(
        (self.request_id, self.method_name, self.args, self.kwargs),
        chunk_size=chunk_size,
    )

  def serialize_async_chunks(
      self, chunk_size: int = _STREAM_CHUNK_BYTES
  ) -> AsyncIterator[bytes]:
    """Serializes request into an async stream of Pickle Protocol 5 chunks."""
    sync_iter = self.serialize_chunks(chunk_size=chunk_size)
    return _iter_async_from_sync_chunks(sync_iter)

  @classmethod
  def deserialize_chunks(cls, chunks: Iterable[bytes]) -> "ExecutionRequest":
    """Deserializes an ExecutionRequest from a synchronous stream of chunks."""
    # SECURITY WARNING: cloudpickle.loads executes arbitrary code via __reduce__
    # during deserialization. In production across untrusted boundaries, verify
    # ALTS/mTLS transport identity or cryptographic HMAC signatures before
    # calling deserialize_chunks(). Where dynamic function shipping is not
    # needed, use `pickle.Unpickler` (`find_class`) to whitelist only trusted
    # domain data types (`int`, `str`, `dict`, `list`, `data_types.*`).
    return cls(*_deserialize_from_chunks(chunks))

  @classmethod
  async def deserialize_async_chunks(
      cls, chunks: AsyncIterable[bytes]
  ) -> "ExecutionRequest":
    """Deserializes an ExecutionRequest from an async stream of chunks."""
    return cls(*(await _deserialize_from_async_chunks(chunks)))


class ExecutionResponse:
  """Universal execution response wrapping a result or a structured error."""

  def __init__(
      self,
      result: Any = None,
      error_message: Optional[str] = None,
      error_type: Optional[str] = None,
      traceback: Optional[str] = None,
      retryable: bool = False,
      request_id: Optional[str] = None,
  ):
    self.result = result
    self.error_message = error_message
    self.error_type = error_type
    self.traceback = traceback
    self.retryable = retryable
    self.request_id = request_id

  def _as_tuple(self) -> Tuple[Any, ...]:
    return (
        self.result,
        self.error_message,
        self.error_type,
        self.traceback,
        self.retryable,
        self.request_id,
    )

  def _record_serialization_error(self, e: Exception) -> None:
    err_msg = (
        f"failed to serialize result of type {type(self.result).__name__}: {e}"
    )
    self.result = None
    self.error_message = err_msg
    self.error_type = "ExecutionResponseSerializationError"
    self.traceback = traceback_lib.format_exc()
    self.retryable = False

  def serialize_chunks(
      self, chunk_size: int = _STREAM_CHUNK_BYTES
  ) -> Iterator[bytes]:
    """Serializes response into Pickle Protocol 5 out-of-band buffer chunks."""
    try:
      return _iter_serialized_chunks(self._as_tuple(), chunk_size=chunk_size)
    except Exception as e:  # pylint: disable=broad-exception-caught
      self._record_serialization_error(e)
      return _iter_serialized_chunks(self._as_tuple(), chunk_size=chunk_size)

  def serialize_async_chunks(
      self, chunk_size: int = _STREAM_CHUNK_BYTES
  ) -> AsyncIterator[bytes]:
    """Serializes response into an async stream of Pickle Protocol 5 chunks."""
    sync_iter = self.serialize_chunks(chunk_size=chunk_size)
    return _iter_async_from_sync_chunks(sync_iter)

  @classmethod
  def deserialize_chunks(cls, chunks: Iterable[bytes]) -> "ExecutionResponse":
    """Deserializes an ExecutionResponse from a synchronous stream of chunks."""
    # SECURITY WARNING: cloudpickle.loads executes arbitrary code during
    # unpickling. Ensure payload authenticity over trusted channels before
    # deserialization, or use custom `pickle.Unpickler` (`find_class`) to
    # whitelist only trusted domain data types.
    return cls(*_deserialize_from_chunks(chunks))

  @classmethod
  async def deserialize_async_chunks(
      cls,
      chunks: AsyncIterable[bytes],
      *,
      allow_empty: bool = False,
  ) -> Optional["ExecutionResponse"]:
    """Deserializes an ExecutionResponse from an async stream of chunks."""
    unpacked = await _deserialize_from_async_chunks(
        chunks, allow_empty=allow_empty
    )
    return None if unpacked is None else cls(*unpacked)

  def unwrap(self) -> Any:
    """Returns the result, or raises RuntimeError if the remote call failed."""
    if self.error_message is not None:
      message = (
          f"RemoteExecutionError [{self.error_type}]: {self.error_message}"
      )
      if self.traceback:
        message = f"{message}\nRemote traceback:\n{self.traceback}"
      raise RuntimeError(message)
    return self.result


class RemoteExecutionServer(abc.ABC):
  """Daemon that binds a target domain object and executes method calls dynamically."""

  def __init__(self, instance: Optional[Any] = None):
    self._instance: Optional[Any] = instance
    self._response_queue: Optional[asyncio.Queue[ExecutionResponse]] = None
    self._response_queue_loop: Optional[asyncio.AbstractEventLoop] = None
    self._request_counter: int = 0
    self._background_tasks: set[asyncio.Task[Any]] = set()

  def _get_response_queue(self) -> asyncio.Queue[ExecutionResponse]:
    """Returns the response queue bound to the currently running event loop."""
    try:
      loop = asyncio.get_running_loop()
    except RuntimeError:
      loop = None
    if self._response_queue is None or (
        loop is not None and self._response_queue_loop is not loop
    ):
      new_q: asyncio.Queue[ExecutionResponse] = asyncio.Queue()
      if self._response_queue is not None:
        while self._response_queue._queue:  # pylint: disable=protected-access
          new_q.put_nowait(
              self._response_queue._queue.popleft()  # pylint: disable=protected-access
          )
      self._response_queue = new_q
      self._response_queue_loop = loop
    return self._response_queue

  async def dispatch_task(self, request: ExecutionRequest) -> str:
    """Dispatches task execution asynchronously on server and returns task ACK ID."""
    if not request.request_id:
      self._request_counter += 1
      request.request_id = f"task_{self._request_counter}"
    task = asyncio.create_task(self._run_and_enqueue(request))
    self._background_tasks.add(task)
    task.add_done_callback(self._background_tasks.discard)
    return request.request_id

  async def _run_and_enqueue(self, request: ExecutionRequest) -> None:
    logging.debug(
        "[RemoteExecutionServer] Starting task %s method=%s",
        request.request_id,
        request.method_name,
    )
    response = await self.execute_request(request)
    if response.error_message:
      logging.debug(
          "[RemoteExecutionServer] Task %s failed: %s\n%s",
          request.request_id,
          response.error_message,
          response.traceback,
      )
    else:
      logging.debug(
          "[RemoteExecutionServer] Task %s finished successfully",
          request.request_id,
      )
    await self._get_response_queue().put(response)

  async def poll_response(
      self, timeout_s: float = LONG_POLL_TIMEOUT_S
  ) -> Optional[ExecutionResponse]:
    """Long-polls server-side response queue for completed task results."""
    try:
      if timeout_s == 0.0:
        return self._get_response_queue().get_nowait()
      return await asyncio.wait_for(
          self._get_response_queue().get(), timeout=timeout_s
      )
    except (asyncio.TimeoutError, asyncio.QueueEmpty):
      return None

  def register_instance(self, instance: Any) -> None:
    """Binds a local Python object (e.g., RolloutWorkerService, TrainerWorker) to the server."""
    self._instance = instance

  @property
  def bound_instance(self) -> Optional[Any]:
    """Returns the bound domain instance."""
    return self._instance

  @abc.abstractmethod
  def start_serving(self, port: int) -> None:
    """Starts network event loop listening on the specified port."""
    pass

  def execute_sync_request(
      self, request: ExecutionRequest
  ) -> ExecutionResponse:
    """Dynamically resolves and executes synchronous method on the bound instance."""
    if self._instance is None:
      return ExecutionResponse(
          error_message="RemoteExecutionServer has no registered instance.",
          error_type="InstanceNotBoundError",
          request_id=request.request_id,
      )

    target_name = request.method_name or "__call__"
    method = getattr(self._instance, target_name, None)
    if method is None or not callable(method):
      return ExecutionResponse(
          error_message=f"Method '{target_name}' not found on bound instance.",
          error_type="AttributeError",
          request_id=request.request_id,
      )

    if inspect.iscoroutinefunction(method):
      return ExecutionResponse(
          error_message=(
              f"Method '{target_name}' is a coroutine function; use asubmit()."
          ),
          error_type="RuntimeError",
          request_id=request.request_id,
      )

    try:
      result = method(*request.args, **request.kwargs)
      return ExecutionResponse(result=result, request_id=request.request_id)
    except Exception as e:  # pylint: disable=broad-exception-caught
      return ExecutionResponse(
          error_message=str(e),
          error_type=type(e).__name__,
          traceback=traceback_lib.format_exc(),
          request_id=request.request_id,
      )

  async def execute_request(
      self, request: ExecutionRequest
  ) -> ExecutionResponse:
    """Dynamically resolves and executes method on the bound instance."""
    if self._instance is None:
      return ExecutionResponse(
          error_message="RemoteExecutionServer has no registered instance.",
          error_type="InstanceNotBoundError",
          request_id=request.request_id,
      )

    target_name = request.method_name or "__call__"
    method = getattr(self._instance, target_name, None)
    if method is None or not callable(method):
      return ExecutionResponse(
          error_message=f"Method '{target_name}' not found on bound instance.",
          error_type="AttributeError",
          request_id=request.request_id,
      )

    try:
      if inspect.iscoroutinefunction(method):
        result = await method(*request.args, **request.kwargs)
      else:
        result = method(*request.args, **request.kwargs)
      return ExecutionResponse(result=result, request_id=request.request_id)
    except Exception as e:  # pylint: disable=broad-exception-caught
      return ExecutionResponse(
          error_message=str(e),
          error_type=type(e).__name__,
          traceback=traceback_lib.format_exc(),
          request_id=request.request_id,
      )


class InProcessRemoteExecutionServer(RemoteExecutionServer):
  """In-process execution engine for single-process testing and v0 dev."""

  def start_serving(self, port: int) -> None:
    pass


class GrpcRemoteExecutionServer(RemoteExecutionServer):
  """RemoteExecutionServer implementation speaking gRPC over physical TCP sockets."""

  def __init__(
      self,
      instance: Optional[Any] = None,
      *,
      stream_chunk_bytes: int = _STREAM_CHUNK_BYTES,
      max_message_bytes: int = _MAX_MESSAGE_BYTES,
  ):
    _validate_stream_config(stream_chunk_bytes, max_message_bytes)
    super().__init__(instance)
    self._server: Optional[Any] = None
    self._serve_loop: Optional[Any] = None
    self._stream_chunk_bytes = stream_chunk_bytes
    self._max_message_bytes = max_message_bytes

  async def _handle_execute(
      self, request_iterator: AsyncIterator[bytes], context: Any
  ) -> AsyncIterator[bytes]:
    """Handles bidirectional chunked streaming execution requests."""
    del context
    try:
      request = await ExecutionRequest.deserialize_async_chunks(
          request_iterator
      )
      response = await self.execute_request(request)
    except Exception as e:  # pylint: disable=broad-exception-caught
      response = ExecutionResponse(
          error_message=str(e),
          error_type=type(e).__name__,
          traceback=traceback_lib.format_exc(),
      )
    async for chunk in response.serialize_async_chunks(
        chunk_size=self._stream_chunk_bytes
    ):
      yield chunk

  async def _handle_dispatch_task(
      self, request_iterator: AsyncIterator[bytes], context: Any
  ) -> bytes:
    del context
    request = await ExecutionRequest.deserialize_async_chunks(request_iterator)
    request_id = await self.dispatch_task(request)
    return cloudpickle.dumps(request_id)

  async def _handle_poll_responses(
      self, request_bytes: bytes, context: Any
  ) -> AsyncIterator[bytes]:
    """Handles server-streaming long-polling for completed task responses."""
    del context
    timeout_s = (
        cloudpickle.loads(request_bytes)  # pylint: disable=g-unsafe-pickle-load
        if request_bytes
        else LONG_POLL_TIMEOUT_S
    )
    response = await self.poll_response(timeout_s=timeout_s)
    if response is None:
      return
    completed = False
    try:
      async for chunk in response.serialize_async_chunks(
          chunk_size=self._stream_chunk_bytes
      ):
        yield chunk
      completed = True
    finally:
      if not completed:
        self._get_response_queue().put_nowait(response)

  async def start_serving_async(self, port: int = 50051) -> Any:
    """Starts an asynchronous gRPC server listening on [::]:port."""
    if not _GRPC_AVAILABLE or _grpc_lib is None or _grpc_aio_lib is None:
      raise RuntimeError("grpc is not installed or available.")

    self._server = _grpc_aio_lib.server(
        options=_grpc_options(self._max_message_bytes)
    )
    handler = _grpc_lib.method_handlers_generic_handler(
        "tunix.ExecutionService",
        {
            "Execute": _grpc_lib.stream_stream_rpc_method_handler(
                self._handle_execute,
                request_deserializer=lambda b: b,
                response_serializer=lambda b: b,
            ),
            "DispatchTask": _grpc_lib.stream_unary_rpc_method_handler(
                self._handle_dispatch_task,
                request_deserializer=lambda b: b,
                response_serializer=lambda b: b,
            ),
            "PollResponses": _grpc_lib.unary_stream_rpc_method_handler(
                self._handle_poll_responses,
                request_deserializer=lambda b: b,
                response_serializer=lambda b: b,
            ),
        },
    )
    self._server.add_generic_rpc_handlers((handler,))
    # NOTE: add_insecure_port is for local loopback / isolated pod testing (experimental v0).
    # For production across trust boundaries, use secure_server_credentials (ALTS/mTLS).
    self._server.add_insecure_port(f"[::]:{port}")
    await self._server.start()
    return self._server

  @property
  def serve_loop(self) -> Optional[Any]:
    """The event loop running the blocking start_serving(), or None."""
    return self._serve_loop

  def start_serving(self, port: int = 50051) -> None:
    """Blocking: starts the gRPC server and serves until it is stopped.

    Runs an event loop for the server's lifetime.
    """
    if _running_loop() is not None:
      raise RuntimeError(
          "GrpcRemoteExecutionServer.start_serving() is blocking and cannot be "
          "called from a running event loop; await start_serving_async() and "
          "hold the server task instead."
      )
    loop = asyncio.new_event_loop()
    self._serve_loop = loop
    try:
      asyncio.set_event_loop(loop)
      loop.run_until_complete(self.start_serving_async(port))
      if self._server is not None:
        loop.run_until_complete(self._server.wait_for_termination())
        loop.run_until_complete(asyncio.sleep(0.1))
    finally:
      self._serve_loop = None
      asyncio.set_event_loop(None)
      loop.close()

  async def stop_serving(self, grace: float = 0.5) -> None:

    if self._server:
      await self._server.stop(grace)


class ActorHandle(abc.ABC):
  """Stateful 1-to-1 routing handle targeting a specific remote worker instance."""

  worker_id: Optional[str] = None

  @classmethod
  def from_address(
      cls,
      target_address: str,
      *,
      rpc_timeout_s: Optional[float] = RPC_TIMEOUT_S,
      stream_chunk_bytes: int = _STREAM_CHUNK_BYTES,
      max_message_bytes: int = _MAX_MESSAGE_BYTES,
  ) -> "ActorHandle":
    """Instantiates a remote actor handle targeting the specified string URI."""
    if target_address.startswith("grpc://") and _GRPC_AVAILABLE:
      return GrpcRemoteActorHandle(
          target_address=target_address,
          rpc_timeout_s=rpc_timeout_s,
          stream_chunk_bytes=stream_chunk_bytes,
          max_message_bytes=max_message_bytes,
      )
    return RemoteActorHandle(target_address=target_address)

  @abc.abstractmethod
  def submit(self, method_name: Optional[str] = None, *args, **kwargs) -> Any:
    """Synchronous / fire-and-forget method execution across actor handle."""
    pass

  @abc.abstractmethod
  async def asubmit(
      self, method_name: Optional[str] = None, *args, **kwargs
  ) -> Any:
    """Asynchronous coroutine returning the completed result or raising exception."""
    pass

  @abc.abstractmethod
  async def dispatch_task(
      self,
      request_id: Optional[str] = None,
      method_name: Optional[str] = None,
      *args,
      **kwargs,
  ) -> str:
    """Dispatches task asynchronously on remote server and receives task ACK ID."""
    pass

  @abc.abstractmethod
  async def poll_responses(
      self, timeout_s: float = LONG_POLL_TIMEOUT_S
  ) -> Optional[ExecutionResponse]:
    """Long-polls remote server response queue for completed result."""
    pass


class RemoteActorHandle(ActorHandle):
  """ActorHandle targeting a remote network worker address over gRPC/Stubby."""

  def __init__(self, target_address: str):
    self.target_address = target_address

  def submit(self, method_name: Optional[str] = None, *args, **kwargs) -> Any:
    del method_name, args, kwargs
    raise NotImplementedError(
        f"Remote execution over {self.target_address} not initialized."
    )

  async def asubmit(
      self, method_name: Optional[str] = None, *args, **kwargs
  ) -> Any:
    del method_name, args, kwargs
    raise NotImplementedError(
        f"Remote execution over {self.target_address} not initialized."
    )

  async def dispatch_task(
      self,
      request_id: Optional[str] = None,
      method_name: Optional[str] = None,
      *args,
      **kwargs,
  ) -> str:
    del method_name, args, request_id, kwargs
    raise NotImplementedError(
        f"Remote execution over {self.target_address} not initialized."
    )

  async def poll_responses(
      self, timeout_s: float = LONG_POLL_TIMEOUT_S
  ) -> Optional[ExecutionResponse]:
    del timeout_s
    raise NotImplementedError(
        f"Remote execution over {self.target_address} not initialized."
    )


class GrpcRemoteActorHandle(RemoteActorHandle):
  """ActorHandle connecting to GrpcRemoteExecutionServer over TCP sockets via gRPC."""

  def __init__(
      self,
      target_address: str,
      *,
      rpc_timeout_s: Optional[float] = RPC_TIMEOUT_S,
      stream_chunk_bytes: int = _STREAM_CHUNK_BYTES,
      max_message_bytes: int = _MAX_MESSAGE_BYTES,
  ):
    if not _GRPC_AVAILABLE or _grpc_aio_lib is None:
      raise RuntimeError("grpc is not installed or available.")
    _validate_stream_config(stream_chunk_bytes, max_message_bytes)
    self.target_address = target_address
    self._host_port = target_address.replace("grpc://", "")
    self._channel: Optional[Any] = None
    self._channel_loop: Optional[asyncio.AbstractEventLoop] = None
    self._rpc: Optional[Any] = None
    self._dispatch_rpc: Optional[Any] = None
    self._poll_rpc: Optional[Any] = None
    self._rpc_timeout_s = rpc_timeout_s
    self._stream_chunk_bytes = stream_chunk_bytes
    self._max_message_bytes = max_message_bytes
    # Blocking submit() runs on a persistent background event loop so repeated
    # calls reuse one channel. gRPC aio channels are bound to the loop that
    # created them, so they cannot be shared with the caller's async loop nor
    # survive a per-call asyncio.run() loop.
    self._sync_loop: Optional[Any] = None
    self._sync_thread: Optional[threading.Thread] = None
    self._sync_channel: Optional[Any] = None
    self._sync_rpc: Optional[Any] = None
    self._sync_lock = threading.Lock()

  def _make_rpc(self, channel: Any) -> Any:
    return channel.stream_stream(
        "/tunix.ExecutionService/Execute",
        request_serializer=lambda b: b,
        response_deserializer=lambda b: b,
    )

  async def _ensure_async_channel(self) -> Any:
    """Ensures the async gRPC channel and stubs are bound to the active loop."""
    assert _grpc_aio_lib is not None
    current_loop = _running_loop()
    if (
        self._channel is None
        or self._channel_loop is not current_loop
        or (self._channel_loop is not None and self._channel_loop.is_closed())
    ):
      old_channel = self._channel
      old_loop = self._channel_loop
      self._channel = _grpc_aio_lib.insecure_channel(
          self._host_port, options=_grpc_options(self._max_message_bytes)
      )
      self._channel_loop = current_loop
      self._rpc = self._make_rpc(self._channel)
      self._dispatch_rpc = self._channel.stream_unary(
          "/tunix.ExecutionService/DispatchTask",
          request_serializer=lambda b: b,
          response_deserializer=cloudpickle.loads,  # pylint: disable=g-unsafe-pickle-load
      )
      self._poll_rpc = self._channel.unary_stream(
          "/tunix.ExecutionService/PollResponses",
          request_serializer=cloudpickle.dumps,
          response_deserializer=lambda b: b,
      )
      if old_channel is not None:
        try:
          if (
              old_loop is not None
              and old_loop.is_running()
              and old_loop is not current_loop
          ):
            old_loop.call_soon_threadsafe(
                lambda ch=old_channel: asyncio.create_task(ch.close())
            )
          else:
            await asyncio.wait_for(old_channel.close(), timeout=1.0)
        except Exception:  # pylint: disable=broad-exception-caught
          pass
    return self._channel

  async def _execute_rpc(
      self,
      rpc: Any,
      method_name: Optional[str],
      args: Sequence[Any],
      kwargs: Dict[str, Any],
  ) -> Any:
    """Streams an ExecutionRequest over `rpc` and unwraps the ExecutionResponse."""
    request = ExecutionRequest(
        method_name=method_name, args=args, kwargs=kwargs
    )
    chunks = request.serialize_async_chunks(chunk_size=self._stream_chunk_bytes)
    call = rpc(chunks, timeout=self._rpc_timeout_s)
    response = await ExecutionResponse.deserialize_async_chunks(call)
    assert response is not None
    return response.unwrap()

  def submit(self, method_name: Optional[str] = None, *args, **kwargs) -> Any:
    """Blocking gRPC invocation; safe to call repeatedly.

    Runs on a persistent background event loop owned by this handle, so repeated
    calls reuse a single channel rather than establishing a new one each time.
    Cannot be called from within a running event loop (use asubmit()).
    """
    if _running_loop() is not None:
      raise RuntimeError(
          "GrpcRemoteActorHandle.submit() is blocking and cannot be called from"
          " a running event loop; use asubmit() instead."
      )
    loop = self._ensure_sync_loop()
    future = asyncio.run_coroutine_threadsafe(
        self._invoke_on_sync_loop(method_name, args, kwargs), loop
    )
    return future.result()

  def _ensure_sync_loop(self) -> Any:
    """Lazily starts (once) the background loop used by blocking submit()."""
    with self._sync_lock:
      if self._sync_loop is None:
        self._sync_loop = asyncio.new_event_loop()
        self._sync_thread = threading.Thread(
            target=self._sync_loop.run_forever,
            name=f"grpc-submit-{self._host_port}",
            daemon=True,
        )
        self._sync_thread.start()
      return self._sync_loop

  async def _invoke_on_sync_loop(
      self,
      method_name: Optional[str],
      args: Sequence[Any],
      kwargs: Dict[str, Any],
  ) -> Any:
    assert _grpc_aio_lib is not None
    if self._sync_rpc is None:
      self._sync_channel = _grpc_aio_lib.insecure_channel(
          self._host_port, options=_grpc_options(self._max_message_bytes)
      )
      self._sync_rpc = self._make_rpc(self._sync_channel)
    return await self._execute_rpc(self._sync_rpc, method_name, args, kwargs)

  async def asubmit(
      self, method_name: Optional[str] = None, *args, **kwargs
  ) -> Any:
    """Asynchronously invokes remote method over gRPC."""
    await self._ensure_async_channel()
    return await self._execute_rpc(self._rpc, method_name, args, kwargs)

  async def dispatch_task(
      self,
      request_id: Optional[str] = None,
      method_name: Optional[str] = None,
      *args,
      **kwargs,
  ) -> str:
    """Asynchronously dispatches task request on remote server, returning task ACK ID."""
    await self._ensure_async_channel()
    assert self._dispatch_rpc is not None
    request = ExecutionRequest(
        request_id=request_id, method_name=method_name, args=args, kwargs=kwargs
    )
    chunks = request.serialize_async_chunks(chunk_size=self._stream_chunk_bytes)
    return await self._dispatch_rpc(chunks, timeout=self._rpc_timeout_s)

  async def poll_responses(
      self, timeout_s: float = LONG_POLL_TIMEOUT_S
  ) -> Optional[ExecutionResponse]:
    """Long-polls remote server response queue for completed task results."""
    await self._ensure_async_channel()
    assert self._poll_rpc is not None
    call = self._poll_rpc(timeout_s, timeout=self._rpc_timeout_s)
    return await ExecutionResponse.deserialize_async_chunks(
        call, allow_empty=True
    )

  async def close(self) -> None:
    if self._channel is not None:
      await self._channel.close()
      self._channel = None
      self._channel_loop = None
      self._rpc = None
      self._dispatch_rpc = None
      self._poll_rpc = None
    sync_loop = self._sync_loop
    if sync_loop is not None:

      async def _close_sync_channel() -> None:
        if self._sync_channel is not None:
          await self._sync_channel.close()

      try:
        await asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(_close_sync_channel(), sync_loop)
        )
      except Exception:  # pylint: disable=broad-exception-caught
        pass
      sync_loop.call_soon_threadsafe(sync_loop.stop)
      if self._sync_thread is not None:
        await asyncio.get_running_loop().run_in_executor(
            None, self._sync_thread.join, 5
        )
        if self._sync_thread.is_alive():
          logging.warning(
              "Background sync thread '%s' failed to join within 5 seconds. "
              "This may lead to leaked thread resources.",
              self._sync_thread.name,
          )
      self._sync_loop = None
      self._sync_thread = None
      self._sync_channel = None
      self._sync_rpc = None


class InProcessActorHandle(ActorHandle):
  """ActorHandle bridging calls directly to an in-process RemoteExecutionServer."""

  def __init__(self, server: RemoteExecutionServer):
    self.server = server

  def submit(self, method_name: Optional[str] = None, *args, **kwargs) -> Any:
    """Executes method synchronously or raises runtime error if coroutine required."""
    request = ExecutionRequest(
        method_name=method_name, args=args, kwargs=kwargs
    )
    target_name = method_name or "__call__"
    method = getattr(self.server.bound_instance, target_name, None)
    if method and inspect.iscoroutinefunction(method):
      try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
          raise RuntimeError(
              "InProcessActorHandle.submit() cannot be called from a running "
              "async event loop for coroutine methods. Use asubmit() instead."
          )
      except RuntimeError as e:
        if "submit() cannot be called" in str(e):
          raise
      response = asyncio.run(self.server.execute_request(request))
      return response.unwrap()

    response = self.server.execute_sync_request(request)
    return response.unwrap()

  async def asubmit(
      self, method_name: Optional[str] = None, *args, **kwargs
  ) -> Any:
    """Executes method asynchronously over in-process server."""
    request = ExecutionRequest(
        method_name=method_name, args=args, kwargs=kwargs
    )
    return await self._run_async(request)

  async def dispatch_task(
      self,
      request_id: Optional[str] = None,
      method_name: Optional[str] = None,
      *args,
      **kwargs,
  ) -> str:
    """Dispatches task execution asynchronously on bound server and returns task ACK ID."""
    request = ExecutionRequest(
        request_id=request_id, method_name=method_name, args=args, kwargs=kwargs
    )
    return await self.server.dispatch_task(request)

  async def poll_responses(
      self, timeout_s: float = LONG_POLL_TIMEOUT_S
  ) -> Optional[ExecutionResponse]:
    """Long-polls bound server's response queue for completed results."""
    return await self.server.poll_response(timeout_s=timeout_s)

  async def _run_async(self, request: ExecutionRequest) -> Any:
    response = await self.server.execute_request(request)
    return response.unwrap()


class ActorPool(abc.ABC):
  """Stateless load-balanced routing across worker farms with out-of-order task streaming."""

  @abc.abstractmethod
  def add_actor(self, actor: Union[str, ActorHandle]) -> ActorHandle:
    """Adds a worker actor handle or string URI target address to the pool."""
    pass

  @abc.abstractmethod
  def submit(self, method_name: Optional[str] = None, *args, **kwargs) -> Any:
    """Submits request to least-loaded or next available worker in the pool."""
    pass

  @abc.abstractmethod
  async def asubmit(
      self, method_name: Optional[str] = None, *args, **kwargs
  ) -> Any:
    """Asynchronously submits request and returns completed result."""
    pass

  @abc.abstractmethod
  def as_completed_stream(
      self,
      tasks: Sequence[Tuple[str, str, Sequence[Any], Dict[str, Any]]],
  ) -> AsyncIterator[Any]:
    """Dispatches a batch of tasks across pool and yields results strictly out-of-order.

    Args:
      tasks: Sequence of task specifications formatted as 4-tuples:
        `(request_id, method_name, args, kwargs)`, where: - request_id: Unique
        request identifier string. - method_name: Target remote method name to
        execute on the worker instance. - args: Positional arguments sequence
        passed to the remote method. - kwargs: Keyword arguments dictionary
        passed to the remote method.
    """
    raise NotImplementedError


def stable_route_hash(route_key: Any) -> int:
  """Maps a sticky routing key to a process-stable non-negative integer.

  Args:
    route_key: The sticky routing key. Non-negative integers are returned as-is;
      any other value is hashed via its `str()` representation.

  Returns:
    A non-negative integer suitable for `% num_actors` bucketing.

  Raises:
    ValueError: If `route_key` is a negative integer.
  """
  if isinstance(route_key, int):
    if route_key < 0:
      raise ValueError(
          "An integer route_key is an explicit shard index and must be"
          f" non-negative, got {route_key}."
      )
    return route_key
  digest = hashlib.blake2b(str(route_key).encode("utf-8"), digest_size=8)
  return int.from_bytes(digest.digest(), "big")


class RoutingActorPool(ActorPool):
  """ActorPool with smart task routing (`route_key affinity, round-robin fallback`).

  Args:
    actors: Initial sequence of worker actor handles or string URI targets.
    router: Optional custom routing callable `(actors, method_name, args,
      kwargs) -> ActorHandle` or router module/object providing per-method
      handlers matching `method_name` with signature `(actors, args, kwargs) ->
      ActorHandle`.
  """

  def __init__(
      self,
      actors: Optional[Sequence[Union[str, ActorHandle]]] = None,
      *,
      router: Optional[Union[Callable[..., ActorHandle], Any]] = None,
  ):
    self._actors: List[ActorHandle] = []
    for a in actors or []:
      self.add_actor(a)
    self._idx = 0
    self.router = router

  @property
  def actors(self) -> List[ActorHandle]:
    return list(self._actors)

  def add_actor(self, actor: Union[str, ActorHandle]) -> ActorHandle:
    if isinstance(actor, str):
      handle = ActorHandle.from_address(actor)
    elif isinstance(actor, ActorHandle):
      handle = actor
    else:
      raise TypeError(f"Expected str or ActorHandle, got {type(actor)}")
    if handle not in self._actors:
      self._actors.append(handle)
    return handle

  def remove_actor(self, actor: ActorHandle) -> bool:
    if actor in self._actors:
      self._actors.remove(actor)
      return True
    return False

  def select_actor(
      self,
      method_name: Optional[str] = None,
      args: Sequence[Any] = (),
      kwargs: Optional[Dict[str, Any]] = None,
  ) -> ActorHandle:
    """Selects target actor via custom router, route_key affinity, or round-robin."""
    return self._get_next_actor(method_name, args, kwargs)

  def _get_next_actor(
      self,
      method_name: Optional[str] = None,
      args: Sequence[Any] = (),
      kwargs: Optional[Dict[str, Any]] = None,
  ) -> ActorHandle:
    """Selects target actor via custom router, route_key affinity, or round-robin.

    Args:
      method_name: Target remote method being invoked.
      args: Positional arguments passed to the method call.
      kwargs: Keyword arguments passed to the method call. If this dictionary
        contains `route_key`, process-stable hash routing
        (`stable_route_hash(route_key) % N`) is used for sticky endpoint
        affinity (popped prior to remote dispatch).

    Returns:
      The selected `ActorHandle` target worker.

    Raises:
      RuntimeError: If the pool contains no registered ActorHandles.
    """

    if not self._actors:

      raise RuntimeError(
          "RoutingActorPool contains no registered ActorHandles."
      )

    kwargs = kwargs or {}
    if self.router is not None:
      if (
          method_name
          and hasattr(self.router, method_name)
          and callable(getattr(self.router, method_name))
      ):
        return getattr(self.router, method_name)(self._actors, args, kwargs)
      elif callable(self.router):
        return self.router(self._actors, method_name, args, kwargs)  # pyrefly: ignore[bad-return]
      else:
        raise TypeError(
            f"Router object {type(self.router)} must provide a method matching "
            f"'{method_name}' or be callable."
        )

    # Check for sticky routing key (e.g. route_key for KV-cache locality)
    route_key = kwargs.get("route_key")

    if route_key is not None:
      # Not builtin `hash()`: it salts `str` per process, so placement would
      # not survive a restart.
      # TODO(tunix-dev): `% len(self._actors)` remaps every key when pool
      # membership changes, not just the keys on the affected actor. Switch to
      # rendezvous (HRW) hashing before adding actor eviction.
      return self._actors[stable_route_hash(route_key) % len(self._actors)]

    # Default fallback: round-robin load balancing across all endpoints
    actor = self._actors[self._idx % len(self._actors)]
    self._idx += 1
    return actor

  def submit(self, method_name: Optional[str] = None, *args, **kwargs) -> Any:
    actor = self._get_next_actor(method_name, args, kwargs)
    kwargs.pop("route_key", None)
    return actor.submit(method_name, *args, **kwargs)

  async def asubmit(
      self, method_name: Optional[str] = None, *args, **kwargs
  ) -> Any:
    actor = self._get_next_actor(method_name, args, kwargs)
    kwargs.pop("route_key", None)
    return await actor.asubmit(method_name, *args, **kwargs)

  async def as_completed_stream(
      self,
      tasks: Sequence[Tuple[str, str, Sequence[Any], Dict[str, Any]]],
  ) -> AsyncIterator[Any]:
    """Dispatches a static batch of tasks across pool workers and yields results as they complete.

    Intended Use Case:
      Simple, one-shot static batch processing where the full list of tasks is
      known upfront (similar to `asyncio.as_completed`). This convenience
      wrapper provides fail-fast behavior: if any task raises an exception, the
      exception is immediately re-raised in the caller's stream and remaining
      tasks are cancelled.

      For dynamic task enqueuing (e.g. submitting new tasks as existing ones
      finish) or fault-isolated streaming (where individual task errors do not
      terminate the stream), use `execution_session()` instead.

    Args:
      tasks: Sequence of task specifications formatted as 4-tuples:
        `(request_id, method_name, args, kwargs)`, where: - request_id: Unique
        request identifier string. - method_name: Target remote method name to
        execute on the worker instance. - args: Positional arguments sequence
        passed to the remote method. - kwargs: Keyword arguments dictionary
        passed to the remote method.
    """
    if not self._actors:
      raise RuntimeError(
          "RoutingActorPool contains no registered ActorHandles."
      )
    if not tasks:
      return
    async with self.execution_session(tasks) as session:
      async for result, exc in session.as_completed():
        if exc is not None:
          raise exc
        yield result

  @contextlib.asynccontextmanager
  async def execution_session(
      self,
      initial_tasks: Optional[
          Sequence[Tuple[str, str, Sequence[Any], Dict[str, Any]]]
      ] = None,
      least_loaded: bool = False,
      *,
      config: Optional["PoolSessionConfig"] = None,
  ) -> AsyncIterator["PoolExecutionSession"]:
    """Creates a dynamic, fault-isolated execution session over the worker pool.

    Intended Use Case:
      Long-running worker pipelines, dynamic task enqueuing, and fault-tolerant
      batch processing. Within the `async with` session block, callers can:
        1. Dynamically enqueue new tasks at any time via `await
        session.submit()`.
        2. Consume completions out-of-order via `session.as_completed()`, which
           yields `(result, exception)` tuples without terminating the stream
           when an individual task fails.
        3. Rely on automatic background worker polling and clean task
        cancellation
           upon session exit.

    Args:
      initial_tasks: Optional sequence of initial task specifications to
        dispatch upon entering the session, formatted as 4-tuples `(request_id,
        method_name, args, kwargs)`, where: - request_id: Unique request
        identifier string. - method_name: Target remote method name to execute
        on the worker instance. - args: Positional arguments sequence passed to
        the remote method. - kwargs: Keyword arguments dictionary passed to the
        remote method.
      config: Optional `PoolSessionConfig` governing eviction, retry,
        concurrency caps, per-task timeout, and zero-worker / pending-worker
        hold policies.
    """
    session = PoolExecutionSession(
        self, least_loaded=least_loaded, config=config
    )
    try:
      if initial_tasks:
        for request_id, method_name, args, kwargs in initial_tasks:
          await session.submit(request_id, method_name, *args, **kwargs)
      yield session
    finally:
      await session.close()


# Bound on remembered route_key placements per least-loaded session.
_MAX_ROUTE_PLACEMENTS = 65536


@dataclasses.dataclass(frozen=True, kw_only=True)
class PoolSessionConfig:
  """Policy and capacity configuration for `PoolExecutionSession`.

  Attributes:
    evict_on_failure: Whether to automatically remove a failing worker from the
      pool when a transport or execution failure occurs.
    retry_on_worker_failure: Whether to re-dispatch tasks that were in-flight on
      a failed worker.
    max_task_retries: Maximum number of retries per `request_id` on worker
      failure.
    on_worker_evicted: Optional callback `(actor, exc)` invoked at most once per
      membership when a worker is evicted.
    max_in_flight_per_worker: Optional default cap on concurrent in-flight tasks
      per worker.
    worker_max_in_flight: Optional per-worker override mapping `ActorHandle` to
      max in-flight tasks.
    has_pending_workers_fn: Optional predicate returning True when replacement
      workers are warming up (e.g. in `PENDING_WEIGHT_SYNC`), allowing tasks to
      be held in `_pending_queue` instead of failing immediately when active
      workers temporarily drop to zero.
    task_timeout_s: Optional per-task execution timeout in seconds once a task
      is dispatched to a worker.
    retain_pending_on_zero_workers: Whether to hold queued/retried tasks in
      `_pending_queue` instead of failing them immediately when the active
      worker pool temporarily drops to zero.
  """

  evict_on_failure: bool = False
  retry_on_worker_failure: bool = False
  max_task_retries: int = 3
  on_worker_evicted: Optional[
      Callable[[ActorHandle, Optional[BaseException]], None]
  ] = None
  max_in_flight_per_worker: Optional[int] = None
  worker_max_in_flight: Optional[Mapping[ActorHandle, int]] = None
  has_pending_workers_fn: Optional[Callable[[], bool]] = None
  task_timeout_s: Optional[float] = None
  retain_pending_on_zero_workers: bool = False

  def __post_init__(self) -> None:
    if self.max_task_retries < 0:
      raise ValueError("max_task_retries must be non-negative")
    if (
        self.max_in_flight_per_worker is not None
        and self.max_in_flight_per_worker <= 0
    ):
      raise ValueError("max_in_flight_per_worker must be positive")
    if self.task_timeout_s is not None and self.task_timeout_s <= 0:
      raise ValueError("task_timeout_s must be positive")
    if self.worker_max_in_flight:
      for limit in self.worker_max_in_flight.values():
        if limit <= 0:
          raise ValueError("worker_max_in_flight values must be positive")


class PoolExecutionSession:
  """Dynamic, fault-isolated execution session for a RoutingActorPool.

  Intended Use Case:
    Managed by `RoutingActorPool.execution_session()`. Provides an interactive
    handle to submit tasks (`submit`) and consume out-of-order completions and
    exceptions (`as_completed`) without terminating the stream on individual
    task
    failures.
  """

  def __init__(
      self,
      pool: RoutingActorPool,
      *,
      least_loaded: bool = False,
      config: Optional[PoolSessionConfig] = None,
  ):
    cfg = config or PoolSessionConfig()
    self._worker_max_in_flight: Dict[ActorHandle, int] = (
        {actor: int(limit) for actor, limit in cfg.worker_max_in_flight.items()}
        if cfg.worker_max_in_flight
        else {}
    )

    self._pool = pool
    self._least_loaded = least_loaded
    # route_key -> actor it was last placed on, for least_loaded affinity.
    self._placements: collections.OrderedDict[Any, ActorHandle] = (
        collections.OrderedDict()
    )
    self._evict_on_failure = cfg.evict_on_failure
    self._retry_on_worker_failure = cfg.retry_on_worker_failure
    self._max_task_retries = cfg.max_task_retries
    self._on_worker_evicted = cfg.on_worker_evicted
    self._has_pending_workers_fn = cfg.has_pending_workers_fn
    self._max_in_flight_per_worker = (
        int(cfg.max_in_flight_per_worker)
        if cfg.max_in_flight_per_worker is not None
        else None
    )
    self._task_timeout_s: Optional[float] = (
        float(cfg.task_timeout_s) if cfg.task_timeout_s is not None else None
    )
    self._retain_pending_on_zero_workers = bool(
        cfg.retain_pending_on_zero_workers
    )
    self._evictions_total = 0
    self._retries_total = 0
    self._response_queue: asyncio.Queue[Any] = asyncio.Queue()
    self._active_workers: set[ActorHandle] = set()
    # Actors currently considered pool members by this session. Eviction
    # removes an actor from this set so `on_worker_evicted` fires at most once
    # per membership, without retaining dead handles after they leave.
    self._membership_lock = threading.Lock()
    self._known_actors: set[ActorHandle] = set(pool.actors)
    self._dispatched_tasks: Dict[ActorHandle, set[str]] = {}
    self._task_dispatch_times: Dict[str, float] = {}
    self._task_last_failed_actor: Dict[str, ActorHandle] = {}
    self._pending_queue: collections.deque[str] = collections.deque()
    self._task_payloads: Dict[
        str, Tuple[Optional[str], Tuple[Any, ...], Dict[str, Any]]
    ] = {}
    self._task_retries: Dict[str, int] = {}
    self._failed_tasks: collections.deque[
        Tuple[
            str,
            Tuple[Optional[str], Tuple[Any, ...], Dict[str, Any]],
            Exception,
        ]
    ] = collections.deque()
    self._poll_tasks: set[asyncio.Task[Any]] = set()
    self._worker_poll_tasks: Dict[ActorHandle, asyncio.Task[Any]] = {}
    self._loop: Optional[asyncio.AbstractEventLoop] = None
    self._in_flight = 0
    self._closed = False
    self._draining_pending = False
    self._sentinel = object()

  def _can_retry_or_hold(self) -> bool:
    return (
        len(self._pool.actors) > 0
        or self._retain_pending_on_zero_workers
        or (
            self._has_pending_workers_fn is not None
            and self._has_pending_workers_fn()
        )
    )

  @property
  def max_in_flight_per_worker(self) -> Optional[int]:
    return self._max_in_flight_per_worker

  @property
  def task_timeout_s(self) -> Optional[float]:
    return self._task_timeout_s

  def set_task_timeout_s(self, task_timeout_s: Optional[float]) -> None:
    """Updates the per-task execution timeout in seconds."""
    if task_timeout_s is not None and task_timeout_s <= 0:
      raise ValueError("task_timeout_s must be positive")
    self._task_timeout_s = (
        float(task_timeout_s) if task_timeout_s is not None else None
    )

  @property
  def retain_pending_on_zero_workers(self) -> bool:
    return self._retain_pending_on_zero_workers

  def set_retain_pending_on_zero_workers(self, val: bool) -> None:
    self._retain_pending_on_zero_workers = bool(val)

  @property
  def evictions_total(self) -> int:
    """Total number of worker evictions observed by this session."""
    return self._evictions_total

  @property
  def retries_total(self) -> int:
    """Total number of task retries triggered by worker failures."""
    return self._retries_total

  @property
  def pending_count(self) -> int:
    """Number of submitted tasks waiting for worker capacity."""
    return len(self._pending_queue)

  @property
  def in_flight_count(self) -> int:
    """Total number of incomplete tasks in the session (dispatched + pending)."""
    return self._in_flight

  def has_pending_or_completed_work(self) -> bool:
    """Returns True if tasks are in-flight, queued for completion, or failed."""
    return (
        self._in_flight > 0
        or not self._response_queue.empty()
        or bool(self._failed_tasks)
    )

  def _bind_current_loop(self) -> asyncio.AbstractEventLoop:
    """Binds the session and its response queue to the currently running event loop."""
    loop = asyncio.get_running_loop()
    if self._loop is not loop:
      prev_loop = self._loop
      self._loop = loop
      new_q: asyncio.Queue[Tuple[Any, Optional[Exception]]] = asyncio.Queue()
      while not self._response_queue.empty():
        try:
          item = self._response_queue.get_nowait()
        except asyncio.QueueEmpty:
          break
        if item is not self._sentinel:
          new_q.put_nowait(item)
      self._response_queue = new_q
      if prev_loop is not None and not prev_loop.is_closed():
        for task in list(self._poll_tasks):
          if not task.done():
            if prev_loop.is_running():
              prev_loop.call_soon_threadsafe(task.cancel)
            else:
              task.cancel()
      self._poll_tasks.clear()
      self._worker_poll_tasks.clear()
      self._active_workers.clear()
      self._draining_pending = False
      for actor, dset in list(self._dispatched_tasks.items()):
        if dset and actor in self._pool.actors:
          self._ensure_worker_polling(actor)
    return loop

  def _run_on_session_loop(self, fn: Callable[[], None]) -> None:
    """Executes `fn` directly if on the session event loop, or via `call_soon_threadsafe`."""
    try:
      running_loop = asyncio.get_running_loop()
    except RuntimeError:
      running_loop = None

    if running_loop is not None and (
        self._loop is None
        or running_loop is self._loop
        or not self._loop.is_running()
    ):
      self._bind_current_loop()
      fn()
    elif self._loop is not None and self._loop.is_running():
      self._loop.call_soon_threadsafe(fn)
    else:
      fn()

  def _get_worker_limit(self, actor: ActorHandle) -> Optional[int]:
    return self._worker_max_in_flight.get(actor, self._max_in_flight_per_worker)

  def _worker_load(self, actor: ActorHandle) -> int:
    return len(self._dispatched_tasks.get(actor, ()))

  def _worker_has_capacity(self, actor: ActorHandle) -> bool:
    limit = self._get_worker_limit(actor)
    return limit is None or self._worker_load(actor) < limit

  def _select_actor_for_task(
      self,
      method_name: Optional[str],
      args: Tuple[Any, ...],
      kwargs: Dict[str, Any],
      *,
      exclude_actor: Optional[ActorHandle] = None,
  ) -> Optional[ActorHandle]:
    """Selects an available actor with capacity for the given task."""
    pool_actors = self._pool.actors
    if not pool_actors:
      raise RuntimeError(
          "RoutingActorPool contains no registered ActorHandles."
      )
    available = [a for a in pool_actors if self._worker_has_capacity(a)]
    if not available:
      return None
    if exclude_actor is not None and len(available) > 1:
      non_excluded = [a for a in available if a is not exclude_actor]
      if non_excluded:
        available = non_excluded

    if self._least_loaded and self._pool.router is None:
      return self._least_loaded_actor(
          kwargs.get("route_key"), available=available
      )
    preferred = self._pool.select_actor(method_name, args, dict(kwargs))
    if (
        self._max_in_flight_per_worker is None
        and not self._worker_max_in_flight
        and preferred in available
    ):
      return preferred

    if kwargs.get("route_key") is not None and preferred in available:
      return preferred

    min_load = min(self._worker_load(a) for a in available)
    if preferred in available and self._worker_load(preferred) == min_load:
      return preferred
    return min(available, key=self._worker_load)

  def _schedule_drain_pending(self) -> None:
    """Schedules an asynchronous drain of the pending task queue."""
    if not self._pending_queue or self._closed or self._draining_pending:
      return
    try:
      running_loop = asyncio.get_running_loop()
    except RuntimeError:
      running_loop = None

    if running_loop is not None and (
        self._loop is None
        or running_loop is self._loop
        or not self._loop.is_running()
    ):
      self._bind_current_loop()
      task = running_loop.create_task(self._drain_pending_queue())
      self._poll_tasks.add(task)
      task.add_done_callback(self._poll_tasks.discard)
    elif self._loop is not None and self._loop.is_running():

      def _spawn() -> None:
        if (
            not self._closed
            and self._pending_queue
            and not self._draining_pending
        ):
          t = self._loop.create_task(self._drain_pending_queue())  # type: ignore[union-attr]
          self._poll_tasks.add(t)
          t.add_done_callback(self._poll_tasks.discard)

      self._loop.call_soon_threadsafe(_spawn)

  def pop_failed_tasks(
      self,
  ) -> List[
      Tuple[
          str,
          Tuple[Optional[str], Tuple[Any, ...], Dict[str, Any]],
          Exception,
      ]
  ]:
    """Drains and returns tasks that failed terminally during the session."""
    failed = list(self._failed_tasks)
    self._failed_tasks.clear()
    return failed

  def add_actor(
      self, actor: ActorHandle, *, max_in_flight: Optional[int] = None
  ) -> None:
    """Dynamically adds a worker actor handle to the session and underlying pool (thread-safe)."""
    if max_in_flight is not None and max_in_flight <= 0:
      raise ValueError("max_in_flight must be positive")
    cap = int(max_in_flight) if max_in_flight is not None else None
    with self._membership_lock:
      was_in_pool = actor in self._pool.actors
      self._pool.add_actor(actor)
      self._known_actors.add(actor)
      active_count = len(self._pool.actors)
    if not was_in_pool:
      logging.info(
          "[rollout-ft] action=join worker=%s active_workers=%d",
          getattr(actor, "worker_id", None) or actor,
          active_count,
      )

    def _apply() -> None:
      if cap is not None:
        self._worker_max_in_flight[actor] = cap
      self._schedule_drain_pending()

    self._run_on_session_loop(_apply)

  def _cancel_worker_polling(self, actor: ActorHandle) -> None:
    """Cancels any active background poll task for `actor`."""
    self._active_workers.discard(actor)
    poll_task = self._worker_poll_tasks.pop(actor, None)
    if poll_task is not None and not poll_task.done():
      try:
        current = asyncio.current_task()
      except RuntimeError:
        current = None
      if poll_task is not current:
        poll_task.cancel()

  def _fail_task(
      self, req_id: str, exc: Exception, *, enqueue_response: bool = True
  ) -> None:
    """Marks `req_id` as terminally failed and cleans up its tracking state."""
    payload = self._task_payloads.pop(req_id, None)
    self._task_retries.pop(req_id, None)
    self._task_dispatch_times.pop(req_id, None)
    self._task_last_failed_actor.pop(req_id, None)
    if payload is not None:
      self._failed_tasks.append((req_id, payload, exc))
    if enqueue_response:
      self._response_queue.put_nowait((None, exc))
    self._in_flight = max(0, self._in_flight - 1)

  def _requeue_or_fail_worker_tasks(
      self,
      dispatched_set: set[str],
      exc: Exception,
      *,
      count_retry: bool = True,
  ) -> None:
    """Synchronously re-queues or fails tasks that were in-flight on a failed worker."""
    pending_req_ids = list(dispatched_set)
    dispatched_set.clear()
    can_retry = self._can_retry_or_hold()
    for req_id in pending_req_ids:
      self._task_dispatch_times.pop(req_id, None)
      retries = self._task_retries.get(req_id, 0)
      if (
          self._retry_on_worker_failure
          and (not count_retry or retries < self._max_task_retries)
          and req_id in self._task_payloads
          and can_retry
      ):
        if count_retry:
          self._task_retries[req_id] = retries + 1
        self._retries_total += 1
        logging.info(
            "[rollout-ft] action=requeue request_id=%s retry=%d/%d reason=%r",
            req_id,
            self._task_retries.get(req_id, 0),
            self._max_task_retries,
            exc,
        )
        self._pending_queue.appendleft(req_id)
      else:
        self._fail_task(req_id, exc)
    if (
        not self._pool.actors
        and self._pending_queue
        and self._in_flight == len(self._pending_queue)
    ):
      if not can_retry:
        self._fail_pending_queue(exc)
      else:
        self._response_queue.put_nowait(self._sentinel)
    self._notify_if_zero_flight()

  def _remove_actor(
      self,
      actor: ActorHandle,
      exc: Optional[BaseException] = None,
      *,
      count_retry: bool = False,
  ) -> bool:
    """Removes `actor` from the pool, cancels polling, and re-queues tasks (thread-safe)."""
    with self._membership_lock:
      removed_from_pool = self._pool.remove_actor(actor)
      was_known = actor in self._known_actors
      self._known_actors.discard(actor)
      active_count = len(self._pool.actors)
    was_tracked = removed_from_pool or was_known
    if not was_tracked:
      return False
    if removed_from_pool:
      logging.info(
          "[rollout-ft] action=leave worker=%s active_workers=%d reason=%r",
          getattr(actor, "worker_id", None) or actor,
          active_count,
          exc,
      )

    failure_exc = (
        exc
        if isinstance(exc, Exception)
        else RuntimeError(str(exc) if exc else "Worker evicted")
    )

    def _apply() -> None:
      self._worker_max_in_flight.pop(actor, None)
      self._evictions_total += 1
      logging.info(
          "[rollout-ft] action=evict worker=%s reason=%r",
          actor,
          exc,
      )
      self._cancel_worker_polling(actor)
      dispatched_set = self._dispatched_tasks.pop(actor, None)
      if dispatched_set or (
          not self._pool.actors
          and self._pending_queue
          and self._in_flight == len(self._pending_queue)
      ):
        self._requeue_or_fail_worker_tasks(
            dispatched_set if dispatched_set is not None else set(),
            failure_exc,
            count_retry=count_retry,
        )
        if self._pending_queue and self._pool.actors:
          self._schedule_drain_pending()
      if self._on_worker_evicted is not None:
        try:
          self._on_worker_evicted(actor, exc)
        except Exception:  # pylint: disable=broad-exception-caught
          logging.exception("Error in on_worker_evicted callback for %s", actor)

    self._run_on_session_loop(_apply)
    return True

  def remove_actor(
      self, actor: ActorHandle, exc: Optional[BaseException] = None
  ) -> bool:
    """Removes `actor`, cancels its poll loop, and re-queues in-flight tasks (thread-safe).

    Returns:
      True if `actor` was a pool member and has been removed, False otherwise.
    """
    return self._remove_actor(actor, exc=exc, count_retry=False)

  def _fail_pending_queue(self, exc: Exception) -> None:
    """Fails all queued tasks when no workers remain in the pool."""
    while self._pending_queue:
      req_id = self._pending_queue.popleft()
      self._fail_task(req_id, exc)
    self._notify_if_zero_flight()

  async def _drain_pending_queue(self) -> None:
    """Dispatches pending tasks to workers as capacity becomes available."""
    if self._draining_pending:
      return
    self._draining_pending = True
    try:
      while self._pending_queue and not self._closed:
        if not self._pool.actors:
          if self._can_retry_or_hold():
            return
          self._fail_pending_queue(
              RuntimeError(
                  "RoutingActorPool contains no registered ActorHandles."
              )
          )
          return
        req_id = self._pending_queue[0]
        payload = self._task_payloads.get(req_id)
        if payload is None:
          self._pending_queue.popleft()
          continue
        method_name, args, orig_kwargs = payload
        actor = self._select_actor_for_task(
            method_name,
            args,
            orig_kwargs,
            exclude_actor=self._task_last_failed_actor.get(req_id),
        )
        if actor is None:
          return
        self._pending_queue.popleft()
        await self._dispatch_to_actor(
            actor,
            req_id,
            method_name,
            args,
            orig_kwargs,
            from_pending=True,
        )
    finally:
      self._draining_pending = False

  async def _handle_worker_failure_tasks(
      self,
      dispatched_set: set[str],
      exc: Exception,
      *,
      count_retry: bool = True,
  ) -> None:
    """Re-queues or fails tasks that were in-flight on a failed worker."""
    self._requeue_or_fail_worker_tasks(
        dispatched_set, exc, count_retry=count_retry
    )
    if self._pending_queue and self._pool.actors:
      await self._drain_pending_queue()

  def _notify_if_zero_flight(self) -> None:
    if self._in_flight == 0:
      self._response_queue.put_nowait(self._sentinel)

  async def _dispatch_to_actor(
      self,
      actor: ActorHandle,
      request_id: str,
      method_name: Optional[str],
      args: Tuple[Any, ...],
      orig_kwargs: Dict[str, Any],
      *,
      from_pending: bool = False,
  ) -> None:
    """Dispatches a task to `actor`, handling eviction and retries on failure."""
    call_kwargs = dict(orig_kwargs)
    call_kwargs.pop("route_key", None)
    dispatched_set = self._dispatched_tasks.setdefault(actor, set())
    dispatched_set.add(request_id)
    self._task_dispatch_times[request_id] = time.monotonic()
    self._ensure_worker_polling(actor)

    try:
      dispatch_coro = actor.dispatch_task(
          request_id, method_name, *args, **call_kwargs
      )
      if self._task_timeout_s is not None:
        await asyncio.wait_for(dispatch_coro, timeout=self._task_timeout_s)
      else:
        await dispatch_coro
      if request_id in dispatched_set:
        self._task_dispatch_times.setdefault(request_id, time.monotonic())
      # Re-ensure worker polling is active in case the previous polling loop
      # exited or died while dispatch_task was awaiting.
      self._ensure_worker_polling(actor)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      # Only decrement _in_flight if _poll_worker_loop hasn't already failed
      # and cleared it.
      was_in_dispatched = request_id in dispatched_set
      if was_in_dispatched:
        dispatched_set.remove(request_id)
        self._task_dispatch_times.pop(request_id, None)
      if self._evict_on_failure:
        self._remove_actor(actor, exc=exc, count_retry=True)
      if (
          was_in_dispatched
          and self._retry_on_worker_failure
          and self._task_retries.get(request_id, 0) < self._max_task_retries
          and self._can_retry_or_hold()
      ):
        self._task_retries[request_id] = (
            self._task_retries.get(request_id, 0) + 1
        )
        self._retries_total += 1
        logging.info(
            "[rollout-ft] action=requeue request_id=%s retry=%d/%d reason=%r",
            request_id,
            self._task_retries[request_id],
            self._max_task_retries,
            exc,
        )
        self._pending_queue.appendleft(request_id)
        if not self._pool.actors:
          self._response_queue.put_nowait(self._sentinel)
          return
        await self._drain_pending_queue()
        return
      if not was_in_dispatched and request_id in self._task_payloads:
        if self._pending_queue and self._pool.actors:
          await self._drain_pending_queue()
        return
      fault_tolerance_enabled = (
          self._evict_on_failure or self._retry_on_worker_failure
      )
      if was_in_dispatched:
        self._fail_task(
            request_id,
            exc,
            enqueue_response=from_pending or fault_tolerance_enabled,
        )
        if not self._pool.actors and self._pending_queue:
          if not self._can_retry_or_hold():
            self._fail_pending_queue(exc)
          else:
            self._response_queue.put_nowait(self._sentinel)
        self._notify_if_zero_flight()
      if not from_pending and not fault_tolerance_enabled:
        raise

  async def submit(
      self,
      request_id: str,
      method_name: Optional[str] = None,
      *args,
      **kwargs,
  ) -> str:
    """Dispatches a task to a worker in the pool and tracks its completion."""
    if self._closed:
      raise RuntimeError("PoolExecutionSession is closed.")
    if not self._pool.actors and not self._can_retry_or_hold():
      raise RuntimeError(
          "RoutingActorPool contains no registered ActorHandles."
      )
    self._bind_current_loop()
    orig_kwargs = dict(kwargs)
    args_tuple = tuple(args)
    self._task_payloads[request_id] = (method_name, args_tuple, orig_kwargs)
    self._in_flight += 1

    if not self._pending_queue and self._pool.actors:
      try:
        actor = self._select_actor_for_task(
            method_name, args_tuple, orig_kwargs
        )
      except Exception:
        self._in_flight = max(0, self._in_flight - 1)
        self._task_payloads.pop(request_id, None)
        raise
      if actor is not None:
        await self._dispatch_to_actor(
            actor,
            request_id,
            method_name,
            args_tuple,
            orig_kwargs,
            from_pending=False,
        )
        return request_id

    self._pending_queue.append(request_id)
    if self._pool.actors:
      await self._drain_pending_queue()
    return request_id

  def _least_loaded_actor(
      self,
      route_key: Any = None,
      *,
      available: Optional[Sequence[ActorHandle]] = None,
  ) -> ActorHandle:
    """Returns the pool actor with the fewest of this session's tasks in flight."""
    actors = list(available) if available is not None else self._pool._actors
    if not actors:
      raise RuntimeError(
          "RoutingActorPool contains no registered ActorHandles."
      )
    if route_key is not None:
      placed = self._placements.get(route_key)
      if placed is not None and placed in actors:
        self._placements.move_to_end(route_key)
        return placed
    loads = [len(self._dispatched_tasks.get(a, ())) for a in actors]
    min_load = min(loads)
    candidates = [a for a, n in zip(actors, loads) if n == min_load]
    actor = candidates[self._pool._idx % len(candidates)]
    self._pool._idx += 1
    if route_key is not None:
      self._placements[route_key] = actor
      if len(self._placements) > _MAX_ROUTE_PLACEMENTS:
        self._placements.popitem(last=False)
    return actor

  def _ensure_worker_polling(self, actor: ActorHandle) -> None:
    """Starts a background polling loop for `actor` if not already running."""
    existing = self._worker_poll_tasks.get(actor)
    if existing is not None and not existing.done():
      self._active_workers.add(actor)
      return
    self._active_workers.add(actor)
    task = asyncio.create_task(self._poll_worker_loop(actor))
    self._poll_tasks.add(task)
    self._worker_poll_tasks[actor] = task

    def _on_done(t: asyncio.Task[Any]) -> None:
      self._poll_tasks.discard(t)
      if self._worker_poll_tasks.get(actor) is t:
        self._worker_poll_tasks.pop(actor, None)

    task.add_done_callback(_on_done)

  async def _poll_worker_loop(self, actor: ActorHandle) -> None:
    discarded = False
    try:
      while not self._closed:
        dispatched_set = self._dispatched_tasks.setdefault(actor, set())
        if not dispatched_set:
          if self._pending_queue:
            await self._drain_pending_queue()
          if not dispatched_set:
            break
        try:
          timeout_s = self._task_timeout_s
          if timeout_s is not None:
            now = time.monotonic()
            expired_rids = {
                rid
                for rid in list(dispatched_set)
                if now - self._task_dispatch_times.get(rid, now) >= timeout_s
            }
            if expired_rids:
              timeout_exc = TimeoutError(
                  f"Task(s) {sorted(expired_rids)} on worker {actor}"
                  f" exceeded task_timeout_s={timeout_s}s."
              )
              if len(expired_rids) < len(dispatched_set):
                # Only a subset of concurrent episodes on this worker timed
                # out while younger tasks remain in flight: re-queue or fail
                # just the expired request_ids without evicting the worker and
                # discarding its healthy in-flight tasks.
                logging.warning(
                    "[rollout-ft] action=partial_timeout worker=%s"
                    " expired_requests=%s remaining_in_flight=%d"
                    " task_timeout_s=%s",
                    getattr(actor, "worker_id", None) or actor,
                    sorted(expired_rids),
                    len(dispatched_set) - len(expired_rids),
                    timeout_s,
                )
                dispatched_set.difference_update(expired_rids)
                for rid in expired_rids:
                  self._task_last_failed_actor[rid] = actor
                await self._handle_worker_failure_tasks(
                    expired_rids, timeout_exc, count_retry=True
                )
                self._notify_if_zero_flight()
                continue
              raise timeout_exc
            oldest_dispatch = min(
                (
                    self._task_dispatch_times.get(rid, now)
                    for rid in list(dispatched_set)
                ),
                default=now,
            )
            remaining_s = timeout_s - (now - oldest_dispatch)
            poll_wait_s = min(LONG_POLL_TIMEOUT_S, remaining_s)
            response = await asyncio.wait_for(
                actor.poll_responses(timeout_s=poll_wait_s),
                timeout=remaining_s,
            )
          else:
            response = await actor.poll_responses(timeout_s=LONG_POLL_TIMEOUT_S)
          if isinstance(response, ExecutionResponse):
            req_id = response.request_id
            if req_id not in dispatched_set:
              # Workers tag every completion with its request_id; an untagged
              # response is a heartbeat/empty poll, anything else is late or
              # unknown. Never attribute it to an arbitrary in-flight task.
              if req_id:
                logging.warning(
                    "Ignoring late or unknown response for request_id=%r from"
                    " worker %s (not in dispatched_set).",
                    req_id,
                    actor,
                )
              continue
            dispatched_set.remove(req_id)
            self._task_dispatch_times.pop(req_id, None)
            self._task_last_failed_actor.pop(req_id, None)
            self._in_flight = max(0, self._in_flight - 1)
            payload = self._task_payloads.pop(req_id, None)
            self._task_retries.pop(req_id, None)
            try:
              res = response.unwrap()
              self._response_queue.put_nowait((res, None))
            except Exception as exc:  # pylint: disable=broad-exception-caught
              if payload is not None:
                self._failed_tasks.append((req_id, payload, exc))
              self._response_queue.put_nowait((None, exc))
            if self._pending_queue:
              await self._drain_pending_queue()
            self._notify_if_zero_flight()
        except asyncio.CancelledError:
          break
        except Exception as exc:  # pylint: disable=broad-exception-caught
          # If poll_responses timed out while younger tasks are still within
          # their individual task_timeout_s budget, only fail/re-queue the
          # expired task(s) unless the poll RPC itself hung past poll_wait_s.
          if (
              isinstance(exc, (TimeoutError, asyncio.TimeoutError))
              and self._task_timeout_s is not None
              and len(dispatched_set) > 1
          ):
            now = time.monotonic()
            expired_rids = {
                rid
                for rid in list(dispatched_set)
                if now - self._task_dispatch_times.get(rid, now)
                >= self._task_timeout_s - 0.05
            }
            if expired_rids and len(expired_rids) < len(dispatched_set):
              timeout_exc = TimeoutError(
                  f"Task(s) {sorted(expired_rids)} on worker {actor}"
                  f" exceeded task_timeout_s={self._task_timeout_s}s."
              )
              logging.warning(
                  "[rollout-ft] action=partial_timeout worker=%s"
                  " expired_requests=%s remaining_in_flight=%d"
                  " task_timeout_s=%s",
                  getattr(actor, "worker_id", None) or actor,
                  sorted(expired_rids),
                  len(dispatched_set) - len(expired_rids),
                  self._task_timeout_s,
              )
              dispatched_set.difference_update(expired_rids)
              for rid in expired_rids:
                self._task_last_failed_actor[rid] = actor
              await self._handle_worker_failure_tasks(
                  expired_rids, timeout_exc, count_retry=True
              )
              self._notify_if_zero_flight()
              continue
          # Transport, timeout, or polling failure on this worker; evict and/or
          # retry in-flight tasks.
          self._active_workers.discard(actor)
          if self._worker_poll_tasks.get(actor) is asyncio.current_task():
            self._worker_poll_tasks.pop(actor, None)
          discarded = True
          if self._evict_on_failure:
            self._remove_actor(actor, exc=exc, count_retry=True)
          else:
            for rid in dispatched_set:
              self._task_last_failed_actor[rid] = actor
            await self._handle_worker_failure_tasks(dispatched_set, exc)
          break
    finally:
      if not discarded:
        self._active_workers.discard(actor)

  async def poll_completed(
      self, timeout_s: float = LONG_POLL_TIMEOUT_S
  ) -> List[Tuple[Any, Optional[Exception]]]:
    """Waits up to `timeout_s` for the first completion, then drains ready items."""
    if self._closed:
      return []
    self._bind_current_loop()
    if self._pending_queue:
      await self._drain_pending_queue()
    if not self._pool.actors and self._response_queue.empty():
      return []

    batch: List[Tuple[Any, Optional[Exception]]] = []
    # When idle (_in_flight == 0), yield briefly (<= 50ms) so concurrent dispatch
    # coroutines can run without stalling for the full 50s long-poll timeout.
    wait_s = (
        min(timeout_s, 0.05)
        if (self._in_flight == 0 and self._response_queue.empty())
        else timeout_s
    )
    deadline = time.monotonic() + wait_s
    try:
      # Block until the first real (result, exc) completion arrives.
      while not batch:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
          return []
        step_wait = (
            min(remaining, 0.2)
            if self._has_pending_workers_fn is not None
            else remaining
        )
        try:
          item = await asyncio.wait_for(
              self._response_queue.get(), timeout=step_wait
          )
        except asyncio.TimeoutError:
          if (
              self._has_pending_workers_fn is not None
              and self._has_pending_workers_fn()
          ):
            return []
          continue
        if item is self._sentinel:
          # Sentinel marks _in_flight reaching 0, active pool emptying, or
          # session close; return early if drained, otherwise skip stale
          # sentinels when new tasks are in flight.
          if (
              self._in_flight == 0 or not self._pool.actors
          ) and self._response_queue.empty():
            return []
          continue
        batch.append(item)
    except asyncio.TimeoutError:
      return []

    # Greedily drain any additional completions already queued by background worker loops.
    while not self._response_queue.empty():
      item = self._response_queue.get_nowait()
      if item is not self._sentinel:
        batch.append(item)
    return batch

  async def as_completed(
      self,
  ) -> AsyncIterator[Tuple[Any, Optional[Exception]]]:
    """Yields (result, exception) tuples as tasks complete out-of-order."""
    self._bind_current_loop()
    if self._pending_queue:
      await self._drain_pending_queue()
    while not self._closed:
      item = await self._response_queue.get()
      if item is self._sentinel:
        if self._in_flight == 0 and self._response_queue.empty():
          break
        continue
      result, exc = item
      yield result, exc

  async def close(self) -> None:
    """Closes the session and cancels all background polling tasks."""
    self._closed = True
    self._response_queue.put_nowait(self._sentinel)
    for t in list(self._poll_tasks):
      if not t.done():
        t.cancel()
    if self._poll_tasks:
      await asyncio.gather(*self._poll_tasks, return_exceptions=True)


def remote(
    cls_or_func: Optional[Any] = None,
    *,
    transport: str = "inprocess",
    address: Optional[str] = None,
) -> Any:
  """Decorator turning classes/functions into Actor factories (like @ray.remote).

  Args:
    cls_or_func: Positional target class or function when decorated without
      parentheses (e.g. `@remote class Foo:`). When called as
      `@remote("grpc://...")`, this receives the target URI string directly.
      When keyword arguments are used, this defaults to `None`.
    transport: Execution engine transport (`"inprocess"`, `"grpc"`, or
      `"stubby"`). Automatically inferred when `address` contains `"://"`.
    address: Optional explicit target URI string (e.g.
      `"grpc://localhost:50051"`).

  Returns:
    An `ActorFactory` class proxy (for decorated classes) or task wrapper
    (for decorated functions) exposing a `.remote(*args, **kwargs)` handle.

  Usage:
    # 1. Bare decorator (without parentheses): `cls_or_func` receives the class
    # or function directly (not a string and not None).
    @remote
    class BareWorker: ...
    handle = BareWorker.remote()

    @remote
    def standalone_task(x: int) -> int: ...

    # 2. Positional string argument: `cls_or_func` receives the URI directly.
    # Dispatches to address="grpc://worker-pod:50051" and transport="grpc".
    @remote("grpc://worker-pod:50051")
    class RemoteWorker: ...
    handle = RemoteWorker.remote()

    # 3. Keyword arguments: `cls_or_func` is None. Returns `decorator` wrapper.
    @remote(transport="inprocess")
    class MyWorker: ...

    @remote(address="grpc://worker-pod:50051")
    class ExplicitWorker: ...

    # 4. Late address binding (dynamic pod discovery at runtime):
    @remote(transport="grpc")
    class DynamicWorker: ...
    handle = DynamicWorker.remote(address="grpc://allocated-pod-42:50051")
  """

  if isinstance(cls_or_func, str):
    address = cls_or_func
    cls_or_func = None

  if address and "://" in address and transport == "inprocess":
    transport = address.split("://")[0]

  def decorator(target: Any) -> Any:
    if inspect.isclass(target):

      class ActorFactory:

        @classmethod
        def remote(cls, *args, **kwargs) -> ActorHandle:
          if transport == "inprocess":
            instance = target(*args, **kwargs)
            server = InProcessRemoteExecutionServer(instance)
            return InProcessActorHandle(server)
          elif transport in ("grpc", "stubby"):
            target_addr = address or kwargs.pop(
                "address", f"{transport}://localhost:50051"
            )
            return ActorHandle.from_address(target_addr)
          else:
            raise ValueError(f"Unsupported transport: {transport}")

      ActorFactory.__name__ = target.__name__
      ActorFactory.__doc__ = target.__doc__
      return ActorFactory
    elif inspect.isfunction(target) or inspect.ismethod(target):
      if transport != "inprocess":
        raise NotImplementedError(
            f"Remote execution over transport='{transport}' is not yet "
            "supported for standalone functions. @remote over gRPC/stubby "
            "currently requires decorating a class (e.g. @remote class "
            "MyWorker: ...)."
        )

      def remote_func(*args, **kwargs) -> Any:
        if inspect.iscoroutinefunction(target):

          class _FunctionContainer:

            async def execute(self, *f_args, **f_kwargs):
              return await target(*f_args, **f_kwargs)

        else:

          class _FunctionContainer:

            def execute(self, *f_args, **f_kwargs):
              return target(*f_args, **f_kwargs)

        server = InProcessRemoteExecutionServer(_FunctionContainer())
        handle = InProcessActorHandle(server)
        if inspect.iscoroutinefunction(target):
          try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
              return handle.asubmit("execute", *args, **kwargs)
          except RuntimeError:
            pass
        return handle.submit("execute", *args, **kwargs)

      remote_func.remote = remote_func  # pyrefly: ignore[missing-attribute]
      return remote_func
    else:
      raise TypeError(
          f"@remote expects a class or function, got {type(target)}"
      )

  if cls_or_func is None:
    return decorator
  return decorator(cls_or_func)
