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
import contextlib
import hashlib
import inspect
import math
import os
import struct
import threading
import time
import traceback as traceback_lib
import uuid
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from absl import logging
import cloudpickle

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

# Cap for a single gRPC message. Set to -1 (unlimited) so large training-batch
# payloads (which can exceed 1 GB for long sequence lengths) are not truncated.
_MAX_MESSAGE_BYTES = -1

# -1 does not make messages unbounded: the gRPC wire format frames each message
# with a 4-byte length, so anything >= 4 GiB is silently truncated (and >2 GiB
# is unreliable). A packed 397B train_step with router replay carries ~4.8 GB
# of routed experts (4M tokens x 60 layers x top-10 x int16), so requests larger
# than this are uploaded in chunks via PutChunk and reassembled server-side.
_CHUNK_BYTES = int(os.environ.get("TUNIX_GRPC_CHUNK_BYTES", 512 * 1024 * 1024))

# Pickle payloads start with the PROTO opcode (0x80), so this never collides
# with a real serialized request.
_CHUNKED_MAGIC = b"TUNIX_CHUNKED\x00"
# PutChunk header: 16-byte upload id, chunk index, total chunk count.
_CHUNK_HEADER = struct.Struct("!16sII")
_UPLOAD_ID_BYTES = 16
# Bounds the list PutChunk allocates from a header's total; 64 TiB at the
# default chunk size.
_MAX_CHUNKS = 1 << 17
# Uploads that receive no chunk and are not claimed by an Execute/DispatchTask
# for this long (client died mid-upload) are dropped so they don't pin
# gigabytes of host memory. Measured from the last chunk, so a slow upload that
# is still making progress never expires.
_CHUNK_UPLOAD_TTL_S = 300.0


def _grpc_options() -> List[Tuple[str, int]]:
  """Channel/server options lifting the message-size cap and enabling keepalive."""
  return [
      ("grpc.max_send_message_length", _MAX_MESSAGE_BYTES),
      ("grpc.max_receive_message_length", _MAX_MESSAGE_BYTES),
      ("grpc.keepalive_time_ms", 20000),
      ("grpc.keepalive_timeout_ms", 10000),
      ("grpc.keepalive_permit_without_calls", 1),
      ("grpc.http2.max_pings_without_data", 0),
      ("grpc.http2.max_ping_strikes", 0),
      ("grpc.http2.min_ping_interval_without_data_ms", 5000),
      ("grpc.http2.min_recv_ping_interval_without_data_ms", 5000),
  ]


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

  def serialize(self) -> bytes:
    """Serializes request to bytes using cloudpickle."""
    return cloudpickle.dumps(
        (self.request_id, self.method_name, self.args, self.kwargs)
    )

  @classmethod
  def deserialize(cls, payload: bytes) -> "ExecutionRequest":
    """Deserializes bytes into an ExecutionRequest."""
    # SECURITY WARNING: cloudpickle.loads executes arbitrary code via __reduce__ during
    # deserialization. In production across untrusted boundaries, verify ALTS/mTLS transport
    # identity or cryptographic HMAC signatures before calling cloudpickle.loads(). Where
    # dynamic function shipping is not needed, use `pickle.Unpickler` (`find_class`) to
    # whitelist only trusted domain data types (`int`, `str`, `dict`, `list`, `data_types.*`).
    request_id, method_name, args, kwargs = cloudpickle.loads(payload)
    return cls(
        request_id=request_id,
        method_name=method_name,
        args=args,
        kwargs=kwargs,
    )


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

  def serialize(self) -> bytes:
    try:
      return cloudpickle.dumps((
          self.result,
          self.error_message,
          self.error_type,
          self.traceback,
          self.retryable,
          self.request_id,
      ))
    except Exception as e:  # pylint: disable=broad-exception-caught
      err_msg = (
          "failed to serialize result of type"
          f" {type(self.result).__name__}: {e}"
      )
      self.result = None
      self.error_message = err_msg
      self.error_type = "ExecutionResponseSerializationError"
      self.traceback = traceback_lib.format_exc()
      self.retryable = False
      return cloudpickle.dumps((
          self.result,
          self.error_message,
          self.error_type,
          self.traceback,
          self.retryable,
          self.request_id,
      ))

  @classmethod
  def deserialize(cls, payload: bytes) -> "ExecutionResponse":
    # SECURITY WARNING: cloudpickle.loads executes arbitrary code during unpickling. Ensure
    # payload authenticity over trusted channels before deserialization, or use custom
    # `pickle.Unpickler` (`find_class`) to whitelist only trusted domain data types.
    result, err_msg, err_type, tb, retryable, request_id = cloudpickle.loads(
        payload
    )
    return cls(
        result=result,
        error_message=err_msg,
        error_type=err_type,
        traceback=tb,
        retryable=retryable,
        request_id=request_id,
    )

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
    self._request_counter: int = 0
    self._background_tasks: set[asyncio.Task[Any]] = set()

  def _get_response_queue(self) -> asyncio.Queue[ExecutionResponse]:
    if self._response_queue is None:
      self._response_queue = asyncio.Queue()
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

  def __init__(self, instance: Optional[Any] = None):
    super().__init__(instance)
    self._server: Optional[Any] = None
    self._serve_loop: Optional[Any] = None
    # upload id -> [last-chunk time, chunks]; filled by PutChunk.
    self._uploads: Dict[bytes, List[Any]] = {}

  async def _handle_put_chunk(self, chunk_bytes: bytes, context: Any) -> bytes:
    del context
    now = time.monotonic()
    for stale in [
        k for k, (t, _) in self._uploads.items()
        if now - t > _CHUNK_UPLOAD_TTL_S
    ]:
      logging.warning("Dropping abandoned chunked upload %s", stale.hex())
      del self._uploads[stale]
    if len(chunk_bytes) < _CHUNK_HEADER.size:
      raise ValueError(
          f"chunk of {len(chunk_bytes)} bytes is shorter than its header"
      )
    upload_id, index, total = _CHUNK_HEADER.unpack_from(chunk_bytes)
    if not 0 < total <= _MAX_CHUNKS:
      raise ValueError(f"invalid chunk count {total} (max {_MAX_CHUNKS})")
    if index >= total:
      raise ValueError(f"chunk index {index} out of range for {total} chunks")
    entry = self._uploads.get(upload_id)
    if entry is None:
      entry = self._uploads[upload_id] = [now, [None] * total]
    elif len(entry[1]) != total:
      raise ValueError(
          f"chunked upload {upload_id.hex()} declared {len(entry[1])} chunks,"
          f" this chunk declares {total}"
      )
    entry[0] = now
    entry[1][index] = chunk_bytes[_CHUNK_HEADER.size :]
    return b""

  def _reassemble(self, request_bytes: bytes) -> bytes:
    """Returns the full payload, joining PutChunk uploads if this is a reference."""
    if not request_bytes.startswith(_CHUNKED_MAGIC):
      return request_bytes
    upload_id = request_bytes[len(_CHUNKED_MAGIC) :]
    if len(upload_id) != _UPLOAD_ID_BYTES:
      raise RuntimeError(f"invalid chunked upload id of {len(upload_id)} bytes")
    entry = self._uploads.pop(upload_id, None)
    if entry is None:
      raise RuntimeError(f"unknown chunked upload {upload_id.hex()}")
    chunks = entry[1]
    missing = [i for i, c in enumerate(chunks) if c is None]
    if missing:
      raise RuntimeError(
          f"chunked upload {upload_id.hex()} is missing chunks {missing}"
      )
    return b"".join(chunks)

  async def _handle_execute(self, request_bytes: bytes, context: Any) -> bytes:
    del context
    try:
      request = ExecutionRequest.deserialize(self._reassemble(request_bytes))
      response = await self.execute_request(request)
    except Exception as e:  # pylint: disable=broad-exception-caught
      response = ExecutionResponse(
          error_message=str(e),
          error_type=type(e).__name__,
          traceback=traceback_lib.format_exc(),
      )
    return response.serialize()

  async def _handle_dispatch_task(
      self, request_bytes: bytes, context: Any
  ) -> bytes:
    del context
    request = ExecutionRequest.deserialize(self._reassemble(request_bytes))
    request_id = await self.dispatch_task(request)
    return cloudpickle.dumps(request_id)

  async def _handle_poll_responses(
      self, request_bytes: bytes, context: Any
  ) -> bytes:
    del context
    timeout_s = (
        cloudpickle.loads(request_bytes)
        if request_bytes
        else LONG_POLL_TIMEOUT_S
    )
    response = await self.poll_response(timeout_s=timeout_s)
    if response is None:
      return b""
    return response.serialize()

  async def start_serving_async(self, port: int = 50051) -> Any:
    """Starts an asynchronous gRPC server listening on [::]:port."""
    if not _GRPC_AVAILABLE or _grpc_lib is None or _grpc_aio_lib is None:
      raise RuntimeError("grpc is not installed or available.")

    self._server = _grpc_aio_lib.server(options=_grpc_options())
    handler = _grpc_lib.method_handlers_generic_handler(
        "tunix.ExecutionService",
        {
            "Execute": _grpc_lib.unary_unary_rpc_method_handler(
                self._handle_execute,
                request_deserializer=lambda b: b,
                response_serializer=lambda b: b,
            ),
            "DispatchTask": _grpc_lib.unary_unary_rpc_method_handler(
                self._handle_dispatch_task,
                request_deserializer=lambda b: b,
                response_serializer=lambda b: b,
            ),
            "PollResponses": _grpc_lib.unary_unary_rpc_method_handler(
                self._handle_poll_responses,
                request_deserializer=lambda b: b,
                response_serializer=lambda b: b,
            ),
            "PutChunk": _grpc_lib.unary_unary_rpc_method_handler(
                self._handle_put_chunk,
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

  @classmethod
  def from_address(
      cls,
      target_address: str,
      *,
      rpc_timeout_s: Optional[float] = RPC_TIMEOUT_S,
  ) -> "ActorHandle":
    """Instantiates a remote actor handle targeting the specified string URI."""
    if target_address.startswith("grpc://") and _GRPC_AVAILABLE:
      return GrpcRemoteActorHandle(
          target_address=target_address, rpc_timeout_s=rpc_timeout_s
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


_WEIGHT_SYNC_RPC_METHODS = frozenset({
    "prepare_weight_sync",
    "release_weight_sync",
    "bind_weight_sync",
    "get_weight_sync_metadata",
    "pre_weight_sync",
    "weight_sync",
    "post_weight_sync",
    "abort_weight_sync",
    "get_weight_sync_status",
})
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "y", "t", "on"})


def _normalize_rpc_timeout(timeout_s: Optional[float]) -> Optional[float]:
  if timeout_s is None or math.isinf(timeout_s) or timeout_s <= 0:
    return None
  return float(timeout_s)


def _weight_sync_timeouts_disabled() -> bool:
  for name in ("WEIGHT_SYNC_DISABLE_TIMEOUTS", "DISABLE_WEIGHT_SYNC_TIMEOUTS"):
    val = os.getenv(name)
    if val is not None and val.strip().lower() in _TRUTHY_ENV_VALUES:
      return True
  return False


class GrpcRemoteActorHandle(RemoteActorHandle):
  """ActorHandle connecting to GrpcRemoteExecutionServer over TCP sockets via gRPC."""

  def __init__(
      self,
      target_address: str,
      *,
      rpc_timeout_s: Optional[float] = RPC_TIMEOUT_S,
  ):
    if not _GRPC_AVAILABLE or _grpc_aio_lib is None:
      raise RuntimeError("grpc is not installed or available.")
    self.target_address = target_address
    self._host_port = target_address.replace("grpc://", "")
    self._channel: Optional[Any] = None
    self._rpc: Optional[Any] = None
    self._rpc_timeout_s = _normalize_rpc_timeout(rpc_timeout_s)
    # Blocking submit() runs on a persistent background event loop so repeated
    # calls reuse one channel. gRPC aio channels are bound to the loop that
    # created them, so they cannot be shared with the caller's async loop nor
    # survive a per-call asyncio.run() loop.
    self._sync_loop: Optional[Any] = None
    self._sync_thread: Optional[threading.Thread] = None
    self._sync_channel: Optional[Any] = None
    self._sync_rpc: Optional[Any] = None
    self._sync_lock = threading.Lock()

  def _effective_rpc_timeout(
      self, method_name: Optional[str] = None
  ) -> Optional[float]:
    if (
        method_name in _WEIGHT_SYNC_RPC_METHODS
        and _weight_sync_timeouts_disabled()
    ):
      return None
    return self._rpc_timeout_s

  def _make_rpc(self, channel: Any) -> Any:
    return channel.unary_unary(
        "/tunix.ExecutionService/Execute",
        request_serializer=lambda b: b,
        response_deserializer=lambda b: ExecutionResponse.deserialize(b),
    )

  async def _encode_request(
      self, channel: Any, request: ExecutionRequest
  ) -> bytes:
    """Serializes `request`, uploading it via PutChunk if it is too big for one message."""
    payload = request.serialize()
    if len(payload) <= _CHUNK_BYTES:
      return payload
    put_chunk = channel.unary_unary(
        "/tunix.ExecutionService/PutChunk",
        request_serializer=lambda b: b,
        response_deserializer=lambda b: b,
    )
    upload_id = uuid.uuid4().bytes
    view = memoryview(payload)
    total = -(-len(payload) // _CHUNK_BYTES)
    if total > _MAX_CHUNKS:
      raise ValueError(
          f"request {request.method_name} needs {total} chunks of"
          f" {_CHUNK_BYTES} bytes, over the {_MAX_CHUNKS} the server accepts;"
          " raise TUNIX_GRPC_CHUNK_BYTES"
      )
    logging.info(
        "Request %s is %.2f GiB; uploading in %d chunks",
        request.method_name,
        len(payload) / 2**30,
        total,
    )
    chunk_timeout = self._effective_rpc_timeout(request.method_name)
    for index in range(total):
      chunk = view[index * _CHUNK_BYTES : (index + 1) * _CHUNK_BYTES]
      await put_chunk(
          _CHUNK_HEADER.pack(upload_id, index, total) + chunk,
          timeout=chunk_timeout,
      )
    return _CHUNKED_MAGIC + upload_id

  def _get_rpc(self) -> Any:
    if self._rpc is None:
      assert _grpc_aio_lib is not None
      self._channel = _grpc_aio_lib.insecure_channel(
          self._host_port, options=_grpc_options()
      )
      self._rpc = self._make_rpc(self._channel)
    return self._rpc

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
          self._host_port, options=_grpc_options()
      )
      self._sync_rpc = self._make_rpc(self._sync_channel)
    request = ExecutionRequest(
        method_name=method_name, args=args, kwargs=kwargs
    )
    response: ExecutionResponse = await self._sync_rpc(
        await self._encode_request(self._sync_channel, request),
        timeout=self._effective_rpc_timeout(method_name),
    )
    return response.unwrap()

  async def asubmit(
      self, method_name: Optional[str] = None, *args, **kwargs
  ) -> Any:
    """Asynchronously invokes remote method over gRPC."""
    rpc = self._get_rpc()
    request = ExecutionRequest(
        method_name=method_name, args=args, kwargs=kwargs
    )
    response: ExecutionResponse = await rpc(
        await self._encode_request(self._channel, request),
        timeout=self._effective_rpc_timeout(method_name),
    )
    return response.unwrap()

  def _make_dispatch_task_rpc(self, channel: Any) -> Any:
    return channel.unary_unary(
        "/tunix.ExecutionService/DispatchTask",
        request_serializer=lambda b: b,
        response_deserializer=lambda b: cloudpickle.loads(b),
    )

  def _make_poll_responses_rpc(self, channel: Any) -> Any:
    return channel.unary_unary(
        "/tunix.ExecutionService/PollResponses",
        request_serializer=lambda b: cloudpickle.dumps(b),
        response_deserializer=lambda b: b,
    )

  async def dispatch_task(
      self,
      request_id: Optional[str] = None,
      method_name: Optional[str] = None,
      *args,
      **kwargs,
  ) -> str:
    """Asynchronously dispatches task request on remote server, returning task ACK ID."""
    if self._channel is None:
      self._get_rpc()
    assert self._channel is not None
    rpc = self._make_dispatch_task_rpc(self._channel)
    request = ExecutionRequest(
        request_id=request_id, method_name=method_name, args=args, kwargs=kwargs
    )
    return await rpc(
        await self._encode_request(self._channel, request),
        timeout=self._effective_rpc_timeout(method_name),
    )

  async def poll_responses(
      self, timeout_s: float = LONG_POLL_TIMEOUT_S
  ) -> Optional[ExecutionResponse]:
    """Long-polls remote server response queue for completed task results."""
    if self._channel is None:
      self._get_rpc()
    assert self._channel is not None
    rpc = self._make_poll_responses_rpc(self._channel)
    resp_bytes = await rpc(timeout_s, timeout=self._rpc_timeout_s)
    if not resp_bytes:
      return None
    return ExecutionResponse.deserialize(resp_bytes)

  async def close(self) -> None:
    if self._channel is not None:
      await self._channel.close()
      self._channel = None
      self._rpc = None
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
      tasks: Sequence[
          Tuple[str, str, Sequence[Any], Dict[str, Any]]
      ],
  ) -> AsyncIterator[Any]:
    """Dispatches a batch of tasks across pool and yields results strictly out-of-order.

    Args:
      tasks: Sequence of task specifications formatted as 4-tuples:
        `(request_id, method_name, args, kwargs)`, where:
          - request_id: Unique request identifier string.
          - method_name: Target remote method name to execute on the worker instance.
          - args: Positional arguments sequence passed to the remote method.
          - kwargs: Keyword arguments dictionary passed to the remote method.
    """
    raise NotImplementedError


def stable_route_hash(route_key: Any) -> int:
  """Maps a sticky routing key to a process-stable non-negative integer.

  Args:
    route_key: The sticky routing key. Non-negative integers are returned
      as-is; any other value is hashed via its `str()` representation.

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

  def replace_actor(
      self, old_actor: ActorHandle, new_actor: ActorHandle
  ) -> None:
    if old_actor in self._actors:
      if new_actor in self._actors:
        if old_actor != new_actor:
          self._actors.remove(old_actor)
      else:
        idx = self._actors.index(old_actor)
        self._actors[idx] = new_actor
    elif new_actor not in self._actors:
      self._actors.append(new_actor)

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
      tasks: Sequence[
          Tuple[str, str, Sequence[Any], Dict[str, Any]]
      ],
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
        `(request_id, method_name, args, kwargs)`, where:
          - request_id: Unique request identifier string.
          - method_name: Target remote method name to execute on the worker instance.
          - args: Positional arguments sequence passed to the remote method.
          - kwargs: Keyword arguments dictionary passed to the remote method.
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
      *,
      evict_on_failure: bool = False,
      retry_on_worker_failure: bool = False,
      max_task_retries: int = 3,
      on_worker_evicted: Optional[
          Callable[[ActorHandle, Optional[BaseException]], None]
      ] = None,
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
      initial_tasks: Optional sequence of initial task specifications to dispatch
        upon entering the session, formatted as 4-tuples `(request_id, method_name, args, kwargs)`, where:
          - request_id: Unique request identifier string.
          - method_name: Target remote method name to execute on the worker instance.
          - args: Positional arguments sequence passed to the remote method.
          - kwargs: Keyword arguments dictionary passed to the remote method.
      evict_on_failure: Whether to automatically remove a failing worker from the pool.
      retry_on_worker_failure: Whether to re-dispatch tasks that were on a failed worker.
      max_task_retries: Maximum number of retries per request_id on worker failure.
      on_worker_evicted: Optional callback `(actor, exc)` invoked when a worker is evicted.
    """
    session = PoolExecutionSession(
        self,
        evict_on_failure=evict_on_failure,
        retry_on_worker_failure=retry_on_worker_failure,
        max_task_retries=max_task_retries,
        on_worker_evicted=on_worker_evicted,
    )
    try:
      if initial_tasks:
        for request_id, method_name, args, kwargs in initial_tasks:
          await session.submit(request_id, method_name, *args, **kwargs)
      yield session
    finally:
      await session.close()


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
      evict_on_failure: bool = False,
      retry_on_worker_failure: bool = False,
      max_task_retries: int = 3,
      on_worker_evicted: Optional[
          Callable[[ActorHandle, Optional[BaseException]], None]
      ] = None,
  ):
    self._pool = pool
    self._evict_on_failure = evict_on_failure
    self._retry_on_worker_failure = retry_on_worker_failure
    self._max_task_retries = max_task_retries
    self._on_worker_evicted = on_worker_evicted
    self._response_queue: asyncio.Queue[Any] = asyncio.Queue()
    self._active_workers: set[ActorHandle] = set()
    self._evicted_actors: set[ActorHandle] = set()
    self._dispatched_tasks: Dict[ActorHandle, set[str]] = {}
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
    self._in_flight = 0
    self._closed = False
    self._sentinel = object()

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

  def add_actor(self, actor: ActorHandle) -> None:
    """Dynamically adds a worker actor handle to the underlying pool."""
    self._evicted_actors.discard(actor)
    self._pool.add_actor(actor)

  def _evict_actor_from_pool(
      self, actor: ActorHandle, exc: Optional[BaseException] = None
  ) -> None:
    self._pool.remove_actor(actor)
    if actor not in self._evicted_actors:
      self._evicted_actors.add(actor)
      if self._on_worker_evicted is not None:
        try:
          self._on_worker_evicted(actor, exc)
        except Exception:  # pylint: disable=broad-exception-caught
          logging.exception("Error in on_worker_evicted callback for %s", actor)

  async def remove_actor(
      self, actor: ActorHandle, exc: Optional[Exception] = None
  ) -> None:
    """Evicts `actor` from the pool and retries or fails its in-flight tasks."""
    self._evict_actor_from_pool(actor, exc)
    dispatched_set = self._dispatched_tasks.get(actor)
    if dispatched_set:
      await self._handle_worker_failure_tasks(
          dispatched_set, exc or RuntimeError("Worker evicted")
      )

  async def _handle_worker_failure_tasks(
      self, dispatched_set: set[str], exc: Exception
  ) -> None:
    pending_req_ids = list(dispatched_set)
    dispatched_set.clear()
    if not pending_req_ids:
      return
    for req_id in pending_req_ids:
      if (
          self._retry_on_worker_failure
          and self._task_retries.get(req_id, 0) < self._max_task_retries
          and req_id in self._task_payloads
          and len(self._pool._actors) > 0
      ):
        method_name, args, orig_kwargs = self._task_payloads[req_id]
        self._task_retries[req_id] = self._task_retries.get(req_id, 0) + 1
        self._in_flight = max(0, self._in_flight - 1)
        try:
          await self.submit(req_id, method_name, *args, **dict(orig_kwargs))
          continue
        except Exception as sub_exc:  # pylint: disable=broad-exception-caught
          payload = self._task_payloads.pop(req_id, None)
          self._task_retries.pop(req_id, None)
          if payload is not None:
            self._failed_tasks.append((req_id, payload, sub_exc))
          self._response_queue.put_nowait((None, sub_exc))
      else:
        payload = self._task_payloads.pop(req_id, None)
        self._task_retries.pop(req_id, None)
        if payload is not None:
          self._failed_tasks.append((req_id, payload, exc))
        self._response_queue.put_nowait((None, exc))
        self._in_flight = max(0, self._in_flight - 1)
    self._notify_if_zero_flight()

  def _notify_if_zero_flight(self) -> None:
    if self._in_flight == 0:
      self._response_queue.put_nowait(self._sentinel)

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
    orig_kwargs = dict(kwargs)
    actor = self._pool._get_next_actor(method_name, args, kwargs)
    kwargs.pop("route_key", None)  # remove route_key from worker method args
    self._task_payloads[request_id] = (method_name, tuple(args), orig_kwargs)

    # Increment in_flight and register request_id BEFORE awaiting dispatch_task
    # so a fast worker completing before dispatch_task's coroutine resumes
    # always finds request_id in _dispatched_tasks[actor].
    self._in_flight += 1
    dispatched_set = self._dispatched_tasks.setdefault(actor, set())
    dispatched_set.add(request_id)
    self._ensure_worker_polling(actor)

    try:
      await actor.dispatch_task(
          request_id, method_name, *args, **kwargs
      )
      # Re-ensure worker polling is active in case the previous polling loop
      # exited or died while dispatch_task was awaiting.
      self._ensure_worker_polling(actor)
      return request_id
    except Exception as exc:
      # Only decrement _in_flight if _poll_worker_loop hasn't already failed and cleared it.
      was_in_dispatched = request_id in dispatched_set
      if was_in_dispatched:
        dispatched_set.remove(request_id)
        self._in_flight = max(0, self._in_flight - 1)
      if self._evict_on_failure:
        self._evict_actor_from_pool(actor, exc)
      if (
          was_in_dispatched
          and self._retry_on_worker_failure
          and self._task_retries.get(request_id, 0) < self._max_task_retries
          and len(self._pool._actors) > 0
      ):
        self._task_retries[request_id] = (
            self._task_retries.get(request_id, 0) + 1
        )
        return await self.submit(
            request_id, method_name, *args, **dict(orig_kwargs)
        )
      if was_in_dispatched:
        payload = self._task_payloads.pop(request_id, None)
        self._task_retries.pop(request_id, None)
        if payload is not None:
          self._failed_tasks.append((request_id, payload, exc))
        self._notify_if_zero_flight()
      raise

  def _ensure_worker_polling(self, actor: ActorHandle) -> None:
    if actor in self._active_workers:
      return
    self._active_workers.add(actor)
    task = asyncio.create_task(self._poll_worker_loop(actor))
    self._poll_tasks.add(task)
    task.add_done_callback(self._poll_tasks.discard)

  async def _poll_worker_loop(self, actor: ActorHandle) -> None:
    discarded = False
    try:
      while not self._closed:
        dispatched_set = self._dispatched_tasks.setdefault(actor, set())
        if not dispatched_set:
          break
        try:
          response = await actor.poll_responses(timeout_s=LONG_POLL_TIMEOUT_S)
          if isinstance(response, ExecutionResponse):
            req_id: Optional[str] = None
            if response.request_id and response.request_id in dispatched_set:
              req_id = response.request_id
              dispatched_set.remove(req_id)
              self._in_flight = max(0, self._in_flight - 1)
            elif dispatched_set:
              req_id = dispatched_set.pop()
              self._in_flight = max(0, self._in_flight - 1)
            payload = (
                self._task_payloads.pop(req_id, None)
                if req_id is not None
                else None
            )
            if req_id is not None:
              self._task_retries.pop(req_id, None)
            try:
              res = response.unwrap()
              self._response_queue.put_nowait((res, None))
            except Exception as exc:  # pylint: disable=broad-exception-caught
              if req_id is not None and payload is not None:
                self._failed_tasks.append((req_id, payload, exc))
              self._response_queue.put_nowait((None, exc))
            self._notify_if_zero_flight()
        except asyncio.CancelledError:
          break
        except Exception as exc:  # pylint: disable=broad-exception-caught
          # Transport or polling failure on this worker; evict and/or retry in-flight tasks.
          self._active_workers.discard(actor)
          discarded = True
          if self._evict_on_failure:
            self._evict_actor_from_pool(actor, exc)
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

    batch: List[Tuple[Any, Optional[Exception]]] = []
    # When idle (_in_flight == 0), yield briefly (<= 50ms) so concurrent dispatch
    # coroutines can run without stalling for the full 50s long-poll timeout.
    wait_s = (
        min(timeout_s, 0.05)
        if (self._in_flight == 0 and self._response_queue.empty())
        else timeout_s
    )
    try:
      # Block until the first real (result, exc) completion arrives.
      while not batch:
        item = await asyncio.wait_for(
            self._response_queue.get(), timeout=wait_s
        )
        if item is self._sentinel:
          # Sentinel marks _in_flight reaching 0 or session close; return early if drained,
          # otherwise skip stale sentinels when new tasks are in flight or queued behind it.
          if self._in_flight == 0 and self._response_queue.empty():
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
