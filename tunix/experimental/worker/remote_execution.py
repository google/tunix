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

"""Universal Actor Model abstraction layer (`RemoteExecutionServer`, `ActorHandle`).

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
import contextlib
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
    NoReturn,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from absl import logging
import cloudpickle
from tunix.experimental.worker import network

try:
  import grpc as _grpc_lib
  import grpc.aio as _grpc_aio_lib

  _GRPC_AVAILABLE = True
except ImportError:
  _grpc_lib = None
  _grpc_aio_lib = None
  _GRPC_AVAILABLE = False


RPC_TIMEOUT_S = network.RPC_TIMEOUT_S
LONG_POLL_TIMEOUT_S = network.LONG_POLL_TIMEOUT_S
is_bulk_lz4_enabled = network.is_bulk_lz4_enabled
set_bulk_lz4_enabled = network.set_bulk_lz4_enabled

_BULK_CAPABILITY_METADATA = (("x-tunix-bulk", "1"),)


class TransientTransportError(ConnectionError):
  """Raised when a remote transport operation fails with a retryable status."""


def _running_loop() -> Optional["asyncio.AbstractEventLoop"]:
  """Returns the currently running event loop, or None if there is none."""
  try:
    return asyncio.get_running_loop()
  except RuntimeError:
    return None


def _client_supports_bulk(context: Any) -> bool:
  """Returns True if the gRPC caller advertised bulk transport support."""
  if context is None:
    return False
  inv_meta = context.invocation_metadata()
  if not inv_meta:
    return False
  for key, val in inv_meta:
    if key == "x-tunix-bulk" and val == "1":
      return True
  return False


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

  def _as_tuple(self) -> Tuple[Any, ...]:
    return (self.request_id, self.method_name, self.args, self.kwargs)

  def serialize_chunks(
      self,
      chunk_size: int = network._STREAM_CHUNK_BYTES,  # pylint: disable=protected-access
  ) -> Iterator[bytes]:
    """Serializes request into Pickle Protocol 5 out-of-band buffer chunks."""
    return network._iter_serialized_chunks(  # pylint: disable=protected-access
        self._as_tuple(), chunk_size=chunk_size
    )

  def serialize_async_chunks(
      self,
      chunk_size: int = network._STREAM_CHUNK_BYTES,  # pylint: disable=protected-access
  ) -> AsyncIterator[bytes]:
    """Serializes request into an async stream of Pickle Protocol 5 chunks."""
    # pylint: disable=protected-access
    total_bytes, manifest, views, raw_buffers, spec_iter = (
        network._prepare_serialized_chunk_specs(
            self._as_tuple(), chunk_size=chunk_size
        )
    )
    return network._iter_async_from_chunk_specs(
        manifest,
        views,
        raw_buffers,
        spec_iter,
        offload=total_bytes >= network._ASYNC_OFFLOAD_THRESHOLD_BYTES,
    )
    # pylint: enable=protected-access

  @classmethod
  def deserialize_chunks(cls, chunks: Iterable[bytes]) -> "ExecutionRequest":
    """Deserializes an ExecutionRequest from a synchronous stream of chunks."""
    # SECURITY WARNING: cloudpickle.loads executes arbitrary code via __reduce__
    # during deserialization. In production across untrusted boundaries, verify
    # ALTS/mTLS transport identity or cryptographic HMAC signatures before
    # calling deserialize_chunks(). Where dynamic function shipping is not
    # needed, use `pickle.Unpickler` (`find_class`) to whitelist only trusted
    # domain data types (`int`, `str`, `dict`, `list`, `data_types.*`).
    return cls(*network._deserialize_from_chunks(chunks))  # pylint: disable=protected-access

  @classmethod
  async def deserialize_async_chunks(
      cls,
      chunks: AsyncIterable[bytes],
      *,
      bulk_server: Optional[network._BulkTransferServer] = None,  # pylint: disable=protected-access
  ) -> "ExecutionRequest":
    """Deserializes an ExecutionRequest from an async stream of chunks.

    Note: When out-of-band bulk TCP transport is used, small out-of-band arrays
    (`< _BULK_PACK_MAX_BYTES = 4 MiB`) are returned as 64-byte-aligned zero-copy
    views sharing a single underlying packed buffer. Callers retaining only a
    small subset of arrays long-term should call `np.copy()` on them to allow
    the shared buffer to be garbage-collected early.
    """
    unpacked = await network._deserialize_from_async_chunks(  # pylint: disable=protected-access
        chunks, bulk_server=bulk_server
    )
    return cls(*unpacked)


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

  def _prepare_chunk_specs(
      self,
      chunk_size: int = network._STREAM_CHUNK_BYTES,  # pylint: disable=protected-access
  ) -> Tuple[
      int,
      bytes,
      List[memoryview],  # pylint: disable=g-bare-generic
      List[pickle.PickleBuffer],
      Iterator[network._ChunkSpec],  # pylint: disable=protected-access
  ]:
    """Prepares serialized views and chunk specs, recording errors on failure."""
    # pylint: disable=protected-access
    try:
      return network._prepare_serialized_chunk_specs(
          self._as_tuple(), chunk_size=chunk_size
      )
    except Exception as e:  # pylint: disable=broad-exception-caught
      self._record_serialization_error(e)
      return network._prepare_serialized_chunk_specs(
          self._as_tuple(), chunk_size=chunk_size
      )
    # pylint: enable=protected-access

  def serialize_chunks(
      self,
      chunk_size: int = network._STREAM_CHUNK_BYTES,  # pylint: disable=protected-access
  ) -> Iterator[bytes]:
    """Serializes response into Pickle Protocol 5 out-of-band buffer chunks."""
    # pylint: disable=protected-access
    try:
      _, chunk_iter = network._iter_serialized_chunks_with_size(
          self._as_tuple(), chunk_size=chunk_size
      )
      return chunk_iter
    except Exception as e:  # pylint: disable=broad-exception-caught
      self._record_serialization_error(e)
      _, chunk_iter = network._iter_serialized_chunks_with_size(
          self._as_tuple(), chunk_size=chunk_size
      )
      return chunk_iter
    # pylint: enable=protected-access

  def serialize_async_chunks(
      self,
      chunk_size: int = network._STREAM_CHUNK_BYTES,  # pylint: disable=protected-access
      *,
      bulk_server: Optional[network._BulkTransferServer] = None,  # pylint: disable=protected-access
  ) -> AsyncIterator[bytes]:
    """Serializes response into an async stream of Pickle Protocol 5 chunks."""
    # pylint: disable=protected-access
    total_bytes, manifest, views, raw_buffers, spec_iter = (
        self._prepare_chunk_specs(chunk_size=chunk_size)
    )
    if (
        bulk_server is not None
        and bulk_server.port > 0
        and network._can_use_bulk_transfer(total_bytes, views)
    ):
      return network._stream_bulk_pull_chunks(views, raw_buffers, bulk_server)
    return network._iter_async_from_chunk_specs(
        manifest,
        views,
        raw_buffers,
        spec_iter,
        offload=total_bytes >= network._ASYNC_OFFLOAD_THRESHOLD_BYTES,
    )
    # pylint: enable=protected-access

  @classmethod
  def deserialize_chunks(cls, chunks: Iterable[bytes]) -> "ExecutionResponse":
    """Deserializes an ExecutionResponse from a synchronous stream of chunks."""
    # SECURITY WARNING: cloudpickle.loads executes arbitrary code during
    # unpickling. Ensure payload authenticity over trusted channels before
    # deserialization, or use custom `pickle.Unpickler` (`find_class`) to
    # whitelist only trusted domain data types.
    return cls(*network._deserialize_from_chunks(chunks))  # pylint: disable=protected-access

  @classmethod
  async def deserialize_async_chunks(
      cls,
      chunks: AsyncIterable[bytes],
      *,
      allow_empty: bool = False,
      bulk_hosts: Sequence[str] = ("localhost",),
      seen_bulk_targets: Optional[Set[Tuple[str, int]]] = None,
  ) -> Optional["ExecutionResponse"]:
    """Deserializes an ExecutionResponse from an async stream of chunks.

    Note: When out-of-band bulk TCP transport is used, small out-of-band arrays
    (`< _BULK_PACK_MAX_BYTES = 4 MiB`) are returned as 64-byte-aligned zero-copy
    views sharing a single underlying packed buffer. Callers retaining only a
    small subset of arrays long-term should call `np.copy()` on them to allow
    the shared buffer to be garbage-collected early.
    """
    unpacked = await network._deserialize_from_async_chunks(  # pylint: disable=protected-access
        chunks,
        allow_empty=allow_empty,
        bulk_hosts=bulk_hosts,
        seen_bulk_targets=seen_bulk_targets,
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

  def __init__(
      self,
      instance: Optional[Any] = None,
      *,
      stream_chunk_bytes: int = network._STREAM_CHUNK_BYTES,  # pylint: disable=protected-access
      max_message_bytes: int = network._MAX_MESSAGE_BYTES,  # pylint: disable=protected-access
      enable_bulk_transport: bool = True,
      bulk_port: int = 0,
      bulk_hosts: Optional[Sequence[str]] = None,
  ):
    """Initializes the gRPC remote execution server.

    Args:
      instance: Optional bound worker object exposing methods to execute.
      stream_chunk_bytes: Slice size per frame on fallback gRPC streaming calls.
      max_message_bytes: Maximum allowed size of a single gRPC message frame.
      enable_bulk_transport: Whether to start an out-of-band zero-copy bulk TCP
        server for payloads >= 512 KiB.
      bulk_port: TCP port for the bulk transfer server (`0` selects an OS
        ephemeral port).
      bulk_hosts: Optional sequence of bare hostnames or IPv4/IPv6 literals on
        this machine to advertise via `GetBulkPort` for multi-NIC TCP striping.
        All entries must route to this server instance. Cannot be specified when
        `enable_bulk_transport=False`.
    """
    if not enable_bulk_transport and bulk_hosts is not None:
      raise ValueError(
          "bulk_hosts cannot be specified when enable_bulk_transport=False."
      )
    # pylint: disable=protected-access
    network._validate_stream_config(stream_chunk_bytes, max_message_bytes)
    self._bulk_hosts: Tuple[str, ...] = network._validate_bulk_hosts(
        bulk_hosts, allow_empty=bulk_hosts is None
    )
    # pylint: enable=protected-access
    super().__init__(instance)
    self._server: Optional[Any] = None
    self._serve_loop: Optional[Any] = None
    self._bulk_server: Optional[network._BulkTransferServer] = None  # pylint: disable=protected-access
    self._stream_chunk_bytes = stream_chunk_bytes
    self._max_message_bytes = max_message_bytes
    self._enable_bulk_transport = enable_bulk_transport
    self._configured_bulk_port = bulk_port

  async def _handle_get_bulk_port(
      self, request_bytes: bytes, context: Any
  ) -> bytes:
    """Returns the TCP port and optional `bulk_hosts` of the bulk server."""
    del request_bytes, context
    bulk_port = (
        self._bulk_server.port
        if (self._enable_bulk_transport and self._bulk_server is not None)
        else 0
    )
    return network._encode_bulk_port_response(bulk_port, self._bulk_hosts)  # pylint: disable=protected-access

  async def _abort_failed_precondition(
      self, context: Any, exc: Exception
  ) -> NoReturn:
    if context is not None and _grpc_lib is not None:
      await context.abort(_grpc_lib.StatusCode.FAILED_PRECONDITION, str(exc))
    raise exc

  async def _handle_execute(
      self, request_iterator: AsyncIterator[bytes], context: Any
  ) -> AsyncIterator[bytes]:
    """Handles bidirectional chunked streaming execution requests."""
    try:
      request = await ExecutionRequest.deserialize_async_chunks(
          request_iterator, bulk_server=self._bulk_server
      )
      response = await self.execute_request(request)
      del request
    except network._BulkTransportError as exc:  # pylint: disable=protected-access
      await self._abort_failed_precondition(context, exc)
    except Exception as e:  # pylint: disable=broad-exception-caught
      response = ExecutionResponse(
          error_message=str(e),
          error_type=type(e).__name__,
          traceback=traceback_lib.format_exc(),
      )
    use_bulk = self._enable_bulk_transport and _client_supports_bulk(context)
    async for chunk in response.serialize_async_chunks(
        chunk_size=self._stream_chunk_bytes,
        bulk_server=self._bulk_server if use_bulk else None,
    ):
      yield chunk
    del response

  async def _handle_dispatch_task(
      self, request_iterator: AsyncIterator[bytes], context: Any
  ) -> bytes:
    """Handles client-streaming task dispatch and returns the task ACK ID."""
    try:
      request = await ExecutionRequest.deserialize_async_chunks(
          request_iterator, bulk_server=self._bulk_server
      )
    except network._BulkTransportError as exc:  # pylint: disable=protected-access
      await self._abort_failed_precondition(context, exc)
    request_id = await self.dispatch_task(request)
    return cloudpickle.dumps(request_id)

  async def _handle_poll_responses(
      self, request_bytes: bytes, context: Any
  ) -> AsyncIterator[bytes]:
    """Handles server-streaming long-polling for completed task responses."""
    timeout_s = (
        cloudpickle.loads(request_bytes)  # pylint: disable=g-unsafe-pickle-load
        if request_bytes
        else LONG_POLL_TIMEOUT_S
    )
    response = await self.poll_response(timeout_s=timeout_s)
    if response is None:
      return
    completed = False
    # pylint: disable=protected-access
    poll_chunk_bytes = min(
        self._stream_chunk_bytes, network._CONCURRENT_STREAM_CHUNK_BYTES
    )
    use_bulk = self._enable_bulk_transport and _client_supports_bulk(context)
    network._inc_active_poll_rpcs()
    try:
      async for chunk in response.serialize_async_chunks(
          chunk_size=poll_chunk_bytes,
          bulk_server=self._bulk_server if use_bulk else None,
      ):
        yield chunk
      completed = True
    finally:
      network._dec_active_poll_rpcs()
      if not completed:
        self._get_response_queue().put_nowait(response)
    # pylint: enable=protected-access

  async def start_serving_async(self, port: int = 50051) -> Any:
    """Starts an asynchronous gRPC server listening on [::]:port."""
    if not _GRPC_AVAILABLE or _grpc_lib is None or _grpc_aio_lib is None:
      raise RuntimeError("grpc is not installed or available.")

    # pylint: disable=protected-access
    self._server = _grpc_aio_lib.server(
        options=network._grpc_options(self._max_message_bytes)
    )
    # pylint: enable=protected-access
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
            "GetBulkPort": _grpc_lib.unary_unary_rpc_method_handler(
                self._handle_get_bulk_port,
                request_deserializer=lambda b: b,
                response_serializer=lambda b: b,
            ),
        },
    )
    self._server.add_generic_rpc_handlers((handler,))
    # NOTE: add_insecure_port is for local loopback / isolated pod testing (experimental v0).
    # For production across trust boundaries, use secure_server_credentials (ALTS/mTLS).
    try:
      self._server.add_insecure_port(f"[::]:{port}")
      # pylint: disable=protected-access
      if self._enable_bulk_transport and self._bulk_server is None:
        self._bulk_server = network._BulkTransferServer()
        self._bulk_server.start(port=self._configured_bulk_port)
      # pylint: enable=protected-access
      await self._server.start()
    except BaseException:
      if self._bulk_server is not None:
        bulk_srv = self._bulk_server
        self._bulk_server = None
        await asyncio.get_running_loop().run_in_executor(None, bulk_srv.stop)
      if self._server is not None:
        grpc_srv = self._server
        self._server = None
        with contextlib.suppress(Exception):
          await grpc_srv.stop(0)
      raise
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
      if self._bulk_server is not None:
        self._bulk_server.stop()
        self._bulk_server = None
      self._serve_loop = None
      asyncio.set_event_loop(None)
      loop.close()

  async def stop_serving(self, grace: float = 0.5) -> None:
    if self._server:
      await self._server.stop(grace)
    if self._bulk_server is not None:
      bulk_srv = self._bulk_server
      self._bulk_server = None
      await asyncio.get_running_loop().run_in_executor(None, bulk_srv.stop)


class ActorHandle(abc.ABC):
  """Stateful 1-to-1 routing handle targeting a specific remote worker instance."""

  @classmethod
  def from_address(
      cls,
      target_address: str,
      *,
      rpc_timeout_s: Optional[float] = RPC_TIMEOUT_S,
      stream_chunk_bytes: int = network._STREAM_CHUNK_BYTES,  # pylint: disable=protected-access
      max_message_bytes: int = network._MAX_MESSAGE_BYTES,  # pylint: disable=protected-access
      enable_bulk_transport: bool = True,
      bulk_hosts: Optional[Sequence[str]] = None,
  ) -> "ActorHandle":
    """Instantiates a remote actor handle targeting the specified string URI.

    Args:
      target_address: Target worker URI (e.g. `grpc://10.0.0.1:50051`).
      rpc_timeout_s: Per-RPC deadline in seconds, or `None` for no timeout.
      stream_chunk_bytes: Slice size per frame on fallback gRPC streaming calls.
      max_message_bytes: Maximum allowed size of a single gRPC message frame.
      enable_bulk_transport: Whether to use out-of-band zero-copy bulk TCP
        sockets for payloads >= 512 KiB.
      bulk_hosts: Optional sequence of bare hostnames or IPv4/IPv6 literals on
        the target worker for multi-NIC bulk TCP striping. When provided, takes
        precedence over server-advertised `bulk_hosts` and the host extracted
        from `target_address`. All entries must route to the same remote server
        instance. Reachable hosts discovered during the initial TCP probe are
        cached until the bulk port is invalidated. Cannot be specified when
        `enable_bulk_transport=False`.

    Returns:
      An `ActorHandle` instance targeting `target_address`.
    """
    if target_address.startswith("grpc://") and _GRPC_AVAILABLE:
      return GrpcRemoteActorHandle(
          target_address=target_address,
          rpc_timeout_s=rpc_timeout_s,
          stream_chunk_bytes=stream_chunk_bytes,
          max_message_bytes=max_message_bytes,
          enable_bulk_transport=enable_bulk_transport,
          bulk_hosts=bulk_hosts,
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

  async def close(self) -> None:
    """Releases any underlying transport resources held by this handle."""
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
      stream_chunk_bytes: int = network._STREAM_CHUNK_BYTES,  # pylint: disable=protected-access
      max_message_bytes: int = network._MAX_MESSAGE_BYTES,  # pylint: disable=protected-access
      enable_bulk_transport: bool = True,
      bulk_hosts: Optional[Sequence[str]] = None,
  ):
    """Initializes a gRPC actor handle targeting `target_address`.

    Args:
      target_address: Target worker URI (e.g. `grpc://10.0.0.1:50051`).
      rpc_timeout_s: Per-RPC deadline in seconds, or `None` for no timeout.
      stream_chunk_bytes: Slice size per frame on fallback gRPC streaming calls.
      max_message_bytes: Maximum allowed size of a single gRPC message frame.
      enable_bulk_transport: Whether to use out-of-band zero-copy bulk TCP
        sockets for payloads >= 512 KiB.
      bulk_hosts: Optional sequence of bare hostnames or IPv4/IPv6 literals on
        the target worker for multi-NIC bulk TCP striping. Overrides
        server-advertised `bulk_hosts` and the host extracted from
        `target_address`. All entries must route to the same remote server
        instance. Cannot be specified when `enable_bulk_transport=False`.

    Raises:
      RuntimeError: If `grpc` is not installed or available.
      ValueError: If streaming or `bulk_hosts` configuration is invalid.
    """
    if not _GRPC_AVAILABLE or _grpc_aio_lib is None:
      raise RuntimeError("grpc is not installed or available.")
    if not enable_bulk_transport and bulk_hosts is not None:
      raise ValueError(
          "bulk_hosts cannot be specified when enable_bulk_transport=False."
      )
    # pylint: disable=protected-access
    network._validate_stream_config(stream_chunk_bytes, max_message_bytes)
    self._configured_bulk_hosts: Tuple[str, ...] = network._validate_bulk_hosts(
        bulk_hosts, allow_empty=bulk_hosts is None
    )
    self.target_address = target_address
    self._host_port = target_address.removeprefix("grpc://")
    self._enable_bulk_transport = enable_bulk_transport
    if enable_bulk_transport:
      try:
        self._bulk_host, _ = network._extract_host_and_port(target_address)
      except ValueError as exc:
        if not self._configured_bulk_hosts:
          raise ValueError(
              "Bulk TCP transport requires a direct host[:port] target address "
              f"(got {target_address!r}); pass enable_bulk_transport=False to "
              "use non-host[:port] gRPC target schemes."
          ) from exc
        self._bulk_host = self._configured_bulk_hosts[0]
      self._active_bulk_hosts: Tuple[str, ...] = (
          self._configured_bulk_hosts or (self._bulk_host,)
      )
    else:
      self._bulk_host = ""
      self._active_bulk_hosts = ()
    # pylint: enable=protected-access
    self._bulk_port: Optional[int] = None
    self._bulk_probe_failed_until: float = 0.0
    self._bulk_fallback_logged: bool = False
    self._resolve_task: Optional[asyncio.Task[int]] = None
    self._bulk_port_gen: int = 0
    self._seen_bulk_targets: Set[Tuple[str, int]] = set()
    self._channel: Optional[Any] = None
    self._channel_loop: Optional[asyncio.AbstractEventLoop] = None
    self._rpc: Optional[Any] = None
    self._dispatch_rpc: Optional[Any] = None
    self._poll_rpc: Optional[Any] = None
    self._get_bulk_port_rpc: Optional[Any] = None
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
    self._sync_get_bulk_port_rpc: Optional[Any] = None
    self._sync_lock = threading.Lock()

  def _make_rpc(self, channel: Any) -> Any:
    return channel.stream_stream(
        "/tunix.ExecutionService/Execute",
        request_serializer=lambda b: b,
        response_deserializer=lambda b: b,
    )

  def _make_get_bulk_port_rpc(self, channel: Any) -> Any:
    return channel.unary_unary(
        "/tunix.ExecutionService/GetBulkPort",
        request_serializer=lambda b: b,
        response_deserializer=lambda b: b,
    )

  def _get_active_resolve_task(
      self, loop: asyncio.AbstractEventLoop
  ) -> Optional[asyncio.Task[int]]:
    task = self._resolve_task
    if task is None or task.done():
      return None
    if task.get_loop() is loop:
      return task
    return None

  def _record_bulk_fallback(self, msg: str, *args: Any) -> None:
    """Negative-caches a failed bulk port probe with cooldown and logs once."""
    self._bulk_port = 0
    self._bulk_probe_failed_until = (
        time.monotonic() + network._BULK_PROBE_RETRY_COOLDOWN_S  # pylint: disable=protected-access
    )
    if not self._bulk_fallback_logged:
      self._bulk_fallback_logged = True
      logging.warning(msg, *args)
    else:
      logging.debug(msg, *args)

  def _invalidate_bulk_port(self) -> None:
    """Evicts cached bulk port and closes pooled sockets to bulk targets."""
    self._bulk_port_gen += 1
    self._resolve_task = None
    targets = set(self._seen_bulk_targets)
    if self._bulk_port is not None and self._bulk_port > 0:
      for host in self._active_bulk_hosts:
        targets.add((host, self._bulk_port))
    self._bulk_port = None
    self._bulk_probe_failed_until = 0.0
    self._active_bulk_hosts = (
        (self._configured_bulk_hosts or (self._bulk_host,))
        if self._enable_bulk_transport
        else ()
    )
    for host, port in targets:
      network._DEFAULT_BULK_SOCKET_POOL.close_target(host, port)  # pylint: disable=protected-access

  async def _probe_and_cache_bulk_port(
      self,
      port: int,
      gen: int,
      advertised_hosts: Sequence[str] = (),
  ) -> int:
    """Probes TCP connectivity across candidate NIC hosts concurrently."""
    if port <= 0:
      if self._bulk_port_gen == gen:
        self._bulk_port = 0
        self._bulk_probe_failed_until = float("inf")
      return 0
    candidate_hosts: Tuple[str, ...] = (
        self._configured_bulk_hosts
        or tuple(advertised_hosts)
        or (self._bulk_host,)
    )
    # pylint: disable=protected-access
    results = await asyncio.gather(
        *(
            network._DEFAULT_BULK_SOCKET_POOL.probe_async(host, port)
            for host in candidate_hosts
        ),
        return_exceptions=True,
    )
    # pylint: enable=protected-access
    for res in results:
      if isinstance(res, BaseException) and not isinstance(res, Exception):
        raise res
    reachable_hosts: List[str] = []
    failed_hosts: List[Tuple[str, BaseException]] = []
    for host, res in zip(candidate_hosts, results):
      if isinstance(res, BaseException):
        failed_hosts.append((host, res))
      else:
        reachable_hosts.append(host)
    if not reachable_hosts:
      if self._bulk_port_gen == gen:
        failures_str = ", ".join(
            f"{f'[{host}]:{port}' if ':' in host else f'{host}:{port}'} ({exc})"
            for host, exc in failed_hosts
        )
        self._record_bulk_fallback(
            "Bulk TCP probe failed across all candidate NICs [%s]; falling"
            " back to gRPC chunks.",
            failures_str,
        )
      return 0
    if self._bulk_port_gen != gen:
      return 0
    if failed_hosts:
      for failed_host, failed_exc in failed_hosts:
        target_str = (
            f"[{failed_host}]:{port}"
            if ":" in failed_host
            else f"{failed_host}:{port}"
        )
        logging.warning(
            "Bulk TCP probe to NIC host %s failed (%s); continuing with"
            " reachable bulk_hosts=%s.",
            target_str,
            failed_exc,
            tuple(reachable_hosts),
        )
    self._bulk_port = port
    self._active_bulk_hosts = tuple(reachable_hosts)
    self._bulk_probe_failed_until = 0.0
    for host in reachable_hosts:
      self._seen_bulk_targets.add((host, port))
    return port

  async def _query_and_probe_bulk_port(
      self,
      get_bulk_port_rpc: Any,
      gen: int,
  ) -> int:
    """Calls `GetBulkPort` and probes the returned TCP port across NIC hosts."""
    try:
      probe_rpc_timeout = (
          min(float(self._rpc_timeout_s), network._BULK_PROBE_TIMEOUT_S)  # pylint: disable=protected-access
          if self._rpc_timeout_s is not None
          else network._BULK_PROBE_TIMEOUT_S  # pylint: disable=protected-access
      )
      resp_bytes = await get_bulk_port_rpc(b"", timeout=probe_rpc_timeout)
      try:
        port, advertised_hosts = network._decode_bulk_port_response(  # pylint: disable=protected-access
            resp_bytes
        )
      except ValueError as exc:
        if self._bulk_port_gen == gen:
          self._record_bulk_fallback(
              "Invalid GetBulkPort response from %s (%s); falling back to gRPC"
              " chunks.",
              self._host_port,
              exc,
          )
        return 0
      return await self._probe_and_cache_bulk_port(
          port, gen, advertised_hosts=advertised_hosts
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      if self._bulk_port_gen == gen:
        if (
            _grpc_lib is not None
            and _grpc_aio_lib is not None
            and isinstance(exc, _grpc_aio_lib.AioRpcError)
            and exc.code() == _grpc_lib.StatusCode.UNIMPLEMENTED
        ):
          self._bulk_port = 0
          self._bulk_probe_failed_until = float("inf")
        else:
          self._record_bulk_fallback(
              "GetBulkPort RPC to %s failed (%s); falling back to gRPC"
              " chunks.",
              self._host_port,
              exc,
          )
    return 0

  async def _ensure_bulk_port(self, get_bulk_port_rpc: Any) -> int:
    """Resolves and probes the target server's out-of-band bulk TCP port."""
    if not self._enable_bulk_transport or get_bulk_port_rpc is None:
      return 0
    loop = asyncio.get_running_loop()
    if self._bulk_port is not None:
      if self._bulk_port > 0:
        return self._bulk_port
      now = time.monotonic()
      if now < self._bulk_probe_failed_until:
        return 0
      # Cooldown expired: schedule a non-blocking background re-probe on the
      # active loop so in-flight RPCs never stall for 2s when the bulk port is
      # firewalled.
      if self._get_active_resolve_task(loop) is None:
        self._bulk_probe_failed_until = (
            now + network._BULK_PROBE_RETRY_COOLDOWN_S  # pylint: disable=protected-access
        )
        self._resolve_task = loop.create_task(
            self._query_and_probe_bulk_port(
                get_bulk_port_rpc, self._bulk_port_gen
            )
        )
      return 0
    task = self._get_active_resolve_task(loop)
    if task is None:
      task = loop.create_task(
          self._query_and_probe_bulk_port(
              get_bulk_port_rpc, self._bulk_port_gen
          )
      )
      self._resolve_task = task
    try:
      return await asyncio.shield(task)
    except asyncio.CancelledError:
      curr_task = asyncio.current_task(loop)
      if curr_task is not None and curr_task.cancelling() > 0:
        raise
      return 0

  async def _prepare_request_chunks(
      self,
      request: ExecutionRequest,
      get_bulk_port_rpc: Any,
      *,
      push_tracker: Optional[network._BulkPushTracker] = None,  # pylint: disable=protected-access
      need_port_for_response: bool = True,
  ) -> Tuple[int, AsyncIterator[bytes]]:
    """Prepares serialized request chunks and resolves bulk port when required."""
    # pylint: disable=protected-access
    total_bytes, manifest, views, raw_buffers, spec_iter = (
        network._prepare_serialized_chunk_specs(
            request._as_tuple(), chunk_size=self._stream_chunk_bytes
        )
    )
    can_bulk_push = (
        self._enable_bulk_transport
        and network._can_use_bulk_transfer(total_bytes, views)
    )
    bulk_port = 0
    if can_bulk_push or need_port_for_response:
      try:
        bulk_port = await self._ensure_bulk_port(get_bulk_port_rpc)
      except BaseException:
        network._release_views_and_buffers(views, raw_buffers)
        raise
    if can_bulk_push and bulk_port > 0:
      return (
          bulk_port,
          network._stream_bulk_push_chunks(
              views,
              raw_buffers,
              self._active_bulk_hosts,
              bulk_port,
              push_tracker=push_tracker,
              seen_bulk_targets=self._seen_bulk_targets,
          ),
      )
    return (
        bulk_port,
        network._iter_async_from_chunk_specs(
            manifest,
            views,
            raw_buffers,
            spec_iter,
            offload=total_bytes >= network._ASYNC_OFFLOAD_THRESHOLD_BYTES,
        ),
    )
    # pylint: enable=protected-access

  def _check_post_rpc_push_tracker(
      self, push_tracker: network._BulkPushTracker  # pylint: disable=protected-access
  ) -> None:
    if push_tracker.error is not None:
      logging.debug(
          "Bulk push post-ACK error after RPC succeeded: %s",
          push_tracker.error,
      )
      self._invalidate_bulk_port()

  def _handle_rpc_exception(
      self,
      exc: BaseException,
      push_tracker: network._BulkPushTracker,  # pylint: disable=protected-access
  ) -> NoReturn:
    """Invalidates cached bulk port only on transport errors and re-raises."""
    if isinstance(exc, asyncio.CancelledError):
      curr_task = asyncio.current_task()
      if curr_task is not None and curr_task.cancelling() > 0:
        raise exc
    if push_tracker.error is not None:
      self._invalidate_bulk_port()
      raise push_tracker.error from exc
    if (
        _grpc_lib is not None
        and _grpc_aio_lib is not None
        and isinstance(exc, _grpc_aio_lib.AioRpcError)
        and exc.code() == _grpc_lib.StatusCode.FAILED_PRECONDITION
    ):
      self._invalidate_bulk_port()
      raise ConnectionError(str(exc.details())) from exc
    if isinstance(exc, network._BulkTransportError):  # pylint: disable=protected-access
      self._invalidate_bulk_port()
    raise exc

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
      self._channel = _grpc_aio_lib.insecure_channel(
          self._host_port,
          options=network._grpc_channel_options(self._max_message_bytes),  # pylint: disable=protected-access
      )
      self._channel_loop = current_loop
      self._rpc = self._make_rpc(self._channel)
      self._get_bulk_port_rpc = self._make_get_bulk_port_rpc(self._channel)
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
          await old_channel.close()
        except Exception:  # pylint: disable=broad-exception-caught
          pass
    return self._channel

  async def _execute_rpc(
      self,
      rpc: Any,
      get_bulk_port_rpc: Any,
      method_name: Optional[str],
      args: Sequence[Any],
      kwargs: Dict[str, Any],
  ) -> Any:
    """Streams an ExecutionRequest over `rpc` and unwraps the ExecutionResponse."""
    request = ExecutionRequest(
        method_name=method_name, args=args, kwargs=kwargs
    )
    push_tracker = network._BulkPushTracker()  # pylint: disable=protected-access
    bulk_port, chunks = await self._prepare_request_chunks(
        request,
        get_bulk_port_rpc,
        push_tracker=push_tracker,
        need_port_for_response=True,
    )
    call_kwargs: Dict[str, Any] = {"timeout": self._rpc_timeout_s}
    if bulk_port > 0:
      call_kwargs["metadata"] = _BULK_CAPABILITY_METADATA
    call = rpc(chunks, **call_kwargs)
    try:
      response = await ExecutionResponse.deserialize_async_chunks(
          call,
          bulk_hosts=self._active_bulk_hosts,
          seen_bulk_targets=self._seen_bulk_targets,
      )
    except BaseException as exc:  # pylint: disable=broad-exception-caught
      self._handle_rpc_exception(exc, push_tracker)
    self._check_post_rpc_push_tracker(push_tracker)
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
          self._host_port,
          options=network._grpc_channel_options(self._max_message_bytes),  # pylint: disable=protected-access
      )
      self._sync_rpc = self._make_rpc(self._sync_channel)
      self._sync_get_bulk_port_rpc = self._make_get_bulk_port_rpc(
          self._sync_channel
      )
    return await self._execute_rpc(
        self._sync_rpc, self._sync_get_bulk_port_rpc, method_name, args, kwargs
    )

  async def asubmit(
      self, method_name: Optional[str] = None, *args, **kwargs
  ) -> Any:
    """Asynchronously invokes remote method over gRPC."""
    await self._ensure_async_channel()
    return await self._execute_rpc(
        self._rpc, self._get_bulk_port_rpc, method_name, args, kwargs
    )

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
    push_tracker = network._BulkPushTracker()  # pylint: disable=protected-access
    _, chunks = await self._prepare_request_chunks(
        request,
        self._get_bulk_port_rpc,
        push_tracker=push_tracker,
        need_port_for_response=False,
    )
    try:
      result = await self._dispatch_rpc(chunks, timeout=self._rpc_timeout_s)
    except BaseException as exc:  # pylint: disable=broad-exception-caught
      self._handle_rpc_exception(exc, push_tracker)
    self._check_post_rpc_push_tracker(push_tracker)
    return result

  async def poll_responses(
      self, timeout_s: float = LONG_POLL_TIMEOUT_S
  ) -> Optional[ExecutionResponse]:
    """Long-polls remote server response queue for completed task results."""
    await self._ensure_async_channel()
    assert self._poll_rpc is not None
    bulk_port = await self._ensure_bulk_port(self._get_bulk_port_rpc)
    call_kwargs: Dict[str, Any] = {"timeout": self._rpc_timeout_s}
    if bulk_port > 0:
      call_kwargs["metadata"] = _BULK_CAPABILITY_METADATA
    call = self._poll_rpc(timeout_s, **call_kwargs)
    try:
      return await ExecutionResponse.deserialize_async_chunks(
          call,
          allow_empty=True,
          bulk_hosts=self._active_bulk_hosts,
          seen_bulk_targets=self._seen_bulk_targets,
      )
    except network._BulkTransportError as exc:  # pylint: disable=protected-access
      self._invalidate_bulk_port()
      if hasattr(call, "cancel"):
        call.cancel()
      raise TransientTransportError(
          f"Transient bulk transport error polling {self._host_port}: {exc}"
      ) from exc
    except BaseException as exc:  # pylint: disable=broad-exception-caught
      if (
          _grpc_lib is not None
          and _grpc_aio_lib is not None
          and isinstance(exc, _grpc_aio_lib.AioRpcError)
          and exc.code()
          in (
              _grpc_lib.StatusCode.UNAVAILABLE,
              _grpc_lib.StatusCode.DEADLINE_EXCEEDED,
          )
      ):
        raise TransientTransportError(
            f"Transient gRPC error polling {self._host_port}:"
            f" {exc.code().name} ({exc.details()})"
        ) from exc
      raise

  async def close(self) -> None:
    loop = _running_loop()
    if loop is not None:
      active_task = self._get_active_resolve_task(loop)
      if active_task is not None:
        active_task.cancel()
    self._invalidate_bulk_port()
    self._seen_bulk_targets.clear()
    if self._channel is not None:
      # `self._channel` is lazily created on the caller's event loop (e.g.
      # `RLProgram.run()`), which is typically already closed by the time
      # `WorkerRegistry.close()` runs on a fresh shutdown loop. Calling
      # `channel.close()` releases the underlying C-core channel so the
      # interpreter does not hang at exit, while ignoring cross-loop poller
      # cleanup errors so shutdown can proceed to `_sync_loop` teardown.
      try:
        await self._channel.close()
      except Exception:  # pylint: disable=broad-exception-caught
        pass
      self._channel = None
      self._channel_loop = None
      self._rpc = None
      self._get_bulk_port_rpc = None
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
      self._sync_get_bulk_port_rpc = None


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
    handle = standalone_task.remote

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
