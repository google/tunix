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
import contextlib
import inspect
import threading
import traceback as traceback_lib
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

# Cap for a single gRPC message. The library default (~4 MiB) is far too small
# for training-batch payloads; raise it and enable keepalive so idle connections
# are detected.
_MAX_MESSAGE_BYTES = 128 * 1024 * 1024


def _grpc_options() -> List[Tuple[str, int]]:
  """Channel/server options lifting the message-size cap and enabling keepalive."""
  return [
      ("grpc.max_send_message_length", _MAX_MESSAGE_BYTES),
      ("grpc.max_receive_message_length", _MAX_MESSAGE_BYTES),
      ("grpc.keepalive_time_ms", 20000),
      ("grpc.keepalive_timeout_ms", 10000),
      ("grpc.keepalive_permit_without_calls", 1),
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
    response = await self.execute_request(request)
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

  async def _handle_execute(self, request_bytes: bytes, context: Any) -> bytes:
    del context
    try:
      request = ExecutionRequest.deserialize(request_bytes)
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
    request = ExecutionRequest.deserialize(request_bytes)
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
  def from_address(cls, target_address: str) -> "ActorHandle":
    """Instantiates a remote actor handle targeting the specified string URI."""
    if target_address.startswith("grpc://") and _GRPC_AVAILABLE:
      return GrpcRemoteActorHandle(target_address=target_address)
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
  ):
    if not _GRPC_AVAILABLE or _grpc_aio_lib is None:
      raise RuntimeError("grpc is not installed or available.")
    self.target_address = target_address
    self._host_port = target_address.replace("grpc://", "")
    self._channel: Optional[Any] = None
    self._rpc: Optional[Any] = None
    self._async_loop: Optional[Any] = None
    self._rpc_timeout_s = rpc_timeout_s
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
    return channel.unary_unary(
        "/tunix.ExecutionService/Execute",
        request_serializer=lambda req: req.serialize(),
        response_deserializer=lambda b: ExecutionResponse.deserialize(b),
    )

  def _get_rpc(self) -> Any:
    """Returns the async stub, binding the channel to the calling loop once.

    Raises:
      RuntimeError: If reused from a different event loop than the one that
        created the channel. A gRPC aio channel belongs to its creating loop;
        using it from another produces failures far from the cause (typically
        a hang or a "attached to a different loop" error deep in the stack),
        so it is rejected here instead. Build a handle per loop.
    """
    running_loop = _running_loop()
    if self._rpc is None:
      assert _grpc_aio_lib is not None
      self._channel = _grpc_aio_lib.insecure_channel(
          self._host_port, options=_grpc_options()
      )
      self._rpc = self._make_rpc(self._channel)
      self._async_loop = running_loop
    elif running_loop is not None and self._async_loop not in (
        None,
        running_loop,
    ):
      raise RuntimeError(
          f"This handle's channel to {self.target_address} belongs to the"
          " event loop that first used it and cannot be shared with another."
          " Construct a handle per event loop."
      )
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
        request, timeout=self._rpc_timeout_s
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
        request, timeout=self._rpc_timeout_s
    )
    return response.unwrap()

  def _make_dispatch_task_rpc(self, channel: Any) -> Any:
    return channel.unary_unary(
        "/tunix.ExecutionService/DispatchTask",
        request_serializer=lambda req: req.serialize(),
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
    return await rpc(request, timeout=self._rpc_timeout_s)

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
  def add_actor(self, actor: Union[str, ActorHandle]) -> None:
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

  def add_actor(self, actor: Union[str, ActorHandle]) -> None:
    if isinstance(actor, str):
      self._actors.append(ActorHandle.from_address(actor))
    elif isinstance(actor, ActorHandle):
      self._actors.append(actor)
    else:
      raise TypeError(f"Expected str or ActorHandle, got {type(actor)}")

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
        contains `route_key`, stable hash routing (`hash(route_key) % N`) is
        used for sticky endpoint affinity (popped prior to remote dispatch).

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
        return self.router(self._actors, method_name, args, kwargs)
      else:
        raise TypeError(
            f"Router object {type(self.router)} must provide a method matching "
            f"'{method_name}' or be callable."
        )

    # Check for sticky routing key (e.g. route_key for KV-cache locality)
    route_key = kwargs.get("route_key")

    if route_key is not None:
      # Stable hash routing ensures identical route_keys consistently hit the same endpoint
      return self._actors[hash(route_key) % len(self._actors)]

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
      on_task_settled: Optional[Callable[["ActorHandle", str], None]] = None,
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
      on_task_settled: Optional `(actor, request_id)` callback invoked once per
        task when it stops occupying its worker. Load balancers use this to
        reclaim per-worker capacity.
    """
    session = PoolExecutionSession(self, on_task_settled=on_task_settled)
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
      on_task_settled: Optional[Callable[[ActorHandle, str], None]] = None,
  ):
    """Initializes the session.

    Args:
      pool: The pool whose workers this session dispatches to.
      on_task_settled: Optional callback invoked with `(actor, request_id)`
        exactly once per task, when it stops occupying the worker -- on
        completion, on dispatch failure, or when the worker's polling loop
        gives up. Load balancers use this to reclaim per-worker capacity.
    """
    self._pool = pool
    self._on_task_settled = on_task_settled
    self._response_queue: asyncio.Queue[Any] = asyncio.Queue()
    self._active_workers: set[ActorHandle] = set()
    self._dispatched_tasks: Dict[ActorHandle, set[str]] = {}
    self._work_events: Dict[ActorHandle, asyncio.Event] = {}
    self._poll_tasks: set[asyncio.Task[Any]] = set()
    self._in_flight = 0
    self._closed = False
    self._sentinel = object()

  def _notify_if_zero_flight(self) -> None:
    if self._in_flight == 0:
      self._response_queue.put_nowait(self._sentinel)

  def _settle(self, actor: ActorHandle, request_id: str) -> None:
    """Reports that `request_id` no longer occupies `actor`."""
    if self._on_task_settled is None:
      return
    try:
      self._on_task_settled(actor, request_id)
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception(
          "on_task_settled callback failed for request %r; the session"
          " continues but capacity accounting may now be wrong.",
          request_id,
      )

  async def submit(
      self,
      request_id: str,
      method_name: Optional[str] = None,
      *args,
      **kwargs,
  ) -> str:
    """Dispatches a task to a worker in the pool and tracks its completion.

    Args:
      request_id: Caller-chosen, non-empty, unique within this session. It is
        what completions are matched against, so the session cannot accept
        responsibility for a task it cannot name.
      method_name: Method to invoke on the worker's bound instance.
      *args: Positional arguments for the method.
      **kwargs: Keyword arguments for the method. `route_key` is consumed for
        routing and never forwarded to the worker.

    Returns:
      The id the task is tracked under.

    Raises:
      RuntimeError: If the session is closed.
      ValueError: If `request_id` is empty.
    """
    if self._closed:
      raise RuntimeError("PoolExecutionSession is closed.")
    if not request_id:
      # Letting the server mint one opens a window where its response cannot
      # be attributed to this task: the id is not known here until the ack
      # returns, and a response arriving first is dropped as unrecognized.
      raise ValueError(
          "submit() requires a non-empty request_id: completions are matched"
          " by id, and a server-minted id is unknown to this session until"
          " its ack returns."
      )
    actor = self._pool._get_next_actor(method_name, args, kwargs)
    # Routing metadata, not a method argument.
    kwargs.pop("route_key", None)

    # Register the id BEFORE dispatching. The worker can finish and enqueue its
    # response while dispatch_task is still awaiting its ack, and the polling
    # loop must be able to attribute that response to this task.
    dispatched_set = self._dispatched_tasks.setdefault(actor, set())
    dispatched_set.add(request_id)

    # Increment in_flight BEFORE dispatch to prevent race conditions with as_completed()
    self._in_flight += 1
    self._ensure_worker_polling(actor)

    try:
      req_id = await actor.dispatch_task(
          request_id, method_name, *args, **kwargs
      )
      if req_id != request_id:
        raise RuntimeError(
            f"Worker acked request {request_id!r} as {req_id!r}; the session"
            " tracks tasks by the id it supplied, so a renamed task could"
            " never be matched to its response."
        )
      # Re-ensure worker polling is active in case the previous polling loop
      # died due to a transport error while dispatch_task was awaiting.
      self._ensure_worker_polling(actor)
      if actor in self._work_events:
        self._work_events[actor].set()
      return req_id
    except Exception:
      dispatched_set.discard(request_id)
      self._in_flight -= 1
      self._settle(actor, request_id)
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
    event = self._work_events.setdefault(actor, asyncio.Event())
    try:
      while not self._closed:
        dispatched_set = self._dispatched_tasks.setdefault(actor, set())
        if not dispatched_set:
          event.clear()
          await event.wait()
          continue
        try:
          response = await actor.poll_responses(timeout_s=LONG_POLL_TIMEOUT_S)
          if isinstance(response, ExecutionResponse):
            if response.request_id not in dispatched_set:
              # Not one of ours: the server's response queue is shared, so
              # another consumer of the same worker may be polling it too.
              # Attributing it to an arbitrary task here would corrupt this
              # session's per-worker accounting.
              logging.warning(
                  "Ignoring response for unknown request_id %r from worker %r;"
                  " it was not dispatched by this session.",
                  response.request_id,
                  actor,
              )
              continue
            dispatched_set.discard(response.request_id)
            try:
              res = response.unwrap()
              self._response_queue.put_nowait((response.request_id, res, None))
            except Exception as exc:  # pylint: disable=broad-exception-caught
              self._response_queue.put_nowait(
                  (response.request_id, None, exc)
              )
            self._in_flight -= 1
            # Free the worker's slot before publishing zero-flight, so a
            # refill triggered by this completion sees the capacity.
            self._settle(actor, response.request_id)
            self._notify_if_zero_flight()
        except asyncio.CancelledError:
          break
        except Exception as exc:  # pylint: disable=broad-exception-caught
          # Transport or polling failure on this worker; fail all dispatched tasks on this worker.
          failed_task_ids = self._dispatched_tasks.pop(actor, set())
          failed_count = len(failed_task_ids)
          if failed_count > 0:
            for failed_id in failed_task_ids:
              self._response_queue.put_nowait((failed_id, None, exc))
            self._in_flight -= failed_count
            for failed_id in failed_task_ids:
              self._settle(actor, failed_id)
            self._notify_if_zero_flight()
          break
    finally:
      self._active_workers.discard(actor)

  async def as_completed(
      self,
  ) -> AsyncIterator[Tuple[Any, Optional[Exception]]]:
    """Yields (result, exception) tuples as tasks complete out-of-order."""
    async for _, result, exc in self.as_completed_with_ids():
      yield result, exc

  async def as_completed_with_ids(
      self,
  ) -> AsyncIterator[Tuple[str, Any, Optional[Exception]]]:
    """Yields (request_id, result, exception) as tasks complete out-of-order.

    Same stream as `as_completed`, but correlated: callers dispatching a batch
    across a pool need the id to reassemble results in their original order.
    """
    while not self._closed:
      item = await self._response_queue.get()
      if item is self._sentinel:
        if self._in_flight == 0 and self._response_queue.empty():
          break
        continue
      request_id, result, exc = item
      yield request_id, result, exc

  async def close(self) -> None:
    """Closes the session and cancels all background polling tasks."""
    self._closed = True
    self._response_queue.put_nowait(self._sentinel)
    for event in self._work_events.values():
      event.set()
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

      remote_func.remote = remote_func
      return remote_func
    else:
      raise TypeError(
          f"@remote expects a class or function, got {type(target)}"
      )

  if cls_or_func is None:
    return decorator
  return decorator(cls_or_func)
