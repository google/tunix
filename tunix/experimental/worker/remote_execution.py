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

Eliminates RPC boilerplate (`rpc_generate`, `rpc_sync_weights`, etc.) across worker
types by serializing arbitrary method invocations (`submit`, `asubmit`) over a universal
execution protocol (`ExecutionRequest`).

Security Notes / Trust Boundaries:
  This module uses `cloudpickle` to serialize and deserialize dynamic execution requests
  and responses (`ExecutionRequest`, `ExecutionResponse`). Because `cloudpickle.loads()`
  executes arbitrary Python code via `__reduce__` gadgets during unpickling, this protocol
  must NEVER be exposed to unauthenticated or untrusted network traffic.
  For production deployment across trust boundaries (e.g. multi-tenant Borg jobs or external
  networks), ensure payloads are authenticated and encrypted via ALTS / mTLS channels (`secure_channel`
  / `secure_server_credentials`) or signed via shared HMAC-SHA256 signatures before unpickling.
  Where dynamic function shipping is not required, use a custom
  `pickle.Unpickler` (`find_class`) to whitelist only trusted domain data types
  (`int`, `str`, `dict`, `list`, `numpy.ndarray`, `data_types.*`).
"""


import abc
import asyncio
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
DEFAULT_TIMEOUT_S = 300.0

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
  """Universal execution request payload wrapping method name, args, and kwargs."""

  def __init__(
      self,
      method_name: Optional[str] = None,
      args: Optional[Sequence[Any]] = None,
      kwargs: Optional[Dict[str, Any]] = None,
  ):
    self.method_name = method_name or "__call__"
    self.args: Tuple[Any, ...] = tuple(args or ())
    self.kwargs: Dict[str, Any] = dict(kwargs or {})


  def serialize(self) -> bytes:
    """Serializes request to bytes using cloudpickle."""
    return cloudpickle.dumps((self.method_name, self.args, self.kwargs))

  @classmethod
  def deserialize(cls, payload: bytes) -> "ExecutionRequest":
    """Deserializes bytes into an ExecutionRequest."""
    # SECURITY WARNING: cloudpickle.loads executes arbitrary code via __reduce__ during
    # deserialization. In production across untrusted boundaries, verify ALTS/mTLS transport
    # identity or cryptographic HMAC signatures before calling cloudpickle.loads(). Where
    # dynamic function shipping is not needed, use `pickle.Unpickler` (`find_class`) to
    # whitelist only trusted domain data types (`int`, `str`, `dict`, `list`, `data_types.*`).
    method_name, args, kwargs = cloudpickle.loads(payload)
    return cls(method_name=method_name, args=args, kwargs=kwargs)


class ExecutionResponse:
  """Universal execution response wrapping a result or a structured error."""

  def __init__(
      self,
      result: Any = None,
      error_message: Optional[str] = None,
      error_type: Optional[str] = None,
      traceback: Optional[str] = None,
      retryable: bool = False,
  ):
    self.result = result
    self.error_message = error_message
    self.error_type = error_type
    self.traceback = traceback
    self.retryable = retryable

  def serialize(self) -> bytes:
    return cloudpickle.dumps((
        self.result,
        self.error_message,
        self.error_type,
        self.traceback,
        self.retryable,
    ))

  @classmethod
  def deserialize(cls, payload: bytes) -> "ExecutionResponse":
    # SECURITY WARNING: cloudpickle.loads executes arbitrary code during unpickling. Ensure
    # payload authenticity over trusted channels before deserialization, or use custom
    # `pickle.Unpickler` (`find_class`) to whitelist only trusted domain data types.
    result, err_msg, err_type, tb, retryable = cloudpickle.loads(payload)
    return cls(
        result=result,
        error_message=err_msg,
        error_type=err_type,
        traceback=tb,
        retryable=retryable,
    )

  def unwrap(self) -> Any:
    """Returns the result, or raises RuntimeError if the remote call failed."""
    if self.error_message is not None:
      message = f"RemoteExecutionError [{self.error_type}]: {self.error_message}"
      if self.traceback:
        message = f"{message}\nRemote traceback:\n{self.traceback}"
      raise RuntimeError(message)
    return self.result


class RemoteExecutionServer(abc.ABC):
  """Daemon that binds a target domain object and executes method calls dynamically."""

  def __init__(self, instance: Optional[Any] = None):
    self._instance: Optional[Any] = instance

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

  def execute_sync_request(self, request: ExecutionRequest) -> ExecutionResponse:
    """Dynamically resolves and executes synchronous method on the bound instance."""
    if self._instance is None:
      return ExecutionResponse(
          error_message="RemoteExecutionServer has no registered instance.",
          error_type="InstanceNotBoundError",
      )

    target_name = request.method_name or "__call__"
    method = getattr(self._instance, target_name, None)
    if method is None or not callable(method):
      return ExecutionResponse(
          error_message=f"Method '{target_name}' not found on bound instance.",
          error_type="AttributeError",
      )

    if inspect.iscoroutinefunction(method):
      return ExecutionResponse(
          error_message=(
              f"Method '{target_name}' is a coroutine function; "
              "use asubmit()."
          ),
          error_type="RuntimeError",
      )

    try:
      result = method(*request.args, **request.kwargs)
      return ExecutionResponse(result=result)
    except Exception as e:  # pylint: disable=broad-exception-caught
      return ExecutionResponse(
          error_message=str(e),
          error_type=type(e).__name__,
          traceback=traceback_lib.format_exc(),
      )

  async def execute_request(self, request: ExecutionRequest) -> ExecutionResponse:
    """Dynamically resolves and executes method on the bound instance."""
    if self._instance is None:
      return ExecutionResponse(
          error_message="RemoteExecutionServer has no registered instance.",
          error_type="InstanceNotBoundError",
      )

    target_name = request.method_name or "__call__"
    method = getattr(self._instance, target_name, None)
    if method is None or not callable(method):
      return ExecutionResponse(
          error_message=(
              f"Method '{target_name}' not found on bound instance."
          ),
          error_type="AttributeError",
      )


    try:
      if inspect.iscoroutinefunction(method):
        result = await method(*request.args, **request.kwargs)
      else:
        result = method(*request.args, **request.kwargs)
      return ExecutionResponse(result=result)
    except Exception as e:  # pylint: disable=broad-exception-caught
      return ExecutionResponse(
          error_message=str(e),
          error_type=type(e).__name__,
          traceback=traceback_lib.format_exc(),
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
    try:
      return response.serialize()
    except Exception as e:  # pylint: disable=broad-exception-caught
      # The result itself is not serializable; return a structured error rather
      # than letting the RPC handler crash with an opaque failure.
      fallback = ExecutionResponse(
          error_message=(
              "failed to serialize result of type "
              f"{type(response.result).__name__}: {e}"
          ),
          error_type="ResultSerializationError",
          traceback=traceback_lib.format_exc(),
      )
      return fallback.serialize()

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
            )
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

    Runs an event loop for the server's lifetime. The previous implementation
    returned immediately after start(), so `asyncio.run` closed the loop and
    tore the server down before it could serve any request. To stop a server
    started this way, schedule stop_serving() on `serve_loop` from another
    thread.
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
      loop.run_until_complete(self._server.wait_for_termination())
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


class GrpcRemoteActorHandle(RemoteActorHandle):
  """ActorHandle connecting to GrpcRemoteExecutionServer over TCP sockets via gRPC."""

  def __init__(
      self,
      target_address: str,
      *,
      default_timeout_s: Optional[float] = DEFAULT_TIMEOUT_S,
  ):
    if not _GRPC_AVAILABLE or _grpc_aio_lib is None:
      raise RuntimeError("grpc is not installed or available.")
    self.target_address = target_address
    self._host_port = target_address.replace("grpc://", "")
    self._channel: Optional[Any] = None
    self._rpc: Optional[Any] = None
    self._default_timeout_s = default_timeout_s
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
          "GrpcRemoteActorHandle.submit() is blocking and cannot be called from "
          "a running event loop; use asubmit() instead."
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
    request = ExecutionRequest(method_name, args, kwargs)
    response: ExecutionResponse = await self._sync_rpc(
        request, timeout=self._default_timeout_s
    )
    return response.unwrap()

  async def asubmit(
      self, method_name: Optional[str] = None, *args, **kwargs
  ) -> Any:
    """Asynchronously invokes remote method over gRPC."""
    rpc = self._get_rpc()
    request = ExecutionRequest(method_name, args, kwargs)
    response: ExecutionResponse = await rpc(
        request, timeout=self._default_timeout_s
    )
    return response.unwrap()

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
    request = ExecutionRequest(method_name, args, kwargs)
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
    request = ExecutionRequest(method_name, args, kwargs)
    return await self._run_async(request)

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
      self, tasks: Sequence[Tuple[str, Sequence[Any], Dict[str, Any]]]
  ) -> AsyncIterator[Any]:
    """Dispatches a batch of tasks across pool and yields results strictly out-of-order."""
    raise NotImplementedError




class RoutingActorPool(ActorPool):
  """ActorPool with smart task routing (`route_key affinity, round-robin fallback`).

  Args:
    actors: Initial sequence of worker actor handles or string URI targets.
    router: Optional custom routing callable
      `(actors, method_name, args, kwargs) -> ActorHandle` or router
      module/object providing per-method handlers matching `method_name` with
      signature `(actors, args, kwargs) -> ActorHandle`.
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

      raise RuntimeError("RoutingActorPool contains no registered ActorHandles.")

    kwargs = kwargs or {}
    if self.router is not None:
      if method_name and hasattr(self.router, method_name) and callable(
          getattr(self.router, method_name)
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
      self, tasks: Sequence[Tuple[str, Sequence[Any], Dict[str, Any]]]
  ) -> AsyncIterator[Any]:
    """Dispatches batch across pool workers and yields results out-of-order as they finish."""
    if not self._actors:
      raise RuntimeError("RoutingActorPool contains no registered ActorHandles.")

    futures = []
    for method_name, args, kwargs in tasks:
      actor = self._get_next_actor(method_name, args, kwargs)
      task_kwargs = dict(kwargs)
      task_kwargs.pop("route_key", None)
      fut = asyncio.create_task(
          actor.asubmit(method_name, *args, **task_kwargs)
      )
      futures.append(fut)

    for completed_fut in asyncio.as_completed(futures):
      yield await completed_fut


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
