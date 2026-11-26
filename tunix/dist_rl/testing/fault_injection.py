# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test-only RPC fault injection primitives and server wrappers for Tunix workers."""

import asyncio
from collections.abc import AsyncIterator, Callable
import contextvars
import dataclasses
import threading
import traceback
from typing import Any

import grpc
from tunix.dist_rl.worker import remote_execution

ANY: str = "*"

_CURRENT_GRPC_CONTEXT: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "_CURRENT_GRPC_CONTEXT", default=None
)


class FaultGate:
  """Loop-native synchronization gate with zero threadpool usage."""

  def __init__(self) -> None:
    self.entered: threading.Event = threading.Event()
    self._released: threading.Event = threading.Event()
    self._lock: threading.Lock = threading.Lock()
    self._async_waiters: list[
        tuple[asyncio.AbstractEventLoop, asyncio.Event]
    ] = []

  async def wait_async(self, *, shield: bool = False) -> None:
    """Waits asynchronously for the gate to be released."""
    loop = asyncio.get_running_loop()
    ev = asyncio.Event()
    with self._lock:
      if self._released.is_set():
        ev.set()
      self._async_waiters.append((loop, ev))
    self.entered.set()
    if shield:
      await asyncio.shield(ev.wait())
    else:
      await ev.wait()

  def wait_sync(self, timeout: float = 30.0) -> None:
    """Waits synchronously for the gate to be released."""
    self.entered.set()
    if not self._released.wait(timeout=timeout):
      raise TimeoutError(f"FaultGate not released within {timeout}s.")

  def release(self) -> None:
    """Releases the gate and wakes all registered sync and async waiters."""
    with self._lock:
      self._released.set()
      waiters = list(self._async_waiters)
    for loop, ev in waiters:
      try:
        loop.call_soon_threadsafe(ev.set)
      except RuntimeError:
        # Loop already closed.
        pass


@dataclasses.dataclass(frozen=True)
class Raise:
  exc: Exception


@dataclasses.dataclass(frozen=True)
class Block:
  gate: FaultGate
  shield: bool = False
  after_super: bool = False


@dataclasses.dataclass(frozen=True)
class Return:
  value: Any


@dataclasses.dataclass(frozen=True)
class LoseReply:
  exc: Exception


FaultAction = Raise | Block | Return | LoseReply


@dataclasses.dataclass
class FaultRule:
  worker_id: str
  method_name: str
  action: FaultAction
  where: Callable[[remote_execution.ExecutionRequest], bool] | None = None
  nth: int | None = None
  times: int | None = 1
  matched_count: int = 0
  fired_count: int = 0


class FaultInjector:
  """Thread-safe rule matcher and gate manager."""

  def __init__(self) -> None:
    self._lock: threading.Lock = threading.Lock()
    self._rules: list[FaultRule] = []

  def on(
      self,
      worker_id: str,
      method_name: str,
      action: FaultAction,
      *,
      where: Callable[[remote_execution.ExecutionRequest], bool] | None = None,
      nth: int | None = None,
      times: int | None = 1,
  ) -> FaultRule:
    """Registers a fault injection rule."""
    rule = FaultRule(
        worker_id=worker_id,
        method_name=method_name,
        action=action,
        where=where,
        nth=nth,
        times=times,
    )
    with self._lock:
      self._rules.append(rule)
    return rule

  def match(
      self, worker_id: str, request: remote_execution.ExecutionRequest
  ) -> FaultAction | None:
    target_method = request.method_name or "__call__"
    with self._lock:
      for rule in self._rules:
        if rule.worker_id != ANY and rule.worker_id != worker_id:
          continue
        if (
            rule.method_name != ANY
            and rule.method_name != target_method
            and rule.method_name != request.method_name
        ):
          continue
        if rule.where is not None and not rule.where(request):
          continue
        rule.matched_count += 1
        if rule.nth is not None:
          if rule.matched_count == rule.nth:
            rule.fired_count += 1
            return rule.action
          continue
        if rule.times is not None and rule.fired_count >= rule.times:
          continue
        rule.fired_count += 1
        return rule.action
    return None

  def release_all_gates(self) -> None:
    with self._lock:
      rules = list(self._rules)
    for rule in rules:
      if isinstance(rule.action, Block):
        rule.action.gate.release()


def _error_response(
    exc: Exception, request_id: str | None
) -> remote_execution.ExecutionResponse:
  return remote_execution.ExecutionResponse(
      error_message=str(exc),
      error_type=type(exc).__name__,
      traceback=(
          "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
          if exc.__traceback__
          else None
      ),
      request_id=request_id,
  )


class FaultInjectingServer(remote_execution.GrpcRemoteExecutionServer):
  """GrpcRemoteExecutionServer intercepting requests via FaultInjector."""

  def __init__(
      self,
      instance: Any,
      *,
      worker_id: str,
      injector: FaultInjector | None = None,
      enable_bulk_transport: bool = False,
      **kwargs: Any,
  ) -> None:
    self._grpc_server: Any = None
    self._bound_port: int = 0
    super().__init__(
        instance,
        enable_bulk_transport=enable_bulk_transport,
        **kwargs,
    )
    self._worker_id = worker_id
    self._injector = injector

  @property
  def _server(self) -> Any:
    return self._grpc_server

  @_server.setter
  def _server(self, srv: Any) -> None:
    if srv is not None and hasattr(srv, "add_insecure_port"):
      orig_add_port = srv.add_insecure_port

      def _recording_add_port(address: str) -> int:
        bound = int(orig_add_port(address))
        self._bound_port = bound
        return bound

      srv.add_insecure_port = _recording_add_port
    self._grpc_server = srv

  @property
  def bound_port(self) -> int:
    return self._bound_port

  @property
  def worker_id(self) -> str:
    return self._worker_id

  @property
  def injector(self) -> FaultInjector | None:
    return self._injector

  async def _handle_execute(
      self, request_iterator: AsyncIterator[bytes], context: Any
  ) -> AsyncIterator[bytes]:
    token = _CURRENT_GRPC_CONTEXT.set(context)
    try:
      async for chunk in super()._handle_execute(request_iterator, context):
        yield chunk
    finally:
      _CURRENT_GRPC_CONTEXT.reset(token)

  async def execute_request(
      self, request: remote_execution.ExecutionRequest
  ) -> remote_execution.ExecutionResponse:
    if self._injector is not None:
      action = self._injector.match(self._worker_id, request)
      if action is not None:
        return await self._execute_fault_action(action, request)
    return await super().execute_request(request)

  async def _execute_fault_action(
      self,
      action: FaultAction,
      request: remote_execution.ExecutionRequest,
  ) -> remote_execution.ExecutionResponse:
    if isinstance(action, Raise):
      return _error_response(action.exc, request.request_id)
    elif isinstance(action, Block):
      if not action.after_super:
        await action.gate.wait_async(shield=action.shield)
        return await super().execute_request(request)
      else:
        response = await super().execute_request(request)
        await action.gate.wait_async(shield=action.shield)
        return response
    elif isinstance(action, Return):
      return remote_execution.ExecutionResponse(
          result=action.value,
          request_id=request.request_id,
      )
    elif isinstance(action, LoseReply):
      _ = await super().execute_request(request)
      context = _CURRENT_GRPC_CONTEXT.get()
      if context is not None:
        await context.abort(grpc.StatusCode.UNAVAILABLE, str(action.exc))
      return _error_response(action.exc, request.request_id)
    raise TypeError(f"Unsupported fault action: {action!r}")


class RunningWorkerServer:
  """Manages a dedicated daemon thread running its own asyncio event loop for a FaultInjectingServer."""

  def __init__(
      self,
      server: FaultInjectingServer,
      port: int = 0,
  ) -> None:
    self.server = server
    self.port = port
    self._loop: asyncio.AbstractEventLoop | None = None
    self._thread: threading.Thread | None = None
    self._ready_event = threading.Event()
    self._startup_error: BaseException | None = None
    self._bound_port: int = 0

  @property
  def bound_port(self) -> int:
    return self._bound_port

  def start(self) -> int:
    """Starts the server thread and returns the bound port."""
    if self._thread is not None:
      return self._bound_port

    self._ready_event.clear()
    self._startup_error = None

    def _runner() -> None:
      loop = asyncio.new_event_loop()
      self._loop = loop
      asyncio.set_event_loop(loop)
      try:
        loop.run_until_complete(self.server.start_serving_async(self.port))
        self._bound_port = self.server.bound_port or self.port
        loop.call_soon(self._ready_event.set)
      except BaseException as exc:  # pylint: disable=broad-exception-caught
        self._startup_error = exc
        self._ready_event.set()
        loop.close()
        return
      try:
        loop.run_forever()
      finally:
        try:
          pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
          for task in pending:
            task.cancel()
          if pending:
            loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
        except Exception:  # pylint: disable=broad-exception-caught
          pass
        finally:
          loop.close()

    self._thread = threading.Thread(
        target=_runner,
        name=f"worker-server-{self.server.worker_id}-{self.port}",
        daemon=True,
    )
    self._thread.start()
    if not self._ready_event.wait(timeout=10.0):
      raise TimeoutError(
          f"Worker server {self.server.worker_id} failed to start within 10s."
      )
    if self._startup_error is not None:
      raise self._startup_error
    return self._bound_port

  def stop(self, grace: float = 0.0) -> None:
    """Idempotently stops the server and terminates its thread."""
    thread = self._thread
    if thread is None:
      return
    loop = self._loop
    if loop is not None and loop.is_running():
      try:
        fut = asyncio.run_coroutine_threadsafe(
            self.server.stop_serving(grace=grace), loop
        )
        fut.result(timeout=5.0)
      except Exception:  # pylint: disable=broad-exception-caught
        pass
      loop.call_soon_threadsafe(loop.stop)
    if thread.is_alive():
      thread.join(timeout=10.0)
    self._thread = None
    self._loop = None

  def __enter__(self) -> "RunningWorkerServer":
    self.start()
    return self

  def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
    self.stop()
