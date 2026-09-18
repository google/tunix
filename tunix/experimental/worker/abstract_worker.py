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

"""Worker abstractions for role-based isolation.

Defines the base interface for all RL pipeline workers. The Worker is the unit
that the Orchestrator talks to, and is a wrapper around the core logic of the
pipeline (e.g. TrainerWorker is a wrapper around the Trainer).
"""

import abc
from collections.abc import Iterator
import contextlib
import contextvars
import functools
import inspect
from typing import Any, Callable

from tunix.experimental.common import datatypes

_ACTIVE_WORKERS: contextvars.ContextVar[frozenset[int]] = (
    contextvars.ContextVar("active_workers", default=frozenset())
)


def _wrap_with_execution_context(fn: Callable[..., Any]) -> Callable[..., Any]:
  """Wraps a worker method to execute within `self.execution_context()`."""
  if inspect.iscoroutinefunction(fn):

    @functools.wraps(fn)
    async def _async_wrapper(self: "Worker", *args: Any, **kwargs: Any) -> Any:
      with self.execution_context():
        return await fn(self, *args, **kwargs)

    return _async_wrapper

  if inspect.isasyncgenfunction(fn):

    @functools.wraps(fn)
    async def _async_gen_wrapper(
        self: "Worker", *args: Any, **kwargs: Any
    ) -> Any:
      with self.execution_context():
        async for item in fn(self, *args, **kwargs):
          yield item

    return _async_gen_wrapper

  if inspect.isgeneratorfunction(fn):

    @functools.wraps(fn)
    def _gen_wrapper(self: "Worker", *args: Any, **kwargs: Any) -> Any:
      with self.execution_context():
        yield from fn(self, *args, **kwargs)

    return _gen_wrapper

  @functools.wraps(fn)
  def _sync_wrapper(self: "Worker", *args: Any, **kwargs: Any) -> Any:
    with self.execution_context():
      return fn(self, *args, **kwargs)

  return _sync_wrapper


@contextlib.contextmanager
def _enter_single(ctx: Any) -> Iterator[None]:
  """Enters a single context manager or zero-arg factory using a native with statement."""
  cm: Any
  if (
      not isinstance(ctx, type)
      and hasattr(ctx, "__enter__")
      and hasattr(ctx, "__exit__")
  ):
    cm = ctx
  elif callable(ctx):
    fn: Any = ctx
    cm = fn()
  else:
    raise TypeError(
        "execution_context must be None, a context manager, a zero-arg callable"
        " returning a context manager, or a sequence thereof; got"
        f" {type(ctx).__name__}."
    )
  with cm:
    yield


class Worker(abc.ABC):
  """Base interface for all Workers."""

  # Default initial state for a worker
  _state: datatypes.WorkerState = datatypes.WorkerState.PENDING
  _execution_context: Any = None

  def __init_subclass__(cls, **kwargs: Any) -> None:
    super().__init_subclass__(**kwargs)
    for name, attr in list(cls.__dict__.items()):
      if name.startswith("_") or name in ("execution_context", "state"):
        continue
      if inspect.isfunction(attr):
        setattr(cls, name, _wrap_with_execution_context(attr))

  @contextlib.contextmanager
  def execution_context(self) -> Iterator[None]:
    """Enters the worker's configured execution context (re-entrant per task/thread)."""
    ctx = getattr(self, "_execution_context", None)
    if not ctx:
      yield
      return

    active = _ACTIVE_WORKERS.get()
    worker_id = id(self)
    if worker_id in active:
      yield
      return

    items = ctx if isinstance(ctx, (list, tuple)) else (ctx,)
    with contextlib.ExitStack() as stack:
      for item in items:
        if item is not None:
          stack.enter_context(_enter_single(item))
      token = _ACTIVE_WORKERS.set(active | {worker_id})
      try:
        yield
      finally:
        _ACTIVE_WORKERS.reset(token)

  @property
  def state(self) -> datatypes.WorkerState:
    """Gets the current lifecycle state of the worker."""
    return self._state

  @state.setter
  def state(self, new_state: datatypes.WorkerState) -> None:
    """Sets a new state, validating the transition."""
    if self._state == new_state:
      return
    if self._state and not self._state.can_transition_to(new_state):
      current_name = (
          self._state.name if hasattr(self._state, "name") else self._state
      )
      target_name = new_state.name if hasattr(new_state, "name") else new_state
      raise RuntimeError(
          f"Invalid transition from {current_name} to {target_name}"
      )
    self._state = new_state

  @abc.abstractmethod
  def initialize(self) -> datatypes.Response:
    """Initializes the worker.

    Allocates memory, loads model weights, and sets up mesh/sharding
    constraints, etc.
    """
    pass

  @abc.abstractmethod
  def compile(self, dummy_data: Any) -> datatypes.Response:
    """Triggers JIT compilation using the provided dummy_data."""
    pass

  @abc.abstractmethod
  def start(self) -> datatypes.Response:
    """Starts the worker's main loop."""
    pass

  @abc.abstractmethod
  def stop(self) -> datatypes.Response:
    """Gracefully stops the worker."""
    pass

  @abc.abstractmethod
  def info(self) -> datatypes.WorkerInfo:
    """Returns the worker's identification and roles."""
    pass

  @abc.abstractmethod
  def heartbeat(self) -> datatypes.HealthReport:
    """Returns the current health status of the worker."""
    pass
