# Copyright 2026 The Tunix Authors.
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

"""In-process driver for the continuous batching engine.

The driver decouples request submission from engine execution: callers submit
requests and get a future back, while a background thread owns the engine and
runs its continuous batching loop until each request finishes.
"""

from collections.abc import Sequence
from concurrent import futures
import threading
import time

from tunix.experimental.generate import engine as engine_lib
from tunix.experimental.generate import request as request_lib

# Resolves to the output of the finished request.
RequestFuture = futures.Future[request_lib.RequestOutput]


class VanillaInProcessDriver:
  """Runs an `LLMEngine` on a background thread and resolves request futures."""

  def __init__(
      self,
      engine: engine_lib.LLMEngine,
      poll_interval_s: float = 0.004,
  ):
    """Initializes the driver.

    Args:
      engine: The engine to drive. The driver owns it once constructed; calling
        into the engine directly races with the loop thread.
      poll_interval_s: How long the loop waits for new work before looking
        again.

    Raises:
      ValueError: If `poll_interval_s` is not positive.
    """
    if poll_interval_s <= 0:
      raise ValueError(f'poll_interval_s must be > 0. Got {poll_interval_s}.')

    self._poll_interval_s = poll_interval_s
    self._engine = engine

    # Serializes every touch of the engine and of the state below, so that
    # submission never interleaves with a step.
    self._engine_lock = threading.Lock()
    self._work_event = threading.Event()
    self._stop_event = threading.Event()
    self._loop_thread: threading.Thread | None = None

    # The future of every request submitted and not yet resolved, by id.
    self._pending: dict[str, RequestFuture] = {}
    # Requests submitted but not yet handed to the engine, in submission order.
    self._submission_queue: list[request_lib.Request] = []

  @property
  def engine(self) -> engine_lib.LLMEngine:
    return self._engine

  def submit_request(
      self, request: request_lib.Request
  ) -> RequestFuture:
    """Submits a request for sampling.

    Args:
      request: The request to sample.

    Returns:
      A future resolving to the request's output.
    """
    with self._engine_lock:
      future = self._queue_request_locked(request)
      self._work_event.set()
    return future

  def submit_requests(
      self, requests: Sequence[request_lib.Request]
  ) -> list[RequestFuture]:
    """Submits a batch of requests for sampling.

    Args:
      requests: The requests to sample.

    Returns:
      One future per request, in the order the requests were given.

    Raises:
      ValueError: If two requests share an id, or one shares the id of a
        request that is still pending. No request is submitted then.
    """
    with self._engine_lock:
      request_ids = [request.request_id for request in requests]
      if len(set(request_ids)) != len(request_ids):
        raise ValueError(f'Request ids must be unique, got {request_ids}.')
      for request in requests:
        self._check_submittable_locked(request)
      future_list = [
          self._queue_request_locked(request) for request in requests
      ]
      if future_list:
        self._work_event.set()
    return future_list

  def _check_submittable_locked(
      self, request: request_lib.Request
  ) -> None:
    if request.request_id in self._pending:
      raise ValueError(f'Request {request.request_id} is still pending.')

  def _queue_request_locked(
      self, request: request_lib.Request
  ) -> RequestFuture:
    """Parks a request until the loop hands it to the engine."""
    self._check_submittable_locked(request)
    future: RequestFuture = futures.Future()
    self._pending[request.request_id] = future
    self._submission_queue.append(request)
    return future

  def _drain_submission_queue_locked(self) -> None:
    queued_requests = self._submission_queue
    self._submission_queue = []
    for request in queued_requests:
      try:
        self._engine.add_request(request)
      except Exception as e:  # pylint: disable=broad-exception-caught
        future = self._pending.pop(request.request_id, None)
        if future is not None:
          future.set_exception(e)

  def start(self) -> None:
    """Starts the loop thread. A no-op if it is already running."""
    if self._loop_thread is not None and self._loop_thread.is_alive():
      return
    self._stop_event.clear()
    self._loop_thread = threading.Thread(
        target=self._loop, name='VanillaInProcessDriverLoop', daemon=True
    )
    self._loop_thread.start()

  def stop(self) -> None:
    """Stops the loop thread, leaving pending futures untouched."""
    self._stop_event.set()
    self._work_event.set()
    if self._loop_thread is not None:
      self._loop_thread.join()
      self._loop_thread = None

  def shutdown(self) -> None:
    """Stops the loop thread and fails every request still pending."""
    self.stop()
    with self._engine_lock:
      self._submission_queue.clear()
      pending = list(self._pending.items())
      self._pending.clear()
      for request_id, _ in pending:
        self._engine.abort_request(request_id)
    for _, future in pending:
      future.set_exception(
          RuntimeError('Driver shut down before request finished.')
      )

  def _loop(self) -> None:
    while not self._stop_event.is_set():
      if not self._wait_for_work():
        continue

      finished = self._step_engine()
      for future, output in finished:
        future.set_result(output)

      if not finished:
        time.sleep(self._poll_interval_s)

  def _has_work_locked(self) -> bool:
    return bool(self._submission_queue) or self._engine.has_unfinished_requests()

  def _wait_for_work(self) -> bool:
    """Blocks until there is something to do, or the driver is stopping."""
    while not self._stop_event.is_set():
      with self._engine_lock:
        if self._has_work_locked():
          return True
        self._work_event.clear()

      self._work_event.wait(timeout=self._poll_interval_s)
    return False

  def _step_engine(
      self,
  ) -> list[tuple[RequestFuture, request_lib.RequestOutput]]:
    """Runs one engine step, returning each finished request's future."""
    with self._engine_lock:
      self._drain_submission_queue_locked()
      if not self._engine.has_unfinished_requests():
        return []

      return [
          (self._pending.pop(output.request_id), output)
          for output in self._engine.step()
      ]

  def __enter__(self) -> 'VanillaInProcessDriver':
    self.start()
    return self

  def __exit__(self, exc_type, exc, tb) -> None:
    self.shutdown()
