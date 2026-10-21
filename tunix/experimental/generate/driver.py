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
RequestFuture = futures.Future[request_lib.EngineResponse]


class VanillaInProcessDriver:
  """Runs an `LLMEngine` on a background thread and resolves request futures."""

  def __init__(
      self,
      engine: engine_lib.LLMEngine,
      poll_interval_s: float = 0.004,
      submission_threshold: int = 0,
      submission_timeout_s: float = 0.0,
  ):
    """Initializes the driver.

    Args:
      engine: The engine to drive. The driver owns it once constructed; calling
        into the engine directly races with the loop thread.
      poll_interval_s: How long the loop waits for new work before looking
        again.
      submission_threshold: Only hand queued requests to the engine once this
        many have accumulated. 0 submits them as soon as they arrive.
      submission_timeout_s: Submit a partial batch anyway once this many
        seconds have elapsed since the first request of the current window
        arrived. 0 disables the timeout, so a partial batch waits until the
        threshold is reached.

    Raises:
      ValueError: If `poll_interval_s` is not positive, or the threshold or
        timeout is negative.
    """
    if poll_interval_s <= 0:
      raise ValueError(f'poll_interval_s must be > 0. Got {poll_interval_s}.')
    if submission_threshold < 0:
      raise ValueError(
          f'submission_threshold must be >= 0. Got {submission_threshold}.'
      )
    if submission_timeout_s < 0:
      raise ValueError(
          f'submission_timeout_s must be >= 0. Got {submission_timeout_s}.'
      )

    self._poll_interval_s = poll_interval_s
    self._submission_threshold = submission_threshold
    self._submission_timeout_s = submission_timeout_s
    self._engine = engine

    # Serializes every touch of the engine and of the state below, so that
    # submission never interleaves with a step.
    self._engine_lock = threading.Lock()
    # Protects `_pending` and the submission queue so `submit_request(s)` can
    # enqueue without waiting for an in-flight `engine.step()`. Acquired after
    # `_engine_lock` when both are held.
    self._queue_lock = threading.Lock()
    self._work_event = threading.Event()
    self._stop_event = threading.Event()
    self._loop_thread: threading.Thread | None = None

    # The future of every request submitted and not yet resolved, by id.
    self._pending: dict[str, RequestFuture] = {}
    # Requests submitted but not yet handed to the engine, in submission order.
    self._submission_queue: list[request_lib.EngineRequest] = []
    # Monotonic timestamp of the first request in the current submission
    # window, used to flush a partial batch. Reset on each drain.
    self._submission_window_start: float | None = None

  @property
  def engine(self) -> engine_lib.LLMEngine:
    return self._engine

  def submit_request(
      self, request: request_lib.EngineRequest
  ) -> RequestFuture:
    """Submits a request for sampling.

    Args:
      request: The request to sample.

    Returns:
      A future resolving to the request's output.
    """
    with self._queue_lock:
      future = self._queue_request_locked(request)
      if self._submission_queue_ready_locked():
        self._work_event.set()
    return future

  def submit_requests(
      self, requests: Sequence[request_lib.EngineRequest]
  ) -> list[RequestFuture]:
    """Submits a batch of requests for sampling.

    Submitting as a batch keeps the requests in one submission window, so they
    reach the engine together and share a continuous batch.

    Args:
      requests: The requests to sample.

    Returns:
      One future per request, in the order the requests were given.

    Raises:
      ValueError: If two requests share an id, or one shares the id of a
        request that is still pending. No request is submitted then.
    """
    with self._queue_lock:
      request_ids = [request.request_id for request in requests]
      if len(set(request_ids)) != len(request_ids):
        raise ValueError(f'Request ids must be unique, got {request_ids}.')
      for request in requests:
        if request.request_id in self._pending:
          raise ValueError(f'Request {request.request_id} is still pending.')

      future_list = [
          self._queue_request_locked(request) for request in requests
      ]

      if self._submission_queue_ready_locked():
        self._work_event.set()

    return future_list

  def _queue_request_locked(
      self, request: request_lib.EngineRequest
  ) -> RequestFuture:
    """Adds a request to the submission queue."""
    if request.request_id in self._pending:
      raise ValueError(f'Request {request.request_id} is still pending.')
    future: RequestFuture = futures.Future()
    self._pending[request.request_id] = future
    if self._submission_window_start is None:
      self._submission_window_start = time.perf_counter()
    self._submission_queue.append(request)
    return future

  def _submission_queue_ready_locked(self) -> bool:
    """Whether the queued requests should be handed to the engine now."""
    if not self._submission_queue:
      return False
    if len(self._submission_queue) >= self._submission_threshold:
      return True

    # Without the timeout, a window that never reaches the threshold would
    # stall in the queue indefinitely.
    return (
        self._submission_timeout_s > 0
        and self._submission_window_start is not None
        and time.perf_counter() - self._submission_window_start
        >= self._submission_timeout_s
    )

  def _drain_submission_queue_locked(self) -> None:
    if not self._submission_queue_ready_locked():
      return
    queued_requests = self._submission_queue
    self._submission_queue = []
    self._submission_window_start = None
    for request in queued_requests:
      self._engine.add_request(request)

  def start(self) -> None:
    """Starts the loop thread. A no-op if it is already running."""
    if self._loop_thread is not None and self._loop_thread.is_alive():
      if not self._stop_event.is_set():
        return
      if self._loop_thread is not threading.current_thread():
        self._loop_thread.join()
    self._stop_event.clear()
    self._loop_thread = threading.Thread(
        target=self._loop, name='VanillaInProcessDriverLoop', daemon=True
    )
    self._loop_thread.start()

  def stop(self) -> None:
    """Stops the loop thread, leaving pending futures untouched."""
    self._stop_event.set()
    self._work_event.set()
    if (
        self._loop_thread is not None
        and self._loop_thread is not threading.current_thread()
    ):
      self._loop_thread.join()
      self._loop_thread = None

  def shutdown(self) -> None:
    """Stops the loop thread and fails every request still pending."""
    self.stop()
    with self._engine_lock, self._queue_lock:
      self._submission_queue.clear()
      self._submission_window_start = None
      pending = list(self._pending.items())
      self._pending.clear()
      for request_id, _ in pending:
        self._engine.abort_request(request_id)
    for _, future in pending:
      if not future.cancelled():
        future.set_exception(
            RuntimeError('Driver shut down before request finished.')
        )

  def _loop(self) -> None:
    try:
      while not self._stop_event.is_set():
        if not self._wait_for_work():
          continue

        for future, output in self._step_engine():
          if not future.cancelled():
            future.set_result(output)
    except Exception as e:  # pylint: disable=broad-exception-caught
      with self._queue_lock:
        self._submission_queue.clear()
        self._submission_window_start = None
        pending = list(self._pending.values())
        self._pending.clear()
      for future in pending:
        if not future.cancelled():
          future.set_exception(e)
      raise

  def _has_work_locked(self) -> bool:
    return (
        self._submission_queue_ready_locked()
        or self._engine.has_unfinished_requests()
    )

  def _wait_for_work(self) -> bool:
    """Blocks until there is something to do, or the driver is stopping."""
    while not self._stop_event.is_set():
      with self._engine_lock, self._queue_lock:
        if self._has_work_locked():
          return True
        self._work_event.clear()

      self._work_event.wait(timeout=self._poll_interval_s)
    return False

  def _step_engine(
      self,
  ) -> list[tuple[RequestFuture, request_lib.EngineResponse]]:
    """Runs one engine step, returning each finished request's future."""
    with self._engine_lock:
      with self._queue_lock:
        self._drain_submission_queue_locked()
      if not self._engine.has_unfinished_requests():
        return []
      outputs = self._engine.step()
      with self._queue_lock:
        finished = []
        for output in outputs:
          future = self._pending.pop(output.request_id, None)
          if future is not None:
            finished.append((future, output))
        return finished

  def __enter__(self) -> 'VanillaInProcessDriver':
    self.start()
    return self

  def __exit__(self, exc_type, exc, tb) -> None:
    self.shutdown()
