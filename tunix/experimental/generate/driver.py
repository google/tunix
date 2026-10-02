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
from typing import Any

from absl import logging
import jaxtyping
from tunix.experimental.generate import engine as engine_lib
from tunix.experimental.generate import request as request_lib
from tunix.experimental.rollout import sampler as sampler_lib

# Resolves to the output of the finished request.
RequestFuture = futures.Future[request_lib.RequestOutput]


class VanillaInProcessDriver:
  """Runs an `LLMEngine` on a background thread and resolves request futures.

  An error the engine raises kills the loop: every pending future fails with
  it, and later submissions raise.
  """

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
    # submission, cancellation and weight updates never interleave with a
    # step.
    self._engine_lock = threading.Lock()
    self._work_event = threading.Event()
    self._stop_event = threading.Event()
    self._loop_thread: threading.Thread | None = None

    # The future of every request submitted and not yet resolved, by id.
    self._pending: dict[str, RequestFuture] = {}
    # Requests submitted but not yet handed to the engine, in submission order.
    self._submission_queue: list[sampler_lib.SamplingRequest] = []
    # Monotonic timestamp of the first request in the current submission
    # window, used to flush a partial batch. Reset on each drain.
    self._submission_window_start: float | None = None
    self._last_error: Exception | None = None

  @property
  def engine(self) -> engine_lib.LLMEngine:
    return self._engine

  @property
  def last_error(self) -> Exception | None:
    """The error that killed the loop, if it died."""
    return self._last_error

  # ----------- Submission -----------
  def submit_request(
      self, request: sampler_lib.SamplingRequest
  ) -> RequestFuture:
    """Submits a request for sampling.

    The engine checks the request once the loop hands it over. A request it
    rejects kills the loop, like any other engine error.

    Args:
      request: The request to sample.

    Returns:
      A future resolving to the request's output.

    Raises:
      RuntimeError: If the loop died.
      ValueError: If a request with the same id is still pending.
    """
    with self._engine_lock:
      future = self._queue_request_locked(request)
      self._work_event.set()
    return future

  def submit_requests(
      self, requests: Sequence[sampler_lib.SamplingRequest]
  ) -> list[RequestFuture]:
    """Submits a batch of requests for sampling.

    Submitting as a batch keeps the requests in one submission window, so they
    reach the engine together and share a continuous batch.

    Args:
      requests: The requests to sample.

    Returns:
      One future per request, in the order the requests were given.

    Raises:
      RuntimeError: If the loop died.
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
      self, request: sampler_lib.SamplingRequest
  ) -> None:
    if self._last_error is not None:
      raise RuntimeError(
          'The driver loop died; no more requests can be submitted.'
      ) from self._last_error
    if request.request_id in self._pending:
      raise ValueError(f'Request {request.request_id} is still pending.')

  def _queue_request_locked(
      self, request: sampler_lib.SamplingRequest
  ) -> RequestFuture:
    """Parks a request until the loop hands it to the engine."""
    self._check_submittable_locked(request)
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

  # ----------- Lifecycle -----------
  def start(self) -> None:
    """Starts the loop thread. A no-op if it is already running."""
    if self._loop_thread is not None and self._loop_thread.is_alive():
      return
    self._stop_event.clear()
    self._loop_thread = threading.Thread(
        target=self._loop, name='VanillaInProcessDriverLoop', daemon=True
    )
    self._loop_thread.start()

  def cancel(self, request_id: str) -> None:
    """Withdraws a request and cancels its future.

    Has no effect if the request is unknown or already resolved.

    Args:
      request_id: The id of the request to withdraw.
    """
    with self._engine_lock:
      if request_id not in self._pending:
        return
      future = self._pending.pop(request_id)
      queued = [
          request
          for request in self._submission_queue
          if request.request_id == request_id
      ]
      if queued:
        # Not handed to the engine yet, so it only has to leave the queue.
        self._submission_queue = [
            request
            for request in self._submission_queue
            if request.request_id != request_id
        ]
        if not self._submission_queue:
          self._submission_window_start = None
      else:
        self._engine.abort_request(request_id)
    future.cancel()

  def update_params(
      self,
      updated_weights: jaxtyping.PyTree,
      filter_types: tuple[Any, ...] | None = None,
  ) -> None:
    """Syncs new weights into the model the engine samples from.

    The update takes the engine lock, so it lands between steps rather than
    underneath one.

    Args:
      updated_weights: The new weights, either all of the model's parameters
        or its LoRA parameters alone.
      filter_types: The variable types to update. All of them when None.
    """
    with self._engine_lock:
      self._engine.update_params(updated_weights, filter_types)

  def reset_kv_caches(self) -> None:
    """Discards the engine's KV between steps, for after the weights change.

    In-flight requests re-prefill, as they do after `update_params`.
    """
    with self._engine_lock:
      self._engine.reset_kv_caches()

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
    self._fail_pending(RuntimeError('Driver shut down.'))

  # ----------- Loop -----------
  def _loop(self) -> None:
    try:
      while not self._stop_event.is_set():
        if not self._wait_for_work():
          continue
        finished = self._step_engine()
        for future, output in finished:
          future.set_result(output)
        if not finished:
          time.sleep(self._poll_interval_s)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      self._record_error(exc)

  def _has_work_locked(self) -> bool:
    return (
        self._submission_queue_ready_locked()
        or self._engine.has_unfinished_requests()
    )

  def _wait_for_work(self) -> bool:
    """Blocks until there is something to do, or the driver is stopping."""
    while not self._stop_event.is_set():
      with self._engine_lock:
        if self._has_work_locked():
          return True
        self._work_event.clear()
      # A queued but not yet ready window still needs waking up once its
      # submission timeout elapses, so never wait longer than the poll
      # interval.
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
      # The futures are claimed in the same critical section as the step, so
      # every output has a pending future.
      return [
          (self._pending.pop(output.request_id), output)
          for output in self._engine.step()
      ]

  def _record_error(self, exc: Exception) -> None:
    logging.exception('VanillaInProcessDriver loop died.')
    with self._engine_lock:
      self._last_error = exc
    self._fail_pending(exc)

  def _fail_pending(self, exc: Exception) -> None:
    with self._engine_lock:
      pending = list(self._pending.values())
      self._pending.clear()
      self._submission_queue = []
      self._submission_window_start = None
    for future in pending:
      future.set_exception(exc)

  def __enter__(self) -> 'VanillaInProcessDriver':
    self.start()
    return self

  def __exit__(self, exc_type, exc, tb) -> None:
    self.shutdown()
