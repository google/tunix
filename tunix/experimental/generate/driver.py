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
prompts and get a future back, while a background thread owns the engine and
runs its continuous batching loop until each request finishes.
"""

from concurrent import futures
import itertools
import threading
import time
from typing import Any, Optional, Sequence, Tuple

from absl import logging
from flax import nnx
import jaxtyping
from tunix.experimental.generate import engine as engine_lib
from tunix.experimental.generate import request as request_lib


# Resolves to the finished request, which carries the generated tokens along
# with any logits and logprobs the engine was configured to return.
RequestFuture = futures.Future


class VanillaInProcessDriver:
  """Runs an `LLMEngine` on a background thread and resolves request futures."""

  def __init__(
      self,
      transformer: 'nnx.Module',
      engine: engine_lib.LLMEngine,
      poll_interval_s: float = 0.004,
      submission_threshold: int = 0,
      submission_timeout_s: float = 0.0,
  ):
    """Initializes the driver.

    Args:
      transformer: The model the engine samples from. Held so that callers can
        reach the model the driver is serving.
      engine: The engine to drive. The driver owns it once constructed; calling
        into the engine directly races with the loop thread.
      poll_interval_s: How long the loop waits for new work before looking
        again.
      submission_threshold: Only hand queued requests to the engine once this
        many have accumulated. 0 submits them as soon as they arrive.
      submission_timeout_s: Submit a partial batch anyway once this many
        seconds have elapsed since the first request of the current window
        arrived. 0 disables the timeout, so a partial batch waits indefinitely.

    Raises:
      ValueError: If a threshold or timeout is negative.
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
    self._transformer = transformer
    self._engine = engine

    # Serializes every touch of the engine, so that submission, cancellation
    # and weight updates never interleave with a step.
    self._engine_lock = threading.Lock()
    self._work_event = threading.Event()
    self._stop_event = threading.Event()
    self._loop_thread: Optional[threading.Thread] = None

    self._pending: dict[str, RequestFuture] = {}
    self._submission_queue: list[request_lib.Request] = []
    # Monotonic timestamp of the first request in the current submission
    # window, used to flush a partial batch. Reset on each drain.
    self._submission_window_start: Optional[float] = None
    self._request_ids = itertools.count()
    self._last_error: Optional[Exception] = None

  @property
  def engine(self) -> engine_lib.LLMEngine:
    return self._engine

  @property
  def transformer(self) -> nnx.Module:
    return self._transformer

  @property
  def last_error(self) -> Optional[Exception]:
    """The error that killed the loop, if it died."""
    return self._last_error

  # ----------- Submission -----------
  def submit_request(
      self,
      prompt_token_ids: Sequence[int],
      max_tokens_to_generate: int | None = None,
      eos_token_ids: Sequence[int] | set[int] | None = None,
  ) -> RequestFuture:
    """Submits a tokenized prompt for sampling.

    Args:
      prompt_token_ids: The token ids of the prompt to sample from. Callers
        tokenize their own prompts, so that the prompt length bound stays with
        the layer that owns the tokenizer.
      max_tokens_to_generate: Optional per-request token limit.
      eos_token_ids: Optional per-request EOS token IDs.

    Returns:
      A future resolving to the finished request.
    """
    with self._engine_lock:
      future = self._queue_request_locked(
          prompt_token_ids,
          max_tokens_to_generate=max_tokens_to_generate,
          eos_token_ids=eos_token_ids,
      )
      self._work_event.set()
    return future

  def submit_requests(
      self,
      prompt_token_ids: Sequence[Sequence[int]],
      max_tokens_to_generate: int | Sequence[int | None] | None = None,
      eos_token_ids: Sequence[int] | set[int] | None = None,
  ) -> list[RequestFuture]:
    """Submits a batch of tokenized prompts for sampling.

    Submitting as a batch keeps the prompts in one submission window, so they
    reach the engine together and share a continuous batch.

    Args:
      prompt_token_ids: The token ids of each prompt to sample from.
      max_tokens_to_generate: Optional per-request token limit (or scalar for
        all).
      eos_token_ids: Optional per-request EOS token IDs.

    Returns:
      One future per prompt, in the order the prompts were given.
    """
    with self._engine_lock:
      future_list = []
      for i, token_ids in enumerate(prompt_token_ids):
        if (
            isinstance(max_tokens_to_generate, int)
            or max_tokens_to_generate is None
        ):
          max_tokens = max_tokens_to_generate
        else:
          max_tokens = max_tokens_to_generate[i]
        future_list.append(
            self._queue_request_locked(
                token_ids,
                max_tokens_to_generate=max_tokens,
                eos_token_ids=eos_token_ids,
            )
        )
      if future_list:
        self._work_event.set()
    return future_list

  def _queue_request_locked(
      self,
      prompt_token_ids: Sequence[int],
      max_tokens_to_generate: int | None = None,
      eos_token_ids: Sequence[int] | set[int] | None = None,
  ) -> RequestFuture:
    """Parks a tokenized prompt until the window is ready to drain."""
    request_id = f'vanilla_{id(self)}_{next(self._request_ids)}'
    req = request_lib.Request(
        request_id,
        list(prompt_token_ids),
        max_tokens_to_generate=max_tokens_to_generate,
        eos_token_ids=eos_token_ids,
    )

    future: RequestFuture = futures.Future()
    self._pending[request_id] = future
    if self._submission_window_start is None:
      self._submission_window_start = time.perf_counter()
    self._submission_queue.append(req)
    return future

  def _submission_queue_ready_locked(self) -> bool:
    """Whether the queued requests should be handed to the engine now."""
    if not self._submission_queue:
      return False
    if self._submission_threshold == 0:
      return True
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
    for req in queued_requests:
      future = self._pending.get(req.request_id)
      if future is None or future.cancelled():
        continue
      self._engine.add_request(req)

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

    Args:
      request_id: The id of the request to withdraw, as carried by the request
        the future resolves to.
    """
    with self._engine_lock:
      future = self._pending.pop(request_id, None)
      if future is not None and not future.done():
        future.cancel()
      self._submission_queue = [
          req
          for req in self._submission_queue
          if req.request_id != request_id
      ]
      if not self._submission_queue:
        self._submission_window_start = None
      self._engine.abort_request(request_id)

  def stop(self) -> None:
    """Stops the loop thread, leaving pending futures untouched."""
    self._stop_event.set()
    self._work_event.set()
    if self._loop_thread is not None:
      self._loop_thread.join()
      self._loop_thread = None

  def shutdown(self) -> None:
    """Stops the loop thread and fails every request still in flight."""
    self.stop()
    with self._engine_lock:
      pending = list(self._pending.values())
      self._pending.clear()
      self._submission_queue = []
      self._submission_window_start = None
    for future in pending:
      if not future.done():
        future.set_exception(RuntimeError('Driver shut down.'))

  def update_params(
      self,
      updated_weights: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    """Syncs new weights into the model the engine samples from.

    The update takes the engine lock, so it lands between steps rather than
    underneath one.

    Args:
      updated_weights: The new weights.
      filter_types: The variable types to update. All of them when None.
    """
    with self._engine_lock:
      self._engine.update_params(updated_weights, filter_types)

  # ----------- Loop -----------
  def _loop(self) -> None:
    try:
      while not self._stop_event.is_set():
        if not self._wait_for_work():
          continue
        finished_requests = self._step_engine()
        if finished_requests:
          for req in finished_requests:
            self._handle_output(req)
        else:
          time.sleep(self._poll_interval_s)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      self._record_error(exc)

  def _wait_for_work(self) -> bool:
    """Blocks until there is something to do, or the driver is stopping."""
    while not self._stop_event.is_set():
      with self._engine_lock:
        has_work = (
            self._submission_queue_ready_locked()
            or self._engine.has_unfinished_requests()
        )
        if has_work:
          return True
        self._work_event.clear()

      # A queued-but-not-ready window still needs waking up once its
      # submission timeout elapses, so never wait longer than the poll
      # interval.
      self._work_event.wait(timeout=self._poll_interval_s)
    return False

  def _step_engine(self) -> list[request_lib.Request]:
    with self._engine_lock:
      self._drain_submission_queue_locked()
      if self._engine.has_unfinished_requests():
        return self._engine.step()
      return []

  def _handle_output(self, req: request_lib.Request) -> None:
    with self._engine_lock:
      future = self._pending.pop(req.request_id, None)
    if future is None or future.done():
      return
    future.set_result(req)

  def _record_error(self, exc: Exception) -> None:
    logging.exception('VanillaInProcessDriver loop died.')
    self._last_error = exc
    with self._engine_lock:
      pending = list(self._pending.values())
      self._pending.clear()
      self._submission_queue = []
      self._submission_window_start = None
    for future in pending:
      if not future.done():
        future.set_exception(exc)

  def __enter__(self) -> 'VanillaInProcessDriver':
    self.start()
    return self

  def __exit__(self, exc_type, exc, tb) -> None:
    self.shutdown()

