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

"""Continuous Batching Asynchronous In-Process Driver for Tunix."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Sequence

from tunix.generate import continuous_sampler


class VanillaInProcessDriver:
  """Thread-safe Asynchronous Continuous Batching Driver matching VLLMInProcessDriver behavior."""

  def __init__(
      self,
      sampler: continuous_sampler.VanillaSampler,
      sampling_config: continuous_sampler.SamplingConfig,
      poll_interval_s: float = 0.005,
      submission_threshold: int = 0,
      submission_timeout_s: float = 0.0,
  ):
    self.sampler = sampler
    self.sampling_config = sampling_config
    self._poll_interval_s = poll_interval_s
    self._submission_threshold = submission_threshold
    self._submission_timeout_s = submission_timeout_s

    self._engine_lock = threading.Lock()
    self._work_event = threading.Event()
    self._stop_event = threading.Event()
    self._loop_thread: threading.Thread | None = None

    self._pending: Dict[str, continuous_sampler.RequestFuture] = {}
    self._submission_queue: list[dict[str, Any]] = []
    self._submission_window_start: float | None = None
    self._last_error: Exception | None = None

    self._sampling_state = self.sampler.init_sample_state(sampling_config)

  def _submission_queue_ready_locked(self) -> bool:
    """Returns true if the submission queue is ready to be processed."""
    if not self._submission_queue:
      return False
    if self._submission_threshold == 0:
      return True
    if len(self._submission_queue) >= self._submission_threshold:
      return True
    if (
        self._submission_timeout_s > 0
        and self._submission_window_start is not None
        and time.perf_counter() - self._submission_window_start
        >= self._submission_timeout_s
    ):
      return True
    return False

  def submit_request(
      self,
      request_id: str,
      prompt: str,
      **kwargs,
  ) -> continuous_sampler.RequestFuture:
    return self.submit_requests([{"id": request_id, "prompt": prompt, **kwargs}])[0]

  def submit_requests(
      self,
      requests: Sequence[dict[str, Any]],
  ) -> list[continuous_sampler.RequestFuture]:
    futures: list[continuous_sampler.RequestFuture] = []
    with self._engine_lock:
      for req_dict in requests:
        futures.append(
            self._queue_request_locked(
                request_id=req_dict["id"],
                req_dict=req_dict,
            )
        )
      if futures:
        self._work_event.set()
    return futures

  def _queue_request_locked(
      self, request_id: str, req_dict: dict[str, Any]
  ) -> continuous_sampler.RequestFuture:
    if request_id in self._pending:
      raise ValueError(f"Request {request_id} already pending.")

    future = continuous_sampler.RequestFuture(request_id)
    self._pending[request_id] = future
    if self._submission_window_start is None:
      self._submission_window_start = time.perf_counter()
    self._submission_queue.append(req_dict)
    return future

  def _drain_submission_queue_locked(self) -> list[dict[str, Any]]:
    if not self._submission_queue_ready_locked():
      return []
    queued_requests = self._submission_queue
    self._submission_queue = []
    self._submission_window_start = None
    return queued_requests

  def start(self) -> None:
    if self._loop_thread and self._loop_thread.is_alive():
      return
    self._stop_event.clear()
    self._loop_thread = threading.Thread(
        target=self._loop, name="VanillaInProcessDriverLoop", daemon=True
    )
    self._loop_thread.start()

  def stop(self) -> None:
    self._stop_event.set()
    self._work_event.set()
    if self._loop_thread is not None:
      self._loop_thread.join()
      self._loop_thread = None

  def shutdown(self) -> None:
    self.stop()
    with self._engine_lock:
      pending = list(self._pending.values())
      self._pending.clear()
    for future in pending:
      future.set_error(RuntimeError("Driver shut down."))

  def cancel(self, request_id: str) -> None:
    """Cancel a pending or active request and decrement group refcounts."""
    with self._engine_lock:
      self._submission_queue = [r for r in self._submission_queue if r["id"] != request_id]
      future = self._pending.pop(request_id, None)
      if future is not None:
        future.set_error(RuntimeError(f"Request {request_id} cancelled."))
      self._sampling_state = self.sampler.cancel_request(self._sampling_state, request_id)
      if not self._submission_queue and len(self._sampling_state.hbm_request_ids) == 0:
        self._work_event.clear()

  def _wait_for_work(self) -> bool:
    while not self._stop_event.is_set():
      with self._engine_lock:
        has_work = self._submission_queue_ready_locked()
        if not has_work:
          has_work = len(self._sampling_state.hbm_request_ids) > 0
      if has_work:
        return True
      self._work_event.wait(timeout=self._poll_interval_s)
      self._work_event.clear()
    return False

  def _step_engine(self) -> list[continuous_sampler.RequestOutput]:
    with self._engine_lock:
      new_requests = self._drain_submission_queue_locked()

    completed_map = self.sampler._sample_step(self._sampling_state, new_requests)
    outputs = list(completed_map.values())
    for out in outputs:
      self._handle_output(out)
    return outputs

  def _handle_output(self, output: continuous_sampler.RequestOutput) -> None:
    with self._engine_lock:
      fut = self._pending.pop(output.request_id, None)
      if fut:
        fut.set_result(output)

  def _loop(self) -> None:
    try:
      while not self._stop_event.is_set():
        if not self._wait_for_work():
          continue
        self._step_engine()
    except Exception as exc:
      self._last_error = exc
