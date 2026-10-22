# Copyright 2025 Google LLC
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

"""Single-process driver for the vLLM V1 EngineCore.

This driver keeps the EngineCore inside the current process and runs the
continuous batching loop on a Python thread. It is intended for TPU setups
where multiprocessing is undesirable (e.g. JAX integration).
"""

from __future__ import annotations

from concurrent.futures import Future
import os
import threading
import time
from typing import Any, Callable, Dict, Optional, Sequence, Union

from absl import logging
from vllm import envs
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput
from vllm.outputs import RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.llm_engine import LLMEngine

# Ensure multiprocessing is disabled before the engine is constructed.
if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
  os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
envs.VLLM_ENABLE_V1_MULTIPROCESSING = False

StreamCallback = Callable[[Union[RequestOutput, PoolingRequestOutput]], None]
RequestFuture = Future[Union[RequestOutput, PoolingRequestOutput]]
QueuedRequest = dict[str, Any]


class VLLMInProcessDriver:
  """Runs a V1 LLMEngine in-process and polls for finished outputs."""

  def __init__(
      self,
      llm_engine: LLMEngine,
      *,
      poll_interval_s: float = 0.004,
      submission_threshold: int = 0,
      submission_timeout_s: float = 0.0,
      log_stats_interval_s: float = 10.0,
      stream_callback: Optional[StreamCallback] = None,
      auto_start: bool = True,
  ) -> None:
    self._llm_engine = llm_engine
    self._poll_interval_s = poll_interval_s
    self._submission_threshold = submission_threshold
    self._submission_timeout_s = submission_timeout_s
    self._stream_callback = stream_callback

    if self._submission_threshold < 0:
      raise ValueError("submission_threshold must be >= 0.")
    if self._submission_timeout_s < 0:
      raise ValueError("submission_timeout_s must be >= 0.")

    self._engine_lock = threading.Lock()
    self._work_event = threading.Event()
    self._stop_event = threading.Event()
    self._loop_thread: Optional[threading.Thread] = None
    self._log_thread: Optional[threading.Thread] = None
    self._log_stats_interval_s: float = log_stats_interval_s

    self._pending: Dict[str, RequestFuture] = {}
    self._submission_queue: list[QueuedRequest] = []
    # Monotonic (``time.perf_counter``) timestamp of the first request in the
    # current submission window; used to flush a partial batch once
    # ``submission_timeout_s`` elapses. Reset to ``None`` on each drain.
    self._submission_window_start: Optional[float] = None
    self._last_error: Optional[Exception] = None

    self._pause_mode: Optional[str] = None
    self._unpaused_event = threading.Event()
    self._unpaused_event.set()
    self._step_idle_event = threading.Event()
    self._step_idle_event.set()
    self._drained_event = threading.Event()
    self._drained_event.set()

    if auto_start:
      self.start()

  @classmethod
  def from_engine_args(
      cls,
      engine_args: EngineArgs,
      *,
      usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
      poll_interval_s: float = 0.005,
      submission_threshold: int = 0,
      submission_timeout_s: float = 0.0,
      log_stats_interval_s: float = 1.0,
      stream_callback: Optional[StreamCallback] = None,
      auto_start: bool = True,
  ) -> "VLLMInProcessDriver":
    logging.debug(
        f"Creating VLLMInProcessDriver with engine_args: {engine_args} and"
        f" usage_context: {usage_context}"
    )
    llm_engine = LLMEngine.from_engine_args(
        engine_args,
        usage_context=usage_context,
        enable_multiprocessing=False,
    )
    return cls(
        llm_engine,
        poll_interval_s=poll_interval_s,
        submission_threshold=submission_threshold,
        submission_timeout_s=submission_timeout_s,
        stream_callback=stream_callback,
        auto_start=auto_start,
    )

  def _submission_queue_ready_locked(self) -> bool:
    if not self._submission_queue:
      return False
    if self._submission_threshold == 0:
      return True
    if len(self._submission_queue) >= self._submission_threshold:
      return True
    # Flush a partial batch if the submission timeout has elapsed since the
    # first request of the current window arrived. Without this, fewer than
    # ``submission_threshold`` requests would stall in the queue indefinitely.
    if (
        self._submission_timeout_s > 0
        and self._submission_window_start is not None
        and time.perf_counter() - self._submission_window_start
        >= self._submission_timeout_s
    ):
      return True
    return False

  @property
  def llm_engine(self) -> LLMEngine:
    return self._llm_engine

  def submit_request(
      self,
      request_id: str,
      prompt: Union[EngineCoreRequest, PromptType],
      params: Union[SamplingParams, PoolingParams],
      *,
      arrival_time: Optional[float] = None,
      lora_request: Optional[LoRARequest] = None,
      tokenization_kwargs: Optional[dict[str, Any]] = None,
      trace_headers: Optional[dict[str, str]] = None,
      priority: int = 0,
  ) -> RequestFuture:
    with self._engine_lock:
      future = self._queue_request_locked(
          request_id=request_id,
          prompt=prompt,
          params=params,
          arrival_time=arrival_time,
          lora_request=lora_request,
          tokenization_kwargs=tokenization_kwargs,
          trace_headers=trace_headers,
          priority=priority,
      )
      self._work_event.set()
    return future

  def submit_requests(
      self,
      requests: Sequence[QueuedRequest],
  ) -> list[RequestFuture]:
    futures: list[RequestFuture] = []
    with self._engine_lock:
      for request in requests:
        futures.append(
            self._queue_request_locked(
                request_id=request["request_id"],
                prompt=request["prompt"],
                params=request["params"],
                arrival_time=request.get("arrival_time"),
                lora_request=request.get("lora_request"),
                tokenization_kwargs=request.get("tokenization_kwargs"),
                trace_headers=request.get("trace_headers"),
                priority=request.get("priority", 0),
            )
        )

      if futures:
        self._work_event.set()
    return futures

  def _queue_request_locked(
      self,
      *,
      request_id: str,
      prompt: Union[EngineCoreRequest, PromptType],
      params: Union[SamplingParams, PoolingParams],
      arrival_time: Optional[float] = None,
      lora_request: Optional[LoRARequest] = None,
      tokenization_kwargs: Optional[dict[str, Any]] = None,
      trace_headers: Optional[dict[str, str]] = None,
      priority: int = 0,
  ) -> RequestFuture:
    if request_id in self._pending:
      raise ValueError(f"Request {request_id} already pending.")

    future: RequestFuture = Future()
    self._pending[request_id] = future
    if self._submission_window_start is None:
      # Start the flush-timeout clock from the first request of this window.
      self._submission_window_start = time.perf_counter()
    self._submission_queue.append({
        "request_id": request_id,
        "prompt": prompt,
        "params": params,
        "arrival_time": arrival_time,
        "lora_request": lora_request,
        "tokenization_kwargs": tokenization_kwargs,
        "trace_headers": trace_headers,
        "priority": priority,
    })
    logging.debug(
        "VLLMInProcessDriver queued request %s for loop-side submission.",
        request_id,
    )
    return future

  def _drain_submission_queue_locked(self, *, force: bool = False) -> None:
    """Submits queued requests to the underlying vLLM engine under lock."""
    if not self._submission_queue:
      return
    if not force and not self._submission_queue_ready_locked():
      return

    queued_requests = self._submission_queue
    self._submission_queue = []
    self._submission_window_start = None
    for request in queued_requests:
      future = self._pending.get(request["request_id"])
      if future is None or future.cancelled():
        continue
      logging.debug(
          "VLLMInProcessDriver submitting queued request %s to vLLM engine.",
          request["request_id"],
      )
      self._llm_engine.add_request(
          request_id=request["request_id"],
          prompt=request["prompt"],
          params=request["params"],
          arrival_time=request.get("arrival_time"),
          lora_request=request.get("lora_request"),
          tokenization_kwargs=request.get("tokenization_kwargs"),
          trace_headers=request.get("trace_headers"),
          priority=request.get("priority", 0),
      )

  def start(self) -> None:
    if self._loop_thread and self._loop_thread.is_alive():
      return
    self._stop_event.clear()
    self._loop_thread = threading.Thread(
        target=self._loop, name="VLLMInProcessDriverLoop", daemon=True
    )
    self._loop_thread.start()

    self._log_thread = threading.Thread(
        target=self._log_loop, name="VLLMLogStats", daemon=True
    )
    self._log_thread.start()

  def _log_loop(self) -> None:
    while not self._stop_event.is_set():
      try:
        self._llm_engine.do_log_stats()
      except Exception:  # pylint: disable=broad-exception-caught
        logging.exception("log_stats failed")
      self._stop_event.wait(self._log_stats_interval_s)

  def cancel(self, request_id: str) -> None:
    with self._engine_lock:
      future = self._pending.pop(request_id, None)
      if future is not None and not future.done():
        future.cancel()
      self._llm_engine.abort_request([request_id])
      if not self._llm_engine.has_unfinished_requests():
        self._work_event.clear()

  def shutdown(self) -> None:
    self.stop()
    with self._engine_lock:
      pending = list(self._pending.values())
      self._pending.clear()
    for future in pending:
      if not future.done():
        future.set_exception(RuntimeError("Driver shut down."))
    with self._engine_lock:
      self._llm_engine.engine_core.shutdown()

  def stop(self) -> None:
    self._stop_event.set()
    self._unpaused_event.set()
    self._drained_event.set()
    self._work_event.set()
    if self._loop_thread is not None:
      self._loop_thread.join()
      self._loop_thread = None
    if self._log_thread is not None:
      self._log_thread.join()
      self._log_thread = None

  @property
  def is_paused(self) -> bool:
    with self._engine_lock:
      return self._pause_mode is not None

  @property
  def pause_mode(self) -> Optional[str]:
    with self._engine_lock:
      return self._pause_mode

  def _apply_pause_engine_effects_locked(
      self, mode: str, clear_cache: bool
  ) -> None:
    """Applies scheduler pause and optional cache resets on the engine."""
    engine_core = getattr(self._llm_engine, "engine_core", None)
    if engine_core is not None and hasattr(engine_core, "pause_scheduler"):
      try:
        engine_core.pause_scheduler(mode=mode, clear_cache=clear_cache)
      except TypeError:
        engine_core.pause_scheduler()
    if clear_cache:
      if hasattr(self._llm_engine, "reset_prefix_cache"):
        self._llm_engine.reset_prefix_cache()
      if engine_core is not None and hasattr(
          engine_core, "reset_encoder_cache"
      ):
        engine_core.reset_encoder_cache()

  def pause(
      self,
      mode: str = "keep",
      *,
      clear_cache: bool = False,
      timeout_s: float = 300.0,
  ) -> None:
    """Pauses the in-process vLLM loop.

    Args:
      mode: Pause mode:
        - "keep": Freeze scheduling at the next step boundary while keeping
          in-flight requests and their KV blocks resident in the engine.
        - "wait": Flush staged submissions and wait for all in-flight requests
          to complete before freezing the loop.
        - "abort": Freeze the loop and abort all queued and in-flight requests.
      clear_cache: Whether to clear the engine prefix/encoder caches once
        paused. Defaults to False so active KV blocks remain valid in "keep"
        mode.
      timeout_s: Maximum seconds to wait for in-flight step / drain completion.

    Raises:
      ValueError: If `mode` is not one of "keep", "wait", or "abort".
      TimeoutError: If waiting for the active step or drain exceeds `timeout_s`.
    """
    if mode not in ("keep", "wait", "abort"):
      raise ValueError(
          f"Unsupported pause mode: {mode!r}. Expected 'keep', 'wait', or"
          " 'abort'."
      )

    if mode == "keep":
      with self._engine_lock:
        self._pause_mode = "keep"
        self._unpaused_event.clear()
        self._work_event.set()
      if not self._step_idle_event.wait(timeout=timeout_s):
        raise TimeoutError(
            "Timed out waiting for in-flight vLLM step to complete during"
            " pause(mode='keep')."
        )
      with self._engine_lock:
        self._apply_pause_engine_effects_locked(mode, clear_cache)
      return

    if mode == "wait":
      with self._engine_lock:
        self._drain_submission_queue_locked(force=True)
        self._pause_mode = "wait"
        if self._llm_engine.has_unfinished_requests():
          self._drained_event.clear()
          self._unpaused_event.set()
          self._work_event.set()
        else:
          self._drained_event.set()
          self._unpaused_event.clear()
          self._work_event.set()
      if not self._drained_event.wait(timeout=timeout_s):
        raise TimeoutError(
            "Timed out waiting for in-flight vLLM requests to drain during"
            " pause(mode='wait')."
        )
      if not self._step_idle_event.wait(timeout=timeout_s):
        raise TimeoutError(
            "Timed out waiting for vLLM loop step to become idle during"
            " pause(mode='wait')."
        )
      with self._engine_lock:
        self._apply_pause_engine_effects_locked(mode, clear_cache)
      return

    # mode == "abort"
    with self._engine_lock:
      self._pause_mode = "abort"
      self._unpaused_event.clear()
      self._work_event.set()
    if not self._step_idle_event.wait(timeout=timeout_s):
      raise TimeoutError(
          "Timed out waiting for in-flight vLLM step to complete during"
          " pause(mode='abort')."
      )
    with self._engine_lock:
      aborted_ids = list(self._pending.keys())
      pending_futures = list(self._pending.values())
      self._submission_queue.clear()
      self._submission_window_start = None
      self._pending.clear()
      if aborted_ids:
        self._llm_engine.abort_request(aborted_ids)
      self._work_event.clear()
      self._apply_pause_engine_effects_locked(mode, clear_cache)
    for future in pending_futures:
      if not future.done():
        future.cancel()

  def resume(self) -> None:
    """Resumes the in-process vLLM loop after a pause."""
    with self._engine_lock:
      engine_core = getattr(self._llm_engine, "engine_core", None)
      if engine_core is not None and hasattr(engine_core, "resume_scheduler"):
        engine_core.resume_scheduler()
      self._pause_mode = None
      self._drained_event.set()
      self._unpaused_event.set()
      if self._submission_queue or self._llm_engine.has_unfinished_requests():
        self._work_event.set()

  def _loop(self) -> None:
    try:
      while not self._stop_event.is_set():
        if not self._unpaused_event.is_set():
          self._unpaused_event.wait(timeout=self._poll_interval_s)
          continue
        if not self._wait_for_work():
          continue
        with self._engine_lock:
          if not self._unpaused_event.is_set():
            continue
          self._step_idle_event.clear()
        outputs = []
        try:
          outputs = self._step_engine()
          logging.log_every_n(
              logging.DEBUG,
              "VLLMInProcessDriver loop step outputs:"
              f" {[output.request_id for output in outputs]}",
              40,
          )
          if outputs:
            for output in outputs:
              self._handle_output(output)
          with self._engine_lock:
            if (
                self._pause_mode == "wait"
                and not self._llm_engine.has_unfinished_requests()
            ):
              self._unpaused_event.clear()
              self._drained_event.set()
        finally:
          self._step_idle_event.set()
        if not outputs:
          time.sleep(self._poll_interval_s)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      self._record_error(exc)

  def _wait_for_work(self) -> bool:
    """Waits until there is engine work to process or the driver is paused/stopped."""
    while not self._stop_event.is_set():
      if not self._unpaused_event.is_set():
        return False
      with self._engine_lock:
        if not self._unpaused_event.is_set():
          return False
        if self._pause_mode == "wait":
          has_work = self._llm_engine.has_unfinished_requests()
          if not has_work:
            self._unpaused_event.clear()
            self._drained_event.set()
            return False
        else:
          has_work = self._submission_queue_ready_locked()
          if not has_work:
            has_work = self._llm_engine.has_unfinished_requests()
        if has_work:
          return True
        self._work_event.clear()

      self._work_event.wait(timeout=self._poll_interval_s)
    return False

  def _step_engine(
      self,
  ) -> list[Union[RequestOutput, PoolingRequestOutput]]:
    logging.log_every_n(
        logging.DEBUG,
        f"VLLMInProcessDriver loop waking up to process one step of requests.",
        100,
    )
    with self._engine_lock:
      if self._pause_mode != "wait":
        self._drain_submission_queue_locked()
      logging.log_every_n(
          logging.DEBUG,
          "VLLMInProcessDriver has"
          f" {self._llm_engine.get_num_unfinished_requests()} pending"
          " requests.",
          100,
      )
      if self._llm_engine.has_unfinished_requests():
        return self._llm_engine.step()
      return []

  def _handle_output(
      self, output: Union[RequestOutput, PoolingRequestOutput]
  ) -> None:
    if not output.finished:
      callback = self._stream_callback
      if callback is not None:
        callback(output)
      return
    with self._engine_lock:
      future = self._pending.pop(output.request_id, None)
    if future is None or future.done():
      return
    logging.debug(
        f"VLLMInProcessDriver completed request id: {output.request_id}."
    )
    future.set_result(output)

  def _record_error(self, exc: Exception) -> None:
    logging.debug("VLLMInProcessDriver encountered an error: %s", exc)
    self._last_error = exc
    with self._engine_lock:
      pending = list(self._pending.values())
      self._pending.clear()
    for future in pending:
      if not future.done():
        future.set_exception(exc)

  @property
  def llm_engine(self) -> LLMEngine:
    return self._llm_engine

  @property
  def last_error(self) -> Optional[Exception]:
    return self._last_error

  def __enter__(self) -> "VLLMInProcessDriver":
    return self

  def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001, ANN201
    self.shutdown()
