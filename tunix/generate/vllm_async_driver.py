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
from typing import Any, Callable, Dict, Optional, Union

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


class VLLMInProcessDriver:
  """Runs a V1 LLMEngine in-process and polls for finished outputs."""

  def __init__(
      self,
      llm_engine: LLMEngine,
      *,
      poll_interval_s: float = 0.005,
      stream_callback: Optional[StreamCallback] = None,
      auto_start: bool = True,
  ) -> None:
    self._llm_engine = llm_engine
    self._poll_interval_s = poll_interval_s
    self._stream_callback = stream_callback

    self._engine_lock = threading.Lock()
    self._work_event = threading.Event()
    self._stop_event = threading.Event()
    self._loop_thread: Optional[threading.Thread] = None

    self._pending: Dict[str, RequestFuture] = {}
    self._last_error: Optional[BaseException] = None

    if auto_start:
      self.start()

  @classmethod
  def from_engine_args(
      cls,
      engine_args: EngineArgs,
      *,
      usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
      poll_interval_s: float = 0.005,
      stream_callback: Optional[StreamCallback] = None,
      auto_start: bool = True,
  ) -> "VLLMInProcessDriver":
    llm_engine = LLMEngine.from_engine_args(
        engine_args,
        usage_context=usage_context,
        enable_multiprocessing=False,
    )
    return cls(
        llm_engine,
        poll_interval_s=poll_interval_s,
        stream_callback=stream_callback,
        auto_start=auto_start,
    )

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
    future: RequestFuture = Future()
    with self._engine_lock:
      if request_id in self._pending:
        raise ValueError(f"Request {request_id} already pending.")
      self._pending[request_id] = future
      self._llm_engine.add_request(
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

  def start(self) -> None:
    if self._loop_thread and self._loop_thread.is_alive():
      return
    self._stop_event.clear()
    self._loop_thread = threading.Thread(
        target=self._loop, name="VLLMInProcessDriverLoop", daemon=False
    )
    self._loop_thread.start()

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
    self._work_event.set()
    if self._loop_thread is not None:
      self._loop_thread.join()
      self._loop_thread = None

  def pause(self) -> None:
    raise RuntimeError("Pause feature WIP")

  def resume(self) -> None:
    raise RuntimeError("Resume feature WIP")

  def _loop(self) -> None:
    try:
      while not self._stop_event.is_set():
        if not self._wait_for_work():
          continue
        outputs = self._step_engine()
        if outputs:
          for output in outputs:
            self._handle_output(output)
        else:
          time.sleep(self._poll_interval_s)
    except BaseException as exc:  # pylint: disable=broad-exception-caught
      self._record_error(exc)

  def _wait_for_work(self) -> bool:
    with self._engine_lock:
      has_work = self._llm_engine.has_unfinished_requests()
      if has_work:
        return True
      self._work_event.clear()

    self._work_event.wait(timeout=self._poll_interval_s)
    return not self._stop_event.is_set()

  def _step_engine(
      self,
  ) -> list[Union[RequestOutput, PoolingRequestOutput]]:
    with self._engine_lock:
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
    future.set_result(output)

  def _record_error(self, exc: BaseException) -> None:
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
  def last_error(self) -> Optional[BaseException]:
    return self._last_error

  def __enter__(self) -> "VLLMInProcessDriver":
    return self

  def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001, ANN201
    self.shutdown()
