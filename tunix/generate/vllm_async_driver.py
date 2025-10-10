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

"""Async driver built on top of vLLM's AsyncLLM.

The driver keeps the engine core in this process but relies on AsyncLLM's async
request/streaming APIs to avoid a CPU polling loop.  A dedicated asyncio event
loop runs in the background thread; requests are submitted via
``AsyncLLM.add_request`` and completions are awaited from
``RequestOutputCollector``.  Callers interact with the driver through the same
``concurrent.futures.Future`` interface as the previous implementation.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future
from dataclasses import dataclass
import os
import threading
from typing import Any, Callable, Dict, Optional, Tuple, Union

from vllm.engine.arg_utils import AsyncEngineArgs
import vllm.envs as envs
from vllm.inputs import PromptType
from vllm.lora.request import LoRARequest
from vllm.outputs import PoolingRequestOutput
from vllm.outputs import RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.async_llm import AsyncLLM

# Ensure multiprocessing is disabled before the engine is constructed.
if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") != "0":
  os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
envs.VLLM_ENABLE_V1_MULTIPROCESSING = False

StreamCallback = Callable[[Union[RequestOutput, PoolingRequestOutput]], None]
RequestFuture = Future


@dataclass
class _PendingRequest:
  future: RequestFuture
  task: Future


class VLLMAsyncDriver:
  """Async driver that delegates to vLLM's AsyncLLM."""

  def __init__(
      self,
      async_llm: AsyncLLM,
      *,
      stream_callback: Optional[StreamCallback] = None,
  ) -> None:
    self._async_llm = async_llm
    self._stream_callback = stream_callback

    self._loop = asyncio.new_event_loop()
    self._loop_thread = threading.Thread(
        target=self._loop_runner, name="VLLMAsyncDriverLoop", daemon=False
    )
    self._loop_thread.start()

    # Kick off cache reset and output handler inside the event loop.
    asyncio.run_coroutine_threadsafe(
        self._async_llm.reset_mm_cache(), self._loop
    ).result()
    self._loop.call_soon_threadsafe(self._async_llm._run_output_handler)

    self._state_lock = threading.Lock()
    self._pending: Dict[str, _PendingRequest] = {}
    self._last_error: Optional[BaseException] = None

  @classmethod
  def from_engine_args(
      cls,
      engine_args: AsyncEngineArgs,
      *,
      usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
      stream_callback: Optional[StreamCallback] = None,
  ) -> "VLLMAsyncDriver":
    async_llm = AsyncLLM.from_engine_args(
        engine_args,
        usage_context=usage_context,
        start_engine_loop=False,
    )
    return cls(async_llm, stream_callback=stream_callback)

  def _loop_runner(self) -> None:
    asyncio.set_event_loop(self._loop)
    self._loop.run_forever()

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
      data_parallel_rank: Optional[int] = None,
      prompt_text: Optional[str] = None,
  ) -> RequestFuture:
    future: RequestFuture = Future()

    async def _submit() -> None:
      try:
        queue = await self._async_llm.add_request(
            request_id=request_id,
            prompt=prompt,
            params=params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            tokenization_kwargs=tokenization_kwargs,
            trace_headers=trace_headers,
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            prompt_text=prompt_text,
        )
        await self._consume_outputs(request_id, queue, future)
      except Exception as exc:  # pylint: disable=broad-except
        self._record_error(exc)
        if not future.done():
          future.set_exception(exc)
        self._cleanup_request(request_id)
        raise

    task = asyncio.run_coroutine_threadsafe(_submit(), self._loop)
    with self._state_lock:
      if request_id in self._pending:
        task.cancel()
        raise ValueError(f"Request {request_id} already pending.")
      self._pending[request_id] = _PendingRequest(future=future, task=task)
    return future

  async def _consume_outputs(
      self,
      request_id: str,
      queue,
      future: RequestFuture,
  ) -> None:
    try:
      while True:
        output = queue.get_nowait()
        if output is None:
          output = await queue.get()

        if not output.finished:
          if self._stream_callback is not None:
            await asyncio.to_thread(self._stream_callback, output)
          continue

        if not future.done():
          future.set_result(output)
        break
    finally:
      self._cleanup_request(request_id)

  def cancel(self, request_id: str) -> None:
    with self._state_lock:
      entry = self._pending.pop(request_id, None)
    if entry is None:
      return
    future, task = entry.future, entry.task
    if not future.done():
      future.cancel()
    task.cancel()
    asyncio.run_coroutine_threadsafe(
        self._async_llm.abort(request_id), self._loop
    )

  def shutdown(self) -> None:
    pending: Tuple[_PendingRequest, ...]
    with self._state_lock:
      pending = tuple(self._pending.values())
      self._pending.clear()

    for entry in pending:
      entry.task.cancel()
      if not entry.future.done():
        entry.future.set_exception(RuntimeError("Driver shut down."))

    # Shut down AsyncLLM on the event loop thread.
    async def _shutdown() -> None:
      self._async_llm.shutdown()

    asyncio.run_coroutine_threadsafe(_shutdown(), self._loop).result()
    self._loop.call_soon_threadsafe(self._loop.stop)
    self._loop_thread.join()
    self._loop.close()

  @property
  def llm_engine(self) -> AsyncLLM:
    return self._async_llm

  @property
  def last_error(self) -> Optional[BaseException]:
    return self._last_error

  def _record_error(self, exc: BaseException) -> None:
    self._last_error = exc

  def _cleanup_request(self, request_id: str) -> None:
    with self._state_lock:
      self._pending.pop(request_id, None)

  def __enter__(self) -> "VLLMAsyncDriver":
    return self

  def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001, ANN201
    self.shutdown()


# Backwards compatibility until downstream components are updated.
VLLMInProcessDriver = VLLMAsyncDriver
