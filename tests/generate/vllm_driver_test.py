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

import asyncio
import functools
import threading
import time
from typing import Iterable

from absl.testing import absltest
from tunix.generate.vllm_async_driver import VLLMInProcessDriver


# TODO(b/453660461): Add extensive concurrency tests.


class _DummyCompletionOutput:

  def __init__(self, request_id: str):
    self.token_ids = [request_id]
    self.logprobs = [0.0]
    self.text = f"response_for_{request_id}"


class _DummyRequestOutput:

  def __init__(self, request_id: str):
    self.request_id = request_id
    self.prompt = None
    self.prompt_token_ids = [request_id]
    self.prompt_logprobs = None
    self.outputs = [_DummyCompletionOutput(request_id)]
    self.finished = True
    self.kv_transfer_params = None
    self.num_cached_tokens = 0
    self.metrics = None


class _StubEngineCore:

  def shutdown(self):
    pass


class _FakeLLMEngine:
  """Minimal synchronous engine that emits completions in a fixed order."""

  def __init__(self, completion_order: Iterable[str]):
    self._completion_order = list(completion_order)
    self._pending: list[str] = []
    self._lock = threading.Lock()
    self.engine_core = _StubEngineCore()

  # The driver only exercises a subset of the LLMEngine surface.
  def add_request(self, request_id: str, *_, **__):
    with self._lock:
      self._pending.append(request_id)

  def has_unfinished_requests(self) -> bool:
    with self._lock:
      return bool(self._pending)

  def step(self):
    with self._lock:
      if not self._completion_order or not self._pending:
        return []

      next_request = self._completion_order[0]
      if next_request not in self._pending:
        # Wait for the next request in the completion order to arrive.
        time.sleep(0.001)
        return []

      self._completion_order.pop(0)
      self._pending.remove(next_request)

    return [_DummyRequestOutput(next_request)]

  def abort_request(self, *_args, **_kwargs):
    pass


class _ControllableLLMEngine:
  """Engine that completes requests after configurable delays."""

  def __init__(self, completion_delays: dict[str, float], *, auto_release=True):
    self._completion_delays = completion_delays
    self._pending: dict[str, float] = {}
    self._lock = threading.Lock()
    self._release_event = threading.Event()
    if auto_release:
      self._release_event.set()
    self._abort_calls: list[list[str]] = []
    self.engine_core = _StubEngineCore()

  def add_request(self, request_id: str, *_, **__):
    ready_time = time.monotonic() + self._completion_delays.get(
        request_id, 0.0
    )
    with self._lock:
      self._pending[request_id] = ready_time

  def has_unfinished_requests(self) -> bool:
    with self._lock:
      return bool(self._pending)

  def step(self):
    if not self._release_event.is_set():
      time.sleep(0.001)
      return []

    now = time.monotonic()
    with self._lock:
      ready = sorted(
          (
              (ready_time, request_id)
              for request_id, ready_time in self._pending.items()
              if now >= ready_time
          )
      )
      for _, request_id in ready:
        self._pending.pop(request_id, None)
    return [_DummyRequestOutput(request_id) for _, request_id in ready]

  def abort_request(self, request_ids):
    with self._lock:
      for request_id in request_ids:
        self._pending.pop(request_id, None)
    self._abort_calls.append(list(request_ids))

  def allow_completions(self):
    self._release_event.set()

  @property
  def abort_calls(self):
    return list(self._abort_calls)


class VllmDriverAsyncTest(absltest.TestCase):

  def test_out_of_order_completions_preserved(self):
    request_ids = [f"req-{i}" for i in range(10)]
    completion_order = [
        "req-0",
        "req-3",
        "req-1",
        "req-7",
        "req-2",
        "req-9",
        "req-4",
        "req-6",
        "req-5",
        "req-8",
    ]

    engine = _FakeLLMEngine(completion_order)
    driver = VLLMInProcessDriver(llm_engine=engine, auto_start=True)
    self.addCleanup(driver.shutdown)

    finished_order: list[str] = []
    futures = []
    for request_id in request_ids:
      future = driver.submit_request(
          request_id=request_id,
          prompt={"prompt_token_ids": [1]},
          params=object(),
      )
      future.add_done_callback(
          lambda f: finished_order.append(f.result().request_id)
      )
      futures.append(future)

    results = [future.result(timeout=5.0) for future in futures]

    # Ensure all requests completed.
    self.assertCountEqual(
        [res.request_id for res in results],
        request_ids,
    )

    # All completions should be observed, but not necessarily in submit order.
    self.assertEqual(finished_order, completion_order)
    self.assertNotEqual(finished_order, request_ids)

  def test_asyncio_run_in_executor_out_of_order(self):
    request_ids = [f"req-{i}" for i in range(10)]
    # Later submissions finish earlier to enforce out-of-order completions.
    delays = {
        request_id: 0.02 * (len(request_ids) - idx)
        for idx, request_id in enumerate(request_ids)
    }
    engine = _ControllableLLMEngine(delays)
    driver = VLLMInProcessDriver(llm_engine=engine, auto_start=True)
    self.addCleanup(driver.shutdown)

    def _call_driver(request_id: str, delay: float):
      time.sleep(delay)
      future = driver.submit_request(
          request_id=request_id,
          prompt={"prompt_token_ids": [request_id]},
          params=object(),
      )
      return future.result(timeout=5.0)

    async def dispatch_requests():
      loop = asyncio.get_running_loop()
      futures = []
      future_to_idx = {}
      for idx, request_id in enumerate(request_ids):
        future = loop.run_in_executor(
            None,
            functools.partial(_call_driver, request_id, delays[request_id]),
        )
        future_to_idx[future] = idx
        futures.append(future)

      completion_order = []
      results_by_idx = {}
      for future in asyncio.as_completed(futures):
        result = await future
        idx = future_to_idx[future]
        completion_order.append(idx)
        results_by_idx[idx] = result

      ordered_results = [results_by_idx[i] for i in range(len(futures))]
      return ordered_results, completion_order

    results, completion_order = asyncio.run(dispatch_requests())

    self.assertLen(results, len(request_ids))
    for request_id, result in zip(request_ids, results):
      self.assertEqual(request_id, result.request_id)

    expected_order = list(range(len(request_ids)))
    self.assertCountEqual(completion_order, expected_order)
    self.assertNotEqual(
        completion_order,
        expected_order,
        msg=(
            "Responses returned strictly in submission order; "
            "expected out-of-order completions."
        ),
    )

  def test_cancel_request_aborts_engine_and_future(self):
    delays = {"req-0": 0.5, "req-1": 0.5}
    engine = _ControllableLLMEngine(delays, auto_release=False)
    driver = VLLMInProcessDriver(llm_engine=engine, auto_start=True)
    self.addCleanup(driver.shutdown)

    future_a = driver.submit_request(
        request_id="req-0",
        prompt={"prompt_token_ids": [0]},
        params=object(),
    )
    future_b = driver.submit_request(
        request_id="req-1",
        prompt={"prompt_token_ids": [1]},
        params=object(),
    )

    driver.cancel("req-0")
    self.assertTrue(future_a.cancelled())
    self.assertIn(["req-0"], engine.abort_calls)

    engine.allow_completions()
    result = future_b.result(timeout=5.0)
    self.assertEqual("req-1", result.request_id)

  def test_stop_mid_generation_and_restart(self):
    engine = _ControllableLLMEngine({"req-0": 0.01}, auto_release=False)
    driver = VLLMInProcessDriver(llm_engine=engine, auto_start=True)
    self.addCleanup(driver.shutdown)

    future = driver.submit_request(
        request_id="req-0",
        prompt={"prompt_token_ids": [0]},
        params=object(),
    )

    # Allow the background loop to observe the pending request, then stop.
    time.sleep(0.01)
    driver.stop()
    self.assertFalse(future.done())

    driver.start()
    engine.allow_completions()

    result = future.result(timeout=5.0)
    self.assertEqual("req-0", result.request_id)


if __name__ == "__main__":
  absltest.main()
