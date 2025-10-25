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


if __name__ == "__main__":
  absltest.main()
