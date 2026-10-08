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

import importlib.metadata
import sys
import threading
import time
from typing import Iterable
from unittest import mock

try:
  import transformers  # pylint: disable=g-import-not-at-top,unused-import
except ImportError:
  pass

_orig_meta_version = importlib.metadata.version


def _version_shim(pkg: str) -> str:
  if pkg == "transformers":
    return "5.0.0"
  return _orig_meta_version(pkg)


importlib.metadata.version = _version_shim

if "openai_harmony" not in sys.modules:
  sys.modules["openai_harmony"] = mock.MagicMock()

from absl.testing import absltest  # pylint: disable=g-import-not-at-top
from tunix.generate.vllm_async_driver import VLLMInProcessDriver  # pylint: disable=g-import-not-at-top


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

  def __init__(self):
    self.paused_calls = []
    self.resumed_calls = 0
    self.encoder_cache_resets = 0

  def pause_scheduler(self, mode: str = "keep", clear_cache: bool = False):
    self.paused_calls.append((mode, clear_cache))

  def resume_scheduler(self):
    self.resumed_calls += 1

  def reset_encoder_cache(self):
    self.encoder_cache_resets += 1

  def shutdown(self):
    pass


class _FakeLLMEngine:
  """Minimal synchronous engine that emits completions in a fixed order."""

  def __init__(
      self,
      completion_order: Iterable[str],
      steps_per_request: dict[str, int] | None = None,
  ):
    self._completion_order = list(completion_order)
    self._pending: list[str] = []
    self._remaining_steps: dict[str, int] = dict(steps_per_request or {})
    self._lock = threading.Lock()
    self.engine_core = _StubEngineCore()
    self.log_called = threading.Event()
    self.step_count = 0
    self.aborted_requests: list[str] = []
    self.prefix_cache_resets = 0
    self.step_entered = threading.Event()
    self.allow_step_proceed: threading.Event | None = None

  # The driver only exercises a subset of the LLMEngine surface.
  def add_request(self, request_id: str, *_, **__):
    with self._lock:
      self._pending.append(request_id)
      self._remaining_steps.setdefault(request_id, 1)

  def has_unfinished_requests(self) -> bool:
    with self._lock:
      return bool(self._pending)

  def get_num_unfinished_requests(self) -> int:
    with self._lock:
      return len(self._pending)

  def step(self):
    self.step_entered.set()
    if self.allow_step_proceed is not None:
      self.allow_step_proceed.wait(timeout=5.0)
    with self._lock:
      self.step_count += 1
      if not self._completion_order or not self._pending:
        return []

      next_request = self._completion_order[0]
      if next_request not in self._pending:
        # Wait for the next request in the completion order to arrive.
        time.sleep(0.001)
        return []

      rem = self._remaining_steps.get(next_request, 1) - 1
      self._remaining_steps[next_request] = rem
      if rem > 0:
        time.sleep(0.001)
        out = _DummyRequestOutput(next_request)
        out.finished = False
        return [out]

      self._completion_order.pop(0)
      self._pending.remove(next_request)

    return [_DummyRequestOutput(next_request)]

  def abort_request(self, request_ids, *_args, **_kwargs):
    with self._lock:
      for req_id in request_ids:
        self.aborted_requests.append(req_id)
        if req_id in self._pending:
          self._pending.remove(req_id)
        if req_id in self._completion_order:
          self._completion_order.remove(req_id)

  def reset_prefix_cache(self):
    self.prefix_cache_resets += 1

  # Log stats API exercised by the driver's log thread.
  def do_log_stats(self):
    # Signal that the log stats method was called.
    self.log_called.set()
    return None


class VllmDriverAsyncTest(absltest.TestCase):

  def test_requests_are_staged_until_loop_starts(self):
    engine = _FakeLLMEngine(["req-0", "req-1"])
    driver = VLLMInProcessDriver(llm_engine=engine, auto_start=False)
    self.addCleanup(driver.shutdown)

    future_0 = driver.submit_request(
        request_id="req-0",
        prompt={"prompt_token_ids": [1]},
        params=object(),
    )
    future_1 = driver.submit_request(
        request_id="req-1",
        prompt={"prompt_token_ids": [1]},
        params=object(),
    )

    self.assertEmpty(engine._pending)
    self.assertFalse(future_0.done())
    self.assertFalse(future_1.done())

    driver.start()

    self.assertEqual(future_0.result(timeout=5.0).request_id, "req-0")
    self.assertEqual(future_1.result(timeout=5.0).request_id, "req-1")

  def test_submission_threshold_delays_queue_drain(self):
    engine = _FakeLLMEngine(["req-0", "req-1"])
    driver = VLLMInProcessDriver(
        llm_engine=engine,
        submission_threshold=2,
        poll_interval_s=0.001,
        auto_start=True,
    )
    self.addCleanup(driver.shutdown)

    future_0 = driver.submit_request(
        request_id="req-0",
        prompt={"prompt_token_ids": [1]},
        params=object(),
    )

    time.sleep(0.01)
    self.assertEmpty(engine._pending)
    self.assertFalse(future_0.done())

    future_1 = driver.submit_request(
        request_id="req-1",
        prompt={"prompt_token_ids": [1]},
        params=object(),
    )

    self.assertEqual(future_0.result(timeout=5.0).request_id, "req-0")
    self.assertEqual(future_1.result(timeout=5.0).request_id, "req-1")

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

  def test_log_thread_calls_do_log_stats(self):
    engine = _FakeLLMEngine([])
    driver = VLLMInProcessDriver(
        llm_engine=engine, log_stats_interval_s=0.01, auto_start=True
    )
    self.addCleanup(driver.shutdown)

    # Wait for the log thread to call into the engine's do_log_stats.
    self.assertTrue(engine.log_called.wait(timeout=1.0))

  def test_submission_timeout_flushes_partial_batch(self):
    # Threshold of 2 but only 1 request arrives: the flush timeout must still
    # submit it instead of letting it hang in the queue forever.
    engine = _FakeLLMEngine(["req-0"])
    timeout_s = 0.05
    driver = VLLMInProcessDriver(
        llm_engine=engine,
        submission_threshold=2,
        submission_timeout_s=timeout_s,
        poll_interval_s=0.001,
        auto_start=True,
    )
    self.addCleanup(driver.shutdown)

    start = time.perf_counter()
    future_0 = driver.submit_request(
        request_id="req-0",
        prompt={"prompt_token_ids": [1]},
        params=object(),
    )

    result = future_0.result(timeout=5.0)
    elapsed = time.perf_counter() - start
    self.assertEqual(result.request_id, "req-0")
    # It flushed because of the timeout (threshold of 2 was never reached), so
    # the request could not have completed before the timeout elapsed.
    self.assertGreaterEqual(elapsed, timeout_s * 0.8)

  def test_submission_timeout_disabled_holds_partial_batch(self):
    # timeout == 0 disables the flush: below-threshold requests stay queued
    # (identical to the pre-existing behavior).
    engine = _FakeLLMEngine(["req-0"])
    driver = VLLMInProcessDriver(
        llm_engine=engine,
        submission_threshold=2,
        submission_timeout_s=0.0,
        poll_interval_s=0.001,
        auto_start=True,
    )
    self.addCleanup(driver.shutdown)

    future_0 = driver.submit_request(
        request_id="req-0",
        prompt={"prompt_token_ids": [1]},
        params=object(),
    )

    time.sleep(0.05)
    self.assertEmpty(engine._pending)
    self.assertFalse(future_0.done())

  def test_submission_timeout_counts_from_first_request(self):
    # The clock starts at the FIRST request of the window. With threshold=3
    # never reached, the partial batch must flush ~timeout after req-0, even
    # though req-1 arrives later (well after req-0 but before the deadline).
    engine = _FakeLLMEngine(["req-0", "req-1"])
    timeout_s = 0.1
    driver = VLLMInProcessDriver(
        llm_engine=engine,
        submission_threshold=3,
        submission_timeout_s=timeout_s,
        poll_interval_s=0.001,
        auto_start=True,
    )
    self.addCleanup(driver.shutdown)

    start = time.perf_counter()
    future_0 = driver.submit_request(
        request_id="req-0",
        prompt={"prompt_token_ids": [1]},
        params=object(),
    )
    time.sleep(timeout_s * 0.6)
    future_1 = driver.submit_request(
        request_id="req-1",
        prompt={"prompt_token_ids": [1]},
        params=object(),
    )

    self.assertEqual(future_0.result(timeout=5.0).request_id, "req-0")
    self.assertEqual(future_1.result(timeout=5.0).request_id, "req-1")
    elapsed = time.perf_counter() - start
    # If the clock had (incorrectly) restarted at req-1, the flush would land at
    # ~0.6*timeout + timeout; assert it fired well before that.
    self.assertLess(elapsed, timeout_s * 1.5)

  def test_negative_submission_timeout_raises(self):
    engine = _FakeLLMEngine([])
    with self.assertRaises(ValueError):
      VLLMInProcessDriver(
          llm_engine=engine,
          submission_timeout_s=-1.0,
          auto_start=False,
      )

  def test_pause_and_resume_freezes_and_resumes_in_place(self):
    engine = _FakeLLMEngine(["req-0", "req-1"])
    driver = VLLMInProcessDriver(
        llm_engine=engine,
        poll_interval_s=0.001,
        auto_start=True,
    )
    self.addCleanup(driver.shutdown)

    driver.pause()
    self.assertTrue(driver.is_paused)

    future_0 = driver.submit_request(
        request_id="req-0",
        prompt={"prompt_token_ids": [1]},
        params=object(),
    )
    future_1 = driver.submit_request(
        request_id="req-1",
        prompt={"prompt_token_ids": [1]},
        params=object(),
    )

    time.sleep(0.03)
    self.assertFalse(future_0.done())
    self.assertFalse(future_1.done())

    driver.resume()
    self.assertFalse(driver.is_paused)

    self.assertEqual(future_0.result(timeout=5.0).request_id, "req-0")
    self.assertEqual(future_1.result(timeout=5.0).request_id, "req-1")

  def test_pause_keep_freezes_in_flight_and_resumes_cleanly(self):
    engine = _FakeLLMEngine(
        ["req-0", "req-1"],
        steps_per_request={"req-0": 3, "req-1": 1},
    )
    engine.allow_step_proceed = threading.Event()
    driver = VLLMInProcessDriver(
        llm_engine=engine, poll_interval_s=0.001, auto_start=True
    )
    self.addCleanup(driver.shutdown)

    future_0 = driver.submit_request(
        request_id="req-0",
        prompt={"prompt_token_ids": [1]},
        params=object(),
    )
    self.assertTrue(engine.step_entered.wait(timeout=2.0))

    # Release the first step and immediately pause(mode="keep") in a thread
    # or after letting step 1 finish.
    pause_done = threading.Event()

    def _do_pause():
      driver.pause(mode="keep", clear_cache=False)
      pause_done.set()

    pause_thread = threading.Thread(target=_do_pause)
    pause_thread.start()
    time.sleep(0.01)
    # While the first step is blocked inside engine.step(), pause must wait.
    self.assertFalse(pause_done.is_set())
    engine.allow_step_proceed.set()
    pause_thread.join(timeout=2.0)
    self.assertTrue(pause_done.is_set())
    self.assertTrue(driver.is_paused)
    self.assertEqual(driver.pause_mode, "keep")
    self.assertEqual(engine.prefix_cache_resets, 0)
    self.assertEqual(engine.engine_core.paused_calls, [("keep", False)])

    # req-0 is still in-flight (took 1 of 3 steps), and step count is frozen.
    steps_at_pause = engine.step_count
    self.assertFalse(future_0.done())
    self.assertIn("req-0", engine._pending)

    # Submitting req-1 while paused stages it without stepping the engine.
    future_1 = driver.submit_request(
        request_id="req-1",
        prompt={"prompt_token_ids": [2]},
        params=object(),
    )
    time.sleep(0.02)
    self.assertEqual(engine.step_count, steps_at_pause)
    self.assertFalse(future_0.done())
    self.assertFalse(future_1.done())

    # Resume and verify both req-0 and req-1 complete.
    driver.resume()
    self.assertFalse(driver.is_paused)
    self.assertIsNone(driver.pause_mode)
    self.assertEqual(engine.engine_core.resumed_calls, 1)
    self.assertEqual(future_0.result(timeout=5.0).request_id, "req-0")
    self.assertEqual(future_1.result(timeout=5.0).request_id, "req-1")

  def test_pause_wait_drains_in_flight_before_returning(self):
    engine = _FakeLLMEngine(
        ["req-0", "req-1"],
        steps_per_request={"req-0": 4, "req-1": 2},
    )
    driver = VLLMInProcessDriver(
        llm_engine=engine,
        # Even below threshold, wait flushes staged requests.
        submission_threshold=4,
        poll_interval_s=0.001,
        auto_start=True,
    )
    self.addCleanup(driver.shutdown)

    future_0 = driver.submit_request(
        request_id="req-0",
        prompt={"prompt_token_ids": [1]},
        params=object(),
    )
    future_1 = driver.submit_request(
        request_id="req-1",
        prompt={"prompt_token_ids": [2]},
        params=object(),
    )

    driver.pause(mode="wait", clear_cache=True)
    self.assertTrue(driver.is_paused)
    self.assertEqual(driver.pause_mode, "wait")
    self.assertTrue(future_0.done())
    self.assertTrue(future_1.done())
    self.assertEqual(future_0.result().request_id, "req-0")
    self.assertEqual(future_1.result().request_id, "req-1")
    self.assertEqual(engine.prefix_cache_resets, 1)
    self.assertEqual(engine.engine_core.encoder_cache_resets, 1)

  def test_pause_abort_cancels_in_flight_and_queued_requests(self):
    engine = _FakeLLMEngine(
        ["req-0", "req-1"],
        steps_per_request={"req-0": 100, "req-1": 100},
    )
    driver = VLLMInProcessDriver(
        llm_engine=engine, poll_interval_s=0.001, auto_start=True
    )
    self.addCleanup(driver.shutdown)

    future_0 = driver.submit_request(
        request_id="req-0",
        prompt={"prompt_token_ids": [1]},
        params=object(),
    )
    self.assertTrue(engine.step_entered.wait(timeout=2.0))
    future_1 = driver.submit_request(
        request_id="req-1",
        prompt={"prompt_token_ids": [2]},
        params=object(),
    )

    driver.pause(mode="abort", clear_cache=True)
    self.assertTrue(driver.is_paused)
    self.assertTrue(future_0.cancelled())
    self.assertTrue(future_1.cancelled())
    self.assertCountEqual(engine.aborted_requests, ["req-0", "req-1"])
    self.assertFalse(engine.has_unfinished_requests())

    # After resume(), new requests succeed normally.
    engine._completion_order.append("req-2")
    driver.resume()
    future_2 = driver.submit_request(
        request_id="req-2",
        prompt={"prompt_token_ids": [3]},
        params=object(),
    )
    self.assertEqual(future_2.result(timeout=5.0).request_id, "req-2")

  def test_invalid_pause_mode_raises(self):
    engine = _FakeLLMEngine([])
    driver = VLLMInProcessDriver(llm_engine=engine, auto_start=False)
    self.addCleanup(driver.shutdown)
    with self.assertRaises(ValueError):
      driver.pause(mode="invalid")


if __name__ == "__main__":
  absltest.main()

