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
import types
from typing import Iterable

from absl.testing import absltest
from tunix.generate.vllm_async_driver import _get_vllm_scheduler
from tunix.generate.vllm_async_driver import _get_vllm_schedulers
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
    self.log_called = threading.Event()

  # The driver only exercises a subset of the LLMEngine surface.
  def add_request(self, request_id: str, *_, **__):
    with self._lock:
      self._pending.append(request_id)

  def has_unfinished_requests(self) -> bool:
    with self._lock:
      return bool(self._pending)

  def get_num_unfinished_requests(self) -> int:
    with self._lock:
      return len(self._pending)

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

  def test_driver_attaches_l3_scheduler_queue_stats_on_outputs(self):
    class _ReqObj:

      def __init__(self, request_id: str, num_preemptions: int = 0):
        self.request_id = request_id
        self.num_preemptions = num_preemptions

    class _FakeScheduler:

      def __init__(self):
        self.waiting = []
        self.skipped_waiting = []
        self.running = []

      def get_kv_cache_usage(self) -> float:
        return 0.75

    class _SchedulerBackedEngine(_FakeLLMEngine):

      def __init__(self):
        super().__init__(["req-run", "req-wait"])
        self.scheduler = _FakeScheduler()
        self._step_idx = 0

      def add_request(self, request_id: str, *_, **__):
        super().add_request(request_id)
        self.scheduler.waiting.append(_ReqObj(request_id))

      def step(self):
        self._step_idx += 1
        time.sleep(0.01)
        if self._step_idx == 1:
          # Step 1: req-run executes and finishes; req-wait stays in waiting
          self.scheduler.waiting = [_ReqObj("req-wait", num_preemptions=1)]
          self.scheduler.running = []
          self._pending.remove("req-run")
          return [_DummyRequestOutput("req-run")]
        # Step 2: req-wait executes and finishes
        self.scheduler.waiting = []
        self.scheduler.running = []
        self._pending.remove("req-wait")
        return [_DummyRequestOutput("req-wait")]

    engine = _SchedulerBackedEngine()
    driver = VLLMInProcessDriver(
        llm_engine=engine, poll_interval_s=0.001, auto_start=True
    )
    self.addCleanup(driver.shutdown)

    fut_run, fut_wait = driver.submit_requests([
        {"request_id": "req-run", "prompt": {"prompt_token_ids": [1]}, "params": object()},
        {"request_id": "req-wait", "prompt": {"prompt_token_ids": [2]}, "params": object()},
    ])
    out_run = fut_run.result(timeout=5.0)
    out_wait = fut_wait.result(timeout=5.0)

    stats_run = getattr(out_run, "_tunix_queue_stats", None)
    stats_wait = getattr(out_wait, "_tunix_queue_stats", None)
    self.assertIsInstance(stats_run, dict)
    self.assertIsInstance(stats_wait, dict)

    # req-run ran on step 1 with 0 scheduler wait steps
    self.assertEqual(stats_run["sched_wait_steps"], 0)
    self.assertEqual(stats_run["sched_wait_s"], 0.0)
    self.assertGreater(stats_run["tpu_exec_s"], 0.0)
    self.assertAlmostEqual(stats_run["kv_cache_usage_max"], 0.75)
    self.assertAlmostEqual(stats_run["waiting_req_ratio"], 0.5)

    # req-wait waited in scheduler.waiting on step 1 and ran on step 2
    self.assertEqual(stats_wait["sched_wait_steps"], 1)
    self.assertGreater(stats_wait["sched_wait_s"], 0.0)
    self.assertGreater(stats_wait["tpu_exec_s"], 0.0)
    self.assertEqual(stats_wait["num_preemptions"], 1)
    self.assertTrue(stats_wait["l3_queued"])

  def test_get_vllm_schedulers_normalizes_list_and_single_and_none(self):
    s1, s2 = object(), object()
    e_single = types.SimpleNamespace(scheduler=s1)
    self.assertEqual(_get_vllm_schedulers(e_single), [s1])
    self.assertIs(_get_vllm_scheduler(e_single), s1)

    e_list = types.SimpleNamespace(scheduler=[s1, s2])
    self.assertEqual(_get_vllm_schedulers(e_list), [s1, s2])
    self.assertIs(_get_vllm_scheduler(e_list), s1)

    e_ec = types.SimpleNamespace(
        scheduler=None, engine_core=types.SimpleNamespace(scheduler=[s2])
    )
    self.assertEqual(_get_vllm_schedulers(e_ec), [s2])
    self.assertIs(_get_vllm_scheduler(e_ec), s2)

    e_none = types.SimpleNamespace()
    self.assertEqual(_get_vllm_schedulers(e_none), [])
    self.assertIsNone(_get_vllm_scheduler(e_none))

  def test_driver_attaches_l3_stats_with_list_schedulers_and_tuple_kv_cache_usage(
      self,
  ):
    class _ReqObj:

      def __init__(self, request_id: str, num_preemptions: int = 0):
        self.request_id = request_id
        self.num_preemptions = num_preemptions

    class _RankScheduler:

      def __init__(self, raw_kv):
        self.waiting = []
        self.skipped_waiting = []
        self.running = []
        self._raw_kv = raw_kv

      def get_kv_cache_usage(self):
        return self._raw_kv

    class _MultiSchedulerEngine(_FakeLLMEngine):

      def __init__(self):
        super().__init__(["req-r0", "req-r1"])
        self.s0 = _RankScheduler((0.85, 0.15))
        self.s1 = _RankScheduler((0.40, 0.05))
        self.scheduler = [self.s0, self.s1]

      def add_request(self, request_id: str, *_, **__):
        super().add_request(request_id)
        if request_id == "req-r0":
          self.s0.waiting.append(_ReqObj(request_id))
        else:
          self.s1.waiting.append(_ReqObj(request_id))

      def step(self):
        time.sleep(0.01)
        reqs = list(self._pending)
        self._pending.clear()
        self.s0.waiting.clear()
        self.s1.waiting.clear()
        return [_DummyRequestOutput(r) for r in reqs]

    engine = _MultiSchedulerEngine()
    driver = VLLMInProcessDriver(
        llm_engine=engine, poll_interval_s=0.001, auto_start=True
    )
    self.addCleanup(driver.shutdown)

    futs = driver.submit_requests([
        {
            "request_id": "req-r0",
            "prompt": {"prompt_token_ids": [1]},
            "params": object(),
        },
        {
            "request_id": "req-r1",
            "prompt": {"prompt_token_ids": [2]},
            "params": object(),
        },
    ])
    out0 = futs[0].result(timeout=5.0)
    out1 = futs[1].result(timeout=5.0)

    stats0 = getattr(out0, "_tunix_queue_stats", None)
    stats1 = getattr(out1, "_tunix_queue_stats", None)
    self.assertIsInstance(stats0, dict)
    self.assertIsInstance(stats1, dict)
    self.assertAlmostEqual(stats0["kv_cache_usage_max"], 0.85)
    self.assertAlmostEqual(stats1["kv_cache_usage_max"], 0.85)

  def test_driver_with_block_manager_fallback(self):
    class _FakeBlockManager:

      def get_num_free_gpu_blocks(self):
        return 20

      def get_num_total_gpu_blocks(self):
        return 100

    class _BlockManagerScheduler:

      def __init__(self):
        self.waiting = []
        self.skipped_waiting = []
        self.running = []
        self.block_manager = _FakeBlockManager()

    class _BlockManagerEngine(_FakeLLMEngine):

      def __init__(self):
        super().__init__(["req-bm"])
        self.scheduler = _BlockManagerScheduler()

      def step(self):
        time.sleep(0.01)
        reqs = list(self._pending)
        self._pending.clear()
        return [_DummyRequestOutput(r) for r in reqs]

    engine = _BlockManagerEngine()
    driver = VLLMInProcessDriver(
        llm_engine=engine, poll_interval_s=0.001, auto_start=True
    )
    self.addCleanup(driver.shutdown)

    futs = driver.submit_requests([
        {
            "request_id": "req-bm",
            "prompt": {"prompt_token_ids": [1]},
            "params": object(),
        },
    ])
    out = futs[0].result(timeout=5.0)
    stats = getattr(out, "_tunix_queue_stats", None)
    self.assertIsInstance(stats, dict)
    self.assertAlmostEqual(stats["kv_cache_usage_max"], 0.8)


if __name__ == "__main__":
  absltest.main()
