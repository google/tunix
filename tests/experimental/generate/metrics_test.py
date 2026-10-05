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

from unittest import mock

from absl.testing import absltest
from tunix.experimental.generate import metrics as metrics_lib
from tunix.experimental.generate import request as request_lib
from tunix.experimental.generate import sampler as sampler_lib
from tunix.experimental.generate import testing_utils
from tunix.experimental.rollout import sampler as rollout_sampler_lib


def _make_request_state(
    request_id: str = 'req-0',
    prompt_len: int = 8,
    max_tokens: int = 4,
    arrival_time: float = 10.0,
) -> request_lib.RequestState:
  return request_lib.RequestState(
      req_id=request_id,
      prompt_token_ids=list(range(prompt_len)),
      sampling_params=rollout_sampler_lib.SamplingParams(max_tokens=max_tokens),
      arrival_time=arrival_time,
  )


class MetricsCollectorTest(absltest.TestCase):

  def test_throughputs_exclude_idle_time_and_decode_only_steps_for_prefill(self):
    collector = metrics_lib.MetricsCollector(log_stats_interval_s=0.0)

    # Step 1: Prefill 100 prompt tokens, generate 2 tokens in 0.2s (50ms sched).
    collector.record_schedule(0.05, num_preemptions=1)
    collector.record_step(
        step_duration_s=0.20,
        num_prompt_tokens=100,
        num_generation_tokens=2,
        finished_requests=(),
        prefix_cache_queries=4,
        prefix_cache_hits=3,
    )

    # Simulate a 5-second weight sync / idle pause outside record_step.
    # Step 2: Decode-only step (0 prompt tokens), generate 18 tokens in 0.3s.
    collector.record_schedule(0.01, num_preemptions=0)
    collector.record_step(
        step_duration_s=0.30,
        num_prompt_tokens=0,
        num_generation_tokens=18,
        finished_requests=(),
        prefix_cache_queries=0,
        prefix_cache_hits=0,
    )

    snap = collector.snapshot(
        num_running_reqs=2,
        num_waiting_reqs=1,
        kv_cache_usage_fraction=0.25,
    )

    self.assertEqual(snap.num_active_steps, 2)
    self.assertEqual(snap.num_prefill_steps, 1)
    # 100 prompt tokens / 0.50s active time = 200 tok/s
    self.assertAlmostEqual(snap.avg_prompt_throughput_tok_per_s, 200.0)
    # 100 prompt tokens / 0.20s prefill-only step time = 500 tok/s
    self.assertAlmostEqual(snap.avg_prefill_throughput_tok_per_s, 500.0)
    # 20 generation tokens / 0.50s active time = 40 tok/s
    self.assertAlmostEqual(snap.avg_generation_throughput_tok_per_s, 40.0)
    # Schedule durations: (50ms + 10ms) / 2 = 30ms
    self.assertAlmostEqual(snap.avg_schedule_duration_ms, 30.0)
    # Engine step durations: (200ms + 300ms) / 2 = 250ms
    self.assertAlmostEqual(snap.avg_engine_step_duration_ms, 250.0)
    self.assertEqual(snap.num_preemptions, 1)
    self.assertAlmostEqual(snap.kv_cache_usage_pct, 25.0)
    self.assertAlmostEqual(snap.prefix_cache_hit_rate_pct, 75.0)

  def test_request_lifecycle_latencies(self):
    collector = metrics_lib.MetricsCollector(log_stats_interval_s=0.0)

    req = _make_request_state(arrival_time=10.0)
    # Scheduled at t=10.25 (queue_time = 0.25s)
    req.queue_time_s = 0.25
    req.first_scheduled_time = 10.25
    # First token at t=10.35 (ttft = 0.35s, prefill_time = 0.10s)
    req.first_token_time = 10.35
    # Finished at t=10.85 (decode_time = 0.50s, e2e = 0.85s)
    req.finished_time = 10.85

    collector.record_step(
        step_duration_s=0.10,
        num_prompt_tokens=8,
        num_generation_tokens=4,
        finished_requests=[req],
    )
    snap = collector.snapshot()

    self.assertEqual(snap.completed_requests, 1)
    self.assertAlmostEqual(snap.avg_request_queue_time_s, 0.25)
    self.assertAlmostEqual(snap.avg_request_ttft_s, 0.35)
    self.assertAlmostEqual(snap.avg_request_prefill_time_s, 0.10)
    self.assertAlmostEqual(snap.avg_request_decode_time_s, 0.50)
    self.assertAlmostEqual(snap.avg_request_e2e_latency_s, 0.85)

  def test_rollout_batch_completion_time_and_step_flush(self):
    collector = metrics_lib.MetricsCollector(log_stats_interval_s=0.0)
    collector.record_schedule(0.01)
    collector.record_step(
        step_duration_s=0.10,
        num_prompt_tokens=16,
        num_generation_tokens=8,
        finished_requests=(),
    )
    collector.record_batch_completion(1.2)
    collector.record_batch_completion(1.8)

    step_snap = collector.flush_step_snapshot(
        num_running_reqs=0,
        num_waiting_reqs=0,
        kv_cache_usage_fraction=0.1,
    )
    self.assertEqual(step_snap.completed_batches, 2)
    self.assertAlmostEqual(step_snap.last_batch_completion_time_s, 1.8)
    self.assertAlmostEqual(step_snap.avg_batch_completion_time_s, 1.5)

    perf_metrics = step_snap.to_perf_metrics()
    self.assertEqual(
        perf_metrics['rollout/last_batch_completion_time_s'][0], 1.8
    )
    self.assertEqual(
        perf_metrics['rollout/avg_batch_completion_time_s'][0], 1.5
    )
    self.assertEqual(
        perf_metrics['rollout/avg_generation_throughput_tok_per_s'][0], 80.0
    )

    # Flushing again with no new batches resets the step window while keeping
    # cumulative totals intact.
    empty_step_snap = collector.flush_step_snapshot()
    self.assertEqual(empty_step_snap.completed_batches, 0)
    self.assertEqual(empty_step_snap.num_active_steps, 0)

    cum_snap = collector.snapshot()
    self.assertEqual(cum_snap.completed_batches, 2)
    self.assertAlmostEqual(cum_snap.avg_batch_completion_time_s, 1.5)

  def test_periodic_console_logging_emits_vllm_style_line(self):
    with mock.patch.object(metrics_lib.time, 'perf_counter', return_value=100.0):
      collector = metrics_lib.MetricsCollector(log_stats_interval_s=10.0)

    collector.record_schedule(0.005)
    collector.record_step(
        step_duration_s=0.10,
        num_prompt_tokens=50,
        num_generation_tokens=10,
        finished_requests=(),
    )
    collector.record_batch_completion(0.45)

    with mock.patch.object(metrics_lib.logging, 'info') as mock_info:
      # Before interval elapsed: does not log.
      with mock.patch.object(
          metrics_lib.time, 'perf_counter', return_value=105.0
      ):
        self.assertFalse(
            collector.maybe_log_stats(
                num_running_reqs=2,
                num_waiting_reqs=1,
                kv_cache_usage_fraction=0.5,
            )
        )
      mock_info.assert_not_called()

      # After interval elapsed: logs vLLM-style line and resets log window.
      with mock.patch.object(
          metrics_lib.time, 'perf_counter', return_value=110.5
      ):
        self.assertTrue(
            collector.maybe_log_stats(
                num_running_reqs=2,
                num_waiting_reqs=1,
                kv_cache_usage_fraction=0.5,
            )
        )
      mock_info.assert_called_once()
      logged_msg = mock_info.call_args[0][0] % mock_info.call_args[0][1:]
      self.assertIn('Avg prompt throughput: 500.0 tokens/s', logged_msg)
      self.assertIn('Avg prefill throughput: 500.0 tokens/s', logged_msg)
      self.assertIn('Avg generation throughput: 100.0 tokens/s', logged_msg)
      self.assertIn('Running: 2 reqs', logged_msg)
      self.assertIn('Waiting: 1 reqs', logged_msg)
      self.assertIn('KV cache usage: 50.0%', logged_msg)


class EngineAndSamplerMetricsIntegrationTest(absltest.TestCase):

  def test_engine_populates_request_and_step_metrics(self):
    engine = testing_utils.make_engine(num_scheduler_steps=2)
    engine.add_request(
        testing_utils.make_request('0', [1, 2, 3, 4], max_tokens=3)
    )
    engine.add_request(
        testing_utils.make_request('1', [1, 2, 5, 6], max_tokens=2)
    )

    while engine.has_unfinished_requests():
      engine.step()

    snap = engine.get_metrics()
    self.assertEqual(snap.completed_requests, 2)
    self.assertEqual(snap.prompt_tokens, 8)
    self.assertEqual(snap.generation_tokens, 5)
    self.assertGreater(snap.num_active_steps, 0)
    self.assertGreater(snap.num_prefill_steps, 0)
    self.assertGreater(snap.avg_schedule_duration_ms, 0.0)
    self.assertGreater(snap.avg_engine_step_duration_ms, 0.0)
    self.assertGreaterEqual(snap.avg_request_queue_time_s, 0.0)
    self.assertGreater(snap.avg_request_ttft_s, 0.0)
    self.assertGreater(snap.avg_request_e2e_latency_s, 0.0)

  def test_sampler_records_batch_completion_time_in_offline_and_server_modes(
      self,
  ):
    for server_mode in (False, True):
      sampler = sampler_lib.Sampler(
          testing_utils.make_engine(), server_mode=server_mode
      )
      try:
        requests = [
            testing_utils.make_request('a', [1, 2, 3], max_tokens=2),
            testing_utils.make_request('b', [4, 5], max_tokens=2),
        ]
        sampler.generate(requests)
        snap = sampler.get_metrics()
        self.assertEqual(snap.completed_batches, 1)
        self.assertGreater(snap.last_batch_completion_time_s, 0.0)
        self.assertGreater(snap.avg_batch_completion_time_s, 0.0)
      finally:
        sampler.stop()


if __name__ == '__main__':
  absltest.main()
