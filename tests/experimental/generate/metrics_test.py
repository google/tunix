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

import asyncio
from unittest import mock

from absl.testing import absltest
import numpy as np
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


def _make_finished_request_state(
    request_id: str,
    *,
    prompt_len: int,
    num_generated: int,
    first_token_time: float,
    finished_time: float,
    arrival_time: float = 10.0,
) -> request_lib.RequestState:
  req = _make_request_state(
      request_id,
      prompt_len=prompt_len,
      max_tokens=num_generated,
      arrival_time=arrival_time,
  )
  req.token_ids.extend(range(num_generated))
  req.first_scheduled_time = arrival_time
  req.first_token_time = first_token_time
  req.finished_time = finished_time
  return req


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

  def test_request_time_per_output_token_and_max_e2e_latency(self):
    collector = metrics_lib.MetricsCollector(log_stats_interval_s=0.0)
    # Decodes 4 tokens after its first in 0.4 s: 100 ms per output token.
    fast = _make_finished_request_state(
        'fast',
        prompt_len=8,
        num_generated=5,
        first_token_time=10.2,
        finished_time=10.6,
    )
    # Generates a single token, so has no time per output token.
    slow = _make_finished_request_state(
        'slow',
        prompt_len=4,
        num_generated=1,
        first_token_time=11.5,
        finished_time=11.5,
    )

    collector.record_step(
        step_duration_s=0.1,
        num_prompt_tokens=12,
        num_generation_tokens=6,
        finished_requests=[fast, slow],
    )
    snap = collector.snapshot()

    self.assertEqual(snap.completed_requests, 2)
    self.assertAlmostEqual(snap.avg_request_time_per_output_token_ms, 100.0)
    self.assertAlmostEqual(snap.avg_request_e2e_latency_s, 1.05)
    self.assertAlmostEqual(snap.max_request_e2e_latency_s, 1.5)

  def test_batch_throughputs_are_over_wall_clock_time_since_batch_start(self):
    collector = metrics_lib.MetricsCollector(log_stats_interval_s=0.0)
    # Work the engine did before the batch started is not the batch's.
    collector.record_step(
        step_duration_s=0.1,
        num_prompt_tokens=50,
        num_generation_tokens=5,
        finished_requests=[_make_request_state('before')],
    )
    collector.record_batch_start('b0')
    collector.record_step(
        step_duration_s=0.1,
        num_prompt_tokens=100,
        num_generation_tokens=4,
        finished_requests=(),
    )
    collector.record_step(
        step_duration_s=0.1,
        num_prompt_tokens=0,
        num_generation_tokens=36,
        finished_requests=[_make_request_state('a'), _make_request_state('b')],
    )
    collector.record_batch_completion(2.0, batch_id='b0')

    snap = collector.snapshot()
    self.assertEqual(snap.completed_batches, 1)
    self.assertEqual(snap.num_timed_batches, 1)
    self.assertAlmostEqual(snap.avg_batch_requests, 2.0)
    # Over the batch's 2 s of wall-clock time, not its 0.2 s of engine steps.
    self.assertAlmostEqual(snap.batch_prefill_throughput_tok_per_s, 50.0)
    self.assertAlmostEqual(snap.batch_generation_throughput_tok_per_s, 20.0)

    perf_metrics = snap.to_perf_metrics()
    self.assertEqual(
        perf_metrics['rollout/batch_prefill_throughput_tok_per_s'][0], 50.0
    )
    self.assertEqual(
        perf_metrics['rollout/batch_generation_throughput_tok_per_s'][0], 20.0
    )
    self.assertEqual(perf_metrics['rollout/avg_batch_requests'][0], 2.0)

  def test_overlapping_batches_each_count_the_work_done_while_in_flight(self):
    collector = metrics_lib.MetricsCollector(log_stats_interval_s=0.0)

    def step(num_prompt_tokens: int, num_generation_tokens: int) -> None:
      collector.record_step(
          step_duration_s=0.1,
          num_prompt_tokens=num_prompt_tokens,
          num_generation_tokens=num_generation_tokens,
          finished_requests=(),
      )

    collector.record_batch_start(0)
    step(10, 10)
    collector.record_batch_start(1)
    step(20, 30)
    collector.record_batch_completion(1.0, batch_id=0)
    step(0, 60)
    collector.record_batch_completion(3.0, batch_id=1)

    snap = collector.snapshot()
    self.assertEqual(snap.num_timed_batches, 2)
    # Batch 0 counts 30 prefill and 40 generated tokens over 1 s, batch 1 20
    # and 90 over 3 s.
    self.assertAlmostEqual(snap.batch_prefill_throughput_tok_per_s, 12.5)
    self.assertAlmostEqual(snap.batch_generation_throughput_tok_per_s, 32.5)

  def test_batch_completion_without_a_start_only_records_its_duration(self):
    collector = metrics_lib.MetricsCollector(log_stats_interval_s=0.0)
    collector.record_step(
        step_duration_s=0.1,
        num_prompt_tokens=10,
        num_generation_tokens=10,
        finished_requests=(),
    )
    collector.record_batch_completion(1.5, batch_id='never-started')

    snap = collector.flush_step_snapshot()
    self.assertEqual(snap.completed_batches, 1)
    self.assertEqual(snap.num_timed_batches, 0)
    perf_metrics = snap.to_perf_metrics()
    self.assertEqual(perf_metrics['rollout/avg_batch_completion_time_s'][0], 1.5)
    self.assertNotIn(
        'rollout/batch_generation_throughput_tok_per_s', perf_metrics
    )

  def test_drops_the_oldest_batch_start_beyond_the_pending_limit(self):
    collector = metrics_lib.MetricsCollector(log_stats_interval_s=0.0)
    with mock.patch.object(metrics_lib, '_MAX_PENDING_BATCH_STARTS', 2):
      for batch_id in range(3):
        collector.record_batch_start(batch_id)
    collector.record_batch_completion(1.0, batch_id=0)
    collector.record_batch_completion(1.0, batch_id=2)

    snap = collector.snapshot()
    self.assertEqual(snap.completed_batches, 2)
    self.assertEqual(snap.num_timed_batches, 1)

  def test_batch_completion_logs_the_batch_throughputs(self):
    collector = metrics_lib.MetricsCollector(log_stats_interval_s=0.0)
    collector.record_batch_start(7)
    collector.record_step(
        step_duration_s=0.1,
        num_prompt_tokens=30,
        num_generation_tokens=10,
        finished_requests=[_make_request_state()],
    )

    with mock.patch.object(metrics_lib.logging, 'info') as mock_info:
      collector.record_batch_completion(2.0, batch_id=7)

    mock_info.assert_called_once()
    logged_msg = mock_info.call_args[0][0] % mock_info.call_args[0][1:]
    self.assertEqual(
        logged_msg,
        'Engine 000: Rollout batch 7 finished in 2.00 s: 1 requests, Prefill'
        ' throughput: 15.0 tokens/s, Generation throughput: 5.0 tokens/s.',
    )

  def test_step_gauges_are_averaged_over_engine_steps(self):
    collector = metrics_lib.MetricsCollector(log_stats_interval_s=0.0)
    for num_running_reqs, kv_cache_usage_fraction in ((4, 0.25), (2, 0.75)):
      collector.record_step(
          step_duration_s=0.1,
          num_prompt_tokens=0,
          num_generation_tokens=num_running_reqs,
          finished_requests=(),
          num_running_reqs=num_running_reqs,
          kv_cache_usage_fraction=kv_cache_usage_fraction,
      )

    # The KV cache is free again once the step's rollouts are done.
    snap = collector.flush_step_snapshot(kv_cache_usage_fraction=0.0)
    self.assertEqual(snap.avg_num_running_reqs, 3.0)
    self.assertEqual(snap.avg_kv_cache_usage_pct, 50.0)
    self.assertEqual(snap.max_kv_cache_usage_pct, 75.0)
    self.assertEqual(snap.kv_cache_usage_pct, 0.0)

    perf_metrics = snap.to_perf_metrics()
    self.assertEqual(perf_metrics['rollout/avg_kv_cache_usage_pct'][0], 50.0)
    # Repeated values aggregate to their peak.
    self.assertEqual(
        perf_metrics['rollout/max_kv_cache_usage_pct'], (75.0, np.max)
    )
    # Read at the flush, after the rollouts are done, it would always be 0.
    self.assertNotIn('rollout/kv_cache_usage_pct', perf_metrics)

  def test_perf_metrics_omit_request_and_batch_averages_without_any(self):
    perf_metrics = (
        metrics_lib.MetricsCollector().flush_step_snapshot().to_perf_metrics()
    )

    self.assertEqual(perf_metrics['rollout/completed_requests'][0], 0.0)
    self.assertEqual(perf_metrics['rollout/completed_batches'][0], 0.0)
    self.assertEqual(
        perf_metrics['rollout/avg_generation_throughput_tok_per_s'][0], 0.0
    )
    for name in (
        'rollout/avg_request_e2e_latency_s',
        'rollout/avg_batch_completion_time_s',
        'rollout/batch_generation_throughput_tok_per_s',
    ):
      self.assertNotIn(name, perf_metrics)

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
    self.assertGreaterEqual(
        snap.max_request_e2e_latency_s, snap.avg_request_e2e_latency_s
    )
    self.assertGreater(snap.avg_request_time_per_output_token_ms, 0.0)
    self.assertGreater(snap.avg_num_running_reqs, 0.0)
    self.assertGreater(snap.max_kv_cache_usage_pct, 0.0)

  def test_sampler_records_batches_in_offline_and_server_modes(self):
    for server_mode in (False, True):
      with self.subTest(server_mode=server_mode):
        sampler = sampler_lib.Sampler(
            testing_utils.make_engine(), server_mode=server_mode
        )
        try:
          requests = [
              testing_utils.make_request('a', [1, 2, 3], max_tokens=2),
              testing_utils.make_request('b', [4, 5], max_tokens=2),
          ]
          sampler.generate(requests)
          # A single request is not a batch of its own.
          sampler.generate(
              [testing_utils.make_request('c', [6, 7], max_tokens=2)]
          )
          asyncio.run(
              sampler.sample(
                  testing_utils.make_request('d', [8, 9], max_tokens=2)
              )
          )

          snap = sampler.get_metrics()
          self.assertEqual(snap.completed_requests, 4)
          self.assertEqual(snap.completed_batches, 1)
          self.assertGreater(snap.last_batch_completion_time_s, 0.0)
          self.assertGreater(snap.avg_batch_completion_time_s, 0.0)
          self.assertEqual(snap.num_timed_batches, 1)
          self.assertAlmostEqual(snap.avg_batch_requests, 2.0)
          # 5 prefill and 4 generated tokens over the batch's wall-clock time.
          self.assertAlmostEqual(
              snap.batch_prefill_throughput_tok_per_s
              * snap.last_batch_completion_time_s,
              5.0,
          )
          self.assertAlmostEqual(
              snap.batch_generation_throughput_tok_per_s
              * snap.last_batch_completion_time_s,
              4.0,
          )
        finally:
          sampler.stop()


if __name__ == '__main__':
  absltest.main()
