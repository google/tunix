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

"""Tests for continuous batching scheduler."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

from tunix.experimental.generate import kv_cache_manager
from tunix.experimental.generate import request as request_lib
from tunix.experimental.generate import scheduler


def _create_cache_config(
    page_size: int = 4,
    num_device_pages: int = 20,
    num_host_pages: int = 20,
    num_layers: int = 1,
    num_kv_heads: int = 2,
    head_dim: int = 16,
) -> kv_cache_manager.CacheConfig:
  bytes_per_layer_page = page_size * (2 * num_kv_heads) * head_dim * 4
  total_device_bytes_per_page = bytes_per_layer_page * num_layers
  total_host_bytes_per_page = bytes_per_layer_page * num_layers
  return kv_cache_manager.CacheConfig(
      max_device_bytes=total_device_bytes_per_page * num_device_pages,
      max_host_bytes=total_host_bytes_per_page * num_host_pages,
      page_size=page_size,
      dtype=jnp.float32,
  )


def _create_kv_cache_manager(
    page_size: int = 4,
    num_device_pages: int = 20,
    num_host_pages: int = 20,
) -> kv_cache_manager.KVCacheManager:
  return kv_cache_manager.KVCacheManager(
      config=_create_cache_config(
          page_size=page_size,
          num_device_pages=num_device_pages,
          num_host_pages=num_host_pages,
      ),
      cache_geometries={
          "cache_0": kv_cache_manager.CacheGeometry(
              num_kv_heads=2, head_dim=16
          )
      },
  )


def _create_scheduler_config(**overrides) -> scheduler.SchedulerConfig:
  kwargs = dict(
      max_num_batch_tokens=64,
      max_seqs_per_batch=4,
      chunked_prefill_length=16,
      num_decode_steps=1,
  )
  kwargs.update(overrides)
  return scheduler.SchedulerConfig(**kwargs)


def _create_request(
    req_id: str,
    prompt_token_ids: list[int],
    max_tokens_to_generate: int = 20,
    eos_token_ids: frozenset[int] = frozenset(),
) -> request_lib.Request:
  return request_lib.Request(
      req_id=req_id,
      prompt_token_ids=prompt_token_ids,
      sampling_params=request_lib.SamplingParams(
          max_tokens_to_generate=max_tokens_to_generate,
          eos_token_ids=eos_token_ids,
      ),
  )


def _admit(
    sched: scheduler.Scheduler,
    kv_mgr: kv_cache_manager.KVCacheManager,
    req: request_lib.Request,
) -> None:
  """Allocates a request's prompt and marks it running."""
  kv_mgr.sync_request_state(req)
  assert kv_mgr.allocate_slots(req, num_new_tokens=len(req.token_ids))
  req.num_in_flight_tokens = len(req.token_ids)
  sched._running_requests.append(req)


class SchedulerConfigTest(parameterized.TestCase):

  def test_valid_config(self):
    config = scheduler.SchedulerConfig(
        max_num_batch_tokens=128,
        max_seqs_per_batch=8,
        chunked_prefill_length=16,
        num_decode_steps=1,
    )
    self.assertEqual(config.max_num_batch_tokens, 128)
    self.assertEqual(config.max_seqs_per_batch, 8)
    self.assertEqual(config.chunked_prefill_length, 16)
    self.assertEqual(config.num_decode_steps, 1)

  def test_negative_chunked_prefill_length_raises(self):
    with self.assertRaisesRegex(ValueError, "must be non-negative"):
      scheduler.SchedulerConfig(
          max_num_batch_tokens=128,
          max_seqs_per_batch=8,
          chunked_prefill_length=-2,
      )

  def test_non_power_of_two_chunked_prefill_length_raises(self):
    with self.assertRaisesRegex(ValueError, "must be a power of 2"):
      scheduler.SchedulerConfig(
          max_num_batch_tokens=128,
          max_seqs_per_batch=8,
          chunked_prefill_length=14,
      )

    with self.assertRaisesRegex(ValueError, "must be a power of 2"):
      scheduler.SchedulerConfig(
          max_num_batch_tokens=128,
          max_seqs_per_batch=8,
          chunked_prefill_length=0,
      )

  def test_chunked_prefill_length_exceeds_max_batch_tokens_raises(self):
    with self.assertRaisesRegex(
        ValueError, "must be less than or equal to max_num_batch_tokens"
    ):
      scheduler.SchedulerConfig(
          max_num_batch_tokens=16,
          max_seqs_per_batch=8,
          chunked_prefill_length=32,
      )


class PreemptionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.kv_mgr = _create_kv_cache_manager(page_size=4, num_device_pages=20)
    self.sched = scheduler.Scheduler(
        config=_create_scheduler_config(),
        kv_cache_manager=self.kv_mgr,
    )

  def test_properties(self):
    self.assertEqual(self.sched.chunked_prefill_length, 16)
    self.assertEqual(self.sched.num_active_requests, 0)

    _admit(
        self.sched,
        self.kv_mgr,
        _create_request(req_id="r0", prompt_token_ids=[10, 20]),
    )
    self.sched._pending_requests.append(
        _create_request(req_id="r1", prompt_token_ids=[30])
    )
    self.assertEqual(self.sched.num_active_requests, 2)

  def test_preempt_moves_newest_request_to_front_of_pending(self):
    r0 = _create_request(req_id="r0", prompt_token_ids=[10, 20])
    r1 = _create_request(req_id="r1", prompt_token_ids=[30, 40])
    pending = _create_request(req_id="p", prompt_token_ids=[50])
    _admit(self.sched, self.kv_mgr, r0)
    _admit(self.sched, self.kv_mgr, r1)
    self.sched._pending_requests.append(pending)

    self.sched._preempt()

    self.assertEqual(list(self.sched._running_requests), [r0])
    self.assertEqual(list(self.sched._pending_requests), [r1, pending])

  def test_preempt_discards_kv_but_keeps_tokens(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20, 30, 40])
    _admit(self.sched, self.kv_mgr, req)
    req.num_completed_tokens = 4
    req.num_in_flight_tokens = 0
    req.token_ids.append(90)

    self.sched._preempt()

    # A preempted request must re-prefill from scratch.
    self.assertEqual(req.num_completed_tokens, 0)
    self.assertEqual(req.num_in_flight_tokens, 0)
    self.assertEqual(req.token_ids, [10, 20, 30, 40, 90])
    self.assertEqual(req.prompt_length, 4)
    self.assertEqual(self.kv_mgr.get_page_idxs("r1"), {"cache_0": ()})

  def test_preempt_all_on_empty_scheduler_is_a_noop(self):
    self.sched.preempt_all()
    self.assertEqual(self.sched.num_active_requests, 0)

  def test_preempt_all_requeues_every_running_request_in_arrival_order(self):
    reqs = [
        _create_request(req_id=f"r{i}", prompt_token_ids=[10 * i, 20, 30])
        for i in range(3)
    ]
    for req in reqs:
      _admit(self.sched, self.kv_mgr, req)
    self.sched._scheduled_requests = tuple(reqs)

    self.sched.preempt_all()

    self.assertEmpty(self.sched._running_requests)
    self.assertEmpty(self.sched._scheduled_requests)
    self.assertEqual(
        [r.request_id for r in self.sched._pending_requests],
        ["r0", "r1", "r2"],
    )
    self.assertEqual(self.sched.num_active_requests, 3)

  def test_preempt_all_releases_prefix_hashes(self):
    req = _create_request(req_id="r1", prompt_token_ids=list(range(12)))
    _admit(self.sched, self.kv_mgr, req)
    self.assertIn("r1", self.kv_mgr._request_to_prefix_hashes)

    self.sched.preempt_all()

    # Stale hashes would otherwise be re-registered against fresh pages and
    # become prefix-cache hits for other requests.
    self.assertNotIn("r1", self.kv_mgr._request_to_prefix_hashes)


class AdmissionTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.kv_mgr = _create_kv_cache_manager(page_size=4, num_device_pages=20)
    self.sched = scheduler.Scheduler(
        config=_create_scheduler_config(),
        kv_cache_manager=self.kv_mgr,
    )
    self.sched._token_budget = 64

  def _create_scheduler(self, kv_mgr=None, **config_overrides):
    sched = scheduler.Scheduler(
        config=_create_scheduler_config(**config_overrides),
        kv_cache_manager=kv_mgr or self.kv_mgr,
    )
    sched._token_budget = sched._config.max_num_batch_tokens
    return sched

  def _complete_step(self, req: request_lib.Request, token: int) -> None:
    """Marks the request's in-flight tokens as computed and samples a token."""
    req.token_ids.append(token)
    req.num_completed_tokens = len(req.token_ids) - 1
    req.num_in_flight_tokens = 0

  def test_allocate_slots_fails_if_token_budget_less_than_one(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20])
    self.sched._token_budget = 0
    self.assertFalse(self.sched._allocate_slots(req))
    self.assertEqual(req.num_in_flight_tokens, 0)
    self.assertEqual(req.num_completed_tokens, 0)

  def test_allocate_slots_full_prefill(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20, 30, 40])

    self.assertTrue(self.sched._allocate_slots(req))

    self.assertEqual(req.num_in_flight_tokens, 4)
    self.assertFalse(req.is_decode)
    self.assertFalse(req.is_chunked_prefill)

  def test_allocate_slots_chunked_prefill(self):
    req = _create_request(req_id="r1", prompt_token_ids=list(range(32)))

    self.assertTrue(self.sched._allocate_slots(req))

    self.assertEqual(req.num_in_flight_tokens, 16)
    self.assertFalse(req.is_decode)
    self.assertTrue(req.is_chunked_prefill)

  def test_allocate_slots_chunked_prefill_needs_a_full_chunk_of_budget(self):
    req = _create_request(req_id="r1", prompt_token_ids=list(range(32)))
    self.sched._token_budget = 8

    self.assertFalse(self.sched._allocate_slots(req))
    self.assertEqual(req.num_in_flight_tokens, 0)

  def test_allocate_slots_single_token_is_decode(self):
    req = _create_request(req_id="r1", prompt_token_ids=[42])

    self.assertTrue(self.sched._allocate_slots(req))

    self.assertEqual(req.num_in_flight_tokens, 1)
    self.assertTrue(req.is_decode)
    self.assertFalse(req.is_chunked_prefill)

  def test_allocate_slots_skips_prefix_cache_hit(self):
    cached = _create_request(req_id="r1", prompt_token_ids=list(range(12)))
    self.assertTrue(self.sched._allocate_slots(cached))
    self._complete_step(cached, 90)
    self.kv_mgr.sync_request_state(cached)

    # The first 3 pages (12 tokens) are cached, so only the last 2 prompt
    # tokens need computing.
    req = _create_request(req_id="r2", prompt_token_ids=list(range(14)))
    self.assertTrue(self.sched._allocate_slots(req))

    self.assertEqual(req.num_completed_tokens, 12)
    self.assertEqual(req.num_in_flight_tokens, 2)

  def test_allocate_slots_with_no_unprocessed_tokens_raises(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20])
    req.num_in_flight_tokens = 2

    with self.assertRaisesRegex(RuntimeError, "no unprocessed tokens"):
      self.sched._allocate_slots(req)

  def test_allocate_slots_insufficient_cache_returns_false(self):
    sched = self._create_scheduler(
        kv_mgr=_create_kv_cache_manager(page_size=4, num_device_pages=1)
    )
    req = _create_request(req_id="r1", prompt_token_ids=list(range(8)))

    self.assertFalse(sched._allocate_slots(req))
    self.assertEqual(req.num_in_flight_tokens, 0)
    self.assertEqual(req.num_completed_tokens, 0)

  def test_schedule_pending_respects_max_seqs_per_batch(self):
    for i in range(6):
      self.sched._pending_requests.append(
          _create_request(req_id=f"r{i}", prompt_token_ids=[i])
      )

    self.sched._schedule_pending_sequences()

    self.assertLen(self.sched._running_requests, 4)
    self.assertLen(self.sched._pending_requests, 2)

  def test_schedule_pending_stops_at_token_budget(self):
    sched = self._create_scheduler(max_num_batch_tokens=16)
    r1 = _create_request(req_id="r1", prompt_token_ids=list(range(8)))
    r2 = _create_request(req_id="r2", prompt_token_ids=list(range(12)))
    sched._pending_requests.extend([r1, r2])

    sched._schedule_pending_sequences()

    # r1 uses 8 of the 16 token budget. r2's 12 tokens don't fit in the
    # remaining 8, and a chunk needs 16.
    self.assertEqual(list(sched._running_requests), [r1])
    self.assertEqual(list(sched._pending_requests), [r2])
    self.assertEqual(sched._token_budget, 8)

  def test_schedule_pending_stops_at_first_request_that_does_not_fit(self):
    sched = self._create_scheduler(
        kv_mgr=_create_kv_cache_manager(page_size=4, num_device_pages=2)
    )
    r1 = _create_request(req_id="r1", prompt_token_ids=list(range(4)))
    r2 = _create_request(req_id="r2", prompt_token_ids=list(range(8)))
    r3 = _create_request(req_id="r3", prompt_token_ids=list(range(4)))
    sched._pending_requests.extend([r1, r2, r3])

    sched._schedule_pending_sequences()

    # r2 needs 2 pages but only 1 is left. r3 would fit, but requests are
    # admitted in arrival order.
    self.assertEqual(list(sched._running_requests), [r1])
    self.assertEqual(list(sched._pending_requests), [r2, r3])

  def test_schedule_running_preempts_newest_on_cache_exhaustion(self):
    sched = self._create_scheduler(
        kv_mgr=_create_kv_cache_manager(page_size=4, num_device_pages=2)
    )
    # Each request fills 1 page, so the cache is full.
    r1 = _create_request(req_id="r1", prompt_token_ids=list(range(4)))
    r2 = _create_request(req_id="r2", prompt_token_ids=list(range(4)))
    sched._pending_requests.extend([r1, r2])
    sched._schedule_pending_sequences()
    self._complete_step(r1, 90)
    self._complete_step(r2, 91)

    # Each decode needs a new page, so r2 (newest) is preempted to make room.
    sched._token_budget = 32
    sched._schedule_running_sequences()

    self.assertEqual(list(sched._running_requests), [r1])
    self.assertEqual(list(sched._pending_requests), [r2])
    self.assertTrue(r1.is_decode)

  def test_schedule_running_preempts_requests_beyond_max_seqs_per_batch(self):
    sched = self._create_scheduler(max_seqs_per_batch=1)
    r1 = _create_request(req_id="r1", prompt_token_ids=[10, 20])
    r2 = _create_request(req_id="r2", prompt_token_ids=[30, 40])
    sched._running_requests.extend([r1, r2])

    sched._schedule_running_sequences()

    self.assertEqual(list(sched._running_requests), [r1])
    self.assertEqual(list(sched._pending_requests), [r2])

  def test_schedule_running_preempts_requests_beyond_token_budget(self):
    sched = self._create_scheduler(max_num_batch_tokens=16)
    r1 = _create_request(req_id="r1", prompt_token_ids=list(range(8)))
    r2 = _create_request(req_id="r2", prompt_token_ids=list(range(8, 16)))
    r3 = _create_request(req_id="r3", prompt_token_ids=list(range(16, 24)))
    sched._running_requests.extend([r1, r2, r3])

    sched._schedule_running_sequences()

    # r1 and r2 use the full 16 token budget, so r3 is preempted even though
    # the cache has room for it.
    self.assertEqual(list(sched._running_requests), [r1, r2])
    self.assertEqual(list(sched._pending_requests), [r3])
    self.assertEqual(sched._token_budget, 0)

  def test_deadlock_raises_runtime_error(self):
    # Cache has 1 allocatable page. A request of 8 tokens cannot fit even alone.
    sched = self._create_scheduler(
        kv_mgr=_create_kv_cache_manager(page_size=4, num_device_pages=1),
        max_num_batch_tokens=32,
    )
    req = _create_request(req_id="r1", prompt_token_ids=list(range(8)))
    sched._running_requests.append(req)

    with self.assertRaisesRegex(
        RuntimeError, "No running requests could be scheduled."
    ):
      sched._schedule_running_sequences()


class ScheduleStepTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.kv_mgr = _create_kv_cache_manager(page_size=4, num_device_pages=20)
    self.config = scheduler.SchedulerConfig(
        max_num_batch_tokens=64,
        max_seqs_per_batch=4,
        chunked_prefill_length=16,
        num_decode_steps=1,
    )
    self.sched = scheduler.Scheduler(
        config=self.config,
        kv_cache_manager=self.kv_mgr,
    )

  def test_queue_new_requests(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20, 30])
    self.sched._queue_new_requests([req])
    self.assertEqual(self.sched.num_active_requests, 1)

  def test_schedule_step_full_prefill(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20, 30, 40])
    scheduled, dist = self.sched.schedule_step([req])

    self.assertLen(scheduled, 1)
    self.assertEqual(scheduled[0].request_id, "r1")
    # 0 decodes, 0 chunked prefills, 1 full prefill -> (0, 0, 1)
    self.assertEqual(dist, (0, 0, 1))
    # 4 prompt tokens allocated
    self.assertEqual(req.num_in_flight_tokens, 4)
    self.assertFalse(req.is_decode)
    self.assertFalse(req.is_chunked_prefill)

  def test_schedule_step_chunked_prefill(self):
    # Prompt is 32 tokens, but chunked_prefill_length is 16 and max_num_batch_tokens is 16
    small_config = scheduler.SchedulerConfig(
        max_num_batch_tokens=16,
        max_seqs_per_batch=4,
        chunked_prefill_length=16,
        num_decode_steps=1,
    )
    sched = scheduler.Scheduler(
        config=small_config,
        kv_cache_manager=self.kv_mgr,
    )
    prompt = list(range(32))
    req = _create_request(req_id="r1", prompt_token_ids=prompt)
    scheduled, dist = sched.schedule_step([req])

    self.assertLen(scheduled, 1)
    # 0 decodes, 1 chunked prefill, 0 full prefill -> (0, 1, 1)
    self.assertEqual(dist, (0, 1, 1))
    self.assertTrue(req.is_chunked_prefill)
    self.assertFalse(req.is_decode)
    self.assertEqual(req.num_in_flight_tokens, 16)

  def test_single_token_prompt_schedules_as_decode(self):
    req = _create_request(req_id="r1", prompt_token_ids=[42])
    scheduled, dist = self.sched.schedule_step([req])

    self.assertLen(scheduled, 1)
    # 1 decode, 0 chunked, 0 full prefill -> (1, 1, 1)
    self.assertEqual(dist, (1, 1, 1))
    self.assertTrue(req.is_decode)
    self.assertFalse(req.is_chunked_prefill)
    self.assertEqual(req.num_in_flight_tokens, 1)

  def test_schedule_step_orders_decodes_then_chunked_then_prefills(self):
    prefill = _create_request(req_id="p", prompt_token_ids=[10, 20, 30])
    chunked = _create_request(req_id="c", prompt_token_ids=list(range(32)))
    decode = _create_request(req_id="d", prompt_token_ids=[42])

    scheduled, dist = self.sched.schedule_step([prefill, chunked, decode])

    self.assertEqual(scheduled, (decode, chunked, prefill))
    # 1 decode, 1 chunked, 1 full prefill -> (1, 2, 3)
    self.assertEqual(dist, (1, 2, 3))

  def test_max_seqs_per_batch_limit(self):
    # Config max_seqs_per_batch is 4
    reqs = [
        _create_request(req_id=f"r{i}", prompt_token_ids=[i])
        for i in range(6)
    ]
    scheduled, dist = self.sched.schedule_step(reqs)

    self.assertLen(scheduled, 4)
    self.assertEqual(self.sched.num_active_requests, 6)
    self.assertLen(self.sched._pending_requests, 2)
    self.assertLen(self.sched._running_requests, 4)

  def test_token_budget_exhaustion(self):
    config = scheduler.SchedulerConfig(
        max_num_batch_tokens=16,
        max_seqs_per_batch=4,
        chunked_prefill_length=16,
        num_decode_steps=1,
    )
    sched = scheduler.Scheduler(config=config, kv_cache_manager=self.kv_mgr)

    req1 = _create_request(req_id="r1", prompt_token_ids=list(range(8)))
    req2 = _create_request(req_id="r2", prompt_token_ids=list(range(100, 112)))

    scheduled, dist = sched.schedule_step([req1, req2])
    # req1 uses 8 tokens, leaving 8.
    # req2 needs 12 tokens, which exceeds remaining budget 8, cannot fit.
    self.assertLen(scheduled, 1)
    self.assertEqual(scheduled[0].request_id, "r1")
    self.assertLen(sched._pending_requests, 1)
    self.assertEqual(sched._pending_requests[0].request_id, "r2")

  def test_aborted_pending_request_is_never_scheduled(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20, 30])
    self.sched._queue_new_requests([req])

    self.sched.abort_request(req)
    scheduled, _ = self.sched.schedule_step([])

    self.assertEmpty(scheduled)
    self.assertEqual(self.sched.num_active_requests, 0)

  def test_aborted_running_request_is_released_next_step(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20, 30, 40])
    self.sched.schedule_step([req])

    self.sched.abort_request(req)
    # The request is only discarded when the scheduler next processes it.
    self.assertEqual(self.sched.num_active_requests, 1)
    scheduled, _ = self.sched.schedule_step([])

    self.assertEmpty(scheduled)
    self.assertEqual(self.sched.num_active_requests, 0)
    self.assertEqual(self.kv_mgr.get_page_idxs("r1"), {"cache_0": ()})


class UpdateFromOutputTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.kv_mgr = _create_kv_cache_manager(page_size=4, num_device_pages=20)
    self.config = scheduler.SchedulerConfig(
        max_num_batch_tokens=64,
        max_seqs_per_batch=4,
        chunked_prefill_length=16,
        num_decode_steps=1,
    )
    self.sched = scheduler.Scheduler(
        config=self.config,
        kv_cache_manager=self.kv_mgr,
    )

  def test_continuation_of_chunked_prefill(self):
    small_config = scheduler.SchedulerConfig(
        max_num_batch_tokens=16,
        max_seqs_per_batch=4,
        chunked_prefill_length=16,
        num_decode_steps=1,
    )
    sched = scheduler.Scheduler(
        config=small_config,
        kv_cache_manager=self.kv_mgr,
    )
    prompt = list(range(32))
    req = _create_request(req_id="r1", prompt_token_ids=prompt)

    # Step 1: schedules first chunk of 16 tokens
    scheduled1, dist1 = sched.schedule_step([req])
    self.assertEqual(dist1, (0, 1, 1))
    self.assertEqual(req.num_in_flight_tokens, 16)
    self.assertEqual(req.num_completed_tokens, 0)

    # Engine step executes: update from output for chunk 1
    sched.update_from_output(
        generated_tokens=np.array([[0]]),
    )
    self.assertEqual(req.num_completed_tokens, 16)
    self.assertEqual(req.num_in_flight_tokens, 0)

    # Step 2: schedules second chunk of 16 tokens
    scheduled2, dist2 = sched.schedule_step([])
    self.assertLen(scheduled2, 1)
    self.assertEqual(scheduled2[0].request_id, "r1")
    self.assertEqual(req.num_in_flight_tokens, 16)

  def test_schedule_step_decode_requests(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20, 30, 40])
    scheduled, dist = self.sched.schedule_step([req])
    self.assertEqual(dist, (0, 0, 1))

    # Complete prefill + first decode token
    self.sched.update_from_output(
        generated_tokens=np.array([[99]]),
    )
    self.assertIn(99, req.token_ids)

    # Next step: should schedule as decode request
    scheduled2, dist2 = self.sched.schedule_step([])
    self.assertLen(scheduled2, 1)
    # 1 decode, 0 chunked, 0 full prefill -> (1, 1, 1)
    self.assertEqual(dist2, (1, 1, 1))
    self.assertEqual(req.num_in_flight_tokens, 1)
    self.assertTrue(req.is_decode)

  def test_request_aborted_mid_step_is_discarded(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20, 30, 40])
    self.sched.schedule_step([req])

    self.sched.abort_request(req)
    completed = self.sched.update_from_output(
        generated_tokens=np.array([[99]]),
    )

    self.assertEmpty(completed)
    self.assertNotIn(99, req.token_ids)
    self.assertEqual(self.sched.num_active_requests, 0)
    self.assertEqual(self.kv_mgr.get_page_idxs("r1"), {"cache_0": ()})

  def test_schedule_ordering_and_distribution(self):
    # Enqueue req1 and step to make it decode
    req1 = _create_request(req_id="r1", prompt_token_ids=[10, 20])
    self.sched.schedule_step([req1])
    self.sched.update_from_output(
        generated_tokens=np.array([[99]]),
    )

    # Now enqueue req2 (full prefill)
    req2 = _create_request(req_id="r2", prompt_token_ids=[30, 40])
    scheduled, dist = self.sched.schedule_step([req2])

    # Expect: req1 (decode) then req2 (prefill)
    self.assertEqual([r.request_id for r in scheduled], ["r1", "r2"])
    self.assertTrue(req1.is_decode)
    self.assertFalse(req2.is_decode)
    # 1 decode, 0 chunked, 1 full prefill -> (1, 1, 2)
    self.assertEqual(dist, (1, 1, 2))

  def test_update_from_output_eos_termination(self):
    req = _create_request(
        req_id="r1", prompt_token_ids=[10, 20], eos_token_ids=frozenset({1})
    )
    self.sched.schedule_step([req])

    completed = self.sched.update_from_output(
        generated_tokens=np.array([[1]]),
    )

    self.assertLen(completed, 1)
    self.assertEqual(completed[0].request_id, "r1")
    self.assertEqual(self.sched.num_active_requests, 0)
    self.assertEqual(req.num_in_flight_tokens, 0)
    self.assertEqual(self.kv_mgr.get_page_idxs("r1"), {"cache_0": ()})

  def test_update_from_output_truncation(self):
    req = _create_request(
        req_id="r1", prompt_token_ids=[10, 20], max_tokens_to_generate=2
    )
    self.sched.schedule_step([req])

    # Step 1: 1 generated token
    completed = self.sched.update_from_output(
        generated_tokens=np.array([[5]]),
    )
    self.assertEmpty(completed)

    # Step 2: 1 generated token reaches max_tokens_to_generate (2)
    self.sched.schedule_step([])
    completed = self.sched.update_from_output(
        generated_tokens=np.array([[6]]),
    )
    self.assertLen(completed, 1)
    self.assertEqual(completed[0].request_id, "r1")
    self.assertEqual(self.sched.num_active_requests, 0)
    self.assertEqual(len(req.token_ids), 4)  # 2 prompt + 2 generated
    self.assertEqual(self.kv_mgr.get_page_idxs("r1"), {"cache_0": ()})

  def test_preemption_on_cache_exhaustion(self):
    tiny_kv_mgr = _create_kv_cache_manager(page_size=4, num_device_pages=2)
    config = scheduler.SchedulerConfig(
        max_num_batch_tokens=32,
        max_seqs_per_batch=4,
        chunked_prefill_length=16,
        num_decode_steps=1,
    )
    sched = scheduler.Scheduler(config=config, kv_cache_manager=tiny_kv_mgr)

    # Both req1 and req2 use 4 tokens (1 page each) -> cache is now full (2 pages used)
    req1 = _create_request(req_id="r1", prompt_token_ids=list(range(4)))
    req2 = _create_request(req_id="r2", prompt_token_ids=list(range(4)))
    sched.schedule_step([req1, req2])
    sched.update_from_output(generated_tokens=np.array([[90], [91]]))

    # Now in next step, both req1 and req2 want to decode (+1 token each).
    # Since pages are full, req2 (newest) must be preempted to free up space!
    scheduled, dist = sched.schedule_step([])
    # req1 is scheduled, req2 was preempted and pushed to pending
    self.assertEqual([r.request_id for r in scheduled], ["r1"])
    self.assertIn(req2, sched._pending_requests)

  def test_sequential_scheduling_steps(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20, 30, 40])
    # Step 1: prefill
    self.sched.schedule_step([req])
    self.assertEqual(req.num_in_flight_tokens, 4)
    self.sched.update_from_output(
        generated_tokens=np.array([[90]]),
    )
    self.assertEqual(req.num_completed_tokens, 4)
    self.assertEqual(req.num_in_flight_tokens, 0)

    # Step 2: decode step 1
    self.sched.schedule_step([])
    self.assertTrue(req.is_decode)
    self.assertEqual(req.num_in_flight_tokens, 1)
    self.sched.update_from_output(
        generated_tokens=np.array([[91]]),
    )
    self.assertEqual(req.num_completed_tokens, 5)
    self.assertEqual(req.num_in_flight_tokens, 0)

    # Step 3: decode step 2
    self.sched.schedule_step([])
    self.assertEqual(req.num_in_flight_tokens, 1)
    self.sched.update_from_output(
        generated_tokens=np.array([[92]]),
    )
    self.assertEqual(req.num_completed_tokens, 6)
    self.assertEqual(req.num_in_flight_tokens, 0)
    self.assertEqual(req.token_ids, [10, 20, 30, 40, 90, 91, 92])

  def test_prefix_caching_computed_pages(self):
    # Page size is 4. Prompt with 12 tokens hashes 2 full pages (withholding last token leaves 11 -> 2 pages).
    req1 = _create_request(req_id="r1", prompt_token_ids=list(range(12)))
    self.sched.schedule_step([req1])
    self.sched.update_from_output(
        generated_tokens=np.array([[90]]),
    )

    # Second request shares the exact same prefix tokens
    req2 = _create_request(req_id="r2", prompt_token_ids=list(range(14)))
    # Scheduling runs r1 first, which syncs its now-full third page into the
    # cache, so r2 matches 3 pages rather than the 2 that were full at r1's
    # first schedule.
    scheduled, dist = self.sched.schedule_step([req2])
    self.assertIn(req2, scheduled)
    # 3 pages (12 tokens) hit prefix cache, so num_completed_tokens advances
    # by 12.
    self.assertEqual(req2.num_completed_tokens, 12)
    # Remaining 2 prompt tokens are allocated in-flight
    self.assertEqual(req2.num_in_flight_tokens, 2)

  def test_update_from_output_with_logits_and_logprobs(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20])
    self.sched.schedule_step([req])

    logits = np.array([[[0.1, 0.9]]])
    logprobs = np.array([[-0.1]])

    self.sched.update_from_output(
        generated_tokens=np.array([[90]]),
        logits=logits,
        logprobs=logprobs,
    )

    self.assertLen(req.token_ids, 3)
    self.assertLen(req.logprobs, 1)
    self.assertLen(req.logits, 1)

  def test_preempted_resumed_request_requires_prefill(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20, 30, 40])
    # Step 1: prefill (4 tokens)
    self.sched.schedule_step([req])
    self.sched.update_from_output(generated_tokens=np.array([[90]]))
    # Step 2: decode step 1 (1 token)
    self.sched.schedule_step([])
    self.sched.update_from_output(generated_tokens=np.array([[91]]))
    self.assertEqual(req.num_completed_tokens, 5)
    self.assertLen(req.token_ids, 6)

    # Preempt r1: Device pages released, moved to pending
    self.sched._preempt()
    self.assertIn(req, self.sched._pending_requests)
    self.assertEmpty(self.sched._running_requests)

    # Resuming r1: it must be scheduled as a prefill (dist has 0 decodes, 1 prefill)
    scheduled, dist = self.sched.schedule_step([])
    self.assertLen(scheduled, 1)
    self.assertEqual(dist, (0, 0, 1))
    self.assertFalse(req.is_decode)
    # r1's own first page was cached before it was preempted, so resuming only
    # re-prefills the 2 tokens the cache does not cover, not all 6.
    self.assertEqual(req.num_completed_tokens, 4)
    self.assertEqual(req.num_in_flight_tokens, 2)

  def test_prefix_cache_hit_leaving_single_token_schedules_as_decode(self):
    # Page size is 4. Prompt with 8 tokens creates 2 full pages.
    # Note: prefix caching withholds last token, so prompt of 9 tokens hashes 8 tokens (2 pages).
    req1 = _create_request(req_id="r1", prompt_token_ids=list(range(9)))
    self.sched.schedule_step([req1])
    self.kv_mgr.sync_request_state(req1)
    self.sched.update_from_output(generated_tokens=np.array([[99]]))

    # req2 has the exact same 9 tokens: 8 tokens hit prefix cache, leaving 1 unprocessed token.
    req2 = _create_request(req_id="r2", prompt_token_ids=list(range(9)))
    scheduled, dist = self.sched.schedule_step([req2])
    self.assertLen(scheduled, 2)
    self.assertEqual(scheduled, (req1, req2))
    # Both req1 and req2 are scheduled as decodes: (2, 2, 2)
    self.assertEqual(dist, (2, 2, 2))
    self.assertTrue(req2.is_decode)
    self.assertEqual(req2.num_completed_tokens, 8)
    self.assertEqual(req2.num_in_flight_tokens, 1)

  def test_per_request_max_tokens_to_generate(self):
    req1 = _create_request(
        req_id="r1", prompt_token_ids=[10], max_tokens_to_generate=2
    )
    req2 = _create_request(
        req_id="r2", prompt_token_ids=[20], max_tokens_to_generate=4
    )
    self.sched.schedule_step([req1, req2])
    # Step 1
    completed = self.sched.update_from_output(
        generated_tokens=np.array([[101], [201]])
    )
    self.assertEmpty(completed)

    # Step 2: req1 completes after 2 generated tokens
    self.sched.schedule_step([])
    completed = self.sched.update_from_output(
        generated_tokens=np.array([[102], [202]])
    )
    self.assertLen(completed, 1)
    self.assertEqual(completed[0].request_id, "r1")
    self.assertEqual(completed[0].token_ids, [10, 101, 102])

    # Step 3: req2 continues
    self.sched.schedule_step([])
    completed = self.sched.update_from_output(
        generated_tokens=np.array([[203]])
    )
    self.assertEmpty(completed)

    # Step 4: req2 completes after 4 generated tokens
    self.sched.schedule_step([])
    completed = self.sched.update_from_output(
        generated_tokens=np.array([[204]])
    )
    self.assertLen(completed, 1)
    self.assertEqual(completed[0].request_id, "r2")
    self.assertEqual(completed[0].token_ids, [20, 201, 202, 203, 204])

  def test_per_request_eos_token_ids(self):
    req1 = _create_request(
        req_id="r1", prompt_token_ids=[10], eos_token_ids=frozenset({99})
    )
    req2 = _create_request(
        req_id="r2", prompt_token_ids=[20], eos_token_ids=frozenset({1})
    )
    self.sched.schedule_step([req1, req2])

    # Token 1 is req2's EOS but not req1's.
    completed = self.sched.update_from_output(
        generated_tokens=np.array([[1], [1]])
    )
    self.assertLen(completed, 1)
    self.assertEqual(completed[0].request_id, "r2")

    # Token 99 is req1's EOS.
    self.sched.schedule_step([])
    completed = self.sched.update_from_output(
        generated_tokens=np.array([[99]])
    )
    self.assertLen(completed, 1)
    self.assertEqual(completed[0].request_id, "r1")

  def test_request_without_eos_never_terminates_on_a_token(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10])
    self.sched.schedule_step([req])

    completed = self.sched.update_from_output(generated_tokens=np.array([[1]]))

    self.assertEmpty(completed)
    self.assertEqual(req.token_ids, [10, 1])


class PreemptAllTest(parameterized.TestCase):
  """Tests preempt_all across full scheduling steps."""

  def setUp(self):
    super().setUp()
    self.kv_mgr = _create_kv_cache_manager(page_size=4, num_device_pages=20)
    self.config = scheduler.SchedulerConfig(
        max_num_batch_tokens=64,
        max_seqs_per_batch=4,
        chunked_prefill_length=16,
        num_decode_steps=1,
    )
    self.sched = scheduler.Scheduler(
        config=self.config,
        kv_cache_manager=self.kv_mgr,
    )

  def _run_to_decode(self, reqs):
    """Schedules `reqs` and steps once so each is mid-decode."""
    self.sched.schedule_step(reqs)
    self.sched.update_from_output(
        generated_tokens=np.array([[100 + i] for i in range(len(reqs))])
    )
    self.sched.schedule_step([])

  def test_preempted_request_stays_logprob_aligned(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20, 30, 40])
    self.sched.schedule_step([req])
    self.sched.update_from_output(
        generated_tokens=np.array([[101]]),
        logprobs=np.array([[-0.5]]),
    )
    self.sched.schedule_step([])
    self.assertLen(req.logprobs, 1)

    self.sched.preempt_all()

    # Rescheduling must re-cover the whole context. The prefix cache may serve
    # part of it back (the pages are only unreferenced, not freed), so what
    # matters is that the accounting adds up to the full sequence and that the
    # step yields exactly one new token, not a duplicate run.
    scheduled, _ = self.sched.schedule_step([])
    self.assertLen(scheduled, 1)
    self.assertEqual(
        req.num_completed_tokens + req.num_in_flight_tokens,
        len(req.token_ids),
    )

    self.sched.update_from_output(
        generated_tokens=np.array([[102]]),
        logprobs=np.array([[-0.25]]),
    )
    self.assertEqual(req.token_ids, [10, 20, 30, 40, 101, 102])
    self.assertLen(req.logprobs, 2)

  def test_preempt_all_with_cache_reset_forces_full_prefill(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20, 30, 40])
    self._run_to_decode([req])

    # Wiping the prefix cache makes the re-prefill unconditional: nothing
    # computed under the old weights may be reused.
    self.sched.preempt_all()
    self.kv_mgr.reset_kv_caches()

    scheduled, dist = self.sched.schedule_step([])
    self.assertLen(scheduled, 1)
    self.assertFalse(scheduled[0].is_decode)
    self.assertEqual(dist, (0, 0, 1))
    self.assertEqual(req.num_completed_tokens, 0)
    self.assertEqual(req.num_in_flight_tokens, 5)

  def test_preempt_all_respects_max_tokens_accounting(self):
    req = _create_request(
        req_id="r1", prompt_token_ids=[10, 20], max_tokens_to_generate=2
    )
    self.sched.schedule_step([req])
    self.sched.update_from_output(generated_tokens=np.array([[101]]))
    self.sched.schedule_step([])

    self.sched.preempt_all()

    # One token of the 2-token budget is already spent, so the re-prefilled
    # request must finish after exactly one more.
    self.sched.schedule_step([])
    finished = self.sched.update_from_output(generated_tokens=np.array([[102]]))
    self.assertLen(finished, 1)
    self.assertEqual(req.token_ids, [10, 20, 101, 102])


if __name__ == "__main__":
  absltest.main()
