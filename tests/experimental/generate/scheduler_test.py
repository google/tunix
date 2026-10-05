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

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

from tunix.experimental.generate import kv_cache_manager
from tunix.experimental.generate import request as request_lib
from tunix.experimental.generate import scheduler
from tunix.experimental.rollout import sampler as sampler_lib


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
  gib = 1 << 30
  return kv_cache_manager.CacheConfig(
      max_device_size_gib=(total_device_bytes_per_page * num_device_pages)
      / gib,
      max_host_size_gib=(total_host_bytes_per_page * num_host_pages) / gib,
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
      max_num_batched_tokens=64,
      max_num_seqs=4,
      max_chunked_prefill_length=16,
      num_scheduler_steps=1,
  )
  kwargs.update(overrides)
  return scheduler.SchedulerConfig(**kwargs)


def _create_request(
    req_id: str,
    prompt_token_ids: list[int],
    max_tokens: int = 20,
) -> request_lib.RequestState:
  return request_lib.RequestState(
      req_id=req_id,
      prompt_token_ids=prompt_token_ids,
      sampling_params=sampler_lib.SamplingParams(max_tokens=max_tokens),
  )


def _admit(
    sched: scheduler.Scheduler,
    kv_mgr: kv_cache_manager.KVCacheManager,
    req: request_lib.RequestState,
) -> None:
  """Allocates a request's prompt and marks it running."""
  kv_mgr.sync_request_state(req)
  assert kv_mgr.allocate_slots(req, num_new_tokens=len(req.token_ids))
  req.status = request_lib.RequestStatus.RUNNING
  sched._running_requests.append(req)


def _advance_in_flight(
    sched: scheduler.Scheduler,
    scheduled: tuple[request_lib.RequestState, ...],
) -> None:
  """Simulates the engine dispatching `scheduled` to the model runner."""
  for req in scheduled:
    start = req.num_computed_tokens - req.num_in_flight_tokens
    if req.is_chunked_prefill:
      req.num_computed_tokens = min(
          len(req.token_ids), start + sched.chunked_prefill_length
      )
    else:
      req.num_computed_tokens = (
          len(req.token_ids) + sched._config.num_scheduler_steps - 1
      )
    req.num_in_flight_tokens = req.num_computed_tokens - start


class SchedulerConfigTest(parameterized.TestCase):

  def test_valid_config(self):
    config = scheduler.SchedulerConfig(
        max_num_batched_tokens=128,
        max_num_seqs=8,
        max_chunked_prefill_length=16,
        num_scheduler_steps=1,
    )
    self.assertEqual(config.max_num_batched_tokens, 128)
    self.assertEqual(config.max_num_seqs, 8)
    self.assertEqual(config.max_chunked_prefill_length, 16)
    self.assertEqual(config.num_scheduler_steps, 1)

  @parameterized.parameters(0, -2)
  def test_non_positive_max_chunked_prefill_length_raises(self, length):
    with self.assertRaisesRegex(ValueError, "must be positive"):
      scheduler.SchedulerConfig(
          max_num_batched_tokens=128,
          max_num_seqs=8,
          max_chunked_prefill_length=length,
      )

  def test_non_power_of_two_max_chunked_prefill_length_rounds_down(self):
    config = scheduler.SchedulerConfig(
        max_num_batched_tokens=128,
        max_num_seqs=8,
        max_chunked_prefill_length=14,
    )
    sched = scheduler.Scheduler(
        config=config, kv_cache_manager=_create_kv_cache_manager()
    )
    self.assertEqual(config.max_chunked_prefill_length, 14)
    self.assertEqual(sched.chunked_prefill_length, 8)

  def test_max_chunked_prefill_length_exceeds_max_batch_tokens_raises(self):
    with self.assertRaisesRegex(
        ValueError, "must be less than or equal to max_num_batched_tokens"
    ):
      scheduler.SchedulerConfig(
          max_num_batched_tokens=16,
          max_num_seqs=8,
          max_chunked_prefill_length=32,
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
    self.assertEqual(r1.status, request_lib.RequestStatus.PENDING)

  def test_preempt_discards_kv_but_keeps_tokens(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20, 30, 40])
    _admit(self.sched, self.kv_mgr, req)
    req.num_computed_tokens = 4
    req.token_ids.append(90)

    self.sched._preempt()

    # A preempted request must re-prefill from scratch.
    self.assertEqual(req.num_computed_tokens, 0)
    self.assertEqual(req.token_ids, [10, 20, 30, 40, 90])
    self.assertEqual(req.prompt_length, 4)
    self.assertEqual(self.kv_mgr.get_page_idxs(req), {"cache_0": ()})

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

    self.sched.preempt_all()

    self.assertEmpty(self.sched._running_requests)
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

  def test_preempt_all_drops_done_requests(self):
    aborted = _create_request(req_id="a", prompt_token_ids=[10, 20])
    running = _create_request(req_id="r", prompt_token_ids=[30, 40])
    _admit(self.sched, self.kv_mgr, aborted)
    _admit(self.sched, self.kv_mgr, running)
    self.sched.abort_request(aborted)

    self.sched.preempt_all()

    self.assertEqual(list(self.sched._pending_requests), [running])
    self.assertEqual(aborted.status, request_lib.RequestStatus.ABORTED)
    self.assertEqual(self.kv_mgr.get_page_idxs(aborted), {"cache_0": ()})


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
    sched._token_budget = sched._config.max_num_batched_tokens
    return sched

  def _complete_step(self, req: request_lib.RequestState, token: int) -> None:
    """Marks the request's scheduled tokens as computed and samples a token."""
    req.token_ids.append(token)
    req.num_computed_tokens = len(req.token_ids) - 1
    req.num_in_flight_tokens = 0

  def test_allocate_slots_full_prefill(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20, 30, 40])

    self.assertTrue(
        self.sched._try_schedule_request(req, n_admitted_requests=0)
    )

    self.assertFalse(req.is_decode)
    self.assertFalse(req.is_chunked_prefill)

  def test_allocate_slots_chunked_prefill(self):
    req = _create_request(req_id="r1", prompt_token_ids=list(range(32)))

    self.assertTrue(
        self.sched._try_schedule_request(req, n_admitted_requests=0)
    )

    self.assertFalse(req.is_decode)
    self.assertTrue(req.is_chunked_prefill)

  def test_allocate_slots_final_chunk_is_full_prefill(self):
    # Exactly one chunk of tokens remains, so this step samples.
    req = _create_request(req_id="r1", prompt_token_ids=list(range(16)))

    self.assertTrue(
        self.sched._try_schedule_request(req, n_admitted_requests=0)
    )

    self.assertFalse(req.is_decode)
    self.assertFalse(req.is_chunked_prefill)

  def test_allocate_slots_single_token_is_decode(self):
    req = _create_request(req_id="r1", prompt_token_ids=[42])

    self.assertTrue(
        self.sched._try_schedule_request(req, n_admitted_requests=0)
    )

    self.assertTrue(req.is_decode)
    self.assertFalse(req.is_chunked_prefill)

  def test_allocate_slots_skips_prefix_cache_hit(self):
    cached = _create_request(req_id="r1", prompt_token_ids=list(range(12)))
    self.assertTrue(
        self.sched._try_schedule_request(cached, n_admitted_requests=0)
    )
    self._complete_step(cached, 90)
    self.kv_mgr.sync_request_state(cached)

    # The first 3 pages (12 tokens) are cached, so only the last 2 prompt
    # tokens need computing.
    req = _create_request(req_id="r2", prompt_token_ids=list(range(14)))
    self.assertTrue(
        self.sched._try_schedule_request(req, n_admitted_requests=0)
    )

    self.assertEqual(req.num_computed_tokens, 12)
    self.assertFalse(req.is_decode)
    self.assertFalse(req.is_chunked_prefill)

  def test_allocate_slots_insufficient_cache_returns_false(self):
    sched = self._create_scheduler(
        kv_mgr=_create_kv_cache_manager(page_size=4, num_device_pages=1)
    )
    req = _create_request(req_id="r1", prompt_token_ids=list(range(8)))

    self.assertFalse(sched._try_schedule_request(req, n_admitted_requests=0))
    self.assertEqual(req.num_computed_tokens, 0)
    self.assertFalse(req.is_decode)
    self.assertFalse(req.is_chunked_prefill)

  def test_schedule_pending_respects_max_num_seqs(self):
    for i in range(6):
      self.sched._pending_requests.append(
          _create_request(req_id=f"r{i}", prompt_token_ids=[i])
      )

    self.sched._schedule_requests()

    self.assertLen(self.sched._running_requests, 4)
    self.assertLen(self.sched._pending_requests, 2)
    for req in self.sched._running_requests:
      self.assertEqual(req.status, request_lib.RequestStatus.RUNNING)

  def test_schedule_pending_chunked_prefill_needs_a_full_chunk_of_budget(self):
    sched = self._create_scheduler(max_num_batched_tokens=16)
    r0 = _create_request(req_id="r0", prompt_token_ids=list(range(8)))
    req = _create_request(req_id="r1", prompt_token_ids=list(range(32)))
    sched._pending_requests.extend([r0, req])

    sched._schedule_requests()

    # r0 uses 8 of the 16 token budget. req needs a full 16-token chunk, so it
    # stays pending.
    self.assertEqual(list(sched._running_requests), [r0])
    self.assertEqual(list(sched._pending_requests), [req])
    self.assertFalse(req.is_chunked_prefill)

  def test_schedule_pending_stops_at_token_budget(self):
    sched = self._create_scheduler(max_num_batched_tokens=16)
    r1 = _create_request(req_id="r1", prompt_token_ids=list(range(8)))
    r2 = _create_request(req_id="r2", prompt_token_ids=list(range(12)))
    sched._pending_requests.extend([r1, r2])

    sched._schedule_requests()

    # r1 uses 8 of the 16 token budget. r2's 12 tokens don't fit in the
    # remaining 8, so it stays in pending.
    self.assertEqual(list(sched._running_requests), [r1])
    self.assertEqual(list(sched._pending_requests), [r2])
    self.assertFalse(r1.is_decode)
    self.assertFalse(r1.is_chunked_prefill)
    self.assertEqual(sched._token_budget, 8)

  def test_schedule_pending_stops_at_first_request_that_does_not_fit(self):
    sched = self._create_scheduler(
        kv_mgr=_create_kv_cache_manager(page_size=4, num_device_pages=2)
    )
    r1 = _create_request(req_id="r1", prompt_token_ids=list(range(4)))
    r2 = _create_request(req_id="r2", prompt_token_ids=list(range(8)))
    r3 = _create_request(req_id="r3", prompt_token_ids=list(range(4)))
    sched._pending_requests.extend([r1, r2, r3])

    sched._schedule_requests()

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
    sched._schedule_requests()
    self._complete_step(r1, 90)
    self._complete_step(r2, 91)

    # Each decode needs a new page, so r2 (newest) is preempted to make room.
    sched._schedule_requests()

    self.assertEqual(list(sched._running_requests), [r1])
    self.assertEqual(list(sched._pending_requests), [r2])
    self.assertTrue(r1.is_decode)

  def test_schedule_running_preempts_requests_beyond_token_budget(self):
    sched = self._create_scheduler(max_num_batched_tokens=16)
    r1 = _create_request(req_id="r1", prompt_token_ids=list(range(8)))
    r2 = _create_request(req_id="r2", prompt_token_ids=list(range(8, 16)))
    r3 = _create_request(req_id="r3", prompt_token_ids=list(range(16, 24)))
    _admit(sched, self.kv_mgr, r1)
    _admit(sched, self.kv_mgr, r2)
    _admit(sched, self.kv_mgr, r3)

    sched._schedule_requests()

    # r1 and r2 use the full 16 token budget. r3 is preempted back to pending.
    self.assertEqual(list(sched._running_requests), [r1, r2])
    self.assertEqual(list(sched._pending_requests), [r3])
    self.assertEqual(sched._token_budget, 0)

  def test_preempted_running_request_resumes_next_step(self):
    sched = self._create_scheduler(max_num_batched_tokens=16)
    prefill = _create_request(req_id="p", prompt_token_ids=list(range(16)))
    decode = _create_request(req_id="d", prompt_token_ids=[100, 101])
    _admit(sched, self.kv_mgr, prefill)
    _admit(sched, self.kv_mgr, decode)
    self._complete_step(decode, 102)

    # The prefill uses the whole budget, so decode is preempted to pending.
    sched._schedule_requests()

    self.assertEqual(list(sched._running_requests), [prefill])
    self.assertEqual(list(sched._pending_requests), [decode])

    # Once the prefill is done, it only needs 1 decode token, so both fit.
    self._complete_step(prefill, 90)
    sched._schedule_requests()

    self.assertEqual(list(sched._running_requests), [prefill, decode])
    self.assertEmpty(sched._pending_requests)
    self.assertTrue(prefill.is_decode)
    self.assertFalse(decode.is_decode)
    self.assertFalse(decode.is_chunked_prefill)


class ScheduleStepTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.kv_mgr = _create_kv_cache_manager(page_size=4, num_device_pages=20)
    self.config = scheduler.SchedulerConfig(
        max_num_batched_tokens=64,
        max_num_seqs=4,
        max_chunked_prefill_length=16,
        num_scheduler_steps=1,
    )
    self.sched = scheduler.Scheduler(
        config=self.config,
        kv_cache_manager=self.kv_mgr,
    )

  def test_num_active_requests_counts_pending(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20, 30])
    self.sched._pending_requests.append(req)
    self.assertEqual(self.sched.num_active_requests, 1)

  def test_schedule_step_full_prefill(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20, 30, 40])
    scheduled, dist = self.sched.schedule_step([req])

    self.assertLen(scheduled, 1)
    self.assertEqual(scheduled[0].request_id, "r1")
    # 0 decodes, 0 chunked prefills, 1 full prefill -> (0, 0, 1)
    self.assertEqual(dist, (0, 0, 1))
    self.assertFalse(req.is_decode)
    self.assertFalse(req.is_chunked_prefill)

  def test_schedule_step_chunked_prefill(self):
    # Prompt is 32 tokens, but chunked_prefill_length is 16 and
    # max_num_batched_tokens is 16.
    small_config = scheduler.SchedulerConfig(
        max_num_batched_tokens=16,
        max_num_seqs=4,
        max_chunked_prefill_length=16,
        num_scheduler_steps=1,
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

  def test_single_token_prompt_schedules_as_decode(self):
    req = _create_request(req_id="r1", prompt_token_ids=[42])
    scheduled, dist = self.sched.schedule_step([req])

    self.assertLen(scheduled, 1)
    # 1 decode, 0 chunked, 0 full prefill -> (1, 1, 1)
    self.assertEqual(dist, (1, 1, 1))
    self.assertTrue(req.is_decode)
    self.assertFalse(req.is_chunked_prefill)

  def test_schedule_step_orders_decodes_then_chunked_then_prefills(self):
    prefill = _create_request(req_id="p", prompt_token_ids=[10, 20, 30])
    chunked = _create_request(req_id="c", prompt_token_ids=list(range(32)))
    decode = _create_request(req_id="d", prompt_token_ids=[42])

    scheduled, dist = self.sched.schedule_step([prefill, chunked, decode])

    self.assertEqual(scheduled, (decode, chunked, prefill))
    # 1 decode, 1 chunked, 1 full prefill -> (1, 2, 3)
    self.assertEqual(dist, (1, 2, 3))

  def test_max_num_seqs_limit(self):
    # Config max_num_seqs is 4
    reqs = [
        _create_request(req_id=f"r{i}", prompt_token_ids=[i])
        for i in range(6)
    ]
    scheduled, _ = self.sched.schedule_step(reqs)

    self.assertLen(scheduled, 4)
    self.assertEqual(self.sched.num_active_requests, 6)
    self.assertLen(self.sched._pending_requests, 2)
    self.assertLen(self.sched._running_requests, 4)

  def test_token_budget_exhaustion(self):
    config = scheduler.SchedulerConfig(
        max_num_batched_tokens=16,
        max_num_seqs=4,
        max_chunked_prefill_length=16,
        num_scheduler_steps=1,
    )
    sched = scheduler.Scheduler(config=config, kv_cache_manager=self.kv_mgr)

    req1 = _create_request(req_id="r1", prompt_token_ids=list(range(8)))
    req2 = _create_request(req_id="r2", prompt_token_ids=list(range(100, 112)))

    scheduled, dist = sched.schedule_step([req1, req2])
    # req1 uses 8 tokens, leaving 8.
    # req2 needs 12 tokens, which exceeds remaining budget 8, so it stays in
    # pending.
    self.assertLen(scheduled, 1)
    self.assertEqual(scheduled[0].request_id, "r1")
    self.assertEqual(dist, (0, 0, 1))
    self.assertEqual(list(sched._running_requests), [req1])
    self.assertEqual(list(sched._pending_requests), [req2])

  def test_aborted_pending_request_is_never_scheduled(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20, 30])
    self.sched._pending_requests.append(req)

    self.sched.abort_request(req)
    scheduled, _ = self.sched.schedule_step([])

    self.assertEmpty(scheduled)
    self.assertEqual(self.sched.num_active_requests, 0)
    self.assertEmpty(self.sched._pending_requests)

  def test_aborted_running_request_is_released_next_step(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20, 30, 40])
    self.sched.schedule_step([req])

    self.sched.abort_request(req)
    # The request is no longer active, but keeps its pages until the next step.
    self.assertEqual(self.sched.num_active_requests, 0)
    self.assertNotEqual(self.kv_mgr.get_page_idxs(req), {"cache_0": ()})
    scheduled, _ = self.sched.schedule_step([])

    self.assertEmpty(scheduled)
    self.assertEmpty(self.sched._running_requests)
    self.assertEqual(self.kv_mgr.get_page_idxs(req), {"cache_0": ()})

  def test_abort_after_finish_keeps_finished_status(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20])
    req.status = request_lib.RequestStatus.FINISHED_EOS

    self.sched.abort_request(req)

    self.assertEqual(req.status, request_lib.RequestStatus.FINISHED_EOS)


class UpdateFromOutputTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.kv_mgr = _create_kv_cache_manager(page_size=4, num_device_pages=20)
    self.config = scheduler.SchedulerConfig(
        max_num_batched_tokens=64,
        max_num_seqs=4,
        max_chunked_prefill_length=16,
        num_scheduler_steps=1,
    )
    self.sched = scheduler.Scheduler(
        config=self.config,
        kv_cache_manager=self.kv_mgr,
    )

  def _scheduler_with_eos(
      self, eos_token_ids: frozenset[int]
  ) -> scheduler.Scheduler:
    return scheduler.Scheduler(
        config=dataclasses.replace(self.config, eos_token_ids=eos_token_ids),
        kv_cache_manager=self.kv_mgr,
    )

  def test_continuation_of_chunked_prefill(self):
    small_config = scheduler.SchedulerConfig(
        max_num_batched_tokens=16,
        max_num_seqs=4,
        max_chunked_prefill_length=16,
        num_scheduler_steps=1,
    )
    sched = scheduler.Scheduler(
        config=small_config,
        kv_cache_manager=self.kv_mgr,
    )
    prompt = list(range(32))
    req = _create_request(req_id="r1", prompt_token_ids=prompt)

    # Step 1: schedules first chunk of 16 tokens as chunked prefill
    scheduled1, dist1 = sched.schedule_step([req])
    self.assertEqual(dist1, (0, 1, 1))
    self.assertTrue(req.is_chunked_prefill)
    self.assertFalse(req.is_decode)

    # Engine step executes: update from output for chunk 1
    _advance_in_flight(sched, scheduled1)
    sched.update_from_output(scheduled1, generated_tokens=np.array([[0]]))
    self.assertEqual(req.num_computed_tokens, 16)

    # Step 2: schedules second chunk of 16 tokens as full prefill
    scheduled2, dist2 = sched.schedule_step([])
    self.assertLen(scheduled2, 1)
    self.assertEqual(scheduled2[0].request_id, "r1")
    self.assertEqual(dist2, (0, 0, 1))
    self.assertFalse(req.is_chunked_prefill)
    self.assertFalse(req.is_decode)

  def test_schedule_step_decode_requests(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20, 30, 40])
    scheduled, dist = self.sched.schedule_step([req])
    self.assertEqual(dist, (0, 0, 1))

    # Complete prefill + first decode token
    _advance_in_flight(self.sched, scheduled)
    self.sched.update_from_output(scheduled, generated_tokens=np.array([[99]]))
    self.assertIn(99, req.token_ids)

    # Next step: should schedule as decode request
    scheduled2, dist2 = self.sched.schedule_step([])
    self.assertLen(scheduled2, 1)
    # 1 decode, 0 chunked, 0 full prefill -> (1, 1, 1)
    self.assertEqual(dist2, (1, 1, 1))
    self.assertTrue(req.is_decode)
    self.assertFalse(req.is_chunked_prefill)

  def test_request_aborted_mid_step_is_discarded(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20, 30, 40])
    scheduled, _ = self.sched.schedule_step([req])
    _advance_in_flight(self.sched, scheduled)

    self.sched.abort_request(req)
    completed = self.sched.update_from_output(
        scheduled, generated_tokens=np.array([[99]])
    )

    self.assertEmpty(completed)
    self.assertNotIn(99, req.token_ids)
    self.assertEqual(self.sched.num_active_requests, 0)

    # Pages are released when the request is dropped at the next step.
    self.sched.schedule_step([])
    self.assertEqual(self.kv_mgr.get_page_idxs(req), {"cache_0": ()})

  def test_schedule_ordering_and_distribution(self):
    # Enqueue req1 and step to make it decode
    req1 = _create_request(req_id="r1", prompt_token_ids=[10, 20])
    scheduled, _ = self.sched.schedule_step([req1])
    _advance_in_flight(self.sched, scheduled)
    self.sched.update_from_output(scheduled, generated_tokens=np.array([[99]]))

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
    sched = self._scheduler_with_eos(frozenset({1}))
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20])
    scheduled, _ = sched.schedule_step([req])
    _advance_in_flight(sched, scheduled)

    completed = sched.update_from_output(
        scheduled,
        generated_tokens=np.array([[1, 7]]),
        logprobs=np.array([[-0.5, -0.25]]),
    )

    self.assertEqual(completed, [req])
    self.assertEqual(req.status, request_lib.RequestStatus.FINISHED_EOS)
    # The EOS token is kept. Tokens sampled after it are dropped.
    self.assertEqual(req.token_ids, [10, 20, 1])
    self.assertEqual(req.logprobs, [-0.5])
    self.assertEqual(sched.num_active_requests, 0)

    # Pages are released when the request is dropped at the next step.
    self.assertNotEqual(self.kv_mgr.get_page_idxs(req), {"cache_0": ()})
    sched.schedule_step([])
    self.assertEmpty(sched._running_requests)
    self.assertEqual(self.kv_mgr.get_page_idxs(req), {"cache_0": ()})

  def test_update_from_output_truncation(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20], max_tokens=2)
    scheduled, _ = self.sched.schedule_step([req])
    _advance_in_flight(self.sched, scheduled)

    # Step 1: 1 generated token
    completed = self.sched.update_from_output(
        scheduled, generated_tokens=np.array([[5]])
    )
    self.assertEmpty(completed)

    # Step 2: 1 generated token reaches max_tokens (2)
    scheduled, _ = self.sched.schedule_step([])
    _advance_in_flight(self.sched, scheduled)
    completed = self.sched.update_from_output(
        scheduled, generated_tokens=np.array([[6]])
    )
    self.assertEqual(completed, [req])
    self.assertEqual(req.status, request_lib.RequestStatus.FINISHED_LENGTH)
    self.assertEqual(self.sched.num_active_requests, 0)
    self.assertEqual(len(req.token_ids), 4)  # 2 prompt + 2 generated

    self.sched.schedule_step([])
    self.assertEqual(self.kv_mgr.get_page_idxs(req), {"cache_0": ()})

  def test_length_limit_takes_precedence_over_later_eos(self):
    sched = self._scheduler_with_eos(frozenset({1}))
    req = _create_request(req_id="r1", prompt_token_ids=[10], max_tokens=1)
    scheduled, _ = sched.schedule_step([req])
    _advance_in_flight(sched, scheduled)

    # The EOS token is sampled after the length limit has been reached.
    completed = sched.update_from_output(
        scheduled, generated_tokens=np.array([[5, 1]])
    )

    self.assertEqual(completed, [req])
    self.assertEqual(req.status, request_lib.RequestStatus.FINISHED_LENGTH)
    self.assertEqual(req.token_ids, [10, 5])

  def test_eos_at_length_limit_finishes_as_eos(self):
    sched = self._scheduler_with_eos(frozenset({1}))
    req = _create_request(req_id="r1", prompt_token_ids=[10], max_tokens=2)
    scheduled, _ = sched.schedule_step([req])
    _advance_in_flight(sched, scheduled)

    completed = sched.update_from_output(
        scheduled, generated_tokens=np.array([[5, 1]])
    )

    self.assertEqual(completed, [req])
    self.assertEqual(req.status, request_lib.RequestStatus.FINISHED_EOS)
    self.assertEqual(req.token_ids, [10, 5, 1])

  def test_update_from_output_applies_rows_in_given_order(self):
    r1 = _create_request(req_id="r1", prompt_token_ids=[10, 20])
    r2 = _create_request(req_id="r2", prompt_token_ids=[30, 40])
    scheduled, _ = self.sched.schedule_step([r1, r2])
    _advance_in_flight(self.sched, scheduled)

    # Rows follow the order of `requests`, not the order they were scheduled.
    self.sched.update_from_output(
        [r2, r1], generated_tokens=np.array([[200], [100]])
    )

    self.assertEqual(r1.token_ids, [10, 20, 100])
    self.assertEqual(r2.token_ids, [30, 40, 200])

  @parameterized.named_parameters(
      dict(
          testcase_name="generated_tokens",
          generated_tokens=np.array([[1], [2]]),
          logits=None,
          logprobs=None,
      ),
      dict(
          testcase_name="logits",
          generated_tokens=np.array([[1]]),
          logits=np.zeros((2, 1, 2)),
          logprobs=None,
      ),
      dict(
          testcase_name="logprobs",
          generated_tokens=np.array([[1]]),
          logits=None,
          logprobs=np.zeros((2, 1)),
      ),
  )
  def test_update_from_output_row_mismatch_raises(
      self, generated_tokens, logits, logprobs
  ):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20])
    scheduled, _ = self.sched.schedule_step([req])

    with self.assertRaisesRegex(ValueError, "rows for 1 requests"):
      self.sched.update_from_output(
          scheduled,
          generated_tokens=generated_tokens,
          logits=logits,
          logprobs=logprobs,
      )

  def test_preemption_on_cache_exhaustion(self):
    tiny_kv_mgr = _create_kv_cache_manager(page_size=4, num_device_pages=2)
    config = scheduler.SchedulerConfig(
        max_num_batched_tokens=32,
        max_num_seqs=4,
        max_chunked_prefill_length=16,
        num_scheduler_steps=1,
    )
    sched = scheduler.Scheduler(config=config, kv_cache_manager=tiny_kv_mgr)

    # Both req1 and req2 use 4 tokens (1 page each) -> cache is now full (2 pages used)
    req1 = _create_request(req_id="r1", prompt_token_ids=list(range(4)))
    req2 = _create_request(req_id="r2", prompt_token_ids=list(range(4)))
    scheduled, _ = sched.schedule_step([req1, req2])
    _advance_in_flight(sched, scheduled)
    sched.update_from_output(scheduled, generated_tokens=np.array([[90], [91]]))

    # Now in next step, both req1 and req2 want to decode (+1 token each).
    # Since pages are full, req2 (newest) must be preempted to free up space!
    scheduled, _ = sched.schedule_step([])
    # req1 is scheduled, req2 was preempted and pushed to pending
    self.assertEqual([r.request_id for r in scheduled], ["r1"])
    self.assertIn(req2, sched._pending_requests)
    self.assertEqual(req2.status, request_lib.RequestStatus.PENDING)

  def test_sequential_scheduling_steps(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20, 30, 40])
    # Step 1: prefill
    scheduled, dist = self.sched.schedule_step([req])
    self.assertEqual(req.status, request_lib.RequestStatus.RUNNING)
    self.assertEqual(dist, (0, 0, 1))
    self.assertFalse(req.is_decode)
    self.assertFalse(req.is_chunked_prefill)
    _advance_in_flight(self.sched, scheduled)
    self.sched.update_from_output(scheduled, generated_tokens=np.array([[90]]))
    self.assertEqual(req.num_computed_tokens, 4)

    # Step 2: decode step 1
    scheduled, dist = self.sched.schedule_step([])
    self.assertEqual(dist, (1, 1, 1))
    self.assertTrue(req.is_decode)
    self.assertFalse(req.is_chunked_prefill)
    _advance_in_flight(self.sched, scheduled)
    self.sched.update_from_output(scheduled, generated_tokens=np.array([[91]]))
    self.assertEqual(req.num_computed_tokens, 5)

    # Step 3: decode step 2
    scheduled, dist = self.sched.schedule_step([])
    self.assertEqual(dist, (1, 1, 1))
    self.assertTrue(req.is_decode)
    self.assertFalse(req.is_chunked_prefill)
    _advance_in_flight(self.sched, scheduled)
    self.sched.update_from_output(scheduled, generated_tokens=np.array([[92]]))
    self.assertEqual(req.num_computed_tokens, 6)
    self.assertEqual(req.token_ids, [10, 20, 30, 40, 90, 91, 92])

  def test_prefix_caching_computed_pages(self):
    # Page size is 4. Prompt with 12 tokens hashes 2 full pages (withholding last token leaves 11 -> 2 pages).
    req1 = _create_request(req_id="r1", prompt_token_ids=list(range(12)))
    scheduled, _ = self.sched.schedule_step([req1])
    _advance_in_flight(self.sched, scheduled)
    self.sched.update_from_output(scheduled, generated_tokens=np.array([[90]]))

    # Second request shares the exact same prefix tokens
    req2 = _create_request(req_id="r2", prompt_token_ids=list(range(14)))
    # Scheduling runs r1 first, which syncs its now-full third page into the
    # cache, so r2 matches 3 pages rather than the 2 that were full at r1's
    # first schedule.
    scheduled, dist = self.sched.schedule_step([req2])
    self.assertIn(req2, scheduled)
    self.assertEqual(dist, (1, 1, 2))
    # 3 pages (12 tokens) hit prefix cache, so num_computed_tokens advances
    # by 12.
    self.assertEqual(req2.num_computed_tokens, 12)
    self.assertFalse(req2.is_decode)
    self.assertFalse(req2.is_chunked_prefill)

  def test_update_from_output_with_logits_and_logprobs(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20])
    scheduled, _ = self.sched.schedule_step([req])
    _advance_in_flight(self.sched, scheduled)

    logits = np.array([[[0.1, 0.9]]])
    logprobs = np.array([[-0.1]])

    self.sched.update_from_output(
        scheduled,
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
    scheduled, _ = self.sched.schedule_step([req])
    _advance_in_flight(self.sched, scheduled)
    self.sched.update_from_output(scheduled, generated_tokens=np.array([[90]]))
    # Step 2: decode step 1 (1 token)
    scheduled, _ = self.sched.schedule_step([])
    _advance_in_flight(self.sched, scheduled)
    self.sched.update_from_output(scheduled, generated_tokens=np.array([[91]]))
    self.assertEqual(req.num_computed_tokens, 5)
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
    self.assertFalse(req.is_chunked_prefill)
    # r1's own first page was cached before it was preempted, so resuming only
    # re-prefills the 2 tokens the cache does not cover, not all 6.
    self.assertEqual(req.num_computed_tokens, 4)

  def test_prefix_cache_hit_leaving_single_token_schedules_as_decode(self):
    # Page size is 4. Prompt with 8 tokens creates 2 full pages.
    # Note: prefix caching withholds last token, so prompt of 9 tokens hashes 8 tokens (2 pages).
    req1 = _create_request(req_id="r1", prompt_token_ids=list(range(9)))
    scheduled, _ = self.sched.schedule_step([req1])
    _advance_in_flight(self.sched, scheduled)
    self.sched.update_from_output(scheduled, generated_tokens=np.array([[99]]))
    self.kv_mgr.sync_request_state(req1)

    # req2 has the exact same 9 tokens: 8 tokens hit prefix cache, leaving 1 unprocessed token.
    req2 = _create_request(req_id="r2", prompt_token_ids=list(range(9)))
    scheduled, dist = self.sched.schedule_step([req2])
    self.assertLen(scheduled, 2)
    self.assertEqual(scheduled, (req1, req2))
    # Both req1 and req2 are scheduled as decodes: (2, 2, 2)
    self.assertEqual(dist, (2, 2, 2))
    self.assertTrue(req2.is_decode)
    self.assertFalse(req2.is_chunked_prefill)
    self.assertEqual(req2.num_computed_tokens, 8)

  def test_per_request_max_tokens(self):
    req1 = _create_request(req_id="r1", prompt_token_ids=[10], max_tokens=2)
    req2 = _create_request(req_id="r2", prompt_token_ids=[20], max_tokens=4)
    scheduled, _ = self.sched.schedule_step([req1, req2])
    _advance_in_flight(self.sched, scheduled)
    # Step 1
    completed = self.sched.update_from_output(
        scheduled, generated_tokens=np.array([[101], [201]])
    )
    self.assertEmpty(completed)

    # Step 2: req1 completes after 2 generated tokens
    scheduled, _ = self.sched.schedule_step([])
    _advance_in_flight(self.sched, scheduled)
    completed = self.sched.update_from_output(
        scheduled, generated_tokens=np.array([[102], [202]])
    )
    self.assertLen(completed, 1)
    self.assertEqual(completed[0].request_id, "r1")
    self.assertEqual(completed[0].token_ids, [10, 101, 102])

    # Step 3: req2 continues
    scheduled, _ = self.sched.schedule_step([])
    _advance_in_flight(self.sched, scheduled)
    completed = self.sched.update_from_output(
        scheduled, generated_tokens=np.array([[203]])
    )
    self.assertEmpty(completed)

    # Step 4: req2 completes after 4 generated tokens
    scheduled, _ = self.sched.schedule_step([])
    _advance_in_flight(self.sched, scheduled)
    completed = self.sched.update_from_output(
        scheduled, generated_tokens=np.array([[204]])
    )
    self.assertLen(completed, 1)
    self.assertEqual(completed[0].request_id, "r2")
    self.assertEqual(completed[0].token_ids, [20, 201, 202, 203, 204])

  def test_any_configured_eos_token_terminates(self):
    sched = self._scheduler_with_eos(frozenset({1, 99}))
    req1 = _create_request(req_id="r1", prompt_token_ids=[10])
    req2 = _create_request(req_id="r2", prompt_token_ids=[20])
    scheduled, _ = sched.schedule_step([req1, req2])
    _advance_in_flight(sched, scheduled)

    completed = sched.update_from_output(
        scheduled, generated_tokens=np.array([[5], [1]])
    )
    self.assertEqual(completed, [req2])

    scheduled, _ = sched.schedule_step([])
    _advance_in_flight(sched, scheduled)
    completed = sched.update_from_output(
        scheduled, generated_tokens=np.array([[99]])
    )
    self.assertEqual(completed, [req1])

  def test_request_without_eos_never_terminates_on_a_token(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10])
    scheduled, _ = self.sched.schedule_step([req])
    _advance_in_flight(self.sched, scheduled)

    completed = self.sched.update_from_output(
        scheduled, generated_tokens=np.array([[1]])
    )

    self.assertEmpty(completed)
    self.assertEqual(req.token_ids, [10, 1])


class PreemptAllTest(parameterized.TestCase):
  """Tests preempt_all across full scheduling steps."""

  def setUp(self):
    super().setUp()
    self.kv_mgr = _create_kv_cache_manager(page_size=4, num_device_pages=20)
    self.config = scheduler.SchedulerConfig(
        max_num_batched_tokens=64,
        max_num_seqs=4,
        max_chunked_prefill_length=16,
        num_scheduler_steps=1,
    )
    self.sched = scheduler.Scheduler(
        config=self.config,
        kv_cache_manager=self.kv_mgr,
    )

  def _run_to_decode(self, reqs):
    """Schedules `reqs` and steps once so each is mid-decode."""
    scheduled, _ = self.sched.schedule_step(reqs)
    _advance_in_flight(self.sched, scheduled)
    self.sched.update_from_output(
        scheduled,
        generated_tokens=np.array([[100 + i] for i in range(len(reqs))]),
    )
    self.sched.schedule_step([])

  def test_preempted_request_stays_logprob_aligned(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20, 30, 40])
    scheduled, _ = self.sched.schedule_step([req])
    _advance_in_flight(self.sched, scheduled)
    self.sched.update_from_output(
        scheduled,
        generated_tokens=np.array([[101]]),
        logprobs=np.array([[-0.5]]),
    )
    self.sched.schedule_step([])
    self.assertLen(req.logprobs, 1)

    self.sched.preempt_all()

    # Rescheduling must re-cover the whole context. The prefix cache serves the
    # first full page (4 tokens) back, and the step yields one new token.
    scheduled, dist = self.sched.schedule_step([])
    self.assertLen(scheduled, 1)
    self.assertEqual(req.num_computed_tokens, 4)
    self.assertTrue(req.is_decode)
    self.assertEqual(dist, (1, 1, 1))

    _advance_in_flight(self.sched, scheduled)
    self.sched.update_from_output(
        scheduled,
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
    self.assertFalse(scheduled[0].is_chunked_prefill)
    self.assertEqual(dist, (0, 0, 1))
    self.assertEqual(req.num_computed_tokens, 0)

  def test_preempt_all_respects_max_tokens_accounting(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20], max_tokens=2)
    scheduled, _ = self.sched.schedule_step([req])
    _advance_in_flight(self.sched, scheduled)
    self.sched.update_from_output(scheduled, generated_tokens=np.array([[101]]))
    self.sched.schedule_step([])

    self.sched.preempt_all()

    # One token of the 2-token budget is already spent, so the re-prefilled
    # request must finish after exactly one more.
    scheduled, _ = self.sched.schedule_step([])
    _advance_in_flight(self.sched, scheduled)
    finished = self.sched.update_from_output(
        scheduled, generated_tokens=np.array([[102]])
    )
    self.assertLen(finished, 1)
    self.assertEqual(req.token_ids, [10, 20, 101, 102])


class ColocatedSchedulingTest(parameterized.TestCase):
  """Tests scheduling the next step while the current step is in flight."""

  def setUp(self):
    super().setUp()
    self.kv_mgr = _create_kv_cache_manager(page_size=4, num_device_pages=20)
    self.config = scheduler.SchedulerConfig(
        max_num_batched_tokens=64,
        max_num_seqs=4,
        max_chunked_prefill_length=16,
        num_scheduler_steps=1,
        eos_token_ids=frozenset({999}),
    )
    self.sched = scheduler.Scheduler(
        config=self.config,
        kv_cache_manager=self.kv_mgr,
    )

  def test_advance_in_flight_schedules_decode_for_next_step(self):
    req = _create_request(req_id="r1", prompt_token_ids=[10, 20, 30, 40])
    scheduled1, dist1 = self.sched.schedule_step([req])
    self.assertEqual(dist1, (0, 0, 1))

    _advance_in_flight(self.sched, scheduled1)
    scheduled2, dist2 = self.sched.schedule_step([])
    self.assertEqual(scheduled2, (req,))
    self.assertEqual(dist2, (1, 1, 1))
    self.assertTrue(req.is_decode)
    self.assertFalse(req.is_chunked_prefill)
    self.assertEqual(req.num_computed_tokens, 4)

    finished = self.sched.update_from_output(
        scheduled1, generated_tokens=np.array([[101]])
    )
    self.assertEmpty(finished)
    self.assertEqual(req.token_ids, [10, 20, 30, 40, 101])
    self.assertEqual(req.num_computed_tokens, 4)

  def test_advance_in_flight_continues_chunked_prefill(self):
    sched = scheduler.Scheduler(
        config=dataclasses.replace(self.config, max_num_batched_tokens=16),
        kv_cache_manager=self.kv_mgr,
    )
    req = _create_request(req_id="r1", prompt_token_ids=list(range(24)))
    scheduled1, dist1 = sched.schedule_step([req])
    self.assertEqual(dist1, (0, 1, 1))

    # While chunk 1 (16 tokens) is in flight, chunk 2 (the remaining 8 tokens)
    # is scheduled as a full prefill.
    _advance_in_flight(sched, scheduled1)
    scheduled2, dist2 = sched.schedule_step([])
    self.assertEqual(scheduled2, (req,))
    self.assertEqual(dist2, (0, 0, 1))
    self.assertFalse(req.is_chunked_prefill)
    self.assertFalse(req.is_decode)
    self.assertEqual(req.num_computed_tokens, 16)

    # Chunk 1's output arrives and must not overwrite chunk 2's schedule.
    sched.update_from_output(scheduled1, generated_tokens=np.array([[0]]))
    self.assertEqual(req.num_computed_tokens, 16)
    self.assertLen(req.token_ids, 24)

  def test_drop_completed_removes_finished_requests_and_updates_distribution(
      self,
  ):
    r1 = _create_request(req_id="r1", prompt_token_ids=[10, 20])
    r2 = _create_request(req_id="r2", prompt_token_ids=list(range(24)))
    r3 = _create_request(req_id="r3", prompt_token_ids=[30, 40])

    scheduled1, _ = self.sched.schedule_step([r1])
    _advance_in_flight(self.sched, scheduled1)
    # Schedule step 2 with r1 (decode), r2 (chunked prefill), r3 (full prefill).
    scheduled2, dist2 = self.sched.schedule_step([r2, r3])
    self.assertEqual(scheduled2, (r1, r2, r3))
    self.assertEqual(dist2, (1, 2, 3))

    # r1 samples EOS in step 1.
    finished = self.sched.update_from_output(
        scheduled1, generated_tokens=np.array([[999]])
    )
    self.assertEqual(finished, [r1])

    scheduled2, dist2 = self.sched.drop_completed(scheduled2, dist2)
    self.assertEqual(scheduled2, (r2, r3))
    self.assertEqual(dist2, (0, 1, 2))
    self.assertEqual(self.kv_mgr.get_page_idxs(r1), {"cache_0": ()})

  def test_preemption_while_in_flight_keeps_sampled_token_and_resumes(self):
    tiny_kv_mgr = _create_kv_cache_manager(page_size=4, num_device_pages=2)
    sched = scheduler.Scheduler(config=self.config, kv_cache_manager=tiny_kv_mgr)
    r1 = _create_request(
        req_id="r1", prompt_token_ids=[1, 2, 3, 4], max_tokens=2
    )
    r2 = _create_request(
        req_id="r2", prompt_token_ids=[5, 6, 7, 8], max_tokens=2
    )

    scheduled1, _ = sched.schedule_step([r1, r2])
    _advance_in_flight(sched, scheduled1)
    # Step 2 needs a second page for r1, which preempts r2 while step 1 is still
    # in flight.
    scheduled2, _ = sched.schedule_step([])
    self.assertEqual(scheduled2, (r1,))
    self.assertEqual(r2.status, request_lib.RequestStatus.PENDING)

    # Step 1 finishes: r2 still receives the token it sampled in step 1, while
    # staying pending with 0 computed KV tokens.
    sched.update_from_output(
        scheduled1, generated_tokens=np.array([[101], [201]])
    )
    self.assertEqual(r2.token_ids, [5, 6, 7, 8, 201])
    self.assertEqual(r2.num_computed_tokens, 0)

    # Step 2 finishes r1, freeing the cache so r2 can re-prefill [5, 6, 7, 8, 201].
    _advance_in_flight(sched, scheduled2)
    sched.schedule_step([])
    sched.update_from_output(scheduled2, generated_tokens=np.array([[102]]))
    sched.drop_completed((), (0, 0, 0))

    scheduled3, dist3 = sched.schedule_step([])
    self.assertEqual(scheduled3, (r2,))
    self.assertEqual(dist3, (0, 0, 1))
    self.assertFalse(r2.is_decode)
    self.assertFalse(r2.is_chunked_prefill)

  def test_preemption_while_in_flight_completes_if_sampled_token_finishes_request(
      self,
  ):
    tiny_kv_mgr = _create_kv_cache_manager(page_size=4, num_device_pages=2)
    sched = scheduler.Scheduler(config=self.config, kv_cache_manager=tiny_kv_mgr)
    r1 = _create_request(
        req_id="r1", prompt_token_ids=[1, 2, 3, 4], max_tokens=2
    )
    r2 = _create_request(
        req_id="r2", prompt_token_ids=[5, 6, 7, 8], max_tokens=1
    )

    scheduled1, _ = sched.schedule_step([r1, r2])
    _advance_in_flight(sched, scheduled1)
    # Step 2 needs a second page for r1, which preempts r2 while step 1 is still
    # in flight.
    scheduled2, _ = sched.schedule_step([])
    self.assertEqual(scheduled2, (r1,))
    self.assertEqual(r2.status, request_lib.RequestStatus.PENDING)

    # Step 1 finishes: r2 processes its generated token and finishes immediately.
    finished = sched.update_from_output(
        scheduled1, generated_tokens=np.array([[101], [201]])
    )
    self.assertEqual(finished, [r2])
    self.assertEqual(r2.status, request_lib.RequestStatus.FINISHED_LENGTH)
    self.assertEqual(r2.token_ids, [5, 6, 7, 8, 201])
    self.assertEqual(r2.num_computed_tokens, 0)

  def test_chunked_prefill_preempted_while_in_flight_discards_output(self):
    tiny_kv_mgr = _create_kv_cache_manager(page_size=4, num_device_pages=5)
    sched = scheduler.Scheduler(config=self.config, kv_cache_manager=tiny_kv_mgr)
    # r1 uses 1 page (4 tokens); r2's first chunk of 16 tokens uses 4 pages -> all 5 pages full.
    r1 = _create_request(
        req_id="r1", prompt_token_ids=[1, 2, 3, 4], max_tokens=2
    )
    r2 = _create_request(
        req_id="r2", prompt_token_ids=list(range(10, 34)), max_tokens=2
    )

    scheduled1, dist1 = sched.schedule_step([r1, r2])
    self.assertEqual(dist1, (0, 1, 2))
    _advance_in_flight(sched, scheduled1)

    # Step 2 needs a second page for r1, which preempts r2 while r2's chunked
    # prefill is still in flight.
    scheduled2, _ = sched.schedule_step([])
    self.assertEqual(scheduled2, (r1,))
    self.assertEqual(r2.status, request_lib.RequestStatus.PENDING)

    # Step 1 finishes: r2 was a chunked prefill so it must not append the dummy
    # output token.
    sched.update_from_output(
        scheduled1, generated_tokens=np.array([[201], [101]])
    )
    self.assertEqual(r1.token_ids, [1, 2, 3, 4, 101])
    self.assertEqual(r2.token_ids, list(range(10, 34)))
    self.assertEqual(r2.num_computed_tokens, 0)


if __name__ == "__main__":
  absltest.main()

