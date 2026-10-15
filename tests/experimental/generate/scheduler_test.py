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
      max_num_batched_tokens=64,
      max_num_seqs=4,
      max_chunked_prefill_length=16,
      num_scheduler_steps=1,
  )
  kwargs.update(overrides)
  return scheduler.SchedulerConfig(**kwargs)


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
        request_lib.RequestState(req_id="r0", prompt_token_ids=[10, 20]),
    )
    self.sched._pending_requests.append(
        request_lib.RequestState(req_id="r1", prompt_token_ids=[30])
    )
    self.assertEqual(self.sched.num_active_requests, 2)

  def test_preempt_moves_newest_request_to_front_of_pending(self):
    r0 = request_lib.RequestState(req_id="r0", prompt_token_ids=[10, 20])
    r1 = request_lib.RequestState(req_id="r1", prompt_token_ids=[30, 40])
    pending = request_lib.RequestState(req_id="p", prompt_token_ids=[50])
    _admit(self.sched, self.kv_mgr, r0)
    _admit(self.sched, self.kv_mgr, r1)
    self.sched._pending_requests.append(pending)

    self.sched._preempt()

    self.assertEqual(list(self.sched._running_requests), [r0])
    self.assertEqual(list(self.sched._pending_requests), [r1, pending])
    self.assertEqual(r1.status, request_lib.RequestStatus.PENDING)

  def test_preempt_discards_kv_but_keeps_tokens(self):
    req = request_lib.RequestState(req_id="r1", prompt_token_ids=[10, 20, 30, 40])
    _admit(self.sched, self.kv_mgr, req)
    req.num_computed_tokens = 4
    req.token_ids.append(90)

    self.sched._preempt()

    # A preempted request must re-prefill from scratch.
    self.assertEqual(req.num_computed_tokens, 0)
    self.assertEqual(req.token_ids, [10, 20, 30, 40, 90])
    self.assertEqual(self.kv_mgr.get_page_idxs(req), {"cache_0": ()})

  def test_preempt_all_on_empty_scheduler_is_a_noop(self):
    self.sched.preempt_all()
    self.assertEqual(self.sched.num_active_requests, 0)

  def test_preempt_all_requeues_every_running_request_in_arrival_order(self):
    reqs = [
        request_lib.RequestState(req_id=f"r{i}", prompt_token_ids=[10 * i, 20, 30])
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
    req = request_lib.RequestState(req_id="r1", prompt_token_ids=list(range(12)))
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
    sched._token_budget = sched._config.max_num_batched_tokens
    return sched

  def _complete_step(self, req: request_lib.RequestState, token: int) -> None:
    """Marks the request's scheduled tokens as computed and samples a token."""
    req.token_ids.append(token)
    req.num_computed_tokens = len(req.token_ids) - 1
    req.num_in_flight_tokens = 0

  def test_allocate_slots_full_prefill(self):
    req = request_lib.RequestState(req_id="r1", prompt_token_ids=[10, 20, 30, 40])

    self.assertTrue(
        self.sched._try_schedule_request(req, n_admitted_requests=0)
    )

    self.assertFalse(req.is_decode)
    self.assertFalse(req.is_chunked_prefill)

  def test_allocate_slots_chunked_prefill(self):
    req = request_lib.RequestState(req_id="r1", prompt_token_ids=list(range(32)))

    self.assertTrue(
        self.sched._try_schedule_request(req, n_admitted_requests=0)
    )

    self.assertFalse(req.is_decode)
    self.assertTrue(req.is_chunked_prefill)

  def test_allocate_slots_final_chunk_is_full_prefill(self):
    # Exactly one chunk of tokens remains, so this step samples.
    req = request_lib.RequestState(req_id="r1", prompt_token_ids=list(range(16)))

    self.assertTrue(
        self.sched._try_schedule_request(req, n_admitted_requests=0)
    )

    self.assertFalse(req.is_decode)
    self.assertFalse(req.is_chunked_prefill)

  def test_allocate_slots_single_token_is_decode(self):
    req = request_lib.RequestState(req_id="r1", prompt_token_ids=[42])

    self.assertTrue(
        self.sched._try_schedule_request(req, n_admitted_requests=0)
    )

    self.assertTrue(req.is_decode)
    self.assertFalse(req.is_chunked_prefill)

  def test_allocate_slots_skips_prefix_cache_hit(self):
    cached = request_lib.RequestState(req_id="r1", prompt_token_ids=list(range(12)))
    self.assertTrue(
        self.sched._try_schedule_request(cached, n_admitted_requests=0)
    )
    self._complete_step(cached, 90)
    self.kv_mgr.sync_request_state(cached)

    # The first 3 pages (12 tokens) are cached, so only the last 2 prompt
    # tokens need computing.
    req = request_lib.RequestState(req_id="r2", prompt_token_ids=list(range(14)))
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
    req = request_lib.RequestState(req_id="r1", prompt_token_ids=list(range(8)))

    self.assertFalse(sched._try_schedule_request(req, n_admitted_requests=0))
    self.assertEqual(req.num_computed_tokens, 0)
    self.assertFalse(req.is_decode)
    self.assertFalse(req.is_chunked_prefill)

  def test_schedule_pending_respects_max_num_seqs(self):
    for i in range(6):
      self.sched._pending_requests.append(
          request_lib.RequestState(req_id=f"r{i}", prompt_token_ids=[i])
      )

    self.sched._schedule_requests()

    self.assertLen(self.sched._running_requests, 4)
    self.assertLen(self.sched._pending_requests, 2)
    for req in self.sched._running_requests:
      self.assertEqual(req.status, request_lib.RequestStatus.RUNNING)

  def test_schedule_pending_chunked_prefill_needs_a_full_chunk_of_budget(self):
    sched = self._create_scheduler(max_num_batched_tokens=16)
    r0 = request_lib.RequestState(req_id="r0", prompt_token_ids=list(range(8)))
    req = request_lib.RequestState(req_id="r1", prompt_token_ids=list(range(32)))
    sched._pending_requests.extend([r0, req])

    sched._schedule_requests()

    # r0 uses 8 of the 16 token budget. req needs a full 16-token chunk, so it
    # stays pending.
    self.assertEqual(list(sched._running_requests), [r0])
    self.assertEqual(list(sched._pending_requests), [req])
    self.assertFalse(req.is_chunked_prefill)

  def test_schedule_pending_stops_at_token_budget(self):
    sched = self._create_scheduler(max_num_batched_tokens=16)
    r1 = request_lib.RequestState(req_id="r1", prompt_token_ids=list(range(8)))
    r2 = request_lib.RequestState(req_id="r2", prompt_token_ids=list(range(12)))
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
    r1 = request_lib.RequestState(req_id="r1", prompt_token_ids=list(range(4)))
    r2 = request_lib.RequestState(req_id="r2", prompt_token_ids=list(range(8)))
    r3 = request_lib.RequestState(req_id="r3", prompt_token_ids=list(range(4)))
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
    r1 = request_lib.RequestState(req_id="r1", prompt_token_ids=list(range(4)))
    r2 = request_lib.RequestState(req_id="r2", prompt_token_ids=list(range(4)))
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
    r1 = request_lib.RequestState(req_id="r1", prompt_token_ids=list(range(8)))
    r2 = request_lib.RequestState(req_id="r2", prompt_token_ids=list(range(8, 16)))
    r3 = request_lib.RequestState(req_id="r3", prompt_token_ids=list(range(16, 24)))
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
    prefill = request_lib.RequestState(req_id="p", prompt_token_ids=list(range(16)))
    decode = request_lib.RequestState(req_id="d", prompt_token_ids=[100, 101])
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


if __name__ == "__main__":
  absltest.main()
