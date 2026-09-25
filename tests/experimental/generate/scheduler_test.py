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
      max_num_batch_tokens=64,
      max_seqs_per_batch=4,
      chunked_prefill_length=16,
      num_decode_steps=1,
  )
  kwargs.update(overrides)
  return scheduler.SchedulerConfig(**kwargs)


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
        request_lib.Request(req_id="r0", prompt_token_ids=[10, 20]),
    )
    self.sched._pending_requests.append(
        request_lib.Request(req_id="r1", prompt_token_ids=[30])
    )
    self.assertEqual(self.sched.num_active_requests, 2)

  def test_preempt_moves_newest_request_to_front_of_pending(self):
    r0 = request_lib.Request(req_id="r0", prompt_token_ids=[10, 20])
    r1 = request_lib.Request(req_id="r1", prompt_token_ids=[30, 40])
    pending = request_lib.Request(req_id="p", prompt_token_ids=[50])
    _admit(self.sched, self.kv_mgr, r0)
    _admit(self.sched, self.kv_mgr, r1)
    self.sched._pending_requests.append(pending)

    self.sched._preempt()

    self.assertEqual(list(self.sched._running_requests), [r0])
    self.assertEqual(list(self.sched._pending_requests), [r1, pending])

  def test_preempt_discards_kv_but_keeps_tokens(self):
    req = request_lib.Request(req_id="r1", prompt_token_ids=[10, 20, 30, 40])
    _admit(self.sched, self.kv_mgr, req)
    req.num_completed_tokens = 4
    req.num_in_flight_tokens = 0
    req.token_ids.append(90)

    self.sched._preempt()

    # A preempted request must re-prefill from scratch.
    self.assertEqual(req.num_completed_tokens, 0)
    self.assertEqual(req.num_in_flight_tokens, 0)
    self.assertEqual(req.token_ids, [10, 20, 30, 40, 90])
    self.assertEqual(self.kv_mgr.get_page_idxs("r1"), {"cache_0": ()})

  def test_preempt_all_on_empty_scheduler_is_a_noop(self):
    self.sched.preempt_all()
    self.assertEqual(self.sched.num_active_requests, 0)

  def test_preempt_all_requeues_every_running_request_in_arrival_order(self):
    reqs = [
        request_lib.Request(req_id=f"r{i}", prompt_token_ids=[10 * i, 20, 30])
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
    req = request_lib.Request(req_id="r1", prompt_token_ids=list(range(12)))
    _admit(self.sched, self.kv_mgr, req)
    self.assertIn("r1", self.kv_mgr._request_to_prefix_hashes)

    self.sched.preempt_all()

    # Stale hashes would otherwise be re-registered against fresh pages and
    # become prefix-cache hits for other requests.
    self.assertNotIn("r1", self.kv_mgr._request_to_prefix_hashes)

  def test_abort_running_request_releases_pages(self):
    req = request_lib.Request(req_id="r1", prompt_token_ids=[10, 20])
    _admit(self.sched, self.kv_mgr, req)

    self.assertTrue(self.sched.abort_request("r1"))

    self.assertEmpty(self.sched._running_requests)
    self.assertEqual(req.num_in_flight_tokens, 0)
    self.assertEqual(self.kv_mgr.get_page_idxs("r1"), {"cache_0": ()})

  def test_abort_pending_request(self):
    req = request_lib.Request(req_id="r1", prompt_token_ids=[10, 20])
    self.sched._pending_requests.append(req)

    self.assertTrue(self.sched.abort_request("r1"))
    self.assertEqual(self.sched.num_active_requests, 0)

  def test_abort_unknown_request_returns_false(self):
    self.assertFalse(self.sched.abort_request("unknown"))


if __name__ == "__main__":
  absltest.main()
