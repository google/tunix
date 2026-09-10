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

"""Unit tests for KVCacheCoordinator."""

import os
from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from tunix.experimental.generate import kv_cache_coordinator
from tunix.experimental.generate import request as request_lib
from tunix.experimental.generate import tiered_page_pool

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

Page = kv_cache_coordinator.Page
_Page = Page
_RequestState = kv_cache_coordinator._RequestState


def _create_coordinator(
    page_size: int = 4,
    num_tpu_pages: int = 10,
    num_cpu_pages: int = 10,
    window_size: int | None = None,
    partition_keys: tuple[str, ...] = ("layer_0",),
) -> tuple[
    kv_cache_coordinator.KVCacheCoordinator,
    tiered_page_pool.TieredPagePoolManager,
]:
  """Creates a KVCacheCoordinator with an initialized TieredPagePoolManager."""
  config = tiered_page_pool.TieredPagePoolConfig(
      page_size=page_size,
      dtype=jnp.float32,
      partition_keys=partition_keys,
      num_tpu_pages=num_tpu_pages,
      num_cpu_pages=num_cpu_pages,
  )
  tpu_pool, cpu_pool = config.init()
  manager = tiered_page_pool.TieredPagePoolManager(config, tpu_pool, cpu_pool)
  coordinator = kv_cache_coordinator.KVCacheCoordinator(
      page_manager=manager,
      window_size=window_size,
  )
  return coordinator, manager


class PageAndRequestStateDataclassTest(absltest.TestCase):

  def test_page_equality_and_hash(self):
    p1 = _Page(page_id=1, location="tpu", ref_count=1, prefix_hash=123)
    p2 = _Page(page_id=1, location="cpu", ref_count=0, prefix_hash=456)
    p3 = _Page(page_id=2, location="tpu", ref_count=1, prefix_hash=123)

    self.assertEqual(p1, p2)
    self.assertNotEqual(p1, p3)
    self.assertEqual(hash(p1), hash(p2))
    self.assertNotEqual(hash(p1), hash(p3))
    self.assertNotEqual(p1, "not_a_page")

  def test_request_state_defaults(self):
    state = _RequestState()
    self.assertEmpty(state.pages)
    self.assertEqual(state.num_hashed_pages, 0)
    self.assertEqual(state.num_released_pages, 0)
    self.assertEqual(state.last_page_hash, 0)


class ReferenceAndEvictionManagementTest(absltest.TestCase):

  def test_touch_page_none(self):
    coordinator, _ = _create_coordinator()
    coordinator._touch_page(None)  # Must be a no-op without error

  def test_touch_page_increments_ref_count(self):
    coordinator, _ = _create_coordinator()
    page = _Page(page_id=0, location="tpu", ref_count=0)
    coordinator._touch_page(page)
    self.assertEqual(page.ref_count, 1)
    coordinator._touch_page(page)
    self.assertEqual(page.ref_count, 2)

  def test_touch_page_removes_from_unreferenced_queues(self):
    coordinator, _ = _create_coordinator()
    page = _Page(page_id=0, location="tpu", ref_count=0)
    coordinator._unreferenced_tpu_pages[page] = None
    coordinator._unreferenced_cpu_pages[page] = None

    coordinator._touch_page(page)
    self.assertNotIn(page, coordinator._unreferenced_tpu_pages)
    self.assertNotIn(page, coordinator._unreferenced_cpu_pages)

  def test_release_page_none(self):
    coordinator, _ = _create_coordinator()
    coordinator._release_page(None)  # Must be a no-op without error

  def test_release_page_zero_or_negative_ref_count_raises(self):
    coordinator, _ = _create_coordinator()
    page = _Page(page_id=5, location="tpu", ref_count=0)
    with self.assertRaisesRegex(
        ValueError, r"Cannot release page 5 with no references\."
    ):
      coordinator._release_page(page)

    page.ref_count = -1
    with self.assertRaisesRegex(
        ValueError, r"Cannot release page 5 with no references\."
    ):
      coordinator._release_page(page)

  def test_release_page_decrements_ref_count(self):
    coordinator, _ = _create_coordinator()
    page = _Page(page_id=0, location="tpu", ref_count=2, prefix_hash=100)
    coordinator._release_page(page)
    self.assertEqual(page.ref_count, 1)
    self.assertNotIn(page, coordinator._unreferenced_tpu_pages)

  def test_release_page_unhashed_freed_immediately(self):
    coordinator, manager = _create_coordinator(num_tpu_pages=5)
    pids = manager.allocate_tpu_pages(1)
    page = _Page(page_id=pids[0], location="tpu", ref_count=1, prefix_hash=None)

    self.assertEqual(manager.num_free_tpu_pages, 4)
    coordinator._release_page(page)

    self.assertEqual(page.ref_count, 0)
    self.assertNotIn(page, coordinator._unreferenced_tpu_pages)
    self.assertEqual(manager.num_free_tpu_pages, 5)

  def test_release_page_tpu_added_to_unreferenced_tpu(self):
    coordinator, manager = _create_coordinator(num_tpu_pages=5)
    pids = manager.allocate_tpu_pages(1)
    page = _Page(page_id=pids[0], location="tpu", ref_count=1, prefix_hash=999)

    coordinator._release_page(page)

    self.assertEqual(page.ref_count, 0)
    self.assertIn(page, coordinator._unreferenced_tpu_pages)
    self.assertEqual(manager.num_free_tpu_pages, 4)

  def test_release_page_cpu_added_to_unreferenced_cpu(self):
    coordinator, manager = _create_coordinator(num_tpu_pages=5, num_cpu_pages=5)
    pids = manager.allocate_tpu_pages(1)
    manager.offload(pids)
    page = _Page(page_id=pids[0], location="cpu", ref_count=1, prefix_hash=888)

    coordinator._release_page(page)

    self.assertEqual(page.ref_count, 0)
    self.assertIn(page, coordinator._unreferenced_cpu_pages)

  def test_free_pages_removes_from_unreferenced_queues(self):
    coordinator, manager = _create_coordinator()
    pids = manager.allocate_tpu_pages(2)
    p0 = _Page(page_id=pids[0], location="tpu", ref_count=0)
    p1 = _Page(page_id=pids[1], location="tpu", ref_count=0)
    coordinator._unreferenced_tpu_pages[p0] = None
    coordinator._unreferenced_cpu_pages[p1] = None

    coordinator._free_pages([p0, p1])
    self.assertNotIn(p0, coordinator._unreferenced_tpu_pages)
    self.assertNotIn(p1, coordinator._unreferenced_cpu_pages)

  def test_free_pages_removes_from_prefix_hash_map(self):
    coordinator, manager = _create_coordinator()
    pids = manager.allocate_tpu_pages(1)
    page = _Page(page_id=pids[0], location="tpu", ref_count=0, prefix_hash=12345)
    coordinator._prefix_hash_to_page[12345] = page

    coordinator._free_pages([page])
    self.assertNotIn(12345, coordinator._prefix_hash_to_page)
    self.assertIsNone(page.prefix_hash)

  def test_free_pages_does_not_remove_other_page_from_prefix_map(self):
    coordinator, manager = _create_coordinator()
    pids = manager.allocate_tpu_pages(2)
    page1 = _Page(page_id=pids[0], location="tpu", ref_count=0, prefix_hash=123)
    page2 = _Page(page_id=pids[1], location="tpu", ref_count=0, prefix_hash=123)
    # Map points to page2
    coordinator._prefix_hash_to_page[123] = page2

    # Freeing page1 should not remove page2 from prefix map
    coordinator._free_pages([page1])
    self.assertEqual(coordinator._prefix_hash_to_page[123], page2)
    self.assertIsNone(page1.prefix_hash)

  def test_free_pages_calls_page_manager_free_with_all_pages(self):
    coordinator, manager = _create_coordinator(num_tpu_pages=10)
    pids = manager.allocate_tpu_pages(3)
    pages = [_Page(page_id=pid, location="tpu") for pid in pids]

    self.assertEqual(manager.num_free_tpu_pages, 7)
    coordinator._free_pages(pages)
    self.assertEqual(manager.num_free_tpu_pages, 10)


class EvictionAndOffloadTest(absltest.TestCase):

  def test_free_unreferenced_tpu_pages_zero_or_negative(self):
    coordinator, _ = _create_coordinator()
    coordinator._free_unreferenced_tpu_pages(0)
    coordinator._free_unreferenced_tpu_pages(-1)

  def test_free_unreferenced_tpu_pages_more_than_available_raises(self):
    coordinator, manager = _create_coordinator(num_tpu_pages=5)
    pids = manager.allocate_tpu_pages(1)
    page = _Page(page_id=pids[0], location="tpu", ref_count=0, prefix_hash=1)
    coordinator._unreferenced_tpu_pages[page] = None

    with self.assertRaisesRegex(
        ValueError, r"Cannot free 2 TPU pages, only 1 available\."
    ):
      coordinator._free_unreferenced_tpu_pages(2)

  def test_free_unreferenced_tpu_pages_offload_when_cpu_available(self):
    coordinator, manager = _create_coordinator(num_tpu_pages=5, num_cpu_pages=5)
    pids = manager.allocate_tpu_pages(2)
    p0 = _Page(page_id=pids[0], location="tpu", ref_count=0, prefix_hash=1)
    p1 = _Page(page_id=pids[1], location="tpu", ref_count=0, prefix_hash=2)
    coordinator._unreferenced_tpu_pages[p0] = None
    coordinator._unreferenced_tpu_pages[p1] = None

    self.assertEqual(manager.num_free_tpu_pages, 3)
    self.assertEqual(manager.num_free_cpu_pages, 5)

    coordinator._free_unreferenced_tpu_pages(2)

    self.assertEmpty(coordinator._unreferenced_tpu_pages)
    self.assertEqual(manager.num_free_tpu_pages, 5)
    self.assertEqual(manager.num_free_cpu_pages, 3)
    self.assertEqual(p0.location, "cpu")
    self.assertEqual(p1.location, "cpu")
    self.assertIn(p0, coordinator._unreferenced_cpu_pages)
    self.assertIn(p1, coordinator._unreferenced_cpu_pages)

  def test_free_unreferenced_tpu_pages_cpu_shortfall_frees_cpu_pages(self):
    # Setup: 2 CPU pages total, both currently in use by unreferenced CPU pages.
    coordinator, manager = _create_coordinator(num_tpu_pages=5, num_cpu_pages=2)
    cpu_pids = manager.allocate_tpu_pages(2)
    manager.offload(cpu_pids)
    cpu_p0 = _Page(
        page_id=cpu_pids[0], location="cpu", ref_count=0, prefix_hash=10
    )
    cpu_p1 = _Page(
        page_id=cpu_pids[1], location="cpu", ref_count=0, prefix_hash=11
    )
    coordinator._unreferenced_cpu_pages[cpu_p0] = None
    coordinator._unreferenced_cpu_pages[cpu_p1] = None
    coordinator._prefix_hash_to_page[10] = cpu_p0
    coordinator._prefix_hash_to_page[11] = cpu_p1

    self.assertEqual(manager.num_free_cpu_pages, 0)

    # Allocate 2 TPU pages that become unreferenced
    tpu_pids = manager.allocate_tpu_pages(2)
    tpu_p0 = _Page(
        page_id=tpu_pids[0], location="tpu", ref_count=0, prefix_hash=20
    )
    tpu_p1 = _Page(
        page_id=tpu_pids[1], location="tpu", ref_count=0, prefix_hash=21
    )
    coordinator._unreferenced_tpu_pages[tpu_p0] = None
    coordinator._unreferenced_tpu_pages[tpu_p1] = None

    # Freeing 2 unreferenced TPU pages requires offloading 2 to CPU.
    # Since num_free_cpu_pages is 0, cpu_shortfall is 2.
    # It must free cpu_p0 and cpu_p1 first, then offload tpu_p0 and tpu_p1.
    coordinator._free_unreferenced_tpu_pages(2)

    self.assertEmpty(coordinator._unreferenced_tpu_pages)
    self.assertNotIn(cpu_p0, coordinator._unreferenced_cpu_pages)
    self.assertNotIn(cpu_p1, coordinator._unreferenced_cpu_pages)
    self.assertIn(tpu_p0, coordinator._unreferenced_cpu_pages)
    self.assertIn(tpu_p1, coordinator._unreferenced_cpu_pages)
    self.assertEqual(tpu_p0.location, "cpu")
    self.assertEqual(tpu_p1.location, "cpu")

  def test_free_unreferenced_tpu_pages_frees_when_no_cpu_pool(self):
    coordinator, manager = _create_coordinator(num_tpu_pages=5, num_cpu_pages=0)
    pids = manager.allocate_tpu_pages(2)
    p0 = _Page(page_id=pids[0], location="tpu", ref_count=0, prefix_hash=1)
    p1 = _Page(page_id=pids[1], location="tpu", ref_count=0, prefix_hash=2)
    coordinator._unreferenced_tpu_pages[p0] = None
    coordinator._unreferenced_tpu_pages[p1] = None

    coordinator._free_unreferenced_tpu_pages(2)

    self.assertEmpty(coordinator._unreferenced_tpu_pages)
    self.assertEqual(manager.num_free_tpu_pages, 5)

  def test_free_unreferenced_cpu_pages_zero_or_negative(self):
    coordinator, _ = _create_coordinator()
    coordinator._free_unreferenced_cpu_pages(0)
    coordinator._free_unreferenced_cpu_pages(-1)

  def test_free_unreferenced_cpu_pages_more_than_available_raises(self):
    coordinator, manager = _create_coordinator(num_tpu_pages=5, num_cpu_pages=5)
    pids = manager.allocate_tpu_pages(1)
    manager.offload(pids)
    page = _Page(page_id=pids[0], location="cpu", ref_count=0, prefix_hash=1)
    coordinator._unreferenced_cpu_pages[page] = None

    with self.assertRaisesRegex(
        ValueError, r"Cannot free 3 CPU pages, only 1 available\."
    ):
      coordinator._free_unreferenced_cpu_pages(3)

  def test_free_unreferenced_cpu_pages_fifo_order_and_count(self):
    coordinator, manager = _create_coordinator(num_tpu_pages=5, num_cpu_pages=5)
    pids = manager.allocate_tpu_pages(3)
    manager.offload(pids)
    pages = [
        _Page(page_id=pid, location="cpu", ref_count=0, prefix_hash=i)
        for i, pid in enumerate(pids)
    ]
    for p in pages:
      coordinator._unreferenced_cpu_pages[p] = None
      if p.prefix_hash is not None:
        coordinator._prefix_hash_to_page[p.prefix_hash] = p

    self.assertEqual(manager.num_free_cpu_pages, 2)
    # Free 2 oldest pages (FIFO: pages[0], pages[1])
    coordinator._free_unreferenced_cpu_pages(2)

    self.assertNotIn(pages[0], coordinator._unreferenced_cpu_pages)
    self.assertNotIn(pages[1], coordinator._unreferenced_cpu_pages)
    self.assertIn(pages[2], coordinator._unreferenced_cpu_pages)
    self.assertLen(coordinator._unreferenced_cpu_pages, 1)
    self.assertEqual(manager.num_free_cpu_pages, 4)


class SlidingWindowAndOutOfWindowPagesTest(absltest.TestCase):

  def test_release_out_of_window_pages_none_window_size(self):
    coordinator, manager = _create_coordinator(window_size=None, page_size=4)
    req = request_lib.Request(req_id="r1", prompt_token_ids=list(range(16)))
    pids = manager.allocate_tpu_pages(4)
    state = coordinator._requests.setdefault(req.request_id, _RequestState())
    state.pages = [
        _Page(page_id=pid, location="tpu", ref_count=1) for pid in pids
    ]
    req.num_completed_tokens = 16

    coordinator.release_out_of_window_pages(req)
    self.assertEqual(state.num_released_pages, 0)
    for p in state.pages:
      self.assertIsNotNone(p)

  def test_release_out_of_window_pages_no_request_or_empty_pages(self):
    coordinator, _ = _create_coordinator(window_size=4, page_size=4)
    req = request_lib.Request(req_id="unknown", prompt_token_ids=[1, 2])
    coordinator.release_out_of_window_pages(req)  # No exception

    coordinator._requests["empty"] = _RequestState(pages=[])
    req2 = request_lib.Request(req_id="empty", prompt_token_ids=[1, 2])
    coordinator.release_out_of_window_pages(req2)  # No exception

  def test_release_out_of_window_pages_within_window_releases_no_pages(self):
    # Case 1: completed tokens within window size -> 0 pages released
    coordinator, manager = _create_coordinator(window_size=8, page_size=4)
    req = request_lib.Request(req_id="r1", prompt_token_ids=list(range(12)))
    pids = manager.allocate_tpu_pages(3)
    state = coordinator._requests.setdefault(req.request_id, _RequestState())
    state.pages = [
        _Page(page_id=pid, location="tpu", ref_count=1) for pid in pids
    ]
    req.num_completed_tokens = 4  # max(0, 4 - 8) = 0 -> lowest_needed_page_idx = 0

    coordinator.release_out_of_window_pages(req)
    self.assertEqual(state.num_released_pages, 0)
    self.assertLen(state.pages, 3)

  def test_release_out_of_window_pages_prompt_larger_than_window_zero_completed(self):
    # Case 2: prompt larger than window size and request has not completed any tokens
    coordinator, manager = _create_coordinator(window_size=4, page_size=4)
    req = request_lib.Request(req_id="r1", prompt_token_ids=list(range(16)))
    pids = manager.allocate_tpu_pages(4)
    state = coordinator._requests.setdefault(req.request_id, _RequestState())
    state.pages = [
        _Page(page_id=pid, location="tpu", ref_count=1) for pid in pids
    ]
    req.num_completed_tokens = 0

    coordinator.release_out_of_window_pages(req)
    self.assertEqual(state.num_released_pages, 0)
    for p in state.pages:
      self.assertIsNotNone(p)

  def test_release_out_of_window_pages_releases_completed_tokens_outside_window(self):
    # Case 3: window pages are released when request has completed tokens
    coordinator, manager = _create_coordinator(window_size=4, page_size=4)
    req = request_lib.Request(req_id="r1", prompt_token_ids=list(range(16)))
    pids = manager.allocate_tpu_pages(4)
    pages = [_Page(page_id=pid, location="tpu", ref_count=1) for pid in pids]
    state = coordinator._requests.setdefault(req.request_id, _RequestState())
    state.pages = list(pages)

    # 12 tokens completed. Window is 4 tokens.
    # lowest_needed_token_idx = 12 - 4 = 8.
    # lowest_needed_page_idx = 8 // 4 = 2.
    # Pages before page index 2 (i.e. pages 0 and 1) should be released.
    req.num_completed_tokens = 12
    coordinator.release_out_of_window_pages(req)

    self.assertEqual(state.num_released_pages, 2)
    self.assertIsNone(state.pages[0])
    self.assertIsNone(state.pages[1])
    self.assertEqual(state.pages[2], pages[2])
    self.assertEqual(state.pages[3], pages[3])
    self.assertEqual(pages[0].ref_count, 0)
    self.assertEqual(pages[1].ref_count, 0)

  def test_release_out_of_window_pages_incremental_calls(self):
    coordinator, manager = _create_coordinator(window_size=4, page_size=4)
    req = request_lib.Request(req_id="r1", prompt_token_ids=list(range(16)))
    pids = manager.allocate_tpu_pages(4)
    pages = [_Page(page_id=pid, location="tpu", ref_count=1) for pid in pids]
    state = coordinator._requests.setdefault(req.request_id, _RequestState())
    state.pages = list(pages)

    # Step 1: 8 completed tokens -> release page 0
    req.num_completed_tokens = 8
    coordinator.release_out_of_window_pages(req)
    self.assertEqual(state.num_released_pages, 1)
    self.assertIsNone(state.pages[0])

    # Step 2: 12 completed tokens -> release page 1
    req.num_completed_tokens = 12
    coordinator.release_out_of_window_pages(req)
    self.assertEqual(state.num_released_pages, 2)
    self.assertIsNone(state.pages[1])

    # Step 3: calling again with same tokens should not release anything new
    coordinator.release_out_of_window_pages(req)
    self.assertEqual(state.num_released_pages, 2)


class ChunkAndHashTest(absltest.TestCase):

  def test_chunk_and_hash_ignores_partial_pages(self):
    coordinator, _ = _create_coordinator(page_size=4)
    tokens = list(range(10))  # 10 tokens: 2 full pages (8 tokens), 2 extra
    hashes = coordinator._chunk_and_hash(tokens)
    self.assertLen(hashes, 2)
    self.assertEqual(hashes, coordinator._chunk_and_hash(tokens[:8]))

  def test_chunk_and_hash_empty_tokens(self):
    coordinator, _ = _create_coordinator(page_size=4)
    hashes = coordinator._chunk_and_hash([])
    self.assertEmpty(hashes)

  def test_chunk_and_hash_with_start_hash(self):
    coordinator, _ = _create_coordinator(page_size=4)
    tokens = list(range(8))
    full_hashes = coordinator._chunk_and_hash(tokens)
    chained_hashes = coordinator._chunk_and_hash(
        tokens[4:], start_hash=full_hashes[0]
    )
    self.assertLen(chained_hashes, 1)
    self.assertEqual(chained_hashes[0], full_hashes[1])


class CacheFullPagesTest(absltest.TestCase):

  def test_cache_full_pages_no_completed_full_pages(self):
    coordinator, manager = _create_coordinator(page_size=4)
    req = request_lib.Request(req_id="r1", prompt_token_ids=list(range(8)))
    pids = manager.allocate_tpu_pages(2)
    state = coordinator._requests.setdefault(req.request_id, _RequestState())
    state.pages = [
        _Page(page_id=pid, location="tpu", ref_count=1) for pid in pids
    ]
    req.num_completed_tokens = 3  # < page_size

    coordinator.cache_full_pages(req)
    self.assertEqual(state.num_hashed_pages, 0)
    self.assertEmpty(coordinator._prefix_hash_to_page)

  def test_cache_full_pages_registers_new_pages(self):
    coordinator, manager = _create_coordinator(page_size=4)
    tokens = [10, 20, 30, 40, 50, 60, 70, 80]
    req = request_lib.Request(req_id="r1", prompt_token_ids=tokens)
    pids = manager.allocate_tpu_pages(2)
    pages = [_Page(page_id=pid, location="tpu", ref_count=1) for pid in pids]
    state = coordinator._requests.setdefault(req.request_id, _RequestState())
    state.pages = list(pages)
    req.num_completed_tokens = 8

    coordinator.cache_full_pages(req)

    self.assertEqual(state.num_hashed_pages, 2)
    expected_h0, expected_h1 = coordinator._chunk_and_hash(tokens)
    self.assertEqual(pages[0].prefix_hash, expected_h0)
    self.assertEqual(pages[1].prefix_hash, expected_h1)
    self.assertEqual(coordinator._prefix_hash_to_page[expected_h0], pages[0])
    self.assertEqual(coordinator._prefix_hash_to_page[expected_h1], pages[1])
    self.assertEqual(state.last_page_hash, expected_h1)

  def test_cache_full_pages_does_not_reassign_already_hashed(self):
    coordinator, manager = _create_coordinator(page_size=4)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    req = request_lib.Request(req_id="r1", prompt_token_ids=tokens)
    pids = manager.allocate_tpu_pages(2)
    pages = [_Page(page_id=pid, location="tpu", ref_count=1) for pid in pids]
    state = coordinator._requests.setdefault(req.request_id, _RequestState())
    state.pages = list(pages)

    req.num_completed_tokens = 4
    coordinator.cache_full_pages(req)
    self.assertEqual(state.num_hashed_pages, 1)

    req.num_completed_tokens = 8
    coordinator.cache_full_pages(req)
    self.assertEqual(state.num_hashed_pages, 2)

    # Calling again at 8 tokens should be a no-op
    coordinator.cache_full_pages(req)
    self.assertEqual(state.num_hashed_pages, 2)

  def test_cache_full_pages_collision_cached_on_tpu(self):
    coordinator, manager = _create_coordinator(page_size=4, num_tpu_pages=10)
    tokens = [1, 2, 3, 4]
    req1 = request_lib.Request(req_id="r1", prompt_token_ids=tokens)
    pid1 = manager.allocate_tpu_pages(1)[0]
    p1 = _Page(page_id=pid1, location="tpu", ref_count=1)
    state1 = coordinator._requests.setdefault(req1.request_id, _RequestState())
    state1.pages = [p1]
    req1.num_completed_tokens = 4
    coordinator.cache_full_pages(req1)
    prefix_hash = p1.prefix_hash

    # Second request with same tokens allocates a new page p2
    req2 = request_lib.Request(req_id="r2", prompt_token_ids=tokens)
    pid2 = manager.allocate_tpu_pages(1)[0]
    p2 = _Page(page_id=pid2, location="tpu", ref_count=1)
    state2 = coordinator._requests.setdefault(req2.request_id, _RequestState())
    state2.pages = [p2]
    req2.num_completed_tokens = 4

    # cache_full_pages should keep old page p1, release p2, touch p1
    coordinator.cache_full_pages(req2)

    self.assertEqual(state2.pages[0], p1)
    self.assertEqual(p1.ref_count, 2)
    self.assertEqual(p2.ref_count, 0)
    # p2 had no prefix_hash so it is freed immediately
    self.assertNotIn(p2, coordinator._unreferenced_tpu_pages)
    self.assertIsNotNone(prefix_hash)
    assert prefix_hash is not None
    self.assertEqual(coordinator._prefix_hash_to_page[prefix_hash], p1)

  def test_cache_full_pages_collision_cached_on_cpu(self):
    coordinator, manager = _create_coordinator(
        page_size=4, num_tpu_pages=10, num_cpu_pages=10
    )
    tokens = [1, 2, 3, 4]
    req1 = request_lib.Request(req_id="r1", prompt_token_ids=tokens)
    pid1 = manager.allocate_tpu_pages(1)[0]
    p1 = _Page(page_id=pid1, location="tpu", ref_count=1)
    state1 = coordinator._requests.setdefault(req1.request_id, _RequestState())
    state1.pages = [p1]
    req1.num_completed_tokens = 4
    coordinator.cache_full_pages(req1)
    prefix_hash = p1.prefix_hash
    self.assertIsNotNone(prefix_hash)
    assert prefix_hash is not None

    # Release req1 and offload p1 to CPU
    coordinator.release_request(req1)
    coordinator._free_unreferenced_tpu_pages(1)
    self.assertEqual(p1.location, "cpu")
    self.assertEqual(p1.ref_count, 0)
    self.assertEqual(coordinator._prefix_hash_to_page[prefix_hash], p1)

    # Second request with identical tokens allocates p2 on TPU
    req2 = request_lib.Request(req_id="r2", prompt_token_ids=tokens)
    pid2 = manager.allocate_tpu_pages(1)[0]
    p2 = _Page(page_id=pid2, location="tpu", ref_count=1)
    state2 = coordinator._requests.setdefault(req2.request_id, _RequestState())
    state2.pages = [p2]
    req2.num_completed_tokens = 4

    # cache_full_pages should free old CPU page p1 and remap prefix_hash to p2
    coordinator.cache_full_pages(req2)

    self.assertEqual(coordinator._prefix_hash_to_page[prefix_hash], p2)
    self.assertEqual(p2.prefix_hash, prefix_hash)
    self.assertEqual(state2.pages[0], p2)
    self.assertIsNone(p1.prefix_hash)


class PrefixMatchingTest(parameterized.TestCase):

  def test_get_cached_pages_scheduled_request_returns_empty(self):
    coordinator, _ = _create_coordinator()
    req = request_lib.Request(req_id="r1", prompt_token_ids=list(range(16)))
    req.num_completed_tokens = 1
    self.assertEmpty(coordinator.get_cached_pages(req))

  def test_get_cached_pages_reserves_final_token(self):
    coordinator, _ = _create_coordinator(page_size=4)
    # len 4: (4 - 1) // 4 = 0 full pages -> returns empty
    req4 = request_lib.Request(req_id="r1", prompt_token_ids=[1, 2, 3, 4])
    self.assertEmpty(coordinator.get_cached_pages(req4))

    # len 5: (5 - 1) // 4 = 1 full page -> can match
    req5 = request_lib.Request(req_id="r2", prompt_token_ids=[1, 2, 3, 4, 5])
    h = coordinator._chunk_and_hash([1, 2, 3, 4])[0]
    page = Page(page_id=0, location="tpu", prefix_hash=h)
    coordinator._prefix_hash_to_page[h] = page
    matched = coordinator.get_cached_pages(req5)
    self.assertEqual(matched, [page])

  def test_get_cached_pages_no_window_prefix_matching(self):
    coordinator, _ = _create_coordinator(page_size=4, window_size=None)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]  # 3 full pages + 1 token
    h0, h1 = coordinator._chunk_and_hash(tokens[:8])
    # Only h0 and h1 in cache, h2 not in cache
    p0 = Page(page_id=0, location="tpu", prefix_hash=h0)
    p1 = Page(page_id=1, location="tpu", prefix_hash=h1)
    coordinator._prefix_hash_to_page[h0] = p0
    coordinator._prefix_hash_to_page[h1] = p1

    req = request_lib.Request(req_id="r1", prompt_token_ids=tokens)
    matched = coordinator.get_cached_pages(req)
    self.assertEqual(matched, [p0, p1])

  def test_get_cached_pages_no_match_returns_empty(self):
    coordinator, _ = _create_coordinator(page_size=4, window_size=None)
    req = request_lib.Request(req_id="r1", prompt_token_ids=[1, 2, 3, 4, 5])
    self.assertEmpty(coordinator.get_cached_pages(req))

  def test_get_cached_pages_full_prefix(self):
    coordinator, _ = _create_coordinator(page_size=4)
    tokens = list(range(1, 18))  # 4 full pages + 1 token
    h0, h1 = coordinator._chunk_and_hash(tokens[:8])
    p0 = Page(page_id=0, location="tpu", prefix_hash=h0)
    p1 = Page(page_id=1, location="tpu", prefix_hash=h1)
    coordinator._prefix_hash_to_page[h0] = p0
    coordinator._prefix_hash_to_page[h1] = p1

    req = request_lib.Request(req_id="r1", prompt_token_ids=tokens)
    matched = coordinator.get_cached_pages(req)
    self.assertEqual(matched, [p0, p1])

  def test_get_cached_pages_middle_matches(self):
    coordinator, _ = _create_coordinator(page_size=4)
    tokens = list(range(1, 18))  # 4 full pages + 1 token
    hashes = coordinator._chunk_and_hash(tokens[:12])
    h1, h2 = hashes[1], hashes[2]
    p1 = Page(page_id=1, location="tpu", prefix_hash=h1)
    p2 = Page(page_id=2, location="tpu", prefix_hash=h2)
    coordinator._prefix_hash_to_page[h1] = p1
    coordinator._prefix_hash_to_page[h2] = p2

    req = request_lib.Request(req_id="r1", prompt_token_ids=tokens)
    matched = coordinator.get_cached_pages(req)
    # Page 0 is miss (None), pages 1 and 2 hit, page 3 is miss (truncated)
    self.assertEqual(matched, [None, p1, p2])

  def test_get_cached_pages_matches_at_end(self):
    coordinator, _ = _create_coordinator(page_size=4)
    tokens = list(range(1, 18))  # 4 full pages + 1 token
    hashes = coordinator._chunk_and_hash(tokens[:16])
    h2, h3 = hashes[2], hashes[3]
    p2 = Page(page_id=2, location="tpu", prefix_hash=h2)
    p3 = Page(page_id=3, location="tpu", prefix_hash=h3)
    coordinator._prefix_hash_to_page[h2] = p2
    coordinator._prefix_hash_to_page[h3] = p3

    req = request_lib.Request(req_id="r1", prompt_token_ids=tokens)
    matched = coordinator.get_cached_pages(req)
    # Pages 0 and 1 are misses (None), pages 2 and 3 hit
    self.assertEqual(matched, [None, None, p2, p3])

  def test_get_cached_pages_gap_between_hits(self):
    coordinator, _ = _create_coordinator(page_size=4)
    tokens = list(range(1, 18))  # 4 full pages + 1 token
    hashes = coordinator._chunk_and_hash(tokens[:12])
    h0, h2 = hashes[0], hashes[2]
    p0 = Page(page_id=0, location="tpu", prefix_hash=h0)
    p2 = Page(page_id=2, location="tpu", prefix_hash=h2)
    coordinator._prefix_hash_to_page[h0] = p0
    coordinator._prefix_hash_to_page[h2] = p2

    req = request_lib.Request(req_id="r1", prompt_token_ids=tokens)
    matched = coordinator.get_cached_pages(req)
    # Page 0 is hit, page 1 is miss (None), page 2 is hit, page 3 is miss (truncated)
    self.assertEqual(matched, [p0, None, p2])

  def test_get_cached_pages_when_prefix_pages_released(self):
    # End-to-end test:
    # 1. Request 1 runs with prompt of 17 tokens (4 full pages + 1 token).
    # 2. Completed tokens = 16. Window size = 8 (2 pages).
    # 3. release_out_of_window_pages releases prefix pages 0 and 1.
    # 4. Prefix pages 0 and 1 are freed from cache (h0 and h1 removed).
    # 5. Only active window pages (h2, h3) remain in prefix cache.
    coordinator, manager = _create_coordinator(
        page_size=4, num_tpu_pages=10, window_size=8
    )
    tokens = list(range(17))
    req1 = request_lib.Request(req_id="r1", prompt_token_ids=tokens)
    coordinator.prepare_request_pages(req1, num_tokens=16)
    req1.num_completed_tokens = 16
    coordinator.cache_full_pages(req1)

    state1 = coordinator._requests[req1.request_id]
    p0 = state1.pages[0]
    p1 = state1.pages[1]
    p2 = state1.pages[2]
    p3 = state1.pages[3]
    self.assertIsNotNone(p0)
    self.assertIsNotNone(p1)
    self.assertIsNotNone(p2)
    self.assertIsNotNone(p3)
    assert p0 is not None and p1 is not None and p2 is not None and p3 is not None
    h0 = p0.prefix_hash
    h1 = p1.prefix_hash
    self.assertIsNotNone(h0)
    self.assertIsNotNone(h1)
    assert h0 is not None and h1 is not None

    # Release skipped pages (pages 0 and 1 outside the sliding window)
    coordinator.release_out_of_window_pages(req1)
    # Free the released unreferenced pages from the cache
    coordinator._free_pages([p0, p1])

    # Now only the tokens in the active window (pages 2 and 3) remain in cache.
    # The prefix pages (pages 0 and 1) were all released and freed.
    self.assertNotIn(h0, coordinator._prefix_hash_to_page)
    self.assertNotIn(h1, coordinator._prefix_hash_to_page)

    # Case A: A new request with a shorter prompt consisting only of the released prefix tokens
    # (e.g. 2 full pages + 1 token). Since prefix pages were released, returns no match.
    req_prefix = request_lib.Request(
        req_id="r2", prompt_token_ids=tokens[:9]
    )
    matched_prefix = coordinator.get_cached_pages(req_prefix)
    self.assertEmpty(matched_prefix)

    # Case B: A new request whose prompt matches up to page 2 of the active window,
    # but tokens diverge on page 3. Since get_cached_pages returns raw hash hits positionally
    # aligned, pages 0 and 1 return None, page 2 returns p2, and page 3 (miss) is truncated.
    diff_tokens = tokens[:12] + [999, 998, 997, 996, 995]
    req_diff = request_lib.Request(req_id="r3", prompt_token_ids=diff_tokens)
    matched_diff = coordinator.get_cached_pages(req_diff)
    self.assertEqual(matched_diff, [None, None, p2])

    # Case C: A new request with identical prompt tokens matches both remaining window pages.
    req_full = request_lib.Request(req_id="r4", prompt_token_ids=tokens)
    matched_full = coordinator.get_cached_pages(req_full)
    self.assertEqual(matched_full, [None, None, p2, p3])



class CalculatePageRequirementsTest(parameterized.TestCase):

  def test_calculate_page_requirements_no_matched_pages(self):
    coordinator, _ = _create_coordinator(page_size=4)
    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    # 7 tokens needed with page_size=4: cdiv(7, 4) = 2 pages
    reqs = coordinator._calculate_page_requirements(req, num_tokens=7)
    self.assertEqual(reqs, (2, 0))

  def test_calculate_page_requirements_matched_tpu_not_counted(self):
    coordinator, _ = _create_coordinator(page_size=4)
    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    p0 = _Page(page_id=0, location="tpu")
    p1 = _Page(page_id=1, location="tpu")

    # total_tokens = 0 + 2 * 4 + 4 = 12 tokens -> 3 total pages
    # target_total_pages (3) - len(current_pages) (0) - len(matched_pages) (2) = 1
    reqs = coordinator._calculate_page_requirements(
        req, num_tokens=4, matched_prefix_pages=[p0, p1]
    )
    self.assertEqual(reqs, (1, 2))

  def test_calculate_page_requirements_matched_cpu_counted(self):
    coordinator, _ = _create_coordinator(page_size=4)
    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    p0 = _Page(page_id=0, location="cpu")
    p1 = _Page(page_id=1, location="tpu")

    # 1 matched CPU page requires 1 TPU slot to load
    # n_needed is 1, matched_cpu is 1 -> 2 TPU pages needed
    reqs = coordinator._calculate_page_requirements(
        req, num_tokens=4, matched_prefix_pages=[p0, p1]
    )
    self.assertEqual(reqs, (2, 1))

  @parameterized.parameters(
      (None,),
      (4,),
      (16,),
  )
  def test_calculate_page_requirements_window_size_agnostic(
      self, window_size: int | None
  ):
    coordinator, _ = _create_coordinator(page_size=4, window_size=window_size)
    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    reqs = coordinator._calculate_page_requirements(req, num_tokens=10)
    # cdiv(10, 4) = 3
    self.assertEqual(reqs, (3, 0))

  @parameterized.parameters(
      (1, 7, 7),
      (2, 7, 4),
      (5, 7, 2),
      (4, 8, 2),
      (4, 9, 3),
  )
  def test_calculate_page_requirements_various_page_sizes(
      self, page_size: int, num_tokens: int, expected_pages: int
  ):
    coordinator, _ = _create_coordinator(page_size=page_size)
    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    self.assertEqual(
        coordinator._calculate_page_requirements(req, num_tokens=num_tokens),
        (expected_pages, 0),
    )

  def test_calculate_page_requirements_with_none_matched_pages(self):
    coordinator, _ = _create_coordinator(page_size=4)
    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    p2 = Page(page_id=2, location="tpu")
    p3 = Page(page_id=3, location="tpu")
    # matched_prefix_pages has 2 None (out-of-window) and 2 TPU pages
    # total_tokens = 0 + (4 * 4) + 4 = 20 tokens -> 5 target pages
    # target_total_pages (5) - len(current_pages) (0) - len(matched_prefix_pages) (4) = 1
    # matched_cpu is 0 -> n_needed is 1
    reqs = coordinator._calculate_page_requirements(
        req, num_tokens=4, matched_prefix_pages=[None, None, p2, p3]
    )
    self.assertEqual(reqs, (1, 2))


class HasSufficientSpaceTest(absltest.TestCase):

  def test_has_sufficient_space_true_when_enough_free_tpu(self):
    coordinator, _ = _create_coordinator(page_size=4, num_tpu_pages=10)
    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    # 7 tokens needed -> 2 TPU pages. Manager has 10 free TPU pages.
    self.assertTrue(coordinator.has_sufficient_space(req, num_tokens=7))

  def test_has_sufficient_space_false_when_not_enough_free_tpu(self):
    coordinator, _ = _create_coordinator(page_size=4, num_tpu_pages=2)
    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    # 12 tokens needed -> 3 TPU pages. Manager has only 2 free TPU pages.
    self.assertFalse(coordinator.has_sufficient_space(req, num_tokens=12))

  def test_has_sufficient_space_true_with_evictable_pages(self):
    coordinator, manager = _create_coordinator(page_size=4, num_tpu_pages=5)
    pids = manager.allocate_tpu_pages(4)  # 1 free TPU page left
    for pid in pids:
      p = Page(page_id=pid, location="tpu", ref_count=0)
      coordinator._unreferenced_tpu_pages[p] = None

    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    # 12 tokens needed -> 3 TPU pages.
    # 1 free TPU + 4 evictable TPU = 5 available >= 3.
    self.assertTrue(coordinator.has_sufficient_space(req, num_tokens=12))

  def test_has_sufficient_space_false_when_matched_unreferenced_cannot_be_evicted(
      self,
  ):
    coordinator, manager = _create_coordinator(page_size=4, num_tpu_pages=5)
    pids = manager.allocate_tpu_pages(4)  # 1 free TPU page left
    pages = []
    for pid in pids:
      p = Page(page_id=pid, location="tpu", ref_count=0)
      coordinator._unreferenced_tpu_pages[p] = None
      pages.append(p)

    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    # Matched pages includes 3 unreferenced TPU pages.
    # total_tokens = 0 + 3*4 + 4 = 16 tokens -> 4 total pages.
    # target_total_pages (4) - len(matched_pages) (3) = 1 new TPU page needed.
    # But matched unreferenced pages = 3.
    # evictable_tpu = 4 - 3 = 1.
    # Total TPU available = 1 free + 1 evictable = 2.
    # If we need 3 tokens -> 1 page needed <= 2 available -> True.
    # But if total_tokens needs 3 new pages:
    # 3 matched + 12 new tokens = 24 tokens -> 6 pages.
    # target_total_pages (6) - 3 matched = 3 TPU pages needed.
    # total available = 1 free + (4 - 3) evictable = 2.
    # 2 available < 3 needed -> False!
    self.assertFalse(
        coordinator.has_sufficient_space(
            req, num_tokens=12, matched_prefix_pages=pages[:3]
        )
    )

  def test_has_sufficient_space_with_matched_cpu_pages(self):
    coordinator, manager = _create_coordinator(page_size=4, num_tpu_pages=2)
    pids = manager.allocate_tpu_pages(1)  # 1 free TPU page left
    p_cpu = Page(page_id=100, location="cpu", ref_count=0)

    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    # 1 matched CPU page + 4 tokens:
    # total_tokens = 4 + 4 = 8 -> 2 pages.
    # n_needed = 2 - 1 = 1 page.
    # 1 matched CPU page requires 1 TPU slot to load.
    # Total TPU needed = 1 + 1 = 2.
    # Total TPU available = 1 free + 0 evictable = 1.
    # 1 < 2 -> False.
    self.assertFalse(
        coordinator.has_sufficient_space(
            req, num_tokens=4, matched_prefix_pages=[p_cpu]
        )
    )

  def test_has_sufficient_space_with_matched_prefix_pages(self):
    coordinator, _ = _create_coordinator(page_size=4, num_tpu_pages=10)
    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    p0 = Page(page_id=0, location="tpu", ref_count=1)

    self.assertTrue(
        coordinator.has_sufficient_space(
            req, num_tokens=4, matched_prefix_pages=[p0]
        )
    )

  def test_has_sufficient_space_rejects_legacy_argument_names(self):
    coordinator, _ = _create_coordinator(page_size=4, num_tpu_pages=10)
    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    p0 = Page(page_id=0, location="tpu", ref_count=1)

    with self.assertRaises(TypeError):
      coordinator.has_sufficient_space(req, n_tokens=4)  # type: ignore[call-arg]

    with self.assertRaises(TypeError):
      coordinator.has_sufficient_space(
          req, num_tokens=4, matched_pages=[p0]  # type: ignore[call-arg]
      )


class PrepareRequestPagesTest(parameterized.TestCase):

  def test_prepare_request_pages_insufficient_tpu_pages_raises(self):
    coordinator, _ = _create_coordinator(page_size=4, num_tpu_pages=2)
    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    with self.assertRaisesRegex(
        ValueError,
        r"Cannot schedule request r1\. Need 3 TPU pages, but only 2 are"
        r" available\.",
    ):
      coordinator.prepare_request_pages(req, num_tokens=10)

  def test_prepare_request_pages_tpu_shortfall_handled_correctly(self):
    coordinator, manager = _create_coordinator(page_size=4, num_tpu_pages=5)
    # Allocate 2 TPU pages that become unreferenced
    pids = manager.allocate_tpu_pages(2)
    p0 = _Page(page_id=pids[0], location="tpu", ref_count=0, prefix_hash=1)
    p1 = _Page(page_id=pids[1], location="tpu", ref_count=0, prefix_hash=2)
    coordinator._unreferenced_tpu_pages[p0] = None
    coordinator._unreferenced_tpu_pages[p1] = None

    # Free TPU pages in manager: 3. Unreferenced TPU pages: 2. Total available: 5.
    # Request needs 4 TPU pages -> shortfall of 1 page -> frees 1 unreferenced TPU page
    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    coordinator.prepare_request_pages(req, num_tokens=16)

    state = coordinator._requests[req.request_id]
    self.assertLen(state.pages, 4)
    for page in state.pages:
      self.assertIsNotNone(page)
      assert page is not None
      self.assertEqual(page.ref_count, 1)
      self.assertEqual(page.location, "tpu")

  def test_prepare_request_pages_loads_matched_cpu_pages(self):
    coordinator, manager = _create_coordinator(
        page_size=4, num_tpu_pages=5, num_cpu_pages=5
    )
    pids = manager.allocate_tpu_pages(1)
    manager.offload(pids)
    cpu_page = _Page(
        page_id=pids[0], location="cpu", ref_count=0, prefix_hash=100
    )
    coordinator._unreferenced_cpu_pages[cpu_page] = None

    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    # Pass cpu_page as matched; calculate_page_requirements will need 1 TPU page for it
    coordinator.prepare_request_pages(
        req, num_tokens=4, matched_prefix_pages=[cpu_page]
    )

    self.assertEqual(cpu_page.location, "tpu")
    self.assertEqual(manager.get_page_location(cpu_page.page_id), "tpu")
    self.assertNotIn(cpu_page, coordinator._unreferenced_cpu_pages)

  def test_prepare_request_pages_marks_scheduled_pages_referenced(self):
    coordinator, _ = _create_coordinator(page_size=4, num_tpu_pages=5)
    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    coordinator.prepare_request_pages(req, num_tokens=8)

    state = coordinator._requests[req.request_id]
    self.assertLen(state.pages, 2)
    for p in state.pages:
      self.assertIsNotNone(p)
      assert p is not None
      self.assertEqual(p.ref_count, 1)
      self.assertNotIn(p, coordinator._unreferenced_tpu_pages)
      self.assertNotIn(p, coordinator._unreferenced_cpu_pages)

  def test_prepare_request_pages_protects_matched_unreferenced_pages(self):
    coordinator, manager = _create_coordinator(page_size=4, num_tpu_pages=2)
    pids = manager.allocate_tpu_pages(1)
    matched_p = _Page(
        page_id=pids[0], location="tpu", ref_count=0, prefix_hash=1
    )
    coordinator._unreferenced_tpu_pages[matched_p] = None

    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    coordinator.prepare_request_pages(
        req, num_tokens=4, matched_prefix_pages=[matched_p]
    )

    # matched_p must not be evicted or reallocated; it must now be referenced
    self.assertEqual(matched_p.ref_count, 1)
    self.assertNotIn(matched_p, coordinator._unreferenced_tpu_pages)

  def test_prepare_request_pages_all_used_pages_touched(self):
    coordinator, manager = _create_coordinator(page_size=4, num_tpu_pages=5)
    pids = manager.allocate_tpu_pages(1)
    p0 = _Page(page_id=pids[0], location="tpu", ref_count=0, prefix_hash=10)
    coordinator._unreferenced_tpu_pages[p0] = None

    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    coordinator.prepare_request_pages(
        req, num_tokens=8, matched_prefix_pages=[p0]
    )

    state = coordinator._requests[req.request_id]
    self.assertLen(state.pages, 3)
    for p in state.pages:
      self.assertIsNotNone(p)
      assert p is not None
      self.assertGreater(p.ref_count, 0)
      self.assertNotIn(p, coordinator._unreferenced_tpu_pages)

  @parameterized.parameters(
      (None,),
      (1,),
      (5,),
  )
  def test_prepare_request_pages_parameterized_window_sizes(
      self, window_size: int | None
  ):
    coordinator, _ = _create_coordinator(
        page_size=4, num_tpu_pages=10, window_size=window_size
    )
    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    coordinator.prepare_request_pages(req, num_tokens=12)
    state = coordinator._requests[req.request_id]
    self.assertLen(state.pages, 3)

  def test_prepare_request_pages_stress_cases(self):
    # Prefill with more tokens than window size; verify only passed matched pages are loaded
    coordinator, manager = _create_coordinator(
        page_size=4, num_tpu_pages=20, num_cpu_pages=20, window_size=4
    )
    tokens = list(range(40))
    req = request_lib.Request(req_id="r1", prompt_token_ids=tokens)

    # 1 matched page passed
    pids = manager.allocate_tpu_pages(1)
    matched_page = _Page(
        page_id=pids[0], location="tpu", ref_count=0, prefix_hash=42
    )
    coordinator._unreferenced_tpu_pages[matched_page] = None

    coordinator.prepare_request_pages(
        req, num_tokens=20, matched_prefix_pages=[matched_page]
    )
    state = coordinator._requests[req.request_id]
    # Total tokens = 0 + 1 * 4 + 20 = 24 tokens -> 6 pages total
    self.assertLen(state.pages, 6)
    self.assertEqual(state.pages[0], matched_page)

  def test_prepare_request_pages_with_out_of_window_none_pages(self):
    coordinator, manager = _create_coordinator(
        page_size=4, num_tpu_pages=10, window_size=8
    )
    pids = manager.allocate_tpu_pages(2)
    p2 = Page(page_id=pids[0], location="tpu", ref_count=0, prefix_hash=200)
    p3 = Page(page_id=pids[1], location="tpu", ref_count=0, prefix_hash=300)
    coordinator._unreferenced_tpu_pages[p2] = None
    coordinator._unreferenced_tpu_pages[p3] = None

    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    # matched_prefix_pages with 2 None entries (out-of-window) and 2 matched pages
    coordinator.prepare_request_pages(
        req, num_tokens=4, matched_prefix_pages=[None, None, p2, p3]
    )

    state = coordinator._requests[req.request_id]
    self.assertEqual(state.num_released_pages, 2)
    self.assertEqual(state.num_hashed_pages, 4)
    self.assertEqual(state.last_page_hash, 300)
    self.assertEqual(state.pages[0], None)
    self.assertEqual(state.pages[1], None)
    self.assertEqual(state.pages[2], p2)
    self.assertEqual(state.pages[3], p3)
    self.assertEqual(p2.ref_count, 1)
    self.assertEqual(p3.ref_count, 1)


class PrepareRequestPagesValidationTest(parameterized.TestCase):

  def test_prepare_request_pages_invalid_when_request_already_scheduled(self):
    coordinator, _ = _create_coordinator(page_size=4)
    req = request_lib.Request(req_id="r1", prompt_token_ids=[1, 2, 3, 4])
    req.num_completed_tokens = 4
    p0 = Page(page_id=0, location="tpu")

    with self.assertRaisesRegex(
        ValueError, r"Invalid matched pages for request r1\."
    ):
      coordinator.prepare_request_pages(
          req, num_tokens=4, matched_prefix_pages=[p0]
      )

  def test_prepare_request_pages_invalid_prefix_with_gap(self):
    coordinator, _ = _create_coordinator(page_size=4)
    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    p0 = Page(page_id=0, location="tpu")
    p2 = Page(page_id=2, location="tpu")

    with self.assertRaisesRegex(
        ValueError, r"Invalid matched pages for request r1\."
    ):
      coordinator.prepare_request_pages(
          req, num_tokens=4, matched_prefix_pages=[p0, None, p2]
      )

  def test_prepare_request_pages_invalid_prefix_with_trailing_none(self):
    coordinator, _ = _create_coordinator(page_size=4)
    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    p0 = Page(page_id=0, location="tpu")

    with self.assertRaisesRegex(
        ValueError, r"Invalid matched pages for request r1\."
    ):
      coordinator.prepare_request_pages(
          req, num_tokens=4, matched_prefix_pages=[p0, None]
      )

  def test_prepare_request_pages_invalid_leading_none_when_no_window_size(self):
    coordinator, _ = _create_coordinator(page_size=4, window_size=None)
    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    p1 = Page(page_id=1, location="tpu")

    with self.assertRaisesRegex(
        ValueError, r"Invalid matched pages for request r1\."
    ):
      coordinator.prepare_request_pages(
          req, num_tokens=4, matched_prefix_pages=[None, p1]
      )

  def test_prepare_request_pages_invalid_suffix_with_cache_hit_before_none(self):
    # window_size = 8, page_size = 4 -> 2 pages in window
    coordinator, _ = _create_coordinator(page_size=4, window_size=8)
    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    p0 = Page(page_id=0, location="tpu")
    p2 = Page(page_id=2, location="tpu")
    p3 = Page(page_id=3, location="tpu")

    # p0 is before None, so suffix doesn't contain all cache hits
    with self.assertRaisesRegex(
        ValueError, r"Invalid matched pages for request r1\."
    ):
      coordinator.prepare_request_pages(
          req, num_tokens=4, matched_prefix_pages=[p0, None, p2, p3]
      )

  def test_prepare_request_pages_invalid_suffix_underfilled_window(self):
    # window_size = 8, page_size = 4 -> 2 pages in window
    coordinator, _ = _create_coordinator(page_size=4, window_size=8)
    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    p2 = Page(page_id=2, location="tpu")

    # Suffix only has 1 page, but window needs 2
    with self.assertRaisesRegex(
        ValueError, r"Invalid matched pages for request r1\."
    ):
      coordinator.prepare_request_pages(
          req, num_tokens=4, matched_prefix_pages=[None, None, p2]
      )

  def test_prepare_request_pages_invalid_suffix_overfilled_window(self):
    # window_size = 8, page_size = 4 -> 2 pages in window
    coordinator, _ = _create_coordinator(page_size=4, window_size=8)
    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    p1 = Page(page_id=1, location="tpu")
    p2 = Page(page_id=2, location="tpu")
    p3 = Page(page_id=3, location="tpu")

    # Suffix has 3 pages, but window only spans 2
    with self.assertRaisesRegex(
        ValueError, r"Invalid matched pages for request r1\."
    ):
      coordinator.prepare_request_pages(
          req, num_tokens=4, matched_prefix_pages=[None, p1, p2, p3]
      )

  def test_prepare_request_pages_invalid_all_none_pages(self):
    coordinator, _ = _create_coordinator(page_size=4, window_size=8)
    req = request_lib.Request(req_id="r1", prompt_token_ids=[])

    with self.assertRaisesRegex(
        ValueError, r"Invalid matched pages for request r1\."
    ):
      coordinator.prepare_request_pages(
          req, num_tokens=4, matched_prefix_pages=[None, None]
      )

  def test_prepare_request_pages_invalid_prefix_exceeds_window(self):
    # window_size = 8, page_size = 4 -> 2 pages max in window
    coordinator, _ = _create_coordinator(page_size=4, window_size=8)
    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    p0 = Page(page_id=0, location="tpu")
    p1 = Page(page_id=1, location="tpu")
    p2 = Page(page_id=2, location="tpu")

    # 3 prefix pages exceed window capacity of 2 pages
    with self.assertRaisesRegex(
        ValueError, r"Invalid matched pages for request r1\."
    ):
      coordinator.prepare_request_pages(
          req, num_tokens=4, matched_prefix_pages=[p0, p1, p2]
      )

  def test_prepare_request_pages_valid_empty_matched_pages(self):
    coordinator, _ = _create_coordinator(page_size=4)
    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    coordinator.prepare_request_pages(
        req, num_tokens=4, matched_prefix_pages=()
    )
    self.assertLen(coordinator._requests[req.request_id].pages, 1)

  def test_prepare_request_pages_valid_contiguous_prefix(self):
    coordinator, manager = _create_coordinator(page_size=4, num_tpu_pages=10)
    pids = manager.allocate_tpu_pages(2)
    p0 = Page(page_id=pids[0], location="tpu", ref_count=0)
    p1 = Page(page_id=pids[1], location="tpu", ref_count=0)
    coordinator._unreferenced_tpu_pages[p0] = None
    coordinator._unreferenced_tpu_pages[p1] = None

    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    coordinator.prepare_request_pages(
        req, num_tokens=4, matched_prefix_pages=[p0, p1]
    )
    self.assertLen(coordinator._requests[req.request_id].pages, 3)

  def test_prepare_request_pages_valid_full_window_suffix(self):
    coordinator, manager = _create_coordinator(
        page_size=4, num_tpu_pages=10, window_size=8
    )
    pids = manager.allocate_tpu_pages(2)
    p2 = Page(page_id=pids[0], location="tpu", ref_count=0)
    p3 = Page(page_id=pids[1], location="tpu", ref_count=0)
    coordinator._unreferenced_tpu_pages[p2] = None
    coordinator._unreferenced_tpu_pages[p3] = None

    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    # 2 None entries and 2 pages exactly matching window_size=8 // page_size=4 = 2
    coordinator.prepare_request_pages(
        req, num_tokens=4, matched_prefix_pages=[None, None, p2, p3]
    )
    self.assertLen(coordinator._requests[req.request_id].pages, 5)


class RequestReleaseAndIndicesTest(absltest.TestCase):

  def test_release_request_calls_release_for_all_pages(self):
    coordinator, manager = _create_coordinator(page_size=4, num_tpu_pages=5)
    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    coordinator.prepare_request_pages(req, num_tokens=8)
    state = coordinator._requests[req.request_id]
    pages = list(state.pages)

    self.assertEqual(manager.num_free_tpu_pages, 3)
    coordinator.release_request(req)

    self.assertNotIn(req.request_id, coordinator._requests)
    for p in pages:
      self.assertIsNotNone(p)
      assert p is not None
      self.assertEqual(p.ref_count, 0)
    # Since pages had no prefix_hash, they should be freed immediately
    self.assertEqual(manager.num_free_tpu_pages, 5)

  def test_release_request_unscheduled_no_op(self):
    coordinator, _ = _create_coordinator()
    req = request_lib.Request(req_id="not_scheduled", prompt_token_ids=[])
    coordinator.release_request(req)  # Should not raise error

  def test_get_page_idxs_unscheduled_returns_empty(self):
    coordinator, _ = _create_coordinator()
    req = request_lib.Request(req_id="unknown", prompt_token_ids=[])
    self.assertEmpty(coordinator.get_page_idxs(req))

  def test_get_page_idxs_returns_correct_indices(self):
    coordinator, manager = _create_coordinator(page_size=4, num_tpu_pages=5)
    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    coordinator.prepare_request_pages(req, num_tokens=8)

    state = coordinator._requests[req.request_id]
    expected_indices = []
    for p in state.pages:
      self.assertIsNotNone(p)
      assert p is not None
      expected_indices.append(manager.get_page_idx(p.page_id))
    indices = coordinator.get_page_idxs(req)
    self.assertEqual(indices, expected_indices)

  def test_get_page_idxs_raises_when_active_page_is_none(self):
    coordinator, manager = _create_coordinator(page_size=4, num_tpu_pages=5)
    req = request_lib.Request(req_id="r1", prompt_token_ids=[])
    coordinator.prepare_request_pages(req, num_tokens=8)
    state = coordinator._requests[req.request_id]
    # Simulate an active page being None
    state.pages[0] = None

    with self.assertRaisesRegex(
        ValueError,
        r"Cannot get page indices for released pages\.",
    ):
      coordinator.get_page_idxs(req)

  def test_get_page_idxs_after_out_of_window_pages_released(self):
    coordinator, manager = _create_coordinator(
        page_size=4, num_tpu_pages=10, window_size=4
    )
    req = request_lib.Request(req_id="r1", prompt_token_ids=list(range(16)))
    coordinator.prepare_request_pages(req, num_tokens=16)
    req.num_completed_tokens = 12

    coordinator.release_out_of_window_pages(req)
    state = coordinator._requests[req.request_id]
    self.assertEqual(state.num_released_pages, 2)
    self.assertIsNone(state.pages[0])
    self.assertIsNone(state.pages[1])

    remaining_indices = coordinator.get_page_idxs(req)
    self.assertLen(remaining_indices, 2)
    p2 = state.pages[2]
    p3 = state.pages[3]
    self.assertIsNotNone(p2)
    self.assertIsNotNone(p3)
    assert p2 is not None and p3 is not None
    self.assertEqual(
        remaining_indices,
        [
            manager.get_page_idx(p2.page_id),
            manager.get_page_idx(p3.page_id),
        ],
    )


class UpdateTpuPoolTest(absltest.TestCase):

  def test_update_tpu_pool_calls_page_manager(self):
    coordinator, manager = _create_coordinator(partition_keys=("layer_0",))
    new_arr = jnp.ones((10, 4), dtype=jnp.float32)
    coordinator.update_tpu_pool({"layer_0": new_arr})
    np.testing.assert_array_equal(
        manager.tpu_pool.partition_pages["layer_0"], new_arr
    )


if __name__ == "__main__":
  absltest.main()
