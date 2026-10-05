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

"""Unit tests for SingleTypeKVCacheManager."""

from collections.abc import Sequence

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from tunix.experimental.generate import single_type_kv_cache_manager
from tunix.experimental.generate import tiered_page_pool

Page = single_type_kv_cache_manager.Page


def _create_manager(
    page_size: int = 4,
    num_device_pages: int = 10,
    num_host_pages: int = 10,
    window_size: int | None = None,
    partition_keys: tuple[str, ...] = ("cache_0",),
) -> single_type_kv_cache_manager.SingleTypeKVCacheManager:
  """Creates a SingleTypeKVCacheManager backed by a TieredPagePoolManager."""
  config = tiered_page_pool.TieredPagePoolConfig(
      page_size=page_size,
      dtype=jnp.float32,
      partition_keys=partition_keys,
      num_device_pages=num_device_pages,
      num_host_pages=num_host_pages,
  )
  return single_type_kv_cache_manager.SingleTypeKVCacheManager(
      page_pool_config=config,
      window_size=window_size,
  )


def _create_unreferenced_device_pages(
    manager: single_type_kv_cache_manager.SingleTypeKVCacheManager,
    num_pages: int,
    prefix_hashes: Sequence[int | None] | None = None,
) -> list[Page]:
  """Allocates device pages and adds them to the unreferenced device pages."""
  pids = manager._page_manager.allocate_device_pages(num_pages)
  pages = []
  for i, pid in enumerate(pids):
    h = prefix_hashes[i] if prefix_hashes is not None else None
    page = Page(page_id=pid, ref_count=0, prefix_hash=h)
    manager._unreferenced_device_pages[page] = None
    pages.append(page)
  return pages


def _create_unreferenced_host_pages(
    manager: single_type_kv_cache_manager.SingleTypeKVCacheManager,
    num_pages: int,
    prefix_hashes: Sequence[int | None] | None = None,
) -> list[Page]:
  """Allocates host pages and adds them to the unreferenced host pages."""
  pids = manager._page_manager.allocate_device_pages(num_pages)
  manager._page_manager.offload(pids)
  pages = []
  for i, pid in enumerate(pids):
    h = prefix_hashes[i] if prefix_hashes is not None else None
    page = Page(page_id=pid, ref_count=0, prefix_hash=h)
    manager._unreferenced_host_pages[page] = None
    pages.append(page)
  return pages


def _assign_request_pages(
    manager: single_type_kv_cache_manager.SingleTypeKVCacheManager,
    request_id: str,
    num_pages: int,
    ref_count: int = 1,
) -> list[Page]:
  """Allocates device pages and assigns them to the given request."""
  pids = manager._page_manager.allocate_device_pages(num_pages)
  pages = [Page(page_id=pid, ref_count=ref_count) for pid in pids]
  manager._request_to_pages[request_id] = list(pages)
  return pages


class PageDataclassTest(absltest.TestCase):

  def test_page_equality_and_hash(self):
    p1 = Page(page_id=1, ref_count=1, prefix_hash=123)
    p2 = Page(page_id=1, ref_count=0, prefix_hash=456)
    p3 = Page(page_id=2, ref_count=1, prefix_hash=123)

    self.assertEqual(p1, p2)
    self.assertNotEqual(p1, p3)
    self.assertEqual(hash(p1), hash(p2))
    self.assertNotEqual(hash(p1), hash(p3))
    self.assertNotEqual(p1, "not_a_page")


class ReferenceAndEvictionManagementTest(parameterized.TestCase):

  def test_touch_page_none(self):
    manager = _create_manager()
    manager._touch_page(None)  # Must be a no-op without error

  def test_touch_page_increments_ref_count(self):
    manager = _create_manager()
    page = Page(page_id=0, ref_count=0)
    manager._touch_page(page)
    self.assertEqual(page.ref_count, 1)
    manager._touch_page(page)
    self.assertEqual(page.ref_count, 2)

  def test_touch_page_removes_from_unreferenced_queues(self):
    manager = _create_manager()
    page = Page(page_id=0, ref_count=0)
    manager._unreferenced_device_pages[page] = None
    manager._unreferenced_host_pages[page] = None

    manager._touch_page(page)
    self.assertNotIn(page, manager._unreferenced_device_pages)
    self.assertNotIn(page, manager._unreferenced_host_pages)

  def test_touch_page_freed_raises(self):
    manager = _create_manager()
    page = Page(page_id=3, is_freed=True)

    with self.assertRaisesRegex(
        ValueError, r"Attempting to touch page 3 which has been freed\."
    ):
      manager._touch_page(page)
    self.assertEqual(page.ref_count, 0)

  def test_release_page_none(self):
    manager = _create_manager()
    manager._release_page(None)  # Must be a no-op without error

  @parameterized.parameters(0, -1)
  def test_release_page_zero_or_negative_ref_count_raises(
      self, ref_count: int
  ):
    manager = _create_manager()
    page = Page(page_id=5, ref_count=ref_count)
    with self.assertRaisesRegex(
        ValueError, r"Cannot release page 5 with no references\."
    ):
      manager._release_page(page)

  def test_release_page_decrements_ref_count(self):
    manager = _create_manager()
    page = Page(page_id=0, ref_count=2, prefix_hash=100)
    manager._release_page(page)
    self.assertEqual(page.ref_count, 1)
    self.assertNotIn(page, manager._unreferenced_device_pages)

  def test_release_page_unhashed_freed_immediately(self):
    manager = _create_manager(num_device_pages=5)
    pids = manager._page_manager.allocate_device_pages(1)
    page = Page(
        page_id=pids[0], ref_count=1, prefix_hash=None
    )

    self.assertEqual(manager._page_manager.num_free_device_pages, 4)
    manager._release_page(page)

    self.assertEqual(page.ref_count, 0)
    self.assertNotIn(page, manager._unreferenced_device_pages)
    self.assertEqual(manager._page_manager.num_free_device_pages, 5)

  def test_release_page_device_added_to_unreferenced_device(self):
    manager = _create_manager(num_device_pages=5)
    pids = manager._page_manager.allocate_device_pages(1)
    page = Page(
        page_id=pids[0], ref_count=1, prefix_hash=999
    )

    manager._release_page(page)

    self.assertEqual(page.ref_count, 0)
    self.assertIn(page, manager._unreferenced_device_pages)
    self.assertEqual(manager._page_manager.num_free_device_pages, 4)

  def test_release_page_host_added_to_unreferenced_host(self):
    manager = _create_manager(num_device_pages=5, num_host_pages=5)
    pids = manager._page_manager.allocate_device_pages(1)
    manager._page_manager.offload(pids)
    page = Page(
        page_id=pids[0], ref_count=1, prefix_hash=888
    )

    manager._release_page(page)

    self.assertEqual(page.ref_count, 0)
    self.assertIn(page, manager._unreferenced_host_pages)

  def test_free_pages_removes_from_unreferenced_queues(self):
    manager = _create_manager()
    pids = manager._page_manager.allocate_device_pages(2)
    p0 = Page(page_id=pids[0], ref_count=0)
    p1 = Page(page_id=pids[1], ref_count=0)
    manager._unreferenced_device_pages[p0] = None
    manager._unreferenced_host_pages[p1] = None

    manager._free_pages([p0, p1])
    self.assertNotIn(p0, manager._unreferenced_device_pages)
    self.assertNotIn(p1, manager._unreferenced_host_pages)

  def test_free_pages_removes_from_prefix_hash_map(self):
    manager = _create_manager()
    pids = manager._page_manager.allocate_device_pages(1)
    page = Page(
        page_id=pids[0], ref_count=0, prefix_hash=12345
    )
    manager._prefix_hash_to_page[12345] = page

    manager._free_pages([page])
    self.assertNotIn(12345, manager._prefix_hash_to_page)
    self.assertIsNone(page.prefix_hash)

  def test_free_pages_does_not_remove_other_page_from_prefix_map(self):
    manager = _create_manager()
    pids = manager._page_manager.allocate_device_pages(2)
    page1 = Page(page_id=pids[0], ref_count=0, prefix_hash=123)
    page2 = Page(page_id=pids[1], ref_count=0, prefix_hash=123)
    manager._prefix_hash_to_page[123] = page2

    manager._free_pages([page1])
    self.assertEqual(manager._prefix_hash_to_page[123], page2)
    self.assertIsNone(page1.prefix_hash)

  def test_free_pages_calls_page_manager_free_with_all_pages(self):
    manager = _create_manager(num_device_pages=10)
    pids = manager._page_manager.allocate_device_pages(3)
    pages = [Page(page_id=pid) for pid in pids]

    self.assertEqual(manager._page_manager.num_free_device_pages, 7)
    manager._free_pages(pages)
    self.assertEqual(manager._page_manager.num_free_device_pages, 10)

  def test_free_pages_referenced_page_raises(self):
    manager = _create_manager()
    pids = manager._page_manager.allocate_device_pages(1)
    page = Page(page_id=pids[0], ref_count=1)

    with self.assertRaisesRegex(
        ValueError, rf"Cannot free page {pids[0]} with 1 references\."
    ):
      manager._free_pages([page])

  def test_free_pages_double_free_raises(self):
    manager = _create_manager()
    pids = manager._page_manager.allocate_device_pages(1)
    page = Page(page_id=pids[0])
    manager._free_pages([page])

    with self.assertRaisesRegex(
        ValueError, rf"Attempting to double-free page {pids[0]}\."
    ):
      manager._free_pages([page])

  def test_free_pages_invalid_page_leaves_state_untouched(self):
    manager = _create_manager(num_device_pages=5)
    (p0,) = _create_unreferenced_device_pages(
        manager, num_pages=1, prefix_hashes=[1]
    )
    manager._prefix_hash_to_page[1] = p0
    pids = manager._page_manager.allocate_device_pages(1)
    referenced = Page(page_id=pids[0], ref_count=1)

    with self.assertRaisesRegex(ValueError, "Cannot free page"):
      manager._free_pages([p0, referenced])

    self.assertFalse(p0.is_freed)
    self.assertEqual(p0.prefix_hash, 1)
    self.assertEqual(manager._prefix_hash_to_page, {1: p0})
    self.assertEqual(list(manager._unreferenced_device_pages), [p0])
    self.assertEqual(manager._page_manager.num_free_device_pages, 3)

  def test_free_pages_duplicate_page_leaves_state_untouched(self):
    manager = _create_manager(num_device_pages=5)
    (page,) = _create_unreferenced_device_pages(
        manager, num_pages=1, prefix_hashes=[1]
    )

    with self.assertRaisesRegex(ValueError, "Cannot free duplicate pages"):
      manager._free_pages([page, page])

    self.assertFalse(page.is_freed)
    self.assertEqual(list(manager._unreferenced_device_pages), [page])
    self.assertEqual(manager._page_manager.num_free_device_pages, 4)

  def test_release_page_after_touch_moves_to_most_recently_used(self):
    manager = _create_manager(num_device_pages=5)
    p0, p1 = _create_unreferenced_device_pages(
        manager, num_pages=2, prefix_hashes=[1, 2]
    )

    manager._touch_page(p0)
    manager._release_page(p0)

    self.assertEqual(list(manager._unreferenced_device_pages), [p1, p0])


class EvictionAndOffloadTest(parameterized.TestCase):

  def test_offload_pages_referenced_page_raises(self):
    manager = _create_manager(num_device_pages=5, num_host_pages=5)
    pids = manager._page_manager.allocate_device_pages(1)
    page = Page(page_id=pids[0], ref_count=1)

    with self.assertRaisesRegex(
        ValueError, rf"Cannot offload page {pids[0]} with 1 references\."
    ):
      manager._offload_pages([page])

    self.assertEqual(manager._page_manager.page_location(pids[0]), "device")
    self.assertEmpty(manager._unreferenced_host_pages)

  @parameterized.parameters(0, -1)
  def test_free_unreferenced_device_pages_zero_or_negative(
      self, num_pages: int
  ):
    manager = _create_manager()
    manager._free_unreferenced_device_pages(num_pages)

  def test_free_unreferenced_device_pages_more_than_available_raises(self):
    manager = _create_manager(num_device_pages=5)
    _create_unreferenced_device_pages(manager, num_pages=1, prefix_hashes=[1])

    with self.assertRaisesRegex(
        ValueError, r"Cannot free 2 device pages, only 1 available\."
    ):
      manager._free_unreferenced_device_pages(2)

  def test_free_unreferenced_device_pages_offload_when_host_available(self):
    manager = _create_manager(num_device_pages=5, num_host_pages=5)
    pages = _create_unreferenced_device_pages(
        manager, num_pages=2, prefix_hashes=[1, 2]
    )
    p0, p1 = pages[0], pages[1]

    self.assertEqual(manager._page_manager.num_free_device_pages, 3)
    self.assertEqual(manager._page_manager.num_free_host_pages, 5)

    manager._free_unreferenced_device_pages(2)

    self.assertEmpty(manager._unreferenced_device_pages)
    self.assertEqual(manager._page_manager.num_free_device_pages, 5)
    self.assertEqual(manager._page_manager.num_free_host_pages, 3)

    p0_location = manager._page_manager.page_location(p0.page_id)
    p1_location = manager._page_manager.page_location(p1.page_id)
    self.assertEqual(p0_location, "host")
    self.assertEqual(p1_location, "host")

    self.assertIn(p0, manager._unreferenced_host_pages)
    self.assertIn(p1, manager._unreferenced_host_pages)

  def test_free_unreferenced_device_pages_host_shortfall_frees_host_pages(self):
    manager = _create_manager(num_device_pages=5, num_host_pages=2)
    host_pages = _create_unreferenced_host_pages(
        manager, num_pages=2, prefix_hashes=[10, 11]
    )
    host_p0, host_p1 = host_pages[0], host_pages[1]
    manager._prefix_hash_to_page[10] = host_p0
    manager._prefix_hash_to_page[11] = host_p1

    self.assertEqual(manager._page_manager.num_free_host_pages, 0)

    device_pages = _create_unreferenced_device_pages(
        manager, num_pages=2, prefix_hashes=[20, 21]
    )
    device_p0, device_p1 = device_pages[0], device_pages[1]

    manager._free_unreferenced_device_pages(2)

    self.assertEmpty(manager._unreferenced_device_pages)
    self.assertNotIn(host_p0, manager._unreferenced_host_pages)
    self.assertNotIn(host_p1, manager._unreferenced_host_pages)
    self.assertIn(device_p0, manager._unreferenced_host_pages)
    self.assertIn(device_p1, manager._unreferenced_host_pages)

    p0_location = manager._page_manager.page_location(device_p0.page_id)
    p1_location = manager._page_manager.page_location(device_p1.page_id)
    self.assertEqual(p0_location, "host")
    self.assertEqual(p1_location, "host")
    self.assertNotIn(10, manager._prefix_hash_to_page)
    self.assertNotIn(11, manager._prefix_hash_to_page)

  def test_free_unreferenced_device_pages_host_shortfall_exceeds_unreferenced(
      self,
  ):
    manager = _create_manager(num_device_pages=5, num_host_pages=2)
    # A referenced host page occupies one host slot but cannot be evicted.
    pm = manager._page_manager
    pm.offload(pm.allocate_device_pages(1))
    (host_p0,) = _create_unreferenced_host_pages(
        manager, num_pages=1, prefix_hashes=[10]
    )
    manager._prefix_hash_to_page[10] = host_p0
    device_p0, device_p1 = _create_unreferenced_device_pages(
        manager, num_pages=2, prefix_hashes=[20, 21]
    )

    # The host shortfall is 2, but only 1 unreferenced host page can be freed.
    manager._free_unreferenced_device_pages(2)

    self.assertTrue(host_p0.is_freed)
    self.assertNotIn(10, manager._prefix_hash_to_page)
    self.assertTrue(device_p0.is_freed)
    self.assertEqual(pm.page_location(device_p1.page_id), "host")
    self.assertEqual(list(manager._unreferenced_host_pages), [device_p1])
    self.assertEmpty(manager._unreferenced_device_pages)
    self.assertEqual(pm.num_free_host_pages, 0)

  def test_free_unreferenced_device_pages_frees_when_no_host_pool(self):
    manager = _create_manager(num_device_pages=5, num_host_pages=0)
    _create_unreferenced_device_pages(
        manager, num_pages=2, prefix_hashes=[1, 2]
    )

    manager._free_unreferenced_device_pages(2)

    self.assertEmpty(manager._unreferenced_device_pages)
    self.assertEqual(manager._page_manager.num_free_device_pages, 5)

  def test_free_unreferenced_device_pages_evicts_lru_first(self):
    manager = _create_manager(num_device_pages=5, num_host_pages=0)
    p0, p1, p2 = _create_unreferenced_device_pages(
        manager, num_pages=3, prefix_hashes=[1, 2, 3]
    )

    manager._free_unreferenced_device_pages(2)

    self.assertTrue(p0.is_freed)
    self.assertTrue(p1.is_freed)
    self.assertFalse(p2.is_freed)
    self.assertEqual(list(manager._unreferenced_device_pages), [p2])

  def test_free_unreferenced_device_pages_partial_room_offloads_newest(self):
    manager = _create_manager(num_device_pages=5, num_host_pages=1)
    p0, p1 = _create_unreferenced_device_pages(
        manager, num_pages=2, prefix_hashes=[1, 2]
    )

    manager._free_unreferenced_device_pages(2)

    pm = manager._page_manager
    self.assertEmpty(manager._unreferenced_device_pages)
    self.assertTrue(p0.is_freed)
    self.assertIsNone(pm.page_location(p0.page_id))
    self.assertEqual(pm.page_location(p1.page_id), "host")
    self.assertEqual(list(manager._unreferenced_host_pages), [p1])
    self.assertEqual(pm.num_free_device_pages, 5)
    self.assertEqual(pm.num_free_host_pages, 0)

  def test_free_unreferenced_device_pages_offloads_in_lru_order(self):
    manager = _create_manager(num_device_pages=5, num_host_pages=2)
    p0, p1, p2 = _create_unreferenced_device_pages(
        manager, num_pages=3, prefix_hashes=[1, 2, 3]
    )

    manager._free_unreferenced_device_pages(3)

    self.assertTrue(p0.is_freed)
    self.assertEqual(list(manager._unreferenced_host_pages), [p1, p2])

  @parameterized.parameters(0, -1)
  def test_free_unreferenced_host_pages_zero_or_negative(self, num_pages: int):
    manager = _create_manager()
    manager._free_unreferenced_host_pages(num_pages)

  def test_free_unreferenced_host_pages_more_than_available_raises(self):
    manager = _create_manager(num_device_pages=5, num_host_pages=5)
    _create_unreferenced_host_pages(manager, num_pages=1, prefix_hashes=[1])

    with self.assertRaisesRegex(
        ValueError, r"Cannot free 3 host pages, only 1 available\."
    ):
      manager._free_unreferenced_host_pages(3)

  def test_free_unreferenced_host_pages_fifo_order_and_count(self):
    manager = _create_manager(num_device_pages=5, num_host_pages=5)
    pages = _create_unreferenced_host_pages(
        manager, num_pages=3, prefix_hashes=[0, 1, 2]
    )
    for p in pages:
      if p.prefix_hash is not None:
        manager._prefix_hash_to_page[p.prefix_hash] = p

    self.assertEqual(manager._page_manager.num_free_host_pages, 2)
    manager._free_unreferenced_host_pages(2)

    self.assertNotIn(pages[0], manager._unreferenced_host_pages)
    self.assertNotIn(pages[1], manager._unreferenced_host_pages)
    self.assertIn(pages[2], manager._unreferenced_host_pages)
    self.assertLen(manager._unreferenced_host_pages, 1)
    self.assertEqual(manager._page_manager.num_free_host_pages, 4)
    self.assertNotIn(0, manager._prefix_hash_to_page)
    self.assertNotIn(1, manager._prefix_hash_to_page)
    self.assertIs(manager._prefix_hash_to_page[2], pages[2])


class SlidingWindowAndOutOfWindowPagesTest(parameterized.TestCase):

  @parameterized.parameters(
      (None, 16),
      (8, 4),
      (4, 0),
  )
  def test_release_out_of_window_within_or_unbounded_releases_no_pages(
      self, window_size: int | None, num_completed_tokens: int
  ):
    manager = _create_manager(window_size=window_size, page_size=4)
    req_id = "req_1"
    _assign_request_pages(manager, req_id, num_pages=4)
    req_pages = manager._request_to_pages[req_id]

    manager._release_out_of_window(
        req_id, num_completed_tokens=num_completed_tokens
    )
    for p in req_pages:
      self.assertIsNotNone(p)

  def test_release_out_of_window_no_request_or_empty_pages(self):
    manager = _create_manager(window_size=4, page_size=4)
    manager._release_out_of_window(request_id="req_999", num_completed_tokens=1)

    manager._request_to_pages["req_1"] = []
    manager._release_out_of_window(request_id="req_1", num_completed_tokens=1)

  def test_release_out_of_window_releases_completed_tokens_outside_window(self):
    manager = _create_manager(window_size=4, page_size=4)
    req_id = "req_1"
    pages = _assign_request_pages(manager, req_id, num_pages=4)
    req_pages = manager._request_to_pages[req_id]

    manager._release_out_of_window(req_id, num_completed_tokens=12)

    self.assertIsNone(req_pages[0])
    self.assertIsNone(req_pages[1])
    self.assertEqual(req_pages[2], pages[2])
    self.assertEqual(req_pages[3], pages[3])
    self.assertEqual(pages[0].ref_count, 0)
    self.assertEqual(pages[1].ref_count, 0)

  def test_release_out_of_window_unaligned_completed_tokens(self):
    manager = _create_manager(window_size=4, page_size=4)
    req_id = "req_1"
    _assign_request_pages(manager, req_id, num_pages=3)
    req_pages = manager._request_to_pages[req_id]

    # The lowest needed token is 7, which lives in page 1.
    manager._release_out_of_window(req_id, num_completed_tokens=11)

    self.assertIsNone(req_pages[0])
    self.assertIsNotNone(req_pages[1])
    self.assertIsNotNone(req_pages[2])

  def test_release_out_of_window_keeps_last_page(self):
    manager = _create_manager(window_size=4, page_size=4)
    req_id = "req_1"
    pages = _assign_request_pages(manager, req_id, num_pages=2)
    req_pages = manager._request_to_pages[req_id]

    # The lowest needed token is 12, which lies past the last page.
    manager._release_out_of_window(req_id, num_completed_tokens=16)

    self.assertIsNone(req_pages[0])
    self.assertEqual(req_pages[1], pages[1])
    self.assertEqual(pages[1].ref_count, 1)

  def test_release_out_of_window_releases_low_index_pages_first(self):
    manager = _create_manager(window_size=4, page_size=4)
    req_id = "req_1"
    pages = _assign_request_pages(manager, req_id, num_pages=3)
    for i, p in enumerate(pages):
      p.prefix_hash = i

    manager._release_out_of_window(req_id, num_completed_tokens=12)

    # Low index pages are evicted first, so that pages near the window can
    # still prefix match.
    self.assertEqual(
        list(manager._unreferenced_device_pages), [pages[0], pages[1]]
    )

  def test_release_out_of_window_incremental_calls(self):
    manager = _create_manager(window_size=4, page_size=4)
    req_id = "req_1"
    _assign_request_pages(manager, req_id, num_pages=4)
    req_pages = manager._request_to_pages[req_id]

    manager._release_out_of_window(req_id, num_completed_tokens=8)
    self.assertIsNone(req_pages[0])

    manager._release_out_of_window(req_id, num_completed_tokens=12)
    self.assertIsNone(req_pages[1])

    manager._release_out_of_window(req_id, num_completed_tokens=12)


class CacheFullPagesTest(absltest.TestCase):

  def test_cache_full_pages_no_completed_full_pages(self):
    manager = _create_manager(page_size=4)
    req_id = "req_1"
    _assign_request_pages(manager, req_id, num_pages=2)

    manager._cache_full_pages(req_id, page_hashes=[])
    self.assertEmpty(manager._prefix_hash_to_page)

  def test_cache_full_pages_registers_new_pages(self):
    manager = _create_manager(page_size=4)
    req_id = "req_1"
    pages = _assign_request_pages(manager, req_id, num_pages=2)

    manager._cache_full_pages(req_id, [100, 200])

    self.assertLen(manager._prefix_hash_to_page, 2)
    self.assertEqual(pages[0].prefix_hash, 100)
    self.assertEqual(pages[1].prefix_hash, 200)
    self.assertEqual(manager._prefix_hash_to_page[100], pages[0])
    self.assertEqual(manager._prefix_hash_to_page[200], pages[1])

  def test_cache_full_pages_more_hashes_than_pages(self):
    manager = _create_manager(page_size=4)
    req_id = "req_1"
    pages = _assign_request_pages(manager, req_id, num_pages=2)

    manager._cache_full_pages(req_id, [100, 200, 300])

    self.assertEqual(
        manager._prefix_hash_to_page, {100: pages[0], 200: pages[1]}
    )

  def test_cache_full_pages_stops_at_released_page(self):
    manager = _create_manager(page_size=4)
    req_id = "req_1"
    pages = _assign_request_pages(manager, req_id, num_pages=2)
    manager._request_to_pages[req_id] = [None, *pages]

    manager._cache_full_pages(req_id, [100, 200, 300])

    self.assertEqual(
        manager._prefix_hash_to_page, {200: pages[0], 300: pages[1]}
    )

  def test_cache_full_pages_does_not_reassign_already_hashed(self):
    manager = _create_manager(page_size=4)
    req_id = "req_1"
    _assign_request_pages(manager, req_id, num_pages=2)

    manager._cache_full_pages(req_id, [100])
    self.assertLen(manager._prefix_hash_to_page, 1)
    first_page = manager._prefix_hash_to_page[100]

    manager._cache_full_pages(req_id, [100, 200])
    self.assertLen(manager._prefix_hash_to_page, 2)
    self.assertEqual(manager._prefix_hash_to_page[100], first_page)

    manager._cache_full_pages(req_id, [100, 200])
    self.assertLen(manager._prefix_hash_to_page, 2)
    self.assertEqual(manager._prefix_hash_to_page[100], first_page)

  def test_cache_full_pages_collision_cached_on_device(self):
    manager = _create_manager(page_size=4, num_device_pages=10)
    req1_id = "req_1"
    p1 = _assign_request_pages(manager, req1_id, num_pages=1)[0]
    manager._cache_full_pages(req1_id, [100])

    req2_id = "req_2"
    p2 = _assign_request_pages(manager, req2_id, num_pages=1)[0]
    req2_pages = manager._request_to_pages[req2_id]

    manager._cache_full_pages(req2_id, [100])

    self.assertEqual(req2_pages[0], p1)
    self.assertEqual(p1.ref_count, 2)
    self.assertEqual(p2.ref_count, 0)
    # The duplicate is never hashed, so it is freed immediately rather than
    # lingering in the unreferenced-device LRU.
    self.assertTrue(p2.is_freed)
    self.assertIsNone(p2.prefix_hash)
    self.assertNotIn(p2, manager._unreferenced_device_pages)
    self.assertEqual(manager._page_manager.num_free_device_pages, 9)
    self.assertEqual(manager._prefix_hash_to_page[100], p1)

  def test_cache_full_pages_collision_cached_on_host(self):
    manager = _create_manager(
        page_size=4, num_device_pages=10, num_host_pages=10
    )
    req1_id = "req_1"
    p1 = _assign_request_pages(manager, req1_id, num_pages=1)[0]
    manager._cache_full_pages(req1_id, [100])

    manager._release_page(p1)
    del manager._request_to_pages[req1_id]
    manager._free_unreferenced_device_pages(1)
    p1_location = manager._page_manager.page_location(p1.page_id)
    self.assertEqual(p1_location, "host")
    self.assertEqual(p1.ref_count, 0)
    self.assertEqual(manager._prefix_hash_to_page[100], p1)

    req2_id = "req_2"
    p2 = _assign_request_pages(manager, req2_id, num_pages=1)[0]
    req2_pages = manager._request_to_pages[req2_id]

    manager._cache_full_pages(req2_id, [100])

    self.assertEqual(manager._prefix_hash_to_page[100], p2)
    self.assertEqual(p2.prefix_hash, 100)
    self.assertEqual(req2_pages[0], p2)
    self.assertIsNone(p1.prefix_hash)
    self.assertTrue(p1.is_freed)
    self.assertNotIn(p1, manager._unreferenced_host_pages)


class SyncRequestStateTest(parameterized.TestCase):

  def test_sync_request_state(self):
    manager = _create_manager(page_size=4, window_size=4)
    req_id = "req_1"
    _assign_request_pages(manager, req_id, num_pages=2)

    manager.sync_request_state(
        req_id, page_hashes=[10, 20], num_completed_tokens=8
    )

    req_pages = manager._request_to_pages[req_id]
    self.assertIsNone(req_pages[0])
    self.assertIsNotNone(req_pages[1])
    self.assertLen(manager._prefix_hash_to_page, 2)

  def test_sync_request_state_caches_before_releasing_out_of_window(self):
    manager = _create_manager(page_size=4, window_size=4)
    req_id = "req_1"
    pages = _assign_request_pages(manager, req_id, num_pages=3)

    manager.sync_request_state(
        req_id, page_hashes=[10, 20], num_completed_tokens=12
    )

    # Pages that leave the window in the same sync are cached first, so they
    # are kept as unreferenced prefix-cache entries instead of being freed.
    self.assertEqual(
        list(manager._unreferenced_device_pages), [pages[0], pages[1]]
    )
    self.assertFalse(pages[0].is_freed)
    self.assertFalse(pages[1].is_freed)
    self.assertEqual(
        manager._prefix_hash_to_page, {10: pages[0], 20: pages[1]}
    )

  def test_sync_request_state_full_attention_only_caches(self):
    manager = _create_manager(page_size=4, window_size=None)
    req_id = "req_1"
    pages = _assign_request_pages(manager, req_id, num_pages=2)

    manager.sync_request_state(
        req_id, page_hashes=[10, 20], num_completed_tokens=8
    )

    self.assertEqual(manager._request_to_pages[req_id], pages)
    for p in pages:
      self.assertEqual(p.ref_count, 1)
    self.assertEqual(
        manager._prefix_hash_to_page, {10: pages[0], 20: pages[1]}
    )


class PrefixMatchingTest(parameterized.TestCase):

  def test_find_longest_cache_hit_empty_cache(self):
    manager = _create_manager(page_size=4, window_size=None)
    self.assertEmpty(manager.find_longest_cache_hit([1, 2, 3]))

  def test_find_longest_cache_hit_full_attention(self):
    manager = _create_manager(page_size=4, window_size=None)
    p0 = Page(page_id=0, prefix_hash=10)
    p1 = Page(page_id=1, prefix_hash=20)
    manager._prefix_hash_to_page[10] = p0
    manager._prefix_hash_to_page[20] = p1

    matched = manager.find_longest_cache_hit([10, 20, 30])
    self.assertEqual(matched, [p0, p1])

  def test_find_longest_cache_hit_full_attention_stops_at_miss(self):
    manager = _create_manager(page_size=4, window_size=None)
    p0 = Page(page_id=0, prefix_hash=10)
    p2 = Page(page_id=2, prefix_hash=30)
    manager._prefix_hash_to_page[10] = p0
    manager._prefix_hash_to_page[30] = p2

    matched = manager.find_longest_cache_hit([10, 20, 30])
    self.assertEqual(matched, [p0])

  def test_find_longest_cache_hit_local_attention_full_prefix(self):
    manager = _create_manager(page_size=4, window_size=8)
    p0 = Page(page_id=0, prefix_hash=10)
    p1 = Page(page_id=1, prefix_hash=20)
    manager._prefix_hash_to_page[10] = p0
    manager._prefix_hash_to_page[20] = p1

    matched = manager.find_longest_cache_hit([10, 20])
    self.assertEqual(matched, [p0, p1])

  def test_find_longest_cache_hit_local_attention_suffix(self):
    manager = _create_manager(page_size=4, window_size=4)
    p2 = Page(page_id=2, prefix_hash=30)
    p3 = Page(page_id=3, prefix_hash=40)
    manager._prefix_hash_to_page[30] = p2
    manager._prefix_hash_to_page[40] = p3

    matched = manager.find_longest_cache_hit([10, 20, 30, 40])
    self.assertEqual(matched, [None, None, p2, p3])

  def test_find_longest_cache_hit_local_attention_incomplete_window(self):
    manager = _create_manager(page_size=4, window_size=8)
    p3 = Page(page_id=3, prefix_hash=40)
    manager._prefix_hash_to_page[40] = p3

    matched = manager.find_longest_cache_hit([10, 20, 30, 40])
    self.assertEmpty(matched)

  def test_find_longest_cache_hit_local_attention_falls_back_past_miss(self):
    # window_size=4 spans 2 pages. The miss at hash 40 breaks the rightmost
    # window, so the lookup falls back to the next full window to the left.
    manager = _create_manager(page_size=4, window_size=4)
    pages = {
        h: Page(page_id=i, prefix_hash=h)
        for i, h in enumerate([10, 20, 30, 50])
    }
    manager._prefix_hash_to_page.update(pages)

    matched = manager.find_longest_cache_hit([10, 20, 30, 40, 50])
    self.assertEqual(matched, [None, pages[20], pages[30]])

  def test_find_longest_cache_hit_local_attention_unaligned_window(self):
    # window_size=6 spans 3 pages, not 2, since the window can straddle pages.
    manager = _create_manager(page_size=4, window_size=6)
    pages = {
        h: Page(page_id=i, prefix_hash=h)
        for i, h in enumerate([20, 30, 40])
    }
    manager._prefix_hash_to_page.update(pages)

    matched = manager.find_longest_cache_hit([10, 20, 30, 40])
    self.assertEqual(matched, [None, pages[20], pages[30], pages[40]])

  @parameterized.parameters(
      (None, 0),
      (4, 2),
      (5, 3),
      (6, 3),
      (8, 3),
  )
  def test_num_pages_in_window(
      self, window_size: int | None, expected_num_pages: int
  ):
    manager = _create_manager(page_size=4, window_size=window_size)
    self.assertEqual(manager._num_pages_in_window, expected_num_pages)


class CalculatePageRequirementsTest(parameterized.TestCase):

  def test_calculate_page_requirements_no_computed_pages(self):
    manager = _create_manager(page_size=4)
    reqs = manager._calculate_page_requirements(
        request_id="req_1", num_tokens=7, num_completed_tokens=0
    )
    self.assertEqual(reqs, (2, 0))

  def test_calculate_page_requirements_computed_device_not_counted(self):
    manager = _create_manager(page_size=4)
    pid0, pid1 = manager._page_manager.allocate_device_pages(2)
    p0 = Page(page_id=pid0)
    p1 = Page(page_id=pid1)

    reqs = manager._calculate_page_requirements(
        request_id="req_1",
        num_tokens=4,
        num_completed_tokens=0,
        computed_pages=[p0, p1],
    )
    self.assertEqual(reqs, (1, 2))

  def test_calculate_page_requirements_computed_host_counted(self):
    manager = _create_manager(page_size=4, num_host_pages=2)
    pid0, pid1 = manager._page_manager.allocate_device_pages(2)
    manager._page_manager.offload([pid0])
    p0 = Page(page_id=pid0)
    p1 = Page(page_id=pid1)

    reqs = manager._calculate_page_requirements(
        request_id="req_1",
        num_tokens=4,
        num_completed_tokens=0,
        computed_pages=[p0, p1],
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
    manager = _create_manager(page_size=4, window_size=window_size)
    reqs = manager._calculate_page_requirements(
        request_id="req_1", num_tokens=10, num_completed_tokens=0
    )
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
    manager = _create_manager(page_size=page_size)
    self.assertEqual(
        manager._calculate_page_requirements(
            request_id="req_1", num_tokens=num_tokens, num_completed_tokens=0
        ),
        (expected_pages, 0),
    )

  def test_calculate_page_requirements_with_none_computed_pages(self):
    manager = _create_manager(page_size=4)
    pid2, pid3 = manager._page_manager.allocate_device_pages(2)
    p2 = Page(page_id=pid2)
    p3 = Page(page_id=pid3)
    reqs = manager._calculate_page_requirements(
        request_id="req_1",
        num_tokens=4,
        num_completed_tokens=0,
        computed_pages=[None, None, p2, p3],
    )
    self.assertEqual(reqs, (1, 2))

  @parameterized.parameters(
      (7, 1, 0),
      (5, 3, 0),
      (8, 1, 1),
      (7, 4, 1),
  )
  def test_calculate_page_requirements_running_request(
      self,
      num_completed_tokens: int,
      num_tokens: int,
      expected_new_pages: int,
  ):
    manager = _create_manager(page_size=4)
    _assign_request_pages(manager, "req_1", num_pages=2)

    reqs = manager._calculate_page_requirements(
        request_id="req_1",
        num_tokens=num_tokens,
        num_completed_tokens=num_completed_tokens,
    )
    self.assertEqual(reqs, (expected_new_pages, 0))

  def test_calculate_page_requirements_counts_released_pages(self):
    manager = _create_manager(page_size=4, window_size=8)
    pages = _assign_request_pages(manager, "req_1", num_pages=2)
    # Page 0 was released for being outside the window.
    manager._request_to_pages["req_1"] = [None, *pages]

    reqs = manager._calculate_page_requirements(
        request_id="req_1", num_tokens=1, num_completed_tokens=12
    )
    self.assertEqual(reqs, (1, 0))


class HasSufficientSpaceTest(absltest.TestCase):

  def test_has_sufficient_space_true_when_enough_free_device(self):
    manager = _create_manager(page_size=4, num_device_pages=10)
    self.assertTrue(
        manager.has_sufficient_space(
            request_id="req_1", num_tokens=7, num_completed_tokens=0
        )
    )

  def test_has_sufficient_space_false_when_not_enough_free_device(self):
    manager = _create_manager(page_size=4, num_device_pages=2)
    self.assertFalse(
        manager.has_sufficient_space(
            request_id="req_1", num_tokens=12, num_completed_tokens=0
        )
    )

  def test_has_sufficient_space_true_with_evictable_pages(self):
    manager = _create_manager(page_size=4, num_device_pages=5)
    _create_unreferenced_device_pages(manager, num_pages=4)

    self.assertTrue(
        manager.has_sufficient_space(
            request_id="req_1", num_tokens=12, num_completed_tokens=0
        )
    )

  def test_has_sufficient_space_false_when_computed_unreferenced_not_evictable(
      self,
  ):
    manager = _create_manager(page_size=4, num_device_pages=5)
    pages = _create_unreferenced_device_pages(manager, num_pages=4)

    self.assertFalse(
        manager.has_sufficient_space(
            request_id="req_1",
            num_tokens=12,
            num_completed_tokens=0,
            computed_pages=pages[:3],
        )
    )

  def test_has_sufficient_space_with_computed_host_pages(self):
    manager = _create_manager(page_size=4, num_device_pages=2)
    pids = manager._page_manager.allocate_device_pages(2)
    manager._page_manager.offload([pids[0]])
    p_host = Page(page_id=pids[0], ref_count=0)

    self.assertFalse(
        manager.has_sufficient_space(
            request_id="req_1",
            num_tokens=4,
            num_completed_tokens=0,
            computed_pages=[p_host],
        )
    )

  def test_has_sufficient_space_with_computed_pages(self):
    manager = _create_manager(page_size=4, num_device_pages=10)
    p0 = Page(page_id=0, ref_count=1)

    self.assertTrue(
        manager.has_sufficient_space(
            request_id="req_1",
            num_tokens=4,
            num_completed_tokens=0,
            computed_pages=[p0],
        )
    )

  def test_has_sufficient_space_running_request_uses_existing_pages(self):
    manager = _create_manager(page_size=4, num_device_pages=2)
    _assign_request_pages(manager, "req_1", num_pages=2)
    self.assertEqual(manager._page_manager.num_free_device_pages, 0)

    self.assertTrue(
        manager.has_sufficient_space(
            "req_1", num_tokens=1, num_completed_tokens=7
        )
    )
    self.assertFalse(
        manager.has_sufficient_space(
            "req_1", num_tokens=1, num_completed_tokens=8
        )
    )


class SwapInBindAndAllocateTest(absltest.TestCase):

  def test_swap_in_pages_ignores_device_and_none_pages(self):
    manager = _create_manager(num_device_pages=5)
    pids = manager._page_manager.allocate_device_pages(1)
    device_page = Page(page_id=pids[0], ref_count=0)

    manager._swap_in_pages([None, device_page])

    self.assertEqual(manager._page_manager.page_location(pids[0]), "device")
    self.assertEqual(manager._page_manager.num_free_device_pages, 4)

  def test_swap_in_pages_unreferenced_host_page_raises(self):
    manager = _create_manager(num_device_pages=5, num_host_pages=5)
    host_page = _create_unreferenced_host_pages(
        manager, num_pages=1, prefix_hashes=[1]
    )[0]

    with self.assertRaisesRegex(
        ValueError, r"Cannot swap in page with no references\."
    ):
      manager._swap_in_pages([host_page])

  def test_swap_in_pages_loads_referenced_host_pages(self):
    manager = _create_manager(num_device_pages=5, num_host_pages=5)
    host_page = _create_unreferenced_host_pages(
        manager, num_pages=1, prefix_hashes=[1]
    )[0]
    manager._touch_page(host_page)

    manager._swap_in_pages([host_page])

    self.assertEqual(
        manager._page_manager.page_location(host_page.page_id), "device"
    )
    self.assertEqual(manager._page_manager.num_free_host_pages, 5)

  def test_swap_in_pages_evicts_unreferenced_device_pages_on_shortfall(self):
    manager = _create_manager(num_device_pages=2, num_host_pages=2)
    host_page = _create_unreferenced_host_pages(
        manager, num_pages=1, prefix_hashes=[1]
    )[0]
    u0, u1 = _create_unreferenced_device_pages(
        manager, num_pages=2, prefix_hashes=[2, 3]
    )
    manager._touch_page(host_page)
    self.assertEqual(manager._page_manager.num_free_device_pages, 0)

    manager._swap_in_pages([host_page])

    pm = manager._page_manager
    self.assertEqual(pm.page_location(host_page.page_id), "device")
    # The LRU unreferenced device page is offloaded to make room.
    self.assertEqual(pm.page_location(u0.page_id), "host")
    self.assertIn(u0, manager._unreferenced_host_pages)
    self.assertEqual(pm.page_location(u1.page_id), "device")
    self.assertIn(u1, manager._unreferenced_device_pages)

  def test_bind_pages_touches_and_extends_request_pages(self):
    manager = _create_manager(num_device_pages=5)
    p0, p1 = _create_unreferenced_device_pages(
        manager, num_pages=2, prefix_hashes=[1, 2]
    )

    manager._bind_pages([None, p0], "req_1")
    manager._bind_pages([p1], "req_1")

    self.assertEqual(manager._request_to_pages["req_1"], [None, p0, p1])
    self.assertEqual(p0.ref_count, 1)
    self.assertEqual(p1.ref_count, 1)
    self.assertEmpty(manager._unreferenced_device_pages)

  def test_allocate_device_pages_zero_is_no_op(self):
    manager = _create_manager(num_device_pages=5)
    manager._allocate_device_pages(0, "req_1")

    self.assertNotIn("req_1", manager._request_to_pages)
    self.assertEqual(manager._page_manager.num_free_device_pages, 5)

  def test_allocate_device_pages_evicts_unreferenced_pages_on_shortfall(self):
    manager = _create_manager(num_device_pages=3, num_host_pages=0)
    u0, u1 = _create_unreferenced_device_pages(
        manager, num_pages=2, prefix_hashes=[1, 2]
    )

    manager._allocate_device_pages(2, "req_1")

    req_pages = manager._request_to_pages["req_1"]
    self.assertLen(req_pages, 2)
    for p in req_pages:
      assert p is not None
      self.assertEqual(p.ref_count, 1)
      self.assertEqual(manager._page_manager.page_location(p.page_id), "device")
    self.assertTrue(u0.is_freed)
    self.assertIsNone(manager._page_manager.page_location(u0.page_id))
    self.assertIn(u1, manager._unreferenced_device_pages)


class AllocateSlotsTest(parameterized.TestCase):

  def test_allocate_slots_insufficient_device_pages_raises(self):
    manager = _create_manager(page_size=4, num_device_pages=2)
    with self.assertRaisesRegex(
        ValueError,
        r"Cannot allocate 10 slots for request req_1\. Insufficient space\.",
    ):
      manager.allocate_slots(
          request_id="req_1", num_tokens=10, num_completed_tokens=0
      )

  def test_allocate_slots_device_shortfall_handled_correctly(self):
    manager = _create_manager(page_size=4, num_device_pages=5)
    _create_unreferenced_device_pages(
        manager, num_pages=2, prefix_hashes=[1, 2]
    )

    manager.allocate_slots(
        request_id="req_1", num_tokens=16, num_completed_tokens=0
    )

    req_pages = manager._request_to_pages["req_1"]
    self.assertLen(req_pages, 4)
    for page in req_pages:
      self.assertIsNotNone(page)
      assert page is not None
      self.assertEqual(page.ref_count, 1)
      self.assertEqual(
          manager._page_manager.page_location(page.page_id), "device"
      )

  def test_allocate_slots_loads_computed_host_pages(self):
    manager = _create_manager(
        page_size=4, num_device_pages=5, num_host_pages=5
    )
    host_page = _create_unreferenced_host_pages(
        manager, num_pages=1, prefix_hashes=[100]
    )[0]

    manager.allocate_slots(
        request_id="req_1",
        num_tokens=4,
        num_completed_tokens=0,
        computed_pages=[host_page],
    )

    pid = host_page.page_id
    self.assertEqual(manager._page_manager.page_location(pid), "device")
    self.assertNotIn(host_page, manager._unreferenced_host_pages)

  def test_allocate_slots_marks_scheduled_pages_referenced(self):
    manager = _create_manager(page_size=4, num_device_pages=5)
    manager.allocate_slots(
        request_id="req_1", num_tokens=8, num_completed_tokens=0
    )

    req_pages = manager._request_to_pages["req_1"]
    self.assertLen(req_pages, 2)
    for p in req_pages:
      self.assertIsNotNone(p)
      assert p is not None
      self.assertEqual(p.ref_count, 1)
      self.assertNotIn(p, manager._unreferenced_device_pages)
      self.assertNotIn(p, manager._unreferenced_host_pages)

  def test_allocate_slots_protects_computed_unreferenced_pages(self):
    manager = _create_manager(page_size=4, num_device_pages=2)
    matched_p = _create_unreferenced_device_pages(
        manager, num_pages=1, prefix_hashes=[1]
    )[0]

    manager.allocate_slots(
        request_id="req_1",
        num_tokens=4,
        num_completed_tokens=0,
        computed_pages=[matched_p],
    )

    self.assertEqual(matched_p.ref_count, 1)
    self.assertNotIn(matched_p, manager._unreferenced_device_pages)

  def test_allocate_slots_all_used_pages_touched(self):
    manager = _create_manager(page_size=4, num_device_pages=5)
    p0 = _create_unreferenced_device_pages(
        manager, num_pages=1, prefix_hashes=[10]
    )[0]

    manager.allocate_slots(
        request_id="req_1",
        num_tokens=8,
        num_completed_tokens=0,
        computed_pages=[p0],
    )

    req_pages = manager._request_to_pages["req_1"]
    self.assertLen(req_pages, 3)
    for p in req_pages:
      self.assertIsNotNone(p)
      assert p is not None
      self.assertGreater(p.ref_count, 0)
      self.assertNotIn(p, manager._unreferenced_device_pages)

  @parameterized.parameters(
      (None,),
      (12,),
      (16,),
  )
  def test_allocate_slots_parameterized_window_sizes(
      self, window_size: int | None
  ):
    manager = _create_manager(
        page_size=4, num_device_pages=10, window_size=window_size
    )
    manager.allocate_slots(
        request_id="req_1", num_tokens=12, num_completed_tokens=0
    )
    req_pages = manager._request_to_pages["req_1"]
    self.assertLen(req_pages, 3)

  def test_allocate_slots_more_than_window_size_raises(self):
    manager = _create_manager(page_size=4, num_device_pages=10, window_size=4)
    with self.assertRaisesRegex(
        ValueError,
        "Cannot allocate more than window size tokens in a single step.",
    ):
      manager.allocate_slots(
          request_id="req_1", num_tokens=5, num_completed_tokens=0
      )

  def test_allocate_slots_stress_cases(self):
    manager = _create_manager(
        page_size=4, num_device_pages=20, num_host_pages=20, window_size=20
    )
    matched_page = _create_unreferenced_device_pages(
        manager, num_pages=1, prefix_hashes=[42]
    )[0]

    manager.allocate_slots(
        request_id="req_1",
        num_tokens=20,
        num_completed_tokens=0,
        computed_pages=[matched_page],
    )
    req_pages = manager._request_to_pages["req_1"]
    self.assertLen(req_pages, 6)
    self.assertEqual(req_pages[0], matched_page)

  def test_allocate_slots_with_out_of_window_none_pages(self):
    manager = _create_manager(
        page_size=4, num_device_pages=10, window_size=8
    )
    pages = _create_unreferenced_device_pages(
        manager, num_pages=2, prefix_hashes=[200, 300]
    )
    p2, p3 = pages[0], pages[1]

    manager.allocate_slots(
        request_id="req_1",
        num_tokens=4,
        num_completed_tokens=0,
        computed_pages=[None, None, p2, p3],
    )

    req_pages = manager._request_to_pages["req_1"]
    # 4 computed pages are bound, plus 1 new page for the 4 scheduled tokens.
    self.assertLen(req_pages, 5)
    self.assertIsNone(req_pages[0])
    self.assertIsNone(req_pages[1])
    self.assertEqual(req_pages[2], p2)
    self.assertEqual(req_pages[3], p3)
    self.assertEqual(p2.ref_count, 1)
    self.assertEqual(p3.ref_count, 1)

  def test_allocate_slots_running_request_appends_pages(self):
    manager = _create_manager(page_size=4, num_device_pages=5)
    manager.allocate_slots("req_1", num_tokens=8, num_completed_tokens=0)
    existing_pages = list(manager._request_to_pages["req_1"])

    # The next token fits in the last page.
    manager.allocate_slots("req_1", num_tokens=1, num_completed_tokens=7)
    self.assertEqual(manager._request_to_pages["req_1"], existing_pages)

    # The next token crosses a page boundary.
    manager.allocate_slots("req_1", num_tokens=1, num_completed_tokens=8)
    req_pages = manager._request_to_pages["req_1"]
    self.assertLen(req_pages, 3)
    self.assertEqual(req_pages[:2], existing_pages)
    self.assertEqual(manager._page_manager.num_free_device_pages, 2)

  def test_allocate_slots_computed_host_page_under_device_pressure(self):
    manager = _create_manager(page_size=4, num_device_pages=2, num_host_pages=2)
    host_page = _create_unreferenced_host_pages(
        manager, num_pages=1, prefix_hashes=[100]
    )[0]
    _create_unreferenced_device_pages(
        manager, num_pages=2, prefix_hashes=[1, 2]
    )

    manager.allocate_slots(
        "req_1",
        num_tokens=1,
        num_completed_tokens=0,
        computed_pages=[host_page],
    )

    # 1 computed page, plus 1 new page for the scheduled token.
    req_pages = manager._request_to_pages["req_1"]
    self.assertLen(req_pages, 2)
    self.assertEqual(req_pages[0], host_page)
    pm = manager._page_manager
    self.assertEqual(pm.page_location(host_page.page_id), "device")
    self.assertEqual(pm.num_free_device_pages, 0)


class PrefixCacheLifecycleTest(absltest.TestCase):

  def test_cached_pages_are_reused_by_next_request(self):
    manager = _create_manager(page_size=4, num_device_pages=5)
    manager.allocate_slots("req_1", num_tokens=8, num_completed_tokens=0)
    req1_pages = list(manager._request_to_pages["req_1"])
    manager.sync_request_state(
        "req_1", page_hashes=[10, 20], num_completed_tokens=8
    )
    manager.release_request("req_1")
    # Cached pages are kept after release.
    self.assertEqual(manager._page_manager.num_free_device_pages, 3)

    hits = manager.find_longest_cache_hit([10, 20, 30])
    self.assertEqual(hits, req1_pages)
    manager.allocate_slots(
        "req_2", num_tokens=4, num_completed_tokens=0, computed_pages=hits
    )

    req2_pages = manager._request_to_pages["req_2"]
    self.assertLen(req2_pages, 3)
    self.assertEqual(req2_pages[:2], req1_pages)
    for p in req2_pages:
      assert p is not None
      self.assertEqual(p.ref_count, 1)
    self.assertEmpty(manager._unreferenced_device_pages)
    self.assertEqual(manager._page_manager.num_free_device_pages, 2)

  def test_local_attention_releases_pages_for_reuse(self):
    manager = _create_manager(page_size=4, num_device_pages=10, window_size=4)
    for step in range(3):
      num_completed_tokens = 4 * step
      manager.allocate_slots(
          "req_1", num_tokens=4, num_completed_tokens=num_completed_tokens
      )
      manager.sync_request_state(
          "req_1",
          page_hashes=[10, 20, 30][: step + 1],
          num_completed_tokens=num_completed_tokens + 4,
      )

    req_pages = manager._request_to_pages["req_1"]
    self.assertIsNone(req_pages[0])
    self.assertIsNone(req_pages[1])
    p2 = req_pages[2]
    assert p2 is not None
    p0 = manager._prefix_hash_to_page[10]
    p1 = manager._prefix_hash_to_page[20]
    # Out-of-window pages are released lowest index first.
    self.assertEqual(list(manager._unreferenced_device_pages), [p0, p1])
    self.assertEqual(manager._page_manager.num_free_device_pages, 7)

    # A new request can match the last window of released and live pages.
    self.assertEqual(
        manager.find_longest_cache_hit([10, 20, 30]), [None, p1, p2]
    )


class RequestReleaseAndIndicesTest(absltest.TestCase):

  def test_release_request_calls_release_for_all_pages(self):
    manager = _create_manager(page_size=4, num_device_pages=5)
    req_id = "req_1"
    manager.allocate_slots(
        req_id, num_tokens=8, num_completed_tokens=0
    )
    pages = list(manager._request_to_pages[req_id])

    self.assertEqual(manager._page_manager.num_free_device_pages, 3)
    manager.release_request(req_id)

    self.assertNotIn(req_id, manager._request_to_pages)
    for p in pages:
      self.assertIsNotNone(p)
      assert p is not None
      self.assertEqual(p.ref_count, 0)
    self.assertEqual(manager._page_manager.num_free_device_pages, 5)

  def test_release_request_unscheduled_no_op(self):
    manager = _create_manager()
    manager.release_request(request_id="req_999")

  def test_release_request_evicts_rightmost_pages_first(self):
    manager = _create_manager(page_size=4, num_device_pages=5)
    pages = _assign_request_pages(manager, "req_1", num_pages=3)
    for i, p in enumerate(pages):
      p.prefix_hash = i

    manager.release_request("req_1")

    # The leftmost pages are evicted last, so they stay available for prefix
    # matching the longest.
    self.assertEqual(
        list(manager._unreferenced_device_pages), [pages[2], pages[1], pages[0]]
    )

  def test_get_page_idxs_unscheduled_returns_empty(self):
    manager = _create_manager()
    self.assertEmpty(manager.get_page_idxs(request_id="req_999"))

  def test_get_page_idxs_returns_correct_indices(self):
    manager = _create_manager(page_size=4, num_device_pages=5)
    req_id = "req_1"
    manager.allocate_slots(
        req_id, num_tokens=8, num_completed_tokens=0
    )

    req_pages = manager._request_to_pages[req_id]
    expected_indices = []
    for p in req_pages:
      self.assertIsNotNone(p)
      assert p is not None
      expected_indices.append(manager._page_manager.page_idx(p.page_id))
    indices = manager.get_page_idxs(req_id)
    self.assertEqual(indices, expected_indices)

  def test_get_page_idxs_returns_placeholder_when_active_page_is_none(self):
    manager = _create_manager(page_size=4, num_device_pages=5)
    req_id = "req_1"
    manager.allocate_slots(
        req_id, num_tokens=8, num_completed_tokens=0
    )
    req_pages = manager._request_to_pages[req_id]
    req_pages[0] = None

    indices = manager.get_page_idxs(req_id)

    # Released pages keep their slot, reported as -1, so that the indices stay
    # aligned with the request's page positions.
    p1 = req_pages[1]
    self.assertIsNotNone(p1)
    assert p1 is not None
    expected_indices = [-1, manager._page_manager.page_idx(p1.page_id)]
    self.assertEqual(
        indices,
        expected_indices
    )

  def test_get_page_idxs_after_out_of_window_pages_released(self):
    manager = _create_manager(
        page_size=4, num_device_pages=10, window_size=4
    )
    req_id = "req_1"
    _assign_request_pages(manager, req_id, num_pages=4)

    manager._release_out_of_window(req_id, num_completed_tokens=12)
    req_pages = manager._request_to_pages[req_id]
    self.assertIsNone(req_pages[0])
    self.assertIsNone(req_pages[1])

    remaining_indices = manager.get_page_idxs(req_id)
    self.assertLen(remaining_indices, 4)
    p2 = req_pages[2]
    p3 = req_pages[3]
    self.assertIsNotNone(p2)
    self.assertIsNotNone(p3)
    assert p2 is not None and p3 is not None
    # The released pages are reported as -1 rather than dropped, so that the
    # indices stay aligned with the request's page positions.
    self.assertEqual(
        remaining_indices,
        [
            -1,
            -1,
            manager._page_manager.page_idx(p2.page_id),
            manager._page_manager.page_idx(p3.page_id),
        ],
    )

  def test_get_page_idxs_unknown_page_raises(self):
    manager = _create_manager()
    manager._request_to_pages["req_1"] = [Page(page_id=999, ref_count=1)]

    with self.assertRaisesRegex(
        ValueError, r"Page 999 not found in page manager\."
    ):
      manager.get_page_idxs("req_1")


class UpdateDevicePoolAndPropertiesTest(absltest.TestCase):

  def test_update_device_pool_calls_page_manager(self):
    manager = _create_manager(partition_keys=("cache_0",))
    new_arr = jnp.ones((10, 4), dtype=jnp.float32)
    manager.update_device_pool({"cache_0": new_arr})
    np.testing.assert_array_equal(
        manager._page_manager.physical_device_pages["cache_0"], new_arr
    )

  def test_properties_and_physical_pages(self):
    manager = _create_manager(
        page_size=4,
        window_size=8,
        partition_keys=("cache_0", "cache_1"),
    )
    self.assertEqual(manager.cache_names, ("cache_0", "cache_1"))
    self.assertEqual(manager.window_size, 8)
    self.assertEqual(manager._num_pages_in_window, 3)
    pages = manager.get_physical_pages()
    self.assertIn("cache_0", pages)
    self.assertIn("cache_1", pages)

  def test_reset_kv_caches_frees_all_pages(self):
    manager = _create_manager(page_size=4, num_device_pages=5, num_host_pages=5)
    manager.allocate_slots("req_1", num_tokens=8, num_completed_tokens=0)
    manager.sync_request_state(
        "req_1", page_hashes=[10], num_completed_tokens=8
    )
    _create_unreferenced_device_pages(manager, num_pages=1, prefix_hashes=[20])
    _create_unreferenced_host_pages(manager, num_pages=1, prefix_hashes=[30])

    manager.reset_kv_caches()

    pm = manager._page_manager
    self.assertEqual(pm.num_free_device_pages, 5)
    self.assertEqual(pm.num_free_host_pages, 5)
    self.assertEmpty(manager._request_to_pages)
    self.assertEmpty(manager._prefix_hash_to_page)
    self.assertEmpty(manager._unreferenced_device_pages)
    self.assertEmpty(manager._unreferenced_host_pages)

  def test_reset_kv_caches_restores_full_capacity_and_purges_prefix_cache(
      self,
  ):
    manager = _create_manager(page_size=4, num_device_pages=5, num_host_pages=5)
    # A finished request leaves its full pages in the prefix cache.
    manager.allocate_slots("req_1", num_tokens=8, num_completed_tokens=0)
    manager.sync_request_state(
        "req_1", page_hashes=[10, 20], num_completed_tokens=8
    )
    manager.release_request("req_1")
    # A running request still holds its pages.
    manager.allocate_slots("req_2", num_tokens=8, num_completed_tokens=0)
    self.assertLen(manager.find_longest_cache_hit([10, 20]), 2)

    manager.reset_kv_caches()

    self.assertEmpty(manager.find_longest_cache_hit([10, 20]))
    self.assertEqual(manager._page_manager.num_free_device_pages, 5)
    self.assertTrue(
        manager.has_sufficient_space(
            "req_3", num_tokens=20, num_completed_tokens=0
        )
    )
    manager.allocate_slots("req_3", num_tokens=20, num_completed_tokens=0)
    self.assertLen(manager.get_page_idxs("req_3"), 5)


if __name__ == "__main__":
  absltest.main()
