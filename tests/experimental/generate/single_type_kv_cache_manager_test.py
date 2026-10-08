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


if __name__ == "__main__":
  absltest.main()
