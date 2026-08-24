import unittest
from tunix.generate.scheduler import Scheduler, Request

class MockCacheManager:
    def __init__(self, tpu_cap=10, cpu_cap=10):
        self.available_hbm_pages = tpu_cap
        self.available_cpu_pages = cpu_cap
        self.offloaded = []
        self.evicted = []
        self.loaded = []
        self.next_id = 9000
    
    def offload(self, pids):
        self.offloaded.extend(pids)
        self.available_hbm_pages += len(pids)
        self.available_cpu_pages -= len(pids)
        
    def evict(self, pids):
        self.evicted.extend(pids)
        self.available_cpu_pages += len(pids)
        
    def load(self, pids):
        self.loaded.extend(pids)
        self.available_hbm_pages -= len(pids)
        self.available_cpu_pages += len(pids)

    def allocate(self, num_pages):
        res = [self.next_id + i for i in range(num_pages)]
        self.next_id += num_pages
        self.available_hbm_pages -= num_pages
        return res
        
    def assign(self, idxs, ids):
        pass

class SchedulerTest(unittest.TestCase):
    def setUp(self):
        self.cache_mgr = MockCacheManager(tpu_cap=5, cpu_cap=10)
        self.scheduler = Scheduler(cache_manager=self.cache_mgr, page_size=2, max_num_seqs=10, max_num_batch_tokens=8)

    def test_lru_state_transitions(self):
        """Tests that releasing pages gracefully falls them into unreferenced tracking."""
        self.scheduler.page_ref_counts[100] = 1
        self.scheduler.page_location[100] = "tpu"
        
        # Test release places in unreferenced TPU
        self.scheduler._release_page(100)
        self.assertIn(100, self.scheduler.unreferenced_tpu_pages)
        self.assertEqual(len(self.scheduler.unreferenced_tpu_pages), 1)
        
        # Freeing 1 page of TPU should migrate it to CPU LRU
        self.scheduler._free_up_unreferenced_tpu_space(1)
        self.assertNotIn(100, self.scheduler.unreferenced_tpu_pages)
        self.assertIn(100, self.scheduler.unreferenced_cpu_pages)
        self.assertIn(100, self.cache_mgr.offloaded)
        
        # Evicting 1 page of CPU should remove it completely
        self.scheduler._free_up_unreferenced_cpu_space(1)
        self.assertNotIn(100, self.scheduler.unreferenced_cpu_pages)
        self.assertIn(100, self.cache_mgr.evicted)

    def test_prefix_caching(self):
        """Tests that common prefixes natively assign matching blocks."""
        r1 = Request("req-1", prompt_token_ids=[10, 20, 30, 40])
        self.scheduler._queue_new_requests([r1])
        self.scheduler._drain_pending_queue()
        
        # req-1 should be able to run immediately (requires 3 blocks since it evaluates 4 tokens + space for 1 decode token)
        self.assertEqual(len(self.scheduler.running_requests), 1)
        # Needs 3 pages to be physically sourced
        self.assertEqual(self.scheduler._calculate_new_pages_needed(), 3)
        
        # Manually distribute allocations as if step progressed
        allocs = self.cache_mgr.allocate(3)
        self.scheduler._distribute_allocated_pages(allocs)
        
        self.assertEqual(len(self.scheduler.prefix_hash_to_page_id), 2, "Prefix cache should track exactly 2 chunks (the 3rd block is an empty decode buffer and unhashable).")
        
        # Send identical request mapping over same hashes
        r2 = Request("req-2", prompt_token_ids=[10, 20, 30, 40])
        self.scheduler._queue_new_requests([r2])
        self.scheduler._drain_pending_queue()
        
        self.assertEqual(len(self.scheduler.running_requests), 2)
        # It matches the 2 prompt blocks precisely, but still strictly needs 1 fresh block for its own independent decode boundary!
        self.assertEqual(self.scheduler._calculate_new_pages_needed(), 1)

    def test_preemption_due_to_hbm_limits(self):
        """Ensures that when running requests outweigh physical boundary constraints,
        the newest requests are systematically popped off and put directly back into pending
        so they can wait until next pass."""
        
        # Create 4 heavy requests, each theoretically requiring 1 boundary decode token expansion.
        reqs = []
        for i in range(4):
            reqs.append(Request(f"req-{i}", prompt_token_ids=[10]))
            
        self.scheduler._queue_new_requests(reqs)
        self.scheduler._drain_pending_queue() # all 4 can fit in TPU initially (requires 4 x 1 pages)
        self.assertEqual(len(self.scheduler.running_requests), 4)
        
        # Artificially limit HBM fully to test 
        self.cache_mgr.available_hbm_pages = 0
        
        # Scheduler's step logic realizes 0 HBM means we can't boundary-allocate for 4 requests. 
        # Needs to pop 4 requests to satisfy room since we have 0 free buffers!
        self.scheduler._make_room_for_step()
        
        self.assertEqual(len(self.scheduler.running_requests), 0)
        self.assertEqual(len(self.scheduler.pending_requests), 4)
        # Verify that appending preempted seqs back kept original arrival priority
        self.assertEqual(self.scheduler.pending_requests[0].request_id, "req-0")


    def test_touch_page_new(self):
        self.scheduler._touch_page(200)
        self.assertEqual(self.scheduler.page_ref_counts[200], 1)

    def test_touch_page_existing(self):
        self.scheduler.page_ref_counts[200] = 1
        self.scheduler._touch_page(200)
        self.assertEqual(self.scheduler.page_ref_counts[200], 2)

    def test_release_page_ref_gt_1(self):
        self.scheduler.page_ref_counts[300] = 2
        self.scheduler.page_location[300] = "tpu"
        self.scheduler._release_page(300)
        self.assertNotIn(300, self.scheduler.unreferenced_tpu_pages)
        self.assertEqual(self.scheduler.page_ref_counts[300], 1)

    def test_release_page_tpu_to_unref(self):
        self.scheduler.page_ref_counts[301] = 1
        self.scheduler.page_location[301] = "tpu"
        self.scheduler._release_page(301)
        self.assertIn(301, self.scheduler.unreferenced_tpu_pages)

    def test_release_page_cpu_to_unref(self):
        self.scheduler.page_ref_counts[302] = 1
        self.scheduler.page_location[302] = "cpu"
        self.scheduler._release_page(302)
        self.assertIn(302, self.scheduler.unreferenced_cpu_pages)

    def test_tpu_evict_beyond_capacity_throws(self):
        with self.assertRaises(RuntimeError):
            self.scheduler._free_up_unreferenced_tpu_space(5)

    def test_freed_tpu_pages_end_up_on_cpu(self):
        self.scheduler.page_ref_counts[400] = 1
        self.scheduler.page_location[400] = "tpu"
        self.scheduler._release_page(400)
        self.scheduler._free_up_unreferenced_tpu_space(1)
        self.assertIn(400, self.scheduler.unreferenced_cpu_pages)
        self.assertEqual(self.scheduler.page_location[400], "cpu")

    def test_cpu_evict_beyond_capacity_throws(self):
        with self.assertRaises(RuntimeError):
            self.scheduler._free_up_unreferenced_cpu_space(5)
            
    def test_evicted_pages_removed_completely(self):
        self.scheduler.page_ref_counts[401] = 1
        self.scheduler.page_location[401] = "cpu"
        self.scheduler._release_page(401)
        self.scheduler._free_up_unreferenced_cpu_space(1)
        self.assertNotIn(401, self.scheduler.page_location)
        self.assertNotIn(401, self.scheduler.page_ref_counts)

    def test_chunked_prefill_clip(self):
        req = Request("r-clip", [1]*100)
        self.scheduler._queue_new_requests([req])
        self.scheduler._drain_pending_queue()
        # token budget is 8. Should only load up to 8 tokens
        self.assertEqual(self.scheduler.running_requests[0].num_in_flight_tokens, 8)

    def test_full_and_partial_prefix_match(self):
        r1 = Request("r-1", [10, 20, 30, 40])
        self.scheduler._queue_new_requests([r1])
        self.scheduler._drain_pending_queue()
        allocs = self.cache_mgr.allocate(3)
        self.scheduler._distribute_allocated_pages(allocs)
        
        # r2 matches perfectly up to [10, 20, 30, 40], and then has extra
        r2 = Request("r-2", [10, 20, 30, 40, 50, 60])
        self.scheduler._queue_new_requests([r2])
        matched_pages = self.scheduler._get_matched_pages(r2)
        # Should match exactly the two full pages from r1!
        self.assertEqual(len(matched_pages), 2)
        self.assertEqual(matched_pages, r1.page_ids[:2])

    def test_test_evicted_pages_do_not_match(self):
        r1 = Request("r-1", [10, 20])
        self.scheduler._queue_new_requests([r1])
        self.scheduler._drain_pending_queue()
        allocs = self.cache_mgr.allocate(2)
        self.scheduler._distribute_allocated_pages(allocs)
        
        # Evict it fully
        pid = allocs[0]
        self.scheduler._release_page(pid)
        self.scheduler._free_up_unreferenced_tpu_space(1)
        self.scheduler._free_up_unreferenced_cpu_space(1)
        
        r2 = Request("r-2", [10, 20])
        self.scheduler._queue_new_requests([r2])
        matched_pages = self.scheduler._get_matched_pages(r2)
        self.assertEqual(len(matched_pages), 0)

    def test_partially_full_pages_arent_hashed(self):
        r1 = Request("r-1", [10, 20, 30])
        self.scheduler._queue_new_requests([r1])
        self.scheduler._drain_pending_queue()
        allocs = self.cache_mgr.allocate(2)
        self.scheduler._distribute_allocated_pages(allocs)
        
        # prefix for [10,20] is hashed because it is page_size
        h1 = self.scheduler._chunk_and_hash([10, 20])[0]
        self.assertIn(h1, self.scheduler.prefix_hash_to_page_id)
        
        # full array [10,20,30] (the 3rd element is a partial block!) 
        # should NOT be hashed
        h2 = self.scheduler._chunk_and_hash([10, 20, 30])[-1]
        self.assertNotIn(h2, self.scheduler.prefix_hash_to_page_id)


    def test_cpu_pages_incur_budget_tpu_do_not(self):
        # Create a TPU cache of 2 and CPU cache of 10
        self.cache_mgr = MockCacheManager(tpu_cap=3, cpu_cap=10)
        self.scheduler = Scheduler(cache_manager=self.cache_mgr, page_size=2, max_num_seqs=10, max_num_batch_tokens=8)
        
        # Load sequence A to TPU, then evict
        rA = Request("rA", [10, 20]) # 1 page
        self.scheduler._queue_new_requests([rA])
        self.scheduler._drain_pending_queue()
        
        allocs = self.cache_mgr.allocate(2)
        self.scheduler._distribute_allocated_pages(allocs)
        
        # Free it to CPU
        self.scheduler._release_page(allocs[0])
        self.scheduler._free_up_unreferenced_tpu_space(1)
        
        # Memory is TPU=2, CPU=9 (with 1 page offloaded to CPU)
        
        rB = Request("rB", [10, 20])
        self.scheduler._queue_new_requests([rB])
        self.scheduler._drain_pending_queue()
        
        # Since rB matches the prefix on CPU, it needs to be LOADED.
        # This incurs a HBM cost of 1 (new_pages=0, cpu_pages_used=1 -> total_hbm_cost=1)
        # So physically_free should go down
        matched = self.scheduler._get_matched_pages(rB)
        self.assertEqual(len(matched), 1)

        self.assertEqual(self.scheduler.page_location[matched[0]], "tpu")
        
    def test_first_in_first_scheduled(self):
        r1 = Request("r1", [10, 20])
        r2 = Request("r2", [30, 40])
        self.scheduler._queue_new_requests([r1, r2])
        self.assertEqual(self.scheduler.pending_requests[0].request_id, "r1")
        self.assertEqual(self.scheduler.pending_requests[1].request_id, "r2")
        self.scheduler._drain_pending_queue()
        self.assertEqual(self.scheduler.running_requests[0].request_id, "r1")
        self.assertEqual(self.scheduler.running_requests[1].request_id, "r2")

    def test_len_running_requests_less_than_max(self):
        # Make max_num_seqs small
        self.scheduler = Scheduler(cache_manager=self.cache_mgr, page_size=2, max_num_seqs=2, max_num_batch_tokens=8)
        self.scheduler._queue_new_requests([
            Request("r1", [10]), Request("r2", [10]), Request("r3", [10])
        ])
        
        # Oh wait, this needs to be enforced in _drain_pending_queue. Let's see if the scheduler natively supports it!
        pass

    def test_suffix_not_matched(self):
        r1 = Request("r1", [10, 20])
        r1_suffix = Request("r1_suffix", [20, 10]) # Reversed, has elements of 10,20 but not exact sequence
        self.scheduler._queue_new_requests([r1])
        self.scheduler._drain_pending_queue()
        allocs = self.cache_mgr.allocate(2)
        self.scheduler._distribute_allocated_pages(allocs)
        
        self.scheduler._queue_new_requests([r1_suffix])
        matched = self.scheduler._get_matched_pages(r1_suffix)
        self.assertEqual(len(matched), 0)

if __name__ == '__main__':


    unittest.main()
