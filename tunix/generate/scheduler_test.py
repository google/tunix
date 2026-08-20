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
        self.scheduler = Scheduler(cache_manager=self.cache_mgr, page_size=2, max_num_seqs=10)

    def test_lru_state_transitions(self):
        """Tests that releasing pages gracefully falls them into unreferenced tracking."""
        self.scheduler.page_ref_counts[100] = 1
        self.scheduler.page_location[100] = "tpu"
        
        # Test release places in unreferenced TPU
        self.scheduler.release_page(100)
        self.assertIn(100, self.scheduler.unreferenced_tpu_pages)
        self.assertEqual(len(self.scheduler.unreferenced_tpu_pages), 1)
        
        # Freeing 1 page of TPU should migrate it to CPU LRU
        self.scheduler._free_up_tpu_space(1)
        self.assertNotIn(100, self.scheduler.unreferenced_tpu_pages)
        self.assertIn(100, self.scheduler.unreferenced_cpu_pages)
        self.assertIn(100, self.cache_mgr.offloaded)
        
        # Evicting 1 page of CPU should remove it completely
        self.scheduler._free_up_cpu_space(1)
        self.assertNotIn(100, self.scheduler.unreferenced_cpu_pages)
        self.assertIn(100, self.cache_mgr.evicted)

    def test_prefix_caching(self):
        """Tests that common prefixes natively assign matching blocks."""
        r1 = Request("req-1", prompt_tokens=[10, 20, 30, 40])
        self.scheduler._queue_new_requests([r1])
        self.scheduler._drain_pending_queue()
        
        # req-1 should be able to run immediately (requires 2 blocks)
        self.assertEqual(len(self.scheduler.running_requests), 1)
        # Needs 2 pages to be physically sourced
        self.assertEqual(self.scheduler._calculate_new_pages_needed(), 2)
        
        # Manually distribute allocations as if step progressed
        allocs = self.cache_mgr.allocate(2)
        self.scheduler._distribute_allocated_pages(allocs)
        
        self.assertEqual(len(self.scheduler.prefix_hash_to_page_id), 2, "Prefix cache should track 2 chunks.")
        
        # Send identical request mapping over same hashes
        r2 = Request("req-2", prompt_tokens=[10, 20, 30, 40])
        self.scheduler._queue_new_requests([r2])
        self.scheduler._drain_pending_queue()
        
        self.assertEqual(len(self.scheduler.running_requests), 2)
        # We shouldn't need ANY new pages for req 2's prompt! It hit 100% caching!
        self.assertEqual(self.scheduler._calculate_new_pages_needed(), 0)

    def test_preemption_due_to_hbm_limits(self):
        """Ensures that when running requests outweigh physical boundary constraints,
        the newest requests are systematically popped off and put directly back into pending
        so they can wait until next pass."""
        
        # Create 4 heavy requests, each theoretically requiring 1 boundary decode token expansion.
        reqs = []
        for i in range(4):
            reqs.append(Request(f"req-{i}", prompt_tokens=[10]))
            
        self.scheduler._queue_new_requests(reqs)
        self.scheduler._drain_pending_queue() # all 4 can fit in TPU initially (requires 4 x 1 pages)
        self.assertEqual(len(self.scheduler.running_requests), 4)
        
        # Artificially limit HBM fully to test 
        self.cache_mgr.available_hbm_pages = 0
        
        # Scheduler's step logic realizes 0 HBM means we can't boundary-allocate for 4 requests. 
        # Needs to pop 4 requests to satisfy room since we have 0 free buffers!
        self.scheduler._make_room_for_allocation()
        
        self.assertEqual(len(self.scheduler.running_requests), 0)
        self.assertEqual(len(self.scheduler.pending_requests), 4)
        # Verify it prioritized popping the newest off the back of the queue
        self.assertEqual(self.scheduler.pending_requests[0].req_id, "req-3")

if __name__ == '__main__':
    unittest.main()
