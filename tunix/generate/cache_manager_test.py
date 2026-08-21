import unittest
from tunix.generate.cache_manager import CacheManager

class MockPageManager:
    def __init__(self, num_pages=50, batch_size=8, max_seq_pages=16):
        self.num_available_pages = num_pages
        self.total_num_pages = num_pages
        self.batch_size = batch_size
        self.max_num_pages_per_seq = max_seq_pages
        self.max_num_seqs = batch_size * 2
        self.calls = []

    def allocate(self, num_pages):
        num = int(num_pages)
        res = [100 + i for i in range(num)]
        self.num_available_pages -= num
        return self, res

    def assign(self, idxs, ids, lens):
        self.calls.append(("assign", idxs, ids, lens))
        return self

    def evict_pages(self, idxs, count):
        self.calls.append(("evict", idxs, count))
        self.num_available_pages += int(count)
        return self

class CacheManagerTest(unittest.TestCase):
    def setUp(self):
        self.hbm = MockPageManager(num_pages=10)
        self.cpu = MockPageManager(num_pages=20)
        self.cache_manager = CacheManager(
            hbm_page_manager=self.hbm,
            offload_page_manager=self.cpu
        )

    def test_allocate(self):
        """Test allocation updates internal capacities and constructs mapping cleanly."""
        hbm_initial = self.cache_manager.available_hbm_pages
        
        assigned_ids = self.cache_manager.allocate(3)
        self.assertEqual(len(assigned_ids), 3)
        self.assertEqual(assigned_ids, [0, 1, 2])
        
        self.assertEqual(self.cache_manager.available_hbm_pages, hbm_initial - 3)
        self.assertEqual(self.cache_manager._page_location[0], "tpu")
        self.assertEqual(self.cache_manager._page_id_to_idx[0], 100) # Provided by Mock
        
    def test_evict(self):
        """Evict should cleanly tear down physical components back to their source pools."""
        assigned_ids = self.cache_manager.allocate(4)
        
        self.cache_manager.evict([0, 2]) # evict partial logical IDs
        
        # Verify internal mapping purged
        self.assertNotIn(0, self.cache_manager._page_location)
        self.assertNotIn(2, self.cache_manager._page_id_to_idx)
        
        # Should persist 1 and 3
        self.assertIn(1, self.cache_manager._page_location)
        self.assertIn(3, self.cache_manager._page_id_to_idx)
        
        # Verify mock HBM got the correct evict call
        self.assertEqual(len(self.hbm.calls), 1)
        self.assertEqual(self.hbm.calls[0][0], "evict")
        # Ensure count matches accurately back down to mock!
        self.assertEqual(int(self.hbm.calls[0][2]), 2) 
        
    def test_assign(self):
        """Test structured batching correctly pads and filters sequence structures."""
        assigned_ids = self.cache_manager.allocate(5) # logical 0, 1, 2, 3, 4 -> phys 100, 101...
        
        # Seq 0 uses [0, 1], Seq 1 uses [2, 3, 4]
        sseq_ids = [0, 1]
        sseq_page_ids = [[0, 1], [2, 3, 4]]
        
        self.cache_manager.assign(sseq_page_ids)
        
        self.assertEqual(len(self.hbm.calls), 1)
        call_type, idxs, ids, lens = self.hbm.calls[0]
        
        self.assertEqual(call_type, "assign")
        self.assertEqual(int(idxs[0]), 0) # seq 0
        self.assertEqual(int(idxs[1]), 1) # seq 1
        
        self.assertEqual(int(lens[0]), 2) # length of seq 0 page array
        self.assertEqual(int(lens[1]), 3) # length of seq 1 page array
        
        self.assertEqual(int(ids[0]), 100) # 0 maps to physical 100
        self.assertEqual(int(ids[2]), 102) # 2 maps to physical 102
        
    def test_allocate_oom(self):
        """Tests that exceeding physically available boundaries throws Runtime exception."""
        with self.assertRaises(RuntimeError):
            self.cache_manager.allocate(11) # Capacity is 10

if __name__ == '__main__':
    unittest.main()
