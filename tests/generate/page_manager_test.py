import dataclasses
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tunix.generate import page_manager as pm_lib
from tunix.generate import utils


class PageManagerTest(parameterized.TestCase):

  def test_ragged_array_row_idxs_capacity_equals_size(self):
    lens = jnp.array([1, 5, 0, 1, 2])
    data = jnp.zeros((9,))
    ragged = pm_lib.RaggedArray(data=data, lens=lens)
    expected = jnp.array([0, 1, 1, 1, 1, 1, 3, 4, 4])
    np.testing.assert_array_equal(ragged.row_idxs, expected)

  def test_ragged_array_row_idxs_capactiy_greater_than_size(self):
    lens = jnp.array([1, 5, 0, 1, 2])
    data = jnp.zeros((11,))
    ragged = pm_lib.RaggedArray(data=data, lens=lens)
    expected = jnp.array([0, 1, 1, 1, 1, 1, 3, 4, 4, 4, 4])
    np.testing.assert_array_equal(ragged.row_idxs, expected)

  def test_ragged_array_intra_offsets(self):
    lens = jnp.array([1, 5, 0, 1, 2])
    data = jnp.zeros((9,))
    ragged = pm_lib.RaggedArray(data=data, lens=lens)
    expected = jnp.array([0, 0, 1, 2, 3, 4, 0, 0, 1])
    np.testing.assert_array_equal(ragged.intra_offsets, expected)

  def test_cache_config_num_pages_aligns_with_dp_size(self):
    config = pm_lib.TpuCpuPageManagerConfig(
        max_tpu_bytes=1000,
        max_cpu_bytes=0,
        dp_size=4,
        logical_page_sharding="dp_axis",
        page_size=10,
        max_seq_len=10,
        max_num_seqs=5,
        dtype=jnp.int32,
    )
    self.assertEqual(config.num_tpu_pages, 24)

  def test_allocate(self):
    config = pm_lib.TpuCpuPageManagerConfig(
        max_tpu_bytes=128,
        max_cpu_bytes=0,
        dp_size=1,
        page_size=1,
        max_seq_len=10,
        max_num_seqs=5,
        dtype=jnp.int32,
    )

    pm = config.init()
    max_num_pages_per_seq = 10
    self.assertEqual(pm.page_indices.shape, (5, max_num_pages_per_seq))

    q_lens = jnp.array([0, 1, 2, 0, 5])
    pm, allocated_idxs = pm.allocate(q_lens)
    pm = pm.assign(allocated_idxs, q_lens)

    np.testing.assert_array_equal(pm.seq_lens, q_lens)
    np.testing.assert_array_equal(pm.page_indices[1][:1], [0])
    np.testing.assert_array_equal(pm.page_indices[2][:2], [1, 2])
    np.testing.assert_array_equal(pm.page_indices[4][:5], [3, 4, 5, 6, 7])
    np.testing.assert_array_equal(allocated_idxs[:8], [0, 1, 2, 3, 4, 5, 6, 7])

    pages_allocated = 8
    num_pages = config.num_tpu_pages
    expected_remaining_pages = num_pages - pages_allocated
    self.assertEqual(pm.tpu_block.num_available_pages, expected_remaining_pages)

  def test_allocate_max_seq_len_indivisible_by_page_size(self):
    page_size = 3
    config = pm_lib.TpuCpuPageManagerConfig(
        max_tpu_bytes=3000,  # plenty
        dp_size=1,
        page_size=page_size,
        max_seq_len=10,
        max_num_seqs=5,
        dtype=jnp.int32,
    )

    pm = config.init()

    # Verify page indices has correct shape
    max_num_pages_per_seq = 4
    self.assertEqual(pm.page_indices.shape, (5, max_num_pages_per_seq))

    q_lens = jnp.array([1, 5, 3, 0, 5])
    pages_needed = jnp.array(utils.cdiv(q_lens, page_size))

    pm, allocated_idxs = pm.allocate(pages_needed)
    pm = pm.assign(allocated_idxs, pages_needed)

    np.testing.assert_array_equal(pm.seq_lens, pages_needed)
    np.testing.assert_array_equal(pm.page_indices[0][:1], [0])
    np.testing.assert_array_equal(pm.page_indices[1][:2], [1, 2])
    np.testing.assert_array_equal(pm.page_indices[2][:1], [3])
    np.testing.assert_array_equal(pm.page_indices[4][:2], [4, 5])
    np.testing.assert_array_equal(allocated_idxs[:6], [0, 1, 2, 3, 4, 5])

  def test_allocate_twice(self):
    config = pm_lib.TpuCpuPageManagerConfig(
        max_tpu_bytes=3000,
        dp_size=1,
        page_size=1,
        max_seq_len=10,
        max_num_seqs=5,
        dtype=jnp.int32,
    )
    pm = config.init()
    pm, first_allocated_idxs = pm.allocate(jnp.array([0, 1, 2, 0, 5]))
    pm = pm.assign(first_allocated_idxs, jnp.array([0, 1, 2, 0, 5]))
    pm, second_allocated_idxs = pm.allocate(jnp.array([1, 1, 0, 0, 2]))
    pm = pm.assign(second_allocated_idxs, jnp.array([1, 1, 0, 0, 2]))

    expected_seq_lens = jnp.array([1, 2, 2, 0, 7])
    np.testing.assert_array_equal(pm.seq_lens, expected_seq_lens)

    self.assertEqual(pm.page_indices[1][0], 0)
    np.testing.assert_array_equal(pm.page_indices[2][:2], [1, 2])
    np.testing.assert_array_equal(pm.page_indices[4][:5], [3, 4, 5, 6, 7])

    self.assertEqual(pm.page_indices[0][0], 8)
    self.assertEqual(pm.page_indices[1][1], 9)
    np.testing.assert_array_equal(pm.page_indices[4][5:7], [10, 11])

  def test_release(self):
    config = pm_lib.TpuCpuPageManagerConfig(
        max_tpu_bytes=3000,
        dp_size=1,
        page_size=1,
        max_seq_len=10,
        max_num_seqs=5,
        dtype=jnp.int32,
    )
    pm = config.init()
    q_lens = jnp.array([0, 1, 2, 0, 5])
    n_pages = jnp.array([0, 1, 2, 0, 5])
    pm, alloc = pm.allocate(n_pages)
    pm = pm.assign(alloc, n_pages)

    initial_avail = pm.tpu_block.num_available_pages
    should_release = jnp.array([False, True, False, False, True])
    pm = pm.release(should_release)

    self.assertEqual(pm.tpu_block.num_available_pages, initial_avail + 6)
    np.testing.assert_array_equal(pm.seq_lens, jnp.array([0, 0, 2, 0, 0]))

  def test_release_for_window(self):
    config = pm_lib.TpuCpuPageManagerConfig(
        max_tpu_bytes=3000,
        dp_size=1,
        page_size=1,
        max_seq_len=10,
        max_num_seqs=5,
        dtype=jnp.int32,
    )
    pm = config.init()
    pm = dataclasses.replace(pm, window_size=3)

    q_lens1 = jnp.array([0, 1, 2, 0, 3])
    pm, alloc1 = pm.allocate(q_lens1)
    pm = pm.assign(alloc1, q_lens1)
    avail_after_first = pm.tpu_block.num_available_pages
    pm = pm.release_for_window()
    self.assertEqual(pm.tpu_block.num_available_pages, avail_after_first)

    q_lens2 = jnp.array([0, 0, 0, 0, 2])
    pm, alloc2 = pm.allocate(q_lens2)
    pm = pm.assign(alloc2, q_lens2)

    avail_before_release = pm.tpu_block.num_available_pages
    pm = pm.release_for_window()

    self.assertEqual(pm.tpu_block.num_available_pages, avail_before_release + 2)
    expected_seq_lens = jnp.array([0, 1, 2, 0, 3])
    np.testing.assert_array_equal(pm.seq_lens, expected_seq_lens)

  def test_load_values_no_subshape(self):
    config = pm_lib.TpuCpuPageManagerConfig(
        max_tpu_bytes=3000,
        dp_size=1,
        page_size=1,
        max_seq_len=10,
        max_num_seqs=5,
        dtype=jnp.int32,
    )
    pm = config.init()
    q_lens = jnp.array([0, 1, 2, 0, 5])
    n_pages = jnp.array([0, 1, 2, 0, 5])
    pm, alloc = pm.allocate(n_pages)
    pm = pm.assign(alloc, n_pages)

    tokens = jnp.arange(10, 18, dtype=jnp.int32)
    pm = pm.load_values(tokens, q_lens, 'tokens')

    out_tokens = pm.to_array(8, q_lens, 'tokens')
    np.testing.assert_array_equal(out_tokens, tokens)

  def test_load_values_with_subshape(self):
    config = pm_lib.TpuCpuPageManagerConfig(
        max_tpu_bytes=3000,
        dp_size=1,
        page_size=1,
        page_subshape=(2, 4),
        max_seq_len=10,
        max_num_seqs=5,
        dtype=jnp.float32,
    )
    pm = config.init()
    q_lens = jnp.array([0, 1, 2, 0, 5])
    n_pages = jnp.array([0, 1, 2, 0, 5])
    pm, alloc = pm.allocate(n_pages)
    pm = pm.assign(alloc, n_pages)

    values = jnp.arange(8 * 2 * 4, dtype=jnp.float32).reshape((8, 2, 4))
    pm = pm.load_values(values, q_lens, 'tokens')

    out_values = pm.to_array(8, q_lens, 'tokens')
    np.testing.assert_array_equal(out_values, values)

  def test_insert_values_no_subshape(self):
    config = pm_lib.TpuCpuPageManagerConfig(
        max_tpu_bytes=3000,
        dp_size=1,
        page_size=2,
        max_seq_len=10,
        max_num_seqs=3,
        dtype=jnp.int32,
    )
    pm = config.init()

    n_pages1 = jnp.array([1, 1, 0])
    pm, alloc = pm.allocate(n_pages1)
    pm = pm.assign(alloc, n_pages1)

    init_values = jnp.array([42, 43, 44])
    pm = pm.load_values(init_values, jnp.array([1, 2, 0]), 'tokens')

    n_pages2 = jnp.array([1, 1, 1])
    pm, alloc2 = pm.allocate(n_pages2)
    pm = pm.assign(alloc2, n_pages2)

    new_values = jnp.array([99, 100, 101], dtype=jnp.int32)
    valid_mask = jnp.array([True, True, True])
    idxs = jnp.array([1, 2, 0])
    pm = pm.insert_values(new_values, idxs=idxs, valid_mask=valid_mask)

    out_values = pm.to_array(6, jnp.array([2, 3, 1]))

    np.testing.assert_array_equal(
        out_values, jnp.array([42, 99, 43, 44, 100, 101], dtype=jnp.int32))

  def test_insert_values_with_subshape(self):
    config = pm_lib.TpuCpuPageManagerConfig(
        max_tpu_bytes=3000,
        dp_size=1,
        page_size=2,
        page_subshape=(2, 4),
        max_seq_len=10,
        max_num_seqs=3,
        dtype=jnp.float32,
    )
    pm = config.init()

    n_pages1 = jnp.array([1, 1, 0])
    pm, alloc = pm.allocate(n_pages1)
    pm = pm.assign(alloc, n_pages1)

    init_values = jnp.arange(3 * 2 * 4, dtype=jnp.float32).reshape((3, 2, 4))
    pm = pm.load_values(init_values, jnp.array([1, 2, 0]), 'tokens')

    n_pages2 = jnp.array([1, 1, 1])
    pm, alloc2 = pm.allocate(n_pages2)
    pm = pm.assign(alloc2, n_pages2)

    new_values = jnp.arange(3 * 2 * 4, dtype=jnp.float32).reshape(
        (3, 2, 4)) + 100.0
    pm = pm.insert_values(new_values, idxs=jnp.array([1, 2, 0]))

    out_values = pm.to_array(6, jnp.array([2, 3, 1]))

    # seq0: init[0], new[0]
    expected_seq0 = jnp.stack([init_values[0], new_values[0]])
    # seq1: init[1], init[2], new[1]
    expected_seq1 = jnp.stack([init_values[1], init_values[2], new_values[1]])
    # seq2: new[2]
    expected_seq2 = jnp.stack([new_values[2]])
    expected = jnp.concatenate([expected_seq0, expected_seq1, expected_seq2],
                               axis=0)
    np.testing.assert_array_equal(out_values, expected)

  def test_offload_and_load(self):
    config = pm_lib.TpuCpuPageManagerConfig(
        max_tpu_bytes=3000,
        max_cpu_bytes=3000,
        dp_size=1,
        page_size=1,
        max_seq_len=10,
        max_num_seqs=5,
        dtype=jnp.int32,
    )
    pm = config.init()

    q_lens = jnp.array([2, 3, 0, 0, 0])
    n_pages = jnp.array([2, 3, 0, 0, 0])
    pm, alloc = pm.allocate(n_pages)
    pm = pm.assign(alloc, n_pages)
    tokens = jnp.array([42, 43, 44, 45, 46], dtype=jnp.int32)
    pm = pm.load_values(tokens, q_lens, 'tokens')

    # Offload first sequence entirely to CPU
    pages_to_offload = 2
    # The first seq got pages 0 and 1
    tpu_page_idxs = jnp.pad(jnp.array([0, 1]), (0, config.num_tpu_pages - 2))
    pm, cpu_page_idxs = pm.offload(pages_to_offload, tpu_page_idxs)

    # Assert CPU block reclaimed 2 pages (so allocated 2)
    self.assertEqual(pm.cpu_block.num_available_pages, config.num_cpu_pages - 2)
    # Assert TPU block released 2 pages
    self.assertEqual(pm.tpu_block.num_available_pages, config.num_tpu_pages - 3)

    # Load back
    pm, new_tpu_page_idxs = pm.load(
        pages_to_offload,
        jnp.pad(jnp.array([0, 1]), (0, config.num_cpu_pages - 2)))
    self.assertEqual(pm.cpu_block.num_available_pages, config.num_cpu_pages)
    self.assertEqual(pm.tpu_block.num_available_pages, config.num_tpu_pages - 5)


if __name__ == '__main__':
  absltest.main()
