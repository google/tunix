import dataclasses
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tunix.generate import page_manager as pm_lib


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

    # This test verifies the number of pages per layer block is
    # properly aligned with dp_size.

    # block1_page_size_bytes = page_size * dtype_size = 40
    # max_pages_per_layer_block = max_bytes // 40 = 25 
    # pages_per_block = (25 // dp_size) * dp_size = 24

    self.assertEqual(config.num_tpu_pages, 24)

  def test_allocate(self):
    config = pm_lib.TpuCpuPageManagerConfig(
        max_tpu_bytes=1300,
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

    np.testing.assert_array_equal(pm.seq_lens, q_lens)
    np.testing.assert_array_equal(pm.page_indices[1][:1], [0])
    np.testing.assert_array_equal(pm.page_indices[2][:2], [1, 2])
    np.testing.assert_array_equal(pm.page_indices[4][:5], [3, 4, 5, 6, 7])
    np.testing.assert_array_equal(allocated_idxs[:8], [0, 1, 2, 3, 4, 5, 6, 7])

    pages_allocated = 8
    num_pages = config.num_tpu_pages
    expected_remaining_pages = num_pages - pages_allocated
    self.assertEqual(
        pm.tpu_block.num_available_pages, expected_remaining_pages 
    )

  def test_allocate_max_seq_len_indivisible_by_page_size(self):
    config = pm_lib.TpuCpuPageManagerConfig(
        max_tpu_bytes=130000,
        dp_size=1,
        page_size=3,
        max_seq_len=10,
        max_num_seqs=5,
        dtype=jnp.int32,
    )

    pm = config.init()

    # Verify page indices has correct shape 
    max_num_pages_per_seq = 4 
    self.assertEqual(pm.page_indices.shape, (5, max_num_pages_per_seq))

    q_lens = jnp.array([1, 5, 3, 0, 5])
    pm, allocated_idxs = pm.allocate(q_lens)

    np.testing.assert_array_equal(pm.seq_lens, q_lens)
    np.testing.assert_array_equal(pm.page_indices[0][:1], [0])
    np.testing.assert_array_equal(pm.page_indices[1][:2], [1, 2])
    np.testing.assert_array_equal(pm.page_indices[2][:1], [3])
    np.testing.assert_array_equal(pm.page_indices[4][:2], [4, 5])
    np.testing.assert_array_equal(allocated_idxs[:6], [0, 1, 2, 3, 4, 5])

    pages_allocated = 6
    num_pages = config.num_tpu_pages
    expected_remaining_pages = num_pages - pages_allocated
    self.assertEqual(
        pm.tpu_block.num_available_pages, expected_remaining_pages 
    )


  def test_allocate_twice(self):
    config = pm_lib.TpuCpuPageManagerConfig(
        max_tpu_bytes=130000,
        dp_size=1,
        page_size=1,
        max_seq_len=10,
        max_num_seqs=5,
        dtype=jnp.int32, 
    )
    pm = config.init()
    pm, first_allocated_idxs = pm.allocate(jnp.array([0, 1, 2, 0, 5]))
    pm, second_allocated_idxs = pm.allocate(jnp.array([1, 1, 0, 0, 2]))

    expected_seq_lens = jnp.array([1, 2, 2, 0, 7])
    np.testing.assert_array_equal(pm.seq_lens, expected_seq_lens)

    self.assertEqual(pm.page_indices[1][0], 0)
    np.testing.assert_array_equal(pm.page_indices[2][:2], [1, 2])
    np.testing.assert_array_equal(pm.page_indices[4][:5], [3, 4, 5, 6, 7])

    self.assertEqual(pm.page_indices[0][0], 8)
    self.assertEqual(pm.page_indices[1][1], 9)
    np.testing.assert_array_equal(pm.page_indices[4][5:7], [10, 11])

    np.testing.assert_array_equal(first_allocated_idxs[:8], [0, 1, 2, 3, 4, 5, 6, 7])
    np.testing.assert_array_equal(second_allocated_idxs[:4], [8, 9, 10, 11])


  def test_release(self):
    config = pm_lib.TpuCpuPageManagerConfig(
        max_bytes=130000,
        dp_size=1,
        page_size=1,
        max_seq_len=10,
        max_num_seqs=5,
        block_specs=[
            pm_lib.BlockSpec(name="block", dtype=jnp.int32),
        ]
    )
    pm = config.init()
    pm = pm.allocate(jnp.array([0, 1, 2, 0, 5]))

    initial_avail = pm.num_available_pages
    should_release = jnp.array([False, True, False, False, True])
    pm = pm.release(should_release)

    self.assertEqual(pm.num_available_pages, initial_avail + 6)
    np.testing.assert_array_equal(pm.seq_lens, jnp.array([0, 0, 2, 0, 0]))
    np.testing.assert_array_equal(
        pm.available_page_indices[initial_avail:pm.num_available_pages],
        jnp.array([0, 3, 4, 5, 6, 7])
    )

  def test_release_for_window(self):
    config = pm_lib.TpuCpuPageManagerConfig(
        max_bytes=130000,
        dp_size=1,
        page_size=1,
        max_seq_len=10,
        max_num_seqs=5,
        block_specs=[
            pm_lib.BlockSpec(name="block", dtype=jnp.int32),
        ]
    )
    pm = config.init()
    pm = dataclasses.replace(pm, window_size=3)

    pm = pm.allocate(jnp.array([0, 1, 2, 0, 3]))
    avail_after_first = pm.num_available_pages
    pm = pm.release_for_window()
    self.assertEqual(pm.num_available_pages, avail_after_first)

    pm = pm.allocate(jnp.array([0, 0, 0, 0, 2]))

    avail_before_release = pm.num_available_pages
    pm = pm.release_for_window()

    self.assertEqual(pm.num_available_pages, avail_before_release + 2)
    expected_seq_lens = jnp.array([0, 1, 2, 0, 3])
    np.testing.assert_array_equal(pm.seq_lens, expected_seq_lens)
    np.testing.assert_array_equal(
        pm.available_page_indices[avail_before_release:pm.num_available_pages],
        jnp.array([3, 4])
    )

  def test_load_values_no_subshape(self):
    config = pm_lib.TpuCpuPageManagerConfig(
        max_bytes=130000,
        dp_size=1,
        page_size=1,
        max_seq_len=10,
        max_num_seqs=5,
        block_specs=[
            pm_lib.BlockSpec(name="tokens", dtype=jnp.int32),
        ]
    )
    pm = config.init()
    q_lens = jnp.array([0, 1, 2, 0, 5])
    pm = pm.allocate(q_lens)
    
    tokens = jnp.arange(10, 18, dtype=jnp.int32)
    pm = pm.load_values(tokens, q_lens, block_id="tokens")

    out_tokens = pm.to_array(8, block_id="tokens")
    np.testing.assert_array_equal(out_tokens, tokens)

  def test_load_values_with_subshape(self):
    config = pm_lib.TpuCpuPageManagerConfig(
        max_bytes=130000,
        dp_size=1,
        page_size=1,
        max_seq_len=10,
        max_num_seqs=5,
        block_specs=[
            pm_lib.BlockSpec(name="block", dtype=jnp.float32, subshape=(2, 4)),
        ]
    )
    pm = config.init()
    q_lens = jnp.array([0, 1, 2, 0, 5])
    pm = pm.allocate(q_lens)

    values = jnp.arange(8 * 2 * 4, dtype=jnp.float32).reshape((8, 2, 4))
    pm = pm.load_values(values, q_lens, block_id="block")

    out_values = pm.to_array(8, block_id="block")
    np.testing.assert_array_equal(out_values, values)

  def test_insert_values_no_subshape(self):
    config = pm_lib.TpuCpuPageManagerConfig(
        max_bytes=130000,
        dp_size=1,
        page_size=2,
        max_seq_len=10,
        max_num_seqs=3,
        block_specs=[
            pm_lib.BlockSpec(name="block", dtype=jnp.int32),
        ]
    )
    pm = config.init()
    
    pm = pm.allocate(jnp.array([1, 2, 0]))
    
    init_values = jnp.array([42, 43, 44])
    pm = pm.load_values(init_values, jnp.array([1, 2, 0]), block_id="block")

    pm = pm.allocate(jnp.array([1, 1, 1]))
        
    new_values = jnp.array([99, 100, 101], dtype=jnp.int32)
    valid_mask = jnp.array([True, True, True])
    pm = pm.insert_values(new_values, valid_mask=valid_mask, block_id="block")
        
    out_values = pm.to_array(6, block_id="block")

    # expected flat array: [42, 99, 43, 44, 100, 101]
    # seq0: 42, 99
    # seq1: 43, 44, 100
    # seq2: 101
    np.testing.assert_array_equal(out_values, jnp.array([42, 99, 43, 44, 100, 101], dtype=jnp.int32))

  def test_insert_values_with_subshape(self):
    config = pm_lib.TpuCpuPageManagerConfig(
        max_bytes=130000,
        dp_size=1,
        page_size=2,
        max_seq_len=10,
        max_num_seqs=3,
        block_specs=[
            pm_lib.BlockSpec(name="block", dtype=jnp.float32, subshape=(2, 4)),
        ]
    )
    pm = config.init()

    pm = pm.allocate(jnp.array([1, 2, 0]))

    init_values = jnp.arange(3 * 2 * 4, dtype=jnp.float32).reshape((3, 2, 4))
    pm = pm.load_values(init_values, jnp.array([1, 2, 0]), block_id="block")

    pm = pm.allocate(jnp.array([1, 1, 1]))

    new_values = jnp.arange(3 * 2 * 4, dtype=jnp.float32).reshape((3, 2, 4)) + 100.0
    pm = pm.insert_values(new_values, block_id="block")

    out_values = pm.to_array(6, block_id="block")

    # Construct expected
    # seq0: init[0], new[0]
    expected_seq0 = jnp.stack([init_values[0], new_values[0]])
    # seq1: init[1], init[2], new[1]
    expected_seq1 = jnp.stack([init_values[1], init_values[2], new_values[1]])
    # seq2: new[2]
    expected_seq2 = jnp.stack([new_values[2]])

    expected = jnp.concatenate([expected_seq0, expected_seq1, expected_seq2], axis=0)

    np.testing.assert_array_equal(out_values, expected)

  def test_batch_copy_pages_only_one_block(self):
    config = pm_lib.TpuCpuPageManagerConfig(
        max_bytes=130000,
        dp_size=1,
        page_size=1,
        max_seq_len=10,
        max_num_seqs=5,
        block_specs=[
            pm_lib.BlockSpec(name="block1", dtype=jnp.int32),
            pm_lib.BlockSpec(name="block2", dtype=jnp.float32, subshape=(1, 1)),
        ]
    )
    hbm_pm = config.init()
    cpu_device = jax.devices('cpu')[0] if jax.devices('cpu') else None
    cpu_config = dataclasses.replace(
        config,
        dp_axis=None,
        tp_axis=None,
        dp_size=1,
        tp_size=1,
        device=cpu_device
    )
    cpu_pm = cpu_config.init()

    q_lens = jnp.array([2, 3, 0, 0, 0])
    hbm_pm = hbm_pm.allocate(q_lens)
    tokens = jnp.array([42, 43, 44, 45, 46], dtype=jnp.int32)
    hbm_pm = hbm_pm.load_values(tokens, q_lens, block_id="block1")

    cpu_pm = cpu_pm.allocate(q_lens)

    cpu_pm = pm_lib.batch_copy_pages(
        src_cache=hbm_pm,
        dst_cache=cpu_pm,
        src_slots=[0, 1],
        dst_slots=[0, 1],
        block_ids=["block1"],
    )

    copied_tokens = cpu_pm.to_array(5, block_id="block1")
    np.testing.assert_array_equal(copied_tokens, tokens)

  def test_batch_copy_pages_with_kv(self):
    config = pm_lib.TpuCpuPageManagerConfig(
        max_bytes=130000,
        dp_size=1,
        page_size=1,
        max_seq_len=10,
        max_num_seqs=5,
        block_specs=[
            pm_lib.BlockSpec(name="tokens", dtype=jnp.int32),
            pm_lib.BlockSpec(name="layer_0", dtype=jnp.float32, subshape=(1, 1)),
            pm_lib.BlockSpec(name="layer_1", dtype=jnp.float32, subshape=(1, 1)),
        ]
    )
    hbm_pm = config.init()
    cpu_device = jax.devices('cpu')[0] if jax.devices('cpu') else None
    cpu_config = dataclasses.replace(
        config,
        dp_axis=None,
        tp_axis=None,
        dp_size=1,
        tp_size=1,
        device=cpu_device
    )
    cpu_pm = cpu_config.init()

    q_lens = jnp.array([2])
    hbm_pm = hbm_pm.allocate(q_lens)

    # Load random values into KV cache
    new_pages = dict(hbm_pm.pages)
    rng = jax.random.PRNGKey(0)
    for i in range(2):
      layer_name = f'layer_{i}'
      new_pages[layer_name] = jax.random.normal(
          rng, hbm_pm.pages[layer_name].shape
      )
    hbm_pm = dataclasses.replace(hbm_pm, pages=new_pages)

    cpu_pm = cpu_pm.allocate(q_lens)
    cpu_pm = pm_lib.batch_copy_pages(
        src_cache=hbm_pm,
        dst_cache=cpu_pm,
        src_slots=[0],
        dst_slots=[0],
        block_ids=["tokens", "layer_0", "layer_1"],
    )

    # Verify KV values were transferred
    src_idxs = hbm_pm.page_indices[0][:2]
    dst_idxs = cpu_pm.page_indices[0][:2]
    for i in range(2):
      layer_name = f'layer_{i}'
      np.testing.assert_array_equal(
          cpu_pm.pages[layer_name][dst_idxs],
          hbm_pm.pages[layer_name][src_idxs]
      )

  def test_batch_copy_pages_with_kv_hbm_to_cpu_dp(self):
    config = pm_lib.TpuCpuPageManagerConfig(
        max_bytes=132000,
        dp_size=4,
        dp_axis='fsdp',
        page_size=1,
        max_seq_len=10,
        max_num_seqs=5,
        block_specs=[
            pm_lib.BlockSpec(name="tokens", dtype=jnp.int32),
            pm_lib.BlockSpec(name="layer_0", dtype=jnp.float32, subshape=(1, 1)),
        ]
    )
    # Just verify they execute without crashing due to DP axis specs
    hbm_pm = config.init()
    cpu_device = jax.devices('cpu')[0] if jax.devices('cpu') else None
    cpu_config = dataclasses.replace(
        config,
        dp_axis=None,
        tp_axis=None,
        dp_size=1,
        tp_size=1,
        device=cpu_device
    )
    cpu_pm = cpu_config.init()

    q_lens = jnp.array([2])
    hbm_pm = hbm_pm.allocate(q_lens)
    cpu_pm = cpu_pm.allocate(q_lens)

    cpu_pm = pm_lib.batch_copy_pages(
        src_cache=hbm_pm,
        dst_cache=cpu_pm,
        src_slots=[0],
        dst_slots=[0],
        block_ids=["tokens", "layer_0"],
    )
    self.assertIsNotNone(cpu_pm)

  def test_batch_copy_pages_with_kv_cpu_to_hbm_dp(self):
    config = pm_lib.TpuCpuPageManagerConfig(
        max_bytes=132000,
        dp_size=4,
        dp_axis='fsdp',
        page_size=1,
        max_seq_len=10,
        max_num_seqs=5,
        block_specs=[
            pm_lib.BlockSpec(name="tokens", dtype=jnp.int32),
            pm_lib.BlockSpec(name="layer_0", dtype=jnp.float32, subshape=(1, 1)),
        ]
    )
    hbm_pm = config.init()
    cpu_device = jax.devices('cpu')[0] if jax.devices('cpu') else None
    cpu_config = dataclasses.replace(
        config,
        dp_axis=None,
        tp_axis=None,
        dp_size=1,
        tp_size=1,
        device=cpu_device
    )
    cpu_pm = cpu_config.init()

    q_lens = jnp.array([2])
    cpu_pm = cpu_pm.allocate(q_lens)
    hbm_pm = hbm_pm.allocate(q_lens)

    hbm_pm = pm_lib.batch_copy_pages(
        src_cache=cpu_pm,
        dst_cache=hbm_pm,
        src_slots=[0],
        dst_slots=[0],
        block_ids=["tokens", "layer_0"],
    )
    self.assertIsNotNone(hbm_pm)

  def test_batch_copy_pages_with_kv_cpu_to_hbm_no_dp(self):
    config = pm_lib.TpuCpuPageManagerConfig(
        max_bytes=132000,
        dp_size=1,
        page_size=1,
        max_seq_len=10,
        max_num_seqs=5,
        block_specs=[
            pm_lib.BlockSpec(name="tokens", dtype=jnp.int32),
            pm_lib.BlockSpec(name="layer_0", dtype=jnp.float32, subshape=(1, 1)),
            pm_lib.BlockSpec(name="layer_1", dtype=jnp.float32, subshape=(1, 1)),
        ]
    )
    hbm_pm = config.init()
    cpu_device = jax.devices('cpu')[0] if jax.devices('cpu') else None
    cpu_config = dataclasses.replace(
        config,
        dp_axis=None,
        tp_axis=None,
        dp_size=1,
        tp_size=1,
        device=cpu_device
    )
    cpu_pm = cpu_config.init()

    q_lens = jnp.array([2])
    cpu_pm = cpu_pm.allocate(q_lens)
    hbm_pm = hbm_pm.allocate(q_lens)

    # Load random values into KV cache on CPU
    new_pages = dict(cpu_pm.pages)
    rng = jax.random.PRNGKey(0)
    for i in range(2):
      layer_name = f'layer_{i}'
      new_pages[layer_name] = jax.random.normal(
          rng, cpu_pm.pages[layer_name].shape
      )
    cpu_pm = dataclasses.replace(cpu_pm, pages=new_pages)

    hbm_pm = pm_lib.batch_copy_pages(
        src_cache=cpu_pm,
        dst_cache=hbm_pm,
        src_slots=[0],
        dst_slots=[0],
        block_ids=["tokens", "layer_0", "layer_1"],
    )
    self.assertIsNotNone(hbm_pm)

    # Verify KV values were transferred
    src_idxs = cpu_pm.page_indices[0][:2]
    dst_idxs = hbm_pm.page_indices[0][:2]
    for i in range(2):
      layer_name = f'layer_{i}'
      np.testing.assert_array_equal(
          hbm_pm.pages[layer_name][dst_idxs],
          cpu_pm.pages[layer_name][src_idxs]
      )

  def test_remove_dp_spec(self):
    spec = jax.sharding.PartitionSpec('dp', 'tp', 'fsdp', None)
    new_spec = pm_lib._remove_dp_spec(spec)
    self.assertEqual(
        new_spec, jax.sharding.PartitionSpec(None, 'tp', None, None)
    )

    spec2 = jax.sharding.PartitionSpec('tp', 'sp')
    self.assertEqual(
        pm_lib._remove_dp_spec(spec2), jax.sharding.PartitionSpec('tp', 'sp')
    )

  def test_put_on_target_device_single_device(self):
    cpu_device = jax.devices('cpu')[0]

    target_tensor = jax.device_put(jnp.zeros((8, 1)), cpu_device)
    src_tensor = jax.device_put(jnp.ones((8, 1)), cpu_device)

    out_tensor = pm_lib._put_on_target_device(src_tensor, target_tensor)
    if hasattr(out_tensor, 'devices'):
      self.assertEqual(list(out_tensor.devices())[0], cpu_device)

  def test_put_on_target_device_named_sharding(self):
    devices = jax.devices()

    mesh = jax.sharding.Mesh(
        np.array(devices).reshape((len(devices), 1)), ('dp', 'tp')
    )

    spec = jax.sharding.PartitionSpec('dp', 'tp')
    sharding = jax.sharding.NamedSharding(mesh, spec)
    target_tensor = jax.lax.with_sharding_constraint(
        jnp.zeros((len(devices), 2)),
        sharding
    )

    src_tensor = jnp.ones((len(devices), 2))
    out_tensor = pm_lib._put_on_target_device(src_tensor, target_tensor)

    self.assertEqual(
        out_tensor.sharding.spec,  # pytype: disable=attribute-error
        jax.sharding.PartitionSpec(None, 'tp')
    )

if __name__ == '__main__':
  absltest.main()
