import dataclasses
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from tunix.generate import page_manager as pm_lib


class PageManagerTest(parameterized.TestCase):
  def test_ragged_array_row_idxs(self):
    lens = jnp.array([1, 5, 0, 1, 2])
    data = jnp.zeros((9,))
    ragged = pm_lib.RaggedArray(data=data, lens=lens)
    expected = jnp.array([0, 1, 1, 1, 1, 1, 3, 4, 4])
    np.testing.assert_array_equal(ragged.row_idxs, expected)

  def test_ragged_array_intra_offsets(self):
    lens = jnp.array([1, 5, 0, 1, 2])
    data = jnp.zeros((9,))
    ragged = pm_lib.RaggedArray(data=data, lens=lens)
    expected = jnp.array([0, 0, 1, 2, 3, 4, 0, 0, 1])
    np.testing.assert_array_equal(ragged.intra_offsets, expected)

  def test_cache_config_num_pages(self):
    config = pm_lib.PageManagerConfig(
        max_bytes=1300,
        dtype=jnp.float32,
        dp_size=4,
        page_size=10,
        num_kv_heads=1,
        head_dim=1,
        num_layers=1,
        max_seq_len=100,
        max_num_seqs=5,
    )

    # This test verifies that the number of pages per layer block
    # is properly aligned with dp_size.

    # Kv_page_size_bytes: 2 * 10 * 1 * 1 * 1 * 4 = 80
    # Token_page_size_bytes = 40
    # Bytes_per_global_page = 80 + 40 = 120
    # max_pages_per_layer_block = 1300 // 120 = 10
    # pages_per_layer_block(10 // 4) * 4 = 8

    self.assertEqual(config.num_pages_per_layer_block, 8)

  def test_allocate(self):
    config = pm_lib.PageManagerConfig(
        max_bytes=130000,
        dtype=jnp.float32,
        dp_size=1,
        page_size=1,
        num_kv_heads=1,
        head_dim=1,
        num_layers=1,
        max_seq_len=10,
        max_num_seqs=5,
    )
    pm = config.init()
    q_lens = jnp.array([0, 1, 2, 0, 5])
    pm = pm.allocate(q_lens)

    np.testing.assert_array_equal(pm.seq_lens, q_lens)
    self.assertEqual(pm.page_indices[1][0], 0)
    self.assertEqual(pm.page_indices[2][:2], [1, 2])
    self.assertEqual(pm.page_indices[4][:5], [3, 4, 5, 6, 7])
    self.assertEqual(
        pm.num_available_pages, int(config.num_pages_per_layer_block) - 8
    )

  def test_allocate_twice(self):
    config = pm_lib.PageManagerConfig(
        max_bytes=130000,
        dtype=jnp.float32,
        dp_size=1,
        page_size=1,
        num_kv_heads=1,
        head_dim=1,
        num_layers=1,
        max_seq_len=10,
        max_num_seqs=5,
    )
    pm = config.init()
    pm = pm.allocate(jnp.array([0, 1, 2, 0, 5]))
    pm = pm.allocate(jnp.array([1, 1, 0, 0, 2]))

    expected_seq_lens = jnp.array([1, 2, 2, 0, 7])
    np.testing.assert_array_equal(pm.seq_lens, expected_seq_lens)

    self.assertEqual(pm.page_indices[1][0], 0)
    np.testing.assert_array_equal(pm.page_indices[2][:2], [1, 2])
    np.testing.assert_array_equal(pm.page_indices[4][:5], [3, 4, 5, 6, 7])

    self.assertEqual(pm.page_indices[0][0], 8)
    self.assertEqual(pm.page_indices[1][1], 9)
    np.testing.assert_array_equal(pm.page_indices[4][5:7], [10, 11])

  def test_release(self):
    config = pm_lib.PageManagerConfig(
        max_bytes=130000,
        dtype=jnp.float32,
        dp_size=1,
        page_size=1,
        num_kv_heads=1,
        head_dim=1,
        num_layers=1,
        max_seq_len=10,
        max_num_seqs=5,
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
    config = pm_lib.PageManagerConfig(
        max_bytes=130000,
        dtype=jnp.float32,
        dp_size=1,
        page_size=1,
        num_kv_heads=1,
        head_dim=1,
        num_layers=1,
        max_seq_len=10,
        max_num_seqs=5,
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

  def test_load_prompt_tokens(self):
    config = pm_lib.PageManagerConfig(
        max_bytes=130000,
        dtype=jnp.float32,
        dp_size=1,
        page_size=1,
        num_kv_heads=1,
        head_dim=1,
        num_layers=1,
        max_seq_len=10,
        max_num_seqs=5,
    )
    pm = config.init()
    q_lens = jnp.array([0, 1, 2, 0, 5])
    pm = pm.allocate(q_lens)

    tokens = jnp.arange(10, 18, dtype=jnp.int32)
    pm = pm.load_prompt_tokens(tokens, q_lens)

    out_tokens = pm.to_array(8)
    np.testing.assert_array_equal(out_tokens, tokens)

  def test_batch_copy_pages_no_kv(self):
    config = pm_lib.PageManagerConfig(
        max_bytes=130000,
        dtype=jnp.float32,
        dp_size=1,
        page_size=1,
        num_kv_heads=1,
        head_dim=1,
        num_layers=1,
        max_seq_len=10,
        max_num_seqs=5,
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
    hbm_pm = hbm_pm.load_prompt_tokens(tokens, q_lens)

    cpu_pm = cpu_pm.allocate(q_lens)

    cpu_pm = pm_lib.batch_copy_pages(
        src_cache=hbm_pm,
        dst_cache=cpu_pm,
        src_slots=[0, 1],
        dst_slots=[0, 1],
        transfer_kv=False,
    )

    copied_tokens = cpu_pm.to_array(5)
    np.testing.assert_array_equal(copied_tokens, tokens)

  def test_batch_copy_pages_with_kv(self):
    config = pm_lib.PageManagerConfig(
        max_bytes=130000,
        dtype=jnp.float32,
        dp_size=1,
        page_size=1,
        num_kv_heads=1,
        head_dim=1,
        num_layers=2,
        max_seq_len=10,
        max_num_seqs=5,
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
        transfer_kv=True,
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
    config = pm_lib.PageManagerConfig(
        max_bytes=132000,
        dtype=jnp.float32,
        dp_size=4,
        dp_axis='fsdp',
        page_size=1,
        num_kv_heads=1,
        head_dim=1,
        num_layers=1,
        max_seq_len=10,
        max_num_seqs=5,
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
        transfer_kv=True,
    )
    self.assertIsNotNone(cpu_pm)

  def test_batch_copy_pages_with_kv_cpu_to_hbm_dp(self):
    config = pm_lib.PageManagerConfig(
        max_bytes=132000,
        dtype=jnp.float32,
        dp_size=4,
        dp_axis='fsdp',
        page_size=1,
        num_kv_heads=1,
        head_dim=1,
        num_layers=1,
        max_seq_len=10,
        max_num_seqs=5,
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
        transfer_kv=True,
    )
    self.assertIsNotNone(hbm_pm)

  def test_batch_copy_pages_with_kv_cpu_to_hbm_no_dp(self):
    config = pm_lib.PageManagerConfig(
        max_bytes=132000,
        dtype=jnp.float32,
        dp_size=1,
        page_size=1,
        num_kv_heads=1,
        head_dim=1,
        num_layers=2,
        max_seq_len=10,
        max_num_seqs=5,
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
        transfer_kv=True,
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
