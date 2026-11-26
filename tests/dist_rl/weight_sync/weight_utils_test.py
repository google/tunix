# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for transport-neutral weight_utils helpers."""

from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.dist_rl.weight_sync import weight_utils


class _DummyRunner:

  def __init__(self, state, *, state_leaves=None):
    self.state = state
    if state_leaves is not None:
      self.state_leaves = state_leaves
    elif isinstance(state, dict):
      self.state_leaves = state
    else:
      self.state_leaves = tuple(jax.tree_util.tree_leaves(state))


class _DummySampler:

  def __init__(self, *, model_runner=None, transformer_state=None):
    self.model_runner = model_runner
    self.transformer_state = transformer_state


class WeightUtilsTest(absltest.TestCase):

  def test_param_key_normalizes_brackets_roots_and_folded_indices(self):
    self.assertEqual(
        weight_utils.param_key(
            "['base']['layers_0']['mlp']['down_proj']['weight'].value"
        ),
        "layers.0.mlp.down_proj.weight",
    )
    self.assertEqual(
        weight_utils.param_key("model.layers[2].self_attn.q_proj.weight"),
        "layers.2.self_attn.q_proj.weight",
    )
    self.assertEqual(
        weight_utils.param_key("base.model.layers.0.w.value"),
        "layers.0.w",
    )
    self.assertEqual(weight_utils.param_key("base"), "")

  def test_is_inference_only_buffer_matches_exact_segments(self):
    cases = (
        ("['base']['decoder']['layers']['attention']['cache']['k']", True),
        ("['layers']['cached_prefill_key']", True),
        ("['layers']['cached_prefill_value']", True),
        (
            "['vllm_model.model.layers.3.self_attn.rotary_emb.cos_sin_cache']",
            True,
        ),
        ("['vllm_model.model.layers.0.hash_indices_table']", True),
        (
            "['vllm_model.model.layers.0.routed_experts"
            + ".tpu_hash_indices_table']",
            True,
        ),
        ("['vllm_model.model.embed_tokens.weight']", False),
        ("['vllm_model.model.layers.0.cache_proj.weight']", False),
    )
    for name, expected in cases:
      with self.subTest(name=name):
        self.assertEqual(weight_utils.is_inference_only_buffer(name), expected)
        self.assertEqual(weight_utils.is_kv_cache(name), expected)

  def test_filter_bindable_drops_inference_only_buffers_and_non_floating(self):
    names = [
        "layers.0.attention.cache.cached_prefill_key",
        "['vllm_model.model.layers.3.self_attn.rotary_emb.cos_sin_cache']",
        "['vllm_model.model.layers.0.hash_indices_table']",
        "['vllm_model.model.layers.0.routed_experts.tpu_hash_indices_table']",
        "['vllm_model.model.layers.0.cache_proj.weight']",
        "layers.0.weight",
        "step_counter",
    ]
    arrays = [
        jnp.ones((2, 4), dtype=jnp.float32),
        jnp.ones((8, 4), dtype=jnp.float32),
        jnp.ones((4,), dtype=jnp.float32),
        jnp.ones((4,), dtype=jnp.float32),
        jnp.ones((2, 4), dtype=jnp.bfloat16),
        jnp.ones((2, 4), dtype=jnp.bfloat16),
        jnp.array([1], dtype=jnp.int32),
    ]
    kept_names, kept_arrays = weight_utils.filter_bindable(names, arrays)
    self.assertEqual(
        kept_names,
        [
            "['vllm_model.model.layers.0.cache_proj.weight']",
            "layers.0.weight",
        ],
    )
    self.assertLen(kept_arrays, 2)

  def test_filter_bindable_strict_drops_only_non_weight_leaves(self):
    names = ["w", "rng_count", "rng_key_data", "scalar_scale"]
    arrays = [
        jnp.ones((2, 4), dtype=jnp.float32),
        jnp.array(3, dtype=jnp.uint32),
        jnp.ones((2,), dtype=jnp.uint32),
        jnp.array(1.0, dtype=jnp.float32),
    ]
    kept_names, _ = weight_utils.filter_bindable(names, arrays, strict=True)
    self.assertEqual(kept_names, ["w"])

  def test_filter_bindable_strict_fails_on_unbindable_weight(self):
    names = ["w", "gpu_weight"]
    arrays = [jnp.ones((2, 4)), jnp.ones((2, 4))]
    real_is_bindable = weight_utils.is_bindable
    with mock.patch.object(
        weight_utils,
        "is_bindable",
        side_effect=lambda arr, **kw: arr is arrays[0]
        and real_is_bindable(arr, **kw),
    ):
      self.assertEqual(
          weight_utils.filter_bindable(names, arrays)[0], ["w"]
      )
      with self.assertRaisesRegex(ValueError, "gpu_weight"):
        weight_utils.filter_bindable(names, arrays, strict=True)

  def test_filter_bindable_warns_inference_only_drops_once_per_set(self):
    arrays = [jnp.ones((2,)), jnp.ones((2,))]
    label = f"test-{self.id()}"
    with mock.patch.object(weight_utils.logging, "warning") as warning:
      for _ in range(3):
        weight_utils.filter_bindable(
            ["w", "layers.0.attention.cache.k"], arrays, label=label
        )
      weight_utils.filter_bindable(
          ["w", "layers.1.attention.cache.k"], arrays, label=label
      )
    logged = [call.args[-1] for call in warning.call_args_list]
    self.assertEqual(
        logged,
        [["layers.0.attention.cache.k"], ["layers.1.attention.cache.k"]],
    )

  def test_filter_bindable_batches_block_until_ready(self):
    a1 = jnp.ones((2, 4), dtype=jnp.bfloat16)
    a2 = jnp.zeros((2, 4), dtype=jnp.bfloat16)
    with mock.patch.object(
        weight_utils.jax,
        "block_until_ready",
        wraps=weight_utils.jax.block_until_ready,
    ) as spy_ready:
      kept_names, kept_arrays = weight_utils.filter_bindable(
          ["w1", "w2"], [a1, a2]
      )
    self.assertEqual(kept_names, ["w1", "w2"])
    self.assertEqual(kept_arrays, [a1, a2])
    spy_ready.assert_called_once_with([a1, a2])

  def test_filter_bindable_and_apply_to_runner_preserve_vllm_lookup_buffers(
      self,
  ):
    old_embed = jnp.zeros((4, 2), dtype=jnp.float32)
    old_lm_head = jnp.zeros((4, 2), dtype=jnp.float32)
    cos_sin_cache = jnp.full((8, 4), 3.0, dtype=jnp.float32)
    hash_table = jnp.ones((4,), dtype=jnp.int32)
    state = {
        "vllm_model.model.embed_tokens.weight": old_embed,
        "vllm_model.lm_head.weight": old_lm_head,
        "vllm_model.model.layers.3.self_attn.rotary_emb.cos_sin_cache": (
            cos_sin_cache
        ),
        "vllm_model.model.layers.0.hash_indices_table": hash_table,
    }
    runner = _DummyRunner(state)
    kept_names, _ = weight_utils.filter_bindable(
        *weight_utils.flatten_weights(runner.state)
    )
    self.assertEqual(
        kept_names,
        [
            "['vllm_model.lm_head.weight']",
            "['vllm_model.model.embed_tokens.weight']",
        ],
    )

    new_lm_head = jnp.full((4, 2), 13.0, dtype=jnp.float32)
    new_embed = jnp.full((4, 2), 11.0, dtype=jnp.float32)
    weight_utils.apply_to_runner(
        "rollout",
        kept_names,
        [new_lm_head, new_embed],
        runner,
        delete_replaced_buffers=True,
    )
    self.assertIs(runner.state, state)
    self.assertIs(runner.state_leaves, state)
    self.assertIs(runner.state["vllm_model.lm_head.weight"], new_lm_head)
    self.assertIs(
        runner.state["vllm_model.model.embed_tokens.weight"], new_embed
    )
    self.assertIs(
        runner.state[
            "vllm_model.model.layers.3.self_attn.rotary_emb.cos_sin_cache"
        ],
        cos_sin_cache,
    )
    self.assertFalse(cos_sin_cache.is_deleted())
    self.assertIs(
        runner.state["vllm_model.model.layers.0.hash_indices_table"], hash_table
    )
    self.assertFalse(hash_table.is_deleted())
    self.assertTrue(old_embed.is_deleted())
    self.assertTrue(old_lm_head.is_deleted())

  def test_apply_to_state_updates_nnx_variables_and_deletes_old_buffers(self):
    class _Model(nnx.Module):

      def __init__(self):
        self.w = nnx.Param(jnp.ones((2, 2), dtype=jnp.float32))

    model = _Model()
    state = nnx.state(model)
    old_buf = state["w"].get_value()
    new_buf = jnp.full((2, 2), 7.0, dtype=jnp.float32)

    weight_utils.apply_to_state(
        "rollout",
        ["['w'].value"],
        [new_buf],
        state,
        delete_replaced_buffers=True,
    )
    np.testing.assert_array_equal(np.asarray(state["w"].get_value()), 7.0)
    self.assertIs(model.w.get_value(), new_buf)
    self.assertTrue(old_buf.is_deleted())
    self.assertFalse(new_buf.is_deleted())

  def test_apply_to_runner_nnx_state_preserves_model_variables(self):
    class _Model(nnx.Module):

      def __init__(self):
        self.w = nnx.Param(jnp.ones((2, 3), dtype=jnp.float32))

    model = _Model()
    old_buf = model.w.get_value()
    runner = _DummyRunner(nnx.state(model))
    new_buf = jnp.full((2, 3), 9.0, dtype=jnp.float32)

    weight_utils.apply_to_runner(
        "rollout",
        ["['base']['w'].value"],
        [new_buf],
        runner,
        delete_replaced_buffers=True,
    )
    self.assertFalse(model.w.get_value().is_deleted())
    np.testing.assert_array_equal(np.asarray(model.w.get_value()), 9.0)
    self.assertEqual(runner.state_leaves, (new_buf,))
    self.assertTrue(old_buf.is_deleted())

  def test_apply_to_runner_flat_dict_handles_empty_or_copied_state_leaves(self):
    for leaves_factory in (
        lambda d: d,
        lambda d: (),
        dict,
    ):
      with self.subTest(leaves_factory=leaves_factory):
        old_w = jnp.ones((3, 2), dtype=jnp.float32)
        state_dict = {"model.embed.weight": old_w}
        runner = _DummyRunner(
            state_dict, state_leaves=leaves_factory(state_dict)
        )
        new_w = jnp.full((3, 2), 3.5, dtype=jnp.float32)
        sampler = _DummySampler(model_runner=runner)

        weight_utils.apply_to_sampler(
            "rollout",
            ["['base']['embed']['weight']"],
            [new_w],
            sampler,
            delete_replaced_buffers=True,
        )
        self.assertIs(runner.state, state_dict)
        self.assertIs(runner.state_leaves, state_dict)
        self.assertIs(state_dict["model.embed.weight"], new_w)
        self.assertTrue(old_w.is_deleted())
        self.assertFalse(new_w.is_deleted())

  def test_apply_to_runner_nested_dict_state_aliases_state_leaves(self):
    old_w1 = jnp.ones((2, 2), dtype=jnp.float32)
    old_w2 = jnp.ones((3,), dtype=jnp.float32)
    state = {"model": {"w1": old_w1, "w2": old_w2}}
    # Mirrors `TPUModelRunner.refresh_state_leaves` for non-`nnx.State` state.
    runner = _DummyRunner(state, state_leaves=state)
    new_w1 = jnp.full((2, 2), 4.0, dtype=jnp.float32)
    new_w2 = jnp.full((3,), 5.0, dtype=jnp.float32)

    weight_utils.apply_to_runner(
        "rollout",
        ["['base']['w1']", "['base']['w2']"],
        [new_w1, new_w2],
        runner,
        delete_replaced_buffers=True,
    )
    self.assertIs(runner.state["model"]["w1"], new_w1)
    self.assertIs(runner.state["model"]["w2"], new_w2)
    self.assertIs(runner.state_leaves, runner.state)
    self.assertTrue(old_w1.is_deleted())
    self.assertTrue(old_w2.is_deleted())

  def test_apply_to_state_validates_all_leaves_before_mutating(self):
    old_param = jnp.ones((2, 2), dtype=jnp.float32)
    raw_leaf = jnp.ones((3,), dtype=jnp.float32)
    state = {"a": nnx.Param(old_param), "b": raw_leaf}

    with self.assertRaisesRegex(TypeError, "expected nnx.Variable leaf"):
      weight_utils.apply_to_state(
          "rollout",
          ["a", "b"],
          [
              jnp.full((2, 2), 7.0, dtype=jnp.float32),
              jnp.full((3,), 7.0, dtype=jnp.float32),
          ],
          state,
          delete_replaced_buffers=True,
      )
    self.assertIs(state["a"].get_value(), old_param)
    self.assertFalse(old_param.is_deleted())

  def test_apply_to_runner_rejects_shape_and_dtype_mismatch_pre_mutation(self):
    old_w1 = jnp.ones((2, 2), dtype=jnp.bfloat16)
    old_w2 = jnp.ones((4, 4), dtype=jnp.bfloat16)
    state = nnx.State({"w1": nnx.Param(old_w1), "w2": nnx.Param(old_w2)})
    runner = _DummyRunner(state)

    with self.assertRaisesRegex(ValueError, "Shape mismatch"):
      weight_utils.apply_to_runner(
          "rollout",
          ["w1", "w2"],
          [
              jnp.full((2, 2), 2.0, dtype=jnp.bfloat16),
              jnp.full((4, 3), 2.0, dtype=jnp.bfloat16),
          ],
          runner,
          delete_replaced_buffers=True,
      )
    self.assertFalse(old_w1.is_deleted())
    self.assertFalse(old_w2.is_deleted())

    with self.assertRaisesRegex(ValueError, "Dtype mismatch"):
      weight_utils.apply_to_runner(
          "rollout",
          ["w1", "w2"],
          [
              jnp.full((2, 2), 2.0, dtype=jnp.bfloat16),
              jnp.full((4, 4), 2.0, dtype=jnp.float16),
          ],
          runner,
          delete_replaced_buffers=True,
      )
    self.assertFalse(old_w1.is_deleted())
    self.assertFalse(old_w2.is_deleted())

  def test_compute_and_verify_checksums_across_mixed_shardings(self):
    devs = jax.devices()
    mesh_a = jax.sharding.Mesh(np.array(devs[:1]), ("tp",))
    mesh_b = jax.sharding.Mesh(
        np.array(devs[1:2] if len(devs) >= 2 else devs[:1]), ("fsdp",)
    )
    arr_embed = jax.device_put(
        jnp.arange(8, dtype=jnp.float32).reshape(2, 4),
        jax.sharding.NamedSharding(mesh_a, jax.sharding.PartitionSpec("tp")),
    )
    arr_l0 = jax.device_put(
        jnp.ones((4,), dtype=jnp.float32) * 2.5,
        jax.sharding.NamedSharding(mesh_b, jax.sharding.PartitionSpec("fsdp")),
    )
    arr_l1 = jnp.ones((2,), dtype=jnp.float32) * 3.0
    arr_lm_head = jnp.ones((3,), dtype=jnp.float32) * 4.0

    # Different tree orders and raw vs canonical names pair up by key.
    src_sums = weight_utils.compute_checksums(
        [
            "['base']['embed']['weight'].value",
            "['base']['layers_0']['w'].value",
            "['base']['layers_1']['w'].value",
            "['lm_head']['weight'].value",
        ],
        [arr_embed, arr_l0, arr_l1, arr_lm_head],
        sample=None,
        canonical_keys=True,
    )
    dst_sums = weight_utils.compute_checksums(
        [
            "lm_head.weight",
            "model.layers[1].w",
            "model.layers[0].w",
            "model.embed.weight",
        ],
        [arr_lm_head, arr_l1, arr_l0, arr_embed],
        sample=None,
        canonical_keys=False,
    )
    weight_utils.verify_weight_checksums(src_sums, dst_sums)

    corrupted = dict(dst_sums)
    corrupted["model.layers[0].w"] = 999.0
    with self.assertRaisesRegex(RuntimeError, "tensor mismatch"):
      weight_utils.verify_weight_checksums(src_sums, corrupted)

  def test_verify_checksums_rejects_sampled_checksums(self):
    names = ["a", "b", "c", "d"]
    arrays = [np.full((2,), i, np.float32) for i in range(4)]
    full = weight_utils.compute_checksums(names, arrays, sample=None)
    sampled = weight_utils.compute_checksums(names, arrays, sample=3)
    with self.assertRaisesRegex(RuntimeError, "cover 3 of 4 tensors"):
      weight_utils.verify_weight_checksums(full, sampled)

  def test_position_weighted_checksum_catches_layout_errors(self):
    w = np.arange(12, dtype=np.float32).reshape(3, 4)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("tp",))
    sharded = jax.device_put(
        jnp.asarray(w),
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("tp")),
    )
    src = weight_utils.compute_checksums(["w"], [sharded], sample=None)
    # Device and host computations agree.
    weight_utils.verify_weight_checksums(
        src, weight_utils.compute_checksums(["w"], [w], sample=None)
    )
    for wrong in (w[::-1].copy(), w.reshape(4, 3).T.copy()):
      dst = weight_utils.compute_checksums(["w"], [wrong], sample=None)
      self.assertEqual(dst["w"], src["w"])  # The L1 sum alone is blind.
      with self.assertRaisesRegex(RuntimeError, r"w@pw"):
        weight_utils.verify_weight_checksums(src, dst)

  def test_position_weights_wrap_without_overflow(self):
    shape = (3, 2000)
    weights = np.asarray(weight_utils._position_weights(shape))
    expected = (np.arange(6000) % 1009 + 1).reshape(shape)
    np.testing.assert_array_equal(weights, expected)

  def test_compute_checksums_empty(self):
    self.assertEqual(
        weight_utils.compute_checksums([], []),
        {
            "__grand_total__": 0.0,
            "__tensor_count__": 0,
            "__element_count__": 0,
        },
    )

  def test_global_shard_indices_and_tensor_metadata_use_ffi_skips_addressable_shards(
      self,
  ):
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:4]), ("data",))
    sharding = jax.sharding.NamedSharding(
        mesh, jax.sharding.PartitionSpec("data")
    )
    tpu_shard = mock.Mock()
    tpu_shard.device.platform = "tpu"
    tpu_shard.index = (slice(0, 2), slice(None))
    local_arr = mock.Mock(
        shape=(8, 2),
        ndim=2,
        dtype=np.dtype(np.float32),
        sharding=sharding,
        addressable_shards=[tpu_shard],
    )
    self.assertEqual(
        weight_utils.global_shard_indices(local_arr, sharding, use_ffi=False),
        (0,),
    )

    class _NoDisassembleArray:
      shape = (8, 2)
      ndim = 2
      dtype = np.dtype(np.float32)

      def __init__(self, arr_sharding):
        self.sharding = arr_sharding

      @property
      def addressable_shards(self):
        raise AssertionError(
            "addressable_shards must not be accessed when use_ffi=True"
        )

    ffi_arr = _NoDisassembleArray(sharding)
    self.assertEqual(
        weight_utils.global_shard_indices(ffi_arr, sharding, use_ffi=True),
        (0, 1, 2, 3),
    )
    meta = weight_utils.tensor_metadata("w", ffi_arr, 0, use_ffi=True)
    self.assertEqual(meta.global_shard_indices, (0, 1, 2, 3))


if __name__ == "__main__":
  jax.config.update("jax_num_cpu_devices", 4)
  absltest.main()
