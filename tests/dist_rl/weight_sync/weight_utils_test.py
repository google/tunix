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

  def test_filter_bindable_drops_kv_cache_and_non_floating(self):
    names = [
        "layers.0.attention.cache.cached_prefill_key",
        "layers.0.weight",
        "step_counter",
    ]
    arrays = [
        jnp.ones((2, 4), dtype=jnp.float32),
        jnp.ones((2, 4), dtype=jnp.bfloat16),
        jnp.array([1], dtype=jnp.int32),
    ]
    kept_names, kept_arrays = weight_utils.filter_bindable(names, arrays)
    self.assertEqual(kept_names, ["layers.0.weight"])
    self.assertLen(kept_arrays, 1)

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

  def test_compute_and_verify_checksums_across_mixed_shardings_and_sample(self):
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

    # Different tree orders ('base' < 'lm_head' vs 'lm_head' < 'model') must
    # select the identical canonical 3-tensor sample under default sample=3.
    src_sums = weight_utils.compute_checksums(
        [
            "['base']['embed']['weight'].value",
            "['base']['layers_0']['w'].value",
            "['base']['layers_1']['w'].value",
            "['lm_head']['weight'].value",
        ],
        [arr_embed, arr_l0, arr_l1, arr_lm_head],
        sample=3,
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
        sample=3,
        canonical_keys=False,
    )
    weight_utils.verify_weight_checksums(src_sums, dst_sums)

    corrupted = dict(dst_sums)
    corrupted["model.layers[0].w"] = 999.0
    with self.assertRaisesRegex(RuntimeError, "tensor mismatch"):
      weight_utils.verify_weight_checksums(src_sums, corrupted)

  def test_compute_checksums_empty(self):
    self.assertEqual(
        weight_utils.compute_checksums([], []),
        {
            "__grand_total__": 0.0,
            "__tensor_count__": 0,
            "__element_count__": 0,
        },
    )


if __name__ == "__main__":
  jax.config.update("jax_num_cpu_devices", 4)
  absltest.main()
