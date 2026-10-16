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

from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.experimental.generate import model_runner as model_runner_lib
from tunix.rl import reshard
from tunix.tests import test_common as tc


def _transformer(seed: int = 0, **config_kwargs) -> tc.ToyTransformer:
  return tc.ToyTransformer(
      config=tc.ModelConfig(**config_kwargs), rngs=nnx.Rngs(seed)
  )


class ModelRunnerTest(absltest.TestCase):

  def _assert_trees_equal(self, x, y):
    jax.tree.map(np.testing.assert_array_equal, x, y)

  def test_transformer_merges_the_state(self):
    transformer = _transformer()
    runner = model_runner_lib.ModelRunner(transformer)

    graphdef, state = runner.model_def_and_state()

    self.assertEqual(graphdef, nnx.graphdef(transformer))
    self._assert_trees_equal(
        nnx.state(nnx.merge(graphdef, state)), nnx.state(transformer)
    )
    self._assert_trees_equal(
        nnx.state(runner.transformer), nnx.state(transformer)
    )

  def test_dtype_comes_from_the_model_config(self):
    runner = model_runner_lib.ModelRunner(_transformer(dtype=jnp.bfloat16))

    self.assertEqual(runner.dtype, jnp.bfloat16)

  def test_dtype_falls_back_to_the_weights(self):
    class NoConfig(nnx.Module):

      def __init__(self):
        self.w = nnx.Param(jnp.ones((1,), dtype=jnp.bfloat16))

    runner = model_runner_lib.ModelRunner(NoConfig())

    self.assertEqual(runner.dtype, jnp.bfloat16)

  def test_set_transformer_state_replaces_the_full_state(self):
    runner = model_runner_lib.ModelRunner(_transformer(seed=0))
    new_state = nnx.variables(_transformer(seed=1), nnx.Param)

    runner.transformer_state = new_state

    self._assert_trees_equal(runner.transformer_state, new_state)
    self._assert_trees_equal(nnx.state(runner.transformer), new_state)

  def test_set_transformer_state_rejects_a_different_structure(self):
    runner = model_runner_lib.ModelRunner(_transformer(num_layers=4))

    with self.assertRaisesRegex(ValueError, 'same structure'):
      runner.transformer_state = nnx.variables(
          _transformer(num_layers=6), nnx.Param
      )

  def test_set_transformer_state_rejects_a_different_dtype(self):
    runner = model_runner_lib.ModelRunner(_transformer())

    with self.assertRaisesRegex(ValueError, 'same shape, dtype and sharding'):
      runner.transformer_state = nnx.variables(
          _transformer(dtype=jnp.bfloat16), nnx.Param
      )

  def test_set_transformer_state_rejects_a_different_sharding_type(self):
    runner = model_runner_lib.ModelRunner(_transformer())
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ('x',))
    new_state = jax.device_put(
        nnx.variables(_transformer(seed=1), nnx.Param),
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
    )

    with self.assertRaisesRegex(ValueError, 'same shape, dtype and sharding'):
      runner.transformer_state = new_state

  def test_set_transformer_state_compares_named_shardings_by_mesh_axes(self):
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ('x',))
    other_mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ('y',))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    transformer = _transformer()
    nnx.update(transformer, jax.device_put(nnx.state(transformer), replicated))
    runner = model_runner_lib.ModelRunner(transformer)
    new_state = nnx.variables(_transformer(seed=1), nnx.Param)

    # Sharding over an axis of size 1 is the same as replicating.
    runner.transformer_state = jax.device_put(
        new_state,
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('x')),
    )
    with self.assertRaisesRegex(ValueError, 'same shape, dtype and sharding'):
      runner.transformer_state = jax.device_put(
          new_state,
          jax.sharding.NamedSharding(other_mesh, jax.sharding.PartitionSpec()),
      )

  def test_set_transformer_state_replaces_only_the_lora_params(self):
    runner = model_runner_lib.ModelRunner(tc.get_lora_model(_transformer()))
    base_state = nnx.state(runner.transformer, nnx.Not(nnx.LoRAParam))
    new_lora_state = jax.tree.map(
        lambda x: x + 1, nnx.state(runner.transformer, nnx.LoRAParam)
    )

    runner.transformer_state = new_lora_state

    self._assert_trees_equal(
        nnx.state(runner.transformer, nnx.LoRAParam), new_lora_state
    )
    self._assert_trees_equal(
        nnx.state(runner.transformer, nnx.Not(nnx.LoRAParam)), base_state
    )

  def test_set_transformer_state_rejects_a_different_lora_structure(self):
    runner = model_runner_lib.ModelRunner(tc.get_lora_model(_transformer()))
    other = tc.get_lora_model(_transformer(), module_path='.*w1')

    with self.assertRaisesRegex(ValueError, 'same structure'):
      runner.transformer_state = nnx.state(other, nnx.LoRAParam)

  def test_set_transformer_state_rejects_other_variable_types(self):
    runner = model_runner_lib.ModelRunner(_transformer())

    with self.assertRaisesRegex(ValueError, 'Only LoRAParam is supported'):
      runner.transformer_state = nnx.State(
          {'w': nnx.BatchStat(jnp.ones((1,)))}
      )

  def test_update_params_replaces_the_weights(self):
    runner = model_runner_lib.ModelRunner(_transformer(seed=0))
    new_state = nnx.state(_transformer(seed=1), nnx.Param)

    runner.update_params(new_state)

    self._assert_trees_equal(runner.transformer_state, new_state)

  def test_update_params_keeps_the_weights_it_is_not_given(self):
    runner = model_runner_lib.ModelRunner(_transformer(seed=0))
    original = nnx.to_pure_dict(runner.transformer_state)
    new_kernel = jnp.ones_like(original['lm_head']['kernel'])

    runner.update_params({'lm_head': {'kernel': new_kernel}})

    updated = nnx.to_pure_dict(runner.transformer_state)
    np.testing.assert_array_equal(updated['lm_head']['kernel'], new_kernel)
    np.testing.assert_array_equal(
        updated['lm_head']['bias'], original['lm_head']['bias']
    )
    self._assert_trees_equal(updated['layers'], original['layers'])

  def test_update_params_casts_to_the_model_dtype(self):
    runner = model_runner_lib.ModelRunner(_transformer(seed=0))
    new_state = nnx.state(_transformer(seed=1, dtype=jnp.bfloat16), nnx.Param)

    runner.update_params(new_state)

    self._assert_trees_equal(
        runner.transformer_state,
        jax.tree.map(lambda x: x.astype(jnp.float32), new_state),
    )

  def test_update_params_reshards_onto_the_filtered_params(self):
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ('x',))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    transformer = _transformer(seed=0)
    nnx.update(transformer, jax.device_put(nnx.state(transformer), replicated))
    runner = model_runner_lib.ModelRunner(transformer)
    new_state = jax.device_put(
        nnx.state(_transformer(seed=1), nnx.Param), replicated
    )

    with mock.patch.object(
        reshard, 'reshard_pytree', wraps=reshard.reshard_pytree
    ) as mock_reshard:
      runner.update_params(new_state, filter_types=(nnx.Param,))

    mock_reshard.assert_called_once()
    self.assertIs(mock_reshard.call_args.args[0], new_state)
    self._assert_trees_equal(
        mock_reshard.call_args.args[1], nnx.state(_transformer(), nnx.Param)
    )
    self._assert_trees_equal(runner.transformer_state, new_state)

  def test_update_params_skips_resharding_without_named_sharding(self):
    runner = model_runner_lib.ModelRunner(_transformer(seed=0))
    new_state = nnx.state(_transformer(seed=1), nnx.Param)

    with mock.patch.object(reshard, 'reshard_pytree') as mock_reshard:
      runner.update_params(new_state, filter_types=(nnx.Param,))

    mock_reshard.assert_not_called()
    self._assert_trees_equal(runner.transformer_state, new_state)


if __name__ == '__main__':
  absltest.main()
