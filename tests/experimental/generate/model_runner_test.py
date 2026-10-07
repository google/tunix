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

import dataclasses
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.experimental.generate import model_runner as model_runner_lib
from tunix.experimental.rollout import sampler as sampler_lib
from tunix.models import paged_attention
from tunix.rl import reshard
from tunix.tests import test_common as tc


def _logits(row: list[float], num_rows: int = 1) -> jnp.ndarray:
  """Returns `[num_rows, vocab]` logits, repeating `row`."""
  return jnp.tile(jnp.array(row, dtype=jnp.float32), (num_rows, 1))


def _transformer(seed: int = 0, **config_kwargs) -> tc.ToyTransformer:
  return tc.ToyTransformer(
      config=tc.ModelConfig(**config_kwargs), rngs=nnx.Rngs(seed)
  )


def _mesh() -> jax.sharding.Mesh:
  return jax.sharding.Mesh(np.array(jax.devices()[:1]), ('x',))


def _runner(
    transformer: nnx.Module, **config_kwargs
) -> model_runner_lib.ModelRunner:
  config_kwargs.setdefault('max_top_k', 8)
  config_kwargs.setdefault('mesh', _mesh())
  return model_runner_lib.ModelRunner(
      transformer,
      model_runner_lib.ModelRunnerConfig(**config_kwargs),
  )


def _params(**kwargs) -> sampler_lib.SamplingParams:
  return sampler_lib.SamplingParams(max_tokens=8, **kwargs)


def _metadata(
    *params: sampler_lib.SamplingParams,
    num_rows: int | None = None,
    max_top_k: int = 8,
) -> model_runner_lib.SamplingMetadata:
  return model_runner_lib.SamplingMetadata.from_sampling_params(
      params, num_rows or len(params), max_top_k, np.random.default_rng(0)
  )


class ModelRunnerConfigTest(absltest.TestCase):

  def test_non_positive_max_top_k_raises(self):
    with self.assertRaisesRegex(ValueError, 'max_top_k'):
      model_runner_lib.ModelRunnerConfig(max_top_k=0, mesh=_mesh())

  def test_max_top_k_of_minus_one_sets_no_limit(self):
    config = model_runner_lib.ModelRunnerConfig(max_top_k=-1, mesh=_mesh())

    self.assertEqual(config.max_top_k, -1)

  def test_non_positive_num_scheduler_steps_raises(self):
    with self.assertRaisesRegex(ValueError, 'num_scheduler_steps'):
      model_runner_lib.ModelRunnerConfig(
          max_top_k=1, mesh=_mesh(), num_scheduler_steps=0
      )


class SamplingMetadataTest(absltest.TestCase):

  def test_packs_each_request_into_its_row(self):
    metadata = _metadata(
        _params(temperature=0.0),
        _params(temperature=0.7, top_p=0.9, top_k=3),
    )

    np.testing.assert_allclose(metadata.temperature, [0.0, 0.7])
    np.testing.assert_allclose(metadata.top_p, [1.0, 0.9])
    np.testing.assert_array_equal(metadata.top_k, [8, 3])

  def test_unset_top_k_falls_back_to_max_top_k(self):
    metadata = _metadata(_params())

    np.testing.assert_array_equal(metadata.top_k, [8])

  def test_padding_rows_sample_greedily(self):
    metadata = _metadata(_params(temperature=0.7), num_rows=3)

    np.testing.assert_allclose(metadata.temperature, [0.7, 0.0, 0.0])
    np.testing.assert_allclose(metadata.top_p, [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(metadata.top_k, [8, 1, 1])

  def test_more_requests_than_rows_raises(self):
    with self.assertRaisesRegex(ValueError, 'rows'):
      _metadata(_params(), _params(), num_rows=1)

  def test_top_k_above_max_top_k_raises(self):
    with self.assertRaisesRegex(ValueError, 'max_top_k'):
      _metadata(_params(top_k=9))

  def test_no_max_top_k_allows_any_top_k(self):
    metadata = _metadata(_params(top_k=100), _params(), max_top_k=-1)

    # An unset `top_k` inherits the engine's lack of a limit.
    np.testing.assert_array_equal(metadata.top_k, [100, -1])

  def test_a_set_seed_is_packed_as_is(self):
    metadata = _metadata(_params(seed=7), _params(seed=7), num_rows=3)

    np.testing.assert_array_equal(metadata.seed, [7, 7, 0])
    self.assertEqual(metadata.seed.dtype, np.uint32)

  def test_an_unset_seed_is_drawn_from_the_rng(self):
    rng = np.random.default_rng(0)
    seeds = [
        int(
            model_runner_lib.SamplingMetadata.from_sampling_params(
                (_params(),), 1, 8, rng
            ).seed[0]
        )
        for _ in range(8)
    ]

    # Every call draws a fresh seed.
    self.assertLen(set(seeds), 8)


class SampleTest(absltest.TestCase):

  def _sample(self, logits, metadata, max_top_k=8, seed=0):
    return model_runner_lib.sample(
        logits,
        jax.random.split(jax.random.PRNGKey(seed), logits.shape[0]),
        metadata,
        max_top_k=max_top_k,
        return_logprobs=True,
    )

  def _sampled_set(self, tokens) -> set[int]:
    return set(np.asarray(tokens).tolist())

  def test_zero_temperature_picks_the_argmax(self):
    logits = jnp.array([[0.0, 2.0, 1.0], [3.0, 0.0, 1.0]])

    tokens, logp = self._sample(
        logits, _metadata(_params(temperature=0.0), _params(temperature=0.0))
    )

    np.testing.assert_array_equal(tokens, [1, 0])
    # Greedy logprobs are taken over the unscaled logits.
    expected = jax.nn.log_softmax(logits, axis=-1)[jnp.arange(2), tokens]
    np.testing.assert_allclose(logp, expected, rtol=1e-6)

  def test_rows_follow_their_own_settings(self):
    # Row 0 is greedy, row 1 samples, both from the same logits.
    logits = _logits([0.0, 1.0, 1.0, 1.0], num_rows=2)
    metadata = _metadata(_params(temperature=0.0), _params())

    samples = [
        self._sample(logits, metadata, seed=seed)[0] for seed in range(200)
    ]

    self.assertEqual(self._sampled_set([s[0] for s in samples]), {1})
    self.assertEqual(
        self._sampled_set([s[1] for s in samples]), {0, 1, 2, 3}
    )

  def test_top_k_keeps_the_k_most_likely_tokens(self):
    logits = _logits([0.0, 3.0, 2.9, 1.0], num_rows=1000)

    tokens, _ = self._sample(logits, _metadata(*[_params(top_k=2)] * 1000))

    self.assertEqual(self._sampled_set(tokens), {1, 2})

  def test_top_k_one_picks_the_argmax(self):
    logits = _logits([0.0, 2.0, 1.0], num_rows=64)

    tokens, _ = self._sample(logits, _metadata(*[_params(top_k=1)] * 64))

    self.assertEqual(self._sampled_set(tokens), {1})

  def test_top_p_keeps_the_smallest_prefix_reaching_top_p(self):
    # Probabilities 0.4, 0.35 and 0.25: the first two already exceed 0.5.
    logits = _logits(np.log([0.4, 0.35, 0.25]).tolist(), num_rows=1000)

    tokens, logp = self._sample(
        logits, _metadata(*[_params(top_p=0.5)] * 1000)
    )

    self.assertEqual(self._sampled_set(tokens), {0, 1})
    # The logprob is taken over the unfiltered distribution.
    np.testing.assert_allclose(
        logp, np.log([0.4, 0.35])[np.asarray(tokens)], rtol=1e-5
    )

  def test_unset_top_k_is_capped_at_max_top_k(self):
    logits = _logits([0.0, 3.0, 2.9, 1.0], num_rows=1000)

    tokens, _ = self._sample(
        logits,
        _metadata(*[_params()] * 1000, max_top_k=2),
        max_top_k=2,
    )

    self.assertEqual(self._sampled_set(tokens), {1, 2})

  def test_max_top_k_above_the_vocab_keeps_every_token(self):
    logits = _logits([0.0, 0.0, 0.0], num_rows=1000)

    tokens, _ = self._sample(logits, _metadata(*[_params()] * 1000))

    self.assertEqual(self._sampled_set(tokens), {0, 1, 2})

  def test_no_max_top_k_keeps_every_token(self):
    logits = _logits([0.0, 0.0, 0.0], num_rows=1000)
    metadata = _metadata(*[_params()] * 1000, max_top_k=-1)

    tokens, _ = self._sample(logits, metadata, max_top_k=-1)

    self.assertEqual(self._sampled_set(tokens), {0, 1, 2})

  def test_row_top_k_still_applies_without_max_top_k(self):
    logits = _logits([0.0, 3.0, 2.9, 1.0], num_rows=1000)
    metadata = _metadata(*[_params(top_k=2)] * 1000, max_top_k=-1)

    tokens, _ = self._sample(logits, metadata, max_top_k=-1)

    self.assertEqual(self._sampled_set(tokens), {1, 2})

  def test_logprobs_use_the_row_temperature(self):
    logits = _logits([0.0, 2.0, 1.0], num_rows=8)

    tokens, logp = self._sample(
        logits, _metadata(*[_params(temperature=2.0, top_k=1)] * 8)
    )

    np.testing.assert_array_equal(tokens, [1] * 8)
    expected = jax.nn.log_softmax(jnp.array([0.0, 1.0, 0.5]))[1]
    np.testing.assert_allclose(logp, [expected] * 8, rtol=1e-6)

  def test_rows_with_the_same_key_sample_the_same_token(self):
    logits = _logits([0.0, 0.0, 0.0, 0.0], num_rows=64)
    keys = jnp.stack([jax.random.PRNGKey(3)] * 64)

    tokens, _ = model_runner_lib.sample(
        logits, keys, _metadata(*[_params()] * 64), max_top_k=8
    )

    self.assertLen(self._sampled_set(tokens), 1)

  def test_without_logprobs_returns_none(self):
    _, logp = model_runner_lib.sample(
        _logits([0.0, 1.0]),
        jax.random.split(jax.random.PRNGKey(0), 1),
        _metadata(_params()),
        max_top_k=8,
    )

    self.assertIsNone(logp)


class ModelRunnerTest(absltest.TestCase):

  def _assert_trees_equal(self, x, y):
    jax.tree.map(np.testing.assert_array_equal, x, y)

  def test_config_is_exposed(self):
    runner = _runner(_transformer(), return_logprobs=True)

    self.assertTrue(runner.config.return_logprobs)

  def test_transformer_merges_the_state(self):
    transformer = _transformer()
    runner = _runner(transformer)

    graphdef, state = runner.model_def_and_state()

    self.assertEqual(graphdef, nnx.graphdef(transformer))
    self._assert_trees_equal(
        nnx.state(nnx.merge(graphdef, state)), nnx.state(transformer)
    )
    self._assert_trees_equal(
        nnx.state(runner.transformer), nnx.state(transformer)
    )

  def test_dtype_comes_from_the_model_config(self):
    runner = _runner(_transformer(dtype=jnp.bfloat16))

    self.assertEqual(runner.dtype, jnp.bfloat16)

  def test_dtype_falls_back_to_the_weights(self):
    class NoConfig(nnx.Module):

      def __init__(self):
        self.w = nnx.Param(jnp.ones((1,), dtype=jnp.bfloat16))

      def __call__(self, tokens, positions, cache, metadata):
        raise NotImplementedError()

    runner = _runner(NoConfig())

    self.assertEqual(runner.dtype, jnp.bfloat16)

  def test_set_transformer_state_replaces_the_full_state(self):
    runner = _runner(_transformer(seed=0))
    new_state = nnx.variables(_transformer(seed=1), nnx.Param)

    runner.transformer_state = new_state

    self._assert_trees_equal(runner.transformer_state, new_state)
    self._assert_trees_equal(nnx.state(runner.transformer), new_state)

  def test_set_transformer_state_rejects_a_different_structure(self):
    runner = _runner(_transformer(num_layers=4))

    with self.assertRaisesRegex(ValueError, 'same structure'):
      runner.transformer_state = nnx.variables(
          _transformer(num_layers=6), nnx.Param
      )

  def test_set_transformer_state_rejects_a_different_dtype(self):
    runner = _runner(_transformer())

    with self.assertRaisesRegex(ValueError, 'same shape, dtype and sharding'):
      runner.transformer_state = nnx.variables(
          _transformer(dtype=jnp.bfloat16), nnx.Param
      )

  def test_set_transformer_state_rejects_a_different_sharding_type(self):
    runner = _runner(_transformer())
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
    runner = _runner(transformer)
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
    runner = _runner(tc.get_lora_model(_transformer()))
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
    runner = _runner(tc.get_lora_model(_transformer()))
    other = tc.get_lora_model(_transformer(), module_path='.*w1')

    with self.assertRaisesRegex(ValueError, 'same structure'):
      runner.transformer_state = nnx.state(other, nnx.LoRAParam)

  def test_set_transformer_state_rejects_other_variable_types(self):
    runner = _runner(_transformer())

    with self.assertRaisesRegex(ValueError, 'Only LoRAParam is supported'):
      runner.transformer_state = nnx.State(
          {'w': nnx.BatchStat(jnp.ones((1,)))}
      )

  def test_update_params_replaces_the_weights(self):
    runner = _runner(_transformer(seed=0))
    new_state = nnx.state(_transformer(seed=1), nnx.Param)

    runner.update_params(new_state)

    self._assert_trees_equal(runner.transformer_state, new_state)

  def test_update_params_keeps_the_weights_it_is_not_given(self):
    runner = _runner(_transformer(seed=0))
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
    runner = _runner(_transformer(seed=0))
    new_state = nnx.state(_transformer(seed=1, dtype=jnp.bfloat16), nnx.Param)

    runner.update_params(new_state)

    self._assert_trees_equal(
        runner.transformer_state,
        jax.tree.map(lambda x: x.astype(jnp.float32), new_state),
    )

  def test_update_params_reshards_onto_the_filtered_params(self):
    runner = _runner(_transformer(seed=0))
    new_state = nnx.state(_transformer(seed=1), nnx.Param)

    with mock.patch.object(
        reshard, 'reshard_pytree', wraps=reshard.reshard_pytree
    ) as mock_reshard:
      runner.update_params(new_state, filter_types=nnx.Param)

    mock_reshard.assert_called_once()
    self.assertIs(mock_reshard.call_args.args[0], new_state)
    self._assert_trees_equal(
        mock_reshard.call_args.args[1], nnx.state(_transformer(), nnx.Param)
    )
    self._assert_trees_equal(runner.transformer_state, new_state)

  def test_update_params_falls_back_when_resharding_fails(self):
    runner = _runner(_transformer(seed=0))
    new_state = nnx.state(_transformer(seed=1), nnx.Param)

    with mock.patch.object(reshard, 'reshard_pytree', side_effect=ValueError):
      runner.update_params(new_state, filter_types=nnx.Param)

    self._assert_trees_equal(runner.transformer_state, new_state)


_VOCAB_SIZE = 16


class _NextTokenTransformer(nnx.Module):
  """A stand-in transformer that follows the RPA calling convention.

  Each token's most likely successor is `token + 1`, then `token + 2`, and the
  positions it was called with are written into the cache, so tests can check
  both what the runner sampled and what it fed in.
  """

  def __init__(self):
    self.scale = nnx.Param(jnp.ones(()))

  def __call__(self, tokens, positions, cache, metadata, mesh):
    del metadata, mesh
    logits = self.scale * (
        10.0 * jax.nn.one_hot((tokens + 1) % _VOCAB_SIZE, _VOCAB_SIZE)
        + 5.0 * jax.nn.one_hot((tokens + 2) % _VOCAB_SIZE, _VOCAB_SIZE)
    )
    # The cache keeps its shape, as the decode loop carries it between steps.
    return logits, {
        'positions': cache['positions'].at[: positions.shape[0]].set(positions)
    }


class _UniformTransformer(nnx.Module):
  """A stand-in transformer that gives every token the same logit."""

  def __init__(self):
    self.scale = nnx.Param(jnp.ones(()))

  def __call__(self, tokens, positions, cache, metadata, mesh):
    del positions, metadata, mesh
    return self.scale * jnp.zeros((tokens.shape[0], _VOCAB_SIZE)), cache


def _rpa_metadata(
    kv_lens: list[int], query_lens: list[int], distribution: list[int]
) -> paged_attention.RPAMetadata:
  num_rows = len(query_lens)
  return paged_attention.RPAMetadata(
      page_indices={
          'layer_0': np.arange(2 * num_rows, dtype=np.int32).reshape(
              num_rows, 2
          )
      },
      kv_lens=np.array(kv_lens, dtype=np.int32),
      query_lens=np.array(query_lens, dtype=np.int32),
      distribution=np.array(distribution, dtype=np.int32),
  )


def _positions_cache(num_tokens: int) -> model_runner_lib.Cache:
  return {'positions': jnp.zeros((num_tokens,), dtype=jnp.int32)}


class RaggedArrayTest(absltest.TestCase):

  def test_row_idxs_and_intra_offsets(self):
    ragged = model_runner_lib.RaggedArray(
        data=jnp.zeros((6,)), lens=jnp.array([1, 3, 0])
    )

    # The two trailing padding elements are attributed to the last row.
    np.testing.assert_array_equal(ragged.row_idxs, [0, 1, 1, 1, 2, 2])
    np.testing.assert_array_equal(ragged.intra_offsets, [0, 0, 1, 2, 0, 1])


class ExecuteStepTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # One decode with 5 cached tokens, then a full prefill of 3 tokens, then a
    # padding row.
    self.metadata = _rpa_metadata(
        kv_lens=[6, 3, 0], query_lens=[1, 3, 0], distribution=[1, 1, 2]
    )
    self.tokens = jnp.array([4, 7, 8, 9], dtype=jnp.int32)
    self.greedy = (_params(temperature=0.0), _params(temperature=0.0))

  def _execute(self, runner, sampling_params=None):
    return runner.execute_step(
        cache=_positions_cache(4),
        tokens=self.tokens,
        metadata=self.metadata,
        sampling_params=sampling_params or self.greedy,
    )

  def test_samples_from_the_last_token_of_each_row(self):
    tokens, logits, logp, _ = self._execute(_runner(_NextTokenTransformer()))

    self.assertIsInstance(tokens, jax.Array)
    # The decode continues from 4, and the prefill from its last token 9.
    np.testing.assert_array_equal(tokens[:2], [[5], [10]])
    self.assertIsNone(logits)
    self.assertIsNone(logp)

  def test_positions_continue_from_the_cached_tokens(self):
    _, _, _, cache = self._execute(_runner(_NextTokenTransformer()))

    np.testing.assert_array_equal(cache['positions'], [5, 0, 1, 2])

  def test_returns_last_token_logits_per_row(self):
    runner = _runner(_NextTokenTransformer(), return_logits=True)

    tokens, logits, _, _ = self._execute(runner)

    np.testing.assert_array_equal(tokens[:2], [[5], [10]])
    self.assertEqual(logits.shape, (3, 1, _VOCAB_SIZE))
    # The prefill's logits come from its last token, 9.
    self.assertEqual(int(np.argmax(logits[1, 0])), 10)

  def test_returns_logprobs_per_row(self):
    runner = _runner(_NextTokenTransformer(), return_logprobs=True)

    _, _, logp, _ = self._execute(runner)

    self.assertEqual(logp.shape, (3, 1))
    expected = jax.nn.log_softmax(
        10.0 * jax.nn.one_hot(5, _VOCAB_SIZE)
        + 5.0 * jax.nn.one_hot(6, _VOCAB_SIZE)
    )[5]
    np.testing.assert_allclose(logp[0, 0], expected, rtol=1e-6)

  def test_each_row_uses_its_own_sampling_params(self):
    # The prefill is limited to its single most likely token, whatever its
    # temperature, while the decode stays greedy.
    runner = _runner(_NextTokenTransformer())

    tokens, _, _, _ = self._execute(
        runner, (_params(temperature=0.0), _params(temperature=5.0, top_k=1))
    )

    np.testing.assert_array_equal(tokens[:2], [[5], [10]])

  def test_single_step_avoids_stack_steps_and_batches_metadata_transfer(self):
    runner = _runner(
        _NextTokenTransformer(), return_logits=True, return_logprobs=True
    )
    with (
        mock.patch.object(
            model_runner_lib,
            '_stack_steps',
            wraps=model_runner_lib._stack_steps,
        ) as mock_stack,
        mock.patch.object(
            runner, '_to_device', wraps=runner._to_device
        ) as mock_to_device,
    ):
      tokens, logits, logp, _ = self._execute(runner)

    mock_stack.assert_not_called()
    # Both `metadata` and `sampling_metadata` are transferred in one call.
    mock_to_device.assert_called_once()
    self.assertEqual(tokens.shape, (3, 1))
    self.assertEqual(logits.shape, (3, 1, _VOCAB_SIZE))
    self.assertEqual(logp.shape, (3, 1))

  def test_sampling_params_must_match_the_scheduled_requests(self):
    runner = _runner(_NextTokenTransformer())

    with self.assertRaisesRegex(ValueError, 'sampling params'):
      self._execute(runner, (_params(),))

  def test_to_device_deduplicates_shared_page_indices(self):
    runner = _runner(_NextTokenTransformer())
    shared_pages = self.metadata.page_indices['layer_0']
    metadata = dataclasses.replace(
        self.metadata,
        page_indices={
            'layer_0': shared_pages,
            'layer_1': shared_pages,
        },
    )

    with mock.patch.object(
        jax, 'device_put', wraps=jax.device_put
    ) as mock_device_put:
      device_metadata = runner._to_device(metadata)

    mock_device_put.assert_called_once()
    put_leaves = mock_device_put.call_args.args[0]
    # 1 shared page table + kv_lens + query_lens + distribution.
    self.assertLen(put_leaves, 4)
    self.assertIs(
        device_metadata.page_indices['layer_0'],
        device_metadata.page_indices['layer_1'],
    )


class ExecuteStepSeedTest(absltest.TestCase):
  """Checks how sampling depends on the request and engine seeds."""

  _NUM_ROWS = 64

  def _execute(self, runner, sampling_params=None, kv_len=1):
    # Every row decodes one token from a uniform distribution.
    num_rows = len(sampling_params) if sampling_params else self._NUM_ROWS
    metadata = _rpa_metadata(
        kv_lens=[kv_len] * num_rows,
        query_lens=[1] * num_rows,
        distribution=[num_rows] * 3,
    )
    tokens, _, _, _ = runner.execute_step(
        cache={},
        tokens=jnp.zeros((num_rows,), dtype=jnp.int32),
        metadata=metadata,
        sampling_params=sampling_params or (_params(),) * num_rows,
    )
    return tokens

  def test_unseeded_sampling_is_reproducible_for_an_engine_seed(self):
    tokens = self._execute(_runner(_UniformTransformer(), seed=3))
    again = self._execute(_runner(_UniformTransformer(), seed=3))

    np.testing.assert_array_equal(tokens, again)

  def test_unseeded_sampling_differs_every_step(self):
    runner = _runner(_UniformTransformer())

    first = self._execute(runner)
    second = self._execute(runner)

    self.assertFalse(np.array_equal(first, second))

  def test_unseeded_sampling_depends_on_the_engine_seed(self):
    tokens = self._execute(_runner(_UniformTransformer(), seed=0))
    other = self._execute(_runner(_UniformTransformer(), seed=1))

    self.assertFalse(np.array_equal(tokens, other))

  def test_seeded_sampling_ignores_the_engine_seed(self):
    seeded = (_params(seed=7),) * 4

    tokens = self._execute(_runner(_UniformTransformer(), seed=0), seeded)
    other = self._execute(_runner(_UniformTransformer(), seed=1), seeded)

    np.testing.assert_array_equal(tokens, other)

  def test_seeded_sampling_ignores_the_batch(self):
    runner = _runner(_UniformTransformer())

    alone = self._execute(runner, (_params(seed=7),))
    # The same request, run in a later step next to others, at the same
    # position, samples the same token.
    batched = self._execute(
        runner, tuple(_params(seed=s) for s in range(8)) + (_params(seed=7),)
    )

    np.testing.assert_array_equal(batched[-1], alone[0])

  def test_seeded_sampling_depends_on_the_position(self):
    runner = _runner(_UniformTransformer())
    seeded = tuple(_params(seed=s) for s in range(self._NUM_ROWS))

    first = self._execute(runner, seeded, kv_len=1)
    second = self._execute(runner, seeded, kv_len=2)

    self.assertFalse(np.array_equal(first, second))

  def test_seeded_sampling_ignores_how_steps_are_split(self):
    seeded = (_params(seed=7),)

    multi_step = self._execute(
        _runner(_UniformTransformer(), num_scheduler_steps=3), seeded
    )
    runner = _runner(_UniformTransformer())
    single_steps = [
        self._execute(runner, seeded, kv_len=kv_len)[0, 0]
        for kv_len in (1, 2, 3)
    ]

    np.testing.assert_array_equal(multi_step[0], single_steps)


def _fake_step_fn(params, cache, tokens, metadata, sampling_metadata):
  """A stand-in for `ModelRunner._model_step_fn` that always samples token 1."""
  del params, tokens, sampling_metadata
  num_rows = len(metadata.query_lens)
  return jnp.ones((num_rows,), dtype=jnp.int32), None, None, cache


class MultiStepExecuteTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # A decode continuing from 4, a chunked prefill ending in 4, a full
    # prefill ending in 9, and a padding row.
    self.metadata = _rpa_metadata(
        kv_lens=[6, 4, 3, 0], query_lens=[1, 4, 3, 0], distribution=[1, 2, 3]
    )
    self.tokens = jnp.array([4, 1, 2, 3, 4, 7, 8, 9], dtype=jnp.int32)
    self.greedy = (_params(temperature=0.0),) * 3

  def _execute(self, runner):
    return runner.execute_step(
        cache=_positions_cache(8),
        tokens=self.tokens,
        metadata=self.metadata,
        sampling_params=self.greedy,
    )

  def test_surviving_rows_keep_decoding(self):
    runner = _runner(_NextTokenTransformer(), num_scheduler_steps=3)

    tokens, _, _, cache = self._execute(runner)

    np.testing.assert_array_equal(tokens[0], [5, 6, 7])
    np.testing.assert_array_equal(tokens[2], [10, 11, 12])
    # The last step decoded the two survivors at the ends of their sequences.
    np.testing.assert_array_equal(cache['positions'][:2], [7, 4])

  def test_returns_logits_and_logprobs_for_every_step(self):
    runner = _runner(
        _NextTokenTransformer(),
        num_scheduler_steps=3,
        return_logits=True,
        return_logprobs=True,
    )

    _, logits, logp, _ = self._execute(runner)

    self.assertEqual(logits.shape, (4, 3, _VOCAB_SIZE))
    self.assertEqual(logp.shape, (4, 3))
    # The full prefill's second step decoded 10, so it favours 11.
    np.testing.assert_array_equal(np.argmax(logits[2, 1]), 11)
    expected = jax.nn.log_softmax(
        10.0 * jax.nn.one_hot(11, _VOCAB_SIZE)
        + 5.0 * jax.nn.one_hot(12, _VOCAB_SIZE)
    )[11]
    np.testing.assert_allclose(logp[2, 1], expected, rtol=1e-6)

  def test_the_metadata_advances_between_steps(self):
    runner = _runner(_NextTokenTransformer(), num_scheduler_steps=3)
    mock_step = mock.Mock(wraps=_fake_step_fn)
    runner._model_step_fn = mock_step
    runner._compiled_model_step_fn = mock_step

    # The decode loop is a compiled `scan`, so its metadata is only concrete
    # with tracing turned off.
    with jax.disable_jit():
      self._execute(runner)

    step_metadata = [call.args[3] for call in mock_step.call_args_list]
    self.assertLen(step_metadata, 3)
    # The scheduled batch runs as-is.
    np.testing.assert_array_equal(step_metadata[0].query_lens, [1, 4, 3, 0])
    np.testing.assert_array_equal(step_metadata[0].distribution, [1, 2, 3])
    # The chunked prefill drops out and the rest decode one token per step.
    for metadata in step_metadata[1:]:
      np.testing.assert_array_equal(metadata.query_lens, [1, 1, 0, 0])
      np.testing.assert_array_equal(metadata.distribution, [2, 2, 2])
    np.testing.assert_array_equal(step_metadata[1].kv_lens, [7, 4, 0, 0])
    np.testing.assert_array_equal(step_metadata[2].kv_lens, [8, 5, 0, 0])
    # The full prefill's pages move up with it into the second row.
    np.testing.assert_array_equal(
        step_metadata[1].page_indices['layer_0'][:2], [[0, 1], [4, 5]]
    )

  def test_sampling_settings_follow_their_rows(self):
    runner = _runner(_NextTokenTransformer(), num_scheduler_steps=2)
    mock_step = mock.Mock(wraps=_fake_step_fn)
    runner._model_step_fn = mock_step
    runner._compiled_model_step_fn = mock_step

    with jax.disable_jit():
      runner.execute_step(
          cache=_positions_cache(8),
          tokens=self.tokens,
          metadata=self.metadata,
          sampling_params=(
              _params(temperature=0.5, top_k=2, seed=1),
              _params(temperature=0.9, seed=2),
              _params(temperature=0.7, top_p=0.8, seed=3),
          ),
      )

    decode_sampling_metadata = mock_step.call_args_list[1].args[4]
    # The full prefill's settings move up with it, and the vacated rows sample
    # greedily like any other padding row.
    np.testing.assert_allclose(
        decode_sampling_metadata.temperature, [0.5, 0.7, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        decode_sampling_metadata.top_p, [1.0, 0.8, 1.0, 1.0]
    )
    np.testing.assert_array_equal(decode_sampling_metadata.top_k, [2, 8, 1, 1])
    np.testing.assert_array_equal(decode_sampling_metadata.seed, [1, 3, 0, 0])

  def test_stops_when_every_row_drops_out(self):
    # A batch of nothing but chunked prefills has no row left to decode, so
    # only the scheduled step runs.
    runner = _runner(_NextTokenTransformer(), num_scheduler_steps=4)
    runner._compiled_decode_loop_fn = mock.Mock()

    tokens, _, _, _ = runner.execute_step(
        cache=_positions_cache(8),
        tokens=self.tokens,
        metadata=_rpa_metadata(
            kv_lens=[4, 4, 0], query_lens=[4, 4, 0], distribution=[0, 2, 2]
        ),
        sampling_params=self.greedy[:2],
    )

    runner._compiled_decode_loop_fn.assert_not_called()
    self.assertEqual(tokens.shape, (3, 4))
    np.testing.assert_array_equal(tokens[:, 1:], 0)


if __name__ == '__main__':
  absltest.main()
