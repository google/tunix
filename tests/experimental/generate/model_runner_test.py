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

from concurrent import futures
import dataclasses
import os
import threading
import time
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.experimental.generate import configs
from tunix.experimental.generate import model_runner as model_runner_lib
from tunix.experimental.generate import request as request_lib
from tunix.models import paged_attention
from tunix.rl import reshard
from tunix.tests import test_common as tc

# Splitting the sampled rows across DP groups needs up to 4 devices.
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'


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
      configs.ModelRunnerConfig(**config_kwargs),
  )


def _params(**kwargs) -> request_lib.SamplingParams:
  return request_lib.SamplingParams(max_tokens=8, **kwargs)


def _metadata(
    *params: request_lib.SamplingParams,
    num_rows: int | None = None,
    max_top_k: int = 8,
) -> model_runner_lib.SamplingMetadata:
  n = len(params)
  return model_runner_lib.SamplingMetadata.from_sampling_params(
      params,
      num_rows or n,
      max_top_k,
      distribution=np.array([[n, n, n]], dtype=np.int32),
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

  def test_unconstrained_requires_no_top_k_or_top_p_truncation(self):
    greedy = _params(temperature=0.0, top_k=1, top_p=0.5)
    for metadata, expected in (
        (_metadata(_params(), greedy, num_rows=4, max_top_k=-1), True),
        (_metadata(_params(), max_top_k=0), True),
        (_metadata(_params(), max_top_k=8), False),
        (_metadata(_params(), _params(top_k=2), max_top_k=-1), False),
        (_metadata(_params(), _params(top_p=0.9), max_top_k=-1), False),
    ):
      self.assertIs(metadata.unconstrained, expected)


class SampleTest(absltest.TestCase):

  def _sample(self, logits, metadata, max_top_k=8, seed=0):
    return model_runner_lib.sample(
        logits,
        jax.random.PRNGKey(seed),
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
    self.assertEqual(self._sampled_set([s[1] for s in samples]), {0, 1, 2, 3})

  def test_top_k_keeps_the_k_most_likely_tokens(self):
    logits = _logits([0.0, 3.0, 2.9, 1.0], num_rows=1000)

    tokens, _ = self._sample(logits, _metadata(*[_params(top_k=2)] * 1000))

    self.assertEqual(self._sampled_set(tokens), {1, 2})

  def test_top_k_one_picks_the_argmax(self):
    logits = _logits([0.0, 2.0, 1.0], num_rows=64)

    tokens, _ = self._sample(logits, _metadata(*[_params(top_k=1)] * 64))

    self.assertEqual(self._sampled_set(tokens), {1})

  def test_non_positive_top_k_disables_top_k_filtering(self):
    logits = _logits([0.0, 0.0, 0.0], num_rows=1000)

    tokens, _ = self._sample(logits, _metadata(*[_params(top_k=0)] * 1000))

    self.assertEqual(self._sampled_set(tokens), {0, 1, 2})

  def test_top_p_keeps_the_smallest_prefix_reaching_top_p(self):
    # Probabilities 0.4, 0.35 and 0.25: the first two already exceed 0.5.
    logits = _logits(np.log([0.4, 0.35, 0.25]).tolist(), num_rows=1000)

    tokens, logp = self._sample(logits, _metadata(*[_params(top_p=0.5)] * 1000))

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

  def test_temperature_scales_logits_before_top_p(self):
    logits = _logits([1.0, 0.0], num_rows=1000)

    low_temp, _ = self._sample(
        logits, _metadata(*[_params(temperature=0.5, top_p=0.8)] * 1000)
    )
    high_temp, _ = self._sample(
        logits, _metadata(*[_params(temperature=2.0, top_p=0.8)] * 1000)
    )

    self.assertEqual(self._sampled_set(low_temp), {0})
    self.assertEqual(self._sampled_set(high_temp), {0, 1})

  def test_logprobs_use_the_row_temperature(self):
    logits = _logits([0.0, 2.0, 1.0], num_rows=8)

    tokens, logp = self._sample(
        logits, _metadata(*[_params(temperature=2.0, top_k=1)] * 8)
    )

    np.testing.assert_array_equal(tokens, [1] * 8)
    expected = jax.nn.log_softmax(jnp.array([0.0, 1.0, 0.5]))[1]
    np.testing.assert_allclose(logp, [expected] * 8, rtol=1e-6)

  def test_rows_sample_independently_under_one_key(self):
    logits = _logits([0.0, 0.0, 0.0, 0.0], num_rows=64)

    tokens, _ = model_runner_lib.sample(
        logits, jax.random.PRNGKey(3), _metadata(*[_params()] * 64), max_top_k=8
    )

    self.assertGreater(len(self._sampled_set(tokens)), 1)

  def test_the_same_key_samples_the_same_tokens(self):
    logits = _logits([0.0, 0.0, 0.0, 0.0], num_rows=64)
    metadata = _metadata(*[_params()] * 64)

    first, _ = self._sample(logits, metadata, seed=3)
    second, _ = self._sample(logits, metadata, seed=3)

    np.testing.assert_array_equal(first, second)

  def test_without_logprobs_returns_none(self):
    _, logp = model_runner_lib.sample(
        _logits([0.0, 1.0]),
        jax.random.PRNGKey(0),
        _metadata(_params()),
        max_top_k=8,
    )

    self.assertIsNone(logp)

  def test_unconstrained_skips_top_k_and_handles_all_neg_inf(self):
    metadata = _metadata(_params(temperature=0.0), _params(), max_top_k=-1)
    for vocab_size in (16, 4096):
      logits = jnp.full((2, vocab_size), -jnp.inf, dtype=jnp.bfloat16)
      with mock.patch.object(jax.lax, 'top_k') as mock_top_k:
        tokens, logp = model_runner_lib.sample(
            logits, jax.random.PRNGKey(0), metadata, max_top_k=-1
        )
        mock_top_k.assert_not_called()
      self.assertEqual(tokens.dtype, jnp.int32)
      np.testing.assert_array_equal(tokens, [0, 0])
      self.assertIsNone(logp)
      _, neg_inf_logp = self._sample(logits, metadata, max_top_k=-1)
      assert neg_inf_logp is not None
      self.assertTrue(bool(jnp.all(jnp.isnan(neg_inf_logp))))

  def test_unconstrained_greedy_ties_extreme_temps_and_logprobs(self):
    for vocab_size, mask_start, mask_stop, ties, solo in (
        (16, 4, 8, [2, 3, 12], 13),
        (4096, 512, 768, [300, 310, 3000], 3500),
    ):
      base = jax.random.normal(
          jax.random.PRNGKey(1), (6, vocab_size), dtype=jnp.bfloat16
      )
      logits = (
          base.at[:, mask_start:mask_stop]
          .set(-jnp.inf)
          .at[0, jnp.array(ties)]
          .set(jnp.bfloat16(12.0))
          .at[2, :]
          .set(-jnp.inf)
          .at[
              jnp.array([2, 3, 3, 5]), jnp.array([solo, ties[0], solo, ties[2]])
          ]
          .set(jnp.array([-100.0, 8.0, 12.0, 11.0], dtype=jnp.bfloat16))
      )
      metadata = _metadata(
          *(_params(temperature=t) for t in (0.0, 0.7, 1e-40, 2e-38, 1e38)),
          num_rows=6,
          max_top_k=-1,
      )
      tokens, logp = self._sample(logits, metadata, max_top_k=-1, seed=2)
      self.assertEqual(tokens.dtype, jnp.int32)
      np.testing.assert_array_equal(
          np.asarray(tokens)[[0, 2, 3, 5]], [ties[0], solo, solo, ties[2]]
      )
      self.assertTrue(
          bool(jnp.all((tokens < mask_start) | (tokens >= mask_stop)))
      )
      f32 = logits.astype(jnp.float32)
      normal_logp = jax.nn.log_softmax(
          f32 / jnp.array([1.0, 0.7, 1.0, 1.0, 1.0, 1.0])[:, None], axis=-1
      )[jnp.arange(6), tokens]
      expected = np.array([
          float(normal_logp[0]),
          float(normal_logp[1]),
          0.0,
          0.0,
          -np.log(vocab_size - (mask_stop - mask_start)),
          float(normal_logp[5]),
      ])
      np.testing.assert_allclose(logp, expected, rtol=1e-5, atol=1e-6)

  def test_hierarchical_unconstrained_matches_softmax_frequencies(self):
    row_logits = (
        jnp.full((4096,), -6.0, dtype=jnp.bfloat16)
        .at[512:768]
        .set(-jnp.inf)
        .at[jnp.array([13, 300, 1025, 2500, 4095])]
        .set(jnp.array([1.5, 0.75, 0.25, -0.25, 1.0], dtype=jnp.bfloat16))
    )
    num_rows, temperature = 16384, 0.8
    metadata = model_runner_lib.SamplingMetadata(
        temperature=jnp.full((num_rows,), temperature, dtype=jnp.float32),
        top_p=jnp.ones((num_rows,), dtype=jnp.float32),
        top_k=jnp.full((num_rows,), -1, dtype=jnp.int32),
        unconstrained=True,
    )
    jit_sample = jax.jit(
        model_runner_lib.sample,
        static_argnames=['max_top_k', 'return_logprobs'],
    )
    tokens, _ = jit_sample(
        jnp.broadcast_to(row_logits, (num_rows, 4096)),
        jax.random.PRNGKey(0),
        metadata,
        max_top_k=-1,
    )
    counts = np.bincount(np.asarray(tokens), minlength=4096)
    expected = jax.nn.softmax(row_logits.astype(jnp.float32) / temperature)
    np.testing.assert_array_equal(counts[512:768], 0)
    np.testing.assert_allclose(counts / num_rows, expected, atol=1.5e-2)


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
      runner.transformer_state = nnx.State({'w': nnx.BatchStat(jnp.ones((1,)))})

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
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ('x',))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    transformer = _transformer(seed=0)
    nnx.update(transformer, jax.device_put(nnx.state(transformer), replicated))
    runner = _runner(transformer)
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
    runner = _runner(_transformer(seed=0))
    new_state = nnx.state(_transformer(seed=1), nnx.Param)

    with mock.patch.object(reshard, 'reshard_pytree') as mock_reshard:
      runner.update_params(new_state, filter_types=(nnx.Param,))

    mock_reshard.assert_not_called()
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

  def __call__(
      self, tokens, positions, cache, metadata, mesh, logits_idxs=None
  ):
    del metadata, mesh
    logits = self.scale * (
        10.0 * jax.nn.one_hot((tokens + 1) % _VOCAB_SIZE, _VOCAB_SIZE)
        + 5.0 * jax.nn.one_hot((tokens + 2) % _VOCAB_SIZE, _VOCAB_SIZE)
    )
    if logits_idxs is not None:
      logits = jnp.take_along_axis(logits, logits_idxs[..., None], axis=1)
    logits = logits.reshape(-1, _VOCAB_SIZE)
    flat_positions = positions.reshape(-1)
    # The cache keeps its shape, as the decode loop carries it between steps.
    return logits, {
        'positions': (
            cache['positions'].at[: flat_positions.shape[0]].set(flat_positions)
        )
    }


class _UniformTransformer(nnx.Module):
  """A stand-in transformer that gives every token the same logit."""

  def __init__(self):
    self.scale = nnx.Param(jnp.ones(()))

  def __call__(
      self, tokens, positions, cache, metadata, mesh, logits_idxs=None
  ):
    del positions, metadata, mesh
    logits = self.scale * jnp.zeros((*tokens.shape, _VOCAB_SIZE))
    if logits_idxs is not None:
      logits = jnp.take_along_axis(logits, logits_idxs[..., None], axis=1)
    return logits.reshape(-1, _VOCAB_SIZE), cache


def _rpa_metadata(
    kv_lens: list[int],
    query_lens: list[int],
    distribution: list[int],
    *,
    decode_only: bool = False,
) -> paged_attention.PagedAttnMetadata:
  num_rows = len(query_lens)
  return paged_attention.PagedAttnMetadata(
      page_indices={
          'layer_0': (
              np.arange(2 * num_rows, dtype=np.int32).reshape(num_rows, 2)
          )
      },
      kv_lens=np.array(kv_lens, dtype=np.int32),
      query_lens=np.array(query_lens, dtype=np.int32),
      distribution=np.array([distribution], dtype=np.int32),
      decode_only=decode_only,
  )


def _positions_cache(num_tokens: int) -> model_runner_lib.Cache:
  return {'positions': jnp.zeros((num_tokens,), dtype=jnp.int32)}


class ExecuteStepTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # One decode with 5 cached tokens, then a full prefill of 3 tokens, then a
    # padding row.
    self.metadata = _rpa_metadata(
        kv_lens=[6, 3, 0], query_lens=[1, 3, 0], distribution=[1, 1, 2]
    )
    self.tokens = jnp.array([[4, 7, 8, 9]], dtype=jnp.int32)
    self.greedy = (_params(temperature=0.0), _params(temperature=0.0))

  def _execute(self, runner, sampling_params=None):
    return runner.execute_step(
        cache=_positions_cache(4),
        tokens=self.tokens,
        metadata=self.metadata,
        sampling_params=sampling_params or self.greedy,
    )

  def test_get_token_positions(self):
    metadata = _rpa_metadata(
        kv_lens=[1, 7, 0], query_lens=[1, 3, 0], distribution=[1, 1, 2]
    )

    positions = model_runner_lib._get_token_positions(
        tokens=jnp.zeros((1, 6), dtype=jnp.int32), metadata=metadata
    )

    # Row 0 starts at 1 - 1 = 0 ([0]); row 1 starts at 7 - 3 = 4 ([4, 5, 6]);
    # the two trailing padding elements belong to row 2 (start 0 -> [0, 1]).
    np.testing.assert_array_equal(positions, [[0, 4, 5, 6, 0, 1]])

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

  def test_batches_metadata_transfer(self):
    runner = _runner(
        _NextTokenTransformer(), return_logits=True, return_logprobs=True
    )
    with mock.patch.object(
        runner, '_to_device', wraps=runner._to_device
    ) as mock_to_device:
      tokens, logits, logp, _ = self._execute(runner)

    # Both `metadata` and `sampling_metadata` are transferred in one call.
    mock_to_device.assert_called_once()
    self.assertEqual(tokens.shape, (3, 1))
    self.assertEqual(logits.shape, (3, 1, _VOCAB_SIZE))
    self.assertEqual(logp.shape, (3, 1))

  def test_sampling_params_must_match_the_scheduled_requests(self):
    runner = _runner(_NextTokenTransformer())

    with self.assertRaisesRegex(ValueError, 'sampling params'):
      self._execute(runner, (_params(),))

  def test_non_positive_num_scheduler_steps_raises(self):
    with self.assertRaisesRegex(ValueError, 'num_scheduler_steps'):
      _runner(_NextTokenTransformer(), num_scheduler_steps=0)


class ExecuteStepSeedTest(absltest.TestCase):
  """Checks how sampling depends on the engine seed."""

  _NUM_ROWS = 64

  def _execute(self, runner, sampling_params=None):
    # Every row decodes one token from a uniform distribution.
    num_rows = len(sampling_params) if sampling_params else self._NUM_ROWS
    metadata = _rpa_metadata(
        kv_lens=[1] * num_rows,
        query_lens=[1] * num_rows,
        distribution=[num_rows] * 3,
    )
    tokens, _, _, _ = runner.execute_step(
        cache={},
        tokens=jnp.zeros((1, num_rows), dtype=jnp.int32),
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


def _fake_step_fn(
    graphdef,
    compile_key,
    params,
    cache,
    tokens,
    metadata,
    sampling_metadata,
    rng,
):
  """A stand-in for `_model_step_impl` that always samples token 1."""
  del graphdef, compile_key, params, tokens, sampling_metadata
  rng, _ = jax.random.split(rng)
  num_rows = len(metadata.query_lens)
  return jnp.ones((num_rows,), dtype=jnp.int32), None, None, cache, rng


class SplitRowsTest(absltest.TestCase):
  """Tests sampling each device's share of the rows."""

  def setUp(self):
    super().setUp()
    if len(jax.devices()) < 4:
      self.skipTest('needs 4 devices')

  def _mesh_2x2(self) -> jax.sharding.Mesh:
    return jax.sharding.Mesh(
        np.array(jax.devices()[:4]).reshape(2, 2), ('dp', 'tp')
    )

  def test_splits_rows_across_the_dp_axes(self):
    mesh = self._mesh_2x2()

    split = jax.jit(lambda x: model_runner_lib._split_rows(x, mesh))(
        jnp.zeros((8, 3))
    )

    # The "tp" devices of a DP group share its rows.
    self.assertTrue(
        split.sharding.is_equivalent_to(
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('dp')),
            2,
        )
    )

  def test_rows_that_do_not_divide_stay_whole(self):
    mesh = self._mesh_2x2()
    x = jax.device_put(
        jnp.zeros((3, 3)),
        jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()),
    )

    out = jax.jit(lambda t: model_runner_lib._split_rows(t, mesh))(x)

    self.assertTrue(out.sharding.is_fully_replicated)

  def test_samples_every_row_on_its_dp_group(self):
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:4]), ('dp',))
    runner = _runner(_NextTokenTransformer(), mesh=mesh, num_scheduler_steps=2)

    # Four decodes, one row per DP group.
    tokens, _, _, _ = runner.execute_step(
        cache=_positions_cache(4),
        tokens=jnp.array([[4], [7], [1], [2]], dtype=jnp.int32),
        metadata=paged_attention.PagedAttnMetadata(
            page_indices={
                'layer_0': np.arange(8, dtype=np.int32).reshape(4, 2)
            },
            kv_lens=np.array([6, 3, 5, 2], dtype=np.int32),
            query_lens=np.array([1, 1, 1, 1], dtype=np.int32),
            distribution=np.array([[1, 1, 1]] * 4, dtype=np.int32),
        ),
        sampling_params=(_params(temperature=0.0),) * 4,
    )

    np.testing.assert_array_equal(tokens, [[5, 6], [8, 9], [2, 3], [3, 4]])
    self.assertTrue(
        tokens.sharding.is_equivalent_to(
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('dp')),
            2,
        )
    )

  def test_multi_group_mixed_rows_pack_and_compact_within_each_group(self):
    mesh = self._mesh_2x2()
    runner = _runner(
        _NextTokenTransformer(),
        mesh=mesh,
        num_scheduler_steps=2,
        return_logprobs=True,
    )

    # Group 0: [decode (4), chunked prefill (1, 2), full prefill (7, 8), pad].
    # Group 1: [decode (2), full prefill (10, 11), pad, pad].
    tokens, _, logp, cache = runner.execute_step(
        cache=_positions_cache(12),
        tokens=jnp.array(
            [[4, 1, 2, 7, 8, 0], [2, 10, 11, 0, 0, 0]], dtype=jnp.int32
        ),
        metadata=paged_attention.PagedAttnMetadata(
            page_indices={
                'layer_0': np.arange(16, dtype=np.int32).reshape(8, 2)
            },
            kv_lens=np.array([6, 2, 2, 0, 4, 2, 0, 0], dtype=np.int32),
            query_lens=np.array([1, 2, 2, 0, 1, 2, 0, 0], dtype=np.int32),
            distribution=np.array([[1, 2, 3], [1, 1, 2]], dtype=np.int32),
        ),
        sampling_params=(
            _params(temperature=1.0, top_k=1),
            _params(temperature=2.0, top_k=1),
            _params(temperature=4.0, top_k=1),
            _params(temperature=5.0, top_k=1),
            _params(temperature=10.0, top_k=1),
        ),
    )

    # Surviving rows (0, 2 in group 0; 4, 5 in group 1) return to their
    # scheduled row positions after compacting within each group.
    np.testing.assert_array_equal(tokens[0], [5, 6])
    np.testing.assert_array_equal(tokens[2], [9, 10])
    np.testing.assert_array_equal(tokens[4], [3, 4])
    np.testing.assert_array_equal(tokens[5], [12, 13])
    # Decode step 2 positions: group 0 survivors at [6, 2], group 1 at [4, 2].
    np.testing.assert_array_equal(cache['positions'][:2], [6, 2])
    np.testing.assert_array_equal(cache['positions'][4:6], [4, 2])
    # Each surviving row keeps its own temperature across both steps.
    assert logp is not None
    for row, temp in ((0, 1.0), (2, 4.0), (4, 5.0), (5, 10.0)):
      expected = float(
          jax.nn.log_softmax(jnp.array([10.0, 5.0] + [0.0] * 14) / temp)[0]
      )
      np.testing.assert_allclose(
          logp[row], [expected, expected], rtol=1e-5, atol=1e-6
      )
    self.assertTrue(
        tokens.sharding.is_equivalent_to(
            jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('dp')),
            2,
        )
    )


class MultiStepExecuteTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # A decode continuing from 4, a chunked prefill ending in 4, a full
    # prefill ending in 9, and a padding row.
    self.metadata = _rpa_metadata(
        kv_lens=[6, 4, 3, 0], query_lens=[1, 4, 3, 0], distribution=[1, 2, 3]
    )
    self.tokens = jnp.array([[4, 1, 2, 3, 4, 7, 8, 9]], dtype=jnp.int32)
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
    jax.clear_caches()
    with mock.patch.object(
        model_runner_lib,
        '_run_decode_scan',
        wraps=model_runner_lib._run_decode_scan,
    ) as decode_loop_traces:
      tokens, _, _, cache = self._execute(runner)
      runner.execute_step(
          cache=_positions_cache(8),
          tokens=self.tokens,
          metadata=self.metadata,
          sampling_params=self.greedy,
      )

    # `_jit_prefill_and_decode_loop` compiles once for a given layout/bucket.
    decode_loop_traces.assert_called_once()
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

    # The decode loop is a compiled `scan`, so its metadata is only concrete
    # with tracing turned off.
    with (
        mock.patch.object(model_runner_lib, '_model_step_impl', mock_step),
        jax.disable_jit(),
    ):
      self._execute(runner)

    step_metadata = [call.args[5] for call in mock_step.call_args_list]
    self.assertLen(step_metadata, 3)
    # The scheduled batch runs as-is.
    np.testing.assert_array_equal(step_metadata[0].query_lens, [1, 4, 3, 0])
    np.testing.assert_array_equal(step_metadata[0].distribution, [[1, 2, 3]])
    self.assertIs(step_metadata[0].decode_only, False)
    # The chunked prefill drops out and the rest decode one token per step.
    for metadata in step_metadata[1:]:
      np.testing.assert_array_equal(metadata.query_lens, [1, 1, 0, 0])
      np.testing.assert_array_equal(metadata.distribution, [[2, 2, 2]])
      self.assertIs(metadata.decode_only, True)
      self.assertEqual(metadata.query_lens.dtype, jnp.int32)
      self.assertEqual(metadata.kv_lens.dtype, jnp.int32)
      self.assertEqual(metadata.distribution.dtype, jnp.int32)
      self.assertEqual(metadata.page_indices['layer_0'].dtype, jnp.int32)
    np.testing.assert_array_equal(step_metadata[1].kv_lens, [7, 4, 0, 0])
    np.testing.assert_array_equal(step_metadata[2].kv_lens, [8, 5, 0, 0])
    # The full prefill's pages move up with it into the second row.
    np.testing.assert_array_equal(
        step_metadata[1].page_indices['layer_0'][:2], [[0, 1], [4, 5]]
    )
    step_rngs = [call.args[7] for call in mock_step.call_args_list]
    self.assertFalse(np.array_equal(step_rngs[0], step_rngs[1]))
    self.assertFalse(np.array_equal(step_rngs[1], step_rngs[2]))

  def test_sampling_settings_follow_their_rows(self):
    runner = _runner(_NextTokenTransformer(), num_scheduler_steps=2)
    mock_step = mock.Mock(wraps=_fake_step_fn)

    with (
        mock.patch.object(model_runner_lib, '_model_step_impl', mock_step),
        jax.disable_jit(),
    ):
      runner.execute_step(
          cache=_positions_cache(8),
          tokens=self.tokens,
          metadata=self.metadata,
          sampling_params=(
              _params(temperature=0.5, top_k=2),
              _params(temperature=0.9),
              _params(temperature=0.7, top_p=0.8),
          ),
      )

    decode_sampling_metadata = mock_step.call_args_list[1].args[6]
    # The full prefill's settings move up with it, and the vacated rows sample
    # greedily like any other padding row.
    np.testing.assert_allclose(
        decode_sampling_metadata.temperature, [0.5, 0.7, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        decode_sampling_metadata.top_p, [1.0, 0.8, 1.0, 1.0]
    )
    np.testing.assert_array_equal(decode_sampling_metadata.top_k, [2, 8, 1, 1])
    self.assertEqual(decode_sampling_metadata.temperature.dtype, jnp.float32)
    self.assertEqual(decode_sampling_metadata.top_p.dtype, jnp.float32)
    self.assertEqual(decode_sampling_metadata.top_k.dtype, jnp.int32)

  def test_decode_loop_preserves_unconstrained(self):
    runner = _runner(
        _NextTokenTransformer(), max_top_k=-1, num_scheduler_steps=2
    )
    mock_step = mock.Mock(wraps=_fake_step_fn)
    with mock.patch.object(model_runner_lib, '_model_step_impl', mock_step):
      for sampling_params, expected in (
          (
              (_params(temperature=0.5), _params(), _params(temperature=0.0)),
              True,
          ),
          ((_params(temperature=0.5, top_p=0.8), _params(), _params()), False),
      ):
        mock_step.reset_mock()
        with jax.disable_jit():
          runner.execute_step(
              cache=_positions_cache(8),
              tokens=self.tokens,
              metadata=self.metadata,
              sampling_params=sampling_params,
          )
        self.assertEqual(
            [c.args[6].unconstrained for c in mock_step.call_args_list],
            [expected, expected],
        )

  def test_stops_when_every_row_drops_out(self):
    # A batch of nothing but chunked prefills has no row left to decode, so
    # only the scheduled step runs.
    runner = _runner(_NextTokenTransformer(), num_scheduler_steps=4)
    with mock.patch.object(
        model_runner_lib, '_jit_prefill_and_decode_loop'
    ) as mock_loop:
      tokens, _, _, _ = runner.execute_step(
          cache=_positions_cache(8),
          tokens=self.tokens,
          metadata=_rpa_metadata(
              kv_lens=[4, 4, 0], query_lens=[4, 4, 0], distribution=[0, 2, 2]
          ),
          sampling_params=self.greedy[:2],
      )

    mock_loop.assert_not_called()
    self.assertEqual(tokens.shape, (3, 4))
    np.testing.assert_array_equal(tokens[:, 1:], 0)

  def test_max_steps_override_runs_single_step_and_repeats_first_token(self):
    runner = _runner(_NextTokenTransformer(), num_scheduler_steps=4)
    with mock.patch.object(
        model_runner_lib, '_jit_prefill_and_decode_loop'
    ) as mock_loop:
      tokens, _, _, _ = runner.execute_step(
          cache=_positions_cache(8),
          tokens=self.tokens,
          metadata=self.metadata,
          sampling_params=self.greedy,
          max_steps=1,
      )
    mock_loop.assert_not_called()
    self.assertEqual(tokens.shape, (4, 4))
    # Column 0 and column -1 both hold the single sampled token for pipelined
    # scatter_last_column reads on surviving rows.
    np.testing.assert_array_equal(tokens[0], [5, 5, 5, 5])
    np.testing.assert_array_equal(tokens[2], [10, 10, 10, 10])
    with self.assertRaisesRegex(ValueError, 'max_steps'):
      runner.execute_step(
          cache=_positions_cache(8),
          tokens=self.tokens,
          metadata=self.metadata,
          sampling_params=self.greedy,
          max_steps=2,
      )

  def test_p10_1_recreated_model_runner_reuses_compiled_cache(self):
    jax.clear_caches()

    pure_tokens = np.array([[4, 9, 0, 0]], dtype=np.int32)
    pure_meta = dataclasses.replace(
        _rpa_metadata(
            kv_lens=[6, 3, 0, 0],
            query_lens=[1, 1, 0, 0],
            distribution=[2, 2, 2],
        ),
        decode_only=True,
    )

    runner_a = _runner(_NextTokenTransformer(), num_scheduler_steps=3, seed=7)
    mixed_tok_a, _, _, _ = self._execute(runner_a)
    pure_tok_a, _, _, _ = runner_a.execute_step(
        cache=_positions_cache(8),
        tokens=pure_tokens,
        metadata=pure_meta,
        sampling_params=self.greedy[:2],
    )

    # A recreated ModelRunner (e.g., new rollout cycle with different seed)
    # must hit the module-level JAX JIT caches with zero retracing.
    runner_b = _runner(_NextTokenTransformer(), num_scheduler_steps=3, seed=99)
    with (
        mock.patch.object(
            model_runner_lib,
            '_model_step_impl',
            wraps=model_runner_lib._model_step_impl,
        ) as step_traces,
        mock.patch.object(
            model_runner_lib,
            '_run_decode_scan',
            wraps=model_runner_lib._run_decode_scan,
        ) as scan_traces,
    ):
      mixed_tok_b, _, _, _ = self._execute(runner_b)
      pure_tok_b, _, _, _ = runner_b.execute_step(
          cache=_positions_cache(8),
          tokens=pure_tokens,
          metadata=pure_meta,
          sampling_params=self.greedy[:2],
      )

    step_traces.assert_not_called()
    scan_traces.assert_not_called()
    np.testing.assert_array_equal(mixed_tok_b, mixed_tok_a)
    np.testing.assert_array_equal(pure_tok_b, pure_tok_a)

  def test_p10_2_pure_decode_multi_step_single_scan_parity(self):
    pure_meta = dataclasses.replace(
        _rpa_metadata(
            kv_lens=[6, 3, 0, 0],
            query_lens=[1, 1, 0, 0],
            distribution=[2, 2, 2],
        ),
        decode_only=True,
    )
    pure_tokens = np.array([[0, 9, 0, 0]], dtype=np.int32)
    prev_generated = jnp.array(
        [[3, 4], [0, 0], [0, 0], [0, 0]], dtype=jnp.int32
    )
    target_idxs = np.array([0, 4, 4, 4], dtype=np.int32)
    sampling = (_params(temperature=0.7, top_k=4), _params(temperature=0.0))

    runner_fused = _runner(
        _NextTokenTransformer(),
        num_scheduler_steps=4,
        return_logits=True,
        return_logprobs=True,
        seed=17,
    )
    with (
        mock.patch.object(
            model_runner_lib,
            '_jit_packed_model_step',
            wraps=model_runner_lib._jit_packed_model_step,
        ) as mock_step0,
        mock.patch.object(
            model_runner_lib,
            '_jit_prefill_and_decode_loop',
            wraps=model_runner_lib._jit_prefill_and_decode_loop,
        ) as mock_mixed_loop,
        mock.patch.object(
            model_runner_lib,
            '_jit_decode_only_loop',
            wraps=model_runner_lib._jit_decode_only_loop,
        ) as mock_pure_loop,
    ):
      toks_f, logits_f, logp_f, cache_f = runner_fused.execute_step(
          cache=_positions_cache(8),
          tokens=pure_tokens,
          metadata=pure_meta,
          sampling_params=sampling,
          prev_generated_tokens=prev_generated,
          target_idxs=target_idxs,
      )

    mock_step0.assert_not_called()
    mock_mixed_loop.assert_not_called()
    mock_pure_loop.assert_called_once()

    runner_ref = _runner(
        _NextTokenTransformer(),
        num_scheduler_steps=4,
        return_logits=True,
        return_logprobs=True,
        seed=17,
    )
    toks_r, logits_r, logp_r, cache_r = runner_ref.execute_step(
        cache=_positions_cache(8),
        tokens=pure_tokens,
        metadata=dataclasses.replace(pure_meta, decode_only=False),
        sampling_params=sampling,
        prev_generated_tokens=prev_generated,
        target_idxs=target_idxs,
    )

    assert logits_f is not None and logits_r is not None
    assert logp_f is not None and logp_r is not None
    np.testing.assert_array_equal(toks_f, toks_r)
    np.testing.assert_allclose(logits_f, logits_r, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(logp_f, logp_r, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(cache_f['positions'], cache_r['positions'])


class DistributedTopKAndPackingTest(absltest.TestCase):
  """Verifies exact two-stage distributed top-k and packed metadata scatter."""

  def setUp(self):
    super().setUp()
    if len(jax.devices()) < 4:
      self.skipTest('needs 4 devices')

  def test_distributed_top_k_matches_jax_top_k_including_ties(self):
    mesh_1x4 = jax.sharding.Mesh(
        np.array(jax.devices()[:4]).reshape(1, 4), ('fsdp', 'tp')
    )
    mesh_2x2 = jax.sharding.Mesh(
        np.array(jax.devices()[:4]).reshape(2, 2), ('dp', 'tp')
    )
    rng = np.random.default_rng(42)
    # Mix distinct logits, cross-shard exact ties, and -inf entries.
    base = rng.normal(size=(4, 64)).astype(np.float32)
    base[0, [3, 19, 35, 51]] = 9.5  # exact 4-way tie across all 4 TP ranks
    base[1, :] = 0.0  # all-equal row
    base[2, 16:48] = -np.inf  # -inf shards
    logits = jnp.asarray(base)

    for mesh, k in ((mesh_1x4, 8), (mesh_1x4, 20), (mesh_2x2, 6)):
      expected_vals, expected_idxs = jax.lax.top_k(logits, k)
      got_vals, got_idxs = jax.jit(
          lambda x, k=k, m=mesh: model_runner_lib._distributed_top_k(
              x, k, mesh=m
          )
      )(logits)
      np.testing.assert_array_equal(got_vals, expected_vals)
      np.testing.assert_array_equal(got_idxs, expected_idxs)

  def test_blocked_local_and_distributed_top_k_bit_exact_with_ties(self):
    mesh_1x4 = jax.sharding.Mesh(
        np.array(jax.devices()[:4]).reshape(1, 4), ('fsdp', 'tp')
    )
    rng = np.random.default_rng(99)
    # 32768 vocab -> 8192 per TP shard (2 blocks of 4096) and 8 blocks locally.
    for vocab_size, dtype in ((8192, np.float32), (32768, jnp.bfloat16)):
      raw = rng.normal(size=(3, vocab_size)).astype(np.float32)
      # Equal-logit ties spanning block boundaries (4095/4096) and TP shards.
      tie_cols = [17, 4095, 4096, 8191, vocab_size - 4096, vocab_size - 1]
      raw[0, tie_cols] = 11.25
      raw[1, :] = -2.0
      raw[2, 4096:8192] = -np.inf
      logits = jnp.asarray(raw, dtype=dtype)

      for k in (1, 16, 64):
        exp_vals, exp_idxs = jax.lax.top_k(logits, k)
        blk_vals, blk_idxs = jax.jit(
            lambda x, k=k: model_runner_lib._blocked_local_top_k(x, k)
        )(logits)
        np.testing.assert_array_equal(blk_vals, exp_vals)
        np.testing.assert_array_equal(blk_idxs, exp_idxs)

        if vocab_size == 32768:
          sharded = jax.device_put(
              logits,
              jax.sharding.NamedSharding(
                  mesh_1x4, jax.sharding.PartitionSpec(None, 'tp')
              ),
          )
          dist_vals, dist_idxs = jax.jit(
              lambda x, k=k: model_runner_lib._distributed_top_k(
                  x, k, mesh=mesh_1x4
              )
          )(sharded)
          np.testing.assert_array_equal(dist_vals, exp_vals)
          np.testing.assert_array_equal(dist_idxs, exp_idxs)

  def test_sample_with_tp_mesh_matches_reference(self):
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()[:4]).reshape(1, 4), ('fsdp', 'tp')
    )
    rng = np.random.default_rng(7)
    logits = jnp.asarray(rng.normal(size=(4, 64)).astype(np.float32))
    key = jax.random.PRNGKey(123)
    meta = model_runner_lib.SamplingMetadata(
        temperature=jnp.array([0.0, 0.7, 1.0, 0.5], dtype=jnp.float32),
        top_p=jnp.array([1.0, 0.9, 0.95, 1.0], dtype=jnp.float32),
        top_k=jnp.array([1, 8, 12, 4], dtype=jnp.int32),
    )
    ref_toks, ref_logp = jax.jit(
        lambda x: model_runner_lib.sample(
            x, key, meta, max_top_k=12, return_logprobs=True, mesh=None
        )
    )(logits)
    sharded_logits = jax.device_put(
        logits,
        jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(None, 'tp')
        ),
    )
    got_toks, got_logp = jax.jit(
        lambda x: model_runner_lib.sample(
            x, key, meta, max_top_k=12, return_logprobs=True, mesh=mesh
        )
    )(sharded_logits)
    np.testing.assert_array_equal(got_toks, ref_toks)
    np.testing.assert_allclose(got_logp, ref_logp, rtol=1e-5, atol=1e-5)

  def test_pack_paged_metadata_deduplicates_and_fuses_scatter(self):
    local_tbl = np.arange(8, dtype=np.int32).reshape(4, 2)
    global_tbl = np.arange(8, 16, dtype=np.int32).reshape(4, 2)
    page_indices: dict[str, jax.Array | np.ndarray] = {}
    for i in range(5):
      page_indices[f'local_{i}'] = local_tbl
    for i in range(2):
      page_indices[f'global_{i}'] = global_tbl
    metadata = paged_attention.PagedAttnMetadata(
        page_indices=page_indices,
        kv_lens=np.array([6, 4, 3, 0], dtype=np.int32),
        query_lens=np.array([1, 4, 3, 0], dtype=np.int32),
        distribution=np.array([[1, 2, 3]], dtype=np.int32),
    )
    packed = model_runner_lib._pack_paged_metadata(metadata)
    self.assertEqual(packed.packed_page_indices.shape, (2, 4, 2))
    unpacked = packed.unpack()
    self.assertEqual(
        set(unpacked.page_indices.keys()), set(page_indices.keys())
    )
    np.testing.assert_array_equal(unpacked.page_indices['local_3'], local_tbl)
    np.testing.assert_array_equal(unpacked.page_indices['global_1'], global_tbl)

    runner = _runner(_NextTokenTransformer(), num_scheduler_steps=2)
    prev_generated = jnp.array(
        [[3, 4], [0, 0], [0, 0], [0, 0]], dtype=jnp.int32
    )
    target_idxs = np.array([0, 8, 8, 8], dtype=np.int32)
    host_tokens = np.array([[0, 1, 2, 3, 4, 7, 8, 9]], dtype=np.int32)
    with mock.patch.object(
        runner, '_to_device', wraps=runner._to_device
    ) as mock_to_device:
      tokens, _, _, _ = runner.execute_step(
          cache=_positions_cache(8),
          tokens=host_tokens,
          metadata=metadata,
          sampling_params=(_params(temperature=0.0),) * 3,
          prev_generated_tokens=prev_generated,
          target_idxs=target_idxs,
      )
    mock_to_device.assert_called_once()
    packed_arg = mock_to_device.call_args.args[0]
    self.assertIsInstance(packed_arg, np.ndarray)
    self.assertEqual(packed_arg.ndim, 1)
    self.assertEqual(packed_arg.dtype, np.int32)
    # Row 0 received scattered token `4` -> predicts `5`, then `6`.
    np.testing.assert_array_equal(tokens[0], [5, 6])
    np.testing.assert_array_equal(tokens[2], [10, 11])

  def test_p13_1_single_h2d_and_fused_prefill_decode_sampling_parity(self):
    metadata = _rpa_metadata(
        kv_lens=[6, 4, 3, 0],
        query_lens=[1, 4, 3, 0],
        distribution=[1, 2, 3],
    )
    host_tokens = np.array([[0, 1, 2, 3, 4, 7, 8, 9]], dtype=np.int32)
    prev_generated = jnp.array(
        [[3, 4], [0, 0], [0, 0], [0, 0]], dtype=jnp.int32
    )
    target_idxs = np.array([0, 8, 8, 8], dtype=np.int32)
    sampling = (
        _params(temperature=0.6, top_p=0.9, top_k=4),
        _params(temperature=0.8),
        _params(temperature=0.4, top_p=0.75, top_k=3),
    )

    runner = _runner(
        _NextTokenTransformer(),
        num_scheduler_steps=3,
        return_logits=True,
        return_logprobs=True,
        seed=41,
    )
    with (
        mock.patch.object(
            runner, '_to_device', wraps=runner._to_device
        ) as mock_h2d,
        mock.patch.object(
            model_runner_lib,
            '_jit_packed_model_step',
            wraps=model_runner_lib._jit_packed_model_step,
        ) as mock_step0,
        mock.patch.object(
            model_runner_lib,
            '_jit_prefill_and_decode_loop',
            wraps=model_runner_lib._jit_prefill_and_decode_loop,
        ) as mock_fused,
    ):
      toks, logits, logp, cache = runner.execute_step(
          cache=_positions_cache(8),
          tokens=host_tokens,
          metadata=metadata,
          sampling_params=sampling,
          prev_generated_tokens=prev_generated,
          target_idxs=target_idxs,
      )

    # One single contiguous 1D int32 H2D transfer + one single fused JIT call.
    mock_h2d.assert_called_once()
    self.assertIsInstance(mock_h2d.call_args.args[0], np.ndarray)
    self.assertEqual(mock_h2d.call_args.args[0].ndim, 1)
    self.assertEqual(mock_h2d.call_args.args[0].dtype, np.int32)
    mock_step0.assert_not_called()
    mock_fused.assert_called_once()

    # Verify JAX-array input packing matches NumPy host packing bit-for-bit.
    runner_jax = _runner(
        _NextTokenTransformer(),
        num_scheduler_steps=3,
        return_logits=True,
        return_logprobs=True,
        seed=41,
    )
    toks_j, logits_j, logp_j, cache_j = runner_jax.execute_step(
        cache=_positions_cache(8),
        tokens=jnp.asarray(host_tokens),
        metadata=metadata,
        sampling_params=sampling,
        prev_generated_tokens=prev_generated,
        target_idxs=jnp.asarray(target_idxs),
    )
    assert logits is not None and logits_j is not None
    assert logp is not None and logp_j is not None
    np.testing.assert_array_equal(toks, toks_j)
    np.testing.assert_allclose(logits, logits_j, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(logp, logp_j, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(cache['positions'], cache_j['positions'])

  def test_p13_4_topk_uses_direct_lax_topk_when_tpu_custom_call_profitable(
      self,
  ):
    self.assertTrue(
        model_runner_lib._tpu_topk_custom_call_profitable(32, 65536, 64)
    )
    self.assertTrue(
        model_runner_lib._tpu_topk_custom_call_profitable(512, 65536, 128)
    )
    self.assertFalse(
        model_runner_lib._tpu_topk_custom_call_profitable(4, 8192, 2048)
    )
    # Direct path lowers as a single rank-2 top_k over 65536 (not 16x4096).
    x = jnp.zeros((32, 65536), dtype=jnp.bfloat16)
    hlo = (
        jax.jit(lambda a: model_runner_lib._blocked_local_top_k(a, 64))
        .lower(x)
        .as_text()
    )
    self.assertIn('32x65536', hlo)
    self.assertNotIn('512x4096', hlo)


class CrossMeshAbstractReuseTest(absltest.TestCase):
  """Tests AbstractMesh JIT trace sharing and cold-start warmup serialization."""

  def setUp(self):
    super().setUp()
    if len(jax.devices()) < 4:
      self.skipTest('needs 4 devices')

  def _disjoint_meshes(
      self,
  ) -> tuple[jax.sharding.Mesh, jax.sharding.Mesh]:
    devices = jax.devices()
    mesh_a = jax.sharding.Mesh(np.array(devices[:2]), ('tp',))
    mesh_b = jax.sharding.Mesh(np.array(devices[2:4]), ('tp',))
    return mesh_a, mesh_b

  def test_disjoint_identical_meshes_share_trace_and_match_outputs(self):
    trace_counter = [0]

    class _CountingTransformer(_NextTokenTransformer):

      def __call__(
          self, tokens, positions, cache, metadata, mesh, logits_idxs=None
      ):
        trace_counter[0] += 1
        return super().__call__(
            tokens, positions, cache, metadata, mesh, logits_idxs=logits_idxs
        )

    mesh_a, mesh_b = self._disjoint_meshes()
    runner_a = _runner(
        _CountingTransformer(),
        mesh=mesh_a,
        num_scheduler_steps=2,
        return_logprobs=True,
        seed=17,
    )
    runner_b = _runner(
        _CountingTransformer(),
        mesh=mesh_b,
        num_scheduler_steps=2,
        return_logprobs=True,
        seed=17,
    )

    self.assertIsInstance(runner_a._compile_key.mesh, jax.sharding.AbstractMesh)
    self.assertEqual(runner_a._compile_key, runner_b._compile_key)
    self.assertEqual(runner_a._static_graphdef, runner_b._static_graphdef)

    # 1. Pure decode-only loop shares one Python trace across disjoint meshes.
    decode_meta = _rpa_metadata(
        kv_lens=[4, 6, 3],
        query_lens=[1, 1, 1],
        distribution=[3, 3, 3],
        decode_only=True,
    )
    decode_tokens = np.array([[2, 5, 9]], dtype=np.int32)
    sampling = (
        _params(temperature=0.0),
        _params(temperature=0.0),
        _params(temperature=0.0),
    )
    shd_a = jax.sharding.NamedSharding(mesh_a, jax.sharding.PartitionSpec())
    shd_b = jax.sharding.NamedSharding(mesh_b, jax.sharding.PartitionSpec())
    cache_a = {'positions': jax.device_put(jnp.zeros((3,), jnp.int32), shd_a)}
    cache_b = {'positions': jax.device_put(jnp.zeros((3,), jnp.int32), shd_b)}

    toks_a, _, logp_a, out_cache_a = runner_a.execute_step(
        cache=cache_a,
        tokens=decode_tokens,
        metadata=decode_meta,
        sampling_params=sampling,
    )
    traces_after_a = trace_counter[0]
    self.assertEqual(traces_after_a, 1)

    toks_b, _, logp_b, out_cache_b = runner_b.execute_step(
        cache=cache_b,
        tokens=decode_tokens,
        metadata=decode_meta,
        sampling_params=sampling,
    )
    self.assertEqual(trace_counter[0], traces_after_a)
    np.testing.assert_array_equal(np.asarray(toks_a), np.asarray(toks_b))
    np.testing.assert_allclose(
        np.asarray(logp_a), np.asarray(logp_b), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_array_equal(
        np.asarray(out_cache_a['positions']),
        np.asarray(out_cache_b['positions']),
    )

    # 2. Mixed prefill + decode loop also shares a single trace across meshes.
    mixed_meta = _rpa_metadata(
        kv_lens=[6, 3, 0], query_lens=[1, 3, 0], distribution=[1, 1, 2]
    )
    mixed_tokens = np.array([[4, 7, 8, 9]], dtype=np.int32)
    mixed_sampling = (_params(temperature=0.0), _params(temperature=0.0))
    m_cache_a = {'positions': jax.device_put(jnp.zeros((4,), jnp.int32), shd_a)}
    m_cache_b = {'positions': jax.device_put(jnp.zeros((4,), jnp.int32), shd_b)}

    mtoks_a, _, mlogp_a, _ = runner_a.execute_step(
        cache=m_cache_a,
        tokens=mixed_tokens,
        metadata=mixed_meta,
        sampling_params=mixed_sampling,
    )
    traces_after_mixed_a = trace_counter[0]
    # Prefill+decode invokes transformer twice in a single JIT trace (step 0 +
    # scan body).
    self.assertEqual(traces_after_mixed_a, traces_after_a + 2)

    mtoks_b, _, mlogp_b, _ = runner_b.execute_step(
        cache=m_cache_b,
        tokens=mixed_tokens,
        metadata=mixed_meta,
        sampling_params=mixed_sampling,
    )
    self.assertEqual(trace_counter[0], traces_after_mixed_a)
    np.testing.assert_array_equal(np.asarray(mtoks_a), np.asarray(mtoks_b))
    np.testing.assert_allclose(
        np.asarray(mlogp_a), np.asarray(mlogp_b), rtol=1e-6, atol=1e-6
    )

  def test_concurrent_cold_start_across_disjoint_meshes_traces_once(self):
    trace_lock = threading.Lock()
    trace_counter = [0]

    class _SlowTracingTransformer(_NextTokenTransformer):

      def __call__(
          self, tokens, positions, cache, metadata, mesh, logits_idxs=None
      ):
        time.sleep(0.05)
        with trace_lock:
          trace_counter[0] += 1
        return super().__call__(
            tokens, positions, cache, metadata, mesh, logits_idxs=logits_idxs
        )

    mesh_a, mesh_b = self._disjoint_meshes()
    runner_a = _runner(
        _SlowTracingTransformer(),
        mesh=mesh_a,
        num_scheduler_steps=3,
        max_top_k=7,
        seed=23,
    )
    runner_b = _runner(
        _SlowTracingTransformer(),
        mesh=mesh_b,
        num_scheduler_steps=3,
        max_top_k=7,
        seed=23,
    )
    meta = _rpa_metadata(
        kv_lens=[3, 5, 7, 2, 4],
        query_lens=[1, 1, 1, 1, 1],
        distribution=[5, 5, 5],
        decode_only=True,
    )
    tokens = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
    sampling = (_params(temperature=0.0),) * 5
    shd_a = jax.sharding.NamedSharding(mesh_a, jax.sharding.PartitionSpec())
    shd_b = jax.sharding.NamedSharding(mesh_b, jax.sharding.PartitionSpec())
    cache_a = {'positions': jax.device_put(jnp.zeros((5,), jnp.int32), shd_a)}
    cache_b = {'positions': jax.device_put(jnp.zeros((5,), jnp.int32), shd_b)}
    barrier = threading.Barrier(2)

    def _run(runner, cache):
      barrier.wait()
      out_toks, _, _, _ = runner.execute_step(
          cache=cache,
          tokens=tokens,
          metadata=meta,
          sampling_params=sampling,
      )
      return np.asarray(out_toks)

    with futures.ThreadPoolExecutor(max_workers=2) as pool:
      fut_a = pool.submit(_run, runner_a, cache_a)
      fut_b = pool.submit(_run, runner_b, cache_b)
      res_a = fut_a.result(timeout=30.0)
      res_b = fut_b.result(timeout=30.0)

    # Cold-start guard serializes the initial trace so sibling slice reuses it.
    self.assertEqual(trace_counter[0], 1)
    np.testing.assert_array_equal(res_a, res_b)

  def test_packed_model_step_fuses_padding_and_shares_single_compilation(self):
    trace_counter = [0]

    class _CountingTransformer(_NextTokenTransformer):

      def __call__(
          self, tokens, positions, cache, metadata, mesh, logits_idxs=None
      ):
        trace_counter[0] += 1
        return super().__call__(
            tokens, positions, cache, metadata, mesh, logits_idxs=logits_idxs
        )

    runner = _runner(
        _CountingTransformer(),
        num_scheduler_steps=4,
        return_logits=True,
        return_logprobs=True,
        seed=29,
    )
    # Case 1: chunked-prefill-only (num_surviving == 0) -> repeat_first=False
    # (steps 1..3 zero-padded in tokens, logits, and logprobs).
    chunked_meta = _rpa_metadata(
        kv_lens=[4, 4, 0, 0],
        query_lens=[4, 4, 0, 0],
        distribution=[0, 2, 2],
        decode_only=False,
    )
    tokens = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int32)
    sampling = (_params(temperature=0.0), _params(temperature=0.0))
    with mock.patch.object(
        model_runner_lib,
        '_jit_packed_model_step',
        wraps=model_runner_lib._jit_packed_model_step,
    ) as mock_step0:
      toks_c, logits_c, logp_c, _ = runner.execute_step(
          cache=_positions_cache(8),
          tokens=tokens,
          metadata=chunked_meta,
          sampling_params=sampling,
      )
    mock_step0.assert_called_once()
    self.assertEqual(trace_counter[0], 1)
    self.assertEqual(toks_c.shape, (4, 4))
    assert logits_c is not None and logp_c is not None
    self.assertEqual(logits_c.shape[:2], (4, 4))
    self.assertEqual(logp_c.shape, (4, 4))
    np.testing.assert_array_equal(np.asarray(toks_c)[:, 1:], 0)
    np.testing.assert_array_equal(np.asarray(logits_c)[:, 1:], 0)
    np.testing.assert_array_equal(np.asarray(logp_c)[:, 1:], 0)

    # Case 2: max_steps=1 with surviving sequences (num_surviving > 0) ->
    # repeat_first=True on tokens (so column -1 carries step 0 for next step's
    # scatter_last_column) while logits/logprobs stay zero-padded, AND reuses
    # the exact same compiled XLA trace without retracing!
    surviving_meta = _rpa_metadata(
        kv_lens=[4, 4, 0, 0],
        query_lens=[4, 4, 0, 0],
        distribution=[0, 0, 2],
        decode_only=False,
    )
    toks_s, logits_s, logp_s, _ = runner.execute_step(
        cache=_positions_cache(8),
        tokens=tokens,
        metadata=surviving_meta,
        sampling_params=sampling,
        max_steps=1,
    )
    self.assertEqual(trace_counter[0], 1)
    toks_s_np = np.asarray(toks_s)
    for col in range(1, 4):
      np.testing.assert_array_equal(toks_s_np[:, col], toks_s_np[:, 0])
    assert logits_s is not None and logp_s is not None
    np.testing.assert_array_equal(np.asarray(logits_s)[:, 1:], 0)
    np.testing.assert_array_equal(np.asarray(logp_s)[:, 1:], 0)


if __name__ == '__main__':
  absltest.main()
