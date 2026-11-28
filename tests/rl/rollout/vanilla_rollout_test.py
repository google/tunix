# Copyright 2025 Google LLC
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

"""Tests for the PRNG seeding of VanillaRollout.generate()."""

from unittest import mock

from absl.testing import absltest
import jax
import numpy as np
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout import vanilla_rollout


def _as_key_data(seed):
  """Maps a seed passed to the Sampler to the key the Sampler would use."""
  # Same mapping as Sampler.__call__: None -> PRNGKey(0), int -> PRNGKey(int).
  if seed is None:
    seed = 0
  if isinstance(seed, int):
    seed = jax.random.PRNGKey(seed)
  if jax.dtypes.issubdtype(seed.dtype, jax.dtypes.prng_key):
    seed = jax.random.key_data(seed)
  return np.asarray(seed)


class VanillaRolloutSeedTest(absltest.TestCase):

  def _make_rollout(self):
    """Builds a VanillaRollout whose Sampler is a mock recording its seeds."""
    patcher = mock.patch.object(vanilla_rollout.sampler, 'Sampler')
    sampler_cls = patcher.start()
    self.addCleanup(patcher.stop)
    rollout = vanilla_rollout.VanillaRollout(
        model=mock.Mock(),
        tokenizer=mock.Mock(),
        cache_config_or_size=base_rollout.CacheConfig(
            cache_size=32, num_layers=1, num_kv_heads=1, head_dim=8
        ),
    )
    return rollout, sampler_cls.return_value

  def _seeds_used(self, seed, num_calls):
    rollout, mock_sampler = self._make_rollout()
    config = base_rollout.RolloutConfig(
        max_prompt_length=8, max_tokens_to_generate=8, seed=seed
    )
    for _ in range(num_calls):
      # One prompt per call, as in agentic RL (one call per trajectory turn).
      rollout.generate(['same prompt'], config)
    return [_as_key_data(c.kwargs['seed']) for c in mock_sampler.call_args_list]

  def test_default_seed_differs_across_calls(self):
    # Before the fix, seed=None reached the sampler as None on every call and
    # was turned into PRNGKey(0) each time, so all rollouts were identical.
    seeds = self._seeds_used(seed=None, num_calls=4)
    self.assertLen(seeds, 4)
    self.assertLen({s.tobytes() for s in seeds}, 4)

  def test_int_seed_differs_across_calls(self):
    seeds = self._seeds_used(seed=42, num_calls=3)
    self.assertLen({s.tobytes() for s in seeds}, 3)

  def test_same_seed_is_reproducible(self):
    first = self._seeds_used(seed=7, num_calls=3)
    second = self._seeds_used(seed=7, num_calls=3)
    for a, b in zip(first, second):
      np.testing.assert_array_equal(a, b)

  def test_prng_key_seed_is_accepted(self):
    seeds = self._seeds_used(seed=jax.random.PRNGKey(3), num_calls=2)
    self.assertLen({s.tobytes() for s in seeds}, 2)

  def test_typed_key_seed_is_accepted(self):
    seeds = self._seeds_used(seed=jax.random.key(3), num_calls=2)
    self.assertLen({s.tobytes() for s in seeds}, 2)

  def test_numpy_and_jax_int_seeds_match_python_int(self):
    expected = self._seeds_used(seed=5, num_calls=2)
    for seed in (np.int32(5), np.int64(5), jax.numpy.int32(5)):
      seeds = self._seeds_used(seed=seed, num_calls=2)
      for a, b in zip(expected, seeds):
        np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
  absltest.main()
