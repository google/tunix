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
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.experimental.generate import testing_utils
from tunix.experimental.rollout import vanilla_rollout
from tunix.rl.rollout import base_rollout

_EOS = 31
_MAX_MODEL_LEN = 24


class _FakeDevice:

  def __init__(self, stats: dict[str, int] | None):
    self._stats = stats

  def memory_stats(self) -> dict[str, int] | None:
    return self._stats


def _rollout_config(**overrides) -> base_rollout.RolloutConfig:
  return base_rollout.RolloutConfig(
      **{
          'max_tokens_to_generate': 8,
          'temperature': 0.0,
          'max_prompt_length': 8,
          'eos_tokens': [_EOS],
          'rollout_hbm_utilization': 0.5,
          'kv_cache_page_size': testing_utils.PAGE_SIZE,
          'max_num_seqs': 4,
          'max_num_batched_tokens': 32,
          'chunked_prefill_length': 32,
          **overrides,
      }
  )


def _expected_tokens(prompt: list[int], offset: float = 0.0) -> list[int]:
  """Returns the tokens a fresh engine generates for `prompt`."""
  engine = testing_utils.make_engine(
      testing_utils.PagedSumTransformer(offset=offset),
      max_model_len=_MAX_MODEL_LEN,
      eos_token_ids=frozenset([_EOS]),
  )
  engine.add_request(testing_utils.make_request('0', prompt))
  outputs = []
  while engine.has_unfinished_requests():
    outputs.extend(engine.step())
  return outputs[0].token_ids.tolist()


class VanillaRolloutTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    fake_device = _FakeDevice(
        {
            'bytes_limit': 200 * testing_utils.BYTES_PER_PAGE,
            'bytes_in_use': 36 * testing_utils.BYTES_PER_PAGE,
        }
    )
    self._local_devices_mock = self.enterContext(
        mock.patch.object(
            jax.sharding.Mesh,
            'local_devices',
            new_callable=mock.PropertyMock,
            return_value=[fake_device],
        )
    )

  def make_rollout(self, server_mode: bool, **overrides):
    rollout = vanilla_rollout.VanillaRollout(
        testing_utils.PagedSumTransformer(),
        testing_utils.WhitespaceTokenizer(),
        rollout_config=_rollout_config(server_mode=server_mode, **overrides),
        mesh=testing_utils.mesh(),
        max_model_len=_MAX_MODEL_LEN,
    )
    self.addCleanup(rollout.close)
    return rollout

  @parameterized.parameters(False, True)
  def test_generate_from_text(self, server_mode):
    rollout = self.make_rollout(server_mode)

    output = rollout.generate(['1 2 3', '5'], _rollout_config())

    self.assertEqual(
        [tokens.tolist() for tokens in output.tokens],
        [_expected_tokens([1, 2, 3]), _expected_tokens([5])],
    )
    self.assertEqual(
        output.text,
        [' '.join(map(str, tokens)) for tokens in output.tokens],
    )
    np.testing.assert_array_equal(
        output.left_padded_prompt_tokens,
        [[0, 0, 0, 0, 0, 1, 2, 3], [0, 0, 0, 0, 0, 0, 0, 5]],
    )
    np.testing.assert_array_equal(output.prompt_lengths, [3, 1])
    self.assertIsNone(output.logprobs)

  @parameterized.parameters(False, True)
  def test_generate_returns_logprobs_when_asked(self, server_mode):
    rollout = self.make_rollout(server_mode, return_logprobs=True)

    output = rollout.generate(['1 2 3'], _rollout_config(return_logprobs=True))

    self.assertIsNotNone(output.logprobs)
    self.assertEqual(output.logprobs[0].shape, output.tokens[0].shape)

  @parameterized.parameters(False, True)
  def test_update_params_changes_the_model(self, server_mode):
    rollout = self.make_rollout(server_mode)

    rollout.update_params(nnx.variables(testing_utils.PagedSumTransformer(offset=1.0), nnx.Param))
    output = rollout.generate(['1 2 3'], _rollout_config())

    self.assertEqual(
        output.tokens[0].tolist(), _expected_tokens([1, 2, 3], offset=1.0)
    )
    self.assertEqual(rollout.model().offset.value, jnp.float32(1.0))

  @parameterized.parameters(0.0, -0.1, 1.5)
  def test_invalid_rollout_hbm_utilization_raises(self, hbm_utilization):
    with self.assertRaisesRegex(ValueError, 'rollout_hbm_utilization'):
      self.make_rollout(False, rollout_hbm_utilization=hbm_utilization)

  def test_resolve_max_device_size_gib_takes_min_across_devices(self):
    self._local_devices_mock.return_value = [
        _FakeDevice({'bytes_limit': 1000, 'bytes_in_use': 100}),
        _FakeDevice({'bytes_limit': 1000, 'bytes_in_use': 300}),
    ]
    size_gib = vanilla_rollout._resolve_max_device_size_gib(
        0.5, testing_utils.mesh()
    )
    self.assertEqual(size_gib, (500 - 300) / (1 << 30))

  def test_resolve_max_device_size_gib_insufficient_hbm_raises(self):
    self._local_devices_mock.return_value = [
        _FakeDevice({'bytes_limit': 1000, 'bytes_in_use': 600}),
    ]
    with self.assertRaisesRegex(ValueError, 'Insufficient HBM'):
      self.make_rollout(False, rollout_hbm_utilization=0.5)

  def test_resolve_max_device_size_gib_missing_memory_stats_raises(self):
    self._local_devices_mock.return_value = [_FakeDevice(None)]
    with self.assertRaisesRegex(ValueError, 'memory_stats'):
      self.make_rollout(False, rollout_hbm_utilization=0.5)

  @parameterized.parameters(False, True)
  def test_get_perf_metrics_returns_rollout_metrics(self, server_mode):
    rollout = self.make_rollout(server_mode)
    rollout.generate(['1 2 3', '5'], _rollout_config())

    perf_metrics = rollout.get_perf_metrics()
    self.assertIn('rollout/avg_generation_throughput_tok_per_s', perf_metrics)
    self.assertIn('rollout/avg_prefill_throughput_tok_per_s', perf_metrics)
    self.assertIn('rollout/avg_schedule_duration_ms', perf_metrics)
    self.assertIn('rollout/avg_engine_step_duration_ms', perf_metrics)
    self.assertIn('rollout/avg_request_queue_time_s', perf_metrics)
    self.assertIn('rollout/last_batch_completion_time_s', perf_metrics)
    self.assertIn('rollout/avg_batch_completion_time_s', perf_metrics)

  @parameterized.parameters(False, True)
  def test_combines_tokenizer_eos_and_configured_eos_tokens(self, server_mode):
    unbounded = _expected_tokens([1, 2, 3])
    custom_eos = unbounded[1]
    rollout = self.make_rollout(server_mode, eos_tokens=[custom_eos])

    output = rollout.generate(['1 2 3'], _rollout_config(eos_tokens=[custom_eos]))

    self.assertEqual(output.tokens[0].tolist(), unbounded[:2])

  def test_wires_host_size_gib_to_cache_config(self):
    host_size_gib = (16 * testing_utils.BYTES_PER_PAGE) / (1 << 30)
    rollout = self.make_rollout(False, host_size_gib=host_size_gib)
    group_mgr = rollout._sampler._engine._kv_cache_manager._kv_cache_group_managers[0]
    self.assertEqual(group_mgr._page_manager.num_free_host_pages, 16)


if __name__ == '__main__':
  absltest.main()
