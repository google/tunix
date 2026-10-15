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

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax.numpy as jnp
import numpy as np
from tunix.experimental.generate import engine as engine_lib
from tunix.experimental.generate import kv_cache_manager as kv_cache_manager_lib
from tunix.experimental.generate import model_runner as model_runner_lib
from tunix.experimental.generate import scheduler as scheduler_lib
from tunix.experimental.generate import testing_utils
from tunix.experimental.rollout import sampler as sampler_lib
from tunix.generate import tokenizer_adapter


class LLMEngineTest(absltest.TestCase):

  def test_non_positive_max_model_len_raises(self):
    with self.assertRaisesRegex(ValueError, 'max_model_len'):
      testing_utils.make_engine(max_model_len=0)

  def test_mismatched_num_scheduler_steps_raises(self):
    with self.assertRaisesRegex(ValueError, 'num_scheduler_steps'):
      engine_lib.LLMEngine(
          testing_utils.PagedSumTransformer(),
          tokenizer=tokenizer_adapter.TokenizerAdapter(
              testing_utils.WhitespaceTokenizer()
          ),
          cache_config=kv_cache_manager_lib.CacheConfig(
              max_device_size_gib=64 * testing_utils.GIB_PER_PAGE,
              page_size=testing_utils.PAGE_SIZE,
              dtype=jnp.float32,
          ),
          scheduler_config=scheduler_lib.SchedulerConfig(
              max_num_batched_tokens=32,
              max_num_seqs=4,
              max_chunked_prefill_length=32,
              num_scheduler_steps=2,
          ),
          model_runner_config=model_runner_lib.ModelRunnerConfig(
              max_top_k=4, mesh=testing_utils.mesh(), num_scheduler_steps=1
          ),
          max_model_len=24,
      )

  def test_exposes_the_model(self):
    transformer = testing_utils.PagedSumTransformer(offset=3.0)
    engine = testing_utils.make_engine(transformer)

    graphdef, state = engine.model_def_and_state()

    self.assertEqual(graphdef, nnx.graphdef(transformer))
    np.testing.assert_array_equal(
        nnx.state(nnx.merge(graphdef, state))['offset'].value, 3.0
    )


class AddRequestTest(parameterized.TestCase):

  def test_new_request_is_unfinished(self):
    engine = testing_utils.make_engine()
    self.assertFalse(engine.has_unfinished_requests())

    engine.add_request(testing_utils.make_request('0', [1, 2]))

    self.assertTrue(engine.has_unfinished_requests())

  def test_text_prompt_is_accepted(self):
    engine = testing_utils.make_engine()

    engine.add_request(
        sampler_lib.SamplingRequest(
            request_id='0',
            prompt='1 2 3',
            sampling_params=sampler_lib.SamplingParams(max_tokens=4),
        )
    )

    self.assertTrue(engine.has_unfinished_requests())

  @parameterized.named_parameters(
      ('token_list', [1, 2]),
      ('float_array', np.asarray([1.0, 2.0])),
      ('2d_array', np.asarray([[1, 2]], dtype=np.int32)),
      ('chat_messages', [{'role': 'user', 'content': 'hi'}]),
  )
  def test_unsupported_prompt_raises(self, prompt):
    request = sampler_lib.SamplingRequest(
        request_id='0',
        prompt=prompt,
        sampling_params=sampler_lib.SamplingParams(max_tokens=4),
    )

    with self.assertRaisesRegex(TypeError, 'prompt'):
      testing_utils.make_engine().add_request(request)

  def test_empty_prompt_raises(self):
    with self.assertRaisesRegex(ValueError, 'empty prompt'):
      testing_utils.make_engine().add_request(testing_utils.make_request('0', []))

  def test_request_over_max_model_len_raises(self):
    engine = testing_utils.make_engine(max_model_len=10)
    engine.add_request(testing_utils.make_request('fits', [1] * 4, max_tokens=6))

    with self.assertRaisesRegex(ValueError, 'max_model_len'):
      engine.add_request(testing_utils.make_request('0', [1] * 5, max_tokens=6))

  def test_top_k_over_max_top_k_raises(self):
    engine = testing_utils.make_engine(max_top_k=4)
    engine.add_request(testing_utils.make_request('fits', [1], top_k=4))

    with self.assertRaisesRegex(ValueError, 'max_top_k'):
      engine.add_request(testing_utils.make_request('0', [1], top_k=5))

  def test_any_top_k_is_allowed_without_a_max_top_k(self):
    engine = testing_utils.make_engine(max_top_k=-1)

    engine.add_request(testing_utils.make_request('0', [1], top_k=1000))

    self.assertTrue(engine.has_unfinished_requests())

  def test_missing_sampling_params_raises(self):
    request = sampler_lib.SamplingRequest(
        request_id='0', prompt=np.asarray([1], dtype=np.int32)
    )

    with self.assertRaisesRegex(ValueError, 'sampling_params'):
      testing_utils.make_engine().add_request(request)

  @parameterized.named_parameters(
      ('zero_max_tokens', dict(max_tokens=0), 'max_tokens'),
      ('negative_temperature', dict(temperature=-0.1), 'temperature'),
      ('zero_top_p', dict(top_p=0.0), 'top_p'),
      ('top_p_over_one', dict(top_p=1.5), 'top_p'),
      ('zero_top_k', dict(top_k=0), 'top_k'),
      ('negative_seed', dict(seed=-1), 'seed'),
      ('seed_past_uint32', dict(seed=1 << 32), 'seed'),
      ('beam_size', dict(beam_size=2), 'beam_size'),
      ('routed_experts', dict(return_routed_experts=True), 'routed_experts'),
      (
          'routed_experts_prompt_start',
          dict(routed_experts_prompt_start=1),
          'routed_experts_prompt_start',
      ),
      ('logprobs', dict(return_logprobs=True), 'return_logprobs'),
      ('logits', dict(return_logits=True), 'return_logits'),
  )
  def test_unsupported_sampling_params_raise(self, sampling_kwargs, message):
    with self.assertRaisesRegex(ValueError, message):
      testing_utils.make_engine().add_request(testing_utils.make_request('0', [1], **sampling_kwargs))

  def test_request_must_ask_for_the_configured_outputs(self):
    engine = testing_utils.make_engine(return_logprobs=True, return_logits=True)
    engine.add_request(
        testing_utils.make_request('fits', [1], return_logprobs=True, return_logits=True)
    )

    with self.assertRaisesRegex(ValueError, 'return_logprobs'):
      engine.add_request(testing_utils.make_request('0', [1], return_logits=True))

  def test_running_request_id_raises(self):
    engine = testing_utils.make_engine()
    engine.add_request(testing_utils.make_request('0', [1, 2]))

    with self.assertRaisesRegex(ValueError, 'already running'):
      engine.add_request(testing_utils.make_request('0', [3]))

  def test_aborted_request_id_can_be_reused(self):
    engine = testing_utils.make_engine()
    engine.add_request(testing_utils.make_request('0', [1, 2]))
    engine.abort_request('0')

    engine.add_request(testing_utils.make_request('0', [3]))

    self.assertTrue(engine.has_unfinished_requests())


class AbortRequestTest(absltest.TestCase):

  def test_aborting_a_new_request_withdraws_it(self):
    engine = testing_utils.make_engine()
    engine.add_request(testing_utils.make_request('0', [1, 2]))

    engine.abort_request('0')

    self.assertFalse(engine.has_unfinished_requests())

  def test_aborting_leaves_other_requests_unfinished(self):
    engine = testing_utils.make_engine()
    engine.add_request(testing_utils.make_request('0', [1, 2]))
    engine.add_request(testing_utils.make_request('1', [1, 2]))

    engine.abort_request('0')

    self.assertTrue(engine.has_unfinished_requests())

  def test_aborting_a_done_request_does_nothing(self):
    engine = testing_utils.make_engine()
    engine.add_request(testing_utils.make_request('0', [1, 2]))
    engine.abort_request('0')

    engine.abort_request('0')

    self.assertFalse(engine.has_unfinished_requests())

  def test_aborting_an_unknown_request_does_nothing(self):
    engine = testing_utils.make_engine()
    engine.add_request(testing_utils.make_request('0', [1, 2]))

    engine.abort_request('unknown')

    self.assertTrue(engine.has_unfinished_requests())


if __name__ == '__main__':
  absltest.main()
