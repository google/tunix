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

import itertools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.experimental.generate import engine as engine_lib
from tunix.experimental.generate import kv_cache_manager as kv_cache_manager_lib
from tunix.experimental.generate import model_runner as model_runner_lib
from tunix.experimental.generate import request as request_lib
from tunix.experimental.generate import scheduler as scheduler_lib
from tunix.experimental.generate import testing_utils


class LLMEngineTest(absltest.TestCase):

  def test_non_positive_max_model_len_raises(self):
    with self.assertRaisesRegex(ValueError, 'max_model_len'):
      testing_utils.make_engine(max_model_len=0)

  def test_max_pages_per_seq_over_num_device_pages_raises(self):
    # max_model_len=24 + num_scheduler_steps=1 spans 7 pages of 4 slots.
    with self.assertRaisesRegex(ValueError, 'needs up to 7 pages'):
      testing_utils.make_engine(
          max_model_len=24, max_device_size_gib=6 * testing_utils.GIB_PER_PAGE
      )

  def test_mismatched_num_scheduler_steps_raises(self):
    with self.assertRaisesRegex(ValueError, 'num_scheduler_steps'):
      engine_lib.LLMEngine(
          testing_utils.PagedSumTransformer(),
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

  def test_truncates_chunked_prefill_length_to_sliding_window(self):
    transformer = testing_utils.PagedSumTransformer(
        geometries={
            'layer_0': kv_cache_manager_lib.CacheGeometry(
                num_kv_heads=1, head_dim=1, window_size=12
            )
        }
    )
    engine = testing_utils.make_engine(
        transformer, chunked_prefill_length=32
    )

    self.assertEqual(engine._scheduler_config.max_chunked_prefill_length, 8)
    self.assertEqual(engine._scheduler._config.max_chunked_prefill_length, 8)


class AddRequestTest(parameterized.TestCase):

  def test_new_request_is_unfinished(self):
    engine = testing_utils.make_engine()
    self.assertFalse(engine.has_unfinished_requests())

    engine.add_request(testing_utils.make_request('0', [1, 2]))

    self.assertTrue(engine.has_unfinished_requests())

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

  @parameterized.named_parameters(
      ('zero_max_tokens', dict(max_tokens=0), 'max_tokens'),
      ('negative_temperature', dict(temperature=-0.1), 'temperature'),
      ('zero_top_p', dict(top_p=0.0), 'top_p'),
      ('top_p_over_one', dict(top_p=1.5), 'top_p'),
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


def reference_generate(
    prompt: list[int],
    max_tokens: int = 8,
    offset: float = 0.0,
    eos_token_ids: frozenset[int] = frozenset(),
) -> list[int]:
  """Returns what `PagedSumTransformer` generates greedily for `prompt`."""
  total = sum(token + offset for token in prompt)
  generated = []
  while len(generated) < max_tokens:
    next_token = round(total) % testing_utils.VOCAB_SIZE
    generated.append(next_token)
    if next_token in eos_token_ids:
      break
    total += next_token + offset
  return generated


def drive(
    engine: engine_lib.LLMEngine,
) -> dict[str, request_lib.EngineResponse]:
  """Steps the engine until it runs out of work, returning what finished."""
  finished = {}
  while engine.has_unfinished_requests():
    for output in engine.step():
      finished[output.request_id] = output
  return finished


def spy_on_execute_step(engine: engine_lib.LLMEngine) -> mock.MagicMock:
  model_runner = engine._model_runner  # pylint: disable=protected-access
  spy = mock.patch.object(
      model_runner, 'execute_step', wraps=model_runner.execute_step
  ).start()
  return spy


_PROMPTS = ([5], [1, 2, 3, 4, 5], [9, 8, 7, 6, 5, 4, 3, 2, 1])


class StepTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._request_counter = itertools.count()

  def tearDown(self):
    mock.patch.stopall()
    super().tearDown()

  def _run(self, engine, prompts, **sampling_kwargs) -> list[list[int]]:
    """Runs `prompts` to completion, returning the tokens each generated."""
    request_ids = [str(next(self._request_counter)) for _ in prompts]
    for request_id, prompt in zip(request_ids, prompts):
      engine.add_request(
          testing_utils.make_request(
              request_id, list(prompt), **sampling_kwargs
          )
      )
    finished = drive(engine)
    self.assertCountEqual(finished, request_ids)
    return [finished[request_id].token_ids.tolist() for request_id in request_ids]

  def test_step_without_requests_does_nothing(self):
    self.assertEmpty(testing_utils.make_engine().step())

  def test_generates_the_reference_continuation(self):
    generated = self._run(testing_utils.make_engine(), _PROMPTS)

    self.assertEqual(generated, [reference_generate(p) for p in _PROMPTS])

  def test_output_describes_the_request(self):
    engine = testing_utils.make_engine()
    engine.add_request(
        testing_utils.make_request('text', [1, 2, 3], max_tokens=4)
    )

    output = drive(engine)['text']

    expected = reference_generate([1, 2, 3], 4)
    self.assertEqual(output.request_id, 'text')
    np.testing.assert_array_equal(output.prompt_token_ids, [1, 2, 3])
    np.testing.assert_array_equal(output.token_ids, expected)
    self.assertEqual(output.finish_reason, 'length')
    self.assertIsNone(output.logprobs)
    self.assertIsNone(output.logits)

  def test_each_request_stops_at_its_own_token_limit(self):
    engine = testing_utils.make_engine()
    engine.add_request(testing_utils.make_request('short', [1, 2], max_tokens=2))
    engine.add_request(testing_utils.make_request('long', [1, 2], max_tokens=7))

    finished = drive(engine)

    self.assertEqual(
        finished['short'].token_ids.tolist(), reference_generate([1, 2], 2)
    )
    self.assertEqual(
        finished['long'].token_ids.tolist(), reference_generate([1, 2], 7)
    )

  def test_stops_at_an_eos_token(self):
    prompt = [1, 2, 3]
    unbounded = reference_generate(prompt)
    eos = unbounded[2]
    engine = testing_utils.make_engine(eos_token_ids=frozenset({eos}))
    engine.add_request(testing_utils.make_request('0', prompt))

    output = drive(engine)['0']

    # The EOS token is kept.
    self.assertEqual(
        output.token_ids.tolist(), unbounded[: unbounded.index(eos) + 1]
    )
    self.assertEqual(output.finish_reason, 'stop')

  def test_chunked_prefill(self):
    engine = testing_utils.make_engine(
        max_num_batched_tokens=8, chunked_prefill_length=4
    )
    spy = spy_on_execute_step(engine)
    prompt = list(range(1, 15))

    generated = self._run(engine, [prompt])

    self.assertEqual(generated, [reference_generate(prompt)])
    first_metadata = spy.call_args_list[0].kwargs['metadata']
    self.assertEqual(first_metadata.query_lens[0], 4)

  def test_page_table_leaves_room_for_the_decode_steps(self):
    engine = testing_utils.make_engine(max_model_len=15, num_scheduler_steps=2)
    spy = spy_on_execute_step(engine)

    self._run(engine, [[1, 2, 3]], max_tokens=4)

    # 15 tokens plus 2 decode steps span 5 pages of 4 slots.
    metadata = spy.call_args_list[0].kwargs['metadata']
    self.assertEqual(metadata.page_indices['layer_0'].shape, (4, 5))

  @parameterized.named_parameters(
      ('within_the_window', 16, 16),
      # Truncated to the largest power of 2 within the window.
      ('past_the_window', 12, 8),
  )
  def test_chunk_fits_in_the_smallest_window(self, window_size, expected):
    geometries = {
        'full': kv_cache_manager_lib.CacheGeometry(num_kv_heads=1, head_dim=1),
        'sliding': kv_cache_manager_lib.CacheGeometry(
            num_kv_heads=1, head_dim=1, window_size=window_size
        ),
    }
    engine = testing_utils.make_engine(
        testing_utils.PagedSumTransformer(geometries=geometries),
        chunked_prefill_length=16,
    )
    spy = spy_on_execute_step(engine)
    engine.add_request(testing_utils.make_request('0', [1, 2, 3]))

    engine.step()

    metadata = spy.call_args_list[0].kwargs['metadata']
    self.assertEqual(metadata.chunk_prefill_size, expected)

  @parameterized.parameters(2, 3)
  def test_multi_step_decode(self, num_scheduler_steps):
    engine = testing_utils.make_engine(num_scheduler_steps=num_scheduler_steps)
    spy = spy_on_execute_step(engine)

    generated = self._run(engine, _PROMPTS)

    self.assertEqual(generated, [reference_generate(p) for p in _PROMPTS])
    # One prefill step, then `num_scheduler_steps` tokens per step.
    self.assertLess(spy.call_count, 8)

  def test_preempted_requests_resume(self):
    # Eight pages hold every request's prompt, but not what they generate.
    engine = testing_utils.make_engine(
        max_device_size_gib=8 * testing_utils.GIB_PER_PAGE
    )
    preempt_spy = mock.patch.object(
        engine._scheduler,  # pylint: disable=protected-access
        '_preempt',
        wraps=engine._scheduler._preempt,  # pylint: disable=protected-access
    ).start()
    prompts = [[i, i + 1, i + 2, i + 3, i + 4, i + 5] for i in range(4)]

    generated = self._run(engine, prompts)

    preempt_spy.assert_called()
    self.assertEqual(generated, [reference_generate(p) for p in prompts])

  def test_prefix_cache_hit_skips_the_cached_tokens(self):
    engine = testing_utils.make_engine()
    prompt = list(range(1, 10))
    self._run(engine, [prompt])
    spy = spy_on_execute_step(engine)

    generated = self._run(engine, [prompt])

    self.assertEqual(generated, [reference_generate(prompt)])
    # Two full pages are cached; the rest of the prompt still runs.
    first_metadata = spy.call_args_list[0].kwargs['metadata']
    self.assertEqual(
        first_metadata.query_lens[0], len(prompt) - 2 * testing_utils.PAGE_SIZE
    )

  def test_requests_join_a_running_batch(self):
    engine = testing_utils.make_engine()
    engine.add_request(testing_utils.make_request('first', [1, 2, 3]))
    engine.step()
    engine.step()
    engine.add_request(testing_utils.make_request('second', [4, 5]))

    finished = drive(engine)

    self.assertEqual(
        finished['first'].token_ids.tolist(), reference_generate([1, 2, 3])
    )
    self.assertEqual(
        finished['second'].token_ids.tolist(), reference_generate([4, 5])
    )

  def test_returns_logits_and_logprobs(self):
    engine = testing_utils.make_engine(return_logits=True, return_logprobs=True)
    engine.add_request(
        testing_utils.make_request(
            '0', [1, 2, 3], return_logits=True, return_logprobs=True
        )
    )

    output = drive(engine)['0']

    tokens = output.token_ids
    logits, logprobs = output.logits, output.logprobs
    assert logits is not None and logprobs is not None
    self.assertLen(logits, len(tokens))
    self.assertLen(logprobs, len(tokens))
    np.testing.assert_array_equal(np.argmax(logits, axis=-1), tokens)
    expected_logprob = float(
        jax.nn.log_softmax(10.0 * jax.nn.one_hot(0, testing_utils.VOCAB_SIZE))[
            0
        ]
    )
    np.testing.assert_allclose(
        logprobs,
        np.full((len(tokens),), expected_logprob),
        atol=1e-5,
    )

  def test_aborted_request_stops_generating(self):
    engine = testing_utils.make_engine()
    engine.add_request(testing_utils.make_request('kept', [1, 2, 3]))
    engine.add_request(testing_utils.make_request('aborted', [4, 5, 6]))
    engine.step()

    engine.abort_request('aborted')
    finished = drive(engine)

    self.assertEqual(list(finished), ['kept'])

  def test_aborted_request_is_not_unfinished(self):
    engine = testing_utils.make_engine()
    engine.add_request(testing_utils.make_request('0', [1, 2, 3]))
    engine.step()
    engine.abort_request('0')

    self.assertFalse(engine.has_unfinished_requests())
    self.assertEmpty(engine.step())

  def test_aborting_a_finished_request_does_nothing(self):
    engine = testing_utils.make_engine()
    self._run(engine, [[1, 2]])

    engine.abort_request('0')

    self.assertFalse(engine.has_unfinished_requests())


if __name__ == '__main__':
  absltest.main()
