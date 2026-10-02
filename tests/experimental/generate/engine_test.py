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
from tunix.experimental.generate import engine as engine_lib
from tunix.experimental.generate import kv_cache_manager as kv_cache_manager_lib
from tunix.experimental.generate import model_runner as model_runner_lib
from tunix.experimental.generate import request as request_lib
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
              max_device_bytes=64 * testing_utils.BYTES_PER_PAGE,
              page_size=testing_utils.PAGE_SIZE,
              dtype=jnp.float32,
          ),
          scheduler_config=scheduler_lib.SchedulerConfig(
              max_num_batched_tokens=32,
              max_num_seqs=4,
              chunked_prefill_length=32,
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
) -> dict[str, request_lib.RequestOutput]:
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

  def tearDown(self):
    mock.patch.stopall()
    super().tearDown()

  def _run(self, engine, prompts, **sampling_kwargs) -> list[list[int]]:
    """Runs `prompts` to completion, returning the tokens each generated."""
    request_ids = [str(i) for i in range(len(prompts))]
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
        sampler_lib.SamplingRequest(
            request_id='text',
            prompt='1 2 3',
            sampling_params=sampler_lib.SamplingParams(
                max_tokens=4, temperature=0.0
            ),
        )
    )

    output = drive(engine)['text']

    expected = reference_generate([1, 2, 3], 4)
    self.assertEqual(output.request_id, 'text')
    np.testing.assert_array_equal(output.prompt_token_ids, [1, 2, 3])
    np.testing.assert_array_equal(output.token_ids, expected)
    self.assertEqual(output.text, ' '.join(str(t) for t in expected))
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
        max_device_bytes=8 * testing_utils.BYTES_PER_PAGE
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

  def test_finished_request_id_can_be_reused(self):
    engine = testing_utils.make_engine()
    self._run(engine, [[1, 2]])

    self.assertEqual(self._run(engine, [[3]]), [reference_generate([3])])

  def test_aborting_a_finished_request_does_nothing(self):
    engine = testing_utils.make_engine()
    self._run(engine, [[1, 2]])

    engine.abort_request('0')

    self.assertFalse(engine.has_unfinished_requests())

  def test_schedules_next_step_while_model_forward_is_in_flight(self):
    engine = testing_utils.make_engine()
    events: list[str] = []

    orig_execute = engine._model_runner.execute_step  # pylint: disable=protected-access
    orig_schedule = engine._scheduler.schedule_step  # pylint: disable=protected-access
    orig_update = engine._scheduler.update_from_output  # pylint: disable=protected-access

    def record_execute(*args, **kwargs):
      events.append('execute')
      return orig_execute(*args, **kwargs)

    def record_schedule(*args, **kwargs):
      events.append('schedule')
      return orig_schedule(*args, **kwargs)

    def record_update(*args, **kwargs):
      events.append('update')
      return orig_update(*args, **kwargs)

    mock.patch.object(
        engine._model_runner, 'execute_step', side_effect=record_execute  # pylint: disable=protected-access
    ).start()
    mock.patch.object(
        engine._scheduler, 'schedule_step', side_effect=record_schedule  # pylint: disable=protected-access
    ).start()
    mock.patch.object(
        engine._scheduler, 'update_from_output', side_effect=record_update  # pylint: disable=protected-access
    ).start()

    engine.add_request(testing_utils.make_request('0', [1, 2], max_tokens=2))
    engine.step()
    engine.step()

    # Step 1 schedules the initial batch, launches `execute`, schedules Step 2
    # while Step 1 is in flight, then updates from Step 1's output. Step 2
    # reuses the pre-scheduled batch directly.
    self.assertEqual(
        events,
        ['schedule', 'execute', 'schedule', 'update', 'execute', 'schedule', 'update'],
    )



def new_weights(offset: float) -> nnx.State:
  return nnx.variables(testing_utils.PagedSumTransformer(offset=offset), nnx.Param)


class UpdateParamsTest(parameterized.TestCase):

  def tearDown(self):
    mock.patch.stopall()
    super().tearDown()

  def test_replaces_the_weights(self):
    engine = testing_utils.make_engine()

    engine.update_params(new_weights(2.0))

    engine.add_request(testing_utils.make_request('0', [1, 2, 3]))
    self.assertEqual(
        drive(engine)['0'].token_ids.tolist(),
        reference_generate([1, 2, 3], offset=2.0),
    )

  @parameterized.named_parameters(
      ('one_request', 1),
      # Identical prompts share their prefix pages, which must be released
      # only once.
      ('shared_prefix', 3),
  )
  def test_mid_decode_update_resumes_under_the_new_weights(
      self, num_requests
  ):
    engine = testing_utils.make_engine()
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    request_ids = [str(i) for i in range(num_requests)]
    for request_id in request_ids:
      engine.add_request(testing_utils.make_request(request_id, list(prompt)))
    # One prefill step, then two decode steps, each sampling one token.
    for _ in range(3):
      engine.step()
    before_sync = reference_generate(prompt, 3)

    engine.update_params(new_weights(1.0))
    finished = drive(engine)

    # Tokens sampled before the sync are kept, and the continuation must be
    # what the new weights generate from the same tokens. It is not if a
    # request decodes on KV left from the old weights.
    after_sync = reference_generate(prompt + before_sync, 5, offset=1.0)
    for request_id in request_ids:
      self.assertEqual(
          finished[request_id].token_ids.tolist(), before_sync + after_sync
      )

  def test_rewinds_running_requests(self):
    engine = testing_utils.make_engine()
    engine.add_request(testing_utils.make_request('0', [1, 2, 3, 4, 5]))
    for _ in range(3):
      engine.step()
    req = engine._requests['0']  # pylint: disable=protected-access
    self.assertGreater(req.num_completed_tokens, 0)

    engine.update_params(new_weights(1.0))

    self.assertEqual(req.num_completed_tokens, 0)
    self.assertEqual(req.num_tokens_scheduled, 0)
    self.assertTrue(engine.has_unfinished_requests())

  def test_clears_the_prefix_cache(self):
    engine = testing_utils.make_engine()
    prompt = list(range(1, 10))
    engine.add_request(testing_utils.make_request('warm', list(prompt)))
    drive(engine)

    engine.update_params(new_weights(1.0))
    spy = spy_on_execute_step(engine)
    engine.add_request(testing_utils.make_request('fresh', list(prompt)))
    finished = drive(engine)

    # Pages cached under the old weights must not be hit.
    first_metadata = spy.call_args_list[0].kwargs['metadata']
    self.assertEqual(first_metadata.query_lens[0], len(prompt))
    self.assertEqual(
        finished['fresh'].token_ids.tolist(),
        reference_generate(prompt, offset=1.0),
    )

  def test_failed_update_leaves_the_engine_usable(self):
    engine = testing_utils.make_engine()
    engine.add_request(testing_utils.make_request('0', [1, 2, 3]))
    engine.step()
    mismatched = nnx.variables(
        testing_utils.PagedSumTransformer(offset=np.zeros((2,))), nnx.Param
    )

    with self.assertRaises(ValueError):
      engine.update_params(mismatched)

    self.assertEqual(
        drive(engine)['0'].token_ids.tolist(), reference_generate([1, 2, 3])
    )


class ResetKVCachesTest(absltest.TestCase):

  def test_in_place_weight_change_resumes_under_the_new_weights(self):
    engine = testing_utils.make_engine()
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    engine.add_request(testing_utils.make_request('0', list(prompt)))
    # One prefill step, then two decode steps, each sampling one token.
    for _ in range(3):
      engine.step()
    before_reset = reference_generate(prompt, 3)

    engine.transformer_state['offset'].value = jnp.asarray(1.0)
    engine.reset_kv_caches()
    finished = drive(engine)

    after_reset = reference_generate(prompt + before_reset, 5, offset=1.0)
    self.assertEqual(
        finished['0'].token_ids.tolist(), before_reset + after_reset
    )


if __name__ == '__main__':
  absltest.main()
