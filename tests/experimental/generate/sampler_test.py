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

import asyncio
from concurrent import futures

from absl.testing import absltest
from flax import nnx
import jax.numpy as jnp
import numpy as np
from tunix.experimental.generate import request as request_lib
from tunix.experimental.generate import sampler as sampler_lib
from tunix.experimental.generate import testing_utils
from tunix.generate import tokenizer_adapter


def run_on_engine(
    prompt: str | list[int],
    *,
    max_tokens: int = 8,
    offset: float = 0.0,
) -> request_lib.RequestOutput:
  """Returns the output a fresh engine produces for `prompt`."""
  engine = testing_utils.make_engine(
      testing_utils.PagedSumTransformer(offset=offset)
  )
  prompt_token_ids = (
      [int(t) for t in prompt.split()]
      if isinstance(prompt, str)
      else list(prompt)
  )
  engine.add_request(
      testing_utils.make_request('0', prompt_token_ids, max_tokens=max_tokens)
  )
  outputs = []
  while engine.has_unfinished_requests():
    outputs.extend(engine.step())
  return outputs[0]


class _SamplerTests(absltest.TestCase):
  """Tests shared by both modes. Subclasses set `server_mode`."""

  server_mode: bool

  def make_sampler(
      self, engine=None, tokenizer=None, **kwargs
  ) -> sampler_lib.Sampler:
    sampler = sampler_lib.Sampler(
        engine or testing_utils.make_engine(),
        tokenizer
        or tokenizer_adapter.TokenizerAdapter(
            testing_utils.WhitespaceTokenizer()
        ),
        server_mode=self.server_mode,
        **kwargs,
    )
    self.addCleanup(sampler.stop)
    return sampler

  def test_single_string_prompt(self):
    sampler = self.make_sampler()
    expected = run_on_engine('1 2 3')

    output = sampler('1 2 3')

    self.assertEqual(
        output.text, [' '.join(str(t) for t in expected.token_ids)]
    )
    self.assertLen(output.tokens, 1)
    self.assertEqual(output.tokens[0].tolist(), expected.token_ids.tolist())
    assert output.prompt_lengths is not None
    self.assertEqual(output.prompt_lengths.tolist(), [3])
    self.assertEqual(
        output.padded_prompt_tokens.tolist(), [[0, 1, 2, 3]]
    )

  def test_outputs_follow_prompt_order_and_left_pad_prompts(self):
    sampler = self.make_sampler()
    prompts = ['5', '1 2 3 4 5', '9 8 7']
    expected_outputs = [
        run_on_engine(prompt, max_tokens=3) for prompt in prompts
    ]

    output = sampler(prompts, max_generation_steps=3)

    self.assertEqual(
        output.text,
        [' '.join(str(t) for t in e.token_ids) for e in expected_outputs],
    )
    self.assertEqual(
        [t.tolist() for t in output.tokens],
        [e.token_ids.tolist() for e in expected_outputs],
    )
    assert output.prompt_lengths is not None
    self.assertEqual(output.prompt_lengths.tolist(), [1, 5, 3])
    self.assertEqual(
        output.padded_prompt_tokens.tolist(),
        [
            [0, 0, 0, 0, 0, 0, 0, 5],
            [0, 0, 0, 1, 2, 3, 4, 5],
            [0, 0, 0, 0, 0, 9, 8, 7],
        ],
    )

  def test_prompt_token_ids_input(self):
    sampler = self.make_sampler()
    token_ids = [[1, 2, 3], [4, 5]]
    expected_outputs = [run_on_engine(ids) for ids in token_ids]

    output = sampler(prompt_token_ids=token_ids)

    self.assertEqual(
        output.text,
        [' '.join(str(t) for t in e.token_ids) for e in expected_outputs],
    )
    self.assertEqual(
        [t.tolist() for t in output.tokens],
        [e.token_ids.tolist() for e in expected_outputs],
    )
    assert output.prompt_lengths is not None
    self.assertEqual(output.prompt_lengths.tolist(), [3, 2])

  def test_pad_output_right_pads_generated_tokens(self):
    # EOS=6 stops generation early ('1 5' -> [6], '1 2' -> [3, 6]).
    engine = testing_utils.make_engine(eos_token_ids=frozenset({6}))
    sampler = self.make_sampler(engine)

    output = sampler(['1 5', '1 2'], max_generation_steps=4, pad_output=True)

    self.assertEqual(
        [t.tolist() for t in output.tokens],
        [[6, 0, 0, 0], [3, 6, 0, 0]],
    )

  def test_concurrent_calls_all_resolve(self):
    sampler = self.make_sampler()
    prompts = [f'{i + 1} {i + 2}' for i in range(4)]

    with futures.ThreadPoolExecutor(max_workers=4) as pool:
      outputs = list(pool.map(sampler, prompts))

    for output, prompt in zip(outputs, prompts):
      expected = run_on_engine(prompt)
      self.assertEqual(
          output.text, [' '.join(str(t) for t in expected.token_ids)]
      )
      self.assertEqual(output.tokens[0].tolist(), expected.token_ids.tolist())

  def test_mesh_is_the_engine_mesh(self):
    engine = testing_utils.make_engine()

    self.assertEqual(self.make_sampler(engine).mesh, engine.mesh)

  def test_update_params_installs_the_new_weights(self):
    sampler = self.make_sampler()

    sampler.update_params(
        nnx.variables(testing_utils.PagedSumTransformer(offset=1.0), nnx.Param)
    )
    output = sampler('1 2 3')

    expected = run_on_engine('1 2 3', offset=1.0)
    self.assertEqual(
        output.text, [' '.join(str(t) for t in expected.token_ids)]
    )
    self.assertEqual(output.tokens[0].tolist(), expected.token_ids.tolist())

  def test_reinitialize_cache_discards_kv_from_the_old_weights(self):
    sampler = self.make_sampler()
    prompt = ' '.join(str(i) for i in range(1, 10))
    # Fills the prefix cache with KV computed under the old weights.
    sampler(prompt)

    sampler.transformer_state['offset'].value = jnp.asarray(
        1.0, dtype=jnp.float32
    )
    sampler.reinitialize_cache()
    output = sampler(prompt)

    expected = run_on_engine(prompt, offset=1.0)
    self.assertEqual(
        output.text, [' '.join(str(t) for t in expected.token_ids)]
    )
    self.assertEqual(output.tokens[0].tolist(), expected.token_ids.tolist())

  def test_multi_sampling(self):
    sampler = self.make_sampler()
    output = sampler(
        ['1 2 3', '4 5'],
        max_generation_steps=4,
        temperature=1.0,
        multi_sampling=2,
    )

    self.assertLen(output.text, 4)
    self.assertLen(output.tokens, 4)
    assert output.prompt_lengths is not None
    self.assertEqual(output.prompt_lengths.tolist(), [3, 3, 2, 2])

  def test_seed_logs_deprecation_warning_and_is_ignored(self):
    sampler = self.make_sampler()
    with self.assertLogs(level='WARNING') as logs:
      output = sampler('1 2 3', seed=7)

    self.assertTrue(any('deprecated and ignored' in m for m in logs.output))
    self.assertEqual(
        output.tokens[0].tolist(), run_on_engine('1 2 3').token_ids.tolist()
    )

  def test_tokenize_deduplicates_leading_bos_tokens(self):
    tokenizer = tokenizer_adapter.TokenizerAdapter(
        testing_utils.WhitespaceTokenizer()
    )
    tokenizer._tokenizer.bos_id = lambda: 2  # pylint: disable=protected-access
    sampler = self.make_sampler(tokenizer=tokenizer)

    self.assertEqual(sampler.tokenize('2 2 1 3'), [2, 1, 3])
    output = sampler('2 2 1 3', max_generation_steps=2)
    assert output.prompt_lengths is not None
    self.assertEqual(output.prompt_lengths.tolist(), [3])
    self.assertEqual(output.padded_prompt_tokens.tolist(), [[0, 2, 1, 3]])

  def _sample_group(
      self, num_generations: int, **kwargs
  ) -> set[tuple[int, ...]]:
    """Samples one prompt `num_generations` times, one call each, like GRPO."""
    sampler = self.make_sampler(
        testing_utils.make_engine(
            testing_utils.UniformTransformer(), max_top_k=-1
        )
    )

    def sample(_):
      return sampler(
          '1 2 3',
          max_generation_steps=8,
          temperature=0.7,
          top_p=1.0,
          **kwargs,
      )

    with futures.ThreadPoolExecutor(max_workers=num_generations) as pool:
      outputs = list(pool.map(sample, range(num_generations)))
    return {tuple(output.tokens[0].tolist()) for output in outputs}

  def test_unseeded_calls_on_the_same_prompt_sample_independently(self):
    self.assertLen(self._sample_group(8), 8)

  def test_a_rejected_request_raises(self):
    sampler = self.make_sampler()

    with self.assertRaisesRegex(ValueError, 'empty'):
      sampler('')

  def test_exposes_transformer_and_state(self):
    transformer = testing_utils.PagedSumTransformer(offset=2.5)
    sampler = self.make_sampler(testing_utils.make_engine(transformer))

    np.testing.assert_array_equal(
        nnx.state(sampler.transformer)['offset'].value, 2.5
    )
    np.testing.assert_array_equal(
        sampler.transformer_state['offset'].value, 2.5
    )

  def test_generate_returns_sampling_responses(self):
    sampler = self.make_sampler()
    requests = [
        testing_utils.make_request('a', [1, 2, 3], max_tokens=3),
        testing_utils.make_request('b', [4, 5], max_tokens=2),
    ]

    responses = sampler.generate(requests)

    self.assertEqual([r.request_id for r in responses], ['a', 'b'])
    self.assertEqual(
        responses[0].token_ids.tolist(),
        run_on_engine([1, 2, 3], max_tokens=3).token_ids.tolist(),
    )
    self.assertEqual(
        responses[1].token_ids.tolist(),
        run_on_engine([4, 5], max_tokens=2).token_ids.tolist(),
    )

  def test_sample_returns_single_and_batch_responses(self):
    sampler = self.make_sampler()
    req_a = testing_utils.make_request('a', [1, 2], max_tokens=2)
    req_b = testing_utils.make_request('b', [3, 4], max_tokens=2)

    single = asyncio.run(sampler.sample(req_a))
    batch = asyncio.run(sampler.sample([req_a, req_b]))

    assert not isinstance(single, list)
    assert isinstance(batch, list)
    self.assertEqual(single.request_id, 'a')
    self.assertEqual([r.request_id for r in batch], ['a', 'b'])
    self.assertEqual(
        single.token_ids.tolist(),
        run_on_engine([1, 2], max_tokens=2).token_ids.tolist(),
    )


class OfflineSamplerTest(_SamplerTests):

  server_mode = False


class ServerModeSamplerTest(_SamplerTests):

  server_mode = True

  def test_cancelling_sample_withdraws_requests_from_driver(self):
    sampler = self.make_sampler(submission_threshold=2)
    req = testing_utils.make_request('cancelled_req', [1, 2])

    async def run_and_cancel():
      task = asyncio.create_task(sampler.sample(req))
      await asyncio.sleep(0.01)
      task.cancel()
      with self.assertRaises(asyncio.CancelledError):
        await task

    asyncio.run(run_and_cancel())

    # The request ID was withdrawn from the driver, so it can be reused.
    response = asyncio.run(
        sampler.sample([req, testing_utils.make_request('other', [3])])
    )
    assert isinstance(response, list)
    self.assertEqual(
        [r.request_id for r in response], ['cancelled_req', 'other']
    )


# Only its subclasses run.
del _SamplerTests


if __name__ == '__main__':
  absltest.main()
