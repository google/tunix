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

from absl.testing import absltest
from flax import nnx
import jax.numpy as jnp
import numpy as np
from tunix.experimental.generate import request as request_lib
from tunix.experimental.generate import sampler as sampler_lib
from tunix.experimental.generate import testing_utils
from tunix.experimental.rollout import sampler as rollout_sampler_lib


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
  request = rollout_sampler_lib.SamplingRequest(
      request_id='0',
      prompt=(
          prompt
          if isinstance(prompt, str)
          else np.asarray(prompt, dtype=np.int32)
      ),
      sampling_params=rollout_sampler_lib.SamplingParams(
          max_tokens=max_tokens, temperature=0.0
      ),
  )
  engine.add_request(request)
  outputs = []
  while engine.has_unfinished_requests():
    outputs.extend(engine.step())
  return outputs[0]


class _SamplerTests(absltest.TestCase):
  """Tests shared by both modes. Subclasses set `server_mode`."""

  server_mode: bool

  def make_sampler(
      self, engine=None, **kwargs
  ) -> sampler_lib.Sampler:
    sampler = sampler_lib.Sampler(
        engine or testing_utils.make_engine(),
        server_mode=self.server_mode,
        **kwargs,
    )
    self.addCleanup(sampler.stop)
    return sampler

  def test_single_string_prompt(self):
    sampler = self.make_sampler()
    expected = run_on_engine('1 2 3')

    output = sampler('1 2 3')

    self.assertEqual(output.text, [expected.text])
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

    self.assertEqual(output.text, [e.text for e in expected_outputs])
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

    self.assertEqual(output.text, [e.text for e in expected_outputs])
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
      self.assertEqual(output.text, [expected.text])
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
    self.assertEqual(output.text, [expected.text])
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
    self.assertEqual(output.text, [expected.text])
    self.assertEqual(output.tokens[0].tolist(), expected.token_ids.tolist())

  def test_multi_sampling(self):
    sampler = self.make_sampler()
    output = sampler(
        ['1 2 3', '4 5'],
        max_generation_steps=4,
        temperature=1.0,
        seed=7,
        multi_sampling=2,
    )

    self.assertLen(output.text, 4)
    self.assertLen(output.tokens, 4)
    assert output.prompt_lengths is not None
    self.assertEqual(output.prompt_lengths.tolist(), [3, 3, 2, 2])


class OfflineSamplerTest(_SamplerTests):

  server_mode = False

  def test_a_rejected_request_raises(self):
    sampler = self.make_sampler()

    with self.assertRaises(ValueError):
      sampler('')


class ServerModeSamplerTest(_SamplerTests):

  server_mode = True


# Only its subclasses run.
del _SamplerTests


if __name__ == '__main__':
  absltest.main()
