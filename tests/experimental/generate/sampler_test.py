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

from absl.testing import absltest
from flax import nnx
import jax.numpy as jnp
from tunix.experimental.generate import request as request_lib
from tunix.experimental.generate import sampler as sampler_lib
from tunix.experimental.generate import testing_utils
from tunix.experimental.rollout import sampler as rollout_sampler_lib


def run_on_engine(
    request: rollout_sampler_lib.SamplingRequest, offset: float = 0.0
) -> request_lib.RequestOutput:
  """Returns the output a fresh engine produces for `request`."""
  engine = testing_utils.make_engine(
      testing_utils.PagedSumTransformer(offset=offset)
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

  def assert_response_matches(
      self, response: object, expected: request_lib.RequestOutput
  ):
    assert isinstance(response, rollout_sampler_lib.SamplingResponse)
    self.assertEqual(response.request_id, expected.request_id)
    self.assertEqual(response.text, expected.text)
    self.assertEqual(
        response.prompt_token_ids.tolist(), expected.prompt_token_ids.tolist()
    )
    self.assertEqual(response.token_ids.tolist(), expected.token_ids.tolist())
    self.assertEqual(response.finish_reason, expected.finish_reason)

  def test_a_single_request_gets_a_single_response(self):
    sampler = self.make_sampler()
    request = testing_utils.make_request('0', [1, 2, 3])

    response = asyncio.run(sampler.sample(request))

    self.assert_response_matches(response, run_on_engine(request))

  def test_responses_follow_the_request_order(self):
    sampler = self.make_sampler()
    requests = [
        testing_utils.make_request('a', [5]),
        testing_utils.make_request('b', [1, 2, 3, 4, 5], max_tokens=3),
        testing_utils.make_request('c', [9, 8, 7]),
    ]

    responses = asyncio.run(sampler.sample(requests))

    assert isinstance(responses, list)
    self.assertLen(responses, len(requests))
    for response, request in zip(responses, requests):
      self.assert_response_matches(response, run_on_engine(request))

  def test_generate_blocks_until_every_request_finishes(self):
    sampler = self.make_sampler()
    requests = [
        testing_utils.make_request('a', [5]),
        testing_utils.make_request('b', [1, 2, 3, 4, 5], max_tokens=3),
    ]

    responses = sampler.generate(requests)

    self.assertLen(responses, len(requests))
    for response, request in zip(responses, requests):
      self.assert_response_matches(response, run_on_engine(request))

  def test_concurrent_calls_all_resolve(self):
    sampler = self.make_sampler()
    requests = [
        testing_utils.make_request(str(i), [i + 1, i + 2]) for i in range(4)
    ]

    async def sample_concurrently():
      return await asyncio.gather(
          *(sampler.sample(request) for request in requests)
      )

    responses = asyncio.run(sample_concurrently())

    for response, request in zip(responses, requests):
      self.assert_response_matches(response, run_on_engine(request))

  def test_mesh_is_the_engine_mesh(self):
    engine = testing_utils.make_engine()

    self.assertEqual(self.make_sampler(engine).mesh, engine.mesh)

  def test_update_params_installs_the_new_weights(self):
    sampler = self.make_sampler()
    request = testing_utils.make_request('0', [1, 2, 3])

    sampler.update_params(
        nnx.variables(testing_utils.PagedSumTransformer(offset=1.0), nnx.Param)
    )
    response = asyncio.run(sampler.sample(request))

    self.assert_response_matches(response, run_on_engine(request, offset=1.0))

  def test_reinitialize_cache_discards_kv_from_the_old_weights(self):
    sampler = self.make_sampler()
    request = testing_utils.make_request('0', list(range(1, 10)))
    # Fills the prefix cache with KV computed under the old weights.
    asyncio.run(sampler.sample(request))

    sampler.delete_cache()
    sampler.transformer_state['offset'].value = jnp.asarray(
        1.0, dtype=jnp.float32
    )
    sampler.reinitialize_cache()
    response = asyncio.run(sampler.sample(request))

    self.assert_response_matches(response, run_on_engine(request, offset=1.0))

class OfflineSamplerTest(_SamplerTests):

  server_mode = False

  def test_a_rejected_request_raises(self):
    sampler = self.make_sampler()

    with self.assertRaises(ValueError):
      asyncio.run(sampler.sample(testing_utils.make_request('0', [])))


class ServerModeSamplerTest(_SamplerTests):

  server_mode = True

  def test_cancelling_a_call_withdraws_its_requests(self):
    # A lone request waits for a second one to reach the threshold.
    sampler = self.make_sampler(submission_threshold=2)
    request = testing_utils.make_request('0', [1, 2])

    async def cancel_while_queued():
      task = asyncio.create_task(sampler.sample(request))
      await asyncio.sleep(0.1)
      task.cancel()
      with self.assertRaises(asyncio.CancelledError):
        await task

    asyncio.run(cancel_while_queued())

    # The id is free again, which it is not while the request is pending.
    responses = asyncio.run(
        sampler.sample([request, testing_utils.make_request('1', [3])])
    )
    assert isinstance(responses, list)
    self.assertEqual([r.request_id for r in responses], ['0', '1'])


# Only its subclasses run.
del _SamplerTests


if __name__ == '__main__':
  absltest.main()
