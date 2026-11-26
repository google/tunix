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
import jax.numpy as jnp
import numpy as np
from tunix.experimental.generate import configs
from tunix.experimental.generate import dp_coordinator as dp_coordinator_lib
from tunix.experimental.generate import request as request_lib
from tunix.experimental.generate import testing_utils


def make_coordinator(
    *,
    max_model_len: int = 24,
    max_device_size_gib: float = 64 * testing_utils.GIB_PER_PAGE,
    max_num_batched_tokens: int = 32,
    chunked_prefill_length: int = 32,
    num_scheduler_steps: int = 1,
    eos_token_ids: frozenset[int] = frozenset(),
    max_num_seqs: int = 4,
) -> dp_coordinator_lib.DPCoordinator:
  """Returns a `DPCoordinator` configured for unit tests."""
  cache_config = configs.CacheConfig(
      max_device_size_gib=max_device_size_gib,
      max_host_size_gib=0.0,
      page_size=testing_utils.PAGE_SIZE,
      enable_prefix_caching=True,
      dtype=jnp.float32,
  )
  scheduler_config = configs.SchedulerConfig(
      max_num_batched_tokens=max_num_batched_tokens,
      max_num_seqs=max_num_seqs,
      max_chunked_prefill_length=chunked_prefill_length,
      num_scheduler_steps=num_scheduler_steps,
  )
  return dp_coordinator_lib.DPCoordinator(
      cache_config=cache_config,
      scheduler_config=scheduler_config,
      cache_geometries=testing_utils.GEOMETRIES,
      mesh=testing_utils.mesh(),
      max_model_len=max_model_len,
      eos_token_ids=eos_token_ids,
  )


class DPCoordinatorTest(absltest.TestCase):

  def test_non_positive_max_model_len_raises(self):
    with self.assertRaisesRegex(ValueError, 'max_model_len'):
      make_coordinator(max_model_len=0)

  def test_max_pages_per_seq_over_num_device_pages_raises(self):
    # max_model_len=24 + num_scheduler_steps=1 spans 7 pages of 4 slots.
    with self.assertRaisesRegex(ValueError, 'needs up to 7 pages'):
      make_coordinator(
          max_model_len=24, max_device_size_gib=6 * testing_utils.GIB_PER_PAGE
      )


class AddRequestTest(absltest.TestCase):

  def test_new_request_is_unfinished(self):
    coordinator = make_coordinator()
    self.assertFalse(coordinator.has_unfinished_requests())

    coordinator.add_request(testing_utils.make_request('0', [1, 2]))

    self.assertTrue(coordinator.has_unfinished_requests())

  def test_empty_prompt_raises(self):
    with self.assertRaisesRegex(ValueError, 'empty prompt'):
      make_coordinator().add_request(testing_utils.make_request('0', []))

  def test_request_over_max_model_len_raises(self):
    coordinator = make_coordinator(max_model_len=10)
    coordinator.add_request(
        testing_utils.make_request('fits', [1] * 4, max_tokens=6)
    )

    with self.assertRaisesRegex(ValueError, 'max_model_len'):
      coordinator.add_request(
          testing_utils.make_request('0', [1] * 5, max_tokens=6)
      )

  def test_running_request_id_raises(self):
    coordinator = make_coordinator()
    coordinator.add_request(testing_utils.make_request('0', [1, 2]))

    with self.assertRaisesRegex(ValueError, 'already running'):
      coordinator.add_request(testing_utils.make_request('0', [3]))

  def test_aborted_request_id_can_be_reused(self):
    coordinator = make_coordinator()
    coordinator.add_request(testing_utils.make_request('0', [1, 2]))
    coordinator.abort_request('0')

    coordinator.add_request(testing_utils.make_request('0', [3]))

    self.assertTrue(coordinator.has_unfinished_requests())


class AbortRequestTest(absltest.TestCase):

  def test_aborting_a_new_request_withdraws_it(self):
    coordinator = make_coordinator()
    coordinator.add_request(testing_utils.make_request('0', [1, 2]))

    coordinator.abort_request('0')

    self.assertFalse(coordinator.has_unfinished_requests())

  def test_aborting_leaves_other_requests_unfinished(self):
    coordinator = make_coordinator()
    coordinator.add_request(testing_utils.make_request('0', [1, 2]))
    coordinator.add_request(testing_utils.make_request('1', [1, 2]))

    coordinator.abort_request('0')

    self.assertTrue(coordinator.has_unfinished_requests())

  def test_aborting_a_done_request_does_nothing(self):
    coordinator = make_coordinator()
    coordinator.add_request(testing_utils.make_request('0', [1, 2]))
    coordinator.abort_request('0')

    coordinator.abort_request('0')

    self.assertFalse(coordinator.has_unfinished_requests())

  def test_aborting_an_unknown_request_does_nothing(self):
    coordinator = make_coordinator()
    coordinator.add_request(testing_utils.make_request('0', [1, 2]))

    coordinator.abort_request('unknown')

    self.assertTrue(coordinator.has_unfinished_requests())


class StepTest(absltest.TestCase):

  def test_step_without_requests_does_nothing(self):
    scheduled, distribution = make_coordinator().schedule_step()
    self.assertEmpty(scheduled)
    self.assertEqual(distribution, (0, 0, 0))

  def test_each_request_stops_at_its_own_token_limit(self):
    coordinator = make_coordinator()
    coordinator.add_request(
        testing_utils.make_request('short', [1, 2], max_tokens=1)
    )
    coordinator.add_request(
        testing_utils.make_request('long', [1, 2], max_tokens=2)
    )

    scheduled, distribution = coordinator.schedule_step()
    coordinator.prepare_batch(scheduled)
    finished = coordinator.update_from_output(
        scheduled,
        distribution,
        np.array([[3], [4], [0], [0]], dtype=np.int32),
    )

    self.assertLen(finished, 1)
    self.assertEqual(finished[0].request_id, 'short')
    self.assertEqual(
        finished[0].status, request_lib.RequestStatus.FINISHED_LENGTH
    )
    self.assertTrue(coordinator.has_unfinished_requests())

  def test_stops_at_an_eos_token(self):
    coordinator = make_coordinator(
        eos_token_ids=frozenset({9}), num_scheduler_steps=2
    )
    coordinator.add_request(
        testing_utils.make_request('0', [1, 2, 3], max_tokens=4)
    )

    scheduled, distribution = coordinator.schedule_step()
    coordinator.prepare_batch(scheduled)
    finished = coordinator.update_from_output(
        scheduled,
        distribution,
        np.array([[9, 10], [0, 0], [0, 0], [0, 0]], dtype=np.int32),
    )

    self.assertLen(finished, 1)
    # The EOS token is kept and trailing tokens are dropped.
    self.assertEqual(finished[0].token_ids, [1, 2, 3, 9])
    self.assertEqual(finished[0].status, request_lib.RequestStatus.FINISHED_EOS)

  def test_stale_output_finishing_a_readmitted_request_keeps_its_kv_count(
      self,
  ):
    coordinator = make_coordinator(eos_token_ids=frozenset({7}))
    req = request_lib.RequestState(
        req_id='0',
        prompt_token_ids=list(range(1, 25)),
        sampling_params=testing_utils.make_request('0', [1]).sampling_params,
    )
    # `req` was preempted while sampling a token, then re-admitted to run its
    # first chunk before that token arrived.
    req.num_stale_output_tokens = 1
    req.status = request_lib.RequestStatus.RUNNING
    req.num_computed_tokens = 16

    coordinator._update_req_from_output(req, np.array([7]))  # pylint: disable=protected-access

    self.assertEqual(req.status, request_lib.RequestStatus.FINISHED_EOS)
    # Only the re-run chunk is in the KV cache, not the whole sequence.
    self.assertEqual(req.num_computed_tokens, 16)
    self.assertEqual(req.num_stale_output_tokens, 0)

  def test_chunked_prefill(self):
    coordinator = make_coordinator(
        max_num_batched_tokens=8, chunked_prefill_length=4
    )
    coordinator.add_request(
        testing_utils.make_request('0', list(range(1, 15)))
    )

    scheduled, distribution = coordinator.schedule_step()
    tokens, prev_to_decode_row, query_lens, kv_lens, _ = (
        coordinator.prepare_batch(scheduled)
    )

    self.assertEqual(distribution, (0, 1, 1))
    self.assertEqual(tokens, [1, 2, 3, 4])
    self.assertEmpty(prev_to_decode_row)
    self.assertEqual(query_lens[0], 4)
    self.assertEqual(kv_lens[0], 4)

  def test_page_table_leaves_room_for_the_decode_steps(self):
    coordinator = make_coordinator(max_model_len=15, num_scheduler_steps=2)
    coordinator.add_request(
        testing_utils.make_request('0', [1, 2, 3], max_tokens=4)
    )

    scheduled, _ = coordinator.schedule_step()
    _, _, _, _, page_indices = coordinator.prepare_batch(scheduled)

    # 15 tokens plus 2 decode steps span 5 pages of 4 slots.
    self.assertEqual(page_indices['layer_0'].shape, (4, 5))

  def test_prepare_batch_maps_in_flight_decode_to_prev_row(self):
    coordinator = make_coordinator()
    coordinator.add_request(testing_utils.make_request('0', [1, 2, 3]))
    coordinator.add_request(testing_utils.make_request('1', [4, 5]))

    scheduled_1, _ = coordinator.schedule_step()
    coordinator.prepare_batch(scheduled_1)

    scheduled_2, _ = coordinator.schedule_step()
    tokens, prev_to_decode_row, query_lens, kv_lens, _ = (
        coordinator.prepare_batch(scheduled_2, prev_scheduled=scheduled_1)
    )

    self.assertEqual(tokens, [0, 0])
    self.assertEqual(prev_to_decode_row, {0: 0, 1: 1})
    np.testing.assert_array_equal(query_lens[:2], [1, 1])
    np.testing.assert_array_equal(kv_lens[:2], [4, 3])

  def test_returns_logits_and_logprobs(self):
    coordinator = make_coordinator()
    coordinator.add_request(
        testing_utils.make_request(
            '0',
            [1, 2, 3],
            max_tokens=1,
            return_logits=True,
            return_logprobs=True,
        )
    )

    scheduled, distribution = coordinator.schedule_step()
    coordinator.prepare_batch(scheduled)
    logits = np.zeros((4, 1, testing_utils.VOCAB_SIZE), dtype=np.float32)
    logprobs = np.array([[-0.5], [0.0], [0.0], [0.0]], dtype=np.float32)
    (req,) = coordinator.update_from_output(
        scheduled,
        distribution,
        np.array([[4], [0], [0], [0]], dtype=np.int32),
        logits=logits,
        logprobs=logprobs,
    )

    self.assertEqual(req.token_ids, [1, 2, 3, 4])
    self.assertLen(req.logits, 1)
    self.assertEqual(req.logprobs, [np.float32(-0.5)])

  def test_aborted_request_is_not_unfinished(self):
    coordinator = make_coordinator()
    coordinator.add_request(testing_utils.make_request('0', [1, 2, 3]))
    coordinator.schedule_step()
    coordinator.abort_request('0')

    self.assertFalse(coordinator.has_unfinished_requests())


class ResetKVCachesTest(absltest.TestCase):

  def test_rewinds_running_requests(self):
    coordinator = make_coordinator()
    coordinator.add_request(
        testing_utils.make_request('0', [1, 2, 3, 4, 5])
    )
    scheduled, _ = coordinator.schedule_step()
    coordinator.prepare_batch(scheduled)
    req = coordinator._requests['0']  # pylint: disable=protected-access
    self.assertGreater(req.num_computed_tokens, 0)

    coordinator.reset_kv_caches()

    self.assertEqual(req.num_computed_tokens, 0)
    self.assertTrue(coordinator.has_unfinished_requests())

  def test_delete_kv_caches_and_reallocate_on_reset(self):
    coordinator = make_coordinator()
    coordinator.add_request(testing_utils.make_request('0', [1, 2, 3, 4, 5]))
    coordinator.schedule_step()

    coordinator.delete_kv_caches()
    for arr in coordinator.get_physical_pages().values():
      self.assertTrue(arr.is_deleted())

    coordinator.reset_kv_caches()
    for arr in coordinator.get_physical_pages().values():
      self.assertFalse(arr.is_deleted())


if __name__ == '__main__':
  absltest.main()
