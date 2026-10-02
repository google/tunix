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

import time

from absl.testing import absltest
from flax import nnx
from tunix.experimental.generate import driver as driver_lib
from tunix.experimental.generate import request as request_lib
from tunix.experimental.generate import testing_utils
from tunix.experimental.rollout import sampler as sampler_lib

# Generous, so that a slow test machine never flakes.
_TIMEOUT_S = 60.0


def run_on_engine(
    *requests: sampler_lib.SamplingRequest,
) -> dict[str, request_lib.RequestOutput]:
  """Returns the outputs a fresh engine produces for `requests`."""
  engine = testing_utils.make_engine()
  for request in requests:
    engine.add_request(request)
  outputs = {}
  while engine.has_unfinished_requests():
    for output in engine.step():
      outputs[output.request_id] = output
  return outputs


class VanillaInProcessDriverTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.engine = testing_utils.make_engine()

  def make_driver(self, **kwargs) -> driver_lib.VanillaInProcessDriver:
    driver = driver_lib.VanillaInProcessDriver(self.engine, **kwargs)
    self.addCleanup(driver.shutdown)
    return driver

  def assert_output_equal(
      self,
      actual: request_lib.RequestOutput,
      expected: request_lib.RequestOutput,
  ):
    self.assertEqual(actual.request_id, expected.request_id)
    self.assertEqual(actual.token_ids.tolist(), expected.token_ids.tolist())
    self.assertEqual(actual.finish_reason, expected.finish_reason)

  def test_non_positive_poll_interval_raises(self):
    with self.assertRaisesRegex(ValueError, 'poll_interval_s'):
      driver_lib.VanillaInProcessDriver(self.engine, poll_interval_s=0.0)

  def test_exposes_the_engine(self):
    self.assertIs(self.make_driver().engine, self.engine)

  def test_submit_request_resolves_to_the_output(self):
    driver = self.make_driver()
    driver.start()
    request = testing_utils.make_request('0', [1, 2, 3])

    output = driver.submit_request(request).result(timeout=_TIMEOUT_S)

    self.assert_output_equal(output, run_on_engine(request)['0'])

  def test_submit_requests_returns_futures_in_order(self):
    driver = self.make_driver()
    driver.start()
    requests = [
        testing_utils.make_request('a', [5]),
        testing_utils.make_request('b', [1, 2, 3, 4, 5], max_tokens=3),
        testing_utils.make_request('c', [9, 8, 7]),
    ]

    outputs = [
        future.result(timeout=_TIMEOUT_S)
        for future in driver.submit_requests(requests)
    ]

    expected = run_on_engine(*requests)
    for output, request in zip(outputs, requests):
      self.assert_output_equal(output, expected[request.request_id])

  def test_requests_submitted_before_start_resolve_once_started(self):
    driver = self.make_driver()
    future = driver.submit_request(testing_utils.make_request('0', [1, 2]))
    self.assertFalse(future.done())

    driver.start()

    self.assertEqual(future.result(timeout=_TIMEOUT_S).request_id, '0')

  def test_stop_leaves_requests_pending(self):
    driver = self.make_driver()
    driver.start()
    driver.stop()
    future = driver.submit_request(testing_utils.make_request('0', [1, 2]))

    self.assertFalse(future.done())
    driver.start()
    self.assertEqual(future.result(timeout=_TIMEOUT_S).request_id, '0')

  def test_finished_request_id_can_be_reused(self):
    driver = self.make_driver()
    driver.start()
    request = testing_utils.make_request('0', [1, 2])
    driver.submit_request(request).result(timeout=_TIMEOUT_S)

    output = driver.submit_request(request).result(timeout=_TIMEOUT_S)

    self.assert_output_equal(output, run_on_engine(request)['0'])

  def test_pending_request_id_raises(self):
    driver = self.make_driver()
    driver.submit_request(testing_utils.make_request('0', [1, 2]))

    with self.assertRaisesRegex(ValueError, 'still pending'):
      driver.submit_request(testing_utils.make_request('0', [3]))

  def test_duplicate_ids_in_a_batch_raise_without_submitting(self):
    driver = self.make_driver()

    with self.assertRaisesRegex(ValueError, 'unique'):
      driver.submit_requests([
          testing_utils.make_request('0', [1]),
          testing_utils.make_request('0', [2]),
      ])

    # Nothing was submitted, so the id is still free.
    driver.submit_request(testing_utils.make_request('0', [1]))

  def test_shutdown_fails_pending_requests(self):
    driver = self.make_driver()
    future = driver.submit_request(testing_utils.make_request('0', [1, 2]))

    driver.shutdown()

    with self.assertRaisesRegex(RuntimeError, 'shut down'):
      future.result(timeout=_TIMEOUT_S)

  def test_context_manager_starts_and_shuts_down(self):
    with self.make_driver() as driver:
      kept = driver.submit_request(testing_utils.make_request('0', [1, 2]))
      self.assertEqual(kept.result(timeout=_TIMEOUT_S).request_id, '0')
      driver.stop()
      dropped = driver.submit_request(testing_utils.make_request('1', [1]))

    with self.assertRaisesRegex(RuntimeError, 'shut down'):
      dropped.result(timeout=_TIMEOUT_S)

  def test_engine_error_kills_the_loop(self):
    driver = self.make_driver()
    # The engine rejects a prompt longer than `max_model_len`.
    rejected = testing_utils.make_request('0', [1] * 30)

    future = driver.submit_request(rejected)
    driver.start()

    with self.assertRaisesRegex(ValueError, 'max_model_len'):
      future.result(timeout=_TIMEOUT_S)
    self.assertIsInstance(driver.last_error, ValueError)

  def test_engine_error_fails_every_pending_request(self):
    driver = self.make_driver()
    futures = driver.submit_requests([
        testing_utils.make_request('ok', [1, 2]),
        testing_utils.make_request('rejected', [1] * 30),
    ])

    driver.start()

    for future in futures:
      with self.assertRaisesRegex(ValueError, 'max_model_len'):
        future.result(timeout=_TIMEOUT_S)

  def test_submitting_after_the_loop_died_raises(self):
    driver = self.make_driver()
    driver.start()
    future = driver.submit_request(testing_utils.make_request('0', [1] * 30))
    with self.assertRaises(ValueError):
      future.result(timeout=_TIMEOUT_S)

    with self.assertRaisesRegex(RuntimeError, 'died'):
      driver.submit_request(testing_utils.make_request('1', [1]))
    with self.assertRaisesRegex(RuntimeError, 'died'):
      driver.submit_requests([testing_utils.make_request('1', [1])])


class SubmissionBatchingTest(absltest.TestCase):

  def make_driver(self, **kwargs) -> driver_lib.VanillaInProcessDriver:
    driver = driver_lib.VanillaInProcessDriver(
        testing_utils.make_engine(), **kwargs
    )
    self.addCleanup(driver.shutdown)
    return driver

  def test_negative_threshold_raises(self):
    with self.assertRaisesRegex(ValueError, 'submission_threshold'):
      self.make_driver(submission_threshold=-1)

  def test_negative_timeout_raises(self):
    with self.assertRaisesRegex(ValueError, 'submission_timeout_s'):
      self.make_driver(submission_timeout_s=-1.0)

  def test_requests_wait_for_the_threshold(self):
    driver = self.make_driver(submission_threshold=2)
    driver.start()
    first = driver.submit_request(testing_utils.make_request('0', [1, 2]))
    time.sleep(0.2)
    self.assertFalse(first.done())

    second = driver.submit_request(testing_utils.make_request('1', [3]))

    self.assertEqual(first.result(timeout=_TIMEOUT_S).request_id, '0')
    self.assertEqual(second.result(timeout=_TIMEOUT_S).request_id, '1')

  def test_a_batch_reaching_the_threshold_is_submitted(self):
    driver = self.make_driver(submission_threshold=2)
    driver.start()

    futures = driver.submit_requests([
        testing_utils.make_request('0', [1, 2]),
        testing_utils.make_request('1', [3]),
    ])

    for future in futures:
      future.result(timeout=_TIMEOUT_S)

  def test_timeout_submits_a_partial_batch(self):
    driver = self.make_driver(
        submission_threshold=100, submission_timeout_s=0.05
    )
    driver.start()

    future = driver.submit_request(testing_utils.make_request('0', [1, 2]))

    self.assertEqual(future.result(timeout=_TIMEOUT_S).request_id, '0')


class CancelTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.driver = driver_lib.VanillaInProcessDriver(testing_utils.make_engine())
    self.addCleanup(self.driver.shutdown)

  def test_cancelling_a_queued_request_withdraws_it(self):
    cancelled = self.driver.submit_request(
        testing_utils.make_request('cancelled', [1, 2])
    )
    kept = self.driver.submit_request(testing_utils.make_request('kept', [3]))

    self.driver.cancel('cancelled')
    self.driver.start()

    self.assertTrue(cancelled.cancelled())
    self.assertEqual(kept.result(timeout=_TIMEOUT_S).request_id, 'kept')

  def test_cancelling_a_running_request_aborts_it(self):
    future = self.driver.submit_request(
        testing_utils.make_request('0', [1, 2])
    )
    # Hands the request to the engine and runs a step, without the loop.
    self.driver._step_engine()  # pylint: disable=protected-access
    self.assertTrue(self.driver.engine.has_unfinished_requests())

    self.driver.cancel('0')

    self.assertTrue(future.cancelled())
    self.assertFalse(self.driver.engine.has_unfinished_requests())

  def test_cancelled_request_id_can_be_reused(self):
    self.driver.submit_request(testing_utils.make_request('0', [1, 2]))
    self.driver.cancel('0')
    self.driver.start()

    future = self.driver.submit_request(testing_utils.make_request('0', [3]))

    self.assertEqual(future.result(timeout=_TIMEOUT_S).request_id, '0')

  def test_cancelling_an_unknown_request_does_nothing(self):
    future = self.driver.submit_request(testing_utils.make_request('0', [1]))

    self.driver.cancel('unknown')
    self.driver.start()

    self.assertEqual(future.result(timeout=_TIMEOUT_S).request_id, '0')

  def test_cancelling_a_resolved_request_does_nothing(self):
    self.driver.start()
    future = self.driver.submit_request(testing_utils.make_request('0', [1]))
    output = future.result(timeout=_TIMEOUT_S)

    self.driver.cancel('0')

    self.assertIs(future.result(), output)


class UpdateParamsTest(absltest.TestCase):

  def test_requests_run_under_the_new_weights(self):
    driver = driver_lib.VanillaInProcessDriver(testing_utils.make_engine())
    self.addCleanup(driver.shutdown)
    driver.start()
    request = testing_utils.make_request('0', [1, 2, 3])

    driver.update_params(
        nnx.variables(testing_utils.PagedSumTransformer(offset=1.0), nnx.Param)
    )
    output = driver.submit_request(request).result(timeout=_TIMEOUT_S)

    reference = testing_utils.make_engine(
        testing_utils.PagedSumTransformer(offset=1.0)
    )
    reference.add_request(request)
    expected = []
    while reference.has_unfinished_requests():
      expected.extend(reference.step())
    self.assertEqual(output.token_ids.tolist(), expected[0].token_ids.tolist())


if __name__ == '__main__':
  absltest.main()
