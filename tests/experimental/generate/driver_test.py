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
from tunix.experimental.generate import driver as driver_lib
from tunix.experimental.generate import request as request_lib
from tunix.experimental.generate import testing_utils

# Generous, so that a slow test machine never flakes.
_TIMEOUT_S = 60.0


def run_on_engine(
    *requests: request_lib.Request,
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

  def test_context_manager_starts_and_shuts_down(self):
    with self.make_driver() as driver:
      kept = driver.submit_request(testing_utils.make_request('0', [1, 2]))
      self.assertEqual(kept.result(timeout=_TIMEOUT_S).request_id, '0')

    after = driver.submit_request(testing_utils.make_request('1', [1]))
    self.assertFalse(after.done())

  def test_shutdown_fails_pending_requests(self):
    driver = self.make_driver()
    future = driver.submit_request(testing_utils.make_request('0', [1, 2]))

    driver.shutdown()

    with self.assertRaisesRegex(RuntimeError, 'shut down'):
      future.result(timeout=_TIMEOUT_S)

  def test_rejected_request_fails_its_future_without_stopping_the_loop(self):
    driver = self.make_driver()
    driver.start()
    bad = driver.submit_request(testing_utils.make_request('bad', []))
    good = driver.submit_request(testing_utils.make_request('good', [1, 2]))

    with self.assertRaisesRegex(ValueError, 'empty'):
      bad.result(timeout=_TIMEOUT_S)
    self.assertEqual(good.result(timeout=_TIMEOUT_S).request_id, 'good')


if __name__ == '__main__':
  absltest.main()
