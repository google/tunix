# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The RPC handles forward their contracts unchanged.

The scoring seam matters most here: in one process the temperature could be
read from a shared config, and across processes it cannot, so it has to travel
with the request and arrive intact.
"""

from absl.testing import absltest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import rpc_workers
from tunix.experimental.worker import remote_execution


class _Scorer:
  """A served inference worker."""

  def __init__(self):
    self.seen = []

  def compute_logps(
      self, request: datatypes.LogprobsRequest
  ) -> datatypes.LogprobsResponse:
    self.seen.append(request)
    if request.temperature <= 1e-5:
      return datatypes.LogprobsResponse(
          request_id=request.request_id,
          per_token_logps=np.zeros((0, 0), dtype=np.float32),
          error=datatypes.ErrorInfo(
              error_type="ValueError", message="bad temperature"
          ),
      )
    return datatypes.LogprobsResponse(
        request_id=request.request_id,
        per_token_logps=np.full((2, 3), request.temperature, np.float32),
    )


class RemoteScoringTest(absltest.TestCase):

  def _handle(self, scorer):
    server = remote_execution.InProcessRemoteExecutionServer(instance=scorer)
    return rpc_workers.RemoteInferenceWorker(
        remote_execution.InProcessActorHandle(server), worker_id="inference"
    )

  def _request(self, temperature: float) -> datatypes.LogprobsRequest:
    return datatypes.LogprobsRequest(
        request_id="req-1",
        prompt_tokens=np.ones((2, 2), dtype=np.int32),
        completion_tokens=np.ones((2, 3), dtype=np.int32),
        temperature=temperature,
    )

  def test_temperature_survives_the_round_trip(self):
    scorer = _Scorer()

    response = self._handle(scorer).compute_logps(self._request(0.7))

    self.assertIsNone(response.error)
    self.assertEqual(scorer.seen[0].temperature, 0.7)
    np.testing.assert_allclose(response.per_token_logps, 0.7)

  def test_a_scoring_failure_comes_back_as_data(self):
    response = self._handle(_Scorer()).compute_logps(self._request(0.0))

    self.assertIsNotNone(response.error)
    self.assertEqual(response.request_id, "req-1")

  def test_the_response_is_wire_safe(self):
    from tunix.experimental.common import rpc_utils  # pylint: disable=g-import-not-at-top

    response = self._handle(_Scorer()).compute_logps(self._request(0.5))

    rpc_utils.validate_wire_safe(response)


if __name__ == "__main__":
  absltest.main()
