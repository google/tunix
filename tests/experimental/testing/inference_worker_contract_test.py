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

"""Runs the InferenceWorker contract suite against the real InferenceWorker."""

import numpy as np

from absl.testing import absltest
from tunix.experimental.testing import inference_worker_contract
from tunix.experimental.worker import inference_worker


class InferenceWorkerContractTest(
    inference_worker_contract.InferenceWorkerContractSuite, absltest.TestCase
):

  def make_worker(self):
    return inference_worker.InferenceWorker(
        inference_worker_contract.StubReferenceScoringCore(),
        pad_id=0,
        eos_id=1,
        model_version=3,
    )

  def test_wrapper_returns_temperature_scaled_core_output(self):
    # Stub-specific: verifies the worker converts numpy->device->numpy and
    # applies the request temperature without mangling the core's output.
    worker = self.make_worker()
    request = self._logprobs_request(batch=2, temperature=2.0)
    result = worker.compute_logprobs(request)
    np.testing.assert_allclose(
        result.per_token_logps,
        request.completion_tokens.astype(np.float32) * 2.0,
    )


if __name__ == "__main__":
  absltest.main()
