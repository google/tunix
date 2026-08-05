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

"""Tests for orchestrator-side gRPC worker proxy contracts."""

from __future__ import annotations

import unittest

import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.orchestrator import grpc_worker_proxies


class _FakeHandle:

  def __init__(self):
    self.calls = []

  def submit(self, method_name, *args, **kwargs):
    self.calls.append((method_name, args, kwargs))
    if method_name != "compute_logps":
      raise AssertionError(f"unexpected method: {method_name}")
    request = args[0]
    return datatypes.LogprobsResponse(
        request_id=request.request_id,
        per_token_logps=np.ones_like(
            request.completion_tokens, dtype=np.float32
        ),
        model_version=3,
    )


class GrpcWorkerProxiesTest(unittest.TestCase):

  def test_inference_proxy_uses_inference_worker_logprobs_contract(self):
    handle = _FakeHandle()
    proxy = grpc_worker_proxies.GrpcInferenceWorkerProxy(handle)

    logps = proxy.per_token_logps(
        [[1, 2]],
        [[3, 4, 5]],
        pad_id=0,
        eos_id=1,
        temperature=0.7,
    )

    np.testing.assert_array_equal(
        logps, np.ones((1, 3), dtype=np.float32)
    )
    method_name, args, kwargs = handle.calls[0]
    self.assertEqual(method_name, "compute_logps")
    self.assertFalse(kwargs)
    request = args[0]
    self.assertIsInstance(request, datatypes.LogprobsRequest)
    self.assertEqual(request.temperature, 0.7)
    self.assertEqual(request.model_role, "reference")


if __name__ == "__main__":
  unittest.main()
