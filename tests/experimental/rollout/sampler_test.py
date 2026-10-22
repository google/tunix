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

"""Tests for Sampler requests and responses in rollout/sampler.py."""

from absl.testing import absltest
import cloudpickle
import numpy as np

from tunix.experimental.rollout import sampler as base_sampler_lib


def _sample_response() -> base_sampler_lib.SamplingResponse:
  return base_sampler_lib.SamplingResponse(
      request_id="sample-req-1",
      text="Hello from Tunix sampler!",
      prompt_token_ids=np.array([1, 2], dtype=np.int32),
      token_ids=np.array([101, 102, 103], dtype=np.int32),
      logprobs=np.array([-0.1, -0.2, -0.05], dtype=np.float32),
      finish_reason="stop",
      routed_experts=np.zeros((1, 2, 3, 2), dtype=np.int32),
      metadata={"cached": True},
  )


def _sampling_request() -> base_sampler_lib.SamplingRequest:
  return base_sampler_lib.SamplingRequest(
      prompt="Solve 2+2",
      request_id="req-sample-42",
      sampling_params=base_sampler_lib.SamplingParams(
          max_tokens=64, temperature=0.7
      ),
      metadata={"priority": "high"},
  )


def _weight_sync_request() -> base_sampler_lib.WeightSyncRequest:
  return base_sampler_lib.WeightSyncRequest(
      request_id="sync-req-1",
      controller_id="raiden-ctrl-0",
      policy_version=14,
      source_metadata={"mesh": "2x4"},
      extra_config={"timeout": 30.0},
      metadata={"trigger": "cron"},
  )


def _load_info() -> base_sampler_lib.LoadInfo:
  return base_sampler_lib.LoadInfo(
      request_id="load-req-1",
      num_requests_waiting=5,
      num_requests_running=2,
      kv_cache_usage_perc=0.45,
      metadata={"status": "ok"},
  )


class SamplerTest(absltest.TestCase):

  def test_weight_sync_request_round_trips_through_cloudpickle(self):
    original = _weight_sync_request()
    restored = cloudpickle.loads(cloudpickle.dumps(original))

    self.assertEqual(restored.request_id, original.request_id)
    self.assertEqual(restored.controller_id, original.controller_id)
    self.assertEqual(restored.policy_version, original.policy_version)
    self.assertEqual(restored.source_metadata, original.source_metadata)
    self.assertEqual(restored.extra_config, original.extra_config)
    self.assertEqual(restored.metadata, original.metadata)

  def test_sampling_request_round_trips_through_cloudpickle(self):
    original = _sampling_request()
    restored = cloudpickle.loads(cloudpickle.dumps(original))

    self.assertEqual(restored.request_id, original.request_id)
    self.assertEqual(restored.prompt, original.prompt)
    self.assertEqual(restored.metadata, original.metadata)
    self.assertIsNotNone(restored.sampling_params)
    self.assertEqual(
        restored.sampling_params.max_tokens, original.sampling_params.max_tokens
    )
    self.assertEqual(
        restored.sampling_params.temperature,
        original.sampling_params.temperature,
    )

  def test_sampling_response_round_trips_through_cloudpickle(self):
    original = _sample_response()
    restored = cloudpickle.loads(cloudpickle.dumps(original))

    self.assertEqual(restored.request_id, original.request_id)
    self.assertEqual(restored.text, original.text)
    self.assertEqual(restored.finish_reason, original.finish_reason)
    self.assertEqual(restored.metadata, original.metadata)
    self.assertIsNone(restored.error)
    np.testing.assert_array_equal(restored.token_ids, original.token_ids)
    np.testing.assert_array_equal(
        restored.prompt_token_ids, original.prompt_token_ids
    )
    np.testing.assert_allclose(restored.logprobs, original.logprobs)
    np.testing.assert_array_equal(
        restored.routed_experts, original.routed_experts
    )

  def test_sampling_response_routed_experts(self):
    # Shape: [Batch, Layers, Length, Top K]
    routed_experts = np.arange(2 * 4 * 10 * 2, dtype=np.int32).reshape(
        (2, 4, 10, 2)
    )
    response = base_sampler_lib.SamplingResponse(
        routed_experts=routed_experts
    )
    self.assertEqual(response.routed_experts.shape, (2, 4, 10, 2))
    np.testing.assert_array_equal(response.routed_experts, routed_experts)

  def test_sampling_response_enforces_shapes(self):
    with self.assertRaisesRegex(
        ValueError, "logprobs shape .* != token_ids shape"
    ):
      base_sampler_lib.SamplingResponse(
          token_ids=np.array([1, 2, 3]),
          logprobs=np.array([-0.1, -0.2]),
      )

  def test_load_info_defaults(self):
    load_info = base_sampler_lib.LoadInfo()
    self.assertEqual(load_info.num_requests_waiting, 0)
    self.assertEqual(load_info.num_requests_running, 0)
    self.assertEqual(load_info.kv_cache_usage_perc, 0.0)
    self.assertEqual(load_info.request_id, "")
    self.assertIsNone(load_info.error)
    self.assertEqual(load_info.metadata, {})

  def test_load_info_round_trips_through_cloudpickle(self):
    original = _load_info()
    restored = cloudpickle.loads(cloudpickle.dumps(original))

    self.assertEqual(restored.request_id, original.request_id)
    self.assertEqual(
        restored.num_requests_waiting, original.num_requests_waiting
    )
    self.assertEqual(
        restored.num_requests_running, original.num_requests_running
    )
    self.assertEqual(
        restored.kv_cache_usage_perc, original.kv_cache_usage_perc
    )
    self.assertEqual(restored.metadata, original.metadata)


if __name__ == "__main__":
  absltest.main()
