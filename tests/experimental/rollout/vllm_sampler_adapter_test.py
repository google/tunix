# Copyright 2026 Google LLC
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

"""Tests for VllmSamplerAdapter with tpu-inference RLVllmSampler."""

import asyncio
from types import SimpleNamespace
from unittest import mock

from absl.testing import absltest
import numpy as np
from tunix.experimental.rollout import sampler as base_sampler_lib
from tunix.experimental.rollout import vllm_sampler_adapter


class VllmSamplerAdapterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_sampler_instance = mock.AsyncMock()
    self.sampler_adapter = vllm_sampler_adapter.VllmSamplerAdapter(
        server_id="vllm_slice_01",
        sampler_instance=self.mock_sampler_instance,
    )

  def test_single_sampling_request(self):
    mock_response = SimpleNamespace(
        request_id="req_01",
        text="completion result",
        token_ids=np.array([101, 102, 103], dtype=np.int32),
        logprobs=np.array([-0.1, -0.2, -0.3], dtype=np.float32),
        finish_reason="stop",
        routed_experts=None,
        error=None,
    )
    self.mock_sampler_instance.sample.return_value = mock_response

    req = base_sampler_lib.SamplingRequest(
        request_id="req_01",
        prompt="Test prompt",
        sampling_params=base_sampler_lib.SamplingParams(
            max_tokens=16,
            temperature=0.7,
            return_logprobs=True,
        ),
    )
    response = asyncio.run(self.sampler_adapter.sample(req))
    self.assertIsInstance(response, base_sampler_lib.SamplingResponse)
    self.assertEqual(response.request_id, "req_01")
    self.assertEqual(response.text, "completion result")
    np.testing.assert_array_equal(response.token_ids, [101, 102, 103])
    np.testing.assert_allclose(response.logprobs, [-0.1, -0.2, -0.3])
    self.assertEqual(response.finish_reason, "stop")
    self.mock_sampler_instance.sample.assert_called_once_with(req)

  def test_batch_sampling_requests(self):
    mock_responses = [
        SimpleNamespace(
            request_id="req_a",
            text="completion a",
            token_ids=np.array([10, 20], dtype=np.int32),
            logprobs=np.array([-0.5, -0.6], dtype=np.float32),
            finish_reason="stop",
            routed_experts=None,
            error=None,
        ),
        SimpleNamespace(
            request_id="req_b",
            text="completion b",
            token_ids=np.array([30, 40], dtype=np.int32),
            logprobs=np.array([-0.7, -0.8], dtype=np.float32),
            finish_reason="length",
            routed_experts=None,
            error=None,
        ),
    ]
    self.mock_sampler_instance.sample.return_value = mock_responses

    reqs = [
        base_sampler_lib.SamplingRequest(
            request_id="req_a",
            prompt="Prompt A",
        ),
        base_sampler_lib.SamplingRequest(
            request_id="req_b",
            prompt="Prompt B",
        ),
    ]
    responses = asyncio.run(self.sampler_adapter.sample(reqs))
    self.assertIsInstance(responses, list)
    self.assertLen(responses, 2)
    self.assertEqual(responses[0].request_id, "req_a")
    self.assertEqual(responses[0].text, "completion a")
    self.assertEqual(responses[1].request_id, "req_b")
    self.assertEqual(responses[1].text, "completion b")
    self.assertEqual(responses[1].finish_reason, "length")

  def test_lifecycle_delegations(self):
    self.mock_sampler_instance.start.return_value = None
    self.mock_sampler_instance.stop.return_value = True
    self.mock_sampler_instance.pause.return_value = True
    self.mock_sampler_instance.resume.return_value = True
    self.mock_sampler_instance.get_mesh.return_value = "mock_mesh"

    asyncio.run(self.sampler_adapter.start())
    self.mock_sampler_instance.start.assert_called_once()

    asyncio.run(self.sampler_adapter.stop())
    self.mock_sampler_instance.stop.assert_called_once()

    asyncio.run(self.sampler_adapter.pause())
    self.mock_sampler_instance.pause.assert_called_once()

    asyncio.run(self.sampler_adapter.resume())
    self.mock_sampler_instance.resume.assert_called_once()

    mesh = asyncio.run(self.sampler_adapter.get_mesh())
    self.assertEqual(mesh, "mock_mesh")

  def test_weight_sync_delegations(self):
    mock_sync = mock.MagicMock()
    mock_sync.bound = True
    mock_sync.work_unit_metadata.return_value = "mock_work_unit_metadata"
    mock_sync.checksums.return_value = {"tensor": 1.0}
    mock_sync.metrics.return_value = {"transfer_ms": 10}
    self.sampler_adapter._synchronizers = [mock_sync]

    meta = asyncio.run(self.sampler_adapter.get_weight_sync_metadata())
    self.assertEqual(meta, ["mock_work_unit_metadata"])

    sync_req = base_sampler_lib.WeightSyncRequest(
        policy_version=1, extra_config={"req_id": "r1", "uuid": 1}
    )
    res_pre = asyncio.run(self.sampler_adapter.pre_weight_sync(sync_req))
    self.assertTrue(res_pre)
    self.mock_sampler_instance.pause.assert_called_once()

    res_sync = asyncio.run(self.sampler_adapter.weight_sync(sync_req))
    self.assertTrue(res_sync)
    mock_sync.h2d.assert_called_once()

    res_post = asyncio.run(self.sampler_adapter.post_weight_sync(sync_req))
    self.assertEqual(res_post, 1)
    self.mock_sampler_instance.resume.assert_called_once()

    status = asyncio.run(self.sampler_adapter.get_weight_sync_status())
    self.assertEqual(status.get("phase"), "committed")

    # Test abort path on next round (with higher uuid)
    sync_req_2 = base_sampler_lib.WeightSyncRequest(
        policy_version=2, extra_config={"req_id": "r2", "uuid": 2}
    )
    asyncio.run(self.sampler_adapter.pre_weight_sync(sync_req_2))
    res_abort = asyncio.run(self.sampler_adapter.abort_weight_sync(sync_req_2))
    self.assertTrue(res_abort)
    status_2 = asyncio.run(self.sampler_adapter.get_weight_sync_status())
    self.assertEqual(status_2.get("phase"), "aborted")

  def test_get_load_info(self):
    self.mock_sampler_instance.get_load_info.return_value = SimpleNamespace(
        num_requests_waiting=3,
        num_requests_running=1,
        kv_cache_usage_perc=25.5,
    )
    load_info = asyncio.run(self.sampler_adapter.get_load_info())
    self.assertIsInstance(load_info, base_sampler_lib.LoadInfo)
    self.assertEqual(load_info.num_requests_waiting, 3)
    self.assertEqual(load_info.num_requests_running, 1)
    self.assertAlmostEqual(load_info.kv_cache_usage_perc, 25.5)

  def test_migrate_kv_cache(self):
    self.mock_sampler_instance.migrate_kv_cache.return_value = True
    res = asyncio.run(
        self.sampler_adapter.migrate_kv_cache(
            source_server_id="src_0",
            target_server_id="dst_0",
            token_ids=[1, 2, 3],
            route_key="key_1",
        )
    )
    self.assertTrue(res)
    self.mock_sampler_instance.migrate_kv_cache.assert_called_once_with(
        route_key="key_1",
        source_server_id="src_0",
        target_server_id="dst_0",
        token_ids=[1, 2, 3],
    )

  def test_uninitialized_raises(self):
    uninit = vllm_sampler_adapter.VllmSamplerAdapter(server_id="empty")
    with self.assertRaises(RuntimeError):
      asyncio.run(uninit.sample(base_sampler_lib.SamplingRequest(prompt="hi")))
    with self.assertRaises(RuntimeError):
      asyncio.run(uninit.stop())
    with self.assertRaises(RuntimeError):
      asyncio.run(uninit.pause())
    with self.assertRaises(RuntimeError):
      asyncio.run(uninit.resume())
    with self.assertRaises(RuntimeError):
      asyncio.run(uninit.get_mesh())
    with self.assertRaises(RuntimeError):
      asyncio.run(uninit.get_weight_sync_metadata())
    with self.assertRaises(RuntimeError):
      asyncio.run(uninit.pre_weight_sync(None))
    with self.assertRaises(RuntimeError):
      asyncio.run(uninit.weight_sync(None))
    with self.assertRaises(RuntimeError):
      asyncio.run(uninit.post_weight_sync(None))
    with self.assertRaises(RuntimeError):
      asyncio.run(uninit.get_transfer_status("req"))
    with self.assertRaises(RuntimeError):
      asyncio.run(uninit.get_load_info())
    with self.assertRaises(RuntimeError):
      asyncio.run(
          uninit.migrate_kv_cache(
              source_server_id="s", target_server_id="t", token_ids=[1]
          )
      )

  def test_sample_none_requests_raises(self):
    with self.assertRaises(ValueError):
      asyncio.run(self.sampler_adapter.sample(None))

  def test_alias_vllm_inference_sampler_adapter(self):
    self.assertIs(
        vllm_sampler_adapter.VllmInferenceSamplerAdapter,
        vllm_sampler_adapter.VllmSamplerAdapter,
    )


if __name__ == "__main__":
  absltest.main()

