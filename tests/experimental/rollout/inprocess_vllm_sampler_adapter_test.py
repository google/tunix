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

"""Tests for InprocessVllmSamplerAdapter with Tunix VllmSampler and Raiden delegate."""

import asyncio
from unittest import mock
from absl.testing import absltest
import numpy as np
from tunix.experimental.rollout import inprocess_vllm_sampler_adapter
from tunix.experimental.rollout import sampler as base_sampler_lib
from tunix.experimental.weight_sync import raiden_weight_sync_delegate
from tunix.experimental.weight_sync import weight_sync
from tunix.generate import base_sampler


class InprocessVllmSamplerAdapterTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_vllm_sampler = mock.MagicMock()
    self.mock_vllm_sampler.return_value = base_sampler.SamplerOutput(
        text=["completion 1"],
        logits=None,
        tokens=[np.array([10, 20, 30], dtype=np.int32)],
        padded_prompt_tokens=np.array([[1, 2]], dtype=np.int32),
        logprobs=None,
    )
    self.mock_vllm_sampler.mesh = "mock_mesh"
    self.mock_vllm_lib = mock.MagicMock()
    self.mock_vllm_lib.VllmSampler.return_value = self.mock_vllm_sampler

    self.patcher = mock.patch.object(
        inprocess_vllm_sampler_adapter,
        "_get_vllm_sampler_cls",
        return_value=self.mock_vllm_lib,
    )
    self.patcher.start()

    self.mock_tokenizer = mock.MagicMock()
    self.mock_config = mock.MagicMock()
    self.mock_config.enable_raiden = False

    self.sampler_adapter = (
        inprocess_vllm_sampler_adapter.InprocessVllmSamplerAdapter(
            server_id="vllm_slice_01",
            tokenizer=self.mock_tokenizer,
            config=self.mock_config,
        )
    )

  def tearDown(self):
    self.patcher.stop()
    super().tearDown()

  def test_implements_sampler_protocol(self):
    self.assertIsInstance(self.sampler_adapter, base_sampler_lib.Sampler)

  def test_explicit_weight_sync_mode_overrides_config(self):
    fallback_config = mock.MagicMock()
    fallback_config.weight_sync_mode = weight_sync.WeightSyncMode.FALLBACK
    mock_delegate = mock.MagicMock(
        spec=raiden_weight_sync_delegate.RaidenWeightSyncDelegate
    )
    adapter = inprocess_vllm_sampler_adapter.InprocessVllmSamplerAdapter(
        server_id="vllm_raiden_slice",
        tokenizer=self.mock_tokenizer,
        config=fallback_config,
        raiden_sync_delegate=mock_delegate,
        weight_sync_mode=weight_sync.WeightSyncMode.RAIDEN,
    )
    self.assertEqual(
        adapter.weight_sync_mode, weight_sync.WeightSyncMode.RAIDEN
    )
    self.assertTrue(adapter.enable_raiden)

  def test_lifecycle_methods(self):
    self.assertTrue(asyncio.run(self.sampler_adapter.start()))
    self.assertTrue(asyncio.run(self.sampler_adapter.pause()))
    self.assertTrue(asyncio.run(self.sampler_adapter.resume()))
    self.assertEqual(asyncio.run(self.sampler_adapter.get_mesh()), "mock_mesh")
    self.assertTrue(asyncio.run(self.sampler_adapter.stop()))
    self.mock_vllm_sampler.stop.assert_called_once()

  def test_single_sampling_request(self):
    req = base_sampler_lib.SamplingRequest(
        request_id="vllm_req_01",
        prompt="hello vllm",
        sampling_params=base_sampler_lib.SamplingParams(
            max_tokens=16,
            temperature=0.7,
        ),
    )
    response = asyncio.run(self.sampler_adapter.sample(req))
    self.assertIsInstance(response, base_sampler_lib.SamplingResponse)
    self.assertEqual(response.request_id, "vllm_req_01")
    self.assertEqual(response.text, "completion 1")
    np.testing.assert_array_equal(response.prompt_token_ids, [1, 2])
    np.testing.assert_array_equal(response.token_ids, [10, 20, 30])
    self.mock_vllm_sampler.assert_called_once()

  def test_batch_sampling_requests(self):
    self.mock_vllm_sampler.return_value = base_sampler.SamplerOutput(
        text=["completion 1", "completion 2"],
        logits=None,
        tokens=[
            np.array([10, 20], dtype=np.int32),
            np.array([40, 50], dtype=np.int32),
        ],
        padded_prompt_tokens=np.array([[1, 2], [3, 4]], dtype=np.int32),
        logprobs=None,
    )
    reqs = [
        base_sampler_lib.SamplingRequest(
            request_id="req_1",
            prompt="prompt 1",
        ),
        base_sampler_lib.SamplingRequest(
            request_id="req_2",
            prompt="prompt 2",
        ),
    ]
    responses = asyncio.run(self.sampler_adapter.sample(reqs))
    self.assertIsInstance(responses, list)
    self.assertLen(responses, 2)
    self.assertEqual(responses[0].text, "completion 1")
    self.assertEqual(responses[1].text, "completion 2")
    np.testing.assert_array_equal(responses[0].prompt_token_ids, [1, 2])
    np.testing.assert_array_equal(responses[1].prompt_token_ids, [3, 4])

  def test_weight_sync_without_raiden_delegate(self):
    mock_weights = {"layer1": "weights"}
    req = base_sampler_lib.WeightSyncRequest(weights=mock_weights)
    res = asyncio.run(self.sampler_adapter.weight_sync(sync_request=req))
    self.assertTrue(res)
    self.mock_vllm_sampler.update_params.assert_called_once_with(mock_weights)

    # Missing sync_request should raise ValueError
    with self.assertRaises(ValueError):
      asyncio.run(self.sampler_adapter.weight_sync(sync_request=None))

    # Missing weights in sync_request should raise ValueError
    empty_req = base_sampler_lib.WeightSyncRequest()
    with self.assertRaises(ValueError):
      asyncio.run(self.sampler_adapter.weight_sync(sync_request=empty_req))

    self.assertIsNone(asyncio.run(self.sampler_adapter.bind_weight_sync()))
    self.assertTrue(asyncio.run(self.sampler_adapter.pre_weight_sync()))
    self.assertTrue(asyncio.run(self.sampler_adapter.post_weight_sync()))
    with self.assertRaises(NotImplementedError):
      asyncio.run(self.sampler_adapter.get_weight_sync_metadata())

  def test_weight_sync_with_raiden_delegate(self):
    mock_delegate = mock.MagicMock(
        spec=raiden_weight_sync_delegate.RaidenWeightSyncDelegate
    )
    mock_delegate.is_bounded.return_value = False
    mock_delegate.bind_weight_sync = mock.AsyncMock(return_value=True)
    mock_delegate.get_weight_sync_metadata = mock.AsyncMock(
        return_value=[{"unit": "rollout"}]
    )
    mock_delegate.pre_weight_sync = mock.AsyncMock(return_value=True)
    mock_delegate.weight_sync = mock.AsyncMock(return_value=5)
    mock_delegate.post_weight_sync = mock.AsyncMock(return_value=True)

    fake_transformer_state = {"param": "tensor"}
    self.mock_vllm_sampler.transformer_state = fake_transformer_state

    raiden_config = mock.MagicMock()
    raiden_config.weight_sync_mode = weight_sync.WeightSyncMode.RAIDEN

    raiden_adapter = inprocess_vllm_sampler_adapter.InprocessVllmSamplerAdapter(
        server_id="vllm_raiden_slice",
        tokenizer=self.mock_tokenizer,
        config=raiden_config,
        raiden_sync_delegate=mock_delegate,
    )

    sync_req = base_sampler_lib.WeightSyncRequest(policy_version=5)

    # 1. bind_weight_sync
    asyncio.run(raiden_adapter.bind_weight_sync(sync_req))
    mock_delegate.bind_weight_sync.assert_awaited_once_with(
        sync_request=sync_req, state=fake_transformer_state
    )
    mock_delegate.is_bounded.return_value = True

    # 2. get_weight_sync_metadata
    metadata = asyncio.run(raiden_adapter.get_weight_sync_metadata())
    self.assertEqual(metadata, [{"unit": "rollout"}])

    # 3. pre_weight_sync
    self.assertTrue(asyncio.run(raiden_adapter.pre_weight_sync(sync_req)))
    mock_delegate.pre_weight_sync.assert_awaited_once_with(
        sync_request=sync_req
    )

    # 4. weight_sync
    version = asyncio.run(raiden_adapter.weight_sync(sync_req))
    self.assertEqual(version, 5)
    mock_delegate.weight_sync.assert_awaited_once_with(sync_request=sync_req)

    # 5. post_weight_sync
    self.assertTrue(asyncio.run(raiden_adapter.post_weight_sync(sync_req)))
    mock_delegate.post_weight_sync.assert_awaited_once_with(
        sync_request=sync_req
    )

  def test_raiden_bind_is_idempotent_when_already_bound(self):
    mock_delegate = mock.MagicMock(
        spec=raiden_weight_sync_delegate.RaidenWeightSyncDelegate
    )
    mock_delegate.is_bounded.return_value = True
    mock_delegate.bind_weight_sync = mock.AsyncMock(return_value=True)
    self.mock_vllm_sampler.transformer_state = {"param": "tensor"}

    raiden_config = mock.MagicMock()
    raiden_config.weight_sync_mode = weight_sync.WeightSyncMode.RAIDEN

    raiden_adapter = inprocess_vllm_sampler_adapter.InprocessVllmSamplerAdapter(
        server_id="vllm_raiden_slice",
        tokenizer=self.mock_tokenizer,
        config=raiden_config,
        raiden_sync_delegate=mock_delegate,
    )

    self.assertTrue(asyncio.run(raiden_adapter.bind_weight_sync()))
    mock_delegate.bind_weight_sync.assert_not_awaited()

  def test_raiden_bind_without_transformer_state_raises(self):
    mock_delegate = mock.MagicMock(
        spec=raiden_weight_sync_delegate.RaidenWeightSyncDelegate
    )
    mock_delegate.is_bounded.return_value = False
    # Ensure vllm_sampler has no transformer_state
    if hasattr(self.mock_vllm_sampler, "transformer_state"):
      del self.mock_vllm_sampler.transformer_state

    raiden_config = mock.MagicMock()
    raiden_config.weight_sync_mode = weight_sync.WeightSyncMode.RAIDEN

    raiden_adapter = inprocess_vllm_sampler_adapter.InprocessVllmSamplerAdapter(
        server_id="vllm_raiden_slice",
        tokenizer=self.mock_tokenizer,
        config=raiden_config,
        raiden_sync_delegate=mock_delegate,
    )

    with self.assertRaisesRegex(RuntimeError, "transformer_state"):
      asyncio.run(raiden_adapter.bind_weight_sync())

  def test_other_sampler_methods(self):
    self.assertEqual(
        asyncio.run(self.sampler_adapter.get_transfer_status("req_1")),
        "SUCCESS",
    )
    self.assertTrue(
        asyncio.run(
            self.sampler_adapter.migrate_kv_cache(
                source_server_id="s1",
                target_server_id="s2",
                token_ids=[1, 2, 3],
            )
        )
    )
    load_info = asyncio.run(self.sampler_adapter.get_load_info())
    self.assertIsInstance(load_info, base_sampler_lib.LoadInfo)

  def test_uninitialized_sampler_raises(self):
    uninit = inprocess_vllm_sampler_adapter.InprocessVllmSamplerAdapter(
        server_id="empty"
    )
    with self.assertRaises(RuntimeError):
      asyncio.run(
          uninit.sample(base_sampler_lib.SamplingRequest(prompt="test"))
      )

  def test_sample_none_requests_raises(self):
    with self.assertRaises(ValueError):
      asyncio.run(self.sampler_adapter.sample(None))


if __name__ == "__main__":
  absltest.main()
