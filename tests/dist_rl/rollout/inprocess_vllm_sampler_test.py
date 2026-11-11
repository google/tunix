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

"""Tests for InprocessVllmSampler with Tunix VllmSampler and Raiden delegate."""

import asyncio
from unittest import mock

from absl.testing import absltest
import numpy as np
from tunix.common import configs
from tunix.dist_rl.rollout import inprocess_vllm_sampler
from tunix.dist_rl.rollout import sampler as base_sampler_lib
from tunix.dist_rl.weight_sync import raiden_weight_sync_delegate
from tunix.dist_rl.weight_sync import weight_sync
from tunix.generate import base_sampler
from tunix.generate import mappings


class InprocessVllmSamplerTest(absltest.TestCase):

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
        inprocess_vllm_sampler,
        "_get_vllm_sampler_cls",
        return_value=self.mock_vllm_lib,
    )
    self.patcher.start()

    self.mock_tokenizer = mock.MagicMock()
    self.config = configs.RolloutConfig(
        weight_sync_mode=weight_sync.WeightSyncMode.IN_MEMORY,
        tensor_parallel_size=1,
        data_parallel_size=1,
        temperature=0.7,
    )

    self.sampler = inprocess_vllm_sampler.InprocessVllmSampler(
        server_id="vllm_slice_01",
        tokenizer=self.mock_tokenizer,
        config=self.config,
    )
    self.sampler.initialize()

  def tearDown(self):
    self.patcher.stop()
    super().tearDown()

  def test_implements_sampler_protocol(self):
    self.assertIsInstance(self.sampler, base_sampler_lib.Sampler)

  def test_missing_tokenizer_raises(self):
    with self.assertRaises(TypeError):
      inprocess_vllm_sampler.InprocessVllmSampler(  # pyrefly: ignore[missing-argument]
          server_id="vllm_slice_01",
          config=self.config,
      )

  def test_real_raiden_delegate_instantiation(self):
    raiden_config = configs.RolloutConfig(
        weight_sync_mode=weight_sync.WeightSyncMode.RAIDEN,
        tensor_parallel_size=1,
        data_parallel_size=1,
    )
    sampler = inprocess_vllm_sampler.InprocessVllmSampler(
        server_id="vllm_raiden_slice",
        tokenizer=self.mock_tokenizer,
        config=raiden_config,
    )
    self.assertTrue(sampler.enable_raiden)
    self.assertIsNotNone(sampler.raiden_sync_delegate)
    self.assertEqual(
        sampler.raiden_sync_delegate._synchronizers[0].job_name,
        "vllm_raiden_slice",
    )
    self.assertFalse(sampler.raiden_sync_delegate.is_bounded())

  def test_weight_sync_mode_read_from_rollout_config(self):
    sampler = inprocess_vllm_sampler.InprocessVllmSampler(
        server_id="rollout_cfg",
        tokenizer=self.mock_tokenizer,
        config=configs.RolloutConfig(weight_sync_mode="none"),
    )
    self.assertEqual(
        sampler.config.weight_sync_mode, weight_sync.WeightSyncMode.NONE
    )
    self.assertFalse(sampler.enable_raiden)
    self.assertIsNone(sampler.raiden_sync_delegate)

    invalid_cfg = configs.RolloutConfig()
    invalid_cfg.weight_sync_mode = "invalid"  # type: ignore[assignment]
    with self.assertRaises(ValueError):
      inprocess_vllm_sampler.InprocessVllmSampler(
          server_id="bad_mode",
          tokenizer=self.mock_tokenizer,
          config=invalid_cfg,
      )

  def test_initialize_builds_vllm_config_from_rollout_config(self):
    mesh = mock.MagicMock()
    config = configs.RolloutConfig(
        weight_sync_mode="none",
        tensor_parallel_size=2,
        data_parallel_size=4,
        return_logprobs=True,
        return_routed_experts=True,
        eos_tokens=[7],
        partial_rollout=True,
        rollout_vllm_model_version="/models/qwen",
        rollout_vllm_hbm_utilization=0.6,
        rollout_vllm_init_with_random_weights=False,
        rollout_vllm_max_num_seqs=32,
        rollout_vllm_kwargs={"max_model_len": 2048, "dtype": "bfloat16"},
        rollout_vllm_sampling_kwargs={"skip_special_tokens": False},
    )
    self.mock_vllm_lib.VllmConfig.reset_mock()
    sampler = inprocess_vllm_sampler.InprocessVllmSampler(
        server_id="engine_cfg",
        tokenizer=self.mock_tokenizer,
        config=config,
        mesh=mesh,
    )
    sampler.initialize()

    vllm_config_kwargs = self.mock_vllm_lib.VllmConfig.call_args.kwargs
    # Concurrent sample() threads need the single-engine-loop driver.
    self.assertTrue(vllm_config_kwargs["server_mode"])
    self.assertIs(vllm_config_kwargs["mesh"], mesh)
    self.assertEqual(vllm_config_kwargs["tensor_parallel_size"], 2)
    self.assertEqual(vllm_config_kwargs["data_parallel_size"], 4)
    self.assertTrue(vllm_config_kwargs["return_logprobs"])
    self.assertTrue(vllm_config_kwargs["return_routed_experts"])
    self.assertEqual(vllm_config_kwargs["eos_tokens"], [7])
    self.assertTrue(vllm_config_kwargs["partial_rollout"])
    self.assertEqual(vllm_config_kwargs["hbm_utilization"], 0.6)
    self.assertFalse(vllm_config_kwargs["init_with_random_weights"])
    self.assertEqual(
        vllm_config_kwargs["sampling_kwargs"], {"skip_special_tokens": False}
    )
    self.assertEqual(
        vllm_config_kwargs["mapping_config"], mappings.MappingConfig()
    )
    engine_kwargs = vllm_config_kwargs["engine_kwargs"]
    self.assertEqual(engine_kwargs["model"], "/models/qwen")
    self.assertEqual(engine_kwargs["max_num_seqs"], 32)
    self.assertEqual(engine_kwargs["max_model_len"], 2048)
    self.assertEqual(engine_kwargs["dtype"], "bfloat16")
    self.assertEqual(engine_kwargs["max_logprobs"], 1)
    self.mock_vllm_lib.VllmSampler.assert_called_with(
        tokenizer=sampler.tokenizer,
        config=self.mock_vllm_lib.VllmConfig.return_value,
    )

  def test_initialize_passes_the_configured_mapping_config(self):
    mapping_config = mappings.MappingConfig()
    sampler = inprocess_vllm_sampler.InprocessVllmSampler(
        server_id="mapping_cfg",
        tokenizer=self.mock_tokenizer,
        config=configs.RolloutConfig(
            tensor_parallel_size=1,
            data_parallel_size=1,
            rollout_mapping_config=mapping_config,
        ),
    )
    sampler.initialize()
    self.assertIs(
        self.mock_vllm_lib.VllmConfig.call_args.kwargs["mapping_config"],
        mapping_config,
    )

  def test_initialize_takes_engine_defaults_from_rollout_config(self):
    with mock.patch.dict("os.environ", {"TPU_MULTIHOST_BACKEND": "ray"}):
      sampler = inprocess_vllm_sampler.InprocessVllmSampler(
          server_id="engine_defaults",
          tokenizer=self.mock_tokenizer,
          config=configs.RolloutConfig(
              kv_cache_size=4096,
              tensor_parallel_size=1,
              data_parallel_size=1,
              rollout_vllm_additional_config={
                  "maxtext_config": {
                      "weight_dtype": "bfloat16",
                      "allow_split": True,
                  }
              },
              rollout_vllm_kwargs={
                  "additional_config": {
                      "sharding": {"ep": 4},
                      "maxtext_config": {"allow_split": False},
                  }
              },
          ),
      )
      sampler.initialize()

    vllm_config_kwargs = self.mock_vllm_lib.VllmConfig.call_args.kwargs
    self.assertEqual(vllm_config_kwargs["tpu_backend_type"], "jax")
    self.assertEqual(vllm_config_kwargs["hbm_utilization"], 0.5)
    self.assertEqual(
        vllm_config_kwargs["additional_config"],
        {
            "sharding": {"ep": 4},
            "maxtext_config": {
                "weight_dtype": "bfloat16",
                "allow_split": False,
            },
        },
    )
    engine_kwargs = vllm_config_kwargs["engine_kwargs"]
    self.assertEqual(engine_kwargs["max_model_len"], 4096)
    self.assertEqual(engine_kwargs["distributed_executor_backend"], "ray")
    self.assertTrue(engine_kwargs["trust_remote_code"])

  def test_initialize_validates_parallelism_and_lora(self):
    with self.assertRaisesRegex(ValueError, "tensor_parallel_size"):
      inprocess_vllm_sampler.InprocessVllmSampler(
          server_id="bad_tp",
          tokenizer=self.mock_tokenizer,
          config=configs.RolloutConfig(),
      ).initialize()

    with self.assertRaisesRegex(
        ValueError, "LoRA is not supported with rollout data parallelism"
    ):
      inprocess_vllm_sampler.InprocessVllmSampler(
          server_id="bad_lora_dp",
          tokenizer=self.mock_tokenizer,
          config=configs.RolloutConfig(
              tensor_parallel_size=1,
              data_parallel_size=2,
              rollout_vllm_lora_config={"max_lora_rank": 8},
          ),
      ).initialize()

  def test_initialize_warns_when_server_mode_is_false(self):
    self.mock_vllm_lib.VllmConfig.reset_mock()
    sampler = inprocess_vllm_sampler.InprocessVllmSampler(
        server_id="warn_slice",
        tokenizer=self.mock_tokenizer,
        config=configs.RolloutConfig(
            tensor_parallel_size=1,
            data_parallel_size=1,
            rollout_vllm_server_mode=False,
        ),
    )
    with self.assertLogs(level="WARNING") as cm:
      sampler.initialize()
    self.assertTrue(
        any("rollout_vllm_server_mode=False" in msg for msg in cm.output)
    )
    self.assertTrue(
        self.mock_vllm_lib.VllmConfig.call_args.kwargs["server_mode"]
    )

    self.mock_vllm_lib.VllmConfig.reset_mock()
    inprocess_vllm_sampler.InprocessVllmSampler(
        server_id="extra_kwargs_slice",
        tokenizer=self.mock_tokenizer,
        config=configs.RolloutConfig(
            tensor_parallel_size=1,
            data_parallel_size=1,
            rollout_vllm_kwargs={
                "server_mode": False,
                "gpu_memory_utilization": 0.85,
            },
        ),
    ).initialize()
    vllm_cfg_kwargs = self.mock_vllm_lib.VllmConfig.call_args.kwargs
    self.assertNotIn("server_mode", vllm_cfg_kwargs["engine_kwargs"])
    self.assertEqual(vllm_cfg_kwargs["hbm_utilization"], 0.85)

  def test_sample_renders_chat_message_prompts_with_chat_template(self):
    self.mock_tokenizer.apply_chat_template.return_value = "<templated>"
    messages = [{"role": "user", "content": "hi"}]
    req = base_sampler_lib.SamplingRequest(
        prompt=messages,
        sampling_params=base_sampler_lib.SamplingParams(max_tokens=9),
    )
    asyncio.run(self.sampler.sample(req))

    self.mock_tokenizer.apply_chat_template.assert_called_once_with(
        messages, add_generation_prompt=True, tokenize=False
    )
    self.assertEqual(
        self.mock_vllm_sampler.call_args.kwargs["input_strings"],
        ["<templated>"],
    )

  def test_lifecycle_methods(self):
    self.assertTrue(asyncio.run(self.sampler.start()))
    self.assertTrue(asyncio.run(self.sampler.pause()))
    self.assertTrue(asyncio.run(self.sampler.resume()))
    self.assertEqual(asyncio.run(self.sampler.get_mesh()), "mock_mesh")
    self.assertTrue(asyncio.run(self.sampler.stop()))
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
    response = asyncio.run(self.sampler.sample(req))
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
    responses = asyncio.run(self.sampler.sample(reqs))
    self.assertIsInstance(responses, list)
    self.assertLen(responses, 2)
    self.assertEqual(responses[0].text, "completion 1")
    self.assertEqual(responses[1].text, "completion 2")
    np.testing.assert_array_equal(responses[0].prompt_token_ids, [1, 2])
    np.testing.assert_array_equal(responses[1].prompt_token_ids, [3, 4])

  def test_weight_sync_without_raiden_delegate(self):
    mock_weights = {"layer1": "weights"}
    req = base_sampler_lib.WeightSyncRequest(weights=mock_weights)
    res = asyncio.run(self.sampler.weight_sync(sync_request=req))
    self.assertTrue(res)
    self.mock_vllm_sampler.update_params.assert_called_once_with(mock_weights)

    # Missing sync_request should raise ValueError
    with self.assertRaises(ValueError):
      asyncio.run(self.sampler.weight_sync(sync_request=None))

    # Missing weights in sync_request should raise ValueError
    empty_req = base_sampler_lib.WeightSyncRequest()
    with self.assertRaises(ValueError):
      asyncio.run(self.sampler.weight_sync(sync_request=empty_req))

    self.assertIsNone(asyncio.run(self.sampler.bind_weight_sync()))
    self.assertTrue(asyncio.run(self.sampler.pre_weight_sync()))
    self.assertTrue(asyncio.run(self.sampler.post_weight_sync()))
    with self.assertRaises(NotImplementedError):
      asyncio.run(self.sampler.get_weight_sync_metadata())

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
    mock_delegate.abort_weight_sync = mock.AsyncMock(return_value=True)

    fake_transformer_state = {"param": "tensor"}
    self.mock_vllm_sampler.transformer_state = fake_transformer_state

    raiden_config = configs.RolloutConfig(
        weight_sync_mode=weight_sync.WeightSyncMode.RAIDEN,
        tensor_parallel_size=1,
        data_parallel_size=1,
    )

    raiden_sampler = inprocess_vllm_sampler.InprocessVllmSampler(
        server_id="vllm_raiden_slice",
        tokenizer=self.mock_tokenizer,
        config=raiden_config,
        raiden_sync_delegate=mock_delegate,
    )
    raiden_sampler.initialize()

    sync_req = base_sampler_lib.WeightSyncRequest(policy_version=5)

    # 1. bind_weight_sync
    asyncio.run(raiden_sampler.bind_weight_sync(sync_req))
    mock_delegate.bind_weight_sync.assert_awaited_once_with(
        sync_request=sync_req,
        state=fake_transformer_state,
        sampler=self.mock_vllm_sampler,
    )
    mock_delegate.is_bounded.return_value = True

    # 2. get_weight_sync_metadata
    metadata = asyncio.run(raiden_sampler.get_weight_sync_metadata())
    self.assertEqual(metadata, [{"unit": "rollout"}])

    # 3. pre_weight_sync
    self.assertTrue(asyncio.run(raiden_sampler.pre_weight_sync(sync_req)))
    mock_delegate.pre_weight_sync.assert_awaited_once_with(
        sync_request=sync_req
    )

    # 4. weight_sync
    version = asyncio.run(raiden_sampler.weight_sync(sync_req))
    self.assertEqual(version, 5)
    mock_delegate.weight_sync.assert_awaited_once_with(sync_request=sync_req)

    # 5. post_weight_sync
    self.assertTrue(asyncio.run(raiden_sampler.post_weight_sync(sync_req)))
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

    raiden_config = configs.RolloutConfig(
        weight_sync_mode=weight_sync.WeightSyncMode.RAIDEN,
        tensor_parallel_size=1,
        data_parallel_size=1,
    )

    raiden_sampler = inprocess_vllm_sampler.InprocessVllmSampler(
        server_id="vllm_raiden_slice",
        tokenizer=self.mock_tokenizer,
        config=raiden_config,
        raiden_sync_delegate=mock_delegate,
    )
    raiden_sampler.initialize()

    self.assertTrue(asyncio.run(raiden_sampler.bind_weight_sync()))
    mock_delegate.bind_weight_sync.assert_not_awaited()

  def test_raiden_bind_without_transformer_state_raises(self):
    mock_delegate = mock.MagicMock(
        spec=raiden_weight_sync_delegate.RaidenWeightSyncDelegate
    )
    mock_delegate.is_bounded.return_value = False
    if hasattr(self.mock_vllm_sampler, "transformer_state"):
      del self.mock_vllm_sampler.transformer_state

    raiden_config = configs.RolloutConfig(
        weight_sync_mode=weight_sync.WeightSyncMode.RAIDEN,
        tensor_parallel_size=1,
        data_parallel_size=1,
    )

    raiden_sampler = inprocess_vllm_sampler.InprocessVllmSampler(
        server_id="vllm_raiden_slice",
        tokenizer=self.mock_tokenizer,
        config=raiden_config,
        raiden_sync_delegate=mock_delegate,
    )
    raiden_sampler.initialize()

    with self.assertRaisesRegex(RuntimeError, "transformer_state"):
      asyncio.run(raiden_sampler.bind_weight_sync())

  def test_exposes_weights_and_cache_hooks_from_vllm_sampler(self):
    fake_state = {"w": 1}
    self.mock_vllm_sampler.transformer_state = fake_state
    self.assertIs(self.sampler.transformer_state, fake_state)
    self.assertIs(
        self.sampler._get_underlying_sampler(), self.mock_vllm_sampler
    )

    self.sampler.update_params({"w": 2})
    self.mock_vllm_sampler.update_params.assert_called_once_with(
        {"w": 2}, filter_types=None
    )
    self.mock_vllm_sampler.update_params.reset_mock()
    self.sampler.update_params({"w": 3}, filter_types=(int,))
    self.mock_vllm_sampler.update_params.assert_called_once_with(
        {"w": 3}, filter_types=(int,)
    )

    self.sampler.delete_cache()
    self.mock_vllm_sampler.delete_cache.assert_called_once_with()

    self.sampler.reset_prefix_cache()
    self.mock_vllm_sampler.reset_prefix_cache.assert_called_once_with()

    self.sampler.reinitialize_cache()
    self.mock_vllm_sampler.reinitialize_cache.assert_called_once_with()

    self.sampler.refresh_state_leaves()
    self.mock_vllm_sampler.refresh_state_leaves.assert_called_once_with()

    uninit = inprocess_vllm_sampler.InprocessVllmSampler(
        server_id="uninit_slice",
        tokenizer=self.mock_tokenizer,
        config=self.config,
    )
    self.assertIsNone(uninit._get_underlying_sampler())
    with self.assertRaisesRegex(RuntimeError, "not initialized"):
      _ = uninit.transformer_state
    with self.assertRaisesRegex(RuntimeError, "not initialized"):
      uninit.update_params({"w": 2})

  def test_other_sampler_methods(self):
    self.assertEqual(
        asyncio.run(self.sampler.get_transfer_status("req_1")),
        "SUCCESS",
    )
    self.assertTrue(
        asyncio.run(
            self.sampler.migrate_kv_cache(
                source_server_id="s1",
                target_server_id="s2",
                token_ids=[1, 2, 3],
            )
        )
    )
    load_info = asyncio.run(self.sampler.get_load_info())
    self.assertIsInstance(load_info, base_sampler_lib.LoadInfo)

  def test_uninitialized_sampler_raises(self):
    uninit = inprocess_vllm_sampler.InprocessVllmSampler(
        server_id="empty",
        tokenizer=self.mock_tokenizer,
        config=self.config,
    )
    with self.assertRaisesRegex(RuntimeError, "not initialized"):
      asyncio.run(
          uninit.sample(base_sampler_lib.SamplingRequest(prompt="test"))
      )

  def test_sample_none_requests_raises(self):
    with self.assertRaises(ValueError):
      asyncio.run(self.sampler.sample(None))

  def test_sample_raw_prompt_raises_type_error(self):
    with self.assertRaises(TypeError):
      asyncio.run(self.sampler.sample("raw prompt"))
    with self.assertRaises(TypeError):
      asyncio.run(self.sampler.sample(["raw prompt"]))

  def test_vllm_sampler_deferred_to_initialize(self):
    self.mock_vllm_lib.VllmSampler.reset_mock()
    sampler = inprocess_vllm_sampler.InprocessVllmSampler(
        server_id="deferred_slice",
        tokenizer=self.mock_tokenizer,
        config=self.config,
    )
    self.assertIsNone(sampler.vllm_sampler)
    self.mock_vllm_lib.VllmSampler.assert_not_called()

    sampler.initialize()
    self.assertIsNotNone(sampler.vllm_sampler)
    self.mock_vllm_lib.VllmSampler.assert_called_once()


class RoutedExpertsTest(absltest.TestCase):
  """Captured MoE routing must reach the matching response."""

  def _sampler(self, routed_experts, engine_captures=True):
    rows = len(routed_experts) if routed_experts else 1
    inner_sampler = mock.MagicMock()
    inner_sampler.return_value = base_sampler.SamplerOutput(
        text=["c"] * rows,
        logits=None,
        tokens=[np.array([10, 20, 30], dtype=np.int32)] * rows,
        padded_prompt_tokens=np.tile(
            np.array([1, 2], dtype=np.int32), (rows, 1)
        ),
        logprobs=None,
        routed_experts=routed_experts,
    )
    sampler = inprocess_vllm_sampler.InprocessVllmSampler(
        server_id="rollout",
        tokenizer=mock.MagicMock(),
        config=configs.RolloutConfig(
            weight_sync_mode=weight_sync.WeightSyncMode.IN_MEMORY,
            return_routed_experts=engine_captures,
        ),
    )
    sampler.vllm_sampler = inner_sampler
    return sampler

  def _requests(self, n):
    return [
        base_sampler_lib.SamplingRequest(
            request_id=f"req-{i}",
            prompt=np.array([1, 2], dtype=np.int32),
            sampling_params=base_sampler_lib.SamplingParams(
                max_tokens=3, return_routed_experts=True
            ),
        )
        for i in range(n)
    ]

  def test_routing_reaches_each_response_in_order(self):
    """Row i's routing must land on row i, or replay trains the wrong tokens."""
    routed = [np.full((3, 2, 2), i, dtype=np.int32) for i in range(2)]
    responses = asyncio.run(self._sampler(routed).sample(self._requests(2)))

    self.assertLen(responses, 2)
    for i, response in enumerate(responses):
      self.assertEqual(response.request_id, f"req-{i}")
      np.testing.assert_array_equal(response.routed_experts, routed[i])

  def test_warns_when_engine_cannot_capture(self):
    """Silently returning None would look like a dense model to the trainer."""
    sampler = self._sampler(None, engine_captures=False)
    with self.assertLogs(level="WARNING") as logs:
      asyncio.run(sampler.sample(self._requests(1)))
    self.assertTrue(
        any("return_routed_experts" in line for line in logs.output),
        f"expected a warning about the engine setting, got {logs.output}",
    )

  def test_forwards_routed_experts_prompt_start(self):
    """Cumulative prompt offset must be forwarded to the vLLM sampler as a list."""
    sampler = self._sampler(None)
    req = base_sampler_lib.SamplingRequest(
        request_id="req-0",
        prompt=np.array([1, 2, 3, 4], dtype=np.int32),
        sampling_params=base_sampler_lib.SamplingParams(
            max_tokens=3,
            return_routed_experts=True,
            routed_experts_prompt_start=12,
        ),
    )
    asyncio.run(sampler.sample([req]))
    call_kwargs = sampler.vllm_sampler.call_args.kwargs
    self.assertEqual(call_kwargs.get("routed_experts_prompt_start"), [12])

  def test_forwards_heterogeneous_batch_routed_experts_prompt_start(self):
    """Per-prompt offsets in heterogeneous batch must be preserved without collapse."""
    sampler = self._sampler([None, None])
    req0 = base_sampler_lib.SamplingRequest(
        request_id="req-0",
        prompt=np.array([1, 2, 3, 4], dtype=np.int32),
        sampling_params=base_sampler_lib.SamplingParams(
            max_tokens=3,
            return_routed_experts=True,
            routed_experts_prompt_start=8,
        ),
    )
    req1 = base_sampler_lib.SamplingRequest(
        request_id="req-1",
        prompt=np.array([1, 2, 3, 4, 5, 6], dtype=np.int32),
        sampling_params=base_sampler_lib.SamplingParams(
            max_tokens=3,
            return_routed_experts=True,
            routed_experts_prompt_start=14,
        ),
    )
    asyncio.run(sampler.sample([req0, req1]))
    call_kwargs = sampler.vllm_sampler.call_args.kwargs
    self.assertEqual(call_kwargs.get("routed_experts_prompt_start"), [8, 14])

  def test_concurrent_sample_calls_do_not_block_event_loop(self):
    """Concurrent sample() calls must run in parallel on the sampler's executor."""
    import threading  # pylint: disable=g-import-not-at-top

    num_concurrent = 4
    barrier = threading.Barrier(num_concurrent, timeout=5.0)

    def _blocking_vllm_call(**kwargs):
      del kwargs
      barrier.wait()
      return base_sampler.SamplerOutput(
          text=["parallel completion"],
          logits=None,
          tokens=[np.array([10, 20], dtype=np.int32)],
          padded_prompt_tokens=np.array([[1, 2]], dtype=np.int32),
          logprobs=None,
      )

    sampler = inprocess_vllm_sampler.InprocessVllmSampler(
        server_id="vllm_concurrent_slice",
        tokenizer=mock.MagicMock(),
        config=configs.RolloutConfig(),
        max_concurrency=num_concurrent,
    )
    sampler.vllm_sampler = mock.MagicMock(side_effect=_blocking_vllm_call)

    async def _run_concurrent():
      reqs = [
          base_sampler_lib.SamplingRequest(
              request_id=f"req-{i}",
              prompt=f"prompt {i}",
          )
          for i in range(num_concurrent)
      ]
      return await asyncio.gather(*(sampler.sample(r) for r in reqs))

    responses = asyncio.run(_run_concurrent())
    self.assertLen(responses, num_concurrent)
    for i, resp in enumerate(responses):
      self.assertEqual(resp.request_id, f"req-{i}")
      self.assertEqual(resp.text, "parallel completion")
    asyncio.run(sampler.stop())

  def test_sample_with_token_id_prompt_and_prompt_lengths(self):
    sampler = self._sampler(None)
    sampler.vllm_sampler.return_value = base_sampler.SamplerOutput(
        text=["completion"],
        logits=None,
        tokens=[np.array([30, 40], dtype=np.int32)],
        padded_prompt_tokens=np.array([[0, 0, 0, 10, 20]], dtype=np.int32),
        prompt_lengths=np.array([3], dtype=np.int32),
        logprobs=None,
    )
    req = base_sampler_lib.SamplingRequest(
        request_id="req_tok",
        prompt=np.array([0, 10, 20], dtype=np.int32),
    )
    resp = asyncio.run(sampler.sample(req))
    call_kwargs = sampler.vllm_sampler.call_args.kwargs
    self.assertIsNone(call_kwargs["input_strings"])
    self.assertLen(call_kwargs["prompt_token_ids"], 1)
    np.testing.assert_array_equal(
        call_kwargs["prompt_token_ids"][0], [0, 10, 20]
    )
    np.testing.assert_array_equal(resp.prompt_token_ids, [0, 10, 20])

  def test_hf_tokenizer_wrapped_with_tokenizer_adapter(self):
    from transformers import tokenization_utils_base  # pylint: disable=g-import-not-at-top
    from tunix.generate import tokenizer_adapter  # pylint: disable=g-import-not-at-top

    class _DummyHFTokenizer(tokenization_utils_base.PreTrainedTokenizerBase):
      pad_token_id = 0
      eos_token_id = 2

      def __init__(self):  # pylint: disable=super-init-not-called
        pass

    hf_tok = _DummyHFTokenizer()

    sampler = inprocess_vllm_sampler.InprocessVllmSampler(
        server_id="vllm_hf_tok_slice",
        tokenizer=hf_tok,
        config=configs.RolloutConfig(),
    )
    self.assertIsInstance(sampler.tokenizer, tokenizer_adapter.TokenizerAdapter)
    inner_sampler = mock.MagicMock()
    inner_sampler.return_value = base_sampler.SamplerOutput(
        text=["completion"],
        logits=None,
        tokens=[np.array([30, 40], dtype=np.int32)],
        padded_prompt_tokens=np.array([[0, 0, 10, 20]], dtype=np.int32),
        prompt_lengths=np.array([2], dtype=np.int32),
        logprobs=None,
    )
    sampler.vllm_sampler = inner_sampler
    req = base_sampler_lib.SamplingRequest(
        request_id="req_hf",
        prompt="hello world",
    )
    resp = asyncio.run(sampler.sample(req))
    np.testing.assert_array_equal(resp.prompt_token_ids, [10, 20])

  def test_forwards_cache_salt_scalar_and_per_prompt(self):
    sampler = self._sampler([None, None])
    req_single = base_sampler_lib.SamplingRequest(
        request_id="req-0",
        prompt="prompt 0",
        sampling_params=base_sampler_lib.SamplingParams(
            max_tokens=3,
            cache_salt="policy_v2",
        ),
    )
    asyncio.run(sampler.sample([req_single]))
    self.assertEqual(
        sampler.vllm_sampler.call_args.kwargs.get("cache_salt"), "policy_v2"
    )

    req_a = base_sampler_lib.SamplingRequest(
        request_id="req-a",
        prompt="prompt a",
        sampling_params=base_sampler_lib.SamplingParams(
            max_tokens=3,
            cache_salt="policy_v2",
        ),
    )
    req_b = base_sampler_lib.SamplingRequest(
        request_id="req-b",
        prompt="prompt b",
        sampling_params=base_sampler_lib.SamplingParams(
            max_tokens=3,
            cache_salt="policy_v3",
        ),
    )
    asyncio.run(sampler.sample([req_a, req_b]))
    self.assertEqual(
        sampler.vllm_sampler.call_args.kwargs.get("cache_salt"),
        ["policy_v2", "policy_v3"],
    )

  def test_weight_sync_pauses_and_resumes_vllm_sampler(self):
    mock_delegate = mock.MagicMock(
        spec=raiden_weight_sync_delegate.RaidenWeightSyncDelegate
    )
    mock_delegate.pre_weight_sync = mock.AsyncMock(return_value=True)
    mock_delegate.post_weight_sync = mock.AsyncMock(return_value=True)
    mock_delegate.abort_weight_sync = mock.AsyncMock(return_value=True)

    inner_sampler = mock.MagicMock()
    sampler = inprocess_vllm_sampler.InprocessVllmSampler(
        server_id="rollout",
        tokenizer=mock.MagicMock(),
        config=configs.RolloutConfig(
            weight_sync_mode=weight_sync.WeightSyncMode.RAIDEN,
            partial_rollout=True,
        ),
        raiden_sync_delegate=mock_delegate,
    )
    sampler.vllm_sampler = inner_sampler
    sync_req = base_sampler_lib.WeightSyncRequest(policy_version=2)

    asyncio.run(sampler.pre_weight_sync(sync_req))
    inner_sampler.pause.assert_called_once_with(mode="keep", clear_cache=False)

    asyncio.run(sampler.post_weight_sync(sync_req))
    inner_sampler.resume.assert_called_once()

    inner_sampler.reset_mock()
    asyncio.run(sampler.pre_weight_sync(sync_req))
    inner_sampler.pause.assert_called_once_with(mode="keep", clear_cache=False)
    asyncio.run(sampler.abort_weight_sync(sync_req))
    inner_sampler.resume.assert_called_once()


if __name__ == "__main__":
  absltest.main()
