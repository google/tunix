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
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Unit tests for RLVllmSampler dynamic duck typing and weight sync."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

import importlib.util
import sys
import types

if importlib.util.find_spec("vllm") is not None:
    from vllm.engine.arg_utils import AsyncEngineArgs  # pylint: disable=g-import-not-at-top
    from tunix.experimental.rollout.vllm_sampler_v2 import RLVllmSampler  # pylint: disable=g-import-not-at-top
else:

    class _StubAsyncEngineArgs(SimpleNamespace):

        def __init__(self, **kwargs):
            defaults = {
                "model": "",
                "tensor_parallel_size": 1,
                "data_parallel_size": 1,
                "expert_parallel_size": 1,
            }
            defaults.update(kwargs)
            super().__init__(**defaults)

    class _StubSamplingParams(SimpleNamespace):
        pass

    _vllm_mod = types.ModuleType("vllm")
    _vllm_envs = types.ModuleType("vllm.envs")
    _vllm_envs.VLLM_LOG_STATS_INTERVAL = 10.0
    _vllm_mod.envs = _vllm_envs
    _vllm_engine = types.ModuleType("vllm.engine")
    _vllm_arg_utils = types.ModuleType("vllm.engine.arg_utils")
    _vllm_arg_utils.AsyncEngineArgs = _StubAsyncEngineArgs
    _vllm_async_engine = types.ModuleType("vllm.engine.async_llm_engine")
    _vllm_async_engine.AsyncLLMEngine = MagicMock()
    _vllm_sampling_params = types.ModuleType("vllm.sampling_params")
    _vllm_sampling_params.SamplingParams = _StubSamplingParams

    with patch.dict(
        sys.modules,
        {
            "vllm": _vllm_mod,
            "vllm.envs": _vllm_envs,
            "vllm.engine": _vllm_engine,
            "vllm.engine.arg_utils": _vllm_arg_utils,
            "vllm.engine.async_llm_engine": _vllm_async_engine,
            "vllm.sampling_params": _vllm_sampling_params,
        },
    ):
        from tunix.experimental.rollout.vllm_sampler_v2 import RLVllmSampler  # pylint: disable=g-import-not-at-top

    AsyncEngineArgs = _StubAsyncEngineArgs


class TestRLVllmSamplerDuckTyping(unittest.TestCase):
    """Tests dynamic attribute handling of arbitrary request objects and dicts."""

    def test_duck_typed_request_processing(self):
        """Verifies that sample() handles raw objects with attributes or dicts."""
        SimpleNamespace(
            prompt="Solve 2+2",
            request_id="req_attr_1",
            sampling_params=SimpleNamespace(
                max_tokens=64,
                temperature=0.5,
                top_p=0.9,
                top_k=-1,
                stop_sequences=[],
                return_logprobs=True,
            ),
        )

        args = AsyncEngineArgs(model="Qwen/Qwen2.5-1.5B")
        sampler = RLVllmSampler(engine_args=args)
        self.assertIsNotNone(sampler)
        self.assertEqual(sampler.engine_args.model, "Qwen/Qwen2.5-1.5B")


class TestRLVllmSamplerInference(unittest.TestCase):
    """Tests sampling batch processing, text decoding, and logprob conversion."""

    def test_sample_with_mocked_engine(self):
        """Verifies full sample() execution flow with a mocked AsyncLLMEngine."""
        args = AsyncEngineArgs(model="Qwen/Qwen2.5-1.5B")
        sampler = RLVllmSampler(engine_args=args)

        # Construct mock AsyncLLMEngine
        mock_engine = MagicMock()

        async def mock_generate_stream(prompt, sampling_params, request_id):
            mock_output_choice = SimpleNamespace(
                text=f"Completion for {request_id}",
                token_ids=[101, 202, 303],
                cumulative_logprob=-0.45,
                finish_reason="stop",
                logprobs=[{
                    101: SimpleNamespace(logprob=-0.1)
                }, {
                    202: SimpleNamespace(logprob=-0.25)
                }, {
                    303: SimpleNamespace(logprob=-0.1)
                }],
            )
            step_out = SimpleNamespace(outputs=[mock_output_choice])
            yield step_out

        mock_engine.generate.side_effect = mock_generate_stream
        sampler._engine = mock_engine
        sampler._is_running = True

        async def run_sample_test():
            reqs = [
                SimpleNamespace(
                    prompt="What is GRPO?",
                    request_id="req_001",
                    sampling_params=SimpleNamespace(max_tokens=64,
                                                    temperature=0.7,
                                                    top_p=0.9,
                                                    return_logprobs=True),
                )
            ]
            results = await sampler.sample(reqs)
            self.assertEqual(len(results), 1)
            res = results[0]
            self.assertEqual(res.request_id, "req_001")
            self.assertEqual(res.text, "Completion for req_001")
            self.assertTrue(
                np.array_equal(res.token_ids,
                               np.array([101, 202, 303], dtype=np.int32)))
            self.assertIsNotNone(res.logprobs)
            self.assertAlmostEqual(res.cumulative_logprob, -0.45)
            self.assertTrue(hasattr(res, "routed_experts"))
            self.assertIsNone(res.error)

        asyncio.run(run_sample_test())

    def test_sample_raw_string_mode(self):
        """Verifies sampling when prompt lists are raw strings."""
        args = AsyncEngineArgs(model="Qwen/Qwen2.5-1.5B")
        sampler = RLVllmSampler(engine_args=args)

        mock_engine = MagicMock()

        async def mock_generate_stream(prompt, sampling_params, request_id):
            yield SimpleNamespace(outputs=[
                SimpleNamespace(text="Output text",
                                token_ids=[1, 2],
                                cumulative_logprob=-0.1,
                                finish_reason="stop",
                                logprobs=None)
            ])

        mock_engine.generate.side_effect = mock_generate_stream
        sampler._engine = mock_engine
        sampler._is_running = True

        async def run_raw_test():
            texts = await sampler.sample(
                ["Prompt string 1", "Prompt string 2"], max_tokens=16)
            self.assertEqual(len(texts), 2)
            self.assertEqual(texts[0], "Output text")
            self.assertEqual(texts[1], "Output text")

        asyncio.run(run_raw_test())

    def test_build_vllm_params_detokenizes_by_default(self):
        """Tunix SamplingParams carry no detokenize field; text must still be returned."""
        args = AsyncEngineArgs(model="Qwen/Qwen2.5-1.5B")
        sampler = RLVllmSampler(engine_args=args)
        req = SimpleNamespace(
            prompt="p",
            sampling_params=SimpleNamespace(max_tokens=8, temperature=1.0),
        )
        self.assertTrue(sampler._build_vllm_params(req, {}).detokenize)

    def test_build_vllm_params_detokenize_opt_out(self):
        """An explicit detokenize=False from the request or kwargs is honored."""
        args = AsyncEngineArgs(model="Qwen/Qwen2.5-1.5B")
        sampler = RLVllmSampler(engine_args=args)
        req = SimpleNamespace(
            prompt="p",
            sampling_params=SimpleNamespace(max_tokens=8, detokenize=False),
        )
        self.assertFalse(sampler._build_vllm_params(req, {}).detokenize)
        req_no_field = SimpleNamespace(
            prompt="p", sampling_params=SimpleNamespace(max_tokens=8)
        )
        self.assertFalse(
            sampler._build_vllm_params(req_no_field, {"detokenize": False})
            .detokenize
        )

    def test_build_vllm_params_forwards_routed_experts_prompt_start(self):
        """Multi-turn router replay offset reaches vLLM from request or kwargs."""
        from tunix.experimental.rollout import sampler as sampler_lib  # pylint: disable=g-import-not-at-top

        args = AsyncEngineArgs(model="Qwen/Qwen2.5-1.5B")
        sampler = RLVllmSampler(engine_args=args)
        # collector.py sets it on tunix SamplingParams and in the kwargs.
        req = SimpleNamespace(
            prompt=[1, 2, 3],
            sampling_params=sampler_lib.SamplingParams(
                max_tokens=8, routed_experts_prompt_start=17
            ),
        )
        self.assertEqual(
            sampler._build_vllm_params(
                req, {"routed_experts_prompt_start": 17}
            ).routed_experts_prompt_start,
            17,
        )
        # kwargs only: the tunix SamplingParams default of 0 must not mask it.
        req_default = SimpleNamespace(
            prompt=[1, 2, 3],
            sampling_params=sampler_lib.SamplingParams(max_tokens=8),
        )
        self.assertEqual(
            sampler._build_vllm_params(
                req_default, {"routed_experts_prompt_start": 5}
            ).routed_experts_prompt_start,
            5,
        )
        # Neither (first turn): route from token 0.
        self.assertEqual(
            sampler._build_vllm_params(
                req_default, {"routed_experts_prompt_start": None}
            ).routed_experts_prompt_start,
            0,
        )

    def test_sample_error_resilience(self):
        """Verifies error isolation when an individual generator stream raises an exception."""
        args = AsyncEngineArgs(model="Qwen/Qwen2.5-1.5B")
        sampler = RLVllmSampler(engine_args=args)

        mock_engine = MagicMock()

        async def mock_failing_stream(prompt, sampling_params, request_id):
            raise RuntimeError("OOM on sequence generation")
            yield None

        mock_engine.generate.side_effect = mock_failing_stream
        sampler._engine = mock_engine
        sampler._is_running = True

        async def run_err_test():
            reqs = [
                SimpleNamespace(prompt="Test error prompt",
                                request_id="err_req",
                                sampling_params=None)
            ]
            results = await sampler.sample(reqs)
            self.assertEqual(len(results), 1)
            res = results[0]
            self.assertIsNotNone(res.error)
            self.assertEqual(res.error.error_type, "RuntimeError")
            self.assertIn("OOM", res.error.message)
            self.assertTrue(res.error.retryable)

        asyncio.run(run_err_test())


class TestRLVllmSamplerWeightSync(unittest.TestCase):
    """Tests RLVllmSampler weight synchronization with duck-typed requests."""

    @patch(
        "tunix.experimental.rollout.vllm_sampler_v2.RLVllmSampler._call_worker_method"
    )
    def test_weight_sync_calls_tpu_worker_apis(self, mock_call_worker_method):

        mock_call_worker_method.return_value = []

        args = AsyncEngineArgs(model="Qwen/Qwen2.5-1.5B",
                               tensor_parallel_size=2)
        sampler = RLVllmSampler(engine_args=args)

        mock_engine = MagicMock()
        mock_engine.pause_background_loop = AsyncMock()
        mock_engine.resume_background_loop = AsyncMock()
        mock_engine.reset_prefix_cache = AsyncMock()
        sampler._engine = mock_engine
        sampler._is_running = True

        async def run_sync_test():
            req_pre = SimpleNamespace(
                model_path="Qwen/Qwen2.5-1.5B",
                controller_id="ctrl_0",
                policy_version=12,
                req_id="transfer_99",
            )
            self.assertEqual(await sampler.get_transfer_status("transfer_99"),
                             "MISSING")

            await sampler.pre_weight_sync(req_pre)
            mock_engine.pause_background_loop.assert_called_once()
            mock_engine.reset_prefix_cache.assert_called_once()
            self.assertEqual(
                mock_call_worker_method.call_args_list,
                [
                    unittest.mock.call("finish_weight_update"),
                    unittest.mock.call(
                        "start_weight_update", free_kv_cache=False
                    ),
                ],
            )
            self.assertEqual(await sampler.get_transfer_status("transfer_99"),
                             "IN_PROGRESS")

            mock_call_worker_method.reset_mock()
            req_sync = {"extra_config": {"dma_channel": 1}}
            await sampler.weight_sync(req_sync)
            mock_call_worker_method.assert_called_once_with(
                "update_weights", {"dma_channel": 1})

            mock_call_worker_method.reset_mock()
            req_post = SimpleNamespace(req_id="transfer_99")
            await sampler.post_weight_sync(req_post)
            mock_call_worker_method.assert_called_once_with(
                "finish_weight_update")
            mock_engine.reset_prefix_cache.assert_called_once()
            mock_engine.resume_background_loop.assert_called_once()
            self.assertEqual(await sampler.get_transfer_status("transfer_99"),
                             "SUCCESS")

        asyncio.run(run_sync_test())

    def test_get_weight_sync_metadata(self):
        """Verifies get_weight_sync_metadata structure."""

        async def run_test():
            args = AsyncEngineArgs(model="Qwen/Qwen2.5-1.5B",
                                   tensor_parallel_size=4)
            sampler = RLVllmSampler(engine_args=args)

            metadata = await sampler.get_weight_sync_metadata()
            expected_sharding = {
                "tensor_parallel_size": 4,
                "data_parallel_size": 1,
                "expert_parallel_size": 1,
            }
            self.assertEqual(metadata["sharding"], expected_sharding)
            self.assertEqual(metadata["model_path"], "Qwen/Qwen2.5-1.5B")
            self.assertIn("policy_version", metadata)

        asyncio.run(run_test())

    def test_pause_resume_and_clear_cache(self):
        """Verifies engine background loop control and prefix cache clearing."""
        args = AsyncEngineArgs(model="Qwen/Qwen2.5-1.5B")
        sampler = RLVllmSampler(engine_args=args)
        mock_engine = MagicMock()
        mock_engine.pause_background_loop = AsyncMock()
        mock_engine.resume_background_loop = AsyncMock()
        mock_engine.reset_prefix_cache = AsyncMock()
        sampler._engine = mock_engine

        async def run_lifecycle_test():
            await sampler.pause()
            self.assertTrue(sampler._is_paused)
            mock_engine.pause_background_loop.assert_called_once()

            await sampler._clear_prefix_cache()
            mock_engine.reset_prefix_cache.assert_called_once()

            await sampler.resume()
            self.assertFalse(sampler._is_paused)
            mock_engine.resume_background_loop.assert_called_once()

        asyncio.run(run_lifecycle_test())

    @patch("tunix.experimental.rollout.vllm_sampler_v2."
           "RLVllmSampler._call_worker_method")
    def test_raiden_h2d_forwards_uuid(self, mock_call_worker_method):
        """Verifies raiden_h2d threads the transfer generation to the worker."""
        mock_call_worker_method.return_value = []
        args = AsyncEngineArgs(model="Qwen/Qwen2.5-1.5B")
        sampler = RLVllmSampler(engine_args=args)

        async def run_test():
            await sampler.raiden_h2d(uuid=123)
            mock_call_worker_method.assert_called_once_with("raiden_h2d",
                                                            uuid=123)

        asyncio.run(run_test())

    @patch("tunix.experimental.rollout.vllm_sampler_v2."
           "RLVllmSampler._call_worker_method")
    def test_raiden_h2d_defaults_uuid_to_none(self, mock_call_worker_method):
        """Omitting uuid still reaches the worker, which waits untargeted."""
        mock_call_worker_method.return_value = []
        args = AsyncEngineArgs(model="Qwen/Qwen2.5-1.5B")
        sampler = RLVllmSampler(engine_args=args)

        async def run_test():
            await sampler.raiden_h2d()
            mock_call_worker_method.assert_called_once_with("raiden_h2d",
                                                            uuid=None)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
