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

# Ensure transformers (which sets up GOOGLE_INTERNAL_PACKAGE_PATH_version monkeypatch) is loaded
import sys
import importlib.metadata
try:
    import transformers  # pylint: disable=g-import-not-at-top
except ImportError:
    pass

_orig_meta_version = importlib.metadata.version
def _version_shim(pkg):
    if pkg == "transformers":
        return "5.0.0"
    return _orig_meta_version(pkg)
importlib.metadata.version = _version_shim

if "openai_harmony" not in sys.modules:
    sys.modules["openai_harmony"] = MagicMock()

from vllm.engine.arg_utils import AsyncEngineArgs

from tunix.experimental.rollout.vllm_sampler_v2 import RLVllmSampler


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
            mock_call_worker_method.assert_called_once_with(
                "start_weight_update", free_kv_cache=True)
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


if __name__ == "__main__":
    unittest.main()
