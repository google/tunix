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
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project

"""Tests for VllmSampler."""

import asyncio
import importlib
import importlib.metadata
import sys
from types import SimpleNamespace
from typing import Any
from unittest import mock

from absl.testing import absltest
import numpy as np

try:
  import transformers  # pylint: disable=g-import-not-at-top,unused-import
except ImportError:
  pass

_orig_meta_version = importlib.metadata.version


def _version_shim(pkg):
  if pkg == "transformers":
    return "5.0.0"
  return _orig_meta_version(pkg)


importlib.metadata.version = _version_shim

try:
  import openai_harmony  # pylint: disable=g-import-not-at-top,unused-import
except ImportError:
  # openai_harmony is not installed in every environment this test runs in,
  # so stub it only when it is missing; where it is installed, a stub would
  # shadow the real package. vLLM uses openai_harmony.Message as a pydantic
  # field type (ChatCompletionMessageParam), so the stub's Message must be a
  # type pydantic can build a schema for. A MagicMock attribute makes
  # importing vLLM fail with PydanticSchemaGenerationError.
  _openai_harmony_stub = mock.MagicMock()
  _openai_harmony_stub.Message = Any
  sys.modules["openai_harmony"] = _openai_harmony_stub

for _name in (
    "xgrammar",
    "xgrammar.structural_tag",
    "xgrammar.openai_tool_call_schema",
):
  try:
    importlib.import_module(_name)
  except ImportError:
    sys.modules[_name] = mock.MagicMock()

# Import before vllm so vLLM patches it directly instead of adding an import
# hook, which breaks under TAP's import profiler.
importlib.import_module("torch._inductor.pattern_matcher")

from tunix.dist_rl.rollout import sampler as base_sampler_lib  # pylint: disable=g-import-not-at-top
from tunix.dist_rl.rollout import vllm_sampler  # pylint: disable=g-import-not-at-top
from tunix.dist_rl.weight_sync import raiden_weight_sync_delegate  # pylint: disable=g-import-not-at-top
from vllm.engine.arg_utils import AsyncEngineArgs  # pylint: disable=g-import-not-at-top
from vllm.engine.async_llm_engine import AsyncLLMEngine  # pylint: disable=g-import-not-at-top


class VllmSamplerTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.engine_args = SimpleNamespace(model="Qwen/Qwen2.5-1.5B")
    self.mock_engine = mock.MagicMock()
    self.mock_engine.pause_background_loop = mock.AsyncMock()
    self.mock_engine.resume_background_loop = mock.AsyncMock()
    self.mock_engine.pause_generation = mock.AsyncMock()
    self.mock_engine.resume_generation = mock.AsyncMock()
    self.mock_engine.reset_prefix_cache = mock.AsyncMock()
    self.mock_engine.collective_rpc = mock.AsyncMock(return_value=[])
    self.mock_engine.do_log_stats = mock.AsyncMock()

    self.sampler = vllm_sampler.VllmSampler(
        server_id="vllm_slice_01",
        engine_args=self.engine_args,
    )
    self.sampler._engine = self.mock_engine
    self.sampler._is_running = True

  def test_single_sampling_request(self):
    async def mock_generate_stream(prompt, sampling_params, request_id):
      del prompt, sampling_params
      mock_output_choice = SimpleNamespace(
          text=f"completion result for {request_id}",
          token_ids=[101, 102, 103],
          cumulative_logprob=-0.6,
          finish_reason="stop",
          logprobs=[
              {101: SimpleNamespace(logprob=-0.1)},
              {102: SimpleNamespace(logprob=-0.2)},
              {103: SimpleNamespace(logprob=-0.3)},
          ],
          routed_experts=None,
      )
      yield SimpleNamespace(outputs=[mock_output_choice])

    self.mock_engine.generate.side_effect = mock_generate_stream

    req = base_sampler_lib.SamplingRequest(
        request_id="req_01",
        prompt="Test prompt",
        sampling_params=base_sampler_lib.SamplingParams(
            max_tokens=16,
            temperature=0.7,
            return_logprobs=True,
        ),
    )
    response = asyncio.run(self.sampler.sample(req))
    self.assertIsInstance(response, base_sampler_lib.SamplingResponse)
    self.assertEqual(response.request_id, "req_01")
    self.assertEqual(response.text, "completion result for req_01")
    np.testing.assert_array_equal(response.token_ids, [101, 102, 103])
    np.testing.assert_allclose(response.logprobs, [-0.1, -0.2, -0.3])
    self.assertEqual(response.finish_reason, "stop")

  def test_batch_sampling_requests(self):
    async def mock_generate_stream(prompt, sampling_params, request_id):
      del prompt, sampling_params
      if request_id == "req_a":
        choice = SimpleNamespace(
            text="completion a",
            token_ids=[10, 20],
            finish_reason="stop",
            logprobs=[
                {10: SimpleNamespace(logprob=-0.5)},
                {20: SimpleNamespace(logprob=-0.6)},
            ],
            routed_experts=None,
        )
      else:
        choice = SimpleNamespace(
            text="completion b",
            token_ids=[30, 40],
            finish_reason="length",
            logprobs=[
                {30: SimpleNamespace(logprob=-0.7)},
                {40: SimpleNamespace(logprob=-0.8)},
            ],
            routed_experts=None,
        )
      yield SimpleNamespace(outputs=[choice])

    self.mock_engine.generate.side_effect = mock_generate_stream

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
    responses = asyncio.run(self.sampler.sample(reqs))
    self.assertIsInstance(responses, list)
    self.assertLen(responses, 2)
    self.assertEqual(responses[0].request_id, "req_a")
    self.assertEqual(responses[0].text, "completion a")
    self.assertEqual(responses[1].request_id, "req_b")
    self.assertEqual(responses[1].text, "completion b")
    self.assertEqual(responses[1].finish_reason, "length")

  def test_sample_raw_string_mode(self):
    async def mock_generate_stream(prompt, sampling_params, request_id):
      del prompt, sampling_params, request_id
      yield SimpleNamespace(
          outputs=[
              SimpleNamespace(
                  text="Output text",
                  token_ids=[1, 2],
                  finish_reason="stop",
                  logprobs=None,
              )
          ]
      )

    self.mock_engine.generate.side_effect = mock_generate_stream
    texts = asyncio.run(
        self.sampler.sample(
            ["Prompt string 1", "Prompt string 2"], max_tokens=16
        )
    )
    self.assertEqual(texts, ["Output text", "Output text"])

  def test_sample_error_resilience(self):
    async def mock_failing_stream(prompt, sampling_params, request_id):
      del prompt, sampling_params, request_id
      raise RuntimeError("OOM on sequence generation")
      yield None

    self.mock_engine.generate.side_effect = mock_failing_stream
    reqs = [
        SimpleNamespace(
            prompt="Test error prompt",
            request_id="err_req",
            sampling_params=None,
        )
    ]
    results = asyncio.run(self.sampler.sample(reqs))
    self.assertLen(results, 1)
    res = results[0]
    self.assertIsNotNone(res.error)
    self.assertEqual(res.error.error_type, "RuntimeError")
    self.assertIn("OOM", res.error.message)
    self.assertTrue(res.error.retryable)

  def test_malformed_response_raises_instead_of_empty_completion(self):
    async def mock_generate_stream(prompt, sampling_params, request_id):
      del prompt, sampling_params, request_id
      yield SimpleNamespace(
          outputs=[
              SimpleNamespace(
                  text="two tokens",
                  token_ids=[1, 2],
                  finish_reason="stop",
                  # One logprob for two tokens: SamplingResponse rejects it.
                  logprobs=[{1: SimpleNamespace(logprob=-0.1)}],
              )
          ]
      )

    self.mock_engine.generate.side_effect = mock_generate_stream
    req = base_sampler_lib.SamplingRequest(request_id="bad", prompt="p")
    with self.assertRaisesRegex(ValueError, "logprobs shape"):
      asyncio.run(self.sampler.sample(req))

  def test_lifecycle_methods(self):
    self.sampler._mesh = "mock_mesh"
    asyncio.run(self.sampler.pause())
    self.assertTrue(self.sampler._is_paused)
    self.mock_engine.pause_background_loop.assert_awaited_once()

    asyncio.run(self.sampler.resume())
    self.assertFalse(self.sampler._is_paused)
    self.mock_engine.resume_background_loop.assert_awaited_once()

    mesh = asyncio.run(self.sampler.get_mesh())
    self.assertEqual(mesh, "mock_mesh")

    asyncio.run(self.sampler.stop())
    self.assertFalse(self.sampler._is_running)
    self.assertIsNone(self.sampler._engine)

  def test_weight_sync_lifecycle(self):
    self.mock_engine.collective_rpc.side_effect = [
        [{
            "unit": {
                "job_name": "rollout",
                "job_replica_id": "",
                "data_name": "",
                "data_replica_idx": 0,
            },
            "variables": (),
            "mesh_shape": (1,),
            "mesh_axes": ("fsdp",),
            "data_address": "",
            "control_plane_rpc_address": "",
        }],
        [],  # start_weight_update
        [{"tensor": 1.0}],  # raiden_h2d
        [],  # refresh_model_state_leaves
        [],  # finish_weight_update
        [],  # start_weight_update (round 2)
        [],  # finish_weight_update (abort round 2)
    ]

    meta = asyncio.run(self.sampler.get_weight_sync_metadata())
    self.assertLen(meta, 1)

    sync_req = base_sampler_lib.WeightSyncRequest(
        policy_version=1, extra_config={"req_id": "r1", "uuid": 1}
    )
    res_pre = asyncio.run(self.sampler.pre_weight_sync(sync_req))
    self.assertTrue(res_pre)
    self.mock_engine.collective_rpc.assert_any_call(
        "start_weight_update", args=(), kwargs={"free_kv_cache": False}
    )

    res_sync = asyncio.run(self.sampler.weight_sync(sync_req))
    self.assertTrue(res_sync)
    self.mock_engine.collective_rpc.assert_any_call(
        "raiden_h2d", args=(), kwargs={"uuid": 1}
    )
    self.mock_engine.collective_rpc.assert_any_call(
        "refresh_model_state_leaves", args=(), kwargs={}
    )

    res_post = asyncio.run(self.sampler.post_weight_sync(sync_req))
    self.assertEqual(res_post, 1)
    self.mock_engine.collective_rpc.assert_any_call(
        "finish_weight_update", args=(), kwargs={}
    )

    status = asyncio.run(self.sampler.get_weight_sync_status())
    self.assertEqual(status.get("phase"), "committed")
    self.assertEqual(
        asyncio.run(self.sampler.get_transfer_status("r1")), "SUCCESS"
    )

    # Test abort path on next round (with higher uuid)
    sync_req_2 = base_sampler_lib.WeightSyncRequest(
        policy_version=2, extra_config={"req_id": "r2", "uuid": 2}
    )
    asyncio.run(self.sampler.pre_weight_sync(sync_req_2))
    res_abort = asyncio.run(self.sampler.abort_weight_sync(sync_req_2))
    self.assertTrue(res_abort)
    status_2 = asyncio.run(self.sampler.get_weight_sync_status())
    self.assertEqual(status_2.get("phase"), "aborted")

  def test_get_load_info(self):
    self.mock_engine.engine = SimpleNamespace(
        make_stats=lambda: SimpleNamespace(
            num_waiting_reqs=3,
            num_running_reqs=1,
            kv_cache_usage=0.255,
        )
    )
    load_info = asyncio.run(self.sampler.get_load_info())
    self.assertIsInstance(load_info, base_sampler_lib.LoadInfo)
    self.assertEqual(load_info.num_requests_waiting, 3)
    self.assertEqual(load_info.num_requests_running, 1)
    self.assertAlmostEqual(load_info.kv_cache_usage_perc, 25.5)

  def test_uninitialized_raises(self):
    uninit = vllm_sampler.VllmSampler(server_id="empty")
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
      asyncio.run(self.sampler.sample(None))

  def test_engine_args_deferred_to_initialize(self):
    with mock.patch(
        "vllm.engine.arg_utils.AsyncEngineArgs",
        return_value=self.engine_args,
    ) as mock_args_cls:
      sampler = vllm_sampler.VllmSampler(
          server_id="deferred_vllm",
          model_name="Qwen/Qwen3-8B",
      )
      self.assertIsNone(sampler.engine_args)
      mock_args_cls.assert_not_called()

      sampler.initialize()
      self.assertIs(sampler.engine_args, self.engine_args)
      mock_args_cls.assert_called_once_with(model="Qwen/Qwen3-8B")

  def test_weight_sync_starts_an_idle_engine(self):
    self.sampler._is_running = False
    mock_llm_cls = mock.MagicMock()
    mock_llm_cls.from_engine_args.return_value = self.mock_engine

    async def run_test():
      await self.sampler.bind_weight_sync()
      mock_llm_cls.from_engine_args.assert_called_once_with(self.engine_args)
      self.mock_engine.collective_rpc.assert_any_call(
          "bind_raiden_sync",
          args=(0, 4, "replica_vllm_slice_01"),
          kwargs={},
      )

      await self.sampler.get_weight_sync_metadata()
      mock_llm_cls.from_engine_args.assert_called_once()
      await self.sampler.stop()

    with mock.patch(
        "vllm.engine.async_llm_engine.AsyncLLMEngine",
        mock_llm_cls,
    ):
      asyncio.run(run_test())

  def test_weight_sync_apis_fail_when_uninitialized(self):
    uninit = vllm_sampler.VllmSampler(server_id="vllm_slice_01")
    for coro in (
        uninit.bind_weight_sync(),
        uninit.pre_weight_sync(None),
        uninit.weight_sync(None),
        uninit.post_weight_sync(None),
        uninit.abort_weight_sync(None),
    ):
      with self.assertRaises(RuntimeError):
        asyncio.run(coro)

  def test_raiden_disabled_phases_short_circuit_before_initialization(self):
    uninit = vllm_sampler.VllmSampler(
        server_id="uninit_no_raiden", weight_sync_mode="none"
    )
    self.assertIsNone(asyncio.run(uninit.bind_weight_sync()))
    self.assertTrue(asyncio.run(uninit.pre_weight_sync(None)))
    self.assertTrue(asyncio.run(uninit.post_weight_sync(None)))
    self.assertTrue(asyncio.run(uninit.abort_weight_sync(None)))
    with self.assertRaises(NotImplementedError):
      asyncio.run(uninit.get_weight_sync_metadata())
    with self.assertRaisesRegex(
        RuntimeError, "supports Raiden weight sync only"
    ):
      asyncio.run(uninit.weight_sync(None))

  def test_bind_weight_sync_initializes_engine_args_lazily(self):
    mock_llm_cls = mock.MagicMock()
    mock_llm_cls.from_engine_args.return_value = self.mock_engine

    async def run_test(sampler):
      await sampler.bind_weight_sync()
      mock_llm_cls.from_engine_args.assert_called_once_with(self.engine_args)
      await sampler.stop()

    with mock.patch(
        "vllm.engine.arg_utils.AsyncEngineArgs",
        return_value=self.engine_args,
    ) as mock_args_cls, mock.patch(
        "vllm.engine.async_llm_engine.AsyncLLMEngine",
        mock_llm_cls,
    ):
      sampler = vllm_sampler.VllmSampler(
          server_id="lazy_vllm", model_name="Qwen/Qwen3-8B"
      )
      asyncio.run(run_test(sampler))
      mock_args_cls.assert_called_once_with(model="Qwen/Qwen3-8B")

  def test_raiden_job_name_derived_from_server_id(self):
    sampler = vllm_sampler.VllmSampler(
        server_id="worker-42",
        engine_args=self.engine_args,
    )
    self.assertEqual(sampler.raiden_job_name, "replica_worker-42")

  def test_canonicalize_variable_names_bracketed_to_dotted(self):
    entry = {
        "unit": {"job_name": "destination"},
        "variables": [
            {"name": "['base']['decoder']['layers_0']['w']", "shape": (10, 10)},
            {"name": "['model']['decoder']['w2'].value", "shape": (20, 20)},
            {"name": "already.canonical.key", "shape": (30, 30)},
        ],
    }
    canonical = raiden_weight_sync_delegate.canonicalize_variable_names(entry)
    names = [v["name"] for v in canonical["variables"]]
    self.assertEqual(
        names, ["decoder.layers.0.w", "decoder.w2", "already.canonical.key"]
    )

  def test_canonicalize_variable_names_empty_key_raises(self):
    entry = {
        "variables": [
            {"name": "['base'].value"},
        ]
    }
    with self.assertRaisesRegex(ValueError, "canonicalised to an empty key"):
      raiden_weight_sync_delegate.canonicalize_variable_names(entry)

  def test_canonicalize_variable_names_collision_raises(self):
    entry = {
        "variables": [
            {"name": "['base']['decoder']['layers_0']['w']"},
            {"name": "['model']['decoder']['layers'][0]['w'].value"},
        ]
    }
    with self.assertRaisesRegex(ValueError, "both canonicalise to"):
      raiden_weight_sync_delegate.canonicalize_variable_names(entry)

  def test_canonicalize_variable_names_passthrough(self):
    empty_entry = {"unit": {"job_name": "dst"}}
    self.assertEqual(
        raiden_weight_sync_delegate.canonicalize_variable_names(empty_entry),
        empty_entry,
    )

  def test_partial_rollout_uses_pause_generation_keep_and_preserves_cache(self):
    sampler = vllm_sampler.VllmSampler(
        server_id="vllm_slice_partial",
        engine_args=self.engine_args,
        partial_rollout=True,
    )
    sampler._engine = self.mock_engine
    sampler._is_running = True

    req = base_sampler_lib.WeightSyncRequest(
        policy_version=1, extra_config={"req_id": "r1", "uuid": 1}
    )
    asyncio.run(sampler.pre_weight_sync(req))
    self.mock_engine.pause_generation.assert_awaited_once_with(
        mode="keep", clear_cache=False
    )
    self.mock_engine.reset_prefix_cache.assert_not_awaited()

    asyncio.run(sampler.post_weight_sync(req))
    self.mock_engine.reset_prefix_cache.assert_not_awaited()
    self.mock_engine.resume_generation.assert_awaited_once()

  def test_build_vllm_params_forwards_cache_salt(self):
    sampler = vllm_sampler.VllmSampler(
        engine_args=self.engine_args, partial_rollout=True
    )
    req_params = SimpleNamespace(
        max_tokens=32,
        temperature=0.6,
        cache_salt="policy_v4",
    )
    req = SimpleNamespace(sampling_params=req_params)
    vllm_params = sampler._build_vllm_params(req, {})
    self.assertEqual(getattr(vllm_params, "cache_salt", None), "policy_v4")

  def test_build_vllm_params_forwards_routed_experts_prompt_start(self):
    sampler = vllm_sampler.VllmSampler(engine_args=self.engine_args)
    req_params = SimpleNamespace(
        max_tokens=32,
        temperature=0.6,
        routed_experts_prompt_start=48,
    )
    req = SimpleNamespace(sampling_params=req_params)
    vllm_params = sampler._build_vllm_params(req, {})
    self.assertEqual(
        getattr(vllm_params, "routed_experts_prompt_start", 0), 48
    )

  def test_build_vllm_params_detokenizes_by_default(self):
    """Requests without a detokenize field still get text back."""
    sampler = vllm_sampler.VllmSampler(engine_args=self.engine_args)
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(max_tokens=32, temperature=0.6)
    )
    vllm_params = sampler._build_vllm_params(req, {})
    self.assertTrue(vllm_params.detokenize)

  def test_build_vllm_params_detokenize_opt_out_via_sampling_params(self):
    """An explicit detokenize=False from sampling_params is honored."""
    sampler = vllm_sampler.VllmSampler(engine_args=self.engine_args)
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(
            max_tokens=32, temperature=0.6, detokenize=False
        )
    )
    vllm_params = sampler._build_vllm_params(req, {})
    self.assertFalse(vllm_params.detokenize)

  def test_build_vllm_params_detokenize_opt_out_via_kwargs(self):
    """An explicit detokenize=False from kwargs is honored."""
    sampler = vllm_sampler.VllmSampler(engine_args=self.engine_args)
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(max_tokens=32, temperature=0.6)
    )
    vllm_params = sampler._build_vllm_params(req, {"detokenize": False})
    self.assertFalse(vllm_params.detokenize)

  def test_build_vllm_params_sampling_params_overrides_kwargs(self):
    """sampling_params.detokenize takes precedence over conflicting kwargs."""
    sampler = vllm_sampler.VllmSampler(engine_args=self.engine_args)
    req = SimpleNamespace(
        sampling_params=SimpleNamespace(
            max_tokens=32, temperature=0.6, detokenize=False
        )
    )
    vllm_params = sampler._build_vllm_params(req, {"detokenize": True})
    self.assertFalse(vllm_params.detokenize)

  def _kwargs_passed_to_build_vllm_params(self, sampler, **sample_kwargs):
    async def mock_generate_stream(prompt, sampling_params, request_id):
      del prompt, sampling_params, request_id
      yield SimpleNamespace(
          outputs=[
              SimpleNamespace(
                  text="t", token_ids=[1], finish_reason="stop", logprobs=None
              )
          ]
      )

    sampler._engine = self.mock_engine
    sampler._is_running = True
    self.mock_engine.generate.side_effect = mock_generate_stream
    with mock.patch.object(
        sampler, "_build_vllm_params", wraps=sampler._build_vllm_params
    ) as build:
      asyncio.run(
          sampler.sample(
              base_sampler_lib.SamplingRequest(prompt="hi"), **sample_kwargs
          )
      )
    return build.call_args.args[1]

  def test_detokenize_flag_off_by_default(self):
    self.assertFalse(self.sampler.detokenize)
    self.assertEqual(self._kwargs_passed_to_build_vllm_params(self.sampler), {})

  def test_detokenize_flag_enabled_is_forwarded(self):
    sampler = vllm_sampler.VllmSampler(
        engine_args=self.engine_args, detokenize=True
    )
    self.assertEqual(
        self._kwargs_passed_to_build_vllm_params(sampler), {"detokenize": True}
    )

  def test_explicit_detokenize_kwarg_wins_over_sampler_flag(self):
    sampler = vllm_sampler.VllmSampler(
        engine_args=self.engine_args, detokenize=True
    )
    self.assertEqual(
        self._kwargs_passed_to_build_vllm_params(sampler, detokenize=False),
        {"detokenize": False},
    )

  def test_duck_typed_request_processing(self):
    async def mock_generate_stream(prompt, sampling_params, request_id):
      del prompt, sampling_params
      yield SimpleNamespace(
          outputs=[
              SimpleNamespace(
                  text=f"ans:{request_id}",
                  token_ids=[1, 2],
                  finish_reason="stop",
                  logprobs=None,
              )
          ]
      )

    self.mock_engine.generate.side_effect = mock_generate_stream
    reqs = [
        SimpleNamespace(
            prompt="Solve 2+2",
            request_id="req_attr_1",
            sampling_params=SimpleNamespace(
                max_tokens=64,
                temperature=0.5,
                top_p=0.9,
                top_k=-1,
                stop_sequences=[],
                return_logprobs=False,
            ),
        ),
        {
            "prompt": "Solve 3+3",
            "request_id": "req_dict_2",
            "sampling_params": {"max_tokens": 32, "temperature": 0.2},
        },
    ]
    results = asyncio.run(self.sampler.sample(reqs))
    self.assertLen(results, 2)
    self.assertEqual(results[0].request_id, "req_attr_1")
    self.assertEqual(results[0].text, "ans:req_attr_1")
    self.assertEqual(results[1].request_id, "req_dict_2")
    self.assertEqual(results[1].text, "ans:req_dict_2")

  def test_pre_weight_sync_forwards_free_kv_cache_and_tracks_transfer_status(
      self,
  ):
    self.assertEqual(
        asyncio.run(self.sampler.get_transfer_status("transfer_99")),
        "MISSING",
    )
    sync_req = base_sampler_lib.WeightSyncRequest(
        policy_version=12,
        extra_config={"req_id": "transfer_99", "uuid": 1},
    )
    asyncio.run(self.sampler.pre_weight_sync(sync_req, free_kv_cache=True))
    self.mock_engine.pause_background_loop.assert_awaited_once()
    self.mock_engine.reset_prefix_cache.assert_awaited_once()
    self.mock_engine.collective_rpc.assert_any_call(
        "start_weight_update", args=(), kwargs={"free_kv_cache": True}
    )
    self.assertEqual(
        asyncio.run(self.sampler.get_transfer_status("transfer_99")),
        "IN_PROGRESS",
    )

    asyncio.run(self.sampler.post_weight_sync(sync_req))
    self.assertEqual(
        asyncio.run(self.sampler.get_transfer_status("transfer_99")),
        "SUCCESS",
    )

  def test_clear_prefix_cache(self):
    self.sampler._cache_valid = True
    asyncio.run(self.sampler._clear_prefix_cache())
    self.assertFalse(self.sampler._cache_valid)
    self.mock_engine.reset_prefix_cache.assert_awaited_once()

  def test_raiden_h2d_forwards_uuid(self):
    asyncio.run(self.sampler.raiden_h2d(uuid=123))
    self.mock_engine.collective_rpc.assert_awaited_once_with(
        "raiden_h2d", args=(), kwargs={"uuid": 123}
    )

  def test_raiden_h2d_defaults_uuid_to_none(self):
    asyncio.run(self.sampler.raiden_h2d())
    self.mock_engine.collective_rpc.assert_awaited_once_with(
        "raiden_h2d", args=(), kwargs={"uuid": None}
    )

  def test_weight_sync_when_raiden_disabled(self):
    sampler = vllm_sampler.VllmSampler(
        server_id="vllm_no_raiden",
        engine_args=self.engine_args,
        weight_sync_mode="none",
    )
    self.assertFalse(sampler.enable_raiden)
    self.assertIsNone(asyncio.run(sampler.bind_weight_sync()))
    with self.assertRaises(NotImplementedError):
      asyncio.run(sampler.get_weight_sync_metadata())
    self.assertTrue(asyncio.run(sampler.pre_weight_sync()))
    with self.assertRaises(RuntimeError):
      asyncio.run(sampler.weight_sync())
    self.assertTrue(asyncio.run(sampler.post_weight_sync()))
    self.assertTrue(asyncio.run(sampler.abort_weight_sync()))


class RoundUuidTest(absltest.TestCase):
  """Covers the extra_config -> Raiden transfer generation extraction."""

  def test_extracts_positive_uuid(self):
    req = base_sampler_lib.WeightSyncRequest(
        policy_version=1, extra_config={"req_id": "r1", "uuid": 7}
    )
    self.assertEqual(raiden_weight_sync_delegate.round_uuid(req), 7)

  def test_absent_extra_config_raises(self):
    with self.assertRaisesRegex(ValueError, "missing a usable transfer uuid"):
      raiden_weight_sync_delegate.round_uuid(None)

  def test_absent_uuid_key_raises(self):
    req = base_sampler_lib.WeightSyncRequest(
        policy_version=1, extra_config={"req_id": "r1"}
    )
    with self.assertRaisesRegex(ValueError, "missing a usable transfer uuid"):
      raiden_weight_sync_delegate.round_uuid(req)

  def test_non_positive_uuid_raises(self):
    for sentinel in (0, -1):
      req = base_sampler_lib.WeightSyncRequest(
          policy_version=1, extra_config={"req_id": "r1", "uuid": sentinel}
      )
      with self.assertRaisesRegex(ValueError, "non-positive transfer uuid"):
        raiden_weight_sync_delegate.round_uuid(req)

  def test_malformed_uuid_raises(self):
    req = base_sampler_lib.WeightSyncRequest(
        policy_version=1, extra_config={"req_id": "r1", "uuid": "not-an-int"}
    )
    with self.assertRaisesRegex(ValueError, "missing a usable transfer uuid"):
      raiden_weight_sync_delegate.round_uuid(req)


class VllmSamplerLogStatsTest(absltest.TestCase):
  """Tests the periodic flush of vLLM engine stats while the sampler runs."""

  def setUp(self):
    super().setUp()
    interval_patcher = mock.patch("vllm.envs.VLLM_LOG_STATS_INTERVAL", 0.01)
    interval_patcher.start()
    self.addCleanup(interval_patcher.stop)
    self.mock_engine = mock.create_autospec(AsyncLLMEngine, instance=True)
    mock_engine_cls = mock.MagicMock()
    mock_engine_cls.from_engine_args.return_value = self.mock_engine
    engine_cls_patcher = mock.patch(
        "vllm.engine.async_llm_engine.AsyncLLMEngine",
        mock_engine_cls,
    )
    engine_cls_patcher.start()
    self.addCleanup(engine_cls_patcher.stop)
    self.sampler = vllm_sampler.VllmSampler(
        engine_args=AsyncEngineArgs(model="Qwen/Qwen2.5-1.5B")
    )

  def test_start_flushes_stats_until_stop(self):
    """start() flushes engine stats periodically; stop() cancels it."""

    async def run_test():
      flushed = asyncio.Event()
      self.mock_engine.do_log_stats.side_effect = flushed.set

      await self.sampler.start()
      await asyncio.wait_for(flushed.wait(), timeout=5)
      log_stats_task = self.sampler._log_stats_task
      await self.sampler.stop()

      self.assertTrue(log_stats_task.cancelled())
      self.assertIsNone(self.sampler._log_stats_task)

    asyncio.run(run_test())

  def test_failed_flush_is_logged_and_flushing_continues(self):
    """A failing flush is logged and does not end the periodic flush."""

    async def run_test():
      recovered = asyncio.Event()
      num_flushes = 0

      def flaky_flush():
        nonlocal num_flushes
        num_flushes += 1
        if num_flushes == 1:
          raise RuntimeError("stats unavailable")
        recovered.set()

      self.mock_engine.do_log_stats.side_effect = flaky_flush

      with self.assertLogs(
          vllm_sampler.VllmSampler.__module__, level="ERROR"
      ) as logs:
        await self.sampler.start()
        await asyncio.wait_for(recovered.wait(), timeout=5)
      await self.sampler.stop()

      self.assertIsInstance(logs.records[0].exc_info[1], RuntimeError)

    asyncio.run(run_test())

  def test_stop_propagates_its_own_cancellation(self):
    """Cancelling stop() is not swallowed, and a retried stop() finishes."""

    async def run_test():
      await self.sampler.start()
      stop_task = asyncio.create_task(self.sampler.stop())
      # Lets stop() run until it waits for the flush task to finish.
      await asyncio.sleep(0)
      self.assertTrue(self.sampler._log_stats_task.cancelling())
      stop_task.cancel()
      with self.assertRaises(asyncio.CancelledError):
        await stop_task

      await self.sampler.stop()
      self.assertFalse(self.sampler._is_running)
      self.assertIsNone(self.sampler._log_stats_task)

    asyncio.run(run_test())


if __name__ == "__main__":
  absltest.main()
