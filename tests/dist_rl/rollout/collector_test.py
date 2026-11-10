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

"""Tests for the trajectory collector."""

import asyncio
import dataclasses
import pathlib
import types
import unittest
from unittest import mock

from absl.testing import absltest
import jax.numpy as jnp
import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.common import test_utils as mocks
from tunix.dist_rl.common import timeouts
from tunix.dist_rl.orchestrator import batch_assembly
from tunix.dist_rl.orchestrator import payload_builder
from tunix.dist_rl.rollout import collector
from tunix.dist_rl.rollout import sampler as sampler_lib
from tunix.dist_rl.trajectory import converter as converter_lib
from tunix.dist_rl.trajectory import file_store
from tunix.dist_rl.trajectory import in_memory_store
from tunix.dist_rl.trajectory import sql_store
from tunix.rl import algorithm_config
from tunix.rl import common as rl_common
from tunix.rl import packing
from tunix.rl.agentic.agents import model_agent
from tunix.rl.agentic.environments import base_environment


class _MultiStepEnv(base_environment.BaseTaskEnv):

  def _initial_observation(self):
    return {"observation": "start"}

  def _step_impl(self, action):
    return base_environment.EnvStepResult(
        observation={"observation": "step"},
        reward=1.0,
        done=False,
        info={},
    )


class _RecordingParser:

  def __init__(self):
    self.calls = []

  def parse(
      self,
      messages=None,
      add_generation_prompt=False,
      is_first_msg=False,
      **kwargs,
  ):
    msgs = messages if messages is not None else kwargs.get("msgs")
    self.calls.append((msgs, add_generation_prompt, is_first_msg))
    return "PARSED"

  def update_assistant_end_tokens(self, tokens):
    return tokens, 0


class _MockTokenizer:

  def encode(self, text, add_special_tokens=False):
    del add_special_tokens
    return [ord(c) % 1000 for c in text] or [101]

  def dedup_bos_ids(self, tokens):
    return tokens


class _MockSampler(sampler_lib.Sampler):

  def __init__(self, token_lengths=None, routed_experts=None):
    self.sampled_params = []
    self.token_lengths = token_lengths or [30, 20]
    self._call_count = 0
    self.routed_experts = routed_experts

  async def sample(self, req, **kwargs):
    if hasattr(req, "sampling_params"):
      self.sampled_params.append(req.sampling_params)
    tok_len = (
        self.token_lengths[self._call_count]
        if self._call_count < len(self.token_lengths)
        else 10
    )
    self._call_count += 1
    tokens = np.arange(tok_len, dtype=np.int32)
    req_prompt = getattr(req, "prompt", None)
    prompt_token_ids = (
        np.asarray(req_prompt, dtype=np.int32)
        if isinstance(req_prompt, np.ndarray)
        else np.array([1, 2], dtype=np.int32)
    )
    return sampler_lib.SamplingResponse(
        request_id=getattr(req, "request_id", ""),
        text=f"action_{self._call_count}",
        token_ids=tokens,
        prompt_token_ids=prompt_token_ids,
        routed_experts=self.routed_experts,
    )


class _MockVanillaSampler(sampler_lib.Sampler):

  def __init__(self):
    self.calls = []

  async def sample(self, req, **kwargs):
    self.calls.append((req, kwargs))
    sp = getattr(req, "sampling_params", None)
    seed = getattr(sp, "seed", None)
    tok_seed = int(seed if seed is not None else 0)
    return sampler_lib.SamplingResponse(
        request_id=getattr(req, "request_id", ""),
        text=f"action_seed_{seed}",
        token_ids=np.array([tok_seed, tok_seed + 1], dtype=np.int32),
        prompt_token_ids=np.array([1, 2], dtype=np.int32),
    )


class _MockVllmSampler(sampler_lib.Sampler):

  def __init__(self):
    self.calls = []

  async def sample(self, req, **kwargs):
    self.calls.append((req, kwargs))
    return sampler_lib.SamplingResponse(
        request_id=getattr(req, "request_id", ""),
        text="action_vllm",
        token_ids=np.array([10, 20], dtype=np.int32),
        prompt_token_ids=np.array([1, 2], dtype=np.int32),
    )


class BuildPromptTest(absltest.TestCase):

  def test_chat_messages_are_parsed(self):
    parser = _RecordingParser()
    msgs = [{"role": "user", "content": "hi"}]
    self.assertEqual(collector._build_prompt(parser, msgs), "PARSED")
    self.assertEqual(parser.calls, [(msgs, True, True)])

  def test_string_prompt_passes_through(self):
    parser = _RecordingParser()
    self.assertEqual(collector._build_prompt(parser, "raw"), "raw")
    self.assertEmpty(parser.calls)

  def test_no_parser_passes_through(self):
    msgs = [{"role": "user", "content": "hi"}]
    self.assertIs(collector._build_prompt(None, msgs), msgs)


class TrajectoryCollectorEngineTest(absltest.TestCase):

  def test_max_response_length_extraction(self):
    sampler = _MockSampler()
    agent = mock.MagicMock()
    env = mock.MagicMock()
    tokenizer = _MockTokenizer()
    parser = _RecordingParser()

    req1 = datatypes.RolloutRequest(
        prompt_id="p1",
        max_response_length=512,
    )
    engine1 = collector.TrajectoryCollectorEngine(
        traj_id="t1",
        request=req1,
        sampler=sampler,
        env_client=env,
        agent=agent,
        tokenizer=tokenizer,
        chat_parser=parser,
    )
    self.assertEqual(engine1.max_response_length, 512)

    # None when absent
    req2 = datatypes.RolloutRequest(
        prompt_id="p2",
        generation_kwargs={},
    )
    engine2 = collector.TrajectoryCollectorEngine(
        traj_id="t2",
        request=req2,
        sampler=sampler,
        env_client=env,
        agent=agent,
        tokenizer=tokenizer,
        chat_parser=parser,
    )
    self.assertIsNone(engine2.max_response_length)

  def test_episode_options_are_forwarded_to_inner_engine(self):
    request = datatypes.RolloutRequest(
        prompt_id="p1",
        max_response_length=512,
        metadata={"episode_timeout": 10800, "overlong_filter": True},
    )
    agent = mock.MagicMock()
    agent.name = "agent"
    engine = collector.TrajectoryCollectorEngine(
        traj_id="t1",
        request=request,
        sampler=_MockSampler(),
        env_client=mock.MagicMock(),
        agent=agent,
        tokenizer=_MockTokenizer(),
        chat_parser=_RecordingParser(),
    )

    with mock.patch.object(
        collector.rl_collect_engine, "TrajectoryCollectEngine"
    ) as inner_engine_cls:
      inner_engine_cls.return_value.collect = mock.AsyncMock(return_value={})
      asyncio.run(engine.run_episode())

    self.assertEqual(inner_engine_cls.call_args.kwargs["timeout"], 10800)
    self.assertTrue(inner_engine_cls.call_args.kwargs["overlong_filter"])

  def test_episode_timeout_defaults_to_constant(self):
    req_empty_meta = datatypes.RolloutRequest(
        prompt_id="p1",
        generation_kwargs={},
    )
    engine1 = collector.TrajectoryCollectorEngine(
        traj_id="t1",
        request=req_empty_meta,
        sampler=_MockSampler(),
        env_client=mock.MagicMock(),
        agent=mock.MagicMock(),
        tokenizer=_MockTokenizer(),
        chat_parser=_RecordingParser(),
    )
    self.assertEqual(
        engine1.episode_timeout, timeouts.DEFAULT_EPISODE_TIMEOUT_S
    )

    req_none_meta = datatypes.RolloutRequest(
        prompt_id="p2",
        generation_kwargs={},
        metadata={"episode_timeout": None},
    )
    engine2 = collector.TrajectoryCollectorEngine(
        traj_id="t2",
        request=req_none_meta,
        sampler=_MockSampler(),
        env_client=mock.MagicMock(),
        agent=mock.MagicMock(),
        tokenizer=_MockTokenizer(),
        chat_parser=_RecordingParser(),
    )
    self.assertEqual(
        engine2.episode_timeout, timeouts.DEFAULT_EPISODE_TIMEOUT_S
    )

  def test_episode_timeout_invalid_raises_value_error(self):
    req = datatypes.RolloutRequest(
        prompt_id="p1",
        generation_kwargs={},
        metadata={"episode_timeout": 0},
    )
    with self.assertRaises(ValueError):
      collector.TrajectoryCollectorEngine(
          traj_id="t1",
          request=req,
          sampler=_MockSampler(),
          env_client=mock.MagicMock(),
          agent=mock.MagicMock(),
          tokenizer=_MockTokenizer(),
          chat_parser=_RecordingParser(),
      )

  def test_overlong_filter_defaults_to_false(self):
    req_empty_meta = datatypes.RolloutRequest(
        prompt_id="p1",
        generation_kwargs={},
    )
    engine1 = collector.TrajectoryCollectorEngine(
        traj_id="t1",
        request=req_empty_meta,
        sampler=_MockSampler(),
        env_client=mock.MagicMock(),
        agent=mock.MagicMock(),
        tokenizer=_MockTokenizer(),
        chat_parser=_RecordingParser(),
    )
    self.assertFalse(engine1.overlong_filter)

    req_none_meta = datatypes.RolloutRequest(
        prompt_id="p2",
        generation_kwargs={},
        metadata={"overlong_filter": None},
    )
    engine2 = collector.TrajectoryCollectorEngine(
        traj_id="t2",
        request=req_none_meta,
        sampler=_MockSampler(),
        env_client=mock.MagicMock(),
        agent=mock.MagicMock(),
        tokenizer=_MockTokenizer(),
        chat_parser=_RecordingParser(),
    )
    self.assertFalse(engine2.overlong_filter)

  def test_overlong_filter_invalid_type_raises_type_error(self):
    for invalid_val in ("False", "True", "false", 1, 0, [True]):
      with self.subTest(invalid_val=invalid_val):
        req = datatypes.RolloutRequest(
            prompt_id="p1",
            generation_kwargs={},
            metadata={"overlong_filter": invalid_val},
        )
        with self.assertRaises(TypeError):
          collector.TrajectoryCollectorEngine(
              traj_id="t1",
              request=req,
              sampler=_MockSampler(),
              env_client=mock.MagicMock(),
              agent=mock.MagicMock(),
              tokenizer=_MockTokenizer(),
              chat_parser=_RecordingParser(),
          )

  def test_dynamic_capping_of_max_tokens_across_turns(self):
    async def _run():
      sampler = _MockSampler(token_lengths=[30, 20])
      agent = model_agent.ModelAgent("test_agent")
      task = {"question": "2+2", "answer": "4"}
      env = _MultiStepEnv(task=task, max_steps=3)
      tokenizer = _MockTokenizer()
      parser = _RecordingParser()

      req = datatypes.RolloutRequest(
          prompt_id="p1",
          prompt="What is 2+2?",
          max_response_length=50,
      )
      engine = collector.TrajectoryCollectorEngine(
          traj_id="t1",
          request=req,
          sampler=sampler,
          env_client=env,
          agent=agent,
          tokenizer=tokenizer,
          chat_parser=parser,
      )

      await engine.run_episode()
      self.assertTrue(engine.is_done)
      self.assertGreaterEqual(len(sampler.sampled_params), 2)
      # Turn 1: remaining budget is 50.
      self.assertEqual(sampler.sampled_params[0].max_tokens, 50)
      # Turn 1 generated 30 tokens, so Turn 2 remaining budget is 50 - 30 = 20.
      self.assertEqual(sampler.sampled_params[1].max_tokens, 20)

    asyncio.run(_run())

  def test_run_episode_records_sampled_tokens_per_turn(self):
    async def _run():
      engine = collector.TrajectoryCollectorEngine(
          traj_id="t1",
          request=datatypes.RolloutRequest(
              prompt_id="p1",
              prompt="What is 2+2?",
              max_response_length=50,
              metadata={"overlong_filter": True},
          ),
          sampler=_MockSampler(token_lengths=[30, 20]),
          env_client=_MultiStepEnv(
              task={"question": "2+2", "answer": "4"}, max_steps=3
          ),
          agent=model_agent.ModelAgent("test_agent"),
          tokenizer=_MockTokenizer(),
          chat_parser=_RecordingParser(),
      )
      return await engine.run_episode()

    item = asyncio.run(_run())

    # The second turn uses up the response budget, so the overlong filter
    # zeroes the training mask; the per-turn lengths are recorded regardless.
    self.assertEqual(item.traj["status"], "MAX_CONTEXT_LIMIT_REACHED")
    self.assertFalse(np.any(item.traj["conversation_masks"]))
    self.assertEqual(item.traj["assistant_turn_lengths"], [30, 20])

  def test_model_call_preserves_explicit_seed_in_generation_kwargs(self):
    async def _run():
      sampler = _MockVanillaSampler()
      req = datatypes.RolloutRequest(
          prompt_id="prompt_42",
          prompt="test prompt",
          group_index=1,
          generation_kwargs={
              "max_generation_steps": 128,
              "seed": 99999,
              "temperature": 0.7,
          },
      )
      mock_agent = mock.MagicMock()
      mock_agent.name = "test_agent"
      engine = collector.TrajectoryCollectorEngine(
          traj_id="traj_1",
          request=req,
          sampler=sampler,
          env_client=mock.MagicMock(),
          agent=mock_agent,
          tokenizer=mock.MagicMock(),
          chat_parser=mock.MagicMock(),
      )

      with mock.patch(
          "tunix.rl.agentic.trajectory.trajectory_collect_engine.TrajectoryCollectEngine"
      ) as mock_engine_cls:
        mock_instance = mock.AsyncMock()
        mock_instance.collect.return_value = {}
        mock_engine_cls.return_value = mock_instance

        await engine.run_episode()

        model_call = mock_engine_cls.call_args.kwargs["model_call"]
        await model_call("prompt text")

      self.assertLen(sampler.calls, 1)
      sampling_req, _ = sampler.calls[0]
      self.assertEqual(sampling_req.sampling_params.seed, 99999)

    asyncio.run(_run())

  def test_model_call_seeds_not_passed_to_vllm(self):
    async def _run():
      sampler = _MockVllmSampler()
      req = datatypes.RolloutRequest(
          prompt_id="prompt_42",
          prompt="test prompt",
          group_index=1,
          generation_kwargs={"max_generation_steps": 128, "temperature": 0.7},
      )
      mock_agent = mock.MagicMock()
      mock_agent.name = "test_agent"
      engine = collector.TrajectoryCollectorEngine(
          traj_id="traj_1",
          request=req,
          sampler=sampler,
          env_client=mock.MagicMock(),
          agent=mock_agent,
          tokenizer=mock.MagicMock(),
          chat_parser=mock.MagicMock(),
      )

      with mock.patch(
          "tunix.rl.agentic.trajectory.trajectory_collect_engine.TrajectoryCollectEngine"
      ) as mock_engine_cls:
        mock_instance = mock.AsyncMock()
        mock_instance.collect.return_value = {}
        mock_engine_cls.return_value = mock_instance

        await engine.run_episode()

        model_call = mock_engine_cls.call_args.kwargs["model_call"]
        await model_call("prompt text")

      self.assertLen(sampler.calls, 1)
      sampling_req, _ = sampler.calls[0]
      self.assertIsNone(sampling_req.sampling_params.seed)

    asyncio.run(_run())

  def test_model_call_propagates_return_routed_experts_in_generation_kwargs(
      self,
  ):
    async def _run():
      mock_routed = np.ones((10, 4, 8), dtype=np.int32)
      sampler = _MockSampler(routed_experts=mock_routed)
      req = datatypes.RolloutRequest(
          prompt_id="prompt_routed",
          prompt="test prompt",
          group_index=0,
          generation_kwargs={
              "max_generation_steps": 128,
              "return_routed_experts": True,
          },
      )
      mock_agent = mock.MagicMock()
      mock_agent.name = "test_agent"
      engine = collector.TrajectoryCollectorEngine(
          traj_id="traj_1",
          request=req,
          sampler=sampler,
          env_client=mock.MagicMock(),
          agent=mock_agent,
          tokenizer=mock.MagicMock(),
          chat_parser=mock.MagicMock(),
      )

      with mock.patch(
          "tunix.rl.agentic.trajectory.trajectory_collect_engine.TrajectoryCollectEngine"
      ) as mock_engine_cls:
        mock_instance = mock.AsyncMock()
        mock_instance.collect.return_value = {}
        mock_engine_cls.return_value = mock_instance

        await engine.run_episode()

        model_call = mock_engine_cls.call_args.kwargs["model_call"]
        output = await model_call("prompt text", env=mock.MagicMock())

      self.assertLen(sampler.sampled_params, 1)
      self.assertTrue(sampler.sampled_params[0].return_routed_experts)
      self.assertIsNotNone(output.routed_experts)
      np.testing.assert_array_equal(output.routed_experts[0], mock_routed)

    asyncio.run(_run())


class _RecordingSampler:

  def __init__(self):
    self.seen_max_tokens = []

  async def sample(self, sampling_req, **kwargs):
    del kwargs
    self.seen_max_tokens.append(sampling_req.sampling_params.max_tokens)
    return sampler_lib.SamplingResponse(
        request_id=getattr(sampling_req, "request_id", "req"),
        text="FINAL_ANSWER: 4",
        prompt_token_ids=np.asarray([1, 2, 3], dtype=np.int32),
        token_ids=np.asarray([4, 5], dtype=np.int32),
        logprobs=np.asarray([0.0, 0.0], dtype=np.float32),
    )


class _FakeInnerEngine:

  next_max_generation_steps = None

  def __init__(self, *, model_call, env, **kwargs):
    del kwargs
    self._model_call = model_call
    self._env = env

  async def collect(self, mode="Trajectory"):
    del mode
    await self._model_call(
        [{"role": "user", "content": "hi"}],
        self._env,
        max_generation_steps=self.next_max_generation_steps,
    )
    return {
        "conversation_text": "",
        "prompt_tokens": np.asarray([1, 2, 3], dtype=np.int32),
        "conversation_tokens": np.array([], dtype=np.int32),
        "conversation_masks": np.array([], dtype=np.float32),
        "old_logprobs": np.array([], dtype=np.float32),
        "trajectory_reward": 0.0,
        "status": "COMPLETED",
        "policy_versions": [0],
    }


class RunEpisodeSamplingParamsTest(absltest.TestCase):

  def _make_collector(self, generation_kwargs: dict[str, object]):
    request = datatypes.RolloutRequest(
        request_id="req_0",
        prompt="What is 2+2?",
        prompt_id="prompt_0",
        group_index=0,
        generation_kwargs=generation_kwargs,
    )
    return collector.TrajectoryCollectorEngine(
        traj_id=request.traj_id,
        request=request,
        sampler=_RecordingSampler(),
        env_client=object(),
        agent=mocks.MockAgent(),
        tokenizer=mocks.MockTokenizer(),
        chat_parser=mocks.MockChatParser(),
    )

  def test_run_episode_uses_request_max_generation_steps_when_unbounded(self):
    engine = self._make_collector({"max_generation_steps": 123})
    _FakeInnerEngine.next_max_generation_steps = None

    with unittest.mock.patch.object(
        collector.rl_collect_engine,
        "TrajectoryCollectEngine",
        _FakeInnerEngine,
    ):
      asyncio.run(engine.run_episode())

    self.assertEqual(engine.sampler.seen_max_tokens, [123])

  def test_run_episode_prefers_explicit_max_generation_steps(self):
    engine = self._make_collector({"max_generation_steps": 123})
    _FakeInnerEngine.next_max_generation_steps = 17

    with unittest.mock.patch.object(
        collector.rl_collect_engine,
        "TrajectoryCollectEngine",
        _FakeInnerEngine,
    ):
      asyncio.run(engine.run_episode())

    self.assertEqual(engine.sampler.seen_max_tokens, [17])

  def test_run_episode_caps_at_request_max_generation_steps_when_episode_budget_larger(
      self,
  ):
    engine = self._make_collector({"max_generation_steps": 123})
    _FakeInnerEngine.next_max_generation_steps = 500

    with unittest.mock.patch.object(
        collector.rl_collect_engine,
        "TrajectoryCollectEngine",
        _FakeInnerEngine,
    ):
      asyncio.run(engine.run_episode())

    self.assertEqual(engine.sampler.seen_max_tokens, [123])

  def test_run_episode_caps_at_episode_budget_when_request_max_generation_steps_larger(
      self,
  ):
    engine = self._make_collector({"max_generation_steps": 123})
    _FakeInnerEngine.next_max_generation_steps = 50

    with unittest.mock.patch.object(
        collector.rl_collect_engine,
        "TrajectoryCollectEngine",
        _FakeInnerEngine,
    ):
      asyncio.run(engine.run_episode())

    self.assertEqual(engine.sampler.seen_max_tokens, [50])


class ConvertTrajectoryItemTest(absltest.TestCase):

  def test_opt_in_episode_summary_preserves_step_and_guard_counts(self):
    request = datatypes.RolloutRequest(
        request_id="eval-1",
        prompt="task",
        prompt_id="task-1",
        generation_kwargs={"max_generation_steps": 8},
        metadata={"record_episode_summary": True},
    )
    agent = mocks.MockAgent()
    agent.trajectory.steps = [
        types.SimpleNamespace(info={"guard_blocked": True,
                                    "guard_reason": "invalid_action"}),
        types.SimpleNamespace(info={"guard_blocked": True,
                                    "guard_reason": "invalid_action"}),
        types.SimpleNamespace(info={"guard_blocked": False}),
    ]
    engine = collector.TrajectoryCollectorEngine(
        traj_id=request.traj_id,
        request=request,
        sampler=_RecordingSampler(),
        env_client=object(),
        agent=agent,
        tokenizer=mocks.MockTokenizer(),
        chat_parser=mocks.MockChatParser(),
    )

    item = engine._convert_to_trajectory({"status": "SUCCEEDED"})

    self.assertEqual(
        item.traj["episode_summary"],
        {
            "num_steps": 3,
            "guard_blocked_steps": 2,
            "guard_reasons": ["invalid_action"],
        },
    )
    self.assertNotIn("record_episode_summary", item.metadata)
    self.assertNotIn("episode_summary", item.metadata)

  def test_convert_to_trajectory_returns_trajectory_item(self):
    request = datatypes.RolloutRequest(
        request_id="req_test",
        prompt="hello",
        prompt_id="prompt_test",
        group_index=2,
        target_policy_version=5,
        generation_kwargs={"max_generation_steps": 64},
        metadata={"custom_key": "custom_val"},
    )
    engine = collector.TrajectoryCollectorEngine(
        traj_id=request.traj_id,
        request=request,
        sampler=_RecordingSampler(),
        env_client=object(),
        agent=mocks.MockAgent(),
        tokenizer=mocks.MockTokenizer(),
        chat_parser=mocks.MockChatParser(),
    )

    rl_traj = {
        "conversation_text": [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "the prompt must not be scored"},
            {"role": "assistant", "content": "first step second step"},
        ],
        "prompt_tokens": np.array([1, 2, 3], dtype=np.int32),
        "conversation_tokens": np.array([10, 11, 12], dtype=np.int32),
        "conversation_masks": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "old_logprobs": np.array([-0.1, -0.2, -0.3], dtype=np.float32),
        "trajectory_reward": 2.5,
        "status": "COMPLETED",
        "policy_versions": [5],
    }

    item = engine._convert_to_trajectory(rl_traj)
    self.assertIsInstance(item, datatypes.TrajectoryItem)
    self.assertEqual(item.prompt_id, "prompt_test")
    self.assertEqual(item.group_index, 2)
    self.assertEqual(item.policy_versions, [5])
    self.assertEqual(item.metadata.get("custom_key"), "custom_val")
    np.testing.assert_array_equal(item.prompt_tokens, [1, 2, 3])
    np.testing.assert_array_equal(item.conversation_tokens, [10, 11, 12])
    np.testing.assert_array_equal(item.conversation_masks, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(item.old_logprobs, [-0.1, -0.2, -0.3])
    self.assertEqual(item.traj, rl_traj)
    # Episode data must live on `traj` only. Mirroring it into metadata let the
    # copy drift from the original and silently corrupted reward scoring.
    self.assertNotIn("text", item.metadata)
    self.assertNotIn("trajectory_reward", item.metadata)
    self.assertEqual(item.traj["trajectory_reward"], 2.5)
    # Episode summaries are opt-in.
    self.assertNotIn("episode_summary", item.traj)
    self.assertEqual(
        datatypes.assistant_text(item.traj["conversation_text"]),
        "first step second step",
    )

  def test_assistant_text_concatenates_assistant_turns_only(self):
    conversation = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    self.assertEqual(datatypes.assistant_text(conversation), "a1a2")

  def test_assistant_text_passes_through_plain_string(self):
    self.assertEqual(
        datatypes.assistant_text("already rendered"), "already rendered"
    )

  def test_assistant_text_handles_empty_and_malformed_entries(self):
    self.assertEqual(datatypes.assistant_text([]), "")
    self.assertEqual(
        datatypes.assistant_text(
            [None, {"role": "assistant"}, {"role": "assistant", "content": "x"}]
        ),
        "x",
    )

  def test_convert_to_trajectory_with_env_tokens_and_masks(self):
    request = datatypes.RolloutRequest(
        request_id="req_multi",
        prompt="hello",
        prompt_id="prompt_multi",
        group_index=0,
        target_policy_version=1,
        generation_kwargs={"max_generation_steps": 64},
    )
    engine = collector.TrajectoryCollectorEngine(
        traj_id=request.traj_id,
        request=request,
        sampler=_RecordingSampler(),
        env_client=object(),
        agent=mocks.MockAgent(),
        tokenizer=mocks.MockTokenizer(),
        chat_parser=mocks.MockChatParser(),
    )

    rl_traj = {
        "conversation_text": "assistant action final response",
        "prompt_tokens": np.array([1, 2], dtype=np.int32),
        "conversation_tokens": np.array([10, 11, 20, 21, 12], dtype=np.int32),
        "conversation_masks": np.array(
            [1.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32
        ),
        "old_logprobs": np.array(
            [-0.1, -0.2, 0.0, 0.0, -0.3], dtype=np.float32
        ),
        "trajectory_reward": 1.0,
        "status": "COMPLETED",
        "policy_versions": [1],
    }

    item = engine._convert_to_trajectory(rl_traj)
    np.testing.assert_array_equal(
        item.conversation_tokens, [10, 11, 20, 21, 12]
    )
    np.testing.assert_array_equal(
        item.conversation_masks, [1.0, 1.0, 0.0, 0.0, 1.0]
    )
    np.testing.assert_allclose(item.old_logprobs, [-0.1, -0.2, 0.0, 0.0, -0.3])

  def test_convert_to_trajectory_rejects_non_dict(self):
    request = datatypes.RolloutRequest(
        request_id="req_fallback",
        prompt="hello",
        prompt_id="prompt_fallback",
        group_index=1,
        target_policy_version=3,
    )
    engine = collector.TrajectoryCollectorEngine(
        traj_id=request.traj_id,
        request=request,
        sampler=_RecordingSampler(),
        env_client=object(),
        agent=mocks.MockAgent(),
        tokenizer=mocks.MockTokenizer(),
        chat_parser=mocks.MockChatParser(),
    )
    mock_traj = types.SimpleNamespace(
        reward=1.5,
        status="COMPLETED",
        text="mock output",
    )
    with self.assertRaisesRegex(TypeError, "Expected rl_traj to be a dict"):
      engine._convert_to_trajectory(mock_traj)

  def test_model_call_respects_min_of_remaining_budget_and_request_max_tokens(
      self,
  ):
    sampler = _MockVllmSampler()
    request = datatypes.RolloutRequest(
        prompt="test",
        prompt_id="p_budget",
        generation_kwargs={"max_tokens": 4, "max_response_length": 16},
    )
    engine = collector.TrajectoryCollectorEngine(
        traj_id=request.traj_id,
        request=request,
        sampler=sampler,
        env_client=object(),
        agent=mocks.MockAgent(),
        tokenizer=mocks.MockTokenizer(),
        chat_parser=mocks.MockChatParser(),
    )
    captured_model_call = None

    def _capture_engine(*args, **kwargs):
      del args
      nonlocal captured_model_call
      captured_model_call = kwargs["model_call"]
      mock_inner = mock.MagicMock()
      mock_inner.collect = mock.AsyncMock(return_value={})
      return mock_inner

    with mock.patch.object(
        collector.rl_collect_engine,
        "TrajectoryCollectEngine",
        side_effect=_capture_engine,
    ):
      asyncio.run(engine.run_episode())

    self.assertIsNotNone(captured_model_call)
    # Remaining budget 12 > request max_tokens 4 -> should use 4
    asyncio.run(captured_model_call("prompt", max_generation_steps=12))
    self.assertEqual(sampler.calls[-1][0].sampling_params.max_tokens, 4)

    # Remaining budget 2 < request max_tokens 4 -> should use 2
    asyncio.run(captured_model_call("prompt", max_generation_steps=2))
    self.assertEqual(sampler.calls[-1][0].sampling_params.max_tokens, 2)


class ResponseBudgetAnnotationTest(absltest.TestCase):
  """Covers the `clipped` / `raw_length` annotations on collected trajectories."""

  class _EosTokenizer(_MockTokenizer):
    eos_token_id = 7

  def _engine(self, max_response_length, tokenizer=None, eos_ids=(7,)):
    request = datatypes.RolloutRequest(
        prompt_id="p1",
        max_response_length=max_response_length,
        generation_kwargs={},
    )
    return collector.TrajectoryCollectorEngine(
        traj_id="t1",
        request=request,
        sampler=_MockSampler(),
        env_client=mock.MagicMock(),
        agent=mock.MagicMock(),
        tokenizer=tokenizer or self._EosTokenizer(),
        chat_parser=_RecordingParser(),
        eos_ids=eos_ids,
    )

  def test_truncated_without_eos_is_clipped(self):
    engine = self._engine(max_response_length=4)
    traj = {"conversation_tokens": np.array([1, 2, 3, 4])}
    metadata = {}

    engine._annotate_response_budget(traj, metadata)

    self.assertTrue(metadata["clipped"])
    self.assertEqual(metadata["raw_length"], 4)
    self.assertNotIn("clipped", traj)
    self.assertNotIn("raw_length", traj)

  def test_budget_reached_but_ending_on_eos_is_not_clipped(self):
    # The boundary the metric hinges on: filling the budget is not truncation
    # if the model still emitted EOS as its final token.
    engine = self._engine(max_response_length=4)
    traj = {"conversation_tokens": np.array([1, 2, 3, 7])}
    metadata = {}

    engine._annotate_response_budget(traj, metadata)

    self.assertFalse(metadata["clipped"])
    self.assertEqual(metadata["raw_length"], 4)

  def test_short_response_is_not_clipped(self):
    engine = self._engine(max_response_length=8)
    traj = {"conversation_tokens": np.array([1, 2, 7])}
    metadata = {}

    engine._annotate_response_budget(traj, metadata)

    self.assertFalse(metadata["clipped"])
    self.assertEqual(metadata["raw_length"], 3)

  def test_overlong_response_clamps_raw_length(self):
    engine = self._engine(max_response_length=3)
    traj = {"conversation_tokens": np.array([1, 2, 3, 4, 5])}
    metadata = {}

    engine._annotate_response_budget(traj, metadata)

    self.assertTrue(metadata["clipped"])
    self.assertEqual(metadata["raw_length"], 3)

  def test_per_request_budget_overrides_program_default(self):
    # DistributedRLEngine lets a dataset item override max_response_length, so
    # the flag must follow the budget this request actually ran with rather
    # than any program-level default.
    tight = self._engine(max_response_length=4)
    loose = self._engine(max_response_length=64)
    tokens = np.array([1, 2, 3, 4])

    tight_meta = {}
    loose_meta = {}
    tight._annotate_response_budget({"conversation_tokens": tokens}, tight_meta)
    loose._annotate_response_budget({"conversation_tokens": tokens}, loose_meta)

    self.assertTrue(tight_meta["clipped"])
    self.assertFalse(loose_meta["clipped"])

  def test_configured_stop_token_ends_the_rollout_cleanly(self):
    # The case this metric exists to distinguish. A Qwen chat run launched with
    # --eos_tokens='<|im_end|>' terminates on that token, not on the
    # tokenizer's own EOS; scoring against the tokenizer default would call
    # every normal termination at the budget a truncation.
    engine = self._engine(max_response_length=4, eos_ids=[151645])
    traj = {"conversation_tokens": np.array([1, 2, 3, 151645])}
    metadata = {}

    engine._annotate_response_budget(traj, metadata)

    self.assertFalse(metadata["clipped"])
    self.assertEqual(metadata["raw_length"], 4)

  def test_configured_stop_set_replaces_the_tokenizer_default(self):
    # The sampler stops on the configured set only, so the tokenizer's EOS
    # (7 here) is just another token and does not end the rollout.
    engine = self._engine(max_response_length=4, eos_ids=[151645])
    traj = {"conversation_tokens": np.array([1, 2, 3, 7])}
    metadata = {}

    engine._annotate_response_budget(traj, metadata)

    self.assertTrue(metadata["clipped"])

  def test_any_member_of_the_stop_set_counts(self):
    engine = self._engine(max_response_length=4, eos_ids=[151643, 151645])
    traj = {"conversation_tokens": np.array([1, 2, 3, 151643])}
    metadata = {}

    engine._annotate_response_budget(traj, metadata)

    self.assertFalse(metadata["clipped"])

  def test_no_budget_leaves_trajectory_unannotated(self):
    engine = self._engine(max_response_length=None)
    traj = {"conversation_tokens": np.array([1, 2, 3])}
    metadata = {}

    with (
        mock.patch.object(
            collector.logging, "_get_next_log_count_per_token", return_value=0
        ),
        self.assertLogs(level="WARNING") as cm,
    ):
      engine._annotate_response_budget(traj, metadata)

    self.assertIn("no max_response_length", cm.output[0])
    self.assertNotIn("clipped", metadata)
    self.assertNotIn("raw_length", metadata)

  def test_non_positive_budget_leaves_trajectory_unannotated(self):
    # Scoring against a zero or negative budget would mark every rollout
    # clipped, which is worse than reporting nothing.
    for invalid_budget in (0, -1):
      engine = self._engine(max_response_length=invalid_budget)
      traj = {"conversation_tokens": np.array([1, 2, 3])}
      metadata = {}

      with (
          mock.patch.object(
              collector.logging,
              "_get_next_log_count_per_token",
              return_value=0,
          ),
          self.assertLogs(level="WARNING") as cm,
      ):
        engine._annotate_response_budget(traj, metadata)

      self.assertIn("not a usable budget", cm.output[0])
      self.assertNotIn("clipped", metadata)
      self.assertNotIn("raw_length", metadata)

  def test_unset_eos_ids_leaves_trajectory_unannotated(self):
    # Even when the tokenizer exposes eos_token_id=7, the framework must not
    # force it when eos_ids is unset/None, because stop tokens are defined at
    # the recipe level (RolloutConfig.eos_tokens).
    engine = self._engine(
        max_response_length=4,
        tokenizer=self._EosTokenizer(),
        eos_ids=None,
    )
    traj = {"conversation_tokens": np.array([1, 2, 3, 7])}
    metadata = {}

    engine._annotate_response_budget(traj, metadata)

    self.assertNotIn("clipped", metadata)
    self.assertNotIn("raw_length", metadata)

  def test_empty_response_is_annotated_as_zero_length(self):
    # A rollout that produced nothing still ran, and must keep its slot in the
    # group denominator; the agentic learner scores it as length 0, unclipped.
    engine = self._engine(max_response_length=4)
    traj = {"conversation_tokens": np.array([], dtype=np.int32)}
    metadata = {}

    engine._annotate_response_budget(traj, metadata)

    self.assertFalse(metadata["clipped"])
    self.assertEqual(metadata["raw_length"], 0)

  def test_raw_length_counts_env_tokens_not_just_assistant_tokens(self):
    # Raw length spans the whole response. `conversation_masks` is the
    # assistant-only loss mask, covering 2 of these 5 tokens; deriving the
    # length from it instead would undercount multi-turn rollouts, which is
    # exactly what rollout/completion_length_mean already does.
    engine = self._engine(max_response_length=8)
    traj = {
        "conversation_tokens": np.array([1, 2, 3, 4, 5]),
        "conversation_masks": np.array([1, 1, 0, 0, 0]),
    }
    metadata = {}

    engine._annotate_response_budget(traj, metadata)

    self.assertEqual(metadata["raw_length"], 5)

  def test_missing_token_stream_leaves_trajectory_unannotated(self):
    engine = self._engine(max_response_length=4)
    traj = {}
    metadata = {}

    engine._annotate_response_budget(traj, metadata)

    self.assertNotIn("clipped", metadata)
    self.assertNotIn("raw_length", metadata)

  def test_convert_to_trajectory_annotates_metadata_without_mutating_traj(self):
    engine = self._engine(max_response_length=4)
    raw_traj = {"conversation_tokens": np.array([1, 2, 3, 4])}

    item = engine._convert_to_trajectory(raw_traj)

    self.assertTrue(item.metadata["clipped"])
    self.assertEqual(item.metadata["raw_length"], 4)
    self.assertTrue(item.clipped)
    self.assertEqual(item.raw_length, 4)
    self.assertNotIn("clipped", raw_traj)
    self.assertNotIn("raw_length", raw_traj)

  def test_response_budget_facts_helper(self):
    self.assertEqual(
        collector.response_budget_facts([10, 20, 30], 4, {99}),
        (3, False),
    )
    self.assertEqual(
        collector.response_budget_facts([10, 20, 30, 40], 4, {99}),
        (4, True),
    )
    self.assertEqual(
        collector.response_budget_facts([10, 20, 30, 99], 4, {99}),
        (4, False),
    )
    self.assertEqual(
        collector.response_budget_facts(
            np.array([1, 2, 3, 151645], dtype=np.int64), 4, {151643, 151645}
        ),
        (4, False),
    )
    self.assertEqual(
        collector.response_budget_facts([1, 2, 3, 4, 5], 4, {99}),
        (4, True),
    )
    self.assertEqual(
        collector.response_budget_facts([10, 20, 30, 40, 99], 4, {99}),
        (4, True),
    )
    self.assertEqual(
        collector.response_budget_facts([10, 20, 30, 99, 108], 4, {99}),
        (4, True),
    )
    self.assertEqual(
        collector.response_budget_facts([], 4, {99}),
        (0, False),
    )
    with self.assertRaises(ValueError):
      collector.response_budget_facts([], 0, {99})
    with self.assertRaises(ValueError):
      collector.response_budget_facts([1, 2], -1, {99})

  def test_convert_to_trajectory_preserves_routed_experts(self):
    request = datatypes.RolloutRequest(
        request_id="req_routed",
        prompt="hello",
        prompt_id="prompt_routed",
        group_index=0,
    )
    engine = collector.TrajectoryCollectorEngine(
        traj_id=request.traj_id,
        request=request,
        sampler=_RecordingSampler(),
        env_client=object(),
        agent=mocks.MockAgent(),
        tokenizer=mocks.MockTokenizer(),
        chat_parser=mocks.MockChatParser(),
    )
    mock_routed = np.ones((10, 4, 8), dtype=np.int32)
    rl_traj = {
        "conversation_text": "step",
        "prompt_tokens": np.array([1, 2], dtype=np.int32),
        "conversation_tokens": np.array([10, 11], dtype=np.int32),
        "conversation_masks": np.array([1.0, 1.0], dtype=np.float32),
        "old_logprobs": np.array([-0.1, -0.2], dtype=np.float32),
        "routed_experts": mock_routed,
        "trajectory_reward": 1.0,
        "status": "COMPLETED",
        "policy_versions": [1],
    }
    item = engine._convert_to_trajectory(rl_traj)
    self.assertNotIn("routed_experts", item.metadata)
    np.testing.assert_array_equal(item.traj["routed_experts"], mock_routed)
    np.testing.assert_array_equal(item.routed_experts, mock_routed)

  def test_multi_turn_token_continuity_passes_prompt_token_ids(self):
    class _TwoTurnEnv(base_environment.BaseTaskEnv):

      def __init__(self):
        super().__init__(task={"question": "q0"}, max_steps=2)
        self._step_idx = 0

      def _initial_observation(self):
        return "obs0"

      def _step_impl(self, action):
        self._step_idx += 1
        return base_environment.EnvStepResult(
            observation="obs1",
            reward=1.0,
            done=self._step_idx >= 2,
            info={},
        )

    class _TokenAwareSampler(sampler_lib.Sampler):

      def __init__(self):
        self.requests = []

      async def sample(self, req, **kwargs):
        del kwargs
        self.requests.append(req)
        turn = len(self.requests)
        prompt_toks = (
            np.asarray(req.prompt, dtype=np.int32)
            if isinstance(req.prompt, np.ndarray)
            else np.array([10, 11], dtype=np.int32)
        )
        return sampler_lib.SamplingResponse(
            request_id=req.request_id,
            text=f"ans_{turn}",
            token_ids=np.array([20 + turn, 21 + turn], dtype=np.int32),
            prompt_token_ids=prompt_toks,
            logprobs=np.array([-0.1, -0.2], dtype=np.float32),
        )

    sampler = _TokenAwareSampler()
    parser = _RecordingParser()
    tokenizer = _MockTokenizer()
    agent = model_agent.ModelAgent("sys")
    env = _TwoTurnEnv()
    req = datatypes.RolloutRequest(
        prompt_id="p_tito",
        prompt="q0",
        max_response_length=64,
    )
    engine = collector.TrajectoryCollectorEngine(
        traj_id="traj_tito",
        request=req,
        sampler=sampler,
        env_client=env,
        agent=agent,
        tokenizer=tokenizer,
        chat_parser=parser,
    )
    item = asyncio.run(engine.run_episode())
    self.assertLen(sampler.requests, 2)
    # Turn 0 uses formatted string prompt from _build_prompt.
    self.assertIsInstance(sampler.requests[0].prompt, str)
    # Turn 1 bypasses _build_prompt and passes continuation token IDs directly.
    self.assertIsInstance(sampler.requests[1].prompt, np.ndarray)
    expected_env_tokens = np.asarray(tokenizer.encode("PARSED"), dtype=np.int32)
    expected_turn1_prompt = np.concatenate([
        np.array([10, 11], dtype=np.int32),
        np.array([21, 22], dtype=np.int32),
        expected_env_tokens,
    ])
    np.testing.assert_array_equal(
        sampler.requests[1].prompt, expected_turn1_prompt
    )
    np.testing.assert_array_equal(item.traj["prompt_tokens"], [10, 11])

  def test_partial_rollout_pins_turn1_cache_salt_and_tracks_policy_versions(
      self,
  ):
    class _TwoTurnEnv(base_environment.BaseTaskEnv):

      def __init__(self):
        super().__init__(task={"question": "q0"}, max_steps=2)
        self._step_idx = 0

      def _initial_observation(self):
        return "obs0"

      def _step_impl(self, action):
        self._step_idx += 1
        return base_environment.EnvStepResult(
            observation="obs1",
            reward=1.0,
            done=self._step_idx >= 2,
            info={},
        )

    current_policy_version = [2]

    class _VersionAdvancingSampler(sampler_lib.Sampler):

      def __init__(self):
        self.requests = []

      async def sample(self, req, **kwargs):
        del kwargs
        self.requests.append(req)
        turn = len(self.requests)
        if turn == 1:
          # Simulate in-flight weight sync advancing policy version between
          # turns.
          current_policy_version[0] = 3
        prompt_toks = (
            np.asarray(req.prompt, dtype=np.int32)
            if isinstance(req.prompt, np.ndarray)
            else np.array([10, 11], dtype=np.int32)
        )
        return sampler_lib.SamplingResponse(
            request_id=req.request_id,
            text=f"ans_{turn}",
            token_ids=np.array([20 + turn, 21 + turn], dtype=np.int32),
            prompt_token_ids=prompt_toks,
            logprobs=np.array([-0.1, -0.2], dtype=np.float32),
        )

    sampler = _VersionAdvancingSampler()
    req = datatypes.RolloutRequest(
        prompt_id="p_partial",
        prompt="q0",
        target_policy_version=2,
        max_response_length=64,
    )
    engine = collector.TrajectoryCollectorEngine(
        traj_id="traj_partial",
        request=req,
        sampler=sampler,
        env_client=_TwoTurnEnv(),
        agent=model_agent.ModelAgent("sys"),
        tokenizer=_MockTokenizer(),
        chat_parser=_RecordingParser(),
        partial_rollout=True,
        policy_version_fn=lambda: current_policy_version[0],
    )
    item = asyncio.run(engine.run_episode())
    self.assertLen(sampler.requests, 2)
    # Both Turn 1 and Turn 2 must use the Turn-1 cache_salt ("policy_v2")
    self.assertEqual(
        sampler.requests[0].sampling_params.cache_salt, "policy_v2"
    )
    self.assertEqual(
        sampler.requests[1].sampling_params.cache_salt, "policy_v2"
    )
    # Per-turn history is recorded in policy_versions on item.traj (SSOT).
    self.assertEqual(item.policy_versions, [2, 3])
    self.assertNotIn("policy_version", item.traj)
    self.assertEqual(item.traj["policy_versions"], [2, 3])
    self.assertNotIn("policy_version", item.metadata)
    self.assertNotIn("policy_versions", item.metadata)

  def test_partial_rollout_handles_arbitrary_v_first_and_advancing_versions(
      self,
  ):
    class _ThreeTurnEnv(base_environment.BaseTaskEnv):

      def __init__(self):
        super().__init__(task={"question": "q0"}, max_steps=3)
        self._step_idx = 0

      def _initial_observation(self):
        return "obs0"

      def _step_impl(self, action):
        self._step_idx += 1
        return base_environment.EnvStepResult(
            observation=f"obs{self._step_idx}",
            reward=1.0,
            done=self._step_idx >= 3,
            info={},
        )

    # Turn 0 starts at arbitrary v_first=42, Turn 1 advances to 45, Turn 2
    # advances to 48.
    turn_versions = [42, 45, 48]
    current_policy_version = [turn_versions[0]]

    class _DynamicVersionSampler(sampler_lib.Sampler):

      def __init__(self):
        self.requests = []

      async def sample(self, req, **kwargs):
        del kwargs
        self.requests.append(req)
        turn = len(self.requests)
        if turn < len(turn_versions):
          current_policy_version[0] = turn_versions[turn]
        prompt_toks = (
            np.asarray(req.prompt, dtype=np.int32)
            if isinstance(req.prompt, np.ndarray)
            else np.array([10, 11], dtype=np.int32)
        )
        return sampler_lib.SamplingResponse(
            request_id=req.request_id,
            text=f"ans_{turn}",
            token_ids=np.array([20 + turn, 21 + turn], dtype=np.int32),
            prompt_token_ids=prompt_toks,
            logprobs=np.array([-0.1, -0.2], dtype=np.float32),
        )

    sampler = _DynamicVersionSampler()
    req = datatypes.RolloutRequest(
        prompt_id="p_arbitrary_start",
        prompt="q0",
        target_policy_version=42,
        max_response_length=64,
    )
    engine = collector.TrajectoryCollectorEngine(
        traj_id="traj_arbitrary_start",
        request=req,
        sampler=sampler,
        env_client=_ThreeTurnEnv(),
        agent=model_agent.ModelAgent("sys"),
        tokenizer=_MockTokenizer(),
        chat_parser=_RecordingParser(),
        partial_rollout=True,
        policy_version_fn=lambda: current_policy_version[0],
    )
    item = asyncio.run(engine.run_episode())
    self.assertEqual(item.traj["policy_versions"], [42, 45, 48])
    self.assertEqual(item.policy_versions, [42, 45, 48])
    self.assertNotIn("policy_version", item.traj)
    self.assertNotIn("policy_version", item.metadata)

  def test_convert_to_trajectory_preserves_store_recovered_step_versions(self):
    request = datatypes.RolloutRequest(
        request_id="req_recovered",
        prompt="hello",
        prompt_id="prompt_recovered",
        group_index=0,
        target_policy_version=46,
    )
    engine = collector.TrajectoryCollectorEngine(
        traj_id=request.traj_id,
        request=request,
        sampler=_RecordingSampler(),
        env_client=object(),
        agent=mocks.MockAgent(),
        tokenizer=mocks.MockTokenizer(),
        chat_parser=mocks.MockChatParser(),
    )
    # Resumed worker invocation only observed Turn 2 live (v=46), while
    # step-backed rl_traj["policy_versions"] carries pre-crash turns [42, 43]
    # restored from TrajectoryStore plus live turn [46].
    rl_traj = {
        "conversation_text": "recovered rollout",
        "prompt_tokens": np.array([1, 2], dtype=np.int32),
        "conversation_tokens": np.array([10, 11], dtype=np.int32),
        "conversation_masks": np.array([1.0, 1.0], dtype=np.float32),
        "old_logprobs": np.array([-0.1, -0.2], dtype=np.float32),
        "trajectory_reward": 1.0,
        "status": "COMPLETED",
        "policy_versions": [42, 43, 46],
    }
    item = engine._convert_to_trajectory(rl_traj)
    self.assertEqual(item.traj["policy_versions"], [42, 43, 46])
    self.assertEqual(item.policy_versions, [42, 43, 46])
    self.assertNotIn("policy_version", item.traj)
    self.assertNotIn("policy_version", item.metadata)

  def test_pause_blocks_model_call_until_resume(self):
    async def _run():
      sampler = _MockVllmSampler()
      req = datatypes.RolloutRequest(
          prompt_id="p_pause",
          prompt="q0",
          generation_kwargs={"max_generation_steps": 16},
      )
      mock_agent = mock.MagicMock()
      mock_agent.name = "test_agent"
      engine = collector.TrajectoryCollectorEngine(
          traj_id="traj_pause",
          request=req,
          sampler=sampler,
          env_client=mock.MagicMock(),
          agent=mock_agent,
          tokenizer=mock.MagicMock(),
          chat_parser=mock.MagicMock(),
      )
      with mock.patch(
          "tunix.rl.agentic.trajectory.trajectory_collect_engine.TrajectoryCollectEngine"
      ) as mock_engine_cls:
        mock_instance = mock.AsyncMock()
        mock_instance.collect.return_value = {}
        mock_engine_cls.return_value = mock_instance
        await engine.run_episode()
        model_call = mock_engine_cls.call_args.kwargs["model_call"]

        engine.pause()
        self.assertTrue(engine.is_paused)
        call_task = asyncio.create_task(model_call("prompt text"))
        await asyncio.sleep(0.02)
        self.assertFalse(call_task.done())
        self.assertEmpty(sampler.calls)

        engine.resume()
        self.assertFalse(engine.is_paused)
        await call_task
        self.assertLen(sampler.calls, 1)

    asyncio.run(_run())


class RolloutCompactRoutedExpertsEncodingTest(absltest.TestCase):
  """Tests rollout-stage compact uint8/uint10 encoding of routed_experts."""

  def _make_single_turn_env(self):
    class _SingleTurnEnv(base_environment.BaseTaskEnv):

      def _initial_observation(self):
        return {"observation": "start"}

      def _step_impl(self, action):
        del action
        return base_environment.EnvStepResult(
            observation={"observation": "done"},
            reward=1.0,
            done=True,
            info={},
        )

    return _SingleTurnEnv(task={"question": "q0", "answer": "a0"}, max_steps=1)

  def test_collector_init_validates_routed_experts_encode_config(self):
    req = datatypes.RolloutRequest(prompt_id="p0", prompt="hi")
    with self.assertRaisesRegex(
        ValueError, "routed_experts_encode_mode must be one of"
    ):
      collector.TrajectoryCollectorEngine(
          traj_id="t0",
          request=req,
          sampler=_MockSampler(),
          env_client=mock.MagicMock(),
          agent=mock.MagicMock(),
          tokenizer=_MockTokenizer(),
          chat_parser=_RecordingParser(),
          routed_experts_encode_mode="bad_mode",
      )
    for bad_num in (0, -2, True, False):
      with self.assertRaisesRegex(
          ValueError, "routed_experts_num_experts must be a positive int"
      ):
        collector.TrajectoryCollectorEngine(
            traj_id="t0",
            request=req,
            sampler=_MockSampler(),
            env_client=mock.MagicMock(),
            agent=mock.MagicMock(),
            tokenizer=_MockTokenizer(),
            chat_parser=_RecordingParser(),
            routed_experts_num_experts=bad_num,
        )

  def test_run_episode_encodes_routed_experts_in_rollout_and_preserves_store(
      self,
  ):
    # Prompt has 2 tokens ([1, 2]), response has 3 tokens ([0, 1, 2]), so
    # total_len = 5 and sampler returns 4 R3 steps [4, 2, 4] (2 prefill + 2
    # decode tokens; final generated token is padded with -1 by collect()).
    raw_routed_steps = np.array(
        [
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            [[8, 9, 10, 11], [12, 13, 14, 15]],
            [[16, 17, 18, 19], [20, 21, 22, 23]],
            [[24, 25, 26, 27], [28, 29, 30, 31]],
        ],
        dtype=np.int16,
    )
    expected_full_routed = np.concatenate(
        [raw_routed_steps, np.full((1, 2, 4), -1, dtype=np.int16)], axis=0
    )
    req = datatypes.RolloutRequest(
        prompt_id="p_routed",
        prompt="What is 2+2?",
        group_index=0,
        max_response_length=16,
        generation_kwargs={"return_routed_experts": True},
    )

    # 1. Default mode="auto" with max_val=31 <= 254 produces compact 3D uint8
    # both on TrajectoryItem.routed_experts and in TrajectoryStore's per-turn
    # prefill_routed_experts.
    store = in_memory_store.InMemoryTrajectoryStore()
    engine_u8 = collector.TrajectoryCollectorEngine(
        traj_id="traj_u8",
        request=req,
        sampler=_MockSampler(
            token_lengths=[3], routed_experts=raw_routed_steps
        ),
        env_client=self._make_single_turn_env(),
        agent=model_agent.ModelAgent("sys"),
        tokenizer=_MockTokenizer(),
        chat_parser=_RecordingParser(),
        trajectory_store=store,
        store_routed_experts=True,
    )
    item_u8 = asyncio.run(engine_u8.run_episode())
    self.assertEqual(item_u8.routed_experts.dtype, np.uint8)
    self.assertEqual(item_u8.routed_experts.shape, (5, 2, 4))
    np.testing.assert_array_equal(
        packing.decode_routed_experts(item_u8.routed_experts, batched=False),
        expected_full_routed,
    )
    stored_trajs = store.get_trajectories(["traj_u8"])
    self.assertLen(stored_trajs, 1)
    # Step 1 is the TunixAgentStep containing prefill_routed_experts (2 prompt
    # tokens + 2 routed decode tokens).
    agent_steps = [
        s for s in stored_trajs[0].steps if s.source.value == "agent"
    ]
    self.assertLen(agent_steps, 1)
    self.assertEqual(agent_steps[0].prefill_routed_experts.dtype, np.uint8)
    self.assertEqual(agent_steps[0].prefill_routed_experts.shape, (4, 2, 4))
    np.testing.assert_array_equal(
        packing.decode_routed_experts(
            agent_steps[0].prefill_routed_experts, batched=False
        ),
        raw_routed_steps,
    )

    # 2. Deterministic tier via routed_experts_num_experts=256 encodes into 4D
    # uint10 [T, L, K//4, 5] both in TrajectoryItem and TrajectoryStore.
    engine_u10 = collector.TrajectoryCollectorEngine(
        traj_id="traj_u10",
        request=req,
        sampler=_MockSampler(
            token_lengths=[3], routed_experts=raw_routed_steps
        ),
        env_client=self._make_single_turn_env(),
        agent=model_agent.ModelAgent("sys"),
        tokenizer=_MockTokenizer(),
        chat_parser=_RecordingParser(),
        trajectory_store=store,
        store_routed_experts=True,
        routed_experts_encode_mode="auto",
        routed_experts_num_experts=256,
    )
    item_u10 = asyncio.run(engine_u10.run_episode())
    self.assertEqual(item_u10.routed_experts.dtype, np.uint8)
    self.assertEqual(item_u10.routed_experts.shape, (5, 2, 1, 5))
    np.testing.assert_array_equal(
        packing.decode_routed_experts(item_u10.routed_experts, batched=False),
        expected_full_routed,
    )
    stored_u10 = store.get_trajectories(["traj_u10"])
    agent_steps_u10 = [
        s for s in stored_u10[0].steps if s.source.value == "agent"
    ]
    self.assertEqual(agent_steps_u10[0].prefill_routed_experts.dtype, np.uint8)
    self.assertEqual(
        agent_steps_u10[0].prefill_routed_experts.shape, (4, 2, 1, 5)
    )
    np.testing.assert_array_equal(
        packing.decode_routed_experts(
            agent_steps_u10[0].prefill_routed_experts, batched=False
        ),
        raw_routed_steps,
    )

    # 3. Explicit mode="int16" preserves 3D int16.
    engine_i16 = collector.TrajectoryCollectorEngine(
        traj_id="traj_i16",
        request=req,
        sampler=_MockSampler(
            token_lengths=[3], routed_experts=raw_routed_steps
        ),
        env_client=self._make_single_turn_env(),
        agent=model_agent.ModelAgent("sys"),
        tokenizer=_MockTokenizer(),
        chat_parser=_RecordingParser(),
        trajectory_store=store,
        store_routed_experts=True,
        routed_experts_encode_mode="int16",
    )
    item_i16 = asyncio.run(engine_i16.run_episode())
    self.assertEqual(item_i16.routed_experts.dtype, np.int16)
    self.assertEqual(item_i16.routed_experts.shape, (5, 2, 4))
    np.testing.assert_array_equal(item_i16.routed_experts, expected_full_routed)
    stored_i16 = store.get_trajectories(["traj_i16"])
    agent_steps_i16 = [
        s for s in stored_i16[0].steps if s.source.value == "agent"
    ]
    self.assertEqual(agent_steps_i16[0].prefill_routed_experts.dtype, np.int16)
    np.testing.assert_array_equal(
        agent_steps_i16[0].prefill_routed_experts,
        raw_routed_steps,
    )

  def test_end_to_end_rollout_encoded_to_adapter_and_assemblers(self):
    raw_g0 = np.array(
        [
            [[0, 1, 2, 3], [4, 5, 6, 7]],
            [[8, 9, 10, 11], [12, 13, 14, 15]],
            [[16, 17, 18, 19], [20, 21, 22, 23]],
            [[24, 25, 26, 27], [28, 29, 30, 31]],
        ],
        dtype=np.int16,
    )
    raw_g1 = np.array(
        [
            [[100, 101, 102, 103], [104, 105, 106, 107]],
            [[108, 109, 110, 111], [112, 113, 114, 115]],
            [[116, 117, 118, 119], [120, 121, 122, 123]],
            [[124, 125, 126, 127], [250, 251, 252, 254]],
        ],
        dtype=np.int16,
    )
    last_payloads = []
    for num_experts, expected_unbatched_shape, expected_batched_tail in (
        (255, (5, 2, 4), (2, 4)),
        (256, (5, 2, 1, 5), (2, 1, 5)),
    ):
      with self.subTest(num_experts=num_experts):
        items = []
        for g_idx, raw_routed in enumerate((raw_g0, raw_g1)):
          req = datatypes.RolloutRequest(
              prompt_id="p_e2e",
              prompt="What is 2+2?",
              group_index=g_idx,
              max_response_length=16,
              generation_kwargs={"return_routed_experts": True},
          )
          engine = collector.TrajectoryCollectorEngine(
              traj_id=f"traj_g{g_idx}",
              request=req,
              sampler=_MockSampler(
                  token_lengths=[3], routed_experts=raw_routed
              ),
              env_client=self._make_single_turn_env(),
              agent=model_agent.ModelAgent("sys"),
              tokenizer=_MockTokenizer(),
              chat_parser=_RecordingParser(),
              routed_experts_encode_mode="auto",
              routed_experts_num_experts=num_experts,
          )
          items.append(asyncio.run(engine.run_episode()))
          self.assertEqual(
              items[-1].routed_experts.shape, expected_unbatched_shape
          )

        algo_cfg = algorithm_config.GRPOConfig(
            num_generations=2,
            routed_experts_encode_mode="auto",
            routed_experts_num_experts=num_experts,
        )
        payloads = payload_builder.create_trainer_payloads(
            items, rewards=[0.0, 1.0], algo_config=algo_cfg
        )
        self.assertTrue(
            np.shares_memory(
                payloads[0].routed_experts, items[0].routed_experts
            )
        )
        self.assertTrue(
            np.shares_memory(
                payloads[1].routed_experts, items[1].routed_experts
            )
        )
        last_payloads = payloads

        # PaddedBatchAssembler
        padded_asm = batch_assembly.PaddedBatchAssembler(
            batch_size=2,
            num_generations=2,
            mini_batch_size=1,
            max_prompt_length=4,
            max_response_length=4,
            pad_id=0,
        )
        padded_batches = list(padded_asm.feed(payloads))
        self.assertLen(padded_batches, 1)
        pb = padded_batches[0].payload
        self.assertEqual(pb.routed_experts.dtype, np.uint8)
        self.assertEqual(
            pb.routed_experts.shape, (2, 8) + expected_batched_tail
        )
        decoded_pb = packing.decode_routed_experts(
            pb.routed_experts, batched=True
        )
        # Left-padded prompt (pad=2, prompt=2) + right-padded completion
        # (3, pad=1): valid R3 steps sit at indices 2..5 (4 steps), index 6 is
        # final token (-1).
        np.testing.assert_array_equal(decoded_pb[0, 2:6], raw_g0)
        np.testing.assert_array_equal(decoded_pb[1, 2:6], raw_g1)
        np.testing.assert_array_equal(decoded_pb[:, :2], -1)
        np.testing.assert_array_equal(decoded_pb[:, 6:], -1)

        # SequencePackedBatchAssembler
        packed_asm = batch_assembly.SequencePackedBatchAssembler(
            batch_size=1,
            num_generations=2,
            mini_batch_size=1,
            max_packed_len=16,
            pad_id=0,
            segment_alignment_boundary=1,
        )
        packed_batches = list(packed_asm.feed(payloads))
        self.assertLen(packed_batches, 1)
        sb = packed_batches[0].payload
        self.assertEqual(sb.routed_experts.dtype, np.uint8)
        self.assertEqual(
            sb.routed_experts.shape, (1, 16) + expected_batched_tail
        )
        decoded_sb = packing.decode_routed_experts(
            sb.routed_experts, batched=True
        )
        # Sequence 0 occupies [0..4] (5 tokens; first 4 have R3 routing, token 4
        # is final token with -1). Sequence 1 occupies [5..9] (first 4 at 5..8).
        np.testing.assert_array_equal(decoded_sb[0, 0:4], raw_g0)
        np.testing.assert_array_equal(decoded_sb[0, 4], -1)
        np.testing.assert_array_equal(decoded_sb[0, 5:9], raw_g1)
        np.testing.assert_array_equal(decoded_sb[0, 9:], -1)

    # Mixed fleet: one int16 item + one uint8 item widens losslessly to int16.
    full_g0 = np.concatenate(
        [raw_g0, np.full((1, 2, 4), -1, dtype=np.int16)], axis=0
    )
    full_g1 = np.concatenate(
        [raw_g1, np.full((1, 2, 4), -1, dtype=np.int16)], axis=0
    )
    payloads_mixed = [
        dataclasses.replace(last_payloads[0], routed_experts=full_g0),
        dataclasses.replace(
            last_payloads[1],
            routed_experts=packing.encode_routed_experts(full_g1),
        ),
    ]
    packed_mixed_asm = batch_assembly.SequencePackedBatchAssembler(
        batch_size=1,
        num_generations=2,
        mini_batch_size=1,
        max_packed_len=16,
        pad_id=0,
        segment_alignment_boundary=1,
    )
    mixed_batch = list(packed_mixed_asm.feed(payloads_mixed))[0].payload
    self.assertEqual(mixed_batch.routed_experts.dtype, np.int16)
    np.testing.assert_array_equal(mixed_batch.routed_experts[0, 0:4], raw_g0)
    np.testing.assert_array_equal(mixed_batch.routed_experts[0, 5:9], raw_g1)

  def test_run_episode_strips_binary_env_config_from_item_store_and_logger(
      self,
  ):
    class _PromptTaskEnv(base_environment.BaseTaskEnv):

      def _initial_observation(self):
        return "Solve Q0"

      def _step_impl(self, action):
        del action
        return base_environment.EnvStepResult(
            observation="",
            reward=1.0,
            done=True,
            info={},
        )

    req = datatypes.RolloutRequest(
        prompt_id="p_gsm",
        prompt="",
        group_index=0,
        max_response_length=16,
        metadata={
            "question": "Q0",
            "answer": "0",
            "env_config": {
                "prompt": "Solve Q0",
                "verifier_bytes": b"\x80\xff",
            },
        },
    )
    with file_store.FileTrajectoryStore(
        self.create_tempdir().full_path
    ) as store:
      engine = collector.TrajectoryCollectorEngine(
          traj_id="traj_bin",
          request=req,
          sampler=_MockSampler(token_lengths=[3]),
          env_client=_PromptTaskEnv(task={}, max_steps=1),
          agent=model_agent.ModelAgent("sys"),
          tokenizer=_MockTokenizer(),
          chat_parser=_RecordingParser(),
          trajectory_store=store,
      )
      item = asyncio.run(engine.run_episode())
      store.flush()
      stored = store.get_trajectories(["traj_bin"])

    self.assertNotIn("env_config", item.metadata)
    self.assertEqual(item.metadata["question"], "Q0")
    self.assertEqual(item.metadata["answer"], "0")
    self.assertEqual(req.metadata["env_config"]["verifier_bytes"], b"\x80\xff")
    self.assertLen(stored, 1)
    self.assertEqual(stored[0].extra, {"question": "Q0", "answer": "0"})

  def test_multi_turn_trajectory_store_compression_and_jax_decode_parity(self):
    rng = np.random.default_rng(42)

    class _MultiStepTextEnv(base_environment.BaseTaskEnv):

      def _initial_observation(self):
        return "start"

      def _step_impl(self, action):
        del action
        return base_environment.EnvStepResult(
            observation="env_feedback",
            reward=1.0,
            done=False,
            info={},
        )

    class _MultiTurnRoutedSampler(sampler_lib.Sampler):

      def __init__(self, asst_lengths, routed_per_turn):
        self._asst_lengths = asst_lengths
        self._routed_per_turn = routed_per_turn
        self._idx = 0

      async def sample(self, req, **kwargs):
        del kwargs
        i = self._idx
        self._idx += 1
        req_prompt = getattr(req, "prompt", None)
        prompt_token_ids = (
            np.asarray(req_prompt, dtype=np.int32)
            if isinstance(req_prompt, np.ndarray)
            else np.asarray([ord(c) % 1000 for c in "PARSED"], dtype=np.int32)
        )
        return sampler_lib.SamplingResponse(
            request_id=getattr(req, "request_id", ""),
            text=f"turn_{i}",
            token_ids=np.arange(1, self._asst_lengths[i] + 1, dtype=np.int32),
            prompt_token_ids=prompt_token_ids,
            routed_experts=self._routed_per_turn[i],
        )

    prompt_len = 6
    env_len = 6
    asst_lengths = [128, 128, 113]
    step_lengths = [
        prompt_len + asst_lengths[0] - 1,
        1 + env_len + asst_lengths[1] - 1,
        1 + env_len + asst_lengths[2] - 1,
    ]

    def _collect_with_mode(
        encode_mode, store_backend, traj_id, routed_turns, num_experts
    ):
      req = datatypes.RolloutRequest(
          prompt_id="p_bench",
          prompt="Solve multi-turn task",
          group_index=0,
          max_response_length=512,
          generation_kwargs={"return_routed_experts": True},
      )
      engine = collector.TrajectoryCollectorEngine(
          traj_id=traj_id,
          request=req,
          sampler=_MultiTurnRoutedSampler(asst_lengths, routed_turns),
          env_client=_MultiStepTextEnv(task={"q": "1"}, max_steps=3),
          agent=model_agent.ModelAgent("sys"),
          tokenizer=_MockTokenizer(),
          chat_parser=_RecordingParser(),
          trajectory_store=store_backend,
          store_routed_experts=True,
          routed_experts_encode_mode=encode_mode,
          routed_experts_num_experts=num_experts
          if encode_mode == "auto"
          else None,
      )
      return asyncio.run(engine.run_episode())

    for (
        model_label,
        num_layers,
        top_k,
        num_experts,
        expected_tier,
        expected_ratio,
    ) in (
        ("Qwen3-30B-A3B (E=128, K=8)", 48, 8, 128, "uint8", 0.50),
        ("DeepSeek-V2 (E=160, K=6)", 30, 6, 160, "uint8", 0.50),
        ("DeepSeek-V3 / Qwen3.5 (E=256, K=8)", 60, 8, 256, "uint10", 0.625),
        ("Large-MoE (E=512, K=8)", 60, 8, 512, "uint10", 0.625),
        ("Qwen3.5-397B-A17B (E=512, K=10)", 60, 10, 512, "uint10", 0.65),
    ):
      with self.subTest(model_label=model_label):
        routed_turns = [
            rng.integers(
                0, num_experts, size=(t_len, num_layers, top_k), dtype=np.int16
            )
            for t_len in step_lengths
        ]
        # Inject -1 (UNSET_ROUTED_EXPERT) at periodic slots for sentinel parity.
        routed_turns[0][0, ::4, :] = -1
        routed_turns[1][-1, ::3, -1] = -1

        tmp_root = pathlib.Path(self.create_tempdir().full_path)
        mem_i16 = in_memory_store.InMemoryTrajectoryStore()
        mem_cmp = in_memory_store.InMemoryTrajectoryStore()
        file_i16_dir = tmp_root / "file_i16"
        file_cmp_dir = tmp_root / "file_cmp"
        f_store_i16 = file_store.FileTrajectoryStore(root_dir=str(file_i16_dir))
        f_store_cmp = file_store.FileTrajectoryStore(root_dir=str(file_cmp_dir))
        sql_i16_db = tmp_root / "sql_i16.db"
        sql_cmp_db = tmp_root / "sql_cmp.db"
        s_store_i16 = sql_store.SqlTrajectoryStore(
            run_id="r1", db_url=f"sqlite:///{sql_i16_db}"
        )
        s_store_cmp = sql_store.SqlTrajectoryStore(
            run_id="r1", db_url=f"sqlite:///{sql_cmp_db}"
        )

        try:
          item_i16 = _collect_with_mode(
              "int16", mem_i16, "t_i16", routed_turns, num_experts
          )
          item_cmp = _collect_with_mode(
              "auto", mem_cmp, "t_cmp", routed_turns, num_experts
          )

          _collect_with_mode(
              "int16", f_store_i16, "t_f_i16", routed_turns, num_experts
          )
          f_store_i16.flush()

          _collect_with_mode(
              "auto", f_store_cmp, "t_f_cmp", routed_turns, num_experts
          )
          f_store_cmp.flush()

          _collect_with_mode(
              "int16", s_store_i16, "t_s_i16", routed_turns, num_experts
          )
          s_store_i16.flush()
          _collect_with_mode(
              "auto", s_store_cmp, "t_s_cmp", routed_turns, num_experts
          )
          s_store_cmp.flush()

          stored_f_i16 = f_store_i16.get_trajectories(["t_f_i16"])[0]
          stored_f_cmp = f_store_cmp.get_trajectories(["t_f_cmp"])[0]

          stored_s_i16 = s_store_i16.get_trajectories(["t_s_i16"])[0]
          stored_s_cmp = s_store_cmp.get_trajectories(["t_s_cmp"])[0]
        finally:
          f_store_i16.close()
          f_store_cmp.close()
          s_store_i16.close()
          s_store_cmp.close()

        # 1. Storage size measurements across InMemory, File (JSON), and SQLite.
        steps_i16 = [s for s in stored_f_i16.steps if s.source.value == "agent"]
        steps_cmp = [s for s in stored_f_cmp.steps if s.source.value == "agent"]
        self.assertLen(steps_cmp, 3)

        ram_bytes_i16 = sum(s.prefill_routed_experts.nbytes for s in steps_i16)
        ram_bytes_cmp = sum(s.prefill_routed_experts.nbytes for s in steps_cmp)
        self.assertAlmostEqual(
            ram_bytes_cmp / ram_bytes_i16, expected_ratio, places=4
        )

        file_bytes_i16 = sum(
            p.stat().st_size
            for p in (file_i16_dir / "traj_t_f_i16").glob("step*")
        )
        file_bytes_cmp = sum(
            p.stat().st_size
            for p in (file_cmp_dir / "traj_t_f_cmp").glob("step*")
        )
        self.assertGreater(file_bytes_cmp, 0)
        self.assertLess(file_bytes_cmp, file_bytes_i16)

        sql_bytes_i16 = len(
            stored_s_i16.model_dump_json(exclude_none=True).encode("utf-8")
        )
        sql_bytes_cmp = len(
            stored_s_cmp.model_dump_json(exclude_none=True).encode("utf-8")
        )
        self.assertLess(sql_bytes_cmp, sql_bytes_i16)
        sql_db_bytes_i16 = sql_i16_db.stat().st_size
        sql_db_bytes_cmp = sql_cmp_db.stat().st_size
        self.assertLess(sql_db_bytes_cmp, sql_db_bytes_i16)

        # 2. Bitwise parity on stored prefill_routed_experts (File & SQL).
        for idx, raw_turn in enumerate(routed_turns):
          self.assertEqual(
              packing.routed_experts_encoding(
                  steps_cmp[idx].prefill_routed_experts, batched=False
              ),
              expected_tier,
          )
          np.testing.assert_array_equal(
              packing.decode_routed_experts(
                  steps_cmp[idx].prefill_routed_experts, batched=False
              ),
              raw_turn,
          )
        steps_sql_cmp = [
            s for s in stored_s_cmp.steps if s.source.value == "agent"
        ]
        for idx, raw_turn in enumerate(routed_turns):
          np.testing.assert_array_equal(
              packing.decode_routed_experts(
                  steps_sql_cmp[idx].prefill_routed_experts, batched=False
              ),
              raw_turn,
          )

        # 3. Bitwise parity on converter.to_tunix_trajectory rehydration and
        # prefill-context vs decode-token stitching across multi-turn steps.
        rehydrated_i16 = converter_lib.to_tunix_trajectory(stored_f_i16)
        rehydrated_cmp = converter_lib.to_tunix_trajectory(stored_f_cmp)
        decoded_prompt = packing.decode_routed_experts(
            rehydrated_cmp.prompt_routed_experts, batched=False
        )
        np.testing.assert_array_equal(
            decoded_prompt, rehydrated_i16.prompt_routed_experts
        )
        np.testing.assert_array_equal(
            decoded_prompt, routed_turns[0][:prompt_len]
        )
        for turn_idx, (s_i16, s_cmp) in enumerate(
            zip(rehydrated_i16.steps, rehydrated_cmp.steps)
        ):
          decoded_asst = packing.decode_routed_experts(
              s_cmp.assistant_routed_experts, batched=False
          )
          np.testing.assert_array_equal(
              decoded_asst, s_i16.assistant_routed_experts
          )
          own_start = prompt_len if turn_idx == 0 else 1 + env_len
          if turn_idx < 2:
            # Non-final turn: decode prefix from turn_idx stitched with the
            # delayed final decode token prefilled in turn_idx + 1.
            expected_asst = np.concatenate(
                [
                    routed_turns[turn_idx][own_start:],
                    routed_turns[turn_idx + 1][:1],
                ],
                axis=0,
            )
          else:
            # Final turn: decode prefix from turn 2 (last token never
            # prefilled).
            expected_asst = routed_turns[turn_idx][own_start:]
          np.testing.assert_array_equal(decoded_asst, expected_asst)
          if s_i16.env_routed_experts is not None:
            decoded_env = packing.decode_routed_experts(
                s_cmp.env_routed_experts, batched=False
            )
            np.testing.assert_array_equal(decoded_env, s_i16.env_routed_experts)
            np.testing.assert_array_equal(
                decoded_env,
                routed_turns[turn_idx + 1][1 : 1 + env_len],
            )

        # 4. Bitwise parity on TrajectoryItem, BatchAssembly, JAX decode, and
        # direct bit-matching against original raw rollout router indices.
        raw_all_routed = np.concatenate(routed_turns, axis=0)
        decoded_item_routed = packing.decode_routed_experts(
            item_cmp.routed_experts, batched=False
        )
        np.testing.assert_array_equal(
            decoded_item_routed, item_i16.routed_experts
        )
        np.testing.assert_array_equal(decoded_item_routed[:-1], raw_all_routed)
        # Final unprefilled decode token of the trajectory is padded with -1.
        np.testing.assert_array_equal(decoded_item_routed[-1], -1)
        p_i16 = payload_builder.create_trainer_payloads(
            [item_i16, dataclasses.replace(item_i16, group_index=1)],
            rewards=[0.0, 1.0],
            algo_config=algorithm_config.GRPOConfig(num_generations=2),
        )
        p_cmp = payload_builder.create_trainer_payloads(
            [item_cmp, dataclasses.replace(item_cmp, group_index=1)],
            rewards=[0.0, 1.0],
            algo_config=algorithm_config.GRPOConfig(
                num_generations=2,
                routed_experts_encode_mode="auto",
                routed_experts_num_experts=num_experts,
            ),
        )
        pad_asm_i16 = batch_assembly.PaddedBatchAssembler(
            batch_size=2,
            num_generations=2,
            mini_batch_size=1,
            max_prompt_length=16,
            max_response_length=400,
            pad_id=0,
        )
        pad_asm_cmp = batch_assembly.PaddedBatchAssembler(
            batch_size=2,
            num_generations=2,
            mini_batch_size=1,
            max_prompt_length=16,
            max_response_length=400,
            pad_id=0,
        )
        pad_payload_i16 = list(pad_asm_i16.feed(p_i16))[0].payload
        pad_payload_cmp = list(pad_asm_cmp.feed(p_cmp))[0].payload
        batch_i16 = pad_payload_i16.routed_experts
        batch_cmp = pad_payload_cmp.routed_experts
        jax_decoded_i16 = np.asarray(
            rl_common.decode_routed_experts_jax(jnp.asarray(batch_i16))
        )
        jax_decoded_cmp = np.asarray(
            rl_common.decode_routed_experts_jax(jnp.asarray(batch_cmp))
        )
        np.testing.assert_array_equal(jax_decoded_cmp, jax_decoded_i16)
        left_pad = 16 - prompt_len
        np.testing.assert_array_equal(
            jax_decoded_cmp[0, left_pad : left_pad + len(raw_all_routed)],
            raw_all_routed,
        )
        self.assertEqual(jax_decoded_cmp.shape[-2:], (num_layers, top_k))

        seq_asm_i16 = batch_assembly.SequencePackedBatchAssembler(
            batch_size=1,
            num_generations=2,
            mini_batch_size=1,
            max_packed_len=800,
            pad_id=0,
            segment_alignment_boundary=1,
        )
        seq_asm_cmp = batch_assembly.SequencePackedBatchAssembler(
            batch_size=1,
            num_generations=2,
            mini_batch_size=1,
            max_packed_len=800,
            pad_id=0,
            segment_alignment_boundary=1,
        )
        seq_batch_i16 = list(seq_asm_i16.feed(p_i16))[0].payload.routed_experts
        seq_batch_cmp = list(seq_asm_cmp.feed(p_cmp))[0].payload.routed_experts
        jax_seq_i16 = np.asarray(
            rl_common.decode_routed_experts_jax(jnp.asarray(seq_batch_i16))
        )
        jax_seq_cmp = np.asarray(
            rl_common.decode_routed_experts_jax(jnp.asarray(seq_batch_cmp))
        )
        np.testing.assert_array_equal(jax_seq_cmp, jax_seq_i16)
        np.testing.assert_array_equal(
            jax_seq_cmp[0, : len(raw_all_routed)], raw_all_routed
        )

    # 5. End-to-end rollout -> store -> batch assembly -> trainer JAX decode
    # with dynamic cross-turn tier promotion (num_experts=None: uint8 -> uint10)
    # for both K % 4 == 0 (K=8) and K % 4 != 0 (K=10).
    for promo_k in (8, 10):
      with self.subTest(dynamic_promotion_top_k=promo_k):
        promo_turns = [
            rng.integers(
                0, 128, size=(step_lengths[0], 16, promo_k), dtype=np.int16
            ),
            rng.integers(
                256, 512, size=(step_lengths[1], 16, promo_k), dtype=np.int16
            ),
            rng.integers(
                0, 64, size=(step_lengths[2], 16, promo_k), dtype=np.int16
            ),
        ]
        raw_promo_all = np.concatenate(promo_turns, axis=0)
        promo_store_i16 = in_memory_store.InMemoryTrajectoryStore()
        promo_store_cmp = in_memory_store.InMemoryTrajectoryStore()
        promo_i16 = _collect_with_mode(
            "int16", promo_store_i16, f"p_i16_{promo_k}", promo_turns, None
        )
        promo_cmp = _collect_with_mode(
            "auto", promo_store_cmp, f"p_cmp_{promo_k}", promo_turns, None
        )
        promo_payloads_i16 = payload_builder.create_trainer_payloads(
            [promo_i16, dataclasses.replace(promo_i16, group_index=1)],
            rewards=[0.0, 1.0],
            algo_config=algorithm_config.GRPOConfig(num_generations=2),
        )
        promo_payloads_cmp = payload_builder.create_trainer_payloads(
            [promo_cmp, dataclasses.replace(promo_cmp, group_index=1)],
            rewards=[0.0, 1.0],
            algo_config=algorithm_config.GRPOConfig(
                num_generations=2,
                routed_experts_encode_mode="auto",
            ),
        )
        promo_seq_i16 = list(
            batch_assembly.SequencePackedBatchAssembler(
                batch_size=1,
                num_generations=2,
                mini_batch_size=1,
                max_packed_len=800,
                pad_id=0,
                segment_alignment_boundary=1,
            ).feed(promo_payloads_i16)
        )[0].payload.routed_experts
        promo_seq_cmp = list(
            batch_assembly.SequencePackedBatchAssembler(
                batch_size=1,
                num_generations=2,
                mini_batch_size=1,
                max_packed_len=800,
                pad_id=0,
                segment_alignment_boundary=1,
            ).feed(promo_payloads_cmp)
        )[0].payload.routed_experts
        jax_promo_i16 = np.asarray(
            rl_common.decode_routed_experts_jax(jnp.asarray(promo_seq_i16))
        )
        jax_promo_cmp = np.asarray(
            rl_common.decode_routed_experts_jax(jnp.asarray(promo_seq_cmp))
        )
        np.testing.assert_array_equal(jax_promo_cmp, jax_promo_i16)
        np.testing.assert_array_equal(
            jax_promo_cmp[0, : len(raw_promo_all)], raw_promo_all
        )


if __name__ == "__main__":
  absltest.main()
