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
import types
import unittest
from unittest import mock

from absl.testing import absltest
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.common import test_utils as mocks
from tunix.experimental.rollout import collector
from tunix.experimental.rollout import sampler as sampler_lib
from tunix.experimental.rollout import vanilla_sampler_adapter
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
    return sampler_lib.SamplingResponse(
        request_id=getattr(req, "request_id", ""),
        text=f"action_{self._call_count}",
        token_ids=tokens,
        prompt_token_ids=np.array([1, 2], dtype=np.int32),
        routed_experts=self.routed_experts,
    )


class _MockVanillaSampler(vanilla_sampler_adapter.VanillaSamplerAdapter):

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


class VanillaRolloutSeedTest(absltest.TestCase):

  def test_generate_vanilla_rollout_seed_reproducible(self):
    seed1 = collector.generate_vanilla_rollout_seed("prompt_42", group_index=0)
    seed2 = collector.generate_vanilla_rollout_seed("prompt_42", group_index=0)
    self.assertEqual(seed1, seed2)
    self.assertIsInstance(seed1, int)
    self.assertGreaterEqual(seed1, 0)
    self.assertLessEqual(seed1, 0x7FFFFFFF)

  def test_generate_vanilla_rollout_seed_intra_group_diversity(self):
    seeds = [
        collector.generate_vanilla_rollout_seed("gsm8k_q1", group_index=i)
        for i in range(4)
    ]
    self.assertEqual(len(seeds), len(set(seeds)))
    for s in seeds:
      self.assertGreaterEqual(s, 0)
      self.assertLessEqual(s, 0x7FFFFFFF)

  def test_generate_vanilla_rollout_seed_arbitrary_prompt_formats(self):
    s1 = collector.generate_vanilla_rollout_seed(42, group_index=0)
    s2 = collector.generate_vanilla_rollout_seed("math-q1", group_index=0)
    s3 = collector.generate_vanilla_rollout_seed("uuid-123-abc", group_index=0)
    self.assertIsInstance(s1, int)
    self.assertIsInstance(s2, int)
    self.assertIsInstance(s3, int)
    self.assertNotEqual(s2, s3)


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
      inner_engine_cls.return_value.collect = mock.AsyncMock(
          return_value={}
      )
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
        engine1.episode_timeout, collector._DEFAULT_EPISODE_TIMEOUT_SECS
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
        engine2.episode_timeout, collector._DEFAULT_EPISODE_TIMEOUT_SECS
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

  def test_model_call_seeds_vanilla_sampler_deterministically(self):
    async def _run():
      sampler = _MockVanillaSampler()
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
      expected_seed = collector.generate_vanilla_rollout_seed(
          "prompt_42", group_index=1
      )
      self.assertEqual(sampling_req.sampling_params.seed, expected_seed)

    asyncio.run(_run())

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

  def test_model_call_grpo_group_seeds_distinct_and_reproducible(self):
    async def _run():
      sampler = _MockVanillaSampler()
      req_0 = datatypes.RolloutRequest(
          prompt_id="prompt_42",
          prompt="test prompt",
          group_index=0,
          generation_kwargs={"max_generation_steps": 128, "temperature": 0.7},
      )
      req_1 = datatypes.RolloutRequest(
          prompt_id="prompt_42",
          prompt="test prompt",
          group_index=1,
          generation_kwargs={"max_generation_steps": 128, "temperature": 0.7},
      )
      req_0_repeat = datatypes.RolloutRequest(
          prompt_id="prompt_42",
          prompt="test prompt",
          group_index=0,
          generation_kwargs={"max_generation_steps": 128, "temperature": 0.7},
      )

      mock_agent = mock.MagicMock()
      mock_agent.name = "test_agent"

      outputs = []
      for req in [req_0, req_1, req_0_repeat]:
        engine = collector.TrajectoryCollectorEngine(
            traj_id=f"traj_{req.prompt_id}_g{req.group_index}",
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
          output = await model_call("prompt text")
          outputs.append(output)

      self.assertLen(sampler.calls, 3)
      req_0_call, _ = sampler.calls[0]
      req_1_call, _ = sampler.calls[1]
      req_0_repeat_call, _ = sampler.calls[2]

      self.assertNotEqual(
          req_0_call.sampling_params.seed, req_1_call.sampling_params.seed
      )
      self.assertEqual(
          req_0_call.sampling_params.seed,
          req_0_repeat_call.sampling_params.seed,
      )
      self.assertFalse(
          np.array_equal(outputs[0].tokens[0], outputs[1].tokens[0])
      )
      self.assertTrue(
          np.array_equal(outputs[0].tokens[0], outputs[2].tokens[0])
      )

    asyncio.run(_run())

  def test_model_call_raises_when_prompt_id_is_none_for_vanilla_sampler(self):
    async def _run():
      sampler = _MockVanillaSampler()
      req = datatypes.RolloutRequest(
          prompt_id=None,
          prompt="test prompt",
          group_index=0,
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
        with self.assertRaisesRegex(
            ValueError, "Vanilla sampler requires a seed or valid prompt_id"
        ):
          await model_call("prompt text")

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
        "policy_version": 0,
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
        "policy_version": 5,
    }

    item = engine._convert_to_trajectory(rl_traj)
    self.assertIsInstance(item, datatypes.TrajectoryItem)
    self.assertEqual(item.prompt_id, "prompt_test")
    self.assertEqual(item.group_index, 2)
    self.assertEqual(item.policy_version, 5)
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
        "conversation_masks": np.array([1.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "old_logprobs": np.array([-0.1, -0.2, 0.0, 0.0, -0.3], dtype=np.float32),
        "trajectory_reward": 1.0,
        "status": "COMPLETED",
        "policy_version": 1,
    }

    item = engine._convert_to_trajectory(rl_traj)
    np.testing.assert_array_equal(item.conversation_tokens, [10, 11, 20, 21, 12])
    np.testing.assert_array_equal(item.conversation_masks, [1.0, 1.0, 0.0, 0.0, 1.0])
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


  def test_model_call_respects_min_of_remaining_budget_and_request_max_tokens(self):
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
        collector.rl_collect_engine, "TrajectoryCollectEngine", side_effect=_capture_engine
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
        "policy_version": 1,
    }
    item = engine._convert_to_trajectory(rl_traj)
    self.assertNotIn("routed_experts", item.metadata)
    np.testing.assert_array_equal(item.traj["routed_experts"], mock_routed)
    np.testing.assert_array_equal(item.routed_experts, mock_routed)


if __name__ == "__main__":
  absltest.main()
