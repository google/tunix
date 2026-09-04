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

  def __init__(self, token_lengths=None):
    self.sampled_params = []
    self.token_lengths = token_lengths or [30, 20]
    self._call_count = 0

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
        generation_kwargs={"max_response_length": 512},
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
          generation_kwargs={"max_response_length": 50},
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
        mock_instance.collect.return_value = mock.MagicMock(steps=[])
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
        mock_instance.collect.return_value = mock.MagicMock(steps=[])
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
        mock_instance.collect.return_value = mock.MagicMock(steps=[])
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
          mock_instance.collect.return_value = mock.MagicMock(steps=[])
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
        mock_instance.collect.return_value = mock.MagicMock(steps=[])
        mock_engine_cls.return_value = mock_instance

        await engine.run_episode()

        model_call = mock_engine_cls.call_args.kwargs["model_call"]
        with self.assertRaisesRegex(
            ValueError, "Vanilla sampler requires a seed or valid prompt_id"
        ):
          await model_call("prompt text")

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
    return types.SimpleNamespace(
        steps=[],
        prompt_tokens=np.asarray([1, 2, 3], dtype=np.int32),
        reward=0.0,
    )


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


if __name__ == "__main__":
  absltest.main()
