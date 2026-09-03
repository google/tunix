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
