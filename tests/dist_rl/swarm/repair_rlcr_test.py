# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Source-described collective penalties, token masks and environment binding."""

import asyncio
import dataclasses
import json

import numpy as np
import pytest
from examples.swarm_repair import custom_collective
from tunix.dist_rl.swarm import collective
from tunix.dist_rl.swarm import rlcr
from tunix.dist_rl.swarm import trace


def episode(index, reward, calls=2):
  return trace.SwarmEpisode(
      task_id="t",
      episode_id=f"e{index}",
      reward=reward,
      final_text="answer",
      calls=tuple(
          trace.CallTrace(
              call_id=f"e{index}/{c}",
              role=f"role{c}",
              policy_id="shared",
              policy_version=0,
              prompt_ids=(99, 100),
              completion_ids=(1, 2, 3),
              rollout_logprobs=(-1.0, -1.0, -1.0),
              loss_mask=(1, 0, 1),
              text="action",
          )
          for c in range(calls)
      ),
  )


def test_penalties_sum_active_error_tokens_and_underhires_before_team_credit():
  episodes = (episode(0, 1), episode(1, 0))
  diagnostics = rlcr.EpisodeDiagnostics(
      calls={
          "e0/0": rlcr.CallDiagnostics(agent_id="CEO", malformed_tool=True),
          "e0/1": rlcr.CallDiagnostics(agent_id="peer", execution_error=True),
      },
      specialist_budget=3,
      specialists_hired=1,
  )
  cfg = rlcr.RLCRConfig(
      parse_error_penalty=0.1,
      execution_error_penalty=0.05,
      underhire_penalty=0.2,
  )
  assert rlcr.collective_return(episodes[0], diagnostics, cfg) == pytest.approx(
      0.3
  )
  batch = rlcr.build_rlcr_credit(
      episodes, diagnostics={"e0": diagnostics}, config=cfg
  )
  expected = 0.15 / (np.sqrt(0.045) + 1e-6)
  np.testing.assert_allclose(
      batch.advantages, [expected, expected, -expected, -expected], rtol=1e-6
  )
  for payload in batch.payloads:
    np.testing.assert_array_equal(payload.completion_mask, [[1, 0, 1]])
    np.testing.assert_array_equal(
        payload.completion_attention_mask, [[1, 1, 1]]
    )


def test_eligibility_preserves_comparison_moments_and_attention_context():
  episodes = (episode(0, -1), episode(1, 0), episode(2, 2))
  cfg = rlcr.RLCRConfig(
      mask_zero_environment_return=True, mask_misplaced_tool_calls=True
  )
  diagnostic = rlcr.EpisodeDiagnostics(
      calls={"e2/0": rlcr.CallDiagnostics(agent_id="CEO", misplaced_tool=True)}
  )
  raw = rlcr.build_rlcr_credit(episodes)
  masked = rlcr.build_rlcr_credit(
      episodes,
      diagnostics={"e2": diagnostic},
      config=cfg,
      token_eligibility=lambda c: (0, 1, 1),
  )
  assert [p.metadata["call_id"] for p in masked.payloads] == [
      "e0/0",
      "e0/1",
      "e2/1",
  ]
  assert masked.advantages == (
      raw.advantages[0],
      raw.advantages[1],
      raw.advantages[-1],
  )
  np.testing.assert_array_equal(
      masked.payloads[-1].completion_mask, [[0, 0, 1]]
  )
  np.testing.assert_array_equal(
      masked.payloads[-1].completion_attention_mask, [[1, 1, 1]]
  )
  with pytest.raises(ValueError, match="aligned binary"):
    rlcr.build_rlcr_credit(episodes, token_eligibility=lambda c: (1,))


def test_repetition_masks_all_calls_of_only_the_affected_agent():
  assert rlcr.has_tandem_repeat([1] * 300)
  assert not rlcr.has_tandem_repeat([1] * 299)
  assert rlcr.has_tandem_repeat(list(range(15)) * 20)
  assert not rlcr.has_tandem_repeat(list(range(32)) * 19)
  first, second = episode(0, 1, 3), episode(1, 0)
  long_call = dataclasses.replace(
      first.calls[0],
      completion_ids=(1,) * 300,
      rollout_logprobs=(-1.0,) * 300,
      loss_mask=(1,) * 300,
  )
  first = dataclasses.replace(first, calls=(long_call, *first.calls[1:]))
  diagnostics = {
      e.episode_id: rlcr.EpisodeDiagnostics(
          calls={
              c.call_id: rlcr.CallDiagnostics(
                  agent_id="peer" if c.call_id.endswith("/2") else "CEO"
              )
              for c in e.calls
          }
      )
      for e in (first, second)
  }
  batch = rlcr.build_rlcr_credit(
      (first, second),
      diagnostics=diagnostics,
      config=rlcr.RLCRConfig(mask_repetitions=True),
  )
  assert [p.metadata["call_id"] for p in batch.payloads] == [
      "e0/2",
      "e1/0",
      "e1/1",
  ]
  assert not rlcr.build_rlcr_credit(
      (episode(0, 1), episode(1, 1 + 1e-10))
  ).payloads


def test_external_environment_rewards_and_global_finish_cancel_peer():
  async def scenario():
    ready, cancelled = asyncio.Event(), asyncio.Event()
    observed, counters = [], {}
    actions = {
        "agent-0": [
            {"action": "spawn", "role": "arbitrary external specialist"},
            {"action": "tool", "tool": "coordinate", "arguments": {}},
            {"action": "finish", "answer": "settled"},
        ],
        "agent-1": [{"action": "tool", "tool": "contribute", "arguments": {}}],
    }

    class Environment:
      tool_instructions = "External tools: coordinate, contribute."
      settled = 0

      def observe(self, agent_id, role):
        return {"public": "cooperate on an external task"}

      async def step(self, agent_id, tool, arguments):
        if tool == "coordinate":
          await ready.wait()
          return custom_collective.ToolResult(observation="ready", reward=1)
        ready.set()
        return custom_collective.ToolResult(
            observation="contribution", reward=2
        )

      async def settle(self, answer):
        self.settled += 1
        assert answer == "settled"
        return 4

    async def generate(agent_id, role, prompt):
      observed.append(prompt)
      index = counters.get(agent_id, 0)
      counters[agent_id] = index + 1
      if index >= len(actions[agent_id]):
        try:
          await asyncio.Event().wait()
        finally:
          cancelled.set()
      return dataclasses.replace(
          episode(0, 0).calls[0],
          call_id=f"{agent_id}/{index}",
          role=role,
          text=json.dumps(actions[agent_id][index]),
      )

    env = Environment()
    rollout = await custom_collective.run_environment(
        env,
        generate,
        task_id="external",
        episode_id="e",
        policy_id="shared",
        policy_version=0,
        config=collective.CollectiveConfig(max_agents=2, deadline_seconds=1),
    )
    assert rollout.runtime.status == "finished" and cancelled.is_set()
    assert rollout.episode.reward == 7 and env.settled == 1
    assert rollout.diagnostics.specialists_hired == 1
    assert len(rollout.episode.calls) == 4
    assert all("hidden_inputs" not in p for p in observed)

  asyncio.run(scenario())
