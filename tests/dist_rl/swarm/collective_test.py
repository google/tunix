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

"""Real asyncio concurrency, deadline cancellation, and flat team credit."""

import asyncio
import dataclasses

import numpy as np
import pytest
from tunix.dist_rl.swarm import collective
from tunix.dist_rl.swarm import trace


def call(call_id, role="solver", *, version=0):
  return trace.CallTrace(
      call_id=call_id,
      role=role,
      policy_id="shared",
      policy_version=version,
      prompt_ids=(1,),
      completion_ids=(2, 3),
      rollout_logprobs=(-0.5, -0.5),
      loss_mask=(1, 1),
      text="42",
  )


def test_dynamic_agents_run_concurrently_and_exchange_state():
  async def scenario():
    both_started = asyncio.Event()
    starts = []

    async def generate(agent_id, role, prompt):
      starts.append(agent_id)
      if len(starts) == 2:
        both_started.set()
      await both_started.wait()  # A serial executor cannot pass this barrier.
      return call(agent_id, role)

    async def agent(context, role):
      if role == "lead":
        child = context.spawn("solver")
        context.put("problem", "6 * 7")
        context.send(child, "solve")
      else:
        assert context.get("problem") == "6 * 7"
        assert context.receive() == (("agent-0", "solve"),)
      response = await context.generate("compute")
      context.commit(response.text)

    runtime = collective.CollectiveRuntime(
        generate, agent, policy_id="shared", policy_version=0
    )
    result = await runtime.run()
    assert result.status == "completed"
    assert result.final_text == "42"
    assert len(result.calls) == 2 and result.requested_calls == 2
    assert len([e for e in result.events if e.kind == "spawn"]) == 2
    with pytest.raises(RuntimeError, match="single-use"):
      await runtime.run()
    episode = result.to_episode(
        task_id="math", episode_id="attempt0", grade=lambda s: float(s == "42")
    )
    assert episode.reward == 1

  asyncio.run(scenario())


def test_deadline_keeps_committed_answer_and_cancels_pending_sampler():
  async def scenario():
    cancelled = asyncio.Event()

    async def generate(agent_id, role, prompt):
      if role == "slow":
        try:
          await asyncio.Event().wait()
        finally:
          cancelled.set()
      return call(agent_id, role)

    async def agent(context, role):
      if role == "lead":
        response = await context.generate("compute")
        context.commit(response.text)
        context.spawn("slow")
      else:
        await context.generate("long running")

    runtime = collective.CollectiveRuntime(
        generate,
        agent,
        policy_id="shared",
        policy_version=0,
        config=collective.CollectiveConfig(deadline_seconds=0.02),
    )
    result = await runtime.run()
    assert result.status == "deadline"
    assert result.final_text == "42"
    assert cancelled.is_set()
    assert result.cancelled_agents == 1 and result.requested_calls == 2
    assert (
        len(result.calls) == 1
    )  # Incomplete completion tokens were not invented.
    assert any(e.kind == "deadline" for e in result.events)

  asyncio.run(scenario())


def test_no_token_deadline_is_explicitly_untrainable():
  async def scenario():
    async def generate(agent_id, role, prompt):
      await asyncio.Event().wait()

    async def agent(context, role):
      await context.generate("wait")

    result = await collective.CollectiveRuntime(
        generate,
        agent,
        policy_id="shared",
        policy_version=0,
        config=collective.CollectiveConfig(deadline_seconds=0.01),
    ).run()
    assert result.status == "deadline"
    with pytest.raises(ValueError, match="whole group"):
      result.to_episode(task_id="t", episode_id="e", grade=lambda _: 1)

  asyncio.run(scenario())


@pytest.mark.parametrize("budget", ["agents", "calls", "messages", "workspace"])
def test_resource_budgets_fail_visibly(budget):
  async def scenario():
    async def generate(agent_id, role, prompt):
      return call(agent_id, role)

    async def agent(context, role):
      if budget == "agents":
        context.spawn("extra")
      elif budget == "calls":
        await context.generate("one")
        await context.generate("two")
      elif budget == "messages":
        context.send(context.agent_id, "one")
        context.send(context.agent_id, "two")
      else:
        context.put("too big", "x" * 100)

    result = await collective.CollectiveRuntime(
        generate,
        agent,
        policy_id="shared",
        policy_version=0,
        config=collective.CollectiveConfig(
            max_agents=1, max_calls=1, max_messages=1, max_workspace_chars=20
        ),
    ).run()
    assert result.status == "failed" and result.errors
    assert "budget" in result.errors[0]
    assert result.requested_calls <= 1

  asyncio.run(scenario())


def test_policy_changes_fail_admission():
  async def scenario():
    async def generate(agent_id, role, prompt):
      return call(agent_id, role, version=1)

    async def agent(context, role):
      await context.generate("compute")

    result = await collective.CollectiveRuntime(
        generate, agent, policy_id="shared", policy_version=0
    ).run()
    assert result.status == "failed"
    assert not result.calls
    assert "policy snapshots" in result.errors[0]

  asyncio.run(scenario())


def test_team_outcome_advantage_is_identical_for_all_roles_and_calls():
  episodes = [
      trace.SwarmEpisode(
          task_id="t",
          episode_id=f"e{i}",
          reward=float(i),
          final_text="42",
          calls=tuple(
              dataclasses.replace(call(f"{i}/{j}", role), valid_action=False)
              for j, role in enumerate(roles)
          ),
      )
      for i, roles in enumerate((("solver",), ("solver", "verifier", "solver")))
  ]
  batch = collective.build_team_credit(episodes)
  expected = 0.5 / (np.sqrt(0.5) + 1e-6)
  np.testing.assert_allclose(
      batch.advantages, [-expected, expected, expected, expected], atol=1e-7
  )
  assert batch.rewards == (0, 1, 1, 1)
