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

"""Environment-independent shared-policy collective adapter.

The modular-repair DSL is not required here. Supply any cancellable environment
and exact-token generation callback. Shell/code tools must execute in an external
sandbox supplied by that environment, never by this adapter. A group's episodes
and diagnostics feed `rlcr.build_rlcr_credit`; the existing SwarmRLProgram then
trains with the registered native CISPO objective.
"""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
import dataclasses
import json
import math
from typing import Protocol

from tunix.dist_rl.swarm import collective
from tunix.dist_rl.swarm import rlcr
from tunix.dist_rl.swarm import trace


@dataclasses.dataclass(frozen=True, kw_only=True)
class ToolResult:
  observation: str
  reward: float = 0.0
  terminal: bool = False
  answer: str | None = None

  def __post_init__(self):
    if not math.isfinite(self.reward):
      raise ValueError("Environment rewards must be finite.")


class Environment(Protocol):
  """One isolated task instance; observations must not expose grading labels.

  `step` commits its mutation/reward atomically before returning; it must honor
  cancellation. `settle` runs once after all peers stop, including at a deadline,
  and returns an additional terminal/anytime reward, not the cumulative reward
  already delivered by tool calls. No task label or grader is passed to agents.
  """

  tool_instructions: str

  def observe(self, agent_id: str, role: str) -> Mapping:
    ...

  async def step(
      self, agent_id: str, tool: str, arguments: Mapping
  ) -> ToolResult:
    ...

  async def settle(self, final_text: str) -> float:
    ...


@dataclasses.dataclass(frozen=True)
class EnvironmentRollout:
  episode: trace.SwarmEpisode
  diagnostics: rlcr.EpisodeDiagnostics
  runtime: collective.CollectiveResult
  reward_events: tuple[float, ...]


INSTRUCTIONS = """Work together on the environment task. Return one JSON action:
{"action":"spawn","role":"a specialist role and task"}
{"action":"send","recipient":"agent-0","message":"progress report"}
{"action":"read","key":"shared note"}
{"action":"write","key":"shared note","value":"text"}
{"action":"tool","tool":"environment tool name","arguments":{}}
{"action":"commit","answer":"current best answer"}
{"action":"finish","answer":"final answer"}
{"action":"exit"}
Only the CEO (agent-0) may spawn specialists or finish the organisation. A commit
is an anytime snapshot, not termination. Environment tools can also terminate the
whole organisation. Communicate through mailboxes and shared notes. No grading
labels are available. All agents use the same policy snapshot.
"""


async def run_environment(
    environment: Environment,
    generate: Callable[[str, str, str], Awaitable[trace.CallTrace]],
    *,
    task_id: str,
    episode_id: str,
    policy_id: str,
    policy_version: int,
    config: collective.CollectiveConfig = collective.CollectiveConfig(),
    max_turns_per_agent: int = 64,
) -> EnvironmentRollout:
  """Run a dynamic team, accumulate all reward events, and retain exact traces.

  `generate(agent_id, role, prompt)` typically routes to a RecordingRuntime per
  agent with unique namespaces. It must enforce model-context/token budgets.
  This adapter supplies calls/agents/messages/workspace/wall-clock bounds.
  """
  if type(max_turns_per_agent) is not int or max_turns_per_agent < 1:
    raise ValueError("The per-agent turn budget must be a positive integer.")
  reward_events, diagnostic_calls = [], {}

  async def agent(ctx, role):
    feedback = "Ready."
    for _ in range(max_turns_per_agent):
      observation = {
          "agent_id": ctx.agent_id,
          "role": role,
          "environment": environment.observe(ctx.agent_id, role),
          "messages": ctx.receive(),
          "feedback": feedback,
      }
      call = await ctx.generate(
          INSTRUCTIONS
          + environment.tool_instructions
          + "\nObservation:\n"
          + json.dumps(observation, sort_keys=True)
      )
      diagnostic_calls[call.call_id] = rlcr.CallDiagnostics(
          agent_id=ctx.agent_id
      )
      try:
        action = json.loads(call.text)
        if not isinstance(action, dict) or not isinstance(
            action.get("action"), str
        ):
          raise ValueError("Expected a JSON action object.")
      except (ValueError, TypeError) as exc:
        diagnostic_calls[call.call_id] = rlcr.CallDiagnostics(
            agent_id=ctx.agent_id, malformed_tool=True
        )
        feedback = f"Malformed action: {exc}"
        continue
      try:
        name = action["action"]
        if name == "spawn":
          if ctx.agent_id != "agent-0" or not isinstance(
              action.get("role"), str
          ):
            raise ValueError("Only the CEO may spawn a named role.")
          feedback = "Spawned " + ctx.spawn(action["role"])
        elif name == "send":
          if not all(
              isinstance(action.get(k), str) for k in ("recipient", "message")
          ):
            raise ValueError("Recipient and message must be strings.")
          ctx.send(action["recipient"], action["message"])
          feedback = "Delivered at recipient's next turn boundary."
        elif name in ("read", "write"):
          if not isinstance(action.get("key"), str):
            raise ValueError("Workspace keys must be strings.")
          if name == "write":
            if not isinstance(action.get("value"), str):
              raise ValueError("Workspace values must be strings.")
            ctx.put(action["key"], action["value"])
          feedback = ctx.get(action["key"]) or "No value."
        elif name == "tool":
          if not isinstance(action.get("tool"), str) or not isinstance(
              action.get("arguments"), dict
          ):
            raise ValueError(
                "Environment calls require a tool name and arguments object."
            )
          result = await environment.step(
              ctx.agent_id, action["tool"], action["arguments"]
          )
          ctx.check_active()
          reward_events.append(result.reward)
          feedback = result.observation
          if result.terminal:
            ctx.finish(result.answer)
            return
        elif name in ("commit", "finish"):
          if not isinstance(action.get("answer"), str):
            raise ValueError("Answers must be strings.")
          if name == "finish":
            if ctx.agent_id != "agent-0":
              raise ValueError("Only the CEO may finish the organisation.")
            ctx.finish(action["answer"])
            return
          ctx.commit(action["answer"])
          feedback = "Anytime snapshot committed."
        elif name == "exit":
          return
        else:
          raise ValueError("Unsupported action.")
      except TimeoutError:
        raise
      except (
          Exception
      ) as exc:  # A tool failure is feedback; cancellation propagates.
        diagnostic_calls[call.call_id] = rlcr.CallDiagnostics(
            agent_id=ctx.agent_id, execution_error=True
        )
        feedback = f"Tool error: {exc}"
      await asyncio.sleep(0)

  result = await collective.CollectiveRuntime(
      generate,
      agent,
      policy_id=policy_id,
      policy_version=policy_version,
      config=config,
  ).run(("CEO",))
  reward_events.append(float(await environment.settle(result.final_text)))
  if not all(math.isfinite(r) for r in reward_events):
    raise ValueError("The environment returned a nonfinite settlement reward.")
  # Completed reward-bearing tools may terminate without a textual answer.
  if not result.calls:
    raise ValueError(
        "No completed action tokens; reject or retry the whole group."
    )
  episode = trace.SwarmEpisode(
      task_id=task_id,
      episode_id=episode_id,
      calls=result.calls,
      reward=math.fsum(reward_events),
      final_text=result.final_text,
  )
  return EnvironmentRollout(
      episode,
      rlcr.EpisodeDiagnostics(
          calls=diagnostic_calls,
          specialist_budget=config.max_agents - 1,
          specialists_hired=sum(e.kind == "spawn" for e in result.events) - 1,
      ),
      result,
      tuple(reward_events),
  )
