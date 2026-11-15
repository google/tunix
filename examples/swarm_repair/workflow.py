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

"""Learned shared-policy collaboration through the collective runtime."""

import asyncio
import collections
import dataclasses
import json
import time

from examples.swarm_repair import environment
from tunix.dist_rl.swarm import collective
from tunix.dist_rl.swarm import runtime

MODES = ("single", "independent", "fixed", "swarm")


class NoCompletedActions(RuntimeError):
  """A whole GRPO group must be retried or rejected, never silently shortened."""


INSTRUCTIONS = """Repair the JSON module workspace using one JSON action per response.
Actions:
{"action":"spawn","role":"repair:module_b.json"}
{"action":"read","path":"module_a.json"}
{"action":"write","path":"module_a.json","module":{"op":"scale","value":2}}
{"action":"send","recipient":"agent-0","message":"module_b.json is repaired"}
{"action":"submit","continue":false}
Only spawn creates new agents. All agents share one policy. Read feedback and
mailbox messages; coordinate edits. Submit commits a snapshot for anytime grading;
continue=true permits further work, false ends this agent. No Python or shell
code is accepted. Budgets are shared across the team, including hidden-tool time.
"""


@dataclasses.dataclass(frozen=True)
class RepairResult:
  episode: object
  status: str
  events: tuple
  errors: tuple
  elapsed_seconds: float
  max_concurrent_calls: int
  cancelled_agents: int
  requested_calls: int
  generated_tokens: int
  mode: str


class RepairWorkflow:

  def __init__(self, config):
    self.config = config

  async def run(
      self,
      task,
      sampler,
      encode,
      *,
      policy_version,
      seed,
      episode_id,
      mode="swarm",
  ):
    if mode not in MODES:
      raise ValueError("Unknown collective baseline mode.")
    cfg = self.config
    contexts, action_counts, submitted = {}, {}, {}
    initialized = False
    token_reservations, observed_tokens = 0, 0
    actor_calls = 0
    inflight, maximum_inflight = 0, 0
    mode_agents = 1 if mode == "single" else cfg["max_agents"]

    async def generate(agent_id, role, prompt):
      nonlocal token_reservations, observed_tokens, actor_calls, inflight, maximum_inflight
      prompt_ids = encode(prompt)
      reserve = len(prompt_ids) + cfg["max_response_length"]
      if len(prompt_ids) > cfg["max_prompt_length"]:
        raise RuntimeError("Encoded repair context exceeds prompt budget.")
      if (
          token_reservations + observed_tokens + reserve
          > cfg["max_total_tokens"]
      ):
        raise RuntimeError("Collective token budget exhausted.")
      token_reservations += reserve
      if agent_id not in contexts:
        from tunix.dist_rl.rollout import sampler as sampler_lib  # pylint: disable=g-import-not-at-top

        contexts[agent_id] = runtime.RecordingRuntime(
            sampler,
            encode,
            sampler_lib.SamplingParams(
                max_tokens=cfg["max_response_length"],
                temperature=1.0,
                top_p=1.0,
                seed=seed + int(agent_id.split("-")[-1]) * 1000,
            ),
            policy_id="shared_repair",
            policy_version=policy_version,
            episode_id=f"{episode_id}/{agent_id}",
            max_prompt_length=cfg["max_prompt_length"],
        )
      try:
        actor_calls += 1
        inflight += 1
        maximum_inflight = max(maximum_inflight, inflight)
        call = await contexts[agent_id].generate(role, prompt)
        observed_tokens += len(call.prompt_ids) + len(call.completion_ids)
        return call
      finally:
        token_reservations -= reserve
        inflight -= 1

    def file_key(agent_id, name):
      return f"{agent_id}/{name}" if mode == "independent" else name

    async def run_agent(ctx, role):
      nonlocal initialized
      if not initialized or mode == "independent":
        for name in environment.FILES:
          ctx.put(
              file_key(ctx.agent_id, name),
              json.dumps({"op": "identity", "value": 0}),
          )
        initialized = True
      feedback, last_action, local_reads = "Ready.", "", {}
      for turn in range(cfg["max_turns_per_agent"]):
        observation = {
            "task": task.public(),
            "agent_id": ctx.agent_id,
            "role": role,
            "mode": mode,
            "turn": turn,
            "last_action": last_action,
            "feedback": feedback,
            "mailbox": ctx.receive(),
            "reads": local_reads,
            "agents_spawned": action_counts.get("spawn", 0),
            "remaining_calls": max(0, cfg["max_calls"] - actor_calls),
        }
        call = await ctx.generate(
            INSTRUCTIONS
            + "\nObservation:\n"
            + json.dumps(observation, sort_keys=True)
        )
        try:
          action = json.loads(call.text)
          if not isinstance(action, dict) or not isinstance(
              action.get("action"), str
          ):
            raise ValueError("Return one JSON action object.")
          name = action["action"]
          last_action = name
          if name == "spawn":
            if mode != "swarm":
              raise ValueError("Dynamic spawn is disabled in this baseline.")
            role_value = action.get("role")
            if role_value not in {f"repair:{p}" for p in environment.FILES}:
              raise ValueError("Spawn role must identify a public module file.")
            feedback = "Spawned " + ctx.spawn(role_value)
          elif name == "read":
            path = action.get("path")
            if path not in environment.FILES:
              raise ValueError("Only declared workspace files can be read.")
            value = ctx.get(file_key(ctx.agent_id, path))
            local_reads[path] = json.loads(value)
            feedback = f"Read {path}: {value}"
          elif name == "write":
            path = action.get("path")
            if path not in environment.FILES:
              raise ValueError("Only declared workspace files can be patched.")
            module = environment.parse_module(action.get("module"))
            ctx.put(
                file_key(ctx.agent_id, path), json.dumps(module, sort_keys=True)
            )
            local_reads[path] = module
            feedback = f"Updated {path}."
          elif name == "send":
            if mode in ("single", "independent"):
              raise ValueError("Communication is disabled in this baseline.")
            recipient, message = action.get("recipient"), action.get("message")
            if not isinstance(message, str):
              raise ValueError("Messages must be strings.")
            ctx.send(recipient, message)
            feedback = "Message sent."
          elif name == "submit":
            if type(action.get("continue", False)) is not bool:
              raise ValueError("continue must be boolean.")
            snapshot = json.dumps(
                {
                    p: json.loads(ctx.get(file_key(ctx.agent_id, p)))
                    for p in environment.FILES
                },
                sort_keys=True,
            )
            submitted[ctx.agent_id] = snapshot
            if mode == "independent":
              counts = collections.Counter(submitted.values())
              snapshot = min(
                  counts, key=lambda candidate: (-counts[candidate], candidate)
              )
            ctx.commit(snapshot)
            feedback = "Snapshot committed."
            if not action.get("continue", False):
              return
          else:
            raise ValueError("Unsupported action.")
          action_counts[name] = action_counts.get(name, 0) + 1
        except (ValueError, TypeError, KeyError) as exc:
          feedback = "Invalid action: " + str(exc)
        # This explicit yield lets spawned peers execute even for local samplers.
        await asyncio.sleep(0)

    runner = collective.CollectiveRuntime(
        generate,
        run_agent,
        policy_id="shared_repair",
        policy_version=policy_version,
        config=collective.CollectiveConfig(
            max_agents=mode_agents,
            max_calls=cfg["max_calls"],
            deadline_seconds=cfg["deadline_seconds"],
            max_messages=cfg["max_messages"],
            max_workspace_chars=cfg["max_workspace_chars"],
        ),
    )
    initial_roles = (
        ("lead", "repair:module_b.json")
        if mode == "fixed"
        else tuple("independent" for _ in range(mode_agents))
        if mode == "independent"
        else ("lead",)
    )
    start = time.monotonic()
    result = await runner.run(initial_roles)
    elapsed = time.monotonic() - start
    if not result.calls:
      raise NoCompletedActions(
          "No completed actions; reject the complete comparison group."
      )
    episode = result.to_episode(
        task_id=task.task_id, episode_id=episode_id, grade=task.grade
    )
    return RepairResult(
        episode,
        result.status,
        result.events,
        result.errors,
        elapsed,
        maximum_inflight,
        result.cancelled_agents,
        result.requested_calls,
        sum(len(c.completion_ids) for c in result.calls),
        mode,
    )
