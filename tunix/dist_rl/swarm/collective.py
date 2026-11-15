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

"""Bounded local collective execution inspired by Avalon/RLCR.

This is a runtime kernel, not a reproduction of Avalon-M1 environments or its
CISPO objective. Agents share one policy snapshot, mailboxes and a workspace.
An anytime answer can be committed before a wall-clock deadline. Incomplete
sampling calls must support asyncio cancellation; their partial tokens are not
silently treated as complete on-policy trajectories.
"""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
import dataclasses
import math

from tunix.dist_rl.swarm import credit
from tunix.dist_rl.swarm import trace


@dataclasses.dataclass(frozen=True, kw_only=True)
class CollectiveConfig:
  max_agents: int = 4
  max_calls: int = 16
  deadline_seconds: float = 30.0
  max_messages: int = 64
  max_workspace_chars: int = 65536

  def __post_init__(self):
    if (
        min(
            self.max_agents,
            self.max_calls,
            self.max_messages,
            self.max_workspace_chars,
        )
        < 1
    ):
      raise ValueError("Collective resource budgets must be positive.")
    if not math.isfinite(self.deadline_seconds) or self.deadline_seconds <= 0:
      raise ValueError("A finite positive deadline is required.")


@dataclasses.dataclass(frozen=True)
class CollectiveEvent:
  kind: str
  agent_id: str
  elapsed_seconds: float
  detail: str = ""


@dataclasses.dataclass(frozen=True)
class CollectiveResult:
  calls: tuple[trace.CallTrace, ...]
  final_text: str
  status: str
  events: tuple[CollectiveEvent, ...]
  errors: tuple[str, ...]
  requested_calls: int
  cancelled_agents: int

  def to_episode(
      self,
      *,
      task_id: str,
      episode_id: str,
      grade: Callable[[str], float],
  ) -> trace.SwarmEpisode:
    """Admit completed action tokens, including failed attempts with reward 0.

    A no-token timeout cannot be trained. The collector must explicitly record
    the failure and retry or reject its whole comparison group; it must not
    silently shrink that group. Ground-truth grading happens only after rollout.
    """
    if not self.calls:
      raise ValueError(
          "No completed action tokens; explicitly reject or retry the whole"
          " group."
      )
    return trace.SwarmEpisode(
        task_id=task_id,
        episode_id=episode_id,
        calls=self.calls,
        reward=float(grade(self.final_text)) if self.final_text else 0.0,
        final_text=self.final_text,
    )


class AgentContext:
  """The capabilities available to one agent; no answer labels are exposed."""

  def __init__(self, runtime: "CollectiveRuntime", agent_id: str):
    self._runtime = runtime
    self.agent_id = agent_id

  async def generate(self, prompt: str) -> trace.CallTrace:
    return await self._runtime._generate(self.agent_id, prompt)

  def check_active(self) -> None:
    """Validate a tool completion before admitting its reward or next action."""
    self._runtime._check_active()

  def spawn(self, role: str) -> str:
    return self._runtime._spawn(role, self.agent_id)

  def send(self, recipient: str, message: str) -> None:
    runtime = self._runtime
    runtime._check_active()
    if recipient not in runtime._mailboxes:
      raise ValueError("Unknown message recipient.")
    if runtime._message_count >= runtime.config.max_messages:
      raise RuntimeError("Collective message budget exhausted.")
    if not message or len(message) > runtime.config.max_workspace_chars:
      raise ValueError("Message is empty or exceeds the text budget.")
    runtime._message_count += 1
    runtime._mailboxes[recipient].append((self.agent_id, message))
    runtime._record("message", self.agent_id, recipient)

  def receive(self) -> tuple[tuple[str, str], ...]:
    self._runtime._check_active()
    mailbox = self._runtime._mailboxes[self.agent_id]
    result = tuple(mailbox)
    mailbox.clear()
    return result

  def put(self, key: str, value: str) -> None:
    runtime = self._runtime
    runtime._check_active()
    if not key:
      raise ValueError("Workspace keys must be nonempty.")
    updated = {**runtime._workspace, key: value}
    if (
        sum(len(k) + len(v) for k, v in updated.items())
        > runtime.config.max_workspace_chars
    ):
      raise RuntimeError("Collective workspace budget exhausted.")
    runtime._workspace[key] = value
    runtime._record("workspace", self.agent_id, key)

  def get(self, key: str) -> str | None:
    self._runtime._check_active()
    return self._runtime._workspace.get(key)

  def commit(self, answer: str) -> None:
    runtime = self._runtime
    runtime._check_active()
    if not answer or len(answer) > runtime.config.max_workspace_chars:
      raise ValueError("Committed answers must be nonempty and bounded.")
    runtime._answer = answer
    runtime._record("commit", self.agent_id)

  def finish(self, answer: str | None = None) -> None:
    """End the whole organisation, optionally committing its final answer.

    Use this for a CEO final answer or an environment termination event.
    `commit` alone preserves an anytime snapshot while other agents continue.
    The runtime cancels pending samplers before returning the terminal result.
    """
    self._runtime._check_active()
    if answer is not None:
      self.commit(answer)
    self._runtime._record("finish", self.agent_id)
    self._runtime._finished.set()


class CollectiveRuntime:
  """Single-use event loop with genuinely concurrent, bounded agent tasks.

  `generate(agent_id, role, prompt)` must return exact sampled-token evidence
  from the shared policy and implement cancellation. Logical agent behavior is
  supplied by the recipe; this runtime does not prescribe a central Director.
  """

  def __init__(
      self,
      generate: Callable[[str, str, str], Awaitable[trace.CallTrace]],
      run_agent: Callable[[AgentContext, str], Awaitable[None]],
      *,
      policy_id: str,
      policy_version: int,
      config: CollectiveConfig = CollectiveConfig(),
  ):
    if not policy_id or policy_version < 0:
      raise ValueError("Collective execution needs one pinned policy snapshot.")
    self.generate = generate
    self.run_agent = run_agent
    self.policy_id = policy_id
    self.policy_version = policy_version
    self.config = config
    self._tasks = {}
    self._roles = {}
    self._mailboxes = {}
    self._workspace = {}
    self._events = []
    self._calls = []
    self._errors = []
    self._answer = ""
    self._requested_calls = 0
    self._message_count = 0
    self._running = False
    self._used = False
    self._finished = asyncio.Event()

  def _check_active(self):
    if not self._running or asyncio.get_running_loop().time() >= self._deadline:
      raise TimeoutError("The collective execution deadline has passed.")
    if self._finished.is_set():
      raise RuntimeError("The collective has already terminated.")

  def _record(self, kind, agent_id, detail=""):
    self._events.append(
        CollectiveEvent(
            kind,
            agent_id,
            asyncio.get_running_loop().time() - self._start,
            detail,
        )
    )

  def _spawn(self, role, parent=""):
    self._check_active()
    if not role:
      raise ValueError("Agents require a role.")
    if len(self._tasks) >= self.config.max_agents:
      raise RuntimeError("Collective agent budget exhausted.")
    agent_id = f"agent-{len(self._tasks)}"
    self._roles[agent_id] = role
    self._mailboxes[agent_id] = []
    self._tasks[agent_id] = asyncio.create_task(
        self.run_agent(AgentContext(self, agent_id), role)
    )
    self._record("spawn", agent_id, parent)
    return agent_id

  async def _generate(self, agent_id, prompt):
    self._check_active()
    if self._requested_calls >= self.config.max_calls:
      raise RuntimeError("Collective sampling budget exhausted.")
    self._requested_calls += 1
    self._record("sample_start", agent_id)
    call = await self.generate(agent_id, self._roles[agent_id], prompt)
    self._check_active()
    if (
        call.policy_id != self.policy_id
        or call.policy_version != self.policy_version
    ):
      raise ValueError("A collective cannot mix policy snapshots.")
    if any(c.call_id == call.call_id for c in self._calls):
      raise ValueError("A collective cannot reuse call IDs.")
    self._calls.append(call)
    self._record("sample_end", agent_id, call.call_id)
    return call

  async def run(
      self, initial_roles: Sequence[str] = ("lead",)
  ) -> CollectiveResult:
    if self._used:
      raise RuntimeError("Collective runtimes are single-use.")
    if not initial_roles or len(initial_roles) > self.config.max_agents:
      raise ValueError("Initial team exceeds the agent budget or is empty.")
    self._used = self._running = True
    self._start = asyncio.get_running_loop().time()
    self._deadline = self._start + self.config.deadline_seconds
    status, cancelled = "completed", 0
    finish_waiter = asyncio.create_task(self._finished.wait())
    try:
      async with asyncio.timeout_at(self._deadline):
        for role in initial_roles:
          self._spawn(role)
        seen = set()
        while True:
          pending = set(self._tasks.values()) - seen
          if not pending:
            break
          done, _ = await asyncio.wait(
              pending | {finish_waiter}, return_when=asyncio.FIRST_COMPLETED
          )
          if self._finished.is_set():
            status = "finished"
            break
          for task in done:
            seen.add(task)
            try:
              task.result()
            except asyncio.CancelledError:
              self._errors.append(
                  "CancelledError: agent cancelled before completion"
              )
              status = "failed"
            except Exception as exc:
              self._errors.append(f"{type(exc).__name__}: {exc}")
              status = "failed"
    except TimeoutError:
      status = "deadline"
      self._record("deadline", "")
    finally:
      self._running = False
      finish_waiter.cancel()
      await asyncio.gather(finish_waiter, return_exceptions=True)
      for task in self._tasks.values():
        if not task.done():
          cancelled += 1
          task.cancel()
      # Sampler adapters must honor cancellation; production transports must
      # cancel the remote request as well before releasing episode resources.
      await asyncio.gather(*self._tasks.values(), return_exceptions=True)
    return CollectiveResult(
        tuple(self._calls),
        self._answer,
        status,
        tuple(self._events),
        tuple(self._errors),
        self._requested_calls,
        cancelled,
    )


def build_team_credit(
    episodes: Sequence[trace.SwarmEpisode],
) -> credit.CreditBatch:
  """Team-GRPO adaptation: one outcome advantage for all team action tokens.

  Uses sample-standard-deviation group normalization and no role-specific
  penalties. This deliberately does not claim to implement Avalon's CISPO.
  """
  return credit.build_credit(
      episodes, normalization="episode", invalid_action_penalty=0.0
  )
