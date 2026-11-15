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

"""User-defined swarm workflows over existing dist_rl sampling and traces.

The API separates workflow, policy ownership, runtime transport, and grading.
It returns the same SwarmEpisode consumed by SwarmRLProgram; algorithms and
credit assignment remain explicit at program construction. It is not a second
training framework or a sandbox for untrusted Python workflows.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
import dataclasses
import math
import numbers
import types

from tunix.dist_rl.swarm import policy
from tunix.dist_rl.swarm import runtime
from tunix.dist_rl.swarm import trace

RuntimeFactory = Callable[[str, int, str], runtime.RecordingRuntime]


@dataclasses.dataclass(frozen=True, kw_only=True)
class SwarmOutput:
  """Public workflow result and optional credit-assignment context."""

  final_text: str
  scaffold_condition: str = "none"
  worker_cost: float = 0.0


class SwarmSession:
  """Bounded recording context passed to one user workflow invocation.

  Each policy owns a RecordingRuntime and disjoint call-ID namespace. No task
  answer, grader or reward is stored on this session. A workflow must await all
  its calls before returning; dangling calls are cancelled and rejected.
  """

  def __init__(
      self,
      spec: SwarmSpec,
      learner_version: int,
      episode_id: str,
      runtime_factory: RuntimeFactory,
  ):
    self._roles = spec.roles
    self._max_calls = spec.max_calls
    self._semaphore = asyncio.Semaphore(spec.max_concurrent_calls)
    self._runtimes = {}
    self._active = set()
    self._results = {}
    self._requested = 0
    self._closed = False
    for index, binding in enumerate(spec.learning_plan.bindings):
      version = learner_version if binding.trainable else binding.version
      instance = runtime_factory(
          binding.policy_id, version, f"{episode_id}/p{index}"
      )
      if (
          instance.policy_id != binding.policy_id
          or instance.policy_version != version
      ):
        raise ValueError("Runtime factory did not honor the policy binding.")
      self._runtimes[binding.policy_id] = instance

  @property
  def calls(self) -> tuple[trace.CallTrace, ...]:
    """Completed traces in request order, independent of completion timing."""
    return tuple(self._results[index] for index in sorted(self._results))

  async def generate(self, role: str, prompt: str) -> trace.CallTrace:
    if self._closed:
      raise RuntimeError("The swarm session is closed.")
    if role not in self._roles:
      raise ValueError(f"Unknown swarm role: {role}.")
    if not isinstance(prompt, str) or not prompt:
      raise ValueError("Generation requires a nonempty public prompt.")
    if self._requested >= self._max_calls:
      raise RuntimeError("Swarm call budget exhausted.")
    index = self._requested
    self._requested += 1
    task = asyncio.current_task()
    self._active.add(task)
    try:
      async with self._semaphore:
        if self._closed:
          raise RuntimeError("The swarm session is closed.")
        call = await self._runtimes[self._roles[role]].generate(role, prompt)
        if any(
            existing.call_id == call.call_id
            for existing in self._results.values()
        ):
          raise ValueError("Runtime factory reused a call-ID namespace.")
        self._results[index] = call
        return call
    finally:
      self._active.discard(task)

  async def _close(self):
    had_pending = bool(self._active)
    if not self._closed and not self._active:
      # Also register tasks queued immediately before a workflow exception.
      # They must be cancelled and awaited, rather than failing unobserved
      # after the collector has moved on to another attempt.
      await asyncio.sleep(0)
      had_pending = bool(self._active) or had_pending
    self._closed = True
    pending = set(self._active)
    for task in pending:
      task.cancel()
    if pending:
      await asyncio.gather(*pending, return_exceptions=True)
    return had_pending


@dataclasses.dataclass(frozen=True, kw_only=True)
class SwarmSpec:
  """Bind logical roles to policies and execute a user-supplied async workflow.

  `runtime_factory(policy_id, version, episode_namespace)` supplies the existing
  RecordingRuntime around a native or external dist_rl Sampler. The factory
  must honor cancellation. `grade(final_text)` is evaluated only after workflow
  completion and can close over the target privately on the collector side.
  """

  roles: Mapping[str, str]
  workflow: Callable[[SwarmSession, str], Awaitable[str | SwarmOutput]]
  learning_plan: policy.LearningPlan
  max_calls: int = 32
  max_concurrent_calls: int = 4
  timeout_seconds: float = 300.0

  def __post_init__(self):
    roles = dict(self.roles)
    known = {b.policy_id for b in self.learning_plan.bindings}
    if not roles or any(
        not role or pid not in known for role, pid in roles.items()
    ):
      raise ValueError("Every nonempty role must bind a declared policy.")
    if self.learning_plan.learner_policy_id not in roles.values():
      raise ValueError("At least one role must bind the learner policy.")
    if not callable(self.workflow):
      raise ValueError("A swarm workflow must be callable.")
    if any(
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or value < 1
        for value in (self.max_calls, self.max_concurrent_calls)
    ):
      raise ValueError("Swarm call and concurrency budgets must be positive.")
    if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
      raise ValueError("A swarm workflow needs a finite positive timeout.")
    object.__setattr__(self, "roles", types.MappingProxyType(roles))

  async def run(
      self,
      *,
      task_id: str,
      episode_id: str,
      prompt: str,
      learner_version: int,
      runtime_factory: RuntimeFactory,
      grade: Callable[[str], float],
  ) -> trace.SwarmEpisode:
    """Execute, validate exact token evidence, then grade a complete attempt.

    Failures propagate to the collector; this API never silently drops a failed
    attempt from its GRPO comparison group. Deadline/anytime semantics belong
    to CollectiveRuntime and a recipe's explicit admission policy instead.
    """
    if (
        not task_id
        or not episode_id
        or not isinstance(prompt, str)
        or not prompt
    ):
      raise ValueError("A swarm run requires IDs and a public prompt.")
    if (
        isinstance(learner_version, bool)
        or not isinstance(learner_version, numbers.Integral)
        or learner_version < 0
    ):
      raise ValueError("Learner versions must be nonnegative integers.")
    session = SwarmSession(self, learner_version, episode_id, runtime_factory)
    try:
      async with asyncio.timeout(self.timeout_seconds):
        output = await self.workflow(session, prompt)
      if isinstance(output, str):
        output = SwarmOutput(final_text=output)
      if not isinstance(output, SwarmOutput):
        raise TypeError("Workflow must return text or SwarmOutput.")
      if not isinstance(output.final_text, str):
        raise TypeError("Workflow final_text must be a string.")
      had_pending = bool(session._active)
      # Register generation tasks created immediately before workflow return.
      # Otherwise a queued create_task(generate(...)) could start after grade.
      await asyncio.sleep(0)
      had_pending = await session._close() or had_pending
      if had_pending:
        raise RuntimeError("Workflow returned with pending generation calls.")
      episode = trace.SwarmEpisode(
          task_id=task_id,
          episode_id=episode_id,
          calls=session.calls,
          reward=0.0,
          final_text=output.final_text,
          scaffold_condition=output.scaffold_condition,
          worker_cost=output.worker_cost,
      )
      self.learning_plan.validate([episode], learner_version)
      return dataclasses.replace(
          episode, reward=float(grade(output.final_text))
      )
    finally:
      await session._close()
