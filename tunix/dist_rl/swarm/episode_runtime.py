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
"""Option 5 unified EpisodeRuntime: `spawn`, `parallel`, budgets, & TITO."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
import dataclasses
import time
from typing import Any

import numpy as np
from tunix.dist_rl.rollout import sampler as sampler_lib
from tunix.dist_rl.swarm import contracts


class BudgetExhaustedError(RuntimeError):
  """Raised internally when an episode reaches CriticalSteps or WallClock cap."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class RoleSamplerAdapter:
  """Transport adapter binding one role to a dist_rl Sampler and tokenizer."""

  sampler: sampler_lib.Sampler
  encode: Callable[[str], Sequence[int]]
  sampling_params: sampler_lib.SamplingParams
  max_prompt_length: int
  fidelity: contracts.ServingFidelity = dataclasses.field(
      default_factory=contracts.ServingFidelity
  )
  attested_snapshot_fn: Callable[[], contracts.SnapshotRef] | None = None


def _try_append_tito(
    prev: contracts.CompactedSequence,
    action: contracts.ModelAction,
) -> contracts.CompactedSequence | None:
  """Merges `action` into `prev` iff participant, snapshot, and prefix match."""
  if (
      prev.participant_id != action.participant_id
      or prev.policy_id != action.policy_id
      or prev.served_snapshot.digest != action.served_snapshot.digest
      or prev.served_snapshot.adapter_id != action.served_snapshot.adapter_id
  ):
    return None
  prev_full = prev.prompt_ids + prev.completion_ids
  if len(action.prompt_ids) < len(prev_full):
    return None
  if action.prompt_ids[: len(prev_full)] != prev_full:
    return None
  obs_tokens = action.prompt_ids[len(prev_full) :]
  obs_len = len(obs_tokens)
  gen_len = len(action.completion_ids)
  act_lps = (
      action.rollout_logprobs
      if action.rollout_logprobs
      else (0.0,) * gen_len
  )
  return contracts.CompactedSequence(
      participant_id=prev.participant_id,
      role=prev.role,
      policy_id=prev.policy_id,
      served_snapshot=prev.served_snapshot,
      action_ids=prev.action_ids + (action.action_id,),
      prompt_ids=prev.prompt_ids,
      completion_ids=prev.completion_ids + obs_tokens + action.completion_ids,
      rollout_logprobs=prev.rollout_logprobs + (0.0,) * obs_len + act_lps,
      loss_mask=prev.loss_mask + (0,) * obs_len + (1,) * gen_len,
      attention_mask=prev.attention_mask + (1,) * (obs_len + gen_len),
  )


def compact_actions(
    actions: Sequence[contracts.ModelAction],
    *,
    roles: Sequence[str] | None = None,
    policy_id: str | None = None,
) -> tuple[contracts.CompactedSequence, ...]:
  """Auto-compacts append-only participant trajectories without `open_session`.

  Actions are filtered by optional `roles` and `policy_id`. Consecutive actions
  belonging to the same `(participant_id, served_snapshot)` whose prompt is an
  exact prefix extension of the previous prompt+completion are merged into one
  TITO row (observation tokens get `loss_mask=0, attention_mask=1`). Concurrent
  siblings or restructured prompts stay separate automatically.
  """
  role_set = set(roles) if roles is not None else None
  active: dict[str, int] = {}
  sequences: list[contracts.CompactedSequence] = []
  for act in actions:
    if not act.loss_eligible:
      continue
    if role_set is not None and act.role not in role_set:
      continue
    if policy_id is not None and act.policy_id != policy_id:
      continue
    gen_len = len(act.completion_ids)
    lps = act.rollout_logprobs if act.rollout_logprobs else (0.0,) * gen_len
    prev_idx = active.get(act.participant_id)
    if prev_idx is not None:
      merged = _try_append_tito(sequences[prev_idx], act)
      if merged is not None:
        sequences[prev_idx] = merged
        continue
    seq = contracts.CompactedSequence(
        participant_id=act.participant_id,
        role=act.role,
        policy_id=act.policy_id,
        served_snapshot=act.served_snapshot,
        action_ids=(act.action_id,),
        prompt_ids=act.prompt_ids,
        completion_ids=act.completion_ids,
        rollout_logprobs=lps,
        loss_mask=(1,) * gen_len,
        attention_mask=(1,) * gen_len,
    )
    active[act.participant_id] = len(sequences)
    sequences.append(seq)
  return tuple(sequences)


class TaskHandle:
  """Handle for one spawned child workflow (basis for both parallel & peers)."""

  def __init__(
      self,
      *,
      child_id: str,
      participant_id: str,
      parent_action_id: str | None,
      cohort_id: str | None,
      task: asyncio.Task[contracts.ChildRunRef],
  ) -> None:
    self.child_id = child_id
    self.participant_id = participant_id
    self.parent_action_id = parent_action_id
    self.cohort_id = cohort_id
    self._task = task

  def done(self) -> bool:
    return self._task.done()

  def cancel(self) -> None:
    if not self._task.done():
      self._task.cancel()

  async def join(self) -> contracts.ChildRunRef:
    return await self._task


@dataclasses.dataclass
class _SharedEpisodeState:
  """Mutable root-owned accounting shared across a root and its children."""

  episode_id: str
  task_id: str
  group_id: str
  slot_index: int
  condition: str
  snapshots: contracts.SnapshotBundle
  bindings: Mapping[str, contracts.PolicySpec]
  adapters: Mapping[str, RoleSamplerAdapter]
  sampler_sem: asyncio.Semaphore
  budget: contracts.EpisodeBudget | None
  start_time: float
  actions: list[contracts.ModelAction] = dataclasses.field(default_factory=list)
  children: list[contracts.ChildRunRef] = dataclasses.field(default_factory=list)
  live_handles: list[TaskHandle] = dataclasses.field(default_factory=list)
  next_action_idx: int = 0
  next_child_idx: int = 0
  next_cohort_idx: int = 0
  total_steps: int = 0
  worker_cost: float = 0.0


class EpisodeRuntime:
  """Unified Option 5 runtime supporting `generate`, `spawn`, and `parallel`."""

  def __init__(
      self,
      *,
      env: Any = None,
      shared: _SharedEpisodeState,
      participant_id: str,
      parent_action_id: str | None = None,
      cohort_id: str | None = None,
      critical_limit: int | None = None,
  ) -> None:
    self.env = env
    self._shared = shared
    self.participant_id = participant_id
    self.parent_action_id = parent_action_id
    self.cohort_id = cohort_id
    self._critical_used = 0
    self._local_total_steps = 0
    self._local_cost = 0.0
    self._critical_limit = critical_limit
    if self._critical_limit is None and isinstance(
        shared.budget, contracts.CriticalStepsBudget
    ):
      self._critical_limit = shared.budget.limit
    self._closed = False

  @classmethod
  def create_root(
      cls,
      *,
      episode_id: str,
      task_id: str,
      group_id: str,
      slot_index: int,
      snapshots: contracts.SnapshotBundle,
      bindings: Mapping[str, contracts.PolicySpec],
      adapters: Mapping[str, RoleSamplerAdapter],
      sampler_sem: asyncio.Semaphore,
      budget: contracts.EpisodeBudget | None = None,
      env: Any = None,
      condition: str = "default",
  ) -> "EpisodeRuntime":
    shared = _SharedEpisodeState(
        episode_id=episode_id,
        task_id=task_id,
        group_id=group_id,
        slot_index=slot_index,
        condition=condition,
        snapshots=snapshots,
        bindings=bindings,
        adapters=adapters,
        sampler_sem=sampler_sem,
        budget=budget,
        start_time=time.monotonic(),
    )
    return cls(env=env, shared=shared, participant_id=f"{episode_id}/root")

  @property
  def critical_steps_used(self) -> int:
    return self._critical_used

  @property
  def remaining_critical_steps(self) -> int | None:
    if self._critical_limit is None:
      return None
    return max(0, self._critical_limit - self._critical_used)

  def _check_budget_before_step(self) -> None:
    if self._closed:
      raise RuntimeError("EpisodeRuntime is already finalized.")
    if self._critical_limit is not None and self._critical_used >= self._critical_limit:
      raise BudgetExhaustedError(
          f"CriticalSteps budget exhausted ({self._critical_used}/{self._critical_limit})."
      )
    budget = self._shared.budget
    if isinstance(budget, contracts.WallClockBudget):
      elapsed = time.monotonic() - self._shared.start_time
      if elapsed >= budget.deadline_s:
        raise BudgetExhaustedError(
            f"WallClock budget exhausted ({elapsed:.2f}s >= {budget.deadline_s:.2f}s)."
        )
      if (
          budget.max_total_calls is not None
          and self._shared.total_steps >= budget.max_total_calls
      ):
        raise BudgetExhaustedError(
            f"Call budget exhausted ({self._shared.total_steps}/{budget.max_total_calls})."
        )

  def _verify_attestation(
      self,
      role: str,
      expected: contracts.SnapshotRef,
      adapter: RoleSamplerAdapter,
      response: sampler_lib.SamplingResponse,
  ) -> None:
    if not adapter.fidelity.require_attestation:
      return
    served_digest = response.metadata.get("served_digest")
    served_adapter = response.metadata.get("served_adapter_id")
    if served_digest is None and adapter.attested_snapshot_fn is not None:
      att = adapter.attested_snapshot_fn()
      served_digest = att.digest
      served_adapter = att.adapter_id
    if served_digest is not None and served_digest != expected.digest:
      raise RuntimeError(
          f"Snapshot attestation mismatch for role {role!r}: expected "
          f"{expected.digest!r}, served {served_digest!r}."
      )
    if (
        expected.adapter_id is not None
        and served_adapter is not None
        and served_adapter != expected.adapter_id
    ):
      raise RuntimeError(
          f"Adapter attestation mismatch for role {role!r}: expected "
          f"{expected.adapter_id!r}, served {served_adapter!r}."
      )

  async def generate(self, role: str, prompt: str) -> contracts.ModelAction:
    """Routes one generation call with two-level admission & attestation."""
    self._check_budget_before_step()
    if role not in self._shared.bindings:
      raise KeyError(f"Unbound swarm role: {role!r}.")
    binding = self._shared.bindings[role]
    adapter = self._shared.adapters[role]
    expected_snap = self._shared.snapshots.snapshot_for_role(role)
    prompt_ids = np.asarray(adapter.encode(prompt), dtype=np.int32)
    if prompt_ids.ndim != 1 or not 0 < len(prompt_ids) <= adapter.max_prompt_length:
      raise ValueError(
          f"Encoded prompt length {len(prompt_ids)} outside (0, {adapter.max_prompt_length}]."
      )
    idx = self._shared.next_action_idx
    self._shared.next_action_idx += 1
    action_id = f"{self._shared.episode_id}/a{idx}"
    params = dataclasses.replace(
        adapter.sampling_params,
        return_logprobs=bool(
            binding.capture_logprobs or adapter.sampling_params.return_logprobs
        ),
        seed=None
        if adapter.sampling_params.seed is None
        else adapter.sampling_params.seed + idx,
    )
    async with self._shared.sampler_sem:
      response = await adapter.sampler.sample(
          sampler_lib.SamplingRequest(
              request_id=action_id,
              prompt=prompt_ids,
              sampling_params=params,
              metadata={
                  "role": role,
                  "participant_id": self.participant_id,
                  "policy_id": expected_snap.model_id,
                  "policy_version": expected_snap.revision,
                  "expected_digest": expected_snap.digest,
                  "expected_adapter_id": expected_snap.adapter_id,
              },
          )
      )
    if response.error is not None or response.finish_reason not in ("stop", "length"):
      raise RuntimeError(
          f"Sampling failed for {action_id}: {response.finish_reason} ({response.error})"
      )
    self._verify_attestation(role, expected_snap, adapter, response)
    if binding.capture_logprobs and response.logprobs is None:
      raise ValueError(f"Role {role!r} requires behavior logprobs from sampler.")
    lps = () if response.logprobs is None else tuple(response.logprobs.tolist())
    action = contracts.ModelAction(
        action_id=action_id,
        participant_id=self.participant_id,
        role=role,
        policy_id=binding.learner if binding.kind == "follow" else expected_snap.model_id,
        served_snapshot=expected_snap,
        prompt_ids=tuple(prompt_ids.tolist()),
        completion_ids=tuple(response.token_ids.tolist()),
        rollout_logprobs=lps,
        text=response.text,
        parent_action_id=self.parent_action_id,
        cohort_id=self.cohort_id,
        finish_reason=response.finish_reason,
        loss_eligible=(binding.kind == "follow"),
    )
    self._shared.actions.append(action)
    self._critical_used += 1
    self._local_total_steps += 1
    self._shared.total_steps += 1
    if binding.kind == "fixed":
      self._local_cost += 1.0
      self._shared.worker_cost += 1.0
    return action

  def annotate(self, action: contracts.ModelAction, **facts: Any) -> None:
    """Attaches public non-secret metadata (e.g. `valid=True`) to an action."""
    action.annotations.update(facts)

  def spawn(
      self,
      worker_workflow: Callable[[Any, "EpisodeRuntime"], Awaitable[Any]],
      job: Any,
      *,
      parent: contracts.ModelAction | None = None,
      cohort_id: str | None = None,
      env: Any = None,
  ) -> TaskHandle:
    """Spawns one isolated child participant with deterministic lineage."""
    if self._closed:
      raise RuntimeError("Cannot spawn after EpisodeRuntime finalization.")
    child_idx = self._shared.next_child_idx
    self._shared.next_child_idx += 1
    child_id = f"{self._shared.episode_id}/c{child_idx}"
    child_ctx = EpisodeRuntime(
        env=self.env if env is None else env,
        shared=self._shared,
        participant_id=child_id,
        parent_action_id=parent.action_id if parent is not None else self.parent_action_id,
        cohort_id=cohort_id,
        critical_limit=self.remaining_critical_steps,
    )

    async def _run_child() -> contracts.ChildRunRef:
      status = "completed"
      output: Any = None
      err_msg: str | None = None
      try:
        if child_ctx.remaining_critical_steps == 0:
          raise BudgetExhaustedError("Zero remaining critical steps for child.")
        output = await worker_workflow(job, child_ctx)
      except BudgetExhaustedError as exc:
        status = "budget_exhausted"
        err_msg = str(exc)
      except asyncio.CancelledError:
        status = "cancelled"
        raise
      except Exception as exc:  # pylint: disable=broad-exception-caught
        status = "failed"
        err_msg = str(exc)
      ref = contracts.ChildRunRef(
          child_id=child_id,
          participant_id=child_id,
          parent_action_id=child_ctx.parent_action_id,
          cohort_id=cohort_id,
          status=status,
          critical_steps=child_ctx.critical_steps_used,
          total_steps=child_ctx._local_total_steps,  # pylint: disable=protected-access
          cost=child_ctx._local_cost,  # pylint: disable=protected-access
          output=output,
          error=err_msg,
      )
      self._shared.children.append(ref)
      return ref

    handle = TaskHandle(
        child_id=child_id,
        participant_id=child_id,
        parent_action_id=child_ctx.parent_action_id,
        cohort_id=cohort_id,
        task=asyncio.create_task(_run_child()),
    )
    self._shared.live_handles.append(handle)
    return handle

  async def parallel(
      self,
      jobs: Sequence[Any],
      worker_workflow: Callable[[Any, "EpisodeRuntime"], Awaitable[Any]],
      *,
      parent: contracts.ModelAction | None = None,
  ) -> list[Any]:
    """Fork-Join sugar over `spawn`: allocates ordered IDs & charges max depth."""
    if not jobs:
      return []
    self._check_budget_before_step()
    c_idx = self._shared.next_cohort_idx
    self._shared.next_cohort_idx += 1
    cohort_id = f"{self._shared.episode_id}/w{c_idx}"
    handles = [
        self.spawn(worker_workflow, job, parent=parent, cohort_id=cohort_id)
        for job in jobs
    ]
    try:
      refs = await asyncio.gather(*(h.join() for h in handles))
    except BaseException:
      for h in handles:
        h.cancel()
      await asyncio.gather(*(h.join() for h in handles), return_exceptions=True)
      raise
    self._critical_used += max((r.critical_steps for r in refs), default=0)
    return [r.output for r in refs]

  async def finalize(
      self,
      public_output: Any,
      *,
      status: str = "completed",
      metadata: Mapping[str, Any] | None = None,
  ) -> contracts.EpisodeRecord:
    """Cancels/joins any dangling children and seals an immutable EpisodeRecord."""
    self._closed = True
    for handle in list(self._shared.live_handles):
      if not handle.done():
        handle.cancel()
    if self._shared.live_handles:
      await asyncio.gather(
          *(h.join() for h in self._shared.live_handles),
          return_exceptions=True,
      )
    return contracts.EpisodeRecord(
        episode_id=self._shared.episode_id,
        task_id=self._shared.task_id,
        group_id=self._shared.group_id,
        slot_index=self._shared.slot_index,
        condition=self._shared.condition,
        snapshots=self._shared.snapshots,
        status=status,
        public_output=public_output,
        actions=tuple(self._shared.actions),
        children=tuple(self._shared.children),
        critical_steps=self._critical_used,
        total_steps=self._shared.total_steps,
        wall_time_s=time.monotonic() - self._shared.start_time,
        worker_cost=self._shared.worker_cost,
        metadata=dict(metadata or {}),
    )


async def agent_env_workflow(
    task: Any,
    ctx: EpisodeRuntime,
    *,
    role: str = "solver",
    max_turns: int = 8,
) -> Any:
  """Built-in single-agent multi-turn tool loop using Option 5 EpisodeRuntime."""
  prompt = await ctx.env.reset(task)
  last_text = ""
  for _ in range(max_turns):
    action = await ctx.generate(role, prompt)
    last_text = action.text
    obs, done, info = await ctx.env.step(action.text)
    if info:
      ctx.annotate(action, **info)
    if done:
      return obs if obs else last_text
    prompt = obs
  return last_text
