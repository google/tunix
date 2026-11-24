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
"""Option 5 L2 control plane: dual-mode StepContext, replay journal, & manifests."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
import dataclasses
import enum
import math
from pathlib import Path
import statistics
from typing import Any

from tunix.dist_rl.swarm import checkpoint
from tunix.dist_rl.swarm import contracts
from tunix.dist_rl.swarm import episode_runtime


class CollectionMode(str, enum.Enum):
  """Execution mode for `StepContext.collect`."""

  STRICT_BUNDLE = "strict_bundle"
  ASYNC_WINDOW = "async_window"


@dataclasses.dataclass(frozen=True, kw_only=True)
class AttemptSpec:
  """One logical episode slot scheduled inside an AttemptPlan."""

  group_id: str
  slot_index: int
  task_id: str
  public_task: Any
  seed: int
  condition: str = "default"
  metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True, kw_only=True)
class AttemptPlan:
  """Ordered collection of logical attempts for one step."""

  attempts: tuple[AttemptSpec, ...]

  def __post_init__(self) -> None:
    object.__setattr__(self, "attempts", tuple(self.attempts))
    if not self.attempts:
      raise ValueError("AttemptPlan requires at least one AttemptSpec.")


def repeat_tasks(
    tasks: Sequence[Mapping[str, Any]],
    *,
    attempts: int = 4,
    base_seed: int = 0,
    condition: str = "default",
) -> AttemptPlan:
  """Expands public task rows into `attempts` comparison slots per group."""
  specs: list[AttemptSpec] = []
  for t_idx, row in enumerate(tasks):
    task_id = str(row["id"])
    public = row["public"]
    for slot in range(attempts):
      specs.append(
          AttemptSpec(
              group_id=f"{task_id}:{condition}",
              slot_index=slot,
              task_id=task_id,
              public_task=public,
              seed=base_seed + t_idx * attempts + slot,
              condition=condition,
          )
      )
  return AttemptPlan(attempts=tuple(specs))


class ManifestStore:
  """Conditional atomic manifest store with pre-commit `step.record` journal."""

  def __init__(self, root: Path) -> None:
    self.root = Path(root)
    self.root.mkdir(parents=True, exist_ok=True)
    self.manifest_path = self.root / "committed_manifest.json"
    self.journal_path = self.root / "pending_record_journal.json"

  def load_committed(self) -> dict[str, Any] | None:
    if not self.manifest_path.is_file():
      return None
    return checkpoint.read_manifest(self.manifest_path)

  def load_journal(self, step_id: int) -> dict[str, Any]:
    if not self.journal_path.is_file():
      return {}
    data = checkpoint.read_manifest(self.journal_path)
    if int(data.get("step_id", -1)) != int(step_id):
      return {}
    return dict(data.get("records", {}))

  def save_journal_entry(self, step_id: int, key: str, value: Any) -> None:
    records = self.load_journal(step_id)
    records[key] = value
    checkpoint.write_manifest(
        self.journal_path,
        {"step_id": int(step_id), "records": records},
    )

  def commit_step(
      self,
      *,
      step_id: int,
      expected_prev_step: int,
      learner_versions: Mapping[str, int],
      recipe_state: Any,
      cursor: int,
      metrics: Mapping[str, float],
  ) -> dict[str, Any]:
    """Atomically commits step state with optimistic prev-step guard."""
    prev = self.load_committed()
    actual_prev = -1 if prev is None else int(prev["step_id"])
    if actual_prev != int(expected_prev_step):
      raise RuntimeError(
          f"Manifest CAS conflict: expected prev step {expected_prev_step}, "
          f"found {actual_prev}."
      )
    state = {
        "step_id": int(step_id),
        "learner_versions": {k: int(v) for k, v in learner_versions.items()},
        "recipe_state": recipe_state,
        "cursor": int(cursor),
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    checkpoint.write_manifest(self.manifest_path, state)
    self.journal_path.unlink(missing_ok=True)
    return state


class StepContext:
  """Option 5 composable step context uniting strict & async-window collection."""

  def __init__(
      self,
      *,
      step_id: int,
      tasks: Sequence[Mapping[str, Any]],
      private_examples: Mapping[str, Any],
      recipe: contracts.Recipe,
      snapshots: contracts.SnapshotBundle,
      adapters: Mapping[str, episode_runtime.RoleSamplerAdapter],
      learners: Mapping[str, Any],
      manifest_store: ManifestStore | None = None,
      state: Any = None,
      cursor: int = 0,
      env_factory: Callable[[AttemptSpec], Any] | None = None,
      snapshot_provider: Callable[[], contracts.SnapshotBundle] | None = None,
      publish_callback: Callable[[Mapping[str, int]], Awaitable[None] | None] | None = None,
  ) -> None:
    self.step_id = int(step_id)
    self.tasks = tuple(tasks)
    self._private_examples = dict(private_examples)
    self.recipe = recipe
    self.snapshots = snapshots
    self._adapters = dict(adapters)
    self._learners = dict(learners)
    self._manifest_store = manifest_store
    self.state = state
    self.cursor = int(cursor)
    self._env_factory = env_factory
    self._snapshot_provider = snapshot_provider
    self._publish_callback = publish_callback
    self._staged_versions: dict[str, int] = {
        k: snap.revision for k, snap in snapshots.learner_snapshots.items()
    }
    self._recorded_cache: dict[str, Any] = (
        manifest_store.load_journal(step_id) if manifest_store is not None else {}
    )

  async def record(self, key: str, operation: Callable[[], Any]) -> Any:
    """Runs or replays a keyed outer step decision across pre-commit crashes."""
    if key in self._recorded_cache:
      return self._recorded_cache[key]
    value = operation()
    if asyncio.iscoroutine(value):
      value = await value
    self._recorded_cache[key] = value
    if self._manifest_store is not None:
      self._manifest_store.save_journal_entry(self.step_id, key, value)
    return value

  def _resolve_collect_bundle(self, mode: CollectionMode) -> contracts.SnapshotBundle:
    if mode == CollectionMode.ASYNC_WINDOW and self._snapshot_provider is not None:
      return self._snapshot_provider()
    return self.snapshots

  async def _execute_single_attempt(
      self,
      spec: AttemptSpec,
      *,
      mode: CollectionMode,
      sampler_sem: asyncio.Semaphore,
      max_retries: int,
  ) -> contracts.EpisodeRecord:
    last_err: Exception | None = None
    for retry in range(max_retries + 1):
      rt = episode_runtime.EpisodeRuntime.create_root(
          episode_id=f"s{self.step_id}/{spec.group_id}/slot{spec.slot_index}/r{retry}",
          task_id=spec.task_id,
          group_id=spec.group_id,
          slot_index=spec.slot_index,
          snapshots=self._resolve_collect_bundle(mode),
          bindings=self.recipe.bindings,
          adapters=self._adapters,
          sampler_sem=sampler_sem,
          budget=self.recipe.budget,
          env=self._env_factory(spec) if self._env_factory is not None else None,
          condition=spec.condition,
      )
      meta = {"mode": mode.value, "seed": spec.seed, **spec.metadata}
      try:
        public_out = await self.recipe.workflow(spec.public_task, rt)
        return await rt.finalize(public_out, status="completed", metadata=meta)
      except episode_runtime.BudgetExhaustedError as exc:
        return await rt.finalize(
            None,
            status="budget_exhausted",
            metadata={**meta, "cutoff_reason": str(exc)},
        )
      except Exception as exc:  # pylint: disable=broad-exception-caught
        last_err = exc
        await rt.finalize(None, status="cancelled")
        if retry >= max_retries:
          raise
    raise RuntimeError(f"Attempt failed: {last_err}")

  async def collect(
      self,
      plan: AttemptPlan,
      *,
      mode: CollectionMode = CollectionMode.STRICT_BUNDLE,
      max_concurrent_episodes: int = 32,
      max_concurrent_sampler_calls: int = 64,
      max_retries: int = 1,
  ) -> list[contracts.EpisodeRecord]:
    """Collects episodes with two-level admission in strict or async-window mode."""
    episode_sem = asyncio.Semaphore(max(1, int(max_concurrent_episodes)))
    sampler_sem = asyncio.Semaphore(max(1, int(max_concurrent_sampler_calls)))

    async def _admit(spec: AttemptSpec) -> contracts.EpisodeRecord:
      async with episode_sem:
        return await self._execute_single_attempt(
            spec,
            mode=mode,
            sampler_sem=sampler_sem,
            max_retries=max_retries,
        )

    return list(await asyncio.gather(*(_admit(s) for s in plan.attempts)))

  async def grade(
      self, episodes: Sequence[contracts.EpisodeRecord]
  ) -> list[contracts.ScoreRecord]:
    """Runs post-collection private grading joined strictly by task_id."""
    scores: list[contracts.ScoreRecord] = []
    for ep in episodes:
      if ep.task_id not in self._private_examples:
        raise KeyError(f"Missing private example for task_id={ep.task_id!r}.")
      result = self.recipe.grade(ep, self._private_examples[ep.task_id])
      if asyncio.iscoroutine(result):
        result = await result
      scores.append(result)
    return scores

  async def update(self, target: str, batch: contracts.ObjectiveBatch) -> dict[str, float]:
    """Validates ownership & policy lag, then stages one learner update."""
    if target not in self.recipe.learners or target not in self._learners:
      raise KeyError(f"Unknown learner target: {target!r}.")
    if batch.target != target:
      raise ValueError(
          f"ObjectiveBatch target {batch.target!r} mismatches {target!r}."
      )
    if not batch.examples:
      return {"skipped_empty": 1.0}
    cur_rev = self._staged_versions.get(target, 0)
    for ex in batch.examples:
      rev = ex.get("policy_version")
      if rev is not None and (cur_rev - int(rev)) > batch.max_policy_lag:
        raise ValueError(
            f"Stale behavior policy version {rev} for target {target!r} "
            f"(current={cur_rev}, max_lag={batch.max_policy_lag})."
        )
    trainer = self._learners[target]
    metrics = trainer.stage_update(batch)
    if asyncio.iscoroutine(metrics):
      metrics = await metrics
    self._staged_versions[target] = cur_rev + 1
    return dict(metrics or {})

  async def commit(self, result: contracts.StepResult) -> dict[str, Any]:
    """Commits all staged learner versions + recipe state atomically & publishes."""
    for name, trainer in self._learners.items():
      if hasattr(trainer, "commit_staged"):
        trainer.commit_staged(self._staged_versions[name])
    manifest = {
        "step_id": self.step_id,
        "learner_versions": dict(self._staged_versions),
        "recipe_state": result.state,
        "cursor": self.cursor + len(self.tasks),
        "metrics": dict(result.metrics),
    }
    if self._manifest_store is not None:
      manifest = self._manifest_store.commit_step(
          step_id=self.step_id,
          expected_prev_step=self.step_id - 1,
          learner_versions=self._staged_versions,
          recipe_state=result.state,
          cursor=self.cursor + len(self.tasks),
          metrics=result.metrics,
      )
    if self._publish_callback is not None:
      pub = self._publish_callback(dict(self._staged_versions))
      if asyncio.iscoroutine(pub):
        await pub
    return manifest


def parl_reward_schedule(
    step_id: int,
    *,
    anneal_steps: int = 40,
    lambda_parallel_init: float = 0.15,
    lambda_finish_init: float = 0.15,
) -> tuple[float, float]:
  """Linear annealing schedule driving PARL shaping coefficients to zero."""
  if anneal_steps <= 0 or step_id >= anneal_steps:
    return (0.0, 0.0)
  frac = max(0.0, 1.0 - float(step_id) / float(anneal_steps))
  return (lambda_parallel_init * frac, lambda_finish_init * frac)


def compute_group_advantages(
    episodes: Sequence[contracts.EpisodeRecord],
    rewards_by_episode: Mapping[str, float],
    *,
    eps: float = 1e-6,
) -> dict[str, float]:
  """Computes group-relative GRPO advantages per `episode.group_id`."""
  by_group: dict[str, list[contracts.EpisodeRecord]] = {}
  for ep in episodes:
    by_group.setdefault(ep.group_id, []).append(ep)
  advs: dict[str, float] = {}
  for grp in by_group.values():
    vals = [float(rewards_by_episode[e.episode_id]) for e in grp]
    if len(vals) <= 1:
      for e in grp:
        advs[e.episode_id] = 0.0
      continue
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals)
    if not math.isfinite(std) or std < 1e-8:
      for e in grp:
        advs[e.episode_id] = 0.0
      continue
    for e, v in zip(grp, vals):
      advs[e.episode_id] = (v - mean) / (std + eps)
  return advs


def build_compacted_grpo_batch(
    *,
    target: str,
    episodes: Sequence[contracts.EpisodeRecord],
    advantages: Mapping[str, float],
    roles: Sequence[str],
    max_policy_lag: int = 0,
    reduction: str = "token_mean",
) -> contracts.ObjectiveBatch:
  """Expands episodes into auto-compacted TITO training rows for `target`."""
  examples: list[dict[str, Any]] = []
  for ep in episodes:
    adv = float(advantages.get(ep.episode_id, 0.0))
    if abs(adv) < 1e-12:
      continue
    seqs = episode_runtime.compact_actions(
        ep.actions, roles=roles, policy_id=target
    )
    for seq in seqs:
      if not any(seq.loss_mask):
        continue
      examples.append({
          "episode_id": ep.episode_id,
          "group_id": ep.group_id,
          "participant_id": seq.participant_id,
          "role": seq.role,
          "policy_version": seq.served_snapshot.revision,
          "prompt_ids": seq.prompt_ids,
          "completion_ids": seq.completion_ids,
          "rollout_logprobs": seq.rollout_logprobs,
          "loss_mask": seq.loss_mask,
          "attention_mask": seq.attention_mask,
          "advantage": adv,
      })
  return contracts.ObjectiveBatch(
      target=target,
      objective_kind="token_grpo",
      examples=tuple(examples),
      reduction=reduction,
      roles=tuple(roles),
      max_policy_lag=max_policy_lag,
  )


def group_grpo(
    *,
    target: str,
    roles: Sequence[str],
    attempts: int = 4,
    mode: CollectionMode = CollectionMode.STRICT_BUNDLE,
    max_policy_lag: int = 0,
) -> Callable[[StepContext], Awaitable[contracts.StepResult]]:
  """Built-in 1-line single-learner GRPO train_step over Option 5 StepContext."""

  async def _step(step: StepContext) -> contracts.StepResult:
    plan = repeat_tasks(step.tasks, attempts=attempts, base_seed=step.step_id * 1000)
    episodes = await step.collect(plan, mode=mode)
    scores = await step.grade(episodes)
    rewards = {s.episode_id: s.reward for s in scores}
    advs = compute_group_advantages(episodes, rewards)
    batch = build_compacted_grpo_batch(
        target=target,
        episodes=episodes,
        advantages=advs,
        roles=roles,
        max_policy_lag=max_policy_lag,
    )
    upd = await step.update(target, batch)
    mean_r = statistics.fmean(rewards.values()) if rewards else 0.0
    return contracts.StepResult(
        state=step.state,
        metrics={"reward_mean": mean_r, "n_rows": float(len(batch.examples)), **upd},
    )

  return _step


def _episode_parl_terms(
    ep: contracts.EpisodeRecord,
    sc: contracts.ScoreRecord,
) -> tuple[float, float, float, float]:
  r_perf = float(sc.components.get("r_perf", sc.reward))
  n_ch = len(ep.children)
  if n_ch == 0:
    return r_perf, 0.0, 0.0, 0.0
  has_par = 1.0 if n_ch >= 2 else 0.0
  r_par = 0.5 + 0.5 * min(1.0, (n_ch - 2) / 4.0) if n_ch >= 2 else 0.0
  done = sum(
      1
      for c in ep.children
      if c.status == "completed" and len(str(c.output or "").strip()) >= 3
  )
  return r_perf, r_par, done / float(n_ch), has_par


def shape_parl_rewards(
    episodes: Sequence[contracts.EpisodeRecord],
    scores: Sequence[contracts.ScoreRecord],
    *,
    lam1: float,
    lam2: float,
) -> tuple[dict[str, float], dict[str, float]]:
  """Computes Kimi K2.5 PARL annealed rewards and CriticalSteps metrics."""
  score_by_ep = {s.episode_id: s for s in scores}
  shaped: dict[str, float] = {}
  perf_vals, par_vals, fin_vals, par_flags = [], [], [], []
  for ep in episodes:
    r_perf, r_par, r_fin, has_par = _episode_parl_terms(
        ep, score_by_ep[ep.episode_id]
    )
    shaped[ep.episode_id] = r_perf + lam1 * r_par + lam2 * r_fin
    perf_vals.append(r_perf)
    par_vals.append(r_par)
    fin_vals.append(r_fin)
    par_flags.append(has_par)
  def _avg(vals: Sequence[float]) -> float:
    return statistics.fmean(vals) if vals else 0.0
  mean_crit = _avg([float(e.critical_steps) for e in episodes])
  mean_tot = _avg([float(e.total_steps) for e in episodes])
  metrics = {
      "parl/r_perf": _avg(perf_vals),
      "parl/r_parallel": _avg(par_vals),
      "parl/r_finish": _avg(fin_vals),
      "parl/r_shaped": _avg(list(shaped.values())),
      "parl/parallel_rate": _avg(par_flags),
      "parl/lambda_1": lam1,
      "parl/lambda_2": lam2,
      "parl/critical_steps": mean_crit,
      "parl/total_steps": mean_tot,
      "parl/parallel_speedup": mean_tot / max(mean_crit, 1e-6),
      "parl/wall_time_s": _avg([float(e.wall_time_s) for e in episodes]),
      "parl/mean_children": _avg([float(len(e.children)) for e in episodes]),
  }
  return shaped, metrics


def parl_grpo(
    *,
    target: str = "director",
    roles: Sequence[str] = ("director",),
    attempts: int = 8,
    lambda_parallel_init: float = 0.15,
    lambda_finish_init: float = 0.15,
    anneal_steps: int = 40,
    mode: CollectionMode = CollectionMode.ASYNC_WINDOW,
    max_policy_lag: int = 1,
) -> Callable[[StepContext], Awaitable[contracts.StepResult]]:
  """Built-in Kimi K2.5 PARL step with annealed shaping & CriticalSteps metrics."""

  async def _step(step: StepContext) -> contracts.StepResult:
    lam1, lam2 = parl_reward_schedule(
        step.step_id,
        anneal_steps=anneal_steps,
        lambda_parallel_init=lambda_parallel_init,
        lambda_finish_init=lambda_finish_init,
    )
    plan = repeat_tasks(step.tasks, attempts=attempts, base_seed=step.step_id * 1000)
    episodes = await step.collect(plan, mode=mode)
    scores = await step.grade(episodes)
    shaped, parl_metrics = shape_parl_rewards(
        episodes, scores, lam1=lam1, lam2=lam2
    )
    advs = compute_group_advantages(episodes, shaped)
    groups = {e.group_id for e in episodes}
    active_groups = {
        e.group_id for e in episodes if abs(advs.get(e.episode_id, 0.0)) >= 1e-12
    }
    active_frac = len(active_groups) / float(max(len(groups), 1))
    batch = build_compacted_grpo_batch(
        target=target,
        episodes=episodes,
        advantages=advs,
        roles=roles,
        max_policy_lag=max_policy_lag,
    )
    upd = await step.update(target, batch)
    return contracts.StepResult(
        state=step.state,
        metrics={
            **parl_metrics,
            "parl/active_group_frac": active_frac,
            "parl/n_director_rows": float(len(batch.examples)),
            **upd,
        },
    )

  return _step
