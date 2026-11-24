# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Decoupled evaluation & CriticalSteps Pareto frontier utilities for PARL."""

from __future__ import annotations

import dataclasses
import statistics
from typing import Any

from examples.swarm_parl import workflow as parl_wf
from tunix.dist_rl.swarm import CollectionMode
from tunix.dist_rl.swarm import CriticalStepsBudget
from tunix.dist_rl.swarm import Recipe
from tunix.dist_rl.swarm import RoleSamplerAdapter
from tunix.dist_rl.swarm import SnapshotBundle
from tunix.dist_rl.swarm import StepContext
from tunix.dist_rl.swarm import repeat_tasks

SUITE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("wide_wikiqa", "wide_"),
    ("multihop_qa", "mh_"),
    ("browsecomp_plus", "bc_"),
)


def summarize_policy_subset(
    prefix: str,
    tag: str,
    eps: list[Any],
    scores: list[Any],
) -> dict[str, float]:
  if not eps:
    return {}
  r_perf = statistics.fmean([s.reward for s in scores])
  em = statistics.fmean([s.components.get("em", 0.0) for s in scores])
  crit = statistics.fmean([float(e.critical_steps) for e in eps])
  tot = statistics.fmean([float(e.total_steps) for e in eps])
  wall = statistics.fmean([float(e.wall_time_s) for e in eps])
  return {
      f"eval/{prefix}{tag}_r_perf": r_perf,
      f"eval/{prefix}{tag}_em": em,
      f"eval/{prefix}{tag}_critical_steps": crit,
      f"eval/{prefix}{tag}_total_steps": tot,
      f"eval/{prefix}{tag}_speedup": tot / max(crit, 1e-6),
      f"eval/{prefix}{tag}_wall_s": wall,
      f"eval/{prefix}{tag}_mean_children": statistics.fmean(
          [float(len(e.children)) for e in eps]
      ),
      f"eval/{prefix}{tag}_parallel_rate": statistics.fmean(
          [1.0 if len(e.children) >= 2 else 0.0 for e in eps]
      ),
  }


def summarize_by_suite(
    tag: str,
    eps: list[Any],
    scores: list[Any],
    eval_priv: dict[str, Any],
) -> dict[str, float]:
  out = summarize_policy_subset("", tag, eps, scores)
  for kind, pref in SUITE_PREFIXES:
    idx = [
        i for i, e in enumerate(eps) if eval_priv[e.task_id].get("kind") == kind
    ]
    if idx:
      out.update(
          summarize_policy_subset(
              pref, tag, [eps[i] for i in idx], [scores[i] for i in idx]
          )
      )
  return out


def attach_base_deltas(
    metrics: dict[str, float],
    base_metrics: dict[str, float],
    *,
    tag: str = "parl",
) -> dict[str, float]:
  merged = dict(metrics)
  for pref in ("", "wide_", "mh_", "bc_"):
    pk = f"eval/{pref}{tag}_r_perf"
    bk = f"eval/{pref}base_serial_r_perf"
    if pk in merged and bk in base_metrics:
      merged[bk] = base_metrics[bk]
      merged[f"eval/{pref}delta_vs_base_serial"] = merged[pk] - base_metrics[bk]
      bc_key = f"eval/{pref}base_serial_critical_steps"
      if bc_key in base_metrics:
        merged[bc_key] = base_metrics[bc_key]
  return merged


async def run_policy_eval(
    *,
    step_id: int,
    eval_pub: list[dict[str, Any]],
    eval_priv: dict[str, Any],
    recipe: Recipe,
    snapshots: SnapshotBundle,
    adapters: dict[str, RoleSamplerAdapter],
    wiki_env: parl_wf.HttpWikiSearchEnv,
    tag: str,
    max_concurrent_episodes: int = 48,
    max_concurrent_sampler_calls: int = 96,
) -> dict[str, float]:
  plan = repeat_tasks(eval_pub, attempts=1, base_seed=step_id * 100)
  step = StepContext(
      step_id=step_id,
      tasks=eval_pub,
      private_examples=eval_priv,
      recipe=recipe,
      snapshots=snapshots,
      adapters=adapters,
      learners={},
      env_factory=lambda _: wiki_env,
  )
  eps = await step.collect(
      plan,
      mode=CollectionMode.STRICT_BUNDLE,
      max_concurrent_episodes=max_concurrent_episodes,
      max_concurrent_sampler_calls=max_concurrent_sampler_calls,
  )
  scores = await step.grade(eps)
  return summarize_by_suite(tag, eps, scores, eval_priv)


async def evaluate_frozen_base_serial(
    *,
    eval_pub: list[dict[str, Any]],
    eval_priv: dict[str, Any],
    recipe: Recipe,
    snapshots: SnapshotBundle,
    adapters: dict[str, RoleSamplerAdapter],
    wiki_env: parl_wf.HttpWikiSearchEnv,
    max_critical_steps: int,
    enable_thinking: bool,
) -> dict[str, float]:
  """Evaluates strictly frozen base model (`worker` pool) as decoupled Serial baseline."""
  serial_recipe = dataclasses.replace(
      recipe,
      workflow=lambda task, ctx: parl_wf.serial_baseline_workflow(
          task,
          ctx,
          max_steps=max_critical_steps,
          role="worker",
          enable_thinking=enable_thinking,
      ),
      budget=CriticalStepsBudget(limit=max_critical_steps),
  )
  return await run_policy_eval(
      step_id=0,
      eval_pub=eval_pub,
      eval_priv=eval_priv,
      recipe=serial_recipe,
      snapshots=snapshots,
      adapters=adapters,
      wiki_env=wiki_env,
      tag="base_serial",
  )


async def run_pareto_sweep(
    *,
    budgets: tuple[int, ...],
    step_id: int,
    eval_pub: list[dict[str, Any]],
    eval_priv: dict[str, Any],
    recipe: Recipe,
    snapshots: SnapshotBundle,
    adapters: dict[str, RoleSamplerAdapter],
    wiki_env: parl_wf.HttpWikiSearchEnv,
    tag: str,
    enable_thinking: bool,
    include_base_serial: bool = True,
) -> list[dict[str, Any]]:
  """Sweeps CriticalSteps budgets to measure Accuracy vs. CriticalSteps Pareto frontier."""
  rows: list[dict[str, Any]] = []
  for limit in budgets:
    b_recipe = dataclasses.replace(recipe, budget=CriticalStepsBudget(limit=limit))
    pol = await run_policy_eval(
        step_id=step_id,
        eval_pub=eval_pub,
        eval_priv=eval_priv,
        recipe=b_recipe,
        snapshots=snapshots,
        adapters=adapters,
        wiki_env=wiki_env,
        tag=tag,
    )
    entry: dict[str, Any] = {"budget": int(limit), **pol}
    if include_base_serial:
      base = await evaluate_frozen_base_serial(
          eval_pub=eval_pub,
          eval_priv=eval_priv,
          recipe=recipe,
          snapshots=snapshots,
          adapters=adapters,
          wiki_env=wiki_env,
          max_critical_steps=limit,
          enable_thinking=enable_thinking,
      )
      entry.update(base)
    rows.append(entry)
  return rows
