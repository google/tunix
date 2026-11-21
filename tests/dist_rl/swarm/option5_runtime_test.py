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
"""Acceptance tests for Option 5 unified Swarm RL runtime, StepContext, & L3."""

from __future__ import annotations

import asyncio
from pathlib import Path
import tempfile
import unittest

import numpy as np
from tunix.dist_rl.rollout import sampler as sampler_lib
from tunix.dist_rl.swarm import AttemptPlan
from tunix.dist_rl.swarm import AttemptSpec
from tunix.dist_rl.swarm import CollectionMode
from tunix.dist_rl.swarm import CriticalStepsBudget
from tunix.dist_rl.swarm import EpisodeRecord
from tunix.dist_rl.swarm import EpisodeRuntime
from tunix.dist_rl.swarm import LearnerSpec
from tunix.dist_rl.swarm import ManifestStore
from tunix.dist_rl.swarm import ObjectiveBatch
from tunix.dist_rl.swarm import PolicySpec
from tunix.dist_rl.swarm import Recipe
from tunix.dist_rl.swarm import RoleSamplerAdapter
from tunix.dist_rl.swarm import ScoreRecord
from tunix.dist_rl.swarm import ServingFidelity
from tunix.dist_rl.swarm import ServingFidelityMode
from tunix.dist_rl.swarm import SnapshotBundle
from tunix.dist_rl.swarm import SnapshotRef
from tunix.dist_rl.swarm import StepContext
from tunix.dist_rl.swarm import StepResult
from tunix.dist_rl.swarm import WallClockBudget
from tunix.dist_rl.swarm import agent_env_workflow
from tunix.dist_rl.swarm import compact_actions
from tunix.dist_rl.swarm import group_grpo
from tunix.dist_rl.swarm import parl_grpo
from tunix.dist_rl.swarm import validate_ownership
from tunix.dist_rl.swarm import verify_serving_fidelity


def _ascii_encode(text: str) -> list[int]:
  return [ord(ch) % 250 + 1 for ch in text]


class _FakeSampler:
  """Deterministic async sampler supporting attestation & scripted responses."""

  def __init__(
      self,
      reply_fn,
      *,
      served_digest: str | None = None,
      served_adapter_id: str | None = None,
      delay_s: float = 0.0,
  ) -> None:
    self._reply_fn = reply_fn
    self._served_digest = served_digest
    self._served_adapter_id = served_adapter_id
    self._delay_s = delay_s
    self.calls: list[sampler_lib.SamplingRequest] = []

  async def sample(
      self, request: sampler_lib.SamplingRequest
  ) -> sampler_lib.SamplingResponse:
    if self._delay_s > 0:
      await asyncio.sleep(self._delay_s)
    self.calls.append(request)
    text = self._reply_fn(request)
    toks = np.asarray(_ascii_encode(text), dtype=np.int32)
    lps = (
        np.full((len(toks),), -0.25, dtype=np.float32)
        if request.sampling_params.return_logprobs
        else None
    )
    meta = {}
    if self._served_digest is not None:
      meta["served_digest"] = self._served_digest
    if self._served_adapter_id is not None:
      meta["served_adapter_id"] = self._served_adapter_id
    return sampler_lib.SamplingResponse(
        request_id=request.request_id,
        prompt_token_ids=request.prompt,
        token_ids=toks,
        text=text,
        finish_reason="stop",
        logprobs=lps,
        metadata=meta,
    )


class _RecordingTrainer:
  """Minimal staged/commit learner stub for StepContext tests."""

  def __init__(self, name: str) -> None:
    self.name = name
    self.staged: list[ObjectiveBatch] = []
    self.committed_versions: list[int] = []

  def stage_update(self, batch: ObjectiveBatch) -> dict[str, float]:
    self.staged.append(batch)
    return {f"{self.name}/rows": float(len(batch.examples))}

  def commit_staged(self, version: int) -> None:
    self.committed_versions.append(version)


class _TwoTurnEnv:
  """Public environment returning append-only tool observations."""

  def __init__(self) -> None:
    self._turn = 0
    self._transcript = ""

  async def reset(self, task: dict[str, str]) -> str:
    self._transcript = f"Q:{task['question']}"
    return self._transcript

  async def step(self, action_text: str) -> tuple[str, bool, dict[str, bool]]:
    self._turn += 1
    self._transcript += action_text
    if self._turn == 1:
      self._transcript += "|OBS:42|"
      return (self._transcript, False, {"valid": True})
    return (action_text, True, {"valid": True})


class Option5RuntimeTest(unittest.TestCase):

  def test_ownership_and_serving_fidelity_gate(self) -> None:
    base = SnapshotRef(model_id="qwen", revision=0, digest="sha-base")
    with self.assertRaises(ValueError):
      LearnerSpec.lora(
          "director",
          base=base,
          adapter="dir",
          fidelity=ServingFidelity(mode=ServingFidelityMode.MERGED_BF16),
      )
    learners = {
        "director": LearnerSpec.lora("director", base=base, adapter="dir"),
        "proposer": LearnerSpec.lora("proposer", base=base, adapter="prop"),
    }
    bindings = {
        "director": PolicySpec.follow("director"),
        "helper": PolicySpec.follow("director"),
        "worker": PolicySpec.fixed(base),
    }
    validate_ownership(learners, bindings)

    # Duplicate adapter collision rejected.
    with self.assertRaises(ValueError):
      validate_ownership(
          {
              "a": LearnerSpec.lora("a", base=base, adapter="dup"),
              "b": LearnerSpec.lora("b", base=base, adapter="dup"),
          },
          {},
      )

    # L3 ServingFidelity catches bf16 87.5% mismatch and passes exact fp32.
    fp32_rep = verify_serving_fidelity(
        learner_top1_ids=[1, 2, 3, 4],
        sampler_top1_ids=[1, 2, 3, 4],
        learner_logprobs=[-0.5, -0.4, -0.3, -0.2],
        sampler_logprobs=[-0.5, -0.400001, -0.3, -0.2],
        fidelity=ServingFidelity(mode=ServingFidelityMode.FP32_NATIVE),
    )
    self.assertTrue(fp32_rep.passed)
    with self.assertRaises(RuntimeError):
      verify_serving_fidelity(
          learner_top1_ids=[1, 2, 3, 4, 5, 6, 7, 8],
          sampler_top1_ids=[1, 2, 3, 9, 5, 6, 7, 8],
          learner_logprobs=[-0.5] * 8,
          sampler_logprobs=[-0.5] * 8,
          fidelity=ServingFidelity(mode=ServingFidelityMode.FP32_NATIVE),
      )

  def test_single_agent_auto_tito_compaction_and_attestation(self) -> None:
    snap = SnapshotRef(model_id="solver", revision=0, digest="dig-0")
    learners = {"solver": LearnerSpec.full("solver", snap)}
    bindings = {"solver": PolicySpec.follow("solver")}
    sampler = _FakeSampler(
        lambda req: "CALL" if len(req.prompt) < 10 else "ANS42",
        served_digest="dig-0",
    )
    adapters = {
        "solver": RoleSamplerAdapter(
            sampler=sampler,
            encode=_ascii_encode,
            sampling_params=sampler_lib.SamplingParams(max_tokens=16),
            max_prompt_length=256,
        )
    }
    bundle = SnapshotBundle(
        step_id=0,
        role_snapshots={"solver": snap},
        learner_snapshots={"solver": snap},
    )

    async def _run() -> EpisodeRecord:
      rt = EpisodeRuntime.create_root(
          episode_id="ep0",
          task_id="t0",
          group_id="g0",
          slot_index=0,
          snapshots=bundle,
          bindings=bindings,
          adapters=adapters,
          sampler_sem=asyncio.Semaphore(2),
          env=_TwoTurnEnv(),
      )
      out = await agent_env_workflow({"question": "6*7"}, rt)
      return await rt.finalize(out)

    ep = asyncio.run(_run())
    self.assertEqual(len(ep.actions), 2)
    compacted = compact_actions(ep.actions, roles=("solver",), policy_id="solver")
    # Two append-only turns auto-compact into ONE TITO row with obs masked to 0!
    self.assertEqual(len(compacted), 1)
    seq = compacted[0]
    self.assertEqual(len(seq.action_ids), 2)
    self.assertEqual(sum(seq.loss_mask), len("CALL") + len("ANS42"))
    self.assertEqual(
        len(seq.completion_ids), len("CALL") + len("|OBS:42|") + len("ANS42")
    )
    self.assertTrue(all(x == 1 for x in seq.attention_mask))

    # Wrong attested digest must raise immediately.
    bad_adapters = {
        "solver": RoleSamplerAdapter(
            sampler=_FakeSampler(lambda _: "BAD", served_digest="wrong-dig"),
            encode=_ascii_encode,
            sampling_params=sampler_lib.SamplingParams(max_tokens=8),
            max_prompt_length=128,
        )
    }

    async def _run_bad() -> None:
      rt = EpisodeRuntime.create_root(
          episode_id="ep1",
          task_id="t0",
          group_id="g0",
          slot_index=0,
          snapshots=bundle,
          bindings=bindings,
          adapters=bad_adapters,
          sampler_sem=asyncio.Semaphore(1),
      )
      await rt.generate("solver", "hello")

    with self.assertRaises(RuntimeError):
      asyncio.run(_run_bad())

  def test_parl_gate_seven_critical_steps_and_bounded_pool(self) -> None:
    dir_snap = SnapshotRef(model_id="director", revision=0, digest="dir-0")
    wrk_snap = SnapshotRef(model_id="worker", revision=0, digest="wrk-0")
    learners = {"director": LearnerSpec.full("director", dir_snap)}
    bindings = {
        "director": PolicySpec.follow("director"),
        "worker": PolicySpec.fixed(wrk_snap),
    }

    async def solve_subtask(job: dict[str, int], ctx: EpisodeRuntime) -> str:
      out = ""
      for k in range(job["steps"]):
        act = await ctx.generate("worker", f"sub-{job['id']}-{k}")
        out = act.text
      return out

    async def parl_wf(task: dict[str, str], ctx: EpisodeRuntime) -> str:
      decision = await ctx.generate("director", f"PLAN:{task['q']}")
      if decision.text == "SPAWN":
        reports = await ctx.parallel(
            [{"id": 0, "steps": 2}, {"id": 1, "steps": 5}],
            solve_subtask,
            parent=decision,
        )
        final = await ctx.generate("director", f"SYNTH:{','.join(reports)}")
        return final.text
      return decision.text

    slot_counter = {"n": 0}

    def dir_reply(req: sampler_lib.SamplingRequest) -> str:
      if req.request_id.endswith("/a0"):
        slot_counter["n"] += 1
        return "SPAWN" if slot_counter["n"] % 2 == 1 else "SERIAL_FAIL"
      return "OK_FINAL"

    adapters = {
        "director": RoleSamplerAdapter(
            sampler=_FakeSampler(dir_reply, served_digest="dir-0"),
            encode=_ascii_encode,
            sampling_params=sampler_lib.SamplingParams(max_tokens=16),
            max_prompt_length=256,
        ),
        "worker": RoleSamplerAdapter(
            sampler=_FakeSampler(lambda _: "SUB_OK", served_digest="wrk-0"),
            encode=_ascii_encode,
            sampling_params=sampler_lib.SamplingParams(max_tokens=16),
            max_prompt_length=256,
        ),
    }

    def grader(ep: EpisodeRecord, priv: dict[str, str]) -> ScoreRecord:
      ok = 1.0 if ep.public_output == priv["ans"] else 0.0
      return ScoreRecord(
          episode_id=ep.episode_id,
          task_id=ep.task_id,
          reward=ok,
          components={"r_perf": ok},
      )

    recipe = Recipe(
        workflow=parl_wf,
        learners=learners,
        bindings=bindings,
        grade=grader,
        train_step=parl_grpo(
            target="director",
            roles=("director",),
            attempts=2,
            mode=CollectionMode.STRICT_BUNDLE,
            max_policy_lag=0,
        ),
        budget=CriticalStepsBudget(limit=8),
    )
    bundle = SnapshotBundle(
        step_id=0,
        role_snapshots={"director": dir_snap, "worker": wrk_snap},
        learner_snapshots={"director": dir_snap},
    )
    trainer = _RecordingTrainer("director")
    step = StepContext(
        step_id=0,
        tasks=[{"id": "q1", "public": {"q": "multi"}}],
        private_examples={"q1": {"ans": "OK_FINAL"}},
        recipe=recipe,
        snapshots=bundle,
        adapters=adapters,
        learners={"director": trainer},
    )
    # Bounded sampler_sem=1 inside collect still makes progress because
    # awaiting parent releases its sampler slot during ctx.parallel!
    plan = AttemptPlan(
        attempts=(
            AttemptSpec(group_id="q1:default", slot_index=0, task_id="q1", public_task={"q": "multi"}, seed=0),
            AttemptSpec(group_id="q1:default", slot_index=1, task_id="q1", public_task={"q": "multi"}, seed=1),
        )
    )
    eps = asyncio.run(step.collect(plan, max_concurrent_episodes=2, max_concurrent_sampler_calls=1))
    spawned_ep = next(e for e in eps if e.children)
    # Exact Gate 4 assertion: 1 dispatch + max(2, 5) + 1 final == 7 critical steps!
    self.assertEqual(spawned_ep.critical_steps, 7)
    self.assertEqual(spawned_ep.total_steps, 9)
    self.assertEqual([c.status for c in spawned_ep.children], ["completed", "completed"])

    # Full parl_grpo step yields nonzero Director rows and ZERO Worker rows.
    res = asyncio.run(recipe.train_step(step))
    self.assertEqual(len(trainer.staged), 1)
    self.assertTrue(all(ex["role"] == "director" for ex in trainer.staged[0].examples))
    self.assertGreater(res.metrics["parl/n_director_rows"], 0.0)
    self.assertGreater(res.metrics["parl/parallel_speedup"], 1.0)

  def test_step_record_crash_replay_and_dual_learner_manifest(self) -> None:
    base = SnapshotRef(model_id="qwen", revision=0, digest="sha-base")
    dir_snap = SnapshotRef(model_id="qwen", revision=0, digest="dir-0", base_digest="sha-base", adapter_id="dir")
    prop_snap = SnapshotRef(model_id="qwen", revision=0, digest="prop-0", base_digest="sha-base", adapter_id="prop")
    learners = {
        "director": LearnerSpec.lora("director", base=base, adapter="dir"),
        "proposer": LearnerSpec.lora("proposer", base=base, adapter="prop"),
    }
    bindings = {"director": PolicySpec.follow("director")}
    recipe = Recipe(
        workflow=lambda task, ctx: ctx.generate("director", task["q"]),
        learners=learners,
        bindings=bindings,
        grade=lambda ep, _: ScoreRecord(episode_id=ep.episode_id, task_id=ep.task_id, reward=1.0),
        train_step=group_grpo(target="director", roles=("director",), attempts=1),
        budget=WallClockBudget(deadline_s=30.0, max_total_calls=10),
    )
    bundle = SnapshotBundle(
        step_id=0,
        role_snapshots={"director": dir_snap},
        learner_snapshots={"director": dir_snap, "proposer": prop_snap},
    )
    with tempfile.TemporaryDirectory() as tmp:
      store = ManifestStore(Path(tmp))
      t_dir, t_prop = _RecordingTrainer("director"), _RecordingTrainer("proposer")
      calls = {"n": 0}

      def _sample_selection() -> list[str]:
        calls["n"] += 1
        return [f"task-{calls['n']}"]

      # Crash before commit: record is persisted in journal.
      step_a = StepContext(
          step_id=0,
          tasks=[{"id": "t0", "public": {"q": "hi"}}],
          private_examples={"t0": {}},
          recipe=recipe,
          snapshots=bundle,
          adapters={},
          learners={"director": t_dir, "proposer": t_prop},
          manifest_store=store,
      )
      sel_a = asyncio.run(step_a.record("selection", _sample_selection))
      self.assertEqual(sel_a, ["task-1"])
      self.assertIsNone(store.load_committed())

      # Replay after crash: `selection` replays cached value without re-calling!
      step_b = StepContext(
          step_id=0,
          tasks=[{"id": "t0", "public": {"q": "hi"}}],
          private_examples={"t0": {}},
          recipe=recipe,
          snapshots=bundle,
          adapters={},
          learners={"director": t_dir, "proposer": t_prop},
          manifest_store=store,
      )
      sel_b = asyncio.run(step_b.record("selection", _sample_selection))
      self.assertEqual(sel_b, ["task-1"])
      self.assertEqual(calls["n"], 1)

      asyncio.run(step_b.update("proposer", ObjectiveBatch(target="proposer", objective_kind="candidate_pair_pg", examples=({"id": 0, "policy_version": 0},))))
      asyncio.run(step_b.update("director", ObjectiveBatch(target="director", objective_kind="token_grpo", examples=({"id": 1, "policy_version": 0},))))
      manifest = asyncio.run(step_b.commit(StepResult(state={"cycle": 1}, metrics={"ok": 1.0})))
      self.assertEqual(manifest["learner_versions"], {"director": 1, "proposer": 1})
      self.assertEqual(store.load_committed()["recipe_state"], {"cycle": 1})
      self.assertEqual(t_dir.committed_versions, [1])
      self.assertEqual(t_prop.committed_versions, [1])

  def test_async_window_mode_tracks_version_lag(self) -> None:
    snap0 = SnapshotRef(model_id="actor", revision=0, digest="v0")
    snap1 = SnapshotRef(model_id="actor", revision=1, digest="v1")
    recipe = Recipe(
        workflow=lambda task, ctx: ctx.generate("actor", task["q"]),
        learners={"actor": LearnerSpec.full("actor", snap0)},
        bindings={"actor": PolicySpec.follow("actor")},
        grade=lambda ep, _: ScoreRecord(episode_id=ep.episode_id, task_id=ep.task_id, reward=1.0),
        train_step=group_grpo(target="actor", roles=("actor",), attempts=2, mode=CollectionMode.ASYNC_WINDOW, max_policy_lag=1),
    )
    tick = {"i": 0}

    def _provider() -> SnapshotBundle:
      tick["i"] += 1
      s = snap0 if tick["i"] == 1 else snap1
      return SnapshotBundle(step_id=1, role_snapshots={"actor": s}, learner_snapshots={"actor": s})

    adapters = {
        "actor": RoleSamplerAdapter(
            sampler=_FakeSampler(lambda _: "GO"),
            encode=_ascii_encode,
            sampling_params=sampler_lib.SamplingParams(max_tokens=8),
            max_prompt_length=64,
        )
    }
    trainer = _RecordingTrainer("actor")
    step = StepContext(
        step_id=1,
        tasks=[{"id": "t0", "public": {"q": "x"}}],
        private_examples={"t0": {}},
        recipe=recipe,
        snapshots=SnapshotBundle(step_id=1, role_snapshots={"actor": snap1}, learner_snapshots={"actor": snap1}),
        adapters=adapters,
        learners={"actor": trainer},
        snapshot_provider=_provider,
    )
    plan = AttemptPlan(
        attempts=(
            AttemptSpec(group_id="g", slot_index=0, task_id="t0", public_task={"q": "x"}, seed=0),
            AttemptSpec(group_id="g", slot_index=1, task_id="t0", public_task={"q": "x"}, seed=1),
        )
    )
    eps = asyncio.run(step.collect(plan, mode=CollectionMode.ASYNC_WINDOW))
    revs = sorted(ep.actions[0].served_snapshot.revision for ep in eps)
    self.assertEqual(revs, [0, 1])
    # Lag=1 accepted when max_policy_lag=1, rejected when max_policy_lag=0.
    asyncio.run(step.update("actor", ObjectiveBatch(target="actor", objective_kind="token_grpo", examples=({"policy_version": 0},), max_policy_lag=1)))
    with self.assertRaises(ValueError):
      asyncio.run(step.update("actor", ObjectiveBatch(target="actor", objective_kind="token_grpo", examples=({"policy_version": 0},), max_policy_lag=0)))


if __name__ == "__main__":
  unittest.main()
