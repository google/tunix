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
"""Unit tests for Option 5 PARL (Kimi K2.5 §3) recipe workflow & grading."""

from __future__ import annotations

import asyncio
import unittest

import numpy as np
from examples.swarm_parl import tasks as parl_tasks
from examples.swarm_parl import workflow as parl_wf
from tunix.dist_rl.rollout import sampler as sampler_lib
from tunix.dist_rl.swarm import CollectionMode
from tunix.dist_rl.swarm import CriticalStepsBudget
from tunix.dist_rl.swarm import LearnerSpec
from tunix.dist_rl.swarm import PolicySpec
from tunix.dist_rl.swarm import Recipe
from tunix.dist_rl.swarm import RoleSamplerAdapter
from tunix.dist_rl.swarm import SnapshotBundle
from tunix.dist_rl.swarm import SnapshotRef
from tunix.dist_rl.swarm import StepContext
from tunix.dist_rl.swarm import parl_grpo


def _encode(text: str) -> list[int]:
  return [ord(ch) % 250 + 1 for ch in text[:1024]]


class _MockWikiEnv:
  async def search(self, query: str) -> str:
    if "France" in query:
      return "[1] France: Paris is the capital of France."
    if "Japan" in query:
      return "[1] Japan: Tokyo is the capital of Japan."
    return "[1] No match."


class _ScriptedSampler:
  def __init__(self, fn) -> None:
    self._fn = fn

  async def sample(
      self, request: sampler_lib.SamplingRequest
  ) -> sampler_lib.SamplingResponse:
    txt = self._fn(request)
    toks = np.asarray(_encode(txt), dtype=np.int32)
    lps = (
        np.full((len(toks),), -0.1, dtype=np.float32)
        if request.sampling_params.return_logprobs
        else None
    )
    return sampler_lib.SamplingResponse(
        request_id=request.request_id,
        prompt_token_ids=request.prompt,
        token_ids=toks,
        text=txt,
        finish_reason="stop",
        logprobs=lps,
    )


class _StubTrainer:
  def __init__(self) -> None:
    self.batches = []

  def stage_update(self, batch):
    self.batches.append(batch)
    return {"train/rows": float(len(batch.examples))}

  def commit_staged(self, version: int) -> None:
    del version


class ParlRecipeTest(unittest.TestCase):

  def test_parser_and_wide_wikiqa_grading(self) -> None:
    cmd = parl_wf.parse_director_action('<parallel>["q1", "q2", "q3"]</parallel>')
    self.assertTrue(cmd.is_valid and not cmd.is_final)
    self.assertEqual(cmd.jobs, ("q1", "q2", "q3"))
    ans = parl_wf.parse_director_action("<answer>(1) Paris\n(2) Tokyo</answer>")
    self.assertTrue(ans.is_valid and ans.is_final)
    self.assertIn("Paris", ans.answer)

    em, f1 = parl_tasks.best_qa_match("(1) Paris (2) Tokyo", ["Paris"])
    self.assertEqual(em, 1.0)
    self.assertGreaterEqual(f1, 1.0)

  def test_parl_end_to_end_parallel_speedup_and_zero_worker_rows(self) -> None:
    d_snap = SnapshotRef(model_id="qwen3-4b", revision=0, digest="d0")
    w_snap = SnapshotRef(model_id="qwen3-4b", revision=0, digest="w0")
    turn_by_ep: dict[str, int] = {}

    def dir_fn(req: sampler_lib.SamplingRequest) -> str:
      ep_prefix = req.request_id.rsplit("/a", 1)[0]
      t = turn_by_ep.get(ep_prefix, 0)
      turn_by_ep[ep_prefix] = t + 1
      if "/slot0/" in req.request_id:
        if t == 0:
          return '<parallel>["Capital of France", "Capital of Japan"]</parallel>'
        return "<answer>(1) Paris (2) Tokyo</answer>"
      return "<answer>(1) London (2) Berlin</answer>"

    worker_turn: dict[str, int] = {}

    def wrk_fn(req: sampler_lib.SamplingRequest) -> str:
      pid = req.metadata["participant_id"]
      t = worker_turn.get(pid, 0)
      worker_turn[pid] = t + 1
      if t == 0:
        return "<search>France</search>" if pid.endswith("/c0") else "<search>Japan</search>"
      return "<report>Paris</report>" if pid.endswith("/c0") else "<report>Tokyo</report>"

    adapters = {
        "director": RoleSamplerAdapter(
            sampler=_ScriptedSampler(dir_fn),
            encode=_encode,
            sampling_params=sampler_lib.SamplingParams(max_tokens=64, return_logprobs=True),
            max_prompt_length=2048,
        ),
        "worker": RoleSamplerAdapter(
            sampler=_ScriptedSampler(wrk_fn),
            encode=_encode,
            sampling_params=sampler_lib.SamplingParams(max_tokens=64, return_logprobs=False),
            max_prompt_length=2048,
        ),
    }
    recipe = Recipe(
        workflow=parl_wf.parl_episode_workflow,
        learners={"director": LearnerSpec.full("director", d_snap)},
        bindings={"director": PolicySpec.follow("director"), "worker": PolicySpec.fixed(w_snap)},
        grade=parl_tasks.grade_parl_episode,
        train_step=parl_grpo(
            target="director",
            roles=("director",),
            attempts=2,
            mode=CollectionMode.STRICT_BUNDLE,
            max_policy_lag=0,
        ),
        budget=CriticalStepsBudget(limit=8),
    )
    trainer = _StubTrainer()
    step = StepContext(
        step_id=0,
        tasks=[{"id": "w0", "public": {"question": "(1) France? (2) Japan?"}}],
        private_examples={"w0": {"kind": "wide_wikiqa", "sub_answers": [["Paris"], ["Tokyo"]]}},
        recipe=recipe,
        snapshots=SnapshotBundle(
            step_id=0,
            role_snapshots={"director": d_snap, "worker": w_snap},
            learner_snapshots={"director": d_snap},
        ),
        adapters=adapters,
        learners={"director": trainer},
        env_factory=lambda _: _MockWikiEnv(),
    )
    res = asyncio.run(recipe.train_step(step))
    self.assertEqual(len(trainer.batches), 1)
    # Only Director rows enter training; ZERO frozen worker rows!
    self.assertTrue(all(ex["role"] == "director" for ex in trainer.batches[0].examples))
    self.assertGreater(res.metrics["parl/parallel_speedup"], 1.0)
    self.assertGreater(res.metrics["parl/r_perf"], 0.4)

  def test_dual_mode_spawn_wave_think_stripping_and_direct_search(self) -> None:
    raw = (
        "<think>I should NOT trigger <search>fake</search> inside thinking. "
        "Instead spawn two specialized subagents.</think>\n"
        '<spawn_wave>[{"role": "Geo_Agent", "persona": "Find capitals", '
        '"task": "Capital of France"}, {"role": "Asia_Agent", "task": "Capital of Japan"}]</spawn_wave>'
    )
    cmd = parl_wf.parse_director_action(raw, max_wave_width=8, allow_direct_tools=True)
    self.assertTrue(cmd.is_valid and not cmd.is_final)
    self.assertEqual(cmd.action_kind, "spawn_wave")
    self.assertEqual(len(cmd.subagents), 2)
    self.assertEqual(cmd.subagents[0].role, "Geo_Agent")
    self.assertEqual(cmd.subagents[0].persona, "Find capitals")

    srch = parl_wf.parse_director_action(
        "<think>Quick direct lookup.</think><search>Inception director</search>",
        allow_direct_tools=True,
    )
    self.assertEqual(srch.action_kind, "search")
    self.assertEqual(srch.query, "Inception director")

    od = parl_wf.parse_director_action(
        "<open_doc>bc-42</open_doc>", allow_direct_tools=True
    )
    self.assertEqual(od.action_kind, "open_doc")
    self.assertEqual(od.docid, "bc-42")

    ep_like = type("Ep", (), {"episode_id": "e1", "task_id": "t1", "public_output": "(1) Paris\n(2) Tokyo"})()
    sc = parl_tasks.grade_parl_episode(ep_like, {"kind": "wide_wikiqa", "sub_answers": [["Paris"], ["Tokyo"]]})
    self.assertEqual(sc.components["em"], 1.0)
    self.assertEqual(sc.components["f1"], 1.0)


if __name__ == "__main__":
  unittest.main()
