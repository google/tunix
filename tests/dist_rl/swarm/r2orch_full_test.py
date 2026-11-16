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

"""Unit and CPU integration tests for Full R2Orch (Self-Play + Ledger + PATS)."""

import asyncio
import json
import random

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from examples.swarm_r2orch import canvas as canvas_lib
from examples.swarm_r2orch import cpu_fixture
from examples.swarm_r2orch import pats
from examples.swarm_r2orch import proposer
from examples.swarm_r2orch import scoring
from examples.swarm_r2orch import skill_ledger
from examples.swarm_r2orch import train as r2orch_train
from examples.swarm_r2orch import workflow
from tunix.dist_rl.swarm import trace


def test_canvas_same_layer_consider_relation_runs_peer_revision_wave():
  canvas = canvas_lib.GraphCanvas(max_agents=4, relay_max_chars=200)
  seen_inputs: list[tuple[str, tuple[str, ...]]] = []

  async def fake_worker(node, upstream):
    up_ids = tuple(a.agent_id for a in upstream)
    seen_inputs.append((node.agent_id, up_ids))
    return canvas_lib.WorkerArtifact(
        agent_id=node.agent_id,
        answer=f"{node.agent_id}:peers={','.join(up_ids)}",
        summary=f"sum({node.agent_id}:{','.join(up_ids)})",
        token_in=60,
        token_out=30,
    )

  async def _exercise():
    r1 = await canvas.step(
        '{"action":"add_agent","role":"SolverA","objective":"Solve algebraically","layer":0,"result_scope":"task_result"}',
        fake_worker,
    )
    r2 = await canvas.step(
        '{"action":"add_agent","role":"SolverB","objective":"Check numerically","layer":0,"result_scope":"task_result"}',
        fake_worker,
    )
    assert r1.accepted and r2.accepted
    peer = await canvas.step(
        '{"action":"consider_relation","source":"agent_1","target":"agent_2"}',
        fake_worker,
    )
    assert peer.accepted and peer.executed_agents == ("agent_1", "agent_2")
    assert canvas.had_parallel_wave
    fin = await canvas.step(
        '{"action":"finish","target":"agent_1"}', fake_worker
    )
    assert fin.accepted and fin.finished
    assert canvas.submitted_answer() == "agent_1:peers=agent_2"

  asyncio.run(_exercise())
  assert ("agent_1", ("agent_2",)) in seen_inputs
  assert ("agent_2", ("agent_1",)) in seen_inputs


def test_skill_ledger_bayesian_withdrawal_admission_and_reinstatement():
  rng = random.Random(7)
  post_zero = skill_ledger.effect_posterior([0.0] * 12, draws=500, rng=rng)
  assert skill_ledger.decide(post_zero, min_tasks=12, epsilon=0.02) == "retire"

  post_pos = skill_ledger.effect_posterior([0.4, 0.5] * 6, draws=500, rng=rng)
  assert skill_ledger.decide(post_pos, min_tasks=12, epsilon=0.02) == "admit"

  seed_card = skill_ledger.SkillCard(
      skill_id="seed_math_1",
      name="Math peer",
      description="Desc",
      trigger="Math",
      plan="Plan",
      pitfall="Pitfall",
      constraint="Constraint",
      kind="orchestration",
      task_types=("math",),
      status="active",
      is_seed=True,
  )
  trial_card = skill_ledger.SkillCard(
      skill_id="trial_math_1",
      name="Verifier check",
      description="Desc",
      trigger="Math",
      plan="Add verifier",
      pitfall="Early finish",
      constraint="Connect nodes",
      kind="verification",
      task_types=("math",),
      status="trial",
      is_seed=False,
  )
  scaffold = skill_ledger.SkillScaffold(
      {"math": [seed_card, trial_card]},
      min_tasks=3,
      epsilon=0.02,
      audit_share=1.0,
  )
  ck_seed = skill_ledger.card_key(seed_card)
  ck_trial = skill_ledger.card_key(trial_card)
  for cyc in (1, 2, 3):
    scaffold.records.append(
        {"cycle": cyc, "task_id": f"t{cyc}", "scope": "math", "card_key": ck_seed, "z": 0.0}
    )
    scaffold.records.append(
        {"cycle": cyc, "task_id": f"u{cyc}", "scope": "math", "card_key": ck_trial, "z": 0.5}
    )
  ev1 = scaffold.apply_ledger_decisions(3)
  assert ev1["retired"] == 1 and ev1["admitted"] == 1
  assert scaffold.cards["math"]["seed_math_1"].status == "retired"
  assert scaffold.cards["math"]["trial_math_1"].status == "active"

  # Windowed audit evidence showing positive gain reinstates the retired card.
  scaffold.window_cycles = 3
  for cyc in (4, 5, 6):
    scaffold.records.append(
        {"cycle": cyc, "task_id": f"a{cyc}", "scope": "math", "card_key": ck_seed, "z": 0.5}
    )
  ev2 = scaffold.apply_ledger_decisions(6)
  assert ev2["reinstated"] == 1
  assert scaffold.cards["math"]["seed_math_1"].status == "active"


def test_proposer_arms_bell_reward_and_gap_clipped_pair_loss():
  end_r = proposer.arms_bell_reward([1.0, 1.0, 1.0], [1.0, 1.0], prior=0.5)
  assert end_r["case"] == "endpoint" and end_r["value"] == pytest.approx(-0.2)

  hid_r = proposer.arms_bell_reward([1.0, 0.0, 0.0], [1.0, 1.0], prior=0.5)
  assert hid_r["case"] == "hidden_bell" and hid_r["value"] > 0.5

  ast_r = proposer.arms_bell_reward(
      [0.0, 0.0, 0.0], [1.0, 0.0], prior=0.5, gamma=0.5
  )
  assert ast_r["case"] == "assisted_bell" and ast_r["value"] == pytest.approx(
      0.5
  )

  model = cpu_fixture.make_proposer_policy()
  loss_fn = proposer.make_proposer_loss_fn(kl_beta=1.0, gap_clip=0.05)
  base_payload = {
      "tokens": jnp.asarray([[5, 0, 6]], dtype=jnp.int32),
      "mask": jnp.asarray([[True, True, True]]),
      "label_ids": jnp.asarray([[1, 2, 3, 4]], dtype=jnp.int32),
      "label_mask": jnp.asarray([[True, True, True, True]]),
      "pair_a": jnp.asarray([0], dtype=jnp.int32),
      "pair_b": jnp.asarray([3], dtype=jnp.int32),
      "advantages": jnp.asarray([1.0], dtype=jnp.float32),
      "pair_valid": jnp.asarray([True]),
  }
  # Initial gap for (0 vs 3) against a uniform reference is 1.1 - 0.8 = 0.3 > 0.05,
  # so the policy-gradient term clips while reference KL remains differentiable.
  uniform_ref = jnp.full((1, 4), -np.log(4.0), dtype=jnp.float32)
  loss_val, grads = nnx.value_and_grad(loss_fn)(
      model, **base_payload, ref_logps=uniform_ref
  )
  assert float(loss_val) > 0.0
  assert float(jnp.linalg.norm(grads["logits"][...])) > 0.0


def test_pats_scrubber_strips_private_gold_fields():
  secret_task = scoring.R2OrchTask(
      task_id="mbpp/secret",
      dataset="mbppplus",
      verifier="mbppplus_tests",
      prompt="Implement foo(x) matching public assert.",
      answers=("SECRET_GOLD_42",),
      code_spec={"plus_test": "assert foo(999) == 'CANARY_PRIVATE_TEST'"},
  )
  plan = skill_ledger.TaskArmPlan(
      task_id="mbpp/secret",
      scope="code_function",
      arms=("hidden", "visible"),
      hidden_context="",
      visible_context="",
      arm_card=None,
      arm_card_key="",
      is_audit=False,
  )
  dummy_call = trace.CallTrace(
      call_id="ep0/0",
      role="Graph Director",
      policy_id="dir",
      policy_version=0,
      prompt_ids=(1, 2),
      completion_ids=(6,),
      rollout_logprobs=(-0.5,),
      loss_mask=(1,),
      text='{"action":"add_agent"}',
      trainable=True,
  )
  ep = trace.SwarmEpisode(
      task_id="mbpp/secret",
      episode_id="ep0",
      calls=(dummy_call,),
      reward=0.0,
      final_text="",
  )
  out = workflow.R2OrchOutcome(
      episode=ep,
      dataset="mbppplus",
      correctness=0.0,
      worker_cost=0.1,
      finished=False,
      director_turns=2,
      worker_calls=1,
      num_nodes=1,
      max_layer=0,
      parallel_wave=False,
      task_id="mbpp/secret",
      skill_arm="hidden",
      action_log=('{"action":"add_agent"} -> Added agent_1.',),
  )
  scrubbed = json.dumps(pats.scrub_public_group(1, secret_task, plan, [out]))
  assert "SECRET_GOLD_42" not in scrubbed
  assert "CANARY_PRIVATE_TEST" not in scrubbed
  assert pats.validate_card_fields({"name": "x", "plan": r"\boxed{1}"}) is None


def test_full_r2orch_cpu_two_cycle_training_and_resume(tmp_path):
  cfg = json.loads(
      (r2orch_train.ROOT / "configs/cpu_full_smoke.json").read_text(
          encoding="utf-8"
      )
  )
  train_tasks = r2orch_train.load_tasks(
      r2orch_train.ROOT / "data/cpu_train_full.jsonl"
  )
  eval_tasks = r2orch_train.load_tasks(
      r2orch_train.ROOT / "data/cpu_eval.jsonl"
  )
  out_dir = tmp_path / "full_run"
  report = r2orch_train.run("cpu", cfg, train_tasks, eval_tasks, out_dir)
  assert report["director_norm_delta_l2"] > 0.0
  assert report["proposer_norm_delta_l2"] > 0.0
  assert report["frozen_worker_max_norm_drift"] == 0.0
  assert report["skill_ledger_final"]["distilled_total"] >= 1
  assert len(report["history"]) == 2

  resumed_cfg = {**cfg, "num_steps": 3}
  resumed_out = tmp_path / "resumed_run"
  resumed_report = r2orch_train.run(
      "cpu",
      resumed_cfg,
      train_tasks,
      eval_tasks,
      resumed_out,
      checkpoint_dir=str(out_dir / "checkpoints"),
      resume=True,
  )
  assert resumed_report["restored_step"] == 2
  assert len(resumed_report["history"]) == 3
  assert resumed_report["frozen_worker_max_norm_drift"] == 0.0
