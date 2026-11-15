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

"""Source-grounded RSI estimators, statistical decisions and resume invariants."""

import json

import numpy as np
import pytest
from examples.r2orch_math import rsi


@pytest.mark.parametrize(
    "h,v,expected",
    [
        ([0, 0, 0], [0, 0], -0.2),
        ([1, 1, 1], [1, 1], -0.2),
        ([1, 0, 0], [1, 1], 0.96),
        ([0, 0, 0], [1, 0], 0.5),
        ([0, 0, 0], [1, 1], 0.375),
        ([], [], None),
    ],
)
def test_recursive_learnability_cases(h, v, expected):
  actual = rsi.recursive_learnability(h, v)
  assert (
      actual is None if expected is None else actual == pytest.approx(expected)
  )


def test_compute_factor_only_scales_positive_reward_and_missing_arm_fallback():
  assert rsi.recursive_learnability([0], [0], time_factor=99) == -0.2
  assert rsi.recursive_learnability([0, 0, 0], [1, 0], time_factor=2) == 1
  assert rsi.recursive_learnability([], [], siblings=[1, 0]) == 1
  with pytest.raises(ValueError, match="binary"):
    rsi.recursive_learnability([0.9], [1])


def test_pair_advantages_center_every_pair_and_share_one_batch_scale():
  values = rsi.proposer_pair_advantages([[1, 0], [0, 0.5], [1, 1]])
  np.testing.assert_allclose(values.sum(axis=1), 0, atol=1e-12)
  assert np.mean(values**2) == pytest.approx(1)
  assert values[0, 0] == pytest.approx(-2 * values[1, 0])
  np.testing.assert_array_equal(values[2], [0, 0])
  np.testing.assert_array_equal(
      rsi.proposer_pair_advantages([[0, 0]]), [[0, 0]]
  )


def test_arm_assignments_are_balanced_and_repeatable():
  a = rsi.arm_assignment(4, "aime/test")
  assert a == rsi.arm_assignment(4, "aime/test")
  assert a.count("hidden") == 3
  assert a.count("visible") == 2
  assert len({rsi.arm_assignment(i, "aime/test") for i in range(10)}) > 1


def test_candidate_sources_are_distinct_and_training_only():
  values = rsi.candidate_list(
      range(4), range(2, 10), allowed_training_ids=range(10)
  )
  assert values == tuple(range(8))
  with pytest.raises(ValueError, match="non-training"):
    rsi.candidate_list(
        ["test"], ["a"], allowed_training_ids=["a"], per_source=1
    )


def test_pair_sampling_uses_restricted_second_distribution():
  pair = rsi.sample_candidate_pair([0, 0, 0], seed=42)
  assert pair == rsi.sample_candidate_pair([0, 0, 0], seed=42)
  assert pair.indices[0] != pair.indices[1]
  np.testing.assert_allclose(pair.logprobs, [-np.log(3), -np.log(2)])


def test_ledger_admission_retirement_audit_and_windowed_reinstatement(tmp_path):
  config = rsi.LedgerConfig(
      min_tasks=3, window_cycles=2, bootstrap_draws=64, audit_share=1.0
  )
  ledger = rsi.SkillLedger(config, seed=10)
  useful = ledger.register("useful", version=1, content="Check the parity.")
  redundant = ledger.register(
      "redundant", version=1, content="Repeat the question."
  )
  for i in range(3):
    ledger.record(
        cycle=0,
        task_id=str(i),
        skill_key=useful,
        hidden=[0, 0, 0],
        visible=[1, 1],
    )
    ledger.record(
        cycle=0,
        task_id=str(i),
        skill_key=redundant,
        hidden=[1, 0, 0],
        visible=[1, 0, 0],
    )
  choices = ledger.decide(cycle=0)
  assert choices[useful] == "admit"
  assert choices[redundant] == "retire"
  assert ledger.choose([redundant], cycle=1, task_id="audit") == (
      redundant,
      True,
  )
  for i in range(3):
    ledger.record(
        cycle=3,
        task_id=str(i),
        skill_key=redundant,
        hidden=[0, 0, 0],
        visible=[1, 1],
    )
  assert ledger.decide(cycle=3)[redundant] == "reinstate"
  assert ledger.skills[redundant]["status"] == "active"
  path = tmp_path / "ledger.json"
  ledger.save(path)
  restored = rsi.SkillLedger.load(path)
  assert restored.records == ledger.records
  assert restored.skills == ledger.skills
  assert restored.summary(redundant, cycle=3) == ledger.summary(
      redundant, cycle=3
  )
  assert restored.choose(
      [useful, redundant], cycle=4, task_id="task"
  ) == ledger.choose([useful, redundant], cycle=4, task_id="task")


def test_ledger_replay_and_content_integrity(tmp_path):
  ledger = rsi.SkillLedger()
  key = ledger.register("skill", version=1, content="Check units.")
  row = dict(cycle=0, task_id="q", skill_key=key, hidden=[0], visible=[1])
  ledger.record(**row)
  ledger.record(**row)
  assert len(ledger.records) == 1
  with pytest.raises(ValueError, match="Conflicting"):
    ledger.record(**{**row, "visible": [0]})
  new_key = ledger.register(
      "skill", version=2, content="Check units and signs."
  )
  assert new_key != key
  assert ledger.summary(new_key, cycle=0)["n"] == 0
  ledger.decide(cycle=0)
  with pytest.raises(ValueError, match="committed"):
    ledger.record(**{**row, "task_id": "late"})
  path = tmp_path / "ledger.json"
  ledger.save(path)
  state = json.loads(path.read_text())
  state["skills"][key]["content"] = "Changed after checkpoint."
  path.write_text(json.dumps(state))
  with pytest.raises(ValueError, match="identity"):
    rsi.SkillLedger.load(path)


def test_bootstrap_mixed_evidence_has_uncertainty_and_reproducible_selection():
  ledger = rsi.SkillLedger(
      rsi.LedgerConfig(min_tasks=2, bootstrap_draws=128), seed=19
  )
  key = ledger.register("k", version=1, content="Try a second approach.")
  ledger.record(cycle=0, task_id="a", skill_key=key, hidden=[0], visible=[1])
  ledger.record(cycle=0, task_id="b", skill_key=key, hidden=[1], visible=[0])
  summary = ledger.summary(key, cycle=0)
  assert summary["mean"] == 0
  assert 0 < summary["positive_effect"] < 1
  assert summary["q_admit"] < 0
  assert ledger.decide(cycle=0)[key] == "undecided"


def test_original_estimators_and_posterior_sampling_match():
  import itertools
  import os
  from examples.r2orch_math.source_adapter import load_source_module

  source = os.environ.get("TUNIX_R2ORCH_SOURCE")
  if not source:
    pytest.skip("Set TUNIX_R2ORCH_SOURCE to the pinned original checkout.")
  rewards = load_source_module(source, "proposer_reward")
  original = load_source_module(source, "skill_ledger")
  for outcomes, prior, factor in itertools.product(
      itertools.product((0, 1), repeat=5), (0.1, 0.5, 0.9), (0.5, 1, 2)
  ):
    hidden, visible = outcomes[:3], outcomes[3:]
    expected = rewards.arms_bell_reward(
        hidden, visible, prior=prior, time_factor=factor
    )
    assert rsi.recursive_learnability(
        hidden, visible, prior=prior, time_factor=factor
    ) == pytest.approx(expected["value"], abs=1e-12)
  ledger = rsi.SkillLedger(seed=41)
  keys = [
      ledger.register(
          f"skill-{i}", version=1, content=f"Specific reusable check {i}"
      )
      for i in range(3)
  ]
  for i in range(8):
    for k in keys[:2]:
      ledger.record(
          cycle=0,
          task_id=f"task-{i}",
          skill_key=k,
          hidden=[i % 2, 0, 1],
          visible=[1, (i + 1) % 2],
      )
  for k in keys:
    expected = original.effect_posterior(
        ledger.contrasts(k, cycle=1),
        draws=ledger.config.bootstrap_draws,
        rng=original.ledger_rng(41, 1, "posterior", k),
        admit_quantile=ledger.config.admit_quantile,
    )
    actual = ledger.summary(k, cycle=1)
    if expected.n:
      assert actual["positive_effect"] == pytest.approx(
          expected.E_delta_plus, abs=1e-12
      )
      assert actual["q_admit"] == pytest.approx(expected.q_admit, abs=1e-12)
  for i in range(20):
    task = f"held-training-task-{i}"
    _, draws = original.thompson_pick(
        {k: ledger.contrasts(k, cycle=1) for k in keys},
        original.ledger_rng(41, 1, "thompson", task),
        prior_sd=ledger.config.prior_sd,
    )
    # Match source selection within retrieval after all active cards were drawn.
    subset = keys[1:]
    assert ledger.choose(subset, cycle=1, task_id=task) == (
        min(subset, key=lambda k: (draws[k], k)),
        False,
    )
  ledger.skills[keys[0]]["status"] = "retired"
  for i in range(20):
    task = f"audit-{i}"
    expected = original.audit_choice(
        {keys[0]: True},
        ledger.config.audit_share,
        original.ledger_rng(41, 1, "audit", task),
    )
    chosen, audit = ledger.choose(keys, cycle=1, task_id=task)
    assert audit == (expected is not None)
    if audit:
      assert chosen == expected
