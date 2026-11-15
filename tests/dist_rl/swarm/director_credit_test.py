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

"""Director credit, frozen-policy isolation, and fixed PPO target contracts."""

import asyncio
import dataclasses
import functools

import numpy as np
import pytest
from tunix.dist_rl.orchestrator import swarm_program
from tunix.dist_rl.swarm import director_credit
from tunix.dist_rl.swarm import policy
from tunix.dist_rl.swarm import trace
from tunix.rl import algorithm_config

PLAN = policy.LearningPlan(
    learner_policy_id="director",
    bindings=(
        policy.PolicyBinding(policy_id="director", trainable=True),
        policy.PolicyBinding(policy_id="worker", trainable=False, version=0),
    ),
)


def episode(index, reward, *, tokens=2, condition="none", cost=0.0, version=3):
  calls = tuple(
      trace.CallTrace(
          call_id=f"e{index}/{role}",
          role=role,
          policy_id=role,
          policy_version=version if role == "director" else 0,
          prompt_ids=(1, 2),
          completion_ids=(3,) * size,
          rollout_logprobs=(-0.5,) * size,
          loss_mask=(1,) * size,
          text="sample",
      )
      for role, size in (("director", tokens), ("worker", 10))
  )
  return trace.SwarmEpisode(
      task_id="task",
      episode_id=f"e{index}",
      calls=calls,
      reward=reward,
      final_text="answer",
      scaffold_condition=condition,
      worker_cost=cost,
  )


def build(episodes, **kwargs):
  return director_credit.build_director_credit(
      episodes,
      plan=PLAN,
      config=director_credit.DirectorCreditConfig(**kwargs),
  )


def test_paper_kl_stride_rotates_per_condition_and_weights_only_reference():
  episodes = [
      episode(i, float((i // 2) % 2), condition="on" if i % 2 else "off")
      for i in range(8)
  ]
  config = director_credit.DirectorCreditConfig(
      reference_stride=4, low_signal_abs_advantage=0
  )
  batch = director_credit.build_director_credit(
      episodes, plan=PLAN, config=config, reference_step=2
  )
  assert [float(row.reference_kl_weights[0, 0]) for row in batch.payloads] == [
      0,
      0,
      0,
      0,
      4,
      4,
      0,
      0,
  ]
  next_batch = director_credit.build_director_credit(
      episodes, plan=PLAN, config=config, reference_step=3
  )
  assert [
      float(row.reference_kl_weights[0, 0]) for row in next_batch.payloads
  ] == [0, 0, 4, 4, 0, 0, 0, 0]
  np.testing.assert_array_equal(batch.advantages, next_batch.advantages)


def test_director_advantage_uses_only_active_tokens_and_sample_variance():
  episodes = [episode(0, 0, tokens=2), episode(1, 1, tokens=4)]
  batch = build(episodes, low_signal_abs_advantage=0)
  # A token-expanded independent oracle catches episode/role normalization and
  # accidental inclusion of the much longer frozen Worker responses.
  centered = np.array([-0.5] * 2 + [0.5] * 4)
  expected = (centered - centered.mean()) / np.sqrt(centered.var(ddof=1) + 1e-8)
  actual = np.concatenate([p.advantages.ravel() for p in batch.payloads])
  np.testing.assert_allclose(actual, expected, atol=1e-7)
  assert {p.metadata["policy_id"] for p in batch.payloads} == {"director"}
  assert len(batch.payloads) == 2
  assert sum(p.completion_mask.sum() for p in batch.payloads) == 6


def test_condition_groups_singletons_cost_and_masked_context():
  episodes = [
      episode(0, 0),
      episode(1, 1, cost=20),
      episode(2, 1, condition="on"),
  ]
  first = episodes[0].calls[0]
  episodes[0] = dataclasses.replace(
      episodes[0],
      calls=(
          dataclasses.replace(first, loss_mask=(0, 1)),
          episodes[0].calls[1],
      ),
  )
  batch = build(episodes, cost_lambda=0.01, low_signal_abs_advantage=0)
  assert batch.rewards == (0.0, 0.8)
  assert len(batch.payloads) == 2  # singleton scaffold condition excluded
  assert batch.payloads[0].advantages[0, 0] == 0
  assert batch.payloads[0].completion_attention_mask[0, 0] == 1
  assert batch.payloads[0].metadata["centered_advantage"] == -0.4


def test_constant_rewards_and_filter_fallback_are_exact_zero():
  batch = build([episode(0, 0.1), episode(1, 0.1)], low_signal_keep_fraction=0)
  assert batch.advantages == (0.0, 0.0)
  assert all(np.count_nonzero(p.advantages) == 0 for p in batch.payloads)


def test_low_signal_selection_precedes_whitening():
  episodes = [
      episode(i, reward, tokens=i + 1) for i, reward in enumerate([0, 0, 1, 1])
  ]
  episodes.extend(
      dataclasses.replace(episode(i, 1), task_id="constant")
      for i in range(4, 8)
  )
  batch = build(episodes, low_signal_keep_fraction=0)
  values = np.concatenate([p.advantages.ravel() for p in batch.payloads])
  assert len(batch.payloads) == 4
  np.testing.assert_allclose(values.mean(), 0, atol=1e-7)
  np.testing.assert_allclose(values.var(ddof=1), 1, atol=1e-7)


def test_policy_versions_are_checked_per_policy_and_per_round():
  episodes = [episode(0, 0), episode(1, 1)]
  PLAN.validate(episodes, learner_version=3)
  with pytest.raises(ValueError, match="stale learner"):
    PLAN.validate(episodes, learner_version=2)
  changed_worker = dataclasses.replace(episodes[1].calls[1], policy_version=1)
  with pytest.raises(ValueError, match="frozen policy"):
    PLAN.validate([
        episodes[0],
        dataclasses.replace(
            episodes[1], calls=(episodes[1].calls[0], changed_worker)
        ),
    ])
  with pytest.raises(ValueError, match="mix learner"):
    PLAN.validate([episodes[0], episode(1, 1, version=4)])
  with pytest.raises(ValueError, match="same policy"):
    dataclasses.replace(
        episodes[0],
        calls=(
            episodes[0].calls[0],
            dataclasses.replace(
                episodes[0].calls[0], call_id="different", policy_version=4
            ),
        ),
    )


class RecordingEngine:

  def __init__(self):
    self.version = 3
    self.updates = 0
    self.events = []
    self.payloads = []

  def configure(self, algo, config):
    self.config = config

  async def prepare(self):
    return self.version

  async def score(self, payload):
    self.events.append("score")
    return np.full_like(payload.old_per_token_logps, -1 + self.updates * 0.1)

  async def reference(self, payload):
    self.events.append("reference")
    return np.full_like(payload.old_per_token_logps, -1.5)

  async def train(self, payload, *, apply_optimizer):
    self.events.append("train")
    self.payloads.append(payload)
    self.updates += int(apply_optimizer)

  async def publish(self):
    self.events.append("publish")
    self.version += 1
    return self.version


@pytest.mark.parametrize("constant_reward", [False, True])
def test_two_policy_updates_keep_old_tis_reference_and_frozen_version(
    constant_reward,
):
  engine = RecordingEngine()
  episodes = [
      episode(i, 1 if constant_reward else i % 2, tokens=i + 1)
      for i in range(4)
  ]

  async def collect(version, step):
    assert version == 3 and step == 0
    return episodes

  program = swarm_program.SwarmRLProgram(
      collect,
      swarm_program.SwarmConfig(
          group_size=4,
          micro_batch_size=2,
          max_prompt_length=2,
          max_response_length=4,
          max_logprob_error=1,
          sampler_is="token",
      ),
      learning_plan=PLAN,
      credit_builder=functools.partial(build, low_signal_keep_fraction=1),
      algo_config=algorithm_config.GRPOConfig(
          beta=0.005, epsilon=0.2, epsilon_high=0.28, epsilon_c=None
      ),
      reference_scorer=engine.reference,
      precompute_sampler_is=True,
      num_policy_minibatches=2,
  )
  asyncio.run(program.run_async(engine, num_steps=1))
  assert engine.updates == 2
  assert program.policy_version == 4
  assert engine.events == [
      "score",
      "score",
      "reference",
      "reference",
      "train",
      "train",
      "publish",
  ]
  for payload in engine.payloads:
    active = payload.completion_mask > 0
    np.testing.assert_allclose(payload.old_per_token_logps[active], -1)
    np.testing.assert_allclose(payload.ref_per_token_logps[active], -1.5)
    np.testing.assert_allclose(
        payload.sampler_is_weights[active], np.exp(-0.5), atol=1e-7
    )
    assert payload.sampler_agreement_applied
  assert program.history[0]["optimizer_steps"] == 2


def test_reference_kl_fails_closed_without_reference_scorer():
  async def collect(version, step):
    raise AssertionError("Should fail before collection")

  program = swarm_program.SwarmRLProgram(
      collect,
      swarm_program.SwarmConfig(),
      algo_config=algorithm_config.GRPOConfig(beta=0.005),
  )
  with pytest.raises(ValueError, match="reference scorer"):
    program.run(RecordingEngine(), num_steps=1)


@pytest.mark.parametrize(
    "fault", ["frozen", "unknown", "duplicate", "stale", "tokens", "mask"]
)
def test_custom_credit_cannot_train_unowned_or_modified_tokens(fault):
  episodes = [episode(0, 0), episode(1, 1)]

  async def collect(version, step):
    return episodes

  def unsafe_builder(episodes):
    credited = build(episodes)
    row = credited.payloads[0]
    metadata = dict(row.metadata)
    if fault == "frozen":
      metadata.update(policy_id="worker", call_id=episodes[0].calls[1].call_id)
    elif fault == "unknown":
      metadata["call_id"] = "never-sampled"
    elif fault == "stale":
      metadata["policy_version"] -= 1
    row = dataclasses.replace(row, metadata=metadata)
    if fault == "tokens":
      row = dataclasses.replace(row, completion_ids=row.completion_ids + 1)
    elif fault == "mask":
      row = dataclasses.replace(
          row, completion_mask=np.full_like(row.completion_mask, 2)
      )
    rows = (row, row) if fault == "duplicate" else (row, credited.payloads[1])
    return dataclasses.replace(credited, payloads=rows)

  engine = RecordingEngine()
  program = swarm_program.SwarmRLProgram(
      collect,
      swarm_program.SwarmConfig(group_size=2),
      learning_plan=PLAN,
      credit_builder=unsafe_builder,
  )
  with pytest.raises(ValueError, match="learner's recorded calls"):
    program.run(engine, num_steps=1)
  assert not engine.events and engine.updates == 0
