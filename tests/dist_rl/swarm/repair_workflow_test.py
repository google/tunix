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

"""Genuine within-team concurrency, anytime grading and shared credit."""

import asyncio
import json

import numpy as np
from examples.swarm_repair import cpu_fixture
from examples.swarm_repair import environment
from examples.swarm_repair import train
from examples.swarm_repair import workflow
from tunix.dist_rl.rollout import sampler as sampler_lib


def config():
  return json.loads((train.ROOT / "configs/cpu_smoke.json").read_text())


def test_fixed_communicating_agents_overlap_and_share_one_policy():
  sampler = cpu_fixture.TinySampler(cpu_fixture.make_policy())
  task = environment.tasks("train", count=1)[0]
  result = asyncio.run(
      workflow.RepairWorkflow(config()).run(
          task,
          sampler,
          sampler.encode,
          policy_version=7,
          seed=42,
          episode_id="parallel",
          mode="fixed",
      )
  )
  assert result.max_concurrent_calls == 2
  assert {c.policy_version for c in result.episode.calls} == {7}
  assert {c.policy_id for c in result.episode.calls} == {"shared_repair"}
  assert len({c.call_id for c in result.episode.calls}) == len(
      result.episode.calls
  )
  assert any(event.kind == "message" for event in result.events)


def test_anytime_commit_survives_deadline_and_pending_tokens_are_excluded():
  task = environment.tasks("train", count=1)[0]
  actions = [
      json.dumps({"action": "write", "path": path, "module": module})
      for path, module in task.oracle_modules().items()
  ]
  actions.append(json.dumps({"action": "submit", "continue": True}))
  observed, cancelled = [], []

  class Backend:

    async def sample(self, request):
      if not actions:
        try:
          await asyncio.sleep(10)
        except asyncio.CancelledError:
          cancelled.append(True)
          raise
      return sampler_lib.SamplingResponse(
          request_id=request.request_id,
          prompt_token_ids=np.asarray(request.prompt),
          token_ids=np.array([1, 0]),
          logprobs=np.array([-1.0, -1.0]),
          text=actions.pop(0),
          finish_reason="stop",
      )

  def encode(prompt):
    observed.append(prompt)
    return [1, 0]

  cfg = {**config(), "deadline_seconds": 0.03}
  result = asyncio.run(
      workflow.RepairWorkflow(cfg).run(
          task,
          Backend(),
          encode,
          policy_version=0,
          seed=42,
          episode_id="deadline",
          mode="single",
      )
  )
  assert result.status == "deadline" and result.episode.reward == 1
  assert len(result.episode.calls) == 3 and result.requested_calls == 4
  assert result.cancelled_agents == 1 and cancelled == [True]
  assert all("hidden_inputs" not in text for text in observed)


def test_team_credit_drops_zero_variance_groups_and_broadcasts_flat_advantage():
  import dataclasses

  sampler = cpu_fixture.TinySampler(cpu_fixture.make_policy())
  task = environment.tasks("train", count=1)[0]
  original = asyncio.run(
      workflow.RepairWorkflow(config()).run(
          task,
          sampler,
          sampler.encode,
          policy_version=0,
          seed=42,
          episode_id="original",
          mode="fixed",
      )
  ).episode
  episodes = [
      dataclasses.replace(
          original,
          episode_id=f"e{i}",
          task_id="variable" if i < 2 else "constant",
          calls=tuple(
              dataclasses.replace(c, call_id=f"e{i}/{j}")
              for j, c in enumerate(original.calls)
          ),
          reward=float(i != 0),
      )
      for i in range(4)
  ]
  batch = train.team_credit(episodes)
  assert {p.metadata["task_id"] for p in batch.payloads} == {"variable"}
  by_episode = {}
  for p in batch.payloads:
    by_episode.setdefault(p.metadata["episode_id"], set()).add(
        float(p.advantages[0, 0])
    )
  assert all(len(values) == 1 for values in by_episode.values())
  assert next(iter(by_episode["e0"])) < 0 < next(iter(by_episode["e1"]))
