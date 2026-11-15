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

"""Define a custom collaboration independently of its policy update ownership.

Run ``python -m examples.r2orch_math.custom_swarm`` for a tiny fixture demo.
Replace the runtime factory with native/Qwen samplers for a real model. The
returned SwarmEpisode is directly consumable by SwarmRLProgram and a credit
builder. This example is not an R²Orch reproduction or a training benchmark.
"""

import asyncio
import dataclasses
import json

from examples.r2orch_math import cpu_fixture
from tunix.dist_rl.rollout import sampler
from tunix.dist_rl.swarm import api
from tunix.dist_rl.swarm import policy
from tunix.dist_rl.swarm import runtime


async def solve_and_review(team, question):
  drafts = await asyncio.gather(
      team.generate("solver", question + "\nSolve independently."),
      team.generate("solver", question + "\nUse a second method."),
  )
  review = await team.generate(
      "reviewer",
      question + "\nReview and finalize:\n" + "\n".join(c.text for c in drafts),
  )
  return review.text


def make_swarm():
  return api.SwarmSpec(
      roles={"solver": "worker", "reviewer": "director"},
      workflow=solve_and_review,
      learning_plan=policy.LearningPlan(
          learner_policy_id="director",
          bindings=(
              policy.PolicyBinding(policy_id="director", trainable=True),
              policy.PolicyBinding(
                  policy_id="worker", trainable=False, version=0
              ),
          ),
      ),
      max_calls=3,
      max_concurrent_calls=2,
  )


async def demo():
  samplers = {
      key: cpu_fixture.TinySampler(cpu_fixture.make_policy())
      for key in ("director", "worker")
  }

  def factory(policy_id, version, namespace):
    return runtime.RecordingRuntime(
        samplers[policy_id],
        cpu_fixture.encode,
        sampler.SamplingParams(max_tokens=2, temperature=1.0, seed=42),
        policy_id=policy_id,
        policy_version=version,
        episode_id=namespace,
        max_prompt_length=8,
    )

  return await make_swarm().run(
      task_id="demo",
      episode_id="demo/0",
      prompt="Compute 1 + 1.",
      learner_version=0,
      runtime_factory=factory,
      grade=lambda final: float(r"\boxed{2}" in final),
  )


if __name__ == "__main__":
  print(json.dumps(dataclasses.asdict(asyncio.run(demo())), indent=2))
