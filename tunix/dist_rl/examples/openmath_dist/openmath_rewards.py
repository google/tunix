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

"""Orchestrator-side reward function for the distributed OpenMath example.

List this module in `reward_functions` and set
`agentic_grpo_config.reward_mode: reward_fn` to score completions on the
orchestrator instead of in `OpenMathEnv`. Don't list it with `reward_mode: env`:
there, reward functions are added to the environment's reward, which already
is this one. `HyperParameters.obtain_reward_fn` uses every public function
defined here as a reward function, so the module defines only `reward_fn`.
"""

from collections.abc import Sequence

from absl import logging
from tunix.dist_rl.examples.openmath_dist import openmath


def reward_fn(
    prompts: Sequence[str],
    completions: Sequence[str],
    answer: Sequence[str],
    **kwargs: object,
) -> Sequence[float]:
  """Scores completions against the gold answers (`+0.1` format + `+1.0` math)."""
  del kwargs
  rewards: list[float] = []
  for prompt, completion, gold_answer in zip(
      prompts, completions, answer, strict=True
  ):
    reward, info = openmath.score_openmath_completion(completion, gold_answer)
    logging.debug(
        "OpenMath reward %.1f (extracted=%s, gold=%s) for prompt:\n%s\n"
        "--- Response ---\n%s\n--- End of response ---",
        reward,
        info["extracted_answer"],
        info["gold_answer"],
        prompt,
        completion,
    )
    rewards.append(reward)
  return rewards
