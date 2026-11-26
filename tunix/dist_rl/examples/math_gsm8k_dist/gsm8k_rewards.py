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

"""Orchestrator-side reward function for the distributed GSM8K example.

List this module in `reward_functions` and set
`agentic_grpo_config.reward_mode: reward_fn` to score completions on the
orchestrator instead of in `GSM8KEnv`. Don't list it with `reward_mode: env`:
there, reward functions are added to the environment's reward, which already
is this one. `HyperParameters.obtain_reward_fn` uses every public function
defined here as a reward function, so the module defines only `reward_fn`.
"""

from collections.abc import Sequence
import logging

from tunix.dist_rl.examples.math_gsm8k_dist import gsm8k


def reward_fn(
    prompts: Sequence[str],
    completions: Sequence[str],
    answer: Sequence[str],
    **kwargs: object,
) -> list[float]:
  """Scores completions against the gold answers.

  This is the reward `GSM8KEnv` gives: 1.0 for a correct answer in the expected
  format, 0.5 for a correct answer only, 0.1 for the format only, and 0.0
  otherwise. Its per-function metric is `rewards/train/reward_fn`.

  Args:
    prompts: Prompt text of each completion.
    completions: Model completions.
    answer: Gold answers, from each record's `metadata.answer`.
    **kwargs: The other reward inputs and record metadata (unused).

  Returns:
    One reward per completion.
  """
  del kwargs  # Unused.
  rewards: list[float] = []
  for prompt, completion, gold_answer in zip(
      prompts, completions, answer, strict=True
  ):
    reward, info = gsm8k.score_gsm8k_completion(completion, gold_answer)
    # Logged with `cluster_config.debug: true`.
    logging.debug(
        "GSM8K reward %.1f (extracted answer %s, gold answer %s) for prompt:\n"
        "%s\n--- Response ---\n%s\n--- End of response ---",
        reward,
        info["extracted_answer"],
        info["gold_answer"],
        prompt,
        completion,
    )
    rewards.append(reward)
  return rewards
