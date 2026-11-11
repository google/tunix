# Copyright 2026 Google LLC
# Copyright 2026 Nanyang Technological University (NTU), Singapore
# Copyright 2026 Dr. MAS Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Async port of the pinned DrMAS Solver–Verifier workflow; see THIRD_PARTY.md."""

from collections.abc import Callable
import dataclasses
import re

from examples.swarm_math import drmas_math
from examples.swarm_math import drmas_reward
from examples.swarm_math import prompts
from tunix.dist_rl.swarm import runtime as runtime_lib
from tunix.dist_rl.swarm import trace

SOLVER = "Solver Agent"
VERIFIER = "Verifier Agent"


@dataclasses.dataclass(frozen=True)
class MathTask:
  task_id: str
  problem: str
  answer: str


def grade(text: str, answer: str) -> bool:
  """Use the source environment reward, including its </think> handling."""
  _, correct = drmas_reward.RewardMathFn(drmas_reward.RewardConfig())(
      {"ground_truth": answer}, text
  )
  return correct


def project_verdict(text: str) -> tuple[str, bool, bool]:
  """Preserve upstream projection, validity and approval as separate decisions.

  Upstream strips whitespace inside the first tag pair; it does not require
  the closing tag to follow the opening tag. Missing/unknown verdicts default
  to approval, while format validity is used separately for the reward penalty.
  Training retains the ORIGINAL sampled tokens even if displayed text changes.
  """
  start, end = text.find("<verify>"), text.find("</verify>")
  valid = start != -1 and end != -1
  projected = text
  if valid:
    projected = text[: start + 8] + text[start + 8 : end].strip() + text[end:]
    valid = re.search(r"[\u4e00-\u9fff]", text) is None
  approved = (
      "<verify>approve</verify>" in projected
      or "<verify>reject</verify>" not in projected
  )
  return projected, valid, approved


class SolverVerifierWorkflow:

  def __init__(
      self,
      *,
      max_solver_turns: int = 2,
      grader: Callable[[str, str], bool] = grade,
  ):
    if max_solver_turns < 1:
      raise ValueError("max_solver_turns must be positive.")
    self.max_solver_turns = max_solver_turns
    self.grader = grader

  async def run(
      self, task: MathTask, runtime: runtime_lib.RecordingRuntime
  ) -> trace.SwarmEpisode:
    context, final = "", ""
    calls = []
    for turn in range(self.max_solver_turns):
      solver = await runtime.generate(
          SOLVER,
          prompts.SOLVER_PROMPT.format(
              env_prompt=task.problem,
              team_context=context,
          ),
      )
      solver = dataclasses.replace(
          solver,
          valid_action=drmas_math.extract_answer(solver.text) is not None,
      )
      calls.append(solver)
      final = solver.text
      context += f'\nThe output of "{SOLVER}": {solver.text}\n'
      if turn == self.max_solver_turns - 1:
        break
      verifier = await runtime.generate(
          VERIFIER,
          prompts.VERIFIER_PROMPT.format(
              env_prompt=task.problem,
              team_context=context,
          ),
      )
      projected, valid, approved = project_verdict(verifier.text)
      calls.append(dataclasses.replace(verifier, valid_action=valid))
      context += f'\nThe output of "{VERIFIER}": {projected}\n'
      if approved:
        break
    return trace.SwarmEpisode(
        task_id=task.task_id,
        episode_id=runtime.episode_id,
        calls=tuple(calls),
        reward=float(self.grader(final, task.answer)),
        final_text=final,
    )


def chat_encoder(tokenizer, max_prompt_length: int):
  """Pinned workflow's system prompt and non-thinking Qwen chat template.

  Match upstream's middle truncation BEFORE sampling, preserving the exact
  resulting context in the trace. Never truncate a recorded training trace.
  """

  def encode(prompt):
    ids = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "You are a helpful and harmless assistant.",
            },
            {"role": "user", "content": prompt},
        ],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if len(ids) > max_prompt_length:
      left = max_prompt_length // 2
      ids = ids[:left] + ids[-(max_prompt_length - left) :]
    return ids

  return encode
