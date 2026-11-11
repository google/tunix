# Copyright 2026 Nanyang Technological University (NTU), Singapore
# Copyright 2026 Dr. MAS Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains the RewardMathFn class, which evaluates mathematical answers
and assigns rewards based on their correctness. It utilizes a language model to
validate answers when necessary.
"""

from examples.swarm_math.drmas_math import extract_answer, grade_answer_mathd, grade_answer_sympy
from dataclasses import dataclass

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"


@dataclass
class RewardConfig:
    apply_format_reward: bool = False

    # General reward constants
    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    format_error_reward: float = 0.0
    unk_error_reward: float = 0.0

    # Bonus reward for calling tools.
    toolcall_bonus: float = 0.0


class RewardMathFn:
    """
    Reward function for evaluating mathematical answers.

    This class implements the RewardFunction protocol to process the input and determine
    the reward based on the correctness of the provided answer compared to the ground truth.
    """

    def __init__(self, config: RewardConfig):
        self.config = config

    def __call__(self, task_info: dict, action: str):
        """
        Calculate the reward for a math task based on the agent's action.

        Args:
            task_info: Dictionary containing problem, data_source, problem_type, and ground_truth
            action: The agent's response/solution

        Returns:
            RewardOutput: The calculated reward with correctness information
        """
        # Extract information from task_info
        problem = task_info.get("problem", "")
        model_response = action
        is_correct = False

        # Handle None or empty response
        if model_response is None or model_response == "":
            # print("DEBUG: Empty or None response")
            return self.config.format_error_reward, is_correct

        # Extract solution.
        if THOUGHT_DELIMITER_END in model_response:
            model_solution = model_response.split(THOUGHT_DELIMITER_END)[1]
        else:
            if self.config.apply_format_reward:
                return self.config.format_error_reward, is_correct
            model_solution = model_response

        model_answer = extract_answer(model_solution)
        if model_answer is None:
            return self.config.format_error_reward, is_correct

        # Process the ground truth(s)
        ground_truths = task_info.get("ground_truth", None)
        if ground_truths is None:
            return self.config.unk_error_reward, is_correct

        # Convert single answer to list for uniform processing
        if isinstance(ground_truths, str | float | int):
            ground_truths = [ground_truths]

        # Process each ground truth
        processed_ground_truths = []
        for truth in ground_truths:
            truth = str(truth)
            if "\\boxed" in truth:
                processed_truth = extract_answer(truth)
                if processed_truth is not None:
                    processed_ground_truths.append(processed_truth)
            else:
                processed_ground_truths.append(truth)

        if not processed_ground_truths:
            return self.config.unk_error_reward, is_correct

        # Check against all possible correct answers
        for ground_truth in processed_ground_truths:
            _is_correct = grade_answer_mathd(model_answer, ground_truth) or grade_answer_sympy(model_answer, ground_truth)
            if _is_correct:
                # Apply tool call bonus if applicable and answer is correct
                reward = self.config.correct_reward
                if task_info.get("has_toolcall", False):
                    reward += self.config.toolcall_bonus

                is_correct=True
                return reward, is_correct

        return self.config.incorrect_reward, is_correct

if __name__ == "__main__":
    reward = RewardMathFn(RewardConfig())
    task_info = {
        "data_source": "",
        "problem": ("Let $P(x)=x^{4}+2 x^{3}-13 x^{2}-14 x+24$ be a polynomial with roots $r_{1}, r_{2}, r_{3}, r_{4}$. Let $Q$ be the quartic polynomial with roots $r_{1}^{2}, r_{2}^{2}, r_{3}^{2}, r_{4}^{2}$, such that the coefficient of the $x^{4}$ term of $Q$ is 1. Simplify the quotient $Q\\left(x^{2}\\right) / P(x)$, leaving your answer in terms of $x$. (You may assume that $x$ is not equal to any of $\\left.r_{1}, r_{2}, r_{3}, r_{4}\\right)$."),
        "problem_type": "math",
        "ground_truth": ["10", "$x^{4}-2 x^{3}-13 x^{2}+14 x+24$"],
        "has_toolcall": True,
    }
    action = "<think>...</think>\nThe answer is \\boxed{24 + 14*x + (-13)*x^2 - 2*x^3 + x^4}."

    output, is_correct = reward(task_info, action)
    print(output)
    print(is_correct)
