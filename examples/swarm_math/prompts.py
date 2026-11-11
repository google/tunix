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


"""Unchanged prompts from the pinned DrMAS source; see THIRD_PARTY.md."""

SOLVER_PROMPT = """
# Task Introduction
{env_prompt}

# Your Teammates' Outputs
{team_context}

# Your Role
You are a "Solver Agent". Your job is to carefully reason through the math problem step by step and derive the correct answer. When reasoning, consider your teammates' outputs (if any) as auxiliary context.\x20

You should give the final answer within \\boxed{{}}.
"""

VERIFIER_PROMPT = """
# Task Introduction
{env_prompt}

# Your Teammates' Outputs
{team_context}

# Your Role
You are a "Verifier Agent". Your responsibility is to critically review the most recent solution provided by the "Solver Agent". Check each reasoning step, formula, and conclusion for accuracy, completeness, and logical consistency.
At the end of your output, you MUST provide your verdict within <verify> </verify> using exactly one of:
(1) <verify>approve</verify> if all steps and the final answer are correct.
(2) <verify>reject</verify> if you detect any issue.
"""
