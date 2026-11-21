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
"""Reusable episode runtime, contracts, serving fidelity, and credit for Swarm RL."""

from tunix.dist_rl.swarm.contracts import CriticalStepsBudget
from tunix.dist_rl.swarm.contracts import EpisodeRecord
from tunix.dist_rl.swarm.contracts import FidelityReport
from tunix.dist_rl.swarm.contracts import LearnerSpec
from tunix.dist_rl.swarm.contracts import ModelAction
from tunix.dist_rl.swarm.contracts import ObjectiveBatch
from tunix.dist_rl.swarm.contracts import PolicySpec
from tunix.dist_rl.swarm.contracts import Recipe
from tunix.dist_rl.swarm.contracts import ScoreRecord
from tunix.dist_rl.swarm.contracts import ServingFidelity
from tunix.dist_rl.swarm.contracts import ServingFidelityMode
from tunix.dist_rl.swarm.contracts import SnapshotBundle
from tunix.dist_rl.swarm.contracts import SnapshotRef
from tunix.dist_rl.swarm.contracts import StepResult
from tunix.dist_rl.swarm.contracts import WallClockBudget
from tunix.dist_rl.swarm.contracts import validate_ownership
from tunix.dist_rl.swarm.contracts import verify_serving_fidelity
from tunix.dist_rl.swarm.episode_runtime import EpisodeRuntime
from tunix.dist_rl.swarm.episode_runtime import RoleSamplerAdapter
from tunix.dist_rl.swarm.episode_runtime import agent_env_workflow
from tunix.dist_rl.swarm.episode_runtime import compact_actions
from tunix.dist_rl.swarm.step_context import AttemptPlan
from tunix.dist_rl.swarm.step_context import AttemptSpec
from tunix.dist_rl.swarm.step_context import CollectionMode
from tunix.dist_rl.swarm.step_context import ManifestStore
from tunix.dist_rl.swarm.step_context import StepContext
from tunix.dist_rl.swarm.step_context import group_grpo
from tunix.dist_rl.swarm.step_context import parl_grpo
from tunix.dist_rl.swarm.step_context import parl_reward_schedule
from tunix.dist_rl.swarm.step_context import repeat_tasks

__all__ = (
    "AttemptPlan",
    "AttemptSpec",
    "CollectionMode",
    "CriticalStepsBudget",
    "EpisodeRecord",
    "EpisodeRuntime",
    "FidelityReport",
    "LearnerSpec",
    "ManifestStore",
    "ModelAction",
    "ObjectiveBatch",
    "PolicySpec",
    "Recipe",
    "RoleSamplerAdapter",
    "ScoreRecord",
    "ServingFidelity",
    "ServingFidelityMode",
    "SnapshotBundle",
    "SnapshotRef",
    "StepContext",
    "StepResult",
    "WallClockBudget",
    "agent_env_workflow",
    "compact_actions",
    "group_grpo",
    "parl_grpo",
    "parl_reward_schedule",
    "repeat_tasks",
    "validate_ownership",
    "verify_serving_fidelity",
)
