# Copyright 2026 Google LLC
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

"""Distributed-RL Rollout Subsystem."""

# pylint: disable=g-importing-member
from tunix.experimental.common.datatypes import RolloutRequest
from tunix.experimental.common.datatypes import WeightSyncMetadata
from tunix.experimental.rollout.collector import TrajectoryCollectorEngine
from tunix.experimental.rollout.manager import RolloutManager
from tunix.experimental.rollout.sampler import Sampler
from tunix.experimental.rollout.vanilla_sampler_adapter import VanillaSamplerAdapter
from tunix.experimental.rollout.vllm_sampler_adapter import VllmSamplerAdapter
from tunix.experimental.trajectory.trajectory import Trajectory
from tunix.experimental.trajectory.trajectory import TrajectoryError
