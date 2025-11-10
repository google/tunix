# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tunix API."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version

try:
  __version__ = version("google-tunix")  # match the name in pyproject.toml
except PackageNotFoundError:
  __version__ = "0.0.0.dev0"  # fallback for editable installs


# pylint: disable=g-multiple-import, g-importing-member

from tunix.distillation.distillation_trainer import DistillationTrainer
from tunix.distillation.distillation_trainer import TrainingConfig as DistillationTrainingConfig
from tunix.generate.sampler import CacheConfig
from tunix.generate.sampler import Sampler
from tunix.generate.tokenizer_adapter import TokenizerAdapter, Tokenizer
from tunix.perf.metrics import PerfMetricsApi
from tunix.perf.metrics import PerfMetricsConfig
from tunix.rl.grpo.grpo_learner import GRPOConfig
from tunix.rl.grpo.grpo_learner import GrpoConfig
from tunix.rl.grpo.grpo_learner import GRPOLearner
from tunix.rl.grpo.grpo_learner import GrpoLearner
from tunix.rl.grpo.grpo_learner import RewardFn
from tunix.rl.ppo.ppo_learner import PPOConfig
from tunix.rl.ppo.ppo_learner import PpoConfig
from tunix.rl.ppo.ppo_learner import PPOLearner
from tunix.rl.ppo.ppo_learner import PpoLearner
from tunix.rl.rl_cluster import ClusterConfig
from tunix.rl.rl_cluster import MetricsBuffer
from tunix.rl.rl_cluster import RLCluster
from tunix.rl.rl_cluster import RLTrainingConfig
from tunix.rl.rl_cluster import Role
from tunix.rl.rollout.base_rollout import RolloutConfig
from tunix.sft.checkpoint_manager import CheckpointManager
from tunix.sft.dpo.dpo_trainer import DPOTrainer
from tunix.sft.dpo.dpo_trainer import DpoTrainer
from tunix.sft.dpo.dpo_trainer import DPOTrainingConfig
from tunix.sft.dpo.dpo_trainer import DpoTrainingConfig
from tunix.sft.metrics_logger import MetricsLogger
from tunix.sft.metrics_logger import MetricsLoggerOptions
from tunix.sft.peft_trainer import PeftTrainer
from tunix.sft.peft_trainer import TrainingConfig

# pylint: enable=g-multiple-import, g-importing-member
