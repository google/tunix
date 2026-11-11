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

import importlib
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from typing import Any, TYPE_CHECKING

try:
  __version__ = version("google-tunix")  # match the name in pyproject.toml
except PackageNotFoundError:
  __version__ = "0.0.0.dev0"  # fallback for editable installs

if TYPE_CHECKING:
  # pylint: disable=g-multiple-import, g-importing-member, g-bad-import-order
  from tunix.common.configs import CacheConfig
  from tunix.common.configs import ClusterConfig
  from tunix.common.configs import RLTrainingConfig
  from tunix.common.configs import RolloutConfig
  from tunix.common.configs import TrainingConfig
  from tunix.common.datatypes import Role
  from tunix.distillation.distillation_trainer import DistillationTrainer
  from tunix.distillation.distillation_trainer import TrainingConfig as DistillationTrainingConfig
  from tunix.generate.sampler import Sampler
  from tunix.generate.tokenizer_adapter import Tokenizer
  from tunix.generate.tokenizer_adapter import TokenizerAdapter
  from tunix.perf.export import PerfMetricsExport
  from tunix.perf.metrics import PerfMetricsConfig
  from tunix.perf.metrics import PerfSpanQuery
  from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig as AgenticGRPOConfig
  from tunix.rl.agentic.agentic_grpo_learner import GRPOLearner as AgenticGRPOLearner
  from tunix.rl.grpo.grpo_learner import GRPOConfig
  from tunix.rl.grpo.grpo_learner import GrpoConfig
  from tunix.rl.grpo.grpo_learner import GRPOLearner
  from tunix.rl.grpo.grpo_learner import GrpoLearner
  from tunix.rl.grpo.grpo_learner import RewardFn
  from tunix.rl.ppo.ppo_learner import PPOConfig
  from tunix.rl.ppo.ppo_learner import PpoConfig
  from tunix.rl.ppo.ppo_learner import PPOLearner
  from tunix.rl.ppo.ppo_learner import PpoLearner
  from tunix.rl.rl_cluster import MetricsBuffer
  from tunix.rl.rl_cluster import RLCluster
  from tunix.rl.rl_cluster import RLEngine
  from tunix.sft.checkpoint_manager import CheckpointManager
  from tunix.sft.dpo.dpo_trainer import DPOTrainer
  from tunix.sft.dpo.dpo_trainer import DpoTrainer
  from tunix.sft.dpo.dpo_trainer import DPOTrainingConfig
  from tunix.sft.dpo.dpo_trainer import DpoTrainingConfig
  from tunix.sft.dpo.dpo_trainer import ORPOTrainer
  from tunix.sft.dpo.dpo_trainer import OrpoTrainer
  from tunix.sft.dpo.dpo_trainer import ORPOTrainingConfig
  from tunix.sft.dpo.dpo_trainer import OrpoTrainingConfig
  from tunix.sft.metrics_logger import MetricsLogger
  from tunix.sft.metrics_logger import MetricsLoggerOptions
  from tunix.sft.peft_trainer import PeftTrainer
  # pylint: enable=g-multiple-import, g-importing-member, g-bad-import-order

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "CacheConfig": ("tunix.common.configs", "CacheConfig"),
    "ClusterConfig": ("tunix.common.configs", "ClusterConfig"),
    "RLTrainingConfig": ("tunix.common.configs", "RLTrainingConfig"),
    "RolloutConfig": ("tunix.common.configs", "RolloutConfig"),
    "TrainingConfig": ("tunix.common.configs", "TrainingConfig"),
    "Role": ("tunix.common.datatypes", "Role"),
    "DistillationTrainer": (
        "tunix.distillation.distillation_trainer",
        "DistillationTrainer",
    ),
    "DistillationTrainingConfig": (
        "tunix.distillation.distillation_trainer",
        "TrainingConfig",
    ),
    "Sampler": ("tunix.generate.sampler", "Sampler"),
    "Tokenizer": ("tunix.generate.tokenizer_adapter", "Tokenizer"),
    "TokenizerAdapter": (
        "tunix.generate.tokenizer_adapter",
        "TokenizerAdapter",
    ),
    "PerfMetricsExport": ("tunix.perf.export", "PerfMetricsExport"),
    "PerfMetricsConfig": ("tunix.perf.metrics", "PerfMetricsConfig"),
    "PerfSpanQuery": ("tunix.perf.metrics", "PerfSpanQuery"),
    "AgenticGRPOConfig": (
        "tunix.rl.agentic.agentic_grpo_learner",
        "GRPOConfig",
    ),
    "AgenticGRPOLearner": (
        "tunix.rl.agentic.agentic_grpo_learner",
        "GRPOLearner",
    ),
    "GRPOConfig": ("tunix.rl.grpo.grpo_learner", "GRPOConfig"),
    "GrpoConfig": ("tunix.rl.grpo.grpo_learner", "GrpoConfig"),
    "GRPOLearner": ("tunix.rl.grpo.grpo_learner", "GRPOLearner"),
    "GrpoLearner": ("tunix.rl.grpo.grpo_learner", "GrpoLearner"),
    "RewardFn": ("tunix.rl.grpo.grpo_learner", "RewardFn"),
    "PPOConfig": ("tunix.rl.ppo.ppo_learner", "PPOConfig"),
    "PpoConfig": ("tunix.rl.ppo.ppo_learner", "PpoConfig"),
    "PPOLearner": ("tunix.rl.ppo.ppo_learner", "PPOLearner"),
    "PpoLearner": ("tunix.rl.ppo.ppo_learner", "PpoLearner"),
    "MetricsBuffer": ("tunix.rl.rl_cluster", "MetricsBuffer"),
    "RLCluster": ("tunix.rl.rl_cluster", "RLCluster"),
    "RLEngine": ("tunix.rl.rl_cluster", "RLEngine"),
    "CheckpointManager": ("tunix.sft.checkpoint_manager", "CheckpointManager"),
    "DPOTrainer": ("tunix.sft.dpo.dpo_trainer", "DPOTrainer"),
    "DpoTrainer": ("tunix.sft.dpo.dpo_trainer", "DpoTrainer"),
    "DPOTrainingConfig": ("tunix.sft.dpo.dpo_trainer", "DPOTrainingConfig"),
    "DpoTrainingConfig": ("tunix.sft.dpo.dpo_trainer", "DpoTrainingConfig"),
    "ORPOTrainer": ("tunix.sft.dpo.dpo_trainer", "ORPOTrainer"),
    "OrpoTrainer": ("tunix.sft.dpo.dpo_trainer", "OrpoTrainer"),
    "ORPOTrainingConfig": ("tunix.sft.dpo.dpo_trainer", "ORPOTrainingConfig"),
    "OrpoTrainingConfig": ("tunix.sft.dpo.dpo_trainer", "OrpoTrainingConfig"),
    "MetricsLogger": ("tunix.sft.metrics_logger", "MetricsLogger"),
    "MetricsLoggerOptions": (
        "tunix.sft.metrics_logger",
        "MetricsLoggerOptions",
    ),
    "PeftTrainer": ("tunix.sft.peft_trainer", "PeftTrainer"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:  # pylint: disable=invalid-name
  target = _LAZY_EXPORTS.get(name)
  if target is None:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
  module_path, attr_name = target
  value = getattr(importlib.import_module(module_path), attr_name)
  globals()[name] = value
  return value


def __dir__() -> list[str]:  # pylint: disable=invalid-name
  return sorted(set(globals()) | set(_LAZY_EXPORTS))
