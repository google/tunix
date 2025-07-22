# Copyright 2025 Google LLC
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

"""Client facing abstraction for interacting with RL training cluster."""

import dataclasses
import enum
from typing import Any, Union

from flax import nnx
import jax
from jax.sharding import Mesh  # pylint: disable=g-importing-member
import optax
from tunix.rl import trainer as rl_trainer
from tunix.rl.inference import inference_worker
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout import vanilla_rollout
from tunix.sft import peft_trainer


type ModelOrPath = Union[nnx.Module, str]


class Role(enum.Enum):
  """Role of the model."""

  ACTOR = "actor"
  CRITIC = "critic"
  REFERENCE = "reference"
  REWARD = "reward"
  ROLLOUT = "rollout"


@dataclasses.dataclass(slots=True, kw_only=True)
class RLTrainingConfig(peft_trainer.TrainingConfig):
  """RLTraining config.

  Attributes:
    actor_optimizer: Optimizer for the actor model.
    critic_optimizer: Optimizer for the critic model. If None, the critic model
      will be trained in the same optimizer as the actor model.
    actor_critic_share_backbone: Whether to share the backbone of the actor and
      critic models.
  """

  actor_optimizer: optax.GradientTransformation
  critic_optimizer: optax.GradientTransformation | None = None
  actor_critic_share_backbone: bool = False  # TODO(tsbao): support this.


@dataclasses.dataclass(kw_only=True, frozen=True)
class ClusterConfig:
  """Cluster config.

  Attributes:
    role_to_mesh: Mapping from model role to mesh. Key config for colocated vs
      disaggregated setup.
    rollout_engine: Rollout engine to use. E.g. "vanilla", "vllm".
    offload_to_cpu: Whether to offload models to CPU at each step..
    training_config: RL training config.
    rollout_config: Rollout config.
  """

  role_to_mesh: dict[Role, Mesh]
  rollout_engine: str = "vanilla"
  offload_to_cpu: bool = False  # TODO(tsbao): support offloading to CPU.

  training_config: RLTrainingConfig
  rollout_config: base_rollout.RolloutConfig


class RLCluster:
  """RLCluster."""

  def __init__(
      self,
      *,
      actor: ModelOrPath,
      critic: ModelOrPath | None = None,
      reference: ModelOrPath | None = None,
      reward: ModelOrPath | None = None,
      tokenizer: Any | None,
      cluster_config: ClusterConfig,
  ):
    self.cluster_config = cluster_config
    self.actor = self._load_model(actor)
    self.critic = self._load_model(critic) if critic else None
    self.reference = self._load_model(reference) if reference else None
    self.reward = self._load_model(reward) if reward else None
    self.tokenizer = tokenizer
    self._init_cluster()

  def _load_model(self, model_or_path: ModelOrPath) -> nnx.Module:
    if isinstance(model_or_path, nnx.Module):
      return model_or_path
    else:
      raise NotImplementedError("Loading from path is not supported yet.")

  def _init_cluster(self):
    """Initializes the RL cluster."""
    # 1. Initialize rollout.
    assert self.cluster_config.rollout_engine in [
        "vanilla",
        "vllm",
    ], f"Unsupported rollout engine: {self.cluster_config.rollout_engine}"
    if self.cluster_config.rollout_engine == "vanilla":
      assert hasattr(
          self.actor, "config"
      ), "Actor model must have a config attribute."
      self._rollout = vanilla_rollout.VanillaRollout(
          self.actor,
          self.tokenizer,
          cache_config=vanilla_rollout.CacheConfig(
              cache_size=self.cluster_config.rollout_config.kv_cache_size,
              num_layers=self.actor.config.num_layers,
              num_kv_heads=self.actor.config.num_kv_heads,
              head_dim=self.actor.config.head_dim,
          ),
      )
    elif self.cluster_config.rollout_engine == "vllm":
      raise NotImplementedError("vLLM rollout engine is not supported yet.")

    # 2. Initialize inference worker.
    inference_models = {}
    if self.critic is not None:
      inference_models["critic"] = self.critic
    if self.reference is not None:
      inference_models["reference"] = self.reference
    if self.reward is not None:
      inference_models["reward"] = self.reward
    self._inference_worker = inference_worker.InferenceWorker(inference_models)

    # 3. Initialize trainer.
    self._actor_trainer = rl_trainer.Trainer(
        model=self.actor,
        optimizer=self.cluster_config.training_config.actor_optimizer,
        training_config=self.cluster_config.training_config,
    )
    if (
        self.critic
        and not self.cluster_config.training_config.actor_critic_share_backbone
    ):
      self._critic_trainer = rl_trainer.Trainer(
          model=self.critic,
          optimizer=self.cluster_config.training_config.critic_optimizer,
          training_config=self.cluster_config.training_config,
      )

  @property
  def rollout(self) -> base_rollout.BaseRollout:
    return self._rollout

  @property
  def inference_worker(self) -> inference_worker.InferenceWorker:
    return self._inference_worker

  @property
  def actor_trainer(self) -> rl_trainer.Trainer:
    return self._actor_trainer

  @property
  def critic_trainer(self) -> rl_trainer.Trainer:
    return self._critic_trainer

  def update_actor(self, train_ds, eval_ds, skip_jit=False):
    with self.cluster_config.role_to_mesh[Role.ACTOR]:
      self.actor_trainer.train(train_ds, eval_ds, skip_jit)

  def update_critic(self, train_ds, eval_ds, skip_jit=False):
    with self.cluster_config.role_to_mesh[Role.CRITIC]:
      self._critic_trainer.train(train_ds, eval_ds, skip_jit)

  def generate(self, prompts: list[str]):
    with self.cluster_config.role_to_mesh[Role.ROLLOUT]:
      return self.rollout.generate(
          prompts,
          self.cluster_config.rollout_config,
      )

  def get_ref_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
  ) -> jax.Array:
    with self.cluster_config.role_to_mesh[Role.REFERENCE]:
      return self.inference_worker.get_ref_per_token_logps(
          prompt_tokens, completion_tokens, pad_id, eos_id
      )

  def get_old_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
  ) -> jax.Array:
    with self.cluster_config.role_to_mesh[Role.ROLLOUT]:
      return self.rollout.get_per_token_logps(prompt_tokens, completion_tokens)
