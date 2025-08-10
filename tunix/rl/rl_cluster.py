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

import collections
import contextlib
import copy
import dataclasses
import enum
import gc
import operator
import os
from typing import Any, Union
from absl import logging
from flax import nnx
from flax.nnx import filterlib
from flax.nnx import statelib
import jax
from jax.sharding import Mesh  # pylint: disable=g-importing-member
import jaxtyping
import optax
# Internal placeholder for vllm rollout worker stub, don't change this line.
from tunix.rl import reshard
from tunix.rl import trainer as rl_trainer
from tunix.rl import utils
from tunix.rl.inference import inference_worker
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout import vanilla_rollout
from tunix.sft import peft_trainer


ModelOrPath = Union[nnx.Module, str]


class Role(enum.Enum):
  """Role of the model."""

  ACTOR = "actor"  # policy model
  CRITIC = "critic"  # value model (only for PPO-style algos, not for GRPO)
  REFERENCE = "reference"  # kept fixed during training
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
    rollout_model_version: Model version for vllm rollout engine.
    rollout_lora_config: LoRA config for vllm rollout engine.
  """

  role_to_mesh: dict[Role, Mesh]
  rollout_engine: str = "vanilla"
  offload_to_cpu: bool = False

  training_config: RLTrainingConfig
  rollout_config: base_rollout.RolloutConfig

  rollout_model_version: str = ""
  rollout_lora_config: dict[str, Any] | None = None


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
    self.r2m = cluster_config.role_to_mesh
    self._init_backbone_sharing_map(actor, reference)

    self._default_memory_kind = jax.devices()[0].default_memory().kind
    self.train_actor = self._load_model(actor, self.r2m[Role.ACTOR])

    if Role.ROLLOUT in self._backbone_sharing_map[Role.ACTOR]:
      self.rollout_actor = self.train_actor
    else:
      self.rollout_actor = self._load_model(actor, self.r2m[Role.ROLLOUT])

    if reference:
      self.reference = self._load_model(reference, self.r2m[Role.REFERENCE])
      if Role.REFERENCE in self._backbone_sharing_map[Role.ACTOR]:
        if not utils.is_sharing_backbone(self.reference, self.train_actor):
          logging.warning(
              "Reference model and actor model are colocated but do not share"
              " the same backbone. This will result in an unnecessary model"
              " copy and increased HBM usage."
          )
    else:
      self.reference = None
    self.critic = (
        self._load_model(critic, self.r2m[Role.CRITIC]) if critic else None
    )
    if Role.CRITIC in self._backbone_sharing_map[Role.ACTOR]:
      critic_state = nnx.state(self.train_actor, filterlib.Not(nnx.LoRAParam))
      nnx.update(self.critic, critic_state)
    self.reward = (
        self._load_model(reward, self.r2m[Role.REWARD]) if reward else None
    )

    self.tokenizer = tokenizer
    self._init_cluster()
    gc.collect()

  def _init_backbone_sharing_map(
      self,
      actor: ModelOrPath,
      reference: ModelOrPath | None = None,
  ):
    """Initializes the backbone sharing map."""
    self._backbone_sharing_map: dict[Role, list[Role]] = (
        collections.defaultdict(list)
    )

    if self.r2m[Role.ACTOR] == self.r2m[Role.ROLLOUT]:
      # Given that we load both actor trainer and rollout from `actor`,
      # if the meshes are the same, they are able to share the same model.
      # TODO(linchai): We may want to enable different shardings for actor
      # trainer and rollout even when they are colocated.
      self._backbone_sharing_map[Role.ACTOR].append(Role.ROLLOUT)
      self._backbone_sharing_map[Role.ROLLOUT].append(Role.ACTOR)

    # TODO(linchai): support loadding model from path and backbone sharing for
    # such case.
    if not isinstance(actor, nnx.Module) or (
        reference and not isinstance(reference, nnx.Module)
    ):
      return
    if peft_trainer.is_lora_enabled(actor):
      if reference and self.r2m[Role.ACTOR] == self.r2m[Role.REFERENCE]:
        self._backbone_sharing_map[Role.ACTOR].append(Role.REFERENCE)
        self._backbone_sharing_map[Role.REFERENCE].append(Role.ACTOR)
      # TODO(linchai): maybe support critic backbone sharing.

    self._propagate_backbone_sharing_map()

  def _load_model(self, model_or_path: ModelOrPath, mesh: Mesh) -> nnx.Module:
    """Loads model with given mesh to the given memory_kind.

    If input is already an NNX model, check if the model is sharded on the
    target mesh. If not, reshard the model.

    Args:
      model_or_path: either a nnx.Module or a path to a model.
      mesh: the mesh to load the model on.

    Returns:
      The model loaded on the given mesh.
    """
    if isinstance(model_or_path, nnx.Module):
      model_mesh = utils.get_pytree_mesh_info(nnx.state(model_or_path))
      original_shardings = jax.tree_util.tree_map(
          lambda x: x.sharding, nnx.state(model_or_path)
      )
      is_on_device = jax.tree_util.tree_reduce(
          operator.or_,
          jax.tree.map(
              lambda x: x.memory_kind == self._default_memory_kind,
              original_shardings,
          ),
      )
      if not mesh.empty and model_mesh != mesh:
        logging.warning("Resharding model from %s to %s", model_mesh, mesh)
        graph, state = nnx.split(model_or_path)
        dst_shardings = jax.tree_util.tree_map(
            lambda x: jax.sharding.NamedSharding(
                mesh,
                x,
                memory_kind=self._default_memory_kind
                if is_on_device
                else "pinned_host",
            ),
            nnx.get_partition_spec(state),
        )
        model_or_path = nnx.merge(
            graph, reshard.reshard_pytree(state, dst_shardings)
        )
      if is_on_device and self.cluster_config.offload_to_cpu:
        graph, state = nnx.split(model_or_path)
        new_params = utils.put_params_on_memory_kind(state, "pinned_host")
        model_or_path = nnx.merge(graph, new_params)
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
          self.rollout_actor, "config"
      ), "Actor model must have a config attribute."
      # We must load the model from CPU before initializing the rollout,
      # otherwise the prefill and decode programs might be initialized on CPU.
      self._maybe_load_model_from_cpu(self.rollout_actor, Role.ROLLOUT)
      self._rollout = vanilla_rollout.VanillaRollout(
          self.rollout_actor,
          self.tokenizer,
          cache_config_or_size=base_rollout.CacheConfig(
              cache_size=self.cluster_config.rollout_config.kv_cache_size,
              num_layers=self.rollout_actor.config.num_layers,
              num_kv_heads=self.rollout_actor.config.num_kv_heads,
              head_dim=self.rollout_actor.config.head_dim,
          ),
      )
      self._maybe_offload_model_to_cpu(self._rollout.model(), Role.ROLLOUT)
    elif self.cluster_config.rollout_engine == "vllm":
      from tunix.rl.rollout import vllm_rollout
      # TODO(linchai): maybe support offloading for vllm rollout.
      self._rollout = vllm_rollout.VllmRollout(
          self.rollout_actor,
          self.tokenizer,
          cache_config_or_size=self.cluster_config.rollout_config.kv_cache_size,
          mesh=self.r2m[Role.ROLLOUT],
          lora_config=self.cluster_config.rollout_lora_config,
          model_version=self.cluster_config.rollout_model_version,
      )
    else:
      raise NotImplementedError(
          f"Rollout engine {self.cluster_config.rollout_engine} not supported"
      )
    del self.rollout_actor

    # 2. Initialize inference worker.
    inference_models = {}
    if self.critic is not None:
      inference_models["critic"] = self.critic
    if self.reference is not None:
      inference_models["reference"] = self.reference
      del self.reference
    if self.reward is not None:
      inference_models["reward"] = self.reward
      del self.reward
    self._inference_worker = inference_worker.InferenceWorker(inference_models)

    # 3. Initialize trainer.
    self._maybe_load_model_from_cpu(self.train_actor, Role.ACTOR)
    actor_config = copy.deepcopy(self.cluster_config.training_config)
    if actor_config.checkpoint_root_directory is not None:
      actor_config.checkpoint_root_directory = os.path.join(
          actor_config.checkpoint_root_directory, "actor"
      )
    self._actor_trainer = rl_trainer.Trainer(
        model=self.train_actor,
        optimizer=self.cluster_config.training_config.actor_optimizer,
        training_config=self.cluster_config.training_config,
    )
    del self.train_actor
    self._maybe_offload_model_to_cpu(self.actor_trainer.model, Role.ACTOR)
    if (
        self.critic
        and Role.CRITIC not in self._backbone_sharing_map[Role.ACTOR]
    ):
      critic_config = copy.deepcopy(self.cluster_config.training_config)
      if critic_config.checkpoint_root_directory is not None:
        critic_config.checkpoint_root_directory = os.path.join(
            critic_config.checkpoint_root_directory, "critic"
        )
      self._critic_trainer = rl_trainer.Trainer(
          model=self.critic,
          optimizer=self.cluster_config.training_config.critic_optimizer,
          training_config=critic_config,
      )
      del self.critic
      self._maybe_offload_model_to_cpu(self._critic_trainer.model, Role.CRITIC)

  def _propagate_backbone_sharing_map(self):
    """Propagates backbone sharing map."""
    for role in self._backbone_sharing_map[Role.ACTOR]:
      for other_role in self._backbone_sharing_map[Role.ACTOR]:
        if other_role != role:
          self._backbone_sharing_map[role].append(other_role)

  def _put_model_on_memory_kind(self, model: nnx.Module, memory_kind: str):
    """Puts model on the given memory kind."""
    assert memory_kind in [
        "pinned_host",
        "device",
    ], f"Unsupported memory kind: {memory_kind}"
    original_variables = nnx.variables(model)
    new_variables = utils.put_params_on_memory_kind(
        original_variables, memory_kind
    )
    nnx.update(model, new_variables)

  def _update_models_sharing_weights(
      self,
      params: jaxtyping.PyTree,
      role: Role,
  ):
    """Updates models sharing weights."""
    for role in self._backbone_sharing_map[role]:
      if role == Role.ROLLOUT:
        if hasattr(self, "rollout_actor"):
          nnx.update(self.rollout_actor, params)
        else:
          self.rollout.update_params(params)
      elif role == Role.REFERENCE:
        ref_model = (
            self.reference
            if hasattr(self, "reference")
            else self.inference_worker.get_model("reference")
        )
        if ref_model:
          nnx.update(
              ref_model,
              statelib.filter_state(params, filterlib.Not(nnx.LoRAParam)),
          )
      elif role == Role.ACTOR:
        actor_model = (
            self.train_actor
            if hasattr(self, "train_actor")
            else self.actor_trainer.model
        )
        nnx.update(actor_model, params)

  def _maybe_load_model_from_cpu(self, model: nnx.Module, role: Role):
    """Loads model from CPU if needed."""
    if not self.cluster_config.offload_to_cpu:
      return
    self._put_model_on_memory_kind(model, "device")
    self._update_models_sharing_weights(nnx.state(model), role)

  def _maybe_offload_model_to_cpu(self, model: nnx.Module, role: Role):
    """Offloads model to CPU if needed."""
    if not self.cluster_config.offload_to_cpu:
      return
    self._put_model_on_memory_kind(model, "pinned_host")
    self._update_models_sharing_weights(nnx.state(model), role)

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
      self._maybe_load_model_from_cpu(self.actor_trainer.model, Role.ACTOR)
      self.actor_trainer.train(train_ds, eval_ds, skip_jit)
      self._maybe_offload_model_to_cpu(self.actor_trainer.model, Role.ACTOR)

  def update_critic(self, train_ds, eval_ds, skip_jit=False):
    with self.cluster_config.role_to_mesh[Role.CRITIC]:
      self._maybe_load_model_from_cpu(self.critic_trainer.model, Role.CRITIC)
      self._critic_trainer.train(train_ds, eval_ds, skip_jit)
      self._maybe_offload_model_to_cpu(self.critic_trainer.model, Role.CRITIC)

  def generate(self, prompts: list[str]):
    """Generates text from the given prompts.

    Args:
      prompts: A list of prompts to generate text from.

    Returns:
      A list of generated text.
    """
    with self.cluster_config.role_to_mesh[Role.ROLLOUT]:
      model = self.rollout.model()
      self._maybe_load_model_from_cpu(model, Role.ROLLOUT)
      if self.cluster_config.offload_to_cpu:
        self.rollout.update_params(nnx.state(model))
      output = self.rollout.generate(
          prompts,
          self.cluster_config.rollout_config,
      )
      model = self.rollout.model()
      self._maybe_offload_model_to_cpu(model, Role.ROLLOUT)
      if self.cluster_config.offload_to_cpu:
        self.rollout.update_params(nnx.state(model))
      return output

  def get_ref_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
  ) -> jax.Array:
    """Gets the per-token logps of the reference model."""
    # TODO(linchai): Need to transfer the prompt and completion tokens to the
    # reference model's mesh if rollout and reference are on different meshes.
    with self.cluster_config.role_to_mesh[Role.REFERENCE]:
      self._maybe_load_model_from_cpu(
          self.inference_worker.get_model("reference"), Role.REFERENCE
      )
      ref_per_token_logps = self.inference_worker.get_ref_per_token_logps(
          prompt_tokens, completion_tokens, pad_id, eos_id
      )
      self._maybe_offload_model_to_cpu(
          self.inference_worker.get_model("reference"), Role.REFERENCE
      )
      return ref_per_token_logps

  def get_old_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
  ) -> jax.Array:
    """Gets the per-token logps of the current policy model."""
    with self.cluster_config.role_to_mesh[Role.ROLLOUT]:
      model = self.rollout.model()
      self._maybe_load_model_from_cpu(model, Role.ROLLOUT)
      if self.cluster_config.offload_to_cpu:
        self.rollout.update_params(nnx.state(model))
      per_token_logps = self.rollout.get_per_token_logps(
          prompt_tokens, completion_tokens
      )
      model = self.rollout.model()
      self._maybe_offload_model_to_cpu(model, Role.ROLLOUT)
      if self.cluster_config.offload_to_cpu:
        self.rollout.update_params(nnx.state(model))
      return per_token_logps

  def sync_weights(self):
    """Syncs the weights of between the sampler model and trainer model."""
    if jax.devices() and jax.default_backend() not in ["tpu", "gpu"]:
      cm = contextlib.ExitStack()
      cm.enter_context(jax.transfer_guard_device_to_host("disallow_explicit"))
      cm.enter_context(jax.transfer_guard_host_to_device("disallow_explicit"))
    else:
      cm = contextlib.nullcontext()
    with cm:
      filter_types = (
          nnx.LoRAParam
          if peft_trainer.is_lora_enabled(self.actor_trainer.model)
          else nnx.Param,
      )
      src_filtered_params = nnx.state(self.actor_trainer.model, filter_types)
      self.rollout.update_params(src_filtered_params, filter_types)

  def get_values(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
  ) -> jax.Array:
    with self.cluster_config.role_to_mesh[Role.CRITIC]:
      return self.inference_worker.get_values(
          prompt_tokens, completion_tokens, pad_id, eos_id
      )

  def get_rewards(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
  ) -> jax.Array:
    with self.cluster_config.role_to_mesh[Role.REWARD]:
      return self.inference_worker.get_rewards(
          prompt_tokens,
          completion_tokens,
          pad_id,
          eos_id,
      )
