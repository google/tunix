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
import functools
import gc
import itertools
import operator
import os
from typing import Any, Callable, Dict, Tuple

from absl import logging
from flax import nnx
from flax.nnx import filterlib
from flax.nnx import statelib
import jax
import jax.numpy as jnp
from jax.sharding import Mesh  # pylint: disable=g-importing-member
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
import jaxtyping
import optax
# Internal placeholder for sglang_jax rollout worker stub, don't change this line.
# Internal placeholder for vllm rollout worker stub, don't change this line.
from tunix.rl import reshard
from tunix.rl import trainer as rl_trainer
from tunix.rl import utils as rl_utils
from tunix.rl.inference import inference_worker
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout import vanilla_rollout
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
from tunix.sft import utils as sft_utils

ModelOrPath = nnx.Module | str


MetricsT = Dict[
    str, Tuple[ArrayLike | str, Callable[[jax.Array], jax.Array] | None]
]  # Metrics to be buffered: name -> (values, optional agg_fn)


@dataclasses.dataclass(slots=True)
class MetricsBuffer:
  global_steps: int
  # Metrics to be buffered: name -> (list of (values), optional agg_fn)
  metrics: dict[
      str, tuple[list[ArrayLike | str], Callable[[ArrayLike], ArrayLike] | None]
  ] = dataclasses.field(default_factory=dict)
  mode: str = "train"


class Mode(enum.Enum):
  """Mode of RolloutConfig."""

  TRAIN = "train"
  EVAL = "eval"

  def __str__(self):
    return self.value


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

  Parameters:
    actor_optimizer: Optimizer for the actor model.
    critic_optimizer: Optimizer for the critic model. If None, the critic model
      will be trained in the same optimizer as the actor model.
    mini_batch_size: The mini-batch size used for policy weight updates. One
      mini-batch corresponds to one optimizer update. `mini_batch_size` must be
      divisible by the global batch size.
    train_micro_batch_size: The micro-batch size used for gradient
      accumulation at training time. `train_micro_batch_size` must be
      divisible by `mini_batch_size`.
    rollout_micro_batch_size: The micro-batch size used for model rollouts.
    compute_logps_micro_batch_size: The micro-batch size used for computing log
      probabilities (e.g. for reference and old policy models).
  """

  actor_optimizer: optax.GradientTransformation
  critic_optimizer: optax.GradientTransformation | None = None
  mini_batch_size: int | None = None
  train_micro_batch_size: int | None = None
  rollout_micro_batch_size: int | None = None
  compute_logps_micro_batch_size: int | None = None

  def __post_init__(self):
    """Validates the configuration after initialization."""
    for name in [
        "mini_batch_size",
        "train_micro_batch_size",
        "rollout_micro_batch_size",
        "compute_logps_micro_batch_size",
    ]:
      rl_utils.check_positive(getattr(self, name), name)

    if self.gradient_accumulation_steps is not None:
      raise ValueError(
          "For RL training, gradient_accumulation_steps should be None. It is "
          "automatically derived from: "
          "`mini_batch_size // train_micro_batch_size`."
      )

    if self.train_micro_batch_size is not None:
      if self.mini_batch_size is None:
        raise ValueError(
            "For RL training, `mini_batch_size` must be set when"
            " `train_micro_batch_size` is set."
        )
      rl_utils.check_divisibility(
          self.train_micro_batch_size,
          self.mini_batch_size,
          f"{self.train_micro_batch_size=}",
          f"{self.mini_batch_size=}",
      )
      self.gradient_accumulation_steps = (
          self.mini_batch_size // self.train_micro_batch_size
      )


@dataclasses.dataclass(kw_only=True, frozen=True)
class ClusterConfig:
  """Cluster config.

  Parameters:
    role_to_mesh: Mapping from model role to mesh. Key config for colocated vs
      disaggregated setup.
    rollout_engine: Rollout engine to use. E.g. "vanilla", "vllm", "sglang_jax".
        Alternatively, if a subclass of `base_rollout.BaseRollout` is provided,
        it will be used as the rollout engine.
    offload_to_cpu: Whether to offload models to CPU at each step..
    training_config: RL training config.
    rollout_config: Rollout config. It may be different for different modes,
      e.g. TRAIN vs EVAL.
    rollout_vllm_model_version: Model version for vllm rollout engine.
    rollout_vllm_lora_config: LoRA config for vllm rollout engine.
    rollout_vllm_hbm_utilization: The percentage of TPU/GPU HBM allocated the
      vllm rollout engine.
    rollout_vllm_init_with_random_weights: Init the vllm TPU backend model with
      random weights instead of loading from the given path.
    rollout_vllm_tpu_backend_type: The TPU Jax backend type for vllm rollout
      engine, E.g. "jax", "torchax" or "pytorch_xla".
    rollout_vllm_swap_space_size_gb: The swap space size (in GiB) for vllm
      rollout engine. This is the amount of CPU memory (RAM) to allocate for
      swapping KV cache blocks from the TPU/GPU memory (HBM). A larger value
      allows for larger batch sizes and longer sequences, potentially at the
      cost of increased latency if swapping occurs.
  """

  role_to_mesh: dict[Role, Mesh]
  rollout_engine: str | type[base_rollout.BaseRollout] = "vanilla"
  offload_to_cpu: bool = False

  training_config: RLTrainingConfig
  rollout_config: (
      dict[Mode, base_rollout.RolloutConfig] | base_rollout.RolloutConfig
  )


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
      rollout_data_type = (
          self.cluster_config.rollout_config[Mode.TRAIN].data_type
          if isinstance(self.cluster_config.rollout_config, dict)
          else self.cluster_config.rollout_config.data_type
      )
      self.rollout_actor = self._load_model(
          actor,
          self.r2m[Role.ROLLOUT],
          rollout_data_type,
      )

    if reference:
      self.reference = self._load_model(reference, self.r2m[Role.REFERENCE])
      if Role.REFERENCE in self._backbone_sharing_map[Role.ACTOR]:
        if not rl_utils.is_sharing_backbone(self.reference, self.train_actor):
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

    # NB: global steps should be adjusted properly based on the actual RL
    # algorithm. E.g. when loading from a checkpoint with additional inner loops
    # that update the model, we should properly update the global steps.
    self.global_steps = 0

    self._rl_metrics_logger = metrics_logger.MetricsLogger(
        self.cluster_config.training_config.metrics_logging_options
    )
    self._buffered_train_metrics: list[MetricsBuffer] = []
    self._buffered_eval_metrics: list[MetricsBuffer] = []
    self._external_metrics_logger = None

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
    if sft_utils.is_lora_enabled(actor):
      if reference and self.r2m[Role.ACTOR] == self.r2m[Role.REFERENCE]:
        self._backbone_sharing_map[Role.ACTOR].append(Role.REFERENCE)
        self._backbone_sharing_map[Role.REFERENCE].append(Role.ACTOR)
      # TODO(linchai): maybe support critic backbone sharing.

    self._propagate_backbone_sharing_map()

  def _load_model(
      self,
      model_or_path: ModelOrPath,
      mesh: Mesh,
      data_type: jnp.dtype | None = None,
  ) -> nnx.Module:
    """Loads model with given mesh to the given memory_kind.

    If input is already an NNX model, check if the model is sharded on the
    target mesh. If not, reshard the model.

    Args:
      model_or_path: either a nnx.Module or a path to a model.
      mesh: the mesh to load the model on.
      data_type: optional data type to cast the model parameters to.

    Returns:
      The model loaded on the given mesh.
    """
    if isinstance(model_or_path, nnx.Module):
      model_mesh = rl_utils.get_pytree_mesh_info(nnx.state(model_or_path))
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
        if data_type and data_type != jax.tree.leaves(state)[0].dtype:
          tmp_state = jax.tree.map(lambda x: x.astype(data_type), state)
        else:
          tmp_state = state
        model_or_path = nnx.merge(
            graph, reshard.reshard_pytree(tmp_state, dst_shardings)
        )
        del tmp_state
        gc.collect()
      if is_on_device and self.cluster_config.offload_to_cpu:
        graph, state = nnx.split(model_or_path)
        new_params = rl_utils.put_params_on_memory_kind(state, "pinned_host")
        model_or_path = nnx.merge(graph, new_params)
      return model_or_path
    else:
      raise NotImplementedError("Loading from path is not supported yet.")

  def _init_cluster(self):
    """Initializes the RL cluster."""
    # 1. Initialize rollout.
    if isinstance(
        self.cluster_config.rollout_engine, str
    ) and self.cluster_config.rollout_engine not in [
        "vanilla",
        "vllm",
        "sglang_jax",
    ]:
      raise ValueError(
          "`cluster_config.rollout_engine` should be one of `'vanilla'` or"
          " `'vllm'` or `'sglang_jax'`. Received:"
          f" '{self.cluster_config.rollout_engine}'."
      )
    if isinstance(self.cluster_config.rollout_config, dict):
      max_kv_cache_size = max(
          self.cluster_config.rollout_config[Mode.TRAIN].kv_cache_size,
          self.cluster_config.rollout_config[Mode.EVAL].kv_cache_size,
      )
    else:
      max_kv_cache_size = self.cluster_config.rollout_config.kv_cache_size

    if self.cluster_config.rollout_engine == "vanilla":
      if not hasattr(self.rollout_actor, "config"):
        raise ValueError("`self.rollout_actor` must have a config attribute.")
      # We must load the model from CPU before initializing the rollout,
      # otherwise the prefill and decode programs might be initialized on CPU.
      self._maybe_load_model_from_cpu(self.rollout_actor, Role.ROLLOUT)
      self._rollout = vanilla_rollout.VanillaRollout(
          self.rollout_actor,
          self.tokenizer,
          cache_config_or_size=base_rollout.CacheConfig(
              cache_size=max_kv_cache_size,
              num_layers=self.rollout_actor.config.num_layers,
              num_kv_heads=self.rollout_actor.config.num_kv_heads,
              head_dim=self.rollout_actor.config.head_dim,
          ),
      )
      self._maybe_offload_model_to_cpu(self._rollout.model(), Role.ROLLOUT)
    elif self.cluster_config.rollout_engine == "vllm":
      from tunix.rl.rollout import vllm_rollout
      loaded_vllm_config = None
      if isinstance(
              self.cluster_config.rollout_config, base_rollout.RolloutConfig
          ):
        loaded_vllm_config = self.cluster_config.rollout_config
      elif isinstance(self.cluster_config.rollout_config, dict):
        loaded_vllm_config = self.cluster_config.rollout_config[Mode.TRAIN]

      if loaded_vllm_config is None:
        raise ValueError("Rollout vllm model config is missing!")

      if loaded_vllm_config.rollout_vllm_model_version is None:
        raise ValueError("Rollout vllm model version or path is missing!")

      # TODO(linchai): maybe support offloading for vllm rollout.
      self._rollout = vllm_rollout.VllmRollout(
          self.rollout_actor,
          self.tokenizer,
          cache_config_or_size=max_kv_cache_size,
          mesh=self.r2m[Role.ROLLOUT],
          rollout_config=loaded_vllm_config,
      )
    elif self.cluster_config.rollout_engine == "sglang_jax":
      from tunix.rl.rollout import sglang_jax_rollout
      if isinstance(
          self.cluster_config.rollout_config, base_rollout.RolloutConfig
      ):
        loaded_sglang_jax_config = self.cluster_config.rollout_config
      elif isinstance(self.cluster_config.rollout_config, dict):
        loaded_sglang_jax_config = self.cluster_config.rollout_config[
            Mode.TRAIN
        ]
      else:
        raise ValueError(
            "Rollout sglang jax model config is missing!"
        )

      self._rollout = sglang_jax_rollout.SglangJaxRollout(
          self.rollout_actor,
          self.tokenizer,
          mesh=self.r2m[Role.ROLLOUT],
          rollout_config=loaded_sglang_jax_config,
      )
    elif (
        isinstance(self.cluster_config.rollout_engine, type)
        and issubclass(
            self.cluster_config.rollout_engine, base_rollout.BaseRollout
        )
    ) or (
        isinstance(self.cluster_config.rollout_engine, functools.partial)
        and issubclass(
            self.cluster_config.rollout_engine.func,
            base_rollout.BaseRollout,
        )
    ):
      self._rollout = self.cluster_config.rollout_engine(
          rollout_actor=self.rollout_actor,
          tokenizer=self.tokenizer,
          mesh=self.r2m[Role.ROLLOUT],
          rollout_config=self.cluster_config.rollout_config,
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
    if (
        self.critic
        and Role.CRITIC not in self._backbone_sharing_map[Role.ACTOR]
    ):
      critic_config = copy.deepcopy(self.cluster_config.training_config)
      critic_config.metric_prefix = "critic/"
      critic_config.pbar_description = "Critic Training"
      if critic_config.checkpoint_root_directory is not None:
        critic_config.checkpoint_root_directory = os.path.join(
            critic_config.checkpoint_root_directory, "critic"
        )
      self._critic_trainer = rl_trainer.Trainer(
          model=self.critic,
          optimizer=self.cluster_config.training_config.critic_optimizer,
          training_config=critic_config,
          custom_checkpoint_metadata_fn=lambda: {
              "global_step": self.global_steps + 1
          },  # offset by 1 since global_step is incremented after the training loop in rl_learner. # pylint: disable=line-too-longå
      )
      del self.critic
      self._maybe_offload_model_to_cpu(self._critic_trainer.model, Role.CRITIC)

    self._maybe_load_model_from_cpu(self.train_actor, Role.ACTOR)
    actor_config = copy.deepcopy(self.cluster_config.training_config)
    actor_config.metric_prefix = "actor/"
    actor_config.pbar_description = "Actor Training"
    if actor_config.checkpoint_root_directory is not None:
      actor_config.checkpoint_root_directory = os.path.join(
          actor_config.checkpoint_root_directory, "actor"
      )
    self._actor_trainer = rl_trainer.Trainer(
        model=self.train_actor,
        optimizer=self.cluster_config.training_config.actor_optimizer,
        training_config=actor_config,
        custom_checkpoint_metadata_fn=lambda: {
            "global_step": self.global_steps + 1
        },  # offset by 1 since global_step is incremented after the training loop in rl_learner. # pylint: disable=line-too-longå
    )
    del self.train_actor
    self._maybe_offload_model_to_cpu(self.actor_trainer.model, Role.ACTOR)

  def _propagate_backbone_sharing_map(self):
    """Propagates backbone sharing map."""
    for role in self._backbone_sharing_map[Role.ACTOR]:
      for other_role in self._backbone_sharing_map[Role.ACTOR]:
        if other_role != role:
          self._backbone_sharing_map[role].append(other_role)

  def _put_model_on_memory_kind(self, model: nnx.Module, memory_kind: str):
    """Puts model on the given memory kind."""
    if memory_kind not in ["pinned_host", "device"]:
      raise ValueError(f"Unsupported memory kind. Received: {memory_kind}")
    original_variables = nnx.variables(model)
    new_variables = rl_utils.put_params_on_memory_kind(
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

  def close(self):
    for m in self._buffered_train_metrics + self._buffered_eval_metrics:
      self._log_metrics(m)
    self.actor_trainer.close()
    if getattr(self, "critic_trainer", None):
      self.critic_trainer.close()

  def _log_metrics(self, metrics_buffer: MetricsBuffer) -> None:
    """Log metrics."""
    for metric_name, (value, op) in metrics_buffer.metrics.items():
      if isinstance(value[0], str):
        continue  # jax.monitoring does not support string values.
      if op is None:
        self._rl_metrics_logger.log(  # pytype: disable=wrong-arg-types
            metric_name, value, metrics_buffer.mode, metrics_buffer.global_steps
        )
      else:
        self._rl_metrics_logger.log(
            metric_name,
            op(value),
            metrics_buffer.mode,
            metrics_buffer.global_steps,
        )
    if self._external_metrics_logger is not None:
      self._external_metrics_logger(metrics_buffer)

  def with_external_metrics_logger(
      self, external_metrics_logger: Callable[[MetricsBuffer], None]
  ):
    self._external_metrics_logger = external_metrics_logger
    return self

  def buffer_metrics(
      self,
      metrics: MetricsT,
      mode: Mode = Mode.TRAIN,
  ) -> None:
    """Buffers rl metrics to be logged.

    Actual logging will happen when global steps are incremented.

    Args:
      metrics: A dictionary mapping metric names to a tuple containing the
        metric value and an optional aggregation function.
      mode: The mode of the workload, either TRAIN or EVAL.
    """
    if mode == Mode.TRAIN:
      buffered_metrics = self._buffered_train_metrics
    else:
      buffered_metrics = self._buffered_eval_metrics

    if not buffered_metrics:
      buffered_metrics.append(MetricsBuffer(self.global_steps, mode=str(mode)))

    # Global steps are incremented, log the previous metrics.
    if self._buffered_train_metrics[0].global_steps != self.global_steps:
      self._buffered_train_metrics.append(
          MetricsBuffer(self.global_steps, mode=str(mode))
      )
      for m in [self._buffered_train_metrics.pop(0)] + (
          [self._buffered_eval_metrics.pop(0)]
          if self._buffered_eval_metrics
          else []
      ):
        self._log_metrics(m)

    cur_metrics = buffered_metrics[-1]
    for metric_name, (value, op) in metrics.items():
      if metric_name not in cur_metrics.metrics:
        cur_metrics.metrics[metric_name] = (
            [value],
            op,
        )
      else:
        cur_metrics.metrics[metric_name][0].append(value)

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

  def generate(
      self,
      prompts: list[str] | list[list[dict[str, str]]],
      apply_chat_template: bool = False,
      mode: Mode = Mode.TRAIN,
      micro_batch_size: int | None = None,
  ) -> base_rollout.RolloutOutput:
    """Generates text from the given prompts.

    Args:
      prompts: A list of prompts to generate text from. If `apply_chat_template`
        is True, this should be a list of conversations (each a list of
        dictionaries with 'role' and 'content'). Otherwise, it should be a list
        of strings.
      apply_chat_template: Whether to apply chat template to the prompts.
      mode: The mode of rollout, either TRAIN or EVAL.
      micro_batch_size: The micro-batch size for generation. If None, no
        micro-batching is performed.

    Returns:
      A `RolloutOutput` object containing the generated text and other info.
    """
    if apply_chat_template:
      if self.tokenizer is None:
        raise ValueError("Tokenizer must be initialized to use chat templates.")
      string_prompts = [
          self.tokenizer.apply_chat_template(
              prompt,  # pytype: disable=wrong-arg-types
              add_generation_prompt=True,
              tokenize=False,
              enable_thinking=False,
          )
          for prompt in prompts
      ]
    else:
      string_prompts = prompts  # pytype: disable=annotation-type-mismatch

    if len(string_prompts) == 0:
      raise ValueError("Cannot generate from an empty list of prompts.")
    micro_batch_size = micro_batch_size or len(string_prompts)

    with self.cluster_config.role_to_mesh[Role.ROLLOUT]:
      model = self.rollout.model()
      self._maybe_load_model_from_cpu(model, Role.ROLLOUT)
      if self.cluster_config.offload_to_cpu:
        self.rollout.update_params(nnx.state(model))

      if isinstance(self.cluster_config.rollout_config, dict):
        rollout_config = self.cluster_config.rollout_config[mode]
      else:
        rollout_config = self.cluster_config.rollout_config
      outputs = [
          self.rollout.generate(string_prompts[s], rollout_config)
          for s in rl_utils.chunk_slices_by_size(
              stop=len(string_prompts), step=micro_batch_size
          )
      ]
      self._maybe_offload_model_to_cpu(model, Role.ROLLOUT)
      if self.cluster_config.offload_to_cpu:
        self.rollout.update_params(nnx.state(model))

    texts = list(itertools.chain.from_iterable(out.text for out in outputs))

    logprobs = None
    if outputs[0].logprobs is not None:
      logprobs = list(
          itertools.chain.from_iterable(out.logprobs for out in outputs)
      )

    logits = None
    if isinstance(outputs[0].logits, jnp.ndarray):
      logits = jnp.concatenate([out.logits for out in outputs], axis=0)

    return base_rollout.RolloutOutput(
        text=texts,
        logits=logits,
        tokens=jnp.concatenate([out.tokens for out in outputs], axis=0),
        left_padded_prompt_tokens=jnp.concatenate(
            [out.left_padded_prompt_tokens for out in outputs], axis=0
        ),
        logprobs=logprobs,
    )

  def get_ref_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
      micro_batch_size: int | None = None,
      completion_mask: jax.Array | None = None,
  ) -> jax.Array:
    """Gets the per-token logps of the reference model."""
    batch_size = prompt_tokens.shape[0]
    if batch_size == 0:
      raise ValueError(
          "Cannot get reference log probabilities from an empty batch."
      )
    micro_batch_size = micro_batch_size or batch_size

    reference_mesh = self.cluster_config.role_to_mesh[Role.REFERENCE]

    dest_prompt_tokens = rl_utils.maybe_move(prompt_tokens, reference_mesh)
    dest_completion_tokens = rl_utils.maybe_move(
        completion_tokens, reference_mesh
    )
    with reference_mesh:
      self._maybe_load_model_from_cpu(
          self.inference_worker.get_model("reference"), Role.REFERENCE
      )
      outs = []
      for batch_slice in rl_utils.chunk_slices_by_size(
          stop=batch_size, step=micro_batch_size
      ):
        outs.append(
            self.inference_worker.get_ref_per_token_logps(
                dest_prompt_tokens[batch_slice],
                dest_completion_tokens[batch_slice],
                pad_id,
                eos_id,
                completion_mask=None
                if completion_mask is None
                else completion_mask[batch_slice],
            )
        )
      ref_per_token_logps = jnp.concatenate(outs, axis=0)
      self._maybe_offload_model_to_cpu(
          self.inference_worker.get_model("reference"), Role.REFERENCE
      )
      return ref_per_token_logps

  def get_old_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      micro_batch_size: int | None = None,
      completion_mask: jax.Array | None = None,
  ) -> jax.Array:
    """Gets the per-token logps of the current policy model."""
    batch_size = prompt_tokens.shape[0]
    if batch_size == 0:
      raise ValueError("Cannot get old log probabilities from an empty batch.")
    micro_batch_size = micro_batch_size or batch_size

    with self.cluster_config.role_to_mesh[Role.ROLLOUT]:
      model = self.rollout.model()
      self._maybe_load_model_from_cpu(model, Role.ROLLOUT)
      if self.cluster_config.offload_to_cpu:
        self.rollout.update_params(nnx.state(model))
      outs = []
      for batch_slice in rl_utils.chunk_slices_by_size(
          stop=batch_size, step=micro_batch_size
      ):
        outs.append(
            self.rollout.get_per_token_logps(
                prompt_tokens[batch_slice],
                completion_tokens[batch_slice],
                completion_mask=None
                if completion_mask is None
                else completion_mask[batch_slice],
            )
        )
      per_token_logps = jnp.concatenate(outs, axis=0)
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
          if sft_utils.is_lora_enabled(self.actor_trainer.model)
          else nnx.Param,
      )
      src_filtered_params = nnx.state(self.actor_trainer.model, filter_types)
      self.rollout.update_params(src_filtered_params, filter_types)

    # sync weights marks the end of a full batch, so increment the global steps.
    self.global_steps += 1

  def get_values(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
      completion_mask: jax.Array | None = None,
  ) -> jax.Array:
    with self.cluster_config.role_to_mesh[Role.CRITIC]:
      return self.inference_worker.get_values(
          prompt_tokens,
          completion_tokens,
          pad_id,
          eos_id,
          completion_mask=completion_mask,
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
