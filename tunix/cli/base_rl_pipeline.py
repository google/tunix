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
"""Base class for RL pipelines."""

import abc
from collections.abc import MutableMapping
import dataclasses
import importlib
import os
import types
from typing import Any

from absl import flags
from absl import logging
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.cli import config
from tunix.cli.utils import data as data_lib
from tunix.cli.utils import model as model_lib
from tunix.examples.data import math_dataset as example_data
from tunix.models import naming
from tunix.models.gemma import model as gemma_lib
from tunix.models.qwen2 import model as qwen_lib
from tunix.perf import export as perf_export
from tunix.perf import metrics as perf_metrics
from tunix.perf.experimental import export as perf_export_v2
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.utils import create_critic_model
from tunix.rl.rollout import base_rollout
from tunix.utils import mesh as mesh_lib

_PATHWAYS_BNS = flags.DEFINE_string(
    "pathways_bns", None, "BNS address of the Pathways server."
)


class BasePipeline(abc.ABC, config.HyperParameters):
  """Base class for RL pipelines."""

  def __init__(self, argv: list[str], **kwargs):
    self.data_module: types.ModuleType | None = None
    super().__init__(argv, **kwargs)

  @property
  @abc.abstractmethod
  def _default_training_mode(self):
    pass

  # ------------------------------------------------------------------
  # Mesh
  # ------------------------------------------------------------------
  _ROLE_TO_MODEL_KEY = {
      rl_cluster_lib.Role.ACTOR: "actor_model_config",
      rl_cluster_lib.Role.CRITIC: "critic_model_config",
      rl_cluster_lib.Role.REFERENCE: "reference_model_config",
      rl_cluster_lib.Role.REWARD: "reward_model_config",
      rl_cluster_lib.Role.ROLLOUT: "rollout_model_config",
  }
  _SPLIT_ROLE_ALIASES = {
      "actor": rl_cluster_lib.Role.ACTOR,
      "critic": rl_cluster_lib.Role.CRITIC,
      "reference": rl_cluster_lib.Role.REFERENCE,
      "reward": rl_cluster_lib.Role.REWARD,
      "rollout": rl_cluster_lib.Role.ROLLOUT,
  }

  def _is_agentic_mode(self, mode: str) -> bool:
    """Checks if the given mode is agentic."""
    return mode.startswith("agentic")

  def _resolve_split_role(self, role_name: str) -> rl_cluster_lib.Role:
    """Resolves a split role name to an alias.

    Args:
      role_name: The role name to resolve.

    Returns:
      The role's alias.

    Raises:
      ValueError: If the role name is not valid.
    """
    normalized = role_name.strip().lower()
    if normalized not in self._SPLIT_ROLE_ALIASES:
      valid_roles = sorted(self._SPLIT_ROLE_ALIASES)
      raise ValueError(
          f"Unknown role name {role_name!r}. Expected one of {valid_roles}."
      )
    return self._SPLIT_ROLE_ALIASES[normalized]

  def _get_same_mesh_as_map(
      self,
  ) -> dict[rl_cluster_lib.Role, rl_cluster_lib.Role]:
    """Returns a mapping from a role to the role whose mesh it should share."""
    same_mesh_as = {}
    for role, model_key in self._ROLE_TO_MODEL_KEY.items():
      model_cfg = self.config.get(model_key, {}) or {}
      if not isinstance(model_cfg, dict):
        continue
      target_name = model_cfg.get("same_mesh_as")
      if target_name is None:
        continue
      if not isinstance(target_name, str):
        raise TypeError(f"Expected {model_key}"
                        "same_mesh_as to be None or str but got"
                        f"{type(target_name).__name__}")
      target_role = self._resolve_split_role(str(target_name))
      if role == rl_cluster_lib.Role.ACTOR:
        raise ValueError("Actor must own its mesh.")
      same_mesh_as[role] = target_role

    return same_mesh_as

  def _is_role_active(self, role: rl_cluster_lib.Role) -> bool:
    """Checks if the given role is active (i.e. used)."""
    if role in (
        rl_cluster_lib.Role.ACTOR,
        rl_cluster_lib.Role.REFERENCE,
        rl_cluster_lib.Role.ROLLOUT,
    ):
      return True
    model_key = self._ROLE_TO_MODEL_KEY[role]
    return model_key in self.config

  def _resolve_mesh_owners(
      self,
  ) -> dict[rl_cluster_lib.Role, rl_cluster_lib.Role]:
    """Resolves the mesh owner for each role.

    Returns:
      A mapping from a role to the role that owns its mesh.
    """

    same_mesh_as = self._get_same_mesh_as_map()
    base_owners = {}
    for role, model_key in self._ROLE_TO_MODEL_KEY.items():
      if not self._is_role_active(role) and role not in same_mesh_as:
        continue

      # We default to the actor as the mesh owner, and override it if the role
      # has an explicit mesh config.
      base_owners[role] = rl_cluster_lib.Role.ACTOR

      model_config = self.config.get(model_key, {})
      has_mesh = (
          isinstance(model_config, MutableMapping)
          and bool(model_config.get("mesh"))
      )
      if has_mesh:
        base_owners[role] = role

    def resolve_owner(
        role: rl_cluster_lib.Role,
        seen: set[rl_cluster_lib.Role],
    ) -> rl_cluster_lib.Role:
      """Resolves the mesh owner for the given role.

      Args:
        role: The role to resolve.
        seen: The set of roles already seen in the current resolution chain.

      Returns:
        The owner of the given role's mesh.
      """
      if role in seen:
        raise ValueError("same_mesh_as contains a cycle.")
      if role not in same_mesh_as:
        return base_owners[role]
      seen.add(role)
      target_role = same_mesh_as[role]
      if target_role not in base_owners:
        raise ValueError(
            f"Role {target_role.value!r} is not active in this config."
        )
      return resolve_owner(target_role, seen)

    role_to_owner = {}
    for role, model_key in self._ROLE_TO_MODEL_KEY.items():
      if role not in base_owners:
        continue

      model_config = self.config.get(model_key, {})
      has_mesh = isinstance(model_config, MutableMapping) and bool(
          model_config.get("mesh")
      )
      if role in same_mesh_as:
        if has_mesh:
          raise ValueError(
              f"{model_key}.mesh is specified, so it must own a separate mesh "
              "and cannot also use same_mesh_as."
          )
      else:
        role_to_owner[role] = resolve_owner(role, set())
        continue
      role_to_owner[role] = resolve_owner(role, set())
    return role_to_owner

  def create_role_to_mesh(self):
    """Builds the role-to-mesh mapping for execution.

    Any role with an explicit ``*.mesh`` config gets a dedicated device slice.
    Roles without a mesh share the actor mesh by default, or can point at
    another role via ``same_mesh_as``.

    All mesh owners participating in the same allocation pass must agree on
    one ``mesh.allocation_policy`` value. That policy is then passed to the
    mesh allocator so users can choose between compact packing and
    performance-oriented cubical packing from config.

    Returns:
      A mapping from logical role to the concrete JAX mesh it should use.

    Raises:
      ValueError: If mesh ownership resolution is invalid or if mesh owners
        request conflicting allocation policies.
    """
    devices = list(jax.devices())
    role_to_owner = self._resolve_mesh_owners()
    owner_order = []
    for role in self._ROLE_TO_MODEL_KEY:
      if role not in role_to_owner:
        continue
      owner = role_to_owner[role]
      if owner not in owner_order:
        owner_order.append(owner)

    mesh_requirements = []
    allocation_policy = None
    for owner in owner_order:
      model_key = self._ROLE_TO_MODEL_KEY[owner]
      axis_shapes, _ = self.parse_mesh_config(model_key)
      owner_policy = self._parse_mesh_allocation_policy(model_key)
      if allocation_policy is None:
        allocation_policy = owner_policy
      elif owner_policy != allocation_policy:
        raise ValueError(
            "All owned meshes must use the same mesh.allocation_policy, got "
            f"{allocation_policy!r} and {owner_policy!r}."
        )
      mesh_requirements.append((model_key, int(np.prod(axis_shapes))))

    allocated_devices = mesh_lib.allocate_named_mesh_device_slices(
        mesh_requirements,
        devices=devices,
        allocation_policy=allocation_policy
        or mesh_lib.normalize_allocation_policy(None),
    )

    owner_to_mesh = {}
    for owner in owner_order:
      model_key = self._ROLE_TO_MODEL_KEY[owner]
      axis_shapes, axis_names = self.parse_mesh_config(model_key)
      assigned_devices = allocated_devices[model_key]
      owner_to_mesh[owner] = mesh_lib.create_mesh(
          axis_shapes, axis_names, devices=assigned_devices
      )
    return {role: owner_to_mesh[owner] for role, owner in role_to_owner.items()}

  # ------------------------------------------------------------------
  # Rollout config
  # ------------------------------------------------------------------
  def create_rollout_config(
      self,
      role_to_mesh: dict[rl_cluster_lib.Role, jax.sharding.Mesh] | None = None,
  ) -> base_rollout.RolloutConfig:
    """Build RolloutConfig from rollout config section in YAML.

    Standard mode: pass rollout_config fields through with kv_cache_size =
    max_prompt_length + total_generation_steps + 256.

    Agentic mode: same base. Same kv_cache_size calculation.

    Engine-specific extras (sglang_jax_config, vllm_config) are also applied.

    Args:
      role_to_mesh: Optional mapping from logical role to JAX mesh.

    Returns:
      The constructed RolloutConfig.
    """
    rollout_cfg = self._config_mapping("rollout_config")
    mode = self._config_string("training_mode", self._default_training_mode)
    engine = self._config_string("rollout_engine", "vanilla")

    valid_fields = {
        f.name for f in dataclasses.fields(base_rollout.RolloutConfig)
    }

    # Base pass-through (same as original create_rollout_config)
    filtered = {k: v for k, v in rollout_cfg.items() if k in valid_fields}
    if "total_generation_steps" in rollout_cfg:
      filtered["max_tokens_to_generate"] = rollout_cfg["total_generation_steps"]

    max_prompt = rollout_cfg.get("max_prompt_length", 0)
    max_response = rollout_cfg.get("total_generation_steps", 0)

    kv_cache_size = 0
    if self._is_agentic_mode(mode):
      agentic_cfg = self._config_mapping(f"{mode}_config")
      kv_cache_size = max_prompt + max_response + 256
      filtered["kv_cache_size"] = kv_cache_size
      logging.info("kv_cache_size: %d", kv_cache_size)

      max_running_requests = agentic_cfg.get("max_concurrency", 16)
    else:
      rl_algm_cfg = self._config_mapping(f"{mode}_config")
      # Standard: kv_cache_size = max_prompt + max_response + 256
      if max_prompt and max_response:
        kv_cache_size = max_prompt + max_response + 256
        filtered["kv_cache_size"] = kv_cache_size
      # Defaults to global batch size * num_generations to allow full
      # concurrency.
      max_running_requests = self.config.get("batch_size", 1) * rl_algm_cfg.get(
          "num_generations", 1
      )

    # Engine-specific extras
    extra = self._rollout_engine_extra(
        engine,
        kv_cache_size,
        max_running_requests,
        role_to_mesh=role_to_mesh,
    )
    filtered.update({k: v for k, v in extra.items() if k in valid_fields})
    return base_rollout.RolloutConfig(**filtered)

  def _rollout_engine_extra(
      self,
      engine: str,
      kv_cache_size: int,
      max_running_requests: int,
      role_to_mesh: dict[rl_cluster_lib.Role, jax.sharding.Mesh] | None = None,
  ) -> dict[str, Any]:
    """Return engine-specific RolloutConfig fields for agentic mode."""
    model_id = self._config_mapping("actor_model_config").get("model_id", "")

    if engine == "sglang_jax":
      sg = self._config_mapping("sglang_jax_config")
      return dict(
          rollout_sglang_jax_model_version=sg.get("model_version", model_id),
          rollout_sglang_jax_mem_fraction_static=sg.get(
              "mem_fraction_static", 0.8
          ),
          rollout_sglang_jax_init_with_random_weights=sg.get(
              "init_with_random_weights", True
          ),
          rollout_sglang_jax_disable_radix_cache=sg.get(
              "disable_radix_cache", True
          ),
          rollout_sglang_jax_enable_deterministic_sampling=sg.get(
              "enable_deterministic_sampling", False
          ),
          rollout_sglang_jax_chunked_prefill_size=sg.get(
              "chunked_prefill_size", 2048
          ),
          rollout_sglang_jax_max_running_requests=sg.get(
              "max_running_requests",
              max_running_requests,
          ),
          rollout_sglang_jax_page_size=sg.get("page_size", 128),
          rollout_sglang_jax_use_sort_for_toppk_minp=sg.get(
              "use_sort_for_toppk_minp", False
          ),
      )

    if engine == "vllm":
      vllm = self._config_mapping("vllm_config")
      if role_to_mesh is None:
        raise ValueError(
            "role_to_mesh must be provided for vllm rollout config."
        )
      rollout_shape = role_to_mesh[rl_cluster_lib.Role.ROLLOUT].devices.shape
      rollout_cfg = self._config_mapping("rollout_config")
      max_num_seqs = rollout_cfg.get(
          "rollout_vllm_max_num_seqs",
          vllm.get("max_num_seqs", 768),
      )
      max_batched_tokens = rollout_cfg.get(
          "rollout_vllm_max_num_batched_tokens",
          vllm.get(
              "max_num_batched_tokens",
              (max_num_seqs * kv_cache_size) // 4,  # pyrefly: ignore[unsupported-operation]
          ),
      )
      submission_threshold = rollout_cfg.get(
          "rollout_vllm_server_mode_submission_threshold",
          vllm.get("server_mode_submission_threshold", 0),
      )
      submission_timeout_s = rollout_cfg.get(
          "rollout_vllm_server_mode_submission_timeout_s",
          vllm.get("server_mode_submission_timeout_s", 0.0),
      )
      os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
      return dict(
          rollout_vllm_model_version=vllm.get("model_version", model_id),
          rollout_vllm_hbm_utilization=vllm.get("hbm_utilization", 0.4),
          rollout_vllm_tpu_backend_type=vllm.get("tpu_backend_type", "jax"),
          rollout_vllm_server_mode=vllm.get("server_mode", True),
          rollout_vllm_server_mode_submission_threshold=submission_threshold,
          rollout_vllm_server_mode_submission_timeout_s=submission_timeout_s,
          rollout_vllm_async_scheduling=vllm.get("async_scheduling", True),
          tensor_parallel_size=vllm.get(
              "tensor_parallel_size",
              rollout_shape[1] if len(rollout_shape) > 1 else 1,
          ),
          data_parallel_size=vllm.get("data_parallel_size", rollout_shape[0]),
          rollout_vllm_max_num_seqs=max_num_seqs,
          rollout_vllm_max_num_batched_tokens=max_batched_tokens,
          rollout_vllm_kwargs=vllm.get(
              "kwargs",
              {
                  "kv_cache_metrics": True,
                  "disable_log_stats": False,
                  "enable_prefix_caching": True,
              },
          ),
      )

    return {}

  def create_cluster_config(
      self,
      *,
      role_to_mesh: dict[rl_cluster_lib.Role, jax.sharding.Mesh],
      rollout_config: base_rollout.RolloutConfig | None = None,
  ):
    """Create ClusterConfig from rollout, optimizer, and rollout engine config sections in YAML.

    Args:
      role_to_mesh: Mapping from role to JAX mesh.
      rollout_config: Optional RolloutConfig. If not provided, will be created
        from YAML config.

    Returns:
      The constructed ClusterConfig.
    """
    if rollout_config is None:
      rollout_config = self.create_rollout_config(role_to_mesh=role_to_mesh)
    return rl_cluster_lib.ClusterConfig(
        role_to_mesh=role_to_mesh,
        rollout_engine=self._config_string("rollout_engine"),
        offload_to_cpu=self._config_bool("offload_to_cpu"),
        training_config=self.create_rl_training_config(),
        rollout_config=rollout_config
    )

  def create_rl_training_config(self):
    base_key = "rl_training_config"
    constructed_rl_training_config = self.obtain_training_config_dict(base_key)

    base_config = self._config_mapping(base_key)
    if base_config.get("actor_optimizer_config"):
      constructed_rl_training_config["actor_optimizer"] = self.create_optimizer(
          base_key, "actor_optimizer_config"
      )
    if base_config.get("critic_optimizer_config"):
      constructed_rl_training_config["critic_optimizer"] = (
          self.create_optimizer(base_key, "critic_optimizer_config")
      )

    return rl_cluster_lib.RLTrainingConfig(**constructed_rl_training_config)

  def create_perf_config(self, cluster_config: rl_cluster_lib.ClusterConfig):
    perf_metrics_options = cluster_config.training_config.perf_metrics_options
    if not perf_metrics_options:
      return None

    perf_config = perf_metrics.PerfMetricsConfig()

    if perf_metrics_options.enable_perf_v1:
      custom_export_fn_path = perf_metrics_options.custom_export_fn_path
      if custom_export_fn_path:
        perf_config.custom_export_fn = self._get_function_from_path(
            custom_export_fn_path
        )
        if perf_config.custom_export_fn is None:
          raise ValueError(
              "Could not load custom export function from"
              f" {custom_export_fn_path}"
          )
      else:
        perf_config.custom_export_fn = (
            perf_export.PerfMetricsExport.from_cluster_config(cluster_config)
        )

    if perf_metrics_options.enable_perf_v2:
      custom_export_fn_path_v2 = perf_metrics_options.custom_export_fn_path_v2
      if custom_export_fn_path_v2:
        perf_config.custom_export_fn_v2 = self._get_function_from_path(
            custom_export_fn_path_v2
        )
        if perf_config.custom_export_fn_v2 is None:
          raise ValueError(
              "Could not load custom export function v2 from"
              f" {custom_export_fn_path_v2}"
          )
      else:
        perf_config.custom_export_fn_v2 = (
            perf_export_v2.PerfMetricsExport.from_cluster_config(
                cluster_config=cluster_config,
                enable_trace_writer=perf_metrics_options.enable_trace_writer,
                trace_dir=perf_metrics_options.trace_dir,
            ).export_metrics
        )
    return perf_config

  def create_rl_cluster(self, tokenizer) -> rl_cluster_lib.RLCluster:
    """Constructs an RLCluster by instantiating models and configurations.

    This method retrieves configurations from YAML and uses them to
    initialize the Reference, Actor, and optional Critic models.

    Args:
      tokenizer: The tokenizer instance to be passed into the RL cluster
      for processing text.

    Returns:
      An initialized reinforcement learning cluster.
    """

    role_to_mesh = self.create_role_to_mesh()
    rollout_config = self.create_rollout_config(role_to_mesh=role_to_mesh)
    reference_model_config = self._mutable_config_mapping(
        "reference_model_config"
    )
    actor_model_config = self._mutable_config_mapping("actor_model_config")
    tokenizer_config = self._config_mapping("tokenizer_config")
    # Should not use LoRA for reference model.
    if reference_model_config.get("lora_config"):
      logging.warning(
          "LoRA config is not supported for the reference model. Disabling"
          " LoRA."
      )
      del reference_model_config["lora_config"]
    reference_model, _ = model_lib.create_model(
        dict(reference_model_config),
        tokenizer_config,
        role_to_mesh[rl_cluster_lib.Role.REFERENCE],
    )
    if actor_model_config.get("lora_config", None):
      actor_model = model_lib.apply_lora_to_model(
          reference_model,
          role_to_mesh[rl_cluster_lib.Role.ACTOR],
          actor_model_config["lora_config"],
      )
    else:
      graph_def, params = nnx.split(reference_model)
      actor_dtype_str = actor_model_config.get("load_dtype")
      if actor_dtype_str:
        try:
          actor_dtype = getattr(jnp, actor_dtype_str)
        except AttributeError:
          raise ValueError(
              f"Invalid load_dtype: {actor_dtype_str}. Must be a valid"
              " jax.numpy type."
          )
        params = jax.tree.map(lambda x: x.astype(actor_dtype), params)
      else:
        params = jax.tree.map(jnp.copy, params)
      actor_model = nnx.merge(graph_def, params)

    critic_model = None
    critic_model_config = self._mutable_config_mapping("critic_model_config")
    if critic_model_config:
      critic_model, _ = model_lib.create_model(
          dict(critic_model_config),
          tokenizer_config,
          role_to_mesh[rl_cluster_lib.Role.CRITIC],
      )

      if critic_model_config.get("lora_config", None):
        critic_model = model_lib.apply_lora_to_model(
            critic_model,
            role_to_mesh[rl_cluster_lib.Role.CRITIC],
            critic_model_config["lora_config"],
        )

      rngs = nnx.Rngs(
          params=jax.random.key(critic_model_config.get("rng_seed", 0))
      )
    
      model_name = critic_model_config.get('model_name')
      config_category = naming.ModelNaming(model_name=model_name).model_config_category
      if hasattr(critic_model, "lm_head"):
          critic_model = create_critic_model(critic_model, rngs=rngs)
      elif config_category in ("qwen2", "qwen3"):
        critic_model = qwen_lib.QwenWithScoreHead(critic_model, rngs=rngs)
      elif config_category in ("gemma", "gemma3", "gemma4"):
        critic_model = gemma_lib.GemmaWithScoreHead(critic_model, rngs=rngs)
      else:
        raise ValueError(f"Unsupported critic model category: {config_category}")

    cluster_config = self.create_cluster_config(
        role_to_mesh=role_to_mesh,
        rollout_config=rollout_config,
    )
    perf_config = self.create_perf_config(cluster_config)
    return rl_cluster_lib.RLCluster(
        actor=actor_model,
        critic=critic_model,
        reference=reference_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
        perf_config=perf_config,
    )

  def _compute_max_steps(self, dataset):
    """Compute max_steps from the dataset length and RL training config.

    Args:
      dataset: The dataset to compute max_steps from.

    Returns:
      The computed max_steps.
    """
    rl_training_config = self._mutable_config_mapping("rl_training_config")

    max_steps = rl_training_config.get("max_steps")
    try:
      dataset_length = len(dataset)
    except TypeError:
      dataset_length = None

      if not max_steps:
        raise ValueError(
            "max_steps must be specified since the"
            "dataset length cannot be determined."
        ) from None

    batch_size = self.config.get("batch_size", 1)
    num_batches = self.config.get("num_batches")
    if not num_batches and not dataset_length:
      raise ValueError(
          "num_batches is not specified and"
          "cannot be determined since the dataset"
          "length is unknown."
      )
    elif not num_batches:
      num_batches = dataset_length // batch_size
      self.config["num_batches"] = num_batches
      logging.info(
          "Dynamically computed num_batches=%d with batch_size=%d",
          num_batches,
          batch_size,
      )
    self.config["num_batches"] = num_batches
    num_train_epochs = self.config.get("num_train_epochs")
    if not num_train_epochs:
      num_train_epochs = 1

    train_fraction = self.config.get("train_fraction")
    if not train_fraction:
      train_fraction = 0.8
    elif train_fraction <= 0.0 or train_fraction > 1.0:
      logging.warning(
          "train_fraction %.2f out of expected range. Setting to 0.8",
          train_fraction,
      )
      train_fraction = 0.8

    allowed_max_steps = int(num_batches * num_train_epochs * train_fraction)
    if not max_steps:
      max_steps = allowed_max_steps
    elif max_steps > allowed_max_steps:
      raise ValueError(
          f"Maximum allowed value for max_steps is {allowed_max_steps}, but"
          f" {max_steps} is specified."
      )

    logging.info(
        "Dynamically computed max_steps=%d based on dataset length %d",
        max_steps,
        dataset_length,
    )

    return max_steps

  def compute_params(self, dataset):
    """Compute max_steps and update the RL training config.

    This method computes max_steps from the dataset length and RL training
    config, and updates the RL training config with the computed max_steps.
    It also updates the actor and critic optimizer configs with the computed
    max_steps.

    Args:
      dataset: The dataset to compute max_steps from.
    """
    rl_training_config = self._mutable_config_mapping("rl_training_config")
    max_steps = self._compute_max_steps(dataset)

    rl_training_config["max_steps"] = max_steps
    self._apply_optimizer_step_limits(
        rl_training_config, "actor_optimizer_config", max_steps
    )
    self._apply_optimizer_step_limits(
        rl_training_config, "critic_optimizer_config", max_steps
    )

  def _apply_optimizer_step_limits(
      self,
      rl_training_config: MutableMapping[str, Any],
      optimizer_key: str,
      max_steps: int,
  ):
    """Apply step limits to optimizer config if it exists."""

    opt: MutableMapping[str, Any] | None = None
    opt_value = rl_training_config.get(optimizer_key)

    if isinstance(opt_value, MutableMapping):
      opt = opt_value
    elif opt_value is not None:
      raise ValueError(f"rl_training_config.{optimizer_key} must be a dict.")

    if opt and not opt.get("decay_steps"):
      opt["decay_steps"] = max_steps
    if opt and not opt.get("warmup_steps"):
      warmup_ratio = self.config.get("warmup_ratio", 0.1)
      warmup_steps = self.config.get("warmup_steps", warmup_ratio * max_steps)
      opt["warmup_steps"] = warmup_steps

  # ------------------------------------------------------------------
  # Standard training
  # ------------------------------------------------------------------

  def _get_tokenizer(self):
    model_config = self.config.get("actor_model_config") or self.config.get(
        "model_config"
    )
    return model_lib.create_tokenizer(
        self.config["tokenizer_config"],
        self.config["tokenizer_config"]["tokenizer_path"],
        model_config=model_config,
    )

  def _get_data_module(
      self,
  ):
    if self.data_module is None:
      self.data_module = importlib.import_module(self.config["data_module"])
    return self.data_module

  def _get_dataset(self, tokenizer):
    apply_chat_template_to_dataset = self.config.get(
        "apply_chat_template_to_dataset"
    )
    if apply_chat_template_to_dataset is None:
      raise ValueError("apply_chat_template_to_dataset must be set.")

    if self.config.get("data_module", None):
      data_module = self._config_string("data_module")
      dataset = data_lib.get_dataset_from_module(
          data_module,
          tokenizer,
          apply_chat_template_to_dataset=apply_chat_template_to_dataset,
          **(self.config.get("data_config") or {}),
      )
    elif self.config["data_source"] == "local":
      dataset = example_data.create_dataset(
          data_source=self.config["data_source"],
          dataset=self.config["data_directory"],
          tokenizer=tokenizer,
          apply_chat_template_to_dataset=apply_chat_template_to_dataset,
      )
    elif self.config["data_source"] == "tfds":
      dataset = example_data.create_dataset(
          data_source=self.config["data_source"],
          dataset=self.config["dataset_name"],
          tfds_download=self.config["tfds_download"],
          split=self.config.get(  # pyrefly: ignore[bad-argument-type]
              "train_split", self.config.get("split", "train")
          ),
          apply_chat_template_to_dataset=apply_chat_template_to_dataset,
      )
    elif self.config["data_source"] == "huggingface":
      dataset = example_data.create_dataset(
          data_source=self.config["data_source"],
          dataset=self.config["dataset_name"],
          tokenizer=tokenizer,
          split=self.config.get(  # pyrefly: ignore[bad-argument-type]
              "train_split", self.config.get("split", "train")
          ),
          apply_chat_template_to_dataset=apply_chat_template_to_dataset,
      )
    else:
      raise ValueError(f"Unsupported data_source {self.config['data_source']}")

    return dataset

  # ------------------------------------------------------------------
  # Agentic helpers
  # ------------------------------------------------------------------

  def _create_chat_parser(self, tokenizer: Any) -> Any:
    """Instantiate a chat parser based on chat_parser_config.type."""
    from tunix.rl.agentic.parser.chat_template_parser import parser as chat_parser_lib  # pylint: disable=g-import-not-at-top

    chat_cfg = self._config_mapping("chat_parser_config")
    parser_type = chat_cfg.get("type", "default")
    if parser_type == "qwen":
      return chat_parser_lib.QwenChatTemplateParser(tokenizer)
    elif parser_type == "gemma4":
      return chat_parser_lib.Gemma4ChatTemplateParser(
          tokenizer,
          enable_thinking=chat_cfg.get("enable_thinking", False)
      )
    return chat_parser_lib.DefaultChatTemplateParser(tokenizer)

  def _load_class_from_path(self, dotted_path: str) -> type[Any]:
    """Load a Python class from a dotted module path.

    Args:
      dotted_path: Dotted module path to the class.

    Returns:
      The loaded Python class.
    """
    module_path, class_name = dotted_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)

  def _load_raw_dataset(self, tokenizer):
    """Load a raw grain.MapDataset from data_module.

    The module must expose ``create_dataset(**data_config) -> grain.MapDataset``
    and optionally a ``batch_fn`` used as ``custom_batch_fn``.

    Args:
      tokenizer: Tokenizer to use.

    Returns:
      A tuple (dataset, batch_fn) containing the loaded dataset and batch
      function.
    """
    data_module = (
        self._get_data_module()
        if self.config.get("data_module", None)
        else None
    )
    dataset = self._get_dataset(tokenizer)
    batch_fn = getattr(data_module, "batch_fn", None) if data_module else None
    return dataset, batch_fn

  def _setup_kubernetes(self) -> None:
    k8s_cfg = self._config_mapping("kubernetes_config")
    if not k8s_cfg:
      return
    os.environ["KUBECONFIG"] = k8s_cfg.get("kubeconfig", "~/.kube/config")
    os.environ["NODE_SELECTOR_KEY"] = k8s_cfg.get(
        "node_selector_key", "cloud.google.com/gke-nodepool"
    )
    os.environ["NODE_SELECTOR_VAL"] = k8s_cfg.get(
        "node_selector_val", "deepswe-cpu-pool"
    )
    try:
      from kubernetes import client as k8s_client_lib  # type: ignore[import-untyped]  # pylint: disable=g-import-not-at-top
      from kubernetes import config as k8s_config_lib  # type: ignore[import-untyped]  # pylint: disable=g-import-not-at-top

      k8s_config_lib.load_kube_config()
      k8s_client_lib.CoreV1Api()
    except Exception as e:  # pylint: disable=broad-except
      logging.warning("Kubernetes config loading failed: %s", e)

  # ------------------------------------------------------------------
  # Agentic training
  # ------------------------------------------------------------------
  @abc.abstractmethod
  def _run(self, mode: str):
    """Execute agentic training (DeepScaleR, DeepSWE, etc.)."""
    pass

  def run_trainer(self):
    """Dispatch to standard or agentic trainer based on training_mode."""
    mode = self.config.get("training_mode", self._default_training_mode)
    self._run(mode=mode)


def setup_jax_pathways(pathways_bns: str):
  """Sets up Jax with Pathways."""
  flags.FLAGS.pathways_ifrt = True
  jax.config.update("jax_xla_backend", "pathways")
  jax.config.update("jax_backend_target", pathways_bns)


def setup_pathways_on_cloud():
  import pathwaysutils  # type: ignore[import-not-found,import-untyped]  # pytype: disable=import-error  # pyright: ignore[reportMissingImports]  # pylint: disable=g-import-not-at-top

  pathwaysutils.initialize()
