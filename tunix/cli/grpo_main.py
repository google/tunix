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

"""Main entry point for GRPO training (standard and agentic).

Set ``training_mode: "grpo"`` (default) for standard single-turn GRPO, or
``training_mode: "agentic_grpo"`` for agentic multi-turn GRPO (DeepScaleR,
DeepSWE, etc.).

Usage::

    # Standard GRPO
    python -m tunix.cli.grpo_main examples/rl/grpo/gsm8k/configs/gemma2_2b.yaml

    # Agentic GRPO — DeepScaleR
    bash examples/deepscaler/run_deepscaler_disagg.sh

    # Agentic GRPO — DeepSWE
    python -m tunix.cli.grpo_main examples/deepswe/configs/qwen3_32b.yaml
"""

import collections
import dataclasses
import importlib
import os
from typing import Any

from absl import app
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
from tunix.perf import export as perf_export
from tunix.perf import metrics as perf_metrics
from tunix.perf.experimental import export as perf_export_v2
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo import grpo_learner
from tunix.rl.rollout import base_rollout

GrpoConfig = grpo_learner.GrpoConfig

_PATHWAYS_BNS = flags.DEFINE_string(
    "pathways_bns", None, "BNS address of the Pathways server."
)


class GrpoPipeline(config.HyperParameters):
  """Runs standard GRPO or agentic GRPO depending on ``training_mode``.

  ``training_mode: "grpo"`` (default) — standard single-turn GRPO using
  GrpoLearner.  All existing YAML configs continue to work unchanged.

  ``training_mode: "agentic_grpo"`` — multi-turn agentic GRPO using
  GRPOLearner.  Additional config sections are recognised:

  * ``agentic_grpo_config``: GRPOConfig fields (num_generations, beta, …)
    plus ``max_turns``, ``context_ratio``, ``per_turn_timeout_secs``.
  * role-specific ``*_model_config.mesh``: any role with an explicit mesh gets
    its own device slice; omitted meshes share the actor mesh by default.
  * role-specific ``same_mesh_as``: optional mesh sharing like
    ``reference_model_config.same_mesh_as: actor``.
  * ``sglang_jax_config`` / ``vllm_config``: engine-specific rollout params.
  * ``chat_parser_config.type``: ``"default"`` or ``"qwen"``.
  * ``agent_class_path`` / ``env_class_path``: dotted Python paths to load
    agent and env classes dynamically.
  * ``data_module``: dotted module path; the module must expose
    ``create_dataset(**data_config) -> grain.MapDataset`` and optionally a
    ``batch_fn`` used as ``custom_batch_fn`` in post_init_dataset.
  * ``kubernetes_config``: optional Kubernetes env-var and kube-config setup.
  """

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

  def _resolve_split_role(self, role_name: str) -> rl_cluster_lib.Role:
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
    same_mesh_as = {}
    for role, model_key in self._ROLE_TO_MODEL_KEY.items():
      model_cfg = self.config.get(model_key, {}) or {}
      target_name = model_cfg.get("same_mesh_as")
      if target_name is None:
        continue
      target_role = self._resolve_split_role(str(target_name))
      if role == rl_cluster_lib.Role.ACTOR:
        raise ValueError("Actor must own its mesh.")
      same_mesh_as[role] = target_role

    return same_mesh_as

  def _is_role_active(self, role: rl_cluster_lib.Role) -> bool:
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
    same_mesh_as = self._get_same_mesh_as_map()
    base_owners = {}
    for role, model_key in self._ROLE_TO_MODEL_KEY.items():
      if not self._is_role_active(role) and role not in same_mesh_as:
        continue
      has_mesh = bool(self.config.get(model_key, {}).get("mesh"))
      base_owners[role] = (
          role
          if role == rl_cluster_lib.Role.ACTOR or has_mesh
          else rl_cluster_lib.Role.ACTOR
      )

    def resolve_owner(
        role: rl_cluster_lib.Role,
        seen: set[rl_cluster_lib.Role],
    ) -> rl_cluster_lib.Role:
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
      has_mesh = bool(self.config.get(model_key, {}).get("mesh"))
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

  def _create_role_to_mesh(self):
    devices = list(jax.devices())
    role_to_owner = self._resolve_mesh_owners()
    owner_order = []
    for role in self._ROLE_TO_MODEL_KEY:
      if role not in role_to_owner:
        continue
      owner = role_to_owner[role]
      if owner not in owner_order:
        owner_order.append(owner)

    owner_to_mesh = {}
    owner_to_device_slice = {}
    device_offset = 0
    for owner in owner_order:
      model_key = self._ROLE_TO_MODEL_KEY[owner]
      axis_shapes, _ = self._parse_mesh_config(model_key)
      required_devices = int(np.prod(axis_shapes))
      next_offset = device_offset + required_devices
      if next_offset > len(devices):
        raise ValueError(
            f"Mesh allocation requires {next_offset} devices after allocating"
            f" {model_key}, but only {len(devices)} are available."
        )
      assigned_devices = devices[device_offset:next_offset]
      owner_to_device_slice[owner] = assigned_devices
      owner_to_mesh[owner] = self.create_mesh(
          model_key, devices=assigned_devices
      )
      device_offset = next_offset

    if device_offset < len(devices):
      logging.warning(
          "Mesh allocation used %d of %d devices; %d devices remain unused.",
          device_offset,
          len(devices),
          len(devices) - device_offset,
      )
    logging.info(
        "Mesh device allocation: %s",
        {
            self._ROLE_TO_MODEL_KEY[owner]: len(owner_to_device_slice[owner])
            for owner in owner_order
        },
    )
    return {role: owner_to_mesh[owner] for role, owner in role_to_owner.items()}

  def create_role_to_mesh(self):
    """Build role→mesh mapping.

    Any role with an explicit ``*.mesh`` config gets a dedicated device slice.
    Roles without a mesh share the actor mesh by default, or can point at
    another role via ``same_mesh_as``.
    """
    return self._create_role_to_mesh()

  # ------------------------------------------------------------------
  # Rollout config
  # ------------------------------------------------------------------

  def create_rollout_config(
      self,
      role_to_mesh: dict[rl_cluster_lib.Role, jax.sharding.Mesh] | None = None,
  ) -> base_rollout.RolloutConfig:
    """Build RolloutConfig from YAML.

    Standard mode: pass rollout_config fields through with kv_cache_size =
    max_prompt_length + total_generation_steps + 256.

    Agentic mode: same base, but multi-turn KV cache =
    max_prompt + total_generation_steps * context_ratio * max_turns.
    Engine-specific extras (sglang_jax_config, vllm_config) are also applied.
    """
    rollout_cfg = self.config["rollout_config"]
    mode = self.config.get("training_mode", "grpo")
    engine = self.config.get("rollout_engine", "vanilla")

    valid_fields = {
        f.name for f in dataclasses.fields(base_rollout.RolloutConfig)
    }

    # Base pass-through (same as original create_rollout_config)
    filtered = {k: v for k, v in rollout_cfg.items() if k in valid_fields}
    if "total_generation_steps" in rollout_cfg:
      filtered["max_tokens_to_generate"] = rollout_cfg["total_generation_steps"]

    max_prompt = rollout_cfg.get("max_prompt_length", 0)
    max_response = rollout_cfg.get("total_generation_steps", 0)

    if mode == "agentic_grpo":
      agentic_cfg = self.config.get("agentic_grpo_config", {})
      max_turns = agentic_cfg.get("max_turns", 1)
      context_ratio = agentic_cfg.get("context_ratio", 1)
      if max_turns > 1:
        kv_cache_size = max_prompt + max_response * context_ratio * max_turns
      else:
        kv_cache_size = max_prompt + max_response + 256
      filtered["kv_cache_size"] = kv_cache_size
      logging.info("kv_cache_size: %d", kv_cache_size)

      # Engine-specific extras
      extra = self._agentic_engine_extra(
          engine,
          kv_cache_size,
          agentic_cfg,
          role_to_mesh=role_to_mesh,
      )
      filtered.update({k: v for k, v in extra.items() if k in valid_fields})
    else:
      # Standard: kv_cache_size = max_prompt + max_response + 256
      if max_prompt and max_response:
        filtered["kv_cache_size"] = max_prompt + max_response + 256

    return base_rollout.RolloutConfig(**filtered)

  def _agentic_engine_extra(
      self,
      engine: str,
      kv_cache_size: int,
      agentic_cfg: dict,
      role_to_mesh: dict[rl_cluster_lib.Role, jax.sharding.Mesh] | None = None,
  ) -> dict:
    """Return engine-specific RolloutConfig fields for agentic mode."""
    model_id = self.config.get("actor_model_config", {}).get("model_id", "")

    if engine == "sglang_jax":
      sg = self.config.get("sglang_jax_config", {})
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
              agentic_cfg.get("max_concurrency", 768),
          ),
          rollout_sglang_jax_page_size=sg.get("page_size", 128),
          rollout_sglang_jax_use_sort_for_toppk_minp=sg.get(
              "use_sort_for_toppk_minp", False
          ),
      )

    if engine == "vllm":
      vllm = self.config.get("vllm_config", {})
      if role_to_mesh is None:
        raise ValueError(
            "role_to_mesh must be provided for vllm rollout config."
        )
      rollout_shape = role_to_mesh[rl_cluster_lib.Role.ROLLOUT].devices.shape
      max_num_seqs = self.config["rollout_config"].get(
          "rollout_vllm_max_num_seqs",
          vllm.get("max_num_seqs", 768),
      )
      max_batched_tokens = self.config["rollout_config"].get(
          "rollout_vllm_max_num_batched_tokens",
          vllm.get(
              "max_num_batched_tokens",
              (max_num_seqs * kv_cache_size) // 4,
          ),
      )
      os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
      return dict(
          rollout_vllm_model_version=vllm.get("model_version", model_id),
          rollout_vllm_hbm_utilization=vllm.get("hbm_utilization", 0.4),
          rollout_vllm_tpu_backend_type=vllm.get("tpu_backend_type", "jax"),
          rollout_vllm_server_mode=vllm.get("server_mode", True),
          rollout_vllm_async_scheduling=vllm.get("async_scheduling", True),
          tensor_parallel_size=(
              rollout_shape[1] if len(rollout_shape) > 1 else 1
          ),
          data_parallel_size=rollout_shape[0],
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

  # ------------------------------------------------------------------
  # Standard GRPO helpers (unchanged)
  # ------------------------------------------------------------------

  def create_cluster_config(
      self,
      *,
      role_to_mesh: dict[rl_cluster_lib.Role, jax.sharding.Mesh],
      rollout_config: base_rollout.RolloutConfig | None = None,
  ):
    if rollout_config is None:
      rollout_config = self.create_rollout_config(role_to_mesh=role_to_mesh)
    return rl_cluster_lib.ClusterConfig(
        role_to_mesh=role_to_mesh,
        rollout_engine=self.config["rollout_engine"],
        offload_to_cpu=self.config["offload_to_cpu"],
        training_config=self.create_rl_training_config(),
        rollout_config=rollout_config,
    )

  def create_rl_training_config(self):
    base_key = "rl_training_config"
    constructed_rl_training_config = self.obtain_training_config_dict(base_key)

    base_config = self.config[base_key]
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

  def create_rl_cluster(self, tokenizer):
    role_to_mesh = self.create_role_to_mesh()
    rollout_config = self.create_rollout_config(role_to_mesh=role_to_mesh)
    # Should not use LoRA for reference model.
    if self.config["reference_model_config"].get("lora_config"):
      logging.warning(
          "LoRA config is not supported for the reference model. Disabling"
          " LoRA."
      )
      del self.config["reference_model_config"]["lora_config"]
    reference_model, tokenizer_path = model_lib.create_model(
        self.config["reference_model_config"],
        self.config["tokenizer_config"],
        role_to_mesh[rl_cluster_lib.Role.REFERENCE],
    )
    if self.config["actor_model_config"].get("lora_config", None):
      actor_model = model_lib.apply_lora_to_model(
          reference_model,
          role_to_mesh[rl_cluster_lib.Role.ACTOR],
          self.config["actor_model_config"]["lora_config"],
      )
    else:
      graph_def, params = nnx.split(reference_model)
      actor_model = nnx.merge(
          graph_def,
          jax.tree.map(jnp.copy, params),
      )

    cluster_config = self.create_cluster_config(
        role_to_mesh=role_to_mesh,
        rollout_config=rollout_config,
    )
    perf_config = self.create_perf_config(cluster_config)
    return rl_cluster_lib.RLCluster(
        actor=actor_model,
        reference=reference_model,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
        perf_config=perf_config,
    )

  def compute_params(self, dataset):
    rl_training_config: dict[str, Any] = self.config.get(
        "rl_training_config", {}
    )

    # Return early if max_steps is already specified.
    max_steps = None
    if rl_training_config.get("max_steps"):
      max_steps = rl_training_config.get("max_steps")
    elif not hasattr(dataset, "__len__"):
      raise ValueError(
          "max_steps must be specified since the dataset length cannot be"
          " determined."
      )

    dataset_length = len(dataset)

    batch_size = self.config.get("batch_size", 1)
    num_batches = self.config.get("num_batches")
    if not num_batches:
      num_batches = dataset_length // batch_size
      logging.info(
          "Dynamically computed num_batches=%d with batch_size=%d",
          num_batches,
          batch_size,
      )
    num_train_epochs = self.config.get("num_train_epochs")
    if not num_train_epochs:
      num_train_epochs = 1

    train_fraction = self.config.get("train_fraction")
    if not train_fraction:
      train_fraction = 0.8
    elif train_fraction <= 0.0 and train_fraction > 1.0:
      logging.warning(
          f"train_fraction {train_fraction:.2f} out of expected range. Setting"
          " to 0.8"
      )
      train_fraction = 0.8

    allowed_max_steps = int(num_batches * num_train_epochs * train_fraction)
    if not max_steps:
      max_steps = allowed_max_steps
    elif max_steps > allowed_max_steps:
      raise ValueError(
          "Maximum allowed value for max_steps is %d", allowed_max_steps
      )

    rl_training_config["max_steps"] = max_steps
    actor_opt: dict[str, Any] = rl_training_config.get(
        "actor_optimizer_config", {}
    )
    if actor_opt and not actor_opt.get("decay_steps"):
      actor_opt["decay_steps"] = max_steps
    if actor_opt and not actor_opt.get("warmup_steps"):
      warmup_ratio = self.config.get("warmup_ratio", 0.1)
      warmup_steps = self.config.get("warmup_steps", warmup_ratio * max_steps)
      actor_opt["warmup_steps"] = warmup_steps
    logging.info(
        "Dynamically computed max_steps=%d based on dataset length %d",
        max_steps,
        dataset_length,
    )

  # ------------------------------------------------------------------
  # Standard GRPO training
  # ------------------------------------------------------------------

  def _run_standard_grpo(self):
    """Execute standard (non-agentic) GRPO training."""
    tokenizer = model_lib.create_tokenizer(
        self.config["tokenizer_config"],
        self.config["tokenizer_config"]["tokenizer_path"],
    )

    if self.config.get("data_module", None):
      dataset = data_lib.get_dataset_from_module(
          self.config["data_module"],
          tokenizer,
      )
    elif self.config["data_source"] == "local":
      dataset = example_data.create_dataset(
          data_source=self.config["data_source"],
          dataset=self.config["data_directory"],
          tokenizer=tokenizer,
      )
    elif self.config["data_source"] == "tfds":
      dataset = example_data.create_dataset(
          data_source=self.config["data_source"],
          dataset=self.config["dataset_name"],
          tfds_download=self.config["tfds_download"],
          split=self.config.get("train_split", self.config.get("split", "train")),
      )
    elif self.config["data_source"] == "huggingface":
      dataset = example_data.create_dataset(
          data_source=self.config["data_source"],
          dataset=self.config["dataset_name"],
          tokenizer=tokenizer,
          split=self.config.get("train_split", self.config.get("split", "train")),
      )
    else:
      raise ValueError(f"Unsupported data_source {self.config['data_source']}")

    self.compute_params(dataset)
    dataset, _ = data_lib.post_init_dataset(
        dataset,
        tokenizer,
        batch_size=self.config["batch_size"],
        num_batches=self.config.get("num_batches", None),
        max_prompt_length=self.config["rollout_config"].get(
            "max_prompt_length", None
        ),
    )
    rl_cluster = self.create_rl_cluster(tokenizer)
    grpo_trainer = grpo_learner.GrpoLearner(
        rl_cluster=rl_cluster,
        reward_fns=self.obtain_reward_fn(),
        algo_config=GrpoConfig(**self.config["grpo_config"]),
    )
    grpo_trainer.train(dataset)

  # ------------------------------------------------------------------
  # Agentic GRPO helpers
  # ------------------------------------------------------------------

  def _create_agentic_grpo_config(self):
    """Build GRPOConfig (agentic) from the agentic_grpo_config YAML section."""
    from tunix.rl.agentic.agentic_grpo_learner import GRPOConfig  # pylint: disable=g-import-not-at-top

    cfg = dict(self.config.get("agentic_grpo_config", {}))

    # episode_timeout = per_turn_timeout_secs * max_turns when not explicit
    if "episode_timeout" not in cfg:
      per_turn = cfg.pop("per_turn_timeout_secs", None)
      max_turns = cfg.get("max_turns", 1)
      if per_turn is not None:
        cfg["episode_timeout"] = per_turn * max_turns

    # max_response_length mirrors rollout_config.total_generation_steps
    if "max_response_length" not in cfg:
      cfg["max_response_length"] = self.config["rollout_config"].get(
          "total_generation_steps", 8192
      )

    # Strip helper keys that are not GRPOConfig fields
    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    cfg.pop("max_turns", None)
    cfg.pop("context_ratio", None)
    return GRPOConfig(**{k: v for k, v in cfg.items() if k in valid})

  def _create_chat_parser(self, tokenizer: Any) -> Any:
    """Instantiate a chat parser based on chat_parser_config.type."""
    from tunix.rl.agentic.parser.chat_template_parser import parser as chat_parser_lib  # pylint: disable=g-import-not-at-top

    parser_type = (self.config.get("chat_parser_config") or {}).get(
        "type", "default"
    )
    if parser_type == "qwen":
      return chat_parser_lib.QwenChatTemplateParser(tokenizer)
    return chat_parser_lib.DefaultChatTemplateParser(tokenizer)

  def _load_class_from_path(self, dotted_path: str) -> type:
    """Load a Python class from a dotted module path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)

  def _load_raw_dataset(self):
    """Load a raw grain.MapDataset from data_module.

    The module must expose ``create_dataset(**data_config) -> grain.MapDataset``
    and optionally a ``batch_fn`` used as ``custom_batch_fn``.
    """
    module = importlib.import_module(self.config["data_module"])
    data_config = dict(self.config.get("data_config", {}))
    dataset = module.create_dataset(**data_config)
    batch_fn = getattr(module, "batch_fn", None)
    return dataset, batch_fn

  def _setup_kubernetes(self) -> None:
    k8s_cfg = self.config.get("kubernetes_config") or {}
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
      from kubernetes import client as k8s_client_lib  # pylint: disable=g-import-not-at-top
      from kubernetes import config as k8s_config_lib  # pylint: disable=g-import-not-at-top

      k8s_config_lib.load_kube_config()
      k8s_client_lib.CoreV1Api()
    except Exception as e:  # pylint: disable=broad-except
      logging.warning("Kubernetes config loading failed: %s", e)

  # ------------------------------------------------------------------
  # Agentic GRPO training
  # ------------------------------------------------------------------

  def _run_agentic_grpo(self):
    """Execute agentic GRPO training (DeepScaleR, DeepSWE, etc.)."""
    from tunix.rl.agentic.agentic_grpo_learner import GRPOLearner  # pylint: disable=g-import-not-at-top

    self._setup_kubernetes()

    tokenizer = model_lib.create_tokenizer(
        self.config["tokenizer_config"],
        self.config["tokenizer_config"]["tokenizer_path"],
    )
    chat_parser = self._create_chat_parser(tokenizer)

    raw_dataset, custom_batch_fn = self._load_raw_dataset()
    self.compute_params(raw_dataset)

    dataset, _ = data_lib.post_init_dataset(
        raw_dataset,
        tokenizer,
        batch_size=self.config.get("batch_size", 1),
        num_batches=self.config.get("num_batches"),
        max_prompt_length=self.config["rollout_config"].get(
            "max_prompt_length"
        ),
        fraction=self.config.get("train_fraction", 1.0),
        num_epochs=self.config.get("num_train_epochs", 1),
        prompt_key=self.config.get("prompt_key", "prompts"),
        custom_batch_fn=custom_batch_fn,
    )

    rl_cluster = self.create_rl_cluster(tokenizer)
    algo_config = self._create_agentic_grpo_config()

    reward_fns = (
        self.obtain_reward_fn() if self.config.get("reward_functions") else None
    )

    learner_kwargs: dict[str, Any] = dict(
        rl_cluster=rl_cluster,
        algo_config=algo_config,
        reward_fns=reward_fns,
        chat_parser=chat_parser,
    )

    agent_class_path = self.config.get("agent_class_path")
    if agent_class_path:
      learner_kwargs["agent_class"] = self._load_class_from_path(
          agent_class_path
      )
      learner_kwargs["agent_kwargs"] = dict(
          self.config.get("agent_kwargs") or {}
      )

    env_class_path = self.config.get("env_class_path")
    if env_class_path:
      learner_kwargs["env_class"] = self._load_class_from_path(env_class_path)
      learner_kwargs["env_kwargs"] = dict(self.config.get("env_kwargs") or {})

    logging.info("Starting agentic GRPO training...")
    GRPOLearner(**learner_kwargs).train(dataset)

  # ------------------------------------------------------------------
  # Dispatcher
  # ------------------------------------------------------------------

  def run_grpo_trainer(self):
    """Dispatch to standard or agentic GRPO based on training_mode."""
    mode = self.config.get("training_mode", "grpo")
    if mode == "agentic_grpo":
      self._run_agentic_grpo()
    elif mode == "grpo":
      self._run_standard_grpo()
    else:
      raise ValueError(
          f"Unknown training_mode: {mode!r}. Expected 'grpo' or 'agentic_grpo'."
      )


def _setup_jax_pathways(pathways_bns: str):
  """Sets up Jax with Pathways."""
  flags.FLAGS.pathways_ifrt = True
  jax.config.update("jax_xla_backend", "pathways")
  jax.config.update("jax_backend_target", pathways_bns)


def _setup_pathways_on_cloud():
  import pathwaysutils  # pylint: disable=g-import-not-at-top

  pathwaysutils.initialize()


def main(argv, **kwargs):
  if _PATHWAYS_BNS.value:
    _setup_jax_pathways(_PATHWAYS_BNS.value)

  if os.getenv("JAX_PLATFORMS") == "proxy":
    _setup_pathways_on_cloud()

  pipeline = GrpoPipeline(argv, **kwargs)
  logging.info(
      "--- Launching GRPO pipeline with following config ---\n"
      "%r\n--------------------------",
      pipeline.config,
  )
  pipeline.run_grpo_trainer()


if __name__ == "__main__":
  app.run(main)
