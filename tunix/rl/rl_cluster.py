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

"""Client facing abstraction for interacting with RL training engine."""

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
import pickle
from typing import Any, Callable, Mapping, Sequence, cast

from absl import logging
import flax
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from flax.nnx import filterlib
from flax.nnx import statelib
import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
from jax.sharding import Mesh  # pylint: disable=g-importing-member
import jaxtyping
import numpy as np
import optax
from tunix.common import configs
from tunix.common import datatypes
from tunix.experimental.rollout import vanilla_rollout as vanilla_rollout_v2
from tunix.generate import tokenizer_adapter
from tunix.generate import utils as generate_utils
# Internal placeholder for sglang_jax rollout worker stub, don't change this line.
from tunix.perf import metrics as perf_metrics
from tunix.perf import trace as perf_trace
from tunix.perf.experimental import constants as perf_constants
from tunix.perf.experimental import tracer as perf_tracer_v2
from tunix.rl import common
from tunix.rl import reshard
from tunix.rl import trainer as rl_trainer
from tunix.rl import utils as rl_utils
from tunix.rl.inference import inference_worker
from tunix.rl.profiler import profiler as profiler_lib
from tunix.rl.rollout import base_rollout
from tunix.rl.rollout import vanilla_rollout
from tunix.sft import metrics_logger
from tunix.sft import peft_trainer
from tunix.sft import sharding_utils
from tunix.sft import utils as sft_utils

Mode = datatypes.Mode
Role = datatypes.Role


RLTrainingConfig = configs.RLTrainingConfig
RLProfileConfig = configs.RLProfileConfig
RLProfiler = profiler_lib.RLProfiler
ProfileScope = profiler_lib.ProfileScope
ClusterConfig = configs.ClusterConfig

ModelOrPath = nnx.Module | str
MetricsT = perf_metrics.MetricsT
MetricsBuffer = perf_metrics.MetricsBuffer


def _env_flag(name: str, default: str = "0") -> bool:
  """True iff the environment variable `name` is set to 1/true/True."""
  return (os.environ.get(name, default) or default).strip() in (
      "1",
      "true",
      "True",
  )


class _GlobalMetrics(dict):
  """`MetricsBuffer.metrics` reduced across processes (per-host learner).

  Marks a buffer whose reducible metrics already hold the values of every
  process, so logging it needs no collective (it may run on a background
  thread).
  """


def _allgather_bytes(payload: bytes) -> list[bytes]:
  """Gathers one byte string from every JAX process, in process order.

  This is a collective: every process must call it at the same point of the
  program, from the thread that issues the other multi-process computations.

  Args:
    payload: This process's bytes.

  Returns:
    The payload of every process, indexed by process index.
  """
  local = np.frombuffer(payload, dtype=np.uint8)
  with jax.transfer_guard("allow"):
    sizes = np.asarray(
        multihost_utils.process_allgather(np.asarray([local.size], np.int32))
    ).reshape(-1)
    # Pad to a power of two so the gather compiles for few distinct shapes.
    width = 1 << max(int(sizes.max()) - 1, 0).bit_length()
    padded = np.zeros((width,), np.uint8)
    padded[: local.size] = local
    gathered = np.asarray(multihost_utils.process_allgather(padded)).reshape(
        sizes.size, width
    )
  return [gathered[i, : sizes[i]].tobytes() for i in range(sizes.size)]


def _union_metric_values(value_lists: Sequence[list[Any]]) -> list[Any]:
  """Concatenates every process's buffered values of one metric.

  Numeric values of different shapes (per-example arrays of different lengths,
  or an empty array from a process without valid items) are flattened into one
  array, so an elementwise reducer (mean, min, max, sum) sees every value once
  instead of failing on a ragged list.

  Args:
    value_lists: The buffered values of each process, in process order.

  Returns:
    The values of all processes.
  """
  values = [value for value_list in value_lists for value in value_list]
  arrays = []
  for value in values:
    if isinstance(value, (str, bytes)) or sft_utils.is_weighted_metric(value):
      return values
    array = np.asarray(value)
    if array.dtype.kind not in "biuf":
      return values
    arrays.append(array)
  if len({array.shape for array in arrays}) > 1:
    return [np.concatenate([array.reshape(-1) for array in arrays])]
  return values


# Gathered instead of `(values, reducer)` for a metric whose values this process
# could not send (they cannot be pickled).
_UNSENDABLE = "unsendable"
# Gathered instead of `(values, reducer)` for a metric whose values are already
# global on this process (computed by a multi-process program).
_ALREADY_GLOBAL = "already global"


def _dropped_metrics(
    per_process: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
  """The metrics of one buffer that no process may log, with the reason.

  A metric is merged only if every process that buffered it sent its values,
  and kept as buffered only if all of those processes report its values as
  already global. Otherwise a logged value would cover only some processes'
  rows, so the metric is dropped. Every process decides this from the same
  gathered payloads, so all processes drop the same metrics.

  Args:
    per_process: The gathered `{name: (values, reducer) or _UNSENDABLE or
      _ALREADY_GLOBAL}` of each process, in process order.

  Returns:
    `{name: reason}` of the metrics to drop.
  """
  dropped = {}
  for name in dict.fromkeys(n for metrics in per_process for n in metrics):
    status = collections.defaultdict(list)
    for process, metrics in enumerate(per_process):
      if name in metrics:
        entry = metrics[name]
        status[entry if isinstance(entry, str) else "sent"].append(process)
    if _UNSENDABLE in status:
      dropped[name] = (
          f"its values could not be sent from process(es) {status[_UNSENDABLE]}"
      )
    elif len(status) > 1:
      dropped[name] = (
          "its values are already global on process(es) "
          f"{status[_ALREADY_GLOBAL]} but not on process(es) {status['sent']}"
      )
  return dropped


def _merge_process_metrics(
    per_process: Sequence[Mapping[str, Any]],
    local_ops: Mapping[str, Callable[..., Any]],
) -> dict[str, tuple[list[Any], Callable[..., Any]]]:
  """Merges the reducible metrics of one buffer across processes.

  Every metric gets the values of all processes, so its reducer runs over what
  one controller would have buffered for the whole global batch: a mean over
  per-example values is global sum / global count, min and max are global
  extrema, `WeightedMetric` values keep their real numerators and
  denominators, and a process that buffered nothing for a metric (no valid
  items) contributes nothing. A metric that some process did not send (see
  `_dropped_metrics`) is not merged.

  Args:
    per_process: `{name: (values, reducer or None)}` of each process, in
      process order. A reducer is None if it could not be sent. A process
      sends `_UNSENDABLE` or `_ALREADY_GLOBAL` instead of `(values, reducer)`
      for a metric it could not send.
    local_ops: This process's reducer of each metric it buffered.

  Returns:
    `{name: (values of all processes, reducer)}`.
  """
  merged = {}
  for name in dict.fromkeys(n for metrics in per_process for n in metrics):
    entries = [metrics[name] for metrics in per_process if name in metrics]
    if any(isinstance(entry, str) for entry in entries):
      continue  # Not sent by every process.
    op = local_ops.get(name) or next(
        (entry[1] for entry in entries if entry[1] is not None), None
    )
    if op is None:
      logging.warning(
          "Per-host learner: skipping metric %s, its reducer is unknown on"
          " process %d.",
          name,
          jax.process_index(),
      )
      continue
    merged[name] = (_union_metric_values([entry[0] for entry in entries]), op)
  return merged


def _sendable_op(op: Callable[..., Any]) -> Callable[..., Any] | None:
  """`op` if it can be pickled (sent to other processes), else None."""
  try:
    pickle.dumps(op)
  except Exception:  # pylint: disable=broad-exception-caught
    return None
  return op


@contextlib.contextmanager
def _profile_scope(
    engine: Any,
    role: Role | str,
    future_output: Any = None,
):
  """Context manager to activate and deactivate targeted profiling around a role execution block.

  Acts as a nullcontext if engine has no profiler.
  """
  profiler = getattr(engine, "profiler", None)
  scope = ProfileScope(future_output)
  if not profiler:
    yield scope
    return

  role_str = role.value if isinstance(role, Role) else role
  profiler.maybe_activate(role_str)
  try:
    yield scope
  finally:
    profiler.maybe_deactivate(role_str, future_output=scope.future_output)


class RLEngine:
  """RLEngine."""

  def __init__(
      self,
      *,
      actor: ModelOrPath,
      critic: ModelOrPath | None = None,
      reference: ModelOrPath | None = None,
      reward: ModelOrPath | None = None,
      tokenizer: Any | None,
      cluster_config: ClusterConfig,
      perf_config: perf_metrics.PerfMetricsConfig | None = None,
  ):
    self.cluster_config = cluster_config
    self.perf_config = perf_config
    self.r2m = cluster_config.role_to_mesh

    self.profiler = None
    if getattr(cluster_config.training_config, "rl_profiler_config", None):
      self.profiler = RLProfiler(
          cluster_config.training_config.rl_profiler_config
      )

    self._init_backbone_sharing_map(actor, reference)
    self._anchor_policy_state = None
    # TUNIX_PER_HOST_SAFE_LOGPS: the first per-host logps chunk is round-trip
    # checked.
    self._per_host_logps_checked = False

    self._default_memory_kind = jax.devices()[0].default_memory().kind
    self.train_actor = self._load_model(actor, self.r2m[Role.ACTOR])

    if self.cluster_config.rollout_config is None:
      raise ValueError("`cluster_config.rollout_config` cannot be None.")
    if isinstance(
        self.cluster_config.rollout_config, dict
    ) and not self.cluster_config.rollout_config.get(Mode.TRAIN):
      raise ValueError(
          "Rollout config is a dict but missing a train config. Provided"
          f" config: {self.cluster_config.rollout_config}"
      )

    if Role.ROLLOUT in self._backbone_sharing_map[Role.ACTOR]:
      self.rollout_actor = self.train_actor
    elif self.cluster_config.rollout_engine in ("vanilla", "vanilla_v2"):
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
    else:
      # Provide initial weights for non vanilla rollout engines.
      self.rollout_actor = self.train_actor

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

    self.tokenizer = tokenizer_adapter.TokenizerAdapter(tokenizer)
    self._rl_metrics_logger = metrics_logger.MetricsLogger(
        self.cluster_config.training_config.metrics_logging_options
    )
    self._buffered_train_metrics: list[MetricsBuffer] = []
    self._buffered_eval_metrics: list[MetricsBuffer] = []
    self._external_metrics_logger = None
    # Per-host learner: RL metrics are reduced across processes before they
    # are logged (see `reduce_metrics_across_processes`).
    self._cross_process_metrics = False

    self._init_engine()
    gc.collect()

    # NB: global steps should be adjusted properly based on the actual RL
    # algorithm. E.g. when loading from a checkpoint with additional inner loops
    # that update the model, we should properly update the global steps.
    self.global_steps = 0

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
                x if x is not None else jax.sharding.PartitionSpec(),
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

  def _init_engine(self):
    """Initializes the RL engine."""
    # 1. Initialize rollout.
    if isinstance(
        self.cluster_config.rollout_engine, str
    ) and self.cluster_config.rollout_engine not in [
        "vanilla",
        "vanilla_v2",
        "vllm",
        "sglang_jax",
    ]:
      raise ValueError(
          "`cluster_config.rollout_engine` should be one of `'vanilla'`, "
          "`'vanilla_v2'`, `'vllm'`, or `'sglang_jax'`. Received:"
          f" '{self.cluster_config.rollout_engine}'."
      )

    if isinstance(self.cluster_config.rollout_config, dict):
      # train_cfg should always be provided.
      train_cfg = self.cluster_config.rollout_config[Mode.TRAIN]
      eval_cfg = self.cluster_config.rollout_config.get(Mode.EVAL)
      max_kv_cache_size = max(
          train_cfg.kv_cache_size,
          eval_cfg.kv_cache_size if eval_cfg is not None else 0,
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
    elif self.cluster_config.rollout_engine == "vanilla_v2":
      if not hasattr(self.rollout_actor, "config"):
        raise ValueError("`self.rollout_actor` must have a config attribute.")
      # The engine compiles its programs for the devices the model is on.
      self._maybe_load_model_from_cpu(self.rollout_actor, Role.ROLLOUT)
      if isinstance(self.cluster_config.rollout_config, dict):
        train_cfg = self.cluster_config.rollout_config[Mode.TRAIN]
        all_cfgs = self.cluster_config.rollout_config.values()
      else:
        train_cfg = self.cluster_config.rollout_config
        all_cfgs = [train_cfg]
      # The engine is built once, from the train config. Configs passed to
      # `generate` only set per-request parameters, so sequences must fit the
      # longest prompt plus generation of any mode.
      self._rollout = vanilla_rollout_v2.VanillaRollout(
          self.rollout_actor,
          self.tokenizer,
          rollout_config=train_cfg,
          mesh=self.r2m[Role.ROLLOUT],
          max_model_len=max(
              cfg.max_prompt_length + cfg.max_tokens_to_generate
              for cfg in all_cfgs
          ),
      )
    elif self.cluster_config.rollout_engine == "vllm":
      # OSS Placeholder for vllm rollout worker, don't change this line.

      if isinstance(self.cluster_config.rollout_config, dict):
        loaded_vllm_config = self.cluster_config.rollout_config[Mode.TRAIN]
      else:
        loaded_vllm_config = self.cluster_config.rollout_config

      if loaded_vllm_config.rollout_vllm_model_version is None:
        raise ValueError("Rollout vllm model version or path is missing!")

      # TODO(linchai): maybe support offloading for vllm rollout.
      with self._get_mesh_and_logical_axis_rules_cm(Role.ROLLOUT):
        # vLLM handles model initialization and loading internally, so we need
        # to provide logical axis rules for vLLM to correctly shard the model on
        # the rollout mesh. This is important for out-of-tree models in vLLM
        # that are implemented with custom logical axis rules, like is the case
        # for MaxText models.
        from tunix.rl.rollout import vllm_rollout  # pylint: disable=g-import-not-at-top  # pyrefly: ignore[missing-module-attribute]

        self._rollout = vllm_rollout.VllmRollout(
            self.rollout_actor,
            self.tokenizer,
            cache_config_or_size=max_kv_cache_size,
            mesh=self.r2m[Role.ROLLOUT],
            rollout_config=loaded_vllm_config,
        )
    elif self.cluster_config.rollout_engine == "sglang_jax":
      from tunix.rl.rollout import sglang_jax_rollout

      if isinstance(self.cluster_config.rollout_config, dict):
        loaded_sglang_jax_config = self.cluster_config.rollout_config[
            Mode.TRAIN
        ]
      else:
        loaded_sglang_jax_config = self.cluster_config.rollout_config

      if (
          sft_utils.is_lora_enabled(self.rollout_actor)
          and not loaded_sglang_jax_config.rollout_sglang_jax_enable_static_lora
      ):
        raise ValueError(
            "Rollout sglang jax lora config is missing: must set"
            " rollout_sglang_jax_lora_target_modules,"
            " rollout_sglang_jax_enable_static_lora,"
            " rollout_sglang_jax_max_lora_rank,"
            " rollout_sglang_jax_lora_scaling."
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
            self.cluster_config.rollout_engine.func,  # pyrefly: ignore[bad-argument-type]
            base_rollout.BaseRollout,
        )
    ):
      if isinstance(self.cluster_config.rollout_config, dict):
        loaded_config = self.cluster_config.rollout_config[Mode.TRAIN]
      else:
        loaded_config = self.cluster_config.rollout_config

      self._rollout = self.cluster_config.rollout_engine(
          rollout_actor=self.rollout_actor,
          tokenizer=self.tokenizer,
          mesh=self.r2m[Role.ROLLOUT],
          rollout_config=loaded_config,
      )
    else:
      raise NotImplementedError(
          f"Rollout engine {self.cluster_config.rollout_engine} not supported"
      )

    # If the rollout engine constructs its own mesh, it could potentially
    # rearanges the devices for better performance. Use that mesh instead of the
    # one provided in the cluster config.
    if hasattr(self._rollout, "mesh") and self._rollout.mesh is not None:
      self.r2m[Role.ROLLOUT] = self._rollout.mesh

    # Initialize the performance tracer after we have all the meshes
    self._perf = perf_trace.NoopTracer()
    self._perf_v2 = perf_tracer_v2.NoopTracer()

    if self.perf_config:
      export_fn_v1 = self.perf_config.custom_export_fn
      export_fn_v2 = self.perf_config.custom_export_fn_v2

      if export_fn_v1 or export_fn_v2:
        devices = list(
            itertools.chain.from_iterable(
                mesh.devices.flatten().tolist()
                for mesh in self.cluster_config.role_to_mesh.values()
            )
        )

        if export_fn_v1:
          self._perf = perf_trace.PerfTracer(devices, export_fn_v1)

        if export_fn_v2:
          self._perf_v2 = perf_tracer_v2.PerfTracer(
              devices,
              export_fn=export_fn_v2,
              concurrent_device_spans=[
                  perf_constants.ROLLOUT,
                  perf_constants.ENVIRONMENT,
              ],
          )

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
      critic_config.metrics_prefix = "critic"
      critic_config.pbar_description = "Critic Training"
      if critic_config.checkpoint_root_directory:
        critic_config.checkpoint_root_directory = os.path.join(
            critic_config.checkpoint_root_directory, "critic"
        )
      with self._get_mesh_and_logical_axis_rules_cm(Role.CRITIC):
        self._critic_trainer = rl_trainer.Trainer(
            model=self.critic,
            optimizer=self.cluster_config.training_config.critic_optimizer,  # pyrefly: ignore[bad-argument-type]
            training_config=critic_config,
            custom_checkpoint_metadata_fn=lambda: {
                "global_step": self.global_steps + 1,
                "role": Role.CRITIC.value,
            },  # offset by 1 since global_step is incremented after the training loop in rl_learner. # pylint: disable=line-too-long
            metrics_logger=self._rl_metrics_logger,
            perf_tracer=self._perf,
            perf_tracer_v2=self._perf_v2,
        )
      del self.critic
      self._maybe_offload_model_to_cpu(self._critic_trainer.model, Role.CRITIC)

    self._maybe_load_model_from_cpu(self.train_actor, Role.ACTOR)
    actor_config = copy.deepcopy(self.cluster_config.training_config)
    actor_config.metrics_prefix = "actor"
    actor_config.pbar_description = "Actor Training"
    if actor_config.checkpoint_root_directory:
      actor_config.checkpoint_root_directory = os.path.join(
          actor_config.checkpoint_root_directory, "actor"
      )
    with self._get_mesh_and_logical_axis_rules_cm(Role.ACTOR):
      self._actor_trainer = rl_trainer.Trainer(
          model=self.train_actor,
          optimizer=self.cluster_config.training_config.actor_optimizer,
          training_config=actor_config,
          custom_checkpoint_metadata_fn=lambda: {
              "global_step": self.global_steps + 1,
              "role": Role.ACTOR.value,
          },  # offset by 1 since global_step is incremented after the training loop in rl_learner. # pylint: disable=line-too-long
          metrics_logger=self._rl_metrics_logger,
          perf_tracer=self._perf,
          perf_tracer_v2=self._perf_v2,
      )
    del self.rollout_actor
    del self.train_actor
    self._maybe_offload_model_to_cpu(self.actor_trainer.model, Role.ACTOR)
    self._anchor_policy_state = rl_utils.put_params_on_memory_kind(
        nnx.state(self.actor_trainer.model), "pinned_host"
    )

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

  def _is_state_on_device(self, state: jaxtyping.PyTree) -> bool:
    shardings = jax.tree.map(
        lambda x: x.sharding if hasattr(x, "sharding") else None, state
    )
    return jax.tree_util.tree_reduce(
        operator.or_,
        jax.tree.map(
            lambda x: x is not None
            and x.memory_kind == self._default_memory_kind,
            shardings,
        ),
        initializer=False,
    )

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

  @property
  def perf(self) -> perf_trace.Tracer:
    """The v1 performance tracer."""
    return self._perf

  @property
  def perf_v2(self) -> perf_tracer_v2.Tracer:
    """The v2 performance tracer."""
    return self._perf_v2

  def close(self):
    """Closes engine components and flushes buffered metrics and profiler traces."""
    for m in self._buffered_train_metrics + self._buffered_eval_metrics:
      self._log_metrics(m)
    self.actor_trainer.close()
    if getattr(self, "critic_trainer", None):
      self.critic_trainer.close()
    if self.profiler:
      self.profiler.close()

  def _log_metrics(self, metrics_buffer: MetricsBuffer) -> None:
    """Log metrics."""
    cross_process = getattr(self, "_cross_process_metrics", False)
    unreduced = cross_process and not isinstance(
        metrics_buffer.metrics, _GlobalMetrics
    )
    skipped = []
    for metric_name, (value, op) in metrics_buffer.metrics.items():
      if unreduced and op is not None:
        # Per-host learner: these values cover this process's rows only;
        # logging them under the global name would misreport the metric.
        skipped.append(metric_name)
        continue
      # Convert to numpy array immediately.
      # This handles nested lists, mixed types, and JAX arrays automatically.
      try:
        agg_value = np.array(value)
      except Exception:
        logging.warning(
            "Skipping metric %s: Could not convert to numpy array.", metric_name
        )
        continue

      if agg_value.dtype.kind in {"U", "S"}:
        self._log_string_metric(metric_name, agg_value, cross_process)
        continue

      if agg_value.dtype.kind == "O":
        # Try to infer if it contains strings by checking the first flattened element
        if agg_value.size > 0 and isinstance(
            agg_value.ravel()[0], (str, np.str_)
        ):
          self._log_string_metric(metric_name, agg_value, cross_process)
          continue

      # Apply aggregation and Log
      if op is not None:
        # Ensure op doesn't crash on empty arrays
        if agg_value.size > 0:
          agg_value = op(agg_value)

      if "/" in metric_name:
        prefix, metric_name = metric_name.split("/", maxsplit=1)
      else:
        prefix = "global"
      self._rl_metrics_logger.log(
          prefix,
          metric_name,
          agg_value,  # pyrefly: ignore[bad-argument-type]
          metrics_buffer.mode,
          metrics_buffer.global_steps,
      )

    if skipped:
      logging.warning(
          "Per-host learner: not logging %d metrics of %s step %d (%s): they"
          " were not reduced across processes.",
          len(skipped),
          metrics_buffer.mode,
          metrics_buffer.global_steps,
          ", ".join(skipped),
      )

    if self._external_metrics_logger is not None:
      self._external_metrics_logger(metrics_buffer)

  def _log_string_metric(
      self, metric_name: str, value: np.ndarray, cross_process: bool
  ) -> None:
    """Logs a string metric (e.g. prompts); never reduced across processes."""
    if cross_process:
      # Per-host learner: per-example data of this process's rows only.
      logging.info(
          "Rollout string metric %s (process %d of %d): %s",
          metric_name,
          jax.process_index(),
          jax.process_count(),
          value,
      )
    else:
      logging.info("Rollout string metric %s: %s", metric_name, value)

  def enable_cross_process_metrics(self) -> None:
    """Turns on the per-host learner's cross-process metric contract.

    Under the per-host learner (TUNIX_PER_HOST_LEARNER=1) every process
    buffers RL metrics (rewards, completion lengths, user metrics, ...) of its
    own prompt shard, and only process 0 exports them. Once this is on, a
    buffer is logged only after `reduce_metrics_across_processes` replaced its
    reducible metrics (those with a reducer) by their global values; reducible
    metrics of a buffer that was not reduced are not logged. Metrics without a
    reducer (prompts, completions and trajectory ids, perf timings) stay
    host-local.
    """
    self._cross_process_metrics = True

  def reduce_metrics_across_processes(
      self, include_current: bool = False
  ) -> None:
    """Replaces the buffered RL metrics of completed steps by global values.

    For the per-host learner (see `enable_cross_process_metrics`; a no-op
    otherwise). Every reducible metric of every buffered, not yet reduced
    step gets the values of all processes (`_merge_process_metrics`), so the
    exported value of its reducer is the one of the global batch: a mean over
    per-example values is global sum / global count, min and max are global
    extrema, `WeightedMetric` values keep their real denominators, and a
    process without values for a metric contributes nothing. A metric fn
    that returns one scalar per call should return per-example values or a
    `WeightedMetric` when its denominator is data dependent.

    A metric is dropped on every process, with a warning, if some process
    could not send its values (they cannot be pickled), or if its values are
    already global (computed by a multi-process program) on some processes but
    not on others: its logged value would cover only some processes' rows.
    All processes decide this from the same gathered payloads. A metric whose
    values are already global on every process that buffered it is logged as
    buffered.

    This is a collective: every process must call it at the same point, on
    the main thread, after the step's rollouts and train steps, so that every
    process gathers in the same order. Logging (also asynchronous logging)
    then needs no collective.

    Args:
      include_current: Also reduce the buffers of the current global step
        (at the end of training).
    """
    if not getattr(self, "_cross_process_metrics", False):
      return
    pending = []
    occurrences = collections.Counter()
    for buf in self._buffered_train_metrics + self._buffered_eval_metrics:
      if isinstance(buf.metrics, _GlobalMetrics) or (
          not include_current and buf.global_steps >= self.global_steps
      ):
        continue
      step_key = (buf.mode, buf.global_steps)
      pending.append(((*step_key, occurrences[step_key]), buf))
      occurrences[step_key] += 1
    payload = {}
    for key, buf in pending:
      payload[key] = {}
      for name, (values, op) in buf.metrics.items():
        if op is None:
          continue
        if any(
            isinstance(x, jax.Array) and not x.is_fully_addressable
            for x in jax.tree.leaves(values)
        ):
          # Computed by a multi-process program: already global.
          payload[key][name] = _ALREADY_GLOBAL
          continue
        try:
          entry = (jax.device_get(values), _sendable_op(op))
          pickle.dumps(entry)
        except Exception:  # pylint: disable=broad-exception-caught
          # Tell the other processes, so that every process drops it.
          payload[key][name] = _UNSENDABLE
          continue
        payload[key][name] = entry
    gathered = [
        pickle.loads(data)
        for data in _allgather_bytes(
            pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        )
    ]
    for key, buf in pending:
      local_ops = {
          name: op for name, (_, op) in buf.metrics.items() if op is not None
      }
      per_process = [metrics.get(key, {}) for metrics in gathered]
      dropped = _dropped_metrics(per_process)
      merged = _merge_process_metrics(per_process, local_ops)
      reduced = _GlobalMetrics()
      for name, entry in buf.metrics.items():
        if name not in dropped:
          reduced[name] = merged.pop(name, entry)
      reduced.update(merged)  # Metrics only other processes buffered.
      for name, reason in dropped.items():
        logging.warning(
            "Per-host learner: not logging metric %s of %s step %d on any"
            " process: %s.",
            name,
            buf.mode,
            buf.global_steps,
            reason,
        )
      buf.metrics = reduced
    remote_only = set().union(*gathered) - set(payload)
    if remote_only:
      logging.warning(
          "Per-host learner: process %d has no metrics buffer for %s; their"
          " metrics are not logged.",
          jax.process_index(),
          sorted(remote_only),
      )

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

  def buffer_metrics_async(
      self,
      metrics: MetricsT,
      mode: Mode = Mode.TRAIN,
      step: int = 0,
  ) -> None:
    """Buffers rl metrics to be logged for async training.

    Actual logging will happen when global steps are incremented.

    Args:
      metrics: A dictionary mapping metric names to a tuple containing the
        metric value and an optional aggregation function.
      mode: The mode of the workload, either TRAIN or EVAL.
      step: The step number for the metrics. Only used in TRAIN mode.
    """
    if mode == Mode.TRAIN:
      buffered_metrics = self._buffered_train_metrics
    else:
      buffered_metrics = self._buffered_eval_metrics

    if not buffered_metrics:
      buffered_metrics.append(MetricsBuffer(self.global_steps, mode=str(mode)))
    else:
      if step != buffered_metrics[-1].global_steps:
        buffered_metrics.append(MetricsBuffer(step, mode=str(mode)))

    cur_metrics = buffered_metrics[-1]
    for metric_name, (value, op) in metrics.items():
      if metric_name not in cur_metrics.metrics:
        cur_metrics.metrics[metric_name] = (
            [value],
            op,
        )
      else:
        cur_metrics.metrics[metric_name][0].append(value)

    # Global steps are incremented, log the previous metrics.
    if (
        self._buffered_train_metrics
        and self._buffered_train_metrics[0].global_steps < self.global_steps
    ):
      for m in [self._buffered_train_metrics.pop(0)]:
        self._log_metrics(m)
    if (
        self._buffered_eval_metrics
        and self._buffered_eval_metrics[0].global_steps < self.global_steps
    ):
      for m in [self._buffered_eval_metrics.pop(0)]:
        self._log_metrics(m)

  def train(
      self,
      role: Role,
      train_ds: Any,
      eval_ds: Any,
      skip_jit: bool = False,
  ) -> None:
    """Runs a training update for the specified role."""
    if role == Role.ACTOR:
      self.update_actor(train_ds, eval_ds, skip_jit=skip_jit)
    elif role == Role.CRITIC:
      self.update_critic(train_ds, eval_ds, skip_jit=skip_jit)
    else:
      raise ValueError(f"Unsupported role for train: {role}")

  def update_actor(self, train_ds, eval_ds, skip_jit=False):
    """Runs a training update step for the actor model."""
    with _profile_scope(
        self, Role.ACTOR, future_output=self.actor_trainer.model
    ):
      with self._get_mesh_and_logical_axis_rules_cm(Role.ACTOR):
        self._maybe_load_model_from_cpu(self.actor_trainer.model, Role.ACTOR)
        with self._perf.span_group("actor_training"):
          self.actor_trainer.train(train_ds, eval_ds, skip_jit)
        self._maybe_offload_model_to_cpu(self.actor_trainer.model, Role.ACTOR)

  def eval_actor(self, eval_ds: Any) -> Any:
    """Runs an explicit actor evaluation phase."""
    if eval_ds is None:
      return None

    with _profile_scope(
        self, Role.ACTOR, future_output=self.actor_trainer.model
    ):
      with self._get_mesh_and_logical_axis_rules_cm(Role.ACTOR):
        self._maybe_load_model_from_cpu(self.actor_trainer.model, Role.ACTOR)
        run_eval = getattr(self.actor_trainer, "_run_eval", None)
        if callable(run_eval):
          res = run_eval(eval_ds)
        else:
          eval_step = getattr(self.actor_trainer, "eval_step", None)
          if not callable(eval_step):
            raise TypeError(
                "actor_trainer must expose _run_eval(...) or eval_step(...)."
            )
          res = None
          for chunk in eval_ds:
            eval_step(chunk)
        self._maybe_offload_model_to_cpu(self.actor_trainer.model, Role.ACTOR)
        return res

  def update_critic(self, train_ds, eval_ds, skip_jit=False):
    with self._get_mesh_and_logical_axis_rules_cm(Role.CRITIC):
      self._maybe_load_model_from_cpu(self.critic_trainer.model, Role.CRITIC)
      with self._perf.span_group("critic_training"):
        self._critic_trainer.train(train_ds, eval_ds, skip_jit)
      self._maybe_offload_model_to_cpu(self.critic_trainer.model, Role.CRITIC)

  def generate(
      self,
      prompts: list[str] | list[list[dict[str, str]]] | None = None,
      apply_chat_template: bool = False,
      mode: Mode = Mode.TRAIN,
      micro_batch_size: int | None = None,
      trace_tags: Mapping[str, Any] | None = None,
      max_generation_steps: int | None = None,
      *,
      prompt_token_ids: Sequence[Sequence[int] | np.ndarray] | None = None,
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
      trace_tags: Optional tags to add to the performance tracer.
      prompt_token_ids: Explicit unpadded token rows. Mutually exclusive with
        text prompts and chat templating; requires backend token-input support.

    Returns:
      A `RolloutOutput` object containing the generated text and other info.
    """
    exact_input = prompt_token_ids is not None
    if exact_input:
      if prompts is not None:
        raise ValueError("Provide exactly one of prompts or prompt_token_ids.")
      prompt_rows: list[Any] = [
          generate_utils.as_token_ids(row)
          for row in prompt_token_ids
      ]
    else:
      if prompts is None:
        raise ValueError("Cannot generate from an empty list of prompts.")
      if apply_chat_template:
        if self.tokenizer is None:
          raise ValueError(
              "Tokenizer must be initialized to use chat templates."
          )
        prompt_rows = [
            self.tokenizer.apply_chat_template(
                prompt,  # pyrefly: ignore[bad-argument-type]
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            for prompt in prompts
        ]
      else:
        prompt_rows = list(prompts)

    if len(prompt_rows) == 0:
      raise ValueError("Cannot generate from an empty list of prompts.")
    micro_batch_size = micro_batch_size or len(prompt_rows)

    with self._generate_mesh_cm() as (mesh, _):
      model = self.rollout.model()
      self._maybe_load_model_from_cpu(model, Role.ROLLOUT)
      if self.cluster_config.offload_to_cpu:
        self.rollout.update_params(nnx.state(model))

      if isinstance(self.cluster_config.rollout_config, dict):
        rollout_config = self.cluster_config.rollout_config[mode]
      else:
        rollout_config = self.cluster_config.rollout_config

      if max_generation_steps is not None:
        rollout_config = dataclasses.replace(
            rollout_config,
            max_tokens_to_generate=max_generation_steps,
        )

      perf_tags = {
          perf_constants.ROLE: Role.ROLLOUT.value,
      }
      if trace_tags:
        perf_tags.update(trace_tags)

      with _profile_scope(self, Role.ROLLOUT) as scope:
        with self._perf.span(
            "rollout", mesh.devices
        ) as span, self._perf_v2.span(
            perf_constants.ROLLOUT,
            mesh.devices,
            tags=perf_tags,
        ) as span_v2:
          outputs = [
              self.rollout.generate(
                  cast(Any, None),
                  rollout_config,
                  prompt_token_ids=prompt_rows[s],
              )
              if exact_input
              else self.rollout.generate(
                  cast(list[str], prompt_rows[s]), rollout_config
              )
              for s in rl_utils.chunk_slices_by_size(
                  stop=len(prompt_rows), step=micro_batch_size
              )
          ]
          scope.set_future_output([o.tokens for o in outputs])
          span.device_end([o.tokens for o in outputs])
          span_v2.async_end([o.tokens for o in outputs])
      self._maybe_offload_model_to_cpu(model, Role.ROLLOUT)
      if self.cluster_config.offload_to_cpu:
        self.rollout.update_params(nnx.state(model))
        gc.collect()

    texts = list(itertools.chain.from_iterable(out.text for out in outputs))

    logprobs = None
    if outputs[0].logprobs is not None:
      logprobs = list(
          itertools.chain.from_iterable(out.logprobs for out in outputs)  # pyrefly: ignore[bad-argument-type]
      )

    logits = None
    if outputs[0].logits is not None:
      logits = list(
          itertools.chain.from_iterable(out.logits for out in outputs)  # pyrefly: ignore[bad-argument-type]
      )

    prompt_lengths = None
    if all(out.prompt_lengths is not None for out in outputs):
      prompt_lengths = np.concatenate(
          [cast(np.ndarray, out.prompt_lengths) for out in outputs]
      )

    return base_rollout.RolloutOutput(
        text=texts,
        logits=logits,
        tokens=list(
            itertools.chain.from_iterable(out.tokens for out in outputs)
        ),
        left_padded_prompt_tokens=np.concatenate(
            [out.left_padded_prompt_tokens for out in outputs], axis=0
        ),
        logprobs=logprobs,
        prompt_lengths=prompt_lengths,
    )

  def per_token_logps(
      self,
      role: Role,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
      micro_batch_size: int | None = None,
      segment_ids: jax.Array | None = None,
      **kwargs: Any,
  ) -> jax.Array:
    """Computes per-token logprobs under the specified model role."""
    if role == Role.REFERENCE:
      return self.get_ref_per_token_logps(
          prompt_tokens=prompt_tokens,
          completion_tokens=completion_tokens,
          pad_id=pad_id,
          eos_id=eos_id,
          micro_batch_size=micro_batch_size,
          segment_ids=segment_ids,
          **kwargs,
      )
    elif role == Role.ACTOR:
      return self.get_actor_per_token_logps(
          prompt_tokens=prompt_tokens,
          completion_tokens=completion_tokens,
          pad_id=pad_id,
          eos_id=eos_id,
          micro_batch_size=micro_batch_size,
          segment_ids=segment_ids,
          **kwargs,
      )
    else:
      raise ValueError(f"Unsupported role for per_token_logps: {role}")

  def get_ref_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
      micro_batch_size: int | None = None,
      segment_ids: jax.Array | None = None,
      segment_positions: jax.Array | None = None,
      token_mask: jax.Array | None = None,
  ) -> jax.Array:
    """Gets the per-token logps of the reference model."""
    batch_size = prompt_tokens.shape[0]
    if batch_size == 0:
      raise ValueError(
          "Cannot get reference log probabilities from an empty batch."
      )
    micro_batch_size = micro_batch_size or batch_size
    if sharding_utils.per_host_safe_logps_enabled():
      return self._per_host_safe_ref_per_token_logps(
          prompt_tokens=prompt_tokens,
          completion_tokens=completion_tokens,
          pad_id=pad_id,
          eos_id=eos_id,
          micro_batch_size=micro_batch_size,
          segment_ids=segment_ids,
          segment_positions=segment_positions,
          token_mask=token_mask,
      )

    with self._get_mesh_and_logical_axis_rules_cm(Role.REFERENCE):
      # This assumes reference model shards same data sharding as actor, which
      # should be true as ref model and policy model shares same architecture.
      dest_prompt_tokens = sharding_utils.shard_input(
          prompt_tokens,
          self.cluster_config.training_config.data_sharding_axis,
      )
      dest_completion_tokens = sharding_utils.shard_input(
          completion_tokens,
          self.cluster_config.training_config.data_sharding_axis,
      )
      dest_segment_ids = (
          None
          if segment_ids is None
          else sharding_utils.shard_input(
              segment_ids,
              self.cluster_config.training_config.data_sharding_axis,
          )
      )
      dest_segment_positions = (
          None
          if segment_positions is None
          else sharding_utils.shard_input(
              segment_positions,
              self.cluster_config.training_config.data_sharding_axis,
          )
      )
      dest_token_mask = (
          None
          if token_mask is None
          else sharding_utils.shard_input(
              token_mask, self.cluster_config.training_config.data_sharding_axis
          )
      )
      self._maybe_load_model_from_cpu(
          self.inference_worker.get_model("reference"), Role.REFERENCE
      )
      temperature = self.get_rollout_config(mode=Mode.TRAIN).temperature
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
                temperature=temperature,
                segment_ids=(
                    None
                    if dest_segment_ids is None
                    else dest_segment_ids[batch_slice]
                ),
                segment_positions=(
                    None
                    if dest_segment_positions is None
                    else dest_segment_positions[batch_slice]
                ),
                token_mask=(
                    None
                    if dest_token_mask is None
                    else dest_token_mask[batch_slice]
                ),
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
  ) -> jax.Array:
    """Gets the per-token logps of the current policy model."""
    batch_size = prompt_tokens.shape[0]
    if batch_size == 0:
      raise ValueError("Cannot get old log probabilities from an empty batch.")
    micro_batch_size = micro_batch_size or batch_size

    with self._get_mesh_and_logical_axis_rules_cm(Role.ROLLOUT):
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
            )
        )
      per_token_logps = jnp.concatenate(outs, axis=0)
      model = self.rollout.model()
      self._maybe_offload_model_to_cpu(model, Role.ROLLOUT)
      if self.cluster_config.offload_to_cpu:
        self.rollout.update_params(nnx.state(model))
        gc.collect()
      return per_token_logps

  def get_actor_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
      micro_batch_size: int | None = None,
      temperature: float | None = None,
      segment_ids: jax.Array | None = None,
      segment_positions: jax.Array | None = None,
      token_mask: jax.Array | None = None,
  ) -> jax.Array:
    """Gets per-token logps from the actor model on the trainer side.

    Mirrors `get_ref_per_token_logps` — must pass through the rollout
    temperature
    so the actor's recomputed logps match the temperature scaling used at
    sampling time (otherwise log_softmax(logits/T_sample) vs log_softmax(logits)
    yields a multi-nat artifact diff vs vllm's `processed_logprobs`).
    """
    if temperature is None:
      temperature = self.get_rollout_config(mode=Mode.TRAIN).temperature
    batch_size = prompt_tokens.shape[0]
    if batch_size == 0:
      raise ValueError(
          "Cannot get actor log probabilities from an empty batch."
      )
    if self._anchor_policy_state is None:
      raise ValueError(
          "Anchor policy state is not initialized. Please run `sync_weights`"
          " first."
      )
    micro_batch_size = micro_batch_size or batch_size
    if sharding_utils.per_host_safe_logps_enabled():
      return self._per_host_safe_actor_per_token_logps(
          prompt_tokens=prompt_tokens,
          completion_tokens=completion_tokens,
          pad_id=pad_id,
          eos_id=eos_id,
          micro_batch_size=micro_batch_size,
          temperature=temperature,
          segment_ids=segment_ids,
          segment_positions=segment_positions,
          token_mask=token_mask,
      )
    with self._get_mesh_and_logical_axis_rules_cm(Role.ACTOR) as (mesh, _):
      dest_prompt_tokens = sharding_utils.shard_input(
          prompt_tokens,
          self.cluster_config.training_config.data_sharding_axis,
      )
      dest_completion_tokens = sharding_utils.shard_input(
          completion_tokens,
          self.cluster_config.training_config.data_sharding_axis,
      )
      dest_segment_ids = (
          None
          if segment_ids is None
          else sharding_utils.shard_input(
              segment_ids,
              self.cluster_config.training_config.data_sharding_axis,
          )
      )
      dest_segment_positions = (
          None
          if segment_positions is None
          else sharding_utils.shard_input(
              segment_positions,
              self.cluster_config.training_config.data_sharding_axis,
          )
      )

      dest_token_mask = (
          None
          if token_mask is None
          else sharding_utils.shard_input(
              token_mask, self.cluster_config.training_config.data_sharding_axis
          )
      )

      # Use the anchor (start-of-global-step) actor weights so old_per_token_logps
      # reference the same policy vllm sampled with even when mini_batch_size <
      # full_batch_size or num_iterations > 1. Only offload the live actor when
      # `offload_to_cpu` is enabled cluster-wide; otherwise the host round-trip
      # was both unnecessary and risked leaving stray weights pinned to host.
      actor_trainer_state_on_device = self._is_state_on_device(
          nnx.state(self.actor_trainer.model)
      )
      if actor_trainer_state_on_device and self.cluster_config.offload_to_cpu:
        self._put_model_on_memory_kind(self.actor_trainer.model, "pinned_host")
        gc.collect()
      graphdef, _ = nnx.split(self.actor_trainer.model)
      anchor_on_device = self._is_state_on_device(self._anchor_policy_state)
      if anchor_on_device:
        anchor_policy_state = self._anchor_policy_state
      else:
        anchor_policy_state = rl_utils.put_params_on_memory_kind(
            self._anchor_policy_state, self._default_memory_kind
        )
      outs = []
      for batch_slice in rl_utils.chunk_slices_by_size(
          stop=batch_size, step=micro_batch_size
      ):
        outs.append(
            common.compute_per_token_logps(
                graphdef,
                anchor_policy_state,
                prompt_tokens=dest_prompt_tokens[batch_slice],
                completion_tokens=dest_completion_tokens[batch_slice],
                pad_id=pad_id,
                eos_id=eos_id,
                stop_gradient=True,
                temperature=temperature,
                chunk_size=self.cluster_config.training_config.compute_logps_chunk_size,
                segment_ids=(
                    None
                    if dest_segment_ids is None
                    else dest_segment_ids[batch_slice]
                ),
                segment_positions=(
                    None
                    if dest_segment_positions is None
                    else dest_segment_positions[batch_slice]
                ),
                token_mask=(
                    None
                    if dest_token_mask is None
                    else dest_token_mask[batch_slice]
                ),
            )
        )
      actor_per_token_logps = jnp.concatenate(outs, axis=0)
      if not anchor_on_device:
        del anchor_policy_state
      if actor_trainer_state_on_device and self.cluster_config.offload_to_cpu:
        # Release the host-side references the log-prob pass leaves behind
        # before the model moves back; without offloading there is nothing
        # to reclaim and a full collection is a pure stall.
        gc.collect()
        self._put_model_on_memory_kind(
            self.actor_trainer.model, self._default_memory_kind
        )
      return actor_per_token_logps

  def _per_host_chunked_logps(
      self,
      compute_fn: Callable[..., jax.Array],
      micro_batch_size: int,
      **inputs: jax.Array | np.ndarray | None,
  ) -> jax.Array:
    """Per-token logps of this process's own rows (TUNIX_PER_HOST_SAFE_LOGPS=1).

    Under the per-host learner (TUNIX_PER_HOST_LEARNER=1) every process
    passes its own rows. The stock reference/actor paths globalise the whole
    process-local batch with `shard_input` (local rows x process count) and
    then slice the global array with process-local offsets, so every process
    but the first would get another process's rows. Here the process-local
    batch is chunked first; each chunk is globalised and run through
    `compute_fn` on every process in lockstep (all processes hold the same
    number of rows), and this process's rows of each result are copied back,
    so the result is host-local like every other TrainExample field.

    Args:
      compute_fn: Called with the globalised chunk of every input (by name).
      micro_batch_size: Process-local rows per chunk.
      **inputs: Process-local inputs (None entries are passed through).

    Returns:
      This process's per-token logps, host-local.
    """
    axis = self.cluster_config.training_config.data_sharding_axis
    pspec = jax.sharding.PartitionSpec(*axis)
    batch_size = inputs["prompt_tokens"].shape[0]
    for name, x in inputs.items():
      if isinstance(x, jax.Array) and not x.is_fully_addressable:
        raise ValueError(
            f"TUNIX_PER_HOST_SAFE_LOGPS: {name} is a global array; the per-host"
            " logps path expects process-local rows."
        )
    if batch_size % micro_batch_size:
      raise ValueError(
          f"TUNIX_PER_HOST_SAFE_LOGPS: the process-local batch of {batch_size}"
          f" rows is not divisible by micro_batch_size={micro_batch_size};"
          " every process must run the same number of equal chunks."
      )
    outs = []
    for batch_slice in rl_utils.chunk_slices_by_size(
        stop=batch_size, step=micro_batch_size
    ):
      chunk = {}
      for name, x in inputs.items():
        if x is None:
          chunk[name] = None
          continue
        chunk[name] = sharding_utils.shard_input(x[batch_slice], axis)
        if getattr(chunk[name].sharding, "spec", None) != pspec:
          raise ValueError(
              f"TUNIX_PER_HOST_SAFE_LOGPS: a {name} chunk of"
              f" {micro_batch_size} rows per process got sharding"
              f" {chunk[name].sharding}, not {pspec}; process-local rows must"
              " never be treated as replicated."
          )
      out = compute_fn(**chunk)
      local = sharding_utils.process_local_rows(out)
      in_rows = sharding_utils.process_row_ranges(chunk["prompt_tokens"])
      out_rows = (
          sharding_utils.process_row_ranges(out)
          if isinstance(out, jax.Array) and not out.is_fully_addressable
          else None
      )
      if out_rows != in_rows or local.shape[0] != micro_batch_size:
        raise RuntimeError(
            f"TUNIX_PER_HOST_SAFE_LOGPS: output rows {out_rows} (shape"
            f" {local.shape}) of process {jax.process_index()} do not match its"
            f" input rows {in_rows}."
        )
      if not self._per_host_logps_checked:
        # Host-local round trip: re-globalise this process's rows exactly as
        # shard_input does and compare every addressable shard (no
        # collective).
        again = jax.make_array_from_process_local_data(
            out.sharding, local, out.shape
        )
        got = sorted(out.addressable_shards, key=lambda z: z.device.id)
        want = sorted(again.addressable_shards, key=lambda z: z.device.id)
        if len(got) != len(want) or not all(
            np.array_equal(np.asarray(a.data), np.asarray(b.data))
            for a, b in zip(got, want)
        ):
          raise RuntimeError(
              "TUNIX_PER_HOST_SAFE_LOGPS: host-local extraction does not invert"
              " jax.make_array_from_process_local_data."
          )
        self._per_host_logps_checked = True
        logging.info(
            "TUNIX_PER_HOST_SAFE_LOGPS: per-host logps on process %d/%d: %d"
            " local rows in chunks of %d, global chunk %s, round trip ok.",
            jax.process_index(),
            jax.process_count(),
            batch_size,
            micro_batch_size,
            out.shape,
        )
      outs.append(local)
    return jnp.asarray(np.concatenate(outs, axis=0))

  def _per_host_safe_ref_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
      micro_batch_size: int,
      segment_ids: jax.Array | None,
      segment_positions: jax.Array | None,
      token_mask: jax.Array | None,
  ) -> jax.Array:
    """`get_ref_per_token_logps` under TUNIX_PER_HOST_SAFE_LOGPS=1."""
    with self._get_mesh_and_logical_axis_rules_cm(Role.REFERENCE):
      ref_model = self.inference_worker.get_model("reference")
      self._maybe_load_model_from_cpu(ref_model, Role.REFERENCE)
      temperature = self.get_rollout_config(mode=Mode.TRAIN).temperature

      def compute(prompt_tokens, completion_tokens, **kwargs):
        return self.inference_worker.get_ref_per_token_logps(
            prompt_tokens,
            completion_tokens,
            pad_id,
            eos_id,
            temperature=temperature,
            **kwargs,
        )

      ref_per_token_logps = self._per_host_chunked_logps(
          compute,
          micro_batch_size,
          prompt_tokens=prompt_tokens,
          completion_tokens=completion_tokens,
          segment_ids=segment_ids,
          segment_positions=segment_positions,
          token_mask=token_mask,
      )
      self._maybe_offload_model_to_cpu(ref_model, Role.REFERENCE)
      return ref_per_token_logps

  def _per_host_safe_actor_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
      micro_batch_size: int,
      temperature: float,
      segment_ids: jax.Array | None,
      segment_positions: jax.Array | None,
      token_mask: jax.Array | None,
  ) -> jax.Array:
    """`get_actor_per_token_logps` under TUNIX_PER_HOST_SAFE_LOGPS=1.

    Same anchor-policy handling as the stock path; only the chunking and the
    host-local result differ.
    """
    with self._get_mesh_and_logical_axis_rules_cm(Role.ACTOR):
      actor_trainer_state_on_device = self._is_state_on_device(
          nnx.state(self.actor_trainer.model)
      )
      if actor_trainer_state_on_device and self.cluster_config.offload_to_cpu:
        self._put_model_on_memory_kind(self.actor_trainer.model, "pinned_host")
        gc.collect()
      graphdef, _ = nnx.split(self.actor_trainer.model)
      anchor_on_device = self._is_state_on_device(self._anchor_policy_state)
      if anchor_on_device:
        anchor_policy_state = self._anchor_policy_state
      else:
        anchor_policy_state = rl_utils.put_params_on_memory_kind(
            self._anchor_policy_state, self._default_memory_kind
        )
      compute = functools.partial(
          common.compute_per_token_logps,
          graphdef,
          anchor_policy_state,
          pad_id=pad_id,
          eos_id=eos_id,
          stop_gradient=True,
          temperature=temperature,
          chunk_size=self.cluster_config.training_config.compute_logps_chunk_size,
      )
      actor_per_token_logps = self._per_host_chunked_logps(
          compute,
          micro_batch_size,
          prompt_tokens=prompt_tokens,
          completion_tokens=completion_tokens,
          segment_ids=segment_ids,
          segment_positions=segment_positions,
          token_mask=token_mask,
      )
      del compute
      if not anchor_on_device:
        del anchor_policy_state
      if actor_trainer_state_on_device and self.cluster_config.offload_to_cpu:
        gc.collect()
        self._put_model_on_memory_kind(
            self.actor_trainer.model, self._default_memory_kind
        )
      return actor_per_token_logps

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
      if self.cluster_config.gc_collect_after_weight_sync:
        gc.collect()
      # The anchor policy state is snapshotted from actor_trainer.model.
      self._anchor_policy_state = rl_utils.put_params_on_memory_kind(
          nnx.state(self.actor_trainer.model), "pinned_host"
      )

    # sync weights marks the end of a full batch, so increment the global steps.
    self.global_steps += 1

  def get_values(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
  ) -> jax.Array:
    with self._get_mesh_and_logical_axis_rules_cm(Role.CRITIC):
      return self.inference_worker.get_values(
          prompt_tokens,
          completion_tokens,
          pad_id,
          eos_id,
      )

  def get_rewards(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
  ) -> jax.Array:
    with self._get_mesh_and_logical_axis_rules_cm(Role.REWARD):
      return self.inference_worker.get_rewards(
          prompt_tokens,
          completion_tokens,
          pad_id,
          eos_id,
      )

  def get_rollout_config(self, mode: Mode) -> base_rollout.RolloutConfig:
    """Returns the rollout config for the given mode."""
    if isinstance(self.cluster_config.rollout_config, dict):
      return self.cluster_config.rollout_config[mode]
    else:
      return self.cluster_config.rollout_config

  def _generate_mesh_cm(self):
    """Mesh context for `generate`.

    By default `generate` enters the ROLLOUT mesh and its logical axis rules.
    With `TUNIX_ROLLOUT_NO_MESH_CONTEXT=1` it yields the ROLLOUT mesh without
    entering it. A host-local in-process engine (TUNIX_VLLM_HOST_LOCAL_ENGINE=1)
    sets its own per-process mesh inside `execute_model`; entering the global
    ROLLOUT mesh around it makes jit read two meshes, and eager ops in the
    engine's decode loop then fail on some processes with "Attempted to
    execute with N argument lists when local device count is M". Other roles
    and the ROLLOUT uses outside `generate` keep the normal context.

    Returns:
      A context manager yielding `(mesh, logical_axis_rules_or_None)`.
    """
    if _env_flag("TUNIX_ROLLOUT_NO_MESH_CONTEXT"):
      return contextlib.nullcontext(
          (self.cluster_config.role_to_mesh[Role.ROLLOUT], None)
      )
    return self._get_mesh_and_logical_axis_rules_cm(Role.ROLLOUT)

  @contextlib.contextmanager
  def _get_mesh_and_logical_axis_rules_cm(self, role: Role):
    """Returns a context manager for the mesh and logical axis rules.

    This is used for models that uses logical sharding, so XLA can generate the
    correct graph based on physical mesh.

    Args:
      role: The role of the model (e.g., ACTOR, CRITIC, REFERENCE, etc.).
    """
    role_logical_axis_rule = self.cluster_config.role_to_logical_axis_rule
    logical_axis_rule_ctx = contextlib.nullcontext()
    if role_logical_axis_rule and role in role_logical_axis_rule:
      logical_axis_rule_ctx = nn_partitioning.axis_rules(
          role_logical_axis_rule[role]
      )
    with contextlib.ExitStack() as stack:
      yield (
          stack.enter_context(self.cluster_config.role_to_mesh[role]),
          stack.enter_context(logical_axis_rule_ctx),
      )


# Alias for backward compatibility.
RLCluster = RLEngine
