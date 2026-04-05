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

"""Drop-in replacement for ``RLCluster`` that uses Ray for process isolation.

``RayRLCluster`` exposes the same public interface as ``RLCluster`` so that the
training script (e.g. ``train_deepscaler_nb.py``) needs only minimal changes:

  1. Import ``RayRLCluster`` instead of ``RLCluster``.
  2. Pass a ``weight_sync_strategy`` in ``ClusterConfig`` (optional; defaults
     to ``NumpyDirectSync``).
  3. Optionally call ``ray.init(...)`` before constructing the cluster.

Internal architecture::

    Orchestrator (main process)
    ├── TrainerActor  (Ray remote, exclusive trainer devices)
    │     actor model + reference model + optimiser
    └── RolloutActor  (Ray remote, exclusive rollout devices)
          rollout engine (vllm / sglang-jax / vanilla)

Weight sync is handled by a pluggable ``WeightSyncStrategy``.

Concurrency model
-----------------
The agentic learner (``AgenticRLLearner``) drives an asyncio-based producer
loop that calls ``generate()`` concurrently with the training loop.  Because
``ray.remote`` calls are non-blocking (they return futures), the orchestrator
can issue a ``generate`` remote call and immediately proceed without blocking
the Python thread, giving the same overlap as the existing single-process
Pathways setup.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable, Mapping

from absl import logging
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.ray import weight_sync as weight_sync_lib
from tunix.rl.ray.ray_trainer_actor import TrainerActor
from tunix.rl.ray.ray_rollout_actor import RolloutActor
from tunix.rl.rollout import base_rollout
from tunix.perf.experimental import tracer as perf_tracer_v2
from tunix.perf import trace as perf_trace
from tunix.perf import metrics as perf_metrics


# ---------------------------------------------------------------------------
# Extended ClusterConfig
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True, kw_only=True)
class RayClusterConfig(rl_cluster_lib.ClusterConfig):
  """Extends ``ClusterConfig`` with Ray-specific settings.

  Attributes:
    weight_sync_strategy: Strategy used to transfer updated policy weights from
      the trainer actor to the rollout actor after each training step.  Defaults
      to ``NumpyDirectSync`` (inline numpy transfer via Ray args).
    trainer_ray_options: Extra kwargs forwarded to ``TrainerActor.options()``,
      e.g. ``{"num_gpus": 4, "num_cpus": 8}``.
    rollout_ray_options: Extra kwargs forwarded to ``RolloutActor.options()``.
    ray_init_kwargs: If not ``None`` and Ray is not yet initialised, ``ray.init``
      is called with these kwargs when ``RayRLCluster`` is constructed.
  """

  weight_sync_strategy: weight_sync_lib.WeightSyncStrategy | None = None
  trainer_ray_options: dict[str, Any] = dataclasses.field(default_factory=dict)
  rollout_ray_options: dict[str, Any] = dataclasses.field(default_factory=dict)
  ray_init_kwargs: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Proxy objects that mimic trainer / rollout interfaces
# ---------------------------------------------------------------------------

class _TrainerProxy:
  """Local proxy that forwards attribute accesses and calls to ``TrainerActor``.

  ``AgenticRLLearner`` accesses ``rl_cluster.actor_trainer`` directly (e.g.
  ``actor_trainer.train_steps``, ``actor_trainer.is_managed_externally``).
  This proxy satisfies those accesses by making synchronous Ray calls.

  Proxy methods that the learner reads are cached after the first call; the
  few write-paths (``is_managed_externally``) are forwarded immediately.
  """

  def __init__(self, actor_handle: Any):
    self._handle = actor_handle
    import ray  # pylint: disable=g-import-not-at-top
    self._ray = ray
    self._meta: dict[str, Any] = {}
    self._refresh()

  def _refresh(self) -> None:
    self._meta = self._ray.get(self._handle.get_actor_trainer_state.remote())

  @property
  def train_steps(self) -> int:
    self._refresh()
    return self._meta["train_steps"]

  @property
  def iter_steps(self) -> int:
    self._refresh()
    return self._meta["iter_steps"]

  def restored_global_step(self) -> int:
    self._refresh()
    return self._meta["restored_global_step"]

  @property
  def is_managed_externally(self) -> bool:
    return self._meta.get("is_managed_externally", False)

  @is_managed_externally.setter
  def is_managed_externally(self, value: bool) -> None:
    self._ray.get(self._handle.set_actor_trainer_managed_externally.remote(value))
    self._meta["is_managed_externally"] = value

  # Allow attribute access for model (used in is_sharing_weights check).
  @property
  def model(self):
    raise AttributeError(
        "actor_trainer.model is not accessible from the orchestrator process "
        "in RayRLCluster. Use get_weights_numpy() on the TrainerActor instead."
    )

  def close(self) -> None:
    self._ray.get(self._handle.close.remote())


class _RolloutProxy:
  """Local proxy that exposes the ``BaseRollout`` interface over Ray calls."""

  def __init__(self, actor_handle: Any):
    self._handle = actor_handle
    import ray  # pylint: disable=g-import-not-at-top
    self._ray = ray
    self._pad_id: int | None = None
    self._eos_id: int | None = None

  def pad_id(self) -> int:
    if self._pad_id is None:
      self._pad_id = self._ray.get(self._handle.pad_id.remote())
    return self._pad_id

  def eos_id(self) -> int:
    if self._eos_id is None:
      self._eos_id = self._ray.get(self._handle.eos_id.remote())
    return self._eos_id

  # model() is intentionally not proxied – the learner should not hold
  # references to rollout model parameters across the process boundary.
  def model(self):
    raise AttributeError(
        "rollout.model() is not accessible from the orchestrator in "
        "RayRLCluster.  Use the weight sync abstraction instead."
    )

  def update_params(self, *args, **kwargs):
    raise AttributeError(
        "rollout.update_params() should not be called directly on a "
        "RayRLCluster proxy; use sync_weights() instead."
    )


# ---------------------------------------------------------------------------
# RayRLCluster
# ---------------------------------------------------------------------------

class RayRLCluster:
  """``RLCluster`` drop-in that splits trainer and rollout into Ray actors.

  Usage::

      import ray
      from tunix.rl.ray.ray_rl_cluster import RayRLCluster, RayClusterConfig
      from tunix.rl.ray.weight_sync import NumpyDirectSync

      ray.init()

      cluster_config = RayClusterConfig(
          role_to_mesh=...,
          rollout_engine="sglang_jax",
          training_config=...,
          rollout_config=...,
          weight_sync_strategy=NumpyDirectSync(),
      )

      rl_cluster = RayRLCluster(
          actor=qwen2_actor,
          reference=qwen2_ref,
          tokenizer=tokenizer,
          cluster_config=cluster_config,
      )

  The rest of the training script (``GRPOLearner``, ``AgenticRLLearner``)
  is unchanged because ``RayRLCluster`` mirrors the ``RLCluster`` API.
  """

  def __init__(
      self,
      *,
      actor: rl_cluster_lib.ModelOrPath,
      critic: rl_cluster_lib.ModelOrPath | None = None,
      reference: rl_cluster_lib.ModelOrPath | None = None,
      reward: rl_cluster_lib.ModelOrPath | None = None,
      tokenizer: Any | None,
      cluster_config: RayClusterConfig,
      perf_config: perf_metrics.PerfMetricsConfig | None = None,
  ):
    self.cluster_config = cluster_config

    # Initialise Ray if not already running.
    import ray  # pylint: disable=g-import-not-at-top
    if not ray.is_initialized():
      init_kwargs = cluster_config.ray_init_kwargs or {}
      logging.info("RayRLCluster: calling ray.init(%s)", init_kwargs)
      ray.init(**init_kwargs)

    self._ray = ray
    self._weight_sync: weight_sync_lib.WeightSyncStrategy = (
        cluster_config.weight_sync_strategy or weight_sync_lib.NumpyDirectSync()
    )

    # Extract numpy weights before forking into actor processes, so that both
    # actors can load the same initial weights without holding references to
    # JAX arrays that live on the main-process devices.
    logging.info("RayRLCluster: extracting initial weights as numpy …")
    actor_weights_np, actor_config, actor_graph = _extract_model_state(actor)
    ref_weights_np = None
    ref_config = None
    ref_graph = None
    if reference is not None:
      ref_weights_np, ref_config, ref_graph = _extract_model_state(reference)

    # Build factory closures – these will be executed *inside* the Ray actors.
    trainer_factory = functools.partial(
        _build_trainer_cluster,
        actor_weights_np=actor_weights_np,
        actor_graph=actor_graph,
        ref_weights_np=ref_weights_np,
        ref_graph=ref_graph,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
        perf_config=perf_config,
    )
    rollout_factory = functools.partial(
        _build_rollout_cluster,
        actor_weights_np=actor_weights_np,
        actor_graph=actor_graph,
        tokenizer=tokenizer,
        cluster_config=cluster_config,
    )

    # Create Ray remote actor classes with any user-supplied resource hints.
    TrainerActorRemote = ray.remote(TrainerActor)
    RolloutActorRemote = ray.remote(RolloutActor)
    if cluster_config.trainer_ray_options:
      TrainerActorRemote = TrainerActorRemote.options(
          **cluster_config.trainer_ray_options
      )
    if cluster_config.rollout_ray_options:
      RolloutActorRemote = RolloutActorRemote.options(
          **cluster_config.rollout_ray_options
      )

    logging.info("RayRLCluster: starting TrainerActor …")
    self._trainer_handle = TrainerActorRemote.remote(trainer_factory)
    logging.info("RayRLCluster: starting RolloutActor …")
    self._rollout_handle = RolloutActorRemote.remote(rollout_factory)

    # Block until both actors are ready.
    ray.get([
        self._trainer_handle.get_global_steps.remote(),
        self._rollout_handle.pad_id.remote(),
    ])
    logging.info("RayRLCluster: both actors ready.")

    # Proxies that satisfy learner attribute access patterns.
    self._actor_trainer_proxy = _TrainerProxy(self._trainer_handle)
    self._rollout_proxy = _RolloutProxy(self._rollout_handle)

    # Metrics are managed locally in the orchestrator process.
    from tunix.sft import metrics_logger  # pylint: disable=g-import-not-at-top
    self._rl_metrics_logger = metrics_logger.MetricsLogger(
        cluster_config.training_config.metrics_logging_options
    )
    self._buffered_train_metrics: list[perf_metrics.MetricsBuffer] = []
    self._buffered_eval_metrics: list[perf_metrics.MetricsBuffer] = []
    self._external_metrics_logger = None

    self.global_steps = 0
    # Read back the restored checkpoint step from the trainer actor.
    self.global_steps = ray.get(
        self._trainer_handle.get_global_steps.remote()
    )

    # Noop perf tracers (perf tracing happens inside actors).
    self._perf = perf_trace.NoopTracer()
    self._perf_v2 = perf_tracer_v2.NoopTracer()

    # Stash tokenizer for chat-template application (happens in orchestrator).
    from tunix.generate import tokenizer_adapter  # pylint: disable=g-import-not-at-top
    self.tokenizer = tokenizer_adapter.TokenizerAdapter(tokenizer)

  # ---------------------------------------------------------------------------
  # Properties (mirror RLCluster interface)
  # ---------------------------------------------------------------------------

  @property
  def actor_trainer(self):
    """Proxy to the remote trainer actor's trainer state."""
    return self._actor_trainer_proxy

  @property
  def rollout(self):
    """Proxy to the remote rollout actor's rollout interface."""
    return self._rollout_proxy

  @property
  def perf(self) -> perf_trace.Tracer:
    return self._perf

  @property
  def perf_v2(self) -> perf_tracer_v2.Tracer:
    return self._perf_v2

  # ---------------------------------------------------------------------------
  # Training
  # ---------------------------------------------------------------------------

  def update_actor(
      self,
      train_ds: list[Any],
      eval_ds: list[Any] | None,
      skip_jit: bool = False,
  ) -> None:
    """Forward a training step to the trainer actor (blocking)."""
    # Convert JAX arrays to numpy before crossing the Ray boundary.
    train_ds_np = _pytree_to_numpy(train_ds)
    eval_ds_np = _pytree_to_numpy(eval_ds) if eval_ds else None
    self._ray.get(
        self._trainer_handle.update_actor.remote(train_ds_np, eval_ds_np, skip_jit)
    )

  def update_critic(
      self,
      train_ds: list[Any],
      eval_ds: list[Any] | None,
      skip_jit: bool = False,
  ) -> None:
    train_ds_np = _pytree_to_numpy(train_ds)
    eval_ds_np = _pytree_to_numpy(eval_ds) if eval_ds else None
    self._ray.get(
        self._trainer_handle.update_critic.remote(train_ds_np, eval_ds_np, skip_jit)
    )

  # ---------------------------------------------------------------------------
  # Generation
  # ---------------------------------------------------------------------------

  def generate(
      self,
      prompts: list[str] | list[list[dict[str, str]]],
      apply_chat_template: bool = False,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
      micro_batch_size: int | None = None,
      trace_tags: Mapping[str, Any] | None = None,
  ) -> base_rollout.RolloutOutput:
    """Forward a generate call to the rollout actor.

    This call is **synchronous** when called from the learner's asyncio loop
    via ``run_in_executor``, which preserves the existing overlap semantics.
    """
    if apply_chat_template:
      if self.tokenizer is None:
        raise ValueError("Tokenizer must be initialised to apply chat template.")
      prompts = [
          self.tokenizer.apply_chat_template(
              p,
              add_generation_prompt=True,
              tokenize=False,
              enable_thinking=False,
          )
          for p in prompts
      ]
      apply_chat_template = False  # already applied

    result_dict = self._ray.get(
        self._rollout_handle.generate.remote(
            prompts=prompts,
            apply_chat_template=apply_chat_template,
            mode_str=mode.value,
            micro_batch_size=micro_batch_size,
            trace_tags=dict(trace_tags) if trace_tags else None,
        )
    )

    # Reconstruct a RolloutOutput from the plain-dict result.
    return base_rollout.RolloutOutput(
        text=result_dict["text"],
        tokens=result_dict["tokens"],
        logits=result_dict["logits"],
        logprobs=result_dict["logprobs"],
        left_padded_prompt_tokens=result_dict["left_padded_prompt_tokens"],
    )

  # ---------------------------------------------------------------------------
  # Reference log-probs
  # ---------------------------------------------------------------------------

  def get_ref_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      pad_id: int,
      eos_id: int,
      micro_batch_size: int | None = None,
      completion_mask: jax.Array | None = None,
  ) -> jax.Array:
    """Delegate to trainer actor; returns a JAX array on the local device."""
    result_np = self._ray.get(
        self._trainer_handle.get_ref_per_token_logps.remote(
            np.asarray(prompt_tokens),
            np.asarray(completion_tokens),
            pad_id,
            eos_id,
            micro_batch_size,
            np.asarray(completion_mask) if completion_mask is not None else None,
        )
    )
    return jnp.asarray(result_np)

  # ---------------------------------------------------------------------------
  # Weight sync
  # ---------------------------------------------------------------------------

  def sync_weights(self) -> None:
    """Transfer updated policy weights from trainer actor to rollout actor.

    Uses the configured ``WeightSyncStrategy``.  After the sync, increments
    ``global_steps`` to match the existing ``RLCluster.sync_weights`` contract.
    """
    logging.info(
        "RayRLCluster.sync_weights: using %s",
        type(self._weight_sync).__name__,
    )
    self._weight_sync.sync(self._trainer_handle, self._rollout_handle)
    self.global_steps += 1
    self._ray.get(self._trainer_handle.set_global_steps.remote(self.global_steps))
    logging.info(
        "RayRLCluster.sync_weights: done. global_steps=%d", self.global_steps
    )

  # ---------------------------------------------------------------------------
  # Metrics
  # ---------------------------------------------------------------------------

  def with_external_metrics_logger(
      self, external_metrics_logger: Callable[[perf_metrics.MetricsBuffer], None]
  ) -> "RayRLCluster":
    self._external_metrics_logger = external_metrics_logger
    return self

  def buffer_metrics_async(
      self,
      metrics: perf_metrics.MetricsT,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
      step: int = 0,
  ) -> None:
    """Buffer metrics locally (same logic as RLCluster.buffer_metrics_async)."""
    if mode == rl_cluster_lib.Mode.TRAIN:
      buffered_metrics = self._buffered_train_metrics
    else:
      buffered_metrics = self._buffered_eval_metrics

    if not buffered_metrics:
      buffered_metrics.append(perf_metrics.MetricsBuffer(self.global_steps, mode=str(mode)))
    else:
      if step != buffered_metrics[-1].global_steps:
        buffered_metrics.append(perf_metrics.MetricsBuffer(step, mode=str(mode)))

    cur = buffered_metrics[-1]
    for metric_name, (value, op) in metrics.items():
      if metric_name not in cur.metrics:
        cur.metrics[metric_name] = ([value], op)
      else:
        cur.metrics[metric_name][0].append(value)

    # Flush old buffers.
    if (
        self._buffered_train_metrics
        and self._buffered_train_metrics[0].global_steps < self.global_steps
    ):
      self._log_metrics(self._buffered_train_metrics.pop(0))
    if (
        self._buffered_eval_metrics
        and self._buffered_eval_metrics[0].global_steps < self.global_steps
    ):
      self._log_metrics(self._buffered_eval_metrics.pop(0))

  def _log_metrics(self, metrics_buffer: perf_metrics.MetricsBuffer) -> None:
    for metric_name, (value, op) in metrics_buffer.metrics.items():
      try:
        agg_value = np.array(value)
      except Exception:
        continue
      if agg_value.dtype.kind in {"U", "S", "O"}:
        continue
      if op is not None and agg_value.size > 0:
        agg_value = op(agg_value)
      prefix, short_name = (
          metric_name.split("/", maxsplit=1)
          if "/" in metric_name
          else ("global", metric_name)
      )
      self._rl_metrics_logger.log(
          prefix, short_name, agg_value, metrics_buffer.mode, metrics_buffer.global_steps
      )
    if self._external_metrics_logger is not None:
      self._external_metrics_logger(metrics_buffer)

  # ---------------------------------------------------------------------------
  # Lifecycle
  # ---------------------------------------------------------------------------

  def close(self) -> None:
    for m in self._buffered_train_metrics + self._buffered_eval_metrics:
      self._log_metrics(m)
    self._ray.get(self._trainer_handle.close.remote())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_model_state(model: nnx.Module) -> tuple[Any, Any, Any]:
  """Extract (numpy_state, graph_def, treedef) from an NNX model."""
  graph_def, state = nnx.split(model)
  numpy_state = jax.tree_util.tree_map(np.asarray, state)
  return numpy_state, model.config if hasattr(model, "config") else None, graph_def


def _build_trainer_cluster(
    *,
    actor_weights_np: Any,
    actor_graph: Any,
    ref_weights_np: Any | None,
    ref_graph: Any | None,
    tokenizer: Any,
    cluster_config: RayClusterConfig,
    perf_config: Any | None,
) -> rl_cluster_lib.RLCluster:
  """Reconstruct actor+reference models and build an RLCluster inside the actor.

  This runs **inside** the TrainerActor Ray process.
  """
  import jax  # pylint: disable=g-import-not-at-top

  logging.info(
      "_build_trainer_cluster: JAX devices = %s", jax.devices()
  )
  trainer_mesh = cluster_config.role_to_mesh[rl_cluster_lib.Role.ACTOR]
  actor_model = _restore_model(actor_graph, actor_weights_np, trainer_mesh)

  ref_model = None
  if ref_weights_np is not None:
    ref_model = _restore_model(ref_graph, ref_weights_np, trainer_mesh)

  # Use vanilla rollout as a placeholder inside the trainer cluster so that
  # RLCluster initialisation succeeds; this rollout is never used for actual
  # generation (the dedicated RolloutActor handles that).
  trainer_config = rl_cluster_lib.ClusterConfig(
      role_to_mesh={**cluster_config.role_to_mesh,
                    rl_cluster_lib.Role.ROLLOUT: trainer_mesh},
      role_to_logical_axis_rule=cluster_config.role_to_logical_axis_rule,
      rollout_engine="vanilla",
      offload_to_cpu=cluster_config.offload_to_cpu,
      training_config=cluster_config.training_config,
      rollout_config=cluster_config.rollout_config,
  )

  return rl_cluster_lib.RLCluster(
      actor=actor_model,
      reference=ref_model,
      tokenizer=tokenizer,
      cluster_config=trainer_config,
      perf_config=perf_config,
  )


def _build_rollout_cluster(
    *,
    actor_weights_np: Any,
    actor_graph: Any,
    tokenizer: Any,
    cluster_config: RayClusterConfig,
) -> rl_cluster_lib.RLCluster:
  """Reconstruct the rollout model and build an RLCluster inside the rollout actor.

  This runs **inside** the RolloutActor Ray process.
  """
  import jax  # pylint: disable=g-import-not-at-top

  logging.info(
      "_build_rollout_cluster: JAX devices = %s", jax.devices()
  )
  rollout_mesh = cluster_config.role_to_mesh[rl_cluster_lib.Role.ROLLOUT]
  rollout_model = _restore_model(actor_graph, actor_weights_np, rollout_mesh)

  # Build a minimal config that only initialises the rollout.
  rollout_only_config = rl_cluster_lib.ClusterConfig(
      role_to_mesh={
          rl_cluster_lib.Role.ACTOR: rollout_mesh,
          rl_cluster_lib.Role.REFERENCE: rollout_mesh,
          rl_cluster_lib.Role.ROLLOUT: rollout_mesh,
      },
      role_to_logical_axis_rule=cluster_config.role_to_logical_axis_rule,
      rollout_engine=cluster_config.rollout_engine,
      offload_to_cpu=cluster_config.offload_to_cpu,
      training_config=cluster_config.training_config,
      rollout_config=cluster_config.rollout_config,
  )

  return rl_cluster_lib.RLCluster(
      actor=rollout_model,
      reference=None,
      tokenizer=tokenizer,
      cluster_config=rollout_only_config,
  )


def _restore_model(
    graph_def: Any,
    numpy_state: Any,
    mesh: jax.sharding.Mesh,
) -> nnx.Module:
  """Reconstruct an NNX model on ``mesh`` from graph + numpy state."""
  from tunix.rl import reshard  # pylint: disable=g-import-not-at-top

  jax_state = jax.tree_util.tree_map(jnp.asarray, numpy_state)
  model = nnx.merge(graph_def, jax_state)

  # Reshard to the target mesh.
  graph, state = nnx.split(model)
  dst_shardings = jax.tree_util.tree_map(
      lambda x: jax.sharding.NamedSharding(mesh, x.sharding.spec)
      if hasattr(x, "sharding") and hasattr(x.sharding, "spec")
      else x.sharding,
      state,
  )
  resharded_state = reshard.reshard_pytree(state, dst_shardings)
  return nnx.merge(graph, resharded_state)


def _pytree_to_numpy(pytree: Any) -> Any:
  """Recursively convert JAX arrays to numpy in a pytree."""
  if pytree is None:
    return None
  return jax.tree_util.tree_map(
      lambda x: np.asarray(x) if isinstance(x, jax.Array) else x,
      pytree,
  )
