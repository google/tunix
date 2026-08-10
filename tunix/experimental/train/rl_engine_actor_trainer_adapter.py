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

"""Adapter exposing the existing RL actor trainer as an AbstractTrainer."""

from __future__ import annotations

import logging
from typing import Any, Callable

from flax import nnx
import numpy as np
from tunix.experimental.common import datatypes
from tunix.experimental.metrics import metrics as exp_metrics
from tunix.experimental.train import abstract_trainer
from tunix.rl import common
from tunix.rl import function_registry
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.rl.agentic import agentic_grpo_learner


class RLEngineActorTrainerAdapter(abstract_trainer.AbstractTrainer):
  """Bridges legacy RLEngine actor updates into the step-level trainer API.

  The experimental TrainerWorker expects an AbstractTrainer with explicit
  `fwd_bwd()` and `update()` calls. The current production RLEngine still owns
  the concrete actor training loop through `update_actor(train_ds, ...)`.
  This adapter therefore stages micro-batches in `fwd_bwd()` and invokes the
  existing actor update once from `update()`, preserving current GRPO semantics
  while moving orchestration onto the new worker lifecycle.
  """

  def __init__(
      self,
      rl_engine: rl_engine_lib.RLEngine,
      config: Any | None = None,
  ):
    self.config = config
    self._rl_engine = rl_engine
    self._pending_payloads: list[Any] = []
    self._policy_version = 0
    self._last_lora_weights: Any = None

  @property
  def actor_trainer(self) -> Any:
    return self._rl_engine.actor_trainer

  @property
  def policy_version(self) -> int:
    return self._policy_version

  @property
  def train_steps(self) -> int:
    return int(getattr(self.actor_trainer, "train_steps", 0))

  def with_loss_fn(
      self, loss_fn: Callable[..., Any], has_aux: bool = False
  ) -> "RLEngineActorTrainerAdapter":
    self.actor_trainer.with_loss_fn(loss_fn, has_aux)
    return self

  def with_gen_model_input_fn(
      self, gen_model_input_fn: Callable[[Any], dict[str, Any]]
  ) -> "RLEngineActorTrainerAdapter":
    self.actor_trainer.with_gen_model_input_fn(gen_model_input_fn)
    return self

  def configure_grpo_loss(
      self,
      *,
      num_generations: int,
      max_response_length: int,
      beta: float = 0.0,
      epsilon: float = 0.2,
      epsilon_high: float | None = None,
      kl_loss_mode: str = "mse_kl",
      force_compute_kl: bool = False,
      loss_agg_mode: str = "sequence-mean-token-mean",
      loss_algo: str = "grpo",
      policy_loss_fn: str = "grpo",
      advantage_estimator: str = "grpo",
      temperature: float | None = None,
      sampler_is: str | None = None,
      sampler_is_threshold: float = 2.0,
  ) -> dict[str, Any]:
    """Configures trainer-side GRPO loss without shipping closures over RPC."""
    logging.info("[RLEngineActorTrainerAdapter] Configuring GRPO loss.")
    configured_algo_config = agentic_grpo_learner.GRPOConfig(
        num_generations=num_generations,
        num_iterations=1,
        beta=beta,
        kl_loss_mode=kl_loss_mode,
        force_compute_kl=force_compute_kl,
        epsilon=epsilon,
        epsilon_high=epsilon if epsilon_high is None else epsilon_high,
        advantage_estimator=advantage_estimator,
        policy_loss_fn=policy_loss_fn,
        loss_agg_mode=loss_agg_mode,
        loss_algo=loss_algo,
        degenerate_group_masking=False,
        use_rollout_logps=False,
        sampler_is=sampler_is,
        sampler_is_threshold=sampler_is_threshold,
        max_response_length=max_response_length,
    )
    if temperature is None:
      temperature = self._rl_engine.get_rollout_config(
          mode=rl_engine_lib.Mode.TRAIN
      ).temperature
    configured_algo_config.temperature = temperature

    policy_loss_impl = function_registry.get_policy_loss_fn(
        configured_algo_config.policy_loss_fn
    )

    def loss_fn(model: nnx.Module, train_example: Any, algo_config: Any):
      del algo_config
      return policy_loss_impl(
          model,
          train_example,
          algo_config=configured_algo_config,
          pad_id=self._rl_engine.rollout.pad_id(),
          eos_id=self._rl_engine.rollout.eos_id(),
          compute_logps_chunk_size=(
              self._rl_engine.cluster_config.training_config.compute_logps_chunk_size
          ),
      )

    self.with_loss_fn(loss_fn, has_aux=True)
    self.with_gen_model_input_fn(
        lambda x: {"train_example": x, "algo_config": configured_algo_config}
    )
    self.actor_trainer.is_managed_externally = True
    self._configure_rl_metrics(configured_algo_config)
    return {
        "configured_loss": "grpo",
        "temperature": temperature,
        "policy_version": self._policy_version,
    }

  def _configure_rl_metrics(
      self, algo_config: agentic_grpo_learner.GRPOConfig
  ) -> None:
    with_metrics = getattr(self.actor_trainer, "with_rl_metrics_to_log", None)
    if callable(with_metrics):
      with_metrics({
          "kl": common.mean_of_means,
          "entropy": common.mean_of_means,
          "reduced_pg_loss": common.mean_of_means,
          "unreduced_pg_loss": common.global_weighted_mean,
          "pg_clipfrac": common.mean_of_means,
          "ppo_kl": common.mean_of_means,
          "kl_loss": common.mean_of_means,
          "is_ratio/mean": common.mean_of_means,
          "is_ratio/max": np.max,
          "is_ratio/min": np.min,
          "log_ratio/abs_mean": common.mean_of_means,
          "pg_loss/unclipped_mean": common.mean_of_means,
          "pg_loss/clipped_mean": common.mean_of_means,
          "advantage/abs_mean": common.mean_of_means,
          "advantage/max": np.max,
          "advantage/min": np.min,
          "advantage/nonzero_frac": common.mean_of_means,
      })

    with_tqdm = getattr(self.actor_trainer, "with_tqdm_metrics_to_display", None)
    if callable(with_tqdm):
      with_tqdm([
          lambda: "kl"
          if algo_config.force_compute_kl or algo_config.beta != 0.0
          else None,
      ])

  def compile(self, dummy_data: Any) -> None:
    del dummy_data
    # The legacy RLEngine compiles on the first concrete batch shape. Keeping
    # this hook idempotent gives the orchestrator a common worker barrier.
    logging.info(
        "[RLEngineActorTrainerAdapter] Compile deferred until first batch."
    )

  def fwd_bwd(self, payload: datatypes.TrainerPayload, **kwargs) -> None:
    del kwargs
    if isinstance(payload, list):
      self._pending_payloads.extend(payload)
    else:
      self._pending_payloads.append(payload)

  def update(self, **kwargs) -> int:
    if not self._pending_payloads:
      logging.info("[RLEngineActorTrainerAdapter] No pending payloads to update.")
      return self.train_steps

    eval_ds = kwargs.pop("eval_ds", None)
    skip_jit = kwargs.pop("skip_jit", False)
    if kwargs:
      raise ValueError(f"Unexpected update kwargs: {sorted(kwargs)}")

    chunks = self._pending_payloads
    self._pending_payloads = []
    try:
      logging.info(
          "[RLEngineActorTrainerAdapter] Applying actor update with %d chunks.",
          len(chunks),
      )
      self._rl_engine.update_actor(chunks, eval_ds, skip_jit=skip_jit)
    except Exception:
      self._pending_payloads = chunks + self._pending_payloads
      raise

    self._policy_version += 1
    self._last_lora_weights = None
    return self.train_steps

  def eval_step(self, payload: datatypes.TrainerPayload, **kwargs) -> None:
    skip_jit = kwargs.pop("skip_jit", False)
    cache_nnx_graph = kwargs.pop("cache_nnx_graph", True)
    if kwargs:
      raise ValueError(f"Unexpected eval_step kwargs: {sorted(kwargs)}")
    _, eval_step = self.actor_trainer.jit_train_and_eval_step(
        skip_jit, cache_nnx_graph
    )
    self.actor_trainer._run_eval(  # pylint: disable=protected-access
        [payload], eval_step
    )

  def save_checkpoint(self, metadata: Any, **kwargs) -> None:
    force = kwargs.pop("force", True)
    if kwargs:
      raise ValueError(f"Unexpected save_checkpoint kwargs: {sorted(kwargs)}")
    checkpoint_manager = getattr(self.actor_trainer, "checkpoint_manager", None)
    if checkpoint_manager is None:
      raise NotImplementedError(
          "Underlying actor trainer has no checkpoint_manager."
      )
    checkpoint_manager.save(
        self.train_steps,
        self.actor_trainer.model,
        self.actor_trainer.optimizer,
        save_only_lora_params=getattr(self.actor_trainer, "_lora_enabled", False),
        custom_metadata=metadata,
        force=force,
    )

  def restore_checkpoint(self, **kwargs) -> Any:
    if kwargs:
      raise ValueError(f"Unexpected restore_checkpoint kwargs: {sorted(kwargs)}")
    return getattr(self.actor_trainer, "_restored_custom_metadata", {})

  def prepare_weight_sync(self, **kwargs) -> None:
    if kwargs:
      raise ValueError(f"Unexpected prepare_weight_sync kwargs: {sorted(kwargs)}")
    self._last_lora_weights = nnx.state(self.actor_trainer.model, nnx.LoRAParam)

  def get_lora_weights(self) -> Any:
    if self._last_lora_weights is None:
      self.prepare_weight_sync()
    return self._last_lora_weights

  def get_metrics(self) -> exp_metrics.MetricsBuffer:
    return exp_metrics.MetricsBuffer(id=self.train_steps)

  def per_token_logps(
      self,
      prompt_ids: Any,
      completion_ids: Any,
      pad_id: int,
      eos_id: int,
  ) -> Any:
    return self._rl_engine.get_actor_per_token_logps(
        prompt_tokens=prompt_ids,
        completion_tokens=completion_ids,
        pad_id=pad_id,
        eos_id=eos_id,
    )

  def reference_logps(
      self,
      prompt_ids: Any,
      completion_ids: Any,
      pad_id: int,
      eos_id: int,
  ) -> Any:
    return self._rl_engine.get_ref_per_token_logps(
        prompt_tokens=prompt_ids,
        completion_tokens=completion_ids,
        pad_id=pad_id,
        eos_id=eos_id,
    )

  def close(self) -> None:
    close = getattr(self._rl_engine, "close", None)
    if callable(close):
      close()
    else:
      self.actor_trainer.close()
