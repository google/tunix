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

"""Orchestrator-backed RLCluster.

`OrchestratorRLCluster` is the second implementation of the cluster surface
(alongside the in-process `RLCluster`): it satisfies the same API a learner
drives, but routes the compute primitives -- generation, training, weight sync,
reference/actor scoring -- to role-based worker handles instead of running them
in-process. Because the agentic learners call these primitives directly on
`rl_cluster`, swapping in this cluster distributes the loop with no learner
changes.

It is built by composition over a base `RLCluster`: the routed primitives go to
the handles (falling back to the base when a handle is absent, for incremental
bring-up), and everything else the loop reads -- `cluster_config`, `rollout`,
`actor_trainer`, `perf_v2`, tokenizer, metric buffering, etc. -- is delegated to
the base. As real workers replace in-process pieces, more of the surface moves
onto handles and less is delegated.

Handle contracts (all optional; absent -> in-process fallback):
    trainer_worker.train(chunks, eval_ds, skip_jit) -> None   # actor only
    trainer_worker.train_critic(chunks, eval_ds, skip_jit) -> None  # optional
    trainer_worker.per_token_logps(prompt_ids, completion_ids,
                                   pad_id, eos_id) -> array   # optional method
    rollout_worker.generate(prompts, apply_chat_template, mode, micro_batch_size,
                            trace_tags, max_generation_steps) -> RolloutOutput
    inference_worker.per_token_logps(prompt_ids, completion_ids,
                                     pad_id, eos_id) -> array
    weight_sync.sync() -> None
"""

from typing import Any, Mapping

from tunix.rl import rl_cluster as rl_cluster_lib


class OrchestratorRLCluster:
  """Worker-backed cluster that routes compute primitives and delegates the rest."""

  def __init__(
      self,
      base: rl_cluster_lib.RLCluster,
      *,
      trainer_worker: Any = None,
      rollout_worker: Any = None,
      inference_worker: Any = None,
      weight_sync: Any = None,
  ):
    """Initializes the orchestrator cluster.

    Args:
      base: The in-process cluster that supplies the full surface (models,
        trainers, rollout, config, metrics, step counter). Routed primitives
        fall back to it when the matching handle is not provided.
      trainer_worker: Optional handle exposing `train(...)` (and optionally
        `per_token_logps(...)`) for the actor trainer.
      rollout_worker: Optional handle exposing `generate(...)`.
      inference_worker: Optional handle exposing `per_token_logps(...)` for the
        frozen reference model.
      weight_sync: Optional handle exposing `sync()`.
    """
    self._base = base
    self._trainer_worker = trainer_worker
    self._rollout_worker = rollout_worker
    self._inference_worker = inference_worker
    self._weight_sync = weight_sync

  # --- Shared bookkeeping (read + write, kept on the base) -------------------

  @property
  def global_steps(self) -> int:
    return self._base.global_steps

  @global_steps.setter
  def global_steps(self, value: int) -> None:
    self._base.global_steps = value

  # --- Generation (rollout) -------------------------------------------------

  def generate(
      self,
      prompts: Any,
      apply_chat_template: bool = False,
      mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
      micro_batch_size: int | None = None,
      trace_tags: Mapping[str, Any] | None = None,
      max_generation_steps: int | None = None,
  ) -> Any:
    if self._rollout_worker is not None:
      return self._rollout_worker.generate(
          prompts=prompts,
          apply_chat_template=apply_chat_template,
          mode=mode,
          micro_batch_size=micro_batch_size,
          trace_tags=trace_tags,
          max_generation_steps=max_generation_steps,
      )
    return self._base.generate(
        prompts,
        apply_chat_template,
        mode,
        micro_batch_size,
        trace_tags,
        max_generation_steps,
    )

  # --- Training (train_step) ------------------------------------------------

  def update_actor(self, train_ds: Any, eval_ds: Any, skip_jit: bool = False) -> None:
    if self._trainer_worker is not None:
      self._trainer_worker.train(train_ds, eval_ds, skip_jit)
    else:
      self._base.update_actor(train_ds, eval_ds, skip_jit)

  def update_critic(self, train_ds: Any, eval_ds: Any, skip_jit: bool = False) -> None:
    if self._trainer_worker is not None and hasattr(
        self._trainer_worker, "train_critic"
    ):
      self._trainer_worker.train_critic(train_ds, eval_ds, skip_jit)
    else:
      self._base.update_critic(train_ds, eval_ds, skip_jit)

  # --- Scoring --------------------------------------------------------------

  def get_ref_per_token_logps(
      self,
      prompt_tokens: Any,
      completion_tokens: Any,
      pad_id: int,
      eos_id: int,
      micro_batch_size: int | None = None,
  ) -> Any:
    if self._inference_worker is not None:
      return self._inference_worker.per_token_logps(
          prompt_ids=prompt_tokens,
          completion_ids=completion_tokens,
          pad_id=pad_id,
          eos_id=eos_id,
      )
    return self._base.get_ref_per_token_logps(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        pad_id=pad_id,
        eos_id=eos_id,
        micro_batch_size=micro_batch_size,
    )

  def get_actor_per_token_logps(
      self,
      prompt_tokens: Any,
      completion_tokens: Any,
      pad_id: int,
      eos_id: int,
      micro_batch_size: int | None = None,
  ) -> Any:
    per_token_logps = getattr(self._trainer_worker, "per_token_logps", None)
    if per_token_logps is not None:
      return per_token_logps(
          prompt_ids=prompt_tokens,
          completion_ids=completion_tokens,
          pad_id=pad_id,
          eos_id=eos_id,
      )
    return self._base.get_actor_per_token_logps(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        pad_id=pad_id,
        eos_id=eos_id,
        micro_batch_size=micro_batch_size,
    )

  # --- Weight sync ----------------------------------------------------------

  def sync_weights(self) -> None:
    if self._weight_sync is not None:
      self._weight_sync.sync()
    else:
      self._base.sync_weights()

  # --- Everything else is delegated to the in-process base cluster ----------

  def __getattr__(self, name: str) -> Any:
    # Only reached for names not defined on this class (the routed primitives,
    # global_steps, and dunders are defined above). Delegates the rest of the
    # cluster surface -- cluster_config, rollout, actor_trainer, perf_v2,
    # tokenizer, buffer_metrics, close, ... -- to the wrapped base cluster.
    base = self.__dict__.get("_base")
    if base is None:
      raise AttributeError(name)
    return getattr(base, name)
