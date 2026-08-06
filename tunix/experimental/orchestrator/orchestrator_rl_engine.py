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

"""Orchestrator-backed RLEngine.

`OrchestratorRLEngine` is the second implementation of the engine surface
(alongside the in-process `RLEngine`): it satisfies the same API a learner
drives, but routes the compute primitives -- generation, training, weight sync,
reference/actor scoring -- to role-based worker handles instead of running them
in-process. Because the agentic learners call these primitives directly on
`rl_engine`, swapping in this engine distributes the loop with no learner
changes.

It can run in two modes. In incremental mode it is built by composition over a
base `RLEngine`: routed primitives go to handles and fall back to the base when
a handle is absent. In remote-only mode the CPU orchestrator provides only
metadata (`cluster_config`, tokenizer, rollout pad/eos ids) and every compute
primitive must be backed by worker handles, so the orchestrator process never
loads the model or initializes TPU.

Handle contracts (all optional; absent -> in-process fallback):
    trainer_worker.fwd_bwd(chunk) -> Response
    trainer_worker.update(skip_jit=...) -> int
    trainer_worker.run_eval(eval_ds) -> Response
    trainer_worker.eval_step(chunk) -> Response
    trainer_worker.per_token_logps(prompt_ids, completion_ids,
                                   pad_id, eos_id) -> array   # optional method
    rollout_worker.generate(prompts, apply_chat_template, mode,
    micro_batch_size,
                            trace_tags, max_generation_steps) -> RolloutOutput
    inference_worker.per_token_logps(prompt_ids, completion_ids,
                                     pad_id, eos_id) -> array
    weight_sync.sync() -> None
"""

from typing import Any, Mapping

from tunix.rl import rl_cluster as rl_engine_lib


class _TokenIdRolloutView:
  """Small rollout metadata view used by remote-only orchestrators."""

  def __init__(self, pad_id: int, eos_id: int):
    self._pad_id = pad_id
    self._eos_id = eos_id

  def pad_id(self) -> int:
    return self._pad_id

  def eos_id(self) -> int:
    return self._eos_id


class OrchestratorRLEngine:
  """Worker-backed cluster that routes compute primitives and delegates the rest."""

  def __init__(
      self,
      base: rl_engine_lib.RLEngine | None = None,
      *,
      trainer_worker: Any = None,
      rollout_worker: Any = None,
      inference_worker: Any = None,
      weight_sync: Any = None,
      cluster_config: Any = None,
      rollout: Any = None,
      actor_trainer: Any = None,
      tokenizer: Any = None,
      r2m: Any = None,
      pad_id: int | None = None,
      eos_id: int | None = None,
  ):
    """Initializes the orchestrator cluster.

    Args:
      base: Optional in-process cluster that supplies the full surface (models,
        trainers, rollout, config, metrics, step counter). Routed primitives
        fall back to it when the matching handle is not provided. For a
        remote-only CPU orchestrator, pass no base and provide the metadata the
        loop needs (`cluster_config`, tokenizer, pad/eos ids) explicitly.
      trainer_worker: Optional handle exposing `fwd_bwd(...)`, `update(...)`,
        and optionally `per_token_logps(...)` for the actor trainer.
      rollout_worker: Optional handle exposing `generate(...)`.
      inference_worker: Optional handle exposing `per_token_logps(...)` for the
        frozen reference model.
      weight_sync: Optional handle exposing `sync()`.
      cluster_config: Optional cluster config metadata for remote-only mode.
      rollout: Optional rollout metadata surface. If omitted and pad/eos ids
        are supplied, a small `pad_id()/eos_id()` view is created.
      actor_trainer: Optional actor trainer metadata surface.
      tokenizer: Optional tokenizer used by learner-side code.
      r2m: Optional role-to-mesh metadata.
      pad_id: Optional tokenizer pad id for remote-only rollout metadata.
      eos_id: Optional tokenizer eos id for remote-only rollout metadata.
    """
    self._base = base
    self._trainer_worker = trainer_worker
    self._rollout_worker = rollout_worker
    self._inference_worker = inference_worker
    self._weight_sync = weight_sync
    self._cluster_config = cluster_config
    self._rollout = rollout
    if self._rollout is None and pad_id is not None and eos_id is not None:
      self._rollout = _TokenIdRolloutView(pad_id, eos_id)
    self._actor_trainer = actor_trainer
    self._tokenizer = tokenizer
    self._r2m = r2m
    self._global_steps = 0
    self._buffered_metrics: list[tuple[Any, dict[str, Any]]] = []
    self._buffered_async_metrics: list[tuple[Any, dict[str, Any]]] = []

  def _require_base(self, method_name: str) -> rl_engine_lib.RLEngine:
    if self._base is None:
      raise RuntimeError(
          f"{method_name} requires either a remote worker/proxy or a base "
          "RLEngine. This OrchestratorRLEngine was constructed in remote-only "
          "mode without the required worker."
      )
    return self._base

  @property
  def has_trainer_worker(self) -> bool:
    return self._trainer_worker is not None

  # --- Shared bookkeeping (read + write, kept on the base) -------------------

  @property
  def global_steps(self) -> int:
    if self._base is not None:
      return self._base.global_steps
    return self._global_steps

  @global_steps.setter
  def global_steps(self, value: int) -> None:
    if self._base is not None:
      self._base.global_steps = value
    self._global_steps = value

  # --- Generation (rollout) -------------------------------------------------

  def generate(
      self,
      prompts: Any,
      apply_chat_template: bool = False,
      mode: rl_engine_lib.Mode = rl_engine_lib.Mode.TRAIN,
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
    return self._require_base("generate").generate(
        prompts,
        apply_chat_template,
        mode,
        micro_batch_size,
        trace_tags,
        max_generation_steps,
    )

  # --- Training (train_step) ------------------------------------------------

  def update_actor(
      self, train_ds: Any, eval_ds: Any, skip_jit: bool = False
  ) -> Any:
    if self._trainer_worker is not None:
      fwd_bwd = getattr(self._trainer_worker, "fwd_bwd", None)
      update = getattr(self._trainer_worker, "update", None)
      if callable(fwd_bwd) and callable(update):
        for chunk in train_ds:
          fwd_bwd(chunk)
        train_step = update(skip_jit=skip_jit)
        if eval_ds is not None:
          self.eval_actor(eval_ds)
        return train_step
      else:
        train = getattr(self._trainer_worker, "train", None)
        if not callable(train):
          raise TypeError(
              "trainer_worker must expose fwd_bwd/update or train(...)."
          )
        return train(train_ds, eval_ds=eval_ds, skip_jit=skip_jit)
    else:
      return self._require_base("update_actor").update_actor(
          train_ds, eval_ds, skip_jit
      )

  def eval_actor(self, eval_ds: Any) -> Any:
    """Runs an explicit actor evaluation phase on the trainer worker."""
    if eval_ds is None:
      return None
    if self._trainer_worker is not None:
      run_eval = getattr(self._trainer_worker, "run_eval", None)
      if callable(run_eval):
        return run_eval(eval_ds)
      eval_step = getattr(self._trainer_worker, "eval_step", None)
      if not callable(eval_step):
        raise TypeError(
            "trainer_worker must expose run_eval(...) or eval_step(...)."
        )
      for chunk in eval_ds:
        eval_step(chunk)
      return None

    base = self._require_base("eval_actor")
    actor_trainer = getattr(base, "actor_trainer", None)
    run_eval = getattr(actor_trainer, "_run_eval", None)
    if callable(run_eval):
      return run_eval(eval_ds)
    eval_step = getattr(actor_trainer, "eval_step", None)
    if not callable(eval_step):
      raise TypeError(
          "base actor_trainer must expose _run_eval(...) or eval_step(...)."
      )
    for chunk in eval_ds:
      eval_step(chunk)
    return None

  def update_critic(
      self, train_ds: Any, eval_ds: Any, skip_jit: bool = False
  ) -> None:
    if self._trainer_worker is not None and hasattr(
        self._trainer_worker, "train_critic"
    ):
      self._trainer_worker.train_critic(train_ds, eval_ds, skip_jit)
    else:
      self._require_base("update_critic").update_critic(
          train_ds, eval_ds, skip_jit
      )

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
      rollout_config = self.get_rollout_config(rl_engine_lib.Mode.TRAIN)
      return self._inference_worker.per_token_logps(
          prompt_ids=prompt_tokens,
          completion_ids=completion_tokens,
          pad_id=pad_id,
          eos_id=eos_id,
          temperature=getattr(rollout_config, "temperature", 1.0),
      )
    return self._require_base("get_ref_per_token_logps").get_ref_per_token_logps(
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
    if callable(per_token_logps):
      return per_token_logps(
          prompt_ids=prompt_tokens,
          completion_ids=completion_tokens,
          pad_id=pad_id,
          eos_id=eos_id,
      )
    return self._require_base(
        "get_actor_per_token_logps"
    ).get_actor_per_token_logps(
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
      self._require_base("sync_weights").sync_weights()

  def configure_grpo_loss(self, **kwargs) -> Any:
    configure = getattr(self._trainer_worker, "configure_grpo_loss", None)
    if callable(configure):
      return configure(**kwargs)
    raise RuntimeError(
        "configure_grpo_loss requires a trainer worker/proxy exposing "
        "configure_grpo_loss(...)."
    )

  def buffer_metrics(self, metrics: Any, **kwargs) -> None:
    if self._base is not None:
      self._base.buffer_metrics(metrics, **kwargs)
      return
    self._buffered_metrics.append((metrics, kwargs))

  def buffer_metrics_async(self, metrics: Any, **kwargs) -> None:
    if self._base is not None:
      self._base.buffer_metrics_async(metrics, **kwargs)
      return
    self._buffered_async_metrics.append((metrics, kwargs))

  def close(self) -> None:
    if self._base is not None:
      self._base.close()

  @property
  def cluster_config(self) -> Any:
    if self._cluster_config is not None:
      return self._cluster_config
    return self._require_base("cluster_config").cluster_config

  @property
  def rollout(self) -> Any:
    if self._rollout is not None:
      return self._rollout
    return self._require_base("rollout").rollout

  @property
  def actor_trainer(self) -> Any:
    if self._actor_trainer is not None:
      return self._actor_trainer
    return self._require_base("actor_trainer").actor_trainer

  @property
  def tokenizer(self) -> Any:
    if self._tokenizer is not None:
      return self._tokenizer
    return self._require_base("tokenizer").tokenizer

  @property
  def r2m(self) -> Any:
    if self._r2m is not None:
      return self._r2m
    return self._require_base("r2m").r2m

  @property
  def perf_v2(self) -> Any:
    return self._require_base("perf_v2").perf_v2

  def get_rollout_config(self, mode: Any) -> Any:
    if self._base is not None:
      return self._base.get_rollout_config(mode)
    rollout_config = self.cluster_config.rollout_config
    if isinstance(rollout_config, dict):
      return rollout_config[mode]
    return rollout_config

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


# Backward-compatibility alias.
OrchestratorRLCluster = OrchestratorRLEngine
