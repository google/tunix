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

"""In-process worker handles backed by an RLEngine (or any AbstractRLEngine).

These handles satisfy the same contracts a remote (RPC) worker would, but run in
the same process by delegating straight to a base ``RLEngine`` /
``AbstractRLEngine``. They let ``OrchestratorRLEngine`` route its compute
primitives to handles today (single process) and give the eventual RPC handles a
behavioral reference to match.

``RLOrchestrator`` (RL Algorithm Layer)
  └── ``OrchestratorRLEngine`` (Coordination/Routing Layer)
        └── ``InProcessRolloutWorker`` (Worker Handle)
              └── ``RLEngine`` /
              ``AbstractRLEngine`` (Base In-process Engine Layer)
"""

from typing import Any, Mapping

from tunix.rl import rl_cluster as rl_engine_lib


class InProcessTrainerWorker:
  """Trainer-worker handle that wraps an in-process base ``RLEngine`` / ``AbstractRLEngine``.

  Contract driven by ``OrchestratorRLEngine``:

      fwd_bwd(payload) -> None
      update(eval_ds=None, skip_jit=False) -> int
      per_token_logps(prompt_ids, completion_ids, pad_id, eos_id) -> array
      sync_weights() -> None
  """

  def __init__(self, rl_engine: Any):
    self._rl_engine = rl_engine
    self._pending_payloads: list[Any] = []

  def fwd_bwd(self, payload: Any) -> None:
    """Stages one actor micro-batch for the next optimizer update."""
    if isinstance(payload, list):
      self._pending_payloads.extend(payload)
    else:
      self._pending_payloads.append(payload)

  def update(self, eval_ds: Any = None, skip_jit: bool = False) -> int:
    """Runs an actor update over the staged micro-batches."""
    if not self._pending_payloads:
      return int(getattr(self._rl_engine.actor_trainer, "train_steps", 0))
    chunks = self._pending_payloads
    self._pending_payloads = []
    try:
      self._rl_engine.train(
          rl_engine_lib.Role.ACTOR, chunks, eval_ds, skip_jit
      )
    except Exception:
      self._pending_payloads = chunks + self._pending_payloads
      raise
    return int(getattr(self._rl_engine.actor_trainer, "train_steps", 0))

  def train(
      self,
      train_ds: Any,
      eval_ds: Any,
      skip_jit: bool = False,
  ) -> None:
    """Compatibility shim for callers that still submit a full train_ds."""
    self.fwd_bwd(train_ds)
    self.update(eval_ds=eval_ds, skip_jit=skip_jit)

  def train_critic(
      self,
      train_ds: Any,
      eval_ds: Any,
      skip_jit: bool = False,
  ) -> None:
    """Runs a training update for the Critic."""
    self._rl_engine.train(rl_engine_lib.Role.CRITIC, train_ds, eval_ds, skip_jit)

  def per_token_logps(
      self,
      prompt_ids: Any,
      completion_ids: Any,
      pad_id: int,
      eos_id: int,
      micro_batch_size: int | None = None,
      segment_ids: Any | None = None,
      **kwargs: Any,
  ) -> Any:
    """Actor-model per-token logprobs over a padded group."""
    call_kwargs = dict(
        prompt_tokens=prompt_ids,
        completion_tokens=completion_ids,
        pad_id=pad_id,
        eos_id=eos_id,
        micro_batch_size=(
            micro_batch_size
            if micro_batch_size is not None
            else (
                self._rl_engine.cluster_config.training_config.compute_logps_micro_batch_size
            )
        ),
        **kwargs,
    )
    if segment_ids is not None:
      call_kwargs["segment_ids"] = segment_ids
    return self._rl_engine.per_token_logps(
        rl_engine_lib.Role.ACTOR, **call_kwargs
    )

  def sync_weights(self) -> None:
    """Synchronizes trainer weights to rollout/inference replicas."""
    self._rl_engine.sync_weights()


class InProcessRolloutWorker:
  """Rollout-worker handle that wraps an in-process base ``RLEngine`` / ``AbstractRLEngine``.

  Contract driven by ``OrchestratorRLEngine``:

      generate(prompts, ...) -> RolloutOutput
      sync_weights() -> None
  """

  def __init__(self, rl_engine: Any):
    self._rl_engine = rl_engine

  def generate(
      self,
      prompts: list[str] | list[list[dict[str, str]]],
      apply_chat_template: bool = False,
      mode: Any = None,
      micro_batch_size: int | None = None,
      trace_tags: Mapping[str, Any] | None = None,
      max_generation_steps: int | None = None,
  ) -> Any:
    """Generates completions for prompts by delegating to base engine."""
    return self._rl_engine.generate(
        prompts=prompts,
        apply_chat_template=apply_chat_template,
        mode=mode,
        micro_batch_size=micro_batch_size,
        trace_tags=trace_tags,
        max_generation_steps=max_generation_steps,
    )

  def sync_weights(self) -> None:
    """Synchronizes rollout weights from trainer."""
    self._rl_engine.sync_weights()


class InProcessInferenceWorker:
  """Inference-worker handle that wraps an in-process base ``RLEngine`` / ``AbstractRLEngine``.

  Contract driven by ``OrchestratorRLEngine``:

      per_token_logps(prompt_ids, completion_ids, pad_id, eos_id) -> array
  """

  def __init__(self, rl_engine: Any):
    self._rl_engine = rl_engine

  def per_token_logps(
      self,
      prompt_ids: Any,
      completion_ids: Any,
      pad_id: int,
      eos_id: int,
      micro_batch_size: int | None = None,
      segment_ids: Any | None = None,
      **kwargs: Any,
  ) -> Any:
    """Reference-model per-token logprobs over a padded group."""
    call_kwargs = dict(
        prompt_tokens=prompt_ids,
        completion_tokens=completion_ids,
        pad_id=pad_id,
        eos_id=eos_id,
        micro_batch_size=(
            micro_batch_size
            if micro_batch_size is not None
            else (
                self._rl_engine.cluster_config.training_config.compute_logps_micro_batch_size
            )
        ),
        **kwargs,
    )
    if segment_ids is not None:
      call_kwargs["segment_ids"] = segment_ids
    return self._rl_engine.per_token_logps(
        rl_engine_lib.Role.REFERENCE, **call_kwargs
    )
