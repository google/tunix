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

"""Orchestrator-side client proxies for routing calls to remote gRPC workers."""

from __future__ import annotations

import logging
from typing import Any
from tunix.rl import rl_cluster as rl_engine_lib
from tunix.experimental.worker import remote_execution


class RemoteActorTrainerProxy:
  """Proxies configurations like loss functions to a remote TrainerWorker."""

  def __init__(self, handle: remote_execution.ActorHandle):
    self.handle = handle

  def with_loss_fn(self, loss_fn: Any, has_aux: bool = False) -> RemoteActorTrainerProxy:
    self.handle.submit("with_loss_fn", loss_fn, has_aux)
    return self

  def with_gen_model_input_fn(self, gen_model_input_fn: Any) -> RemoteActorTrainerProxy:
    self.handle.submit("with_gen_model_input_fn", gen_model_input_fn)
    return self


class GrpcRolloutWorkerProxy:
  """Proxies generate() and updates configs from local rollout config."""

  def __init__(self, handle: remote_execution.ActorHandle, cluster_config: Any):
    self.handle = handle
    self.cluster_config = cluster_config

  def generate(
      self,
      prompts: Any,
      apply_chat_template: bool = False,
      mode: Any = rl_engine_lib.Mode.TRAIN,
      micro_batch_size: int | None = None,
      trace_tags: Any = None,
      max_generation_steps: int | None = None,
      **kwargs,
  ) -> Any:
    del apply_chat_template, micro_batch_size, trace_tags
    # Resolve configs from local base config
    rollout_cfg = self.cluster_config.rollout_config
    if isinstance(rollout_cfg, dict):
      rollout_cfg = rollout_cfg[mode]

    # Convert to wire format
    return self.handle.submit(
        "generate",
        prompts,
        max_generation_steps=(
            max_generation_steps
            if max_generation_steps is not None
            else rollout_cfg.max_tokens_to_generate
        ),
        temperature=rollout_cfg.temperature,
        top_p=rollout_cfg.top_p,
        top_k=rollout_cfg.top_k,
        **kwargs,
    )

class GrpcWeightSyncProxy:
  """Orchestrates weight sync from Trainer to Rollout worker over gRPC."""

  def __init__(
      self,
      trainer_handle: remote_execution.ActorHandle,
      rollout_handle: remote_execution.ActorHandle,
      *,
      sync_lora_weights: bool = True,
  ):
    self.trainer_handle = trainer_handle
    self.rollout_handle = rollout_handle
    self.sync_lora_weights = sync_lora_weights

  def sync(self) -> None:
    if self.sync_lora_weights:
      logging.info("[WeightSyncProxy] Pulling LoRA weights from Trainer...")
      self.trainer_handle.submit("prepare_weight_sync")
      lora_state = self.trainer_handle.submit("get_lora_weights")
      logging.info("[WeightSyncProxy] Pushing LoRA weights to Rollout worker...")
      self.rollout_handle.submit("pre_weight_sync")
      self.rollout_handle.submit("weight_sync", lora_state)
      self.rollout_handle.submit("post_weight_sync")
    else:
      logging.info("[WeightSyncProxy] Running rollout sync barrier.")
      self.rollout_handle.submit("pre_weight_sync")
      self.rollout_handle.submit("weight_sync")
      self.rollout_handle.submit("post_weight_sync")
    logging.info("[WeightSyncProxy] Sync completed.")


class GrpcTrainerWorkerProxy:
  """Proxies trainer calls to a remote experimental TrainerWorker."""

  def __init__(self, handle: remote_execution.ActorHandle):
    self.handle = handle

  def configure_grpo_loss(self, **kwargs) -> Any:
    return self.handle.submit("configure_grpo_loss", **kwargs)

  def fwd_bwd(self, chunk: Any) -> Any:
    return self.handle.submit("fwd_bwd", chunk)

  def update(self, eval_ds: Any = None, skip_jit: bool = False) -> int:
    return self.handle.submit("update", eval_ds=eval_ds, skip_jit=skip_jit)

  def train(self, chunks: list[Any], eval_ds: Any = None, skip_jit: bool = False) -> None:
    for chunk in chunks:
      self.fwd_bwd(chunk)
    self.update(eval_ds=eval_ds, skip_jit=skip_jit)

  def per_token_logps(
      self,
      prompt_ids: Any,
      completion_ids: Any,
      pad_id: int,
      eos_id: int,
  ) -> Any:
    return self.handle.submit(
        "per_token_logps",
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        pad_id=pad_id,
        eos_id=eos_id,
    )


class GrpcInferenceWorkerProxy:
  """Proxies reference logprob calls to a remote GRPO trainer worker."""

  def __init__(self, handle: remote_execution.ActorHandle):
    self.handle = handle

  def per_token_logps(
      self,
      prompt_ids: Any,
      completion_ids: Any,
      pad_id: int,
      eos_id: int,
  ) -> Any:
    # Routes to the reference model logprob calculation on Trainer worker
    return self.handle.submit(
        "reference_logps",
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        pad_id=pad_id,
        eos_id=eos_id,
    )
