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

"""Distributed GRPO orchestrator built by reusing the agentic learner.

Rather than reimplementing the RL control loop, the orchestrator *subclasses*
`GRPOLearner` and reuses its `train()` verbatim -- the producer/consumer loop,
grouping, chunking, gradient accumulation, mu-replay, eval cadence, and step
bookkeeping all come from the agentic path. It overrides only the seams where a
distributed run diverges, routing them to role-based worker handles instead of
the in-process `RLCluster`:

  * `_train_micro_batch` -> a TrainerWorker handle (a real handle wraps
    `peft_trainer`'s grad-accumulating `train`; a remote one is an RPC stub).
  * `_sync_weights` -> a weight-sync handle, reusing the base's fence,
    step-advance, and prompt-feed around it.

Remaining seams (`_model_call` -> rollout workers, `_maybe_run_eval`/
`_process_micro_batch`) are wired the same way in follow-up steps. Everything
else -- reward/advantage math, TrainExample assembly, metrics, checkpointing --
is inherited unchanged.

Handle contracts:
    trainer_worker.train(chunks: list, eval_ds, skip_jit: bool) -> None
    weight_sync.sync() -> None
"""

from typing import Any

from tunix.rl.agentic import agentic_grpo_learner


class AgenticGRPOOrchestrator(agentic_grpo_learner.GRPOLearner):
  """GRPOLearner whose divergent seams are routed to role-based workers."""

  def __init__(
      self, *, trainer_worker: Any, weight_sync: Any = None, **kwargs
  ):
    """Initializes the orchestrator.

    Args:
      trainer_worker: Handle exposing `train(chunks, eval_ds, skip_jit)`; the
        seam through which the reused loop drives training.
      weight_sync: Optional handle exposing `sync()`; drives the weight sync via
        workers. If None, the sync falls back to the in-process cluster (useful
        during incremental bring-up).
      **kwargs: Forwarded to `GRPOLearner.__init__` (rl_cluster, algo_config,
        reward_fns, ...). The in-process cluster still backs the not-yet-routed
        stages (metrics, step counter, generation) during incremental bring-up.
    """
    super().__init__(**kwargs)
    self._trainer_worker = trainer_worker
    self._weight_sync = weight_sync

  def _train_micro_batch(self, chunks, eval_ds, skip_jit) -> None:
    """Routes the trainer pass to the trainer worker (reused loop calls this)."""
    self._trainer_worker.train(chunks, eval_ds, skip_jit)

  def _sync_weights(self) -> None:
    """Routes the weight sync to the worker coordinator (fence/advance reused)."""
    if self._weight_sync is not None:
      self._weight_sync.sync()
    else:
      super()._sync_weights()
