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

"""A trainer handle backed by a step-level trainer instead of a cluster.

The worker-backed cluster routes training to whatever trainer handle it is
given. `InProcessTrainerWorker` forwards to the surrounding `RLCluster`, which
owns a model and derives its own update boundaries. This handle instead drives
an `AbstractTrainer` directly: it converts each train example into a trainer
payload, runs one forward/backward per micro-batch, and applies the optimizer
every `grad_accumulation_steps` -- the same boundary rule the agentic learner
uses for unpacked batches.

That makes the training substrate swappable the way the rest of the stack is:
any object implementing the step-level trainer API can serve the orchestrated
loop, including one that owns no model of the cluster's kind. It is also the
shape a genuinely remote trainer worker needs, since the step API -- not an
`RLCluster` -- is what crosses that boundary.
"""

from typing import Any, Optional

import numpy as np

from tunix.experimental.common import datatypes
from tunix.experimental.worker import abstract_worker

WorkerState = datatypes.WorkerState


def to_trainer_payload(example: Any) -> datatypes.RLTrainerPayload:
  """Flattens an RL train example into the trainer's payload layout.

  Token ids are the left-padded prompt followed by the right-padded
  completion; the loss mask covers completion positions only, since prompt
  tokens are conditioning rather than predictions.
  """
  prompt_ids = np.asarray(example.prompt_ids)
  completion_ids = np.asarray(example.completion_ids)
  prompt_mask = np.asarray(example.prompt_mask)
  completion_mask = np.asarray(example.completion_mask)

  return datatypes.RLTrainerPayload(
      token_ids=np.concatenate([prompt_ids, completion_ids], axis=-1),
      token_mask=np.concatenate(
          [prompt_mask.astype(np.int32), completion_mask.astype(np.int32)],
          axis=-1,
      ),
      loss_mask=np.concatenate(
          [
              np.zeros_like(prompt_mask, dtype=np.int32),
              completion_mask.astype(np.int32),
          ],
          axis=-1,
      ),
      advantages=np.asarray(example.advantages),
      ref_per_token_logps=_optional(example.ref_per_token_logps),
      old_per_token_logps=_optional(example.old_per_token_logps),
      sampler_is_weights=_optional(
          getattr(example, "sampler_is_weights", None)
      ),
  )


def _optional(value: Any) -> Optional[np.ndarray]:
  return None if value is None else np.asarray(value)


class AbstractTrainerHandle(abstract_worker.Worker):
  """Trainer handle driving a step-level trainer.

  Handle contract:
      train(chunks, eval_ds, skip_jit) -> None
  """

  def __init__(
      self,
      trainer: Any,
      *,
      grad_accumulation_steps: int = 1,
      worker_id: str = "trainer",
      payload_fn: Any = to_trainer_payload,
  ):
    """Initializes the handle.

    Args:
      trainer: The step-level trainer to drive.
      grad_accumulation_steps: Micro-batches per optimizer update.
      worker_id: Identifier reported to the control plane.
      payload_fn: Converts one train example into a trainer payload.

    Raises:
      ValueError: If `grad_accumulation_steps` is not positive.
    """
    if grad_accumulation_steps < 1:
      raise ValueError(
          "grad_accumulation_steps must be >= 1, got"
          f" {grad_accumulation_steps}."
      )
    self._trainer = trainer
    self._grad_accumulation_steps = grad_accumulation_steps
    self._worker_id = worker_id
    self._payload_fn = payload_fn
    self._micro_steps = 0
    self._updates = 0

  @property
  def trainer(self) -> Any:
    return self._trainer

  @property
  def updates_applied(self) -> int:
    return self._updates

  # --- Handle contract ------------------------------------------------------

  def train(self, chunks: Any, eval_ds: Any = None, skip_jit: bool = False):
    """Runs forward/backward over each micro-batch, updating on the boundary."""
    del eval_ds, skip_jit  # Evaluation cadence is the caller's concern.
    for example in _as_iterable(chunks):
      self._trainer.fwd_bwd(self._payload_fn(example))
      self._micro_steps += 1
      if self._micro_steps % self._grad_accumulation_steps == 0:
        self._trainer.update()
        self._updates += 1

  def per_token_logps(
      self, prompt_ids: Any, completion_ids: Any, pad_id: int, eos_id: int
  ) -> Any:
    """Not served here: the step API has no scoring verb."""
    raise NotImplementedError(
        "This trainer handle drives the step-level training API, which has no"
        " scoring entry point; route actor scoring to a worker that does."
    )

  # --- Control plane --------------------------------------------------------

  def initialize(self) -> datatypes.Response:
    self.state = WorkerState.INITIALIZING
    try:
      return datatypes.Response()
    finally:
      self.state = WorkerState.READY

  def compile(self, dummy_data: Any = None) -> datatypes.Response:
    self.state = WorkerState.COMPILING
    try:
      self._trainer.compile(dummy_data)
      return datatypes.Response()
    finally:
      self.state = WorkerState.READY

  def start(self) -> datatypes.Response:
    self.state = WorkerState.READY
    return datatypes.Response()

  def stop(self) -> datatypes.Response:
    self.state = WorkerState.STOPPED
    self._trainer.close()
    return datatypes.Response()

  def info(self) -> datatypes.WorkerInfo:
    return datatypes.WorkerInfo(
        worker_id=self._worker_id, roles=frozenset({"trainer"})
    )

  def heartbeat(self) -> datatypes.HealthReport:
    return datatypes.HealthReport(state=self.state)


def _as_iterable(chunks: Any) -> Any:
  """Accepts a single example or a sequence of them."""
  if hasattr(chunks, "completion_ids"):
    return [chunks]
  return chunks
