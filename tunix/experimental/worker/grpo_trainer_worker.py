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

"""GRPO-specific trainer worker extensions."""

from __future__ import annotations

from typing import Any

from tunix.experimental.common import datatypes
from tunix.experimental.worker import trainer_worker


class GRPOTrainerWorker(trainer_worker.TrainerWorker):
  """TrainerWorker with GRPO/RL methods required by the demo orchestrator."""

  def info(self) -> datatypes.WorkerInfo:
    info = super().info()
    return datatypes.WorkerInfo(
        worker_id=info.worker_id,
        roles=info.roles
        | frozenset({"grpo", "actor_logps", "reference_logps"}),
        resources=info.resources,
    )

  def configure_grpo_loss(self, **kwargs) -> datatypes.Response:
    self._ensure_ready()  # pylint: disable=protected-access
    metadata = self._trainer.configure_grpo_loss(  # pylint: disable=protected-access
        **kwargs
    )
    return self._response(**metadata)  # pylint: disable=protected-access

  def per_token_logps(
      self,
      prompt_ids: Any,
      completion_ids: Any,
      pad_id: int,
      eos_id: int,
  ) -> Any:
    self._ensure_ready()  # pylint: disable=protected-access
    return self._trainer.per_token_logps(  # pylint: disable=protected-access
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        pad_id=pad_id,
        eos_id=eos_id,
    )

  def actor_logps(
      self,
      prompt_ids: Any,
      completion_ids: Any,
      pad_id: int,
      eos_id: int,
  ) -> Any:
    return self.per_token_logps(prompt_ids, completion_ids, pad_id, eos_id)

  def reference_logps(
      self,
      prompt_ids: Any,
      completion_ids: Any,
      pad_id: int,
      eos_id: int,
  ) -> Any:
    self._ensure_ready()  # pylint: disable=protected-access
    return self._trainer.reference_logps(  # pylint: disable=protected-access
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        pad_id=pad_id,
        eos_id=eos_id,
    )

  def get_lora_weights(self) -> Any:
    self._ensure_ready()  # pylint: disable=protected-access
    return self._trainer.get_lora_weights()  # pylint: disable=protected-access
