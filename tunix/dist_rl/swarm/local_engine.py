# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""In-process transport adapter around the existing TrainerWorker.

Only transport and weight publication are local. Forward/backward, gradient
accumulation, token normalization and optimizer updates use dist_rl unchanged.
"""

from collections.abc import Callable

import numpy as np
from tunix.dist_rl.common import datatypes
from tunix.dist_rl.worker import trainer_worker
from tunix.rl import algo_core
from tunix.rl import function_registry


class LocalSwarmEngine:

  def __init__(
      self, worker: trainer_worker.TrainerWorker, sync: Callable[[], None]
  ):
    self.worker = worker
    self.sync = sync
    self.version = 0
    self.config = None

  def configure(self, algo_config, config):
    self.config = config
    self.worker.with_loss_fn(
        function_registry.get_policy_loss_fn(algo_config.policy_loss_fn),
        has_aux=True,
    )
    self.worker.with_gen_model_input_fn(
        algo_core.build_gen_model_input_fn(
            algo_config, pad_id=config.pad_id, eos_id=config.eos_id
        )
    )

  async def prepare(self):
    self.sync()
    return self.version

  async def score(self, payload):
    response = self.worker.per_token_logps(
        datatypes.LogprobsRequest(
            prompt_tokens=payload.prompt_ids,
            completion_tokens=payload.completion_ids,
            token_mask=np.concatenate(
                [payload.prompt_mask, payload.completion_attention_mask], axis=1
            ),
            pad_id=self.config.pad_id,
            eos_id=self.config.eos_id,
            temperature=1.0,
        )
    )
    return response.per_token_logps

  async def train(self, payload, *, apply_optimizer):
    self.worker.fwd_bwd(datatypes.TrainRequest(payload=payload))
    return self.worker.update() if apply_optimizer else None

  async def publish(self):
    self.sync()
    self.version += 1
    return self.version
