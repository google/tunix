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

"""Telling a remote trainer which loss to use, without sending a function.

The loss is installed on the trainer as a Python closure built
orchestrator-side, capturing the algorithm config and a few values read off
the cluster. That works while the trainer is in the same process and does not
work at all once it is not: a closure over live objects is not something to
put on a wire, and pickling one would ship whatever it happened to capture.

So the orchestrator sends a description instead -- which registered loss, and
the values it needs -- and the trainer builds the function on its own side
from the same registry. Both processes run the same code, so naming the loss
is enough to identify it; what has to travel is the configuration, which is
plain data.

This is also why the description is worth having in one process: it is an
explicit statement of everything the loss depends on, where a closure leaves
that implicit in whatever it captured.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Optional

from tunix.rl import function_registry


@dataclasses.dataclass(frozen=True)
class LossSpec:
  """Names a loss and carries the values it needs to be rebuilt.

  Attributes:
    policy_loss_fn: Registry name of the loss to use.
    algo_config: The algorithm's configuration, passed to the loss.
    pad_id: Padding token id the loss masks on.
    eos_id: End-of-sequence token id the loss masks on.
    compute_logps_chunk_size: Chunk size for the loss's own scoring pass.
  """

  policy_loss_fn: str
  algo_config: Any
  pad_id: int
  eos_id: int
  compute_logps_chunk_size: Optional[int] = None

  def build_loss_fn(self) -> Callable[..., Any]:
    """Reconstructs the loss function on whichever side calls this."""
    policy_loss_fn = function_registry.get_policy_loss_fn(self.policy_loss_fn)
    algo_config = self.algo_config
    pad_id = self.pad_id
    eos_id = self.eos_id
    chunk_size = self.compute_logps_chunk_size

    def loss_fn(model, train_example, algo_config=algo_config):
      return policy_loss_fn(
          model,
          train_example,
          algo_config=algo_config,
          pad_id=pad_id,
          eos_id=eos_id,
          compute_logps_chunk_size=chunk_size,
      )

    return loss_fn

  def build_model_input_fn(self) -> Callable[[Any], dict[str, Any]]:
    """Reconstructs the payload adapter the loss expects."""
    algo_config = self.algo_config
    return lambda payload: {
        "train_example": payload,
        "algo_config": algo_config,
    }

  def install_on(self, trainer: Any) -> None:
    """Builds the loss here and wires it onto a local trainer.

    Args:
      trainer: The trainer to configure, in this process.
    """
    trainer.with_loss_fn(self.build_loss_fn(), has_aux=True)
    trainer.with_gen_model_input_fn(self.build_model_input_fn())
    trainer.is_managed_externally = True
