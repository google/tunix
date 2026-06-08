# Copyright 2025 Google LLC
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

from typing import Any, Optional, Sequence, Tuple
import threading
import multiprocess as mp

import jax
import jaxtyping

from tunix.rl.rollout import base_rollout

class RolloutGenerateRequest:
    """Request from rl_proc to rollout_proc to perform inference."""
    # prompts: Sequence[str]
    # response_queue: Any  # for RolloutOutput
    # rollout_config: base_rollout.RolloutConfig | None = None
    def __init__(self, prompts: Sequence[str], response_queue: Any):
      super().__init__()
      self.prompts = prompts
      self.response_queue = response_queue

class RolloutUpdateParamsRequest:
    """Request from rl_proc to rollout_proc to update model weights."""
    # sender_info: Any  # sender_info from experimental_remote_copy_prepare
    # shardings: Any    # List of shardings or serialized sharding info
    def __init__(self, sender_info: Any, shardings: Any):
      super().__init__()
      self.sender_info = sender_info
      self.shardings = shardings

class RolloutPadIdRequest:
    def __init__(self, response_queue: Any):
      super().__init__()
      self.response_queue = response_queue

class RolloutEosIdRequest:
    def __init__(self, response_queue: Any):
      super().__init__()
      self.response_queue = response_queue

class RolloutEngineClient:
  def __init__(self,
               worker_request_channels: list[Any]):
    # must be pickle-able
    self.worker_request_channels = worker_request_channels

    self.num_workers = len(worker_request_channels)
    self.dispatch_id = 0
    self.semaphore = threading.Semaphore(self.num_workers * 800)
    self.manager = mp.Manager()
    self.cache = {}

  def generate(
      self, prompts, rollout_config: base_rollout.RolloutConfig, **kwargs
  ) -> base_rollout.RolloutOutput:
    if "trajectory_id" in kwargs:
      request_id = int(kwargs["trajectory_id"])
    else:
      request_id = self.dispatch_id
      self.dispatch_id += 1

    with self.semaphore:
      response = self.manager.Queue() # TODO: try use manager.Value()
      self.worker_request_channels[request_id % self.num_workers].put(
          RolloutGenerateRequest(
              prompts=prompts,
              response_queue=response,
              # rollout_config=rollout_config,
          )
      )
      # print(f"enqueued request {request_id}")
      output = response.get()
      return output

  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
      completion_mask: jax.Array | None = None,
      **kwargs
  ) -> jax.Array:
    raise NotImplementedError("Not implemented for RolloutEngineGroup.")

  def update_params(
      self,
      params: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ):
    print(f"ignore update_params().")
    pass

  def pad_id(self) -> int:
    if "pad_id" not in self.cache:
      response = self.manager.Queue()
      self.worker_request_channels[0].put(
          RolloutPadIdRequest(
              response_queue=response,
          )
      )
      self.cache["pad_id"] = response.get()
      print(f"pad_id() => {self.cache["pad_id"]}")
    return self.cache["pad_id"]

  def eos_id(self) -> int:
    if "eos_id" not in self.cache:
      response = self.manager.Queue()
      self.worker_request_channels[0].put(
          RolloutEosIdRequest(
              response_queue=response,
          )
      )
      self.cache["eos_id"] = response.get()
      print(f"eos_id() => {self.cache["eos_id"]}")
    return self.cache["eos_id"]

  def model(self) -> Any:
    return None
