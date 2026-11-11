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

"""Topology-independent recording over the existing dist_rl sampler API."""

from collections.abc import Callable, Sequence
import dataclasses

import numpy as np
from tunix.dist_rl.rollout import sampler as sampler_lib
from tunix.dist_rl.swarm import trace


class RecordingRuntime:
  """One policy snapshot shared by all logical roles in an episode.

  The workflow owns role prompts and control flow. The runtime owns sampling
  and exact token provenance. No ground-truth answers enter this interface.
  """

  def __init__(
      self,
      sampler: sampler_lib.Sampler,
      encode: Callable[[str], Sequence[int]],
      sampling_params: sampler_lib.SamplingParams,
      *,
      policy_id: str,
      policy_version: int,
      episode_id: str,
      max_prompt_length: int,
  ):
    if max_prompt_length <= 0:
      raise ValueError("max_prompt_length must be positive.")
    self.sampler = sampler
    self.encode = encode
    self.params = dataclasses.replace(sampling_params, return_logprobs=True)
    self.policy_id = policy_id
    self.policy_version = policy_version
    self.episode_id = episode_id
    self.max_prompt_length = max_prompt_length
    self._calls = 0

  async def generate(self, role: str, prompt: str) -> trace.CallTrace:
    ids = np.asarray(self.encode(prompt), dtype=np.int32)
    if ids.ndim != 1 or not 0 < len(ids) <= self.max_prompt_length:
      raise ValueError("Encoded prompt is empty or exceeds the context budget.")
    call_id = f"{self.episode_id}/{self._calls}"
    # Distinct deterministic request seeds; never reset all calls to one draw.
    params = dataclasses.replace(
        self.params,
        seed=None
        if self.params.seed is None
        else self.params.seed + self._calls,
    )
    response = await self.sampler.sample(
        sampler_lib.SamplingRequest(
            request_id=call_id,
            prompt=ids,
            sampling_params=params,
            metadata={
                "role": role,
                "policy_id": self.policy_id,
                "policy_version": self.policy_version,
            },
        )
    )
    if not isinstance(response, sampler_lib.SamplingResponse):
      raise TypeError("A single call must return one SamplingResponse.")
    if response.error is not None or response.finish_reason not in (
        "stop",
        "length",
    ):
      raise RuntimeError(
          f"Sampling failed for {call_id}: {response.finish_reason}"
      )
    if response.request_id != call_id or not np.array_equal(
        response.prompt_token_ids, ids
    ):
      raise ValueError("Sampler changed the request identity or prompt tokens.")
    if response.logprobs is None or len(response.token_ids) > params.max_tokens:
      raise ValueError(
          "Sampler must return unpadded completion IDs and logprobs."
      )
    self._calls += 1
    return trace.CallTrace(
        call_id=call_id,
        role=role,
        policy_id=self.policy_id,
        policy_version=self.policy_version,
        prompt_ids=tuple(ids.tolist()),
        completion_ids=tuple(response.token_ids.tolist()),
        rollout_logprobs=tuple(response.logprobs.tolist()),
        loss_mask=(1,) * len(response.token_ids),
        text=response.text,
    )
