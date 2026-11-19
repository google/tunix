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

from collections.abc import Callable, Mapping, Sequence
import dataclasses

import numpy as np
from tunix.dist_rl.rollout import sampler as sampler_lib
from tunix.dist_rl.swarm import trace


@dataclasses.dataclass(frozen=True, kw_only=True)
class PolicyBinding:
  """Binds a logical role (or '*' fallback) to a trainable or frozen policy."""

  policy_id: str
  policy_version: int
  sampler: sampler_lib.Sampler
  encode: Callable[[str], Sequence[int]]
  sampling_params: sampler_lib.SamplingParams
  max_prompt_length: int
  trainable: bool = True

  def __post_init__(self):
    if not self.policy_id or self.policy_version < 0:
      raise ValueError("PolicyBinding needs a policy_id and version >= 0.")
    if self.max_prompt_length <= 0:
      raise ValueError("max_prompt_length must be positive.")
    if self.trainable:
      object.__setattr__(
          self,
          "sampling_params",
          dataclasses.replace(self.sampling_params, return_logprobs=True),
      )


class RecordingRuntime:
  """Role-routed policy runtime over one or more bound policies in an episode.

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
    default_binding = PolicyBinding(
        policy_id=policy_id,
        policy_version=policy_version,
        sampler=sampler,
        encode=encode,
        sampling_params=sampling_params,
        max_prompt_length=max_prompt_length,
        trainable=True,
    )
    self.sampler = sampler
    self.encode = encode
    self.params = default_binding.sampling_params
    self.policy_id = policy_id
    self.policy_version = policy_version
    self.episode_id = episode_id
    self.max_prompt_length = max_prompt_length
    self._bindings: dict[str, PolicyBinding] = {"*": default_binding}
    self._calls = 0

  @classmethod
  def from_bindings(
      cls,
      bindings: Mapping[str, PolicyBinding],
      *,
      episode_id: str,
  ) -> "RecordingRuntime":
    """Constructs a runtime routing each role to its configured PolicyBinding."""
    if not episode_id or not bindings:
      raise ValueError("from_bindings requires episode_id and nonempty bindings.")
    first = next(iter(bindings.values()))
    instance = cls(
        first.sampler,
        first.encode,
        first.sampling_params,
        policy_id=first.policy_id,
        policy_version=first.policy_version,
        episode_id=episode_id,
        max_prompt_length=first.max_prompt_length,
    )
    instance._bindings = dict(bindings)
    return instance

  def _resolve_binding(self, role: str) -> PolicyBinding:
    binding = self._bindings.get(role) or self._bindings.get("*")
    if binding is None:
      raise KeyError(f"Unbound swarm role: {role!r}.")
    return binding

  async def generate(self, role: str, prompt: str) -> trace.CallTrace:
    binding = self._resolve_binding(role)
    ids = np.asarray(binding.encode(prompt), dtype=np.int32)
    if ids.ndim != 1 or not 0 < len(ids) <= binding.max_prompt_length:
      raise ValueError("Encoded prompt is empty or exceeds the context budget.")
    call_index = self._calls
    self._calls += 1
    call_id = f"{self.episode_id}/{call_index}"
    # Distinct deterministic request seeds; reserve index before await so
    # concurrent same-layer worker calls never collide.
    params = dataclasses.replace(
        binding.sampling_params,
        seed=None
        if binding.sampling_params.seed is None
        else binding.sampling_params.seed + call_index,
    )
    response = await binding.sampler.sample(
        sampler_lib.SamplingRequest(
            request_id=call_id,
            prompt=ids,
            sampling_params=params,
            metadata={
                "role": role,
                "policy_id": binding.policy_id,
                "policy_version": binding.policy_version,
                "trainable": binding.trainable,
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
    if len(response.token_ids) > params.max_tokens or (
        binding.trainable and response.logprobs is None
    ):
      raise ValueError(
          "Sampler must return unpadded completion IDs and logprobs."
      )
    n_out = len(response.token_ids)
    logprobs = (
        () if response.logprobs is None else tuple(response.logprobs.tolist())
    )
    return trace.CallTrace(
        call_id=call_id,
        role=role,
        policy_id=binding.policy_id,
        policy_version=binding.policy_version,
        prompt_ids=tuple(ids.tolist()),
        completion_ids=tuple(response.token_ids.tolist()),
        rollout_logprobs=logprobs,
        loss_mask=(1,) * n_out if binding.trainable else (0,) * n_out,
        text=response.text,
        trainable=binding.trainable,
    )
