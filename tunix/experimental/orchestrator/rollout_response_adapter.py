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

"""Turns a worker's `RolloutResponse` into the shape the postprocess consumes.

`RolloutResponse` is the wire type: numpy arrays and primitives only. The
agentic postprocess (`_process_results` / `GRPOAdapter.postprocess_group`) reads
trajectory-shaped items -- `item.traj[...]` -- because that is what the
in-process agentic path produces. This module is the single seam between the
two, so that:

  * `Trajectory` stays worker-internal and never crosses a process boundary;
  * `RolloutResponse` is the only wire type;
  * one postprocess serves both producers (local trajectories and remote
    responses), so nothing downstream is duplicated.

Two fields deliberately do not travel on the response:

  * `original_input` -- the dataset row. The orchestrator issued the request, so
    it already has it; sending it back would be redundant and would couple the
    worker to the dataset schema. It is supplied here from the request side.
  * the completion *text* -- `RolloutResponse.from_trajectory` drops strings to
    keep the wire numpy-only. Reward functions that score text need it, so it is
    reconstructed by detokenizing the assistant spans with the orchestrator's own
    tokenizer (a worker may instead pass it through `metadata["completion_text"]`,
    which wins when present).

Keeping the conversion here is also what makes a metadata-only future cheap: a
response that carries no `segments` (payload parked in a store, referenced from
`metadata`) only needs this one function taught how to fetch it.
"""

import dataclasses
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from tunix.experimental.common import datatypes

ASSISTANT_SOURCE = "assistant"


@dataclasses.dataclass
class TrajectoryItem:
  """A postprocess-shaped view of one completed rollout.

  Mirrors the `.traj` dict the agentic path yields, so the same postprocess can
  consume rollouts produced locally or by a remote worker.

  Attributes:
    traj: The trajectory fields the postprocess reads.
    group_id: Which group this rollout belongs to. Exposed as an attribute
      because the reused consumer reads it that way, not out of the dict. The
      consumer derives a step number from it arithmetically, so a pooled path
      feeding that consumer must issue numeric group ids.
  """

  traj: dict[str, Any]
  group_id: Any = None


def _concat(arrays: Sequence[Any], dtype: Any) -> np.ndarray:
  if not arrays:
    return np.zeros(0, dtype=dtype)
  return np.concatenate([np.asarray(a, dtype=dtype) for a in arrays], axis=0)


def completion_tokens(response: datatypes.RolloutResponse) -> np.ndarray:
  """All generated tokens (assistant and env spans) in emission order."""
  return _concat([seg.tokens for seg in response.segments], np.int32)


def completion_masks(response: datatypes.RolloutResponse) -> np.ndarray:
  """Loss mask over `completion_tokens`; 1 only where the model emitted."""
  return _concat([seg.loss_mask for seg in response.segments], np.int32)


def completion_logps(
    response: datatypes.RolloutResponse,
) -> Optional[np.ndarray]:
  """Sampler logprobs over `completion_tokens`, zero-filled on env spans.

  Returns None when no span carried logprobs, which is how the postprocess
  distinguishes "the sampler did not report logprobs" from "they were all zero".
  """
  if not any(seg.logps is not None for seg in response.segments):
    return None
  spans = []
  for seg in response.segments:
    if seg.logps is None:
      spans.append(np.zeros(np.asarray(seg.tokens).shape[0], dtype=np.float32))
    else:
      spans.append(np.asarray(seg.logps, dtype=np.float32))
  return _concat(spans, np.float32)


def assistant_text(
    response: datatypes.RolloutResponse, tokenizer: Any = None
) -> str:
  """The model-emitted text, for reward functions that score strings.

  Prefers `metadata["completion_text"]` when a worker chose to send it;
  otherwise detokenizes the assistant spans locally.
  """
  from_metadata = (response.metadata or {}).get("completion_text")
  if isinstance(from_metadata, str):
    return from_metadata
  if tokenizer is None:
    return ""
  tokens = _concat(
      [seg.tokens for seg in response.segments if seg.source == ASSISTANT_SOURCE],
      np.int32,
  )
  if tokens.size == 0:
    return ""
  return tokenizer.decode(tokens.tolist())


def to_trajectory_item(
    response: datatypes.RolloutResponse,
    request: Optional[datatypes.RolloutRequest] = None,
    *,
    tokenizer: Any = None,
    original_input: Optional[Mapping[str, Any]] = None,
) -> TrajectoryItem:
  """Converts one worker response into a postprocess-ready item.

  Args:
    response: The worker's wire-safe rollout result.
    request: The request that produced it, supplying the identifiers the
      response does not echo (`prompt_id`, `group_id`) and, by default, the
      dataset row.
    tokenizer: Used to reconstruct the completion text when the worker did not
      send it. Optional; without it text-scoring rewards see an empty string.
    original_input: The dataset row. Defaults to `request.prompt` when it is a
      mapping, else `{"prompts": request.prompt}`.

  Returns:
    A `TrajectoryItem` whose `.traj` carries the keys the postprocess reads.
  """
  if original_input is None:
    original_input = _original_input_from(request)

  text = assistant_text(response, tokenizer)
  traj: dict[str, Any] = {
      # Token-level payload.
      "prompt_tokens": np.asarray(response.prompt_tokens, dtype=np.int32),
      "conversation_tokens": completion_tokens(response),
      "conversation_masks": completion_masks(response),
      "old_logprobs": completion_logps(response),
      # Scalars the postprocess needs.
      "policy_version": response.policy_version,
      "trajectory_reward": response.env_reward,
      "status": response.status,
      # Text view for reward functions.
      "conversation_text": [{"role": "assistant", "content": text}],
      # Provenance the orchestrator owns, not the worker.
      "original_input": dict(original_input),
      "request_id": response.request_id,
  }
  group_id = None
  if request is not None:
    traj["prompt_id"] = request.prompt_id
    traj["group_id"] = request.group_id
    group_id = request.group_id
  return TrajectoryItem(traj=traj, group_id=group_id)


def _original_input_from(
    request: Optional[datatypes.RolloutRequest],
) -> Mapping[str, Any]:
  if request is None:
    return {"prompts": ""}
  if isinstance(request.prompt, Mapping):
    return request.prompt
  return {"prompts": request.prompt}


def to_trajectory_items(
    responses: Sequence[datatypes.RolloutResponse],
    requests: Optional[Sequence[datatypes.RolloutRequest]] = None,
    *,
    tokenizer: Any = None,
) -> list[TrajectoryItem]:
  """Converts a group of responses, pairing each with its request by id.

  Pairing is by `request_id` rather than by position, so responses may arrive in
  any completion order.
  """
  by_id = {r.request_id: r for r in (requests or [])}
  items = []
  for response in responses:
    items.append(
        to_trajectory_item(
            response,
            by_id.get(response.request_id),
            tokenizer=tokenizer,
        )
    )
  return items
