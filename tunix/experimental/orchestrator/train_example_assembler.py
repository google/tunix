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

"""Pure pad/mask/assemble core shared by the learner and the orchestrator.

`pad_samples` is the shared pad/mask primitive: it left-pads prompts and
right-pads completions to the ShapeConfig buckets and returns a numpy
`PaddedBatch`. Both callers use it, so the error-prone padding logic lives in one
place (the existing `_process_results` calls it in place of its inline loop; the
orchestrator reaches it through `trajectories_to_train_example`).

`trajectories_to_train_example` wraps `pad_samples` and attaches the group's
advantages and per-row policy versions to produce a numpy `TrainExampleV1`. It is
deliberately pure: no rl_cluster, no device arrays, no I/O. The impure stages that
surround this core in the existing learner -- reference/actor log-prob scoring,
reward/advantage computation, metrics, trajectory logging, and TIS -- stay with
their owners; their outputs (advantages now, ref-logps later) are layered on
around it.

Each caller first normalizes its own representation into `SampleTokens` (the
existing learner from its Token-dicts, the orchestrator from RolloutResult
segments).
"""

import dataclasses

import numpy as np
from tunix.experimental.common import datatypes
from tunix.rl.agentic import utils as agentic_utils


@dataclasses.dataclass(kw_only=True)
class SampleTokens:
  """One group member, normalized for assembly (unpadded, numpy/host).

  Attributes:
    prompt_tokens: Unpadded prompt token ids.
    completion_tokens: Unpadded completion (conversation) token ids.
    completion_mask: Per-token loss mask over the completion (1 = trainable).
    policy_version: Weight version used to generate this sample.
    old_logprobs: Optional per-token sampler log-probs over the completion.
  """

  prompt_tokens: np.ndarray
  completion_tokens: np.ndarray
  completion_mask: np.ndarray
  policy_version: int
  old_logprobs: np.ndarray | None = None


@dataclasses.dataclass(kw_only=True)
class PaddedBatch:
  """Padded, batched token arrays for one group (numpy/host).

  Attributes:
    prompt_ids: int32 [G, P] left-padded prompt tokens.
    prompt_mask: int32 [G, P] (1 where not pad).
    completion_ids: int32 [G, C] right-padded completion tokens.
    completion_mask: int32 [G, C] right-padded loss mask.
    old_per_token_logps: float32 [G, C] sampler log-probs, or None.
  """

  prompt_ids: np.ndarray
  prompt_mask: np.ndarray
  completion_ids: np.ndarray
  completion_mask: np.ndarray
  old_per_token_logps: np.ndarray | None = None


def pad_samples(
    samples: list[SampleTokens],
    *,
    tokenizer_info: datatypes.TokenizerInfo,
    shape_config: datatypes.ShapeConfig,
    use_rollout_logps: bool = False,
) -> PaddedBatch:
  """Left-pads prompts and right-pads completions to the ShapeConfig buckets.

  Args:
    samples: The group's members (length G), in group order.
    tokenizer_info: Supplies the pad id used for padding.
    shape_config: Supplies max_prompt_length and max_response_tokens.
    use_rollout_logps: Whether to carry the samplers' old log-probs.

  Returns:
    A PaddedBatch of numpy arrays.

  Raises:
    ValueError: If `samples` is empty.
  """
  if not samples:
    raise ValueError("cannot pad zero samples")

  pad_id = tokenizer_info.pad_id
  max_prompt_length = shape_config.max_prompt_length
  max_response_tokens = shape_config.max_response_tokens

  padded_prompt_ids = []
  padded_completion_ids = []
  padded_completion_masks = []
  padded_old_logprobs = []

  for sample in samples:
    padded_prompt, padded_completion, _ = (
        agentic_utils.pad_prompt_and_completion(
            sample.prompt_tokens,
            sample.completion_tokens,
            max_prompt_length,
            max_response_tokens,
            pad_id,
        )
    )
    padded_prompt_ids.append(padded_prompt)
    padded_completion_ids.append(padded_completion[:max_response_tokens])
    padded_completion_masks.append(
        agentic_utils.right_pad(sample.completion_mask, max_response_tokens, 0)[
            :max_response_tokens
        ]
    )
    if use_rollout_logps:
      if sample.old_logprobs is not None:
        logprobs = np.asarray(sample.old_logprobs)
        padded_old_logprobs.append(
            agentic_utils.right_pad(
                logprobs, max_response_tokens, 0.0, dtype=logprobs.dtype
            )[:max_response_tokens]
        )
      else:
        padded_old_logprobs.append(
            np.zeros(max_response_tokens, dtype=np.float32)
        )

  prompt_ids = np.asarray(padded_prompt_ids, dtype=np.int32)
  completion_ids = np.asarray(padded_completion_ids, dtype=np.int32)
  completion_mask = np.asarray(padded_completion_masks, dtype=np.int32)
  prompt_mask = (prompt_ids != pad_id).astype(np.int32)
  old_per_token_logps = (
      np.asarray(padded_old_logprobs, dtype=np.float32)
      if use_rollout_logps and padded_old_logprobs
      else None
  )
  return PaddedBatch(
      prompt_ids=prompt_ids,
      prompt_mask=prompt_mask,
      completion_ids=completion_ids,
      completion_mask=completion_mask,
      old_per_token_logps=old_per_token_logps,
  )


def trajectories_to_train_example(
    samples: list[SampleTokens],
    advantages: np.ndarray,
    *,
    tokenizer_info: datatypes.TokenizerInfo,
    shape_config: datatypes.ShapeConfig,
    use_rollout_logps: bool = False,
) -> datatypes.TrainExampleV1:
  """Assembles a group's normalized samples into a numpy TrainExampleV1.

  Pads via `pad_samples`, then attaches advantages and per-row policy versions.
  The completion mask becomes the payload's `loss_mask`. `ref_per_token_logps` is
  left None (the inference stage injects it when beta != 0); `old_per_token_logps`
  is populated from the samples only when `use_rollout_logps` is set.

  Args:
    samples: The group's members (length G), in group order.
    advantages: Per-sample advantages, shape [G] (or [G, C]).
    tokenizer_info: Supplies the pad id used for padding.
    shape_config: Supplies max_prompt_length and max_response_tokens.
    use_rollout_logps: Whether to carry the samplers' old log-probs.

  Returns:
    A TrainExampleV1 with padded ids/masks, advantages, and policy versions.
  """
  padded = pad_samples(
      samples,
      tokenizer_info=tokenizer_info,
      shape_config=shape_config,
      use_rollout_logps=use_rollout_logps,
  )
  policy_version = np.asarray(
      [sample.policy_version for sample in samples], dtype=np.int32
  )
  return datatypes.TrainExampleV1(
      loss_mask=padded.completion_mask,
      prompt_ids=padded.prompt_ids,
      prompt_mask=padded.prompt_mask,
      completion_ids=padded.completion_ids,
      advantages=np.asarray(advantages, dtype=np.float32),
      old_per_token_logps=padded.old_per_token_logps,
      policy_version=policy_version,
  )
