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

"""Pure assembler: normalized rollout samples -> a numpy TrainExampleV1.

This is the pad/mask/assemble core of the existing `_process_results`, lifted into
one dependency-free function so the existing learner and the orchestrator share a
single implementation. It is deliberately pure: no rl_cluster, no device arrays,
no I/O. The impure stages that surround it -- reference/actor log-prob scoring,
reward/advantage computation, metrics, trajectory logging, and TIS -- stay with
their owners; their outputs (advantages now, ref-logps later) are injected around
this core.

Both callers first normalize their own representation into `SampleTokens` (the
existing learner from its Token-dicts, the orchestrator from RolloutResult
segments), then call `trajectories_to_train_example`. Padding reuses the same
`tunix.rl.agentic.utils` helpers, so the two callers produce identical arrays.
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


def trajectories_to_train_example(
    samples: list[SampleTokens],
    advantages: np.ndarray,
    *,
    tokenizer_info: datatypes.TokenizerInfo,
    shape_config: datatypes.ShapeConfig,
    use_rollout_logps: bool = False,
) -> datatypes.TrainExampleV1:
  """Assembles a group's normalized samples into a numpy TrainExampleV1.

  Prompts are left-padded and completions right-padded to the ShapeConfig
  buckets; the completion mask becomes the payload's `loss_mask`. Advantages and
  per-row policy versions are carried through. `ref_per_token_logps` is left None
  (the inference stage injects it when beta != 0); `old_per_token_logps` is
  populated from the samples only when `use_rollout_logps` is set.

  Args:
    samples: The group's members (length G), in group order.
    advantages: Per-sample advantages, shape [G] (or [G, C]).
    tokenizer_info: Supplies the pad id used for padding.
    shape_config: Supplies max_prompt_length and max_response_tokens.
    use_rollout_logps: Whether to carry the samplers' old log-probs.

  Returns:
    A TrainExampleV1 with padded ids/masks, advantages, and policy versions.

  Raises:
    ValueError: If `samples` is empty.
  """
  if not samples:
    raise ValueError("cannot assemble a TrainExample from zero samples")

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
            list(sample.prompt_tokens),
            list(sample.completion_tokens),
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
  prompt_mask = (prompt_ids != pad_id).astype(np.int32)
  completion_ids = np.asarray(padded_completion_ids, dtype=np.int32)
  completion_mask = np.asarray(padded_completion_masks, dtype=np.int32)
  old_per_token_logps = (
      np.asarray(padded_old_logprobs, dtype=np.float32)
      if use_rollout_logps and padded_old_logprobs
      else None
  )
  policy_version = np.asarray(
      [sample.policy_version for sample in samples], dtype=np.int32
  )

  return datatypes.TrainExampleV1(
      loss_mask=completion_mask,
      prompt_ids=prompt_ids,
      prompt_mask=prompt_mask,
      completion_ids=completion_ids,
      advantages=np.asarray(advantages, dtype=np.float32),
      old_per_token_logps=old_per_token_logps,
      policy_version=policy_version,
  )
