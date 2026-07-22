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
"""Checkpointing options for Tunix."""

import dataclasses
from typing import Protocol
from absl import logging

import orbax.checkpoint as ocp_v0
from orbax.checkpoint import v1 as ocp


SaveDecisionPolicyType = (
    ocp_v0.checkpoint_managers.SaveDecisionPolicy
    | ocp.training.save_decision_policies.SaveDecisionPolicy
)
PreservationPolicyType = (
    ocp_v0.checkpoint_managers.PreservationPolicy
    | ocp.training.preservation_policies.PreservationPolicy
)
StepNameFormatType = (
    ocp_v0.path.step.NameFormat
    | ocp.path.step.NameFormat
)
AsyncOptionsType = (
    ocp_v0.AsyncOptions
    | ocp.options.AsyncOptions
)


class CheckpointingOptions(Protocol):
  """Structural protocol for representing checkpointing options.

  Any configuration object that fulfills this protocol (such as legacy v0
  `ocp.CheckpointManagerOptions`, Tunix `TunixCheckpointingOptions`, or a custom
  implementation) is supported and can be supplied directly to the Checkpointer.
  """

  @property
  def save_decision_policy(self) -> SaveDecisionPolicyType | None:
    """Returns the policy that defines when to save a checkpoint."""
    ...

  @property
  def preservation_policy(self) -> PreservationPolicyType | None:
    """Returns the policy that defines when to preserve a checkpoint."""
    ...

  @property
  def step_name_format(self) -> StepNameFormatType | None:
    """Returns the format for step names."""
    ...

  @property
  def enable_async_checkpointing(self) -> bool | None:
    """Returns whether to use async checkpointing."""
    ...

  @property
  def async_options(self) -> AsyncOptionsType | None:
    """Returns the options for async operations."""
    ...


@dataclasses.dataclass(frozen=True)
class TunixCheckpointingOptions:
  """Concrete implementation of checkpointing options for Tunix."""
  save_decision_policy: SaveDecisionPolicyType | None = None
  preservation_policy: PreservationPolicyType | None = None
  step_name_format: StepNameFormatType | None = None
  enable_async_checkpointing: bool | None = None
  async_options: AsyncOptionsType | None = None


# Default checkpointing options for Tunix:
# - Save every 180 seconds.
# - Keep the latest 3 checkpoints.
# - Use simple integer step names.
# - Use async checkpointing.
# - Timeout for async operations is 1200 seconds.
DEFAULT_CHECKPOINTING_OPTIONS = TunixCheckpointingOptions(
    save_decision_policy=ocp.training.save_decision_policies.ContinuousCheckpointingPolicy(
        minimum_interval_secs=180,
    ),
    preservation_policy=ocp.training.preservation_policies.LatestN(n=3),
    step_name_format=ocp.path.step.standard_name_format(),
    enable_async_checkpointing=True,
    async_options=ocp.options.AsyncOptions(timeout_secs=1200),
)


def create_checkpointing_options(
    *,
    save_decision_policy: SaveDecisionPolicyType | None = None,
    preservation_policy: PreservationPolicyType | None = None,
    step_name_format: StepNameFormatType | None = None,
    enable_async_checkpointing: bool | None = None,
    async_options: AsyncOptionsType | None = None,
) -> TunixCheckpointingOptions:
  """Creates a TunixCheckpointingOptions instance."""
  return TunixCheckpointingOptions(
      save_decision_policy=save_decision_policy,
      preservation_policy=preservation_policy,
      step_name_format=step_name_format,
      enable_async_checkpointing=enable_async_checkpointing,
      async_options=async_options,
  )


def resolve_checkpointing_defaults(
    options: CheckpointingOptions | None = None,
) -> TunixCheckpointingOptions:
  """Resolves options adhering to CheckpointingOptions protocol.

  This function accepts any object fulfilling the `CheckpointingOptions`
  protocol and cleanly extracts fields essential for Tunix. Legacy v0 fields
  (`save_interval_steps` or `max_to_keep`) are applied strictly as second-tier
  fallbacks, matching the explicit internal configuration logic used by Orbax V0
  for backwards compatibility.

  Args:
    options: The options object to resolve.

  Returns:
    A resolved `TunixCheckpointingOptions` instance.
  """
  if options is None:
    return DEFAULT_CHECKPOINTING_OPTIONS

  if (save_policy := options.save_decision_policy) is None:
    # save_interval_steps is a v0 CheckpointManagerOptions construct only. We
    # fall back to it for backward compatibility if v1 policies are not set.
    # TODO(b/497926314): Remove this fallback once we no longer support v0.
    if (
        save_interval := getattr(options, "save_interval_steps", None)
    ) is not None:
      logging.warning(
          "Using v0 ocp.CheckpointManagerOptions is deprecated, along with"
          " save_interval_steps. Please use a checkpointing_options with"
          " save_decision_policy instead."
      )
      save_policy = ocp.training.save_decision_policies.FixedIntervalPolicy(
          save_interval
      )
    else:
      save_policy = DEFAULT_CHECKPOINTING_OPTIONS.save_decision_policy

  if (preserve_policy := options.preservation_policy) is None:
    # max_to_keep is a v0 CheckpointManagerOptions construct only. We fall
    # back to it for backward compatibility if v1 policies are not set.
    # TODO(b/497926314): Remove this fallback once we no longer support v0.
    if (max_to_keep := getattr(options, "max_to_keep", None)) is not None:
      logging.warning(
          "Using v0 ocp.CheckpointManagerOptions is deprecated, along with"
          " max_to_keep. Please use a checkpointing_options with"
          " preservation_policy instead."
      )
      preserve_policy = ocp.training.preservation_policies.LatestN(max_to_keep)
    else:
      preserve_policy = DEFAULT_CHECKPOINTING_OPTIONS.preservation_policy

  if (step_name_format := options.step_name_format) is None:
    step_name_format = DEFAULT_CHECKPOINTING_OPTIONS.step_name_format

  if (
      enable_async := options.enable_async_checkpointing
  ) is None:
    enable_async = DEFAULT_CHECKPOINTING_OPTIONS.enable_async_checkpointing

  if (
      options.async_options is not None
      and options.async_options.timeout_secs is not None
  ):
    # We want to only allow configuration of timeout_secs, and not the entire
    # async_options, so we create a new AsyncOptions object here.
    async_options = ocp.options.AsyncOptions(
        timeout_secs=options.async_options.timeout_secs
    )
  else:
    async_options = DEFAULT_CHECKPOINTING_OPTIONS.async_options

  return create_checkpointing_options(
      save_decision_policy=save_policy,
      preservation_policy=preserve_policy,
      step_name_format=step_name_format,
      enable_async_checkpointing=enable_async,
      async_options=async_options,
  )
