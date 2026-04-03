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
"""Checkpointing options for Tunix."""

from collections.abc import Mapping
import dataclasses
from typing import Generic, Any, Protocol, TypeVar
from absl import logging

from orbax.checkpoint import v1 as ocp


# The subset of config options that are used to allow for configuration of
# save_decision_policy and preservation_policy.
CONFIG_OPTIONS = frozenset([
    "save_interval_steps",
    "max_to_keep",
    "enable_async_checkpointing",
    "timeout_secs",
])


class AsyncOptionsProtocol(Protocol):
  """Interface for async options."""

  @property
  def timeout_secs(self) -> int | None:
    ...


MetadataT = TypeVar("MetadataT")


class NameFormatProtocol(Protocol, Generic[MetadataT]):
  """Interface for step name format."""

  ...


class CheckpointingOptions(Protocol):
  """Interface for checkpointing options."""

  @property
  def save_decision_policy(
      self,
  ) -> ocp.training.save_decision_policies.SaveDecisionPolicy | None:
    ...

  @property
  def preservation_policy(
      self,
  ) -> ocp.training.preservation_policies.PreservationPolicy | None:
    ...

  @property
  def step_name_format(
      self,
  ) -> NameFormatProtocol | None:
    ...

  @property
  def enable_async_checkpointing(self) -> bool | None:
    ...

  @property
  def async_options(self) -> AsyncOptionsProtocol | None:
    ...


@dataclasses.dataclass(frozen=True)
class _CheckpointingOptions:
  """Configuration options for v1 Orbax checkpointing.

  Attributes:
    save_decision_policy: A policy that defines when to save a checkpoint.
    preservation_policy: A policy that defines when to preserve a checkpoint.
    step_name_format: A format for step names. If not explicitly set, defaults
      to simple integer steps.
    enable_async_checkpointing: Whether to use async checkpointing.
    async_options: The options for async operations.
  """
  save_decision_policy: (
      ocp.training.save_decision_policies.SaveDecisionPolicy | None
  ) = None
  preservation_policy: (
      ocp.training.preservation_policies.PreservationPolicy | None
  ) = None
  step_name_format: (
      NameFormatProtocol | None
  ) = None
  enable_async_checkpointing: bool | None = None
  async_options: AsyncOptionsProtocol | None = None


# Default checkpointing options for Tunix:
# - Save every 180 seconds.
# - Keep the latest 3 checkpoints.
# - Use simple integer step names.
# - Use async checkpointing.
# - Timeout for async operations is 1200 seconds.
DEFAULT_CHECKPOINTING_OPTIONS = _CheckpointingOptions(
    save_decision_policy=ocp.training.save_decision_policies.ContinuousCheckpointingPolicy(
        minimum_interval_secs=180,
    ),
    preservation_policy=ocp.training.preservation_policies.LatestN(n=3),
    step_name_format=ocp.path.step.standard_name_format(),
    enable_async_checkpointing=True,
    async_options=ocp.options.AsyncOptions(timeout_secs=1200),
)


def create_checkpointing_options(
    save_decision_policy: (
        ocp.training.save_decision_policies.SaveDecisionPolicy | None
    ) = None,
    preservation_policy: (
        ocp.training.preservation_policies.PreservationPolicy | None
    ) = None,
    step_name_format: (
        ocp.path.step.NameFormat | None
    ) = None,
    enable_async_checkpointing: bool | None = None,
    async_options: AsyncOptionsProtocol | None = None,
) -> CheckpointingOptions:
  """Creates a CheckpointingOptions instance."""
  return _CheckpointingOptions(
      save_decision_policy=save_decision_policy,
      preservation_policy=preservation_policy,
      step_name_format=step_name_format,
      enable_async_checkpointing=enable_async_checkpointing,
      async_options=async_options,
  )


def resolve_checkpointing_defaults(
    options: CheckpointingOptions | None = None,
) -> _CheckpointingOptions:
  """Resolves checkpointing options with defensive fallbacks and Tunix defaults.

  This function handles both our custom `_CheckpointingOptions` dataclass and
  standard `ocp.CheckpointManagerOptions`, which satisfy our structural
  Protocol.

  Args:
    options: The options object to resolve.

  Returns:
    A resolved `_CheckpointingOptions` instance.
  """
  if options is None:
    return DEFAULT_CHECKPOINTING_OPTIONS

  if (save_policy := getattr(options, "save_decision_policy", None)) is None:
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

  if (preserve_policy := getattr(options, "preservation_policy", None)) is None:
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

  if (step_name_format := getattr(options, "step_name_format", None)) is None:
    step_name_format = DEFAULT_CHECKPOINTING_OPTIONS.step_name_format

  if (
      enable_async := getattr(options, "enable_async_checkpointing", None)
  ) is None:
    enable_async = DEFAULT_CHECKPOINTING_OPTIONS.enable_async_checkpointing

  if (timeout := getattr(options, "timeout_secs", None)) is None:
    if (async_opts := getattr(options, "async_options", None)) is not None:
      timeout = getattr(async_opts, "timeout_secs", None)

  if timeout is not None:
    # We want to only allow configuration of timeout_secs, and not the entire
    # async_options, so we create a new AsyncOptions object here.
    async_options = ocp.options.AsyncOptions(timeout_secs=timeout)
  else:
    async_options = DEFAULT_CHECKPOINTING_OPTIONS.async_options

  return _CheckpointingOptions(
      save_decision_policy=save_policy,
      preservation_policy=preserve_policy,
      step_name_format=step_name_format,
      enable_async_checkpointing=enable_async,
      async_options=async_options,
  )


def dict_to_checkpointing_options(
    options: Mapping[str, Any],
) -> _CheckpointingOptions:
  """Converts a mapping of options to V1 CheckpointingOptions.

  This functionality enforces that we use Tunix defaults for all options except
  for those explicitly provided in the mapping. It also validates that no
  unsupported options are provided.

  Args:
    options: The options to convert.

  Returns:
    A `_CheckpointingOptions` instance.

  Raises:
    ValueError: If any of the options are not supported.
  """
  # Validate that the options are supported.
  invalid_options = set(options.keys()) - CONFIG_OPTIONS
  if invalid_options:
    raise ValueError(
        f"The following options {invalid_options} are not supported for"
        " Checkpointing, please refer to the following set of configurable"
        f" options: {CONFIG_OPTIONS}, alongside documentation found at"
        " https://tunix.readthedocs.io/en/latest/launching.html#training-configuration-training-config."
    )

  save_interval_steps = options.get("save_interval_steps")
  max_to_keep = options.get("max_to_keep")

  if save_interval_steps is not None:
    save_decision_policy = (
        ocp.training.save_decision_policies.FixedIntervalPolicy(
            save_interval_steps
        )
    )
  else:
    save_decision_policy = None

  if max_to_keep is not None:
    preservation_policy = ocp.training.preservation_policies.LatestN(
        max_to_keep
    )
  else:
    preservation_policy = None

  timeout_secs = options.get("timeout_secs")
  async_options = (
      ocp.options.AsyncOptions(timeout_secs=timeout_secs)
      if timeout_secs is not None
      else None
  )

  return _CheckpointingOptions(
      save_decision_policy=save_decision_policy,
      preservation_policy=preservation_policy,
      enable_async_checkpointing=options.get("enable_async_checkpointing"),
      async_options=async_options,
  )
