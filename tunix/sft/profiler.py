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

"""Profiler class and backend protocol for Tunix trainers."""

from __future__ import annotations

import dataclasses
import threading
from typing import Protocol

from absl import logging


@dataclasses.dataclass(frozen=True)
class ProfilerOptions:
  """Options for configuring the profiler."""

  # Directory to write the profile to.
  log_dir: str
  # Number of steps to skip before profiling.
  skip_first_n_steps: int
  # Number of steps to profile.
  profiler_steps: int
  # Whether to set tracer level options (host_tracer_level, python_tracer_level).
  set_profile_options: bool = True
  # https://github.com/jax-ml/jax/blob/0b1b909dd66a113ee0d7e54e55d0efef480e2a8a/docs/profiling.md?plain=1#L285
  host_tracer_level: int = 2  # set to 2 to capture HBM profiles.
  # https://github.com/jax-ml/jax/blob/0b1b909dd66a113ee0d7e54e55d0efef480e2a8a/docs/profiling.md?plain=1#L300
  python_tracer_level: int = 1
  # Maximum number of hosts to profile. Only supported for Pathways workloads
  # using pathwaysutils.
  max_num_hosts: int | None = None
  # Whether to enable continuous profiling via JAX advanced_configuration.
  enable_continuous_profiling: bool = False


class ProfilerBackend(Protocol):
  """Backend interface for starting and stopping step-window profiler traces."""

  def is_main_process(self) -> bool:
    """Returns True if the current process should record profiles."""
    ...

  def start(self, options: ProfilerOptions) -> None:
    """Starts profiling with the given options."""
    ...

  def stop(self) -> None:
    """Stops the active profiling session."""
    ...


class Profiler:
  """Activate/deactivate a profiler based on the ProfilerOptions."""

  _lock = threading.Lock()
  _is_active: bool = False

  def __init__(
      self,
      initial_step: int,
      max_step: int | None,
      profiler_options: ProfilerOptions | None,
      backend: ProfilerBackend | None = None,
  ):
    if profiler_options is None:
      self._do_not_profile = True
      return
    if backend is None:
      raise ValueError("backend must be provided when profiler_options is set.")
    self._backend: ProfilerBackend = backend
    if not self._backend.is_main_process():
      self._do_not_profile = True
      return
    self._profiler_options = profiler_options
    self._do_not_profile = False
    # This is step number, starting from 0.
    self._first_profile_step = (
        initial_step + profiler_options.skip_first_n_steps
    )
    # This is step number + 1, starting from 1.
    self._last_profile_step = self._set_last_profile_step(
        profiler_options.profiler_steps, max_step
    )
    # We use >= instead of > because last_profile_step is step number + 1.
    if self._first_profile_step >= self._last_profile_step:
      # An empty profiling window is a misconfiguration of the profiler, not
      # of the training run: warn and skip profiling rather than abort a run
      # that would otherwise train fine.
      logging.warning(
          "Profiler disabled: first profile step %d is not before the last"
          " profile step %d (initial_step=%d, max_step=%d,"
          " skip_first_n_steps=%d, profiler_steps=%d).",
          self._first_profile_step,
          self._last_profile_step,
          initial_step,
          max_step,
          profiler_options.skip_first_n_steps,
          profiler_options.profiler_steps,
      )
      self._do_not_profile = True
      return
    self._started_by_this_instance = False

  def maybe_activate(self, step: int) -> None:
    """Start the profiler."""
    if self._do_not_profile or step != self._first_profile_step:
      return
    with Profiler._lock:
      if Profiler._is_active:
        logging.warning(
            "A profiler is already active. Skipping activation of this"
            " profiler."
        )
        return
      logging.info("Starting profiler at step %d.", step)
      self._backend.start(self._profiler_options)
      Profiler._is_active = True
      self._started_by_this_instance = True

  def maybe_deactivate(self, step: int) -> None:
    """End the profiler."""
    if self._do_not_profile or step != self._last_profile_step:
      return
    with Profiler._lock:
      if not self._started_by_this_instance:
        logging.warning(
            "This profiler instance did not start the profiler. Skipping"
            " deactivation."
        )
        return
      logging.info("Stopping profiler at step %d.", step)
      self._backend.stop()
      Profiler._is_active = False
      self._started_by_this_instance = False

  def _set_last_profile_step(
      self, profiler_steps: int, max_step: int | None
  ) -> int:
    calculated_last_step = self._first_profile_step + profiler_steps
    if max_step is None:
      return calculated_last_step
    return min(calculated_last_step, max_step)
