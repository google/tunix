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

"""Profiler class for Tunix trainers."""

import dataclasses
import threading

from absl import logging
import jax


@dataclasses.dataclass(frozen=True)
class ProfilerOptions:
  # Directory to write the profile to.
  log_dir: str
  # Number of steps to skip before profiling.
  skip_first_n_steps: int
  # Number of steps to profile.
  profiler_steps: int
  # Whether to set the profile options.
  set_profile_options: bool = True
  # https://github.com/jax-ml/jax/blob/0b1b909dd66a113ee0d7e54e55d0efef480e2a8a/docs/profiling.md?plain=1#L285
  host_tracer_level: int = 2  # set to 2 to capture HBM profiles.
  # https://github.com/jax-ml/jax/blob/0b1b909dd66a113ee0d7e54e55d0efef480e2a8a/docs/profiling.md?plain=1#L300
  python_tracer_level: int = 1


class Profiler:
  """Activate/deactivate a profiler based on the ProfilerOptions."""

  _lock = threading.Lock()
  _is_active: bool = False

  def __init__(
      self,
      initial_step: int,
      max_step: int | None,
      profiler_options: ProfilerOptions | None,
  ):
    if jax.process_index() != 0 or profiler_options is None:
      self._do_not_profile = True
      return
    self._profiler_options = profiler_options
    self._do_not_profile = False
    self._output_path = profiler_options.log_dir
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
      raise ValueError(
          f"First profile step {self._first_profile_step} cannot be greater"
          f" than the last profile step {self._last_profile_step}."
      )
    self._started_by_this_instance = False

  def maybe_activate(self, step: int):
    """Start the profiler."""
    if self._do_not_profile or step != self._first_profile_step:
      return
    with Profiler._lock:
      if Profiler._is_active:
        logging.warning(
            "A JAX profiler is already active. Skipping activation of this"
            " profiler."
        )
        return
      logging.info("Starting JAX profiler at step %d.", step)
      if self._profiler_options.set_profile_options:
        profile_options = jax.profiler.ProfileOptions()
        profile_options.host_tracer_level = (
            self._profiler_options.host_tracer_level
        )
        profile_options.python_tracer_level = (
            self._profiler_options.python_tracer_level
        )
        jax.profiler.start_trace(
            log_dir=self._output_path, profiler_options=profile_options
        )
      else:
        jax.profiler.start_trace(log_dir=self._output_path)
      Profiler._is_active = True
      self._started_by_this_instance = True

  def maybe_deactivate(self, step: int):
    """End the profiler."""
    if self._do_not_profile or step != self._last_profile_step:
      return
    with Profiler._lock:
      if not self._started_by_this_instance:
        logging.warning(
            "This profiler instance did not start JAX profiler. Skipping"
            " deactivation."
        )
        return
      logging.info("Stopping JAX profiler at step %d.", step)
      jax.profiler.stop_trace()
      Profiler._is_active = False
      self._started_by_this_instance = False

  def _set_last_profile_step(self, profiler_steps, max_step):
    calculated_last_step = self._first_profile_step + profiler_steps
    if max_step is None:
      return calculated_last_step
    return min(calculated_last_step, max_step)
