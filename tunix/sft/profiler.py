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
from absl import logging
import jax


@dataclasses.dataclass
class ProfilerOptions:
  # Directory to write the profile to.
  log_dir: str
  # Number of steps to skip before profiling.
  skip_first_n_steps: int
  # Number of steps to profile.
  profiler_steps: int


class Profiler:
  """Activate/deactivate a profiler based on the ProfilerOptions."""

  def __init__(
      self,
      initial_step: int,
      max_step: int | None,
      profiler_options: ProfilerOptions | None,
  ):
    if jax.process_index() != 0 or profiler_options is None:
      self._do_not_profile = True
      return
    self._do_not_profile = False
    self._output_path = profiler_options.log_dir
    self._first_profile_step = self._set_first_profile_step(
        profiler_options.skip_first_n_steps, initial_step
    )
    self._last_profile_step = self._set_last_profile_step(
        profiler_options.profiler_steps, max_step
    )

  def activate(self):
    """Start the profiler."""
    if self._do_not_profile:
      return
    logging.info("Starting JAX profiler.")
    jax.profiler.start_trace(self._output_path)

  def deactivate(self):
    """End the profiler."""
    if self._do_not_profile:
      return
    logging.info("Stopping JAX profiler.")
    jax.profiler.stop_trace()

  def should_activate(self, step: int):
    """Returns True if the profiler should be activated at the given step."""
    if self._do_not_profile:
      return False
    return step == self._first_profile_step

  def should_deactivate(self, step: int):
    """Returns True if the profiler should be deactivated at the given step."""
    if self._do_not_profile:
      return False
    return step == self._last_profile_step

  def _set_first_profile_step(self, skip_first_n_steps, initial_step):
    return initial_step + skip_first_n_steps

  def _set_last_profile_step(self, profiler_steps, max_step):
    calculated_last_step = self._first_profile_step + profiler_steps
    if max_step is None:
      return calculated_last_step
    return min(calculated_last_step, max_step)
