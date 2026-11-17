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

"""JAX profiler backend for Tunix trainers."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from absl import logging
import jax
from tunix.sft import profiler


def _supports_max_num_hosts(max_num_hosts: int | None) -> bool:
  """Returns True if max_num_hosts is set and supported by JAX start_trace."""
  if max_num_hosts is None:
    return False
  try:
    sig = inspect.signature(jax.profiler.start_trace)
    if "max_num_hosts" in sig.parameters:
      return True
    logging.warning(
        "max_num_hosts is specified in ProfilerOptions but "
        "jax.profiler.start_trace does not accept it. This is "
        "only supported for Pathways workloads using pathwaysutils."
    )
  except (ValueError, TypeError) as e:
    logging.warning(
        "Failed to inspect signature of jax.profiler.start_trace: %s. "
        "Not passing max_num_hosts.",
        e,
    )
  return False


def _build_jax_profile_options(
    options: profiler.ProfilerOptions,
) -> jax.profiler.ProfileOptions | None:
  """Builds `jax.profiler.ProfileOptions` from `ProfilerOptions` if enabled."""
  if not (options.set_profile_options or options.enable_continuous_profiling):
    return None
  profile_options = jax.profiler.ProfileOptions()
  if options.set_profile_options:
    profile_options.host_tracer_level = options.host_tracer_level
    profile_options.python_tracer_level = options.python_tracer_level
  if options.enable_continuous_profiling:
    profile_options.advanced_configuration = {
        "enable_continuous_profiling": True
    }
  return profile_options


class JaxProfilerBackend:
  """JAX implementation of `ProfilerBackend` using `jax.profiler`."""

  def is_main_process(self) -> bool:
    """Returns True on JAX process 0."""
    return jax.process_index() == 0

  def start(self, options: profiler.ProfilerOptions) -> None:
    """Starts a JAX profiler trace for `options`."""
    profile_options = _build_jax_profile_options(options)
    if _supports_max_num_hosts(options.max_num_hosts):
      jax.profiler.start_trace(
          log_dir=options.log_dir,
          profiler_options=profile_options,
          max_num_hosts=options.max_num_hosts,  # pyrefly: ignore[unexpected-keyword]
      )
    else:
      jax.profiler.start_trace(
          log_dir=options.log_dir,
          profiler_options=profile_options,
      )

  def stop(self) -> None:
    """Stops the active JAX profiler trace."""
    jax.profiler.stop_trace()


if TYPE_CHECKING:
  _BACKEND_TYPE_CHECK: profiler.ProfilerBackend = JaxProfilerBackend()
