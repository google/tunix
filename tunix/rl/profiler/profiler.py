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

"""RL Profiler abstraction supporting standard JAX and ML Diagnostics backends for Pathways setup."""

from typing import Optional

import jax
from tunix.common import configs

try:
  import google_cloud_mldiagnostics as mldiag  # pytype: disable=import-error  pylint: disable=g-import-not-at-top

  mldiag_xprof = getattr(mldiag, "xprof", None)

  if mldiag_xprof is None and hasattr(mldiag, "src"):
    _src = getattr(mldiag, "src")
    _inner = getattr(_src, "google_cloud_mldiagnostics", None)
    mldiag_xprof = getattr(_inner, "xprof", None)

  _HAS_MLDIAG = mldiag_xprof is not None
except ImportError:
  mldiag_xprof = None
  _HAS_MLDIAG = False


def _parse_steps(steps_str: str) -> dict[str, list[int]]:
  """Basic string splitting to get targeted profile steps."""
  parsed = {}
  if not steps_str:
    return parsed

  # Example format: "trainer:5,10;sampler:2"
  for role_steps in steps_str.split(";"):
    if ":" not in role_steps:
      continue
    role, steps = role_steps.split(":", 1)
    role = role.strip()
    parsed[role] = []
    for s in steps.split(","):
      s = s.strip()
      if s.isdigit():
        parsed[role].append(int(s))
  return parsed


class _JAXProfilerBackend:
  """Duck-typed wrapper for JAX tracing with Pathways session ID plumbing."""

  def __init__(self, output_dir: str):
    self._output_dir = output_dir

  def start(self, session_id: Optional[str] = None) -> None:
    options = jax.profiler.ProfileOptions()
    if session_id:
      # Plumb session_id so PathwaysUtils intercepts it for separated profiles
      options.session_id = session_id
    jax.profiler.start_trace(self._output_dir, profiler_options=options)

  def stop(self) -> None:
    jax.profiler.stop_trace()


class RLProfiler:
  """Basic RL Profiler handling targeted execution bounds."""

  def __init__(self, config: configs.RLProfileConfig):
    self.config = config
    self._active_steps = _parse_steps(config.profile_steps)

    if getattr(config, "managed_mldiagnostics", False) and _HAS_MLDIAG:
      # Hardcoded None to capture multi-role (Trainer + Sampler) Pathways topologies
      self._profiler = mldiag_xprof(  # type: ignore[not-callable]
          process_index_list=None
      )
    else:
      # Fallback chain: Explicit diagnostics dir -> Standard output dir
      output_dir = getattr(config, "mldiagnostics_dir", "") or config.output_dir
      self._profiler = _JAXProfilerBackend(output_dir)

  def should_profile(self, step: int, role: str) -> bool:
    return step in self._active_steps.get(role, [])

  def maybe_activate(self, step: int, role: str) -> None:
    if self.should_profile(step, role):
      self._profiler.start(session_id=f"{role}_step{step}")

  def maybe_deactivate(self, step: int, role: str) -> None:
    if self.should_profile(step, role):
      self._profiler.stop()
