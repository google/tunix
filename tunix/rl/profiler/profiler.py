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

import collections
import contextlib
import logging
import threading
from typing import Any, Optional
import jax
from tunix.common import configs

try:
  import google_cloud_mldiagnostics as mldiag  # pyrefly: ignore[missing-import]  # pylint: disable=g-import-not-at-top

  mldiag_xprof = getattr(mldiag, "xprof", None)

  if mldiag_xprof is None and hasattr(mldiag, "src"):
    _src = getattr(mldiag, "src")
    _inner = getattr(_src, "google_cloud_mldiagnostics", None)
    mldiag_xprof = getattr(_inner, "xprof", None)

  _HAS_MLDIAG = mldiag_xprof is not None
except ImportError:
  mldiag_xprof = None
  _HAS_MLDIAG = False


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


class ProfileScope:
  """Handle yielded by profile() to receive deferred future outputs."""

  __slots__ = ("future_output",)

  def __init__(self, future_output: Any = None):
    self.future_output = future_output

  def set_future_output(self, future_output: Any) -> None:
    self.future_output = future_output


class RLProfiler:
  """Basic RL Profiler handling targeted execution bounds."""

  def __init__(self, config: configs.RLProfileConfig):
    self.config = config
    self._call_counts: dict[str, int] = collections.defaultdict(int)

    if getattr(config, "managed_mldiagnostics", False):
      if not _HAS_MLDIAG or mldiag_xprof is None:
        raise RuntimeError(
            "RLProfileConfig.managed_mldiagnostics is True, but "
            "google_cloud_mldiagnostics is not available."
        )
      # Hardcoded None to capture multi-role (Trainer + Sampler) Pathways
      # topologies.
      self._profiler = mldiag_xprof(  # type: ignore[not-callable]
          process_index_list=None
      )
      # Pre-flight check: trigger lazy initialization immediately at step 0 so
      # missing ML run / GCS path errors fail fast at startup instead of
      # mid-run.
      if hasattr(self._profiler, "_ensure_initialized"):
        self._profiler._ensure_initialized()
    else:
      # Fallback chain: Explicit diagnostics dir -> Standard output dir
      output_dir = getattr(config, "mldiagnostics_dir", "") or config.output_dir
      self._profiler = _JAXProfilerBackend(output_dir)

    self._is_active: bool = False
    self._active_role: Optional[str] = None
    self._stop_thread: Optional[threading.Thread] = None
    self._lock: threading.Lock = threading.Lock()

  def should_profile(self, role: str) -> bool:
    return (
        self.config.start_invocation
        <= self._call_counts[role]
        < self.config.start_invocation + self.config.num_invocations
    )

  @contextlib.contextmanager
  def profile(self, role: str, future_output: Any = None):
    """Context manager to activate and deactivate targeted profiling.

    Args:
      role: Role string (e.g. 'actor', 'rollout') requesting trace activation.
      future_output: Device array/model to block on asynchronously before
        stopping. Can also be set dynamically inside the block via the yielded
        scope.

    Yields:
      ProfileScope handle for setting future_output deferredly.
    """
    scope = ProfileScope(future_output)
    self.maybe_activate(role)
    try:
      yield scope
    finally:
      self.maybe_deactivate(role, future_output=scope.future_output)

  def maybe_activate(self, role: str) -> None:
    """Activates the profiler for a given role if within profile window.

    Note: In nested execution loops (like actor/critic training),
    this traces exactly one micro-batch/update step per invocation
    to prevent trace file bloat, rather than the entire global step block.

    Args:
      role: Role string (e.g. 'actor', 'rollout') requesting trace activation.
    """
    if not self.should_profile(role):
      return

    # If a previous background stop thread is still finishing, wait for it so
    # that the trace cleanly ends before starting a new one.
    thread_to_join = None
    with self._lock:
      if self._stop_thread is not None and self._stop_thread.is_alive():
        thread_to_join = self._stop_thread

    if thread_to_join is not None:
      thread_to_join.join(timeout=self.config.timeout_secs)

    with self._lock:
      if self._is_active:
        logging.warning(
            "RLProfiler: Ignoring start for '%s' because '%s' is already"
            " active.",
            role,
            self._active_role,
        )
        return

      session_id = f"{role}_invocation_{self._call_counts[role]}"
      self._profiler.start(session_id=session_id)
      self._is_active = True
      self._active_role = role

  def maybe_deactivate(self, role: str, future_output: Any = None) -> None:
    """Deactivates the profiler for a given role.

    If active in this role: if `future_output` is provided, spawns a daemon
    thread that executes `jax.block_until_ready(future_output)` followed by
    `_profiler.stop()` and resets state. If `future_output` is None, stops
    synchronously in a finally block.

    Always increments the internal invocation counter for `role` in a finally
    block so that call counts advance on every invocation.

    Args:
      role: Role string (e.g. 'actor', 'rollout') requesting deactivation.
      future_output: Optional device tensor or pytree to asynchronously await
        via `jax.block_until_ready` before stopping the trace.
    """
    try:
      with self._lock:
        if not (self._is_active and self._active_role == role):
          return

        # If deactivation has already been dispatched to a stop thread for this
        # active session, do not dispatch again.
        if self._stop_thread is not None and self._stop_thread.is_alive():
          return

        if future_output is not None:
          target_future_output = future_output
          future_output = None

          def _async_stop():
            nonlocal target_future_output
            try:
              try:
                jax.block_until_ready(target_future_output)
              finally:
                target_future_output = None
                self._profiler.stop()
            except Exception:  # pylint: disable=broad-exception-caught
              logging.exception(
                  "RLProfiler: Background trace deactivation failed."
              )
              raise
            finally:
              with self._lock:
                self._is_active = False
                self._active_role = None
                self._stop_thread = None

          self._stop_thread = threading.Thread(target=_async_stop, daemon=True)
          self._stop_thread.start()
        else:
          try:
            self._profiler.stop()
          finally:
            self._is_active = False
            self._active_role = None
    finally:
      with self._lock:
        self._call_counts[role] += 1

  def close(self) -> None:
    """Joins any active background stop thread and stops any trace still active."""
    thread_to_join = None
    with self._lock:
      if self._stop_thread is not None and self._stop_thread.is_alive():
        thread_to_join = self._stop_thread

    if thread_to_join is not None:
      thread_to_join.join(timeout=self.config.timeout_secs)
      if thread_to_join.is_alive():
        logging.warning(
            "RLProfiler: Background stop thread did not finish within %ss"
            " timeout.",
            self.config.timeout_secs,
        )

    with self._lock:
      if self._is_active:
        if self._stop_thread is not None and self._stop_thread.is_alive():
          logging.warning(
              "RLProfiler: Stop thread still alive during close; skipping"
              " redundant stop."
          )
        else:
          try:
            self._profiler.stop()
          finally:
            self._is_active = False
            self._active_role = None
            self._stop_thread = None
