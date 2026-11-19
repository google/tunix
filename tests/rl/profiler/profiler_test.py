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

"""Tests for RL Profiler invocation bounding and lifecycle."""

from unittest import mock
from absl.testing import absltest
from tunix.common import configs
from tunix.common.datatypes import Role
from tunix.rl.profiler import profiler


class RLProfilerTest(absltest.TestCase):

  def test_should_profile_window_matching(self):
    config = configs.RLProfileConfig(
        start_invocation=2, num_invocations=2, output_dir="/tmp/test"
    )
    p = profiler.RLProfiler(config)

    # Initial call count is 0 for both roles.
    self.assertFalse(p.should_profile(Role.ACTOR.value))
    self.assertFalse(p.should_profile(Role.ROLLOUT.value))

    # Advance actor counter manually to test bounds.
    p._call_counts[Role.ACTOR.value] = 1
    self.assertFalse(p.should_profile(Role.ACTOR.value))

    p._call_counts[Role.ACTOR.value] = 2
    self.assertTrue(p.should_profile(Role.ACTOR.value))

    p._call_counts[Role.ACTOR.value] = 3
    self.assertTrue(p.should_profile(Role.ACTOR.value))

    p._call_counts[Role.ACTOR.value] = 4
    self.assertFalse(p.should_profile(Role.ACTOR.value))

    # Rollout counter was untouched and remains 0.
    self.assertFalse(p.should_profile(Role.ROLLOUT.value))

  def test_maybe_activate_matching_invocation_starts_trace(self):
    config = configs.RLProfileConfig(
        start_invocation=0, num_invocations=1, output_dir="/tmp/test"
    )
    p = profiler.RLProfiler(config)
    p._profiler = mock.MagicMock()

    p.maybe_activate(Role.ACTOR.value)

    p._profiler.start.assert_called_once_with(
        session_id=f"{Role.ACTOR.value}_invocation_0"
    )
    self.assertTrue(p._is_active)
    self.assertEqual(p._active_role, Role.ACTOR.value)

  def test_maybe_activate_non_matching_invocation_ignores(self):
    config = configs.RLProfileConfig(
        start_invocation=2, num_invocations=1, output_dir="/tmp/test"
    )
    p = profiler.RLProfiler(config)
    p._profiler = mock.MagicMock()

    p.maybe_activate(Role.ACTOR.value)

    p._profiler.start.assert_not_called()
    self.assertFalse(p._is_active)
    self.assertIsNone(p._active_role)

  def test_maybe_activate_mutual_exclusion_warns_and_drops(self):
    config = configs.RLProfileConfig(
        start_invocation=0, num_invocations=5, output_dir="/tmp/test"
    )
    p = profiler.RLProfiler(config)
    p._profiler = mock.MagicMock()

    p.maybe_activate(Role.ACTOR.value)
    self.assertTrue(p._is_active)
    self.assertEqual(p._active_role, Role.ACTOR.value)

    with self.assertLogs(level="WARNING") as logs:
      p.maybe_activate(Role.ROLLOUT.value)

    self.assertEqual(p._profiler.start.call_count, 1)
    self.assertEqual(p._active_role, Role.ACTOR.value)
    self.assertIn("Ignoring start", "".join(logs.output))

  def test_maybe_activate_start_failure_does_not_commit_state(self):
    config = configs.RLProfileConfig(
        start_invocation=0, num_invocations=1, output_dir="/tmp/test"
    )
    p = profiler.RLProfiler(config)
    p._profiler = mock.MagicMock()
    p._profiler.start.side_effect = RuntimeError("start failed")

    with self.assertRaises(RuntimeError):
      p.maybe_activate(Role.ACTOR.value)

    self.assertFalse(p._is_active)
    self.assertIsNone(p._active_role)

  def test_maybe_deactivate_synchronous(self):
    config = configs.RLProfileConfig(
        start_invocation=0, num_invocations=1, output_dir="/tmp/test"
    )
    p = profiler.RLProfiler(config)
    p._profiler = mock.MagicMock()

    p.maybe_activate(Role.ACTOR.value)
    p.maybe_deactivate(Role.ACTOR.value)  # future_output=None

    p._profiler.stop.assert_called_once()
    self.assertFalse(p._is_active)
    self.assertIsNone(p._active_role)
    self.assertEqual(p._call_counts[Role.ACTOR.value], 1)

  def test_maybe_deactivate_advances_counter_for_unprofiled_call(self):
    config = configs.RLProfileConfig(
        start_invocation=2, num_invocations=1, output_dir="/tmp/test"
    )
    p = profiler.RLProfiler(config)
    p._profiler = mock.MagicMock()

    # Invocations 0 and 1: not profiled, but counter increments.
    p.maybe_activate(Role.ACTOR.value)
    p.maybe_deactivate(Role.ACTOR.value)
    self.assertEqual(p._call_counts[Role.ACTOR.value], 1)
    p._profiler.start.assert_not_called()

    p.maybe_activate(Role.ACTOR.value)
    p.maybe_deactivate(Role.ACTOR.value)
    self.assertEqual(p._call_counts[Role.ACTOR.value], 2)
    p._profiler.start.assert_not_called()

    # Invocation 2: matches start_invocation=2, profiles.
    p.maybe_activate(Role.ACTOR.value)
    p._profiler.start.assert_called_once_with(
        session_id=f"{Role.ACTOR.value}_invocation_2"
    )
    p.maybe_deactivate(Role.ACTOR.value)
    self.assertEqual(p._call_counts[Role.ACTOR.value], 3)
    p._profiler.stop.assert_called_once()

  def test_maybe_deactivate_async_with_future_output(self):
    config = configs.RLProfileConfig(
        start_invocation=0, num_invocations=1, output_dir="/tmp/test"
    )
    p = profiler.RLProfiler(config)
    p._profiler = mock.MagicMock()
    mock_future_output = [mock.MagicMock()]

    with mock.patch.object(profiler.jax, "block_until_ready") as mock_block:
      p.maybe_activate(Role.ACTOR.value)
      p.maybe_deactivate(Role.ACTOR.value, future_output=mock_future_output)

      # Invocation count advances immediately on caller thread.
      self.assertEqual(p._call_counts[Role.ACTOR.value], 1)

      # Background thread was spawned.
      self.assertIsNotNone(p._stop_thread)
      p._stop_thread.join(timeout=5.0)

      mock_block.assert_called_once_with(mock_future_output)
      p._profiler.stop.assert_called_once()
      self.assertFalse(p._is_active)
      self.assertIsNone(p._active_role)

  def test_maybe_deactivate_resets_state_even_if_stop_throws(self):
    config = configs.RLProfileConfig(
        start_invocation=0, num_invocations=1, output_dir="/tmp/test"
    )
    p = profiler.RLProfiler(config)
    p._profiler = mock.MagicMock()
    p.maybe_activate(Role.ACTOR.value)
    p._profiler.stop.side_effect = RuntimeError("Failed to stop")

    with self.assertRaises(RuntimeError):
      p.maybe_deactivate(Role.ACTOR.value)

    self.assertFalse(p._is_active)
    self.assertIsNone(p._active_role)
    self.assertEqual(p._call_counts[Role.ACTOR.value], 1)

  def test_maybe_deactivate_wrong_role_ignored(self):
    config = configs.RLProfileConfig(
        start_invocation=0, num_invocations=1, output_dir="/tmp/test"
    )
    p = profiler.RLProfiler(config)
    p._profiler = mock.MagicMock()
    p.maybe_activate(Role.ACTOR.value)

    p.maybe_deactivate(Role.CRITIC.value)

    p._profiler.stop.assert_not_called()
    self.assertTrue(p._is_active)
    self.assertEqual(p._active_role, Role.ACTOR.value)
    self.assertEqual(p._call_counts[Role.CRITIC.value], 1)
    self.assertEqual(p._call_counts[Role.ACTOR.value], 0)

  def test_close_joins_active_thread(self):
    config = configs.RLProfileConfig(
        start_invocation=0, num_invocations=1, output_dir="/tmp/test"
    )
    p = profiler.RLProfiler(config)
    p._profiler = mock.MagicMock()
    mock_thread = mock.MagicMock()
    mock_thread.is_alive.return_value = True

    def mock_join(timeout=None):
      del timeout
      p._is_active = False
      p._active_role = None

    mock_thread.join.side_effect = mock_join
    p._stop_thread = mock_thread
    p._is_active = True
    p._active_role = Role.ACTOR.value

    p.close()

    mock_thread.join.assert_called_once_with(timeout=60.0)
    p._profiler.stop.assert_not_called()
    self.assertFalse(p._is_active)
    self.assertIsNone(p._active_role)

  def test_close_uses_configured_timeout(self):
    config = configs.RLProfileConfig(
        start_invocation=0,
        num_invocations=1,
        timeout_secs=15.0,
        output_dir="/tmp/test",
    )
    p = profiler.RLProfiler(config)
    p._profiler = mock.MagicMock()
    mock_thread = mock.MagicMock()
    mock_thread.is_alive.return_value = True

    def mock_join(timeout=None):
      del timeout
      p._is_active = False
      p._active_role = None

    mock_thread.join.side_effect = mock_join
    p._stop_thread = mock_thread
    p._is_active = True
    p._active_role = Role.ACTOR.value

    p.close()

    mock_thread.join.assert_called_once_with(timeout=15.0)

  def test_maybe_activate_uses_configured_timeout_when_joining_thread(self):
    config = configs.RLProfileConfig(
        start_invocation=0,
        num_invocations=1,
        timeout_secs=15.0,
        output_dir="/tmp/test",
    )
    p = profiler.RLProfiler(config)
    p._profiler = mock.MagicMock()
    mock_thread = mock.MagicMock()
    mock_thread.is_alive.return_value = True

    def mock_join(timeout=None):
      del timeout
      p._is_active = False
      p._active_role = None

    mock_thread.join.side_effect = mock_join
    p._stop_thread = mock_thread

    p.maybe_activate(Role.ACTOR.value)

    mock_thread.join.assert_called_once_with(timeout=15.0)

  def test_close_stops_active_trace_if_not_stopped_by_thread(self):
    config = configs.RLProfileConfig(
        start_invocation=0, num_invocations=1, output_dir="/tmp/test"
    )
    p = profiler.RLProfiler(config)
    p._profiler = mock.MagicMock()
    p._is_active = True
    p._active_role = Role.ACTOR.value

    p.close()

    p._profiler.stop.assert_called_once()
    self.assertFalse(p._is_active)
    self.assertIsNone(p._active_role)

  def test_close_noop_when_inactive(self):
    config = configs.RLProfileConfig(
        start_invocation=0, num_invocations=1, output_dir="/tmp/test"
    )
    p = profiler.RLProfiler(config)
    p._profiler = mock.MagicMock()

    p.close()

    p._profiler.stop.assert_not_called()

  def test_multi_invocation_window_profiles_each_matching_invocation(self):
    config = configs.RLProfileConfig(
        start_invocation=1, num_invocations=2, output_dir="/tmp/test"
    )
    p = profiler.RLProfiler(config)
    p._profiler = mock.MagicMock()
    mock_future_output = [mock.MagicMock()]

    # Invocation 0: before window (not profiled)
    p.maybe_activate(Role.ACTOR.value)
    p.maybe_deactivate(Role.ACTOR.value, future_output=mock_future_output)
    self.assertEqual(p._call_counts[Role.ACTOR.value], 1)
    p._profiler.start.assert_not_called()
    p._profiler.stop.assert_not_called()

    # Invocation 1: matching invocation -> starts and stops
    p.maybe_activate(Role.ACTOR.value)
    p._profiler.start.assert_called_once_with(
        session_id=f"{Role.ACTOR.value}_invocation_1"
    )
    p.maybe_deactivate(Role.ACTOR.value, future_output=mock_future_output)
    if p._stop_thread is not None:
      p._stop_thread.join(timeout=5.0)
    self.assertEqual(p._call_counts[Role.ACTOR.value], 2)
    self.assertEqual(p._profiler.stop.call_count, 1)
    self.assertFalse(p._is_active)

    # Invocation 2: matching invocation -> starts and stops
    p.maybe_activate(Role.ACTOR.value)
    self.assertEqual(p._profiler.start.call_count, 2)
    p.maybe_deactivate(Role.ACTOR.value, future_output=mock_future_output)
    if p._stop_thread is not None:
      p._stop_thread.join(timeout=5.0)
    self.assertEqual(p._call_counts[Role.ACTOR.value], 3)
    self.assertEqual(p._profiler.stop.call_count, 2)
    self.assertFalse(p._is_active)

    # Invocation 3: past window -> not profiled
    p.maybe_activate(Role.ACTOR.value)
    p.maybe_deactivate(Role.ACTOR.value, future_output=mock_future_output)
    self.assertEqual(p._call_counts[Role.ACTOR.value], 4)
    self.assertEqual(p._profiler.start.call_count, 2)
    self.assertEqual(p._profiler.stop.call_count, 2)

  def test_multi_invocation_window_interleaved_multi_role(self):
    config = configs.RLProfileConfig(
        start_invocation=1, num_invocations=2, output_dir="/tmp/test"
    )
    p = profiler.RLProfiler(config)
    p._profiler = mock.MagicMock()
    mock_future_output = [mock.MagicMock()]

    # Iteration 0 (inv 0): not profiled
    p.maybe_activate(Role.ROLLOUT.value)
    p.maybe_deactivate(Role.ROLLOUT.value, future_output=mock_future_output)
    p.maybe_activate(Role.ACTOR.value)
    p.maybe_deactivate(Role.ACTOR.value, future_output=mock_future_output)
    self.assertEqual(p._profiler.start.call_count, 0)

    # Iteration 1 (inv 1): both rollout and actor are profiled
    p.maybe_activate(Role.ROLLOUT.value)
    p.maybe_deactivate(Role.ROLLOUT.value, future_output=mock_future_output)
    p.maybe_activate(Role.ACTOR.value)
    p.maybe_deactivate(Role.ACTOR.value, future_output=mock_future_output)
    self.assertEqual(p._profiler.start.call_count, 2)

    # Iteration 2 (inv 2): both rollout and actor are profiled
    p.maybe_activate(Role.ROLLOUT.value)
    p.maybe_deactivate(Role.ROLLOUT.value, future_output=mock_future_output)
    p.maybe_activate(Role.ACTOR.value)
    p.maybe_deactivate(Role.ACTOR.value, future_output=mock_future_output)
    self.assertEqual(p._profiler.start.call_count, 4)

    if p._stop_thread is not None:
      p._stop_thread.join(timeout=5.0)
    self.assertEqual(p._profiler.stop.call_count, 4)

  def test_async_deactivation_clears_thread_reference(self):
    config = configs.RLProfileConfig(
        start_invocation=0, num_invocations=1, output_dir="/tmp/test"
    )
    p = profiler.RLProfiler(config)
    p._profiler = mock.MagicMock()
    mock_future_output = [mock.MagicMock()]

    p.maybe_activate(Role.ACTOR.value)
    p.maybe_deactivate(Role.ACTOR.value, future_output=mock_future_output)
    self.assertIsNotNone(p._stop_thread)
    p._stop_thread.join(timeout=5.0)

    self.assertIsNone(p._stop_thread)
    self.assertFalse(p._is_active)

  def test_unprofiled_call_during_active_stop_thread_does_not_duplicate_stop(
      self,
  ):
    config = configs.RLProfileConfig(
        start_invocation=0, num_invocations=1, output_dir="/tmp/test"
    )
    p = profiler.RLProfiler(config)
    p._profiler = mock.MagicMock()
    mock_future_output = [mock.MagicMock()]

    # Invocation 0 is profiled.
    p.maybe_activate(Role.ACTOR.value)
    p.maybe_deactivate(Role.ACTOR.value, future_output=mock_future_output)
    self.assertEqual(p._call_counts[Role.ACTOR.value], 1)

    # Invocation 1 is not profiled (window is [0, 1)).
    # Even if stop thread from invocation 0 is alive, calling maybe_deactivate
    # for invocation 1 must not spawn another thread or double-stop.
    p.maybe_activate(Role.ACTOR.value)
    p.maybe_deactivate(Role.ACTOR.value, future_output=mock_future_output)
    self.assertEqual(p._call_counts[Role.ACTOR.value], 2)

    if p._stop_thread is not None:
      p._stop_thread.join(timeout=5.0)
    self.assertEqual(p._profiler.stop.call_count, 1)

  def test_sequential_roles_future_output_handoff(self):
    config = configs.RLProfileConfig(
        start_invocation=0, num_invocations=1, output_dir="/tmp/test"
    )
    p = profiler.RLProfiler(config)
    p._profiler = mock.MagicMock()
    mock_future_output = [mock.MagicMock()]

    # Rollout profiles invocation 0.
    p.maybe_activate(Role.ROLLOUT.value)
    p.maybe_deactivate(Role.ROLLOUT.value, future_output=mock_future_output)

    # Actor profiles invocation 0. Pending rollout thread is joined.
    p.maybe_activate(Role.ACTOR.value)
    p.maybe_deactivate(Role.ACTOR.value, future_output=mock_future_output)

    if p._stop_thread is not None:
      p._stop_thread.join(timeout=5.0)

    self.assertEqual(p._profiler.start.call_count, 2)
    self.assertEqual(p._profiler.stop.call_count, 2)

  def test_async_deactivation_with_mock_array_block_until_ready(self):
    config = configs.RLProfileConfig(
        start_invocation=0, num_invocations=1, output_dir="/tmp/test"
    )
    p = profiler.RLProfiler(config)
    p._profiler = mock.MagicMock()

    mock_array = mock.MagicMock()
    p.maybe_activate(Role.ACTOR.value)
    p.maybe_deactivate(Role.ACTOR.value, future_output=[mock_array])

    self.assertIsNotNone(p._stop_thread)
    p._stop_thread.join(timeout=5.0)

    mock_array.block_until_ready.assert_called_once()
    p._profiler.stop.assert_called_once()

  def test_close_handles_stop_thread_timeout_gracefully(self):
    config = configs.RLProfileConfig(
        start_invocation=0, num_invocations=1, output_dir="/tmp/test"
    )
    p = profiler.RLProfiler(config)
    p._profiler = mock.MagicMock()
    mock_thread = mock.MagicMock()
    mock_thread.is_alive.return_value = True
    p._stop_thread = mock_thread
    p._is_active = True
    p._active_role = Role.ACTOR.value

    with self.assertLogs(level="WARNING") as logs:
      p.close()

    mock_thread.join.assert_called_once_with(timeout=60.0)
    p._profiler.stop.assert_not_called()
    self.assertIn("Background stop thread did not finish", "".join(logs.output))

  def test_maybe_deactivate_async_block_until_ready_exception_still_stops_profiler(
      self,
  ):
    config = configs.RLProfileConfig(
        start_invocation=0, num_invocations=1, output_dir="/tmp/test"
    )
    p = profiler.RLProfiler(config)
    p._profiler = mock.MagicMock()
    mock_future_output = [mock.MagicMock()]

    with mock.patch.object(
        profiler.jax,
        "block_until_ready",
        side_effect=RuntimeError("Device error"),
    ) as mock_block:
      p.maybe_activate(Role.ACTOR.value)
      p.maybe_deactivate(Role.ACTOR.value, future_output=mock_future_output)

      self.assertIsNotNone(p._stop_thread)
      p._stop_thread.join(timeout=5.0)

      mock_block.assert_called_once_with(mock_future_output)
      # Profiler stop MUST still be called despite block_until_ready exception.
      p._profiler.stop.assert_called_once()
      self.assertFalse(p._is_active)
      self.assertIsNone(p._active_role)
      self.assertEqual(p._call_counts[Role.ACTOR.value], 1)

  def test_mldiagnostics_initialization_success(self):
    config = configs.RLProfileConfig(
        start_invocation=1,
        num_invocations=1,
        managed_mldiagnostics=True,
    )
    mock_xprof = mock.MagicMock()
    with mock.patch.object(profiler, "_HAS_MLDIAG", True), mock.patch.object(
        profiler, "mldiag_xprof", mock_xprof
    ):
      p = profiler.RLProfiler(config)
      mock_xprof.assert_called_once_with(process_index_list=None)
      mock_xprof.return_value._ensure_initialized.assert_called_once()
      self.assertEqual(p._profiler, mock_xprof.return_value)

  def test_mldiagnostics_uninitialized_run_raises_at_startup(self):
    config = configs.RLProfileConfig(
        start_invocation=1,
        num_invocations=1,
        managed_mldiagnostics=True,
    )
    mock_xprof = mock.MagicMock()
    mock_xprof.return_value._ensure_initialized.side_effect = RuntimeError(
        "No active ML run found"
    )
    with mock.patch.object(profiler, "_HAS_MLDIAG", True), mock.patch.object(
        profiler, "mldiag_xprof", mock_xprof
    ):
      with self.assertRaisesRegex(RuntimeError, "No active ML run found"):
        profiler.RLProfiler(config)

  def test_mldiagnostics_missing_sdk_raises_runtime_error(self):
    config = configs.RLProfileConfig(
        start_invocation=1,
        num_invocations=1,
        managed_mldiagnostics=True,
    )
    with mock.patch.object(profiler, "_HAS_MLDIAG", False):
      with self.assertRaisesRegex(
          RuntimeError, "google_cloud_mldiagnostics is not available"
      ):
        profiler.RLProfiler(config)

  @mock.patch.object(profiler.jax.profiler, "start_trace")
  @mock.patch.object(profiler.jax.profiler, "stop_trace")
  def test_jax_profiler_backend(self, mock_stop, mock_start):
    backend = profiler._JAXProfilerBackend("/tmp/output")
    backend.start(session_id="test_session")
    mock_start.assert_called_once()
    options = mock_start.call_args[1].get("profiler_options")
    self.assertEqual(options.session_id, "test_session")

    backend.stop()
    mock_stop.assert_called_once()

  def test_profile_context_manager_static_future_output(self):
    config = configs.RLProfileConfig(
        start_invocation=0, num_invocations=1, output_dir="/tmp/test"
    )
    p = profiler.RLProfiler(config)
    p._profiler = mock.MagicMock()

    mock_tensor = mock.MagicMock()
    with mock.patch.object(profiler.jax, "block_until_ready") as mock_block:
      with p.profile(Role.ACTOR.value, future_output=mock_tensor):
        p._profiler.start.assert_called_once_with(
            session_id="actor_invocation_0"
        )
      p.close()
      mock_block.assert_called_once_with(mock_tensor)
      p._profiler.stop.assert_called_once()

    self.assertEqual(p._call_counts[Role.ACTOR.value], 1)

  def test_profile_context_manager_dynamic_scope_future_output(self):
    config = configs.RLProfileConfig(
        start_invocation=0, num_invocations=1, output_dir="/tmp/test"
    )
    p = profiler.RLProfiler(config)
    p._profiler = mock.MagicMock()

    mock_tokens = [mock.MagicMock()]
    with mock.patch.object(profiler.jax, "block_until_ready") as mock_block:
      with p.profile(Role.ROLLOUT.value) as scope:
        p._profiler.start.assert_called_once_with(
            session_id="rollout_invocation_0"
        )
        scope.set_future_output(mock_tokens)
      p.close()
      mock_block.assert_called_once_with(mock_tokens)
      p._profiler.stop.assert_called_once()

    self.assertEqual(p._call_counts[Role.ROLLOUT.value], 1)

  def test_profile_context_manager_exception_safety(self):
    config = configs.RLProfileConfig(
        start_invocation=0, num_invocations=1, output_dir="/tmp/test"
    )
    p = profiler.RLProfiler(config)
    p._profiler = mock.MagicMock()

    with self.assertRaisesRegex(ValueError, "step failure"):
      with p.profile(Role.ACTOR.value):
        raise ValueError("step failure")

    p.close()
    p._profiler.stop.assert_called_once()
    self.assertEqual(p._call_counts[Role.ACTOR.value], 1)


if __name__ == "__main__":
  absltest.main()
