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

"""Unit tests for `profiler`."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from tunix.sft import profiler
import tempfile


class ProfilerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    try:
      self.log_dir = self.create_tempdir().full_path
    except Exception:
      self.log_dir = tempfile.TemporaryDirectory().name
    # Reset class variable state to ensure test isolation.
    profiler.Profiler._is_active = False

  @mock.patch.object(jax, 'process_index', return_value=0)
  @mock.patch.object(jax.profiler, 'start_trace')
  @mock.patch.object(jax.profiler, 'stop_trace')
  def test_profiler_active_with_options_process_0(
      self, mock_stop_trace, mock_start_trace, _
  ):
    profiler_options = profiler.ProfilerOptions(
        log_dir=self.log_dir, skip_first_n_steps=10, profiler_steps=5
    )
    p = profiler.Profiler(
        initial_step=0, max_step=100, profiler_options=profiler_options
    )

    self.assertFalse(p._do_not_profile)
    self.assertEqual(p._output_path, self.log_dir)
    self.assertEqual(p._first_profile_step, 10)
    self.assertEqual(p._last_profile_step, 15)

    p.maybe_activate(10)  # Activate at the correct step
    mock_start_trace.assert_called_once_with(
        log_dir=self.log_dir, profiler_options=mock.ANY
    )

    p.maybe_deactivate(15)  # Deactivate at the correct step
    mock_stop_trace.assert_called_once()

  @mock.patch.object(jax, 'process_index', return_value=1)
  @mock.patch.object(jax.profiler, 'start_trace')
  @mock.patch.object(jax.profiler, 'stop_trace')
  def test_profiler_inactive_for_non_zero_process(
      self, mock_stop_trace, mock_start_trace, _
  ):
    profiler_options = profiler.ProfilerOptions(
        log_dir=self.log_dir, skip_first_n_steps=10, profiler_steps=5
    )
    p = profiler.Profiler(
        initial_step=0, max_step=100, profiler_options=profiler_options
    )

    self.assertTrue(p._do_not_profile)
    mock_start_trace.assert_not_called()
    mock_stop_trace.assert_not_called()

    p.maybe_activate(10)  # Call with an arbitrary step, should not activate
    mock_start_trace.assert_not_called()

    p.maybe_deactivate(15)  # Call with an arbitrary step, should not deactivate
    mock_stop_trace.assert_not_called()

  @mock.patch.object(jax, 'process_index', return_value=0)
  @mock.patch.object(jax.profiler, 'start_trace')
  @mock.patch.object(jax.profiler, 'stop_trace')
  def test_profiler_inactive_without_options(
      self, mock_stop_trace, mock_start_trace, _
  ):
    p = profiler.Profiler(initial_step=0, max_step=100, profiler_options=None)

    self.assertTrue(p._do_not_profile)
    mock_start_trace.assert_not_called()
    mock_stop_trace.assert_not_called()

    p.maybe_activate(10)  # Call with an arbitrary step, should not activate
    mock_start_trace.assert_not_called()

    p.maybe_deactivate(15)  # Call with an arbitrary step, should not deactivate
    mock_stop_trace.assert_not_called()

  @parameterized.named_parameters(
      dict(
          testcase_name='activate_at_first_step',
          initial_step=0,
          skip_first_n_steps=5,
          current_step=5,
          expect_start_called=True,
      ),
      dict(
          testcase_name='not_activate_before_first_step',
          initial_step=0,
          skip_first_n_steps=5,
          current_step=4,
          expect_start_called=False,
      ),
      dict(
          testcase_name='not_activate_after_first_step',
          initial_step=0,
          skip_first_n_steps=5,
          current_step=6,
          expect_start_called=False,
      ),
      dict(
          testcase_name='not_active_when_do_not_profile',
          initial_step=0,
          skip_first_n_steps=5,
          current_step=5,
          profiler_options_none=True,
          expect_start_called=False,
      ),
  )
  @mock.patch.object(jax, 'process_index', return_value=0)
  @mock.patch.object(jax.profiler, 'start_trace')
  def test_maybe_activate(
      self,
      mock_start_trace,
      _,
      initial_step,
      skip_first_n_steps,
      current_step,
      expect_start_called,
      profiler_options_none=False,
  ):
    profiler_options = None
    if not profiler_options_none:
      profiler_options = profiler.ProfilerOptions(
          log_dir=self.log_dir,
          skip_first_n_steps=skip_first_n_steps,
          profiler_steps=10,
      )
    p = profiler.Profiler(
        initial_step=initial_step,
        max_step=100,
        profiler_options=profiler_options,
    )
    p.maybe_activate(current_step)
    if expect_start_called:
      mock_start_trace.assert_called_once_with(
          log_dir=self.log_dir, profiler_options=mock.ANY
      )
    else:
      mock_start_trace.assert_not_called()

  @parameterized.named_parameters(
      dict(
          testcase_name='not_deactivate_before_last_step',
          initial_step=0,
          skip_first_n_steps=5,
          profiler_steps=10,
          current_step=14,
          expect_stop_called=False,
      ),
      dict(
          testcase_name='not_deactivate_after_last_step',
          initial_step=0,
          skip_first_n_steps=5,
          profiler_steps=10,
          current_step=16,
          expect_stop_called=False,
      ),
      dict(
          testcase_name='not_active_when_do_not_profile',
          initial_step=0,
          skip_first_n_steps=5,
          profiler_steps=10,
          current_step=15,
          profiler_options_none=True,
          expect_stop_called=False,
      ),
  )
  @mock.patch.object(jax, 'process_index', return_value=0)
  @mock.patch.object(jax.profiler, 'stop_trace')
  @mock.patch.object(jax.profiler, 'start_trace')
  def test_maybe_deactivate(
      self,
      mock_start_trace,
      mock_stop_trace,
      _,
      initial_step,
      skip_first_n_steps,
      profiler_steps,
      current_step,
      expect_stop_called,
      profiler_options_none=False,
  ):
    profiler_options = None
    if not profiler_options_none:
      profiler_options = profiler.ProfilerOptions(
          log_dir=self.log_dir,
          skip_first_n_steps=skip_first_n_steps,
          profiler_steps=profiler_steps,
      )
    p = profiler.Profiler(
        initial_step=initial_step,
        max_step=100,
        profiler_options=profiler_options,
    )
    if not profiler_options_none:
      p.maybe_activate(initial_step + skip_first_n_steps)
      mock_start_trace.assert_called_once()
    p.maybe_deactivate(current_step)
    if expect_stop_called:
      mock_stop_trace.assert_called_once()
    else:
      mock_stop_trace.assert_not_called()

  @mock.patch.object(jax, 'process_index', return_value=0)
  @mock.patch.object(jax.profiler, 'stop_trace')
  @mock.patch.object(jax.profiler, 'start_trace')
  def test_multiple_profiler_instances(
      self, mock_start_trace, mock_stop_trace, _
  ):
    profiler_options = profiler.ProfilerOptions(
        log_dir=self.log_dir, skip_first_n_steps=5, profiler_steps=10
    )
    p1 = profiler.Profiler(
        initial_step=0, max_step=100, profiler_options=profiler_options
    )
    p2 = profiler.Profiler(
        initial_step=0, max_step=100, profiler_options=profiler_options
    )

    # First profiler activates.
    p1.maybe_activate(5)
    mock_start_trace.assert_called_once()

    # Second profiler tries to activate but should be skipped.
    p2.maybe_activate(5)
    mock_start_trace.assert_called_once()  # Still called only once.

    # Second profiler tries to deactivate but should be skipped.
    p2.maybe_deactivate(15)
    mock_stop_trace.assert_not_called()

    # First profiler deactivates.
    p1.maybe_deactivate(15)
    mock_stop_trace.assert_called_once()

  @parameterized.named_parameters(
      dict(
          testcase_name='first_5_last_10_max_1',
          max_step=1,
          initial_step=0,
          skip_first_n_steps=5,
          profiler_steps=5,
      ),
      dict(
          testcase_name='first_1_last_1_max_1',
          max_step=1,
          initial_step=0,
          skip_first_n_steps=1,
          profiler_steps=0,
      ),
      dict(
          testcase_name='first_1_last_2_max_1',
          max_step=1,
          initial_step=0,
          skip_first_n_steps=1,
          profiler_steps=1,
      ),
      dict(
          testcase_name='first_1_last_1_max_10',
          max_step=10,
          initial_step=0,
          skip_first_n_steps=1,
          profiler_steps=0,
      ),
  )
  def test_invalid_step_numbers(
      self, max_step, initial_step, skip_first_n_steps, profiler_steps
  ):
    profiler_options = profiler.ProfilerOptions(
        log_dir=self.log_dir,
        skip_first_n_steps=skip_first_n_steps,
        profiler_steps=profiler_steps,
    )
    with self.assertRaises(ValueError):
      profiler.Profiler(
          initial_step=initial_step,
          max_step=max_step,
          profiler_options=profiler_options,
      )


if __name__ == '__main__':
  absltest.main()
