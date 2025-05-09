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
from tunix.sft import profiler


class ProfilerTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.log_dir = self.create_tempdir().full_path

  @mock.patch('jax.process_index', return_value=0)
  @mock.patch('jax.profiler.start_trace')
  @mock.patch('jax.profiler.stop_trace')
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

    p.activate()
    mock_start_trace.assert_called_once_with(self.log_dir)

    p.deactivate()
    mock_stop_trace.assert_called_once()

  @mock.patch('jax.process_index', return_value=1)
  @mock.patch('jax.profiler.start_trace')
  @mock.patch('jax.profiler.stop_trace')
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

    p.activate()
    mock_start_trace.assert_not_called()

    p.deactivate()
    mock_stop_trace.assert_not_called()

  @mock.patch('jax.process_index', return_value=0)
  @mock.patch('jax.profiler.start_trace')
  @mock.patch('jax.profiler.stop_trace')
  def test_profiler_inactive_without_options(
      self, mock_stop_trace, mock_start_trace, _
  ):
    p = profiler.Profiler(initial_step=0, max_step=100, profiler_options=None)

    self.assertTrue(p._do_not_profile)
    mock_start_trace.assert_not_called()
    mock_stop_trace.assert_not_called()

    p.activate()
    mock_start_trace.assert_not_called()

    p.deactivate()
    mock_stop_trace.assert_not_called()

  @parameterized.named_parameters(
      dict(
          testcase_name='should_activate_at_first_step',
          initial_step=0,
          skip_first_n_steps=5,
          current_step=5,
          expected_should_activate=True,
      ),
      dict(
          testcase_name='should_not_activate_before_first_step',
          initial_step=0,
          skip_first_n_steps=5,
          current_step=4,
          expected_should_activate=False,
      ),
      dict(
          testcase_name='should_not_activate_after_first_step',
          initial_step=0,
          skip_first_n_steps=5,
          current_step=6,
          expected_should_activate=False,
      ),
      dict(
          testcase_name='not_active_when_do_not_profile',
          initial_step=0,
          skip_first_n_steps=5,
          current_step=5,
          profiler_options_none=True,
          expected_should_activate=False,
      ),
  )
  @mock.patch('jax.process_index', return_value=0)
  def test_should_activate(
      self,
      _,
      initial_step,
      skip_first_n_steps,
      current_step,
      expected_should_activate,
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
    self.assertEqual(p.should_activate(current_step), expected_should_activate)

  @parameterized.named_parameters(
      dict(
          testcase_name='should_deactivate_at_last_step',
          initial_step=0,
          skip_first_n_steps=5,
          profiler_steps=10,
          current_step=15,
          expected_should_deactivate=True,
      ),
      dict(
          testcase_name='should_not_deactivate_before_last_step',
          initial_step=0,
          skip_first_n_steps=5,
          profiler_steps=10,
          current_step=14,
          expected_should_deactivate=False,
      ),
      dict(
          testcase_name='should_not_deactivate_after_last_step',
          initial_step=0,
          skip_first_n_steps=5,
          profiler_steps=10,
          current_step=16,
          expected_should_deactivate=False,
      ),
      dict(
          testcase_name='not_active_when_do_not_profile',
          initial_step=0,
          skip_first_n_steps=5,
          profiler_steps=10,
          current_step=15,
          profiler_options_none=True,
          expected_should_deactivate=False,
      ),
  )
  @mock.patch('jax.process_index', return_value=0)
  def test_should_deactivate(
      self,
      _,
      initial_step,
      skip_first_n_steps,
      profiler_steps,
      current_step,
      expected_should_deactivate,
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
    self.assertEqual(
        p.should_deactivate(current_step), expected_should_deactivate
    )


if __name__ == '__main__':
  absltest.main()
