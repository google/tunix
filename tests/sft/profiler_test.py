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

import ast
import pathlib
import sys
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
from tunix.sft import profiler
from tunix.sft import profiler_jax


class _FakeProfilerBackend:
  """In-memory `ProfilerBackend` double for testing backend injection."""

  def __init__(self, main_process: bool = True):
    self._main_process = main_process
    self.is_main_process_calls: int = 0
    self.started_with: list[profiler.ProfilerOptions] = []
    self.stop_count: int = 0

  def is_main_process(self) -> bool:
    self.is_main_process_calls += 1
    return self._main_process

  def start(self, options: profiler.ProfilerOptions) -> None:
    self.started_with.append(options)

  def stop(self) -> None:
    self.stop_count += 1


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
        initial_step=0,
        max_step=100,
        profiler_options=profiler_options,
        backend=profiler_jax.JaxProfilerBackend(),
    )

    self.assertFalse(p._do_not_profile)
    self.assertEqual(p._profiler_options.log_dir, self.log_dir)
    self.assertEqual(p._first_profile_step, 10)
    self.assertEqual(p._last_profile_step, 15)

    p.maybe_activate(10)  # Activate at the correct step
    mock_start_trace.assert_called_once_with(
        log_dir=self.log_dir, profiler_options=mock.ANY
    )

    p.maybe_deactivate(15)  # Deactivate at the correct step
    mock_stop_trace.assert_called_once()

  @parameterized.named_parameters(
      dict(
          testcase_name='no_options_set',
          set_profile_options=False,
          enable_continuous_profiling=False,
          expect_profile_options=False,
          expected_advanced_config=None,
      ),
      dict(
          testcase_name='only_set_profile_options',
          set_profile_options=True,
          enable_continuous_profiling=False,
          expect_profile_options=True,
          expected_advanced_config={},
      ),
      dict(
          testcase_name='only_enable_continuous_profiling',
          set_profile_options=False,
          enable_continuous_profiling=True,
          expect_profile_options=True,
          expected_advanced_config={'enable_continuous_profiling': True},
      ),
      dict(
          testcase_name='both_options_enabled',
          set_profile_options=True,
          enable_continuous_profiling=True,
          expect_profile_options=True,
          expected_advanced_config={'enable_continuous_profiling': True},
      ),
  )
  @mock.patch.object(jax, 'process_index', return_value=0)
  @mock.patch.object(jax.profiler, 'start_trace')
  def test_profiler_active_with_profile_options_or_continuous_profiling(
      self,
      mock_start_trace,
      _,
      set_profile_options,
      enable_continuous_profiling,
      expect_profile_options,
      expected_advanced_config,
  ):
    profiler_options = profiler.ProfilerOptions(
        log_dir=self.log_dir,
        skip_first_n_steps=10,
        profiler_steps=5,
        set_profile_options=set_profile_options,
        host_tracer_level=3,
        python_tracer_level=2,
        enable_continuous_profiling=enable_continuous_profiling,
    )
    p = profiler.Profiler(
        initial_step=0,
        max_step=100,
        profiler_options=profiler_options,
        backend=profiler_jax.JaxProfilerBackend(),
    )
    p.maybe_activate(10)

    mock_start_trace.assert_called_once()
    called_options = mock_start_trace.call_args.kwargs['profiler_options']
    if not expect_profile_options:
      self.assertIsNone(called_options)
    else:
      self.assertIsNotNone(called_options)
      self.assertEqual(
          called_options.advanced_configuration, expected_advanced_config
      )
      if set_profile_options:
        self.assertEqual(called_options.host_tracer_level, 3)
        self.assertEqual(called_options.python_tracer_level, 2)
      else:
        default_options = jax.profiler.ProfileOptions()
        self.assertEqual(
            called_options.host_tracer_level, default_options.host_tracer_level
        )
        self.assertEqual(
            called_options.python_tracer_level,
            default_options.python_tracer_level,
        )

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
        initial_step=0,
        max_step=100,
        profiler_options=profiler_options,
        backend=profiler_jax.JaxProfilerBackend(),
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
        backend=profiler_jax.JaxProfilerBackend(),
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
        backend=profiler_jax.JaxProfilerBackend(),
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
        initial_step=0,
        max_step=100,
        profiler_options=profiler_options,
        backend=profiler_jax.JaxProfilerBackend(),
    )
    p2 = profiler.Profiler(
        initial_step=0,
        max_step=100,
        profiler_options=profiler_options,
        backend=profiler_jax.JaxProfilerBackend(),
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
  @mock.patch.object(jax, 'process_index', return_value=0)
  @mock.patch.object(jax.profiler, 'start_trace')
  def test_invalid_step_numbers_disable_profiling(
      self,
      mock_start_trace,
      _,
      max_step,
      initial_step,
      skip_first_n_steps,
      profiler_steps,
  ):
    profiler_options = profiler.ProfilerOptions(
        log_dir=self.log_dir,
        skip_first_n_steps=skip_first_n_steps,
        profiler_steps=profiler_steps,
    )
    with self.assertLogs(level='WARNING') as logs:
      p = profiler.Profiler(
          initial_step=initial_step,
          max_step=max_step,
          profiler_options=profiler_options,
          backend=profiler_jax.JaxProfilerBackend(),
      )
    self.assertTrue(any('Profiler disabled' in line for line in logs.output))
    self.assertTrue(p._do_not_profile)

    p.maybe_activate(initial_step + skip_first_n_steps)
    mock_start_trace.assert_not_called()

  def test_profiler_with_max_num_hosts_supported(self):
    def dummy_start_trace(
        log_dir, *, profiler_options=None, max_num_hosts=None
    ):
      del log_dir, profiler_options, max_num_hosts

    mock_start_trace = mock.create_autospec(dummy_start_trace)

    with mock.patch.object(jax.profiler, 'start_trace', mock_start_trace):
      profiler_options = profiler.ProfilerOptions(
          log_dir=self.log_dir,
          skip_first_n_steps=10,
          profiler_steps=5,
          max_num_hosts=5,
      )
      p = profiler.Profiler(
          initial_step=0,
          max_step=100,
          profiler_options=profiler_options,
          backend=profiler_jax.JaxProfilerBackend(),
      )
      p.maybe_activate(10)

      mock_start_trace.assert_called_once_with(
          log_dir=self.log_dir,
          profiler_options=mock.ANY,
          max_num_hosts=5,
      )

  @mock.patch.object(profiler_jax.logging, 'warning')
  def test_profiler_with_max_num_hosts_unsupported(self, mock_warning):
    def dummy_start_trace(log_dir, *, profiler_options=None):
      del log_dir, profiler_options

    mock_start_trace = mock.create_autospec(dummy_start_trace)

    with mock.patch.object(jax.profiler, 'start_trace', mock_start_trace):
      profiler_options = profiler.ProfilerOptions(
          log_dir=self.log_dir,
          skip_first_n_steps=10,
          profiler_steps=5,
          max_num_hosts=5,
      )
      p = profiler.Profiler(
          initial_step=0,
          max_step=100,
          profiler_options=profiler_options,
          backend=profiler_jax.JaxProfilerBackend(),
      )
      p.maybe_activate(10)

      mock_start_trace.assert_called_once_with(
          log_dir=self.log_dir,
          profiler_options=mock.ANY,
      )
      mock_warning.assert_called_once_with(
          'max_num_hosts is specified in ProfilerOptions but '
          'jax.profiler.start_trace does not accept it. This is '
          'only supported for Pathways workloads using pathwaysutils.'
      )

  @mock.patch.object(profiler_jax.inspect, 'signature')
  @mock.patch.object(jax.profiler, 'start_trace')
  @mock.patch.object(profiler_jax.logging, 'warning')
  def test_profiler_with_signature_inspection_failure(
      self, mock_warning, mock_start_trace, mock_signature
  ):
    mock_signature.side_effect = ValueError('Incompatible object')
    profiler_options = profiler.ProfilerOptions(
        log_dir=self.log_dir,
        skip_first_n_steps=10,
        profiler_steps=5,
        max_num_hosts=5,
    )
    p = profiler.Profiler(
        initial_step=0,
        max_step=100,
        profiler_options=profiler_options,
        backend=profiler_jax.JaxProfilerBackend(),
    )
    p.maybe_activate(10)

    # It should fall back to calling start_trace without max_num_hosts.
    mock_start_trace.assert_called_once_with(
        log_dir=self.log_dir,
        profiler_options=mock.ANY,
    )
    # It should log a warning about the failed inspection.
    mock_warning.assert_called_once_with(
        'Failed to inspect signature of jax.profiler.start_trace: %s. '
        'Not passing max_num_hosts.',
        mock_signature.side_effect,
    )

  @mock.patch.object(jax.profiler, 'start_trace', autospec=True)
  @mock.patch.object(jax.profiler, 'stop_trace', autospec=True)
  def test_custom_backend_injection_and_non_main_process(
      self, mock_stop_trace, mock_start_trace
  ):
    profiler_options = profiler.ProfilerOptions(
        log_dir=self.log_dir, skip_first_n_steps=2, profiler_steps=3
    )
    disabled_backend = _FakeProfilerBackend(main_process=True)
    with mock.patch.dict(
        sys.modules, {'jax': None, 'tunix.sft.profiler_jax': None}
    ):
      p_disabled = profiler.Profiler(
          initial_step=0,
          max_step=10,
          profiler_options=None,
          backend=disabled_backend,
      )
      self.assertTrue(p_disabled._do_not_profile)
      self.assertEqual(disabled_backend.is_main_process_calls, 0)

      with self.assertRaisesRegex(
          ValueError, 'backend must be provided when profiler_options is set'
      ):
        profiler.Profiler(
            initial_step=0,
            max_step=10,
            profiler_options=profiler_options,
            backend=None,
        )

      main_backend = _FakeProfilerBackend(main_process=True)
      p_main = profiler.Profiler(
          initial_step=0,
          max_step=10,
          profiler_options=profiler_options,
          backend=main_backend,
      )
    self.assertFalse(p_main._do_not_profile)
    self.assertEqual(main_backend.is_main_process_calls, 1)
    p_main.maybe_activate(2)
    self.assertEqual(main_backend.started_with, [profiler_options])
    p_main.maybe_deactivate(5)
    self.assertEqual(main_backend.stop_count, 1)

    worker_backend = _FakeProfilerBackend(main_process=False)
    p_worker = profiler.Profiler(
        initial_step=0,
        max_step=10,
        profiler_options=profiler_options,
        backend=worker_backend,
    )
    self.assertTrue(p_worker._do_not_profile)
    p_worker.maybe_activate(2)
    p_worker.maybe_deactivate(5)
    self.assertEmpty(worker_backend.started_with)
    self.assertEqual(worker_backend.stop_count, 0)
    mock_start_trace.assert_not_called()
    mock_stop_trace.assert_not_called()

  def test_jax_profiler_backend_helpers_and_no_jax_import_in_profiler(self):
    opts_disabled = profiler.ProfilerOptions(
        log_dir=self.log_dir,
        skip_first_n_steps=1,
        profiler_steps=2,
        set_profile_options=False,
        enable_continuous_profiling=False,
        max_num_hosts=None,
    )
    self.assertIsNone(profiler_jax._build_jax_profile_options(opts_disabled))

    def start_trace_with_hosts(
        log_dir, *, profiler_options=None, max_num_hosts=None
    ):
      del log_dir, profiler_options, max_num_hosts

    with mock.patch.object(
        jax.profiler,
        'start_trace',
        mock.create_autospec(start_trace_with_hosts),
    ):
      self.assertFalse(profiler_jax._supports_max_num_hosts(None))
      self.assertTrue(profiler_jax._supports_max_num_hosts(4))

    backend = profiler_jax.JaxProfilerBackend()
    with mock.patch.object(jax, 'process_index', return_value=0, autospec=True):
      self.assertTrue(backend.is_main_process())
    with (
        mock.patch.object(
            jax.profiler, 'start_trace', autospec=True
        ) as mock_start,
        mock.patch.object(
            jax.profiler, 'stop_trace', autospec=True
        ) as mock_stop,
    ):
      backend.start(opts_disabled)
      mock_start.assert_called_once_with(
          log_dir=self.log_dir, profiler_options=None
      )
      backend.stop()
      mock_stop.assert_called_once()

    source = pathlib.Path(profiler.__file__).read_text(encoding='utf-8')
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
      if isinstance(node, ast.Import):
        imported_roots.update(alias.name.split('.')[0] for alias in node.names)
      elif isinstance(node, ast.ImportFrom) and node.module:
        imported_roots.add(node.module.split('.')[0])
    self.assertNotIn('jax', imported_roots)

    top_level_imports: set[str] = set()
    for node in tree.body:
      if isinstance(node, (ast.Import, ast.ImportFrom)):
        top_level_imports.update(alias.name for alias in node.names)
      if isinstance(node, ast.ImportFrom) and node.module:
        top_level_imports.add(node.module)
    self.assertNotIn('profiler_jax', top_level_imports)


if __name__ == '__main__':
  absltest.main()
