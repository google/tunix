# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import sys
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tunix.common import datatypes
from tunix.common import test_utils
from tunix.perf import metrics
from tunix.perf import span


class MetricsTest(parameterized.TestCase):

  def test_perf_metrics_options_defaults(self):
    options = metrics.PerfMetricsOptions()
    self.assertTrue(options.enable_perf_v1)
    self.assertFalse(options.enable_perf_v2)
    self.assertEqual(options.custom_export_fn_path, "")
    self.assertEqual(options.custom_export_fn_path_v2, "")
    self.assertTrue(options.enable_trace_writer)
    self.assertEqual(options.trace_dir, "")

  @parameterized.named_parameters(
      dict(
          testcase_name="v1_disabled_but_path_set",
          enable_perf_v1=False,
          custom_export_fn_path="some.path",
          enable_perf_v2=False,
          custom_export_fn_path_v2="",
          expected_regex=(
              "custom_export_fn_path is set but enable_perf_v1 is False"
          ),
      ),
      dict(
          testcase_name="v2_disabled_but_path_set",
          enable_perf_v1=True,
          custom_export_fn_path="",
          enable_perf_v2=False,
          custom_export_fn_path_v2="some.path",
          expected_regex=(
              "custom_export_fn_path_v2 is set but enable_perf_v2 is False"
          ),
      ),
      dict(
          testcase_name="trace_writer_enabled_but_no_perf",
          enable_perf_v1=False,
          custom_export_fn_path="",
          enable_perf_v2=False,
          custom_export_fn_path_v2="",
          expected_regex=(
              "enable_trace_writer is True but neither perf v1 nor v2 is"
              " enabled."
          ),
      ),
  )
  def test_perf_metrics_options_validation_error(
      self,
      enable_perf_v1,
      custom_export_fn_path,
      enable_perf_v2,
      custom_export_fn_path_v2,
      expected_regex,
  ):
    with self.assertRaisesRegex(ValueError, expected_regex):
      metrics.PerfMetricsOptions(
          enable_perf_v1=enable_perf_v1,
          custom_export_fn_path=custom_export_fn_path,
          enable_perf_v2=enable_perf_v2,
          custom_export_fn_path_v2=custom_export_fn_path_v2,
      )

  def test_perf_metrics_options_validation_success(self):
    # Should not raise exception
    metrics.PerfMetricsOptions(
        enable_perf_v1=True,
        custom_export_fn_path="some.path",
        enable_perf_v2=True,
        custom_export_fn_path_v2="some.other.path",
        enable_trace_writer=True,
    )
    with mock.patch.object(
        metrics.logging, "warning", autospec=True
    ) as mock_warning:
      metrics.PerfMetricsOptions(
          enable_perf_v1=True,
          enable_trace_writer=False,
          trace_dir=self.create_tempdir().full_path,
      )
    mock_warning.assert_called_once()

  def test_perf_span_query_selectors(self):
    root = span.SpanGroup("root")
    step0 = span.SpanGroup("step", root)
    step1 = span.SpanGroup("step", root)
    mb0 = span.SpanGroup("mb", step1)
    mb1 = span.SpanGroup("mb", step1)
    query = metrics.PerfSpanQuery(
        {"main": types.SimpleNamespace(root=root)}, "main"
    )
    self.assertEqual(query.get_main_thread_id(), "main")
    self.assertEqual(query.get_timeline_ids(), ["main"])
    self.assertEqual(query().main().first_group("step").get(), [step0])
    self.assertEqual(
        query().timeline("main").last_group("step").nth_group("mb", 0).get(),
        [mb0],
    )
    self.assertEqual(
        query().main().last_group("step").all_groups("mb").get(),
        [mb0, mb1],
    )
    with self.assertRaisesRegex(ValueError, "timeline 'missing' not found"):
      query().timeline("missing").get()

  def test_metrics_buffer_accepts_tensor_like_values(self):
    values = np.array([1.0, 3.0], dtype=np.float32)
    custom = test_utils.CustomTensor(np.array([5.0], dtype=np.float32))
    self.assertIsInstance(values, datatypes.TensorLike)
    self.assertIsInstance(custom, datatypes.TensorLike)
    self.assertNotIsInstance(1.0, datatypes.TensorLike)
    buffer = metrics.MetricsBuffer(
        global_steps=1, metrics={"loss": ([values, custom, 2.0], np.mean)}
    )
    buffered_values, agg_fn = buffer.metrics["loss"]
    self.assertIs(agg_fn, np.mean)
    self.assertIs(buffered_values[0], values)
    self.assertIs(buffered_values[1], custom)

  def test_module_imports_without_ml_frameworks(self):
    orig = sys.modules["tunix.perf.metrics"]
    roots = (
        "jax",
        "flax",
        "optax",
        "torch",
        "jaxtyping",
        "vllm",
        "pathwaysutils",
        "tpu_sync",
    )
    prefixes = tuple(f"{r}." for r in roots)
    blocked = dict.fromkeys(
        (*roots, *(k for k in sys.modules if k.startswith(prefixes))),
        None,
    )
    try:
      with mock.patch.dict(sys.modules, blocked):
        sys.modules.pop("tunix.perf.metrics", None)
        reloaded = importlib.import_module("tunix.perf.metrics")
        self.assertEqual(reloaded.MetricsBuffer(global_steps=0).metrics, {})
    finally:
      setattr(sys.modules["tunix.perf"], "metrics", orig)


if __name__ == "__main__":
  absltest.main()
