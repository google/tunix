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

"""Tests for the [PERF] stage-timing markers."""

import io
import os
import unittest
from unittest import mock

from tunix.rl import perf_log


class PerfLogTest(unittest.TestCase):

  def test_phase_prints_stage_and_seconds(self):
    out = io.StringIO()
    with mock.patch("sys.stdout", out):
      with perf_log.phase("rollout_generate", step=3) as info:
        info["rows"] = 256
    line = out.getvalue().strip()
    self.assertIn("[PERF]", line)
    self.assertIn("step=3", line)
    self.assertIn("stage=rollout_generate", line)
    self.assertIn("seconds=", line)
    self.assertIn("rows=256", line)

  def test_phase_calls_sink_with_duration(self):
    calls = []
    with mock.patch("sys.stdout", io.StringIO()):
      with perf_log.phase("rescore_b", sink=lambda s, dt: calls.append((s, dt))):
        pass
    self.assertEqual(len(calls), 1)
    self.assertEqual(calls[0][0], "rescore_b")
    self.assertGreaterEqual(calls[0][1], 0.0)

  def test_sink_failure_does_not_raise(self):
    def bad_sink(stage, seconds):
      raise RuntimeError("sink down")

    out = io.StringIO()
    with mock.patch("sys.stdout", out):
      with perf_log.phase("weight_sync", sink=bad_sink):
        pass
    self.assertIn("WARN metric sink failed", out.getvalue())

  def test_kill_switch_silences_output(self):
    out = io.StringIO()
    with mock.patch.dict(os.environ, {"CANON_PERF_LOG": "0"}):
      with mock.patch("sys.stdout", out):
        with perf_log.phase("rollout_generate", step=1):
          pass
    self.assertEqual(out.getvalue(), "")

  def test_phase_still_prints_when_body_raises(self):
    out = io.StringIO()
    with mock.patch("sys.stdout", out):
      with self.assertRaises(ValueError):
        with perf_log.phase("optimizer_transaction"):
          raise ValueError("boom")
    self.assertIn("stage=optimizer_transaction", out.getvalue())

  def test_timed_decorator(self):
    out = io.StringIO()

    @perf_log.timed("prefill_rescore")
    def fn(x):
      return x + 1

    with mock.patch("sys.stdout", out):
      self.assertEqual(fn(1), 2)
    self.assertIn("stage=prefill_rescore", out.getvalue())


if __name__ == "__main__":
  unittest.main()
