#!/usr/bin/env python3
"""Negative controls for the P59 XProf drop census."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest


PATH = Path(__file__).with_name("inspect_xprof_capture.py")
SPEC = importlib.util.spec_from_file_location("p59_xprof_inspector", PATH)
assert SPEC is not None and SPEC.loader is not None
INSPECTOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = INSPECTOR
SPEC.loader.exec_module(INSPECTOR)


def _profile(*, dropped=False, decode=False, semantic=True):
  planes = []
  for index in range(8):
    modules = [
        "jit_zt_tr_dp_parallel_bwd_layer_27" if semantic else "jit_bwd_layer"
    ]
    if decode:
      modules.append("jit_run_model")
    lines = [types.SimpleNamespace(
        name="XLA Modules",
        events=[types.SimpleNamespace(name=name) for name in modules],
    )]
    if dropped:
      lines.append(types.SimpleNamespace(
          name="XLA TraceMe",
          events=[types.SimpleNamespace(name="Trace Buffers Dropped")],
      ))
    planes.append(types.SimpleNamespace(
        name=f"/device:TPU:{index}",
        lines=lines,
        stats=(("dropped_traces", "10"),) if dropped and index == 0 else (),
    ))
  return types.SimpleNamespace(planes=planes)


class InspectXprofCaptureTest(unittest.TestCase):

  def test_exact_eight_plane_semantic_backward_passes(self):
    record = INSPECTOR.inspect_profile(_profile())
    self.assertEqual(record["verdict"], "PASS")
    self.assertEqual(record["tpu_plane_count"], 8)

  def test_dropped_trace_event_and_stat_are_fatal(self):
    record = INSPECTOR.inspect_profile(_profile(dropped=True))
    self.assertEqual(record["verdict"], "FAIL")
    self.assertTrue(any("dropped_traces" in reason for reason in record["reasons"]))
    self.assertTrue(any("dropped_events" in reason for reason in record["reasons"]))

  def test_decode_or_missing_semantic_label_is_fatal(self):
    record = INSPECTOR.inspect_profile(_profile(decode=True, semantic=False))
    self.assertEqual(record["verdict"], "FAIL")
    self.assertTrue(any("decode_present" in reason for reason in record["reasons"]))
    self.assertTrue(any("semantic_backward_absent" in reason for reason in record["reasons"]))


if __name__ == "__main__":
  unittest.main()
