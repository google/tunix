#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "tunix/perf/profile_window.py"
SPEC = importlib.util.spec_from_file_location("v1_profile_window", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
profile_window = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = profile_window
SPEC.loader.exec_module(profile_window)


class ProfileWindowTest(unittest.TestCase):

  def test_exports_only_target_and_only_newest_committed_step(self):
    calls = []

    def sink(timelines):
      calls.append(timelines)
      return {"written": (1, None)}

    export = profile_window.single_step_export_fn(sink, target_step=2)
    timeline = types.SimpleNamespace(
        id="host-1", born=10.0, committed_steps=[]
    )
    for step in range(4):
      timeline.committed_steps.append({step: f"span-{step}"})
      result = export({timeline.id: timeline})
      if step == 2:
        self.assertEqual(result, {"written": (1, None)})
      else:
        self.assertEqual(result, {})
    self.assertEqual(len(calls), 1)
    snapshot = calls[0]["host-1"]
    self.assertEqual(snapshot.committed_steps, [{2: "span-2"}])
    self.assertEqual(len(timeline.committed_steps), 4)

  def test_rejects_negative_target(self):
    with self.assertRaisesRegex(ValueError, "non-negative"):
      profile_window.single_step_export_fn(lambda unused: {}, target_step=-1)


if __name__ == "__main__":
  unittest.main()
