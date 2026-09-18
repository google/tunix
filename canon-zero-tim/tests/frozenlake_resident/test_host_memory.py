"""Focused tests for the isolated P45 host-memory contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = _ROOT / "tunix/rl/host_memory.py"
_SPEC = importlib.util.spec_from_file_location("p45_host_memory", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
host_memory = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = host_memory
_SPEC.loader.exec_module(host_memory)


class P45HostMemoryTest(unittest.TestCase):

  def test_reads_cgroup_v2_and_proc_status(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      (root / "memory.current").write_text("123\n")
      (root / "memory.peak").write_text("456\n")
      (root / "memory.max").write_text("max\n")
      status = root / "status"
      status.write_text("Name:\tpython\nVmHWM:\t8 kB\nVmRSS:\t4 kB\n")

      snapshot = host_memory.snapshot(
          cgroup_root=str(root), proc_status_path=str(status)
      )

    self.assertEqual(snapshot["cgroup_current_bytes"], 123)
    self.assertEqual(snapshot["cgroup_peak_bytes"], 456)
    self.assertIsNone(snapshot["cgroup_limit_bytes"])
    self.assertEqual(snapshot["process_rss_bytes"], 4 * 1024)
    self.assertEqual(snapshot["process_hwm_bytes"], 8 * 1024)

  def test_contract_is_disabled_by_default_and_rejects_bad_interval(self):
    self.assertEqual(
        host_memory.contract({}),
        (False, 0),
    )
    with self.assertRaisesRegex(ValueError, "positive integer"):
      host_memory.contract({
          "CANON_P45_HOST_MEMORY_TELEMETRY": "1",
          "CANON_P45_HOST_GC_INTERVAL": "0",
      })

  def test_gc_runs_only_on_committed_step_cadence(self):
    calls = []

    def collector():
      calls.append(True)
      return 7

    self.assertIsNone(
        host_memory.maybe_collect_garbage(
            step=9, interval=10, collector=collector
        )
    )
    self.assertEqual(
        host_memory.maybe_collect_garbage(
            step=10, interval=10, collector=collector
        ),
        7,
    )
    self.assertEqual(len(calls), 1)

if __name__ == "__main__":
  unittest.main()
