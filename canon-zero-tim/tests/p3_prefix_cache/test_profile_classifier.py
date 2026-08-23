#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / (
    "canon-zero-tim/tasks/v1-phase3-prefix-cache/scripts/"
    "classify_p3_profile.py"
)
SPEC = importlib.util.spec_from_file_location("p3_apc_profile", MODULE)
assert SPEC is not None and SPEC.loader is not None
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


class ProfileClassifierTest(unittest.TestCase):

  def _fixture(self, expect_apc: bool = True):
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name)
    state = root / "state"
    raw = root / "raw.log"
    (state / "xprof" / "plugins" / "profile" / "run").mkdir(
        parents=True
    )
    (state / "perf").mkdir()
    (state / "xprof" / "plugins" / "profile" / "run" / "device.xplane.pb").write_bytes(b"xplane")
    (state / "xprof" / "plugins" / "profile" / "run" / "device.trace.json.gz").write_bytes(b"trace")
    (state / "perf" / "perfetto_trace_v2_1.pb").write_bytes(b"perfetto")
    status = "GB_GC_CERTIFICATION_GREEN" if expect_apc else "CONTROL_GREEN"
    (state / "alignment.classification.json").write_text(
        json.dumps({"status": status, "expect_apc": expect_apc}),
        encoding="utf-8",
    )
    raw.write_text(
        "[P3.XPROF] phase=diagnostic started completed_rounds=1 capture_round=1\n"
        "[CANON_ALIGN_PRE] step=0 verdict=PASS\n"
        "[CANON_ALIGN_PRE] step=1 verdict=PASS\n"
        "[P3.XPROF] phase=diagnostic stopped completed_rounds=2 captured_round=1\n"
        "[P3.XPROF] semantic_perfetto_exported completed_rounds=2\n"
        "[CANON_ALIGN_PRE] step=2 verdict=PASS\n",
        encoding="utf-8",
    )
    return temporary, raw, state

  def test_profile_with_device_and_semantic_artifacts_is_green(self):
    tmp, raw, state = self._fixture()
    with tmp:
      result = profile.classify(raw, state, True)
      self.assertEqual(result["status"], "PROFILE_GREEN")
      self.assertEqual(result["captured_round"], 1)

  def test_missing_xplane_is_rejected(self):
    tmp, raw, state = self._fixture()
    with tmp:
      next((state / "xprof").rglob("*.xplane.pb")).unlink()
      with self.assertRaisesRegex(profile.ProfileError, "no device xplane"):
        profile.classify(raw, state, True)

  def test_alignment_fail_is_rejected(self):
    tmp, raw, state = self._fixture()
    with tmp:
      raw.write_text(
          raw.read_text(encoding="utf-8")
          + "[CANON_ALIGN_PRE] step=9 verdict=FAIL\n",
          encoding="utf-8",
      )
      with self.assertRaisesRegex(profile.ProfileError, "alignment FAIL"):
        profile.classify(raw, state, True)


if __name__ == "__main__":
  unittest.main()
