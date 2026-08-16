#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "src/engine_shims/p38_tail_capture.py"
SPEC = importlib.util.spec_from_file_location("p38_tail_capture", MODULE)
assert SPEC and SPEC.loader
capture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(capture)


class TailCaptureTest(unittest.TestCase):

  def test_checkpoint_and_record_contract(self):
    self.assertEqual(len(capture.P38_TAIL_CHECKPOINTS), 6)
    self.assertEqual(capture.P38_TAIL_CHECKPOINTS[0], "raw_target_logit")
    self.assertEqual(
        capture.P38_TAIL_CHECKPOINTS[-1], "production_target_logprob"
    )
    with tempfile.TemporaryDirectory() as directory:
      state = {"records": 0, "bytes": 0}
      index, digest = capture.write_tail_record(
          directory,
          state,
          {"tail_values": np.arange(12, dtype=np.float32).reshape(2, 6)},
          {"arm": "A", "diagnostic_round": 0},
          1 << 20,
      )
      self.assertEqual(index, 0)
      record = json.loads(
          Path(directory, "p38_tail_000000.json").read_text(encoding="utf-8")
      )
      self.assertEqual(record["schema"], "p38-tail-values-v1")
      self.assertEqual(record["npz_sha256"], digest)
      with self.assertRaisesRegex(RuntimeError, "byte bound"):
        capture.write_tail_record(
            directory,
            state,
            {"tail_values": np.zeros((1 << 18, 6), dtype=np.float32)},
            {"arm": "B", "diagnostic_round": 0},
            state["bytes"] + 8,
        )


if __name__ == "__main__":
  unittest.main()
