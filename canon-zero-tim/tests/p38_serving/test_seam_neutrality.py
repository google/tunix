#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / (
    "tasks/p38-pathways-decode-prefill-carrier/scripts/"
    "classify_p38_seam_neutrality.py"
)
SPEC = importlib.util.spec_from_file_location("p38_seam_neutrality", MODULE)
assert SPEC is not None and SPEC.loader is not None
neutrality = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(neutrality)


def _row(step: int) -> dict:
  return {
      "step": step,
      "hashes": {"tokens": f"tokens-{step}", "action_mask": f"mask-{step}"},
      "masked_hashes": {
          "S_decode": f"a-{step}",
          "S_prefill": f"b-{step}",
          "T_old": f"c-{step}",
      },
      "boundaries": {
          "S_decode_vs_S_prefill": {"differing_elements": 0},
          "S_prefill_vs_T_old": {"differing_elements": 0},
      },
  }


class SeamNeutralityTest(unittest.TestCase):

  def _write(self, root: Path, name: str, rows: list[dict]) -> Path:
    path = root / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path

  def test_exact_three_rounds_pass(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      rows = [_row(index) for index in range(3)]
      off = self._write(root, "off.jsonl", rows)
      observed = self._write(root, "observed.jsonl", rows)
      result = neutrality.classify(off, observed)
      self.assertEqual(result["status"], "PASS")
      self.assertEqual(len(result["rounds"]), 3)

  def test_endpoint_drift_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      off_rows = [_row(index) for index in range(3)]
      observed_rows = [_row(index) for index in range(3)]
      observed_rows[1]["masked_hashes"]["S_decode"] = "fault"
      off = self._write(root, "off.jsonl", off_rows)
      observed = self._write(root, "observed.jsonl", observed_rows)
      with self.assertRaisesRegex(ValueError, "endpoint drift"):
        neutrality.classify(off, observed)

  def test_token_drift_and_missing_round_are_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      rows = [_row(index) for index in range(3)]
      short = self._write(root, "short.jsonl", rows[:2])
      full = self._write(root, "full.jsonl", rows)
      with self.assertRaisesRegex(ValueError, "exactly three"):
        neutrality.classify(short, full)
      changed = [_row(index) for index in range(3)]
      changed[2]["hashes"]["tokens"] = "fault"
      changed_path = self._write(root, "changed.jsonl", changed)
      with self.assertRaisesRegex(ValueError, "tokens drift"):
        neutrality.classify(full, changed_path)

  def test_non_hash_contract_drift_is_rejected(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      off_rows = [_row(index) for index in range(3)]
      observed_rows = [_row(index) for index in range(3)]
      for row in off_rows + observed_rows:
        row["N_action"] = 17
      observed_rows[1]["N_action"] = 18
      off = self._write(root, "off.jsonl", off_rows)
      observed = self._write(root, "observed.jsonl", observed_rows)
      with self.assertRaisesRegex(ValueError, "alignment contract drift"):
        neutrality.classify(off, observed)


if __name__ == "__main__":
  unittest.main()
