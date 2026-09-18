#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("classify_m15_e0v_onehost_arm.py")
SPEC = importlib.util.spec_from_file_location("classify_m15_e0v_onehost_arm", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
classifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = classifier
SPEC.loader.exec_module(classifier)


def _record(round_index: int, *, a_b: int = 0, b_c: int = 0) -> dict:
  hashes = {
      key: key + "-sha"
      for key in ("S_decode", "S_prefill", "T_old", "tokens", "action_mask", "policy_version")
  }
  masked = {key: key + "-masked-sha" for key in ("S_decode", "S_prefill", "T_old")}
  return {
      "verdict": "PASS",
      "blocking_reds": [],
      "diagnostic_round": round_index,
      "N_action": 12,
      "boundaries": {
          "S_decode_vs_S_prefill": {
              "valid": True,
              "finite": True,
              "differing_bytes": a_b,
          },
          "S_prefill_vs_T_old": {
              "valid": True,
              "finite": True,
              "differing_bytes": b_c,
          },
      },
      "hashes": hashes,
      "masked_hashes": masked,
  }


class OnehostArmClassifierTest(unittest.TestCase):

  def _classify(
      self,
      arm: str,
      *,
      a_b: tuple[int, int, int] = (0, 0, 0),
      b_c: tuple[int, int, int] = (0, 0, 0),
      cache_hit: bool = True,
  ) -> dict:
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      enabled = int(arm == "on")
      raw = root / "raw.log"
      lines = [
          f"[P3_APC_CONFIG] enabled={enabled} workload=frozenlake "
          "reader=train_frozenlake_qwen3"
      ]
      if arm == "on" and cache_hit:
        lines.append("Prefix cache hit rate: 80.0%")
      for index in range(1, 4):
        lines.extend((
            "[CANON_APC_M15_B_CONTRACT] reset_prefix_cache=True "
            "all_num_cached_tokens_zero=True",
            "[CANON_P38] PRECHECK_ROUND_COMPLETE "
            f"round={index}/3 backward=0 optimizer_commits=0",
        ))
      raw.write_text("\n".join(lines) + "\n", encoding="utf-8")
      report = root / "pre_alignment.jsonl"
      report.write_text(
          "".join(
              json.dumps(_record(index, a_b=a_b[index], b_c=b_c[index])) + "\n"
              for index in range(3)
          ),
          encoding="utf-8",
      )
      return classifier.classify(raw_path=raw, report_path=report, arm=arm)

  def test_control_exact_passes(self):
    self.assertEqual(self._classify("off")["status"], "CONTROL_GREEN")

  def test_treatment_exact_is_a_complete_outcome(self):
    self.assertEqual(self._classify("on")["status"], "TREATMENT_EXACT")

  def test_treatment_red_is_preserved_as_a_complete_outcome(self):
    report = self._classify("on", a_b=(0, 7, 0))
    self.assertEqual(report["status"], "TREATMENT_RED")
    self.assertFalse(report["first_red_localized"])

  def test_control_red_fails_closed(self):
    with self.assertRaisesRegex(classifier.OnehostArmError, "control A-B"):
      self._classify("off", a_b=(1, 0, 0))

  def test_b_c_red_or_missing_cache_hit_fails_closed(self):
    with self.assertRaisesRegex(classifier.OnehostArmError, "B-C"):
      self._classify("on", b_c=(0, 1, 0))
    with self.assertRaisesRegex(classifier.OnehostArmError, "no cache hit"):
      self._classify("on", cache_hit=False)


if __name__ == "__main__":
  unittest.main()
