#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / (
    "canon-zero-tim/tasks/v1-phase3-prefix-cache/scripts/"
    "compare_p3_perf_pair.py"
)
SPEC = importlib.util.spec_from_file_location("p3_apc_perf_pair", MODULE)
assert SPEC is not None and SPEC.loader is not None
pair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pair)


def _records(token_suffix: str = "") -> list[dict]:
  records = []
  for index, n_action in enumerate((10, 20, 30)):
    hashes = {
        name: f"{name}-{index}{token_suffix}"
        for name in (
            "tokens", "action_mask", "policy_version", "S_decode",
            "S_prefill", "T_old",
        )
    }
    records.append({
        "diagnostic_round": index,
        "N_action": n_action,
        "verdict": "PASS",
        "hashes": hashes,
    })
  return records


def _raw(round_totals: tuple[tuple[float, ...], ...]) -> str:
  lines = [
      "[P3.APC] performance_contract=greedy-matched-v1 "
      "temperature=0.0 max_concurrency=1"
  ]
  for index, totals in enumerate(round_totals):
    for seconds in totals:
      lines.append(
          f"[PERF] stage=rollout_generate seconds={seconds} rows=1"
      )
    lines.append(
        f"[PERF] step=0 stage=rescore_b seconds={index + 0.5} rows=4"
    )
    lines.append(
        f"[CANON_P38] PRECHECK_ROUND_COMPLETE round={index + 1}/3"
    )
  lines.append("[P3.APC] docker_exit=42 elapsed_seconds=99")
  return "\n".join(lines) + "\n"


class PerfPairTest(unittest.TestCase):

  def _fixture(self, apc_suffix: str = ""):
    temporary = tempfile.TemporaryDirectory()
    root = Path(temporary.name)
    control_raw = root / "control.raw"
    apc_raw = root / "apc.raw"
    control_report = root / "control.jsonl"
    apc_report = root / "apc.jsonl"
    control_class = root / "control.class.json"
    apc_class = root / "apc.class.json"
    control_raw.write_text(
        _raw(((10.0, 2.0), (4.0, 3.0), (6.0,))), encoding="utf-8"
    )
    apc_raw.write_text(
        _raw(((10.5, 2.0), (3.0, 2.0), (4.0,))), encoding="utf-8"
    )
    control_report.write_text(
        "".join(json.dumps(x) + "\n" for x in _records()), encoding="utf-8"
    )
    apc_report.write_text(
        "".join(json.dumps(x) + "\n" for x in _records(apc_suffix)),
        encoding="utf-8",
    )
    control_class.write_text(json.dumps({
        "status": "CONTROL_GREEN", "expect_apc": False,
    }), encoding="utf-8")
    apc_class.write_text(json.dumps({
        "status": "GB_GC_CERTIFICATION_GREEN", "expect_apc": True,
    }), encoding="utf-8")
    return temporary, (
        control_raw, control_report, control_class,
        apc_raw, apc_report, apc_class,
    )

  def test_matched_pair_reports_steady_keep(self):
    tmp, args = self._fixture()
    with tmp:
      result = pair.compare(*args)
      self.assertEqual(result["status"], "MATCHED_INPUTS")
      self.assertEqual(result["decision"], "KEEP_ONEHOST_PROXY")
      self.assertEqual(result["steady_speedup_percent"], 30.769)

  def test_token_hash_mismatch_is_rejected(self):
    tmp, args = self._fixture(apc_suffix="-different")
    with tmp, self.assertRaisesRegex(pair.PairError, "hashes differ"):
      pair.compare(*args)


if __name__ == "__main__":
  unittest.main()
