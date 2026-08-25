#!/usr/bin/env python3
"""Host positives and negatives for the M15 APC target classifier."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).with_name("classify_m15_apc_target_run.py")
SPEC = importlib.util.spec_from_file_location(
    "classify_m15_apc_target_run", MODULE_PATH
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

SOURCE = "1" * 40


def _raw(arm: str) -> str:
  enabled = 1 if arm == "on" else 0
  command = " ".join([
      "python3", "-u", "-m", "examples.frozenlake.train_frozenlake_qwen3",
      "--mesh_dp=8", "--mesh_tp=8", "--batch_size=32",
      "--mini_batch_size=32", "--num_generations=8",
      "--max_prompt_length=4096", "--max_response_length=8192",
      "--max_concurrency=256", "--vllm_max_num_seqs=32",
      "--vllm_max_num_batched_tokens=256", "--env_max_steps=15",
      "--temperature=0.7", "--top_k=0", "--top_p=1.0", "--seed=42",
      "--p57_workload_candidate=m15", "--p57_data_split=main",
  ])
  lines = [
      f"[sync] HEAD={SOURCE}",
      f"[run] cmd: {command}",
      f"[P3_APC_CONFIG] enabled={enabled} workload=frozenlake reader=train_frozenlake_qwen3",
      "[VLLM.LOGPROB_REQUEST] return_logprobs=1 sampled=1 prompt=None host_extraction=enabled",
      "[CAN" "ON_APC_M15_A_CONTRACT] prompt_logprobs=None logprobs=1 skip_reading_prefix_cache=False",
      "[CAN" "ON_APC_M15_B_CONTRACT] reset_prefix_cache=True all_num_cached_tokens_zero=True",
      f"[CAN" f"ON_APC_M15_TARGET_CONTRACT] arm={arm} topology=DP8xTP8 workload=m15/main backward=0 optimizer_commits=0",
      "[CANON_P38] CONTROLLED_EXIT code=42 backward=0 optimizer_commits=0",
  ]
  if arm == "on":
    lines.append("Prefix cache hit rate: 91.0%")
  return "\n".join(lines) + "\n"


def _record(*, ab_bytes: int, bc_bytes: int = 0) -> dict:
  return {
      "step": 0,
      "diagnostic_round": 0,
      "N_action": 100,
      "action_geometry": {
          "valid": True,
          "min_logical_kv_prefix_length": 988,
          "max_logical_kv_prefix_length": 2000,
          "rows_reaching_1686": 4,
      },
      "boundaries": {
          "S_decode_vs_S_prefill": {
              "valid": True,
              "finite": True,
              "differing_bytes": ab_bytes,
              "differing_elements": 1 if ab_bytes else 0,
          },
          "S_prefill_vs_T_old": {
              "valid": True,
              "finite": True,
              "differing_bytes": bc_bytes,
              "differing_elements": 1 if bc_bytes else 0,
          },
      },
      "hashes": {
          name: name
          for name in (
              "S_decode", "S_prefill", "T_old", "tokens", "action_mask",
              "policy_version",
          )
      },
  }


class Fixture:

  def __init__(self, *, arm: str, red: bool):
    self.holder = tempfile.TemporaryDirectory()
    self.root = Path(self.holder.name)
    self.raw = self.root / "run.log"
    self.report = self.root / "pre_alignment.jsonl"
    self.capture = self.root / "capture.json"
    self.capsule = self.root / "mismatch.npz"
    self.raw.write_text(_raw(arm))
    self.report.write_text(json.dumps(_record(ab_bytes=3 if red else 0)) + "\n")
    capsule_receipt = None
    joins = []
    if red:
      self.capsule.write_bytes(b"bounded-capsule")
      capsule_receipt = {
          "path": str(self.capsule),
          "sha256": hashlib.sha256(self.capsule.read_bytes()).hexdigest(),
          "diagnostic_round": 0,
      }
      joins = [{"source_row": 192, "request_id": "request-192"}]
    self.capture.write_text(json.dumps({
        "verdict": "PASS",
        "scope": "p38-serving-capture",
        "program_path": "standard",
        "source_commit": SOURCE,
        "request_journal_records": 4,
        "incident_ledger_records": 4,
        "records": [{"seq": 0}],
        "mismatch_capsule": capsule_receipt,
        "incident_exact_joins": joins,
        "incident_missing_joins": [],
        "prefix_bounds": [1152, 1216, 1280, 1408, 1696],
    }))
    self.arm = arm
    self.red = red

  def close(self):
    self.holder.cleanup()

  def classify(self):
    return MODULE.classify(
        raw_path=self.raw,
        report_path=self.report,
        capture_classification_path=self.capture,
        arm=self.arm,
        expected_source_commit=SOURCE,
        capsule_path=self.capsule if self.red else None,
    )


class ClassifyM15ApcTargetRunTest(unittest.TestCase):

  def _fixture(self, *, arm: str, red: bool) -> Fixture:
    fixture = Fixture(arm=arm, red=red)
    self.addCleanup(fixture.close)
    return fixture

  def test_accepts_apc_off_exact_control(self):
    result = self._fixture(arm="off", red=False).classify()
    self.assertEqual(result["status"], "CONTROL_GREEN")

  def test_accepts_representative_apc_on_exact_observation(self):
    result = self._fixture(arm="on", red=False).classify()
    self.assertEqual(result["status"], "TARGET_NOT_REPRODUCED")

  def test_accepts_replayable_apc_on_red(self):
    result = self._fixture(arm="on", red=True).classify()
    self.assertEqual(result["status"], "FRESH_TARGET_RED_FROZEN")
    self.assertEqual(result["capture"]["joined_source_rows"], [192])

  def test_rejects_red_without_exact_incident_join(self):
    fixture = self._fixture(arm="on", red=True)
    capture = json.loads(fixture.capture.read_text())
    capture["incident_exact_joins"] = []
    fixture.capture.write_text(json.dumps(capture))
    with self.assertRaisesRegex(MODULE.ClassificationError, "no exact incident"):
      fixture.classify()

  def test_rejects_red_with_missing_capsule(self):
    fixture = self._fixture(arm="on", red=True)
    fixture.capsule.unlink()
    with self.assertRaisesRegex(MODULE.ClassificationError, "capsule is absent"):
      fixture.classify()

  def test_rejects_b_c_red(self):
    fixture = self._fixture(arm="on", red=True)
    fixture.report.write_text(json.dumps(_record(ab_bytes=3, bc_bytes=1)) + "\n")
    with self.assertRaisesRegex(MODULE.ClassificationError, "B-C changed"):
      fixture.classify()

  def test_rejects_wrong_runtime_source(self):
    fixture = self._fixture(arm="off", red=False)
    fixture.raw.write_text(fixture.raw.read_text().replace(SOURCE, "2" * 40))
    with self.assertRaisesRegex(MODULE.ClassificationError, "source receipt"):
      fixture.classify()

  def test_rejects_non_cache_readable_a(self):
    fixture = self._fixture(arm="on", red=True)
    fixture.raw.write_text(
        fixture.raw.read_text().replace(
            "skip_reading_prefix_cache=False",
            "skip_reading_prefix_cache=True",
        )
    )
    with self.assertRaisesRegex(MODULE.ClassificationError, "cache-readable"):
      fixture.classify()

  def test_rejects_b_without_full_reset_receipt(self):
    fixture = self._fixture(arm="on", red=True)
    fixture.raw.write_text(
        fixture.raw.read_text().replace(
            "all_num_cached_tokens_zero=True",
            "all_num_cached_tokens_zero=False",
        )
    )
    with self.assertRaisesRegex(MODULE.ClassificationError, "B full-reset"):
      fixture.classify()

  def test_rejects_underdepth_exact_treatment(self):
    fixture = self._fixture(arm="on", red=False)
    record = _record(ab_bytes=0)
    record["action_geometry"]["max_logical_kv_prefix_length"] = 1600
    record["action_geometry"]["rows_reaching_1686"] = 0
    fixture.report.write_text(json.dumps(record) + "\n")
    with self.assertRaisesRegex(MODULE.ClassificationError, "deep band"):
      fixture.classify()

  def test_rejects_wrong_m15_geometry(self):
    fixture = self._fixture(arm="off", red=False)
    fixture.raw.write_text(fixture.raw.read_text().replace("--mesh_tp=8", "--mesh_tp=4"))
    with self.assertRaisesRegex(MODULE.ClassificationError, "command geometry"):
      fixture.classify()

  def test_rejects_optimizer_marker(self):
    fixture = self._fixture(arm="off", red=False)
    fixture.raw.write_text(fixture.raw.read_text() + "OPTIMIZER_COMMIT step=0\n")
    with self.assertRaisesRegex(MODULE.ClassificationError, "optimizer commit"):
      fixture.classify()


if __name__ == "__main__":
  unittest.main()
