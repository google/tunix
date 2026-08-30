from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts"
    / "classify_short_carrier_screen.py"
)
SPEC = importlib.util.spec_from_file_location("carrier_screen", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CLASSIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLASSIFIER)
SOURCE_SHA = "1" * 40
HOST = "v5p-host"


def _write_fixture(root: Path, *, clipped: bool = False) -> None:
  manifest = {
      "source_commit": SOURCE_SHA,
      "expected_hostname": HOST,
      "model_id": CLASSIFIER.EXPECTED_MODEL,
      "contract_name": "local-qwen4b-dp1-tp4-zero-admission",
      "stage": "rollout-only",
      "onehost_xprof_arm": "zero-hp",
      "onehost_seam_probe": True,
      "q4_tp4_zero_admission": True,
      "q4_tp4_seam_diagnostic": "",
      "q4_tp4_continue_kv_diagnostic": False,
      "q4_tp4_short_backward": True,
      "q4_tp4_carrier_screen": True,
      "compilation_cache_dir": "",
      "max_prompt_length": 1792,
      "max_response_length": 8192,
      "max_turns": 16,
      "generations": 16,
      "global_trajectories": 16,
      "task_image": CLASSIFIER.EXPECTED_TASK_IMAGE,
      "whitelist_sha256": CLASSIFIER.EXPECTED_WHITELIST_SHA256,
      "role_topology": {"dp": 1, "tp": 4, "devices": 4},
      "sampling_contract": {
          "temperature": 1.0,
          "top_k": 0,
          "top_p": 1.0,
          "source": "explicit-cli",
      },
  }
  (root / "run_manifest.json").write_text(json.dumps(manifest))
  (root / "probe_process_status.json").write_text(
      json.dumps({"profile": "seam", "training_process_status": 0})
  )
  (root / "raw.log").write_text(CLASSIFIER.ROLLOUT_PASS_MARKER + "\n")
  rows = []
  for pair_index in range(16):
    reward = 1.0 if pair_index == 1 else 0.0
    rows.append({
        "status": "MAX_CONTEXT_LIMIT_REACHED" if clipped else "SUCCEEDED",
        "compact_filtered": clipped,
        "complete": not clipped,
        "group_id": "0",
        "pair_index": pair_index,
        "raw_final_reward": reward,
        "trajectory": {
            "conversation_tokens": [10, 11, 12],
            "conversation_masks": [0, 0, 0] if clipped else [1, 1, 1],
            "old_logprobs": [-0.1, -0.2, -0.3],
        },
    })
  trajectory = root / "batch-000000.trajectories.jsonl.gz"
  with gzip.open(trajectory, "wt", encoding="utf-8") as stream:
    for row in rows:
      stream.write(json.dumps(row) + "\n")
  metrics = {
      "trajectory_sha256": CLASSIFIER._sha256(trajectory),
      "trajectories": 16,
      "prompt_groups": 1,
      "compact_filtered_trajectories": 16 if clipped else 0,
      "mixed_prompt_groups": 0 if clipped else 1,
      "nonzero_advantages": 0 if clipped else 16,
  }
  (root / "batch_metrics.jsonl").write_text(json.dumps(metrics) + "\n")


class ShortCarrierScreenTest(unittest.TestCase):

  def test_pass_requires_real_mixed_nonclipped_rows(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _write_fixture(root)
      result, status = CLASSIFIER.classify(
          root, source_sha=SOURCE_SHA, expected_hostname=HOST
      )
      self.assertEqual(status, 0)
      self.assertEqual(result["outcome"], "CARRIER_SCREEN_PASS")
      self.assertEqual(result["raw_rewards"].count(1.0), 1)
      self.assertEqual(result["raw_rewards"].count(0.0), 15)
      self.assertEqual(result["action_tokens"], [3] * 16)
      self.assertEqual(result["eligible_solved_rows"], [1])
      self.assertEqual(len(result["eligible_unsolved_rows"]), 15)

  def test_clipped_rows_are_inconclusive(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _write_fixture(root, clipped=True)
      result, status = CLASSIFIER.classify(
          root, source_sha=SOURCE_SHA, expected_hostname=HOST
      )
      self.assertEqual(status, 3)
      self.assertEqual(result["verdict"], "INCONCLUSIVE")
      self.assertIn("eligible_solved_rows=0", result["reasons"])
      self.assertIn("eligible_unsolved_rows=0", result["reasons"])


if __name__ == "__main__":
  unittest.main()
