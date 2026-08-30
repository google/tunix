from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = (
    ROOT / "canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts"
)
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "trajectory_replay_classifier", SCRIPTS / "classify_trajectory_replay.py"
)
assert SPEC is not None and SPEC.loader is not None
CLASSIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLASSIFIER)
SOURCE_SHA = "1" * 40
HOST = "v5p-host"


def _exact_boundary(n_action: int) -> dict:
  return {
      "valid": True,
      "finite": True,
      "differing_elements": 0,
      "differing_bytes": 0,
      "total_elements": n_action,
      "element_fraction": 0.0,
      "byte_fraction": 0.0,
      "max_abs": 0.0,
      "first_mismatch": None,
      "mismatches": [],
      "mismatches_truncated": False,
  }


def _write_fixture(root: Path) -> None:
  manifest = {
      "source_commit": SOURCE_SHA,
      "expected_hostname": HOST,
      "model_id": "Qwen/Qwen3-4B-Instruct-2507",
      "contract_name": "local-qwen4b-dp1-tp4-zero-admission",
      "role_topology": {"dp": 1, "tp": 4, "devices": 4},
      "onehost_seam_probe": True,
      "onehost_xprof_arm": "zero-hp",
      "stage": "backward-no-commit",
      "q4_tp4_zero_admission": True,
      "q4_tp4_seam_diagnostic": "",
      "q4_tp4_continue_kv_diagnostic": False,
      "q4_tp4_short_backward": True,
      "q4_tp4_trajectory_replay": True,
      "replay_journal_sha256": CLASSIFIER.EXPECTED_JOURNAL_SHA256,
      "alignment_precheck_only": False,
      "alignment_controlled_exit": False,
      "continue_decode_steps": "8",
      "sampling_contract": {
          "source": "explicit-cli",
          "temperature": 1.0,
          "top_k": 0,
          "top_p": 1.0,
      },
      "compilation_cache_dir": (
          "/mnt/disks/tunix-data/jax-compilation-cache/"
          "p58-q4-tp4-systemopt-b2g2-k2560"
      ),
      "max_prompt_length": 2048,
      "max_response_length": 512,
      "max_turns": 16,
      "task_image": CLASSIFIER.EXPECTED_TASK_IMAGES[0],
      "task_images": list(CLASSIFIER.EXPECTED_TASK_IMAGES),
      "task_image_id": "not-applicable-recorded-trajectory-replay",
      "whitelist_sha256": (
          "26e06ab7469987b4bc0c66d683e8468c"
          "2f10ae7d6842b0e138e563adcf87e257"
      ),
      "global_prompts": 2,
      "global_trajectories": 4,
      "system_optimization": {
          "carrier": "P28+P30+P71-fwd",
          "p59_rank_parallel_backward": False,
          "p59_reason": "DP1 one-host cannot execute rank-parallel backward",
          "p28_segmented_forward": True,
          "p28_segmented_train": True,
          "p30_sparse_grad_assembly": True,
          "p30_reuse_segmented_engine": True,
          "p71_scan": "fwd",
      },
  }
  (root / "run_manifest.json").write_text(json.dumps(manifest))
  (root / "probe_process_status.json").write_text(json.dumps({
      "profile": "seam",
      "training_process_status": 0,
  }))
  (root / "raw.log").write_text(
      "[P58.23.SYSTEM_OPT] PASS carrier=P28+P30+P71-fwd\n"
      "[P58.23.REPLAY] LOAD_PASS groups=2 generations=2 trajectories=4\n"
      "[P58.23.REPLAY] SAMPLING_PROVENANCE_PASS temperature=1.0 "
      "top_p=1.0 top_k=0\n"
      "[P58.23.REPLAY] PRODUCER_BYPASS verdict=PASS environment=0 "
      "rollout_decode=0 rescore_b=1 trainer_old=1\n"
      "[P58.23.REPLAY] ADVANTAGE_PASS groups=2 generations=2 "
      "values=[1.0, -1.0, 1.0, -1.0] injected=0\n"
      "[P58.23.REPLAY] POST_BACKWARD_BATCH_PASS trajectories=4 "
      "microsteps=2 N_action=1254\n"
  )
  expected = (
      (0, 0, 0, 432, 363, 1.0),
      (0, 1, 1, 333, 264, 0.0),
      (1, 2, 0, 432, 363, 1.0),
      (1, 3, 1, 333, 264, 0.0),
  )
  rows = []
  provenance_rows = []
  for group_id, source_row, pair_index, length, actions, reward in expected:
    masks = [1] * actions + [0] * (length - actions)
    replay = {
        "source_group_id": group_id,
        "source_row": source_row,
        "source_pair_index": pair_index,
        "prefix_length": length,
        "prefix_action_tokens": actions,
    }
    rows.append({
        "schema": "canon.local.deepswe.trajectory.v1",
        "status": "SUCCEEDED",
        "compact_filtered": False,
        "complete": True,
        "group_id": str(group_id),
        "pair_index": pair_index,
        "raw_final_reward": reward,
        "task_identity": {
            "docker_image": CLASSIFIER.EXPECTED_TASK_IMAGES[group_id]
        },
        "trajectory": {
            "prompt_length": 1745,
            "conversation_tokens": [10] * length,
            "conversation_masks": masks,
            "old_logprobs": [-0.25] * length,
            "replay_provenance": replay,
        },
    })
    provenance_rows.append({
        **replay,
        "source_completion_length": length + 1,
        "terminal_reward": reward,
        "prompt_tokens_sha256": "2" * 64,
        "prefix_tokens_sha256": "3" * 64,
        "prefix_action_mask_sha256": "4" * 64,
        "prefix_old_logprobs_sha256": "5" * 64,
    })
  trajectory = root / "batch-000000.trajectories.jsonl.gz"
  with gzip.open(trajectory, "wt", encoding="utf-8") as output:
    for row in rows:
      output.write(json.dumps(row) + "\n")
  metrics = {
      "trajectory_sha256": CLASSIFIER.base._sha256(trajectory),
      "trajectory_path": str(trajectory),
      "trajectories": 4,
      "complete_trajectories": 4,
      "compact_filtered_trajectories": 0,
      "solved_trajectories": 2,
      "prompt_groups": 2,
      "mixed_prompt_groups": 2,
      "effective_prompt_groups": 2,
      "nonzero_advantages": 4,
      "groups": [
          {"raw_rewards": [1.0, 0.0], "category": "mixed"},
          {"raw_rewards": [1.0, 0.0], "category": "mixed"},
      ],
  }
  (root / "batch_metrics.jsonl").write_text(json.dumps(metrics) + "\n")
  n_action = 1254
  boundaries = {
      "S_decode_vs_S_prefill": _exact_boundary(n_action),
      "S_prefill_vs_T_old": _exact_boundary(n_action),
  }
  (root / "pre_alignment.jsonl").write_text(json.dumps({
      "N_action": n_action,
      "boundaries": boundaries,
  }) + "\n")
  (root / "alignment.jsonl").write_text(json.dumps({
      "N_action": n_action,
      "boundaries": boundaries,
  }) + "\n")
  (root / "backward_no_commit.json").write_text(json.dumps({
      "verdict": "PASS",
      "commits": 0,
      "gradient_finite": True,
      "gradient_nonzero": True,
      "gradient_repeat_exact": True,
      "repeat_count": 2,
      "xprof_arm": "zero-hp",
      "model_changed_paths": [],
      "optimizer_changed_paths": [],
      "accumulator_changed_paths": [],
      "reference_changed_paths": [],
      "train_steps_before": 0,
      "train_steps_after": 0,
      "work_hashes": {"actor_update_calls": 2},
  }))
  (root / "replay_provenance.json").write_text(json.dumps({
      "schema": "canon.p58.recorded-trajectory-replay.v1",
      "evidence_kind": "recorded-trajectory-prefix-backward-diagnostic",
      "journal_sha256": CLASSIFIER.EXPECTED_JOURNAL_SHA256,
      "source_manifest_sha256": CLASSIFIER.EXPECTED_SOURCE_MANIFEST_SHA256,
      "source_model_id": "Qwen/Qwen3-4B-Instruct-2507",
      "source_sampling_contract": {
          "temperature": 1.0,
          "top_p": 1.0,
          "top_k": 0,
      },
      "source_sampling_identity": (
          "p58s22lr3_20260829t2256z@"
          "16c224aa80eb6b3a544be19f693c0542ab4b0dcb:"
          "rows7,0x2:B2G2"
      ),
      "prompt_identity": "same-strict-exact-real-prompt-repeated-twice",
      "environment_calls": 0,
      "rollout_decode_calls": 0,
      "rows": provenance_rows,
  }))


class TrajectoryReplayClassifierTest(unittest.TestCase):

  def test_passes_only_exact_mixed_no_commit_replay(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _write_fixture(root)
      report = CLASSIFIER.classify(
          root, source_sha=SOURCE_SHA, expected_hostname=HOST
      )
      self.assertEqual(report["verdict"], "PASS")
      self.assertEqual(
          report["outcome"],
          "ZERO_TIM_RECORDED_TRAJECTORY_BACKWARD_NO_COMMIT_PASS",
      )
      self.assertIn("not a fresh rollout", report["claim"])

  def test_rejects_reward_or_environment_drift(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      _write_fixture(root)
      metrics_path = root / "batch_metrics.jsonl"
      metrics = json.loads(metrics_path.read_text())
      metrics["groups"][0]["raw_rewards"] = [0.0, 0.0]
      metrics_path.write_text(json.dumps(metrics) + "\n")
      with self.assertRaisesRegex(ValueError, "mixed reward"):
        CLASSIFIER.classify(
            root, source_sha=SOURCE_SHA, expected_hostname=HOST
        )

      _write_fixture(root)
      with (root / "raw.log").open("a") as output:
        output.write("[SWEEnv group=0] creating RepoEnv\n")
      with self.assertRaisesRegex(ValueError, "invoked environment"):
        CLASSIFIER.classify(
            root, source_sha=SOURCE_SHA, expected_hostname=HOST
        )


if __name__ == "__main__":
  unittest.main()
