"""Tests for the FrozenLake DP2xTP2 one-host classifier and launch contract."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = ROOT / "canon-zero-tim/tasks/v2-frozenlake-onehost/scripts"
CLASSIFIER_PATH = SCRIPT_DIR / "classify_frozenlake_dp2tp2.py"
RUNNER_PATH = SCRIPT_DIR / "run_frozenlake_dp2tp2_onehost.sh"
INNER_PATH = SCRIPT_DIR / "run_frozenlake_dp2tp2_inner.sh"
SPEC = importlib.util.spec_from_file_location(
    "v2_frozenlake_onehost_classifier", CLASSIFIER_PATH
)
assert SPEC is not None and SPEC.loader is not None
classifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = classifier
SPEC.loader.exec_module(classifier)


def _boundary() -> dict:
  return {
      "valid": True,
      "finite": True,
      "differing_bytes": 0,
      "differing_elements": 0,
      "max_abs": 0.0,
  }


def _alignment(*, pre: bool, action_tokens: int = 5000) -> dict:
  boundaries = {"T_old_vs_T_current": _boundary()}
  if pre:
    boundaries = {
        "S_decode_vs_S_prefill": _boundary(),
        "S_prefill_vs_T_old": _boundary(),
    }
  return {
      "N_action": action_tokens,
      "verdict": "PASS",
      "blocking_reds": [],
      "reds": [],
      "boundaries": boundaries,
      "hashes": {
          "tokens": "1" * 64,
          "action_mask": "2" * 64,
          "S_decode": "3" * 64,
          "S_prefill": "4" * 64,
          "T_old": "4" * 64,
      },
  }


def _manifest(
    workload: str = "m15", arm: str = "r3", mode: str = "measure"
) -> dict:
  spec = classifier._WORKLOADS[workload]  # pylint: disable=protected-access
  return {
      "schema": "canon.v2-frozenlake-onehost.run.v1",
      "source_commit": "a" * 40,
      "source_diff_sha256": "b" * 64,
      "image_id_sha256": "c" * 64,
      "model_snapshot_sha256": "d" * 64,
      "dataset_train_sha256": "e" * 64,
      "dataset_test_sha256": "f" * 64,
      "runner_sha256": "0" * 64,
      "workload": workload,
      "workload_name": spec["name"],
      "arm": arm,
      "stage": "backward-no-commit",
      "model_id": "Qwen/Qwen3-8B",
      "model_dir_name": "qwen8b_tp2",
      "topology": {"dp": 2, "tp": 2, "devices": 4},
      "global_prompts": 4,
      "num_generations": 4,
      "global_trajectories": 16,
      "gradient_groups": 8,
      "local_m": 256,
      "global_m": 512,
      "max_prompt_length": spec["prompt"],
      "max_response_length": spec["response"],
      "max_turns": spec["max_turns"],
      "data_shuffle_seed": 42,
      "vllm_global_seed": 0,
      "vllm_hbm_utilization": spec["vllm_hbm_utilization"],
      "selectors": classifier._ARMS[arm],  # pylint: disable=protected-access
      "checked_vma": True,
      "wandb_mode": "disabled",
      "classification_mode": mode,
  }


def _update(arm: str = "r3") -> dict:
  norms = [float(index + 1) for index in range(8)]
  return {
      "contract_name": "frozenlake-m15-onehost-dp2-tp2",
      "dp_size": 2,
      "tp_size": 2,
      "global_m": 512,
      "verdict": "PASS",
      "mode": "backward-no-commit",
      "microsteps": 8,
      "commits": 0,
      "train_steps_before": 0,
      "train_steps_after": 0,
      "gradient_finite": True,
      "gradient_activity": [True] * 8,
      "micro_gradient_norms": norms,
      "dp_replicas_exact": True,
      "dp_axis": "dp",
      "dp_reduction_transactions": 1 if arm in ("r2", "r3") else 8,
      "dp_reduction_rounds_per_transaction": 2,
      "dp_rank_pullbacks_per_transaction": 2,
      "dp_pullback_invocations_per_transaction": 1,
      "model_changed_paths": [],
      "optimizer_changed_paths": [],
      "accumulator_changed_paths": [],
      "reference_changed_paths": [],
      "optimizer_placement": "pinned-host-offload",
      "optimizer_memory_kinds_before": ["pinned_host"],
      "hbm_before": [
          {"peak_bytes_in_use": 60, "bytes_limit": 100} for _ in range(4)
      ],
      "hbm_after_reverse": [
          {"peak_bytes_in_use": 80, "bytes_limit": 100} for _ in range(4)
      ],
  }


class FrozenLakeOneHostClassifierTest(unittest.TestCase):

  def _fixture(
      self, root: Path, *, arm: str = "r3", mode: str = "measure"
  ) -> Path:
    (root / "run_manifest.json").write_text(
        json.dumps(_manifest(arm=arm, mode=mode)), encoding="utf-8"
    )
    (root / "runtime.json").write_text(
        json.dumps({"verdict": "PASS"}), encoding="utf-8"
    )
    (root / "pre_alignment.jsonl").write_text(
        json.dumps(_alignment(pre=True)) + "\n", encoding="utf-8"
    )
    (root / "alignment.jsonl").write_text(
        "".join(json.dumps(_alignment(pre=False)) + "\n" for _ in range(8)),
        encoding="utf-8",
    )
    update = _update(arm)
    (root / "updates.json").write_text(json.dumps(update), encoding="utf-8")
    marker = "forward_group_done" if arm == "r0" else "forward_group_issued"
    lines = [
        f"[P32.DP2] {marker} group={index}/8 rows=(0, 8) "
        f"n_real=({3900 + index * 30}, {4000 + index * 30})"
        for index in range(1, 9)
    ]
    lines.append("[P66.VMA] outer_check_enabled program=test")
    lines.append(
        "[V2.FL.SEED] CONTRACT_PASS data_shuffle_seed=42 "
        "vllm_global_seed=0 per_request_seed=unsupported"
    )
    lines.append("[V2.FL.WANDB] DISABLED_LOCAL_PASS test")
    lines.append(
        "[V2.FL.SAMPLER] CONTRACT_PASS sampler_is=none "
        "use_rollout_logps=1 tis_weights=absent"
    )
    (root / "raw.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    anchor = root / "anchors.json"
    anchor.write_text(
        json.dumps({
            "schema": "canon.v2-frozenlake-onehost.gradient-anchors.v1",
            "anchors": {},
        }),
        encoding="utf-8",
    )
    return anchor

  def _classify(
      self, mutate=None, *, anchor=False, arm="r3", require_anchor=False
  ) -> dict:
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      anchor_path = self._fixture(
          root, arm=arm, mode="certify" if require_anchor else "measure"
      )
      if mutate is not None:
        mutate(root)
      if anchor:
        registry = json.loads(anchor_path.read_text(encoding="utf-8"))
        update = json.loads((root / "updates.json").read_text(encoding="utf-8"))
        registry["anchors"][f"m15:{arm}"] = {
            "run_id": "measurement-run-r1",
            "micro_gradient_norms": update["micro_gradient_norms"],
        }
        anchor_path.write_text(json.dumps(registry), encoding="utf-8")
      return classifier.classify(
          root,
          workload="m15",
          arm=arm,
          docker_exit=0,
          anchor_registry=anchor_path,
          require_anchor=require_anchor,
      )

  def test_clean_unregistered_run_is_measurement_only(self):
    result = self._classify()
    self.assertEqual(result["verdict"], "MEASUREMENT_ONLY")
    self.assertEqual(result["landed_shape"]["max_n_real"], 4240)
    self.assertFalse(result["landed_shape"]["cap_coverage"])
    self.assertEqual(result["receipts"]["p66_outer_check_enabled"], 1)

  def test_registered_bitwise_anchor_promotes_run(self):
    result = self._classify(anchor=True)
    self.assertEqual(result["verdict"], "PASS")
    self.assertTrue(result["gradient"]["anchor_exact"])

  def test_certification_refuses_an_unregistered_anchor(self):
    result = self._classify(require_anchor=True)
    self.assertEqual(result["verdict"], "FAIL")
    self.assertIn("gradient_anchor_unregistered", result["reasons"])

  def test_one_ulp_anchor_drift_rejects(self):
    def mutate(root: Path) -> None:
      update = json.loads((root / "updates.json").read_text(encoding="utf-8"))
      update["micro_gradient_norms"][3] += 0.000001
      (root / "updates.json").write_text(json.dumps(update), encoding="utf-8")

    # Register the original expected values after restoring them explicitly.
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      anchor_path = self._fixture(root)
      original = _update()["micro_gradient_norms"]
      registry = json.loads(anchor_path.read_text(encoding="utf-8"))
      registry["anchors"]["m15:r3"] = {
          "run_id": "measurement-run-r1",
          "micro_gradient_norms": original,
      }
      anchor_path.write_text(json.dumps(registry), encoding="utf-8")
      mutate(root)
      result = classifier.classify(
          root,
          workload="m15",
          arm="r3",
          docker_exit=0,
          anchor_registry=anchor_path,
          require_anchor=False,
      )
    self.assertEqual(result["verdict"], "FAIL")
    self.assertIn("gradient_anchor_bitwise", result["reasons"])

  def test_alignment_hbm_vma_length_and_reduction_negatives_fire(self):
    def wrong_vllm_hbm_utilization(root: Path) -> None:
      manifest = json.loads(
          (root / "run_manifest.json").read_text(encoding="utf-8")
      )
      manifest["vllm_hbm_utilization"] = 0.20
      (root / "run_manifest.json").write_text(
          json.dumps(manifest), encoding="utf-8"
      )

    def bad_alignment(root: Path) -> None:
      row = _alignment(pre=True)
      row["boundaries"]["S_prefill_vs_T_old"]["differing_bytes"] = 4
      (root / "pre_alignment.jsonl").write_text(
          json.dumps(row) + "\n", encoding="utf-8"
      )

    def bad_hbm(root: Path) -> None:
      update = _update()
      update["hbm_after_reverse"][0]["peak_bytes_in_use"] = 91
      (root / "updates.json").write_text(json.dumps(update), encoding="utf-8")

    def no_vma(root: Path) -> None:
      raw = (root / "raw.log").read_text(encoding="utf-8")
      (root / "raw.log").write_text(
          raw.replace("[P66.VMA] outer_check_enabled program=test\n", ""),
          encoding="utf-8",
      )

    def no_seed_receipt(root: Path) -> None:
      raw = (root / "raw.log").read_text(encoding="utf-8")
      (root / "raw.log").write_text(
          raw.replace(
              "[V2.FL.SEED] CONTRACT_PASS data_shuffle_seed=42 "
              "vllm_global_seed=0 per_request_seed=unsupported\n",
              "",
          ),
          encoding="utf-8",
      )

    def no_token_hash(root: Path) -> None:
      row = _alignment(pre=True)
      del row["hashes"]["tokens"]
      (root / "pre_alignment.jsonl").write_text(
          json.dumps(row) + "\n", encoding="utf-8"
      )

    def no_disabled_wandb_receipt(root: Path) -> None:
      raw = (root / "raw.log").read_text(encoding="utf-8")
      (root / "raw.log").write_text(
          raw.replace("[V2.FL.WANDB] DISABLED_LOCAL_PASS test\n", ""),
          encoding="utf-8",
      )

    def no_sampler_receipt(root: Path) -> None:
      raw = (root / "raw.log").read_text(encoding="utf-8")
      (root / "raw.log").write_text(
          raw.replace(
              "[V2.FL.SAMPLER] CONTRACT_PASS sampler_is=none "
              "use_rollout_logps=1 tis_weights=absent\n",
              "",
          ),
          encoding="utf-8",
      )

    def too_short(root: Path) -> None:
      raw = (root / "raw.log").read_text(encoding="utf-8")
      raw = "\n".join(
          reline if "n_real=" not in reline else reline.split("n_real=")[0] + "n_real=(1000, 2000)"
          for reline in raw.splitlines()
      ) + "\n"
      (root / "raw.log").write_text(raw, encoding="utf-8")

    def wrong_reduction(root: Path) -> None:
      update = _update()
      update["dp_reduction_transactions"] = 8
      (root / "updates.json").write_text(json.dumps(update), encoding="utf-8")

    for name, mutation, reason in (
        ("vllm_hbm", wrong_vllm_hbm_utilization, "manifest:"),
        ("alignment", bad_alignment, "pre_alignment_not_exact"),
        ("hbm", bad_hbm, "hbm_headroom_ratio="),
        ("vma", no_vma, "p66_outer_check_receipts=0"),
        ("seed", no_seed_receipt, "seed_receipts=0"),
        ("wandb", no_disabled_wandb_receipt, "disabled_wandb_receipts=0"),
        ("sampler", no_sampler_receipt, "sampler_receipts=0"),
        ("hash", no_token_hash, "input_hash_inventory"),
        ("length", too_short, "landed_n_real_max=2000"),
        ("reduction", wrong_reduction, "update:"),
    ):
      with self.subTest(name=name):
        result = self._classify(mutation)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertTrue(any(item.startswith(reason) for item in result["reasons"]))

  def test_runner_is_onehost_only_clean_and_observe_only(self):
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    inner = INNER_PATH.read_text(encoding="utf-8")
    self.assertIn("idle_seconds", runner)
    self.assertIn('if [ "$idle_seconds" -ge 120 ]', runner)
    self.assertIn("acceptance launches require a clean", runner)
    self.assertIn("stopping_own=$container", runner)
    self.assertNotIn("kubectl", runner + inner)
    self.assertNotIn("gcloud", runner + inner)
    self.assertNotIn("WANDB_API_KEY", runner + inner)
    self.assertIn("--model qwen8b_tp2", runner)
    self.assertIn("CANON_P66_P59_CHECK_VMA", inner)
    self.assertIn('"backward-no-commit"', inner)
    self.assertIn("measure|certify", runner)
    self.assertIn("--require-anchor", runner)


if __name__ == "__main__":
  unittest.main()
