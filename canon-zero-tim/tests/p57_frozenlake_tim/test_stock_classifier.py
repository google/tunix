"""Decision-table tests for P57 stock stochastic rollout calibration."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "canon-zero-tim/tasks/p57-frozenlake-tim-causal-study/scripts/classify_stock_discovery.py"
SPEC = importlib.util.spec_from_file_location("p57_stock_classifier", MODULE_PATH)
assert SPEC and SPEC.loader
classifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = classifier
SPEC.loader.exec_module(classifier)


def _records(recipe, solve_rate):
  generations = 8
  records = []
  solved_groups = int(100 * solve_rate * 2)
  solved = 0
  for group in range(100):
    # Each selected solved group is mixed 4/8, so solve=.20 gives 40 mixed
    # groups and enough nonzero-advantage coverage.
    group_solved = [
        group < solved_groups and pair < 4 for pair in range(generations)
    ]
    for pair, success in enumerate(group_solved):
      solved += int(success)
      records.append({
          "group_id": group,
          "pair_index": pair,
          "policy_version": 0,
          "status": "SUCCEEDED" if success else "MAX_STEPS_REACHED",
          "reward": 1.0 if success else 0.0,
          "invalid_actions": 0,
          "ineffective_actions": 0,
          "turns": 5,
          "prompt_tokens": 1000,
          "completion_tokens": 1000,
          "assistant_tokens": 100,
          "context_tokens": 2000,
          "p57_index": group,
          "grid_side": classifier.p57_workloads.recipe(recipe).min_grid_side,
          "shortest_path": 4,
          "map_sha256": f"{group:064x}",
      })
  if solved / len(records) != solve_rate:
    raise AssertionError((solved / len(records), solve_rate))
  return records


def _receipt(*, solve_rates=None):
  solve_rates = solve_rates or {"m10": 0.18, "m15": 0.20, "m20": 0.24}
  results = {}
  for recipe in classifier._RECIPE_ORDER:
    records = _records(recipe, solve_rates[recipe])
    spec = classifier.p57_workloads.recipe(recipe)
    results[recipe] = {
        "recipe": {
            "name": spec.name,
            "min_grid_side": spec.min_grid_side,
            "max_grid_side": spec.max_grid_side,
            "max_turns": spec.max_turns,
            "context_hard_cap": spec.context_hard_cap,
            "frozen_probability": spec.frozen_probability,
            "eligible": spec.eligible,
        },
        "dataset_eval_sha256": recipe[0] * 64,
        "prompts": 100,
        "generations": 8,
        "trajectories": len(records),
        "batches": 100,
        "wall_seconds": 12.0,
        "train_steps_before": 0,
        "train_steps_after": 0,
        "records": records,
    }
  return {
      "schema": "p57-frozenlake-stock-rollout-calibration-v2",
      "arm": "mismatch",
      "inference_regime": "stock-fast",
      "zero_tim_off_attestation": classifier._ZERO_TIM_OFF_ATTESTATION,
      "rollout_weight_sync": classifier._STOCK_SYNC_RECEIPT,
      "fixed_lm_head": "0",
      "source_commit": "a" * 40,
      "mode": "stochastic",
      "temperature": 0.7,
      "generations": 8,
      "recipe_order": list(classifier._RECIPE_ORDER),
      "physical_max_prompt_length": 16384,
      "physical_max_response_length": 16384,
      "train_steps_before": 0,
      "train_steps_after": 0,
      "global_steps_before": 0,
      "global_steps_after": 0,
      "backward_calls": 0,
      "optimizer_commits": 0,
      "checkpoint_writes": 0,
      "results": results,
  }


class P57StockClassifierTest(unittest.TestCase):

  def _write(self, receipt, *, log=False):
    temporary = tempfile.TemporaryDirectory()
    self.addCleanup(temporary.cleanup)
    path = Path(temporary.name) / ("stochastic.log" if log else "stochastic.json")
    payload = json.dumps(receipt, sort_keys=True, separators=(",", ":"))
    path.write_text(
        f"noise\n{classifier._MARKER}{payload}\n" if log else payload,
        encoding="utf-8",
    )
    return path

  def test_selects_eligible_recipe_closest_to_twenty_percent(self):
    result = classifier.classify(self._write(_receipt()))
    self.assertEqual(result["verdict"], "PASS")
    self.assertEqual(result["selection"], "FREEZE_M15")
    self.assertEqual(result["selected_recipe"], "m15")
    self.assertEqual(result["eligible_recipes"], ["m10", "m15", "m20"])

  def test_tie_prefers_m15_then_m10_then_m20(self):
    receipt = _receipt(solve_rates={"m10": 0.18, "m15": 0.22, "m20": 0.24})
    result = classifier.classify(self._write(receipt))
    self.assertEqual(result["selection"], "FREEZE_M15")
    receipt = _receipt(solve_rates={"m10": 0.18, "m15": 0.24, "m20": 0.26})
    result = classifier.classify(self._write(receipt))
    self.assertEqual(result["selection"], "FREEZE_M10")

  def test_accepts_compact_log_marker(self):
    self.assertEqual(
        classifier.classify(self._write(_receipt(), log=True))["verdict"],
        "PASS",
    )

  def test_rejects_missing_trajectory_or_training_mutation(self):
    receipt = _receipt()
    receipt["results"]["m10"]["records"].pop()
    receipt["optimizer_commits"] = 1
    result = classifier.classify(self._write(receipt))
    self.assertEqual(result["verdict"], "FAIL")
    self.assertTrue(any("coverage" in reason for reason in result["reasons"]))
    self.assertTrue(any("contract drifted" in reason for reason in result["reasons"]))

  def test_rejects_canonical_or_incomplete_stock_attestation(self):
    receipt = _receipt()
    receipt["inference_regime"] = "canonical"
    result = classifier.classify(self._write(receipt))
    self.assertEqual(result["verdict"], "FAIL")

    receipt = _receipt()
    receipt["zero_tim_off_attestation"] = {
        **classifier._ZERO_TIM_OFF_ATTESTATION,
        "zero_switches": ["CANON_P38_FIXED_LM_HEAD"],
    }
    result = classifier.classify(self._write(receipt))
    self.assertEqual(result["verdict"], "FAIL")

  def test_rejects_missing_or_fabricated_stock_weight_sync(self):
    receipt = _receipt()
    del receipt["rollout_weight_sync"]
    self.assertEqual(
        classifier.classify(self._write(receipt))["verdict"], "FAIL"
    )
    receipt = _receipt()
    receipt["rollout_weight_sync"] = {
        **classifier._STOCK_SYNC_RECEIPT,
        "exact_weight_attestation": "pass",
    }
    self.assertEqual(
        classifier.classify(self._write(receipt))["verdict"], "FAIL"
    )

  def test_context_cap_excess_makes_recipe_ineligible(self):
    receipt = _receipt()
    receipt["results"]["m15"]["records"][0]["context_tokens"] = 13000
    result = classifier.classify(self._write(receipt))
    self.assertEqual(result["verdict"], "PASS")
    self.assertFalse(result["eligibility"]["m15"]["no_context_cap_excess"])

  def test_physical_prompt_or_response_cap_hit_makes_recipe_ineligible(self):
    receipt = _receipt()
    receipt["results"]["m10"]["records"][0]["prompt_tokens"] = 16384
    receipt["results"]["m10"]["records"][1]["completion_tokens"] = 16384
    result = classifier.classify(self._write(receipt))
    self.assertEqual(result["verdict"], "PASS")
    self.assertFalse(result["eligibility"]["m10"]["no_physical_cap_hit"])
    self.assertEqual(result["summaries"]["m10"]["physical_prompt_cap_hits"], 1)
    self.assertEqual(result["summaries"]["m10"]["physical_response_cap_hits"], 1)


if __name__ == "__main__":
  unittest.main()
