"""Materialized-map contracts for P57 FrozenLake recipes."""

from __future__ import annotations

import ast
import copy
from pathlib import Path
import unittest

from examples.frozenlake import p57_workloads


class P57WorkloadsTest(unittest.TestCase):

  def test_generation_contract_is_shared_with_real_workload_entrypoint(self):
    self.assertEqual(p57_workloads.GENERATIONS_PER_PROMPT, 8)
    entrypoint = (
        Path(__file__).resolve().parents[3]
        / "examples/frozenlake/train_frozenlake_qwen3.py"
    ).read_text()
    self.assertIn(
        "p57_workloads.GENERATIONS_PER_PROMPT if CANON_P57_RUN_KIND else 8",
        entrypoint,
    )
    self.assertNotIn(
        "expected_generations = 2 if CANON_P57_EVALUATION else 8",
        entrypoint,
    )
    self.assertIn(
        'next_action = "complete" if completed_step == MAX_STEPS else "isolated-eval"',
        entrypoint,
    )

  def test_primary_entrypoint_registers_constant_learning_rate_receipt(self):
    entrypoint_path = (
        Path(__file__).resolve().parents[3]
        / "examples/frozenlake/train_frozenlake_qwen3.py"
    )
    tree = ast.parse(entrypoint_path.read_text(), filename=str(entrypoint_path))

    registrations = []
    scalar_adamw_rates = []
    for node in ast.walk(tree):
      if not isinstance(node, ast.Call):
        continue
      if (
          isinstance(node.func, ast.Attribute)
          and node.func.attr == "register_learning_rate_schedule"
      ):
        registrations.append(node)
      if (
          isinstance(node.func, ast.Attribute)
          and isinstance(node.func.value, ast.Name)
          and node.func.value.id == "optax"
          and node.func.attr == "adamw"
      ):
        scalar_adamw_rates.extend(
            keyword.value
            for keyword in node.keywords
            if keyword.arg == "learning_rate"
        )

    self.assertEqual(len(registrations), 1)
    registration = registrations[0]
    self.assertEqual(len(registration.args), 1)
    schedule = registration.args[0]
    self.assertIsInstance(schedule, ast.Call)
    self.assertIsInstance(schedule.func, ast.Attribute)
    self.assertIsInstance(schedule.func.value, ast.Name)
    self.assertEqual(schedule.func.value.id, "optax")
    self.assertEqual(schedule.func.attr, "constant_schedule")
    self.assertEqual(len(schedule.args), 1)
    self.assertIsInstance(schedule.args[0], ast.Name)
    self.assertEqual(schedule.args[0].id, "LEARNING_RATE")

    self.assertEqual(len(scalar_adamw_rates), 1)
    self.assertIsInstance(scalar_adamw_rates[0], ast.Name)
    self.assertEqual(scalar_adamw_rates[0].id, "LEARNING_RATE")

  def test_registered_recipe_table(self):
    self.assertEqual(tuple(p57_workloads.RECIPES), ("l0", "m10", "m15", "m20"))
    m10 = p57_workloads.recipe("m10")
    self.assertEqual(m10.grid_sides(), (5, 6, 7, 8, 9, 10))
    self.assertEqual((m10.max_turns, m10.context_hard_cap), (10, 8192))
    self.assertFalse(p57_workloads.recipe("l0").eligible)

  def test_recipes_materialize_deterministically_and_balanced(self):
    for name, spec in p57_workloads.RECIPES.items():
      with self.subTest(recipe=name):
        first = p57_workloads.materialize_records(
            name, "calibration", "eval", 100
        )
        second = p57_workloads.materialize_records(
            name, "calibration", "eval", 100
        )
        self.assertEqual(first, second)
        counts = {side: 0 for side in spec.grid_sides()}
        for row in first:
          counts[row["size"]] += 1
          minimum, maximum = spec.path_envelope(row["size"])
          self.assertLessEqual(minimum, row["shortest_path"])
          self.assertLessEqual(row["shortest_path"], maximum)
          self.assertFalse(row["is_slippery"])
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertEqual(
            p57_workloads.attest_records(
                first, name, "calibration", "eval", expected_count=100
            ),
            p57_workloads.attest_records(
                second, name, "calibration", "eval", expected_count=100
            ),
        )

  def test_calibration_selection_and_main_are_disjoint(self):
    records = {
        split: p57_workloads.materialize_dataset_pair(
            "m15", split, train_count=100, eval_count=20
        )
        for split in ("calibration", "selection", "main")
    }
    for field in ("seed", "map_sha256"):
      values = {
          split: {row[field] for role in pair for row in role}
          for split, pair in records.items()
      }
      self.assertFalse(values["calibration"] & values["main"])
      self.assertFalse(values["calibration"] & values["selection"])
      self.assertFalse(values["selection"] & values["main"])

  def test_attestation_rejects_map_mutation(self):
    rows = p57_workloads.materialize_records(
        "m10", "calibration", "eval", 2
    )
    corrupted = copy.deepcopy(rows)
    corrupted[0]["map_sha256"] = "0" * 64
    with self.assertRaisesRegex(ValueError, "row contract drifted"):
      p57_workloads.attest_records(
          corrupted, "m10", "calibration", "eval", expected_count=2
      )

  def test_primary_dataset_hashes_are_frozen(self):
    p45_train = p57_workloads.materialize_p45_records("train", 10_000)
    p45_eval = p57_workloads.materialize_p45_records("eval", 100)
    self.assertEqual(
        p57_workloads.attest_p45_records(
            p45_train, "train", expected_count=10_000
        ),
        p57_workloads.PRIMARY_DATASET_SHA256[
            ("p45", "legacy", "train", 10_000)
        ],
    )
    self.assertEqual(
        p57_workloads.attest_p45_records(
            p45_eval, "eval", expected_count=100
        ),
        p57_workloads.PRIMARY_DATASET_SHA256[
            ("p45", "legacy", "eval", 100)
        ],
    )
    m15_train, m15_eval = p57_workloads.materialize_dataset_pair(
        "m15", "main", train_count=10_000, eval_count=100
    )
    self.assertEqual(
        p57_workloads.attest_records(
            m15_train, "m15", "main", "train", expected_count=10_000
        ),
        p57_workloads.PRIMARY_DATASET_SHA256[
            ("m15", "main", "train", 10_000)
        ],
    )
    self.assertEqual(
        p57_workloads.attest_records(
            m15_eval, "m15", "main", "eval", expected_count=100
        ),
        p57_workloads.PRIMARY_DATASET_SHA256[
            ("m15", "main", "eval", 100)
        ],
    )

  def test_p45_attestation_rejects_row_mutation(self):
    rows = p57_workloads.materialize_p45_records("eval", 100)
    corrupted = copy.deepcopy(rows)
    corrupted[0]["seed"] += 1
    with self.assertRaisesRegex(ValueError, "P45 dataset row drifted"):
      p57_workloads.attest_p45_records(
          corrupted, "eval", expected_count=100
      )


if __name__ == "__main__":
  unittest.main()
