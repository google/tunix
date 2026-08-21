"""Materialized-map contracts for P57 FrozenLake recipes."""

from __future__ import annotations

import copy
import unittest

from examples.frozenlake import p57_workloads


class P57WorkloadsTest(unittest.TestCase):

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


if __name__ == "__main__":
  unittest.main()
