"""CPU contracts for the P43 Qwen3-8B DeepSWE debug workload."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "p43_deepswe_contract", ROOT / "tunix/rl/deepswe_contract.py"
)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import DeepSWE contract")
contract = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = contract
SPEC.loader.exec_module(contract)


class P43ContractTest(unittest.TestCase):

  def test_debug_geometry_and_rank_groups(self):
    workload = contract.P43_DEBUG_WORKLOAD
    workload.validate()
    self.assertEqual(workload.model_id, "Qwen/Qwen3-8B")
    self.assertEqual(workload.global_trajectories, 16)
    self.assertEqual(workload.local_trajectories, 4)
    self.assertEqual((workload.dp_size, workload.tp_size), (4, 8))
    self.assertEqual(workload.global_m, 1024)
    self.assertEqual(len(workload.rank_major_rows()), 4)
    self.assertTrue(
        all(len(group) == 4 for group in workload.rank_major_rows())
    )

  def test_debug_and_pilot_are_mutually_exclusive(self):
    with self.assertRaisesRegex(ValueError, "mutually exclusive"):
      contract.active_workload({
          "CANON_P39_64CHIP_PILOT": "1",
          "CANON_P43_DEEPSWE_DEBUG": "1",
      })

  def test_debug_stage_contract(self):
    common = {
        "CANON_P39_64CHIP_PILOT": "0",
        "CANON_P43_DEEPSWE_DEBUG": "1",
    }
    for stage, no_commit, steps in (
        ("rollout-only", "1", 1),
        ("one-update", "0", 1),
        ("three-update", "0", 3),
    ):
      values = {
          **common,
          "CANON_P34_RUN_STAGE": stage,
          "CANON_P34_NO_COMMIT": no_commit,
      }
      self.assertEqual(contract.requested_max_steps(values), steps)
    with self.assertRaisesRegex(ValueError, "P43 64-chip debug"):
      contract.requested_max_steps({
          **common,
          "CANON_P34_RUN_STAGE": "full",
          "CANON_P34_NO_COMMIT": "0",
      })

  def test_p34_and_p39_defaults_are_unchanged(self):
    self.assertIs(contract.active_workload({}), contract.P34_WORKLOAD)
    self.assertIs(
        contract.active_workload({"CANON_P39_64CHIP_PILOT": "1"}),
        contract.P39_PILOT_WORKLOAD,
    )
    with self.assertRaisesRegex(ValueError, "only for P43"):
      contract.requested_max_steps({
          "CANON_P34_RUN_STAGE": "rollout-only",
          "CANON_P34_NO_COMMIT": "1",
      })


if __name__ == "__main__":
  unittest.main()
