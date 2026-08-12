"""CPU contracts for the dual-topology Qwen3-4B DeepSWE debug recipe."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "p44_deepswe_contract", ROOT / "tunix/rl/deepswe_contract.py"
)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import DeepSWE contract")
contract = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = contract
SPEC.loader.exec_module(contract)


class P44ContractTest(unittest.TestCase):

  def test_topologies_share_one_normalized_recipe(self):
    small = contract.P44_PARITY_64_WORKLOAD
    large = contract.P44_PARITY_256_WORKLOAD
    small.validate()
    large.validate()
    self.assertEqual(
        contract.p44_recipe_signature(small),
        contract.p44_recipe_signature(large),
    )
    self.assertEqual(small.model_id, "Qwen/Qwen3-4B")
    self.assertEqual(large.model_id, "Qwen/Qwen3-4B")
    self.assertEqual(small.global_trajectories, 16)
    self.assertEqual(large.global_trajectories, 16)
    self.assertEqual((small.dp_size, small.tp_size), (4, 8))
    self.assertEqual((large.dp_size, large.tp_size), (16, 8))
    self.assertEqual(small.local_trajectories, 4)
    self.assertEqual(large.local_trajectories, 1)
    self.assertEqual(small.global_m, 1024)
    self.assertEqual(large.global_m, 4096)

  def test_active_workload_requires_explicit_topology(self):
    common = {"CANON_P44_DEEPSWE_PARITY": "1"}
    self.assertIs(
        contract.active_workload({**common, "CANON_P44_TOPOLOGY": "64"}),
        contract.P44_PARITY_64_WORKLOAD,
    )
    self.assertIs(
        contract.active_workload({**common, "CANON_P44_TOPOLOGY": "256"}),
        contract.P44_PARITY_256_WORKLOAD,
    )
    with self.assertRaisesRegex(ValueError, "exactly 64 or 256"):
      contract.active_workload(common)

  def test_p39_p43_p44_are_mutually_exclusive(self):
    with self.assertRaisesRegex(ValueError, "mutually exclusive"):
      contract.active_workload({
          "CANON_P43_DEEPSWE_DEBUG": "1",
          "CANON_P44_DEEPSWE_PARITY": "1",
          "CANON_P44_TOPOLOGY": "64",
      })

  def test_both_topologies_share_the_bounded_stage_ladder(self):
    for topology in ("64", "256"):
      common = {
          "CANON_P44_DEEPSWE_PARITY": "1",
          "CANON_P44_TOPOLOGY": topology,
      }
      for stage, no_commit, steps in (
          ("rollout-only", "1", 1),
          ("one-update", "0", 1),
          ("three-update", "0", 3),
      ):
        with self.subTest(topology=topology, stage=stage):
          self.assertEqual(contract.requested_max_steps({
              **common,
              "CANON_P34_RUN_STAGE": stage,
              "CANON_P34_NO_COMMIT": no_commit,
          }), steps)
      with self.assertRaisesRegex(ValueError, "P44 Qwen3-4B parity"):
        contract.requested_max_steps({
            **common,
            "CANON_P34_RUN_STAGE": "full",
            "CANON_P34_NO_COMMIT": "0",
        })

  def test_existing_contract_defaults_are_unchanged(self):
    self.assertIs(contract.active_workload({}), contract.P34_WORKLOAD)
    self.assertIs(
        contract.active_workload({"CANON_P39_64CHIP_PILOT": "1"}),
        contract.P39_PILOT_WORKLOAD,
    )
    self.assertIs(
        contract.active_workload({"CANON_P43_DEEPSWE_DEBUG": "1"}),
        contract.P43_DEBUG_WORKLOAD,
    )


if __name__ == "__main__":
  unittest.main()
