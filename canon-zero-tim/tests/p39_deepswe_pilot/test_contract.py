"""CPU contracts for the bounded P39 DeepSWE pilot."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest


ROOT = Path(__file__).resolve().parents[3]


def _load(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


contract = _load("p39_deepswe_contract", ROOT / "tunix/rl/deepswe_contract.py")


def _devices(cross_host=False):
  devices = []
  for x in range(4):
    for y in range(4):
      for z in range(4):
        process = y * 2 + z // 2 if cross_host else (x // 2) * 8 + y * 2 + z // 2
        devices.append(types.SimpleNamespace(
            id=x * 16 + y * 4 + z,
            coords=(x, y, z),
            process_index=process,
        ))
  return devices


class P39ContractTest(unittest.TestCase):

  def test_pilot_geometry_and_rank_groups(self):
    workload = contract.P39_PILOT_WORKLOAD
    workload.validate()
    self.assertEqual(workload.global_trajectories, 64)
    self.assertEqual(workload.local_trajectories, 16)
    self.assertEqual(workload.global_m, 1024)
    self.assertEqual(len(workload.rank_major_rows()), 16)
    self.assertTrue(all(len(group) == 4 for group in workload.rank_major_rows()))

  def test_host_complete_role_split(self):
    rollout, trainer, report = contract.split_4x4x4_role_devices(_devices())
    self.assertEqual((len(rollout), len(trainer)), (32, 32))
    self.assertTrue(report["disjoint"])
    self.assertTrue(report["exhaustive"])
    self.assertTrue(report["host_complete"])

  def test_cross_host_role_split_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "crosses host boundaries"):
      contract.split_4x4x4_role_devices(_devices(cross_host=True))

  def test_pilot_rejects_unbounded_stages(self):
    with self.assertRaisesRegex(ValueError, "only one-update or three-update"):
      contract.requested_max_steps({
          "CANON_P39_64CHIP_PILOT": "1",
          "CANON_P34_RUN_STAGE": "full",
          "CANON_P34_NO_COMMIT": "0",
      })


if __name__ == "__main__":
  unittest.main()
