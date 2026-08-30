"""CPU contracts for the P34 DeepSWE topology and launch profile."""

from __future__ import annotations

import os
import importlib.util
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[3]


def _load_module(name, path):
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


deepswe_contract = _load_module(
    "p34_deepswe_contract", ROOT / "tunix/rl/deepswe_contract.py"
)


def _devices(cross_host: bool = False):
  result = []
  for x in range(4):
    for y in range(8):
      for z in range(8):
        process = y * 4 + z // 2 if cross_host else (x // 2) * 32 + y * 4 + z // 2
        result.append(
            types.SimpleNamespace(
                id=x * 64 + y * 8 + z,
                coords=(x, y, z),
                process_index=process,
            )
        )
  return result


class DeepSWEContractTest(unittest.TestCase):

  def test_signed_workload_and_rank_groups(self):
    workload = deepswe_contract.P34_WORKLOAD
    workload.validate()
    self.assertEqual(workload.name, workload.contract_name)
    self.assertEqual(workload.global_trajectories, 64)
    self.assertEqual(workload.local_trajectories, 4)
    self.assertEqual(workload.max_num_seqs_per_dp, 4)
    self.assertEqual(workload.max_num_batched_tokens_per_dp, 256)
    self.assertEqual(workload.max_num_seqs_per_dp * workload.dp_size, 64)
    self.assertEqual(
        workload.max_num_batched_tokens_per_dp * workload.dp_size,
        workload.global_m,
    )
    self.assertEqual(workload.rank_major_rows(), tuple(
        tuple(group * 16 + rank for rank in range(16))
        for group in range(4)
    ))

  def test_every_deepswe_contract_exposes_the_shared_adapter_identity(self):
    workloads = (
        deepswe_contract.P34_WORKLOAD,
        deepswe_contract.P39_PILOT_WORKLOAD,
        deepswe_contract.P43_DEBUG_WORKLOAD,
        deepswe_contract.P44_PARITY_64_WORKLOAD,
        deepswe_contract.P44_PARITY_128_WORKLOAD,
        deepswe_contract.P46_Q32_64_WORKLOAD,
        deepswe_contract.P46_Q32_256_WORKLOAD,
        deepswe_contract.P58_Q4_TIM_128_WORKLOAD,
    )
    for workload in workloads:
      with self.subTest(contract=workload.contract_name):
        self.assertEqual(workload.name, workload.contract_name)
    self.assertNotIn(
        "name",
        deepswe_contract.p44_recipe_signature(
            deepswe_contract.P44_PARITY_128_WORKLOAD
        ),
    )

  def test_legal_physical_halves_are_disjoint_and_exhaustive(self):
    rollout, trainer, report = deepswe_contract.split_4x8x8_role_devices(
        _devices()
    )
    self.assertEqual((len(rollout), len(trainer)), (128, 128))
    self.assertTrue(report["disjoint"])
    self.assertTrue(report["exhaustive"])
    self.assertTrue(report["host_complete"])
    self.assertEqual(report["slice_extents"], (4, 8, 8))

  def test_half_split_rejects_a_host_crossing_roles(self):
    with self.assertRaisesRegex(ValueError, "crosses host boundaries"):
      deepswe_contract.split_4x8x8_role_devices(_devices(cross_host=True))

  def test_profile_keeps_local_and_global_m_distinct(self):
    text = (
        ROOT
        / "canon-zero-tim/cluster/profiles/qwen3-32b-dp16-tp8-deepswe.env"
    ).read_text()
    self.assertIn("export CANON_LOGPROB_M=256", text)
    self.assertIn("export MIN_TOKEN_BUCKET=4096", text)
    self.assertIn("export CANON_P34_ABCPROD=256", text)
    self.assertIn("export CANON_VJP2_MAX_SEQS=1", text)
    self.assertIn("export CANON_P34_MAX_NUM_SEQS=4", text)
    self.assertIn("export CANON_P34_MAX_BATCHED_TOKENS=256", text)
    self.assertNotIn("export CANON_P34_MAX_NUM_SEQS=64", text)
    self.assertNotIn("export CANON_P34_MAX_BATCHED_TOKENS=8192", text)
    self.assertNotIn("CANON_LOGPROB_M=4096", text)

  def test_promotion_stage_budget_is_fail_closed(self):
    self.assertEqual(
        deepswe_contract.requested_max_steps({
            "CANON_P34_RUN_STAGE": "three-update",
            "CANON_P34_NO_COMMIT": "0",
        }),
        3,
    )
    with self.assertRaisesRegex(ValueError, "stage/no-commit mismatch"):
      deepswe_contract.requested_max_steps({
          "CANON_P34_RUN_STAGE": "backward-no-commit",
          "CANON_P34_NO_COMMIT": "0",
      })


if __name__ == "__main__":
  unittest.main()
