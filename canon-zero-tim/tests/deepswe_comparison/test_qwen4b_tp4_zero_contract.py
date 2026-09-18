#!/usr/bin/env python3
"""Static contracts for the isolated P58.20 Qwen3-4B TP4 carrier."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = (
    ROOT
    / "canon-zero-tim/src/engine_shims/models/qwen4b_tp4/p22xf_contract.py"
)


def _load():
  spec = importlib.util.spec_from_file_location(
      "p58_qwen4b_tp4_contract", CONTRACT_PATH
  )
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


contract = _load()


class Qwen4bTp4ZeroContractTest(unittest.TestCase):

  def test_signed_projection_shapes_and_padding(self):
    by_name = {site.family: site for site in contract.SITES}
    self.assertEqual(
        (by_name["q_proj"].k_local, by_name["q_proj"].n_local),
        (2560, 1024),
    )
    self.assertEqual(
        (by_name["k_proj"].k_local, by_name["k_proj"].n_local),
        (2560, 256),
    )
    self.assertEqual(
        (by_name["down_proj"].k_local, by_name["down_proj"].n_local),
        (2432, 2560),
    )
    self.assertEqual(contract.MATMUL_K_PADDING, {2432: 2560})
    self.assertEqual(
        contract.MATMUL_N_PADDING, {2432: 2560, 37984: 38144}
    )
    self.assertEqual(contract.SWIGLU_FEATURE_PADDING, {2432: 2560})

  def test_preflight_rejects_wrong_tp_identity(self):
    values = {
        "CANON_PALLAS_ALL_PROJ": "1",
        "CANON_FIXED_AR": "1",
        "CANON_QWEN3_HIDDEN_SIZE": "2560",
        "CANON_QWEN3_INTERMEDIATE_SIZE": "9728",
        "CANON_QWEN3_NUM_ATTENTION_HEADS": "32",
        "CANON_QWEN3_NUM_KV_HEADS": "8",
        "CANON_QWEN3_HEAD_DIM": "128",
        "CANON_QWEN3_TP_SIZE": "4",
    }
    with mock.patch.dict(os.environ, values, clear=True):
      contract.preflight(require_enabled=True)
      os.environ["CANON_QWEN3_TP_SIZE"] = "8"
      with self.assertRaisesRegex(RuntimeError, "model contract mismatch"):
        contract.preflight(require_enabled=True)

  def test_installer_keeps_tp4_model_separate_from_tp8(self):
    installer = (ROOT / "canon-zero-tim/install.sh").read_text()
    self.assertIn('if [ "$MODEL" = qwen4b_tp4 ]; then', installer)
    self.assertIn(
        'models/qwen4b/qwen3_p22xh.py" "$OUT/"', installer
    )
    manifest = (
        ROOT
        / "canon-zero-tim/src/engine_shims/models/qwen4b_tp4/MANIFEST.sha256"
    ).read_text()
    self.assertIn("p22xf_contract.py", manifest)
    self.assertIn("qwen3_p22xh.py", manifest)


if __name__ == "__main__":
  unittest.main()
