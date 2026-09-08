"""Fail-closed tests for the Qwen3-8B TP1 one-host overlay."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "canon-zero-tim/src/engine_shims/models/qwen8b_tp1"


def _load_contract():
  name = "v2_qwen8b_tp1_contract"
  spec = importlib.util.spec_from_file_location(
      name, MODEL_DIR / "p22xf_contract.py"
  )
  if spec is None or spec.loader is None:
    raise RuntimeError("cannot import Qwen3-8B TP1 projection contract")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


CONTRACT = _load_contract()


class Qwen8BTp1ContractTest(unittest.TestCase):

  def test_projection_shapes_are_exact(self):
    CONTRACT.validate_manifest(CONTRACT.SITES)
    self.assertEqual(CONTRACT.TP_SIZE, 1)
    self.assertEqual(CONTRACT.MATMUL_N_PADDING, {151936: 152064})
    self.assertEqual(
        {(site.family, site.k_local, site.n_local) for site in CONTRACT.SITES},
        {
            ("q_proj", 4096, 4096),
            ("k_proj", 4096, 1024),
            ("v_proj", 4096, 1024),
            ("o_proj", 4096, 4096),
            ("gate_proj", 4096, 12288),
            ("up_proj", 4096, 12288),
            ("down_proj", 12288, 4096),
        },
    )

  def test_wrong_tp_environment_is_rejected(self):
    values = {
        "CANON_QWEN3_HIDDEN_SIZE": "4096",
        "CANON_QWEN3_INTERMEDIATE_SIZE": "12288",
        "CANON_QWEN3_NUM_ATTENTION_HEADS": "32",
        "CANON_QWEN3_NUM_KV_HEADS": "8",
        "CANON_QWEN3_HEAD_DIM": "128",
        "CANON_QWEN3_TP_SIZE": "2",
        "CANON_PALLAS_ALL_PROJ": "1",
        "CANON_FIXED_AR": "1",
    }
    with mock.patch.dict(os.environ, values, clear=True):
      with self.assertRaisesRegex(RuntimeError, "CANON_QWEN3_TP_SIZE='2'"):
        CONTRACT.preflight(require_enabled=True)

  def test_manifest_reuses_only_the_reviewed_8b_wrapper(self):
    for line in (MODEL_DIR / "MANIFEST.sha256").read_text().splitlines():
      digest, name = line.split()
      source = (
          ROOT / "canon-zero-tim/src/engine_shims/models/qwen8b/qwen3_p22xh.py"
          if name == "qwen3_p22xh.py"
          else MODEL_DIR / name
      )
      self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
    installer = (ROOT / "canon-zero-tim/install.sh").read_text()
    self.assertIn(
        'if [ "$MODEL" = qwen8b_tp1 ] || [ "$MODEL" = qwen8b_tp2 ]; then',
        installer,
    )
    probe = (
        ROOT
        / "canon-zero-tim/tests/v2_frozenlake_onehost/"
        "probe_qwen8b_tp1_overlay.py"
    ).read_text()
    self.assertIn('"CANON_QWEN3_TP_SIZE": "1"', probe)
    self.assertIn("V2_QWEN8B_TP1_IMPORT_PASS", probe)


if __name__ == "__main__":
  unittest.main()
