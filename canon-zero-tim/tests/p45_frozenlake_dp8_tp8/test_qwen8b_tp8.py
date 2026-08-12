"""Static gates for the isolated Qwen3-8B TP8 engine overlay."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
TP4_DIR = ROOT / "canon-zero-tim/src/engine_shims/models/qwen8b"
TP8_DIR = ROOT / "canon-zero-tim/src/engine_shims/models/qwen8b_tp8"


def _load_contract(name: str, path: Path):
  spec = importlib.util.spec_from_file_location(name, path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import projection contract from {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


tp4 = _load_contract("p45_qwen8b_tp4_contract", TP4_DIR / "p22xf_contract.py")
tp8 = _load_contract("p45_qwen8b_tp8_contract", TP8_DIR / "p22xf_contract.py")


class Qwen8BTP8Test(unittest.TestCase):

  def test_tp4_overlay_remains_unchanged_and_separate(self):
    self.assertEqual(tp4.TP_SIZE, 4)
    self.assertEqual((tp4.BM, tp4.BN, tp4.BK), (128, 256, 256))
    self.assertEqual(
        {(site.family, site.k_local, site.n_local) for site in tp4.SITES},
        {
            ("q_proj", 4096, 1024),
            ("k_proj", 4096, 256),
            ("v_proj", 4096, 256),
            ("o_proj", 1024, 4096),
            ("gate_proj", 4096, 3072),
            ("up_proj", 4096, 3072),
            ("down_proj", 3072, 4096),
        },
    )

  def test_all_seven_tp8_projection_shapes_are_exact(self):
    tp8.validate_manifest(tp8.SITES)
    self.assertEqual(tp8.TP_SIZE, 8)
    self.assertEqual((tp8.BM, tp8.BN, tp8.BK), (128, 128, 128))
    self.assertEqual(
        {(site.family, site.k_local, site.n_local) for site in tp8.SITES},
        {
            ("q_proj", 4096, 512),
            ("k_proj", 4096, 128),
            ("v_proj", 4096, 128),
            ("o_proj", 512, 4096),
            ("gate_proj", 4096, 1536),
            ("up_proj", 4096, 1536),
            ("down_proj", 1536, 4096),
        },
    )

  def test_tp8_requires_no_matmul_or_swiglu_padding(self):
    self.assertEqual(tp8.MATMUL_K_PADDING, {})
    self.assertEqual(tp8.MATMUL_N_PADDING, {})
    self.assertEqual(tp8.SWIGLU_FEATURE_PADDING, {})
    self.assertEqual(tp8.INTERMEDIATE_SIZE // tp8.TP_SIZE, 1536)
    self.assertEqual(1536 % tp8.BK, 0)
    self.assertEqual(1536 % tp8.BN, 0)
    self.assertEqual(1536 % 256, 0)

  def test_tp4_environment_is_rejected(self):
    model_env = {
        "CANON_QWEN3_HIDDEN_SIZE": "4096",
        "CANON_QWEN3_INTERMEDIATE_SIZE": "12288",
        "CANON_QWEN3_NUM_ATTENTION_HEADS": "32",
        "CANON_QWEN3_NUM_KV_HEADS": "8",
        "CANON_QWEN3_HEAD_DIM": "128",
        "CANON_QWEN3_TP_SIZE": "4",
        "CANON_PALLAS_ALL_PROJ": "1",
        "CANON_FIXED_AR": "1",
    }
    with mock.patch.dict(os.environ, model_env, clear=True):
      with self.assertRaisesRegex(RuntimeError, "CANON_QWEN3_TP_SIZE='4'"):
        tp8.preflight(require_enabled=True)

  def test_tp8_manifests_match_and_rmsnorm_wrapper_is_model_local(self):
    for model_dir in (TP4_DIR, TP8_DIR):
      for line in (model_dir / "MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split()
        self.assertEqual(
            hashlib.sha256((model_dir / name).read_bytes()).hexdigest(), digest
        )
    self.assertNotEqual(
        (TP4_DIR / "p22xf_contract.py").resolve(),
        (TP8_DIR / "p22xf_contract.py").resolve(),
    )
    self.assertIn(
        "model=qwen3-8b-tp8", (TP8_DIR / "qwen3_p22xh.py").read_text()
    )

  def test_p45_profile_selects_only_the_tp8_overlay(self):
    profile = (
        ROOT
        / "canon-zero-tim/cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-resident.env"
    ).read_text()
    self.assertIn("export CANON_MODEL_DIR_NAME=qwen8b_tp8", profile)
    tp4_profile = (
        ROOT / "canon-zero-tim/cluster/profiles/qwen3-8b-dp16-tp4-admission.env"
    ).read_text()
    self.assertIn("export CANON_MODEL_DIR_NAME=qwen8b", tp4_profile)
    self.assertNotIn("qwen8b_tp8", tp4_profile)


if __name__ == "__main__":
  unittest.main()
