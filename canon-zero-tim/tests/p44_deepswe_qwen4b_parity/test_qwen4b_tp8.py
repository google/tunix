"""Static and arithmetic gates for the Qwen3-4B TP8 engine package."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "canon-zero-tim/src/engine_shims/models/qwen4b"
SPEC = importlib.util.spec_from_file_location(
    "p44_qwen4b_projection_contract", MODEL_DIR / "p22xf_contract.py"
)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import Qwen3-4B projection contract")
model = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = model
SPEC.loader.exec_module(model)


class Qwen4BTP8Test(unittest.TestCase):

  def test_registered_geometry_is_tp8_divisible(self):
    source = (ROOT / "tunix/models/qwen3/model.py").read_text()
    registry = (ROOT / "tunix/models/registry.py").read_text()
    self.assertIn("def qwen3_4b(cls)", source)
    self.assertIn("model_id='Qwen/Qwen3-4B'", registry)
    self.assertEqual(model.TP_SIZE, 8)
    for width in (
        model.HIDDEN_SIZE,
        model.INTERMEDIATE_SIZE,
        model.NUM_ATTENTION_HEADS,
        model.NUM_KV_HEADS,
    ):
      self.assertEqual(width % model.TP_SIZE, 0)

  def test_all_projection_sites_use_exact_tp8_shapes(self):
    model.validate_manifest(model.SITES)
    by_name = {site.family: site for site in model.SITES}
    self.assertEqual((by_name["q_proj"].k_local, by_name["q_proj"].n_local), (2560, 512))
    self.assertEqual((by_name["down_proj"].k_local, by_name["down_proj"].n_local), (1216, 2560))
    self.assertEqual((model.BM, model.BN, model.BK), (128, 128, 128))

  def test_canonical_vjp_uses_the_model_pinned_chunk(self):
    source = (
        ROOT / "canon-zero-tim/src/engine_shims/p22xk_vjp_ops.py"
    ).read_text()
    self.assertIn("from p22xf_contract import BK", source)
    self.assertNotIn("from p22_pallas_matmul import BK", source)
    self.assertNotEqual(1216 % model.BK, 0)
    self.assertEqual(model.MATMUL_K_PADDING, {1216: 1280})
    self.assertEqual(model.MATMUL_N_PADDING, {1216: 1280, 18992: 19200})
    self.assertEqual(model.MATMUL_K_PADDING[1216] % model.BK, 0)
    self.assertEqual(model.MATMUL_N_PADDING[1216] % model.BN, 0)

  def test_matmul_padding_covers_both_mlp_directions(self):
    by_name = {site.family: site for site in model.SITES}
    gate = by_name["gate_proj"]
    down = by_name["down_proj"]
    self.assertEqual(
        (gate.k_local, model.MATMUL_N_PADDING[gate.n_local]),
        (2560, 1280),
    )
    self.assertEqual(
        (model.MATMUL_K_PADDING[down.k_local], down.n_local),
        (1280, 2560),
    )
    self.assertEqual(model.MATMUL_N_PADDING[18992], 19200)
    self.assertEqual(151936 // model.TP_SIZE, 18992)
    self.assertEqual(19200 % 256, 0)
    wrapper = (
        ROOT / "canon-zero-tim/src/engine_shims/p22xi_padded_matmul.py"
    ).read_text()
    self.assertIn("return out[:m, :n]", wrapper)

  def test_swiglu_feature_padding_is_model_pinned(self):
    self.assertEqual(model.SWIGLU_FEATURE_PADDING, {1216: 1280})
    self.assertNotEqual(1216 % 256, 0)
    self.assertEqual(1280 % 256, 0)
    wrapper = (
        ROOT / "canon-zero-tim/src/engine_shims/p22xj_padded_swiglu.py"
    ).read_text()
    self.assertIn("SWIGLU_FEATURE_PADDING", wrapper)
    self.assertIn("return out[:m, :f]", wrapper)

  def test_model_manifest_matches(self):
    for line in (MODEL_DIR / "MANIFEST.sha256").read_text().splitlines():
      digest, name = line.split()
      self.assertEqual(
          hashlib.sha256((MODEL_DIR / name).read_bytes()).hexdigest(), digest
      )


if __name__ == "__main__":
  unittest.main()
