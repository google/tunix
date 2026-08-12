"""Static and arithmetic gates for the Qwen3-32B TP8 package."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "canon-zero-tim/src/engine_shims/models/qwen32b"


def _contract():
  path = MODEL_DIR / "p22xf_contract.py"
  spec = importlib.util.spec_from_file_location("p34_qwen32b_contract", path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


class Qwen32BTP8Test(unittest.TestCase):

  def test_all_local_projection_shapes_divide_128_tiles(self):
    contract = _contract()
    contract.validate_manifest(contract.SITES)
    self.assertEqual((contract.BM, contract.BN, contract.BK), (128, 128, 128))
    self.assertEqual(
        {site.family: (site.k_local, site.n_local) for site in contract.SITES},
        {
            "q_proj": (5120, 1024),
            "k_proj": (5120, 128),
            "v_proj": (5120, 128),
            "o_proj": (1024, 5120),
            "gate_proj": (5120, 3200),
            "up_proj": (5120, 3200),
            "down_proj": (3200, 5120),
        },
    )

  def test_old_256_output_tile_is_a_registered_negative(self):
    contract = _contract()
    offenders = [
        site.family for site in contract.SITES if site.n_local % 256
    ]
    self.assertEqual(offenders, ["k_proj", "v_proj", "gate_proj", "up_proj"])

  def test_swiglu_feature_padding_is_model_pinned(self):
    contract = _contract()
    self.assertEqual(contract.SWIGLU_FEATURE_PADDING, {3200: 3328})
    self.assertNotEqual(3200 % 256, 0)
    self.assertEqual(3328 % 256, 0)

  def test_projection_wrapper_forwards_model_tiles(self):
    linear = (ROOT / "canon-zero-tim/src/engine_shims/linear_p22xf.py").read_text()
    matmul = (
        ROOT / "canon-zero-tim/src/engine_shims/p22_pallas_matmul.py"
    ).read_text()
    self.assertIn("block_n=BN", linear)
    self.assertIn("block_k=BK", linear)
    self.assertIn("BN = 256", matmul)
    self.assertIn("BK = 256", matmul)
    self.assertIn("block_n: int = BN", matmul)
    self.assertIn("block_k: int = BK", matmul)

  def test_package_contains_model_specific_rmsnorm_wrapper(self):
    wrapper = (MODEL_DIR / "qwen3_p22xh.py").read_text()
    self.assertIn("model=qwen3-32b", wrapper)
    self.assertIn("HIDDEN_SIZE", wrapper)


if __name__ == "__main__":
  unittest.main()
