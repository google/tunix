#!/usr/bin/env python3
"""Tests for the fixed-head executable receipt classifier."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts"
    / "classify_p38_fixed_lm_head_receipts.py"
)
spec = importlib.util.spec_from_file_location("p38_fixed_receipts", SCRIPT)
assert spec and spec.loader
receipts = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = receipts
spec.loader.exec_module(receipts)


def _log(
    endpoint: str = "tied_embed",
    *,
    hidden: int = 2048,
    tp_size: int = 4,
    vjp: bool = True,
) -> str:
  local_vocab, padded_local_vocab = receipts.GEOMETRIES[
      (endpoint, hidden, tp_size)
  ]
  lines = []
  if endpoint == "tied_embed":
    lines.append("[P28.G5C] TIED_EMBEDDING_HEAD on shared_leaves=1")
  for semantic_m in (*receipts.REQUEST_M, receipts.LEARNER_M):
    chunks = 16 if semantic_m == receipts.LEARNER_M else 1
    lines.append(
        "[PATHTRACE] CANON_P38_FIXED_LM_HEAD=1 "
        f"semantic_M={semantic_m} fixed_M=256 K={hidden} TP={tp_size} "
        f"local_N={local_vocab} fixed_N={padded_local_vocab} "
        "BM=128 BN=256 BK=256 "
        f"chunks={chunks} endpoint={endpoint}"
    )
  if vjp:
    lines.append(
        "[PATHTRACE] CANON_P38_FIXED_LM_HEAD_VJP=1 semantic_M=4096 "
        "fixed_M=256 chunks=16 accumulation=lax.scan order=ascending "
        f"K={hidden} TP={tp_size} local_N={local_vocab} "
        f"fixed_N={padded_local_vocab} endpoint={endpoint}"
    )
  return "\n".join(lines) + "\n"


class FixedLmHeadReceiptTest(unittest.TestCase):

  def test_tied_full_contract_passes(self):
    report = receipts.classify(
        _log(),
        endpoint="tied_embed",
        hidden=2048,
        tp_size=4,
        require_vjp=True,
    )
    self.assertEqual(report["verdict"], "P38_FIXED_LM_HEAD_RECEIPTS_PASS")
    self.assertEqual(report["reasons"], [])

  def test_untied_full_contract_passes(self):
    report = receipts.classify(
        _log("untied_lm_head", hidden=4096),
        endpoint="untied_lm_head",
        hidden=4096,
        tp_size=4,
        require_vjp=True,
    )
    self.assertEqual(report["verdict"], "P38_FIXED_LM_HEAD_RECEIPTS_PASS")

  def test_missing_request_shape_fails(self):
    text = "\n".join(
        line for line in _log().splitlines() if "semantic_M=64 " not in line
    )
    report = receipts.classify(
        text,
        endpoint="tied_embed",
        hidden=2048,
        tp_size=4,
        require_vjp=True,
    )
    self.assertIn(64, report["missing_M"])
    self.assertIn("missing_primal_M=64", report["reasons"])

  def test_old_unscoped_receipts_fail(self):
    text = _log().replace(" endpoint=tied_embed", "")
    report = receipts.classify(
        text,
        endpoint="tied_embed",
        hidden=2048,
        tp_size=4,
        require_vjp=True,
    )
    self.assertTrue(report["missing_M"])
    self.assertIn("missing", report["foreign_endpoints"])

  def test_wrong_endpoint_fails(self):
    text = _log().replace("endpoint=tied_embed", "endpoint=untied_lm_head")
    report = receipts.classify(
        text,
        endpoint="tied_embed",
        hidden=2048,
        tp_size=4,
        require_vjp=True,
    )
    self.assertIn("untied_lm_head", report["foreign_endpoints"])

  def test_vjp_and_tied_marker_are_independent_gates(self):
    no_vjp = receipts.classify(
        _log(vjp=False),
        endpoint="tied_embed",
        hidden=2048,
        tp_size=4,
        require_vjp=True,
    )
    self.assertIn("missing_fixed_order_vjp", no_vjp["reasons"])
    no_marker = receipts.classify(
        _log().replace(
            "[P28.G5C] TIED_EMBEDDING_HEAD on shared_leaves=1\n", ""
        ),
        endpoint="tied_embed",
        hidden=2048,
        tp_size=4,
        require_vjp=True,
    )
    self.assertIn("missing_tied_embedding_adapter_marker", no_marker["reasons"])

  def test_tp8_tied_qwen4b_contract_passes(self):
    report = receipts.classify(
        _log("tied_embed", hidden=2560, tp_size=8),
        endpoint="tied_embed",
        hidden=2560,
        tp_size=8,
        require_vjp=True,
    )
    self.assertEqual(report["verdict"], "P38_FIXED_LM_HEAD_RECEIPTS_PASS")
    self.assertEqual(report["local_vocab"], 18992)
    self.assertEqual(report["padded_local_vocab"], 19200)

  def test_tp8_untied_qwen32b_contract_passes(self):
    report = receipts.classify(
        _log("untied_lm_head", hidden=5120, tp_size=8),
        endpoint="untied_lm_head",
        hidden=5120,
        tp_size=8,
        require_vjp=True,
    )
    self.assertEqual(report["verdict"], "P38_FIXED_LM_HEAD_RECEIPTS_PASS")

  def test_tp8_untied_qwen8b_contract_passes(self):
    report = receipts.classify(
        _log("untied_lm_head", hidden=4096, tp_size=8),
        endpoint="untied_lm_head",
        hidden=4096,
        tp_size=8,
        require_vjp=True,
    )
    self.assertEqual(report["verdict"], "P38_FIXED_LM_HEAD_RECEIPTS_PASS")
    self.assertEqual(report["local_vocab"], 18992)
    self.assertEqual(report["padded_local_vocab"], 19200)

  def test_request_only_eval_does_not_require_learner_or_vjp(self):
    text = "\n".join(
        line
        for line in _log("untied_lm_head", hidden=4096, tp_size=8).splitlines()
        if "semantic_M=4096 " not in line
    )
    report = receipts.classify(
        text,
        endpoint="untied_lm_head",
        hidden=4096,
        tp_size=8,
        require_vjp=False,
        include_learner=False,
    )
    self.assertEqual(report["verdict"], "P38_FIXED_LM_HEAD_RECEIPTS_PASS")
    self.assertIsNone(report["learner_M"])

  def test_unregistered_endpoint_geometry_fails(self):
    with self.assertRaisesRegex(ValueError, "unsupported fixed-head geometry"):
      receipts.classify(
          "",
          endpoint="untied_lm_head",
          hidden=2560,
          tp_size=8,
          require_vjp=False,
      )


if __name__ == "__main__":
  unittest.main()
