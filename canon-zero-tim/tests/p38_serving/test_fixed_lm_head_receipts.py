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


def _log(endpoint: str = "tied_embed", *, vjp: bool = True) -> str:
  hidden = 2048 if endpoint == "tied_embed" else 4096
  lines = []
  if endpoint == "tied_embed":
    lines.append("[P28.G5C] TIED_EMBEDDING_HEAD on shared_leaves=1")
  for semantic_m in (*receipts.REQUEST_M, receipts.LEARNER_M):
    chunks = 16 if semantic_m == receipts.LEARNER_M else 1
    lines.append(
        "[PATHTRACE] CANON_P38_FIXED_LM_HEAD=1 "
        f"semantic_M={semantic_m} fixed_M=256 K={hidden} "
        "local_N=37984 fixed_N=38144 BM=128 BN=256 BK=256 "
        f"chunks={chunks} endpoint={endpoint}"
    )
  if vjp:
    lines.append(
        "[PATHTRACE] CANON_P38_FIXED_LM_HEAD_VJP=1 semantic_M=4096 "
        "fixed_M=256 chunks=16 accumulation=lax.scan order=ascending "
        f"K={hidden} endpoint={endpoint}"
    )
  return "\n".join(lines) + "\n"


class FixedLmHeadReceiptTest(unittest.TestCase):

  def test_tied_full_contract_passes(self):
    report = receipts.classify(
        _log(), endpoint="tied_embed", hidden=2048, require_vjp=True
    )
    self.assertEqual(report["verdict"], "P38_FIXED_LM_HEAD_RECEIPTS_PASS")
    self.assertEqual(report["reasons"], [])

  def test_untied_full_contract_passes(self):
    report = receipts.classify(
        _log("untied_lm_head"),
        endpoint="untied_lm_head",
        hidden=4096,
        require_vjp=True,
    )
    self.assertEqual(report["verdict"], "P38_FIXED_LM_HEAD_RECEIPTS_PASS")

  def test_missing_request_shape_fails(self):
    text = "\n".join(
        line for line in _log().splitlines() if "semantic_M=64 " not in line
    )
    report = receipts.classify(
        text, endpoint="tied_embed", hidden=2048, require_vjp=True
    )
    self.assertIn(64, report["missing_M"])
    self.assertIn("missing_primal_M=64", report["reasons"])

  def test_old_unscoped_receipts_fail(self):
    text = _log().replace(" endpoint=tied_embed", "")
    report = receipts.classify(
        text, endpoint="tied_embed", hidden=2048, require_vjp=True
    )
    self.assertTrue(report["missing_M"])
    self.assertIn("missing", report["foreign_endpoints"])

  def test_wrong_endpoint_fails(self):
    text = _log().replace("endpoint=tied_embed", "endpoint=untied_lm_head")
    report = receipts.classify(
        text, endpoint="tied_embed", hidden=2048, require_vjp=True
    )
    self.assertIn("untied_lm_head", report["foreign_endpoints"])

  def test_vjp_and_tied_marker_are_independent_gates(self):
    no_vjp = receipts.classify(
        _log(vjp=False), endpoint="tied_embed", hidden=2048, require_vjp=True
    )
    self.assertIn("missing_fixed_order_vjp", no_vjp["reasons"])
    no_marker = receipts.classify(
        _log().replace(
            "[P28.G5C] TIED_EMBEDDING_HEAD on shared_leaves=1\n", ""
        ),
        endpoint="tied_embed",
        hidden=2048,
        require_vjp=True,
    )
    self.assertIn("missing_tied_embedding_adapter_marker", no_marker["reasons"])


if __name__ == "__main__":
  unittest.main()
