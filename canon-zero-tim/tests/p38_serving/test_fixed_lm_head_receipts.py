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
    learner_m: int = receipts.DEFAULT_LEARNER_M,
    p59_local_dp_size: int | None = None,
) -> str:
  local_vocab, padded_local_vocab = receipts.GEOMETRIES[
      (endpoint, hidden, tp_size)
  ]
  lines = []
  if endpoint == "tied_embed":
    lines.append("[P28.G5C] TIED_EMBEDDING_HEAD on shared_leaves=1")
  for semantic_m in receipts.REQUEST_M:
    lines.append(
        "[PATHTRACE] CANON_P38_FIXED_LM_HEAD=1 "
        f"semantic_M={semantic_m} fixed_M=256 K={hidden} TP={tp_size} "
        f"local_N={local_vocab} fixed_N={padded_local_vocab} "
        "BM=128 BN=256 BK=256 "
        f"chunks=1 endpoint={endpoint}"
    )
  if p59_local_dp_size is None:
    lines.append(
        "[PATHTRACE] CANON_P38_FIXED_LM_HEAD=1 "
        f"semantic_M={learner_m} fixed_M=256 K={hidden} TP={tp_size} "
        f"local_N={local_vocab} fixed_N={padded_local_vocab} "
        "BM=128 BN=256 BK=256 "
        f"chunks={learner_m // 256} endpoint={endpoint}"
    )
  else:
    local_m = learner_m // p59_local_dp_size
    lines.append(
        "[PATHTRACE] CANON_P38_FIXED_LM_HEAD=1 "
        f"semantic_M={local_m} fixed_M=256 K={hidden} TP={tp_size} "
        f"local_N={local_vocab} fixed_N={padded_local_vocab} "
        "BM=128 BN=256 BK=256 chunks=1 "
        f"endpoint={endpoint} p59_local=1 global_M={learner_m} "
        f"dp={p59_local_dp_size}"
    )
  if vjp:
    if p59_local_dp_size is None:
      lines.append(
          "[PATHTRACE] CANON_" "P38_FIXED_LM_HEAD_VJP=1 "
          f"semantic_M={learner_m} fixed_M=256 "
          f"chunks={learner_m // 256} "
          "accumulation=lax.scan order=ascending "
          f"K={hidden} TP={tp_size} local_N={local_vocab} "
          f"fixed_N={padded_local_vocab} endpoint={endpoint}"
      )
    else:
      lines.append(
          "[PATHTRACE] CANON_" "P38_FIXED_LM_HEAD_VJP=1 "
          f"semantic_M={learner_m} "
          f"local_M={learner_m // p59_local_dp_size} "
          "fixed_M=256 chunks=1 accumulation=lax.scan order=ascending "
          "tp_input_reduction=all_gather_rank_order_f32_barrier "
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

  def test_frozenlake_tp8_learner_m2048_passes_and_wrong_m_fails(self):
    text = _log(
        "untied_lm_head", hidden=4096, tp_size=8, learner_m=2048
    )
    report = receipts.classify(
        text,
        endpoint="untied_lm_head",
        hidden=4096,
        tp_size=8,
        require_vjp=True,
        learner_m=2048,
    )
    self.assertEqual(report["verdict"], "P38_FIXED_LM_HEAD_RECEIPTS_PASS")
    self.assertEqual(report["learner_M"], 2048)
    wrong = receipts.classify(
        text,
        endpoint="untied_lm_head",
        hidden=4096,
        tp_size=8,
        require_vjp=True,
        learner_m=4096,
    )
    self.assertEqual(wrong["verdict"], "P38_FIXED_LM_HEAD_RECEIPTS_FAIL")
    self.assertIn("missing_primal_M=4096", wrong["reasons"])
    self.assertIn("missing_fixed_order_vjp", wrong["reasons"])

  def test_p59_local_dp8_m2048_requires_exact_local_receipts(self):
    text = _log(
        "untied_lm_head",
        hidden=4096,
        tp_size=8,
        learner_m=2048,
        p59_local_dp_size=8,
    )
    report = receipts.classify(
        text,
        endpoint="untied_lm_head",
        hidden=4096,
        tp_size=8,
        require_vjp=True,
        learner_m=2048,
        p59_local_dp_size=8,
    )
    self.assertEqual(report["verdict"], "P38_FIXED_LM_HEAD_RECEIPTS_PASS")
    self.assertEqual(report["p59_local_M"], 256)
    self.assertEqual(report["matching_p59_local_primal_records"], 1)

    corruptions = {
        "global": text.replace("global_M=2048", "global_M=4096"),
        "local": text.replace("local_M=256", "local_M=128"),
        "chunks": text.replace(
            "chunks=1 endpoint=untied_lm_head p59_local=1",
            "chunks=2 endpoint=untied_lm_head p59_local=1",
        ),
        "reduction": text.replace(
            "tp_input_reduction=all_gather_rank_order_f32_barrier ", ""
        ),
    }
    for label, corrupted in corruptions.items():
      with self.subTest(label=label):
        red = receipts.classify(
            corrupted,
            endpoint="untied_lm_head",
            hidden=4096,
            tp_size=8,
            require_vjp=True,
            learner_m=2048,
            p59_local_dp_size=8,
        )
        self.assertEqual(
            red["verdict"], "P38_FIXED_LM_HEAD_RECEIPTS_FAIL"
        )

  def test_p59_local_dp16_m4096_passes(self):
    report = receipts.classify(
        _log(
            "tied_embed",
            hidden=2048,
            tp_size=4,
            learner_m=4096,
            p59_local_dp_size=16,
        ),
        endpoint="tied_embed",
        hidden=2048,
        tp_size=4,
        require_vjp=True,
        learner_m=4096,
        p59_local_dp_size=16,
    )
    self.assertEqual(report["verdict"], "P38_FIXED_LM_HEAD_RECEIPTS_PASS")
    self.assertEqual(report["p59_local_M"], 256)

  def test_p59_local_rejects_non_m256_global_dp_pair(self):
    with self.assertRaisesRegex(ValueError, "local_M=256"):
      receipts.classify(
          "",
          endpoint="untied_lm_head",
          hidden=4096,
          tp_size=8,
          require_vjp=True,
          learner_m=4096,
          p59_local_dp_size=8,
      )

  def test_p59_local_mode_cannot_drop_learner_or_vjp_receipts(self):
    for include_learner, require_vjp in ((False, True), (True, False)):
      with self.subTest(
          include_learner=include_learner, require_vjp=require_vjp
      ):
        with self.assertRaisesRegex(ValueError, "requires learner primal"):
          receipts.classify(
              "",
              endpoint="untied_lm_head",
              hidden=4096,
              tp_size=8,
              require_vjp=require_vjp,
              include_learner=include_learner,
              learner_m=2048,
              p59_local_dp_size=8,
          )

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

  def test_learner_m2048_rejects_neighboring_geometry(self):
    with self.assertRaisesRegex(ValueError, "registered only"):
      receipts.classify(
          _log("tied_embed", hidden=2560, tp_size=8, learner_m=2048),
          endpoint="tied_embed",
          hidden=2560,
          tp_size=8,
          require_vjp=True,
          learner_m=2048,
      )


if __name__ == "__main__":
  unittest.main()
