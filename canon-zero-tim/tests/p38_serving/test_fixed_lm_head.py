#!/usr/bin/env python3
"""CPU/static contracts for the P38.2x fixed lm_head construction."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SHIM = ROOT / "canon-zero-tim/src/engine_shims"
MODEL_CONTRACT = SHIM / "models/qwen8b/p22xf_contract.py"
SOURCE = SHIM / "linear_p22xk.py"
FIXED_SOURCE = SHIM / "p38_fixed_lm_head.py"
PROBE = (
    ROOT
    / "canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts"
    / "probe_p38_fixed_lm_head.py"
)
RUN_STEP = ROOT / "canon-zero-tim/cluster/steps/90_run.sh"

sys.path.insert(0, str(SHIM))
import p38_fixed_lm_head as fixed  # noqa: E402


def _load(path: Path, name: str):
  spec = importlib.util.spec_from_file_location(name, path)
  assert spec and spec.loader
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


class FixedLmHeadContractTest(unittest.TestCase):

  def test_registered_production_shape(self):
    self.assertEqual(fixed.REQUEST_M, (8, 16, 32, 64, 128, 256))
    self.assertEqual(fixed.LEARNER_M, (4096,))
    self.assertEqual(fixed.SEMANTIC_M, (8, 16, 32, 64, 128, 256, 4096))
    for m in fixed.SEMANTIC_M:
      with self.subTest(m=m):
        self.assertEqual(
            fixed.validate_global_contract(
                (m, 4096), (4096, 151936), "bfloat16", "bfloat16", tp_size=4
            ),
            m,
        )
        self.assertEqual(
            fixed.validate_local_contract((m, 4096), (4096, 37984)), m
        )
    self.assertEqual((fixed.BM, fixed.BN, fixed.BK), (128, 256, 256))
    self.assertEqual(fixed.PADDED_LOCAL_VOCAB, 38144)

  def test_qwen1p7b_production_shape(self):
    for m in fixed.SEMANTIC_M:
      with self.subTest(m=m):
        self.assertEqual(
            fixed.validate_global_contract(
                (m, 2048), (2048, 151936), "bfloat16", "bfloat16",
                tp_size=4,
            ),
            m,
        )
        self.assertEqual(
            fixed.validate_local_contract((m, 2048), (2048, 37984)), m
        )

  def test_shape_dtype_and_topology_negatives(self):
    base = ((16, 4096), (4096, 151936), "bfloat16", "bfloat16")
    cases = (
        ((1, 4096), base[1], base[2], base[3], 4),
        ((7, 4096), base[1], base[2], base[3], 4),
        ((24, 4096), base[1], base[2], base[3], 4),
        ((257, 4096), base[1], base[2], base[3], 4),
        ((512, 4096), base[1], base[2], base[3], 4),
        ((2048, 4096), base[1], base[2], base[3], 4),
        ((8192, 4096), base[1], base[2], base[3], 4),
        ((16, 2048), base[1], base[2], base[3], 4),
        ((16, 3072), (3072, 151936), base[2], base[3], 4),
        (base[0], (4096, 152064), base[2], base[3], 4),
        (base[0], base[1], "float32", base[3], 4),
        (base[0], base[1], base[2], "float32", 4),
        (base[0], base[1], base[2], base[3], 8),
    )
    for xshape, wshape, xdtype, wdtype, tp in cases:
      with self.subTest(case=(xshape, wshape, xdtype, wdtype, tp)):
        with self.assertRaises(ValueError):
          fixed.validate_global_contract(
              xshape, wshape, xdtype, wdtype, tp_size=tp
          )

  def test_flag_preflight_requires_full_canonical_stack(self):
    clean = {fixed.ENV: "1"}
    with mock.patch.dict(os.environ, clean, clear=True):
      with self.assertRaisesRegex(RuntimeError, "dependencies missing"):
        fixed.preflight(require_enabled=True)
    enabled = {fixed.ENV: "1", **fixed.REQUIRED}
    with mock.patch.dict(os.environ, enabled, clear=True):
      fixed.preflight(require_enabled=True)
      os.environ["CANON_MM_ALGO"] = "1"
      with self.assertRaisesRegex(RuntimeError, "conflicting diagnostics"):
        fixed.preflight(require_enabled=True)
    with mock.patch.dict(os.environ, {fixed.ENV: "bogus"}, clear=True):
      with self.assertRaisesRegex(RuntimeError, "must be unset, 0, or 1"):
        fixed.preflight(require_enabled=False)

  def test_model_contract_has_only_lm_head_n_padding(self):
    model = _load(MODEL_CONTRACT, "p38_fixed_lm_head_qwen8b_contract")
    self.assertEqual(model.MATMUL_K_PADDING, {})
    self.assertEqual(model.MATMUL_N_PADDING, {37984: 38144})
    model.validate_manifest(model.SITES)

  def test_qwen1p7b_contract_has_only_lm_head_n_padding(self):
    model = _load(
        SHIM / "models/qwen1p7b/p22xf_contract.py",
        "p38_fixed_lm_head_qwen1p7b_contract",
    )
    self.assertEqual(model.MATMUL_K_PADDING, {})
    self.assertEqual(model.MATMUL_N_PADDING, {37984: 38144})
    model.validate_manifest(model.SITES)

  def test_hook_is_default_off_and_flag_scoped(self):
    text = SOURCE.read_text()
    self.assertIn('_p38_fixed_lm_head_value == "1"', text)
    self.assertIn("_p22xk_linear_module.JaxLmHead.__call__ =", text)
    self.assertIn("_p38_embed_module.JaxEmbed.decode =", text)
    self.assertIn("self.weight.value.T", text)
    self.assertIn('endpoint="untied_lm_head"', text)
    self.assertIn('endpoint="tied_embed"', text)
    self.assertIn("P38_FIXED_LM_HEAD_ACTIVE", text)
    self.assertIn("P38_FIXED_TIED_HEAD_ACTIVE", text)
    self.assertNotIn("CANON_P38_FIXED_LM_HEAD", text.split("JaxEinsum =", 1)[0])

  def test_fixed_head_requires_a_registered_endpoint(self):
    self.assertEqual(
        fixed.ENDPOINTS, ("untied_lm_head", "tied_embed", "direct_probe")
    )
    text = FIXED_SOURCE.read_text()
    self.assertIn("endpoint must be one of", text)
    self.assertIn("f\"K={hidden_size} endpoint={endpoint}\"", text)
    self.assertIn("f\"endpoint={endpoint}\"", text)

  def test_learner_uses_fixed_chunks_without_stock_fallback(self):
    text = FIXED_SOURCE.read_text()
    self.assertIn("@jax.custom_vjp", text)
    self.assertIn("learner_fixed_vjp.defvjp(learner_fwd, learner_bwd)", text)
    self.assertIn("weight_cotangent + chunk_weight_cotangent", text)
    self.assertIn("lax.scan(", text)
    self.assertIn("out = learner_fixed_vjp(a_local, w_local)", text)
    self.assertNotIn("original_lm_head", text)

  def test_exact_target_does_not_require_a_mismatch_join(self):
    text = RUN_STEP.read_text()
    self.assertIn(
        '[ "${CANON_P38_FIXED_LM_HEAD:-0}" != "1" ]', text
    )
    self.assertIn("p38_join_args+=(--require-mismatch-join)", text)

  def test_probe_verdict_and_negative(self):
    # Stub heavy JAX/shim imports so the host-only classifier is testable.
    modules = {
        "p22xi_padded_matmul": mock.Mock(matmul=mock.Mock()),
        "p22xk_vjp_ops": mock.Mock(matmul=mock.Mock()),
        "probe_p38_lm_head": mock.Mock(
            DECODE_M=16,
            PREFILL_M=256,
            _different_elements=mock.Mock(),
            _flip_one_bit=mock.Mock(),
            _load_weight=mock.Mock(),
            _max_abs=mock.Mock(),
        ),
    }
    with mock.patch.dict(sys.modules, modules):
      probe = _load(PROBE, "p38_fixed_lm_head_probe_test")
    exact = [{"fixed_differing_elements": 0}]
    red = [{"fixed_differing_elements": 1}]
    self.assertEqual(
        probe.classify(exact, 1), "FIXED_LM_HEAD_ONEHOST_CONSTRUCTION_PASS"
    )
    self.assertEqual(probe.classify(red, 1), "FIXED_LM_HEAD_NOT_INVARIANT")
    self.assertEqual(probe.classify(exact, 0), "FAIL_NEGATIVE_CONTROL")
    self.assertEqual(
        probe.classify(exact, 1, [{"differing_elements": 1}]),
        "FIXED_LM_HEAD_LEARNER_CHUNK_NOT_INVARIANT",
    )

  def test_vjp_probe_verdicts_are_fail_closed(self):
    exact = dict(
        hidden_differing=0,
        weight_differing=0,
        repeat_hidden_differing=0,
        repeat_weight_differing=0,
        gradients_finite=True,
        gradients_nonzero=True,
        negative_differing=1,
    )
    self.assertEqual(fixed.classify_vjp(**exact), fixed.VJP_PASS)
    for key, verdict in (
        ("hidden_differing", "FIXED_LM_HEAD_CHUNK_VJP_NOT_INVARIANT"),
        ("weight_differing", "FIXED_LM_HEAD_CHUNK_VJP_NOT_INVARIANT"),
        ("repeat_hidden_differing", "FIXED_LM_HEAD_VJP_NOT_DETERMINISTIC"),
        ("repeat_weight_differing", "FIXED_LM_HEAD_VJP_NOT_DETERMINISTIC"),
    ):
      with self.subTest(key=key):
        values = dict(exact)
        values[key] = 1
        self.assertEqual(fixed.classify_vjp(**values), verdict)
    values = dict(exact, gradients_finite=False)
    self.assertEqual(fixed.classify_vjp(**values), "FAIL_NONFINITE_GRADIENT")
    values = dict(exact, gradients_nonzero=False)
    self.assertEqual(
        fixed.classify_vjp(**values), "INCONCLUSIVE_NO_GRADIENT_SIGNAL"
    )
    values = dict(exact, negative_differing=0)
    self.assertEqual(fixed.classify_vjp(**values), "FAIL_NEGATIVE_CONTROL")


if __name__ == "__main__":
  unittest.main()
