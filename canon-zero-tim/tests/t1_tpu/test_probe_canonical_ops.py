"""CPU-only contract tests for the canonical production-operator gate."""

from __future__ import annotations

import types
import unittest
from unittest import mock

import probe_canonical_ops as target


def _module(name: str, **attributes):
    return types.SimpleNamespace(__name__=name, **attributes)


class CanonicalOperatorContractTest(unittest.TestCase):
    def test_empty_environment_value_uses_the_registered_default(self):
        with mock.patch.dict(target.os.environ, {"DEPTH_TEST": ""}):
            self.assertEqual(target._envd("DEPTH_TEST", "1,2,4,8"), "1,2,4,8")

    def test_depth_parser_rejects_empty_duplicate_and_nonpositive_values(self):
        self.assertEqual(target._parse_depths("1,2,4,8"), (1, 2, 4, 8))
        for raw in ("", "0", "1,1", "x"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                target._parse_depths(raw)

    def test_gate_requires_complete_exact_live_rows(self):
        good = [
            {
                "differing_bytes": 0,
                "gradient_finite": True,
                "gradient_nonzero": 7,
            }
            for _ in range(4)
        ]
        self.assertTrue(target._admit_rows(good, 4))
        self.assertFalse(target._admit_rows(good[:-1], 4))
        for field, value in (
            ("differing_bytes", 1),
            ("gradient_finite", False),
            ("gradient_nonzero", 0),
        ):
            changed = [dict(row) for row in good]
            changed[2][field] = value
            self.assertFalse(target._admit_rows(changed, 4))

    def test_promotion_attestation_accepts_only_the_live_terminal_functions(self):
        matmul = lambda *_args, **_kwargs: None
        swiglu = lambda *_args, **_kwargs: None
        rmsnorm = lambda *_args, **_kwargs: None
        xf = _module("xf", pallas_matmul=matmul)
        xi = _module("xi", P22XI_XF_MODULE=xf)
        q2base = _module("q2base", P22XJ_XG_MODULE=_module("xg", pallas_swiglu=swiglu))
        q3base = _module("q3base", pallas_rmsnorm=rmsnorm)
        column = lambda *_args: None
        contract_parallel = lambda *_args: None
        column.__globals__["pallas_matmul"] = matmul
        contract_parallel.__globals__["pallas_matmul"] = matmul
        linear = _module(
            "linear",
            P22XK_MATMUL_ACTIVE=True,
            P22XK_LINEAR_BASE=xi,
            traced_canonical_vjp_matmul=matmul,
            _column_parallel=column,
            _contract_parallel=contract_parallel,
        )
        qwen2 = _module(
            "qwen2",
            P22XK_SWIGLU_ACTIVE=True,
            P22XK_QWEN2_BASE=q2base,
            traced_canonical_vjp_swiglu=swiglu,
        )
        qwen3 = _module(
            "qwen3",
            P22XK_RMSNORM_ACTIVE=True,
            P22XK_QWEN3_BASE=q3base,
            traced_canonical_vjp_rmsnorm=rmsnorm,
        )
        contract = _module(
            "contract",
            HIDDEN_SIZE=4096,
            INTERMEDIATE_SIZE=12288,
            TP_SIZE=4,
            match_site=lambda *_args: object(),
            preflight=lambda **_kwargs: None,
        )
        target._attest_promoted_modules(linear, qwen2, qwen3, contract)
        linear.P22XK_MATMUL_ACTIVE = False
        with self.assertRaisesRegex(RuntimeError, "promotion sentinel"):
            target._attest_promoted_modules(linear, qwen2, qwen3, contract)


if __name__ == "__main__":
    unittest.main()
