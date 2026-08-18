#!/usr/bin/env python3
"""Real-weight one-v5p construction gate for P38.2x fixed lm_head."""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np

from p22xi_padded_matmul import matmul as padded_pallas_matmul
from p22xk_vjp_ops import matmul as canonical_vjp_matmul
from p38_fixed_lm_head import (
    BK,
    BM,
    BN,
    FIXED_M,
    HIDDEN,
    LOCAL_VOCAB,
    PADDED_LOCAL_VOCAB,
    TP_SIZE,
    VOCAB,
    fixed_lm_head,
    preflight,
)
from probe_p38_lm_head import (
    DECODE_M,
    PREFILL_M,
    _different_elements,
    _flip_one_bit,
    _load_weight,
    _max_abs,
)


def classify(rows: list[dict[str, Any]], negative_differing: int) -> str:
    if negative_differing != 1:
        return "FAIL_NEGATIVE_CONTROL"
    if any(row["fixed_differing_elements"] for row in rows):
        return "FIXED_LM_HEAD_NOT_INVARIANT"
    return "FIXED_LM_HEAD_ONEHOST_CONSTRUCTION_PASS"


def _lowering_receipt(lowered: Any) -> dict[str, Any]:
    text = lowered.as_text()
    return {
        "stablehlo_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "stablehlo_bytes": len(text.encode()),
        "has_custom_call": "custom_call" in text,
    }


def _promoted_matmul(x, y, **kwargs):
    def forward(a, b):
        return padded_pallas_matmul(a, b, **kwargs)

    return canonical_vjp_matmul(x, y, forward=forward)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seeds", type=int, default=4)
    args = parser.parse_args()
    if args.seeds < 1:
        raise RuntimeError("--seeds must be positive")

    preflight(require_enabled=True)
    devices = jax.devices()
    if jax.default_backend() != "tpu" or len(devices) != TP_SIZE:
        raise RuntimeError(
            f"P38 fixed lm_head requires exactly {TP_SIZE} TPU devices"
        )
    mesh = Mesh(np.asarray(devices), ("model",))
    replicated = NamedSharding(mesh, P(None, None))
    vocab_sharded = NamedSharding(mesh, P(None, "model"))
    weight = _load_weight(args.model, vocab_sharded)

    @partial(
        jax.jit,
        in_shardings=(replicated, vocab_sharded),
        out_shardings=vocab_sharded,
    )
    def fixed_head(hidden: jax.Array, kernel: jax.Array) -> jax.Array:
        return fixed_lm_head(
            hidden,
            kernel,
            mesh=mesh,
            tp_axis="model",
            local_matmul=_promoted_matmul,
        )

    @partial(
        jax.jit,
        in_shardings=(replicated, vocab_sharded),
        out_shardings=vocab_sharded,
    )
    def stock_head(hidden: jax.Array, kernel: jax.Array) -> jax.Array:
        return jnp.einsum("TD,DV->TV", hidden, kernel)

    compare = jax.jit(_different_elements)
    max_abs = jax.jit(_max_abs)
    rows: list[dict[str, Any]] = []
    lowerings: dict[str, dict[str, Any]] = {}
    last_fixed = None
    for seed in range(args.seeds):
        key = jax.random.PRNGKey(seed + 3802)
        hidden = jax.random.normal(key, (PREFILL_M, HIDDEN), dtype=jnp.float32)
        hidden = jax.device_put(hidden.astype(jnp.bfloat16), replicated)
        decode_hidden = hidden[:DECODE_M]
        if not lowerings:
            lowerings = {
                "fixed_decode": _lowering_receipt(
                    fixed_head.lower(decode_hidden, weight)
                ),
                "fixed_prefill": _lowering_receipt(
                    fixed_head.lower(hidden, weight)
                ),
            }

        fixed_decode = fixed_head(decode_hidden, weight)
        fixed_prefill = fixed_head(hidden, weight)
        stock_decode = stock_head(decode_hidden, weight)
        stock_prefill = stock_head(hidden, weight)
        for value in (fixed_decode, fixed_prefill, stock_decode, stock_prefill):
            value.block_until_ready()

        fixed_prefill_rows = fixed_prefill[:DECODE_M]
        stock_prefill_rows = stock_prefill[:DECODE_M]
        row = {
            "seed": seed,
            "fixed_differing_elements": int(
                compare(fixed_decode, fixed_prefill_rows)
            ),
            "fixed_max_abs": float(max_abs(fixed_decode, fixed_prefill_rows)),
            "stock_differing_elements": int(
                compare(stock_decode, stock_prefill_rows)
            ),
            "stock_max_abs": float(max_abs(stock_decode, stock_prefill_rows)),
            "decode_intervention_differing_elements": int(
                compare(stock_decode, fixed_decode)
            ),
            "prefill_intervention_differing_elements": int(
                compare(stock_prefill_rows, fixed_prefill_rows)
            ),
        }
        rows.append(row)
        print(
            f"[P38.FIXED_LM_HEAD] seed={seed} "
            f"{json.dumps(row, sort_keys=True)}",
            flush=True,
        )
        last_fixed = fixed_prefill_rows

    assert last_fixed is not None
    negative = jax.jit(_flip_one_bit)(last_fixed)
    negative.block_until_ready()
    negative_differing = int(compare(last_fixed, negative))
    verdict = classify(rows, negative_differing)
    report = {
        "schema_version": 1,
        "verdict": verdict,
        "claim_scope": "onehost-real-weight-fixed-lm-head-construction-only",
        "backend": jax.default_backend(),
        "device_count": len(devices),
        "semantic_m": [DECODE_M, PREFILL_M],
        "fixed_shape": [FIXED_M, HIDDEN, PADDED_LOCAL_VOCAB],
        "weight_shape": [HIDDEN, VOCAB],
        "local_vocab": LOCAL_VOCAB,
        "tiles": {"BM": BM, "BN": BN, "BK": BK},
        "weight_dtype": str(weight.dtype),
        "weight_sharding": str(weight.sharding),
        "negative_control_differing_elements": negative_differing,
        "lowerings": lowerings,
        "seeds": rows,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[P38.FIXED_LM_HEAD] {json.dumps(report, sort_keys=True)}", flush=True)
    if verdict != "FIXED_LM_HEAD_ONEHOST_CONSTRUCTION_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
