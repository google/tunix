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
    LEARNER_M,
    LOCAL_VOCAB,
    PADDED_LOCAL_VOCAB,
    REQUEST_M,
    TP_SIZE,
    VOCAB,
    fixed_lm_head,
    preflight,
)
from probe_p38_lm_head import (
    _different_elements,
    _flip_one_bit,
    _load_weight,
    _max_abs,
)


def classify(
    rows: list[dict[str, Any]],
    negative_differing: int,
    learner_rows: list[dict[str, Any]] | None = None,
) -> str:
    if negative_differing != 1:
        return "FAIL_NEGATIVE_CONTROL"
    if any(row["fixed_differing_elements"] for row in rows):
        return "FIXED_LM_HEAD_NOT_INVARIANT"
    if any(row["differing_elements"] for row in (learner_rows or [])):
        return "FIXED_LM_HEAD_LEARNER_CHUNK_NOT_INVARIANT"
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
    parser.add_argument("--learner-seeds", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN)
    args = parser.parse_args()
    if args.seeds < 1 or args.learner_seeds < 1:
        raise RuntimeError("--seeds and --learner-seeds must be positive")

    preflight(require_enabled=True)
    devices = jax.devices()
    if jax.default_backend() != "tpu" or len(devices) != TP_SIZE:
        raise RuntimeError(
            f"P38 fixed lm_head requires exactly {TP_SIZE} TPU devices"
        )
    mesh = Mesh(np.asarray(devices), ("model",))
    replicated = NamedSharding(mesh, P(None, None))
    vocab_sharded = NamedSharding(mesh, P(None, "model"))
    hidden_size = int(args.hidden_size)
    weight_source = (
        "model.embed_tokens.weight"
        if hidden_size == 2048
        else "lm_head.weight"
    )
    weight = _load_weight(
        args.model, vocab_sharded, hidden_size, weight_source
    )

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
            endpoint="direct_probe",
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
        hidden = jax.random.normal(
            key, (FIXED_M, hidden_size), dtype=jnp.float32
        )
        hidden = jax.device_put(hidden.astype(jnp.bfloat16), replicated)
        bucket_hidden = {m: hidden[:m] for m in REQUEST_M}
        if not lowerings:
            lowerings = {
                f"fixed_m{m}": _lowering_receipt(
                    fixed_head.lower(bucket_hidden[m], weight)
                )
                for m in REQUEST_M
            }

        fixed_outputs = {
            m: fixed_head(bucket_hidden[m], weight) for m in REQUEST_M
        }
        stock_decode = stock_head(bucket_hidden[16], weight)
        stock_prefill = stock_head(hidden, weight)
        for value in (*fixed_outputs.values(), stock_decode, stock_prefill):
            value.block_until_ready()

        fixed_reference = fixed_outputs[FIXED_M]
        fixed_bucket_differences = {
            str(m): int(compare(fixed_outputs[m], fixed_reference[:m]))
            for m in REQUEST_M
        }
        fixed_bucket_max_abs = {
            str(m): float(max_abs(fixed_outputs[m], fixed_reference[:m]))
            for m in REQUEST_M
        }
        fixed_prefill_rows = fixed_reference[:16]
        stock_prefill_rows = stock_prefill[:16]
        row = {
            "seed": seed,
            "fixed_bucket_differing_elements": fixed_bucket_differences,
            "fixed_bucket_max_abs": fixed_bucket_max_abs,
            "fixed_differing_elements": sum(fixed_bucket_differences.values()),
            "fixed_max_abs": max(fixed_bucket_max_abs.values()),
            "stock_differing_elements": int(
                compare(stock_decode, stock_prefill_rows)
            ),
            "stock_max_abs": float(max_abs(stock_decode, stock_prefill_rows)),
            "decode_intervention_differing_elements": int(
                compare(stock_decode, fixed_outputs[16])
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

    learner_rows: list[dict[str, Any]] = []
    learner_m = LEARNER_M[0]
    for seed in range(args.learner_seeds):
        key = jax.random.PRNGKey(seed + 3817)
        hidden = jax.random.normal(
            key, (learner_m, hidden_size), dtype=jnp.float32
        )
        hidden = jax.device_put(hidden.astype(jnp.bfloat16), replicated)
        if "fixed_m4096" not in lowerings:
            lowerings["fixed_m4096"] = _lowering_receipt(
                fixed_head.lower(hidden, weight)
            )
        learner_output = fixed_head(hidden, weight)
        learner_output.block_until_ready()
        differing = 0
        largest = 0.0
        for chunk in range(learner_m // FIXED_M):
            start = chunk * FIXED_M
            stop = start + FIXED_M
            reference = fixed_head(hidden[start:stop], weight)
            reference.block_until_ready()
            differing += int(compare(learner_output[start:stop], reference))
            largest = max(
                largest,
                float(max_abs(learner_output[start:stop], reference)),
            )
        row = {
            "seed": seed,
            "semantic_m": learner_m,
            "chunks": learner_m // FIXED_M,
            "differing_elements": differing,
            "max_abs": largest,
        }
        learner_rows.append(row)
        print(
            f"[P38.FIXED_LM_HEAD] learner_seed={seed} "
            f"{json.dumps(row, sort_keys=True)}",
            flush=True,
        )

    assert last_fixed is not None
    negative = jax.jit(_flip_one_bit)(last_fixed)
    negative.block_until_ready()
    negative_differing = int(compare(last_fixed, negative))
    verdict = classify(rows, negative_differing, learner_rows)
    report = {
        "schema_version": 2,
        "verdict": verdict,
        "claim_scope": "onehost-real-weight-fixed-lm-head-construction-only",
        "backend": jax.default_backend(),
        "device_count": len(devices),
        "request_m": list(REQUEST_M),
        "learner_m": list(LEARNER_M),
        "fixed_shape": [FIXED_M, hidden_size, PADDED_LOCAL_VOCAB],
        "weight_shape": [hidden_size, VOCAB],
        "weight_source": weight_source,
        "local_vocab": LOCAL_VOCAB,
        "tiles": {"BM": BM, "BN": BN, "BK": BK},
        "weight_dtype": str(weight.dtype),
        "weight_sharding": str(weight.sharding),
        "negative_control_differing_elements": negative_differing,
        "lowerings": lowerings,
        "seeds": rows,
        "learner_seeds": learner_rows,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"[P38.FIXED_LM_HEAD] {json.dumps(report, sort_keys=True)}", flush=True)
    if verdict != "FIXED_LM_HEAD_ONEHOST_CONSTRUCTION_PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
