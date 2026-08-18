#!/usr/bin/env python3
"""P38.2x fixed-shape Pallas construction for Qwen3-8B TP4 lm_head."""

from __future__ import annotations

import os
from collections.abc import Callable


ENV = "CANON_P38_FIXED_LM_HEAD"
# Pinned tpu_inference request buckets for this max-concurrency-256 target plus
# the one learner chunk shape exercised by canonical_qwen3_adapter. Request
# buckets are padded to FIXED_M; the learner shape is mapped as exact FIXED_M
# chunks. Non-registered row counts remain fail-closed instead of retracing.
REQUEST_M = (8, 16, 32, 64, 128, 256)
LEARNER_M = (4096,)
SEMANTIC_M = REQUEST_M + LEARNER_M
FIXED_M = 256
HIDDEN = 4096
VOCAB = 151936
TP_SIZE = 4
LOCAL_VOCAB = VOCAB // TP_SIZE
PADDED_LOCAL_VOCAB = 38144
BM = 128
BN = 256
BK = 256

REQUIRED = {
    "CANON_FIXED_AR": "1",
    "CANON_FIXED_AR_EMBED": "1",
    "CANON_PALLAS_ALL_PROJ": "1",
    "CANON_PALLAS_ALL_RMSNORM": "1",
    "CANON_PALLAS_SWIGLU": "1",
    "CANON_PALLAS_MPAD": "1",
    "CANON_PALLAS_SWIGLU_MPAD": "1",
    "CANON_PALLAS_CANONICAL_VJP": "1",
}
CONFLICTS = (
    "CANON_MM_ALGO",
    "CANON_PALLAS_MATMUL",
    "CANON_PALLAS_MATERIALIZE",
    "CANON_POSTRPA_M",
    "CANON_CUT",
    "CANON_TAIL",
)


def preflight(*, require_enabled: bool) -> None:
    value = os.environ.get(ENV, "")
    if value not in ("", "0", "1"):
        raise RuntimeError(f"{ENV} must be unset, 0, or 1, got {value!r}")
    if require_enabled and value != "1":
        raise RuntimeError(f"{ENV}=1 required")
    if value != "1":
        return
    wrong = [
        f"{name}={os.environ.get(name)!r}"
        for name, expected in REQUIRED.items()
        if os.environ.get(name, "") != expected
    ]
    if wrong:
        raise RuntimeError(
            "P38 fixed lm_head canonical dependencies missing: " + ", ".join(wrong)
        )
    active = [name for name in CONFLICTS if os.environ.get(name, "")]
    if active:
        raise RuntimeError(
            "P38 fixed lm_head conflicting diagnostics: " + ",".join(active)
        )


def validate_global_contract(
    input_shape,
    weight_shape,
    input_dtype,
    weight_dtype,
    *,
    tp_size: int,
) -> int:
    """Validate the caller-global Qwen3-8B contract and return semantic M."""
    input_shape = tuple(map(int, input_shape))
    weight_shape = tuple(map(int, weight_shape))
    if len(input_shape) != 2 or input_shape[0] not in SEMANTIC_M:
        raise ValueError(
            f"P38 fixed lm_head requires semantic M in {SEMANTIC_M}, got {input_shape}"
        )
    if input_shape[1] != HIDDEN or weight_shape != (HIDDEN, VOCAB):
        raise ValueError(
            "P38 fixed lm_head requires input/weight "
            f"[(M,{HIDDEN}),({HIDDEN},{VOCAB})], got {input_shape}/{weight_shape}"
        )
    if str(input_dtype) != "bfloat16" or str(weight_dtype) != "bfloat16":
        raise ValueError(
            "P38 fixed lm_head requires bf16 input/weight, got "
            f"{input_dtype}/{weight_dtype}"
        )
    if int(tp_size) != TP_SIZE:
        raise ValueError(
            f"P38 fixed lm_head requires TP{TP_SIZE}, got TP{int(tp_size)}"
        )
    return input_shape[0]


def validate_local_contract(input_shape, weight_shape) -> int:
    """Validate one shard_map local shard and return semantic M."""
    input_shape = tuple(map(int, input_shape))
    weight_shape = tuple(map(int, weight_shape))
    if len(input_shape) != 2 or input_shape[0] not in SEMANTIC_M:
        raise ValueError(f"P38 fixed lm_head local M invalid: {input_shape}")
    if input_shape[1] != HIDDEN or weight_shape != (HIDDEN, LOCAL_VOCAB):
        raise ValueError(
            "P38 fixed lm_head local shape mismatch: "
            f"{input_shape}/{weight_shape}, expected (M,{HIDDEN})/"
            f"({HIDDEN},{LOCAL_VOCAB})"
        )
    return input_shape[0]


def fixed_lm_head(
    inputs,
    weight,
    *,
    mesh,
    tp_axis: str,
    local_matmul: Callable,
):
    """Run every registered outer shape through one fixed Pallas shape."""
    from jax import lax
    import jax.numpy as jnp
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    preflight(require_enabled=True)
    if mesh is None:
        raise RuntimeError("P38 fixed lm_head requires the live model mesh")
    if tp_axis not in mesh.shape:
        raise RuntimeError(
            f"P38 fixed lm_head mesh lacks axis {tp_axis!r}: {mesh.shape}"
        )
    tp_size = int(mesh.shape[tp_axis])
    semantic_m = validate_global_contract(
        inputs.shape,
        weight.shape,
        inputs.dtype,
        weight.dtype,
        tp_size=tp_size,
    )

    def local(a_local, w_local):
        local_m = validate_local_contract(a_local.shape, w_local.shape)
        if a_local.dtype != jnp.bfloat16 or w_local.dtype != jnp.bfloat16:
            raise ValueError(
                "P38 fixed lm_head local dtype mismatch: "
                f"{a_local.dtype}/{w_local.dtype}"
            )
        def run_fixed(a_fixed):
            return local_matmul(
                a_fixed,
                w_local,
                block_m=BM,
                block_n=BN,
                block_k=BK,
                shape_invariant_numerics=True,
            )

        if local_m < FIXED_M:
            a_fixed = jnp.pad(
                a_local,
                ((0, FIXED_M - local_m), (0, 0)),
                constant_values=0,
            )
            chunks = 1
            out = run_fixed(a_fixed)[:local_m, :]
        elif local_m == FIXED_M:
            chunks = 1
            out = run_fixed(a_local)
        else:
            chunks = local_m // FIXED_M
            a_chunks = a_local.reshape((chunks, FIXED_M, HIDDEN))
            out = lax.map(run_fixed, a_chunks).reshape(
                (local_m, LOCAL_VOCAB)
            )
        print(
            "[PATHTRACE] CANON_P38_FIXED_LM_HEAD=1 "
            f"semantic_M={local_m} fixed_M={FIXED_M} K={HIDDEN} "
            f"local_N={LOCAL_VOCAB} fixed_N={PADDED_LOCAL_VOCAB} "
            f"BM={BM} BN={BN} BK={BK} chunks={chunks}",
            flush=True,
        )
        if tuple(map(int, out.shape)) != (local_m, LOCAL_VOCAB):
            raise RuntimeError(
                f"P38 fixed lm_head output shape mismatch: {out.shape}"
            )
        return out

    try:
        mapped = shard_map(
            local,
            mesh=mesh,
            in_specs=(P(None, None), P(None, tp_axis)),
            out_specs=P(None, tp_axis),
            check_vma=False,
        )
    except TypeError:
        mapped = shard_map(
            local,
            mesh=mesh,
            in_specs=(P(None, None), P(None, tp_axis)),
            out_specs=P(None, tp_axis),
            check_rep=False,
        )
    output = mapped(inputs, weight)
    if tuple(map(int, output.shape)) != (semantic_m, VOCAB):
        raise RuntimeError(
            f"P38 fixed lm_head global output mismatch: {output.shape}"
        )
    return output
