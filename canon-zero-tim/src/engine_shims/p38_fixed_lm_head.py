#!/usr/bin/env python3
"""Fixed-shape Pallas construction for registered Qwen3 output heads."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass


ENV = "CANON_P38_FIXED_LM_HEAD"
# Pinned tpu_inference request buckets for this max-concurrency-256 target plus
# the one learner chunk shape exercised by canonical_qwen3_adapter. Request
# buckets are padded to FIXED_M; the learner shape is mapped as exact FIXED_M
# chunks. Non-registered row counts remain fail-closed instead of retracing.
REQUEST_M = (8, 16, 32, 64, 128, 256)
LEARNER_M = (4096,)
SEMANTIC_M = REQUEST_M + LEARNER_M
FIXED_M = 256
HIDDEN = 4096  # Historical Qwen3-8B default retained for old probes.
VOCAB = 151936
BM = 128
BN = 256
BK = 256


@dataclass(frozen=True)
class Geometry:
    """One model/topology/output-endpoint executable contract."""

    model: str
    hidden: int
    tp_size: int
    endpoint: str
    local_vocab: int
    padded_local_vocab: int


def _geometry(
    model: str, hidden: int, tp_size: int, endpoint: str
) -> Geometry:
    local_vocab = VOCAB // tp_size
    padded_local_vocab = ((local_vocab + BN - 1) // BN) * BN
    return Geometry(
        model=model,
        hidden=hidden,
        tp_size=tp_size,
        endpoint=endpoint,
        local_vocab=local_vocab,
        padded_local_vocab=padded_local_vocab,
    )


GEOMETRIES = {
    (2048, 4): _geometry("qwen3-1p7b", 2048, 4, "tied_embed"),
    (4096, 4): _geometry("qwen3-8b", 4096, 4, "untied_lm_head"),
    (4096, 8): _geometry("qwen3-8b-tp8", 4096, 8, "untied_lm_head"),
    (2560, 8): _geometry("qwen3-4b", 2560, 8, "tied_embed"),
    (5120, 8): _geometry("qwen3-32b", 5120, 8, "untied_lm_head"),
}
SUPPORTED_HIDDEN = tuple(sorted({hidden for hidden, _tp in GEOMETRIES}))
# Historical aliases retained for the TP4 probes and evidence readers. New
# code must resolve a registered Geometry instead of consulting these aliases.
TP_SIZE = 4
LOCAL_VOCAB = VOCAB // TP_SIZE
PADDED_LOCAL_VOCAB = ((LOCAL_VOCAB + BN - 1) // BN) * BN

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
VJP_PASS = "FIXED_LM_HEAD_ONEHOST_VJP_PASS"
ENDPOINTS = ("untied_lm_head", "tied_embed", "direct_probe")


def _validate_hidden(hidden: int) -> int:
    hidden = int(hidden)
    if hidden not in SUPPORTED_HIDDEN:
        raise ValueError(
            "P38 fixed lm_head requires hidden size in "
            f"{SUPPORTED_HIDDEN}, got {hidden}"
        )
    configured = os.environ.get("CANON_QWEN3_HIDDEN_SIZE", "")
    if configured:
        try:
            configured_hidden = int(configured)
        except ValueError as error:
            raise ValueError(
                "P38 fixed lm_head requires an integer "
                f"CANON_QWEN3_HIDDEN_SIZE, got {configured!r}"
            ) from error
        if configured_hidden != hidden:
            raise ValueError(
                "P38 fixed lm_head hidden size disagrees with the profile: "
                f"shape={hidden} env={configured_hidden}"
            )
    return hidden


def resolve_geometry(
    hidden: int, tp_size: int, *, endpoint: str | None = None
) -> Geometry:
    """Return one registered geometry and reject model/topology drift."""
    hidden = _validate_hidden(hidden)
    tp_size = int(tp_size)
    try:
        geometry = GEOMETRIES[(hidden, tp_size)]
    except KeyError as error:
        registered = ", ".join(
            f"K{item.hidden}/TP{item.tp_size}/{item.endpoint}"
            for item in GEOMETRIES.values()
        )
        raise ValueError(
            "P38 fixed lm_head model/topology is not registered: "
            f"K{hidden}/TP{tp_size}; registered={registered}"
        ) from error
    configured_tp = os.environ.get("CANON_QWEN3_TP_SIZE", "")
    if configured_tp:
        try:
            parsed_tp = int(configured_tp)
        except ValueError as error:
            raise ValueError(
                "P38 fixed lm_head requires an integer "
                f"CANON_QWEN3_TP_SIZE, got {configured_tp!r}"
            ) from error
        if parsed_tp != geometry.tp_size:
            raise ValueError(
                "P38 fixed lm_head TP size disagrees with the profile: "
                f"mesh={geometry.tp_size} env={parsed_tp}"
            )
    if endpoint is not None and endpoint not in (geometry.endpoint, "direct_probe"):
        raise ValueError(
            "P38 fixed lm_head endpoint disagrees with the registered model: "
            f"model={geometry.model} expected={geometry.endpoint} got={endpoint}"
        )
    return geometry


def classify_vjp(
    *,
    hidden_differing: int,
    weight_differing: int,
    repeat_hidden_differing: int,
    repeat_weight_differing: int,
    gradients_finite: bool,
    gradients_nonzero: bool,
    negative_differing: int,
) -> str:
    """Return one fail-closed verdict from compact fixed-lm-head VJP metrics."""
    if negative_differing != 1:
        return "FAIL_NEGATIVE_CONTROL"
    if not gradients_finite:
        return "FAIL_NONFINITE_GRADIENT"
    if not gradients_nonzero:
        return "INCONCLUSIVE_NO_GRADIENT_SIGNAL"
    if repeat_hidden_differing or repeat_weight_differing:
        return "FIXED_LM_HEAD_VJP_NOT_DETERMINISTIC"
    if hidden_differing or weight_differing:
        return "FIXED_LM_HEAD_CHUNK_VJP_NOT_INVARIANT"
    return VJP_PASS


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
    """Validate one caller-global Qwen3 contract and return semantic M."""
    input_shape = tuple(map(int, input_shape))
    weight_shape = tuple(map(int, weight_shape))
    if len(input_shape) != 2 or input_shape[0] not in SEMANTIC_M:
        raise ValueError(
            f"P38 fixed lm_head requires semantic M in {SEMANTIC_M}, got {input_shape}"
        )
    hidden = _validate_hidden(input_shape[1])
    if weight_shape != (hidden, VOCAB):
        raise ValueError(
            "P38 fixed lm_head requires input/weight "
            f"[(M,{hidden}),({hidden},{VOCAB})], got {input_shape}/{weight_shape}"
        )
    if str(input_dtype) != "bfloat16" or str(weight_dtype) != "bfloat16":
        raise ValueError(
            "P38 fixed lm_head requires bf16 input/weight, got "
            f"{input_dtype}/{weight_dtype}"
        )
    resolve_geometry(hidden, tp_size)
    return input_shape[0]


def validate_local_contract(
    input_shape,
    weight_shape,
    *,
    tp_size: int = TP_SIZE,
    admitted_m: tuple[int, ...] = SEMANTIC_M,
) -> int:
    """Validate one shard_map local shard and return semantic M."""
    input_shape = tuple(map(int, input_shape))
    weight_shape = tuple(map(int, weight_shape))
    admitted_m = tuple(map(int, admitted_m))
    if len(input_shape) != 2 or input_shape[0] not in admitted_m:
        raise ValueError(f"P38 fixed lm_head local M invalid: {input_shape}")
    hidden = _validate_hidden(input_shape[1])
    geometry = resolve_geometry(hidden, tp_size)
    if weight_shape != (hidden, geometry.local_vocab):
        raise ValueError(
            "P38 fixed lm_head local shape mismatch: "
            f"{input_shape}/{weight_shape}, expected (M,{hidden})/"
            f"({hidden},{geometry.local_vocab})"
        )
    return input_shape[0]


def _p59_local_contract(inputs, weight, *, mesh, tp_axis: str):
    """Resolve one already-DP/TP-mapped P59 head call or return None.

    P59's outer shard_map has already sliced both the DP row dimension and the
    TP vocabulary dimension. Re-entering the engine's concrete shard_map is
    illegal and would partition the TP-local weight twice. This admission is
    deliberately structural: ordinary serving with the P59 flag present has
    no two-axis manual outer context and therefore stays on the global path.
    """
    if os.environ.get("CANON_P59_RANK_PARALLEL_BACKWARD", "") != "1":
        return None

    import jax
    import jax.numpy as jnp

    context = jax.sharding.get_abstract_mesh()
    if tuple(context.axis_names) != ("data", "model"):
        return None
    axis_types = dict(zip(context.axis_names, context.axis_types))
    if (
        axis_types.get("data") is not jax.sharding.AxisType.Manual
        or axis_types.get("model") is not jax.sharding.AxisType.Manual
    ):
        return None
    if tp_axis != "model" or tp_axis not in mesh.shape or "data" not in mesh.shape:
        raise RuntimeError(
            "P59 local fixed lm_head requires engine data/model axes"
        )
    dp_size = int(context.shape["data"])
    tp_size = int(context.shape["model"])
    if dp_size <= 1 or tp_size <= 1:
        raise RuntimeError(
            "P59 local fixed lm_head requires non-unit DP and TP"
        )
    if (
        int(mesh.shape["data"]) != dp_size
        or int(mesh.shape[tp_axis]) != tp_size
    ):
        raise RuntimeError(
            "P59 local fixed lm_head context and engine topology differ"
        )
    input_shape = tuple(map(int, inputs.shape))
    weight_shape = tuple(map(int, weight.shape))
    if len(input_shape) != 2:
        raise ValueError(
            f"P59 local fixed lm_head input rank changed: {input_shape}"
        )
    hidden_size = _validate_hidden(input_shape[1])
    geometry = resolve_geometry(hidden_size, tp_size)
    if weight_shape != (hidden_size, geometry.local_vocab):
        raise ValueError(
            "P59 local fixed lm_head weight shape mismatch: "
            f"{weight_shape} != {(hidden_size, geometry.local_vocab)}"
        )
    if inputs.dtype != jnp.bfloat16 or weight.dtype != jnp.bfloat16:
        raise ValueError(
            "P59 local fixed lm_head requires bf16 input/weight, got "
            f"{inputs.dtype}/{weight.dtype}"
        )
    global_m = tuple(
        candidate
        for candidate in LEARNER_M
        if candidate % dp_size == 0 and candidate // dp_size == input_shape[0]
    )
    if len(global_m) != 1:
        raise ValueError(
            "P59 local fixed lm_head rows do not reconstruct one learner M: "
            f"local_M={input_shape[0]} dp={dp_size} learner_M={LEARNER_M}"
        )
    return dp_size, global_m[0], geometry


def fixed_lm_head(
    inputs,
    weight,
    *,
    mesh,
    tp_axis: str,
    local_matmul: Callable,
    endpoint: str,
):
    """Run every registered outer shape through one fixed Pallas shape."""
    import jax
    from jax import lax
    import jax.numpy as jnp
    from jax.experimental.shard_map import shard_map
    from jax.sharding import PartitionSpec as P

    preflight(require_enabled=True)
    if endpoint not in ENDPOINTS:
        raise ValueError(
            f"P38 fixed lm_head endpoint must be one of {ENDPOINTS}, got "
            f"{endpoint!r}"
        )
    if mesh is None:
        raise RuntimeError("P38 fixed lm_head requires the live model mesh")
    if tp_axis not in mesh.shape:
        raise RuntimeError(
            f"P38 fixed lm_head mesh lacks axis {tp_axis!r}: {mesh.shape}"
        )
    tp_size = int(mesh.shape[tp_axis])
    p59_local = _p59_local_contract(
        inputs, weight, mesh=mesh, tp_axis=tp_axis
    )
    if p59_local is None:
        semantic_m = validate_global_contract(
            inputs.shape,
            weight.shape,
            inputs.dtype,
            weight.dtype,
            tp_size=tp_size,
        )
        p59_dp_size = 0
        p59_global_m = semantic_m
    else:
        p59_dp_size, p59_global_m, _ = p59_local
        semantic_m = int(inputs.shape[0])
    hidden_size = _validate_hidden(inputs.shape[1])
    geometry = resolve_geometry(hidden_size, tp_size, endpoint=endpoint)

    def local(a_local, w_local):
        local_m = validate_local_contract(
            a_local.shape,
            w_local.shape,
            tp_size=tp_size,
            admitted_m=(semantic_m,),
        )
        if a_local.dtype != jnp.bfloat16 or w_local.dtype != jnp.bfloat16:
            raise ValueError(
                "P38 fixed lm_head local dtype mismatch: "
                f"{a_local.dtype}/{w_local.dtype}"
            )
        def run_fixed_with_weight(a_fixed, weight_local):
            return local_matmul(
                a_fixed,
                weight_local,
                block_m=BM,
                block_n=BN,
                block_k=BK,
                shape_invariant_numerics=True,
            )

        def run_fixed(a_fixed):
            return run_fixed_with_weight(a_fixed, w_local)

        def learner_forward(a_learner, weight_local):
            a_chunks = a_learner.reshape((-1, FIXED_M, hidden_size))
            return lax.map(
                lambda a_chunk: run_fixed_with_weight(a_chunk, weight_local),
                a_chunks,
            ).reshape((a_learner.shape[0], geometry.local_vocab))

        @jax.custom_vjp
        def learner_fixed_vjp(a_learner, weight_local):
            return learner_forward(a_learner, weight_local)

        def learner_fwd(a_learner, weight_local):
            output = learner_forward(a_learner, weight_local)
            return output, (a_learner, weight_local)

        def learner_bwd(residual, cotangent):
            print(
                "[PATHTRACE] CANON_P38_FIXED_LM_HEAD_VJP=1 "
                "semantic_M=4096 fixed_M=256 chunks=16 "
                "accumulation=lax.scan order=ascending "
                f"K={hidden_size} TP={tp_size} "
                f"local_N={geometry.local_vocab} "
                f"fixed_N={geometry.padded_local_vocab} "
                f"endpoint={endpoint}",
                flush=True,
            )
            a_learner, weight_local = residual
            a_chunks = a_learner.reshape((-1, FIXED_M, hidden_size))
            cotangent_chunks = cotangent.reshape(
                (-1, FIXED_M, geometry.local_vocab)
            )

            def accumulate(weight_cotangent, values):
                a_chunk, output_cotangent = values
                _, pullback = jax.vjp(
                    run_fixed_with_weight, a_chunk, weight_local
                )
                a_cotangent, chunk_weight_cotangent = pullback(
                    output_cotangent
                )
                # The loop-carried dependency is the backward contract: chunk
                # q is added only after chunks [0, q) have completed.
                return weight_cotangent + chunk_weight_cotangent, a_cotangent

            weight_cotangent, a_cotangent_chunks = lax.scan(
                accumulate,
                jnp.zeros_like(weight_local),
                (a_chunks, cotangent_chunks),
            )
            return (
                a_cotangent_chunks.reshape(a_learner.shape),
                weight_cotangent,
            )

        learner_fixed_vjp.defvjp(learner_fwd, learner_bwd)

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
            out = learner_fixed_vjp(a_local, w_local)
        p59_receipt = (
            f" p59_local=1 global_M={p59_global_m} dp={p59_dp_size}"
            if p59_local is not None
            else ""
        )
        print(
            "[PATHTRACE] CANON_P38_FIXED_LM_HEAD=1 "
            f"semantic_M={local_m} fixed_M={FIXED_M} K={hidden_size} "
            f"TP={tp_size} local_N={geometry.local_vocab} "
            f"fixed_N={geometry.padded_local_vocab} "
            f"BM={BM} BN={BN} BK={BK} chunks={chunks} "
            f"endpoint={endpoint}{p59_receipt}",
            flush=True,
        )
        if tuple(map(int, out.shape)) != (local_m, geometry.local_vocab):
            raise RuntimeError(
                f"P38 fixed lm_head output shape mismatch: {out.shape}"
            )
        return out

    if p59_local is not None:

        @jax.custom_vjp
        def p59_local_head(a_local, w_local):
            return local(a_local, w_local)

        def p59_local_head_fwd(a_local, w_local):
            output_local = local(a_local, w_local)
            return output_local, (a_local, w_local)

        def p59_local_head_bwd(residual, cotangent):
            a_local, w_local = residual
            _, pullback = jax.vjp(local, a_local, w_local)
            da_local, dw_local = pullback(cotangent)
            gathered = lax.all_gather(
                da_local.astype(jnp.float32),
                tp_axis,
                axis=0,
                tiled=False,
            )
            da = gathered[0]
            for rank in range(1, tp_size):
                da = (
                    lax.optimization_barrier(da)
                    + lax.optimization_barrier(gathered[rank])
                )
            da = da.astype(da_local.dtype)
            print(
                "[PATHTRACE] CANON_" "P38_FIXED_LM_HEAD_VJP=1 "
                f"semantic_M={p59_global_m} local_M={semantic_m} "
                f"fixed_M={FIXED_M} chunks={semantic_m // FIXED_M} "
                "accumulation=lax.scan order=ascending "
                "tp_input_reduction=all_gather_rank_order_f32_barrier "
                f"K={hidden_size} "
                f"TP={tp_size} local_N={geometry.local_vocab} "
                f"fixed_N={geometry.padded_local_vocab} "
                f"endpoint={endpoint}",
                flush=True,
            )
            return da, dw_local

        p59_local_head.defvjp(p59_local_head_fwd, p59_local_head_bwd)
        output = p59_local_head(inputs, weight)
    else:
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
    p59_local_tp = p59_local is not None
    expected_output_shape = (
        (semantic_m, geometry.local_vocab)
        if p59_local_tp
        else (semantic_m, VOCAB)
    )
    if tuple(map(int, output.shape)) != expected_output_shape:
        raise RuntimeError(
            "P38 fixed lm_head output boundary mismatch: "
            f"{output.shape} != {expected_output_shape}"
        )
    return output
