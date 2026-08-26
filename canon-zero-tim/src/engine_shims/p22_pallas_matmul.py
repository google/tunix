#!/usr/bin/env python3
"""P22.XE fixed-tile Pallas TPU matmul.

Additive and default-off.  Gate 0 uses Pallas interpret with exact power-of-two
known answers; TPU/HLO and real-model integration are separate gates.
"""

from __future__ import annotations

import argparse
import hashlib
import os


ENV = "CANON_PALLAS_MATMUL"
CONFLICTS = (
    "CANON_PALLAS_MATERIALIZE",
    "CANON_CUT",
    "CANON_TAIL",
    "CANON_POSTRPA_M",
)
BM = 128
BN = 256
BK = 256


def preflight(*, require_enabled: bool) -> None:
    value = os.environ.get(ENV, "")
    if value not in ("", "1"):
        raise SystemExit(f"P22.XE preflight: {ENV} must be unset or 1, got {value!r}")
    if require_enabled and value != "1":
        raise SystemExit(f"P22.XE preflight: {ENV}=1 required")
    if value == "1" and os.environ.get("CANON_FIXED_AR", "") != "1":
        raise SystemExit("P22.XE preflight: CANON_FIXED_AR=1 required")
    if value == "1":
        active = [name for name in CONFLICTS if os.environ.get(name, "")]
        if active:
            raise SystemExit("P22.XE preflight: conflicting diagnostics: " + ",".join(active))


def _imports():
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    return jax, jnp, pl, pltpu


def p66_vma_align_operands(jax, *values):
    """Give operands of a local Pallas operation one common VMA type.

    JAX dot primitives require all operands to have matching varying manual
    axes.  A pcast from replicated to varying is a runtime identity; its
    transpose supplies the psum that the replicated operand needs.  P66 only
    admits plain-varying state here, never an already reduced/unreduced value.
    """
    if os.environ.get("CANON_P66_P59_CHECK_VMA", "0") != "1":
        return values
    mats = tuple(jax.typeof(value).mat for value in values)
    if any(mat.unreduced or mat.reduced for mat in mats):
        raise ValueError(
            "P66 Pallas VMA does not admit reduced/unreduced operands: "
            + ", ".join(str(mat) for mat in mats)
        )
    varying = frozenset().union(*(mat.varying for mat in mats))
    aligned = []
    for value, mat in zip(values, mats, strict=True):
        for axis in sorted(varying - mat.varying):
            value = jax.lax.pcast(value, axis, to="varying")
        aligned.append(value)
    return tuple(aligned)


def p66_vma_output_manual_axis_type(jax, *values):
    """Return the explicit Pallas output VMA type for aligned operands."""
    if os.environ.get("CANON_P66_P59_CHECK_VMA", "0") != "1":
        return None
    mats = tuple(jax.typeof(value).mat for value in values)
    if any(mat.unreduced or mat.reduced for mat in mats):
        raise ValueError(
            "P66 Pallas VMA output does not admit reduced/unreduced operands: "
            + ", ".join(str(mat) for mat in mats)
        )
    varying = frozenset().union(*(mat.varying for mat in mats))
    return jax.sharding.ManualAxisType(varying=varying)


def matmul(
    x,
    y,
    *,
    interpret: bool = False,
    shape_invariant_numerics: bool = True,
    block_m: int = BM,
    block_n: int = BN,
    block_k: int = BK,
):
    """Compute bf16 [M,K] @ [K,N] with fixed BM/BN/BK and f32 accumulation."""
    jax, jnp, pl, pltpu = _imports()
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(f"P22.XE expects rank-2 inputs, got {x.shape}, {y.shape}")
    m, k = map(int, x.shape)
    ky, n = map(int, y.shape)
    if k != ky:
        raise ValueError(f"P22.XE contracted dimensions differ: {k} vs {ky}")
    if x.dtype != jnp.bfloat16 or y.dtype != jnp.bfloat16:
        raise ValueError(f"P22.XE requires bf16 inputs, got {x.dtype}, {y.dtype}")
    if min(block_m, block_n, block_k) <= 0:
        raise ValueError("P22.XE block sizes must be positive")
    if m % block_m or n % block_n or k % block_k:
        raise ValueError(
            "P22.XE shape must divide BM/BN/BK="
            f"{block_m}/{block_n}/{block_k}, got {(m, k, n)}"
        )
    x, y = p66_vma_align_operands(jax, x, y)

    def _kernel(x_ref, y_ref, out_ref, acc_ref):
        @pl.when(pl.program_id(2) == 0)
        def _init():
            acc_ref[...] = jnp.zeros_like(acc_ref)

        acc_ref[...] = acc_ref[...] + jnp.dot(
            x_ref[...], y_ref[...], preferred_element_type=jnp.float32
        )
        out_ref[...] = acc_ref[...].astype(out_ref.dtype)

    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct(
            (m, n),
            jnp.bfloat16,
            manual_axis_type=p66_vma_output_manual_axis_type(jax, x, y),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[
                pl.BlockSpec((block_m, block_k), lambda i, _j, q: (i, q)),
                pl.BlockSpec((block_k, block_n), lambda _i, j, q: (q, j)),
            ],
            out_specs=pl.BlockSpec((block_m, block_n), lambda i, j, _q: (i, j)),
            grid=(m // block_m, n // block_n, k // block_k),
            scratch_shapes=[pltpu.VMEM((block_m, block_n), jnp.float32)],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            # P56.4.7: producer fusion changes materialization, not
            # values -- the fused producers are elementwise-exact and
            # the kernel reads identical operand values either way.
            allow_input_fusion=(
                (True, True)
                if os.environ.get("CANON_PALLAS_INPUT_FUSION", "") == "1"
                else (False, False)
            ),
            shape_invariant_numerics=shape_invariant_numerics,
        ),
        interpret=interpret,
        name=f"canon_matmul_bm{block_m}_bn{block_n}_bk{block_k}",
    )(x, y)


def _bits(array) -> bytes:
    import numpy as np

    return np.asarray(array).tobytes(order="C")


def _assert_equal(label: str, actual, expected) -> None:
    import numpy as np

    a = np.asarray(actual)
    b = np.asarray(expected)
    if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
        differing = -1
        if a.shape == b.shape and a.dtype == b.dtype:
            differing = int((a.view(np.uint8) != b.view(np.uint8)).sum())
        raise AssertionError(f"{label}: arrays differ; differing_bytes={differing}")


def _known_inputs(m: int, k: int, n: int, *, filler_sign: float = 1.0):
    """Power-of-two bf16 values whose products/sums are exactly representable in f32."""
    import numpy as np

    rows = np.arange(m, dtype=np.int32)[:, None]
    inner = np.arange(k, dtype=np.int32)[None, :]
    x = (((rows * 3 + inner * 5) % 5) - 2).astype(np.float32) / 16.0
    if m > 128 and filler_sign != 1.0:
        x[128:] *= np.float32(filler_sign)
    inner_y = np.arange(k, dtype=np.int32)[:, None]
    cols = np.arange(n, dtype=np.int32)[None, :]
    y = (((inner_y * 7 + cols * 11) % 5) - 2).astype(np.float32) / 16.0
    return x, y


def _expected_bf16(x, y):
    _jax, jnp, _pl, _pltpu = _imports()
    import numpy as np

    # Products are multiples of 2^-8 and K<=512 here, so float64 is an exact
    # mathematical oracle before the one final bf16 rounding.
    exact = np.asarray(x, np.float64) @ np.asarray(y, np.float64)
    return np.asarray(jnp.asarray(exact, dtype=jnp.bfloat16))


def interpret_selftest() -> None:
    preflight(require_enabled=True)
    _jax, jnp, _pl, _pltpu = _imports()
    import numpy as np

    shapes = ((256, 256, 256), (512, 256, 256), (256, 512, 512), (512, 512, 512))
    outputs = {}
    expected_count = len(shapes)
    completed = 0
    for m, k, n in shapes:
        x_np, y_np = _known_inputs(m, k, n)
        x = jnp.asarray(x_np, dtype=jnp.bfloat16)
        y = jnp.asarray(y_np, dtype=jnp.bfloat16)
        out = np.asarray(matmul(x, y, interpret=True))
        expected = _expected_bf16(x_np, y_np)
        _assert_equal(f"known-answer MKN={m},{k},{n}", out, expected)
        outputs[(m, k, n)] = out
        completed += 1
        print(
            f"P22.XE.INTERPRET M={m} K={k} N={n} exact=1 "
            f"sha256={hashlib.sha256(_bits(out)).hexdigest()}",
            flush=True,
        )
    if completed != expected_count:
        raise AssertionError(f"P22.XE expected={expected_count} completed={completed}")

    for k, n in ((256, 256), (512, 512)):
        _assert_equal(
            f"cross-M shared rows K={k} N={n}",
            outputs[(256, k, n)][:128],
            outputs[(512, k, n)][:128],
        )

    # Pad-content control: only rows >=128 differ; shared rows must not.
    x1_np, y_np = _known_inputs(512, 256, 256, filler_sign=1.0)
    x2_np, _ = _known_inputs(512, 256, 256, filler_sign=-1.0)
    y = jnp.asarray(y_np, dtype=jnp.bfloat16)
    out1 = np.asarray(matmul(jnp.asarray(x1_np, jnp.bfloat16), y, interpret=True))
    out2 = np.asarray(matmul(jnp.asarray(x2_np, jnp.bfloat16), y, interpret=True))
    _assert_equal("pad-content shared rows", out1[:128], out2[:128])
    print("P22.XE.INTERPRET pad-content shared_rows_exact=1", flush=True)

    negative_count = 0
    try:
        matmul(jnp.zeros((64, 256), jnp.bfloat16),
               jnp.zeros((256, 256), jnp.bfloat16), interpret=True)
    except ValueError:
        negative_count += 1

    corrupted = outputs[(256, 256, 256)].view(np.uint16).copy()
    corrupted[0, 0] ^= np.uint16(1)
    try:
        _assert_equal("one-bit negative", corrupted.view(outputs[(256, 256, 256)].dtype),
                      outputs[(256, 256, 256)])
    except AssertionError:
        negative_count += 1

    try:
        _assert_equal("wrong crop negative", outputs[(256, 256, 256)][:-1],
                      outputs[(256, 256, 256)])
    except AssertionError:
        negative_count += 1

    if negative_count != 3:
        raise AssertionError(f"P22.XE negative expected=3 completed={negative_count}")
    print(
        f"P22.XE.INTERPRET PASS measurements={completed}/{expected_count} negatives=3/3",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interpret-selftest", action="store_true")
    args = parser.parse_args()
    if args.interpret_selftest:
        interpret_selftest()
        return
    raise SystemExit("choose --interpret-selftest")


if __name__ == "__main__":
    main()
