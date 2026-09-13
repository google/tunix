#!/usr/bin/env python3
"""Canonical custom-VJPs whose primals remain the promoted Pallas forwards."""

from __future__ import annotations

from p22xk_contract import preflight


def canonical_matmul(x, y):
    """Pure-JAX replica of the model-pinned padded-BK accumulation order."""
    import jax.numpy as jnp
    from p22xf_contract import BK, BN
    from p22xi_padded_matmul import padded_matmul_extents

    if x.ndim != 2 or y.ndim != 2 or int(x.shape[1]) != int(y.shape[0]):
        raise ValueError(f"P22.XK matmul shapes invalid: {x.shape}, {y.shape}")
    if x.dtype != jnp.bfloat16 or y.dtype != jnp.bfloat16:
        raise ValueError(f"P22.XK matmul requires bf16, got {x.dtype}, {y.dtype}")
    m, k, n = int(x.shape[0]), int(x.shape[1]), int(y.shape[1])
    kp, _ = padded_matmul_extents(k, n, block_k=BK, block_n=BN)
    if kp != k:
        x = jnp.pad(x, ((0, 0), (0, kp - k)), constant_values=0)
        y = jnp.pad(y, ((0, kp - k), (0, 0)), constant_values=0)
    acc = jnp.zeros((m, n), jnp.float32)
    for q in range(kp // BK):
        lo, hi = q * BK, (q + 1) * BK
        acc = acc + jnp.dot(
            x[:, lo:hi], y[lo:hi, :], preferred_element_type=jnp.float32
        )
    return acc.astype(jnp.bfloat16)


def canonical_swiglu(gate, up):
    import jax
    import jax.numpy as jnp

    if gate.shape != up.shape or gate.dtype != jnp.bfloat16 or up.dtype != jnp.bfloat16:
        raise ValueError(f"P22.XK SwiGLU contract failed: {gate.shape}/{up.shape} "
                         f"{gate.dtype}/{up.dtype}")
    return (jax.nn.silu(gate) * up).astype(jnp.bfloat16)


# tasks/zero_tim_perf3 E2d.  The coats below differentiate canonical_matmul,
# the 16-K-block replica of the forward's accumulation order, so every
# projection's backward is 16 sliced dots whose weight cotangents are padded
# back into full-size buffers and summed: in the 2026-09-11 one-host 8B
# update window that structure was the add_bitcast_fusion bf16[1,4096,6144]/
# [1,6144,4096] family (21.5% of jit_zt_tr_bwd_chunk) plus the
# fusion bf16[256,256] sea (4032 per launch, 5%).  The backward does not
# need the forward's K-blocking: dX = cot . W^T and dW = X^T . cot as two
# plain f32-accumulating dots give the same gradient up to summation order.
# The plain dots are the code default (tasks/zero_tim_perf3 Phase H);
# CANON_MATMUL_VJP_PLAIN=0 keeps the K-block replica pullback as the explicit
# A/B path until the tp8 64-chip wave has exercised the default.
PLAIN_VJP_ENV = "CANON_MATMUL_VJP_PLAIN"
_PLAIN_RECEIPT = set()


def plain_matmul_vjp_enabled():
    """Default on (tasks/zero_tim_perf3 Phase H): the projection pullbacks are
    JAX's transposes of one plain bf16 dot.  ``CANON_MATMUL_VJP_PLAIN=0`` keeps
    the K-block replica pullback as an explicit A/B path until the plain
    pullback has been exercised at tp8 on the 64-chip wave."""
    import os

    value = os.environ.get(PLAIN_VJP_ENV, "1")
    if value not in ("0", "1"):
        raise ValueError(f"{PLAIN_VJP_ENV} must be unset, 0 or 1, got {value!r}")
    if value not in _PLAIN_RECEIPT:
        _PLAIN_RECEIPT.add(value)
        print(f"[PATHTRACE] {PLAIN_VJP_ENV}={value} projection VJPs use "
              + ("plain bf16 dots (dX = cot.W^T, dW = X^T.cot; default)" if value == "1"
                 else "the K-block replica pullback (explicit A/B path)"), flush=True)
    return value == "1"


def plain_matmul(x, y):
    """One plain bf16 dot (f32 MXU accumulation, rounded once to bf16).

    Only its transpose is ever executed: the E2d pullback is JAX's own
    transpose of this dot, two plain bf16 dots.
    """
    import jax.numpy as jnp

    return jnp.dot(x, y, preferred_element_type=jnp.bfloat16)


def plain_matmul_pullback(a, b, cotangent):
    """dX, dW of X @ W through JAX's transpose of one plain dot.

    Going through jax.vjp instead of writing the two dots by hand keeps the
    manual-axis typing of the canonical path: under check_vma=True the
    replicated projection input is implicitly cast to varying over TP inside
    the dot, and JAX transposes that cast into the TP psum, so dX comes back
    invariant over model like the primal input.  A hand-written dot(cot, W^T)
    is typed varying over model and carries only this rank's partial sum
    (ztp3_e2d_v_20260911_r1: "Custom VJP bwd rule must produce an output with
    the same type as the args tuple ... bfloat16[256,4096]{V:(data,model)}
    corresponding to an input of type bfloat16[256,4096]{V:data}").
    """
    import jax

    _, pullback = jax.vjp(plain_matmul, a, b)
    return pullback(cotangent)


def matmul(x, y, *, forward):
    """Use ``forward`` verbatim for primal and canonical_matmul only for VJP."""
    import jax

    preflight(require_enabled=True)

    @jax.custom_vjp
    def op(a, b):
        return forward(a, b)

    def fwd(a, b):
        return forward(a, b), (a, b)

    def bwd(residual, cotangent):
        a, b = residual
        if plain_matmul_vjp_enabled():
            return plain_matmul_pullback(a, b, cotangent)
        _, pullback = jax.vjp(canonical_matmul, a, b)
        return pullback(cotangent)

    op.defvjp(fwd, bwd)
    return op(x, y)


def swiglu(gate, up, *, forward):
    """Use promoted padded SwiGLU for primal and canonical_swiglu for VJP."""
    import jax

    preflight(require_enabled=True)

    @jax.custom_vjp
    def op(g, u):
        return forward(g, u)

    def fwd(g, u):
        return forward(g, u), (g, u)

    def bwd(residual, cotangent):
        g, u = residual
        _, pullback = jax.vjp(canonical_swiglu, g, u)
        return pullback(cotangent)

    op.defvjp(fwd, bwd)
    return op(gate, up)


def norm_matmul(x, gamma, y, *, epsilon: float, forward):
    """P56.4.6 coat: fused Pallas primal, composed canonical-replica VJP.

    The backward differentiates canonical_matmul(canonical_rmsnorm(.)) --
    the same two replicas the separate XK coats differentiate, composed
    by the chain rule in the same order the un-fused path applies them.
    canonical_rmsnorm is the declared bit-exact semantic of the Pallas
    norm, so the recomputed intermediate matches the stored one.
    """
    import jax
    from p22_pallas_rmsnorm import canonical_rmsnorm

    preflight(require_enabled=True)

    def oracle(a, g, b):
        return canonical_matmul(canonical_rmsnorm(a, g, epsilon=epsilon), b)

    @jax.custom_vjp
    def op(a, g, b):
        return forward(a, g, b)

    def fwd(a, g, b):
        return forward(a, g, b), (a, g, b)

    def bwd(residual, cotangent):
        a, g, b = residual
        if plain_matmul_vjp_enabled():
            h, norm_pullback = jax.vjp(
                lambda a_, g_: canonical_rmsnorm(a_, g_, epsilon=epsilon), a, g
            )
            dh, db = plain_matmul_pullback(h, b, cotangent)
            da, dg = norm_pullback(dh)
            return da, dg, db
        _, pullback = jax.vjp(oracle, a, g, b)
        return pullback(cotangent)

    op.defvjp(fwd, bwd)
    return op(x, gamma, y)


def rmsnorm(x, weight, *, epsilon: float, forward):
    """Use promoted Pallas RMSNorm for primal and fixed-BF replica for VJP."""
    import jax
    from p22_pallas_rmsnorm import canonical_rmsnorm

    preflight(require_enabled=True)

    def oracle(a, w):
        return canonical_rmsnorm(a, w, epsilon=epsilon)

    @jax.custom_vjp
    def op(a, w):
        return forward(a, w)

    def fwd(a, w):
        return forward(a, w), (a, w)

    def bwd(residual, cotangent):
        a, w = residual
        _, pullback = jax.vjp(oracle, a, w)
        return pullback(cotangent)

    op.defvjp(fwd, bwd)
    return op(x, weight)
