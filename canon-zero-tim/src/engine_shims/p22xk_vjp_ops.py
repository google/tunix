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
