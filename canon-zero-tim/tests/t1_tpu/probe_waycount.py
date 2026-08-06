"""Admission probe: does the third-program drift appear at THIS topology's reduction width?

Background.  The drift that forces CANON_FIXED_AR / CANON_FIXED_AR_EMBED is not a property of
the device count, the mesh rank, or the device order.  Bisection on the v5p-8 probe host
isolated it to the WIDTH of a single reduction:

    reduction over 2 ways  ->  jit(f).primal == jit(value_and_grad(f)).primal   (SAME)
    reduction over 4 ways  ->  DIFFERS
    fully replicated       ->  SAME
    2x2 mesh, TP on one 2-wide axis          -> SAME
    2x2 mesh, TP over both axes (4 way)      -> DIFFERS

Only 2 and 4 were ever measured.  Any deployment whose tensor-parallel width is not 4 is
therefore running on an untested assumption.  This probe sweeps every width the visible
device count admits and reports, per width, whether the drift appears and whether the
fixed-order ppermute tree removes it.

Reads nothing, writes nothing, needs no model and no engine: a sharded MLP stack is enough
to reproduce the effect.  Run it FIRST on a new cluster -- it costs seconds and it decides
whether the canonical switch set transfers at all.

    python3 probe_waycount.py

Environment:
    CANON_WAYCOUNT_WIDTHS   comma-separated widths to test (default: all divisors of the
                            visible device count that are >= 2)
    CANON_WAYCOUNT_DEPTHS   comma-separated stack depths (default: 8,15,24)

XLA_FLAGS must contain --xla_allow_excess_precision=false, exactly as in a canonical run;
without it this probe measures a different program family than production.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

D, F, T = 512, 2048, 256
AXIS = "m"


def _divisors_ge2(n):
    return [w for w in range(2, n + 1) if n % w == 0]


def _rms(x, g):
    xf = x.astype(jnp.float32)
    return ((xf * jax.lax.rsqrt(jnp.mean(xf * xf, -1, keepdims=True) + 1e-6))
            * g.astype(jnp.float32)).astype(jnp.bfloat16)


def _make_fwd(mesh, n, fixed):
    """Transformer-like MLP stack whose row projection reduces over `n` ways."""

    def _tree(p):
        # fixed-order (((p0+p1)+...)+p_{n-1}) -- the F4 form.  Same addends as the XLA
        # all-reduce, but an order that does not depend on ring position or row chunk.
        parts = [p]
        cur = p
        for _ in range(n - 1):
            cur = jax.lax.ppermute(cur, AXIS, [(i, (i + 1) % n) for i in range(n)])
            parts.append(cur)
        j = jax.lax.axis_index(AXIS)
        ordered = jnp.stack(parts)[jnp.argsort((j - jnp.arange(n)) % n)]
        acc = ordered[0]
        for k in range(1, n):
            acc = acc + ordered[k]
        return acc

    def fwd(x, w):
        for p in w:
            h = _rms(x, p["g"])
            gate = jax.nn.silu(
                jnp.einsum("td,df->tf", h, p["Wg"]).astype(jnp.float32)).astype(jnp.bfloat16)
            act = gate * jnp.einsum("td,df->tf", h, p["Wu"])
            if fixed:
                def _loc(a_, wd_):
                    return _tree((a_ @ wd_)[None])[0]
                try:
                    sm = shard_map(_loc, mesh=mesh, in_specs=(P(None, AXIS), P(AXIS, None)),
                                   out_specs=P(None, None), check_vma=False)
                except TypeError:
                    sm = shard_map(_loc, mesh=mesh, in_specs=(P(None, AXIS), P(AXIS, None)),
                                   out_specs=P(None, None), check_rep=False)
                m = sm(act, p["Wd"])
            else:
                m = jnp.einsum("tf,fd->td", act, p["Wd"])
            x = x + m
        return x

    return fwd


def _differing_bytes(a, b):
    return int((np.ascontiguousarray(a).view(np.uint8)
                != np.ascontiguousarray(b).view(np.uint8)).sum())


def main():
    flags = os.environ.get("XLA_FLAGS", "")
    if "--xla_allow_excess_precision=false" not in flags:
        print("[waycount] REFUSING: XLA_FLAGS lacks --xla_allow_excess_precision=false; "
              "this probe would measure a different program family than a canonical run.",
              file=sys.stderr)
        return 2

    devs = jax.devices()
    nd = len(devs)
    widths_env = os.environ.get("CANON_WAYCOUNT_WIDTHS", "")
    widths = ([int(w) for w in widths_env.split(",") if w.strip()]
              if widths_env else _divisors_ge2(nd))
    depths = [int(d) for d in os.environ.get("CANON_WAYCOUNT_DEPTHS", "8,15,24").split(",")]

    print(f"[waycount] devices={nd} kind={devs[0].device_kind} "
          f"widths={widths} depths={depths}", flush=True)
    if not widths:
        print("[waycount] REFUSING: no reduction width >= 2 is available "
              f"({nd} device(s) visible)", file=sys.stderr)
        return 2

    rng = np.random.default_rng(0)
    rows = 0
    for n in widths:
        if n > nd:
            print(f"[waycount] SKIP width={n}: exceeds {nd} visible devices", flush=True)
            continue
        mesh = Mesh(mesh_utils.create_device_mesh((n,), devs[:n]), (AXIS,))
        put = lambda a, s: jax.device_put(jnp.asarray(a, jnp.bfloat16), NamedSharding(mesh, s))
        for L in depths:
            w = [dict(g=put(rng.normal(size=(D,)) * 0.1 + 1.0, P(None)),
                      Wg=put(rng.normal(size=(D, F)) * 0.02, P(None, AXIS)),
                      Wu=put(rng.normal(size=(D, F)) * 0.02, P(None, AXIS)),
                      Wd=put(rng.normal(size=(F, D)) * 0.02, P(AXIS, None))) for _ in range(L)]
            x0 = put(rng.normal(size=(T, D)) * 0.5, P(None, None))
            for fixed in (False, True):
                fwd = _make_fwd(mesh, n, fixed)
                a = np.asarray(jax.device_get(jax.jit(fwd)(x0, w)))

                def loss(x, ww, _f=fwd):
                    y = _f(x, ww)
                    return jnp.sum(y.astype(jnp.float32)), y

                (_, bd), _ = jax.jit(
                    jax.value_and_grad(loss, argnums=0, has_aux=True))(x0, w)
                b = np.asarray(jax.device_get(bd))
                nb = _differing_bytes(a, b)
                arm = "F4-fixed-order" if fixed else "XLA-all-reduce"
                print(f"[waycount] width={n:2d} depth={L:3d} {arm:15s} "
                      f"differing_bytes={nb:7d}/{a.nbytes} "
                      f"{'SAME' if nb == 0 else 'DIFFERS'}", flush=True)
                rows += 1

    expected = sum(len(depths) * 2 for n in widths if n <= nd)
    print(f"[waycount] measurements={rows} expected={expected}", flush=True)
    if rows != expected:
        print("[waycount] VERDICT: INCONCLUSIVE -- measurement count mismatch", flush=True)
        return 1
    print("[waycount] VERDICT: COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
