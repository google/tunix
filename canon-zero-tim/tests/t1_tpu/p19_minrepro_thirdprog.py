"""Minimal reproducer: jit(f) primal  vs  jax.value_and_grad(f) primal, as a function of depth.

Claim under test (no JAX/XLA API guarantee is being asserted -- see JAX FAQ / OpenXLA semantics):
on TPU, for a sharded transformer-like stack, the primal returned by value_and_grad becomes
bitwise-DIFFERENT from the standalone jitted forward once the stack exceeds a depth threshold,
even with --xla_allow_excess_precision=false.  Small dims so each depth costs seconds.
"""
import os, sys

if "--pathways_enforce_subset_devices_form_subslice=false" not in sys.argv:
    sys.argv.append("--pathways_enforce_subset_devices_form_subslice=false")

try:
    import pathwaysutils  # noqa: F401
except ImportError:
    pass

import jax, jax.numpy as jnp, numpy as np
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

D, F, T = 512, 2048, 256
devs = jax.devices(); N = len(devs)
mesh = Mesh(mesh_utils.create_device_mesh((N,), devs), ("m",))
rng = np.random.default_rng(0)
put = lambda a, s: jax.device_put(jnp.asarray(a, jnp.bfloat16), NamedSharding(mesh, s))

def make_weights(L):
    w = []
    for _ in range(L):
        w.append(dict(g=put(rng.normal(size=(D,))*0.1+1.0, P(None)),
                      Wg=put(rng.normal(size=(D, F))*0.02, P(None, "m")),
                      Wu=put(rng.normal(size=(D, F))*0.02, P(None, "m")),
                      Wd=put(rng.normal(size=(F, D))*0.02, P("m", None))))
    return w

def rms(x, g):
    xf = x.astype(jnp.float32)
    return ((xf * jax.lax.rsqrt(jnp.mean(xf*xf, -1, keepdims=True) + 1e-6))
            * g.astype(jnp.float32)).astype(jnp.bfloat16)

def fwd(x, w):
    for p in w:
        h = rms(x, p["g"])
        m = jnp.einsum("tf,fd->td",
                       jax.nn.silu(jnp.einsum("td,df->tf", h, p["Wg"]).astype(jnp.float32)
                                   ).astype(jnp.bfloat16) * jnp.einsum("td,df->tf", h, p["Wu"]),
                       p["Wd"])
        x = x + m
    return x

x0 = put(rng.normal(size=(T, D))*0.5, P(None, None))
print(f"[minrepro] devices={N} dims T={T} D={D} F={F} "
      f"XLA_FLAGS={os.environ.get('XLA_FLAGS','')!r}", flush=True)
for L in (4, 8, 12, 14, 15, 16, 20, 24):
    w = make_weights(L)
    f_plain = jax.jit(lambda x, w: fwd(x, w))
    def loss(x, w):
        y = fwd(x, w)
        return jnp.sum(y.astype(jnp.float32)), y
    f_vg = jax.jit(jax.value_and_grad(loss, argnums=0, has_aux=True))
    a = np.asarray(jax.device_get(f_plain(x0, w)))
    (_, b_dev), _g = f_vg(x0, w)
    b = np.asarray(jax.device_get(b_dev))
    nb = int((np.ascontiguousarray(a).view(np.uint8) != np.ascontiguousarray(b).view(np.uint8)).sum())
    mx = float(np.abs(a.astype(np.float32) - b.astype(np.float32)).max())
    print(f"[minrepro] L={L:3d}: primal差异 {nb:6d}/{a.nbytes} bytes  max|Δ|={mx:.3e}  "
          f"{'SAME' if nb == 0 else '<<<<< DIFFERS'}", flush=True)
