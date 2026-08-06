"""Mesh geometry is the switch: 1D-4dev TP DIFFERS, 2x2 TP SAME.  Confirm across depth &
mesh construction (plain reshape vs mesh_utils.create_device_mesh)."""
import os as _os
import sys as _sys

if "--pathways_enforce_subset_devices_form_subslice=false" not in _sys.argv:
    _sys.argv.append("--pathways_enforce_subset_devices_form_subslice=false")

try:
    import pathwaysutils
    pathwaysutils.initialize()
except Exception:
    pass

import jax, jax.numpy as jnp, numpy as np
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
D, F, T = 512, 2048, 256
devs = jax.devices()
rng = np.random.default_rng(0)
def rms(x, g):
    xf = x.astype(jnp.float32)
    return ((xf*jax.lax.rsqrt(jnp.mean(xf*xf,-1,keepdims=True)+1e-6))*g.astype(jnp.float32)).astype(jnp.bfloat16)
def fwd(x, w):
    for p in w:
        h = rms(x, p["g"])
        gate = jax.nn.silu(jnp.einsum("td,df->tf", h, p["Wg"]).astype(jnp.float32)).astype(jnp.bfloat16)
        x = x + jnp.einsum("tf,fd->td", gate*jnp.einsum("td,df->tf", h, p["Wu"]), p["Wd"])
    return x
def check(tag, mesh, sg, sd, L):
    put = lambda a, s: jax.device_put(jnp.asarray(a, jnp.bfloat16), NamedSharding(mesh, s))
    w = [dict(g=put(rng.normal(size=(D,))*0.1+1.0, P(None)),
              Wg=put(rng.normal(size=(D,F))*0.02, sg), Wu=put(rng.normal(size=(D,F))*0.02, sg),
              Wd=put(rng.normal(size=(F,D))*0.02, sd)) for _ in range(L)]
    x0 = put(rng.normal(size=(T,D))*0.5, P(None,None))
    a = np.asarray(jax.device_get(jax.jit(fwd)(x0, w)))
    def loss(x, w):
        y = fwd(x, w); return jnp.sum(y.astype(jnp.float32)), y
    (_, bd), _ = jax.jit(jax.value_and_grad(loss, argnums=0, has_aux=True))(x0, w)
    b = np.asarray(jax.device_get(bd))
    nb = int((np.ascontiguousarray(a).view(np.uint8)!=np.ascontiguousarray(b).view(np.uint8)).sum())
    print(f"[m2d] L={L:3d} {tag:44s} 差异 {nb:6d} {'SAME ***' if nb==0 else 'DIFFERS'}", flush=True)

m1_plain = Mesh(np.array(devs[:4]).reshape(4), ("m",))
m1_cdm   = Mesh(mesh_utils.create_device_mesh((4,), devs[:4]), ("m",))
m22_plain= Mesh(np.array(devs[:4]).reshape(2,2), ("a","b"))
m22_cdm  = Mesh(mesh_utils.create_device_mesh((2,2), devs[:4]), ("a","b"))
for L in (8, 15, 24, 32):
    check("1D-4dev(plain reshape) TP", m1_plain, P(None,"m"), P("m",None), L)
    check("1D-4dev(create_device_mesh) TP", m1_cdm, P(None,"m"), P("m",None), L)
    check("2x2(plain) TP on b", m22_plain, P(None,"b"), P("b",None), L)
    check("2x2(cdm) TP on b", m22_cdm, P(None,"b"), P("b",None), L)
    check("2x2(plain) 2D-TP on (a,b)", m22_plain, P(None,("a","b")), P(("a","b"),None), L)
