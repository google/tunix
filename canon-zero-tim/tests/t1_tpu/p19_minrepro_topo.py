"""Device-count / mesh-geometry bisection.  2 devices SAME, 4 devices DIFFERS (even replicated)."""
import os as _os

from pathways_bootstrap import initialize_pathways

initialize_pathways()

import jax, jax.numpy as jnp, numpy as np
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
D, F, T, L = 512, 2048, 256, 8
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
def check(tag, mesh, spec_g, spec_d):
    put = lambda a, s: jax.device_put(jnp.asarray(a, jnp.bfloat16), NamedSharding(mesh, s))
    w = [dict(g=put(rng.normal(size=(D,))*0.1+1.0, P(None)),
              Wg=put(rng.normal(size=(D,F))*0.02, spec_g),
              Wu=put(rng.normal(size=(D,F))*0.02, spec_g),
              Wd=put(rng.normal(size=(F,D))*0.02, spec_d)) for _ in range(L)]
    x0 = put(rng.normal(size=(T,D))*0.5, P(None,None))
    a = np.asarray(jax.device_get(jax.jit(fwd)(x0, w)))
    def loss(x, w):
        y = fwd(x, w); return jnp.sum(y.astype(jnp.float32)), y
    (_, bd), _ = jax.jit(jax.value_and_grad(loss, argnums=0, has_aux=True))(x0, w)
    b = np.asarray(jax.device_get(bd))
    nb = int((np.ascontiguousarray(a).view(np.uint8) != np.ascontiguousarray(b).view(np.uint8)).sum())
    print(f"[topo] {tag:48s} differing_bytes={nb:6d} "
          f"{'SAME ***' if nb==0 else 'DIFFERS'}", flush=True)

for nd in (2, 3, 4):
    try:
        m = Mesh(np.array(devs[:nd]).reshape(nd), ("m",))
        check(f"{nd}-device 1D mesh, TP-sharded", m, P(None,"m"), P("m",None))
        check(f"{nd}-device 1D mesh, replicated", m, P(None,None), P(None,None))
    except Exception as e:
        print(f"[topo] {nd}-device: EXC {type(e).__name__}: {str(e)[:70]}", flush=True)
try:
    m22 = Mesh(np.array(devs[:4]).reshape(2,2), ("a","b"))
    check("4-device 2x2 mesh, TP on axis b", m22, P(None,"b"), P("b",None))
    check("4-device 2x2 mesh, replicated", m22, P(None,None), P(None,None))
except Exception as e:
    print(f"[topo] 2x2: EXC {type(e).__name__}: {str(e)[:70]}", flush=True)
# Device-order sensitivity (the C6a echo).
try:
    m_perm = Mesh(np.array([devs[0],devs[2],devs[1],devs[3]]).reshape(4), ("m",))
    check("4-device 1D mesh, order [0,2,1,3], TP", m_perm, P(None,"m"), P("m",None))
except Exception as e:
    print(f"[topo] perm: EXC {type(e).__name__}: {str(e)[:70]}", flush=True)
