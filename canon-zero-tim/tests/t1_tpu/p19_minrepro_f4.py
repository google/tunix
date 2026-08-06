"""Does an F4-style fixed-order reduction make the 4-way case bitwise across programs?
Baseline: 1D-4dev TP (XLA's own all-reduce) -> DIFFERS.  Arm: same math, but the row-proj
reduction done as an explicit fixed-order ppermute tree inside shard_map."""
import os as _os
import sys as _sys

_os.environ["FLAGS_pathways_enforce_subset_devices_form_subslice"] = "false"
_os.environ["PATHWAYS_ENFORCE_SUBSET_DEVICES_FORM_SUBSLICE"] = "false"
if "--FLAGS_pathways_enforce_subset_devices_form_subslice=false" not in _sys.argv:
    _sys.argv.append("--FLAGS_pathways_enforce_subset_devices_form_subslice=false")
if "--pathways_enforce_subset_devices_form_subslice=false" not in _sys.argv:
    _sys.argv.append("--pathways_enforce_subset_devices_form_subslice=false")

try:
    import pathwaysutils
    pathwaysutils.initialize()
except Exception:
    pass

import jax, jax.numpy as jnp, numpy as np
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
D, F, T = 512, 2048, 256
import os as _os
# N was hardcoded to 4 (the v5p-8 probe host).  The reduction WIDTH is the variable that
# decides whether the third-program drift appears, so a package targeting other topologies
# must be able to set it.  Default keeps the documented 4-way behaviour on a 4-chip host.
devs = jax.devices()
N = int(_os.environ.get("CANON_MINREPRO_N") or min(4, len(devs)))
assert N <= len(devs), f"CANON_MINREPRO_N={N} exceeds {len(devs)} visible devices"
mesh = Mesh(mesh_utils.create_device_mesh((N,), devs[:N]), ("m",))
rng = np.random.default_rng(0)
put = lambda a, s: jax.device_put(jnp.asarray(a, jnp.bfloat16), NamedSharding(mesh, s))
def rms(x, g):
    xf = x.astype(jnp.float32)
    return ((xf*jax.lax.rsqrt(jnp.mean(xf*xf,-1,keepdims=True)+1e-6))*g.astype(jnp.float32)).astype(jnp.bfloat16)

def _tree(p):                                  # fixed-order (((p0+p1)+p2)+p3), F4 form
    parts=[p]; cur=p
    for _ in range(N-1):
        cur = jax.lax.ppermute(cur, "m", [(i,(i+1)%N) for i in range(N)]); parts.append(cur)
    j = jax.lax.axis_index("m")
    ordered = jnp.stack(parts)[jnp.argsort((j - jnp.arange(N)) % N)]
    acc = ordered[0]
    for k in range(1, N): acc = acc + ordered[k]
    return acc

def make(fixed):
    def fwd(x, w):
        for p in w:
            h = rms(x, p["g"])
            gate = jax.nn.silu(jnp.einsum("td,df->tf", h, p["Wg"]).astype(jnp.float32)).astype(jnp.bfloat16)
            act = gate * jnp.einsum("td,df->tf", h, p["Wu"])
            if fixed:
                def _loc(a_, wd_):
                    return _tree((a_ @ wd_)[None])[0]
                try:
                    sm = shard_map(_loc, mesh=mesh, in_specs=(P(None,"m"), P("m",None)),
                                   out_specs=P(None,None), check_vma=False)
                except TypeError:
                    sm = shard_map(_loc, mesh=mesh, in_specs=(P(None,"m"), P("m",None)),
                                   out_specs=P(None,None), check_rep=False)
                m = sm(act, p["Wd"])
            else:
                m = jnp.einsum("tf,fd->td", act, p["Wd"])
            x = x + m
        return x
    return fwd

for L in (8, 15, 24):
    w = [dict(g=put(rng.normal(size=(D,))*0.1+1.0, P(None)),
              Wg=put(rng.normal(size=(D,F))*0.02, P(None,"m")),
              Wu=put(rng.normal(size=(D,F))*0.02, P(None,"m")),
              Wd=put(rng.normal(size=(F,D))*0.02, P("m",None))) for _ in range(L)]
    x0 = put(rng.normal(size=(T,D))*0.5, P(None,None))
    for fixed in (False, True):
        fwd = make(fixed)
        a = np.asarray(jax.device_get(jax.jit(fwd)(x0, w)))
        def loss(x, w):
            y = fwd(x, w); return jnp.sum(y.astype(jnp.float32)), y
        (_, bd), _ = jax.jit(jax.value_and_grad(loss, argnums=0, has_aux=True))(x0, w)
        b = np.asarray(jax.device_get(bd))
        nb = int((np.ascontiguousarray(a).view(np.uint8)!=np.ascontiguousarray(b).view(np.uint8)).sum())
        print(f"[f4] L={L:3d} {'F4 固定序树' if fixed else 'XLA all-reduce':18s} "
              f"差异 {nb:6d}/{a.nbytes} {'SAME ***' if nb==0 else 'DIFFERS'}", flush=True)
