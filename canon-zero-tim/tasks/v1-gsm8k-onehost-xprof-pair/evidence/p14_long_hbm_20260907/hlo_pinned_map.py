"""Phase 14: does the K7-pinned two-mesh mapped program carry entry/exit copies?

Builds the pinned map on the CPU two-mesh fixture of
tests/rl/test_p59_pinned_map_shardings.py, lowers it, and tallies the
compiled HLO's copy / collective instructions against the relabel path
(the same mapped program compiled without pinned shardings, fed
engine-spelled operands).
"""
import os, sys, re, collections
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=16").strip()
sys.path.insert(0, "/mnt/disks/tunix-data/worktrees/v2_dispatch_0906/tests/rl")
sys.path.insert(0, "/mnt/disks/tunix-data/worktrees/v2_dispatch_0906")
os.environ["CANON_P66_P59_CHECK_VMA"] = "0"
import jax, jax.numpy as jnp, numpy as np
import test_p59_pinned_map_shardings as t
from tunix.rl import canonical_qwen3_adapter as A

def closure_var(fn, name):
    return fn.__closure__[fn.__code__.co_freevars.index(name)].cell_contents

def tally(text):
    ops = collections.Counter()
    for line in text.splitlines():
        m = re.match(r"\s*(?:ROOT )?%?[\w.\-]+ = \S+ (\w[\w\-]*)\(", line)
        if m:
            ops[m.group(1)] += 1
    return ops

trainer, engine = t._meshes()
invoke, args = t._build(trainer, engine)
compiled = closure_var(invoke, "compiled")
mesh = closure_var(invoke, "mesh")
module_name = closure_var(invoke, "module_name")
print("pinned:", invoke._p59_pinned, "mesh:", mesh.axis_names)
with A._p59_localize_engine_shard_maps(mesh, module_name):
    lowered = compiled.lower(*args)
text = lowered.compile().as_text()
ops = tally(text)
print("pinned program ops:", {k: v for k, v in ops.items() if k in ("copy", "all-to-all", "collective-permute", "all-gather", "reduce-scatter", "all-reduce", "dynamic-slice", "dynamic-update-slice")})
print("  total instructions:", sum(ops.values()))
# entry copies: copies whose operand is a parameter
entry_copies = [l.strip()[:120] for l in text.splitlines() if re.search(r"= \S+ copy\(", l) and "parameter" in l]
print("  copies of parameters:", len(entry_copies))
for l in [l.strip()[:140] for l in text.splitlines() if re.search(r" copy\(", l)][:12]:
    print("   ", l)

# relabel path: same mapped fn compiled without pinning, operands in engine spelling
mapped = closure_var(invoke, "mapped")
aligned = tuple(A._p59_align_to_mesh(v, mesh, module_name) for v in args)
plain = jax.jit(mapped)
with A._p59_localize_engine_shard_maps(mesh, module_name):
    text2 = plain.lower(*aligned).compile().as_text()
ops2 = tally(text2)
print("relabel-path program ops:", {k: v for k, v in ops2.items() if k in ("copy", "all-to-all", "collective-permute", "all-gather", "reduce-scatter", "all-reduce", "dynamic-slice", "dynamic-update-slice")})
print("  total instructions:", sum(ops2.values()))
