# Cluster admission

What to measure before trusting this package on a topology it has never run on — and why each
measurement is not optional.

**Everything signed in `EVIDENCE.md` was measured on one configuration**: a directly-attached
4-chip v5p host, single slice, tensor-parallel width 4, no Pathways. The *method* transfers —
enumerate the degrees of freedom that split a program, pin each one. The *settings* do not.
Scaling up does not enlarge the old problems; it introduces new degrees of freedom.

Run in this order. It is sorted by cost, cheapest first, and by likelihood of going red — the
two happen to agree.

---

## Step 1 — Is this the engine the patches were cut against? (`probe-only`, no TPU)

```
[probe] SUMMARY same=6 drift=0 missing=0
[rope]  ROPE_FIX=not_needed | applied
```

A drift here does not merely risk a failed patch. It means the produced files can no longer be
byte-identical to the ones carrying the signed evidence, so a pass proves the chain builds —
not that it reproduces anything. Decide deliberately; record the override.

## Step 2 — Does the chain load here? (`install-only`, no TPU)

```
[verify] A. byte identity of overlay targets -> 6x OK
[verify] B. live import -> P22XK_MATMUL_ACTIVE=True, P22XK_RMSNORM_ACTIVE=True, ...
```

This is the check that catches a shim chain whose sibling resolution landed somewhere empty —
the failure that otherwise presents as a completely green run against the stock engine.

## Step 3 — Separate platform drift from the production operator path (`gate-only`, minutes of TPU)

`tests/t1_tpu/probe_waycount.py` is a generic diagnostic.  It deliberately uses a handwritten
RMSNorm/einsum/MLP chain, so it can show that Pathways has a forward-only versus gradient-program
carrier without proving that the promoted production Qwen operators have the same carrier.

`tests/t1_tpu/probe_canonical_ops.py` is the hard admission gate.  It imports the live overlay and
calls the exact P22.XK RMSNorm, gate/up projections, SwiGLU, down projection and F4 reduction at
the installed model dimensions.  It uses the full production `(replica, model)` mesh and
differentiates the weights, not a convenient no-weight surrogate.

Before accepting any numerical row, require exactly one marker from every JAX-based probe:

```
[T1.PATHWAYS] required=1 initialized=1 status=ok
```

In proxy mode, a missing marker or `initialized=0` voids the topology run.  The shared bootstrap
executes before JAX import and exits nonzero if proxy registration fails; it never silently falls
back to a directly attached backend.

The old four-device probe established that TP reduction width can carry a third-program
difference. It did **not** establish that reduction is the only carrier on Pathways. The first
64-device attempt made this distinction mandatory: a replicated two-device diagnostic arm was
also dirty, so reduction was not necessary for that observation.

The release probe now uses every visible device. For TP width `w`, it constructs the
topology-aware mesh `(num_devices / w, w)` with axes `(replica, tp)`. On 64 devices this means
`(32,2)`, the production `(16,4)`, and the future-facing diagnostic `(8,8)`, not an invalid
multi-host subset such as `devices[:2]` or
`devices[:4]`. Require these attestation lines before reading a numerical row:

```
[waycount.mesh] width=4 shape=(16, 4) devices=64 unique=64 full_slice=1
[waycount.mesh] width=4 group=00 ids=[...] coords=[...]
[waycount.mesh] width=8 shape=(8, 8) devices=64 unique=64 full_slice=1
...
```

Every `(width, depth)` point reuses exactly the same host arrays across three arms:

| arm | purpose |
|---|---|
| `replicated` | Detect Pathways/compiler third-program drift that does not require TP reduction. |
| `stock-ar` | Measure the production-shaped TP reduction path. |
| `f4-fixed` | Replace only the TP reduction order with the fixed global-rank tree. |

Historical directly-attached four-device observations, in differing bytes out of 262144:

| configuration | forward-only vs forward+backward primal |
|---|---|
| 1D 4-device, TP over 4 | **6712 – 8196** |
| 2×2 mesh, TP across both axes (4 way total) | **6769 – 8104** |
| 4-way TP, device order `[0,2,1,3]` | **8083** |
| 2×2 mesh, TP on one 2-wide axis | **0**, except 946 at depth 32 with `create_device_mesh` |
| 2-device 1D mesh, TP over 2 (plain reshape) | **320** |
| 4 devices, fully replicated | **0** |

`differing_bytes` is a **binary bitwise gate only**. It saturates as differences spread and must
not be used to rank two dirty arms. Use `rel_l2`, `one_minus_cos`, and `max_abs` for magnitude.
In particular, `91371 > 90582` does not show that F4 made anything worse.

The current 64-chip discovery artifact completed widths `2,4,8`: 18 rows for depths `8,15` and
three arms. Every row is dirty, including the replicated arms. TP8 here is a generic platform
diagnostic only. The installed Qwen8B production contract, P1b and T2 remain TP4.

Read the generic P1 table as a paired diagnostic:

- `replicated SAME`, `stock-ar DIFFERS`, `f4-fixed SAME`: TP reduction order is a sufficient
  carrier here and F4 removes it.
- `replicated DIFFERS`: a non-reduction Pathways/compiler carrier is present. F4 may still
  remove an additional TP component, but stock-versus-F4 alone cannot identify it.
- `replicated SAME`, `stock-ar SAME`: reduction-order drift was absent at this point; a green
  F4 arm proves no repair.
- `f4-fixed DIFFERS`: F4 alone does not close the generic JAX graph.  This does **not** yet decide
  the promoted Qwen path because the generic graph did not execute the P22.XK operators.

The probe exits zero only after all `widths × depths × 3` rows complete. A partial table is
`INCONCLUSIVE`, not a numerical verdict.

Before P1b, P1a must compile the exact promoted RMSNorm through Mosaic and emit:

```
[mosaic.compat] VERSIONS jax=... jaxlib=... pathwaysutils=...
[mosaic.compat] COMPILE PASS ...
[mosaic.compat] VERDICT: PASS
```

An unsupported stable-Mosaic version is an infrastructure failure, not a numerical P1b red.
The canonical client image contains JAX/JAXLIB `0.10.2`. Both JobSet manifests therefore pin the
official `20260730-jax_0.10.2` Pathways proxy and server by immutable digest. A tag that merely
contains `jax_0.9.1` is not compatible, even if generic non-Mosaic XLA probes complete.

The production-operator gate must then print exactly one row at each registered depth:

```
[canonical-op] depth= 1 ... differing_bytes=0/... gradient_finite=1 gradient_nonzero=... SAME
[canonical-op] depth= 2 ... differing_bytes=0/... gradient_finite=1 gradient_nonzero=... SAME
[canonical-op] depth= 4 ... differing_bytes=0/... gradient_finite=1 gradient_nonzero=... SAME
[canonical-op] depth= 8 ... differing_bytes=0/... gradient_finite=1 gradient_nonzero=... SAME
[canonical-op] measurements=4 expected=4
[canonical-op] VERDICT: PASS
```

Any missing row, dead/nonfinite gradient, promotion-sentinel failure or nonzero byte difference
is a hard red. A P1a or P1b red taints T2 and stops the single Pathways session. A P1b green admits
only this bounded canonical MLP operator chain; it is not a full-model or training claim.

## Step 4 — What order did placement actually pick, and is this multi-slice?

`tests/t1_tpu/probe_mesh_order.py`.

Two independent hazards:

**Device order.** Topology-aware placement permutes what you pass in (`[0,1,2,3]` →
`[0,2,1,3]` on the probe host), and two different mesh *shapes* produce different permutations.
So rollout and trainer "using the same expression" does not make their orders agree. Print the
order **after** the mesh is built and assert it on both sides with
`CANON_EXPECT_MODEL_MESH_IDS`. An order copied from a different mesh shape is worse than none.
The one-dimensional P2 order does not attest the `(16,4)` training mesh; the full TP-group
listing from P1 is the authoritative placement record for that shape.

**Slice structure.** `MULTI_SLICE=1` means collectives cross slices and XLA lowers a
hierarchical reduction — intra-slice, then inter-slice. That is a **new program-splitting
mechanism with zero coverage in this work**, not a bigger version of a known one. The
fixed-order tree pins the order *within* one mesh axis and has nothing to say about the
level-crossing. Treat every bitwise claim on a multi-slice topology as UNVERIFIED until
re-measured there.

## Step 5 — What bucket does this dp geometry need?

`tests/t1_tpu/probe_bucket_contract.py`.

`MIN_TOKEN_BUCKET` is a **global** token count that the runner divides by `dp_size`. Copying
`256` from the `dp=1` recipe into a `dp=64` deployment gives each replica a bucket of **4** —
the pinning that the entire result rests on would be gone, while every switch still reads
"on". At `dp=64` targeting a per-replica `M=256`, the value is `16384`.

Note the probe answers *given a `dp_size`*. Determine the engine's actual data-parallel width
for your configuration; it is not necessarily the trainer's mesh axis of the same name.

## Step 6 — What does the tree cost at this width?

`tests/t1_tpu/probe_f4_cost.py`. Analytic, derived from the implementation:

```
ring all-reduce      2(n-1)/n * B moved,  ~1 buffer live
F4 fixed-order tree    (n-1)  * B moved,  ~n buffers live      ratio ~ n/2
```

`n=4` is 2× and was accepted. `n=8` is 4× at every reduction site, twice per layer, and has
never been measured. If that is unaffordable, the tree only needs to be *rank-ordered*, not
linear: recursive doubling gives the same rank-fixed order in `log2(n)` rounds. That is a new
implementation and would have to re-clear the full THIRDPROG and `A=B` gate set first.

## Step 7 — Is the DP update repeatable under a frozen placement?

`tests/t2_dp/run.sh`, or `CANON_MODE=dp-gate-only` on GKE.  On GKE the DP probe is imported by
the T1 unified runner after P1b, so it reuses the already initialized Pathways client.  Step 75
only validates the persisted same-session markers; it must not create a second IFRT proxy client.

For P32 the exact geometry is DP16×TP4 with 16 trajectories per DP replica. The hard admission
contract is:

- two executions with the same sample→rank mapping produce array-exact gradients;
- every DP replica sees the same post-reduction gradient;
- the injected rank-dependent negative is rejected;
- one AdamW arithmetic step emits stable SHA-256s for gradient, parameter and both moments;
- the measured mesh id sequence can be pinned and reproduced in a fresh run.

The first 64-chip T2 process measured an exact `(16,4)` order and passed the numerical checks, but
the expected-id variable was empty in that process. `jobset-64chip.yaml` now pins the observed
order and requires it at preflight. Only a fresh run that reproduces the pin upgrades T2 from
discovery evidence to admission evidence. Other manifests must measure and pin their own ids.

The probe also redistributes the same global samples across DP ranks. This is an observation, not
an initial hard gate. Local CPU evidence already shows the regrouped gradient changes even under
the fixed-order reference, because local partial sums were grouped differently *before* the
collective. Therefore fixed DP all-reduce order alone cannot provide arbitrary placement
invariance. Either freeze placement or accumulate canonical per-example contributions.

---

## Not covered by any probe

Say these out loud in any report from a new topology.

- **Full-model Pathways behavior.** One single-slice 64-device Pathways discovery process now
  covers the bounded P1a/P1b operator chain and T2 arithmetic. The train-mesh pin still needs an
  independent process repeat, and no full model, rollout or training step has run there.
- **FSDP parameter sharding.** A mesh axis named `fsdp` may shard parameters and all-gather
  them per layer in the forward. That is another collective in the forward path, and it has
  never been characterised here. Isolate it: run gates before training.
- **Cross-slice hierarchical reduction.** See step 4.
- **Replicated DP16 segmented VJP.** The current trainer's first mesh axis is `fsdp` and its
  segmented adapter walks the global trajectory list. The P32 profile refuses training until a
  true `(dp,tp)` adapter proves local-16 reverse, one DP reduction and one global commit.

## Reporting

A zero exit code is not an admission. Report the numbers, the overrides used, and the raw log
paths with their SHA-256 — see `cluster/README.md` §4.
