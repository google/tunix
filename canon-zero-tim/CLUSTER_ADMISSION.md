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

## Step 3 — Does the drift appear at *this* reduction width? (`gate-only`, seconds of TPU)

`tests/t1_tpu/probe_waycount.py`.

Before accepting any numerical row, require exactly one marker from every JAX-based probe:

```
[T1.PATHWAYS] required=1 initialized=1 status=ok
```

In proxy mode, a missing marker or `initialized=0` voids the topology run.  The shared bootstrap
executes before JAX import and exits nonzero if proxy registration fails; it never silently falls
back to a directly attached backend.

The deciding variable is neither device count nor mesh rank nor device order: it is the width
of a **single reduction**. Measured on the probe host, in differing bytes out of 262144:

| configuration | forward-only vs forward+backward primal |
|---|---|
| 1D 4-device, TP over 4 | **6712 – 8196** |
| 2×2 mesh, TP across both axes (4 way total) | **6769 – 8104** |
| 4-way TP, device order `[0,2,1,3]` | **8083** |
| 2×2 mesh, TP on one 2-wide axis | **0**, except 946 at depth 32 with `create_device_mesh` |
| 2-device 1D mesh, TP over 2 (plain reshape) | **320** |
| 4 devices, fully replicated | **0** |

**Read the magnitudes, not a SAME/DIFFERS label.** Four-way sits stably at 7000–8000 bytes.
Two-way is usually zero but not reliably so — the 320 and 946 above are real, one to two
orders of magnitude smaller than four-way and not explained. So the honest statement is that
reduction width is the *dominant* variable, not that two-way is guaranteed exact. This is why
`probe_waycount` prints byte counts: a reader who collapses them to a label loses precisely the
information that separates these cases.

Only widths 2 and 4 were ever measured. **8 is unknown**, and 8 is a width people reach for.

Read it this way: at every width you intend to use, the `F4-fixed-order` arm must be zero, or
at least an order of magnitude below its `XLA-all-reduce` counterpart at the same width.
And note the trap — if the `XLA-all-reduce` arm is *already* `SAME` at your width, the
fixed-order tree is a no-op there. A green result then tells you the problem was absent, not
that the fix works. A configuration at width 2 cannot validate this package.

## Step 4 — What order did placement actually pick, and is this multi-slice?

`tests/t1_tpu/probe_mesh_order.py`.

Two independent hazards:

**Device order.** Topology-aware placement permutes what you pass in (`[0,1,2,3]` →
`[0,2,1,3]` on the probe host), and two different mesh *shapes* produce different permutations.
So rollout and trainer "using the same expression" does not make their orders agree. Print the
order **after** the mesh is built and assert it on both sides with
`CANON_EXPECT_MODEL_MESH_IDS`. An order copied from a different mesh shape is worse than none.

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

`tests/t2_dp/run.sh`, or `CANON_MODE=dp-gate-only` on GKE.

For P32 the exact geometry is DP16×TP4 with 16 trajectories per DP replica. The hard admission
contract is:

- two executions with the same sample→rank mapping produce array-exact gradients;
- every DP replica sees the same post-reduction gradient;
- the injected rank-dependent negative is rejected;
- one AdamW arithmetic step emits stable SHA-256s for gradient, parameter and both moments;
- the measured mesh id sequence can be pinned and reproduced in a fresh run.

The probe also redistributes the same global samples across DP ranks. This is an observation, not
an initial hard gate. Local CPU evidence already shows the regrouped gradient changes even under
the fixed-order reference, because local partial sums were grouped differently *before* the
collective. Therefore fixed DP all-reduce order alone cannot provide arbitrary placement
invariance. Either freeze placement or accumulate canonical per-example contributions.

---

## Not covered by any probe

Say these out loud in any report from a new topology.

- **Pathways proxy backend.** Every result here was measured against a directly-attached TPU.
  Pathways is a different runtime with its own compilation and dispatch path. Whether program
  identity behaves the same under it is simply unknown.
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
