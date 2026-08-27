# Running GSM8K DP16xTP4 (target) with the 2026-08 optimization wave

Answers one question: what does the reverse-path optimization wave buy at
target scale? Read this before rendering a Phase4 GSM8K JobSet.

## What you get for free (no flag, no renderer change)

Rendering from a source SHA that contains this wave already gives the
target every flagless optimization:

- batched nonzero receipts (the v1-hp profile already requires
  `CANON_BATCHED_EVIDENCE=1`, so the grouped batching is on);
- the three jitted gradient programs that replaced ~1,242 eager per-leaf
  dispatches per group;
- the per-group dispatch-preparation hoist;
- the reducer program cache, which removes a per-update host re-trace of
  every reducer program.

On the DP4xTP1 one-host carrier these together moved a warm update from
23.2 s to 18.7 s and cut host dispatches by 63%. **They should scale
better at target, not worse**: the re-trace and per-dispatch costs are
host-side and the target pays them across 16 DP ranks with Pathways
round-trips, which is exactly the regime the one-host carrier
under-represents. That makes a plain "old SHA vs new SHA" comparison the
first and most informative run — nothing to configure.

## What needs a decision (and why it is not on by default here)

| selector | value | status at DP16xTP4 |
|---|---|---|
| `CANON_DP_COLLECTIVE_REDUCE` | `1` (fp32 psum) | cuts the DP reduce section 62% at DP4 with bitwise-identical norms there; the **DP16 envelope against the FP64 oracle has not been run**, and the host adversarial spectrum showed psum up to 15.7% worse mean error than the tree on fp32 leaves. Treat as an arm, check the norms and the loss curve. |
| `CANON_DP_COMPARE_MODE` | `fingerprint-hybrid` | gradient values unchanged bitwise by construction; kill-tested; validated at DP4. Receipt cadence drops from every group to the first groups of an update. |
| `CANON_DP_DISTINCT_SCHEDULE` | `first-group-warmup` | as above; only exercised after the third update, so short runs cannot price it. |
| `CANON_DP_FINITE_FETCH` | `batched-commit` | as above; the non-finite gate stays pinned at the optimizer commit. |
| `CANON_P71_SCAN` | `fwd` | one scanned forward-tape program per chunk; bitwise at DP4xTP1 and DP16xTP1, **never tested at TP>1**. |
| `CANON_P71_SCAN` | `bwd` | **cannot run here**: the unrolled backward blocks fail closed on a non-unit model axis. TP4 refuses by design. |

## Delivering a selector to the pods

Verified against the renderer source, not inferred. The JobSet has two
replicated jobs, `pathways-head` and `pathways-worker`; our Python
learner — and therefore every `CANON_*` reader — runs in the head job's
`jax-tpu` container, which is exactly the container the Phase4
renderer's `_set_env` patches. Workers run the Pathways binaries and
never read these flags (XLA-level flags are the separate proxy/server
delivery problem, not this one).

Two mechanics worth knowing before editing:

- `_set_env` **refuses to overwrite an already-rendered key**, so
  additions is for names the base manifest does not already carry; a
  value that is already rendered must be changed where it is rendered.
- `CANON_BATCHED_EVIDENCE=1` is exported by the base profile
  (`cluster/profiles/qwen3-1p7b-dp16-tp4-gsm8k.env`); the v1-hp profile
  only enforces it. That is why the batched receipts need no action.

A docker `-e` in a one-host runner reaches nothing here. To arm one of
the selectors, add it to the additions dict in
`canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts/render_three_full_recipes.py`
next to the existing `CANON_P59_CHECKED_VMA` / `CANON_P67_P66_VMA_P59_ONLY`
entries, for example:

    additions["CANON_DP_COLLECTIVE_REDUCE"] = "1"

That file belongs to the Phase4 recipe lane; coordinate before editing it.
After rendering, confirm delivery from the run itself, never from the
launch command: the psum arm shows all-reduce rather than ppermute in the
collective census, and the forward-scan arm shows one `zt_tr_fwd_scan`
module per chunk instead of the per-layer tape dispatches.

## Reading the result

1. **Numerics first.** Commit gradient norms and the strict alignment
   gate decide whether a run counts at all; a faster arm with a norm
   drift is a failed arm, and its timings are failure evidence.
   Note that this profile runs the overflow-safe norm
   (`CANON_P63_OVERFLOW_SAFE_CLIP=1`), which is byte-identical to the
   stock norm whenever the stock norm is finite.
2. **Then the reverse share.** The pre-wave target baseline was a warm
   reverse of 511.0 s inside a 613.6 s official step (83.3%), parsed
   with `read-xprof/scripts/parse_perf_stages.py`. Parse the new run the
   same way and compare the share, not just the wall.
3. **Watch HBM on the first armed run.** The jitted assembly packs one
   chunk's operands, which cost up to ~3.4 GB of transient on the
   one-host geometry; the target's per-rank trees are TP4-sharded but
   staged across sixteen DP ranks, so the transient profile is not the
   same and has not been measured there.
4. **Then attribution.** The one-host wave's biggest single item was the
   removal of ~5,408 synchronous device-to-host receipt reads per update;
   at target each of those round trips is far more expensive than on one
   host, so the reverse-side saving is the number this run exists to
   measure.
