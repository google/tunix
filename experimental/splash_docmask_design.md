# Splash document mask for packed rows — design and measurements

Companion to the runbook at the end of `reported_issue.md`. This file records
*why* the design looks the way it does, including the two approaches that were
measured and dropped and the one that was measured and found silently wrong.

Status: **FALSIFIED on TPU, 2026-08-03.** The design is both incorrect and
slower than what it replaces. Do not ship it and do not run the A/B; the
measurements that motivated it were taken against the wrong baseline. Details
in "Why this does not work" below. Everything before that section is the
original reasoning, kept because the mechanism it describes is real — it is the
cost model that was wrong.

The production plumbing was never landed, so nothing under `tunix/` is affected.

## Why this does not work

Measured on v4-8 through the real packer, one Qwen3 attention block, 4 rows x
2048, block 256, `p20f_diag.py`:

| arm | ms | vs today | correct? |
|---|---|---|---|
| `CausalMask` (today) | **2.327** | 1.000 | — |
| `NumpyMask` holding the IDENTICAL causal mask | 3.136 | **1.347x** | bitwise identical |
| `NumpyMask` holding the document mask | 3.017 | 1.296x | **DIFFERENT** |
| document mask + padding segment (correct) | 3.329 | 1.431x | identical, but `grid_width` back to 8 |

**Two independent failures.**

**1. The mask is not a superset, because padding is a segment.** The packer's
`segment_layout` lists the real segments only. Padding gets `segment_id = 0`,
and splash's segment test is a bare `q_ids == kv_ids` with no special case for
zero, so **pad attends pad** across the padding region's whole causal triangle.
A row that is entirely padding — the packer emits them, e.g.
`((700,650,600), (550,500,450,400), (350,300,250,200,150), ())` gives
`row3: id0 x 2048` — is a full triangle that the union over layout rows does
not cover. Measured: 1,063,936 uncovered pairs, 3,656,487 differing output
elements in rows 2 and 3. Every earlier bitwise check used hand-built layouts
whose rows were exactly full, so the padding segment never appeared.

**2. The cost is the mask REPRESENTATION, not the mask content.** `CausalMask`
subclasses `_ComputableMask`: it carries a `mask_function` and a `q_sequence`,
and splash rebuilds the triangle **in-register**. `NumpyMask` is opaque, so
every partial block becomes a 256x256 tile fetched from HBM. Swapping
`CausalMask` for a `NumpyMask` with byte-identical content costs **1.347x** on
its own. Halving `grid_width` claws back only part of that (1.296x), and once
the padding segment is included to make it correct, `grid_width` returns to 8
and the arm is a pure 1.431x loss.

**The block-count model was measuring the wrong quantity.** It counted work
blocks and ignored the per-block cost of the mask representation, so `0.861x`
was never achievable. Any earlier A/B whose baseline was itself a `NumpyMask`
compared 1.347x against 1.347x and reported a win that does not exist relative
to production.

**What would have to change for this to work.** The mask has to stay
computable: a `_ComputableMask` subclass whose `mask_function(q_ids, kv_ids)`
derives causal-and-same-segment from an encoded `q_sequence` (segment index in
the high bits, position in the low bits), so no tile is ever fetched. The open
problem is that splash's mask is shared across the batch while each packed row
has a different layout, which is exactly why this version took a union — and a
union is what makes `grid_width` degrade to the worst row. `ChunkedCausalMask`
is the shape of the answer for equal-length documents; variable lengths are
unsolved here.

## The problem

Splash attention builds its block schedule from a **static** mask at trace time.
A packed row holding several documents is handed a plain causal mask, so splash
schedules the whole `budget²/2` triangle. `segment_ids` then zeroes the
cross-segment blocks — *after* they have been computed. Packing buys memory and
padding, but attention is still charged as if the row were one long sequence.

Concretely, at `budget=2048`, `block=256`: `grid_width` is 8 for every row,
whether the row holds one 2048-token document or eight 256-token ones.

## The fix

Build the mask from the chunk's actual segment layout, with segment extents
rounded **outward** to block boundaries, and hand splash that mask instead.

```
row = [700, 600, 500, 248]      block = 256

causal (today)          block-rounded document mask
  ████████                ██
  ████████                ███
  ████████                ████
  ████████                ░░░░███       ░ = dropped from the schedule
  ████████                ░░░░████
  ████████                ░░░░░░░██
  ████████                ░░░░░░░███
  ████████                ░░░░░░░░██
  grid_width 8            grid_width 4
```

Two properties make this both safe and cheap.

**Safe — the mask is a superset.** Rounding outward can only *add* positions, and
`segment_ids` still performs the exact elementwise masking downstream. So the
output is bitwise unchanged; only the schedule shrinks. Measured on TPU with the
layout `(1024, 512, 512)`: the exact mask, a coarser superset, the coarsest
superset, and plain causal all produced **bitwise identical** `out` and `grad`,
while a negative control that was *not* a superset — `(768, 768, 512)` — differed
immediately. That negative control is what gives the assertion resolution.

**Cheap — the rounding direction is chosen so compile count stays bounded.** Only
the *segment* direction is rounded, never the causal one. `segment_ids` enforces
same-segment, **not causality**, so rounding the causal half would let
non-causal pairs through — verified as a negative control, it does exactly that.
Leaving the causal half alone means the diagonal blocks stay causal triangles,
and every one of them is *the same* triangle, so JAX dedupes
`partial_mask_blocks` down to **1**.

That last point is the whole reason this works, and it is not obvious.

## Compile count is governed by TWO shapes, not one

The first version of this analysis assumed `grid_width` alone drove the number of
compiled programs. That is wrong. `partial_mask_blocks.shape[0]` is a shape too,
and a segment boundary landing mid-block produces a uniquely-shaped partial
block:

| mask | `grid_width` | `partial_mask_blocks` |
|---|---|---|
| aligned `[1024, 1024]` | 4 | **1** |
| unaligned `[700, 600, 500, 200]` | 4 | **11** |
| unaligned `[400, 250, 180, 150]` | 2 | **7** |

The exact per-chunk union over real-looking layouts produced **7 distinct
`(grid_width, partial_blocks)` pairs with no upper bound** — one compiled program
each. With outward rounding, 11 chunks across 7 length distributions collapsed to
`[(2,1), (4,1), (6,1), (8,1)]` — **4 programs**, `partial_blocks` always 1.

## Alternatives measured and dropped

| approach | attention | programs | token cost | packer change |
|---|---|---|---|---|
| causal (today) | 1.000 | 1 | — | — |
| band mask keyed on longest segment | 0.924 | ≤8 | 0% | none |
| block-diagonal, exact union | 0.861 | 7, unbounded | 0% | none |
| **block-diagonal, boundary-rounded** | **0.861** | **4** | **0%** | **none** |
| block-diagonal, quantized segment lengths | 0.888 | 4 | +2.6% | required |

Boundary rounding is no worse than any of the others on all four axes.

The `0.861` is from a block-count model, **not a wall-clock measurement** — that
is what the runbook's A/B is for.

**Rebalancing chunks is not worth it.** Giving every row its own kernel — the
theoretical floor for any regrouping strategy — is only `0.968×`. Karmarkar-Karp
balancing measured `1.029×` (*worse*), sorting rows by longest segment ascending
`1.014×` (worse), descending `1.000×` (a wash). A 3% ceiling does not justify
touching the packer. The one actionable finding: do not sort ascending.

## The bug this design exists to avoid

An earlier version passed the mask through a module-level global. All three gate
arms passed. It was silently wrong.

A global read inside a jitted function is a **trace-time constant**, baked into
the compiled program. jit's cache key is built from the **arguments'** shapes and
dtypes; a global is not an argument, so changing it does not trigger a retrace.

| run | checksum |
|---|---|
| declare layout A `(1024, 1024)` only | `528823221325` |
| declare layout B `(2048,)` only | `528749157605` |
| **declare A, run, then declare B and run** | **`528823221325`** ← still A's |

The real data in that test was a single 2048-token segment, which A's mask cuts
in half. The second call computed a **wrong answer, with no error, no retrace,
and no observable symptom.**

The three-arm gate could not see this because each arm ran in a fresh process and
therefore traced exactly once. **Any declarative switch needs a gate that changes
it twice in the same process.** `p20b_gate_tpu.py --mode A_then_B` is that gate.

The fix is to pass the kernel as an ordinary pytree argument: its leaves are the
MaskInfo arrays, so jit sees them and caches on their *shapes* — one program per
mask shape. Making the field static instead would put the mask *values* in the
cache key and give one program per layout, i.e. one per step.

## Files

| file | what it is |
|---|---|
| `p20_splash_mask.py` | the new module, shipped verbatim as `tunix/rl/splash_mask.py` (`docmask` / `build_kernel` / `kernel_for` / `attach`); the only place that reads the env switch |
| `p20_patch_packer.py` | generates the packer-side patch: `segment_layout` field, `_emit` stamping, the `attach` call |
| `p20b_patch_thread.py` | generates the model-side patch (10 mechanical `splash_kernel` parameter threads) |
| `p20c_patch_route.py` | generates the routing patch (`TrainExample` → `algo_core` → `common`) |
| `p20b_gate_tpu.py` | neutrality / correctness / **same-process switch** / compile count |
| `p20c_gate_route.py` | the full route, verified in both switch positions |
| `profile_v5p_docmask.sh` | the end-to-end A/B, xprof + perfetto |

Reproducing the patched tree (this is what `profile_v5p_docmask.sh` mounts):

```bash
python3 experimental/p20_patch_packer.py . "$OUT"          # common, utils, rl_learner
cp "$OUT"/{common,utils,rl_learner}.py tunix/rl/           # feed forward
python3 experimental/p20b_patch_thread.py \
        tunix/models/qwen3/model.py "$OUT/model.py"        # model
python3 experimental/p20c_patch_route.py . "$OUT"          # common, algo_core
cp experimental/p20_splash_mask.py "$OUT/splash_mask.py"   # module, verbatim
```

Verified 2026-08-03: run against this branch, that chain reproduces the six
mounted files **byte for byte**.

The patch generators hard-fail if an anchor is missing, and their post-condition
counts are **derived** (`after = before − Σanchor + Σreplacement`) rather than
hand-written. Three earlier hand-typed counts were wrong; the guard caught all
three and refused to write.

Three scripts from the earlier rounds are deliberately **not** here:
`p20_patch_docmask.py` (generates the module-global design proved wrong below),
`p20_gate_tpu.py` (the three-arm gate that missed it), and `p20_gate_cpu.py`
(calls `model._docmask`, which moved to `splash_mask.py`, and pulls in four
scripts from superseded rounds). Their results are recorded in the table below;
the scripts themselves would only mislead.

## Gate results (2026-08-03)

| gate | result |
|---|---|
| P20.1 layout stamped, not a pytree leaf, `None` when off, negative control | ✅ CPU |
| P20.2 `partial_blocks` ≡ 1; 4 programs over 7 distributions; superset per chunk; negative control has resolution | ✅ CPU |
| P20.3 module-global channel | 🔴 **FAIL** — silently wrong, see above; its timings are void |
| P20.4 G1 neutrality: unpatched vs patched-with-`kernel=None` | ✅ both `528749157605` |
| P20.4 G2 a declared mask actually takes effect | ✅ A ≠ B |
| P20.4 G3 **same-process switch** `A_then_B` | ✅ returns B, jit cache 1→2 |
| P20.4 G4 compile count: 3 same-shape → +1; 2 different-shape → +2 | ✅ |
| P20.5 full route, both switch positions | ✅ CPU, `(gw, pb) = (3, 1)` |

## The second bug: `attach` got a list, not an example

`pack_sequences` is typed `-> Iterator[list[TrainExample]]`, and `_mark`
returns `[merged.replace(...)]`. The learner therefore hands `attach` a LIST.
The first version read `segment_layout` straight off it, got `None` from
`getattr`, and returned it untouched -- **on every step of a full end-to-end
run**, with no error and no log line. Attention time was unchanged, which looks
exactly like an optimisation that simply did not help.

The P20.5 route gate missed it because it fed a hand-built single
`TrainExample` instead of the container the pipeline actually produces. Gates
have to be built from the real structure, not a plausible stand-in.

Two fixes, both in `p20_splash_mask.py`:

* `attach` unwraps a list/tuple elementwise.
* Doing nothing is now **loud**. Every no-op path calls `_skip(why)`, which
  warns the first three times; every successful attach logs its `grid_width`
  and `partial_mask_blocks`. `splash_mask.stats()` returns
  `{'attached': n, 'skipped': n, 'shapes': {(gw, pb): chunks}}`.

`p20d_gate_list.py` asserts on the real container and on the loud-failure
behaviour, in both switch positions.

**Read `shapes` before reading any timing.** If every entry is `(8, 1)` at
budget 2048 then `grid_width` never shrank and no speed-up is possible --
that is a property of the length distribution, not a bug.

## What is not here, and what is not known

**Not in this change:** the production plumbing. Those edits exist only as the
patch generators above, applied to files mounted read-only into the container.
Nothing under `tunix/` is modified. The plumbing should not land before the
end-to-end numbers exist.

**Not verified:**

* **End-to-end benefit.** `0.861×` on attention is a block-count model. Attention
  is ~79% of a DecoderLayer's fwd+bwd, and a training step is more than decoder
  layers, so the step-level effect is smaller and is not predicted here.
* **Compile time for a whole train step** — only attention-level was measured, and
  a 32s-attention-vs-4s-layer discrepancy there is still unexplained.
* **The real gsm8k length distribution.** The 4-program result is over seven
  *synthetic* distributions. If real chunks are more fragmented, the compile
  budget needs re-estimating — the runbook's step 4 is that check.

**Known rough edges:**

* `num_heads` arrives via `TUNIX_SPLASH_NUM_HEADS` rather than the model config.
  Missing it is a hard error, not a silent skip, but this is not how it should
  ship.
* `build_kernel` hardcodes `head_shards=q_seq_shards=1`. Correct only for `tp=1`
  with no sequence sharding; any other mesh must pass the real values or
  `manual_sharding_spec` will disagree with the shard_map.
* **Zero benefit is an expected outcome for some recipes.** gsm8k runs with
  `budget = L_max = 2048`, so a chunk holding one near-`L_max` sequence has no
  cross-segment blocks to drop. The gain comes entirely from real sequences being
  shorter than the cap.

## Upstream

JAX's `_process_dynamic_mask` accepts a runtime mask but `del`s `shrink_grid`,
so the grid-compression path is unavailable to dynamic masks —
`_shrink_mask_info` is pure numpy (`np.nonzero` per q-block row) and cannot run
on traced arrays. That is why the mask here is built on the host and the kernel
is passed in, rather than the mask being passed as an array. Worth an upstream
issue.
