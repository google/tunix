# P38.2i: request journal and concurrency discriminator

- Status: locally complete; target P38s12a/P38s12b are not run.

## Evidence entering this phase

P38s11 is the first admitted full-coverage stock red: it covered all 32
prompts / 256 trajectories, reproduced finite `S_decode != S_prefill`, kept
`S_prefill == T_old`, emitted no capture error, and stopped before backward.
The carrier retained the historical shape: 27 differing elements among
48,449 actions, onset at logical KV 1686, turns 3--4, and maximum absolute
difference about 0.1044. P38s10 was exact only because its first four prompts
solved after about 85 action tokens per trajectory; it never reached the known
carrier depth.

The P38s11 capsule and serving archive permit exact offline prefix joins:

| source row | snapshot | request | DP | computed tokens | physical blocks |
|---|---:|---|---:|---:|---|
| 199 | 0/1 | `372-9b9cf482` | 14 | 1179/1216 | `45,43,42,41,40` |
| 206 | 0/1 | `390-9ab9fff6` | 15 | 1230/1267 | `62,61,60,59,33` |
| 199 | 2 | `529-ac6158ef` | 2 | 1348 | `65,60,14,72,73,75` |
| 206 | 2 | `532-baab38d4` | 3 | 1463 | `34,35,51,1,2,18` |

Those joins are provenance, not a causal result. The four global snapshot
anchors did not observe the red rows at their actual mismatch times. P38s11
also exposed two classifier defects: production block tables may be flattened,
and a valid snapshot may join more than one selected source row.

## Deliverable A: P38s12a, known-red capture upgrade

Keep the known-red environment unchanged: stock engine, all 32 prompts, eight
generations, engine DP16, maximum concurrency 256, prefix cache disabled,
precheck-only, backward zero, optimizer commits zero. Do not rerun unified KV;
its production arm was already red.

Add only diagnostic coverage:

1. Select up to eight red capsule rows and persist group/generation identity.
2. Replace the unreachable upper stratum with four bands covering the observed
   carrier: `[1536,1664)`, `[1664,1792)`, `[1792,1920)`, and `[1920,2048)`.
3. Persist one host-only request journal record when each scheduled request
   first enters each band. Each record contains exact token history/SHA,
   request/DP/slot, physical block IDs, co-batch membership, and explicitly
   named **observation** generations for pages.
4. The journal must not call `jax.device_get`, hash the KV cache, or otherwise
   add a device program boundary. Records from one scheduler call are appended
   and fsynced together.
5. The classifier restores flattened block tables, admits multiple unique row
   joins per snapshot, requires a nonempty journal, and requires every selected
   capsule row to join at least one journal event.
6. Render stock only. No U YAML is needed for this target run.

Observation generations say only what this bounded journal saw. They are not
allocator generations and cannot prove an unobserved free/reuse event. Full KV
page hashing remains deferred until an exact red request is selected; that
observer needs its own neutrality gate because fetching KV buffers can perturb
the program being measured.

## Local exit gate

- classifier unit and negative-control suite passes, including flattened block
  tables, multiple source rows, ambiguous joins, missing journal, and multiple
  turn requests for one source row;
- renderer emits both legacy paired manifests by default and exactly one stock
  manifest with `--stock-only`; the stock manifest pins eight capsule rows,
  the four new bands, and the journal path inside the archive directory;
- outer postflight rejects an absent journal and accepts a valid journal;
- pinned Qwen3-1.7B and Qwen3-8B overlays apply patches 01--13, match the
  manifest, compile, and prove the journal never touches a device buffer;
- full P33 CPU/adjacent gates, shell syntax, credential scan, executable-text
  scan, ordinary-source whitespace scan, and patch application pass.

All local exit gates passed on 2026-08-13. The pinned Qwen3-1.7B and Qwen3-8B
overlays each passed 23 tests and matched all 29 manifest entries. The complete
pinned-image CPU/adjacent suite reached `[P33.WORKLOAD] CPU_GATE PASS`; the
classifier passed 30 tests, the renderer passed seven tests, and the outer
postflight rejected a marker-only but missing journal. This is construction
evidence only; no target result is implied.

## P38s12a target exit gate

One Attempt-0 stock run must return the complete byte-zero terminal log and
evidence directory. It must reproduce finite A-B red with exact B-C, cover 256
trajectories, emit four pre/post records, emit a nonempty request journal, join
every selected red row, classify PASS, archive all records, and stop before
backward with zero optimizer commits. Any missing selected-row join is
`INCONCLUSIVE_CAPTURE_SELECTION`, not evidence of equality.

Only after exact red-row joins exist may the next step attempt strict E0
replay. E0 remains a whole-vector identity gate: replayed A must equal
production A at every action position, replayed B must equal production B,
and A-B must remain red. A geometry-only replay that misses this gate is a
falsifier, not an E0 result.

## Deliverable B: P38s12b, pure concurrency arm

Run this only after P38s12a evidence is admitted. Change one variable: maximum
concurrency from 256 to 32 while consuming the same complete 32-prompt data in
eight sequential four-prompt units. Keep source, weights, prompt order,
generation count, prefix cache, precision, kernels, and capture schema fixed.

The arm is informative only if at least one trajectory reaches logical KV
1686. If no trajectory reaches that depth, classify it
`INCONCLUSIVE_DEPTH_NOT_REACHED`. If a depth-sufficient concurrency-32 arm is
exact, repeat it once before claiming concurrency is necessary. A red arm
shows that small concurrency does not remove the carrier. An exact repeated
arm only establishes concurrency/churn as a necessary trigger; it does not
identify the faulty operator.

## Claim ceiling and rollback

This phase improves selection and separates trigger environment from numerical
cause. It does not prove a scheduler bug, stale KV, RoPE, residual/cast, page
reuse, or a numerical repair. Leave all P38 capture variables unset to restore
the ordinary runtime. The patch and capsule expansion are default-off and do
not change training, evaluation, prefix cache, precision, optimizer placement,
or canonical kernels.
