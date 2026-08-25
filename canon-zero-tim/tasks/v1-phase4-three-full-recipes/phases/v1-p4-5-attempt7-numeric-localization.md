# V1.P4.5 — Attempt-7 P59 numerical localization

## Objective

Explain the first numerical failure after strict Zero-TIM forward alignment and
the complete P59 reverse. Do not treat max-scaled L2/clipping as the cause or
repair. The first target carrier is backward-only and performs zero optimizer
commits.

## Known facts and claim ceiling

- GSM8K `g64s` is strict for 191,439 action values with zero alignment FAIL and
  reaches 16/16 P59 reverse groups. Its first incoming scaled group reports the
  old naive FP32 norm as `inf`; it never commits an optimizer update.
- P45 `f45s` is strict for 48,082 action values and enters real TP8 backward.
  The old replica comparator returns false, but its log cannot distinguish a
  common NaN/Inf from unequal finite replicas.
- The current max-scaled L2 and element-finiteness code is host/exact-image
  construction evidence only. It has not shown that P59 magnitude or scaling
  is correct on DP16xTP4 or DP8xTP8.
- A finite-but-enormous gradient is not automatically valid. For roughly
  1.72e9 nonzero FP32 elements, uniform RMS must be about 4.4e14 before the
  naive sum of squares overflows; a single element must exceed about 1.84e19
  before its square overflows. Normal training gradients are not expected near
  those scales.
- Remote G5a run `canon-p62-gsm8k-num-p62d3-505bfb95` adds one bounded fact:
  the loss cotangent and group-0 engine/rank-local trees are finite, but the
  latter has `max_abs=5.7923e21` and `stable_norm=5.3814e22`. The preserved
  artifact contains only six selected lines and cannot establish later
  boundaries, complete execution, or a zero-commit terminal.

## Frozen GSM8K algebra and shape ledger

Runtime receipts must print and validate every value below rather than infer it
from topology names.

| Quantity | Registered value | Meaning |
|---|---:|---|
| caller-global trajectories | 256 | 32 prompts x 8 generations |
| DP size / TP size | 16 / 4 | target trainer topology |
| rank-local trajectories | 16 | 256 / DP16 |
| gradient groups | 16 | one trajectory slot from every DP rank per group |
| caller-global M | 4096 | DP16 x local M256 |
| shard-local M | 256 | rows seen by one DP rank |
| canonical-kernel M | 256 | fixed numerical kernel program |
| scheduler token capacity | 256 per rank / 4096 global | separate from semantic valid rows |
| logical loss scale | `1 / denominator`; expected denominator 256 | must be read from the loss output |
| streamed group multiplier | `scale * local_trajectories`; expected 1/16 | applied after fixed DP reduction |
| accumulator denominator | 16 | one add per gradient group |
| final intended gradient | `(1/256) * sum(j=0..15,r=0..15, g[j,r])` | no implicit extra DP divide or sum allowed |

TP reductions are separate from the DP average. The fixed LM-head and other
TP-sharded pullbacks may sum TP-local cotangents in fixed rank order, but they
must not introduce another DP reduction.

## Pre-registered discriminator

| First red observation | Interpretation | Next action | Forbidden claim |
|---|---|---|---|
| input/token/alignment hashes differ | carrier invalid | repair capture/join | backward root cause |
| advantage, ratio, loss or loss cotangent is non-finite/extreme | objective or denominator seam | inspect GRPO loss inputs and scale | engine VJP fault |
| loss cotangents sane; engine VJP first red | custom head/layer pullback seam | isolate first bad endpoint and compare with FP64 oracle | DP reducer fault |
| rank-local gradients sane; post-DP gradient grows by DP | implicit plus manual DP reduction | inspect sharding/collective ownership | optimizer/norm fault |
| post-DP gradient sane; scaled group first red | wrong streamed multiplier/dtype | inspect `scale * local_trajectories` | accumulator fault |
| all scaled groups sane; accumulator first red | denominator/cadence error | inspect add count and `GradientAccumulator.get()` | reducer fault |
| finite max-abs and stable norm plausible; only naive norm is `inf` | norm algorithm case | consider stable clipping only after target evidence | general P59 correctness |
| DP2xTP2 sane but magnitude grows with DP16 topology | topology-dependent DP scaling | inspect implicit collectives and sharding | local carrier proves target |
| same leaf is first red on DP2xTP2 and target | custom VJP/local mapping candidate | bounded endpoint oracle and causal arm | full training solved |

## Diagnostic contract

The diagnostic is default-off, exact-workload admitted, and fail-closed.

1. It runs the strict canonical forward and all grouped P59 backward work.
2. It accumulates only into a disposable gradient accumulator and then discards
   it; model, optimizer state, reference state, and train-step counters must be
   byte/fingerprint unchanged. `optimizer_commits=0` is mandatory.
3. It emits compact receipts for group 0, group 15, and the first failing group:
   all-finite, nonzero count, max-abs, overflow-safe L2, naive-L2 finiteness,
   leaf count, top max-abs leaf paths, and first non-finite rank/leaf/path.
4. Boundaries are: loss inputs/scale, loss cotangents, engine VJP output,
   trainer rank-local adjoint, fixed DP-reduced gradient, scaled microgradient,
   and final averaged accumulator.
5. Non-finite values are always fatal. Finite-but-extreme values stop before
   optimizer and are evidence, not silently clipped.
6. Observers must not print tokens, model values, complete gradients, secrets,
   or full environments. Full-tree artifacts remain opt-in and immutable.

## Gates

- G0: flag/parser/profile truth table, wrong workload/topology/stage negatives,
  exact writer-to-reader delivery, and registry audit.
- G1: deterministic small-tree unit tests catch non-finite, finite-huge,
  wrong multiplier, wrong denominator, and DP double-sum injections.
- G2: forced-CPU DP2xTP2 installed fixed-head/projection/P59 composition emits
  every boundary and the negative control trips.
- G3: pinned-image dependency-complete gate is green with exact diagnostic
  markers and `optimizer_commits=0`.
- G4: real one-host v5p DP2xTP2 matched carrier uses fixed weights and input;
  ordinary JAX/Tunix and P28+P59 gradients are compared by per-leaf finiteness,
  max-abs, relative L2, cosine, and bounded FP64 endpoint oracles. Serial and
  parallel need not be bitwise identical.
- G5a: remote DP16xTP4 GSM8K partial excerpt is preserved as incomplete; it
  must not admit a repair or optimizer transaction.
- G5b: user-run fresh DP16xTP4 GSM8K diagnostic preserves strict alignment,
  completes 16 groups, identifies the first numerical red or proves every
  boundary sane, persists the complete raw log, and performs zero optimizer
  commits. The classifier must validate exactly one profile marker and reject
  unknown non-JSON P62 markers.
- G6: only after G5b may a one-commit candidate be designed. Full recipes stay
  blocked until that transaction and its prior gates are green.

G2 topology correction: the pre-registered `DP2xTP2 installed fixed-head`
phrase names a geometry that is not registered for either target model; the
1.7B head is TP4 and the 8B head is TP8. It is therefore not legal to invent a
TP2 head just to satisfy the carrier. G2 is fulfilled by the stronger actual
registered DP2xTP4 plus DP2xTP8 installed-head/projection composition. G4
retains the physical four-chip DP2xTP2 scaling/RPA mechanism check. This
correction narrows claims to real supported geometries; it does not relax a
numerical criterion.

## Result log

- 2026-08-25: phase pre-registered. Verified by source/log analysis that the
  Attempt-7 `norm=inf` belongs to the first scaled group, before final
  accumulator averaging; not verified whether the first bad value originates
  in loss cotangents, custom VJP, DP reduction, or scaling because those
  boundary magnitudes were not serialized.
- 2026-08-25: G0-G3 verified by host and pinned-image execution. Production
  recipes were restored to stock Optax clipping. P62 is exact-workload,
  default-off and zero-commit; wrong profile/stage/shape/denominator flags are
  rejected. Complete exact-image raw SHA is
  `604c95e5953f97fa8465e03f38b15589bd38fbf618b04c5652be0328b446689e`
  with one `V1_HP_EXACT_IMAGE_PASS` terminal and `p62_numeric=6`.
- G2's composition ceiling was checked explicitly after the complete gate:
  the focused pinned-image r2 runs real fixed-head -> report adjoint -> fixed
  reducer and installed projection/attention at DP2xTP4 and DP2xTP8, emits 10
  P62 boundary receipts, and catches two injected NaN first reds. Raw SHA is
  `8fb3720e3ac39cf80535833e1786585950ab13bd7015b4c9c9aa66da0dc60b92`.
  The r1 carrier failure (negative accessed a JAX-donated buffer after
  finalize) is preserved and is not a numerical verdict. This remains a
  seam-composition gate, not a full-Qwen target invocation.
- 2026-08-25: G4 verified by real one-host v5p DP2xTP2 run
  `a7_numeric_dp2tp2_20260825_r2`. Real installed RPA/staged-spec carrier and
  matched gradient carrier both pass; FP64 relative-L2 is `3.77417983e-08`,
  cosine is `1`, wrong multiplier and duplicate-DP-sum negatives both have
  relative-L2 about `1`, and optimizer commits are zero. Not verified by this
  gate: full Qwen DP16xTP4 magnitude or the first target red, because the
  carrier is a bounded linear/reduction mechanism test.
- G5a is incomplete historical evidence and remains preserved. G5b is complete:
  fresh target run `canon-p62-gsm8k-num-c1-e2c51a89` executed all 16 groups
  and the final DP16 accumulator with `all_finite=true`,
  `first_nonfinite=null`, denominator 16, 16/16 strict alignment PASS, and
  zero optimizer commits. The final accumulator has
  `stable_norm=4.688506872758702e20`, `max_abs=7.188024281236767e19`, and
  `naive_norm=inf`; this identifies FP32 sum-of-squares overflow after a valid
  finite backward. G6 may now design the root-cause repair.
- The offline G5 classifier is implemented and host-negative-tested. It keeps
  `ROOT_LOCALIZED_NONFINITE`, `FINITE_NAIVE_L2_OVERFLOW`, and
  `ALL_BOUNDARIES_FINITE_NO_COMMIT` distinct; alignment/shape/scale/optimizer
  violations are `FATAL_CONTRACT`, and an unlocalized truncated log is
  `INCONCLUSIVE_INCOMPLETE`.
- 2026-08-25 G5a correction: the signed six-line target artifact passes its
  own SHA256. The repaired classifier admits and exactly validates its
  `profile_resolved` marker, then still returns `FATAL_CONTRACT` because the
  artifact omits strict alignment, groups 1-15, fixed-DP/scaled/accumulator
  boundaries, and discard. It is not backward-correctness or naive-norm-only
  evidence.
- 2026-08-25 publication audit: rebased the scoped P62 CLs on operator runtime
  tip `eb58954f`, then through the publication-time M15 evidence/documentation
  tip, and preserved the incoming APC carrier and Attempt-0 receipt. Host V1 34/34, P59
  37/37, and deterministic flag registry 371/371 pass. The complete merged
  pinned-image tree exits zero with one terminal containing both
  `p62_numeric=6` and `apc_m15_carrier=33`; verified by the release terminal,
  but not registered as a new signed raw artifact. The previously signed P62
  r1, G2 r2, and one-host r2 evidence remains byte-verified. G5 is still not
  run and no optimizer transaction is admitted.
- 2026-08-25 G5b carrier admission: the P62 run now seeds the exact validated
  profile receipt into its unique `$CANON_STATE/run.log`, appends the complete
  workload output, and automatically classifies that exact file in pod before
  exit. The terminal receipt binds verdict, workload/transport return codes,
  log SHA/bytes/lines, classification path, and classification SHA. The
  renderer records both durable paths. A selected-line excerpt cannot pass.
- Classifier negatives require one exact profile and admission marker, strict
  pre-alignment, complete reverse groups 1-16, loss cotangent, group-0/group-15
  engine/rank-local/fixed-DP/scaled seams, final denominator 16, and one
  discard. Finite naive-L2 overflow is downgraded to
  `INCONCLUSIVE_INCOMPLETE` until all completion seams exist; unknown textual
  P62 markers and optimizer activity are fatal. P62 also enters the exact
  fixed-head DP16 local-receipt classifier path.
- Parent base `bdfa50e1` passes V1 38/38, P57 144/144, P59 37/37, APC
  31/31, M15 APC target 9/9, flag audit 371/371, Bash syntax, and diff hygiene.
  Dependency-complete pinned image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  exits zero with unique terminal `V1_HP_EXACT_IMAGE_PASS ... p62_numeric=6
  ... apc_m15_carrier=39 ... manifests=3`. This execution was observed in the
  terminal but was not saved as a new signed raw artifact. No TPU, JobSet,
  optimizer commit, source commit, or push occurred. G5b target remains
  pending.
- Final operator base `41a2043c` adds only a one-line checkpoint exemption for
  the separate zero-commit M15 APC target carrier. Its focused host and pinned-
  image tests pass 9/9; P62 runtime and classifier blobs are unchanged from the
  complete green parent gate.
