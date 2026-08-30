# P58.22 — Qwen3-4B continue-decode KV/state repair

Status: completed alignment prerequisite; serial backward superseded by
P58.23.

## Construction checkpoint

The default-off discriminator and first repair are implemented locally on source base
`16c224aa80eb6b3a544be19f693c0542ab4b0dcb`.  Append-only runner patch 35
selects at most one serial continue-decode request in the fixed logical-prefix
window `[2280,3072)`, then delegates live-A and clean-B capture to the existing
P38 integer KV fingerprint implementation.  The wrapper fixes candidates,
pages, output bytes, and read bytes to `1/192/128MiB/640MiB`; the classifier
requires P58.20 A-B RED, exact B-C, one exact token-prefix join, no backward,
capture end strictly beyond the observed first mismatch, effective
device-to-slice sharding equality, and zero commits before returning either write/state-suspect or
read/program-suspect.  After target `p58s22kv6`, the same diagnostic also has
a bounded alignment-only result: it sets the existing P38 precheck and
controlled-exit controls as a pair, records them in the manifest, and accepts
strict A=B=C only with exit 42, both stop markers, no backward artifact, and
zero-commit semantics.  This never claims backward admission.

The previous target-derived construction gates passed: Qwen3-4B TP4 installed
all 37 overlay files; the assembled runner probe emitted
`P58_CONTINUE_KV_OVERLAY_PASS cases=4/4`; P34
static emits `P34_STATIC_PASS suites=10`; the flag registry is
`400/400/400` with `FLAG_AUDIT_PASS`; and the complete digest-pinned image
gate exits zero with `P58_EXACT_IMAGE_CPU_PASS ... continue_kv_observer=1 ...
regressions=1`.  The two signed prefix bounds are registered diagnostic flags,
default absent, and exact-value negative controls pass.  This is construction
evidence only.  The effective-sharding and prefix-window repair repeated those
gates before `p58s22kv4`.  The later partial-rescore repair has a focused
exact-image marker `P58_CONTINUE_KV_OVERLAY_PASS cases=8/8`; the complete
pinned-image rerun exits zero with `continue_kv_observer=1` and
`regressions=1`.

## Target-derived instrumentation incidents

- `p58s22kv_20260829t0624z` stopped before model load because the P58 selector
  inherited P38's generic serving-capture requirement.  P58 now explicitly
  permits its own bounded observer directory while ordinary P38 still requires
  generic serving capture; no trajectory, fingerprint, backward, or commit was
  produced.
- `p58s22kv2_20260829t0635z` reached four real TPU devices, the Qwen3-4B model,
  and candidate prefix 2207, then stopped because the inherited 16-page budget
  could not hold 142 logical pages.  The exact P58 bound is now 192 pages,
  `ceil(3072/16)`; ordinary P38 remains capped independently.
- `p58s22kv3_20260829t0647z` produced both immutable A/B records for identical
  2,270-token prefixes.  A read-only comparison found zero aggregate/sample
  fingerprint cells different, but this is **not** the discriminator result:
  the old classifier compared `NamedSharding` repr strings even though the
  additional A/B axes had size one, and the capture ended before the actual
  first RED at prefix 2,286.  The repair records a canonical device-to-slice
  effective sharding map and moves the lower selector bound from 2200 to 2280
  so the next capture must include the RED seam.  The failed artifact remains
  diagnostic construction evidence only.
- `p58s22kv4_20260829t071538z` selected the correct pre-RED seam at prefix
  2,285, captured live A through token 2,472, and reproduced the immutable
  first RED at 2,286 with exact B-C.  Clean rescore ran, but the generic P38 B
  hook waited for `seq_len == num_tokens`; prompt-logprob scoring intentionally
  need not execute the final input token, so no B record was written before
  the strict alignment gate stopped.  P58 now captures B at the first clean
  chunk where `seq_len >= A.target_seq_len` and the full host token prefix
  matches exactly.  Ordinary P38 retains the full-request requirement.  A
  focused pinned-image positive plus generic-negative probe passes 6/6.
- `p58s22kv6_20260829t0802z` captured live A at candidate prefix 2285 through
  target 2472, but correctly refused B: clean rescore token IDs first diverged
  at position 2242.  The live and durable trajectory streams decode to the
  same text (`<parameter=path=./PIL/Image.py>`) but use different BPE
  segmentations (`97183` versus `28,1725`).  Thus this is not yet a KV-state
  discriminator.  Each later agent turn rendered text and re-tokenized the
  whole conversation for serving, while the durable trainer sequence
  concatenated original sampled token IDs plus environment token IDs.  The
  repair passes the exact accumulated token IDs to vLLM from turn two onward,
  scoped atomically to `CANON_P58_Q4_TP4_ZERO_ADMISSION=1`; normal DeepSWE,
  Native/IS, TP8, and other models retain the text path.  Focused agentic and
  sampler tests pass, including missing-environment and unsigned-route
  negatives.  A fresh real alignment-only rerun is still required.

## Bound evidence

P58.20 (`p58s20dev_20260829t0330z`) is finite A-B RED and exact B-C with
`CANON_CONTINUE_DECODE=8`; its first RED is the first action token following
an environment result.  P58.21 (`p58s21std_20260829t0357z`) changes only that
value to standard decode and is bitwise A=B=C over 2,553 action tokens.  The
P58.21 full 8,192-token backward did not finish compiling inside the bounded
7,200-second process window, so backward remains unproved.

## Repaired target evidence

- Development target `p58s22kv9d_20260829t0846z` kept continue-decode at `8`,
  prefix cache off, and the full Qwen3-4B TP4 Zero overlay.  Exact cross-turn
  token continuity admitted 2,413 real action tokens and returned strict
  A=B=C: both boundaries had zero differing elements and bytes.  The signed
  P38 controlled exit returned code 42 before backward and the packaged
  classifier reported `EXACT_TOKEN_CONTINUITY_ALIGNMENT_PASS`.  This is real
  alignment evidence only; it does not prove backward or TP8.
- First short-backward target `p58s22bw1_20260829t0906z` used 2,048 prompt plus
  2,048 response tokens.  Both fixed-seed trajectories reached the official
  max-context outcome, so compact filtering produced `N_action=0`; the strict
  gate stopped before backward.  The long alignment target's completed row
  used 2,862 response tokens, so the compilation carrier now uses 2,048 prompt
  plus 3,072 response tokens (train width 5,120).  It does not admit clipped
  rows or change overlong reward semantics.
- `p58s22bw2_20260829t0924z` stopped before rollout because the training
  entrypoint's fail-closed expectations accidentally reversed the new prompt
  and response maxima.  The runner and manifest were correct.  The mapping is
  repaired and a regression now asserts the directional 2,048/3,072 contract;
  the complete digest-pinned image gate passes after the repair.
- `p58s22bw3_20260829t0931z` produced one successful trajectory plus one
  officially compact-filtered overlong row and returned strict A=B=C over
  2,413 action tokens.  The 5,120-token backward entered the real VJP compile
  but did not finish inside 7,200 seconds, so its result is
  `ZERO_TIM_BACKWARD_INCOMPLETE`, not PASS.
- `p58s22bw4_20260829T114504Z` and `p58s22bw5_20260829T115556Z` tried two
  clean-census tasks at width 4,096.  Both fixed-seed rows in both runs were
  compact-filtered at the context limit, so each stopped with
  `INCONCLUSIVE_NO_ACTION_TOKENS` before backward.
- The default-off rollout-only carrier screen then proved that it exits before
  canonical rescore/backward and never commits.  `p58s22cs1_20260829T120815Z`,
  `p58s22cs2_20260829T121419Z`, and `p58s22cs3_20260829T122139Z` screened
  Scrapy LocalCache, Scrapy JSON-set, and DataLad repr respectively.  Each had
  two `MAX_CONTEXT_LIMIT_REACHED` rows at a 2,048-token response cap and
  correctly returned `CARRIER_SCREEN_INCONCLUSIVE`; none is training evidence.
- These three independent negatives show that a 2,048-token completion cap is
  not a useful fixed-seed carrier for this Qwen3-4B agent.  The active target
  `p58s22bw6_20260829T123125Z` therefore reuses the already-proven Pillow
  trajectory with minimal observed padding: prompt/response `1792/2880`, train
  width `4672`.  Its signed six-hour process bound and persistent JAX cache
  `/mnt/disks/tunix-data/jax-compilation-cache/p58-q4-tp4-short-backward` are
  compilation controls only.  The target is in progress and must not be
  promoted without the packaged classifier PASS.

## Work

1. Keep `CANON_CONTINUE_DECODE=8` and add one default-off, P58.20-only bounded
   KV/state diagnostic.  It must fingerprint the live A cache at the
   continue-decode-to-environment seam and the clean B cache for the identical
   logical token prefix using the existing integer `p38_kv_fingerprint`
   implementation.  It must record request identity, logical prefix, block
   table/pages, per-page valid extent, effective device-to-slice sharding,
   program chronology, and exact artifact checksums.  Unknown paths, missing
   A/B join, token-prefix drift, capture ending before the first RED, B-C RED,
   observer budget overflow, or observer-on/off output drift fail closed.
2. Preserve the exact token-prefix fail-closed join.  `kv6` established a
   pre-KV BPE segmentation drift, so do not compare cache fingerprints until
   live and clean prefixes are byte-identical.
3. Reuse original sampled and environment token IDs on every subsequent
   agentic model call, then rerun baseline value `8`.  Do not change numerical
   flags or relax alignment.
   Standard decode is a control, not the shipped high-performance admission.
4. For the real backward gate, use the separately attested one-host-only short
   carrier with prompt/response maxima `1792/2880` (train sequence `4672`).
   The signed Pillow prompt is 1,737 tokens and its known successful completion
   is 2,862 tokens with 2,413 admitted action tokens.  The run must include a
   real environment seam, strict A=B=C, finite nonzero repeat-exact backward,
   unchanged model and TPU-resident optimizer state, and zero commits.  The
   persistent compilation cache and 21,600-second bound are mandatory and
   recorded; they do not alter the executable's math.

The short carrier is only a compilation-cost control for direct-host
admission.  It must not change P58 production 16K/50-turn geometry, TP8
profiles, model math, loss, sampling, precision, or optimizer placement.

## Exit

P58.22 exits only after repaired `continue_decode=8` passes the short real-R2E
one-host strict gate and its observer-off repeat remains bitwise exact.  Only
then may a distinct TP8 phase begin.  A diagnostic fingerprint result alone,
or a standard-decode PASS, is not admission.
