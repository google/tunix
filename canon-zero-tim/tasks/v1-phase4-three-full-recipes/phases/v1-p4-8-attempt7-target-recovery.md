# V1.P4.8 — Attempt-7 target recovery

Status: active; no target relaunch authorized.

## Objective

Recover the two independent Attempt-7 failures without weakening strict
Zero-TIM or changing optimizer mathematics to hide a non-finite gradient:

1. make the Pathways XProf capture use a unique GCS directory and restore the
   complete trace into the existing local postflight evidence directory;
2. determine whether GSM8K's very large but finite gradient scale is legitimate
   or caused by topology/ownership duplication using a frozen, no-commit
   replay; and
3. identify P45's earliest rank-1 non-finite boundary before proposing a
   numerical repair; and
4. capture the exact strict-prealignment P45 training payload once, then allow
   hash-bound diagnostic replay without repeating rollout or B-arm rescore.

## Frozen contracts

- Strict alignment remains an absolute hard gate. Any `CANON_ALIGN` failure
  kills that arm.
- P63 remains unchanged: it may repair FP32 sum-of-squares overflow only when
  every gradient value is finite. NaN/Inf remains fatal.
- No localization carrier performs an optimizer commit.
- B-arm rescore remains a full reset/recompute and is never replaced by an APC
  cache hit.
- Existing workload shapes stay fixed: GSM8K DP16xTP4 uses global M4096 and
  local M256; P45 DP8xTP8 uses global M2048 and local M256.
- Failed and inconclusive evidence is append-only. A new code image or payload
  receives a new run label.
- A P64 capture still executes all 32 backward groups. Replay executes only
  group 0, performs no optimizer commit, and is always labelled
  `certification=0`; it cannot replace a fresh strict target certification.
- The capsule is immutable and binds all 17 training/observation arrays, the
  exact P45 geometry, a bounded live-model fingerprint, and independent file
  hashes. Replay fails closed on any identity, shape, dtype, model, or hash
  mismatch.

## Work packages and gates

### G1 — XProf GCS capture and durable local restore

- Derive one exact `gs://` XProf directory from the immutable JobSet/run label;
  a local `/tmp` directory is invalid under Pathways.
- After workload completion, synchronously copy the GCS capture into the
  existing `${CANON_STATE}/xprof-update` postflight directory.
- Require both XPlane and trace JSON artifacts plus a source/destination
  receipt. Missing tools, copy errors, empty captures, or unexpected paths are
  infrastructure failures, not numerical failures.
- Positive and negative host tests must cover the exact rendered GCS path,
  wrong path/profile, missing artifacts, and the final local classifier.

### G2 — GSM8K fixed-replay scale carrier

- Freeze checkpoint, seed, batch tokens, masks, advantages, denominator, and
  one trajectory payload. Do not regenerate rollout independently per arm.
- Compare the ordinary/native reference backward and the P59 grouped backward
  under the same topology. Record per-leaf ownership, max-abs, scaled-L2 norm,
  denominator, and representative exact hashes.
- Use an FP64 oracle on a bounded projection/layer slice to validate both arms'
  gradient direction and scale. Serial/P59 byte identity is not required; a
  topology-dependent multiplicative scale or duplicated replicated leaf is a
  red.
- Terminal must state `commits=0` and preserve the complete raw log.

### G3 — P45 rank-1 first-red localization

- Add default-off observation at successively earlier boundaries: scalar loss
  and cotangent, report adjoint, fixed LM-head input/output gradients, then
  layer inputs/outputs.
- Report rank, leaf/path, dtype, shape, finite count, first non-finite index,
  max finite magnitude, and the preceding finite boundary. Observers must not
  replace, clamp, cast, or rescale runtime values.
- Stop after the earliest reproducible transition from finite to non-finite;
  do not jump directly to a repair.

### G3a — Frozen P45 training capsule

- Capture only after strict prealignment passes, before backward mutates any
  accumulator. Preserve prompt/completion IDs and masks, advantages, old
  logprobs, S-decode/S-prefill/T-old, policy versions, sampling values, and
  completion-valid masks in one atomic immutable capsule.
- Upload the capsule and model-binding sidecar to a unique append-only GCS URI.
  Replay must download and verify both SHA-256 values before constructing a
  training batch.
- Replay bypasses environment, rollout, and B-arm rescore, but rechecks the
  frozen A/B/C values and live model binding before executing only reverse
  group 0. It remains a localization accelerator, never certification.

### G4 — Admission

- Re-run focused tests, V1/P57/P59/APC suites, flag audit, syntax checks, and
  `git diff --check`.
- Run the complete pinned-image gate on the exact final runtime tree.
- A one-host or 64-chip TPU carrier requires separate explicit user approval.

## Decision table

| Observation | Verdict | Next action |
|---|---|---|
| GCS capture restores complete XPlane + trace and classifier passes | XProf infrastructure repaired | admit G1; do not infer numerical correctness |
| GSM native and P59 match FP64 within registered tolerance and have the same ownership/denominator | scale carrier green | retain P59; treat large finite scale as workload evidence |
| GSM shows duplicated ownership or a constant topology factor | P59/topology red | repair at the first ownership/reduction boundary and repeat G2 |
| P45 identifies a finite-to-nonfinite transition | localized numerical red | pre-register one surgical repair at that boundary |
| P45 only reaches the final staged tree | inconclusive | add one earlier observer; do not sanitize at clipping |
| P45 capture is strict-green and hash-bound | reusable diagnostic input | replay group 0 only; retain capture as the certification source |
| Capsule or model binding differs | fatal input mismatch | do not run backward; issue a fresh capture label |
| Any strict alignment failure | hard FAIL | revert the candidate change and preserve evidence |

## Rollback

The XProf transport concern must remain separable from numerical observers.
Disable/revert localization flags and profile opt-ins without touching P63 or
strict alignment. Preserve every target and negative-control artifact.

## Result log

- 2026-08-25: Attempt 7 GSM8K completed two real optimizer commits with 35/35
  strict alignment records green. Both gradient trees were independently
  all-finite and used P63 only because naive FP32 sum-of-squares overflowed.
  Step 2 stopped before training when Pathways rejected the local XProf path;
  verified by `evidence/v1_hp_three_full_attempt7_20260825/gsm8k_g8_error.log`.
- 2026-08-25: Attempt 7 P45 passed rollout/alignment but stopped before its
  first optimizer commit with 253 non-finite staged gradient leaves on DP rank
  1. This is verified by
  `evidence/v1_hp_three_full_attempt7_20260825/p45_p8_error.log`; the earliest
  generating boundary is not verified because the current log observes only
  the finalized staged tree.
- 2026-08-25: the later committed M15 Attempt-7 artifact verifies the same
  failure family on a different rank. M15 passed strict prealignment for
  118,816 action tokens with both byte deltas zero, then its first staged DP
  reduction found 122 non-finite leaves on rank 3 before any optimizer commit.
  The rollout solve ratio was 0.156. Raw log SHA is
  `9f221091fd685b7303bc8203fffc4e931191faecc260145d27f167d2eddc9492` in
  `evidence/v1_hp_three_full_attempt7_20260825/m15_m8_error.log`. This is
  verified target evidence; its earlier generating boundary remains
  unverified because M15 has no P64-style first-red carrier.
- 2026-08-25: G1 construction is verified by host and pinned-image tests. Both
  V1 profiles now derive an attempt-isolated `gs://.../p33/<job>/attempt-<n>/`
  XProf path. Step 90 restores the capture into the corresponding local state
  directory and requires nonempty XPlane plus trace JSON files before the full
  classifier runs. Wrong identity, transport failure, and either missing
  artifact class are fail-closed. No real Pathways capture has yet exercised
  the transport, so this is not target evidence.
- 2026-08-25: G3 carrier construction is verified. Default-off P64 is admitted
  only for strict original-P45 DP8xTP8, APC-off, P59 fixed-head,
  `backward-no-commit`. It observes ordered loss cotangent, rank-indexed group
  input cotangent, engine VJP, trainer rank-local adjoint, fixed DP reduction,
  scaled microgradient, and final accumulator boundaries; the first real
  non-finite value aborts immediately and the all-finite terminal discards the
  accumulator. The production P45 resident profile remains unchanged and
  closed to no-commit execution.
- 2026-08-25: host gates pass V1 59/59, P57 144/144, P59 37/37, APC 31/31,
  registry 373/373, syntax and diff hygiene. Pinned-image r1 stopped on a test
  fixture whose toy accumulation count was 2 instead of its registered 4; it
  is `INCONCLUSIVE_TEST_FIXTURE`, not a numerical verdict. Focused correction
  passed, then the complete immutable-image rerun exited zero with terminal
  `V1_HP_EXACT_IMAGE_PASS ... p62_numeric=6 p64_numeric=4 p63_clip=1 ...
  manifests=3`. The terminal was observed in the execution transcript; no new
  filesystem-backed raw-log artifact was created by this invocation.
- 2026-08-25 (pre-G2 checkpoint): G2 fixed GSM8K replay and all new TPU/target
  observations were still unverified. No JobSet, TPU, optimizer commit, source
  commit, or push occurred in that recovery checkpoint.
- 2026-08-25: G2 construction is now verified on 64 forced CPU devices in the
  immutable pinned image. One frozen seed-42 capsule supplies both ordinary
  JAX and P59 DP16xTP4 arms. Relative-L2 versus FP64 is `5.8479e-8` ordinary
  and `7.2023e-8` P59; inter-arm relative-L2 is `9.1470e-8`, both cosine values
  exceed `0.99999999999999`, all 16 groups have 16 distinct rank partials, and
  wrong-denominator plus duplicate-DP-sum negatives each separate at about
  `15.0`. The gradients need not be byte-identical and are not. Durable
  receipt and checksum are under
  `evidence/v1_hp_gsm_fixed_replay_exact_image_20260825_r1/`; receipt SHA is
  `f226097c0c0f0239bec23d91dfe09c31d90ed9a87ef4d2cdf39d9aec71be0f6e`.
- 2026-08-25: after G2 was added to the complete gate, the final pinned-image
  rerun exited zero with `V1_HP_EXACT_IMAGE_PASS ... p64_numeric=4
  p63_clip=1 gsm_scale_replay=1 ... manifests=3`. This admits only bounded
  topology/scaling correctness. Full-Qwen GSM target comparison and the real
  P45 first-red observation remain unverified.
- 2026-08-25: final audit strengthened the P64 classifier before target use.
  A finite completion now requires complete group-0 and group-31 boundary
  chains, one final accumulator, and one discard; a first non-finite receipt
  must be terminal. Host V1 passes 64/64 and the exact immutable-image rerun
  exits zero with `V1_HP_EXACT_IMAGE_PASS ... p64_numeric=4 p63_clip=1
  gsm_scale_replay=1 ... manifests=3`. Verified by the host and pinned-image
  gates; not verified on TPU because no P64 JobSet has been launched.
- 2026-08-25: G3a construction is verified by host and immutable pinned-image
  gates. `tunix/rl/p64_training_capsule.py:252` persists 17 arrays only after a
  strict prealignment PASS, verifies per-array and whole-file SHA-256, and
  binds the payload to the exact P45 geometry plus a bounded live-model sample.
  `agentic_rl_learner.py:2853` bypasses environment/rollout/B rescore only in
  replay, while `canonical_qwen3_adapter.py:6810` executes reverse group 0
  with `optimizer_commits=0` and `certification=0`; capture at
  `agentic_grpo_learner.py:1869` still executes all 32 groups after storing the
  source. Host gates pass V1 67/67, P57 144/144, P59 37/37,
  APC 31/31, flags 378/378, syntax, and diff hygiene. The durable 966-line
  pinned-image log exits zero with one `V1_HP_EXACT_IMAGE_PASS ...
  p64_capsule=3 ... manifests=3`, raw SHA
  `c8121b6668e4fbdcceec14214966c7ba8ef55ba30ff4b9b1a52e1baa7c70177c`;
  receipt and checksums are under
  `evidence/v1_hp_p64_capsule_exact_image_20260825_r1/`. Verified by host and
  pinned image; not verified on TPU because no P64 capture or replay JobSet was
  launched. No optimizer commit, source commit, push, or TPU use occurred.
