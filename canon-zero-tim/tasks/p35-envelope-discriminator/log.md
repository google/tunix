# P35 envelope discriminator log

## 2026-08-09 — Task bind and metric correction

- Bound the phased task to `canon-zero-tim/tasks/p35-envelope-discriminator/` on commit
  `ad309a810e35121d7d25db67c32c2712d9f8e086`.
- Reconciled the actual `T_old` hot path: outer JIT plus `lax.map`, then complete
  `runner.model_fn` per canonical 256-token group. The P28 per-layer segmented reverse is not the
  r18 value path.
- Corrected r18 interpretation: `differing_bytes / N_action` is dimensionally invalid. Existing
  artifacts support byte fractions only; element/token mismatch rates require new instrumentation.
- Started an additive alignment-report change that keeps legacy fields while adding element-level
  counts, exact denominators, fractions and action-masked hashes.
- No TPU experiment, cloud mutation, commit or push was performed.

## 2026-08-09 — P35.1 local gate complete

- Added exact element and byte denominators/fractions plus masked hashes to the pre-backward and
  four-boundary reports. Legacy `differing_bytes` fields and console lines remain available.
- Added a fail-closed P35 classifier for the four A/B/C outcomes.
- Exact-image result: alignment 13/13 PASS; classifier 5/5 PASS.
- Local Python compilation, `git diff --check` and executable English-only scan PASS.
- Advanced the active phase to P35.2. The target producer and 64-chip run remain NOT RUN.

## 2026-08-09 — P35.2 serving B-arm primitive

- Added a default-unused grouped native-prefill primitive. It submits an exact number of fixed
  request groups through the same real serving API and resets prefix cache before each group.
- Added an RL-cluster passthrough and CPU controls for two complete groups and a rejected partial
  group.
- This produces the A-vs-B scheduling variable. It does not yet attest actual page tables or
  trainer-to-engine weight equality, so the target runner remains not admitted.

## 2026-08-09 — Publication approval

- The user explicitly approved commit and push.
- Remote `origin/yuxzhang/canon-zero-tim` was fetched and remained exactly at the local base
  `ad309a810e35121d7d25db67c32c2712d9f8e086`; no rebase was required.

Artifacts to preserve:

- `canon-zero-tim/debug_logs/p33_r18_gsm8k_full.raw.log`
- `canon-zero-tim/debug_logs/p33_r18_fl_align.raw.log`
- the complete generic way-count target log referenced by the package evidence manifest

Rollback: disable the P35 runner and retain all old logs. Do not overwrite r18 artifacts.
