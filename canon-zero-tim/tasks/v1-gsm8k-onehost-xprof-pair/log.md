# Log

## 2026-08-24 — implementation start

- Rejected the historical P58 DeepSWE Qwen3-4B DP1×TP4 no-commit carrier for
  this purpose: it launches `train_deepswe_nb.py` and does not exercise the
  current GSM8K/P59 update path.
- Reused the proven P59 DP4×TP1 geometry and the existing GSM8K vanilla stock
  admission, but moved both arms onto the same whole-update XProf window.
- Added a default-off matched-work selector and exact pre-update token/
  advantage receipts. Runtime and TPU validation remain pending.

## 2026-08-24 — one-host hardening

- A first Native development run completed 3/3 updates and produced a healthy
  159 MB XPlane plus 35 MB trace, but correctly exposed two postflight bugs:
  root-owned artifacts and reuse of the P55 segmented-backward/19-track census
  for a stock DP4 trainer. Added container-side permission normalization and
  arm-aware XPlane/Perfetto censuses. Reclassification proves 8/8 planes each
  contain 16 stock `jit__train_step` modules and zero decode modules.
- The first Zero-HP attempt failed before training because the runner omitted
  the registered P60 deterministic 1024/256 carrier. Both arms now require
  `CANON_P60_DETERMINISTIC_AB=1`; a narrow signed Native admission preserves
  stock numerical execution while fixing seed/schedule/shape.
- The second Zero-HP attempt reached a strict all-zero pre-gate and exposed
  that the receipt observed `ObservedTrainExample` rather than its underlying
  train example. The receipt now unwraps the host-only sidecar before hashing.
- Final Zero-HP run `zero_dev3_20260824` is GREEN: 3/3 updates, 51/51 strict
  alignment PASS, 8/8 planes with all five P59 backward families and no decode,
  813,929,492-byte XPlane, 33,974,285-byte trace JSON and 12,436-byte semantic
  Perfetto.

## 2026-08-24 — one-host final result

- Final Native run `native_dev2_20260824` is GREEN: 3/3 updates; every one of
  eight TensorCore planes contains exactly 16 stock `jit__train_step` modules
  and no decode module. Its XPlane is 159,449,612 bytes
  (`a2367fc94d4fa3643b5895e9cef068383c8d464bfc8b945bc95e0ab14186e4a6`),
  trace JSON is 34,779,292 bytes
  (`48dab817baf38549db84429f358f7f1f0ccb7d363ca59bceb175088f1886214f`),
  and semantic Perfetto is 15,302 bytes
  (`f63521cdd7263d0d3e0c1b4b92305aded078b07756d4bd2c52b0c0b6bb9e16c7`).
- Zero-HP `zero_dev3_20260824` remains GREEN on all eight planes. The five P59
  backward families are present, decode is absent, and the profiled transaction
  includes the fixed reducer and optimizer commit.
- The pair classifier returns `INCONCLUSIVE_INPUT_MISMATCH`, not FAIL. Both
  arms used the same source diff, image, model snapshot, DP4xTP1 topology,
  update 1 window, prompt ids, prompt mask, policy version and static shape.
  Completion ids/masks and advantages differ. Same seed therefore did not
  freeze stochastic work across numerically different inference programs.
- The `xprof-trace-analysis` comparison confirmed the expected program-shape
  change (Native monolithic `jit__train_step`; Zero-HP decomposed canonical
  forward/P59 backward), but is not a timing verdict because work differs.
  It also showed why completeness is gated on XPlane: Native trace JSON exposes
  11 modules on the selected plane while the full XPlane proves 16/16 on all
  planes.
- The repository `read-xprof` workflow was also applied to the full XPlanes.
  Native has 128 total `jit__train_step` programs across eight planes and
  18.47 device-seconds of monolithic update work. Zero-HP has the expected
  decomposed modules: canonical forward, P59 parallel backward, fixed reducer,
  replica comparison, and optimizer transaction. Its largest all-plane module
  families are forward layer (49.82s), fixed reduction (24.92s), parallel layer
  backward (13.17s), and replica comparison (9.54s). These numbers are
  attribution only and cannot be converted into an A/B speed ratio because the
  profiled work hashes differ.
- `scripts/analyze_gsm8k_xprof_pair.sh` was exercised on the real artifacts. It
  returned the expected exit 3, preserved both arm PASS verdicts, named the
  four differing arrays, produced the compact XProf comparison, and wrote a
  SHA ledger. This removes wildcard/manual-path ambiguity from the handoff.

## 2026-08-24 — operator handoff expansion

- Rewrote `HANDOFF.md` as a cold-start execution recipe: exact Native and
  Zero-HP commands, clean-vs-dirty evidence grade, expected arm-specific
  backward markers, output-root derivation, pair exit-code handling, artifact
  authority, and the complete return manifest. No runtime behavior changed.
