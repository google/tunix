# P38s18l Evidence Status

- **Status**: Analysis-grade / Partial run (`INCONCLUSIVE_REDUCTION_JOIN`)
- **Run ID**: `p38s18l` (JobSet `canon-p38-fl-stock-p38s18l-9a834574`)
- **Commit**: `9a83457417fc995079a4beaf7c0c1694f4da605f`
- **Topology**: 64 TPU chips (`DP16xTP4`, concurrency 256)
- **Diagnostic Mode**: `--seam-mode layer` (All 36 Transformer layers intermediate state capture)
- **Raw GCS Live Snapshot**: `gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-fl-stock-p38s18l-9a834574/attempt-0/live/000020/`
- **Derived Compact GCS Archive**: `gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-fl-stock-p38s18l-9a834574/attempt-0/derived/p38s18l-seam-reduction-v1/p38s18l-seam-reduction-v1.tar.gz`
  - Archive SHA256: `90e8bb9b368436a6f73c6e0490da0f9350d8ef0a621792342b705e6201cb6143`
  - Manifest SHA256: `dbbfca0d552ae8e5c410b559eedfb361d0691b8b0ad0bb8bc297420a3b25b410`
- **Reducer Verdict**: `INCONCLUSIVE_REDUCTION_JOIN` (Ambiguous join key detected across call records [319, 398] for prefix hash `729d2e6ec52e...`).
- **V2 selector execution**: reported 22 candidates and zero eligible
  two-round snapshots (`000020` round 0 only; `000021` missing manifest and
  paired NPZs). Commit `e0c1aef7` contains only a prose summary, not the raw
  selector inventory; the local P38.2q amendment must seal and audit those
  bytes before this fact becomes committed mechanical evidence.
- **Core Findings**:
  1. **B-C Boundary (`S_prefill` vs `T_old`)**: `0` mismatches (STRICT EXACT 0 DIFF across all rounds).
  2. **Layer Seam State**: no official classification was produced because
     one round-0 A key matched both records 319 and 398. The old hand-authored
     hidden-exact subset is a candidate direction only.
  3. **Claim ceiling**: neither complete hidden-chain equality nor lm_head,
     gather, normalizer, processing, or subtraction isolation is admitted.
     The no-source result cannot satisfy the 47/94 join gate and cannot promote
     a tail observer. After the selection-only bundle is audited, P38s18l is
     retired and any new acquisition follows P38.2r's combined seam-and-tail
     per-round durability contract.
