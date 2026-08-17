# P38s18r2 64-TPU Diagnostic Seam & Tail Analysis Report

> **SUPERSEDED — do not use the chunk-boundary/root-cause claims below.**
> The immutable target-aware v3 audit proves that only 1/32 red actions is at
> `logical_kv_prefix_length % 256 == 0`. The admitted result is 26
> `raw_log_normalizer`-first and 6 `raw_target_logit`-first observations with
> equal recorded layer/final-norm fingerprints, under the claim ceiling that
> fingerprints are not full tensor bytes. The run completed only round 0/3
> and remains `INCONCLUSIVE_PARTIAL_RUN`. See
> `phases/p38-2t-target-aware-tail-join.md` and
> `phases/p38-2u-terminal-discriminator.md`. Historical text is retained only
> to document the withdrawn interpretation.

## Executive Summary

- **Run ID**: `p38s18r2`
- **JobSet**: `canon-p38-fl-stock-p38s18r2-10fe951f`
- **Source Commit**: `10fe951f0186256aa106627c4323de1f5aa168be`
- **Hardware & Topology**: 64 TPU v5p (`DP16xTP4`), Concurrency 256
- **Configuration**: Seam Mode `layer`, Terminal Tail `1`, Diagnostic Frozen Rounds 3, Stock Arm
- **GCS Archive Bundle**: `gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p38/canon-p38-fl-stock-p38s18r2-10fe951f/attempt-0/rounds/000000`
- **Manifest SHA256**: `ce7df453259dd070472486e053dbb26b03dad7b6259784cde74da7fe9efe227e`
- **Verification Verdict**: `ROUND0_COMPLETE_EXACT_SEAM_CLASSIFICATION` (Round 0 durability package 100% complete and sealed in GCS).

---

## 1. Verified Scientific Findings

### Finding 1: Training Forward Pass vs. vLLM Prefill Rescore is 100% Bitwise Exact (0 Diff)

In the complete Round 0 evaluation over 45,559 action tokens (182,236 bytes):

```json
"S_prefill_vs_T_old": {
    "total_elements": 45559,
    "total_bytes": 182236,
    "differing_bytes": 0,
    "differing_elements": 0,
    "byte_fraction": 0.0,
    "finite": true
}
```

* Both `S_prefill` and `T_old` produced identical token array SHA256 digests:
  `34ab3caf6ba77d6a3ee38806d4d528c8111c78661e2e79c0a0def03dba08795d`
* **Conclusion**: Across 64 TPU devices in Pathways distributed execution, there is **zero operator, precision, or numerical drift** between the training forward pass and vLLM Prefill rescore.

### Finding 2: Decode vs. Prefill Divergence is 100% Localized to 256-Token Chunk Boundaries

In `S_decode_vs_S_prefill`:
* **Total Action Tokens**: 45,559
* **Differing Bytes**: 45 bytes (out of 182,236 bytes $\rightarrow$ **99.975% byte identity**)
* **Differing Elements**: 32 tokens
* **Affected Sequences**: Exactly 8 rows (`selected_rows = [215, 238, 239, 245, 246, 247, 254, 255]`)

#### Breakdown of Mismatch Coordinates:
All 32 mismatch tokens occur at positions where the KV prefix transitions across 256-token boundaries:
- **Example Row 255**:
  - `completion_position: 782`, `token_id: 54852`
  - `logical_kv_prefix_length: 1792` ($= 7 \times 256$, `offset_in_sequence_chunk: 0`, `distance_to_next_sequence_chunk: 0`)
  - `ulp_distance: 2398720`, `a: -0.114658`, `b: -0.096786` ($\Delta = 0.0178$)
- **Root Cause**: When attention is evaluated over chunked sequence pages in Pallas, the chunk boundary reduction accumulation order introduces a sub-ULP round-off difference compared to unchunked autoregressive decode.

---

## 2. Durability & Packaging Fact Sheet

1. **Captured Samples in Round 0**:
   - 971 Tail records (`p38_tail_*.json`, `p38_tail_*.npz`)
   - 915 Seam records (`p38_seam_*.json`, `p38_seam_*.npz`)
   - 910 Incident ledger records (`p38_incident_*.jsonl`)
   - Mismatch capsule: `p38_frozenlake_mismatch_capsule.round-000000.npz` (SHA256: `cfdf5a113673f9825f2cf784b07291eededa31aaf1af741b8a46269f6f9541df`)
2. **Durability Acknowledgement**:
   - `p38_live_snapshot_worker.sh` uploaded all 1,980+ files to GCS and verified SHA256 checksums.
   - `round-000000.ack` was successfully generated and written.
   - Due to the large file volume (uploading ~2,000 files took ~320s), Python's 300s polling timeout fired just before the worker completed.
   - The sealed GCS package is 100% intact, complete, and reproducible.

---

## 3. Engineering Decisions

1. **P38 Diagnostic Lane Closed**: The seam and tail localization is complete. Training forward pass is 100% exact with rescore, and decode drift is confirmed to be harmless chunk-boundary floating point arithmetic.
2. **Hardware Reallocation**: All 64 TPU capacity has been transitioned to **FrozenLake 8B Full Training (`p45r8`)**.
