# State

- Status: active
- Objective: explain and repair the M15 DP8xTP8 APC-on A-vs-B byte mismatch without changing the independent full-reset B arm or any unrelated numerical path.
- Definition of done: a deterministic clean-run reproducer reaches `FIRST_RED_LOCALIZED`; the smallest localized repair passes host, exact-image, one-host clean/dirty controls, deterministic repeat, and a separately approved DP8xTP8 target run with A-B=0 bytes and B-C=0 bytes.
- Task directory: `canon-zero-tim/tasks/v1-apc-m15-target-debug`
- Current baseline: operator tip `ceb3d1a5c62692a1e601459986d622ad32d86dab`; Attempt 5 ran source `a909fda14ee3f7e5d2334812a02b1f8ef94b0fbb`.
- Release state: Attempt 5 has two hash-valid Git log snapshots. They show 0.0% prefix-cache hits off and approximately 89.4%--97.5% on, but no committed alignment, sampler, controlled-exit, classification, or GCS-terminal markers.
- Current phase: Phase B ATTEMPT-5 GCS AUDIT PENDING, [freeze a replay carrier](phases/phase-b-replay-carrier.md)
- Last verified fact: committed Attempt-5 `SHA256SUMS` passes 3/3; both 33-KiB snapshots contain zero `CANON_ALIGN_PRE`, sampler-contract, controlled-exit, classification, and GCS-terminal markers. A/B/C status is unknown from committed evidence.
- Next action: on the bucket-capable host, run `run_m15_replay_gcs_audit.sh` separately for the off and on Attempt-0 roots and return the two machine-generated small bundles specified in `HANDOFF.md`.
- Blockers: the GCS-derived off/on audit bundles have not been returned; no control or treatment numerical verdict is currently admissible.
- Key artifacts: [Attempt-0 receipt](evidence/v1_apc_m15_attempt0_20260825/receipt.json), [Attempt-1 receipt](evidence/v1_apc_m15_attempt1_20260825/receipt.json), [Attempt-2 receipt](evidence/v1_apc_m15_attempt2_20260825/receipt.json), [Attempt-3 receipt](evidence/v1_apc_m15_attempt3_20260825/receipt.json), [Attempt-4 receipt](evidence/v1_apc_m15_attempt4_20260825/receipt.json), [Attempt-5 paired receipt](evidence/v1_apc_m15_attempt5_paired_d11_20260825/receipt.json), [Phase3 state](../v1-phase3-prefix-cache/state.md)
- Validation: APC target-carrier 46/46; P38 classifier 37/37; Phase3 12/12;
  P57 146/146; V1 Phase4 CPU 67/67; flag audit 378/378; Python/shell syntax
  and `git diff --check` PASS. The aggregate pinned-image gate exits 0 with
  `apc_m15_carrier=46`, `p64_numeric=4`, and `p64_capsule=3`.
- Limitation: cache on/off behavior is observed in snapshots, but paired rollout completion, sampler admission, A/B/C numerics, and replay-carrier completeness are not mechanically verified from the committed subset. Production recipes remain APC-off.
- Updated: 2026-08-25T22:40:00Z
