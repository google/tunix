# State

- Status: active
- Objective: explain and repair the M15 DP8xTP8 APC-on A-vs-B byte mismatch without changing the independent full-reset B arm or any unrelated numerical path.
- Definition of done: a deterministic clean-run reproducer reaches `FIRST_RED_LOCALIZED`; the smallest localized repair passes host, exact-image, one-host clean/dirty controls, deterministic repeat, and a separately approved DP8xTP8 target run with A-B=0 bytes and B-C=0 bytes.
- Task directory: `canon-zero-tim/tasks/v1-apc-m15-target-debug`
- Current baseline: operator tip `9f91d93001dd5b44659f062626eb93fc65e6fcb4`; Attempt 6 paired run `d12-9f91d930` complete and audited.
- Release state: Attempt 6 paired run complete on 64 TPUs (DP8xTP8). Off control classified `CONTROL_GREEN` ($A-B=0, B-C=0, N_{action}=117415$); On treatment classified `FRESH_TARGET_RED_FROZEN` ($A-B=1770\text{ bytes}, B-C=0, N_{action}=119565$, 748 differing elements). Replay carrier frozen with 256-row producer and 3,027 serving calls. Both arms passed upstream GCS audit `run_m15_replay_gcs_audit.sh`.
- Current phase: Phase C/D (First-red localization and deterministic replay harness), [freeze a replay carrier](phases/phase-b-replay-carrier.md)
- Last verified fact: `run_m15_replay_gcs_audit.sh` executed against both Attempt-0 GCS roots (`canon-v1-apc-m15-off-d12-9f91d930/attempt-0` and `canon-v1-apc-m15-on-d12-9f91d930/attempt-0`). Off returned `status=CONTROL_GREEN` (`receipt_sha256=c9550f73...`); On returned `status=FRESH_TARGET_RED_FROZEN` (`receipt_sha256=557801a3...`). First red joined to source row 245, call 565 (first mismatch call 188), request `400-bc7daec5`, 296 exact joins.
- Next action: construct deterministic single-host or TPU replay harness to localize first red to specific cache/RoPE/KV page operation.
- Blockers: None. Replay carrier is fully frozen and verified.
- Key artifacts: [Attempt-0 receipt](evidence/v1_apc_m15_attempt0_20260825/receipt.json), [Attempt-1 receipt](evidence/v1_apc_m15_attempt1_20260825/receipt.json), [Attempt-2 receipt](evidence/v1_apc_m15_attempt2_20260825/receipt.json), [Attempt-3 receipt](evidence/v1_apc_m15_attempt3_20260825/receipt.json), [Attempt-4 receipt](evidence/v1_apc_m15_attempt4_20260825/receipt.json), [Attempt-5 paired receipt](evidence/v1_apc_m15_attempt5_paired_d11_20260825/receipt.json), [Attempt-6 paired receipt](evidence/v1_apc_m15_attempt6_paired_d12_20260825/receipt.json), [Phase3 state](../v1-phase3-prefix-cache/state.md)
- Validation: APC target-carrier 46/46; P38 classifier 37/37; Phase3 12/12; P57 146/146; V1 Phase4 CPU 67/67; flag audit 378/378; Python/shell syntax and `git diff --check` PASS. Attempt-6 GCS audits PASS with status `CONTROL_GREEN` and `FRESH_TARGET_RED_FROZEN`.
- Limitation: Replay carrier is frozen with verified call sequences and producer tokens, but exact replay execution and mechanism localization have not yet run. Production recipes remain APC-off.
- Updated: 2026-08-26T00:05:00Z
