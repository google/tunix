# State

- Status: active
- Objective: publish and render one simultaneous three-recipe full wave using the P66 checked-VMA backward repair, with an independent fail-closed first-update gate on every recipe.
- Definition of done: GSM8K DP16xTP4 plus P45/M15 DP8xTP8 complete their signed horizons with every strict Zero-TIM gate green and durable optimizer, timing, XProf, Perfetto, cache, evaluation, and checkpoint evidence.
- Task directory: `canon-zero-tim/tasks/v1-phase4-three-full-recipes`
- Directory state: clean isolated worktree at `/home/yuxuan/code_rl_repro/worktrees/p66_gsm8k_convergence_0825`, branch `local/p66-gsm8k-onehost-convergence`. Four local P66/P4.9 CLs are rebased onto fetched operator tip `75e97a1db4a4bb328fa174f75869f039defc4b98` and pass post-rebase admission; publication and final rendering are pending. Nothing was launched.
- Current phase: V1.P4.9 checked-VMA three-full launch wave. Runtime/profile promotion, first-update hard receipts, host admission, and immutable-image construction admission are complete. Publication, final render, target execution, convergence, and performance analysis are pending.
- Last verified fact: P64 target evidence localized the old P45 first red to the Group-0 engine VJP on nonzero cotangents. P66 G1/G1.5 then identifies erased VMA ownership in old P59 and passes the same-point ordinary-JAX oracle with worst relative-L2 `0.0052568`. The final rebased tree passes V1 74/74, P57 146/146, P59 37/37, P66 16/16, P61 6/6, APC 31/31, flags 383/383, syntax, diff hygiene, and immutable image `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`. Its unique terminal includes `p59_checked_vma_real_shim=4`, `first_update_gate=4`, `apc_m15_carrier=46`, and `manifests=3`.
- Next action: fetch once more, require the operator tip has not moved, push/read back the four-CL stack, then use `scripts/prepare_checked_vma_three_full_wave.sh` and let the user launch GSM8K/P45/M15 together.
- Blockers: no approved published SHA contains P66/P4.9, so no final manifest is launchable. DP16xTP4 and DP8xTP8 optimizer correctness, convergence, and performance remain target-unverified. GCS XProf transport remains construction-only until a real Pathways capture restores.
- Key artifacts: `phases/v1-p4-9-checked-vma-full-wave.md`; `scripts/prepare_checked_vma_three_full_wave.sh`; `RUNBOOK.md`; `evidence/v1_hp_p64_remote_64tpu_20260825/`; `../p66-onehost-gsm8k-convergence/evidence/p66-g15-onehost-20260826/receipt.json`.
- Updated: 2026-08-26T00:59:00Z
