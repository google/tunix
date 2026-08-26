# State

- Status: active; Attempt 10 strict Step-0 forward is target-green, but both full runs stopped at the first gradient sink; the checkpoint-admission repair is host/full-image green and published by the current CL, with exact identity determined by remote read-back
- Objective: admit P67 P59-only VMA scoping into the exact P45/M15 FrozenLake full profiles, then run both 300-update targets with strict Zero-TIM and backward-health gates unchanged.
- Definition of done: GSM8K DP16xTP4 plus P45/M15 DP8xTP8 complete their signed horizons with every strict Zero-TIM gate green and durable optimizer, timing, XProf, Perfetto, cache, evaluation, and checkpoint evidence.
- Task directory: `canon-zero-tim/tasks/v1-phase4-three-full-recipes`
- Directory state: isolated worktree `/mnt/disks/tunix-data/worktrees/v1_fl_tp8_ab_diag_0826`, branch `local/v1-autoscale-recovery-0826`, fetched base `3820b168` before the repair. The current CL contains the repair and ledgers; only its remote-read exact SHA is launchable, never a dirty tree or base SHA.
- Current phase: V1.P4.12 final-only checkpoint admission recovery.
- Last verified fact: Attempt 10 P45 (48,753 actions) and M15 (122,162 actions) both have strict A-B/B-C `0/0`. Each completed reverse group 1/32 through all 36 layers, then the first gradient sink rejected final-only interval 300 because `peft_trainer.py` duplicated a stale interval-10 whitelist. Local code now reuses the canonical checkpoint parser; checkpoint 15/15, Phase4 89/89, P57 146/146, and complete immutable-image V1 gates pass.
- Next action: after remote exact-SHA read-back, render fresh P45/M15 full identities and relaunch both when separately authorized.
- Blockers: no target run has exercised the repaired first sink, complete 32/32 reverse, first-update precommit, AdamW, weight sync, policy step 1, convergence, or final checkpoint.
- Key artifacts: `phases/v1-p4-11-frozenlake-full-checkpoint-contract.md`; `evidence/v1_hp_three_full_attempt10_20260826/`; `RUNBOOK.md`; the two-full renderer.
- Updated: 2026-08-26T20:31:08Z
