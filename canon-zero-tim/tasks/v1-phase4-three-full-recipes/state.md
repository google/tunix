# State

- Status: active; f45w09 completed a healthy strict Step-0 train/update and then failed only in held-out eval because the standard wrapper had not actually selected eval-off; no-eval/no-checkpoint P45+M15 fast-run runtime is host- and exact-image-green and published as `a8449b3ddc2187806341b280f9d659028b3936c6`; target not run
- Objective: run optimized Zero P45 and M15/main for 300 updates with eval and checkpoint I/O removed, while preserving strict Zero-TIM, backward-health, optimizer, timing, W&B, cache, XProf, and Perfetto gates.
- Definition of done: both P45/M15 DP8xTP8 fast concept runs complete 300 updates with zero strict alignment FAIL, healthy optimizer receipts, complete timing/profile artifacts, and explicit evaluation/checkpoint-disabled receipts. Evaluation curves, resume points, and final checkpoints are intentionally out of scope.
- Task directory: `canon-zero-tim/tasks/v1-phase4-three-full-recipes`
- Directory state: writable worktree `/home/yuxuan/code_rl_repro/worktrees/p57_zero_noeval_0828`, branch `local/p57-zero-noeval-0828`; runtime commit `a8449b3ddc2187806341b280f9d659028b3936c6` was pushed by ordinary fast-forward and exactly read back from the publication branch.
- Current phase: fast concept-run admission for no-eval/no-checkpoint optimized Zero P45+M15.
- Last verified fact: f45w09 source `19d10537` passed strict Step-0 train/backward/AdamW and failed after commit in eval rescore; local implementation scopes eval-off and checkpoint-disabled to both optimized Zero workloads. P57 155/155, Phase4 90/90, P45-owned 32/32, flags 393/393, and the complete immutable-image gate pass with terminal `V1_HP_EXACT_IMAGE_PASS ... manifests=3`; the image transcript is not a durable raw artifact and no TPU result exists for this repair.
- Next action: the other operator checks out exact runtime source `a8449b3ddc2187806341b280f9d659028b3936c6`, renders two fresh immutable manifests, and launches P45+M15 together under the user-owned run lane.
- Blockers: publication and target validation are pending. Checkpoint-free runs cannot resume and produce no held-out/final-checkpoint evidence by design.
- Key artifacts: incident log `evidence/incident_20260828_failures/f45w09_head.log`; `HANDOFF.md`; `RUNBOOK.md`; the two-full renderer.
- Updated: 2026-08-28T01:10:29Z
