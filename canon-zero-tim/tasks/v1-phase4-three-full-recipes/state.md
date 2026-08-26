# State

- Status: active; P66 serving-scope repair published, P45/M15 DP8xTP8 revalidation pending
- Objective: publish and render one simultaneous three-recipe full wave using the P66 checked-VMA backward repair, with an independent fail-closed first-update gate on every recipe.
- Definition of done: GSM8K DP16xTP4 plus P45/M15 DP8xTP8 complete their signed horizons with every strict Zero-TIM gate green and durable optimizer, timing, XProf, Perfetto, cache, evaluation, and checkpoint evidence.
- Task directory: `canon-zero-tim/tasks/v1-phase4-three-full-recipes`
- Directory state: clean isolated worktree at `/home/yuxuan/code_rl_repro/worktrees/p66_gsm8k_convergence_0825`, branch `local/p66-gsm8k-onehost-convergence`. Runtime CL `41f50d23` plus carrier/ledger CL `299fca0c` were pushed by ordinary fast-forward and read back exactly at `299fca0c9f2fff9679076c6d938185696c0f3d2d`. No repaired DP8xTP8 or full-recipe result exists yet.
- Current phase: evaluate the user-launched P45 and M15 full trains independently at the strict pre-backward and optimizer gates.
- Last verified fact: published source `c2833eea` makes P45/TP8 fail at step 0 with A-B/B-C `396/0` bytes and M15/TP8 fail with `20/0` across 123381 action tokens. Both are APC-off and stop before backward. The local fix scopes the P66 completed-sum `pmean` to the live P59 manual data/model context. V1 74/74, P59 37/37, P66 16/16 and the pinned-image installed TP4/TP8 ring/gather negative pass. Real one-host TP4 ring and gather carriers each pass three strict rounds at `0/0`; the failed real TP8 executable is not yet reverified.
- Next action: wait for the user's fresh P45/M15 run IDs, pull their logs, and require strict A-B/B-C `0/0` per recipe before assessing backward or optimizer evidence.
- Blockers: the Attempt 8 target hard-red has not yet been reverified on repaired runtime bytes. DP16xTP4/DP8xTP8 optimizer correctness, convergence, performance, and GCS XProf transport remain target-unverified.
- Key artifacts: `phases/v1-p4-9-checked-vma-full-wave.md`; `scripts/prepare_checked_vma_three_full_wave.sh`; `RUNBOOK.md`; `evidence/v1_hp_p64_remote_64tpu_20260825/`; `../p66-onehost-gsm8k-convergence/evidence/p66-g15-onehost-20260826/receipt.json`.
- Updated: 2026-08-26T02:31:00Z
