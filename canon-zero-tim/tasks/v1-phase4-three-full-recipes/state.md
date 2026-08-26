# State

- Status: active; P4.11 source is host/full-image green; publication identity is the exact remote-read SHA containing this file; full targets are not launched
- Objective: admit P67 P59-only VMA scoping into the exact P45/M15 FrozenLake full profiles, then run both 300-update targets with strict Zero-TIM and backward-health gates unchanged.
- Definition of done: GSM8K DP16xTP4 plus P45/M15 DP8xTP8 complete their signed horizons with every strict Zero-TIM gate green and durable optimizer, timing, XProf, Perfetto, cache, evaluation, and checkpoint evidence.
- Task directory: `canon-zero-tim/tasks/v1-phase4-three-full-recipes`
- Directory state: isolated worktree `/mnt/disks/tunix-data/worktrees/v1_fl_tp8_ab_diag_0826`, branch `local/v1-autoscale-recovery-0826`, clean base `e5c596a4e7621e7442606cfc4dbbb39005eba4eb` before P4.11 edits. The base is provenance only and must never be rendered; no P4.11 manifest render or launch has occurred.
- Current phase: V1.P4.11 production admission and two-full-recipe handoff.
- Last verified fact: Wave 5 P45 p66-off and serving-scope both recorded strict A-B/B-C `0/0`, adequate deep-prefix coverage, controlled exit 42, and zero backward/optimizer commits. The serving-scope arm retained checked VMA with `CANON_P67_P66_VMA_P59_ONLY=1`; its 48,594-action target result is the accepted production candidate. M15 with P67 and both workloads' full backward/optimizer paths remain target-unrun.
- Next action: after approved publication and exact remote read-back, render two fresh immutable manifests and let the user launch P45 and M15 together.
- Blockers: full targets remain unrun; dirty-source YAML and the pre-edit base SHA are never launchable.
- Key artifacts: `phases/v1-p4-11-p67-frozenlake-full-promotion.md`; Wave 5 evidence directory; `RUNBOOK.md`; upcoming two-full renderer.
- Updated: 2026-08-26T08:58:18Z
