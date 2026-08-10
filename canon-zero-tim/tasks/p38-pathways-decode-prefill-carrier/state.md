# State

- Status: active
- Objective: Localize and remove the 64-chip Pathways decode-versus-prefill bitwise carrier while
  keeping every committing training gate fail-closed.
- Definition of done: One source-pinned flag-on run reports `S_decode_vs_S_prefill=0`,
  `S_prefill_vs_T_old=0`, and `T_old_vs_T_current=0` before a full workload is allowed to commit.
- Task directory: `canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/`
- Directory state: untracked in the current worktree; no commit or push has been authorized.
- Current phase: P38.2 flag-on production-boundary reproduction (pending publication and target
  operator); P38.1 is complete.
- Last verified fact: the pinned-image alignment suite passed 18/18 and the full P33 CPU gate
  passed, including the injected failed-workload stdout artifact control. Committing-training
  gate semantics were not relaxed.
- Next action: Review this evidence-only patch, then obtain explicit approval to commit/push. The
  target operator can subsequently run the GSM8K `alignment-short` manifest first by following
  `HANDOFF.md`; FrozenLake is the fallback if the sparse GSM8K carrier is not sampled.
- Blockers: A target flag-on reproduction requires an approved published source commit and the
  external 64-chip cluster operator. No target run is authorized by this checkpoint.
- Key artifacts: `../../debug_logs/p33_r35_gsm8k_full.raw.log`,
  `../../debug_logs/p33_r35_frozenlake_full.raw.log`, `plan.md`,
  `phases/p38-1-evidence-hardening.md`, `HANDOFF.md`
- Updated: 2026-08-10 UTC
