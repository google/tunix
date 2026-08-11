# State

- Status: active
- Objective: Localize and remove the 64-chip Pathways decode-versus-prefill bitwise carrier while
  keeping every committing training gate fail-closed.
- Definition of done: One source-pinned flag-on run reports `S_decode_vs_S_prefill=0`,
  `S_prefill_vs_T_old=0`, and `T_old_vs_T_current=0` before a full workload is allowed to commit.
- Task directory: `canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/`
- Directory state: tracked through `1a9310f3`; P38.2e/P38.2f implementation is
  in the current working tree and is not committed or published.
- Current phase: P38.2e schedule-aware GSM8K commit and P38.2f FrozenLake
  mismatch capture are active; P38.2a remains an unresolved localization track.
- Last verified fact: the complete P33 CPU gate is green (67 workload tests,
  26 alignment tests, all adjacent suites and negative controls). Focused
  commit tests prove that LR-zero commits advance Adam
  state without changing parameters, positive-LR commits report post-rounding
  changes, and a FrozenLake red boundary writes a bounded NPZ capsule whose
  transport and array hashes can be recovered from stdout. No target hardware
  run has exercised this new code. The unchanged overlay exact-image gate also
  remains green for Qwen3-1.7B and Qwen3-8B.
- Local limitation: the legacy GSM8K L3 runner has a pre-existing two-versus-
  eight trajectory contract mismatch, so it was not used to manufacture a
  hardware PASS. Real-model compile time and peak HBM for the new scalar commit
  evidence remain NOT RUN.
- Next action: obtain explicit commit/push approval, then let the external
  operator render a fresh source-pinned queue.
  Apply only FrozenLake backward-no-commit and GSM8K full. Recover the capsule
  from the FrozenLake raw log before any further target rerun.
- Blockers: commit/push requires explicit approval. The target capsule and
  schedule-aware update verdict require the external 64-chip operator; local
  controls cannot promote either target gate.
- Key artifacts: `../../debug_logs/p33_r35_gsm8k_full.raw.log`,
  `../../debug_logs/p33_r35_frozenlake_full.raw.log`, `plan.md`,
  `phases/p38-1-evidence-hardening.md`, `HANDOFF.md`
- Key local artifacts: `/mnt/disks/tunix-data/logp_probe_1host/p38_onehost_0810_r2.result.json`,
  `/mnt/disks/tunix-data/logp_probe_1host/p38_tail_0810_r1.result.json`,
  `/mnt/disks/tunix-data/logp_probe_1host/p38_aval_0810_r1.result.json`
- Updated: 2026-08-11 UTC
- Rollback: leave `CANON_P38_MISMATCH_CAPSULE` empty and revert the P38.2e/P38.2f
  working-tree changes. The loss, precision, prefix cache, and optimizer
  schedule are unchanged.
