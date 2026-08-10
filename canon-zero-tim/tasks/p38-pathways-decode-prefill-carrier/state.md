# State

- Status: active
- Objective: Localize and remove the 64-chip Pathways decode-versus-prefill bitwise carrier while
  keeping every committing training gate fail-closed.
- Definition of done: One source-pinned flag-on run reports `S_decode_vs_S_prefill=0`,
  `S_prefill_vs_T_old=0`, and `T_old_vs_T_current=0` before a full workload is allowed to commit.
- Task directory: `canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/`
- Directory state: tracked; implementation commit `671250a5` is published on
  `yuxzhang/canon-zero-tim`.
- Current phase: P38.2a model-free aval and sharding discriminator is active.
- Last verified fact: the live model-free sampling/scoring discriminator ran
  on direct-attached DP1xTP4 and classified `MODEL_FREE_NOT_REPRODUCED`.
  M16/M256 sampling had distinct HLO digests, M256/M256 scoring had identical
  HLO digests, all five comparisons were exact, and the one-bit negative
  control was detected. This does not replace the target DP16 Pathways arm.
- Next action: review and publish the source-pinned P38.2a target renderer,
  then let the external operator run exactly one DP16xTP4 Attempt-0 model-free
  discriminator before the two production probes.
  P38.2b GSM8K and P38.2c FrozenLake both remain required because r35 exposed
  different amplitude signatures.
- Blockers: a target flag-on reproduction still requires the external 64-chip
  cluster operator. The one-host result cannot remove that requirement.
- Key artifacts: `../../debug_logs/p33_r35_gsm8k_full.raw.log`,
  `../../debug_logs/p33_r35_frozenlake_full.raw.log`, `plan.md`,
  `phases/p38-1-evidence-hardening.md`, `HANDOFF.md`
- Key local artifacts: `/mnt/disks/tunix-data/logp_probe_1host/p38_onehost_0810_r2.result.json`,
  `/mnt/disks/tunix-data/logp_probe_1host/p38_tail_0810_r1.result.json`,
  `/mnt/disks/tunix-data/logp_probe_1host/p38_aval_0810_r1.result.json`
- Updated: 2026-08-10 UTC
