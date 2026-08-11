# P41.4 — FrozenLake/Qwen3-8B resident capacity admission

## Question

Can the existing four-chip v5p host retain the Qwen3-8B AdamW state on TPU
through one real FrozenLake backward and optimizer commit without changing
precision, loss, gradient order, alignment policy, or the production default?

## Frozen workload

- topology: DP1xTP4;
- model: Qwen3-8B actor in fp32 and frozen reference in bf16;
- one P27 update: four prompts, two generations, eight trajectories;
- four gradient microbatches of two trajectories;
- prompt/response limits: 2048/64;
- evaluation and checkpointing disabled;
- optimizer placement: `CANON_OPT_STATE_RESIDENT=1` and
  `CANON_P30_OPT_STATE_OFFLOAD=0`;
- all canonical forward, alignment, VJP2, transaction, and replica gates stay
  enabled and fail closed.

## Pre-registered exit gate

PASS requires all of the following in the one-update report:

1. exactly one commit and train step `0 -> 1`;
2. four finite, nonzero gradient microbatches;
3. optimizer placement is `device-resident` before and after commit;
4. the optimizer transaction is valid and changes at least one parameter;
5. reference state and reset accumulator are unchanged;
6. DP replica equality is true;
7. optimizer timing and TPU HBM snapshots are present;
8. the strict alignment gate and canonical postflight pass;
9. the process does not OOM or time out.

An alignment failure is recorded as an alignment blocker, not as a capacity
failure. An OOM before the report is a resident-capacity failure. A missing or
partial report is inconclusive and must not be called PASS.

## Claim boundary

A pass admits only one local DP1xTP4 Qwen3-8B resident update. It does not
measure resident speedup, prove multi-update stability, admit DP16xTP4
Pathways, or change the offload production default. A failure does not affect
the signed GSM8K P41.2 result.

## Rollback

Leave `CANON_OPT_STATE_RESIDENT=0`. No optimizer arithmetic, precision, or
training default changes are part of this phase.

## Result — `p41fl1`

Status: **NOT ADMITTED**.

- Docker exit: 0; backward, one optimizer commit, engine weight sync, and
  canonical postflight completed without OOM.
- Alignment: four records, 491 total action tokens; every A/B/C boundary was
  byte-exact and every ratio/clip/TIS identity gate passed.
- Placement: `device-resident` before reverse and after commit; optimizer H2D
  and D2H were both zero.
- Update: aggregate gradient finite with 7,558,745,320 nonzero elements;
  6,934,505,968 parameter elements changed; reference and reset accumulator
  remained unchanged.
- Transaction: 56.99 seconds, of which Adam commit was 56.93 seconds.
- HBM: peak 97,955,232,768 bytes per chip against a 102,803,437,568-byte
  limit, leaving 4,848,204,800 bytes (4.52 GiB).
- Pre-registered gate failure: gradient activity was
  `[false, true, false, false]`, with norms
  `[0, 25.702417, 0, 0]`. The aggregate update is real, but the required four
  active microbatches were not observed. The threshold is not changed after
  seeing the run.

Artifacts:

- `/mnt/disks/tunix-data/logp_probe_1host/p41_frozenlake_p41fl1/raw.log`
  (`b6e6eb3c2d2c65da3f531fa08e4ddea6dbe95c96ca1cd34b5dc029f1b878b6d1`)
- `/mnt/disks/tunix-data/logp_probe_1host/p41_frozenlake_p41fl1/alignment.jsonl`
  (`15e23a481ca16a11f3fd3419740de1c3cc3bd733195c1d8830b00d7a9f1ba661`)
- `/mnt/disks/tunix-data/logp_probe_1host/p41_frozenlake_p41fl1/update.json`
  (`579ef37a66d14dfb5ad17d0a6205f637eec7bc324530884d941c5d7991a7d20d`)
- `/mnt/disks/tunix-data/logp_probe_1host/p41_frozenlake_p41fl1/resident.classification.json`
  (`7dea9871c8eb023a14ebe3be48ba855cb02ad70757d54d9c707a06e96fe4746e`)
