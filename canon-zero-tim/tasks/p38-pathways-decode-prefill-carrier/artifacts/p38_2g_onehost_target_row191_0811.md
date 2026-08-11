# P38.2g one-host target row 191

Date: 2026-08-11 UTC

## Result

The hash-verified P38e1 source row 191 completed on real Qwen3-8B DP1xTP4 in
417 seconds. Measurement integrity passed, but the mask-derived local serving
schedule did not reproduce the captured production decode boundary.

- Actor and engine weights: exact across 399 leaves and 8,190,735,360 elements.
- Prefix cache: disabled.
- Backward: not executed.
- Optimizer commits: zero.
- R0, R1, and REF repeats: exact at raw target, processed target, normalizer,
  and logprob stages.
- One-bit negative control: exactly one element detected.
- Captured row: `S_decode_vs_S_prefill` has 3 of 517 action elements red;
  `S_prefill_vs_T_old` is exact.
- Local R0 versus R1: exact at every measured stage.
- Local R0/R1 versus REF: 395 of 517 logprobs red. The local R0/R1 raw target
  and normalizer are also red for every action.
- REF logprob SHA equals the captured `S_prefill`/`T_old` SHA exactly.
- Classification: `LOCAL_CARRIER_NOT_ISOLATED`; production repair not admitted.

This result validates the fixed-chunk/prefill side and rejects the
mask-derived R0 schedule as an exact proxy for production decode. The capsule
does not contain the original serving block tables, page allocation, or
per-call scheduler metadata. R2/R3 KV-unified arms remain gated and must not be
interpreted in this local envelope.

## Command

```bash
bash canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38_frozenlake_replay.sh \
  canon-zero-tim/debug_logs/p38_p38e1_frozenlake_mismatch_capsule.npz \
  p38e1_row191_stock_0811
```

## Artifacts

- Raw log: `/mnt/disks/tunix-data/logp_probe_1host/p38_fl_replay_p38e1_row191_stock_0811.raw.log`
  (`sha256=bc1057eaa0c3c11bc6506bd0a93a50c6a5695b62789fccf59068f8ab0fb01151`)
- Replay report: `/mnt/disks/tunix-data/logp_probe_1host/p38_fl_replay_p38e1_row191_stock_0811/replay.json`
  (`sha256=5fb7f3481e53555c66bb17c2d2faf2655a96070faf51812e7c69f279dc45be9e`)
- Classification: `/mnt/disks/tunix-data/logp_probe_1host/p38_fl_replay_p38e1_row191_stock_0811/replay.classification.json`
  (`sha256=5ab9c23a0f172df2cb10ab1aa51506a194e8c4ffd08e3bda8eaddcea56a5c2d6`)
- Schedule: `/mnt/disks/tunix-data/logp_probe_1host/p38_fl_replay_p38e1_row191_stock_0811/schedule.json`
  (`sha256=56b4437ebb4b9eb12ca46c5a2b2d3e191877a9ca3e3d0e3df822b823fa1068ff`)

## Non-numerical warning

vLLM emitted an ignored cleanup-time `AttributeError` after the complete report
was persisted. Docker exited zero and the classifier passed. This warning did
not occur in the measured forward path and is not a numerical verdict.

## Rollback

Leave `CANON_P38_FROZENLAKE_REPLAY` unset. The run was forward-only and did not
write model, optimizer, checkpoint, or W&B state.
