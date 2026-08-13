# P46.4 — remote execution and evidence return

- Status: active; remote data/log reconciliation is complete, and publication
  still requires explicit approval

## Entry gate

1. Confirm the reconciled base and remote both equal
   `99c3f7af761c859caa6c81ab509446cc3cc47dc0`.
2. Commit and push only after explicit user approval; never target `main`.
3. Read back the exact 40-character SHA from
   `origin/yuxzhang/canon-zero-tim` and require a clean detached execution
   checkout at that SHA.
4. Run `bash canon-zero-tim/tests/p46_deepswe_profiles/run_cpu.sh` and the
   topology/model-specific exact-image gates.
5. Confirm `training_data_sharding_axis` is derived from
   `train_axis_names[0]`; reject a production hard-coded `fsdp` axis.
6. Render through `cluster/render_p46_deepswe_profiles.py`; do not edit the
   rendered YAML or insert credential values.

## Ordered target campaign

1. Prefer 64 chips and run logical shard 0, physical shard 0 of
   `q4-clean-eval`. Require full trajectory records, confirmed R2E pod cleanup,
   and `P46_EVAL_SUBSHARD_PASS`; this is not a full logical report.
2. Run `q4-debug` for exactly three updates. Require three durable trajectory
   batches, finite forward/backward evidence, three monotonic optimizer commits,
   device-resident optimizer state, the `dp` data-sharding marker and P44
   classifier PASS.
3. If the curriculum campaign is wanted, complete 58 logical reports through
   463 resumable physical JobSets. Never classify a task before exact N16.
4. Run `q32-train` with Qwen3-32B, 16K, B8/G8, 1000 updates, the original
   1851-row clean whitelist, and a 5400-second rollout-batch boundary.

## Hard stops and return package

Timeout, OOM, IFRT, topology, dataset fingerprint, weight, replica,
non-finite, optimizer transaction, cleanup leak, duplicate sample identity,
artifact digest or classifier failures remain hard errors. Finite alignment
residuals in full training remain visible warning-only evidence.

For training, absence of `[DEEPSWE.DATA_SHARDING] PASS`, a canonical data axis
other than `dp`, negative remaining timeout, or `KeyError: 'fsdp'` is a source
or manifest provenance failure. `observed_trajectories=N` alone never proves
the N records are valid; promotion reads full statuses and durable artifacts.

Return the rendered YAML and digest, exact source/image/data pins, complete
head/worker/R2E logs, JobSet descriptions/events, persistent run directory,
trajectory and report digests, classifier JSON, optimizer placement, per-device
HBM, and the first fatal traceback when applicable.

## Claim ceiling

A P46 local PASS proves no target behavior. A single evaluation physical shard
does not prove a logical N16 report. Q4 three-update does not prove Q32
trainability. A 64-chip result does not prove DP16 behavior, and a 256-chip
result does not imply bitwise or performance parity with 64 chips. No P46 run
promotes zero-TIM without separate evidence.
