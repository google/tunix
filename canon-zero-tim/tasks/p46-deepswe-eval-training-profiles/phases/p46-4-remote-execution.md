# P46.4 — remote execution and evidence return

- Status: active; implementation published, target campaign pending

## Entry gate

1. Read back the exact 40-character SHA from
   `origin/yuxzhang/canon-zero-tim` and require a clean detached execution
   checkout at that SHA.
2. Require implementation commit
   `e1b4009394c49ea015919bda0cfdb97c12c221b5` in that SHA's ancestry.
3. Run `bash canon-zero-tim/tests/p46_deepswe_profiles/run_cpu.sh` and the
   topology/model-specific exact-image gates.
4. Confirm `training_data_sharding_axis` is derived from
   `train_axis_names[0]`; reject a production hard-coded `fsdp` axis.
5. Render through `cluster/render_p46_deepswe_profiles.py`; do not edit the
   rendered YAML or insert credential values.

## Ordered target campaign

1. On whichever allocation is available, run logical shard 0, physical shard 0
   of `q4-clean-eval`: DP8 x TP8 on 64 chips or DP32 x TP8 on 256 chips.
   Require full trajectory records, confirmed R2E pod cleanup, and
   `P46_EVAL_SUBSHARD_PASS`; this is not a full logical report.
2. Run `q4-debug` for exactly three updates on the available registered
   topology: DP4 x TP8 per role on 64 chips or DP16 x TP8 per role on 256
   chips. Require three durable trajectory batches, finite forward/backward
   evidence, three monotonic optimizer commits, device-resident optimizer
   state, the `dp` data-sharding marker and P44 classifier PASS.
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
