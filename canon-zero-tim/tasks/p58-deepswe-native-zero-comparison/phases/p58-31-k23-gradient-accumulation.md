# P58.31 — K23 gradient-accumulation geometry

Status: `LOCAL CONSTRUCTION PASS / DIRECT BACKWARD REPLAY PASS / TARGET NOT RUN`

## K23 evidence boundary

K23 crossed the P58.30 grouped-trainer axis boundary on the real 128-device
disaggregated target. The complete immutable log records 128 trajectories,
eight reward-one trajectories, 47 final nonzero advantages, three effective
prompt groups, and strict A=B=C over 396,233 action tokens. Six
`MODEL_TIMEOUT` rows were compact-filtered. The incident report's 393,135
token count is stale; preserve that immutable prose, but use the complete
`run.log` value when describing K23.

The backward path reached all 36 layers, emitted
`[P59.DP8] gradient_reducer_ready dp_axis=dp dp_size=8`, and completed group
1/16 with eight rank contributions, exact replicas, and a finite nonzero
gradient. It then stopped before writing an accumulator:

```text
ValueError: segmented update accumulation changed: 8 != 16
```

This is not a rollout, alignment, pullback, or gradient-numerics failure. No
optimizer transaction or checkpoint occurred. Preserve the immutable package
at `canon-zero-tim/evidence/p58_k23_gradient_accumulation_mismatch_incident/`.

## Root cause

P58 has 128 global trajectories and DP8. Each rank-major streamed gradient
group contains one trajectory from each DP rank, so the trajectory microbatch
width is 8 and the update contains 16 groups:

```text
128 global trajectories / 8 trajectories per group = 16 accumulation groups
```

The launcher incorrectly set `train_trajectory_micro_batch_size` to the 16
local trajectories. `RLTrainingConfig` therefore derived eight accumulation
steps while the segmented trainer had registered 16 streamed groups. The
precomputed-gradient safety check correctly rejected the mismatch after the
first expensive pullback.

## Repair

`DeepSWEWorkload` now names the two independent quantities explicitly:

- `train_trajectory_micro_batch_size = dp_size`;
- `gradient_groups = local_trajectories`.

The P58 launcher uses those meanings and requires
`gradient_accumulation_steps == gradient_groups` before cluster construction.
The learner uses the same width for its P34 segmented geometry and validates
the registered group count before entering backward. A bad future recipe now
fails in startup rather than after a 36-layer pullback.

Required pre-backward receipts are:

```text
[DEEPSWE.ACCUMULATION] PASS global_trajectories=128 trajectory_micro_batch=8 gradient_groups=16 gradient_accumulation_steps=16
[CANON_P34_DP8] accumulator_contract_ready trajectories=128 micro=8 groups=16 gradient_accumulation_steps=16
```

No flag, model, clean-data selector, sampler, loss, precision, optimizer,
topology, timeout, TiTO, compact-filter, or Zero-HP setting changed.

## Validation and promotion

Host P34 static passes all ten suites. Contract/script regressions cover every
registered DeepSWE geometry and the exact P58 128/8/16 boundary. The complete
digest-pinned P58 image gate passes, including the real `RLTrainingConfig`,
`PeftTrainer` precomputed-gradient transaction, first-update, and agentic
geometry boundaries.

Development run `p58k23accumdev_20260901T004406Z` executed the bounded
direct-v5p P58.23 replay on this dirty source diff. It used Qwen3-4B
DP1xTP4, B2xG2, and four immutable mixed-reward trajectories. The official
classifier returned `PASS`: A=B=C over 1,254 action tokens, both gradient
norms `8.544539451599121`, finite/nonzero/repeat-exact backward, device
optimizer state, no changed model/optimizer/accumulator/reference paths, and
zero commits. Warmup segmented value-and-grad took 123.528 seconds; the
profiled cached repeat took 12.565 seconds. Peak HBM was 56,370,843,648 bytes
per the returned backward report.

The return bundle and checksum are under
`/mnt/disks/tunix-data/deepswe-onehost-xprof/p58_zero-hp_p58k23accumdev_20260901T004406Z/`.
Because the run intentionally admitted a dirty development diff, this is
direct-TPU development evidence, not clean-source signed acceptance. It does
not certify DP8/TP8 or the target accumulator stream.

Source commit/push is explicitly approved for this delivery. After final
clean remote readback, matching-image publication and target launch still
require separate approval. The next full target must emit both startup
receipts above, complete all 16 `reverse_group_done` transactions, preserve
finite nonzero/repeat-safe gradients and exact replicas, commit exactly one
first optimizer transaction, and write the durable checkpoint before
promotion. Until those target receipts exist, say `LOCAL CONSTRUCTION PASS`
or `INCONCLUSIVE`, never “DeepSWE training PASS.”
