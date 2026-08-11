# P38.2g one-host synthetic causal replay

Date: 2026-08-11 UTC

## Scope

These runs used real Qwen3-8B weights on the authorized direct-attached
DP1xTP4 v5p host. The capsules were deterministic synthetic token/mask inputs,
not recovered P38.2f production capsules. The result admits the one-host
measurement path and rejects one depth-specific interpretation; it does not
identify or repair the production FrozenLake carrier.

Prefix caching was disabled. Runtime KV caching was enabled. Every arm used an
independent fresh cache, ran twice, and exited before backward with zero
optimizer commits.

## Deep and shallow controls

| Control | Prompt length | Classification | R0 vs R1 logps | R0 vs REF logps |
|---|---:|---|---|---|
| deep | 1788 | `LOCAL_CARRIER_NOT_ISOLATED` | 0/8 elements, max 0 | 8/8, max 0.7455787658691406 |
| shallow | 256 | `LOCAL_CARRIER_NOT_ISOLATED` | 0/8 elements, max 0 | 8/8, max 7.218132019042969 |

For the deep control, R0 versus REF raw targets differed in 7/8 elements with
maximum absolute difference 0.4375. For the shallow control they differed in
8/8 elements with maximum absolute difference 8.5234375. All R0/R1 raw target,
normalizer, and logprob comparisons were bitwise exact at both depths.

The shallow control is decisive: the synthetic R0/REF red is not specific to
the observed production onset around logical KV 1791. It measures a broader
incremental-cache-envelope versus fixed-chunk-reference boundary. Therefore it
cannot authorize R2/R3 or a KV-unified production candidate.

## Integrity gates

- Qwen3-8B actor and live engine: 399/399 mapped leaves, 8,190,735,360
  elements, bitwise equal, no mismatch indices.
- Mesh order: `[0, 2, 1, 3]`.
- R0, R1, and REF repeated bitwise exactly at every observed stage.
- One-bit negative control changed one element and was detected.
- Classifier verdict: `PASS`, scope `measurement-integrity-only`,
  `production_repair_admitted=false`.
- Deep run elapsed 264 seconds; shallow run elapsed 263 seconds.

## Evidence

Deep control:

- raw log: `/mnt/disks/tunix-data/logp_probe_1host/p38_fl_replay_synthetic_kv1792_0811d.raw.log`
- raw SHA-256: `1e578758eae97ce9b37901b4cb980850397e450daa088d255e47daa978bf099a`
- report: `/mnt/disks/tunix-data/logp_probe_1host/p38_fl_replay_synthetic_kv1792_0811d/replay.json`
- report SHA-256: `c611b9d69671e9bf9bbf2a9a52cb5bb1b25f9e9151f2ec0a9a275e4f605efd46`

Shallow control:

- raw log: `/mnt/disks/tunix-data/logp_probe_1host/p38_fl_replay_synthetic_kv256_0811a.raw.log`
- raw SHA-256: `a01b58e6dbeec87dda97946673dff9a3f09701a139f4b7aa4df97a6e17c143c9`
- report: `/mnt/disks/tunix-data/logp_probe_1host/p38_fl_replay_synthetic_kv256_0811a/replay.json`
- report SHA-256: `32ffb6f754878f66df7fca5e8b6af6cbe969e09033fb2a40817037d5c11765e9`

Three earlier immutable admission attempts failed before numerical execution:
missing alignment-gate environment, an invalid `dp` mesh-axis name for a
model `fsdp` PartitionSpec, and `num_generations=1`. The corrected runner uses
gate-only alignment, a size-one `fsdp` axis (no FSDP weight split), and the
minimum legal `num_generations=2`.

## Next action

Capture and verify the target P38.2f capsule on source-pinned Pathways. Run the
same one-host binary with that capsule. If the target capsule repeats the same
shallow-style R0/R1 versus REF split, add an exact serving-envelope control
before interpreting any KV-unified arm. R2/R3 remain gated.

## Rollback

Leave `CANON_P38_FROZENLAKE_REPLAY` unset. No production default, precision,
loss, prefix-cache policy, attention kernel, or optimizer behavior changed.
