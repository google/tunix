# P58.23 — Qwen3-4B optimized B2xG2 one-host backward

Status: completed locally; publication and TP8 promotion are separately gated.

## Decision

P58.22 proved real Qwen3-4B-Instruct-2507 DP1xTP4 rollout and strict A=B=C
with `CANON_CONTINUE_DECODE=8`, but its serial 4,672/5,120-token backward
carriers spent hours compiling without a terminal backward receipt.  P58.23
does not repeat that serial reference.  It exercises the current optimized
training path at one fixed K=2,560 using the same immutable strict-exact real
prompt repeated as two physical prompt groups, with two generations per group.

The global geometry is **B2xG2**, never batch size one:

- prompt batch `B=2`, generations `G=2`, four trajectories total;
- each prompt group has rewards `[1, 0]`, so all four RLOO advantages are
  finite and nonzero and each group sums to zero;
- prompt/response maxima are `2,048/512`, total K=`2,560`;
- global `batch_size=2` and `mini_batch_size=2`; device-memory microbatches
  remain `1` and do not change the global batch semantics;
- Qwen3-4B-Instruct-2507, seed 42, TP4, BF16, prefix cache off, TPU-resident
  optimizer, backward-no-commit, and strict A=B=C remain fixed.

The optimized carrier is P28 segmented forward/train plus G6 update, P29
full-train, P30 sparse/reuse/release/reshard, and the certified P71 forward
scan.  P59 remains off because a DP1 host cannot execute rank-parallel
backward; P59 is a separate DP8 target claim.  No serial reference arm is
launched.

## Immutable replay source

The preparation script is:

`tasks/p58-deepswe-native-zero-comparison/scripts/prepare_q4_b2g2_replay.py`

It validates the original run receipts before emitting a deterministic gzip
journal at:

`/mnt/disks/tunix-data/deepswe-replay-sources/p58-q4-b2g2-k2560-v2`

Required SHA-256 values are:

- combined `run_manifest.json`:
  `482d7934a95207d0d77bb4857fbb200d7b367cbf437dda6585937b20909afa8f`;
- combined `batch-000000.trajectories.jsonl.gz`:
  `091a9273c2067876fbee1996ee853e3c8e861352e307cd5fb94fea2563aec456`.

Both physical groups repeat the same real R2E Scrapy pair whose source run
already proved strict A=B=C.  Every row is truncated only at a complete
assistant-action boundary below 512 tokens; no synthetic trajectory or
injected advantage is admitted.  This exercises global B=2/RLOO/backward
shape and math but deliberately does not claim prompt diversity.  The earlier
v1 source remains preserved as failure evidence: its second Coverage group
was alignment-red in its own historical run and made only rows 2/3 fail in
`p58s23optb2g2c`; it is forbidden for acceptance.

## Gates and execution

Before TPU execution, require P34 static, the deterministic 408/408 flag
audit, all 37 exact-image Qwen3-4B TP4 files, the P58 exact-image regression
gate, and the four-row replay loader contract.  The cold one-host process is
bounded to 1,800 seconds and uses compilation-cache namespace:

`/mnt/disks/tunix-data/jax-compilation-cache/p58-q4-tp4-systemopt-b2g2-k2560`

Run only through:

```bash
P58_ONEHOST_ALLOW_DIRTY=1 \
  bash canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/run_onehost_deepswe_zero_trajectory_replay_docker.sh \
  <fresh-label>
```

A PASS requires four attested replay rows, two mixed groups, strict A=B=C,
four finite nonzero advantages, finite nonzero optimized backward, unchanged
parameters and optimizer state, zero commits, the exact system-optimization
tuple, and a verified return bundle.  A timeout or missing terminal receipt is
`ZERO_TIM_BACKWARD_INCOMPLETE`, not numerical failure and not PASS.

## Claim ceiling

This phase joins already-proved real rollout/alignment evidence to a bounded
real-trajectory trainer replay.  The replay itself does not execute R2E or
decode new tokens.  DP1xTP4 cannot certify P59, TP8, Pathways, 128-chip
disaggregation, a full optimizer update, or production readiness.  No commit,
push, image publication, Kubernetes mutation, or remote launch is authorized
by this phase record.

## Accepted target evidence

The final direct-v5p target is `p58s23optb2g2g_20260830t0132z`, rooted at:

`/mnt/disks/tunix-data/deepswe-onehost-xprof/p58_zero-hp_p58s23optb2g2g_20260830t0132z`

It returned `ZERO_TIM_RECORDED_TRAJECTORY_BACKWARD_NO_COMMIT_PASS` with:

- global B2xG2, four successful trajectory rows, and `N_action=1254`;
- strict byte-exact A=B=C before and after backward;
- two trajectory microsteps and exact repeated gradient norms
  `[8.544539451599121, 8.544539451599121]`;
- finite nonzero gradients, optimizer commits zero, train step `0 -> 0`, and
  no changed model/reference/optimizer/accumulator paths;
- device-resident optimizer memory and peak HBM 56,370,843,648 bytes;
- warmup segmented backward 122.657 seconds and compiled profiled repeat
  12.418 seconds;
- return-bundle SHA-256
  `7d33ee791146d2309c16866d8e30f15f0f012e05e88f6c795b587938f973f795`.

The earlier `p58s23optb2g2f_20260830t0121z` executed the same core backward
successfully but its classifier consumed a microstep-local 627-action receipt.
The final fix emits and requires an aggregate post-backward B2xG2 receipt; no
training math changed between that run and the accepted target.  Global batch
size one is forbidden throughout.
