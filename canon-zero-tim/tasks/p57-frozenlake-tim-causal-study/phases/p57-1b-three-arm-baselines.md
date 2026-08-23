# P57.1b — Two-workload, three-arm concept study

- Status: active

## Finding

- Confirmed: `p57cal6` selected M15 at 24.625% initial solve with 56% mixed
  groups, context max 7,403, completion max 6,223, and no cap hit.
- Confirmed: the historical P45 workload is the original deterministic
  parameter generator (seed 42/123, grid side 2–9, p 0.60–0.85), five turns,
  prompt/response 4,096/2,048, and a historical 450-update recipe. The user
  superseded the earlier 450-update/no-eval plan before the replacement launch.
  All paired treatments now use 300 updates with seven in-process held-out
  evaluations. P57
  `l0` is only a rematerialized envelope anchor and is not a byte-identical
  substitute.
- Confirmed: the first four-job launch is `INCONCLUSIVE` before step 0. The
  manifests and outer preflights were correct, but the Python runtime validator
  still hardcoded the older `(mismatch,m15,selection)` discovery cell. The
  repair must admit only the five preregistered stock tuples and reject every
  unregistered arm/workload/split combination.
- Confirmed: the matrix repair moved native jobs into committed training, but
  exposed a second stale inner guard in post-backward alignment. Stock
  `mismatch` and stock `is` both intentionally run without canonical Engine
  Module C; only the former was exempted. `i45a` is `INCONCLUSIVE`, while its
  successful pre-backward and backward evidence narrows the repair to this
  post-backward attestation.
- Confirmed: the first four 450-update jobs (`n45c/n15c/i45c/i15c`) all
  completed Step-0 rollout and then failed before trainer alignment because
  policy seeding made the collector replace the FrozenLake trajectory task
  (which owns the rendered prompt) with an environment task containing only
  durable metadata. The compatibility repair keeps a prompt-bearing DeepSWE
  environment task exact, merges durable environment metadata into a
  prompt-bearing FrozenLake trajectory task, and preserves the missing-prompt
  fail-closed control. Both pinned-image suites pass; target repair validation
  remains pending.
- Decision: test three treatments independently on both workloads:

| Runtime arm | Numerical program | Sampler correction | Old denominator | TIS weights |
|---|---|---|---|---|
| `mismatch` | native/stock-fast | none | rollout A | absent |
| `is` | same native/stock-fast | token TIS | trainer C | `min(exp(C-A), 2)` |
| `zero` | complete zero-TIM bundle | none | rollout A | absent |

Standard GSPO current/old ratio clipping at epsilon 0.003/0.005 is shared base
algorithm behavior and remains enabled in every cell.

## Shape and workload ledger

Both workloads use Qwen3-8B, DP8xTP8, global 32 prompts x eight generations =
256 rollout rows, 32 local trajectories per DP rank, resident optimizer state,
temperature 0.7, AdamW 1e-6, GSPO-token/RLOO, and rollout-only in-process
evaluation every 50 updates plus the final step 300. Every paired command pins
experiment/data-shuffle seed 42 and rollout-engine global seed 0. Dataset rows
and canonical SHA-256 identities are checked before rollout; the exact hashes
are registered in `RUNBOOK.md`.

| Workload | Dataset | Turns | Prompt/response | Horizon |
|---|---|---:|---:|---:|
| P45 | original seed-42/123 parameter generator | 5 | 4,096 / 2,048 | 300 |
| M15 | materialized `m15/main` split | 15 | 4,096 / 8,192 | 300 |

Native and zero executables intentionally have different canonical-kernel M
contracts; within a workload, caller-global rows, semantic rows, scheduler
capacity, data order, and optimizer recipe remain equal. The study estimates
the whole zero-TIM bundle effect and cannot attribute a result to one kernel.

## Execution

1. First queue: render four fresh 300-update jobs and, after explicit launch
   approval, launch P45 and M15 under both `mismatch` and `is`. These are four
   independent 64-chip jobs using the same native/stock-fast numerical program;
   their controlled treatment is token importance sampling and the
   corresponding old-logprob identity. The partial 200-update identities remain
   immutable evidence and are not resumed into this superseding campaign.
2. Deferred Zero-TIM pair: do not include P45 `zero` or M15 `zero` in the first
   queue. After the first four runs are packaged, require a separate user
   decision before launching either complete Zero-TIM/no-IS cell.
3. Each JobSet runs held-out rollout-only evaluation at
   `0,50,100,150,200,250,300`. Every point uses 100 prompts x eight generations
   at temperature 0.7 over the same signed eval rows. Step 0 is the initial
   rollout policy; at step 50 the rollout engine holds the weights after 50
   updates and update-51 weights are not yet synced. The final point runs after
   update 300 and its final weight sync. Never compare P45 accuracy directly
   with M15 accuracy as an arm effect.
4. Compute within-workload contrasts: IS benefit (`is - mismatch`), zero-TIM
   benefit (`zero - mismatch`), and zero-TIM versus mitigation (`zero - is`).

Each wave uses the same immutable source/image/model and fresh arm-specific
checkpoint tags. A healthy run is not intentionally paused. The primary
300-update recipes save only at the final update and `LatestN(1)` retains that
single actor+optimizer checkpoint. Partial stops are rejected because they
would have no durable recovery point. No YAML hand edits are admitted.

## Exit gate

- Command: `bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh` and
  `bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh`.
- Pass before launch: all six rendered cells pass resolved-env preflight; each
  command has exactly one expected sampler mode and exactly one `--seed=42`;
  a seed-43 mutation is rejected; native cells attest the entire
  zero-TIM bundle off; zero cells attest the registered canonical bundle; the
  opposite sampler mode and wrong workload horizon are rejected. The pinned
  image must emit
  `P57_STOCK_RUNTIME_MATRIX_PASS variants=5 stages=train,eval`,
  `P57_STOCK_POST_BACKWARD_MODULE_C_PASS arms=mismatch,is`, and the
  unknown-arm negative marker.
- Target pass for the current four-job queue: all four **fresh** 300-update
  horizons and seven-point evaluation classifiers complete;
  the two `mismatch` jobs have exactly one no-IS purity receipt and the two `is`
  jobs have exactly one token-IS purity receipt. All four attest the stock-fast
  zero-TIM-off path, a finite A-B dose, valid B-C, checkpoints, and complete
  classifier artifacts.
- Target data/seed pass: exactly one seed receipt reports data shuffle 42 and
  vLLM global seed 0; exactly one dataset receipt reports the registered P45 or
  M15 train/eval hashes. Per-request stochastic seed remains unsupported, so
  cross-launch token streams are not required to be byte-identical.
- Fail: any missing receipt, nonfinite/B-C/transaction/checkpoint failure,
  restart without an explicit resume decision, or treatment leakage is
  `INCONCLUSIVE`; preserve the run and stop.

## Claim ceiling

The first six one-seed curves are a concept study. A positive result supports
that zero-TIM or token TIS changes learning under these exact recipes; a null
supports only robustness to the measured dose. A campaign-level stability
claim requires preregistered paired seeds and counterbalanced launch order.
