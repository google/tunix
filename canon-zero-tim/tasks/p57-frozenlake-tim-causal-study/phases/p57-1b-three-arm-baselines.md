# P57.1b — Two-workload, three-arm concept study

- Status: active

## Finding

- Confirmed: `p57cal6` selected M15 at 24.625% initial solve with 56% mixed
  groups, context max 7,403, completion max 6,223, and no cap hit.
- Confirmed: the historical P45 workload is the original deterministic
  parameter generator (seed 42/123, grid side 2–9, p 0.60–0.85), five turns,
  prompt/response 4,096/2,048, and a historical 450-update recipe. This study
  intentionally truncates every P45 treatment to the preregistered common
  200-update horizon. P57 `l0` is only a rematerialized envelope anchor and is
  not a byte-identical substitute.
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
temperature 0.7, AdamW 1e-6, GSPO-token/RLOO, and no in-process evaluation.

| Workload | Dataset | Turns | Prompt/response | Horizon |
|---|---|---:|---:|---:|
| P45 | original seed-42/123 parameter generator | 5 | 4,096 / 2,048 | 200 |
| M15 | materialized `m15/main` split | 15 | 4,096 / 8,192 | 200 |

Native and zero executables intentionally have different canonical-kernel M
contracts; within a workload, caller-global rows, semantic rows, scheduler
capacity, data order, and optimizer recipe remain equal. The study estimates
the whole zero-TIM bundle effect and cannot attribute a result to one kernel.

## Execution

1. Primary no-IS pair: render and, after explicit approval, launch P45 and M15
   under both `mismatch` and `zero`. These are four independent 64-chip jobs.
   Native and zero may run concurrently when four disjoint slices are available;
   otherwise preserve the same source and campaign contract across the queue.
2. IS add-on: after the primary pair evidence is packaged, launch P45 `is` and
   M15 `is`.
3. Run one isolated deterministic final-checkpoint evaluation for every valid
   cell. Never compare P45 accuracy directly with M15 accuracy as an arm effect.
4. Compute within-workload contrasts: IS benefit (`is - mismatch`), zero-TIM
   benefit (`zero - mismatch`), and zero-TIM versus mitigation (`zero - is`).

Each wave uses the same immutable source/image/model and fresh arm-specific
checkpoint tags. A healthy run is not intentionally paused; checkpoints every
10 updates with LatestN(1) are recovery only. No YAML hand edits are admitted.

## Exit gate

- Command: `bash canon-zero-tim/tests/p57_frozenlake_tim/run_cpu.sh` and
  `bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh`.
- Pass before launch: all six rendered cells pass resolved-env preflight; each
  command has exactly one expected sampler mode; native cells attest the entire
  zero-TIM bundle off; zero cells attest the registered canonical bundle; the
  opposite sampler mode and wrong workload horizon are rejected.
- Target pass for the primary pair: all four full horizons complete with exactly
  one no-IS purity receipt; native has a finite A-B dose and valid B-C, while
  zero has strict A=B=C; every job has checkpoints and complete classifier
  artifacts.
- Fail: any missing receipt, nonfinite/B-C/transaction/checkpoint failure,
  restart without an explicit resume decision, or treatment leakage is
  `INCONCLUSIVE`; preserve the run and stop.

## Claim ceiling

The first six one-seed curves are a concept study. A positive result supports
that zero-TIM or token TIS changes learning under these exact recipes; a null
supports only robustness to the measured dose. A campaign-level stability
claim requires preregistered paired seeds and counterbalanced launch order.
