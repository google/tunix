# Plan

## Outcome

Compare two otherwise identical synchronous disaggregated DeepSWE-derived
Qwen3-4B-Instruct training arms on the promoted P46 clean task set:

- `native`: preserve the untreated serving and trainer numerical programs from
  the pinned `yuxzhang/deepswe-quality-fix` reference and measure their finite
  inference-trainer mismatch as the treatment dose;
- `zero`: enable the complete canonical numerical bundle and require exact
  A=B=C at every admitted boundary.

This is a causal comparison of two complete numerical systems. It is not an
ablation of one kernel, and it is not an exact reproduction of the published
DeepSWE-32B run: the model, generation count, clean-data selector, and context
length differ deliberately. The compact trajectory-filtering rule does match
the published DeepSWE recipe and the pinned Tunix quality-fix reference.

The execution order changed again by user decision on 2026-08-21: waive the
optional one-host gate and the separate three-update stop, then run the native
full 1,000-update campaign directly. Commits 1–3 remain mandatory online
monitoring boundaries inside the same job. The zero implementation remains
available for review and regression testing but is not launch-authorized until
its optimization work is complete and the user explicitly reactivates it. A
native-only result is an integration/training result, not a causal comparison.

## Phases

| Phase | Deliverable | Exit gate | Status |
|---|---|---|---|
| P58.1 | Frozen shared recipe and loss-aggregation contract | Fixed-16K formula oracle, compact-filter mask policy, effective-row-weighted accumulation, and DP8 invariance are specified and locally tested; public-source discrepancy is recorded | completed |
| P58.2 | Default-off paired profiles, renderer, metrics, trajectory journal, and negative controls | Host tests and pinned-image tests prove that the two rendered arms differ only in the registered numerical treatment bundle | completed |
| P58.3 | One-host observer and artifact sanity | Full redacted trajectory schema, W&B metric schema, logprob/alignment observers, checkpoint transactions, and no-update neutrality pass without a production claim | waived — not PASS |
| P58.4N | Native 128-chip three-update canary | Native completes exactly three optimizer commits on rollout DP8 x TP8 plus trainer DP8 x TP8, records finite nonzero A-B, keeps B-C exact, and emits a signed classifier PASS | superseded — p58c05 failed before execution; not PASS |
| P58.4Z | Zero 128-chip three-update canary | Zero completes exactly three commits on the identical recipe with strict A=B=C and a signed classifier PASS | deferred — do not launch |
| P58.5N | Native 128-chip full campaign | Native completes exactly 1,000 optimizer commits; the first three are monitored without stopping; durable trajectory, evaluation, checkpoint, alignment, optimizer, and classifier evidence passes | active |
| P58.5Z | Zero full or paired comparison | Activated only after zero optimization and a new explicit user decision | deferred — do not launch |

Exactly one phase may be active. Commit, push, image publication, Kubernetes
render/application, and TPU execution each remain separately user-gated.

P58.1 and P58.2 are closed by the pinned-image marker recorded in `state.md`.
P58.3 has CPU coverage for journal continuity and observer/classifier logic but
no real Qwen/R2E one-host evidence; the user explicitly waived it rather than
calling it PASS. P58.4N was superseded after p58c05 failed Kueue admission
before any workload pod or update existed. P58.5N is the only active phase.
Both zero phases remain deferred even though the renderer and CPU tests cover
that arm.

P58.5N attempts `p58f01` through `p58f03` remain `INCONCLUSIVE`. P58f01 exposed
missing sandbox LocalQueue inheritance and reset-time policy provenance;
p58f02 showed that the chosen CPU flavor required `cpu-np`; and p58f03 proved
that the CPU routing repair works by completing 128 real trajectories in
616.3 seconds. P58f03 stopped after durable journaling but before trainer
forward because the native arm was incorrectly sent through a
canonical-adapter-only weight-attestation method. The published repair provides a
shared exact-live-weight observer while preserving native numerical
untreatedness. The phase remains active; after final branch readback the next
fresh attempt is `p58f04`.

## Frozen shared recipe

| Field | Shared value |
|---|---|
| Model | `Qwen/Qwen3-4B-Instruct-2507` |
| Data | promoted 1,012-task P46 exact-N16 list, SHA-256 `ec297c9cbc39cd67db15b0b9db6a229b15671b848df5ec3101de9ef8df7c9973` |
| Prompts / generations | B=8, G=16, 128 trajectories per update |
| Sandbox concurrency | 64, producing the unchanged 128 rows in two waves |
| Prompt / response / turns | 4,096 / 16,384 / 50 |
| Sampling | temperature 1.0; top-p and top-k must resolve identically in both arms and be printed in the signed receipt |
| Topology per arm | 128 TPU total: rollout DP8 x TP8 = 64 and trainer DP8 x TP8 = 64, synchronous disaggregated |
| Optimizer placement | TPU resident; host offload forbidden |
| Objective | RLOO, `sequence-mean-token-scale`, epsilon 0.20 / 0.28, beta 0 |
| Optimizer | Adam learning rate 1e-6, betas 0.9/0.99, weight decay 0.01, global grad clip 1.0 |
| Update structure | prompt-counted `batch_size=8`, `mini_batch_size=8`; trajectory mini-batch 128; trajectory micro-batch 16; eight equal-size accumulation calls |
| Policy iterations | one; no off-policy replay |
| Deadlines | batch 3,600 s; episode 3,000 s; sandbox active 3,300 s; turn 300 s; step/reward 600 s; cleanup 300 s |
| Prefix cache | off |
| Horizon | direct full campaign: exactly 1,000 committed updates; updates 1–3 are monitoring milestones, not a separate job |

The implementation must not set `mini_batch_size=128`: that CLI is counted in
prompts. The 128 count belongs to the trajectory-level fields.

## Algorithm-neutrality contract

Both arms retain rollout log probabilities because those are part of the
training ratio and the treatment measurement. Both arms disable optional
corrections and sample-selection interventions:

- `sampler_is=None`; its threshold is inactive;
- `group_clip_filter_threshold=None` (renderer sentinel `-1` only if that is
  the parser's registered representation);
- `degenerate_group_masking=false`;
- DeepSWE compact filtering remains enabled: trajectories ending in
  `MAX_STEPS_REACHED`, `MAX_CONTEXT_LIMIT_REACHED`, `TIMEOUT`, `ENV_TIMEOUT`,
  `MODEL_TIMEOUT`, or `REWARD_TIMEOUT` are retained in the trajectory journal
  but receive an all-zero policy-loss mask;
- a compact-filtered local trajectory skips final reward evaluation, matching
  the pinned quality-fix path; this is not reward relabeling;
- no flat-group resampling;
- `num_iterations=1`; `off_policy_steps=0`.

All-zero and all-one reward groups with valid policy masks remain in the batch.
Under RLOO they produce zero advantage naturally; they are measured rather
than resampled. Compact-filtered rows are a separate axis from reward-group
signal and are excluded from policy loss. W&B records raw per-batch solved
count/ratio, all-zero/all-one/mixed/effective group counts and ratios,
compact-filtered row/group counts and ratios by status, structurally invalid
rows, reward histogram, nonzero-advantage count/ratio, completion/turn lengths,
latency, throughput, cleanup, policy ratio/clip diagnostics, gradient/update
norms, and A-B/B-C/A-C receipts. Timeout metrics split sandbox start,
environment reset/step, model generation, final reward, and trajectory-deadline
stages. Scheduler dimensions are a fixed vocabulary (`unschedulable`, CPU,
memory, ephemeral storage, or other); raw scheduler text remains in the
bounded run log and is never sent to W&B.
Complete redacted trajectories are journaled atomically to durable storage;
W&B stores compact metrics plus artifact path and digest.

The trajectory journal uses a monotonically increasing `batch_index` separate
from `optimizer_step`. An all-filtered batch advances only the former. Resume
validates contiguous files, metrics parity, and each SHA-256 before choosing
the next batch index; partial or tampered journals stop fail-closed.

Sandbox construction is also fail-closed. A Kubernetes pod start timeout must
be propagated through environment reset as signed `ENV_TIMEOUT` only after
pod deletion is confirmed. Returning a RepoEnv with `container=None` and
continuing setup against a deleted pod is structurally invalid, not a
trajectory result. Repeated batches with zero confirmed Running pods require
cluster scheduling/capacity evidence before another launch. An
`all_sandbox_start_timeout_batch` means effective environment throughput was
zero and does not diagnose model-serving throughput. Conversely, model
throughput is considered only when sandbox-start timeout metrics are zero and
`MODEL_TIMEOUT` is observed after real environment admission.

## Paired-treatment invariants

- Source, base checkpoint, clean-data bytes and ordering, prompt assignment,
  seed schedule, sampling parameters, topology, optimizer, objective, deadline,
  checkpoint cadence, and observer schema are byte- or value-equal.
- `native` changes only the registered numerical implementation bundle. Finite
  A-B is expected and logged; nonfinite values, B-C divergence, replica drift,
  transaction errors, OOM, and corrupted artifacts remain fatal.
- `zero` enables the complete canonical serving/forward/backward bundle. Any
  A-B, B-C, or A-C discrepancy is fatal; warning-only is not a zero arm.
- Observers may not change token selection, reward, advantages, loss masks,
  optimizer math, or commit count. A one-host test only checks this neutrality;
  it cannot prove the 128-chip treatment.
- Exact rollout/trainer weight attestation is arm-aware but equally strict.
  Zero delegates to its registered canonical adapter. Native uses the same pure
  trainer-to-engine leaf mapping and bitwise comparison as an observer only;
  it must neither register the canonical adapter nor replace serving functions.
  Missing/mismatched weights, mesh drift, an unsigned route, or a leaked native
  canonical adapter are fatal before A/B/C.
- Every rollout batch must contain exactly 128 raw trajectory records. A
  signed compact-filter status may produce a zero policy mask and is not a
  malformed row. Missing, duplicated, structurally empty, or parser-invalid
  records are fatal. If all 128 rows are compact-filtered, the update performs
  no optimizer commit and does not resample; partial filtering uses the exact
  effective-row denominator described in P58.1.

Postflight is arm-aware. Native records the stock JAX sharded trainer
transaction without pretending it used the zero arm's explicit fixed-tree DP
reducer. Zero records explicit DP8 fixed-tree reduction receipts. Both require
device-resident optimizer state, exact commit accounting, and unchanged state
for all-filtered zero-commit batches.

## Loss decision

P58 provisionally freezes `sequence-mean-token-scale`, subject to the P58.1
exit gate. The reason is not merely the current default: it implements the
published DeepSWE/Dr.GRPO intent of a fixed maximum-context divisor, matches
the exact Tunix notebook path used by the prior workload, and avoids silently
changing effective gradient scale by roughly 16,384 at this context length.

The experiment must be described as “DeepSWE-derived” rather than “exact
official DeepSWE.” The public rLLM shell launcher uses `seq-mean-token-sum`,
while the algorithm description and pinned Tunix path use the fixed-context
scale selected here. If exact historical rLLM launcher reproduction becomes a
goal, it needs a separate third arm and a separately calibrated learning rate;
it must not be substituted into either P58 arm.

## Claim ceiling

- A paired result estimates the effect of the measured native-vs-zero numerical
  bundle under this Qwen3-4B, P46-clean, B8/G16, 16K, DP8 x TP8 recipe.
- It does not isolate RoPE, attention, lm-head, precision, or one individual
  numerical component.
- It does not reproduce Qwen3-32B DeepSWE or establish 256-chip behavior.
- Missing native mismatch dose is `NO_TREATMENT`; nonexact zero is invalid;
  incomplete trajectories or transactions make the affected run inconclusive.

## Rollback

P58 profiles, flags, renderer, tests, observers, and journals must be additive
and default off. Removing the P58 task/profile concern restores the P44/P46
paths unchanged.
