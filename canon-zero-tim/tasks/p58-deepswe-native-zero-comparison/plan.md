# Plan

## Outcome

Compare two otherwise identical synchronous disaggregated DeepSWE-derived
Qwen3-4B-Instruct numerical systems on the promoted P46 clean task set, with
one separately registered Native mitigation variant:

- `native`: preserve the untreated serving and trainer numerical programs from
  the pinned `yuxzhang/deepswe-quality-fix` reference and measure their finite
  inference-trainer mismatch as the treatment dose;
- `zero`: enable the complete canonical numerical bundle and require exact
  A=B=C at every admitted boundary.
- `native-is`: preserve the same Native serving/trainer programs but apply
  token-level sampler/trainer truncated importance weights at threshold 2.0.
  This is a mitigation arm, not part of the original two-system estimand.

This is a causal comparison of two complete numerical systems. It is not an
ablation of one kernel, and it is not an exact reproduction of the published
DeepSWE-32B run: the model, generation count, clean-data selector, and context
length differ deliberately. The compact trajectory-filtering rule does match
the published DeepSWE recipe and the pinned Tunix quality-fix reference.

The execution order changed again by user decision on 2026-08-23. The native
full campaign remains incomplete historical evidence; the current work is
P58.6, a matched one-host Native/optimized-Zero XProf carrier, followed by
P58.7, an optimized strict-Zero Qwen3-4B full recipe. P58.8 then repairs the
TP4/TP8 P59 nested-mesh admission failure and the P57 Zero/full telemetry
identity exposed by the first Phase4 target logs. Review then found that the
new exact-image gate exercised only a synthetic nested map, not the installed
fixed-head/projection shim VJP that failed in GSM8K. P58.9 now refines the
latest published tree by adding the Native-IS recipe as a fail-closed selector
and restoring exact Attempt-0 semantics until attempt-isolated durable roots
exist. P58.10 makes seed 42 an explicit shared recipe field and wires that
same value into both dataset shuffle and rollout sampling, with durable
provenance and a deliberately limited reproducibility claim.

On 2026-08-24 the operator reported a sharp training-reward drop in the live
Native/no-IS campaign and judged it collapsed. The onset update is unknown; do
not attribute the event to a fixed optimizer step. Preserve and stop
that exact run; do not resume its checkpoint. Native raw is removed from the
active launch queue. The next training target is a fresh Native+IS full run
from the original frozen base checkpoint after the local P58.9 source is
explicitly published and read back.

Native+IS was subsequently prepared and published. On 2026-08-26 the user
reactivated the strict Zero-HP target. P58.11 admitted the latest shared
checked-VMA, first-update, and overflow-safe clip repairs. Its source passed
construction and was launched as `p58z01`, but the first
Step-0 generation exposed the P58.10 per-request seed incompatibility with
vLLM JAX. At that point P58.12 became the only active phase: preserve seed 42 at engine
scope, require exact route provenance, and make the independent empty-body
cleanup failure bounded and fail-closed. Target `p58z02` proved that repair by
returning all 128 rollout rows, then failed in the first trainer-logprob
forward because Qwen3-4B TP8 did not admit fixed-head semantic M=2,048.
P58.13 admitted the exact 4B geometry and `p58z03` proved its M=2,048 path,
but the same attempt exposed a role-placement error before trainer execution.
P58.14 rebound the adapter's explicit trainer placement, and `p58z04` proved
those receipts plus all 128 rollouts. It also exposed a deeper captured-device
closure: vLLM's nested model/logits JITs still named rollout devices. P58.15
reconstructed the same weight-free model graph and nested JITs on trainer
devices. `p58z06` reached that reconstruction before rollout and exposed that
Pathways dummy loading adds `_is_loaded=True` provenance to the live NNX State
while the abstract clone has no loader marker. P58.16 then became the only
active phase. It normalizes only that exact true-valued marker while retaining exact
logical tree, metadata, leaf, shape, and dtype checks in both full and
segmented trainer paths. Every serving and numerical/algorithmic contract is
unchanged.

`p58z07` subsequently proved P58.16 and completed Step-0 rollout, then exposed
a finite strict pre-backward `S_decode_vs_S_prefill` RED while
`S_prefill_vs_T_old` remained exact. P58.17 is now the only active phase. Its
bounded DP1xTP4 same-task Zero-HP carrier has run and automatically joined,
classified, and packaged the returned trajectory/alignment evidence. The
carrier reproduced finite RED but also found B-C RED, so it is not the exact
remote signature. The phase does not claim fixed-token replay or use TP4 to
certify the production DP8xTP8 geometry. Its remaining deliverable is a
default-off exact-geometry Step-0/no-commit checked-VMA discriminator, not a
new full-training retry.

The exact-geometry discriminator is refined in P58.18 into three independent,
matched Step-0 JobSets: logical `ON-A/OFF/ON-B`. Per the operator's capacity
plan they are prepared for concurrent submission. Concurrent execution makes
the two ON arms replication controls around one matched OFF treatment; it is
not a temporal ABA sandwich. All three retain the same frozen recipe and stop
before backward or optimizer state mutation.

The sealed `p58aba01` wave returned A-B RED with exact B-C in all three arms,
so checked-VMA is not sufficient.  P58.19 is now the only active phase.  It
uses one 128-chip JobSet with three sequential frozen-weight rounds to locate
the first reproducible coarse model boundary.  This is observation, not a
numerical treatment: checked-VMA, sampling, precision, fixed-M geometry, loss,
and the Zero-HP serving program remain unchanged, while backward and optimizer
commit remain unreachable.

## Phases

| Phase | Deliverable | Exit gate | Status |
|---|---|---|---|
| P58.1 | Frozen shared recipe and loss-aggregation contract | Fixed-16K formula oracle, compact-filter mask policy, effective-row-weighted accumulation, and DP8 invariance are specified and locally tested; public-source discrepancy is recorded | completed |
| P58.2 | Default-off paired profiles, renderer, metrics, trajectory journal, and negative controls | Host tests and pinned-image tests prove that the two rendered arms differ only in the registered numerical treatment bundle | completed |
| P58.3 | One-host observer and artifact sanity | Full redacted trajectory schema, W&B metric schema, logprob/alignment observers, checkpoint transactions, and no-update neutrality pass without a production claim | waived — not PASS |
| P58.4N | Native 128-chip three-update canary | Native completes exactly three optimizer commits on rollout DP8 x TP8 plus trainer DP8 x TP8, records a finite nonzero serving-path mismatch dose, keeps every stock numerical boundary finite, and emits a signed classifier PASS | superseded — p58c05 failed before execution; not PASS |
| P58.4Z | Zero 128-chip three-update canary | Zero completes exactly three commits on the identical recipe with strict A=B=C and a signed classifier PASS | deferred — do not launch |
| P58.5N | Native 128-chip full campaign | Native completes exactly 1,000 optimizer commits; the first three are monitored without stopping; durable trajectory, evaluation, checkpoint, alignment, optimizer, and classifier evidence passes | incomplete historical campaign; superseded as active implementation queue |
| P58.5Z | Zero full or paired comparison | Activated only after zero optimization and a new explicit user decision | deferred — do not launch |
| P58.6 | Matched one-host Native/Zero-HP XProf and Perfetto carriers | Both immutable arms pass on the same direct four-chip host and pair work hashes match | implementation + pinned-image PASS; direct TPU pair not run |
| P58.7 | Optimized strict-Zero Qwen3-4B full recipe | Exactly 1,000 DP8 x TP8 commits, zero alignment failures, complete P59/fixed-head/XProf/Perfetto/postflight receipts | implementation + construction PASS; target not run |
| P58.8 | P59 TP4/TP8 nested-mesh and P57 Zero/full telemetry repair | Installed fixed-head + projection shim VJPs run through P59 parallel/report/fixed-reducer no-commit paths; local-output positive/negative placement controls pass; four independent CLs are rebuilt on latest tip `ccbcf572` | local four-CL release and committed-tree manifest audit PASS; push/hardware target not run |
| P58.9 | Native token-IS plus Attempt-0 refinement | Native raw/IS and Zero-HP render as a closed set; partial/mixed tuples fail; runtime proves old-logp/TIS provenance; JobSet retry and unconsumed keepalive overrides are absent; full pinned-image gate passes | implementation `2aedd73c` published/read back; Native-IS selected after operator-observed Native-raw reward collapse; onset step unknown, target not run |
| P58.10 | Fixed dataset and rollout seed | All three P58 recipes render exactly one `--seed=42`; training, W&B, manifests, classifiers, and first-batch marker agree; missing/duplicate/drifted values fail closed | implementation `9597de3d` published/read back; pinned-image PASS; target not run |
| P58.11 | Qwen3-4B strict Zero-HP checked-VMA admission | Exact P58 Zero/full profile alone derives checked-VMA, first-update, and overflow-safe clip; Native/neighbor negatives pass; complete pinned-image gate passes; target remains separately launch-gated | source/construction completed; `p58z01` target failed before first generation on seed route |
| P58.12 | JAX engine seed route and bounded abort cleanup | P58 uses `EngineArgs.seed=42` with no JAX per-request seed; exact route/manifest/postflight and cleanup retry regressions plus complete pinned-image gate pass | completed source repair; `p58z02` proved route and reached trainer forward |
| P58.13 | Qwen3-4B trainer-logprob M2048 and P59-only VMA scoping | Exact `(2560,8)` fixed-head registration, Qwen3-32B negative, FrozenLake Wave-5 P67 bundle, profile/environment/Python fail-closed gates, and complete pinned-image PASS | completed source repair; `p58z03` proved fixed-head admission and exposed P58.14 before execution |
| P58.14 | Disaggregated canonical trainer-mesh binding | Adapter receives trainer state, executes differentiable forward on the matching trainer role, retains rollout serving placement, rejects DP/TP/partial-overlap drift, and passes disjoint plus colocated image regressions | completed source repair; `p58z04` proved explicit placement and exposed P58.15 nested JIT closure |
| P58.15 | Disaggregated nested model/logits JIT and segmented-backward binding | Reconstructed graph is state-contract exact, both hidden JITs and segmented backward execute on trainer devices, three placement receipts are classifier-enforced, and dependency-image regressions pass | completed source repair; `p58z06` exposed P58.16 loader-metadata admission before rollout |
| P58.16 | Loader-metadata-aware NNX logical State contract | Only exact `_is_loaded=True` provenance is normalized; every other metadata/path/type/leaf/shape/dtype contract stays exact; full and segmented forced-device tests plus classifier and complete image gate pass | completed — implementation published/read back; `p58z07` proved the state/JIT/scorer contracts and exposed P58.17 after rollout |
| P58.17 | Decode-vs-prefill seam probe | Exact historical mismatch join plus one real-task DP1xTP4 Zero-HP carrier, followed by a single-selector 128-chip DP8xTP8+DP8xTP8 checked-VMA-off Step-0 discriminator with durable 128-row evidence and zero VJP/backward/commit | implementation `b54bd81a` published/read back — `p58s17` returned finite A-B and B-C RED over 4,808 action tokens; `p58z08` reran the ordinary checked-VMA-on full arm and is not discriminator evidence; exact-geometry selector passes construction gates, target still not run |
| P58.18 | Exact-geometry checked-VMA matched triplicate | Render and verify three independent ON-A/OFF/ON-B 128-chip Step-0 carriers whose recipe and treatment signatures differ only in the selector; each returns 128 durable rows, exact B-C, one finite/exact A-B classification, controlled exit, and zero VJP/backward/commit | completed — sealed `p58aba01` PASS / `CHECKED_VMA_NOT_SUFFICIENT`; all arms A-B RED, B-C exact, backward/commits zero |
| P58.19 | Three-round coarse decode/prefill seam localization | One default-off selector renders one exact 128-chip carrier with three sequential frozen-weight layer/tail observation rounds; every round has finite A-B, exact B-C, bounded classification, verified durable seal, and zero VJP/backward/commit | active — `p58s19b` reached the observer but `[3072,4608)` emitted zero records; local P58.19c changes the signed window to evidence-derived `[1686,4096)`; construction validation is complete and source publication is approved for this delivery; target retry is not run |

Exactly one phase may be active. Commit, push, image publication, Kubernetes
render/application, and TPU execution each remain separately user-gated.

P58.1 and P58.2 are closed by the pinned-image marker recorded in `state.md`.
P58.3 has CPU coverage for journal continuity and observer/classifier logic but
no real Qwen/R2E one-host evidence; the user explicitly waived it rather than
calling it PASS. P58.4N was superseded after p58c05 failed Kueue admission.
P58.5N never completed and is not a valid full Native baseline. P58.19 is the
only active phase; P58.6 through P58.18 retain their historical phase records.
P58.7's historical target remains not run and is superseded for new Zero
launches by P58.11 plus the P58.12 seed-route correction. P58.9 and P58.10
source are published and read back. P58.13 implementation
`bea1aabde39c43c13ca4eaefab989301c6e8b46c` is also published and read back;
its `p58z03` target is immutable failure evidence. P58.14 implementation
`dce0e93777548b7623e4f41702144f8d00f242f5` is published and `p58z04` is
immutable P58.15 trigger evidence. P58.15 implementation
`f60cdd569c2737df6cb2968125c8e42680938981` is published, and `p58z06` is
immutable P58.16 trigger evidence. P58.16 implementation
`dba5211ac4945fefb50337603c800d9f8e3d37b5` is published/read back, and
immutable `p58z07` proved it before triggering P58.17. The P58.17 one-host
carrier is historical diagnostic evidence, and P58.18 is sealed target
evidence. P58.19 implementation
`f58a97748a8895835fba4944f5c5a34ba8bee352` is published/read back; its
exact-geometry carrier remains separately launch-gated.
No remote execution is authorized by this plan alone.

P58.5N attempts `p58f01` through `p58f11` remain `INCONCLUSIVE`. P58f01 exposed
missing sandbox LocalQueue inheritance and reset-time policy provenance;
p58f02 showed that the chosen CPU flavor required `cpu-np`; and p58f03 proved
that the CPU routing repair works by completing 128 real trajectories in
616.3 seconds. P58f03 stopped after durable journaling but before trainer
forward because the native arm was incorrectly sent through a
canonical-adapter-only weight-attestation method. The published repair provides a
shared exact-live-weight observer while preserving native numerical
untreatedness. P58f04 proved that repair with exact 398-leaf live weights after
a 557.2-second, 128-row rollout. It then stopped before trainer forward because
the shared processed-`S_prefill` contract accepted only the canonical
processed-logprob engine even though native correctly kept that numerical flag
off. P58f05 proved the independent native stock-B observer over all 2,048
prompt rows after a 486.4-second, 128-row rollout and exact 398-leaf weight
attestation. It then stopped before trainer forward because the alignment
policy's P58 admission enumerated only short update stages and incorrectly
rejected the signed `full/1000` tuple. The published repair separates P58 from
the debug-stage admission and requires its existing admission, native arm, and
exact stage/horizon signature. Its implementation SHA is
`5132d7ad0d3bc7c53de09e20bae835dca18a211a`. P58f06 proved that repair and
executed alignment after another healthy 128-row rollout, exact live weights,
and 2,048-row Native B observation. Its A-B and B-C arrays were shape-valid and
finite across 405,827 action tokens, but the P58 policy still narrowed warnings
to A-B and blocked on B-C before trainer forward. The correction admits both
finite Native serving-path boundaries as treatment observations. P58f07 proved
that repair: all 128 real
RepoEnv trajectories completed, the pre-backward gate admitted both finite
serving warnings, and the trainer entered real value-and-grad/backward. It then
stopped on trainer `T_old_vs_T_current` because the gate still required an
observer-only stock rescore to match the value-and-grad primal exactly. The
local correction keeps the original 128-trajectory quality-fix observer and
classifies every shape-valid finite Native program mismatch as measurement.
Zero remains exact. P58f08 did not exercise that repair: six concurrent
Pathways heads already occupied the six available `cpu-np` nodes, Kubernetes
packed the new host-network head onto an occupied node, and fixed port 29001
connected its CL/956357083 worker to a foreign CL/42 ResourceManager. Moving
the head to `deepswe-cpu-pool` was also tested and rejected because the worker
could not maintain its scheduler pipe across the node-pool subnet boundary.
P58f09 kept the proven host-network transport on `cpu-np`, attached to the
correct Pathways server, and completed all 128 Step-0 rollout slots in 1,699.1
seconds. It then failed before durable journaling or any trainer program:
reset-deadline rows had terminated before first observation, leaving
`agent.trajectory.task=None`; learner `merge_micro_batches()` dereferenced that
value as a mapping.

The published p58f09 implementation repair therefore preserves head and TPU-worker host networking,
adds required hostname-level anti-affinity over every JobSet `pathways-head`
replicated-job Pod, and validates the exact JobSet DNS/RM route. The collector
falls back to the environment's original task only when the agent never
observed one, and fails closed if neither source is a dictionary. Admitted
compact rows remain journaled with zero policy masks; no row is dropped or
resampled. No model, data, numerical, topology, deadline, optimizer, or
Native/Zero flag changed.

P58f10 ran the source containing that repair and entered Step-0 rollout, but
the batch timeout prevented post-rollout merge, so the original-input fallback
remains target-unproven despite exact-image coverage. It exposed an independent
scheduling mismatch: B8 x G16 produces 128 trajectories while
`max_concurrency=64` admitted only 64 at a time. The resulting two sequential
waves did not drain before the 3,600-second hard batch deadline; only 5/8
prompt groups were complete when the orchestrator failed closed. The repair
sets concurrency to 128, exactly matching both the raw batch and rollout
capacity DP8 x max-seqs16. Episode, cleanup, and batch deadlines remain
3,000/300/3,600 seconds. Per-trajectory timeout/context outcomes still return
as compact zero-mask rows; only failure of the complete one-wave batch to drain
remains fatal. After fetching/readback of the final operator tip, the next
attempt is `p58f11`.

P58f11 proved the one-wave repair: all 128 trajectories and all 8 prompt groups
completed in 1,209.2 seconds. One generation ended during environment reset,
and the existing compact trajectory path preserved it. That row exposed the
next schema defect: `SWEEnv` retained the dataset row only as `self.entry`,
while inherited `self.task` contained only the pre-reset `policy_version`.
The collector fallback therefore produced a dictionary without `prompts`, and
learner reward-input merge stopped with `KeyError: 'prompts'` before the P58
journal or any trainer program.

The published correction makes the normalized prompt a durable part of
`SWEEnv.task` before reset and makes the policy-seeded environment task the
single original-input source for training trajectories, including successful
and pre-observation termination paths. This prevents mixed schemas inside a
G16 group. Missing `prompts` on a policy-seeded task fails at collection.
Timeout/context masks, rewards, advantages, filtering, resampling, Native/Zero
flags, data, topology, loss, optimizer, deadlines, and horizon are unchanged.
At that historical checkpoint the next attempt was fresh `p58f12`; p58f11 was
not resumable.

P58f12 proved that prompt provenance repair and wrote the first valid 128-row
Step-0 journal after it. It did not prove model rollout or training: all 128
RepoEnv Pods remained `scheduling_gated` until sandbox-start timeout, so every
row was signed compact-filtered `ENV_TIMEOUT`, action/completion token counts
were zero, and `generate()` was never called. The Native processed-B observer
then required rollout sampling-transform provenance and failed before
alignment, backward, optimizer, or checkpoint. The journal is durable
diagnostic evidence, not resumable trainer state.

The local correction implements the already frozen all-filtered contract all
the way through the outer learner. A signed P58 batch with zero completion
targets validates its structure and observer signature, skips engine rescore,
and records explicit empty-target provenance; it does not synthesize
log-probabilities. Only durable all-compact provenance admits zero action
tokens through alignment. For an ordinary all-compact model/context/runtime
batch, the trainer's existing zero-gradient transaction makes no optimizer
commit, the outer learner suppresses weight sync plus policy/RL/trainer step
advance, `batch_index` advances, and the next clean prompt batch is consumed
without resampling.

A batch whose durable metrics prove `all_sandbox_start_timeout_batch=true` is
an infrastructure outage, not an ordinary training sample. Once its journal
and bounded timeout metrics exist, a separate circuit breaker emits
`[P58.SANDBOX_CAPACITY] BLOCKED` with `optimizer_commits=0` and
`prompts_consumed_after_batch=0`, then raises `BLOCKED_SANDBOX_CAPACITY` before
rescore, alignment, trainer execution, weight sync, or consumption of a later
prompt batch. Evidence claiming that condition while row counts, timeout
counts, compact masks, effective groups, environment-timeout status, or token
targets disagree is fatal. Any nonempty target still requires real
post-`generate()` sampling provenance and engine rescore; unsigned
zero-action, nonfinite, shape, or nonzero-gradient cases remain fatal.

Fresh `p58f13` is the next attempt only after exact final remote readback and a
separately approved live one-sandbox admission probe.
The probe must use a real frozen-clean-list task image, queue
`multislice-queue`, node pool `cpu-np`, and production requests 2 CPU/4 GiB;
the read-only verifier must observe the Pod Running with no scheduling gate.
That is only one-Pod evidence. Before the full JobSet, the operator must also
confirm ClusterQueue/ResourceFlavor/autoscaler capacity for 128 sandboxes:
at least 256 requested CPU and 512 GiB requested memory, plus head and cluster
overhead. P58f12's 128/128 `scheduling_gated` result cannot be fixed by tuning
vLLM concurrency, and removing the Kueue queue label is forbidden.

Read-only review of `origin/main` at
`c7d8950f12a9c55a976bf2e1a0d8b447d71c20b3` found Agent Sandbox/SandboxFleet
commit `e789573964b6f695ded85fe519040bd06a2b9f37`. It is not integrated into this
repair: it does not supply CPU quota, its prewarm failures are warning-only,
and its current-plus-lookahead sizing can request 256 sandboxes for B8 x G16.
Any later P58 port needs a separate default-off, Kueue-aware, fail-closed phase
with current-batch-only capacity and exact cleanup receipts. `main` remains
untouched.

## Frozen shared recipe

| Field | Shared value |
|---|---|
| Model | `Qwen/Qwen3-4B-Instruct-2507` |
| Data | promoted 1,012-task P46 exact-N16 list, SHA-256 `ec297c9cbc39cd67db15b0b9db6a229b15671b848df5ec3101de9ef8df7c9973` |
| Prompts / generations | B=8, G=16, 128 trajectories per update |
| Sandbox concurrency | 128, exactly one wave matching the raw batch and rollout DP8 x max-seqs16 capacity |
| Prompt / response / turns | 4,096 / 16,384 / 50 |
| Sampling | temperature 1.0; top-p and top-k must resolve identically in both arms and be printed in the signed receipt |
| Topology per arm | 128 TPU total: rollout DP8 x TP8 = 64 and trainer DP8 x TP8 = 64, synchronous disaggregated |
| Optimizer placement | TPU resident; host offload forbidden |
| Objective | RLOO, `sequence-mean-token-scale`, epsilon 0.20 / 0.28, beta 0 |
| Optimizer | Adam learning rate 1e-6, betas 0.9/0.99, weight decay 0.01, global grad clip 1.0 |
| Update structure | prompt-counted `batch_size=8`, `mini_batch_size=8`; trajectory mini-batch 128; trajectory micro-batch 16; eight equal-size accumulation calls |
| Trainer observer | Stock quality-fix prompt-counted rescore: one 128-trajectory observation for B8 x G16; observer-only under `use_rollout_logps=true` |
| Policy iterations | one; no off-policy replay |
| Deadlines | batch 3,600 s; episode 3,000 s; sandbox active 3,300 s; turn 300 s; step/reward 600 s; cleanup 300 s |
| Prefix cache | off |
| Horizon | direct full campaign: exactly 1,000 committed updates; updates 1–3 are monitoring milestones, not a separate job |

The implementation must not set `mini_batch_size=128`: that CLI is counted in
prompts. The 128 count belongs to the trajectory-level fields.

## Algorithm-neutrality contract

All recipes retain rollout log probabilities because those are part of the
training ratio and treatment measurement. Native raw and Zero disable optional
corrections. The active Native-IS mitigation changes only the registered
sampler correction:

- Native raw and Zero: `sampler_is=None`; threshold inactive;
- Native-IS: `sampler_is=token`, threshold `2.0`, trainer logps are old policy
  logps, and token TIS weights correct the trainer-vs-rollout sampler gap;
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
from `optimizer_step`. An ordinary all-filtered batch advances only the former;
a full sandbox-start outage writes its current journal and stops. Resume
validates contiguous files, metrics parity, and each SHA-256 before choosing
the next batch index; partial or tampered journals stop fail-closed. P58f12 has
no trainer checkpoint and is diagnostic evidence, not resumable training.

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
- `native` preserves the registered stock numerical implementation bundle.
  Finite A-B, B-T_old, and T_old-T_current differences and their finite ratio
  consequences are treatment observations. Nonfinite values, invalid shapes,
  replica drift, transaction errors, OOM, and corrupted artifacts remain fatal.
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
- Processed B observation is also arm-aware and mutually exclusive. Native
  keeps `CANON_PROMPT_PROCESSED_LOGPROBS=0`, `CANON_ENGINE_MODULE_C=0`, and all
  zero-TIM numerical switches disabled/absent; its independent, signed stock
  observer only transforms post-rollout prompt logits and uses absolute
  request-history targets. Zero keeps that observer off and uses the complete
  canonical processed-logprob engine. Either mixed tuple is a hard contract
  error. The observer may not affect generation, trainer forward, loss,
  backward, optimizer math, or commit count.
- Every rollout batch must contain exactly 128 raw trajectory records. A
  signed compact-filter status may produce a zero policy mask and is not a
  malformed row. Missing, duplicated, structurally empty, or parser-invalid
  records are fatal. If all 128 rows are compact-filtered by ordinary
  model/context/runtime outcomes, the update performs no optimizer commit,
  weight sync, or trainer/RL/policy-version step advance; it does not resample
  and consumes the next prompt batch. Its `batch_index` advances while
  `optimizer_step` remains the actual committed trainer step. If all 128
  failed before sandbox start, persist the journal and stop
  `BLOCKED_SANDBOX_CAPACITY` without consuming another prompt. Partial
  filtering uses the exact effective-row denominator described in P58.1.

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
