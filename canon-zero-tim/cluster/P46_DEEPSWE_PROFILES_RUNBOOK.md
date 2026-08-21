# P46 DeepSWE evaluation and training profiles

P46 is the operator entrypoint for the current DeepSWE campaign. It maintains
three immutable workload families with a workload-specific topology allowlist:
Qwen3-4B debug/evaluation render on 64-chip `4x4x4` or 128-chip `4x4x8`, while
Qwen3-32B training renders on 64-chip `4x4x4` or 256-chip `4x8x8`. The renderer
writes the signed parameters directly into the JobSet; do not add a second
shell override layer.

## 2026-08-21 returned-snapshot correction

The current operator path is **legacy-v5 adoption into a fresh census tag**,
not the generic frozen-v6 path below. Attempt
`canon-p46-eval-census-128-p46c128a0` selected a directory named
`p46e12806-v6-final`, but its rows are
`canon.p46.deepswe-eval.trajectory.v5` with sampler
`stock@ac2c31bc7f6f82d33b3a62d62e1c390c8338b60e`. Omitting the explicit
historical sampling SHA caused the correct fingerprint failure before model
runtime.

Do not reuse `p46q4census01`: the old code wrote its incorrect immutable
resume contract before import validation. Preserve it as incident evidence.
Use a fresh destination such as `p46q4census02`; make a sealed v5-only copy
containing trajectory JSONLs plus `SHA256SUMS` and no
`resume_contract.json`; then render `--legacy-import-id` with explicit
`--sampling-source-commit ac2c31bc7f6f82d33b3a62d62e1c390c8338b60e`.
Require `LEGACY_IMPORT_PASS records=<actual>` before runtime. Imported durable
identities are skipped, so this does not restart washing from zero.

The fixed entrypoint validates all legacy-v5 rows before writing the new resume
contract, and the renderer refuses either import mode without an explicit
sampling-source SHA. Repair implementation
`f823bb6a9aabf023e651788452d94ff656c827e1` must be present in the freshly
read-back operator-branch ancestry before a remote launch. Full commands,
exact error evidence and cardinality limits are in
`P46_CENSUS_SNAPSHOT_RESUME_INCIDENT.md`.

## P46.7 breadth-first census and generic v6 handoff

This section applies only when inspection proves the source records are real
trajectory-v6 and a matching sealed `resume_contract.json` exists. Directory
names do not determine schema. Implementation
`365b46c1cd150839e3be1fd50adb33325fe3189f` is published and was read back
exactly from `yuxzhang/canon-zero-tim`; executors still repeat the fresh
ancestry/read-back gate before launch. It supersedes the strict-first launch
ordering below but does not replace the strict completion gate.

The returned campaign should first cover every still-never-attempted
task/sample identity once. Render Q4 evaluation with all three controls:

```text
--full-campaign
--first-pass-census
--resume-tag <fresh-v6-migration-tag>
```

For a genuine v6 source only, on the first launch add
`--frozen-v6-import-id <sealed-old-v6-snapshot-id>` and pass the old
`resume_contract.json` source SHA through `--sampling-source-commit`. A newer
census-capable `--source-commit` is the harness SHA. They must not be silently
collapsed.

### Why the old v6 tag cannot be resumed in place

Trajectory-v6 fingerprints bind both `resume_tag` and `harness_commit`.
Checking out the old SHA lacks census code; checking out the new SHA against
the old tag correctly causes a contract mismatch. Therefore:

1. Wait until every old producer is terminal and no sandbox pod remains.
2. Copy old `outputs/resume_contract.json` and the entire
   `outputs/trajectories/` tree into
   `<fresh-root>/imports/<import-id>/`; never move or edit the old evidence.
3. From inside that snapshot, write `SHA256SUMS` containing the relative path
   `resume_contract.json` and every relative `trajectories/*.jsonl` path, then
   make the snapshot read-only.
4. Use a fresh destination tag with no trajectory JSONL. Require
   `[P46.RESUME] FROZEN_V6_IMPORT_PASS` before TPU runtime preparation.

The importer validates old resume contract self-consistency, all digests,
every per-logical fingerprint/run tag, clean identity/nonce, consecutive
attempt sequence, reward/validity fields, reward-only logprob absence and
sampler provenance. Only destination harness SHA and resume tag may differ.
Raw trajectories and `sampled_by=stock@<old-source-sha>` are retained; copied
rows add source tag/harness/fingerprint/path/line/record-digest provenance.

### Census runtime and result interpretation

The signed workload remains Qwen3-4B-Instruct-2507, 1,851 clean prompts, N16,
16,384 response tokens, 50 steps, reward-only, prefix cache off, concurrency
64, and one 3,600-second physical-wave budget. Census only changes retry
ordering and is deliberately outside the sampling fingerprint:

- any durable attempt, valid or invalid, suppresses another census attempt;
- model/context/max-step/signed trajectory timeouts remain valid unsolved;
- `FAILED`, environment/reward timeout and malformed results remain invalid,
  are persisted, and appear in `deferred_identities.jsonl`;
- a bounded wave timeout does not stop later-wave traversal; and
- an interrupted census relaunch runs only identities with no durable row.

`P46_EVAL_CENSUS_INCOMPLETE` is expected while unattempted identities remain
and exits nonzero. Census is done only at:

```text
P46_EVAL_CENSUS_PASS tasks=1851 scheduled_identities=29616 unattempted=0
```

This marker proves breadth coverage, not exact-N washing. Immutable snapshots
under `outputs/census/` explicitly claim
`breadth_first_coverage_only_not_final_washing`; provisional mixed/all-fail/
all-pass files must not be used as final training data.

After census PASS, use the same destination tag, new harness SHA, old sampling
source SHA, topology, model/image/data and sampling fields. Omit
`--first-pass-census` and `--frozen-v6-import-id`. Strict mode then retries all
identities lacking a valid result and retains its unchanged completion gate:

```text
P46_EVAL_CAMPAIGN_PASS tasks=1851 n_sample=16 valid_trajectories=29616 logical_shards=58
```

The executable freeze and render commands are maintained in
`tasks/p46-deepswe-eval-training-profiles/HANDOFF.md`. Existing
`p46e12808`/`p46e12806` registry state predates P46.7; do not assume its
rendered manifest contains this mode, and do not mutate it without separate
operator authority.

## P46.6 strict campaign baseline (used after P46.7 census)

This section supersedes the legacy shard-by-shard promotion sequence later in
this document and remains the strict-repair stage after census. Q4 full
clean-data washing uses one persistent runtime, not another standalone
`l0/p0` test. Returned 128-chip run
`p46e12804` exposed the old greedy action-tag bug and the cost of repeating
model init/JIT. The later legacy-v5 campaign `p46e12805` ran the fixed action
adapter at sampler source `18d5d2ac1603a26a221af9d5fc430b084ec002df`.
Its archived first wave contains 64 valid identities and ten reward-one
outcomes, and its log reached the second wave. The operator reports that job is
still running: do not stop, delete, relabel, or copy its live output.

P46.6 makes the Q4 repairer explicit as
`action_compat_mode=q4_r2egym_xml_v2`; ordinary agents and Qwen3-32B training
remain `strict_xml`. Deterministically repaired Q4 syntax and rejected model
tool calls are model outcomes, not retryable infrastructure. Raw response,
canonical action, repair count, and model-action-error count are persisted in
config-v4/trajectory-v6. Only malformed harness structure or a repairer-created
corruption is a harness failure.

Production washing is rendered with `--full-campaign`. It initializes Q4/vLLM
once, then runs 463 sequential waves under the unchanged per-wave contract:
four tasks x N16, concurrency 64, 16,384 response tokens, at most 50 steps,
and a 3600-second deadline. Every trajectory is fsynced immediately. Relaunch
with the same explicit resume tag, published SHA, topology and evaluation
contract to resume after interruption; the Kubernetes run id may be new.
Completion still requires 1851 tasks, 29,616 valid identities, 58 logical
summaries, all referenced digests, postflight cleanup, and the global
`P46_EVAL_CAMPAIGN_PASS` marker.

Resume-tag hardening and frozen legacy-v5 adoption are implemented by
`c3a960acdc94173440144559bb95f1de36d31537`. Publication checkpoint
`dc6b5b32a90ad0e12b1b9ae50ef7cc060b450abf` was read back from
`origin/yuxzhang/canon-zero-tim` with that implementation in its ancestry.
Because the branch may advance, every executor must still resolve the exact
current HEAD and repeat the ancestry gate. Never launch from a dirty worktree
and never modify or push `main`.

This package has local CPU, direct one-host development evidence, and returned
256-chip evaluation attempts. Neither returned attempt is a physical-shard
PASS: `p46e25608` exposed the old invalid-attempt resume bug, while
`p46e25609` exposed a Q4-to-R2E action-tag adapter bug in every trajectory. A
rendered YAML, cardinality marker, or `SUCCEEDED` status is not a target PASS.
Do not apply a new manifest until the retry repair, action-adapter repair and
Q4 64/128 topology migration have been published to
`origin/yuxzhang/canon-zero-tim`, read back, and separately approved for
launch.

Required implementation ancestry:

```text
e1b4009394c49ea015919bda0cfdb97c12c221b5
a4d165e854cc4c2320d8120e89aed185eaf61465
a642ab267425a5b08b0cebb6e12c607f50f71831
c3a960acdc94173440144559bb95f1de36d31537
365b46c1cd150839e3be1fd50adb33325fe3189f
```

P46.5 true reward-only evaluation is published by `a4d165e8`; do not infer its
presence from `e1b40093` alone. Before using the reward-only instructions
below, require an exact operator SHA containing `a4d165e8`,
`evaluation_mode=reward_only`, the true no-logprob request construction, and
`probe_reward_only_v5p.py`. Never apply a manifest rendered from the dirty
development worktree.

Operator HEAD `63b092b001864e4e9a4822b4354a665bb00b1c6b` is the historical
returned-evidence checkpoint. It contains the archived `p46e25608` log and the
false-positive behavior, not the repair. The invalid-attempt retry and campaign
finalizer are published by
`a642ab267425a5b08b0cebb6e12c607f50f71831`. A remote agent must stop unless
the exact read-back operator SHA contains that commit and all of
`attempt_index`, `P46_EVAL_PHYSICAL_INCOMPLETE`, and
`P46_DEEPSWE_PROFILES_CPU_PASS cases=77`, plus
`finalize_deepswe_eval.py`. Do not invent or substitute a repair SHA in
advance.

The remote branch may advance with documentation or returned evidence. Resolve
and record its exact current 40-character HEAD at execution time, and require
that the resolved HEAD contains both implementation commits above.

## Returned-run correction and required fix

The archived Qwen3-32B P34r03 run is a failed attempt, not a rollout PASS. It
constructed the correct DP16 x TP8 role meshes and returned this cardinality
marker:

```text
[P34.LOGPS_BATCH] configured_prompts=8 generations=8 execution_trajectories=64 observed_trajectories=64
```

However, all 64 returned records were clipped as `ENV_TIMEOUT`; all 64 hung
environment steps were killed only after their shared trajectory budget had
already become negative. The run then failed before forward/backward or an
optimizer commit with `KeyError: 'fsdp'`. Its trainer mesh was named `dp,tp`,
while `RLTrainingConfig.data_sharding_axis` still named `fsdp`.

The published implementation repair has two parts:

1. one shared rollout-batch deadline, per-trajectory deadline, bounded model
   request cancellation and bounded R2E cleanup; and
2. trainer input sharding derived from the actual leading trainer-mesh axis.

Every training launch after publication must emit exactly one pre-rollout
marker whose data axis is `dp` and whose mesh contains `dp,tp`:

```text
[DEEPSWE.DATA_SHARDING] PASS axes=('dp',) mesh=('dp', 'tp')
```

Absence of this marker, `axes=('fsdp',)` on a canonical DeepSWE profile, any
negative remaining timeout, or any `KeyError: 'fsdp'` means the wrong source
SHA/profile ran. Archive the manifest and logs and stop; do not retry the same
manifest. A log-prob cardinality marker proves only that the expected number of
objects reached the trainer-side boundary. It does not prove those objects are
complete, valid, solved, or capable of producing a nonzero advantage.

The same distinction applies to returned evaluation run `p46e25608` at source
`bdc9681824743911d0691659604dec090dd42bc4`. Its 256-chip Qwen3-4B reward-only
physical shard `l0-p0` attempted all 64 task/sample identities, but only 62
were valid. These exact attempts ended `MODEL_TIMEOUT`:

```text
namanjain12/aiohttp_final:006fbe03fede4eaa1eeba7b8393cbf4d63cb44b6 sample=6
namanjain12/aiohttp_final:04deab71cc804311016159548e5dcdfb9c2698d3 sample=5
```

The old resume code treated any durable record, including an invalid attempt,
as completion and nevertheless emitted `P46_EVAL_SUBSHARD_PASS` with
`pending_logical_tasks=30`. That marker is revoked. The repair makes attempts
durable while allowing only consecutive retries before the first valid result;
after a valid result, another attempt is a duplicate error. A physical shard
passes only with exactly 64 valid identities and no remaining valid sample.
Any timeout or invalid-only identity emits
`P46_EVAL_PHYSICAL_INCOMPLETE` and exits nonzero.

Because the repair changes the source SHA and evaluation fingerprint, do not
copy or silently resume the 62 old `p46e25608` records. The first fixed run
must use a new run id and rerun all 64 `l0-p0` identities. Later relaunches of
that same fixed SHA, run id, topology and shard may retry only the identities
whose latest attempts remain invalid.

The later returned run `p46e25609` is also not evaluation or data-washing
evidence. It ran on **256 chips**, not 64: Qwen3-4B-Instruct-2507 at DP32 x
TP8. Its artifact contains exactly four tasks x N16 = 64 unique identities and
1,102 nonempty action/observation steps; all sampled logprobs are null. Those
structural facts prove durable reward-only trajectory capture only.

The terminal histogram is 59 `SUCCEEDED`, four
`MAX_CONTEXT_LIMIT_REACHED`, and one `MODEL_TIMEOUT`, with total reward zero.
The old evaluator stopped after the first 64-attempt wave because it treated
all five signed budget terminals as retryable invalid records, then emitted
`P46_EVAL_PHYSICAL_INCOMPLETE pending_valid_samples=5` and exited nonzero.
Context, max-step, model-timeout, and whole-trajectory budget terminals are now
completed unsolved evaluation outcomes under the fixed wall-clock contract;
retrying them would resample a failed identity and bias N16. Model timeout is
explicitly labeled `validity_reason=completed_model_timeout`. `ENV_TIMEOUT`,
`REWARD_TIMEOUT`, `FAILED`, malformed structure, and known harness failures
remain invalid and retryable; a harness failure overrides terminal status.

More importantly, the full JSONL shows the Q4 model repeatedly used the hybrid
tag form `<parameter=command=view>` instead of
`<parameter=command>view</parameter>`. The pinned R2E parser accepted the text
as a parameter named `command=view`, causing the CLI to receive
`--command=view`. Across the shard there are 347 `unrecognized arguments`
observations, 363 file-editor usage errors, 172 `/parameter` shell errors and
40 missing-required-argument errors. Every one of the 64 trajectories contains
at least one recognizable leaked parameter tag. Therefore none of this shard
may be classified or promoted, even where the terminal status says
`SUCCEEDED`.

The correction canonicalizes only the observed R2E tool dialect before the
pinned parser, keeps the raw model response in the artifact, records the
canonical executed action, verifies the pinned file-editor positional contract
during R2E installation, and marks any surviving adapter signature invalid
with `validity_reason=r2egym_action_parameter_adapter`. The trajectory schema
is `canon.p46.deepswe-eval.trajectory.v4`. Because source SHA is fingerprinted,
the first published fixed run uses a new run id and reruns all 64 `l0/p0`
identities; never resume or reclassify `p46e25609` in place.

## Workload matrix

| Family | Model and purpose | Signed work | Hard batch boundary |
|---|---|---|---:|
| `q4-debug` | Qwen3-4B-Instruct-2507 end-to-end training debug | 16K response, B4 x G4, exactly 3 optimizer updates | 3600 s |
| `q4-clean-eval` | Qwen3-4B-Instruct-2507 clean-data evaluation | 16K response, logical 32 tasks x N16; physical 4 tasks x N16 | 3600 s per physical shard |
| `q32-train` | Qwen3-32B full training | 16K response, B8 x G8, exactly 1000 optimizer updates | 5400 s |

Both training families keep AdamW state on TPU
(`CANON_OPT_STATE_RESIDENT=1`, `CANON_P30_OPT_STATE_OFFLOAD=0`, and
`--no-optimizer-offload`). There is no automatic host-offload fallback.

Training uses separated rollout and trainer roles:

| Allocation | Per-role mesh | Per-role devices | Local trajectories | Global M |
|---|---|---:|---:|---:|
| 64 chips | DP4 x TP8 | 32 | Q4: 4; Q32: 16 | 1024 |
| 128 chips | DP8 x TP8 | 64 | Q4: 2; Q32: not admitted | 2048 |
| 256 chips | DP16 x TP8 | 128 | Q4: not admitted; Q32: 4 | 4096 |

Evaluation has no trainer role and uses every visible device: DP8 x TP8 on 64
chips or DP16 x TP8 on 128 chips. Its semantic batch is still exactly four
tasks x 16 samples = 64 trajectories. Prefer 64 chips for evaluation when both
allocations are available; the 128-chip form exists so the same workload can
run when only that slice is schedulable.

## True reward-only evaluation candidate

`q4-clean-eval` has one configuration source:

```text
CANON_P46_EVALUATION_MODE=reward_only
```

That mode derives no sampled or prompt logprob request, no host logprob
extraction, no rescore, no trainer, no alignment and no optimizer. A caller
that supplies reward-only while enabling any trainer/alignment/logprob or
optimizer switch is rejected before the profile can overwrite the
contradiction. In vLLM, integer zero still requests a logprob; the only off
request is:

```text
SamplingParams.logprobs=None
SamplingParams.prompt_logprobs=None
```

Trajectory logprob fields are absent or `null`, never numeric `0.0`. Every
configuration, trajectory, task report and summary carries both:

```text
trajectory_mode=reward_only_no_logprobs
sampled_by=stock@<exact-source-sha>
```

It also carries `sampling_rng_mode=engine_global_sequential`. The TPU/JAX vLLM
backend rejects per-request seeds. Engine seed 42 drives an ordered RNG split
stream; `sample_nonce` is only a stable task/sample artifact identity. Do not
claim that each pair is independently replayable. The L2 one-host diagnostic
can restore the exact idle engine RNG snapshot before its two arms, but target
evaluation does not rewind the engine between requests.

Promotion is layered:

1. **L1 mechanical, mandatory:** exact mode/config, `None/None`, extraction
   bypass, legal schema, complete non-logprob fields, provenance, durable
   artifact and cleanup evidence.
2. **L2 token stream, diagnostic:** exact identity is preferred. A clean suffix
   divergence from one sample boundary is recorded as
   `LAW1_SUFFIX_DIVERGENCE` and does not block reward-only evaluation.
3. **L3 target statistics, mandatory for default promotion:** the same N16
   task/sample identities in observer and reward-only arms, exact paired
   McNemar/binomial PASS, plus valid trajectories/hour and Kubernetes cleanup
   comparison on 64 chips.

The direct-attached development gate passed Qwen3-4B DP1 x TP4 with one pinned
clean R2E Docker task:

```text
P46_REWARD_ONLY_ONEHOST_PASS l1=PASS l2=IDENTICAL_OBSERVER
report=/mnt/disks/tunix-data/deepswe-reward-only-evidence/reward-only-onehost-20260813T061510Z-696010/report.json
report_sha256=db3305413817ffe5c4d0085098475a12753cea6b698e15e4263b0c7d0835ba7c
```

It proved a real rollout, parser and `search` tool action, final reward 0, a
valid trajectory, null logprobs and zero residual Docker containers. The
median two-token diagnostic calls were 0.0330 s with logprobs versus 0.0310 s
reward-only; sampler payloads were 117 versus 70 bytes. This proves the flag
really changes request/extraction and payload. It does not prove a useful
cluster throughput gain, Kubernetes cleanup, TP8, DP8/DP32 or L3.

After publication, reproduce the local gate only on a direct-attached four-chip
v5p host with the pinned local model, dataset and R2E checkout:

```bash
bash canon-zero-tim/tests/p46_deepswe_profiles/run_onehost_reward_only_v5p.sh
```

The isolated smoke is one turn, 256 generated tokens per call and 512 total so
model, parser, tool, final reward and cleanup fit cheaply. Those limits do not
change the production evaluator, which remains 16K and 50 turns. Prefix cache
also remains off in the signed profile; do not combine optimizations until
their own gate is reviewed.

## Immutable data and sampling contract

All three families start from the reviewed clean whitelist:

```text
canon-zero-tim/clean_data/final_filter_result/task_report_good_qwen3_128_retry_20260713_090141.jsonl
sha256=2f95c2e6df3526f68bd3eed3ab9aece7077ef85c74251c77f7b3474b0b307ed7
rows=1851
```

It joins exactly against `R2E-Gym/R2E-Gym-Subset`, split `train`, revision
`2e8108ff942f24fcb5686badfaf7f9a8808566d5`, with 4578 source rows. The
evaluator uses Qwen3-4B-Instruct-2507, temperature 1.0, top-p 1.0, top-k 0,
prefix cache off, engine seed 42 and a stable identity nonce for every
`(docker_image, sample_index)`. The nonce is not a per-request sampling seed.

The evaluation report classifies a task only after exactly 16 valid samples:

- `partial`: Q4 solved between 1 and 15 samples; written to `q4_learnable`;
- `all_fail`: Q4 solved 0 of 16; kept in the separate Q32 hard tier;
- `all_pass`: Q4 solved 16 of 16; reported, not used as a Q4 learning tier;
- `broken` or `incomplete`: never promoted as curriculum data.

`q32_candidates` is the union of `partial` and `all_fail`. These reports are
advisory artifacts. Neither report silently replaces the 1851-row training
whitelist. Using one in a later training run requires a new reviewed manifest
with its exact SHA-256.

## Evaluation lifecycle and artifacts

One logical report covers 32 tasks x 16 samples. Full-campaign mode executes it
as eight physical waves inside one resident-runtime JobSet, each four tasks x
16 samples with concurrency 64. The final logical shard contains 27 tasks and
its last wave is three tasks x N16 = 48. The full 1851-task campaign is 58
logical shards and 463 physical waves in one resumable JobSet.
It requires 29,616 valid task/sample identities. The evaluator keeps a
16,384-token total response budget, at most 50 environment/model steps and a
3600-second deadline for every physical wave. Sixty-four is only the normal
physical-shard size, not the evaluation stopping condition.

Every trajectory attempt is appended and fsynced before another result is
accepted. Records contain the complete redacted trajectory, exact task/sample
identity, consecutive `attempt_index`, seed, status, reward, elapsed time,
source SHA, data revision, whitelist digest, client-image digest, topology,
model and sampling fingerprint. Only a valid record completes an identity.
Resume accepts only an identical fingerprint, permits the next consecutive
attempt after an invalid record, and rejects nonconsecutive attempts or any
attempt after the first valid result. A full campaign additionally binds an
explicit lowercase `resume_tag` to one immutable `resume_contract.json` and
holds a kernel-released exclusive lease, so two launches cannot write the same
campaign concurrently. A timed-out wave returns nonzero after preserving every
complete attempt; reapply the original manifest or use a new launch run id with
the same resume tag, topology, harness SHA and sampling-source SHA to retry only
missing identities.
An in-flight trajectory with no complete fsynced row restarts from its
beginning; token-level continuation is not supported. Genuine
infrastructure failures retry in-process only within that wave's shared
3600-second wall-clock budget.

Each resumed launch checks out the original manifest SHA even if the branch has
advanced, after proving that SHA remains in the fetched branch ancestry. Setup
state is isolated by launch run id. Before generating new trajectories, the
exclusive owner removes and confirms only R2E sandbox pods labelled with the
same resume tag. Each launch writes a new immutable attempt log, so an old
timeout remains available but cannot poison the current postflight grep.

A legacy config-v3/trajectory-v5 campaign is never read from its live output
directory. After that JobSet reaches a natural terminal state and no producer
pod remains, an operator may copy its trajectory tree once into
`<resume-root>/imports/<legacy-run-id>/trajectories/` and seal the copy with
`SHA256SUMS`. Under the new campaign lease, the importer verifies every file
digest, exact derived v5 fingerprint, ordered clean task, sample nonce,
attempt sequence, outcome, reward-only logprob absence, and per-logical-shard
provenance. It then emits immutable v6 rows plus an import receipt before TPU
initialization. The old sampler SHA remains `sampled_by`; the new checkout SHA
is recorded separately as `harness_commit`. Directly copying v5 rows into
`outputs/trajectories/`, importing a live directory, or changing any signed
model/data/sampling/topology field is forbidden and fails closed.

Reports are immutable and digest-bearing. Concurrent final shard writers are
accepted only when their bytes are identical; any content drift is a hard
failure. Durable output is under the mounted disk at:

```text
/mnt/disks/linchai_data/deepswe_eval/<resume-tag>/outputs/resume_contract.json
/mnt/disks/linchai_data/deepswe_eval/<resume-tag>/outputs/resume_lease.json
/mnt/disks/linchai_data/deepswe_eval/<resume-tag>/imports/<legacy-run-id>/
/mnt/disks/linchai_data/deepswe_eval/<resume-tag>/outputs/imports/*.receipt.json
/mnt/disks/linchai_data/deepswe_eval/<resume-tag>/outputs/trajectories/
/mnt/disks/linchai_data/deepswe_eval/<resume-tag>/outputs/census/
/mnt/disks/linchai_data/deepswe_eval/<resume-tag>/outputs/reports/
/mnt/disks/linchai_data/deepswe_eval/<resume-tag>/logs/campaign.attempt-*.log
```

The useful terminal markers are:

```text
P46_EVAL_SUBSHARD_PASS ...
P46_EVAL_LOGICAL_REPORT_PASS ... summary_sha256=<64 hex>
P46_EVAL_PHYSICAL_INCOMPLETE ... pending_valid_samples=<positive integer>
P46_EVAL_CAMPAIGN_PASS tasks=1851 n_sample=16 valid_trajectories=29616 ...
```

The first marker proves only that one physical shard contains exactly 64 valid
task/sample identities; the attempt count may exceed 64 after valid retries.
The second proves one complete exact-N logical report. The third is a failed,
resumable physical gate and must never be interpreted as PASS. Only the fourth
proves all 58 logical reports were merged into the digest-bearing secondary
evaluation output. None proves training, Qwen3-32B quality, or production
readiness.

Return full trajectories, not only the head log. On the coordinator after a
physical attempt finishes, build a compact return package without modifying
the JSONL files:

```bash
RUN_ROOT="/mnt/disks/linchai_data/deepswe_eval/$RUN_ID"
TRAJ_DIR="$RUN_ROOT/outputs/trajectories"
test -d "$TRAJ_DIR"
find "$TRAJ_DIR" -type f -name '*.jsonl' -print | sort
find "$TRAJ_DIR" -type f -name '*.jsonl' -exec sha256sum {} + | \
  sort -k2 > "$RUN_ROOT/trajectory.SHA256SUMS"
find "$TRAJ_DIR" -type f -name '*.jsonl' -exec wc -l {} + \
  > "$RUN_ROOT/trajectory.WC"
tar -C "$RUN_ROOT" -czf "$RUN_ROOT/trajectory-return-${RUN_ID}.tar.gz" \
  outputs/trajectories logs
sha256sum "$RUN_ROOT/trajectory-return-${RUN_ID}.tar.gz"
```

For a completed logical report, include `outputs/reports` in the archive as
well. Return the absolute JSONL/archive paths, the two inventory files, line
counts and SHA-256 values. Do not commit a large trajectory archive to git
unless the operator explicitly asks; preserving it on the mounted disk and
returning its exact path/digest is sufficient.

## Fetch and verify before rendering

From the remote execution checkout:

```bash
git fetch origin yuxzhang/canon-zero-tim
git switch --detach origin/yuxzhang/canon-zero-tim
SOURCE_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git ls-remote origin refs/heads/yuxzhang/canon-zero-tim | awk '{print $1}')"
test "$SOURCE_SHA" = "$REMOTE_SHA"
test -z "$(git status --porcelain)"
bash canon-zero-tim/tests/p46_deepswe_profiles/run_cpu.sh
```

Require `P46_DEEPSWE_PROFILES_CPU_PASS`. Also run the pinned-image gates named
in the P34/P44 runbooks with the actual registry-digest client image. Do not
print or modify `HF_TOKEN`, `WANDB_API_KEY`, or `.env`.

For evaluation execution after the `p46e25609` action-adapter correction, the
marker must be exactly `P46_DEEPSWE_PROFILES_CPU_PASS cases=77`, and this
source audit must
pass before rendering:

```bash
rg -n 'attempt_index|P46_EVAL_PHYSICAL_INCOMPLETE|physical_pending|validity_reason' \
  examples/deepswe/deepswe_eval_artifacts.py \
  examples/deepswe/eval_deepswe.py
rg -n 'canonicalize_r2egym_action|DEEPSWE.R2E_ACTION_COMPAT' \
  examples/deepswe/r2egym_action_compat.py \
  examples/deepswe/swe_agent.py
test -f examples/deepswe/finalize_deepswe_eval.py
```

If any marker is absent, the detached SHA still has the false-positive resume
semantics. Stop without applying a JobSet.

Before rendering, confirm the published checkout actually contains the
returned-run repair:

```bash
rg -n 'training_data_sharding_axis|DEEPSWE.DATA_SHARDING' \
  examples/deepswe/train_deepswe_nb.py
python3 -m unittest \
  canon-zero-tim/tests/p34_deepswe/test_script_contract.py \
  canon-zero-tim/tests/p44_deepswe_qwen4b_parity/test_integration_contract.py
```

The source must derive `training_data_sharding_axis` from
`train_axis_names[0]`; a hard-coded production `("fsdp",)` is stale and must
not launch.

## Render one JobSet

Set operator-specific inputs without embedding credential values:

```bash
TOPOLOGY=64
BASE=canon-zero-tim/cluster/jobset-64chip.yaml
RUN_ID=p46q4d01
RESUME_TAG=p46q4wash01
CPU_NODEPOOL=deepswe-cpu-pool
TPU_NODEPOOL=mlperf-v5p-64-np-0
MODEL_PVC=haoyugao-cpu-np-pvc
```

For Q4 on 128 chips, set `TOPOLOGY=128`, use
`canon-zero-tim/cluster/jobset-256cluster-64chip.yaml` only as the structural
Pathways template, and select the `4x4x8` worker node pool. The renderer
rewrites the resource manager, worker topology and worker count to 32; verify
those rendered fields before apply. Q4 explicitly rejects topology 256.

For Q32 on 256 chips, reset `TOPOLOGY=256`, use the same structural template,
and select the `4x8x8` worker node pool. Q32 explicitly rejects topology 128.

Render Qwen3-4B three-update debug:

```bash
python3 canon-zero-tim/cluster/render_p46_deepswe_profiles.py \
  --base "$BASE" \
  --output "/tmp/p46-q4-debug-${TOPOLOGY}.yaml" \
  --workload q4-debug \
  --topology "$TOPOLOGY" \
  --source-commit "$SOURCE_SHA" \
  --source-branch yuxzhang/canon-zero-tim \
  --client-image "$CLIENT_IMAGE_DIGEST" \
  --run-id "$RUN_ID" \
  --cpu-nodepool "$CPU_NODEPOOL" \
  --worker-nodepool "$TPU_NODEPOOL" \
  --model-pvc "$MODEL_PVC"
```

Render the production Qwen3-4B strict washing campaign. Under P46.7 this is
used after census PASS. Choose the campaign resume tag once and keep it
unchanged; run id names
this Kubernetes attempt and may change on a later resume. Do not also pass
shard indices:

```bash
python3 canon-zero-tim/cluster/render_p46_deepswe_profiles.py \
  --base "$BASE" \
  --output "/tmp/p46-eval-campaign-${TOPOLOGY}-${RUN_ID}.yaml" \
  --workload q4-clean-eval \
  --topology "$TOPOLOGY" \
  --source-commit "$SOURCE_SHA" \
  --source-branch yuxzhang/canon-zero-tim \
  --client-image "$CLIENT_IMAGE_DIGEST" \
  --run-id "$RUN_ID" \
  --resume-tag "$RESUME_TAG" \
  --cpu-nodepool "$CPU_NODEPOOL" \
  --worker-nodepool "$TPU_NODEPOOL" \
  --model-pvc "$MODEL_PVC" \
  --full-campaign
```

The renderer writes `CANON_P46_FULL_CAMPAIGN=1`,
`CANON_P46_RESUME_TAG=$RESUME_TAG`, launch-isolated setup state, one shared
artifact root and immutable per-attempt logs. On 128 chips it must render
`4x4x8`, 32 workers, and an evaluation mesh of DP16 x TP8.

## Preserve and adopt the running `p46e12805` campaign

Do not stop the existing `p46e12805` JobSet. It is a legacy-v5 producer and
must keep writing only its current root:

```text
/mnt/disks/linchai_data/deepswe_eval/p46e12805/
```

Wait for it to terminate naturally. Before copying anything, archive its final
JobSet status/events and prove that its coordinator and sandbox producer pods
are no longer running. A live JobSet, a terminating pod, or an output file that
is still growing means **wait**. Never point `--legacy-import-id` at this live
root.

After terminal-state proof, create a new, write-once snapshot under a fresh
resume tag. These commands copy; they do not move or delete the old evidence:

```bash
LEGACY_RUN_ID=p46e12805
LEGACY_ROOT=/mnt/disks/linchai_data/deepswe_eval/$LEGACY_RUN_ID
RESUME_TAG=p46q4wash01
RESUME_ROOT=/mnt/disks/linchai_data/deepswe_eval/$RESUME_TAG
SNAPSHOT_ROOT=$RESUME_ROOT/imports/$LEGACY_RUN_ID

test -d "$LEGACY_ROOT/outputs/trajectories"
test ! -e "$SNAPSHOT_ROOT"
test -z "$(find "$LEGACY_ROOT/outputs/trajectories" -type l -print -quit)"
install -d "$SNAPSHOT_ROOT/trajectories"
cp -a "$LEGACY_ROOT/outputs/trajectories/." "$SNAPSHOT_ROOT/trajectories/"
(
  cd "$SNAPSHOT_ROOT"
  find trajectories -type f -name '*.jsonl' -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 -r sha256sum > SHA256SUMS.tmp
  test -s SHA256SUMS.tmp
  mv SHA256SUMS.tmp SHA256SUMS
)
chmod -R a-w "$SNAPSHOT_ROOT"
```

Use a fresh resume tag with no existing target trajectory files. The first
resume-capable launch pins the new harness commit separately from the old
sampling lineage and performs the import before model initialization:

```bash
HARNESS_SHA=c3a960acdc94173440144559bb95f1de36d31537
SAMPLING_SOURCE_SHA=18d5d2ac1603a26a221af9d5fc430b084ec002df
RUN_ID=p46r01a0

python3 canon-zero-tim/cluster/render_p46_deepswe_profiles.py \
  --base "$BASE" \
  --output "/tmp/p46-eval-adopt-${TOPOLOGY}-${RUN_ID}.yaml" \
  --workload q4-clean-eval \
  --topology "$TOPOLOGY" \
  --source-commit "$HARNESS_SHA" \
  --sampling-source-commit "$SAMPLING_SOURCE_SHA" \
  --source-branch yuxzhang/canon-zero-tim \
  --client-image "$CLIENT_IMAGE_DIGEST" \
  --run-id "$RUN_ID" \
  --resume-tag "$RESUME_TAG" \
  --legacy-import-id "$LEGACY_RUN_ID" \
  --cpu-nodepool "$CPU_NODEPOOL" \
  --worker-nodepool "$TPU_NODEPOOL" \
  --model-pvc "$MODEL_PVC" \
  --full-campaign
```

Required pre-TPU marker:

```text
[P46.RESUME] LEGACY_IMPORT_PASS import_id=p46e12805 records=<n> valid_records=<n> manifest_sha256=<sha256> receipt=<absolute-path>
```

The exact receipt is
`<resume-root>/outputs/imports/p46e12805.receipt.json`. If fingerprint,
manifest, task order, identity, provenance, or attempt validation fails, stop;
do not edit either the old trajectories or the receipt. Subsequent relaunches
omit `--legacy-import-id` but retain the same `HARNESS_SHA`,
`SAMPLING_SOURCE_SHA`, resume tag, topology, client image, and all evaluation
fields. The evaluator will skip imported valid identities and retry only
missing or policy-invalid ones.

Render one Qwen3-4B physical shard only for a separately requested diagnostic,
not for the P46.6 production washing launch:

```bash
python3 canon-zero-tim/cluster/render_p46_deepswe_profiles.py \
  --base "$BASE" \
  --output "/tmp/p46-eval-${TOPOLOGY}-l0-p0.yaml" \
  --workload q4-clean-eval \
  --topology "$TOPOLOGY" \
  --source-commit "$SOURCE_SHA" \
  --source-branch yuxzhang/canon-zero-tim \
  --client-image "$CLIENT_IMAGE_DIGEST" \
  --run-id "$RUN_ID" \
  --cpu-nodepool "$CPU_NODEPOOL" \
  --worker-nodepool "$TPU_NODEPOOL" \
  --model-pvc "$MODEL_PVC" \
  --logical-shard-index 0 \
  --physical-shard-index 0
```

Render the two validation-only 64-chip parity arms. They are one clean task x
N16 each and write into separate directories under the same run id:

```bash
test "$TOPOLOGY" = 64
for EVAL_MODE in logprob_observer reward_only; do
  python3 canon-zero-tim/cluster/render_p46_deepswe_profiles.py \
    --base "$BASE" \
    --output "/tmp/p46-parity-${EVAL_MODE}-64.yaml" \
    --workload q4-clean-eval \
    --topology 64 \
    --source-commit "$SOURCE_SHA" \
    --source-branch yuxzhang/canon-zero-tim \
    --client-image "$CLIENT_IMAGE_DIGEST" \
    --run-id "$RUN_ID" \
    --cpu-nodepool "$CPU_NODEPOOL" \
    --worker-nodepool "$TPU_NODEPOOL" \
    --model-pvc "$MODEL_PVC" \
    --logical-shard-index 0 \
    --physical-shard-index 0 \
    --evaluation-mode "$EVAL_MODE" \
    --parity-canary
done
```

The renderer rejects `logprob_observer` without `--parity-canary`, rejects the
canary on 128 chips, and rejects evaluation-only controls on either training
family. Apply each arm only with explicit launch approval. After both arms
return, obtain their JobSet wall times from Kubernetes rather than summing
per-trajectory latency, and build the promotion report:

```bash
PARITY_ROOT="/mnt/disks/linchai_data/deepswe_eval/$RUN_ID/parity"
python3 examples/deepswe/deepswe_reward_only_parity.py \
  --observer-jsonl \
    "$PARITY_ROOT"/logprob_observer/outputs/trajectories/*.jsonl \
  --reward-only-jsonl \
    "$PARITY_ROOT"/reward_only/outputs/trajectories/*.jsonl \
  --observer-wall-secs "$OBSERVER_WALL_SECS" \
  --reward-only-wall-secs "$REWARD_ONLY_WALL_SECS" \
  --output "$PARITY_ROOT/l3-report.json"
```

The classifier requires exactly the same 16 valid identities, one task, one
`sampled_by=stock@<SHA>`, sampled-token logprobs in every observer trajectory,
no numeric logprob in reward-only artifacts, and paired statistical PASS. It
also reports valid trajectories/hour for both arms.

Render Qwen3-32B full training:

```bash
python3 canon-zero-tim/cluster/render_p46_deepswe_profiles.py \
  --base "$BASE" \
  --output "/tmp/p46-q32-train-${TOPOLOGY}.yaml" \
  --workload q32-train \
  --topology "$TOPOLOGY" \
  --source-commit "$SOURCE_SHA" \
  --source-branch yuxzhang/canon-zero-tim \
  --client-image "$CLIENT_IMAGE_DIGEST" \
  --run-id "$RUN_ID" \
  --cpu-nodepool "$CPU_NODEPOOL" \
  --worker-nodepool "$TPU_NODEPOOL" \
  --model-pvc "$MODEL_PVC"
```

The renderer refuses to overwrite an existing output, rejects floating images,
bad source SHAs, topology drift, unreviewed whitelist input, empty evaluation
shards, and any signed command-field change. A remote agent may render and
server-side dry-run these files after pulling the publication; applying them
still requires the operator's explicit launch approval.

## Promotion order and claim ceiling

P46.6 operator decision: do not repeat the standalone parity/l0-p0/three-update
sequence before washing. After P46.6 is published and separately approved for
cluster launch, render the single `--full-campaign` JobSet above. Monitor each
bounded wave and inspect trajectories during the run; do not wait for the end
to discover malformed artifacts. On interruption, preserve logs/events,
verify cleanup, and resume with the same tag and source contract. The numbered
legacy
sequence below remains historical rationale and claim ceilings; it is not the
current launch order.

The full JobSet must finish with both:

```text
P46_EVAL_CAMPAIGN_PASS tasks=1851 n_sample=16 valid_trajectories=29616 logical_shards=58 ...
[P46.EVAL.POSTFLIGHT] PASS
```

Final outputs are written directly under
`outputs/campaign/p46-campaign.{q4_learnable,q32_candidates,all_pass,all_fail}.jsonl`.
`q4_learnable` is exactly reward counts 1/16 through 15/16. Q32 remains a
separate later launch after these manifests and digests are reviewed.

The remote agent must advance one gate at a time. Q4 has first-class 64/128
variants; Q32 has first-class 64/256 variants. Use whichever admitted
allocation is available. Prefer 64 only when both admitted allocations are
simultaneously available because it is cheaper, not because it is a
prerequisite. Keep one topology for a given resumable evaluation tag because
topology is part of its fingerprint.

1. Before any full evaluation shard, require a clean published P46.5 SHA and
   complete the 64-chip paired N16 L3 canary. Compare identical task/sample
   identities with `classify_l3_paired_solve_rate`, require its exact paired
   verdict `PASS`, compare valid trajectories/hour, and prove all R2E pods were
   deleted in both arms. The current direct one-host PASS supplies L1/L2 only.
   Use only the validation-only manifests rendered above from the same SHA; do
   not use a historical solve rate as the control.
2. Run one `q4-clean-eval` physical shard at logical index 0 and physical
   index 0 on the available topology. The 64-chip form is DP8 x TP8; the
   128-chip form is DP16 x TP8. Both still evaluate exactly four tasks x N16
   with concurrency 64 and a one-hour boundary. Require
   `P46_EVAL_SUBSHARD_PASS` and
   `[P46.EVAL.POSTFLIGHT] PASS`, exactly 64 unique valid
   `(task, sample_index)` identities, full redacted conversations, finite
   rewards and proof that every R2E pod was deleted. Durable invalid attempts
   may make the total record count exceed 64; they must have consecutive
   attempt indices before the selected valid result. A
   `P46_EVAL_PHYSICAL_INCOMPLETE` marker, timeout or invalid-only identity
   preserves resumable evidence but is not a PASS.
3. Manually inspect at least one successful and one failed trajectory from the
   persistent JSONL. Confirm that assistant actions alternate with real R2E
   observations, statuses agree with terminal events, and reward 1.0 is used
   only for a valid solved trajectory. Summary-only JSONL is insufficient.
4. Run `q4-debug` on the available topology for exactly three updates. The
   64-chip form splits into DP4 x TP8 rollout/trainer roles; the 128-chip form
   splits into DP8 x TP8 roles. Both retain B4 x G4, 16 trajectories and the
   one-hour shared batch boundary. Require the `dp` data-axis marker once,
   three `P44.LOGPS_BATCH` markers, three trajectory files and digests, three
   batch-metrics rows, finite nonzero gradient activity, train steps
   `0->1->2->3`, exactly three commits, device-resident optimizer state, at
   least 8 GiB classifier-observed HBM margin, and a P44 classifier JSON whose
   `verdict` is `PASS`.
5. Complete all 58 logical N16 evaluation reports through all 463 resumable
   physical JobSets. This is the required data-washing completion gate, not an
   optional extension of the first 64-trajectory smoke. Logical indices 0-56
   use physical indices 0-7; logical 57 uses physical indices 0-6, with 48
   valid identities in final l57/p6. Run one physical JobSet at a time, retry
   the same index after `P46_EVAL_PHYSICAL_INCOMPLETE`, and never classify a
   task from a partial N16 sample set. Require 29,616 valid identities and all
   58 digest-bearing logical reports before declaring evaluation complete.
   Then run the fail-closed campaign finalizer:

   ```bash
   RUN_ROOT="/mnt/disks/linchai_data/deepswe_eval/$RUN_ID"
   python3 examples/deepswe/finalize_deepswe_eval.py \
     --summary-json "$RUN_ROOT"/outputs/reports/*.summary.json \
     --output-dir "$RUN_ROOT/outputs/campaign"
   ```

   Require `P46_EVAL_CAMPAIGN_PASS tasks=1851 n_sample=16
   valid_trajectories=29616 logical_shards=58`. Archive
   `outputs/trajectories`, `outputs/reports`, `outputs/campaign` and `logs`,
   and return the global summary/candidate digests. Any missing/duplicate task,
   digest mismatch, cross-shard contract drift, non-exact N16 report, broken
   task or incomplete task makes finalization fail closed.
6. Only after steps 4 and 5 pass, launch `q32-train` against the original clean
   1851-row whitelist. Require
   the same `dp` data-axis marker before rollout, 16K response, B8 x G8, at most
   64 concurrent sandboxes, a 5400-second shared rollout-batch deadline, and
   TPU-resident optimizer state. Start at update 0 and let the signed profile
   run to 1000 updates; do not edit the rendered YAML to create an ad hoc
   one-update or context variant.

If Q4 evaluation exposes a cleanup leak or malformed trajectory, stop before
training. If Q4 three-update times out, OOMs, has a nonfinite value, lacks
gradient activity, fails optimizer placement/transaction checks, or fails its
classifier, stop before Q32. Do not switch to host optimizer, increase a
deadline, resample a zero-signal batch, inject reward, or loosen a gate inside
the same attempt.

For every attempted gate, return the exact source SHA, image digest, rendered
YAML plus SHA-256, run directory, full head/worker/R2E logs, JobSet events,
trajectory/report SHA-256 values, classifier JSON, optimizer placement, HBM
evidence and the first fatal traceback. Applying any JobSet still requires the
operator's explicit launch approval.

A 64-chip PASS proves the DP4 Q4/Q32 training carrier or DP8 Q4 evaluation
carrier that actually ran. A 128-chip Q4 PASS proves its DP8 training or DP16
evaluation carrier. A 256-chip PASS is admitted only for Q32 and proves its
DP16 training carrier. Results are functionally comparable because model, data, sampler,
loss, optimizer, batch and deadline semantics are identical, but they are not
bitwise or performance-equivalent across DP sizes. Local CPU gates and rendered
YAML prove no TPU, Pathways, R2E, HBM, convergence, or zero-TIM claim.
