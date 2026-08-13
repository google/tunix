# P46.5 — true reward-only DeepSWE evaluation

- Status: active

## Finding

- Confirmed: P46 clean evaluation has no trainer, reference model, backward or
  optimizer, but its `VllmConfig.return_logprobs=False` path still sets both
  `SamplingParams.logprobs=0` and `prompt_logprobs=0`. In this vLLM API, zero
  is a request to return the sampled-token/prompt-token logprobs rather than an
  absence request. `VllmSampler.detokenize()` also invokes sampled-token
  logprob extraction unconditionally.
- Confirmed: the evaluation profile inherits alignment and processed-logprob
  environment switches from the training profile even though its trainer
  admissions are zero. They are not consumed by the standalone evaluator, but
  they make provenance ambiguous and allow future drift.
- Decision: `evaluation_mode=reward_only` is the only production configuration
  input. It derives no rollout/prompt logprobs, no rescore, no trainer, no
  alignment and no optimizer. Any contradictory inherited switch is rejected.

## Execution

1. Make the vLLM no-logprob request use `None/None` and skip host extraction.
2. Add fail-closed reward-only configuration and trajectory validation. Numeric
   logprob payloads, including `0.0`, are forbidden; absent or `null` is valid.
3. Persist `trajectory_mode=reward_only_no_logprobs` and
   `sampled_by=stock@<source-sha>` in configuration, trajectory, report and
   summary evidence.
4. Add layered gates:
   - L1 is mandatory: mode/config/schema/provenance/no-logprob/cleanup wiring.
   - L2 token identity is reported but non-blocking; a clean suffix divergence
     after one sampling point is a Law-1 observation, not a failure.
   - L3 is target-only: paired N16 task arms must have solve-rate differences
     consistent with binomial uncertainty, followed by trajectories/hour.
5. Provide a direct-attached one-host probe for the real request switch,
   per-call latency and artifact-size delta. Do not promote it to 64/256 or R2E
   throughput evidence.

## Exit gate

- Local: P46 CPU tests prove `None/None`, extraction bypass, contradiction
  rejection, schema/provenance and the L1/L2 classifier contract.
- One-host: a real v5p report records on/off request parameters, timing,
  artifact bytes and L2 identity/divergence without making L3 claims.
- Target: one 64-chip paired N16 shard supplies L3 and valid
  trajectories/hour before reward-only becomes the Q4 clean-evaluation
  default.

## Result

The reward-only implementation and local L1/L2 gates are published by
`a4d165e854cc4c2320d8120e89aed185eaf61465`, rebased on operator commit
`23bb2a3c1a77fa4037f3ec81b783e48d1af22951`. The implementation:

- sends `logprobs=None,prompt_logprobs=None` to vLLM and bypasses host
  extraction entirely;
- derives trainer/alignment/optimizer/logprob invariants from the single
  `evaluation_mode=reward_only` input and rejects contradictory caller input;
- never serializes numeric fake logprobs, and carries
  `trajectory_mode=reward_only_no_logprobs`, `sampled_by=stock@<SHA>` and
  `sampling_rng_mode=engine_global_sequential` through every artifact; and
- keeps the production 64/256 lifecycle, clean-data fingerprint, deadlines,
  durability and cleanup behavior unchanged.

CPU evidence is `P46_DEEPSWE_PROFILES_CPU_PASS cases=31`; the two direct
sampler tests for `None/None` and extraction bypass also pass. A real
direct-attached v5p-8 host then passed Qwen3-4B DP1 x TP4 L1/L2 with one pinned
clean R2E Docker task:

```text
P46_REWARD_ONLY_ONEHOST_PASS l1=PASS l2=IDENTICAL_OBSERVER
status=SUCCEEDED reward=0.0 steps=1 cleanup_new_containers=[]
```

The diagnostic arms restored the same engine RNG snapshot. Their median
two-token call times were 0.0330 s with sampled logprobs and 0.0310 s in
reward-only mode; serialized sampler payloads were 117 and 70 bytes. These
measurements prove request/extraction and payload differences, not a target
throughput improvement. TPU/JAX rejects per-request seeds, so `sample_nonce`
is an artifact identity only; production sampling uses engine seed 42 and its
ordered split stream.

Durable development evidence:

```text
/mnt/disks/tunix-data/deepswe-reward-only-evidence/reward-only-onehost-20260813T061510Z-696010/report.json
sha256=db3305413817ffe5c4d0085098475a12753cea6b698e15e4263b0c7d0835ba7c
/mnt/disks/tunix-data/deepswe-reward-only-evidence/reward-only-onehost-20260813T061510Z-696010/eval/onehost.trajectories.jsonl
sha256=2497cb614a92a888c34c4ec4b019d05a3e10d9024b61c7c9853861f725a1bfa8
```

The one-host run also found and repaired three stock-evaluation integration
faults without changing training: single-row semantic lists now receive only
the outer singleton batch dimension expected by `SWEEnv`; direct
`SamplerOutput` is adapted to `RolloutOutput`; and the isolated smoke caps one
turn at 256 tokens inside its 512-token total budget so parser, tool, final
reward and cleanup all execute. Production remains 16K and 50 turns.

P46.5 stays active. L3 paired N16 solve-rate consistency and
valid-trajectories/hour on one 64-chip small shard are not run and remain the
promotion gate. A validation-only renderer now emits exact 64-chip, one-task x
N16 `logprob_observer` and `reward_only` arms from the same SHA; observer mode
is rejected outside that canary. The L3 artifact gate requires the same 16
valid identities, numeric sampled logprobs only in the observer arm, one stock
sampler SHA, exact paired statistics, and measured valid trajectories/hour.
Both validation-only 64-chip manifests render from the local worktree with
distinct output roots and fingerprints. Development render digests are
`5815be34b17a5274e605e5b178f8c811f7ff6d4fdf1ef943fdd8c54400e9ac05`
for `logprob_observer` and
`03c03ade29ffea1a490a90a96eaa1a09df831ca75705baaad2b4cd45b3c3ef8a`
for `reward_only`; they are review evidence only, not launch artifacts.
The one-host PASS is development evidence from a dirty
worktree at the base SHA, not a clean publication claim.

## Returned 256-chip correction

Operator HEAD `63b092b001864e4e9a4822b4354a665bb00b1c6b` archives the first
256-chip target attempt, `p46e25608`, run from source
`bdc9681824743911d0691659604dec090dd42bc4`. Qwen3-4B reward-only DP32 x
TP8 initialized and l0/p0 attempted all 64 identities. The unique terminal
status audit is 62 `SUCCEEDED` and two `MODEL_TIMEOUT`; all 62 valid rewards
are zero. The old evaluator nevertheless emitted
`P46_EVAL_SUBSHARD_PASS ... pending_logical_tasks=30` and postflight returned
zero because any durable record, including an invalid attempt, completed the
resume identity. That PASS is revoked.

The local unpublished repair adds consecutive `attempt_index` provenance.
Invalid attempts remain immutable evidence but do not complete an identity;
the next retry is admitted only before the first valid result. Nonconsecutive
indices and attempts after a valid result fail closed. The physical evaluator
recomputes missing valid identities after collection and emits
`P46_EVAL_PHYSICAL_INCOMPLETE` with a nonzero exit until the exact shard count
is valid. Task reports and L3 statistics select the valid retry while retaining
attempt/invalid-attempt counts. A global finalizer additionally rejects
missing/duplicate logical shards or tasks, digest drift, cross-shard contract
changes and any non-exact N16 report before producing merged candidate
manifests. Local P46 evidence is now 33/33 PASS.

The fixed first target run must use a new run id and rerun all 64 l0/p0
identities because the source SHA/fingerprint changes. After that smoke passes,
the evaluation does not stop: complete all 1851 x N16 = 29,616 valid
trajectories through 58 logical reports and 463 sequential/resumable physical
JobSets. Each trajectory retains the 16,384-token total response budget and at
most 50 environment/model steps; each physical JobSet remains bounded at 3600
seconds. No washed candidate whitelist exists until all exact-N16 reports are
complete and separately reviewed.

Only the head log was returned in git. The persistent full trajectory location
for the old run is expected at
`/mnt/disks/linchai_data/deepswe_eval/p46e25608/outputs/trajectories/`, but it
was not archived here. The next remote return must include the JSONL absolute
paths, line counts, per-file SHA-256, compressed trajectory/log archive and
archive SHA-256; a head log alone cannot prove action/observation/tool-call
quality.
