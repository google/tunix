# P34 DeepSWE Qwen3-32B DP16xTP8 operator runbook

Status: local package gates pass; the 4x8x8 target has not run. The DP16
processed-logprob contract accepts only global compact M256 and global padded
M4096. Both shard to the same local canonical M256 program; every other global
row count fails closed.

The production configuration and branch provenance are frozen in
`tasks/p39-deepswe-production/plan.md`. The workload behavior comes from
`yuxzhang/deepswe-quality-fix`; the P34 renderer replaces its FSDP-named
topology with DP16xTP8 replicated parameters and pins every algorithm field
that would otherwise depend on a Python default.

This runbook renders manifests only. Do not apply a manifest until the implementation branch is
committed, pushed, read back at the exact SHA, and the 256-device experiment is approved.

The optional P39 64-chip resident-optimizer pilot is not a prerequisite for the
256-chip run. When a complete 4x8x8 slice is available, the operator may defer
that pilot and exercise the actual DP16xTP8 production topology directly.

Operator decision 2026-08-12: launch one real `full` run directly.  The
production profile uses device-resident optimizer state
(`CANON_OPT_STATE_RESIDENT=1`, `CANON_P30_OPT_STATE_OFFLOAD=0`) and the command
uses the unambiguous `--no-optimizer-offload` flag.  There is no automatic host
fallback.  Qwen3-32B HBM margin at 32K response length remains UNVERIFIED; an
OOM is infrastructure-INCONCLUSIVE and a host-offload relaunch requires a new
reviewed manifest.  Watch `[P41.OPTIMIZER] placement=device-resident` and the
init/update HBM records.

Finite A-B, B-C and later alignment residuals are warning-only in `full` so
that the convergence run continues.  The raw residuals, exact hashes and
warning counts remain evidence.  Shape-invalid or nonfinite alignment,
topology, exact-weight, replica, optimizer transaction, artifact, OOM and IFRT
failures still stop the run.  The claim level is convergence-only, never
zero-TIM.

## Required operator inputs

- Exact 40-character source commit on `yuxzhang/canon-zero-tim`.
- Client image pinned by registry SHA-256 digest. Tags such as `:latest` are rejected.
- The checked-in clean whitelist must be present unchanged at
  `clean_data/final_filter_result/task_report_good_qwen3_128_retry_20260713_090141.jsonl`.
  Its only admitted SHA-256 is
  `2f95c2e6df3526f68bd3eed3ab9aece7077ef85c74251c77f7b3474b0b307ed7`.
- CPU and TPU node-pool names and the model/output PVC name.
- A new 1-16 character lowercase run id for every manifest.
- A cluster-scoped `very-high` PriorityClass with value `1000` and
  `PreemptLowerPriority`; both Pathways head and worker Pods use it.

The Kubernetes `HF_TOKEN` and `WANDB_API_KEY` secret references are inherited from the reviewed
base manifest. The renderer never accepts secret values and `00_env.sh` never writes either token
to the resolved environment file.

## Render the immutable full run

First perform the read-only priority preflight on the DeepSWE cluster:

```bash
kubectl get priorityclass very-high \
  -o jsonpath='{.metadata.name}{" value="}{.value}{" policy="}{.preemptionPolicy}{"\n"}'
```

The required result is exactly
`very-high value=1000 policy=PreemptLowerPriority`. Missing or different output
stops the launch; this runbook does not create or mutate cluster scheduling
policy.

```bash
python3 canon-zero-tim/cluster/render_p34_jobset.py \
  --base canon-zero-tim/cluster/jobset-256cluster-64chip.yaml \
  --output /tmp/p34-full.yaml \
  --source-commit "$SOURCE_SHA" \
  --source-branch yuxzhang/canon-zero-tim \
  --client-image "$CLIENT_IMAGE_DIGEST" \
  --run-id "$RUN_ID" \
  --stage full \
  --cpu-nodepool deepswe-cpu-pool \
  --worker-nodepool mlperf-v5p-256-np-0 \
  --model-pvc haoyugao-cpu-np-pvc \
  --whitelist clean_data/final_filter_result/task_report_good_qwen3_128_retry_20260713_090141.jsonl \
  --whitelist-sha256 2f95c2e6df3526f68bd3eed3ab9aece7077ef85c74251c77f7b3474b0b307ed7
```

`full` is exactly 1000 updates; the renderer does not accept an arbitrary
budget.  The renderer rejects any other full-run whitelist path or digest and
pins `R2E-Gym/R2E-Gym-Subset`, split `train`, revision
`2e8108ff942f24fcb5686badfaf7f9a8808566d5`, 4578 source rows, 1851 clean
whitelist rows, 1851 unique images and 1851 retained rows.

## Short diagnostic modes

1. `backward-no-commit`: initializes topology/model/rollout, checks all four forward boundaries,
   computes each DP16 group twice, requires full-array gradient equality, and commits no state.
2. `one-update`: requires one fixed DP transaction per group and exactly one optimizer commit.
3. `three-update`: requires commits and synchronized rollout weights at steps 1, 2, and 3.
4. `full`: the selected operator path for this campaign.

These short modes remain useful diagnostics but are not prerequisites for this
campaign.  Do not queue one-update and three-update allocations before the
reviewed full manifest.  The same topology, weight, alignment, backward,
optimizer, replica, IFRT and W&B checks execute from update zero onward.

## Durable trajectory and quality telemetry

Every full-run batch writes, before backward:

- `debug/batch-XXXXXX.trajectories.jsonl.gz`: 64 redacted raw trajectories,
  including group/pair identity, Docker image identity, complete conversation
  and tool observations, tokens/masks/old logprobs, status, reward and
  advantage;
- `debug/batch_metrics.jsonl`: solve ratio plus all-solved, all-failed, mixed,
  incomplete and effective prompt-group counts; and
- `debug/run_manifest.json`: exact source, model, topology, dataset, whitelist
  and schema identity.

`effective_prompt_groups == 0` and a finite zero gradient are quality
warnings.  They do not trigger resampling, signal injection or skip-commit.
Artifact write failure is fatal.  The same metrics are sent to online W&B
under `deepswe/*`.

During the run, inspect without mutating artifacts:

```bash
jq -c '{step,trajectory_solve_ratio,all_solved_prompt_groups,all_failed_prompt_groups,mixed_prompt_groups,incomplete_prompt_groups,effective_prompt_groups,status_histogram}' \
  "$RUN_ROOT/debug/batch_metrics.jsonl"
gzip -cd "$RUN_ROOT/debug/batch-000000.trajectories.jsonl.gz" | head -n 1 | jq .
```

The first stage combines P34.5 forward evidence and P34.6 backward evidence in one allocation.
Raw forward markers remain useful if backward fails, but the stage classifier does not report PASS
unless the entire backward-no-commit contract completes.

## Fail-closed facts

- Do not re-enable the retired Step 65 fresh-process JAX device probe. Its
  temporary client disconnected after discovery and could cancel the shared
  Pathways session, killing otherwise healthy workers. The production profile
  intentionally leaves `CANON_EXPECTED_SLICE_DEVICES` unset. The real training
  process performs the authoritative fail-closed admission instead:
  `split_4x8x8_role_devices` requires exactly 256 unique devices, physical
  extents `(4, 8, 8)`, two disjoint and exhaustive 128-device role halves, and
  no host split across roles before either mesh is constructed. On that check
  failing, archive the `jax-tpu`, `pathways-proxy`, and `pathways-rm` logs
  before deleting the JobSet, then check for incomplete registration or stale
  clients holding the slice.
- R2E-Gym is provisioned by `cluster/steps/35_install_r2egym.sh`: a pinned
  checkout (`CANON_R2EGYM_COMMIT`) with the vendored
  `patches/r2egym/r2egym.patch` applied, pip-installed in the pod together
  with `kubernetes`. The DeepSWE profile enables it; GSM8K/FrozenLake
  profiles skip the step entirely. Any drift -- wrong commit, missing patch,
  surviving source pins -- fails closed. The reference MLPerf launch cloned
  the floating upstream HEAD at runtime; the pin replaces that.
- Independently of provisioning, `swe_agent` imports without r2egym (parser
  fails closed at use with the exact remedy) and
  `apply_repoenv_kubernetes_poll_patch` logs a skip and returns an empty
  path. This is defense in depth for non-DeepSWE contexts, not the supply
  route.
- JobSet restart count, head backoff, worker backoff, and pod restart are all zero.
- Pathways head and worker Pods both retain `priorityClassName: very-high`; the
  renderer rejects missing or mismatched values. Priority does not replace the
  separate IFRT readiness/session admission gate.
- The Pathways 4x8x8 slice is divided by physical coordinates into two host-complete 2x8x8 roles.
- Each role is logical DP16xTP8; parameters are replicated over DP and sharded only over TP.
- `CANON_LOGPROB_M=256`, `MIN_TOKEN_BUCKET=4096`, and `CANON_VJP2_MAX_SEQS=1` are distinct signed
  values. TPU inference scheduler limits are per DP rank: 4 sequences and 256 batched tokens.
  Under DP16 these become exactly 64 global requests and one global M4096 token bucket.
- FSDP, TIS, sampler importance correction, prefix caching, runtime dependency installation, and
  floating source/image/whitelist inputs are rejected.
- `CANON_PRE_ALIGN_GATE=1` is mandatory. Every update flushes exactly one
  A-B/B-C record before backward. Finite residuals produce
  `PASS_WITH_ALIGNMENT_WARNINGS` and continue; invalid shapes, empty action
  sets and nonfinite values stop before gradient computation.
- A missing evidence row is `INCONCLUSIVE`, never PASS.
- Before every A/B/C comparison, all mapped trainer-anchor leaves must be
  bitwise equal to the live rollout-engine leaves. The run emits and persists
  exactly one `weight_attestation.jsonl` record per update; a missing,
  duplicate, or non-exact record stops promotion.

Before interpreting any numerical row, the target log must contain exactly:

```text
Prepared token paddings: [4096]
Precompile worker0 backbone --> {'num_tokens': 4096, 'num_reqs': 64}
```

Any additional prepared bucket, a different request capacity, or a runtime compile/cache miss for
a larger backbone shape is a contract failure. `max_num_batched_tokens=256` limits one scheduler
step per DP rank; it does not shorten the 4096-token prompt or 32768-token response because long
sequences use chunked prefill/decode steps.

Before using target resources, reproduce the contract in the pinned local image:

```bash
bash canon-zero-tim/tests/p34_deepswe/run_exact_image.sh
```

The required terminal marker is:

```text
P34_EXACT_IMAGE_CPU_PASS unit_cases=55 alignment_cases=3 pallas_cases=2 contract_cases=5 scheduler_cases=1 overlay=qwen32b
```

## Rollback

Do not render or apply the P34 JobSet, or leave all P34 admission variables at zero. Existing P33
profiles and launch paths remain unchanged. Preserve every failed manifest, raw log, classifier
and digest record.
