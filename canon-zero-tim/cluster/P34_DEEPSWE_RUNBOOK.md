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

The optional P39 64-chip resident-optimizer pilot is not a prerequisite for a
256-chip run that retains pinned-host optimizer offload. When a complete 4x8x8
slice is available, the operator may defer that pilot and exercise the actual
DP16xTP8 production topology directly. This does not promote resident optimizer
state: Qwen3-32B must keep `CANON_P30_OPT_STATE_OFFLOAD=1` until a separate
capacity experiment passes its HBM gate.

## Required operator inputs

- Exact 40-character source commit on `yuxzhang/canon-zero-tim`.
- Client image pinned by registry SHA-256 digest. Tags such as `:latest` are rejected.
- Gold whitelist on the mounted PVC, plus its lowercase SHA-256 digest.
- CPU and TPU node-pool names and the model/output PVC name.
- A new 1-16 character lowercase run id for every manifest.
- A cluster-scoped `very-high` PriorityClass with value `1000` and
  `PreemptLowerPriority`; both Pathways head and worker Pods use it.

The Kubernetes `HF_TOKEN` and `WANDB_API_KEY` secret references are inherited from the reviewed
base manifest. The renderer never accepts secret values and `00_env.sh` never writes either token
to the resolved environment file.

## Render one immutable stage

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
  --output /tmp/p34-backward-no-commit.yaml \
  --source-commit "$SOURCE_SHA" \
  --source-branch yuxzhang/canon-zero-tim \
  --client-image "$CLIENT_IMAGE_DIGEST" \
  --run-id "$RUN_ID" \
  --stage backward-no-commit \
  --cpu-nodepool deepswe-cpu-pool \
  --worker-nodepool mlperf-v5p-256-np-0 \
  --model-pvc haoyugao-cpu-np-pvc \
  --whitelist /mnt/disks/linchai_data/deepswe/gold.jsonl \
  --whitelist-sha256 "$WHITELIST_SHA256"
```

Allowed stages are `backward-no-commit`, `one-update`, `three-update`, and `full`. Use a different
run id and output path for each one. `full` is exactly 1000 updates; the renderer does not accept
an arbitrary budget.

## Available stage modes

1. `backward-no-commit`: initializes topology/model/rollout, checks all four forward boundaries,
   computes each DP16 group twice, requires full-array gradient equality, and commits no state.
2. `one-update`: requires one fixed DP transaction per group and exactly one optimizer commit.
3. `three-update`: requires commits and synchronized rollout weights at steps 1, 2, and 3.
4. `full`: starts only after a separately reviewed promotion decision.

The list above is an evidence ladder, not a requirement to reserve four
separate slices. A resource-constrained convergence campaign may use one
reviewed `full` manifest because the same topology, cross-role weight,
pre-alignment, backward, optimizer-transaction, replica, IFRT, and W&B checks
run inside its first update and continue running afterward. The checked-in
production profile is currently strict: every finite A-B or B-C mismatch still
stops before backward. A convergence-first run that must continue through a
finite alignment residual therefore requires a separate, default-off,
reviewed DeepSWE warning-only admission before rendering `full`; do not edit
the resolved environment or rendered YAML by hand. Nonfinite values and every
non-alignment gate remain hard failures in that mode.

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
- `CANON_PRE_ALIGN_GATE=1` is mandatory. Every update must flush exactly one
  passing A-B/B-C record before backward. A red record stops before gradient
  computation and optimizer commit.
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
P34_EXACT_IMAGE_CPU_PASS unit_cases=55 pallas_cases=2 contract_cases=5 scheduler_cases=1 overlay=qwen32b
```

## Rollback

Do not render or apply the P34 JobSet, or leave all P34 admission variables at zero. Existing P33
profiles and launch paths remain unchanged. Preserve every failed manifest, raw log, classifier
and digest record.
