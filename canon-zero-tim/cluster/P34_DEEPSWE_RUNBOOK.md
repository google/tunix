# P34 DeepSWE Qwen3-32B DP16xTP8 operator runbook

Status: local package gates pass; the 4x8x8 target has not run. The DP16
processed-logprob contract accepts only global compact M256 and global padded
M4096. Both shard to the same local canonical M256 program; every other global
row count fails closed.

This runbook renders manifests only. Do not apply a manifest until the implementation branch is
committed, pushed, read back at the exact SHA, and the 256-device experiment is approved.

## Required operator inputs

- Exact 40-character source commit on `yuxzhang/canon-zero-tim`.
- Client image pinned by registry SHA-256 digest. Tags such as `:latest` are rejected.
- Gold whitelist on the mounted PVC, plus its lowercase SHA-256 digest.
- CPU and TPU node-pool names and the model/output PVC name.
- A new 1-16 character lowercase run id for every manifest.

The Kubernetes `HF_TOKEN` and `WANDB_API_KEY` secret references are inherited from the reviewed
base manifest. The renderer never accepts secret values and `00_env.sh` never writes either token
to the resolved environment file.

## Render one immutable stage

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

## Admission order

1. `backward-no-commit`: initializes topology/model/rollout, checks all four forward boundaries,
   computes each DP16 group twice, requires full-array gradient equality, and commits no state.
2. `one-update`: requires one fixed DP transaction per group and exactly one optimizer commit.
3. `three-update`: requires commits and synchronized rollout weights at steps 1, 2, and 3.
4. `full`: starts only after a separately reviewed promotion decision.

The first stage combines P34.5 forward evidence and P34.6 backward evidence in one allocation.
Raw forward markers remain useful if backward fails, but the stage classifier does not report PASS
unless the entire backward-no-commit contract completes.

## Fail-closed facts

- JobSet restart count, head backoff, worker backoff, and pod restart are all zero.
- The Pathways 4x8x8 slice is divided by physical coordinates into two host-complete 2x8x8 roles.
- Each role is logical DP16xTP8; parameters are replicated over DP and sharded only over TP.
- `CANON_LOGPROB_M=256`, `MIN_TOKEN_BUCKET=4096`, and `CANON_VJP2_MAX_SEQS=1` are distinct signed
  values. TPU inference scheduler limits are per DP rank: 4 sequences and 256 batched tokens.
  Under DP16 these become exactly 64 global requests and one global M4096 token bucket.
- FSDP, TIS, sampler importance correction, prefix caching, runtime dependency installation, and
  floating source/image/whitelist inputs are rejected.
- A missing evidence row is `INCONCLUSIVE`, never PASS.

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
P34_EXACT_IMAGE_CPU_PASS unit_cases=38 pallas_cases=1 contract_cases=5 scheduler_cases=1 overlay=qwen32b
```

## Rollback

Do not render or apply the P34 JobSet, or leave all P34 admission variables at zero. Existing P33
profiles and launch paths remain unchanged. Preserve every failed manifest, raw log, classifier
and digest record.
