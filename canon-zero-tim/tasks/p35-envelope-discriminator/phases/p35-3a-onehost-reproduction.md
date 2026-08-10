# P35.3a one-host envelope reproduction

Status: completed; local carrier not reproduced

## Question

Does the production `S_prefill != T_old` boundary reproduce on the existing direct-attached
four-chip v5p host, or is the known carrier specific to the 64-chip Pathways program envelope?

## Frozen scope

- Existing `t1v-n-4a77ebd0-w-0` only; no TPU VM lifecycle change.
- Qwen3-1.7B, DP1xTP4, prompt/response caps 1024/256.
- Two prompts x eight generations = 16 trajectories, matching one DP16 rank's target-local
  trajectory count and the target outer-map depth.
- Canonical local M256, prefix cache disabled, the published P35.3 source, and the pinned
  `tunix_frozenlake_image:vllm-tpu0.25.0` engine overlay.
- `CANON_P35_ENVELOPE=1` and `CANON_P35_EXACT_REPLAY=1`; stop before backward. W&B is disabled,
  and no optimizer update or checkpoint is admitted.

## Interpretation

1. If A versus C is red and the full six-arm report completes, classify the first local red
   boundary using the unchanged fail-closed classifier.
2. If A versus C is exact, the target-only producer is expected to stop at its known-red
   reproduction guard. Record `LOCAL_NOT_REPRODUCED`; do not call it a P35.3 PASS because the six
   replay arms were not measured.
3. A setup, overlay, metadata or runtime failure is infrastructure/contract evidence, not a
   numerical verdict.
4. A local red or exact result does not replace the source-pinned 64-chip r29 Attempt 0.

## Exit gate

The raw log must attest four TPU devices, the published source SHA, canonical overlay byte
identity and runtime PATHTRACE. It must then either produce one complete P35/P35.3 report pair or
emit the exact known-red reproduction rejection after A and C were measured. Preserve the raw log,
reports if any, SHA-256 values, command and rollback.

## Result

Attempt r3 completed the DP1xTP4 production boundary through rollout and A/C measurement. The
sampler/trainer metric was numerically zero, and the P35 selector's actual element-bitwise scan
found no red action element. It therefore stopped at the pre-registered known-red reproduction
guard before B and before the six-arm replay. The corrected mechanical verdict is
`LOCAL_NOT_REPRODUCED`; this is not a P35.3 PASS and is not a Pathways target verdict.

The original wrapper counted the same exception twice because a Python traceback contains both
the source `raise` line and the terminal exception line. Its schema-v1 output is retained as
`INCONCLUSIVE`. The schema-v2 reclassification anchors only the terminal exception line, requires
clean postflight plus nonzero canonical PATHTRACE, and preserves the immutable raw SHA.

Reproduce the classification without TPU work:

```bash
python3 canon-zero-tim/tasks/p35-envelope-discriminator/scripts/classify_p35_onehost.py \
  --raw /mnt/disks/tunix-data/logp_probe_1host/p35_onehost_0809_dp1tp4_r3.raw.log \
  --output /mnt/disks/tunix-data/logp_probe_1host/p35_onehost_0809_dp1tp4_r3.result.v2.json \
  --source-commit cf4c12e4003199cd80c73603f8b54a0f80f49657
```

Artifacts:

- raw log SHA-256: `13f77d5b13110b995582089a7a0f40be85f04dcb0e50116ee5ba240070534af6`
- schema-v2 result SHA-256: `516c1ad9c7bc3a963c856e674421df236b30a5a71b637e204310ae63903c8908`
- observed canonical traces: fixed AR 168, fixed embed 1, canonical logprob M 1
- postflight: no C7/C8 violation
- excluded by design: backward, optimizer update, W&B mutation and any 64-chip claim

## Rollback

Leave both P35 environment switches unset and stop invoking the isolated one-host runner. The
experiment changes no production default, precision, weight, optimizer state, checkpoint, W&B
run or cloud resource.
