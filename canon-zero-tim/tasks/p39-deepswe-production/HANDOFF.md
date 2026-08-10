# P39 DeepSWE production handoff

Read `plan.md`, then `state.md`, then `log.md`.

## Current revision

- Integration base: `0fe5f6609df06895d93cbf2e54cada22ad7f2697`
- Implementation started from: `697a29ab4b27015297af8e3dbb37c49db3560445`
- Working branch: `codex/p39-deepswe-production-contract-0810`
- Publish target, only after user approval: `yuxzhang/canon-zero-tim`

## What changed

1. The P34-only no-sampler/TIS policy is now an explicit alignment contract;
   neighboring unregistered workloads still reject `sampler_is=None`.
2. P34 renders `CANON_PRE_ALIGN_GATE=1` and a persistent
   `pre_alignment.jsonl`. The P34 classifier requires exactly one passing
   pre-alignment record per update before accepting the four-boundary gradient
   records.
3. The renderer command pins the DeepSWE algorithm and timeout fields that
   previously depended on defaults.
4. A CPU test feeds the rendered environment through the real `00_env.sh` and
   proves that one missing gate or sampler attestation is rejected.

## Reproduce local validation

```bash
bash canon-zero-tim/tests/p34_deepswe/run_exact_image.sh
```

The required terminal marker remains the one in
`cluster/P34_DEEPSWE_RUNBOOK.md`. It passed in the pinned image after the P39
changes. A local pass does not promote target status.

## First target command

Render `backward-no-commit` exactly as documented in
`cluster/P34_DEEPSWE_RUNBOOK.md`. Do not use `full` as a probe. The target run
must be Attempt 0, have zero retries, produce exactly one pre-alignment record,
four post-backward alignment records, deterministic repeated full gradients,
and zero optimizer commits.

## Stop conditions

- Any pre-backward A-B or B-C differing byte.
- Any C-old/C-current differing byte.
- Missing report, measurement count, proxy XLA flag, scheduler bucket,
  Pathways marker, W&B online marker or fixed reducer marker.
- Infrastructure disconnect: classify `INCONCLUSIVE`; do not read later rows.

## Rollback

Do not apply the rendered JobSet or keep all P34 admissions at zero. Preserve
all failed evidence.
