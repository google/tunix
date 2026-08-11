# P39 DeepSWE production handoff

Scope: this handoff covers only Qwen3-32B DeepSWE on the 4x8x8 target. For the
parallel GSM8K/FrozenLake workstream, read
`../p38-pathways-decode-prefill-carrier/HANDOFF.md`. Never use one
workstream's green result to promote the other.

Read `plan.md`, then `state.md`, then `log.md`.

## Current revision

- Hardening base: `5ee6dbfb5601cf1d1f864ccf6859764ba1f321fe`
- Implementation started from: `697a29ab4b27015297af8e3dbb37c49db3560445`
- Working branch: `codex/p39-deepswe-production-contract-0810`
- Publish target, only after user approval: `yuxzhang/canon-zero-tim`
- Workload reference only: `yuxzhang/deepswe-quality-fix` at
  `023978b976dd6d94e7a42948c3f3a68e34d73744`

The operator source is never the workload-reference branch. After publication,
use the exact 40-character `git rev-parse HEAD` from a clean
`yuxzhang/canon-zero-tim` worktree.

The P39.2 pilot changes are currently workspace-local. The final publication
SHA is therefore intentionally absent from this handoff. Do not launch from
the working branch or substitute the stale hardening-base SHA.

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
5. The renderer emits the source label as an explicitly quoted YAML string, so
   numeric-exponent SHA prefixes such as `022893e2` cannot change type in a
   Kubernetes YAML 1.1 consumer.
6. Before every P34 A/B/C comparison, the learner bitwise-compares every
   mapped trainer-anchor leaf with the live rollout-engine leaf on device. The
   exact result is fsynced to `weight_attestation.jsonl`; one mismatching leaf
   stops before rescore, backward, or optimizer commit.

## Reproduce local validation

```bash
bash canon-zero-tim/tests/p34_deepswe/run_exact_image.sh
```

The required terminal marker remains the one in
`cluster/P34_DEEPSWE_RUNBOOK.md`. It passed in the pinned image after the P39
changes. It must be rerun from the final publication SHA because shared
alignment and agentic-learner code changed after the original P39 gate. A
local pass does not promote target status.

## First target command

The next target is the bounded 64-chip pilot specified in
`phases/p39-2-64chip-tp8-resident-pilot.md` and operated through
`../../cluster/P39_DEEPSWE_64CHIP_PILOT_RUNBOOK.md`. Its CPU gate and the P34
static regression gate pass locally; the target remains NOT RUN. It uses one
4x4x4 slice split into two 32-device roles, each DP4xTP8, and tests
device-resident optimizer state. This is a new systems contract; do not bypass
the existing DP16xTP8 preflight or relabel the pilot as P34 green.

After the pilot classifies resident capacity, render the 4x8x8 DP16xTP8 target
exactly as documented in `cluster/P34_DEEPSWE_RUNBOOK.md`. Use resident
optimizer only if the pilot leaves the pre-registered HBM margin; otherwise
retain pinned-host offload. The 256-chip target must still independently prove
DP16 collective and replica behavior.

## Stop conditions

- Any nonfinite, shape, topology, metadata, cross-role weight, gradient,
  optimizer-transaction, replica, HBM, IFRT, or W&B failure.
- Finite alignment differences remain warnings only in the explicitly admitted
  P39 64-chip pilot and must remain visible in evidence. They do not promote a
  zero-TIM claim.
- Missing, duplicate, or non-exact rollout/trainer weight attestation.
- Missing report, measurement count, proxy XLA flag, scheduler bucket,
  Pathways marker, W&B online marker or fixed reducer marker.
- Infrastructure disconnect: classify `INCONCLUSIVE`; do not read later rows.

## Target infrastructure facts

- The renderer rejects a floating client image; provide an image reference
  pinned by registry SHA-256 digest.
- The model and output PVC is mounted only into the `jax-tpu` head. Pathways
  workers do not each download a 32B checkpoint, but their server-image layers
  still require adequate node ephemeral storage.
- Before apply, verify the model PVC, whitelist path and digest, worker image
  pull health, and relevant Kueue storage policy. An eviction or image-pull
  failure is infrastructure-inconclusive, not a numerical result.

## Rollback

Do not apply the rendered JobSet or keep all P34 admissions at zero. Preserve
all failed evidence.
