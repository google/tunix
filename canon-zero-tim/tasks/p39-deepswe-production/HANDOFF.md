# P39 DeepSWE production handoff

Scope: this handoff covers only Qwen3-32B DeepSWE on the 4x8x8 target. For the
parallel GSM8K/FrozenLake workstream, read
`../p38-pathways-decode-prefill-carrier/HANDOFF.md`. Never use one
workstream's green result to promote the other.

Read `plan.md`, then `state.md`, then `log.md`.

## Current revision

- Latest operator-branch base used by this unpublished worktree:
  `6905ca7c8551eeb8be772c40213e57e91bcfb0a7`
- P39.5 bounded-lifecycle implementation: local changes only; no commit or
  publication SHA exists yet
- P39.4 publication revision: the exact 40-character HEAD after pulling
  `yuxzhang/canon-zero-tim`; do not substitute the base SHA listed above
- p34r02 mesh-admission repair commit:
  `562f55b077bdadbcfa160177715b0d8ca903f457`
- Hardening base: `5ee6dbfb5601cf1d1f864ccf6859764ba1f321fe`
- Implementation started from: `697a29ab4b27015297af8e3dbb37c49db3560445`
- Working branch: `codex/p46-deepswe-32b-full`
- Publish target, only after user approval: `yuxzhang/canon-zero-tim`
- Workload reference only: `yuxzhang/deepswe-quality-fix` at
  `023978b976dd6d94e7a42948c3f3a68e34d73744`

The operator source is never the workload-reference branch. After publication,
use the exact 40-character `git rev-parse HEAD` from a clean
`yuxzhang/canon-zero-tim` worktree.

The older P39.2 pilot and P34 gates are already published.  The operator
approved publication of the P39.4 direct full-run changes on 2026-08-12.
Always fetch `yuxzhang/canon-zero-tim`, verify the exact
40-character SHA from a clean worktree, and render only from the final reviewed
publication HEAD.  Do not launch from this local branch or substitute an older SHA.

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
7. P39.4 fixes the optimizer CLI boolean trap and renders
   `--no-optimizer-offload`; P34 full requires device-resident optimizer state
   and zero P30 host round trips.
8. P39.4 pins the R2E-Gym subset revision and the checked-in 1851-image clean
   whitelist.  Source, whitelist and retained row counts are exact gates.
9. Every full-run batch durably stores 64 real redacted trajectories plus
   solve/all-solved/all-failed/mixed/incomplete/effective-group metrics before
   backward.  Eight-of-eight, not four-of-eight, defines an all-solved group.
10. `effective_prompt_groups == 0` remains visible but does not resample,
    inject signal or skip the normal optimizer transaction.
11. The local p34r02 repair explicitly clears
    `CANON_EXPECT_MODEL_MESH_IDS`.  P34 must never inherit the legacy
    four-device `0,2,1,3` default into its 128-device Pathways rollout role;
    renderer validation and `00_env.sh` now enforce that before launch.
12. Target `p34r03` proved real Qwen3-32B rollout entry but exposed an
    unbounded lifecycle: 60 environment timeouts received negative remaining
    time, update zero did not produce a trajectory batch, and four vLLM
    requests were still active after more than four hours.
13. P39.5 gives vLLM requests a real abort path, shares one wall clock across
    reset/model/step/reward, bounds cleanup, confirms R2E pod deletion, and
    gives the complete prompt batch one watchdog.
14. The rendered defaults are now explicit: Qwen3-4B debug is B4/G4, 4096
    response, five turns, three updates and a 3600-second rollout batch on
    either 64 or 256 devices; Qwen3-32B full remains B8/G8, 32K response,
    50 turns, 1000 updates and a 5400-second rollout batch. Both use
    temperature 1.0, clean data, device optimizer and durable trajectories.

## Reproduce local validation

```bash
bash canon-zero-tim/tests/p34_deepswe/run_exact_image.sh
```

The required terminal marker is
`P34_EXACT_IMAGE_CPU_PASS unit_cases=55 alignment_cases=3 pallas_cases=2
contract_cases=5 scheduler_cases=1 overlay=qwen32b`. It passed in the pinned
image after the P39 changes. It must be rerun from the final publication SHA because shared
alignment and agentic-learner code changed after the original P39 gate. A
local pass does not promote target status.

Also run the Qwen3-4B exact-image gate from the final publication SHA. It must
end with `P44_EXACT_IMAGE_CPU_PASS overlay=qwen4b`; the gate includes the
request-abort and reset/model/reward/cleanup deadline controls.

## Current target decision

The operator has a complete 4x8x8 slice.  The P39 64-chip pilot is deferred,
not passed, and is not a prerequisite.  The selected 256-chip run starts with
device-resident optimizer state; Qwen3-32B HBM capacity remains unverified and
there is no automatic offload fallback.

The selected production geometry is one 4x8x8 slice split by the real training
process into two host-complete 128-device roles. Rollout and trainer are each
DP16xTP8. This is one Pathways JobSet and one client session, not two
independent clusters. The in-process split rejects any device count other than
256, any physical extent other than 4x8x8, overlapping roles, incomplete
coverage, or a host divided between roles.

The selected objective is one continuous `full` convergence campaign, exactly
1000 updates.  Separate one-update and three-update jobs are not prerequisites.
Finite A-B, B-C and downstream alignment residuals are warning-only and remain
fully recorded.  Do not hand-edit a rendered manifest.  Follow
`../../cluster/P34_DEEPSWE_RUNBOOK.md` exactly.

Attempt `p34r03` used source
`65b0cd0a84807f2308409d1867022407ae53f8fb` and passed the repaired dynamic
mesh, Qwen3-32B initialization, online W&B and clean data before entering real
rollout. Its complete returned log is
`../../debug_logs/p34_p34r03_deepswe_full.raw.log`, SHA-256
`426019f66f812e0bb80874cbcfb19fe183846b6565251bb5f043d505425dd2a1`.
It is not a successful rollout: update zero ran for more than four hours, 60
trajectories reported negative-remaining-time `ENV_TIMEOUT`, the final lines
still showed four active vLLM requests, and there is no completed trajectory
batch, backward or optimizer record. P39.5 repairs this locally; the target
retry has not run.

After publication, resolve the exact new remote HEAD. First render the
Qwen3-4B `three-update` debug recipe for the available 64/256 allocation and
inspect its trajectory/deadline/cleanup evidence. Then render Qwen3-32B
`full`. Do not launch from the unpublished local worktree and do not reuse the
p34r03 manifest.

## Stop conditions

- Any nonfinite, shape, topology, metadata, cross-role weight,
  optimizer-placement/transaction, replica, artifact, OOM, IFRT, or W&B
  failure.
- Finite alignment differences do not stop P34 full.  They must remain visible
  in the records and cannot promote a zero-TIM claim.
- A finite zero gradient or `effective_prompt_groups == 0` does not stop or
  skip commit.  Record it as a quality warning without injecting signal.
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
- Before apply, verify the model PVC, checked-in whitelist path and digest, worker image
  pull health, and relevant Kueue storage policy. An eviction or image-pull
  failure is infrastructure-inconclusive, not a numerical result.

## Rollback

Do not apply the rendered JobSet, or keep all P34 admissions at zero.  A host
offload retry is a separate reviewed configuration, not an in-process fallback.
Preserve every failed manifest and evidence directory.
