# P58.17 — Decode-vs-prefill seam probe

Status: one-host diagnostic complete and exact DP8xTP8 mechanism discriminator
implemented/host-tested and published as
`b54bd81a26e418ef3ff32f34d25ae8d81d9ac3f9`; matching image not published;
target not run; no Kubernetes/TPU mutation.

## Trigger

Immutable `p58z07` proved the P58.16 loader-metadata repair, returned all 128
Step-0 trajectory slots, and then stopped at the strict pre-backward gate. On
379,496 admitted action tokens, `S_prefill_vs_T_old` was exact while
`S_decode_vs_S_prefill` differed in 32,952 elements / 71,797 serialized bytes.
The first mismatch was finite and small (`4.35257e-3`) at a turn-8 action-run
start after an environment observation; the reported maximum `11.87498` was a
later amplified value, not the first divergence.

All 1,024 bounded mismatch records join exactly to two durable trajectories
(artifact rows 49 and 62). At the reported positions, shift 0 has median
absolute delta `0.0040245`, while shifts -1 and +1 are about `0.4952` and
`0.4922`. A simple one-token displacement is therefore refuted. Both rows
belong to the same signed Pillow task now frozen in
`clean_data/p58_seam_probe/p58z07_group3_pillow.jsonl`.

## Deliverable

Provide a default-off, mutation-free DP1xTP4 Qwen3-4B Zero-HP carrier that:

- selects only the frozen Pillow task;
- uses real R2E rollout, G2, response 4,096, 16 turns, serial scheduling,
  prefix cache off, and continue-decode 8;
- retains the strict decode-vs-prefill pre-backward gate and durable full
  trajectory journal;
- never commits optimizer state;
- accepts either a finite RED reproduction or an exact TP4 result as a
  bounded diagnostic outcome, while malformed, non-finite, empty-action, or
  unjoinable evidence fails closed;
- creates `P58_SEAM_PROBE_RETURN.tar.gz` plus its SHA-256 for a remote executor
  to return without manually selecting files.

The public serving API does not force arbitrary sampled token IDs through the
incremental decode path. This carrier therefore does not claim fixed-token
replay. Historical artifact joining is exact; the new one-host rollout is a
same-task, same-program carrier whose sampled token stream may differ.

The exact-geometry follow-up is now a second default-off deliverable. The one
selector `CANON_P58_CHECKED_VMA_DIAGNOSTIC=off` is admitted only for the
Qwen3-4B Zero-HP/full 128-chip carrier. It preserves Qwen3-4B-Instruct-2507,
the 1,012-task clean whitelist, B8xG16, 16K response, 50 turns, fixed seed 42,
rollout DP8xTP8, trainer DP8xTP8, prefix cache off, fixed lm-head,
continue-decode 8, and the other Zero-HP serving fields. It atomically derives
checked VMA, its P66 compatibility alias, P67 scoping, first-update gate, and
P63 clip to zero. It writes all 128 real trajectories and A/B/C pre-alignment,
then exits with code 42 before fixed-head VJP, P59/P66 backward, or an
optimizer commit. Production Zero-HP behavior is unchanged when the selector
is absent.

## Exit gate

1. Python/shell syntax, focused classifier/selector tests, renderer -> profile
   -> authoritative reload -> Python contract, real `p58z07` artifact
   classification, flag registry, and diff hygiene pass.
2. One direct-attached v5p host returns a checksum-valid bundle with one real
   trajectory journal and `N_action > 0`. This gate is complete in `p58s17`.
3. `FINITE_RED_REPRODUCED` localizes the carrier to TP4 and begins local
   first-seam diagnosis. `EXACT_ON_THIS_CARRIER` records a TP4 non-reproduction
   and requires a separately authorized DP8xTP8 exact-geometry follow-up.
4. The exact-geometry selector target returns either
   `A_B_EXACT_WITH_CHECKED_VMA_OFF` or `A_B_RED_WITH_CHECKED_VMA_OFF`, with
   B-C exact, 128 durable rows, and zero VJP/backward/optimizer evidence.
5. Neither outcome certifies backward, optimizer correctness, Zero-TIM
   production readiness, or convergence.

No TPU/Kubernetes launch, image publication, commit, or push is authorized by
this phase file.

## Local checkpoint

- 70 focused host tests pass across classifier, selector/manifest, renderer,
  profile, sampler, paired XProf, Zero-HP classifier, sandbox capacity, and
  flag registry adjacency.
- Flag audit passes at declared/actual/unique `389/389/389` after registering
  `CANON_P58_CHECKED_VMA_DIAGNOSTIC`.
- Immutable P58z07 evidence classifies as
  `FINITE_RED_REPRODUCED` with `N_action=379496`, 32,952 differing elements,
  and 1,024/1,024 mismatch records joined.
- The real direct-attached carrier `p58s17` completed two `SUCCEEDED` R2E
  Pillow trajectories with `N_action=4,808`, zero compact-filter rows, and
  zero optimizer commits. Its automatic classifier returned diagnostic
  `PASS / FINITE_RED_REPRODUCED`.
- `S_decode_vs_S_prefill` differs at 2,488/4,808 elements with
  `max_abs=1.3662147521972656`. Its first mismatch is at action coordinate
  `[0,0]`, prompt/KV prefix 1,737, with `0.0` versus
  `-0.08071136474609375`. Shift-0 median absolute delta is `0.02804`, versus
  `0.18280/0.19583` for shifts -1/+1, again refuting a simple token offset.
- Unlike `p58z07`, this TP4 carrier also has a finite
  `S_prefill_vs_T_old` RED (988 elements). It therefore reproduces a finite
  seam failure but not the exact remote signature. It cannot decide whether
  P67's topology-shaped VMA predicate leaks only on the DP8xTP8 serving mesh.
- The accepted bundle is
  `/mnt/disks/tunix-data/deepswe-onehost-xprof/p58_zero-hp_p58s17_20260827t1045z/P58_SEAM_PROBE_RETURN.tar.gz`,
  SHA-256
  `6285b5d2e8958ee85bd4b4190beaa240c7239ad6d07165a0948d7ba7f2b32eee`.
- Two launch defects were repaired before `p58s17`: the trainer mesh now uses
  JAX topology-aware device ordering matching vLLM (`0,2,1,3`), and the
  Zero carrier overlays the generated canonical runner as a real private
  package instead of placing a flat shim directory on `PYTHONPATH`. The
  Qwen3/linear/embed/attention/RPA installer files remain excluded because
  their signed geometry is TP8, not TP4.
- Prompt-rescore now executes the canonical runner and emits all three
  `CANON_PROMPT_*` receipts. Alignment normalizes inactive `top_k=None` and
  `top_p=None` to `0` and `1.0`, matching the rescore call contract.
- The final runner additionally pins the one-row whitelist SHA-256
  `7294da90559ebace771b7bd3fd8be01de87e0ae9bcb7ae1e317dbe5a6ed0db9f`
  in the manifest and asks an executor to return only the tarball plus its
  adjacent checksum. That provenance hardening was added after the local
  development run and is covered by focused tests; it does not change the
  recorded `p58s17` numerical values.
- The complete dependency-bearing pinned-image gate passes after the local
  fixes and provenance hardening. Its terminal marker is
  `P58_EXACT_IMAGE_CPU_PASS ... onehost_xprof=1 zero_hp_full=1
  checked_vma_diagnostic=1 qwen4b_fixed_head=1 checked_vma=1
  vma_p59_only=1 ... disaggregated_trainer_mesh=4 ... regressions=1`; the full alignment suite
  passes 43/43 and the focused probe/one-host suites pass 11/11. This is CPU
  construction/regression evidence, not TP8 target evidence.

## Decision and next carrier

Do not launch another 1,000-update job to diagnose this pre-backward failure.
The next expensive arm is the implemented exact 128-chip disaggregated
DP8xTP8+DP8xTP8 selector and stops after Step-0 pre-alignment with zero
optimizer commits. Recovery to strict A-B/B-C `0/0` supports the P67/VMA leak
family; a remaining finite A-B RED with exact B-C rejects checked VMA as a
sufficient cause and promotes seam replay. Partial bundles, Native arms,
non-HP Zero, warning-only alignment, wrong stage/horizon, VJP/backward, or an
inherited `CANON_P32_WORKLOAD` fail closed. No executor may hand-edit the
rendered YAML or set subordinate flags directly.

After source publication and matching digest-pinned image publication are
separately approved, prepare the YAML without contacting Kubernetes:

```bash
export P58_EXPECT_SOURCE_SHA=<exact-published-40-char-sha>
bash canon-zero-tim/tasks/p58-deepswe-native-zero-comparison/scripts/prepare_p58_checked_vma_off_diagnostic.sh \
  <short-fresh-run-id> \
  <matching-digest-pinned-image> \
  <worker-pool-or-auto> \
  /tmp/p58-checked-vma-off.yaml
```

The wrapper requires a clean tree whose HEAD and
`origin/yuxzhang/canon-zero-tim` both equal `P58_EXPECT_SOURCE_SHA`; it renders
only and refuses an existing output. Server dry-run and apply remain separate
operator-authorized actions. Return the whole persistent run root, especially
`run.log`, `env.sh`, `weight_attestation.jsonl`, `pre_alignment.jsonl`,
`debug/run_manifest.json`, the Step-0 trajectory journal and batch metrics,
and `p58_checked_vma_off.classification.json`. `updates.jsonl` must be absent
or empty. Construction PASS remains `TARGET NOT RUN` until this real carrier
completes.
