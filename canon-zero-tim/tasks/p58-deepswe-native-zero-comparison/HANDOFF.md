# P58 DeepSWE native-first training handoff

## Current checkpoint

P58 was developed in isolated worktree
`/home/yuxuan/code_rl_repro/worktrees/p58_deepswe_native_zero_0821` and approved
for publication to `yuxzhang/canon-zero-tim`. The implementation commit is
`c5bdc9d993dfaf1a6956335609fbf259f9ed95f7`; its first post-push readback
matched exactly with ahead/behind `0/0`. Always obtain the runnable
40-character source SHA by fetching and reading back that branch after push;
the latest branch may contain this later publication-evidence-only commit. Do
not infer the runnable SHA from the historical base or a mutable tag.

The user changed the execution order: waive the optional one-host sanity and
run only the native 128-chip three-update canary first. Zero is not optimized
enough for launch and is explicitly deferred. No image publication,
Kubernetes render output, JobSet apply, model download, or TPU run was made by
the publishing agent.

Native attempts `p58c01` and `p58c02` are bootstrap `INCONCLUSIVE` results.
P58c01 failed in `00_env.sh`; its published fix preserves native
`CANON_P32_DP_REDUCTION_ADMITTED=0`, exports three unrelated FrozenLake zeros,
and passes the renderer-to-real-`00_env.sh` regression. The fix implementation commit
`acd3136267214b367a6755d0ba28d80e883d6753` was pushed and its first remote
readback matched exactly with ahead/behind `0/0`. Fetch again and use the
final operator-branch SHA because this publication note is a later docs commit.

P58c02 then initialized Pathways and stopped before importing the model: direct
file execution of `/app/examples/deepswe/canonical_entrypoint.py` did not put
`/app` on `sys.path`, so its package-qualified `examples.deepswe` target could
not be found. The local fix derives the repository root from `__file__`, adds
it before the package import, and changes native stock preflight to exercise
the identical direct-file entrypoint. The exact command now exits zero from
`/tmp` in the pinned image, and the complete exact-image gate passes. These
changes were published as `82d82f72a7220d945737d95f6266b5b7e2cfe706`;
the first post-push readback matched exactly with ahead/behind `0/0`. Fetch the
final operator tip because this publication checkpoint advances it once more.

Never modify or push `main`. The eventual publication target is
`yuxzhang/canon-zero-tim` after reconciling the unrelated remote tip.

## What was implemented

- additive P58 DP8 x TP8 per-role workload/profile and a `4x4x8` paired
  renderer for `native|zero` and `three-update|full`;
- frozen Qwen3-4B-Instruct-2507 B8 x G16 recipe on the 1,012-task clean list;
- explicit 16,384 `sequence-mean-token-scale` norm and effective-row
  denominator matching the pinned DeepSWE quality-fix compact-filter path;
- denominator-weighted eight-way gradient accumulation for the stock trainer;
- matching global denominator behavior in the canonical segmented path;
- all-filtered no-commit for both paths, with no resampling;
- durable full-trajectory P58 journal, separate `batch_index` and
  `optimizer_step`, restart continuity/digest verification, per-batch solve and
  signal metrics, and W&B forwarding;
- native stock-engine verification and absence checks for the complete
  canonical numerical bundle;
- native finite A-B warning boundary with B-C strict, and zero all-boundary
  strictness;
- native stock optimizer transaction receipts plus zero explicit fixed-tree
  transaction receipts;
- P58 fail-closed postflight classifier and automatic invocation from
  `90_run.sh`; and
- negative/regression controls for P34/P44 and the shared trainer/loss paths.

The exact run instructions and artifact interpretation are in
`canon-zero-tim/cluster/P58_DEEPSWE_TIM_RUNBOOK.md`.

## Validated locally

Pinned local image ID:

```text
sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a
```

Terminal marker:

```text
P58_EXACT_IMAGE_CPU_PASS loss_oracle=1 weighted_accumulation=1 compact_filter=1 durable_journal=1 paired_renderer=1 alignment_policy=1 regressions=1
```

The gate covers the P58 loss oracle, unequal-effective-row gradients, real
trainer accumulation, stock/canonical all-filter discard, journal resume,
native-dose/zero-exact classifier negatives, both renderer arms/stages,
environment resolution, the full alignment suite, and relevant P34/P44
regressions.

One earlier attempt to run the complete legacy P34 static wrapper reached its
final device-probe test and timed out because this host has no TPU. The nine
preceding P34 suites passed. That device-probe result is `INCONCLUSIVE` and is
not represented as TPU evidence; the final P58 image gate runs the directly
relevant P34 contract/environment/renderer regressions instead.

No one-host real Qwen/R2E rollout and no 128-chip target canary has run. A
fresh inventory found Qwen3-4B weights but no `libtpu.so`, so this host cannot
run the requested direct-attached v5p validation. The current claim is
implementation plus CPU/exact-image validation only.

## Next executor sequence — native only

1. Read `state.md`, `plan.md`, this handoff, the P58.4N phase file, and
   `cluster/P58_DEEPSWE_TIM_RUNBOOK.md` completely.
2. Fetch `yuxzhang/canon-zero-tim`, detach at its exact remote-tracking SHA,
   prove a second remote readback matches, and require a clean tree. Never use
   `main`.
3. Rerun syntax, `git diff --check`, the P58 renderer/profile/environment
   tests, and the pinned exact-image gate.
4. Publish or select a client image by immutable registry digest and verify the
   mounted Qwen3-4B-Instruct-2507 weights and frozen clean-list digest without
   printing credentials.
5. Render only `arm=native, stage=three-update` with fresh run-id `p58c03`;
   preserve its YAML and digest,
   run server-side dry-run, and apply only that JobSet under the user's
   native-first decision.
6. Require the native classifier JSON to say `PASS`, including finite nonzero
   A-B, exact B-C, exactly three commits, device optimizer, complete journal,
   cleanup, and checkpoint/transaction receipts. Package the run before any
   follow-up decision.
7. Do not render or apply zero. Do not start 1,000 updates merely because the
   native canary passes; return evidence for a new user decision.

Do not reuse either failed `p58c01` or `p58c02` YAML/run root. Neither contains
resumable trajectory state. They remain immutable failure evidence at
`evidence/p58c01/run.log` and `evidence/p58c02/run.log`.

## Important operational semantics

- `use_rollout_logps=true` is shared. Sampler-IS, TIS correction, group
  clip/filter, degenerate-group masking, and flat-group resampling are off.
- All-zero/all-one reward groups remain. They naturally produce zero RLOO
  advantage and are logged.
- Compact-filter statuses are not malformed trajectories. They remain in the
  full journal but have zero policy mask. Structural missing/duplicate/parser
  failures remain fatal.
- If an entire batch is compact-filtered, `batch_index` advances but
  `optimizer_step` and commit count do not. Relaunching into a complete
  journal continues at the next batch index; a partial/digest-mismatched
  journal stops fail-closed.
- The native arm is stock numerical training plus observation. It must not
  inherit `CANON_FIXED_AR`, `CANON_LOGPROB_M`, the canonical module, VJP2, or
  the excess-precision pin. The zero arm retains the complete bundle.
- `CANON_P34_TRAJECTORY_CAPTURE=0` is intentional. P58 uses its own full
  trajectory journal and does not enable the older large P34 alignment-tensor
  capture mode.
- Optimizer state is TPU device-resident in both arms. Host offload is a hard
  configuration error.

## Claim ceiling

A native 128-chip PASS proves only that the untreated Qwen3-4B clean-data
training path completed the signed three-update integration gate. It does not
estimate a native-versus-zero effect, prove zero-TIM, isolate one kernel,
reproduce DeepSWE-32B, prove packing, or establish 256-chip production
behavior. Native exact A-B is `NO_TREATMENT`; missing evidence or interrupted
execution is inconclusive.
