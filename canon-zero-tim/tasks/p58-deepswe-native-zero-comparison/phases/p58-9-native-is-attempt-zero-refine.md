# P58.9 — Native token-IS and Attempt-0 refinement

## Status

`IMPLEMENTED LOCALLY / CONSTRUCTION PASS / NATIVE-IS TARGET SELECTED / PUBLICATION AUTHORIZED`

This phase is an unpublished refinement built in
`/home/yuxuan/code_rl_repro/worktrees/p58_is_zero_refine_0824` on exact
operator tip `614156c1ab067192ab65b2969543e23904f192be`. The user explicitly
authorized commit and push on 2026-08-24; replay over the latest operator tip,
validation, push, and exact remote readback remain. `main`, the earlier dirty
P58 worktree, Kubernetes, images, and TPU jobs were not modified.

## Objective

Maintain three disjoint P58 production recipes on the same frozen 128-chip
DeepSWE geometry:

1. `native-raw`: stock serving/trainer numerical programs and rollout logps;
2. `native-is`: the identical Native program plus registered token-level
   sampler/trainer truncated importance weights with threshold `2.0`;
3. `zero-hp`: strict Zero numerical program plus the existing default-off
   high-performance bundle.

All three retain Qwen3-4B-Instruct-2507, the exact 1,012-task list, B8 x G16,
16K response, 50 turns, rollout DP8 x TP8, trainer DP8 x TP8, one-hour batch
deadline, TPU-resident optimizer, no prefix cache, no group filter, no flat
group resampling, and a 1,000-commit full horizon.

## Executable selector contract

| Recipe | Renderer selector | Disable tuple | Runtime recipe |
|---|---|---|---|
| Native raw | `--arm native` | `1:1` | `sampler_is=None`, old logps are rollout logps |
| Native IS | `--arm native --sampler-is` | `0:0` | `sampler_is=token`, threshold `2.0`, old logps are trainer logps, TIS weights present |
| Zero HP | `--arm zero --high-performance` | `1:1` | strict Zero; sampler IS absent |

Partial tuples, IS on Zero, IS on Zero-HP, group clipping, or optimizer
offload fail closed.  Native-IS emits exactly one
`[P58.TIM_RECIPE] ... recipe=native-is` marker on the first effective batch;
Native-raw emits the corresponding `native-raw` marker.  Postflight requires
exactly one of the two markers for every Native run.

## Retry decision

P58 is restored to exact Attempt-0:

```yaml
failurePolicy:
  maxRestarts: 0
  restartStrategy: Recreate
```

A JobSet retry recreates the whole JobSet but reuses the persistent run root.
P58 does not yet have attempt-scoped roots and reports, so `maxRestarts: 3`
could mix evidence or stop immediately on existing files.  Five renderer-only
Pathways/IFRT/GRPC keepalive environment names were also removed: exact image
inspection found no consumer for them.  They must not be reintroduced without
a code consumer plus a bounded failure/recovery test.

## Validation checkpoint

Host results on 2026-08-24 UTC:

- renderer: 20/20 PASS;
- profile: 7/7 PASS;
- sampler recipe: 7/7 PASS;
- stock prompt observer: 6/6 PASS;
- Python compilation, Bash syntax, and `git diff --check`: PASS;
- host environment-contract collection: `INCONCLUSIVE`, because this shell
  lacks `metrax`; no assertion ran before the import boundary;
- complete dependency-bearing pinned-image gate: PASS in image
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  with terminal marker `P58_EXACT_IMAGE_CPU_PASS loss_oracle=1
  weighted_accumulation=1 compact_filter=1 durable_journal=1
  paired_renderer=1 alignment_policy=1 stock_observer=1 onehost_xprof=1
  zero_hp_full=1 apc=1 p59_tp4_tp8=2 p59_real_shim=4 p57_wandb=1
  regressions=1`.

Required next gates before any launch:

1. inspect the three rendered YAMLs and their SHA-256 values;
2. separately approve publication and read back the exact remote SHA;
3. launch Native-IS first only if that is the user-selected experiment;
4. treat any missing recipe marker, wrong tuple, Zero alignment difference,
   corrupted journal, optimizer mismatch, OOM, or IFRT failure as blocking.

Host and pinned-image gates prove construction only.  No one-host TPU,
DP8 x TP8 Pathways, 1,000-commit, convergence, or performance claim exists.

## 2026-08-24 execution decision

The operator reports a sharp training-reward drop in the currently running
Native/no-IS campaign and judges it collapsed. The onset update is not
established and must not be labeled as a fixed optimizer step. The returned
run evidence is not yet available in this worktree, so this is a
reported target observation rather than a local root-cause/classifier verdict.
It is sufficient to change execution order:

- stop and archive the exact Native-raw JobSet;
- mark the run failed/collapsed and never resume its optimizer checkpoint;
- retire Native raw from the launch queue;
- promote Native+IS as the next and only P58 training launch;
- initialize Native+IS from the original frozen base checkpoint under a fresh
  run id, run root, W&B run, and checkpoint directory.

Stopping the exact Native-raw JobSet is authorized by this decision after its
identity and full reward-curve evidence are preserved. Native+IS launch
remains blocked until publication and exact remote readback complete. After
readback and separate launch approval, render `--stage full --arm native
--sampler-is`, verify the `0:0` tuple and `recipe=native-is` marker contract,
then apply only that fresh YAML.
