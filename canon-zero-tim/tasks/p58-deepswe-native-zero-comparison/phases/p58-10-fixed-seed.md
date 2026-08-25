# P58.10 — fixed dataset and rollout seed

## Status

`LOCAL IMPLEMENTATION / PINNED-IMAGE PASS / UNPUBLISHED / TARGET NOT RUN`

Built in `/home/yuxuan/code_rl_repro/worktrees/p58_fixed_seed_0824` on branch
`local/p58-fixed-seed-0824`, based exactly on operator tip
`687b2bd6d0815b5628af39e7adbf949e429e72ae`. The older P58 worktree contained
unrelated dirty P59/V1 work and was left untouched.

## Objective and contract

Make seed 42 an explicit common field of the P58 matched recipe rather than
relying on the notebook parser's default. Every Native-raw, Native+IS, and
Zero-HP render contains exactly one `--seed=42`. The training entry point
requires 42 for P58 and passes it to both the Hugging Face dataset shuffle and
`RolloutConfig.seed`; vLLM receives that value through its existing sampler
interface.

The runtime emits:

```text
[P58.SEED] PASS dataset_seed=42 rollout_seed=42 scope=config-level async_completion_order=not-claimed
```

W&B records the CLI seed, rollout seed, and scope. The durable run manifest
records `dataset_seed=42`, `rollout_seed=42`, and the same scope; P58 target
and one-host classifiers require the new provenance.

## Claim ceiling

This fixes the configured data ordering and sampling seed. It does not promise
bitwise-equal end-to-end trajectories between jobs because vLLM request
scheduling, R2E sandbox runtime, and asynchronous completion order remain
external sources of nondeterminism. No per-trajectory derived-seed protocol or
deterministic sandbox scheduler was added.

## Validation

- Python compilation and `git diff --check`: PASS.
- Focused renderer, sampler-recipe, and one-host tests: 33/33 PASS.
- Bare-host artifact/classifier imports: `INCONCLUSIVE` because this shell
  lacks `metrax`; no assertion executed past that dependency boundary.
- Complete pinned-image gate in
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`:
  PASS with terminal `P58_EXACT_IMAGE_CPU_PASS ... paired_renderer=1 ...
  onehost_xprof=1 zero_hp_full=1 ... regressions=1`.

No real one-host TPU rollout, Pathways target, optimizer commit, image
publication, Kubernetes mutation, commit, or push occurred.

## Next gate

Obtain explicit commit/push approval, publish only to
`yuxzhang/canon-zero-tim`, and read back the exact remote SHA. Then render the
fresh Native+IS full job and reject it unless the YAML has one `--seed=42` and
the runtime produces the signed seed marker and manifest fields.
