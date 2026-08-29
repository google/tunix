# V1 system-optimization full-workload runbook

This runbook renders manifests only. It does not launch TPU or Kubernetes
work. Every launch needs its own explicit approval, a fresh label/run ID, and
remote read-back of the published source and digest-pinned image.

## FrozenLake P45 and M15 full wave

From the physical repository root, after the approved source is committed and
the worktree is clean:

```bash
bash canon-zero-tim/tasks/v1-phase4-three-full-recipes/scripts/prepare_p67_frozenlake_two_full_wave.sh \
  <approved-40-sha> \
  <fresh-output-dir> \
  <fresh-campaign-root> \
  <fresh-p45-run-id> \
  <fresh-m15-run-id>
```

The preparation receipt must say
`V1_P67_FROZENLAKE_WAVE_READY ... manifests=2 ... launch=not-executed`.
Inspect `manifest-index.json` and both YAML files. The effective trainer env
must include the exact optimization tuple recorded in `state.md`, must retain
checked-VMA/P67/first-update protection, and must not contain
`CANON_DP_COLLECTIVE_REDUCE`.

## DeepSWE Qwen3-4B Zero-HP full

From the same clean, physical repository root:

```bash
bash canon-zero-tim/tasks/v1-system-optimization-workload-rollout/prepare_deepswe_zero_hp_full.sh \
  <approved-40-sha> \
  <registry/image@sha256:digest> \
  <fresh-output.yaml> \
  <fresh-run-id> \
  <worker-nodepool> \
  <model-pvc>
```

The preparation receipt must say
`V1_DEEPSWE_ZERO_HP_RFULL_READY ... launch=not-executed`. Inspect the YAML and
confirm the same exact tuple, the DeepSWE P67 selector, and collective-reducer
absence.

## Target receipts required after separately approved launches

- source SHA, image digest, unique run ID/label, physical launch path, and
  manifest SHA256;
- `CANON_P59_CHECKED_VMA=1`, `CANON_P67_P66_VMA_P59_ONLY=1`, and the unchanged
  checked-VMA protection receipt in the trainer log;
- complete first-update gate with finite gradients and no numerical red;
- per-step `p32_vag_reverse` and global wall receipts, plus per-group/per-chunk
  timing or a warm reverse XProf capture;
- FrozenLake and DeepSWE target performance remain unverified until those
  receipts exist. One-host GSM8K numbers cannot certify DP8xTP8 target gains.
