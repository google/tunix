# P44.6 — target promotion ladder

- Status: active

## Inputs

- Exact remote head of `origin/yuxzhang/canon-zero-tim` containing repair
  implementation commit `5f0cf7e04b34932d8c9deb2463f3b205e3ad8b51`
  and the future P44.9 SwiGLU feature-padding publication commit recorded in
  `../HANDOFF.md`. Published head `e4ead609` does not yet satisfy this input.
- One exact 64-device `4x4x4` slice or 256-device `4x8x8` slice.
- Registry-digest client image, existing Qwen3-4B checkpoint, pinned R2E-Gym,
  and reviewed gold whitelist/digest.

## Execution

1. Fetch, read back, and detach at the exact repair publication SHA.
2. Run P44/P43/P39/P34 package and exact-image gates.
3. Render only `rollout-only` for the available topology and server-side
   dry-run it.
4. Launch only with operator approval; require exact device-inventory,
   Qwen3-4B `1216->1280` SwiGLU feature-padding PATHTRACE,
   trajectory-counted logprob, trajectory artifact, and batch-metric evidence.
5. Promote that topology independently to one-update and then three-update
   only after each classifier PASS.

## Exit gate

Both topologies independently classify rollout-only, one-update, and
three-update as PASS from the same functional recipe. A pass on one topology
does not waive any stage on the other.

## Result

Pending P44.9 publication, remote read-back, and target execution. The
pre-repair 256-device attempts `p44r02`, `p44r03`, and `p44r04` stopped before
a completed rollout-only stage and cannot be promoted or reclassified.
