# P44.8 — optional direct-attached v5p Qwen3-4B one-host smoke

- Status: prerequisite-blocked; not run

## Inventory

- Host JAX TPU initialization: unavailable; `libtpu.so` is absent.
- Direct device nodes: no `/dev/vfio` or `/dev/accel*` entries are visible.
- Existing local Qwen3-4B checkpoint: not found under the reviewed local model
  roots, including `/mnt/disks/linchai_data/models/Qwen3-4B`.
- R2E-Gym Python package: not importable on the host.
- Kubernetes: `kubectl` is absent and no readable user kubeconfig is present.
- Immutable dependency image: available and used successfully for CPU gates,
  but it cannot manufacture missing accelerator devices or a real R2E
  environment.

## Decision

`BLOCKED_REAL_ENVIRONMENT`. Do not implement or claim a fake one-host E2E. No
model download, cluster mutation, rollout, forward, backward, or optimizer
update was performed.

This optional smoke does not block publishing the separately validated P44
64/256 Pathways repair. If a future session exposes exactly four direct TPU
devices plus existing Qwen3-4B weights and real R2E access, resume here with a
default-off DP1xTP4 colocated profile and preserve the documented claim
ceiling.
