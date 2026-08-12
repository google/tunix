# P44.8 — optional direct-attached v5p Qwen3-4B one-host smoke

- Status: real TPU matmul construction gate passed; full trajectory E2E prerequisite-blocked

## Inventory

- Host Python JAX TPU initialization: unavailable because `libtpu.so` is absent.
- A privileged pinned-image container can access `/dev/vfio` and exposes
  exactly four direct-attached `TPU v5` devices.
- Local Qwen3-4B training checkpoints exist under `/mnt/disks/tunix-data`, but
  the reviewed Hugging Face cache contains tokenizer artifacts rather than a
  complete initial-weight snapshot suitable for the requested model-init E2E.
- R2E-Gym Python package: not importable on the host.
- Kubernetes: `kubectl` is absent and no readable user kubeconfig is present.
- Immutable dependency image: available and used for CPU, exact-image, and
  real four-device TPU matmul forward/VJP gates. It contains neither R2E-Gym
  nor Kubernetes access.

## Decision

The P44.10 target-shaped matmul forward/custom-VJP gate is PASS on four real
TPU v5 devices; see `p44-10-r05-matmul-padding.md`. Full DeepSWE one-host E2E
remains `BLOCKED_REAL_ENVIRONMENT`. Do not implement or claim a fake R2E
environment. No model download, cluster mutation, real trajectory, whole-model
backward, or optimizer update was performed.

This optional smoke does not block publishing the separately validated P44
64/256 Pathways repair. If a future session exposes exactly four direct TPU
devices plus existing Qwen3-4B weights and real R2E access, resume here with a
default-off DP1xTP4 colocated profile and preserve the documented claim
ceiling.

## 2026-08-12 inventory correction

The prerequisites were later found in different local roots: a complete
Qwen3-4B-Instruct-2507 Hugging Face snapshot, pinned R2E-Gym checkout, cached
`R2E-Gym-V1`, reviewed whitelist, and Docker daemon are available on the
direct-attached host. P44.11 therefore supersedes this blocker and records the
real rollout/backward result. The observations above remain the historical
reason the earlier attempt stopped and must not be cited as current inventory.
