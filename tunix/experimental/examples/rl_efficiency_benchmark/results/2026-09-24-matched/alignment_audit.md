# Aligned 2+2 benchmark audit

This paired run completed one warmup step and 20 measured steps,
with eight trajectories per train/log-prob microbatch on both stacks.
Agentic uses one prompt group with eight
generations; dist sets both microbatch sizes to eight trajectories. Each
step contains 64 prompt groups and 512 trajectories, and each stack made
64 training calls and one optimizer update per step. Both use
the same two-chip trainer/actor and separate two-chip rollout allocation,
model files, dataset, precision, sampling configuration, and synchronized
`stages` timing mode. Both processes exited with code zero. All 12 recorded
parity checks passed, including observed `[8, 2048]` training and log-prob tensor shapes,
64 calls per step, 512 trajectories per step, and 2,097,152 padded
token slots per step. See
[`stages-comparison.json`](stages-comparison.json) for the checks and
[`stages-agentic/manifest.json`](stages-agentic/manifest.json) and
[`stages-dist/manifest.json`](stages-dist/manifest.json) for the matching
source, dataset, and model configuration.

The previous attempt at 16 trajectories per microbatch is excluded: the
dist trainer failed on its first forward/backward call with
`RESOURCE_EXHAUSTED`, trying to reserve 75.81 GiB where 56.77 GiB was
available. The agentic 16-trajectory run completed, but no matched speed
comparison exists for that size. The unsuccessful 16-trajectory run is not
included in this result snapshot.

The stacks still use distinct local versus RPC and weight-sync paths.
At `micro_groups=1`, agentic converts each completed group to a training
example in its producer path, while dist assembles scored groups in its
orchestrator; preprocessing and log-prob work can therefore overlap the
consumer differently. These implementation paths are part of the measured
systems and remain distinct despite equal train/log-prob tensor shapes.
Online sampling can yield different tokens, lengths, turns, rewards, and
subsequent weights. Agentic fuses optimizer update into the final jitted
microstep; dist has a separate update RPC, so no symmetric optimizer-only
time is available. One sequential pair with 20 measured steps cannot
establish run-to-run confidence intervals.
