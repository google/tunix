# P44.12 — bounded Qwen3-4B three-update defaults

- Status: published in `e1b4009394c49ea015919bda0cfdb97c12c221b5`;
  local frozen-image PASS; 64/256 target NOT RUN
- Development base:
  `6905ca7c8551eeb8be772c40213e57e91bcfb0a7`

## Objective

Use Qwen3-4B as the fast end-to-end DeepSWE debug job before the real
Qwen3-32B campaign. The 64- and 256-device forms must do the same work and
terminate each update-zero-or-later rollout batch within one hour. A successful
rollout inspection may go directly to three updates; a separate one-update
allocation is not required.

## Signed recipe

- Qwen3-4B, TP8, separated rollout/trainer roles;
- 4 prompts x 4 generations = 16 trajectories;
- 4096 prompt, 4096 response, five turns;
- temperature 1.0, top-k disabled, top-p 1.0;
- exactly three optimizer updates, TPU-resident optimizer, no host fallback;
- reviewed R2E-Gym subset revision and the 1851-image clean whitelist;
- trajectory 3000 s, model turn 300 s, environment step 600 s, final reward
  600 s, cleanup 300 s, R2E pod active deadline 3300 s, complete prompt batch
  3600 s;
- durable redacted trajectories and solve/group metrics before backward;
- finite alignment differences warning-only; nonfinite, topology, weight,
  replica, optimizer, OOM, IFRT, artifact and cleanup failures remain fatal;
- zero-signal batches are recorded and follow the normal transaction; no
  resampling, signal injection or skip-commit.

The two physical allocations differ only in topology fields:

| Allocation | Role mesh | Local trajectories | Global M | Workers |
|---|---|---:|---:|---:|
| 64 | DP4 x TP8 | 4 | 1024 | 16 |
| 256 | DP16 x TP8 | 1 | 4096 | 64 |

## Lifecycle repair

P44 inherits the P39.5 request-abort, one-trajectory clock, final-reward,
cleanup, rollout-batch and R2E delete-confirm controls. A timed-out vLLM
request is aborted. The learner never submits an environment step with a
nonpositive timeout, never consumes a partial prompt batch, and never treats
unconfirmed sandbox deletion as success.

P44 artifact mode is selected by `CANON_P44_DEEPSWE_PARITY=1`.
`CANON_P34_TRAJECTORY_CAPTURE=0` is expected because artifact modes are
mutually exclusive; it does not disable P44 trajectory capture.

## Evidence

- P44 CPU: PASS, 41 cases, including identical normalized 64/256 recipes,
  clean-whitelist enforcement and timeout environment validation.
- P44 exact image: `P44_EXACT_IMAGE_CPU_PASS overlay=qwen4b`; seven targeted
  agentic tests (five deadline/cleanup controls) and one unfinished vLLM
  request-abort test pass.
- P34 static/trajectory/update, P39 and P43 regressions: PASS.
- Dummy P44 64 and 256 `three-update` manifests both rendered with PASS
  markers and contain `--max_steps=3`, the signed clean-data gate,
  `--no-optimizer-offload`, the 3600-second batch deadline and R2E 3300-second
  active deadline.

## Boundary and next action

No target, cloud resource, real Kubernetes pod, optimizer update, commit or
push was created. The local exact image has no TPU device and proves only
contract/control behavior.

After publication approval, detach at the exact remote operator HEAD and
render `three-update` for the available topology. Return all three compressed
trajectory batches, metrics, manifest, classifier, update records and the
deadline/cleanup log markers. A timeout is a failed/inconclusive attempt; do
not hand-edit the YAML or continue with a partial batch.
