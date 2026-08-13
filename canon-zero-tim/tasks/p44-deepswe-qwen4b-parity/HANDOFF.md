# P44 Qwen3-4B DeepSWE parity-debug handoff

The operator-facing source of truth is
`../../cluster/P44_DEEPSWE_QWEN4B_PARITY_RUNBOOK.md`.

## Objective

Run the same Qwen3-4B DeepSWE functional recipe on either 64 devices
(`4x4x4`, DP4 x TP8 per role) or 256 devices (`4x8x8`, DP16 x TP8 per role).
Both allocations expose the same bounded stages, write the same
real-trajectory schemas and grouped solve metrics, and are judged by one
topology-aware classifier. For the current campaign a successful rollout
inspection may go directly to `three-update`; `one-update` is not a
prerequisite.

This is a fast systems-debug lane. It does not replace or admit the Qwen3-32B
production workload and does not claim bitwise, performance, quality, or
zero-TIM equivalence between allocations.

## Publication contract

- Required remote branch: `yuxzhang/canon-zero-tim`
- Repair development baseline: `7ea2176f807e3e13fde17499e15fef2bd497363b`
- P44.7 placement/batching repair implementation commit:
  `5f0cf7e04b34932d8c9deb2463f3b205e3ad8b51`
- P44.9 development baseline:
  `e4ead609498771987c011a9cbc16fec7e4b17f69`. It archives `p44r04` but does
  not itself contain the SwiGLU feature-padding repair.
- P44.9 SwiGLU feature-padding implementation commit:
  `1a058b461496e039a3857c094b109b794027783a`.
- Pre-publication operator baseline:
  `d8184123448d0add72b72f09d0a6faf5d326c26e`. It archives `p44r05` plus P38
  capture/precheck hardening and publication records.
- P44.10 matmul K/N-padding and P44.11 one-host integration implementation
  commit: `29cea119259f1f7fe583a3e3dd1cb190acc0bf63`. The local v5p evidence predates
  this commit and records a base SHA plus tracked development changes, so it
  remains development evidence rather than clean execution-source evidence.
- First publication metadata read-back:
  `0b492277167004743c07f2fe77705d27c1f8cb01`. Later ledger-only commits may
  advance the branch; the execution-source rule below remains authoritative.
- Exact execution source: resolve the current remote head with
  `git ls-remote origin refs/heads/yuxzhang/canon-zero-tim`, detach at that
  exact SHA, require that it contains the P44.7 and P44.9 repairs above plus
  implementation commit `29cea119259f1f7fe583a3e3dd1cb190acc0bf63`, and record the
  resolved head in the rendered JobSet and returned evidence. The publication
  metadata commit may be newer than either implementation commit; do not
  silently substitute any SHA for another.
- P44.12 bounded-lifecycle implementation commit:
  `e1b4009394c49ea015919bda0cfdb97c12c221b5`. Resolve the exact current
  operator HEAD dynamically and require this commit in its ancestry.
- Local development branch: `codex/p46-deepswe-32b-full`
- Remote execution owner: the launch agent/operator, not the implementation
  agent

The launch agent must fetch the required remote branch, detach at its exact
read-back SHA, verify a clean checkout, and pass that same SHA to the renderer.
Do not launch a local development worktree or an unverified symbolic branch.

## Current evidence

- P44 shared-recipe, Qwen3-4B TP8 overlay, both topology renderers/preflights,
  artifact schemas, both dataset entrypoints, Pathways `logical_task` host
  placement, prompt batching, trajectory-counted logprob batching, and
  classifier controls plus the isolated one-host contract: PASS locally (41
  tests). The cases require runtime
  evidence that TP8-local SwiGLU width `1216` is padded to `1280`, plus both
  matmul `K=1216 Kp=1280` and `N=1216 Np=1280` directions, and reject a run
  that omits any of that evidence.
- Qwen4B overlay exact-image CPU gate: PASS in local immutable image ID
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.
  It passes the two affected real learner unit tests and exact Pallas-interpret
  SwiGLU forward/VJP comparison at `1216->1280` and both matmul K/N padding
  directions, including unregistered-width negative controls. This is not a
  remote registry digest; rerun with the launch image digest.
- Adjacent P43/P39/P34 regressions pass: P43 22/22, P39 15/15, and P34 10
  suites. Qwen8B remains on the unpadded `3072->3072` path; Qwen32B has a
  separately pinned `3200->3328` path. All three overlays reinstall 29/29 and
  pass exact forward/VJP image probes. This is local contract evidence, not a
  Qwen3-32B target execution claim.
- P44.12 defaults are locked in the renderer: B4/G4, response 4096, five
  turns, temperature 1.0, clean 1851-row whitelist, device optimizer and
  three updates. One prompt batch has a shared 3600-second limit; the
  trajectory/model/step/reward/cleanup budgets are 3000/300/600/600/300
  seconds and the R2E pod active deadline is 3300 seconds. The frozen-image
  gate covers unfinished-request abort plus reset/model/reward/cleanup
  deadlines. No target has executed these changes.
- Remote 64-device stages: NOT RUN.
- Remote 256-device attempt `p44r02`: FAILED before mesh construction because
  the old splitter treated degenerate `process_index=0` as host identity. It
  nevertheless proved 256 Pathways devices, pinned R2E provisioning, Qwen3-4B
  model access, CLI admission, and gold filtering. It proves no rollout or
  training execution.
- Optional one-host smoke: four direct TPU v5 devices passed the repeatable
  P44.10 Pallas forward/custom-VJP gate for all five local projection shapes.
  P44.11 then located complete model/R2E/cache/Docker prerequisites and ran
  Qwen3-4B DP1 x TP4 through two real Docker tool actions, durable trajectory
  and solve-metric artifacts, trainer forward, and backward-no-commit.
  Rollout integration is PASS; backward is `INCONCLUSIVE_NO_SIGNAL` because
  both bounded trajectories ended at the context limit with zero reward and
  the finite gradient norm was zero. State was unchanged, commits were zero,
  optimizer state was device-resident, and peak HBM was about 35.92 GiB per
  device. Persistent artifacts and hashes are in
  `phases/p44-11-onehost-deepswe-integration.md`.
- Remote 256-device attempt `p44r03`: FAILED after exact host-complete
  placement because a one-host `CANON_EXPECT_MODEL_MESH_IDS` value leaked into
  the Pathways launch. Published profile cleanup unsets it.
- Remote 256-device attempt `p44r04`: FAILED after dynamic mesh construction,
  Qwen3-4B checkpoint load, W&B connection, and entry into the MLP because
  TP8-local SwiGLU width `1216` was not divisible by the unchanged BF256
  kernel. It proves no completed rollout or training stage. The locally
  validated P44.9 repair zero-pads only the registered 4B feature tail to
  `1280`, calls the unchanged kernel/VJP, and slices back to `1216`.
- Remote 256-device attempt `p44r05`: FAILED after proving the SwiGLU repair
  on all 36 layers. Mosaic rejected Qwen3-4B `BN64/BK64` matmul block specs.
  The local P44.10 repair uses BN/BK128, pads only registered matmul K/N width
  `1216` to `1280`, slices output N back, and makes both directions mandatory
  classifier evidence. The implementation is committed, but it has no clean
  post-publication or 64/256 execution evidence yet.

## First operator action in the current P46 campaign

Read `../p46-deepswe-eval-training-profiles/HANDOFF.md` first. Follow the
runbook's fetch and immutable-input preflight, resolve the exact current remote
HEAD, and require a clean detached checkout containing `e1b40093`. Use the
available 64- or 256-device allocation: first inspect one Q4 clean-evaluation
shard, then render Q4-Instruct `q4-debug` three-update. Do not insert a separate
one-update job by default. Both topologies use the same functional recipe but
remain independent target evidence.
Require the exact `[P34.DEVICE_INVENTORY]`, SwiGLU feature-padding, both
matmul-padding `[PATHTRACE]`, `[P44.LOGPS_BATCH]`, trajectory-batch,
rollout-deadline and RepoEnv cleanup lines from the runbook. Return all three
trajectory/metric batches and the complete failure package on any red or
inconclusive result. Do not edit the recipe or train on a partial timed-out
batch. Use a fresh run id, never an archived manifest.

If 257 devices are available, P44 still renders an exact 256-device
`4x8x8` target. The additional device is outside the mesh.
