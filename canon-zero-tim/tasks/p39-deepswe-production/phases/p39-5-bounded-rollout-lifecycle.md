# P39.5 — bounded DeepSWE rollout lifecycle and locked YAML defaults

- Status: implementation published at
  `e1b4009394c49ea015919bda0cfdb97c12c221b5`; local frozen-image gates PASS;
  target retry NOT RUN
- Development base: `yuxzhang/canon-zero-tim` at
  `6905ca7c8551eeb8be772c40213e57e91bcfb0a7`

## Target evidence and diagnosis

Attempt `p34r03`, source
`65b0cd0a84807f2308409d1867022407ae53f8fb`, passed the 256-device
host-complete split, Qwen3-32B initialization, the pinned dataset revision and
the clean 4578-to-1851 join. Training began at update zero. The returned log
then remained in rollout for more than four hours. All 64 trajectories returned
as `ENV_TIMEOUT`; trainer setup subsequently raised `KeyError: 'fsdp'` before
forward, backward or optimizer commit.

The archived log is
`debug_logs/p34_p34r03_deepswe_full.raw.log`, SHA-256
`426019f66f812e0bb80874cbcfb19fe183846b6565251bb5f043d505425dd2a1`.
It contains 64 `ENV_TIMEOUT` records, all reporting negative remaining time.
This establishes a lifecycle bug: the old code could submit an environment
step after the trajectory budget expired, did not abort unfinished vLLM
requests, did not bound final reward or cleanup, and had no shared deadline
for collecting one complete prompt batch.

The same attempt also establishes a separate trainer wiring bug: the live
training mesh is `('dp','tp')`, while the stale training-data sharding axis was
`fsdp`. The published repair derives the data axis from the live mesh and
emits `[DEEPSWE.DATA_SHARDING] PASS` before training.

## Repair

1. Thread a per-turn request deadline through AgenticRL, RLCluster,
   VllmRollout and VllmSampler. On expiry the server-mode driver aborts every
   unfinished request before returning the timeout.
2. Start one trajectory wall clock before environment reset. Bound reset,
   model, environment step and final reward by the remaining time; never pass
   zero or negative timeout to a blocking call.
3. Always close the environment in `finally`. Bound cleanup and fail the run
   if it cannot finish, since continuing could leak CPU sandboxes.
4. Put one shared watchdog around all prompt groups in a rollout batch. A
   timeout reports the number of completed groups and cancels the producer;
   it never trains on a partial batch.
5. Label R2E pods by run, set explicit CPU/memory requests and limits, add an
   active deadline, and delete-and-confirm on startup failure, terminal phase
   and cleanup.
6. Keep durable trajectory/solve metrics enabled. `effective_prompt_groups ==
   0` remains record-and-commit telemetry, finite B-C stays warning-only, and
   TPU-resident optimizer state remains the default.

## Locked launch recipes

| Field | Qwen3-4B debug | Qwen3-32B full |
|---|---:|---:|
| prompts x generations | 4 x 4 | 8 x 8 |
| response / turns | 16384 / 5 | 16384 / 50 |
| updates | 3 | 1000 |
| temperature | 1.0 | 1.0 |
| per turn | 300 s | 300 s |
| trajectory | 3000 s | 4800 s |
| step / reward | 600 / 600 s | 1800 / 1800 s |
| cleanup | 300 s | 300 s |
| R2E active deadline | 3300 s | 5100 s |
| complete rollout batch | 3600 s | 5400 s |

The Qwen3-4B recipe is identical on 64 and 256 devices except for registered
topology, DP-local partitioning, worker count and DP-derived global carrier
geometry. It uses the same reviewed clean whitelist as the 32B run and writes
16 real trajectories plus grouped solve metrics per update. P46 also supplies
a clean Qwen3-4B evaluation profile on both topologies: DP8 x TP8 or DP32 x
TP8, always four tasks by 16 samples with a one-hour shard deadline.

## Local evidence

- `P34_STATIC_PASS suites=10`
- `P34_TRAJECTORY_CPU_PASS tests=5`
- `P34_UPDATE_CPU_PASS tests=5`
- `P39_DEEPSWE_PILOT_CPU_PASS`
- `P43_DEEPSWE_DEBUG_CPU_PASS`
- `P44_DEEPSWE_QWEN4B_PARITY_CPU_PASS` (41 cases)
- `P34_EXACT_IMAGE_CPU_PASS unit_cases=55 alignment_cases=3 pallas_cases=2
  contract_cases=5 scheduler_cases=1 overlay=qwen32b`
- `P44_EXACT_IMAGE_CPU_PASS overlay=qwen4b`, including seven targeted agentic
  tests (five deadline/cleanup controls) and one real server-mode
  unfinished-request abort test
- Dummy, non-launchable renders for P34 full and P44 64/256 three-update all
  ended in their renderer PASS markers and contained the values in the table.

## Claim ceiling and next action

These tests prove local contracts and control flow only. They do not prove a
real Qwen3-4B update, Qwen3-32B training, TP8 target execution, Pathways
liveness, sandbox deletion against the real cluster, HBM capacity or model
quality. No target cloud retry occurred after the published repair.

The launch agent must detach at the exact remote HEAD containing the published
repair and follow the P46 handoff. Run clean Qwen3-4B evaluation, then the
Qwen3-4B-Instruct three-update debug profile on the available registered 64- or
256-chip allocation, inspect every trajectory batch and cleanup marker, then
start a fresh Qwen3-32B full attempt. A batch timeout is a failed/inconclusive
attempt, not a reason to loosen the deadlines or train on partial data.
