# P39.4 — Qwen3-32B direct full training contract

- Status: local implementation PASS, publication approved; target NOT RUN
- Source base: `yuxzhang/canon-zero-tim` at
  `4a2cb8cd2bff2e1e9f5f82a6d2e0575d166759bd` (implementation started at
  `4e4ca2891a01448f09428affd1eb2434bbd61657`; the intervening FrozenLake/P38
  commits did not overlap this phase)

## Operator decision

Use the available 4x8x8 slice for one real Qwen3-32B DeepSWE full-training
run.  Do not require separate one-update or three-update allocations first.
The existing short stages remain diagnostics, not launch prerequisites.

The run is one separated-role Pathways session: rollout DP16xTP8 on 128
devices and trainer DP16xTP8 on the other 128 devices.  Optimizer state is
device-resident by default; host offload is an explicit later fallback, never
an automatic runtime transition.

## Signed workload

- 8 prompts x 8 generations = 64 trajectories per update;
- Qwen3-32B, full parameters, 4096 prompt tokens, 32768 response tokens,
  50 turns, 1000 updates;
- RLOO, `sequence-mean-token-scale`, rollout logprobs enabled, sampler IS and
  TIS disabled, prefix cache disabled;
- R2E-Gym subset pinned at dataset revision
  `2e8108ff942f24fcb5686badfaf7f9a8808566d5`, split `train`;
- clean whitelist
  `clean_data/final_filter_result/task_report_good_qwen3_128_retry_20260713_090141.jsonl`,
  1851 rows and 1851 unique Docker images, SHA-256
  `2f95c2e6df3526f68bd3eed3ab9aece7077ef85c74251c77f7b3474b0b307ed7`.

The full launcher must reject a different dataset revision, split, source-row
count, whitelist path, whitelist digest, whitelist row count, unique-image
count, or filtered-row count.

## Observability and continuation policy

Persist every completed training batch before backward as durable compressed
trajectory JSONL plus a fsynced batch-metrics row.  Each trajectory records its
group/pair identity, status, reward, advantage and the complete redacted raw
trajectory mapping.  Each batch records solve ratio, all-solved, all-failed,
mixed, incomplete and effective prompt-group counts.  Artifact corruption or
write failure is fatal.

`effective_prompt_groups == 0` is telemetry, not a resampling or skip-commit
gate.  Do not inject learning signal.  A finite zero gradient may complete the
normal optimizer transaction and is recorded as a quality warning.

Finite A-B and B-C differences are warning-only.  The same finite-only policy
also keeps all downstream alignment residuals visible without stopping the
convergence run.  NaN/Inf, invalid shapes, exact cross-role weight failure,
topology or replica failure, optimizer placement/transaction failure, OOM,
IFRT failure, missing online W&B, and evidence corruption remain hard errors.
This policy can prove only a convergence/systems run; it cannot promote a
zero-TIM claim.

## Exit gate

The local implementation is complete only after P34 static, trajectory,
update, renderer, classifier and alignment negative controls pass, plus the
adjacent P39/P43/P44 CPU gates affected by the shared CLI and artifact code.
No local CPU or one-host result proves Qwen3-32B, TP8 kernels, DP16 reduction,
Pathways health, HBM capacity or training quality.

## Local exit evidence

- `P34_STATIC_PASS suites=10`
- `P34_TRAJECTORY_CPU_PASS tests=5`
- `P34_UPDATE_CPU_PASS tests=5`
- `P39_DEEPSWE_PILOT_CPU_PASS`
- `P43_DEEPSWE_DEBUG_CPU_PASS`
- `P44_DEEPSWE_QWEN4B_PARITY_CPU_PASS`
- `P34_EXACT_IMAGE_CPU_PASS unit_cases=55 alignment_cases=3 pallas_cases=2
  contract_cases=5 scheduler_cases=1 overlay=qwen32b`

The exact-image negative control proves that finite A-B/B-C residuals produce
`PASS_WITH_ALIGNMENT_WARNINGS`, while a nonfinite B-C residual remains
`FAIL`. These are local contract tests, not a target-training result.
