# Plan

## Outcome

Provide one operator-facing DeepSWE package with three workload families:

1. Qwen3-4B-Instruct-2507, 16K response, B4/G4, exactly three updates, and a 3600-second rollout-batch deadline.
2. Qwen3-4B-Instruct-2507 clean-data evaluation at 16K, logically 32 tasks x 16 samples but executed as resumable 4-task x 16-sample shards with a 3600-second shard deadline.
3. Qwen3-32B, 16K response, B8/G8, exactly 1000 updates, and a 5400-second rollout-batch deadline.

Training keeps TPU-resident optimizer state. The reviewed 1851-row source whitelist remains immutable. Evaluation produces versioned reports and candidate whitelists; it never silently replaces the production input. Q4 uses signed 64/128 variants and Q32 uses signed 64/256 variants; an admitted pair may differ only in registered topology, worker count, DP-local partitioning, and DP-derived carrier geometry. Main, credentials, cloud resources, precision, loss, reward, sampler semantics, and optimizer math remain out of scope.

## Phases

| Phase | Deliverable | Exit gate | Status |
|---|---|---|---|
| P46.1 | Hardened evaluator and frozen workload contracts | Evaluator unit tests prove config fingerprinting, exact-N resume, full trajectory serialization, invalid-sample isolation, and curriculum classification | completed |
| P46.2 | Three workload-specific dual-topology JobSet families | Six dummy manifests render; normalized Q4 64/128 and Q32 64/256 pairs differ only by the topology allowlist; Q4 uses 3600 s and Q32 uses 5400 s | completed |
| P46.3 | Operator documentation and release regressions | New P46 CPU gate plus P44/P39/P34 adjacent gates and `git diff --check` pass | completed |
| P46.4 | Remote execution campaign | Q4 evaluation smoke, Q4 three-update training, clean-data promotion, and Q32 training each return durable target evidence | pending |
| P46.5 | True reward-only Q4 evaluation | L1 local and one-host gates prove real logprob-request/extraction bypass and artifact provenance; 64-chip paired N16 supplies L3 and trajectories/hour | active |

## Decisions

- Confirmed: the reference branch is `yuxzhang/swe-evaluation-dev` at `5113c0fb788a2c1f31344f6c3b1265d069bf11ea`; it provides n-sample reporting, streaming summaries, resume, and per-task failure isolation, but not the complete artifact and fingerprint contract required here.
- Decision: selectively port behavior; do not merge the reference branch wholesale.
- Decision: `32 x 16` is an evaluation report unit, not a 512-trajectory training update. Execute one concurrency wave of 4 tasks x 16 samples at a time and aggregate eight shards.
- Decision: Qwen3-32B remains B8/G8 and max concurrency 64. B16/G8 is not admitted until a real target proves 128 concurrent sandboxes can finish within the boundary.
- Decision: one-hour and ninety-minute deadlines bound rollout collection, not model initialization or first XLA compilation.
- Decision: a timed-out or incomplete batch never commits an optimizer update. Every evaluation attempt is durable, but only a valid result completes a task/sample identity. Resume uses a consecutive `attempt_index` after invalid results and rejects any attempt after the first valid result.
- Decision: `MAX_STEPS_REACHED`, `MAX_CONTEXT_LIMIT_REACHED`, `MODEL_TIMEOUT`, and the signed whole-trajectory `TIMEOUT` are completed unsolved model outcomes under the fixed evaluation wall-clock budget. They count toward exact N16 and are never resampled. `MODEL_TIMEOUT` is distinguished as `validity_reason=completed_model_timeout`. `ENV_TIMEOUT`, `REWARD_TIMEOUT`, `FAILED`, malformed structure, and recognized harness-adapter failures remain invalid and retryable; an infrastructure/adapter failure overrides an otherwise accepted terminal status.
- Decision: Q4's observed inline-valued R2E tags are canonicalized before the pinned parser for the signed tool set only. Raw `model_response` remains durable; the executed step `action` is canonical. Any surviving inline parameter tag or tool-CLI XML leak produces `validity_reason=r2egym_action_parameter_adapter`, does not count toward N16, and blocks physical-shard completion.
- Confirmed: returned 256-chip run `p46e25608` attempted 64 l0/p0 identities but, under the then-published policy, produced 62 valid results and two invalid `MODEL_TIMEOUT` attempts. The old resume logic incorrectly treated those policy-invalid records as complete and printed a false physical-shard PASS. The fixed evaluator still emits `P46_EVAL_PHYSICAL_INCOMPLETE` and exits nonzero whenever any identity remains invalid under the current policy; historical artifacts are never reclassified in place.
- Confirmed: returned 256-chip run `p46e25609` structurally captured 64 unique reward-only trajectories but every trajectory contains at least one recognizable action-parameter adapter leak. Its 59 SUCCEEDED/four context-limit/one model-timeout status histogram and zero rewards are not clean curriculum evidence. The action/status repair changes source fingerprint and trajectory schema, so `p46e25609` is not resumed or reclassified in place.
- Decision: source SHA is part of the evaluation fingerprint. The first run after publishing the invalid-attempt repair uses a new run id and reruns all 64 l0/p0 identities; it does not transplant the old 62 records. Retries under that fixed fingerprint rerun only invalid identities.
- Decision: Qwen3-4B evaluation is curriculum evidence, not Qwen3-32B ground truth. Q4 all-fail tasks remain a separate Q32 hard tier rather than being deleted.
- Confirmed: 1851 tasks at N16 and 64 trajectories per physical wave require 58 logical reports and 463 physical JobSets. The final logical report has 27 tasks and only physical indices 0-6.
- Decision: one 64-trajectory physical shard is only a smoke/resume unit. The data-washing deliverable is complete only after all 29,616 valid identities and all 58 digest-bearing logical reports exist. The evaluator remains Qwen3-4B-Instruct-2507 with a 16,384-token total response budget, at most 50 environment/model steps, and a 3600-second physical-shard deadline.
- Decision: run one physical JobSet at a time and retry the same l/p index on physical incompleteness. Do not fan out 463 JobSets concurrently. Candidate whitelists remain advisory until a separate digest review and operator decision; they never silently replace the original 1851-row training whitelist.
- Decision: 58 per-logical reports are necessary but not sufficient for the washing deliverable. A fail-closed campaign finalizer must verify one common contract, exact logical indices 0-57, 1851 unique tasks, exact valid N16 and every referenced digest, then emit one global summary plus merged `q4_learnable`, `q32_candidates`, `all_pass` and `all_fail` manifests. Only `P46_EVAL_CAMPAIGN_PASS tasks=1851 valid_trajectories=29616 logical_shards=58` closes the evaluation campaign.
- Decision: P46 maintains parameterized JobSet renderers rather than checked-in launch YAML with stale source/image/node-pool pins. A remote agent renders concrete YAML only after reading back the publication SHA.
- Decision: because the 256-chip Q4 allocation is unavailable, Q4 debug and clean evaluation admit exactly 64-chip `4x4x4` and 128-chip `4x4x8`. The 128-chip debug split is two host-complete DP8 x TP8 roles; evaluation uses all devices as DP16 x TP8. Q4 rejects 256. Q32 remains exactly 64/256 and rejects 128.
- Publication status: implementation commit
  `e1b4009394c49ea015919bda0cfdb97c12c221b5` is published to the operator
  branch. Remote execution resolves the current branch HEAD dynamically and
  requires this implementation commit in its ancestry.
- Confirmed: before P46.5, the nominal no-logprob vLLM path passed integer zero
  for both sampled and prompt logprobs and still called host extraction. The
  published P46.5 path uses `None/None`; zero is a legal logprob value and is
  forbidden as a missing-value sentinel.
- Decision: token identity in the one-host on/off pair is L2 diagnostic
  evidence, not a hard gate. L3 paired N16 solve-rate consistency and valid
  trajectories/hour remain 64-chip target evidence.
- Confirmed: TPU/JAX vLLM rejects per-request sampling seeds. P46 records
  `sampling_rng_mode=engine_global_sequential`; `sample_nonce` is a stable
  task/sample identity and is never represented as an independently replayable
  sampling seed. The one-host L2 diagnostic restores the exact idle engine RNG
  key before each arm.
- Confirmed: the unpublished one-host development gate passed on four direct
  v5p devices with a real pinned clean R2E Docker task, a valid zero-reward
  trajectory, and no residual containers. This completes local L1/L2 only;
  it does not satisfy L3 or target throughput.
- Decision: `logprob_observer` is not a fourth workload family and cannot be a
  production evaluation default. The renderer admits it only as a 64-chip,
  one-task x N16 parity canary. Both canary arms use the same source SHA,
  engine seed, clean task/sample identities, 16K/50-turn limits and lifecycle;
  only the sampled-logprob observation request differs. The artifact classifier
  rejects any missing/invalid identity, cross-SHA comparison, observer arm
  without sampled logprobs, or reward-only arm with numeric logprobs.
- Publication status: P46.5 implementation commit
  `a4d165e854cc4c2320d8120e89aed185eaf61465` is published to the operator
  branch on top of `23bb2a3c`. Remote execution resolves the current branch
  HEAD dynamically and requires `a4d165e8` in its ancestry. Never publish to
  `main`.
- Publication status: invalid-attempt retry, exact-valid physical completion
  and the 1851 x N16 campaign finalizer are published by
  `a642ab267425a5b08b0cebb6e12c607f50f71831`. Remote execution requires this
  commit in the exact read-back operator ancestry before starting a new run id.
- Publication status: the R2E action canonicalizer, trajectory-v4
  fixed-budget terminal policy, and Q4 64/128 topology migration are published
  by implementation commit
  `267a35ef41198dab55fd892a681c3a34b9331a78`. Remote execution requires this
  commit in the exact read-back operator ancestry and must use a new run id.
