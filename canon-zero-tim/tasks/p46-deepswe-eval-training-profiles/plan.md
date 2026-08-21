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
| P46.5 | True reward-only Q4 evaluation | L1 local and one-host gates prove real logprob-request/extraction bypass and artifact provenance; 64-chip paired N16 supplies L3 and trajectories/hour | deferred by operator; L3 remains unproven |
| P46.6 | Persistent full clean-data washing | One Q4 runtime completes 463 bounded waves, 29,616 exact identities, 58 logical summaries, and the global learnable list; an explicit resume tag recovers missing identities from fsynced v6 artifacts under one immutable contract and exclusive lease | active |
| P46.7 | Breadth-first census before strict repair | A default-off full reward-only census gives every never-attempted identity one durable attempt, defers invalid retries without reclassifying them, survives bounded wave timeouts, and emits separate coverage-only artifacts; schema-correct sealed migration preserves prior sampler evidence under a fresh harness/tag before the unchanged strict finalizer runs | published base `365b46c1`; returned v5-import incident repair `f823bb6a` |

## Decisions

- Confirmed: the reference branch is `yuxzhang/swe-evaluation-dev` at `5113c0fb788a2c1f31344f6c3b1265d069bf11ea`; it provides n-sample reporting, streaming summaries, resume, and per-task failure isolation, but not the complete artifact and fingerprint contract required here.
- Decision: selectively port behavior; do not merge the reference branch wholesale.
- Decision: `32 x 16` is an evaluation report unit, not a 512-trajectory training update. Execute one concurrency wave of 4 tasks x 16 samples at a time and aggregate eight shards.
- Decision: Qwen3-32B remains B8/G8 and max concurrency 64. B16/G8 is not admitted until a real target proves 128 concurrent sandboxes can finish within the boundary.
- Decision: one-hour and ninety-minute deadlines bound rollout collection, not model initialization or first XLA compilation.
- Decision: a timed-out or incomplete batch never commits an optimizer update. Every evaluation attempt is durable, but only a valid result completes a task/sample identity. Resume uses a consecutive `attempt_index` after invalid results and rejects any attempt after the first valid result.
- Decision: `MAX_STEPS_REACHED`, `MAX_CONTEXT_LIMIT_REACHED`, `MODEL_TIMEOUT`, and the signed whole-trajectory `TIMEOUT` are completed unsolved model outcomes under the fixed evaluation wall-clock budget. They count toward exact N16 and are never resampled. `MODEL_TIMEOUT` is distinguished as `validity_reason=completed_model_timeout`. `ENV_TIMEOUT`, `REWARD_TIMEOUT`, `FAILED`, and malformed trajectory structure remain invalid and retryable.
- Decision: Q4 evaluation explicitly enables `action_compat_mode=q4_r2egym_xml_v2`. It repairs only observed, deterministic inline-value/nested-key forms and the signed top-level `view/create/str_replace/insert/undo_edit` mapping. Raw `model_response`, canonical executed `action`, repair count, and model-action-error count remain durable. Unambiguous model syntax/tool errors are valid model outcomes, normally reward zero, and are never resampled as infrastructure. Only an adapter-internal corruption is a hard harness failure.
- Decision: Qwen3-32B training and every ordinary `SWEAgent()` keep `action_compat_mode=strict_xml`; the Q4 repairer is not a default DeepSWE component and cannot alter Q32 trajectories.
- Confirmed: returned 256-chip run `p46e25608` attempted 64 l0/p0 identities but, under the then-published policy, produced 62 valid results and two invalid `MODEL_TIMEOUT` attempts. The old resume logic incorrectly treated those policy-invalid records as complete and printed a false physical-shard PASS. The fixed evaluator still emits `P46_EVAL_PHYSICAL_INCOMPLETE` and exits nonzero whenever any identity remains invalid under the current policy; historical artifacts are never reclassified in place.
- Confirmed: returned 256-chip run `p46e25609` structurally captured 64 unique reward-only trajectories but every trajectory contains at least one recognizable action-parameter adapter leak. Its 59 SUCCEEDED/four context-limit/one model-timeout status histogram and zero rewards are not clean curriculum evidence. The action/status repair changes source fingerprint and trajectory schema, so `p46e25609` is not resumed or reclassified in place.
- Decision: source SHA is part of the evaluation fingerprint. The first run after publishing the invalid-attempt repair uses a new run id and reruns all 64 l0/p0 identities; it does not transplant the old 62 records. Retries under that fixed fingerprint rerun only invalid identities.
- Decision: Qwen3-4B evaluation is curriculum evidence, not Qwen3-32B ground truth. Q4 all-fail tasks remain a separate Q32 hard tier rather than being deleted.
- Confirmed: 1851 tasks at N16 and 64 trajectories per physical wave require 58 logical reports and 463 physical waves. The final logical report has 27 tasks and its final wave contains 48 identities.
- Decision: one 64-trajectory physical shard is only a smoke/resume unit. The data-washing deliverable is complete only after all 29,616 valid identities and all 58 digest-bearing logical reports exist. The evaluator remains Qwen3-4B-Instruct-2507 with a 16,384-token total response budget, at most 50 environment/model steps, and a 3600-second physical-shard deadline.
- Decision: production washing uses one `--full-campaign` JobSet and one resident Q4 runtime, processing the same 463 waves sequentially. Each wave has one shared 3600-second wall-clock budget across its initial attempt and genuine-infrastructure retries; every trajectory is fsynced immediately. `resume_tag` is the stable PVC campaign identity while `run_id` names one Kubernetes launch. A relaunch with the same resume tag, original harness SHA, sampling-source SHA, topology and exact contract resumes only missing identities. It never fans out sandbox waves concurrently.
- Decision: config-v4/trajectory-v6 pins the resume tag in every fingerprint and writes one immutable `resume_contract.json`. A nonblocking `flock` permits only one active full-campaign writer. Setup state is per launch, logs are immutable per attempt, and resumed launches delete only orphaned R2E pods carrying the same resume-tag label before creating new sandboxes. An incomplete in-flight trajectory restarts from its beginning; token-level continuation is not claimed.
- Decision: do not stop the operator-reported running legacy-v5 `p46e12805` campaign. After natural termination and proof that no producer pod remains, copy its trajectory tree into a fresh `<resume-tag>/imports/p46e12805/` snapshot and seal every JSONL in `SHA256SUMS`. Never import the live root or place v5 rows directly under v6 outputs.
- Decision: the only admitted v5-to-v6 adoption derives and verifies each per-logical-shard v3 fingerprint from the exact old sampling source, clean task order and signed contract. It preserves `sampled_by=stock@18d5d2ac...`, pins the resume-capable checkout separately as `harness_commit=c3a960ac...`, emits immutable per-row provenance plus a receipt, and must be the first trajectory evidence for a fresh resume tag. Any drift fails before TPU initialization.
- Decision: the first post-return pass is breadth-first rather than strict retry-first. `CANON_P46_CENSUS_FIRST_PASS=1` is one default-off orchestration mode and is deliberately absent from the sampling/config fingerprint: it schedules only identities with no durable attempt, runs each at most once during census, defers invalid `FAILED`/environment/reward outcomes without converting them to reward zero, and continues to later physical waves after a bounded wave timeout. Census completion means all 29,616 identities have durable evidence, not that washing is complete.
- Decision: census outputs are immutable snapshots below `outputs/census/` with `trajectory_mode=reward_only_no_logprobs`, a coverage-only claim, provisional complete-task categories, and explicit deferred-invalid/unattempted lists. They never populate canonical `outputs/reports/` or `outputs/campaign/`, and `P46_EVAL_CENSUS_PASS` never substitutes for strict `P46_EVAL_CAMPAIGN_PASS`.
- Decision: a new harness cannot append directly to the existing v6 resume tag because config-v4 fingerprints bind both `resume_tag` and `harness_commit`. After the old producer is terminal and absent, copy its `outputs/resume_contract.json` plus every raw trajectory JSONL into a sealed snapshot. The explicit frozen-v6 importer verifies the old contract, every row/digest, logical fingerprint, run tag, task/sample identity, attempt sequence, reward-only shape and sampler provenance; it then copies the evidence into a fresh tag, changing only harness/tag fields and adding record-level migration provenance. Source rows remain untouched, `sampled_by=stock@<old source SHA>` is preserved, and any data/sampling/topology drift fails before TPU initialization.
- Confirmed: the snapshot returned for `p46c128a0` is trajectory-v5 despite its `p46e12806-v6-final` directory name. Snapshot kind is determined by every row's `schema`, never by its directory name. This source must use legacy-v5 adoption and must not be given a synthetic `resume_contract.json` or routed through frozen-v6 migration.
- Decision: every legacy-v5 or frozen-v6 import requires an explicit historical `--sampling-source-commit`; the renderer must never infer sampler lineage from the live harness `--source-commit`. Before writing the destination resume contract, the entrypoint validates the sealed snapshot kind and checks every legacy-v5 row's sampling, identity, attempt, reward-only and outcome contract; a frozen-v6 source is prechecked against its immutable source resume contract and manifest.
- Decision: `p46q4census01` is immutable failed-attempt evidence because old code claimed it with the wrong inferred sampler before import failure. Never delete, overwrite or reuse it. Recovery uses a fresh tag (`p46q4census02`) and imports the complete terminal raw v5 tree. Imported durable identities are skipped; only absent identities run.
- Decision: actual import cardinality comes only from the sealed raw tree and `LEGACY_IMPORT_PASS` receipt. The incident reports 510 raw rows; stale 6,460+ registry text and the 22,918-row five-field derived outcome table cannot be used as resume evidence.
- Decision: after `P46_EVAL_CENSUS_PASS`, strict repair uses the same new resume tag, harness SHA and sampling source with census/import flags omitted. The original `remaining_samples` policy then retries every identity lacking a valid result. Only the unchanged 1,851-task/29,616-valid/58-summary finalizer produces washed lists.
- Decision: the launcher fetches the operator branch, proves the original manifest SHA remains in its ancestry, and checks out that SHA instead of moving `FETCH_HEAD`. This permits safe recovery after the branch advances without allowing a different source contract to consume old evidence.
- Confirmed: returned 128-chip `p46e12804` captured 64 identities in about 21 minutes but spent about ten minutes initializing/compiling. Repeating that cost for 463 JobSets is rejected; the persistent runtime is a throughput fix without changing sampling, reward, N16, 16K, or 50-step semantics.
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
- Publication status: P46.6 action compatibility v2, config-v3/trajectory-v5,
  model-outcome validity, and `--full-campaign` are committed as
  `a989af34054434e6567f88e99b45ed67faf15a44` on base
  `c33ba5f50d606210ca9f2c94fca003b63ea6e326`. Remote execution requires this
  implementation commit in the exact read-back operator ancestry.
- Publication status: crash-safe resume and reviewed frozen-v5 adoption are
  published as `c3a960acdc94173440144559bb95f1de36d31537` on base
  `2ec1cb768c7454c7d0ecf798ff1a5aff890ceae7`. Operator-branch checkpoint
  `dc6b5b32a90ad0e12b1b9ae50ef7cc060b450abf` was read back with both the
  implementation and synchronized handoff in its ancestry. Executors still
  resolve the current branch HEAD dynamically and repeat the ancestry gate.
- Publication status: P46.7 breadth-first census and frozen-v6 migration are
  published as `365b46c1cd150839e3be1fd50adb33325fe3189f` on base
  `eae3d6d47e07bbb631106284da40a5e90763faee`. Exact operator-branch read-back
  resolved the same SHA with local/remote divergence `0/0`; target execution
  still requires a fresh read-back and explicit launch authority.
