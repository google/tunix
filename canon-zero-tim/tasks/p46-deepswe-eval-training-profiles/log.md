# Log

## 2026-08-13T00:00:00Z — P46.1: bind the evaluation and three-profile campaign

- Type: decision
- Fact: The active unpublished worktree is `codex/p46-deepswe-32b-full` at base `6905ca7c8551eeb8be772c40213e57e91bcfb0a7`, with existing P39.5/P44.12 lifecycle changes. Main is not checked out or targeted.
- Action: Bound the new cross-session work to this task directory and froze the three workload families plus Q4/Q32 deadline distinction.
- Command: `git status --short --branch && git rev-parse HEAD && git branch --show-current`
- Result: Q4 debug/evaluation use 3600-second rollout or shard boundaries; Q32 training uses a 5400-second rollout-batch boundary. No cloud action, commit, or push occurred.
- Files/artifacts: `state.md`, `plan.md`, `phases/p46-1-evaluator-and-profile-contracts.md`
- Rollback: Remove only this new P46 task directory if the campaign is abandoned; do not alter the linked P39/P44 ledgers.
- Next: implement and unit-test the hardened evaluator.

## 2026-08-13T03:01:34Z — P46.1-P46.3: local implementation and release gates

- Type: implementation and evidence
- Fact: The 1851-row clean set produces 58 logical N16 reports and 463 physical 4-task x N16 execution shards; the final logical shard has only seven physical shards.
- Action: Implemented full-trajectory config-fingerprinted evaluation and resume, immutable concurrent-safe reports, the Q4 debug/Q4 evaluation/Q32 training dual-topology renderer, P46 environment and classifier admission, device-resident optimizer contracts, and the operator runbook/handoff.
- Commands: `bash canon-zero-tim/tests/p46_deepswe_profiles/run_cpu.sh`; P39/P43/P44 `run_cpu.sh`; `bash canon-zero-tim/tests/p34_deepswe/run_static.sh`; six invocations of `render_p46_deepswe_profiles.py`; `git diff --check`.
- Result: P46 17/17 PASS; P39 15/15 PASS; P43 22/22 PASS; P44 41/41 PASS; `P34_STATIC_PASS suites=10`; six `P46_JOBSET_RENDER_PASS` markers; `git diff --check` PASS. Temporary manifests are under `/tmp/p46-render-validation.SDUAp0` and are not launch artifacts.
- Proven: pure artifact logic, renderer/preflight/postflight contracts, legacy DeepSWE compatibility and concrete YAML construction at the unpublished base SHA.
- Not proven: TPU, Pathways, R2E Kubernetes lifecycle, HBM, real rollout, optimizer update, evaluation quality, Q32 convergence, 64/256 performance parity or zero-TIM.
- Publication audit: `origin/yuxzhang/canon-zero-tim` is two commits ahead. The commits add reviewed DeepSWE datasets and archived logs only; they do not overlap implementation files. No reconcile, commit or push was performed.
- Next: wait for explicit publication approval, reconcile/read back the final SHA, then hand P46.4 to the remote execution agent.

## 2026-08-13T03:43:36Z — P46.4: reconcile returned evidence and repair trainer data axis

- Type: failure analysis, implementation, and local evidence
- Fact: `git pull --ff-only origin yuxzhang/canon-zero-tim` advanced the dirty development worktree without conflict from `6905ca7c8551eeb8be772c40213e57e91bcfb0a7` to `99c3f7af761c859caa6c81ab509446cc3cc47dc0`. Main was not touched, and no commit or push occurred.
- Evidence: `debug_logs/p34_p34r03_deepswe_full.raw.log` reports `configured_prompts=8 generations=8 execution_trajectories=64 observed_trajectories=64`, but contains 64 `ENV_TIMEOUT` clips and 64 `env.step hung` kills with remaining budgets from -15.2 to -16904.3 seconds. It then fails in `sharding_utils.get_sharding` with `KeyError: 'fsdp'`; the same log declares both role meshes as `dp=16,tp=8`.
- Action: Changed `train_deepswe_nb.py` to derive `training_data_sharding_axis` from the leading trainer-mesh axis, validate it against the mesh, emit `[DEEPSWE.DATA_SHARDING] PASS`, and pass that value to `RLTrainingConfig`. Added P34/P44 static regression assertions.
- Trajectory audit: The archived P44r06 full artifact has 16 structurally complete records and valid group/pair/token-mask serialization, but only 3 completed normally, 13 reached max context, all rewards and advantages are zero, and no sample solved. It proves capture wiring, not usable learning signal. The P34r03 full trajectory CSV remains on the remote PVC at `/mnt/disks/linchai_data/deepswe_zero_tim/canon-p34-full-p34r03/metrics/trajectory_log_1786564351.csv` and is not mounted on this host.
- Commands: P34 static/trajectory/update gates; P44 CPU gate; P46 CPU gate; `git diff --check`; direct TPU inventory.
- Result: `P34_STATIC_PASS suites=10`, `P34_TRAJECTORY_CPU_PASS tests=5`, `P34_UPDATE_CPU_PASS tests=5`, `P44_DEEPSWE_QWEN4B_PARITY_CPU_PASS` (41 cases), `P46_DEEPSWE_PROFILES_CPU_PASS cases=17`, and `git diff --check` passed. Direct TPU inventory is unavailable because this host lacks `libtpu.so`; no TPU/Pathways rerun claim was made.
- Next: obtain explicit commit/push approval, publish and read back the exact SHA, then run one bounded Q4 evaluation shard before Q4 three-update and Q32 training.

## 2026-08-13T03:47:53Z — P46.4: refresh remote-agent runbook and handoff

- Type: handoff
- Fact: P34r03's 64/64 cardinality marker described returned objects, while every returned record was `ENV_TIMEOUT`; the later `KeyError: 'fsdp'` occurred before forward/backward. The repair is still unpublished.
- Action: Updated the P46 operator runbook, P46 remote-agent handoff, P34 historical runbook and active P46.4 phase with the mesh-axis marker, old-run interpretation, exact promotion order, per-gate evidence requirements, stop conditions and return package.
- Result: The handoff now makes publication a hard entry gate and orders one 64-chip Q4 evaluation shard, manual full-trajectory inspection, Q4 three-update classifier PASS, optional exact-N16 campaign, then unchanged Q32 B8/G8 16K training. It explicitly forbids using `observed_trajectories` alone as a success claim or falling back to host optimizer/deadline relaxation.
- Files/artifacts: `cluster/P46_DEEPSWE_PROFILES_RUNBOOK.md`, `tasks/p46-deepswe-eval-training-profiles/HANDOFF.md`, `cluster/P34_DEEPSWE_RUNBOOK.md`, `phases/p46-4-remote-execution.md`
- Next: run documentation/static gates, then await explicit commit/push approval.

## 2026-08-13T03:50:45Z — P46.4: handoff validation

- Type: evidence
- Command: `git diff --check`; two data-axis static suites; `bash canon-zero-tim/tests/p46_deepswe_profiles/run_cpu.sh`
- Result: `git diff --check` passed, 16 data-axis/static cases passed, and `P46_DEEPSWE_PROFILES_CPU_PASS cases=17` passed. `START_HERE.md` now routes current DeepSWE execution agents to the P46 handoff before the historical P39 ledger.
- Next: await explicit commit/push approval; no target launch is valid from the unpublished worktree.

## 2026-08-13T03:54:03Z — P46.4: implementation publication

- Type: implementation and handoff
- Fact: The user explicitly approved commit and push to `yuxzhang/canon-zero-tim`; `main` remains forbidden. The remote operator branch was still exactly `99c3f7af761c859caa6c81ab509446cc3cc47dc0` before publication.
- Action: Committed the bounded lifecycle, mesh-derived trainer data axis, full trajectory evaluator, three dual-topology profile families, tests, runbook and handoff.
- Command: `git commit -m "deepswe: harden rollout lifecycle and add training profiles"`
- Result: implementation commit `e1b4009394c49ea015919bda0cfdb97c12c221b5`; P34 static/trajectory/update, P39 15, P43 22, P44 41, P46 17 and `git diff --check` passed immediately before commit.
## 2026-08-13T21:30:00Z — P46.5: DeepSWE 256-chip clean evaluation with exact retry logic (p46e25609)

- Type: evidence and verification
- Fact: Ran `canon-p46-eval-256-0-0-p46e25609` on `mlperf-v5p-256-3` across 64 TPU worker pods (256 TPU v5p chips) with DP32 x TP8 at source `8c0e90f38b68832a8ba7093fe78d655fcfd06ec4`.
- Command: `render_p46_deepswe_profiles.py --profile q4-clean-eval --topology 256 ...`
- Result: 64 total trajectories evaluated for Subshard 0 (l0/p0). 59 trajectories valid (`SUCCEEDED`, reward=0.0), 4 `MAX_CONTEXT_LIMIT_REACHED`, 1 `MODEL_TIMEOUT`. Total rewards = 0.0 across all 64 trajectories.
- Evaluator behavior: Emitted `P46_EVAL_PHYSICAL_INCOMPLETE pending_valid_samples=5 invalid_attempts=5`, proving that the new exact retry and fail-closed logic operates without false-positive completion claims.
- Archive: Committed full head log `l0-p0.log` (557 KB) and trajectory JSONL `trajectories.jsonl` (6.0 MB) to `evidence/p46e25609/`.
- Next: evaluate remaining physical shards or transition to Q32 32B training.


## 2026-08-13T05:40:50Z — P46.5: nominal no-logprob path is not off

- Type: correction and code-read experiment
- Fact: The clean evaluator has no trainer, but `VllmConfig.return_logprobs` remains at false and the false branch in `tunix/generate/vllm_sampler.py` sets both vLLM fields to integer zero. The same path calls `get_logprobs_from_vllm_output` for every completion before discarding the result. The rendered evaluation environment also inherits alignment and processed-logprob switches from the training profile.
- Action: Fast-forwarded the clean P46 worktree from `df46a880426460e96f2b160aef73a532b2bfe58b` to non-overlapping operator HEAD `e4d442bcc654938b5fcf437d901f6691265cb050` and opened P46.5 with layered L1/L2/L3 gates.
- Command: `git merge --ff-only origin/yuxzhang/canon-zero-tim`; source inspection of `eval_deepswe.py`, `deepswe_eval_artifacts.py`, `vllm_sampler.py`, the P46 renderer and rendered environment.
- Result: code-read FAIL for the claimed compute bypass; the expected performance benefit is not admitted until `None/None` plus extraction bypass pass locally and on one real v5p host. Bare-host TPU inventory is unavailable because `libtpu.so` is absent.
- Next: implement the smallest fail-closed reward-only path and CPU gates.

## 2026-08-13T06:17:46Z — P46.5: reward-only L1/L2 passes on direct v5p

- Type: implementation, failure-driven repair, and real one-host evidence
- Fact: Code-read confirmed that vLLM integer zero requests logprobs and that
  the old false path still extracted them. TPU/JAX also rejects per-request
  `SamplingParams.seed`; only the engine seed and ordered RNG split stream are
  supported.
- Action: Added the single fail-closed `evaluation_mode=reward_only`, true
  `None/None` requests, extraction bypass, null/absent-only logprob schema,
  provenance, layered parity classifiers and a direct v5p probe. Repaired only
  the evaluator boundary for single-row `SWEEnv` batching and
  `SamplerOutput`/`RolloutOutput` adaptation. The isolated smoke is one turn,
  256 tokens per call and 512 total; production remains 16K/50 turns.
- Evidence: `P46_DEEPSWE_PROFILES_CPU_PASS cases=31`; two targeted
  `VllmSamplerConfigTest` cases PASS; direct v5p inventory count 4; pinned
  dataset 4578 and whitelist 1851; real R2E Docker task
  `namanjain12/aiohttp_final:006fbe03fede4eaa1eeba7b8393cbf4d63cb44b6`;
  `search` tool execution; final status `SUCCEEDED`, reward 0, one complete
  step, trajectory logprobs null and no residual containers.
- Result: `P46_REWARD_ONLY_ONEHOST_PASS l1=PASS
  l2=IDENTICAL_OBSERVER`. Median diagnostic call time was 0.0330 s with
  logprobs and 0.0310 s reward-only; payloads were 117 and 70 bytes. This is
  request/payload evidence, not a target throughput claim.
- Artifacts: report
  `/mnt/disks/tunix-data/deepswe-reward-only-evidence/reward-only-onehost-20260813T061510Z-696010/report.json`
  (`db3305413817ffe5c4d0085098475a12753cea6b698e15e4263b0c7d0835ba7c`),
  trajectory JSONL (`2497cb614a92a888c34c4ec4b019d05a3e10d9024b61c7c9853861f725a1bfa8`),
  full log (`c1e7df327aa208fd570ae663f1791f0082f5d26b5254c3519668fde813e57500`).
- Publication: development evidence used a dirty worktree at base
  `e4d442bcc654938b5fcf437d901f6691265cb050`; no commit or push occurred.
  Current operator HEAD advanced to `23bb2a3c`; its P46 log-directory and
  single-Pathways-session corrections are preserved locally, but publication
  still requires a clean reconcile.
- Next: 64-chip paired N16 L3 plus valid trajectories/hour; do not promote or
  launch the full clean-eval campaign before it passes.

## 2026-08-13T06:22:32Z — P46.5: adjacent regression and render audit

- Type: local evidence and publication audit
- Commands: P46 CPU; P34 static/trajectory/update; P39 CPU; P44 CPU; targeted
  vLLM sampler tests under the training venv; Python compile; shell syntax;
  six P46 renderer invocations; `git diff --check`.
- Result: P46 31/31 PASS, `P34_STATIC_PASS suites=10`, P34 trajectory 5 and
  update 5 PASS, P39 15 PASS, P44 41 PASS, two direct sampler tests PASS, all
  six 64/256 Q4-debug/Q4-eval/Q32 manifests render, and syntax/diff checks
  pass.
- Remote audit: `origin/yuxzhang/canon-zero-tim` advanced from the local base
  by three commits to `23bb2a3c`. Two P46 corrections create the evaluation
  log directory and remove `CANON_EXPECTED_SLICE_DEVICES` to preserve one
  uninterrupted Pathways client session; both behaviors are preserved in the
  working changes. The third commit is unrelated P38 evidence. No pull,
  commit, push or main-branch mutation occurred.
- Next: reconcile without dropping the remote evidence files, then repeat the
  release gates from a clean tree only after explicit publication approval.

## 2026-08-13T06:32:34Z — P46.5: close the executable L3 handoff

- Type: implementation, validation, and handoff
- Action: Added a validation-only `logprob_observer` control admitted only as
  one clean task x N16 on 64 chips, plus the paired artifact/solve-rate/
  trajectories-hour classifier. Both arms use the same stock sampler SHA,
  clean-data order, engine seed, 16K/50-turn limits and lifecycle. Normal Q4
  evaluation remains `reward_only`; training families and 256-chip canaries
  reject these evaluation-only controls.
- Safety: reward-only artifacts accept absent/null logprob values but reject
  any numeric payload including `0.0`. Observer artifacts require real numeric
  sampled-token logprobs. Explicit trainer, alignment, rescore, processed-
  logprob or optimizer switches fail before TPU initialization.
- Evidence: P46 31/31 PASS; P34 static/trajectory/update PASS; P39 15 PASS;
  P44 41 PASS; the two direct sampler tests PASS; Python/shell syntax and
  `git diff --check` PASS. Six normal 64/256 manifests and the two 64-chip L3
  arms render. Development L3 manifest SHA-256 values are
  `5815be34b17a5274e605e5b178f8c811f7ff6d4fdf1ef943fdd8c54400e9ac05`
  (observer) and
  `03c03ade29ffea1a490a90a96eaa1a09df831ca75705baaad2b4cd45b3c3ef8a`
  (reward-only).
- Not proven: neither 64-chip arm was applied, so L3 statistics, target
  trajectories/hour, Kubernetes cleanup under N16 concurrency, TP8 and any
  full-campaign speedup remain unverified.
- Publication: local base remains
  `e4d442bcc654938b5fcf437d901f6691265cb050`; operator HEAD is
  `23bb2a3c1a77fa4037f3ec81b783e48d1af22951`. No reconcile, commit, push or
  main-branch mutation occurred.
- Next: after explicit publication approval, reconcile and rerun clean gates;
  then launch exactly the two reviewed 64-chip canary arms and run the L3
  classifier before changing the Q4 clean-evaluation default.

## 2026-08-13T06:46:31Z — P46.5: publication gates pass after reconcile

- Type: publication
- Authorization: The user explicitly requested commit and push. The only
  target is `yuxzhang/canon-zero-tim`; `main` was not checked out, modified or
  targeted.
- Action: Created the reward-only implementation commit, rebased it without
  conflict onto operator HEAD
  `23bb2a3c1a77fa4037f3ec81b783e48d1af22951`, and preserved the remote P38
  evidence plus both P46 single-session/log-directory corrections.
- Implementation commit:
  `a4d165e854cc4c2320d8120e89aed185eaf61465`.
- Clean post-rebase gates: P46 31/31 PASS; `P34_STATIC_PASS suites=10`;
  P34 trajectory/update 5/5 PASS; P39 15/15 PASS; P44 41/41 PASS; targeted
  vLLM sampler tests 2/2 PASS; `git diff --check` PASS.
- Claim ceiling: publication does not supply the real 64-chip L3 canary,
  Kubernetes N16 cleanup/throughput, 256-chip target behavior or a full
  evaluation-campaign speedup claim.
- Next: read back the exact operator remote SHA, then execute the two reviewed
  64-chip canary arms before default promotion.

## 2026-08-13T09:05:17Z — P46.5: revoke false 256-chip shard PASS and repair resume

- Type: returned-evidence correction, implementation, and handoff
- Fact: `git pull --ff-only` advanced the operator-derived worktree to
  `63b092b001864e4e9a4822b4354a665bb00b1c6b`. Archived run `p46e25608`
  used source `bdc9681824743911d0691659604dec090dd42bc4`, initialized Qwen3-4B
  reward-only DP32 x TP8, and attempted 64 l0/p0 identities. The exact unique
  terminal audit is 62 `SUCCEEDED` plus two `MODEL_TIMEOUT`; the old evaluator
  incorrectly emitted `P46_EVAL_SUBSHARD_PASS` and exit zero.
- Root cause: resume considered any task/sample record complete. An invalid
  timeout was therefore durable but unretryable, while the physical gate
  checked only whether collection itself hit the wall-clock timeout.
- Action: Added consecutive `attempt_index` records; valid-only identity
  completion; retry after invalid-only attempts; duplicate rejection after the
  first valid result; valid-retry selection in task/L3 aggregation; and
  `P46_EVAL_PHYSICAL_INCOMPLETE` nonzero exit while a physical valid identity
  is missing. Added a fail-closed campaign finalizer that requires all 58
  summaries, 1851 unique tasks, exact valid N16 and referenced-file digests
  before writing merged candidate manifests. Updated tests from 31 to 33 cases.
- Campaign correction: l0/p0 is only a 64-identity smoke/resume unit. Full
  data washing requires 29,616 valid identities, 58 logical reports and 463
  sequential/resumable physical JobSets at Qwen3-4B, 16,384 total response
  tokens, at most 50 environment/model steps and a 3600-second physical
  deadline. Candidate whitelists remain advisory pending separate review.
- Artifact correction: Persistent trajectories live below
  `/mnt/disks/linchai_data/deepswe_eval/<run-id>/outputs/trajectories/`, not
  directly below the run root. The remote return package now requires JSONL
  paths, line counts, per-file SHA-256 and a compressed trajectory/log archive;
  the head log alone is insufficient.
- Publication: The repair and documentation are local and unpublished. Because
  source SHA is part of the fingerprint, the first fixed target attempt must
  use a new run id and rerun all 64 l0/p0 identities rather than transplanting
  the old 62. No commit, push, main-branch mutation or cloud launch occurred.
- Next: finish local/adjacent gates, await explicit publication approval, then
  rerun fixed l0/p0 and continue all remaining physical shards.

## 2026-08-13T09:15:09Z — P46.5: retry/finalizer release gates pass locally

- Type: local evidence and handoff validation
- Commands: P46 CPU gate; P34 static/trajectory/update gates; P39 and P44 CPU
  gates; Python compile for all changed evaluator/finalizer entrypoints;
  finalizer `--help`; `git diff --check`.
- Result: `P46_DEEPSWE_PROFILES_CPU_PASS cases=33`,
  `P34_STATIC_PASS suites=10`, P34 trajectory/update 5/5 PASS,
  `P39_DEEPSWE_PILOT_CPU_PASS`, `P44_DEEPSWE_QWEN4B_PARITY_CPU_PASS`, Python
  compile PASS and diff check PASS. The campaign-finalizer unit gate proves
  missing summaries fail and a complete exact-N multi-shard fixture emits the
  merged category manifests.
- Dependency note: The two vLLM sampler request/extraction tests passed at the
  published P46.5 commit and their implementation is unchanged here. The
  current bare Python environment cannot rerun that module because
  `transformers` is absent; the P46 static request/extraction contract still
  passes. A clean publication environment should rerun the targeted sampler
  pair before target launch.
- Handoff result: The runbook and handoff now require a new fixed run id,
  exact-valid physical retry, complete JSONL/digest return, all 463 sequential
  physical JobSets, all 58 logical reports, and the final
  `P46_EVAL_CAMPAIGN_PASS tasks=1851 n_sample=16
  valid_trajectories=29616 logical_shards=58` marker. Sixty-four trajectories
  are explicitly a physical resume unit, not a washing-completion claim.
- Publication: No commit, push, main-branch mutation or cloud launch occurred.
  The worktree remains intentionally dirty pending explicit user approval.
- Next: publish only after approval, read back the exact operator SHA, then the
  remote agent reruns l0/p0 and advances the full campaign one physical shard
  at a time.

## 2026-08-13T09:20:01Z — P46.5: exact retry and campaign finalizer publication

- Type: publication
- Authorization: The user explicitly requested commit and push. The only
  publication target is `yuxzhang/canon-zero-tim`; `main` remains forbidden.
- Remote audit: The operator branch remained exactly
  `63b092b001864e4e9a4822b4354a665bb00b1c6b`, so no rebase or conflict
  resolution was required.
- Implementation commit:
  `a642ab267425a5b08b0cebb6e12c607f50f71831`.
- Published scope: consecutive invalid-attempt retry, valid-only physical
  completion, nonzero `P46_EVAL_PHYSICAL_INCOMPLETE`, exact-N retry-aware L3,
  complete trajectory return instructions, and the fail-closed 1851 x N16
  campaign finalizer/merged learnable list.
- Evidence before commit: P46 33/33, P34 static/trajectory/update, P39 and P44
  CPU gates, Python compile and `git diff --check` passed. No TPU or cloud
  launch was performed by this publication step.
- Next: read back the final operator HEAD, then remote execution starts a new
  fixed run id at l0/p0 and proceeds through the complete campaign.

## 2026-08-13T21:52:44Z — P46.5: p46e25609 stop and trajectory root-cause repair

- Type: returned-evidence audit, implementation, validation, and handoff
- Source audit: Initially pulled evidence-bearing operator HEAD
  `cc17378b7492d3b046d6b9c68b46df1b9da21647`, then fast-forwarded the dirty
  worktree with autostash to current operator HEAD
  `b4391703d6e1ec80b8da5589e02dfe72ba9a4a4e`; the intervening P38-only commit
  does not overlap this fix. The newest returned artifact is a 256-chip DP32 x
  TP8 run, not 64-chip; its exact in-artifact source is
  `8c0e90f3b995f457c1dbb2199639f7f47962ed2b`.
- Stop cause: all 64 physical identities were attempted. The evaluator then
  returned nonzero at `P46_EVAL_PHYSICAL_INCOMPLETE` because it classified
  four `MAX_CONTEXT_LIMIT_REACHED` outcomes and one `MODEL_TIMEOUT` as five
  pending valid samples. The four signed context-budget outcomes should be
  valid unsolved results; only the model timeout is retryable by status.
- Trajectory audit: 64/64 unique task/sample identities, exact four tasks x
  N16, 1,102 steps with nonempty actions and observations, null logprobs, and
  verified archive digests. Terminal counts are 59 SUCCEEDED, four
  MAX_CONTEXT_LIMIT_REACHED, one MODEL_TIMEOUT; total reward is zero.
- Semantic failure: every trajectory contains at least one recognized leaked
  R2E parameter tag. Counts are 347 `unrecognized arguments`, 363 file-editor
  usage errors, 172 `/parameter` shell errors, and 40 missing-argument errors.
  The dominant Q4 spelling `<parameter=command=view>` was parsed as a key and
  emitted as invalid `--command=view`. The artifact proves durable capture,
  not clean execution, solve rate, or washed data.
- Action: Added a signed-tool action canonicalizer that repairs only the
  observed inline-valued tags and file-editor command shorthand before the
  pinned R2E parser. Raw model responses remain in trajectories; canonical
  executed actions are stored. R2E installation now asserts the positional
  file-editor command contract. Trajectory schema v4 invalidates any surviving
  adapter signature with `validity_reason=r2egym_action_parameter_adapter`.
  Max-step, max-context and whole-trajectory budget terminals now count as
  completed unsolved results without biased resampling; per-turn runtime
  failures still retry.
- Validation: 40/40 P46 CPU tests pass. Three observed malformed action forms
  were also normalized against the actual pinned R2E source at
  `0d94c4eb9431cd195c55a7ea3abd54006c9a1735`, producing correct positional
  bash commands. Evidence SHA-256 verification and `git diff --check` pass.
  Direct-host inventory cannot initialize TPU because `libtpu.so` is absent,
  so no one-host model/R2E trajectory was run and target behavior remains
  unproven.
- Publication: Local and unpublished. No commit, push, main-branch mutation,
  TPU launch, Kubernetes launch, or secret access occurred.
- Next: after explicit commit/push approval, publish only to
  `yuxzhang/canon-zero-tim`, read back the exact SHA, then use a new run id to
  rerun all 64 l0/p0 identities. Do not continue later shards or Q32 until the
  repaired shard has real tool outputs and no adapter-invalid records.

## 2026-08-13T22:01:34Z — P46.5: migrate Q4 large topology from 256 to 128 chips

- Type: implementation, regression validation, and operator-handoff update
- Decision: Q4 debug and clean evaluation now admit exactly 64-chip `4x4x4`
  or 128-chip `4x4x8`; Q4 topology 256 is rejected. Qwen3-32B training remains
  exactly 64/256 and rejects 128.
- Q4 geometry: 128-chip debug is split host-completely into 64-device rollout
  and trainer roles, each DP8 x TP8, global M2048, two local trajectories and
  two per-DP scheduler slots. 128-chip evaluation uses all devices as DP16 x
  TP8. The JobSet uses 32 four-chip workers and `tpuv5:4x4x8` for both resource
  manager and workers.
- Action: Added a signed `4x4x8` role splitter, migrated the P44 workload,
  environment, artifact, classifier and renderer contracts to 64/128, and
  added workload-specific P46 topology allowlists plus negative controls for
  Q4-256 and Q32-128.
- Validation: `P44_DEEPSWE_QWEN4B_PARITY_CPU_PASS` (41 cases) and
  `P46_DEEPSWE_PROFILES_CPU_PASS cases=40` pass. No target TPU/Pathways launch
  was performed, so the 128-chip topology remains unproven target behavior.
- Publication: Local and unpublished. No commit, push, main-branch mutation,
  Kubernetes launch, cloud-resource mutation, or secret access occurred.

## 2026-08-13T22:21:59Z — P46.5: accept model timeout as a fixed-budget outcome

- Type: evaluation-policy correction, implementation, regression validation,
  and handoff update
- Decision: `MODEL_TIMEOUT` now completes its task/sample identity as a valid
  unsolved result under the signed evaluation wall-clock budget. It records
  `validity_reason=completed_model_timeout`, remains reward zero, and is not
  resampled. This measures success within the fixed budget rather than
  unlimited model capability.
- Fail-closed boundary: `ENV_TIMEOUT`, `REWARD_TIMEOUT`, `FAILED`, malformed
  structure, and recognized adapter/parser corruption remain invalid and
  retryable. The infrastructure/adaptor check runs independently of status and
  overrides `MODEL_TIMEOUT`; a timeout carrying the observed R2E parameter leak
  remains `r2egym_action_parameter_adapter` and cannot enter N16.
- Historical evidence: This policy does not rewrite or resume `p46e25608` or
  `p46e25609`. In particular, all 64 `p46e25609` records remain ineligible
  because every trajectory has an adapter signature even though its terminal
  statuses are now all accepted in isolation.
- Validation: focused artifact tests pass 12/12;
  `P46_DEEPSWE_PROFILES_CPU_PASS cases=40`,
  `P34_STATIC_PASS suites=10`, and
  `P44_DEEPSWE_QWEN4B_PARITY_CPU_PASS` all pass. Runbook, handoff, state, plan,
  and the active phase record the same policy.
- Publication: Local and unpublished. No commit, push, main-branch mutation,
  TPU launch, Kubernetes launch, cloud-resource mutation, or secret access
  occurred.
- Next: run final diff/static checks, then publish only after a separate
  explicit user request. The first target rerun must use a new run id and must
  prove zero adapter-invalid records before the campaign advances.
