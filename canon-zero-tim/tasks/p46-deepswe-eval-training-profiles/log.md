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

## 2026-08-13T22:26:58Z — P46.5/P44.13 implementation publication

- Type: publication
- Authorization: The user explicitly requested commit and push. The only
  target is `yuxzhang/canon-zero-tim`; `main` was not checked out, modified, or
  targeted.
- Remote audit: `origin/yuxzhang/canon-zero-tim` remained exactly
  `b4391703d6e1ec80b8da5589e02dfe72ba9a4a4e`, matching the local baseline, so
  no rebase or conflict resolution was required.
- Implementation commit:
  `267a35ef41198dab55fd892a681c3a34b9331a78`.
- Published scope: narrow R2E action canonicalization and fail-closed adapter
  detection; model-timeout-as-valid-unsolved fixed-budget semantics; Q4
  64/128 topology contracts and negative controls; and synchronized P44/P46
  runbook, handoff, state, plan, and phase records.
- Pre-publication evidence: focused artifacts 12/12,
  `P46_DEEPSWE_PROFILES_CPU_PASS cases=40`, `P34_STATIC_PASS suites=10`,
  `P44_DEEPSWE_QWEN4B_PARITY_CPU_PASS`, Python compilation, and
  `git diff --check` all pass.
- Claim ceiling: publication contains no new TPU, Pathways, Kubernetes,
  repaired R2E trajectory, HBM, throughput, backward, or optimizer-update
  evidence. `p46e25609` remains wholly ineligible historical evidence.
- Next: read back the exact final operator HEAD, then rerun Q4 evaluation
  `l0/p0` under a new run id on an admitted 64/128 topology and require zero
  adapter-invalid records before advancing.

## 2026-08-14T00:00:00Z — P46.6 persistent full-washing implementation

- Type: returned-evidence diagnosis, replan, implementation, and local
  validation. No cluster launch, commit, push, or main-branch mutation.
- Baseline: clean operator worktree at
  `c33ba5f50d606210ca9f2c94fca003b63ea6e326` before local edits.
- Returned evidence: `p46e12804`, source
  `2c160bf931d4d94756f5200472de8070615c0e9f`, Qwen3-4B-Instruct-2507,
  128 chips, DP16 x TP8, four clean tasks x N16. Exact status was 54
  `SUCCEEDED`, nine `MAX_CONTEXT_LIMIT_REACHED`, one `MODEL_TIMEOUT`; rewards
  were seven solved and 57 unsolved. Fifty-nine records were accepted and five
  rejected by the old adapter policy.
- Root cause: `_INLINE_PARAMETER` greedily consumed the real tail closing tag
  in `<parameter=cmd=ls</parameter>` and synthesized
  `</parameter</parameter>`. Returned data also contains nested
  `parameter=path`, top-level editor command shorthands, and one accepted
  `--parameter path` tool error. The first is our harness bug; the latter forms
  are Q4 model dialect/capability outcomes and must not cause resampling bias.
- Action fix: compatibility v2 repairs only exact observed forms, leaves
  contradictory commands untouched, and records action mode/repair/error
  provenance. Q4 eval opts in; default `SWEAgent()` and Q32 remain
  `strict_xml`. Model tool errors are valid outcomes. Adapter-created
  corruption is a hard failure. Schemas advance to config-v3/trajectory-v5.
- Throughput finding: the returned job took about 21 minutes and paid roughly
  ten minutes of model initialization/JIT for one 64-trajectory shard.
  Repeating that across 463 JobSets is rejected.
- Campaign implementation: `--full-campaign` creates one resident Q4 runtime
  and processes all 58 logical shards / 463 sequential one-hour waves. Every
  trajectory remains fsynced; real infrastructure failures retry only inside
  the current wave's shared one-hour budget; timeout stops nonzero and resumes
  with the same run id. Finalization remains exact 1851 x N16 = 29,616.
- Local evidence: P46 CPU suite passes. Its complete-scale fake-runtime test
  observes exactly one runtime, 463 waves, 29,616 identities, and a final
  48-identity wave. This is orchestration evidence only, not target washing.
- Operator decision: P46.5 L3 is deferred without being declared passed. The
  next target run, after publication and separate launch approval, is one
  admitted-topology full-washing campaign rather than another l0/p0 smoke.

## 2026-08-14T00:00:01Z — P46.6 implementation commit

- Type: publication preparation under explicit user commit/push approval.
- Remote precondition: freshly fetched
  `origin/yuxzhang/canon-zero-tim` remained exactly
  `c33ba5f50d606210ca9f2c94fca003b63ea6e326`; local and remote divergence was
  `0/0`, so no rebase or conflict resolution was required.
- Implementation commit:
  `a989af34054434e6567f88e99b45ed67faf15a44`.
- Validated scope: P46 49/49 unittest PASS with stable release marker
  `P46_DEEPSWE_PROFILES_CPU_PASS cases=40`; P34 static suites=10 PASS; P44
  41/41 PASS; returned-artifact replay covered 1,076 responses and 1,521
  deterministic repairs with zero introduced double closing tags; and
  `git diff --check` passed.
- Safety: only `yuxzhang/canon-zero-tim` is the approved push target. `main`
  was not checked out, modified, merged, or targeted. No cluster resource was
  launched or mutated.
- Claim ceiling: commit and CPU/replay evidence do not prove target TPU,
  Kubernetes cleanup, throughput, or completed data washing.

## 2026-08-14T00:00:02Z — P46.6 operator-branch read-back

- Type: publication read-back.
- Push result: `yuxzhang/canon-zero-tim` advanced from
  `c33ba5f50d606210ca9f2c94fca003b63ea6e326` to
  `9ec08b47b10f2663cc7649f6918627ff5d78a923` without force.
- Read-back: a fresh fetch returned the same exact
  `9ec08b47b10f2663cc7649f6918627ff5d78a923`; implementation
  `a989af34054434e6567f88e99b45ed67faf15a44` and its handoff commit are both
  ancestors. Local worktree was clean and had zero divergence afterward.
- Target discipline: `main` was never a refspec or checkout target. No cluster
  launch occurred. The next authorized action is to render the new-run-id
  `--full-campaign` manifest and obtain separate launch approval.

## 2026-08-14T00:00:03Z — P46.6 crash-safe resume-tag hardening

- Type: local implementation and recovery evidence; uncommitted and unpushed.
- Fact: the existing evaluator fsynced completed trajectory rows and recovered
  identities, but the JobSet layer still fetched a moving branch tip, reused
  setup state, overwrote `campaign.log`, accepted only legacy single-wave PASS
  markers, and did not reconcile R2E sandboxes after coordinator death.
- Action: separated stable `resume_tag` from per-launch `run_id`; added an
  immutable config-v4/trajectory-v6 resume contract, nonblocking single-writer
  lease, launch-isolated setup state, unique trajectory files, immutable
  attempt logs, exact full-campaign postflight, original-SHA checkout after
  ancestry verification, bounded R2E lifecycle patch activation in evaluation,
  same-tag orphan sandbox deletion, and runbook/handoff instructions.
- Recovery semantics: complete fsynced identities are skipped. Invalid
  infrastructure attempts retry consecutively. A trajectory killed before a
  complete row restarts from its beginning; no token-level continuation is
  claimed. Contract/source/topology drift and concurrent same-tag writers fail
  closed.
- Evidence: `P46_DEEPSWE_PROFILES_CPU_PASS cases=60`. Tests cover a first
  launch preserving 17/64 identities and a second launch running exactly the
  missing 47, torn-tail recovery across unique files, contract drift,
  exclusive lease, stable tag across distinct launch ids, pinned-SHA rendering,
  immutable attempt logs, legacy marker compatibility and exact 58-logical/
  29,616-identity campaign postflight.
- Adjacent gates: `P34_STATIC_PASS suites=10`,
  `P44_DEEPSWE_QWEN4B_PARITY_CPU_PASS`, Python compilation, shell syntax and
  `git diff --check` all pass.
- Claim ceiling: CPU evidence does not prove PVC locking semantics, Kubernetes
  orphan cleanup or a target 64/128-chip resume. No cluster action, commit or
  push occurred.
- Next: run adjacent P34/P44 and diff gates, then wait for explicit commit/push
  approval before any remote executor uses the new resume tag.

## 2026-08-14T09:23:34Z — P46.6 crash-safe resume and legacy-v5 adoption commit

- Type: implementation, transition decision, validation and publication
  preparation under explicit user commit/push approval.
- Remote audit: a fresh fetch found local base and
  `origin/yuxzhang/canon-zero-tim` exactly aligned at
  `2ec1cb768c7454c7d0ecf798ff1a5aff890ceae7` with divergence `0/0`. Main was
  neither checked out nor targeted.
- Operator constraint: do not stop the already running legacy-v5
  `p46e12805` campaign. No Kubernetes, JobSet, pod, PVC or cloud mutation was
  performed. Its archived first wave has 64 valid records, ten reward-one and
  54 reward-zero, at sampler source
  `18d5d2ac1603a26a221af9d5fc430b084ec002df`; the archived log reached the
  second physical wave. Current live cardinality is unknown.
- Implementation commit:
  `c3a960acdc94173440144559bb95f1de36d31537`.
- Resume action: stable `resume_tag` and per-launch `run_id`; immutable resume
  contract; single-writer lease; fsynced unique trajectory files; launch-local
  setup state; immutable attempt logs; exact full-campaign postflight; pinned
  checkout after branch-ancestry verification; and same-tag orphan sandbox
  cleanup.
- Legacy transition: a terminal v5 producer is copied, never moved, into
  `<resume-tag>/imports/<legacy-run-id>/`; `SHA256SUMS` freezes every JSONL.
  The importer verifies the derived v3 fingerprint for each logical shard,
  clean task order, sample nonce, attempt sequence, reward-only payload and
  provenance. It writes immutable v6 rows and an import receipt while
  preserving the old sampler SHA and separately pinning the new harness SHA.
  Live directories, cross-contract rows and imports after target evidence
  fail closed.
- Evidence: `P46_DEEPSWE_PROFILES_CPU_PASS cases=65`,
  `P34_STATIC_PASS suites=10`, and
  `P44_DEEPSWE_QWEN4B_PARITY_CPU_PASS` all pass. Python compilation, shell
  syntax and `git diff --check` pass. Tests include 17/64 + missing-47 resume,
  torn-tail recovery, contract/lease/attempt-log gates, exact 58-logical
  postflight, frozen snapshot digest/drift rejection, idempotent adoption, and
  per-logical-shard fingerprint preservation.
- Claim ceiling: no real 64/128-chip controlled kill/restart or legacy import
  has run. CPU evidence does not prove PVC `flock`, Kubernetes orphan cleanup,
  import throughput or completion of the 29,616-identity wash.
- Next: commit the synchronized runbook/handoff, push both commits only to
  `yuxzhang/canon-zero-tim`, and read back the exact remote SHA. Leave
  `p46e12805` untouched until natural termination.

## 2026-08-14T09:26:55Z — P46.6 publication read-back

- Type: publication checkpoint under explicit user commit/push approval.
- Pushed implementation `c3a960acdc94173440144559bb95f1de36d31537`
  and synchronized handoff `dc6b5b32a90ad0e12b1b9ae50ef7cc060b450abf`
  only to `origin/yuxzhang/canon-zero-tim`; `main` was neither checked out nor
  targeted.
- Fresh remote read-back resolved exactly to
  `dc6b5b32a90ad0e12b1b9ae50ef7cc060b450abf`, both commits passed ancestry
  checks, and local/remote divergence was `0/0`.
- No Kubernetes, JobSet, pod, PVC or cloud mutation occurred. The existing
  `p46e12805` producer remains untouched and must run to natural termination.
- Next: after terminal-state and no-producer proof, make a copied,
  digest-sealed, read-only legacy snapshot; never move or import the live
  directory. Require `LEGACY_IMPORT_PASS` before any resumed TPU model init.

## 2026-08-20T21:03:17Z — P46.6 returned campaign extraction audit

- Type: read-only latest-operator-branch evidence review. No Kubernetes,
  JobSet, pod, PVC, cloud, commit, push, or `main` mutation occurred.
- Baseline: clean review worktree at exact operator HEAD
  `eae3d6d47e07bbb631106284da40a5e90763faee`. Existing uncommitted P46 ledger
  edits in the older `p46_deepswe_32b_full` worktree were preserved and not
  rebased, stashed, reset, or overwritten.
- Returned package: commit `7fcae26e5a75dca14abdcfefc2796f2759b5cd2d`
  adds `p46r01a0_128chip_campaign_report.md`, one ad-hoc metrics JSON and files
  named full-RL, golden-SFT and DPO. All JSON lines parse and all 1,136 task
  identities join to the signed 1,851-row clean source at SHA-256
  `2f95c2e6df3526f68bd3eed3ab9aece7077ef85c74251c77f7b3474b0b307ed7`.
- Completion verdict: **INCOMPLETE**. The extraction contains 22,918 rows and
  1,136 tasks, versus the signed completion gate of 29,616 valid identities
  and 1,851 tasks. It has no global PASS marker, no 58 immutable logical
  summaries and no referenced digest manifest. The official finalizer rejects
  it with `campaign requires every logical summary: expected=58 actual=1`.
- Identity audit: the 22,918 rows collapse to 18,121 unique
  `(task_id, sample_idx)` identities; 4,797 rows are duplicate copies of an
  identity. At summary level, 1,123 tasks have indices 0-15, 13 represented
  tasks are incomplete, and 715 clean-source tasks are absent. Thirty-two
  distinct identities across four coveragepy tasks have invalid `FAILED`
  status and require retry under the signed classifier.
- Provisional curriculum audit: after deterministic identity deduplication and
  status-only classification, the 1,851-source view is 609 mixed 1/16-15/16,
  514 all-fail, zero all-pass and 728 incomplete. These are diagnostic counts,
  not publishable manifests, because raw trajectory validity/provenance and
  exact report digests are unavailable.
- Trajectory verdict: **INCONCLUSIVE**. Every row in the file named
  `deepswe_full_rl_22918.jsonl` has exactly five fields:
  `task_id`, `repo`, `sample_idx`, `status`, and `reward`. There are no turns,
  model responses, canonical actions, observations, patches, elapsed times,
  attempt indices, logprob-null evidence, action-compat diagnostics, source
  commit, config fingerprint, or cleanup evidence. The golden-SFT file is the
  same five-field summary restricted to reward one; the DPO file pairs two
  such summaries. They are not directly usable RL, SFT, or DPO examples.
- Internal consistency defects: the report states 4,374 DPO pairs while the
  metrics JSON and file contain 4,560; 865 DPO rows duplicate an exact pair.
  The 2,280 golden rows contain 432 duplicate identities, leaving 1,848 unique
  solved identities. The raw-row solve ratio 2,280/22,918 is therefore not the
  exact-N campaign metric; deduplicated valid summary rows give 1,848/18,089.
- Positive evidence and claim ceiling: outcome values are finite binary
  rewards; sample indices stay in 0-15; represented tasks all come from the
  clean source; the status mix is plausible. None of this proves full
  trajectory correctness, reward execution, Q4 sampler/model provenance,
  exact N16 washing, sandbox cleanup, or training-data readiness.
- Next: retrieve and seal the raw persistent campaign tree, resume every
  missing/invalid identity under the immutable tag, run all 58 logical report
  gates and the official finalizer, and only then publish digest-bearing
  `q4_learnable`, `q32_candidates`, `all_pass`, and `all_fail` manifests.

## 2026-08-20T21:26:42Z — P46.7 breadth-first census and frozen-v6 migration

- Type: local implementation under the user-approved phase workflow. No
  Kubernetes, JobSet, pod, PVC, cloud, commit, push, credential, or `main`
  mutation occurred.
- Trigger: strict resume prioritizes repeated retryable-invalid identities in
  the current wave and can delay coverage of the remaining clean prompts. The
  requested first pass instead needs every never-attempted identity sampled
  once; model timeout, context limit, max-step and signed trajectory timeout
  stay valid unsolved outcomes, while `FAILED`/environment/reward failures may
  be repaired later.
- Action: added default-off `--first-pass-census` /
  `CANON_P46_CENSUS_FIRST_PASS=1`. It is restricted to full reward-only Q4
  evaluation, is not part of the sampling fingerprint, skips any identity
  with a durable attempt, continues to later waves after a bounded timeout,
  and writes immutable coverage-only census summaries plus explicit deferred
  and unattempted identity lists below `outputs/census/`.
- Safety: strict mode, retry validity, exact-N aggregation and the official
  campaign finalizer are unchanged. Census invalids are neither solved nor
  coerced to reward zero. `CENSUS_PASS` requires all 1,851 tasks and 29,616
  identities attempted at least once but does not claim washed data.
- Migration: a new harness cannot append to old config-v4 evidence in place.
  Added an explicit frozen-v6 importer that requires a copied terminal
  `resume_contract.json`, all trajectory JSONLs and an exact `SHA256SUMS`,
  verifies every old contract/row/attempt, permits only harness SHA and fresh
  resume-tag changes, preserves `sampled_by=stock@<old source SHA>` and raw
  payloads, and emits immutable per-row migration provenance plus a receipt.
- Local evidence so far: relevant Python modules compile; 42 artifact,
  renderer and environment-contract tests pass, including idempotent
  invalid-to-valid v6 migration, sampling drift/fresh-tag rejection, and
  mutually exclusive import controls. Complete P46 and adjacent gates remain
  to be run after documentation synchronization.
- Next: synchronize HANDOFF/runbook and current gate markers; run complete P46
  CPU, adjacent P34/P44, static/diff checks; then await separate commit/push
  approval. A remote executor must not use this mode until an exact published
  SHA is read back from `origin/yuxzhang/canon-zero-tim`.

## 2026-08-20T21:29:50Z — P46.7 local release gates and execution handoff

- Synchronized `HANDOFF.md`, `P46_DEEPSWE_PROFILES_RUNBOOK.md`, state, plan,
  phase ledger and flag registry with an executable freeze/import/census/strict
  sequence. The handoff records the live-lineage boundary: a pre-P46.7
  `p46e12808`/`p46e12806` manifest does not acquire census behavior; its raw
  v6 tree may be copied only after a terminal/no-producer proof, or after a
  separately authorized operator stop.
- `bash canon-zero-tim/tests/p46_deepswe_profiles/run_cpu.sh` passes 75 tests
  and emits `P46_DEEPSWE_PROFILES_CPU_PASS cases=75`. Covered cases include
  29,616-identity strict orchestration, breadth-first invalid deferral,
  timeout-then-unattempted resume, census postflight, immutable census
  snapshots, frozen-v6 idempotence/provenance, fresh-tag and sampling-drift
  rejection, renderer/preflight controls and the unchanged strict finalizer.
- Adjacent gates pass:
  `P34_STATIC_PASS suites=10`, `P34_TRAJECTORY_CPU_PASS tests=5`,
  `P34_UPDATE_CPU_PASS tests=5`, and
  `P44_DEEPSWE_QWEN4B_PARITY_CPU_PASS`. `git diff --check` passes.
- Claim ceiling: these are CPU contract gates only. No old snapshot was copied,
  no TPU/vLLM/R2E workload ran, no trajectory was migrated on target, no
  census/strict campaign completed, and no commit/push/main/cloud mutation
  occurred.
- Next: wait for separate commit/push approval. After publication, the remote
  agent must read back the exact operator SHA, rerun `cases=75`, prove the old
  producer terminal/absent, seal the copied v6 snapshot, and require
  `FROZEN_V6_IMPORT_PASS` before the first census runtime starts.

## 2026-08-20T21:33:54Z — P46.7 implementation publication/read-back

- Under explicit user commit/push approval, committed the census scheduler,
  frozen-v6 migration, tests and synchronized execution documents as
  `365b46c1cd150839e3be1fd50adb33325fe3189f` and pushed only to
  `origin/yuxzhang/canon-zero-tim`.
- Pre-push fetch proved the operator remote and local base were exactly
  `eae3d6d47e07bbb631106284da40a5e90763faee` with divergence `0/0`; no
  conflict reconciliation or history rewrite was needed.
- Post-push fetch/read-back resolved both local HEAD and the operator remote to
  exact `365b46c1cd150839e3be1fd50adb33325fe3189f`, divergence `0/0`, with the
  implementation ancestry gate passing.
- `main` was never checked out, targeted, merged, rebased, or pushed. No
  Kubernetes, JobSet, pod, PVC, cloud resource or credential was touched.
- Next: publish this read-back ledger checkpoint, then the execution agent may
  follow HANDOFF P46.7 only after fresh branch ancestry, old-producer terminal
  and sealed-snapshot gates pass.

## 2026-08-21T01:08:25Z — P46.7 returned v5 snapshot incident repair

- Fast-forwarded the clean review worktree from
  `5f2d016147a55c032ea7b89b156a583d3b4ca7e8` to exact operator HEAD
  `91844a412cc288e18574e0812726263930726b12`. The only returned change was the
  incident report; no local changes were discarded and `main` was untouched.
- Exact returned failure: 128 TPU chips reached Running, but coordinator
  `p46c128a0` failed before model/runtime initialization while legacy import
  expected a fingerprint derived from the live harness SHA. The actual rows
  are trajectory-v5 with
  `sampled_by=stock@ac2c31bc7f6f82d33b3a62d62e1c390c8338b60e`; the launch
  omitted that explicit historical sampling SHA.
- The directory name `p46e12806-v6-final` was misleading. Import mode is now
  selected from actual row schema. A legacy-v5 snapshot must contain only raw
  trajectory JSONLs plus `SHA256SUMS`; a real frozen-v6 snapshot must also
  contain the matching immutable resume contract.
- `p46q4census01` is non-reusable incident evidence because old code wrote its
  incorrect immutable resume contract before import validation. Recovery uses
  a fresh tag such as `p46q4census02`; it never deletes, overwrites or repairs
  the failed tag.
- Local repair: renderer requires explicit `--sampling-source-commit` for
  either import mode; environment preflight rejects a v5 staging directory
  containing `resume_contract.json`; and the evaluator validates every sealed
  source row before acquiring the campaign lease or writing the destination
  resume contract. Wrong sampler, mixed schema and wrong importer fail before
  claiming the new tag.
- Resume semantics: a successful `LEGACY_IMPORT_PASS records=<actual>` adopts
  all validated durable identities, and census runs only identities absent
  from that imported set. The incident reports 510 raw records; a greater
  reusable count requires a greater sealed raw tree. The 22,918 five-field
  derived table cannot seed resume.
- Release evidence: `P46_DEEPSWE_PROFILES_CPU_PASS cases=77`,
  `P34_STATIC_PASS suites=10`, `P34_TRAJECTORY_CPU_PASS tests=5`,
  `P34_UPDATE_CPU_PASS tests=5`, and
  `P44_DEEPSWE_QWEN4B_PARITY_CPU_PASS`; Python/shell compilation and
  `git diff --check` pass. Regressions prove a wrong sampler does not create
  target `resume_contract.json`, a snapshot change after pre-lease validation
  is rejected, all legacy rows are checked, and 17 imported identities cause
  census to execute exactly the remaining 47 of a 64-identity wave.
- Claim ceiling: this proves CPU resume orchestration, not target PVC I/O,
  Pathways, R2E cleanup, 128-chip throughput or actual imported cardinality.
  No cluster, PVC, commit, push or credential mutation occurred.

## 2026-08-21T01:14:16Z — P46.7 v5 resume implementation commit

- Fresh fetch resolved local base, `FETCH_HEAD`, and
  `origin/yuxzhang/canon-zero-tim` to exact
  `91844a412cc288e18574e0812726263930726b12` with divergence `0/0`.
- Committed the pre-lease legacy-v5 validation, explicit sampler-lineage
  renderer gate, v5-only environment preflight, 77-case regressions and
  synchronized incident/runbook/handoff as implementation
  `f823bb6a9aabf023e651788452d94ff656c827e1`.
- `main` was neither checked out nor targeted. No cluster, PVC or credential
  mutation occurred. Remote execution still requires fresh operator-branch
  ancestry/read-back and separate launch authority.

## 2026-08-21T02:39:05Z — P46.7 sealed legacy source contract hardening

- Clean preflight passed on local branch `local/p46-results-review-0820` at
  exact base `6c3ab1f2d2ffeaf47667c07fc4151532574e6279`. Review found that this
  base accepted any syntactically valid legacy fingerprint and did not bind
  `run_tag`, while the execution docs still pinned older `f823bb6a`.
- Replaced that broad relaxation with
  `canon.p46.deepswe-eval.legacy-source-contract.v1`. Stable Q4 model, exact
  dataset/whitelist, N16/16K/50-step sampling, timeout, action, RNG and topology
  facts must match. Each observed logical shard has one opaque historical
  fingerprint/run-tag cohort and exact cardinality. Absolute historical paths,
  destination harness/tag and the unrecorded old client image are deliberately
  outside stable semantics.
- Added `seal_p46_legacy_v5_snapshot.py`. It reads the reviewed 1,851-task
  order, rejects mixed cohorts, writes deterministic
  `legacy_source_contract.json`, and seals it plus every JSONL in
  `SHA256SUMS`. Environment preflight now requires both seal files.
- Import provenance/receipt schema advances to v2 and records the source
  contract digest and sealed cohorts. Missing/tampered contracts, semantic
  drift, mixed cohort, file/cardinality drift and wrong sampler all fail before
  target resume-tag creation/runtime.
- `bash canon-zero-tim/tests/p46_deepswe_profiles/run_cpu.sh` ran 79 tests and
  passed. New regressions prove historical path drift is admitted, mixed
  fingerprint/run-tag cohorts are rejected, and an old trajectory-only
  `SHA256SUMS` is not a valid import seal. The terminal marker in the runner is
  updated to `P46_DEEPSWE_PROFILES_CPU_PASS cases=79`.
- Adjacent release gates pass:
  `P34_STATIC_PASS suites=10`, `P34_TRAJECTORY_CPU_PASS tests=5`,
  `P34_UPDATE_CPU_PASS tests=5`, and
  `P44_DEEPSWE_QWEN4B_PARITY_CPU_PASS`.
- Synchronized the runbook, incident recovery, HANDOFF, state, plan and active
  phase. Remote execution is explicitly blocked until this candidate is
  committed/pushed under separate authority and read back by exact operator
  SHA. No commit, push, cluster, PVC, credential or `main` mutation occurred.

## 2026-08-21T02:49:00Z — sealed-contract implementation publication

- Under explicit user commit/push authority, created implementation commit
  `9cebe0d1671f6da1748bc53ed0da07a5f970fb37` on exact remote base
  `6c3ab1f2d2ffeaf47667c07fc4151532574e6279` after fetch proved divergence
  `0/0`.
- Updated operator documents to require `9cebe0d1` in freshly fetched branch
  ancestry while using the exact read-back branch HEAD as launch source SHA.
- Publication targets only `yuxzhang/canon-zero-tim`. No cluster, PVC,
  credential or `main` mutation is authorized or performed by this checkpoint.
