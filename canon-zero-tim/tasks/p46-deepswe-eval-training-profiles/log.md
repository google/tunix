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
- Next: remote execution reads back the exact operator HEAD containing `e1b40093`, then starts with one Q4 clean-evaluation shard on whichever 64/256 topology is available.

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
