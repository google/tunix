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
