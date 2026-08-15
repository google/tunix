# Log

## 2026-08-12 UTC — P45.1: bind DP8xTP8 resident FrozenLake

- Type: decision
- Fact: `render_p33_jobsets.py`, `dp_workloads.py`, the FrozenLake recipe, and the P33 segmented path currently encode DP16xTP4. The renderer also forces `CANON_OPT_STATE_RESIDENT=0`.
- Action: bound an isolated P45 task for a new 64-chip DP8xTP8 full/eval carrier. Existing DP16xTP4 debug and offload entries remain unchanged.
- Command: omitted
- Result: contract frozen before implementation; no TPU/cloud run or training mutation occurred.
- Files/artifacts: `state.md`; `plan.md`; `phases/p45-1-contract-and-renderer.md`
- Rollback: remove only the new P45 profile/spec and workload registration; do not alter the existing P33 profiles.
- Next: implement and run focused CPU/render gates.

## 2026-08-12 UTC — P45.1: separate the coincident DP16 cadences

- Type: correction
- Fact: the old value `16` represented three different quantities at once: DP size, global trajectory microbatch, and local gradient-group count. Under DP8 the correct values are 8, 8, and 32 respectively.
- Action: made the trajectory microbatch topology-derived, made the optimizer transaction length follow `CANON_LOCAL_TRAJECTORIES`, and made the learner expect 32 fixed rank-major groups for the P45 workload.
- Command: omitted
- Result: P45 preserves one trajectory per rank per group and the original global batch of 256 trajectories; no target run occurred.
- Files/artifacts: `dp_workloads.py`; `agentic_rl_learner.py`; `peft_trainer.py`; P45 plan and phase file
- Rollback: remove the P45 workload/profile; DP16 continues to resolve every affected cadence to 16.
- Next: rerun focused and full adjacent CPU gates.

## 2026-08-12 UTC — P45.1/P45.2: implement and admit locally

- Type: implementation and validation
- Fact: under DP8, global trajectory microbatch 8 and local gradient-group count 32 are different quantities. Reusing local trajectories as the outer microbatch would have failed the learner before backward.
- Action: registered the isolated workload/profile, generalized topology-aware recipe/adapter/learner/classifier contracts, kept 32 ordered groups per update, added resident optimizer timing checks, and wrote an exact operator handoff.
- Commands: fixed-image `canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_cpu.sh`; fixed-image `canon-zero-tim/tests/p33_workloads/run_cpu.sh`; merged-profile `00_env.sh` preflight; `git diff --check`; shell syntax checks.
- Result: on the final source, P45 77-test suite PASS; alignment 29/29 PASS; merged profile PASS with DP8xTP8, local trajectories 32, global M2048, optimizer resident, eval on, warning-only on; complete adjacent P33/P38 CPU gate PASS.
- Files/artifacts: `HANDOFF.md`; `cluster/render_p45_frozenlake.py`; `cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-resident.env`; `tests/p45_frozenlake_dp8_tp8/`; phase files.
- Rollback: stop using the isolated P45 renderer/profile. Existing P33/P38 workload entries and target evidence remain separate.
- Next: 64-chip target run; capture first-update HBM, placement, timing, evaluation, and complete raw evidence.

## 2026-08-12 UTC — publish P45 implementation

- Type: publication
- Fact: the focused fixed-image gate passed after rebasing onto remote commit `115ef814`; the remote DeepSWE changes did not overlap the P45 implementation.
- Action: published the implementation, tests, renderer/profile, and operator handoff to `yuxzhang/canon-zero-tim`.
- Result: implementation commit `fae4e67f` is the minimum P45 source anchor; no target JobSet was launched by this publication.
- Files/artifacts: `HANDOFF.md`; `state.md`; isolated P45 renderer/profile/tests.
- Rollback: stop rendering the isolated P45 carrier; the existing DP16xTP4 entries remain available.
- Next: execute P45.3 on one 64-chip slice and return the complete evidence bundle.

## 2026-08-12 UTC — make P45 the explicit resident operator route

- Type: documentation correction
- Fact: the P42 evaluation runbook still selected `render_p33_jobsets.py`, which deliberately resolves DP16xTP4 with `CANON_OPT_STATE_RESIDENT=0` and `CANON_P30_OPT_STATE_OFFLOAD=1`. Its title made it easy to mistake that historical evaluation carrier for the new resident full-training route.
- Action: added a dedicated P45 operator runbook, marked P42 as the legacy/offload route, linked the P45 handoff and active target phase to the new entry point, and documented the runtime placement markers that distinguish the two carriers.
- Result: the fixed-image gate passed 77 workload/renderer tests and 31 alignment tests, followed by `[P45.PROFILE] ADMITTED_PREFLIGHT_PASS` and `[P45.FROZENLAKE] CPU_GATE PASS`. The operator path now fails conceptually before launch: new resident full/eval runs select only `render_p45_frozenlake.py`; a `pinned-host-offload` runtime marker is explicitly classified as selection of the wrong carrier. A host-Python attempt was non-authoritative because that environment lacks `datasets` and `metrax`; the required pinned image contains the reviewed dependencies. No JobSet was launched and no placement code changed.
- Files/artifacts: `../../cluster/P45_FROZENLAKE_RESIDENT_RUNBOOK.md`; `../../cluster/P42_FROZENLAKE_EVAL_RUNBOOK.md`; `HANDOFF.md`; `state.md`; `phases/p45-3-target-run.md`
- Rollback: remove only the P45 routing documentation and restore the P42 banner; renderer/profile behavior is unchanged.
- Next: run documentation/render gates, then execute P45.3 and capture first-update resident/HBM evidence.

## 2026-08-12 UTC — P45 r3 invalidates model-overlay admission

- Type: failed target gate and re-plan
- Fact: `p45r3` from source `b26135f2` resolved the intended DP8xTP8 P45 profile but inherited `CANON_MODEL_DIR_NAME=qwen8b`. That overlay is TP4-only and rejected `CANON_QWEN3_TP_SIZE='8'` during model import.
- Result: exit 1 before rollout with `PATHTRACE=0`. The attempt contains no evidence about resident optimizer HBM, evaluation, or training. The previous CPU/render gate was insufficient because it never installed/imported the selected engine overlay.
- Files/artifacts: `../../debug_logs/p45_p45r3_frozenlake_resident.raw.log`; `phases/p45-2b-qwen8b-tp8-overlay.md`
- Decision: add an isolated TP8 overlay and an exact-image contract/forward/VJP/negative gate; preserve the existing TP4 overlay.
- Rollback: no production defaults changed by this checkpoint.
- Next: complete P45.2b locally before re-admitting P45.3.

## 2026-08-12 UTC — P45.2b exact-image admission complete

- Type: implementation and validation
- Action: added the isolated `qwen8b_tp8` model overlay with TP8-local projection contracts and BM/BN/BK `128/128/128`; bound only the P45 profile to it; added static, installed-chain import, seven-site shape, TP4-negative, and Pallas forward/VJP gates. Corrected two stale P33 renderer tests to the already-approved resident optimizer default while preserving an explicit offload-drift negative.
- Commands: `bash canon-zero-tim/tests/p45_frozenlake_dp8_tp8/run_exact_image.sh`; complete pinned-image `canon-zero-tim/tests/p33_workloads/run_cpu.sh`; model SHA verification; `git diff --check`.
- Result: exact image ID `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`; 29 installed files matched; `linear_p22xk` TP8 import PASS; seven projection sites PASS; no padding; TP4 negative PASS; Pallas interpret forward/VJP exact; P45 83 tests plus 31 alignment tests PASS; full adjacent P33 CPU gate PASS.
- Claim boundary: this closes the r3 model-overlay wiring failure only. It does not prove 64-chip model execution, resident optimizer HBM, evaluation, or training.
- Rollback: stop selecting `qwen8b_tp8` in the P45 profile and remove only the new overlay/tests; the TP4 `qwen8b` overlay was not modified.
- Next: fetch the published branch head, launch one fresh P45 target attempt, and capture the first committed update.

## 2026-08-15 UTC — P45r5 sustained training milestone (step 47 / align 1535) and host OOM diagnosis

- Type: target hardware execution and capacity milestone
- Command: `canon-p45-fl-eval-p45r5-42139ffa` on 64 TPU (`DP8xTP8`, resident optimizer, concurrency 256).
- Result: ran continuously for ~60 hours (2.5 days), completing 47 training steps and 1535 alignment checks (1535/1535 PASS).
- Termination: pod terminated at `Sat, 15 Aug 2026 06:03:58 UTC` with `Exit Code: 137 (OOMKilled)` on `jax-tpu` container.
- Root Cause: Linux host memory cgroup limit `memory: 200G` exceeded due to multi-day accumulation of trajectory objects, JAX compilation cache, and logging buffers. TPU HBM remained fully healthy.
- Recommendation: for future continuous multi-day runs, increase head pod memory limit to 350GiB+ and introduce periodic Python GC in the training step loop.
