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
