# Plan

## Outcome

Create a separate production-shaped FrozenLake no-eval full-training carrier
on the same 64-chip v5p slice using DP8xTP8 and device-resident Adam state. Preserve the
global optimization contract (32 prompts, 8 generations, 256 trajectories,
learning rate `1e-6`, 450 steps) and the local canonical logprob shape M256.
Do not mutate or promote the existing DP16xTP4 carrier-debug evidence.

## Phases

| Phase | Deliverable | Exit gate | Status |
|---|---|---|---|
| P45.1 | DP8xTP8 workload, profile, recipe/adapter generalization, and isolated full/eval renderer entries | focused Python and renderer tests pass; old DP16xTP4 renders remain contract compatible | completed |
| P45.2 | Local static and render admission | generated manifests attest DP8xTP8, M2048/M256, 32 local trajectories, resident optimizer, evaluation selection, and hard safety gates | completed |
| P45.2b | Isolated Qwen3-8B TP8 engine overlay and exact-image admission | target image installs `qwen8b_tp8`; seven TP8 projection shapes, canonical forward/VJP, manifest integrity, and TP4 rejection all pass | completed |
| P45.3a | GCS checkpoint/resume admission | fresh/resume manifests are fail-closed, save every 10 committed steps with `LatestN(1)`, restore actor/Adam/step metadata, and sync the restored actor into vLLM before the first rollout | local implementation complete; target pending |
| P45.3b | Long-run host-memory and warm-step hardening | P45 alone renders a 350G `jax-tpu` limit; evaluation remains exactly once per policy cadence; cgroup/RSS evidence is emitted at eval and committed-step boundaries; large per-step references are released before bounded GC; the wired P32 grouped report adjoint is enabled | local implementation complete; target pending |
| P45.3c | Explicit no-eval full training plus checkpoint continuation | FULL renders cadence 0; a real one-host v5p model/Adam roundtrip passes; a fresh 64-chip step 10/11 and identical-source resume pass without any evaluation marker | local implementation complete; target pending |
| P45.3 | 64-chip no-eval full target run | first real update reports device-resident state, optimizer H2D/D2H zero, finite update, HBM headroom, online W&B, and continued training with no evaluation; step 10 publishes one restorable GCS checkpoint; identical-source resume restores step 10 and commits step 11; host memory remains below its limit with a measurable post-GC trend | pending on P45.3a/P45.3b/P45.3c |

## Decisions

- Confirmed: P41 ran one real Qwen3-8B resident update on DP1xTP4 without OOM, but peak HBM left only 4.52 GiB per chip and did not admit the production default.
- Confirmed: P33 and P45 remain distinct DP16xTP4 and DP8xTP8 carriers. Optimizer placement defaults were later changed to resident across profiles, so topology/profile/overlay identity must be checked independently of placement.
- Decision: introduce a new DP8xTP8 resident carrier instead of changing the existing P33 debug or full entries.
- Decision: preserve global batch, generations, learning rate, sequence limits,
  and step budget. The current full-training target changes only evaluation
  cadence to 0. Set the topology-derived global trajectory microbatch to DP8 so
  every rank still contributes one trajectory per fixed group; this produces
  32 ordered gradient groups instead of the DP16 carrier's 16.
- Decision: full-training alignment remains warning-only for observed A/B/C drift, but non-finite values, invalid placement, failed transaction, replica mismatch, and OOM remain hard failures.
- Confirmed: `p45r3` selected the existing `qwen8b` engine overlay and failed before rollout because that overlay is deliberately TP4-only. The prior P45 CPU/render gate did not install or import the model overlay and therefore could not detect this gap.
- Decision: preserve `qwen8b` unchanged for the admitted TP4 path. Add a separate `qwen8b_tp8` overlay with TP8-local projection shapes `(4096,512)`, `(4096,128)`, `(512,4096)`, `(4096,1536)`, and `(1536,4096)`, using BM/BN/BK `128/128/128`. Since 1536 divides both 128 and 256, no matmul or SwiGLU feature padding is admitted.
- Hypothesis: TP8 materially increases per-chip HBM reserve and resident placement removes optimizer host transfers; end-to-end speedup remains unproven because TP8 communication and 32 rank-local gradient groups may offset part of the gain.
- Decision: P45 checkpoints use the dedicated durable root
  `gs://yuxzhang-tunix-models/canon-zero-tim/checkpoints/frozenlake`, a stable
  operator-supplied campaign tag, a fixed interval of 10 committed updates,
  and `LatestN(1)`. JobSet run IDs identify launch attempts and must not be
  used as resume identity.
- Decision: checkpoint mode is explicit. `new` refuses an existing complete
  checkpoint; `resume` refuses an empty prefix or a mismatched source/config
  contract. Resume restores actor parameters, Adam state and global step, then
  performs and attests one actor-to-vLLM weight sync before any rollout.
- Decision: forced close-time checkpoints are disabled for P45. Otherwise an
  off-interval graceful exit could save step 17 and, under `LatestN(1)`, delete
  the last admitted step-10 recovery point.
- Confirmed: P45 uses the agentic evaluation path, which materializes the
  complete held-out rollout and guards it with `_last_eval_train_step`; the
  base `RLLearner` evaluation queue is not on this workload's execution path.
  P45.3b preserves that semantic and adds a focused exactly-once regression
  gate rather than rewriting generic evaluation.
- Confirmed: P45r7 exposed a stronger incompatibility than the earlier
  exactly-once analysis covered. Streaming evaluation feeds completed groups
  into canonical prefill rescore while other requests remain live, but that
  rescore requires a driver-wide idle prefix-cache reset. The inner reset
  timed out after 300 seconds; `eval_future.result()` only propagated it.
- Decision: current full training uses no in-training evaluation. The FULL
  manifest explicitly passes cadence 0 and the learner defines every
  nonpositive cadence as disabled. The EVAL manifest is retained only as a
  quarantined future repair target.
- Decision: do not directly resume the P45r7 checkpoint into the new no-eval
  source. Its exact metadata freezes the old source and cadence 10; a direct
  restore must fail closed. Start a new tag unless a separate migration is
  reviewed.
- Decision: the P45 renderer, not the shared 64-chip base manifest, raises the
  `jax-tpu` memory limit from 200G to 350G. The larger limit is crash headroom,
  not evidence that the p45r5 growth mechanism is fixed.
- Decision: P45 emits cgroup and process-RSS snapshots at evaluation and
  committed-step boundaries, drops per-step rollout/evaluation references,
  and runs Python cyclic GC once per committed step. Do not call
  `jax.clear_caches()` in the production loop.
- Decision: enable `CANON_P28_BATCHED_REPORT=1` because its compiled adjoint is
  implemented in P45's P32 grouped reverse path. Do not enable
  `CANON_BATCHED_EVIDENCE` or `CANON_P28_BATCHED_REVERSE` until their P32
  grouped mirrors and verify gates exist; setting them now would be a
  non-functional performance claim.
- Claim boundary: p45r5 proves sustained resident training through step 47 and
  a 200G host OOM. Its archived log does not contain a complete RSS/cgroup
  timeline, so trajectory, logging, and compilation-cache accumulation remain
  hypotheses until P45.3b target evidence separates them.
- Claim ceiling: resume is committed-step training continuation. It does not
  restore an in-flight rollout or the vLLM sampling RNG and is not a bitwise
  continuation of the interrupted trajectory stream.
