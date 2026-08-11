# Log

## 2026-08-11 — P41.4 started

- Reopened the task for a separate FrozenLake/Qwen3-8B capacity admission.
- Pre-registered a resident-only DP1xTP4 one-update canary. It preserves the
  strict FrozenLake alignment gate and does not change the production default.
- Added a fail-closed classifier requiring four active gradient microbatches,
  one valid commit, device-resident optimizer state before and after commit,
  finite/nonzero gradients, unchanged reference/reset accumulator, DP replica
  equality, timing evidence, and TPU HBM snapshots.
- Added a self-contained local-wheel runtime admission for Gymnasium so the
  image's protected numerical stack cannot change during the canary.

## 2026-08-11 — P41.4 result

- `p41fl1` completed one real Qwen3-8B resident update, engine weight sync,
  and canonical postflight without OOM.
- All four alignment records were exact across A/B/C. The aggregate gradient
  was finite/nonzero and changed 6,934,505,968 parameter elements.
- Optimizer state remained on device; H2D and D2H were zero. The optimizer
  transaction took 56.99 seconds.
- Peak HBM was 97,955,232,768 of 102,803,437,568 bytes per chip, leaving only
  4.52 GiB.
- The phase is not admitted because only one of four stochastic microbatches
  produced nonzero advantage/gradient, violating the pre-registered 4/4
  activity gate. The observed aggregate update is recorded separately and the
  threshold was not changed after the run.
- Recommendation: retain pinned-host offload as FrozenLake production default.

## 2026-08-11 UTC — P41.1: bind the optimizer-residency experiment

- Type: decision
- Fact: `TrainingConfig.optimizer_offload=False` keeps optimizer state on device; the canonical GSM8K and FrozenLake recipes currently force `True` and the P33 classifier requires `pinned_host`.
- Action: bound a default-off, evidence-driven placement experiment for both recipes, with GSM8K as the first hardware canary.
- Command: omitted
- Result: task active; no source change or TPU workload has run yet.
- Files/artifacts: `state.md`, `plan.md`, `phases/p41-1-placement-and-canary.md`
- Rollback: keep the new flag unset or `0` to retain the existing behavior.
- Next: implement and CPU-test the placement contract.

## 2026-08-11 UTC — P41.1: optimizer placement contract passes CPU gates

- Type: code change
- Fact: both workload profiles now default to pinned-host offload and accept an explicit, mutually exclusive device-resident candidate; both recipes consume the selected placement.
- Action: added placement validation, runtime before/after attestation, classifier support, bounded state fingerprints, optimizer phase timing, profile controls, and a one-host pair runner.
- Command: `sudo docker run --rm ... tunix_frozenlake_image:vllm-tpu0.25.0 bash canon-zero-tim/tests/p33_workloads/run_cpu.sh`
- Result: exact-image P33 gate 73/73 PASS; focused workload 43/43, renderer 12/12, classifier 13/13, P41 classifier 2/2, profile 2/2, and two-commit offload-vs-device bitwise equivalence 1/1 PASS.
- Files/artifacts: source diff; `scripts/run_onehost_pair.sh`; `scripts/classify_onehost_pair.py`
- Rollback: keep `CANON_OPT_STATE_RESIDENT=0`; remove only P41 placement/timing fields and restore recipe offload expressions if the hardware gate fails.
- Next: run the bounded DP1xTP4 GSM8K pair.

## 2026-08-11 UTC — P41.2: first launch refused before TPU startup

- Type: correction
- Fact: attempt `p41a1` exited after preflight because the runner required the host HF snapshot directory to contain `config.json`, although the run intentionally bind-mounts the signed local model directory onto that snapshot path inside the container.
- Action: replaced the invalid host-file assertion with the established 40-character snapshot-ref validation used by the existing one-host runner.
- Command: `bash canon-zero-tim/tasks/p41-optimizer-residency/scripts/run_onehost_pair.sh p41a1`
- Result: exit 1 before evidence-directory creation, Docker startup, or TPU access; no numerical result.
- Files/artifacts: none; the failed precondition occurred before the artifact directory was created.
- Rollback: restore the assertion only if the runner stops bind-mounting the complete local model directory.
- Next: rerun with a new label.

## 2026-08-11 UTC — P41.2: second launch exposed the legacy L3 geometry mismatch

- Type: correction
- Fact: attempt `p41a2` completed rollout and exact pre-update alignment, then the segmented-update gate rejected two trajectories because its legacy default expected `8->4x2`.
- Action: added the default-off `CANON_P41_OPTIMIZER_BENCH=1` contract. It accepts only the bounded GSM8K L3 update canary and fixes this comparison at `2->1x2`; P33, P34, and runs without the flag retain their existing contracts.
- Command: `bash canon-zero-tim/tasks/p41-optimizer-residency/scripts/run_onehost_pair.sh p41a2`
- Result: `logp_diff=(0,0)`, `prob_diff=(0,0)`, and importance weight exactly one before the fail-closed geometry rejection. No backward or optimizer commit ran, so this is not an optimizer or OOM result.
- Files/artifacts: `/mnt/disks/tunix-data/logp_probe_1host/p41_optimizer_p41a2/offload/raw.log`
- Rollback: unset `CANON_P41_OPTIMIZER_BENCH`; the legacy gate again rejects the two-trajectory geometry.
- Next: run the CPU negative controls, then rerun both hardware arms with a new label.

## 2026-08-11 UTC — P41.2: third and fourth launch corrections

- Type: correction
- Fact: `p41a3` was killed with exit 137 immediately after TPU discovery following a forced container stop. Host memory had 429 GiB available and a subsequent four-device TPU compile/execute smoke test passed, so it is retained as an infrastructure-only transient. `p41a4` then reached exact rollout alignment but exposed that the recipe CLI left `train_trajectory_micro_batch_size=None`.
- Action: explicitly pass the existing `--train_trajectory_micro_batch_size=2` argument and include actual geometry values in future fail-closed errors.
- Result: neither attempt reached backward or optimizer commit; neither is an optimizer placement result.
- Files/artifacts: `/mnt/disks/tunix-data/logp_probe_1host/p41_optimizer_p41a3/`; `/mnt/disks/tunix-data/logp_probe_1host/p41_optimizer_p41a4/`.
- Rollback: remove the explicit trajectory microbatch argument together with the P41 benchmark contract.
- Next: rerun the two hardware arms under a new label after the TPU runtime smoke test.

## 2026-08-11 UTC — P41.2: attempts p41a5 through p41a9 reach the real reverse path

- Type: correction
- Fact: attempts `p41a5` through `p41a8` exposed three independent geometry/profile readers that still held legacy defaults. After synchronizing the bounded P41 geometry and the signed GSM8K profile, `p41a9` completed exact rollout alignment and all 28 reverse layers.
- Result: `p41a9` then failed before leaf accumulation because the ordinary depth-1 optimization had allocated an empty gradient accumulator for the one-microbatch precomputed transaction. No optimizer commit ran and this is not a numerical or OOM result.
- Files/artifacts: `/mnt/disks/tunix-data/logp_probe_1host/p41_optimizer_p41a5/` through `/mnt/disks/tunix-data/logp_probe_1host/p41_optimizer_p41a9/`.
- Rollback: unset `CANON_P41_OPTIMIZER_BENCH`; all legacy transaction lengths remain unchanged.
- Next: require a model-shaped accumulator for every explicit segmented G6 transaction, regress the ordinary depth-1 fast path, and relaunch under a new label.

## 2026-08-11 UTC — P41.2: single-microbatch accumulator regression passes

- Type: code change
- Fact: the precomputed segmented transaction always consumes `GradientAccumulator`, even when its transaction length is one; the ordinary non-packing depth-1 path does not.
- Action: made accumulator allocation depend on the explicit segmented G6 contract as well as ordinary accumulation depth, and added a one-microbatch commit regression.
- Command: `python3 -m pytest -q tests/sft/peft_trainer_test.py -k 'p41_ or p28_g6_precomputed_four_microstep_update or p30_optimizer_offload_matches_two_device_commits or accumulator_grads_skipped_for_depth1 or accumulator_grads_allocated_when_used'`
- Result: 7 passed. The fix changes allocation only; gradient values, dtype, sharding, order, and optimizer arithmetic are unchanged.
- Files/artifacts: `tunix/sft/peft_trainer.py`; `tests/sft/peft_trainer_test.py`.
- Rollback: remove `_requires_precomputed_gradient_accumulator` and the P41 benchmark contract together.
- Next: run a minimal TPU smoke test, then launch `p41a10`.

## 2026-08-11 UTC — P41.2: p41a10 completes the offload update before report failure

- Type: correction
- Fact: `p41a10` passed exact A/B/C alignment for 128 action tokens, completed all 28 reverse layers, accumulated a finite nonzero gradient, and completed the pinned-host H2D, Adam commit, and D2H transaction. The run then failed while serializing evidence because the DP1 adapter did not emit the DP-reduction identity fields expected by the shared report schema.
- Result: the optimizer computation succeeded and `train_steps=1`; the evidence report did not land, so this attempt is not a completed offload arm and the resident arm did not start. No OOM occurred.
- Files/artifacts: `/mnt/disks/tunix-data/logp_probe_1host/p41_optimizer_p41a10/offload/raw.log`.
- Action: make the DP1 adapter explicitly report a single-replica identity contract (`replica_equality=true`, zero DP reductions, one rank pullback) instead of relying on an implicit default.
- Rollback: remove the DP1 report fields together with the shared P41 classifier; no model or optimizer math changes.
- Next: run the adapter regression and relaunch with a fresh label.

## 2026-08-11 UTC — P41.2: p41a11 completes both placements but not a controlled pair

- Type: expensive run
- Fact: both `p41a11` arms independently passed exact A/B/C alignment, one finite nonzero reverse transaction, one real parameter update, accumulator reset, reference immutability, and placement attestation.  Device residency did not OOM.
- Result: pinned-host offload used 47.8987 seconds for the optimizer transaction versus 39.1333 seconds resident (1.224x); the full reverse-plus-commit window improved from 168.3911 to 159.7685 seconds (1.054x).  Resident peak HBM was 34,676,120,576 bytes per chip versus 33,093,301,760 bytes offloaded, an increase of 1,582,818,816 bytes per chip.
- Correction: the pair classifier correctly failed because the two independent processes sampled different token sequences (`b9eb...` versus `78ca...`), producing different gradients and updates.  This is an uncontrolled-input result, not evidence of placement-dependent numerical drift.
- Files/artifacts: `/mnt/disks/tunix-data/logp_probe_1host/p41_optimizer_p41a11/`; `offload/update.json` SHA-256 `ec2bc076569e0da6da41e5aacd01754fd565710633f9f1f223feaf2a6817ade1`; `resident/update.json` SHA-256 `07c35d78874aa78ca1272bdd9f3fba5b91ec9cfef9d1a0cd5e3ef975e63f7`.
- Rollback: keep `CANON_OPT_STATE_RESIDENT=0`; no production placement changed.
- Next: freeze the rollout seed only in the bounded P41 benchmark and rerun the pair.

## 2026-08-11 UTC — P41.2: p41a12 rejects unsupported per-request seed

- Type: correction
- Fact: the first `p41a12` arm reached the real TPU vLLM request path, where the pinned JAX backend rejected `SamplingParams.seed` with `ValueError: JAX does not support per-request seed.`
- Result: no trajectory, backward pass, or optimizer update ran; this is not a numerical, placement, or OOM result.
- Action: use greedy sampling only in the bounded P41 benchmark to make both processes consume the same rollout, rather than changing production sampling or patching the backend's seed contract.
- Files/artifacts: `/mnt/disks/tunix-data/logp_probe_1host/p41_optimizer_p41a12/offload/raw.log`.
- Rollback: unset `CANON_P41_OPTIMIZER_BENCH`; production GSM8K retains temperature 1.0 and no per-request seed.
- Next: run the focused CPU gate and relaunch with a new label.

## 2026-08-11 UTC — P41.2: p41a13 rejects non-neutral top-k

- Type: correction
- Fact: `p41a13` accepted temperature-zero greedy mode, completed engine compilation, and then the canonical adapter rejected `top_k=1` because its differentiable score path admits only neutral top-k/top-p transforms.
- Result: no trajectory, backward pass, or optimizer update ran; this is not a numerical, placement, or OOM result.
- Action: retain greedy behavior through `temperature=0.0`, while keeping the adapter-compatible neutral transforms `top_k=0` and `top_p=1.0`.
- Files/artifacts: `/mnt/disks/tunix-data/logp_probe_1host/p41_optimizer_p41a13/offload/raw.log`.
- Rollback: unset `CANON_P41_OPTIMIZER_BENCH`; production GSM8K retains its original temperature and transforms.
- Next: rerun the focused gate and hardware pair with a new label.

## 2026-08-11 UTC — P41.2: p41a14 rejects zero-temperature gradient

- Type: correction
- Fact: `p41a14` passed exact A/B/C alignment with greedy rollout and completed all 28 reverse layers, but the differentiable processed-logprob temperature transform produced a non-finite microgradient at temperature zero.
- Result: G6 rejected `active=True norm=nan` before optimizer commit.  This is a valid negative result: greedy sampling cannot serve as the placement-equivalence workload because it changes the gradient transform.
- Action: restore the production temperature and neutral transforms.  Make the P41 workload repeatable with an explicit vLLM engine seed and serial request scheduling (`max_concurrency=1`), neither of which changes gradient arithmetic.
- Files/artifacts: `/mnt/disks/tunix-data/logp_probe_1host/p41_optimizer_p41a14/offload/raw.log`.
- Rollback: unset `CANON_P41_OPTIMIZER_BENCH`; production scheduling and engine defaults remain unchanged.
- Next: pass the focused gate and launch a new pair.

## 2026-08-11 UTC — P41.2/P41.3: p41a15 passes the controlled hardware pair

- Type: phase gate
- Fact: an explicit vLLM engine seed plus serial request scheduling produced identical temperature-1 trajectories without changing sampling transforms or gradient arithmetic.
- Result: both DP1xTP4 Qwen3-1.7B arms passed exact A/B/C for 128 action tokens, finite nonzero gradient norm `17.8889`, one real update, reference immutability, accumulator reset, and placement attestation.  Token hash, microgradient, commit evidence, and final model/optimizer fingerprints were bitwise equal.  Device residency did not OOM.
- Performance: optimizer transaction `46.9757s -> 39.1262s` (1.2006x); measured reverse-plus-commit `168.0918s -> 159.1951s` (5.29 percent lower); peak HBM `33,093,301,760 -> 34,676,120,576` bytes per chip (+1.47 GiB).
- Command: `bash canon-zero-tim/tasks/p41-optimizer-residency/scripts/run_onehost_pair.sh p41a15`
- CPU regression: exact-image P33 `74/74`, alignment `28/28`, profile `10/10`, `git diff --check`, and `py_compile` passed.
- Files/artifacts: `/mnt/disks/tunix-data/logp_probe_1host/p41_optimizer_p41a15/`; pair SHA-256 `cbda39947f729c2526f2ade074360f91705766712609ac106472d42f46af9599`; offload evidence SHA-256 `73ecc99b6f45ad3571fd0978ea3c2611ad46883b2f617f67178e88a7db931df1`; resident evidence SHA-256 `a8c9e585e893016f4fae8bdc89ad6d4844fa950999541d9b3a250d2257b452d7`.
- Claim boundary: GSM8K device residency is admitted as a default-off candidate for this one-update geometry.  FrozenLake wiring is present, but Qwen3-8B capacity and backward behavior remain untested; no production default changed.
- Rollback: set `CANON_OPT_STATE_RESIDENT=0` and `CANON_P30_OPT_STATE_OFFLOAD=1` to retain pinned-host offload.
- Next: none for P41; a FrozenLake resident canary requires a separate capacity admission.
