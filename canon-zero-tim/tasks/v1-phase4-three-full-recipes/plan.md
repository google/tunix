# Plan

## Outcome

Prepare exactly three uninterrupted 64-chip Zero-TIM full trains. A full run is
the first target-topology certification; there is no separate short canary.
Every run stays strict and carries one warmed update-window XProf capture plus
the semantic Perfetto timeline.

## Phases

| Phase | Deliverable | Exit gate | Status |
|---|---|---|---|
| V1.P4.1 | Integrated default-off P56/P59/APC implementations | syntax, focused CPU, manifest | complete |
| V1.P4.2 | Three immutable manifests and intent verifier | exactly three renderer PASS records | complete |
| V1.P4.3 | Host/real-env/exact-image admission | all positive and negative markers | complete |
| V1.P4.4 | Attempt-6 P59 staged-spec repair plus uniform APC-off/JAX-cache receipt hardening | host + pinned-image + one-host TPU mechanism | superseded before target publication; admission evidence preserved |
| V1.P4.5 | Attempt-7 first-red numerical localization for P59 grouped backward | a complete durable DP16xTP4 no-commit log passes the profile/alignment/16-group/reduction/scaling/accumulator/discard contract and explains the extreme magnitude | complete |
| V1.P4.6 | Hybrid overflow-safe global-norm clipping for the three strict full recipes | stock-finite outputs remain byte-exact; finite-overflow matches FP64; NaN/Inf stays fatal; host and pinned-image gates pass | complete |
| V1.P4.7 | Publish one reviewed immutable SHA and execute the three full target recipes | approved commit/push and remote readback; three fresh manifests; GSM8K 200 plus P45/M15 300 complete with zero strict FAIL and complete signed evidence | partial target evidence; superseded by recovery after GSM XProf infrastructure stop and P45 numerical stop |
| V1.P4.8 | Recover Attempt 7 without weakening the numerical contract | GCS-backed XProf capture restores locally with signed artifacts; GSM scale carrier distinguishes topology/duplication from legitimate magnitude; a hash/model-bound P45 capsule accelerates first-red replay; P45 first-red carrier identifies the earliest non-finite boundary; host and pinned-image gates pass before any target relaunch | active; published at `548db7e9`; G1/G2/P64 construction PASS, real GCS transport and P45 target capture/observation pending; apply matrix awaiting user choice |

## Decisions

- Decision: P59 is accepted under the user's ordinary-JAX FP64 gradient-correctness policy; serial/update trajectory differences remain disclosed.
- Decision: attempt-2 target evidence VETOES APC for M15/main; user elected the same APC-off production policy for P45. All three full recipes are APC-off. B rescore always resets the cache and the strict gate is unchanged.
- Decision: all three manifests retain the P33 JAX persistent-cache bucket. Exact restore/save receipts are mandatory carrier evidence; miss/error remains a performance limitation, not a numerical verdict.
- Decision: the profiled update is excluded from steady-state performance means.
- Decision: launch all three full-horizon jobs in one wave with no short canary and no cross-recipe first-commit dependency. Each recipe independently passes strict alignment plus its registered P59-local/fixed-head/token/APC/optimizer receipts while the other healthy runs continue from the same exact source SHA.
- Decision correction (2026-08-25): max-scaled L2 is an overflow-safe observer, not an admitted optimizer repair. Attempt 7 did not establish that the finite gradient magnitude was legitimate. No full recipe may use stable clipping to turn an unexplained `norm=inf` into an optimizer transaction; first localize the earliest bad numerical boundary in a zero-commit carrier.
- Decision correction (2026-08-25 G5a): the six-line `p62d3` excerpt is an incomplete observation, not a classified G5 result. `all_finite=true` for group 0 distinguishes NaN/Inf from finite values but does not validate a `5.38e22` gradient norm. A fresh G5b must preserve the full raw log and zero-commit terminal before any numerical repair.
- Decision (2026-08-25 G5b): the complete 16-group DP16xTP4 carrier proves every pre-optimizer gradient boundary and the final accumulator finite with exact denominator 16. The remaining `naive_norm=inf` is FP32 sum-of-squares overflow. Admit a default-off hybrid repair: preserve the stock transform when its norm is finite; use max-scaled L2 only when the stock norm is non-finite and an independent all-finite predicate is true. A real NaN/Inf never takes the fallback.
- Decision (2026-08-25 Attempt 7): GSM8K's two all-finite P63 optimizer commits are valid numerical evidence, but the run stopped for an XProf path contract bug: Pathways accepts only a GCS trace directory. Repair capture to a unique `gs://` path and synchronously restore the resulting XPlane/trace artifacts into the existing local postflight directory. Do not classify this infrastructure stop as a Zero-TIM red.
- Decision (2026-08-25 Attempt 7): P45's rank-1 non-finite staged gradients are a true pre-optimizer numerical red. P63 must not sanitize them. Add observation-only, no-commit localization from loss/report cotangent through the fixed head and earliest layer boundary before considering a repair.
- Decision (2026-08-25 Attempt 7): the next GSM correctness carrier uses one frozen input/trajectory and identical checkpoint across native/serial-reference and P59 arms. It measures denominator, per-leaf scale, topology ownership, and an FP64 oracle; it does not require historical serial AdamW byte identity and performs no optimizer commit.
- Decision (2026-08-25 P64 admission): the P45 target-localization carrier is complete only if group 0 and group 31 each traverse engine VJP, trainer rank-local, fixed DP reduction, and scaled-microgradient boundaries, followed by exactly one final accumulator and discard receipt. A first non-finite receipt must be the terminal receipt. This is an observation-only, zero-commit carrier and does not authorize a production repair.
- Decision (2026-08-25 P64 capsule): one fresh strict P45 target may capture the
  exact training payload before backward. Capture retains the original
  certification burden and executes all 32 groups. A later hash/model-bound
  replay may bypass environment, rollout, and B rescore and execute group 0
  only, but is permanently labelled diagnostic and cannot count as a new
  strict certification or optimizer result.
- Decision checkpoint (2026-08-25 four-job request): the requested concurrent
  matrix is GSM8K full, P64 P45 capture, P45 full, and M15 full. Fresh remote
  evidence now proves both unchanged FrozenLake full recipes already fail in
  Step 0 after strict prealignment (P45 rank 1, M15 rank 3). Publication and
  rendering may proceed, but applying the two known-red full recipes requires
  a final explicit matrix choice; do not describe them as expected-green
  production runs.
