# Plan

## Outcome

Prepare the optimized P45 and M15/main 64-chip full trains as direct
300-update efficiency-first concept runs. P45 remains strict. M15 temporarily
uses its registered finite A-B warning lane and is alignment-degraded, while
retaining hard token-continuity, B-C, nonfinite, backward-health, optimizer,
timing, W&B, JAX-cache, XProf, and Perfetto gates. Both deliberately omit
in-process held-out evaluation and all checkpoint I/O.

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
| V1.P4.8 | Recover Attempt 7 without weakening the numerical contract | GCS-backed XProf capture restores locally with signed artifacts; GSM scale carrier distinguishes topology/duplication from legitimate magnitude; a hash/model-bound P45 capsule accelerates first-red replay; P45 first-red carrier identifies the earliest non-finite boundary; host and pinned-image gates pass before any target relaunch | superseded by the P66/P67 recovery phases; construction and failure evidence preserved |
| V1.P4.9 | Promote the P66 checked-VMA repair into the three exact full recipes and prepare one simultaneous wave | exact-profile admission; first-update precommit/optimizer hard receipts; host and immutable-image gates; three fresh manifests only after publication | implementation complete; original three-job target sequence superseded by P4.10-P4.14; evidence preserved |
| V1.P4.10 | Localize the post-P66 FrozenLake TP8 forward regression with a production-geometry, pre-backward matched pair | P45 p66-off vs serving-scope render/resolved-env; finite A−B, exact B−C, depth floor, controlled zero-commit exit; scoped TP4/TP8 P59 and full pinned-image before publication | complete; Wave 5 both arms target-green at strict A−B/B−C `0/0`, serving-scope candidate accepted |
| V1.P4.11 | Promote P67 P59-only VMA scoping into the exact P45 and M15 full recipes | two-manifest production admission, host/full-image gates, then independent 300-update target verdicts | implementation complete; host/full-image PASS, publication identity requires remote read-back, and both 300-update targets remain pending |
| V1.P4.12 | Repair the stale G6 checkpoint admission exposed by Attempt 10 | one checkpoint source of truth; legacy-10 and primary-300 positives; wrong identity/cadence negatives; host and immutable-image gates; then fresh target first update | source published by current CL; host and immutable image PASS; target first gradient sink/AdamW not rerun |
| V1.P4.13 | Repair the missing FrozenLake effective-learning-rate observation exposed by P45 Wave 02 | keep scalar AdamW unchanged; register the same constant only for receipts; pin entrypoint structure; host and P45 immutable-image gates; then fresh target weight sync | source published by current CL; P57 147/147, Phase4 89/89, P45 exact image PASS; post-fix target not run |
| V1.P4.14 | Admit exact no-eval/no-checkpoint P45+M15 Zero fast concept runs after f45w09 | both manifests fail closed on eval disabled, checkpoint mode disabled, and empty residual fields; host plus immutable-image gates; then two fresh 300-update targets | active; runtime `a8449b3d` published and exactly read back, host/exact-image PASS; TPU target not run |
| V1.P4.15 | Make exact TITO the signed M15 full default while retaining the finite A-B concept lane | exact-M15 selector in YAML/profile/runtime; prompt-token equality hard gate; neighboring negatives; host/pinned-image; one-host mechanism; then DP8xTP8 full target | active; runtime `3fc7ef8b` published/read back, host and post-rebase pinned-image construction PASS; one-host/target not run |

## Decisions

- Decision: P59 is accepted under the user's ordinary-JAX FP64 gradient-correctness policy; serial/update trajectory differences remain disclosed.
- Decision: attempt-2 target evidence VETOES APC for M15/main; user elected the same APC-off production policy for P45. All three full recipes are APC-off. B rescore always resets the cache and the strict gate is unchanged.
- Decision: all three manifests retain the P33 JAX persistent-cache bucket. Exact restore/save receipts are mandatory carrier evidence; miss/error remains a performance limitation, not a numerical verdict.
- Decision: the profiled update is excluded from steady-state performance means.
- Historical decision (superseded by P4.14): launch all three full-horizon jobs in one wave with no short canary and no cross-recipe first-commit dependency. The current launch set is the two optimized FrozenLake fast runs only.
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
- Decision (2026-08-26 P66 recovery): P66 G1/G1.5 supersedes the old
  padding/RMS hypothesis and localizes the Attempt-7 TP>1 gradient explosion to
  erased VMA/replication ownership in old P59. Promote the repaired path only
  behind the exact three full profiles and treat their full runs as first
  target-topology optimizer/convergence certification.
- Decision (2026-08-26 launch matrix): prepare GSM8K, P45, and M15 as three
  simultaneous full-horizon jobs. Each independently requires a pre-AdamW
  finite/nonzero/`stable_norm <= 1e6` first-accumulator receipt and a valid
  first optimizer receipt before weight sync/checkpoint. One recipe's red does
  not cancel the other two.
- Decision (2026-08-26 Attempt 8): P45 (`396/0` bytes) and M15 (`20/0`
  bytes) share one APC-off, TP8, pre-backward signature: A decode differs from
  B full prefill while B remains exact to T-old. The P66 checked-VMA
  completed-sum marker must be scoped to the live P59 manual context and must
  never enter ordinary serving. Hold all relaunches until the dirty repair
  returns both strict boundaries to zero on real TPU; host/image negatives do
  not certify target restoration.
- Decision (2026-08-26 Wave 5): P45 `p66-off` and `serving-scope` both
  recovered strict A−B/B−C `0/0` on real DP8xTP8 serving geometry. Promote
  only the scoped candidate into the exact FrozenLake full profile. The user
  waived another M15 scope precheck; P45 and M15/main full runs are the first
  M15 serving and both-workload backward/optimizer/convergence target gates.
- Decision (2026-08-26 Attempt 10): both FrozenLake full runs prove strict
  Step-0 A-B/B-C `0/0`, then stop at the first of 32 gradient sinks because
  the trainer duplicated a historical checkpoint-interval whitelist. This is
  not a Zero-TIM or numerical backward red. Reuse the canonical checkpoint
  parser in the G6 guard; do not change the final-only cadence or bypass the
  guard. The full backward, AdamW, convergence, and checkpoint claims remain
  target-unverified.
- Decision (2026-08-26 P45 Wave 02): checkpoint admission is fixed and the
  target now proves all 32 reverse groups, a finite/nonzero denominator-32
  accumulator, and one finite AdamW mutation. The subsequent
  `effective_learning_rate=None` red is an observer-registration omission:
  preserve scalar AdamW and register the identical constant with the receipt
  API. Do not weaken the first-update gate or claim outer weight sync/policy
  step 1 until a fresh target crosses them.
- Decision (2026-08-28 f45w09): the optimized Zero P45 and M15/main recipes
  are efficiency-first 300-update concept runs. Disable held-out evaluation
  and checkpoint creation only for their exact v1-hp identities; retain all
  strict Zero-TIM, backward-health, optimizer, timing, W&B, cache, XProf, and
  Perfetto gates. Native/IS and historical/evaluation carriers keep their
  existing contracts. A failed fast run has no resume point and must restart
  at step 0 with a fresh identity.
- Decision (2026-08-30 V1.P4.15): DeepSWE TP4 proves that later-turn text
  retokenization can break A/B input identity, but that evidence does not
  transfer to M15. First add an observer-neutral M15 prompt-token verifier. An
  exact-token input path is admitted only after M15 token drift is observed.
  Finite A-B remains warning-only for the concept run; token mismatch, B-C,
  nonfinite, backward, replica, and optimizer faults remain fatal.
- Decision override (2026-08-30 V1.P4.15): user explicitly selected exact TITO
  as the default for the signed M15 full recipe before a real M15 verify
  verdict. This authorizes the model-input change but does not transfer P58
  certification. P45, GSM8K, Native, IS, eval, diagnostics, and every other
  topology keep the selector absent; exact token inequality is fatal and the
  first DP8xTP8 full run is the target gate.
