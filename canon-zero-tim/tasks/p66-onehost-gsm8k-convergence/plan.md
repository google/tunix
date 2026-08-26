# P66 plan: one-host GSM8K trainer/backward bisection

## Scope

P66 is a correctness-first bisection of the trainer/backward optimization bundle on one v5p host. It does not alter the P61 evidence or retroactively change its thresholds. Serving/rollout flags remain fixed while trainer dependencies are isolated.

## Frozen arm geometry

- Model/topology: Qwen3-1.7B, DP4xTP1.
- Workload: existing P59/P61 one-update GSM8K carrier, 64 trajectories in 16 groups.
- Input discipline: deterministic request scheduling; compare all seven update hashes and exact model-before tree before interpreting gradient/update differences.
- Control: P59 rank-parallel backward off.
- Candidate: P59 rank-parallel backward on.
- Both arms: one real positive-learning-rate AdamW commit, 17/17 strict alignment, complete pre/gradient/post tree capture.
- Thresholds: unchanged P61 Tier-1-derived thresholds and frozen caps.
- Performance: ineligible while full-tree device-to-host capture is enabled.

## Phases

| Phase | State | Gate | Outcome |
|---|---|---|---|
| P66.0 Reconcile prior evidence | COMPLETE | Verify P61n2 input/pre-state contract and published-source gap | Historical result is usable; reproduction gap is localized |
| P66.1 Restore classifier chain | COMPLETE | Comparator unit tests, reject-path manifest test, P59 focused host tests, `git diff --check` | Durable KEEP/REJECT/INCONCLUSIVE bundle verified |
| P66.2 Current-source P59 A/B | COMPLETE | One-host v5p, fresh immutable labels, arm order control then candidate, strict gates and full-tree classifier | Gradient envelope KEEP; AdamW update-trajectory REJECT reproduced |
| P66.3 TP backward causal closure | ACTIVE — G1/G1.5 COMPLETE, G2 PENDING APPROVAL | VMA structural probe, one-host S/U/P/R causal scan, then same-evaluation-point head/norm/layer/embed oracle; signed P64 target replay follows only after source freeze and separate approval | Four-arm verdict `H1_VMA_SUPPORTED`; final-source G1.5 six-endpoint and observer-neutrality PASS; target not run |
| P66.4 TP fixed-head isolation | PENDING/CONDITIONAL | DP1xTP4 fixed LM-head/projection VJP gate with global-path negative control | Keep TP repair separate from DP P59 result |
| P66.5 Convergence health | PENDING/CONDITIONAL | Native-200 vs admitted Zero-full-200; strict receipts plus reward/update curves | Decide whether to restore full GSM8K train campaign |

Exactly one phase is active. A red gate is repaired or rejected at that phase; no phase is skipped by relabeling a failure.

## Dependency bisection order

1. P59 rank-parallel backward alone. COMPLETE: gradient-correct under the frozen envelope, but not update-trajectory equivalent.
2. Ordinary monolithic versus segmented reverse. INCONCLUSIVE: ordinary XLA
   requires 450.29 GiB HBM, so this carrier is retired rather than retried.
3. P59 VMA/manual-TP composition versus fixed-AR gather transpose. G1
   COMPLETE: H1 supported and fixed gather acquitted. G1.5 same-point
   pullback oracle is COMPLETE; G2 P64 target replay remains pending source
   freeze and separate launch approval.
4. Gradient sparse assembly and leading-DP reshard closure, only if the first
   red is later than engine VJP.
5. Reuse/alias dependency closure.

Within a phase, only one dependency closure changes between arms. If a flag requires prerequisites, the entire minimal dependency closure is named in the receipt; partial bundles fail closed.

## Evidence contract

Every attempt keeps its raw logs, arm classifications, capture manifests, tree blobs, comparison JSON, scripts, baseline SHA, and a top-level SHA256 manifest. Failed and inconclusive attempts are never deleted. Run labels are never reused.

## Launch discipline

- One-host v5p only. Any P64/64-device target replay requires a separate user
  launch approval and cannot be inferred from this phase.
- Confirm no conflicting P66/P51 container before launch.
- Launch without a trailing pipeline; inspect raw logs afterward.
- Never run control and candidate concurrently on the TPU host.
- No commit or push without explicit user approval.
