# P38.2h — fixed lm-head backward-no-commit admission

Status: locally complete and ready for review. The real-v5p VJP gate, complete
P33/P38 CPU gate, pinned exact-image gate, renderer tests, and operator return
positive/negative controls pass. Commit, push, and the 64-TPU launch each
require separate user approval.

## Entering evidence

P38s23r3 measured three FrozenLake forward-only rounds with the fixed Pallas
lm-head enabled. Across 146,042 action tokens, action-masked A-B and B-C were
bitwise exact. Every round stopped before backward with zero optimizer commits.
The round archives are independently sealed, but the returned package omitted
the preregistered root `COLLECTED.json`, `COMPLETE.json`, and root
`SHA256SUMS`; therefore this is a strong forward causal-repair result with an
incomplete run-level durability seal, not a backward or production claim.

The fixed lm-head already calls the promoted P22.XK custom VJP. Its primal is
the fixed Pallas matmul; its backward differentiates the canonical fixed-BK
pure-JAX replica. The untested part is the new outer learner composition:
semantic M4096 is split into 16 M256 calls that share one weight. Forward
exactness does not establish that `dHidden` and the 16 contributions to
`dWeight` compose deterministically or match a fixed-order M256 reference.

## Deliverables

1. A real-Qwen3-8B one-v5p gate computes a nonzero selected-logit loss through
   the complete TP4 fixed lm-head at M4096, runs its VJP twice, and compares
   both `dHidden` and `dWeight` against 16 completed M256 VJPs accumulated in
   explicit chunk order.
2. The one-host gate requires finite/nonzero gradients, array-exact repeat,
   array-exact chunk reference, and a one-element normal-value negative
   control. A BF16 subnormal bit flip is forbidden because TPU flush-to-zero
   can make that negative control inert.
3. Only after the one-host gate passes, render one strict FrozenLake
   DP16xTP4 backward-no-commit target with fixed lm-head enabled, prefix cache
   off, evaluation off, one update, and no P38 precheck-only/observer env.
4. The target must execute the actual-model segmented VJP, all 16 gradient
   groups, fixed-order DP reduction and replica gates, while recording zero
   optimizer commits and byte-unchanged model/optimizer/accumulator/reference
   state.
5. A compact classifier consumes the complete head log plus P33 pre-alignment,
   alignment, and no-commit reports. It emits a SHA-sealed verdict; the remote
   operator does not choose numbers or conclusions.

## Local gate

```bash
set -euo pipefail
python3 -m unittest \
  canon-zero-tim/tests/p38_serving/test_fixed_lm_head.py -v
python3 -m py_compile \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/probe_p38_fixed_lm_head_vjp.py
bash -n \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38_fixed_lm_head_vjp_onehost.sh
bash canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38_fixed_lm_head_vjp_onehost.sh \
  p38_2h_<unique>
bash canon-zero-tim/tests/p38_serving/test_p38h_backward_operator_scripts.sh
bash canon-zero-tim/tests/p33_workloads/run_cpu.sh
```

The admitted real-v5p result is `p38_2h_vjp3_20260819` with verdict
`FIXED_LM_HEAD_ONEHOST_VJP_PASS`. Both `dHidden` and shared `dWeight` are
array-exact against the ascending 16-chunk oracle and across a repeated run;
gradients are finite/nonzero and the normal-value one-bit negative is detected.

## One-host decision table

| Verdict | Decision |
|---|---|
| `FIXED_LM_HEAD_ONEHOST_VJP_PASS` | prepare the 64-TPU actual-model target |
| `FIXED_LM_HEAD_CHUNK_VJP_NOT_INVARIANT` | stop; repair the outer M4096 chunk VJP/reduction order |
| `FIXED_LM_HEAD_VJP_NOT_DETERMINISTIC` | stop; investigate execution nondeterminism before any target |
| nonfinite/no-signal/negative-control failure | probe invalid or unsafe; no target |

## 64-TPU target gate

The target is not the historical P38 diagnostic renderer. It must omit
`CANON_P38_PRECHECK_ONLY`, `CANON_P38_CONTROLLED_EXIT`, diagnostic rounds, and
all P38 serving observers; those settings deliberately stop before backward.
It reuses the ordinary P33 FrozenLake `backward-no-commit` transaction and
adds only `CANON_P38_FIXED_LM_HEAD=1` plus its single-variable label.

Admission requires:

- Attempt 0 and the exact approved source SHA;
- fixed-lm-head receipts for request buckets and learner M4096;
- exact pre-backward A-B and B-C;
- finite, nonzero actual-model loss/gradients;
- 16 expected gradient groups and all DP reducer/replica checks;
- `[CANON_P33_DP16] backward_no_commit verdict=PASS commits=0` exactly once;
- train step unchanged and no changed model, optimizer, accumulator, or
  reference paths;
- no evaluation, prefix cache, warning-only policy, checkpoint write, or
  optimizer commit.

## Claim ceiling and rollback

A green one-host result proves only the fixed-lm-head VJP construction. A green
64-TPU result admits the repair through actual-model backward and the
mutation-free transaction boundary. It still does not establish optimizer
commit, full training, checkpointing, quality, or performance.

Rollback is to omit `CANON_P38_FIXED_LM_HEAD`; the hook is default-off. No
production profile or full-training default changes in this phase.
