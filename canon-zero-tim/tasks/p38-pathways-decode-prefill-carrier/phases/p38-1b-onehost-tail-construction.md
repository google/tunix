# P38.1b — One-host production-tail construction gate

- Status: complete

## Question

Can the direct-attached four-chip v5p host validate the remaining
decode-versus-prefill hypothesis without promoting a local result into a
Pathways claim?

## Reconciled facts

- The r35 logs already contain
  `runner_sampling_adapter_same_object=True`. Python callable identity is
  therefore not the missing evidence by itself.
- Both serving paths already terminate in the canonical Pallas log-softmax,
  whose calls set `shape_invariant_numerics=True` and disable input fusion.
- The previous real-Qwen DP1xTP4 attempt classified the local A/C boundary as
  `LOCAL_NOT_REPRODUCED`.
- FrozenLake r35 measured a sparse boundary with `max_abs=0.10390`; differing
  byte density does not establish a one-ULP carrier.

## Deliverables

1. Add a default-off precheck-only stop that writes the ordinary A/B/C
   pre-backward record and exits before backward when the record is exact.
2. Run the current Qwen3-1.7B DP1xTP4 production boundary on the authorized
   one-host v5p using the complete canonical switch set and the signed GSM8K
   prompt/response contract (`1024/1024`).
3. If the local production boundary is red, preserve exact mismatch evidence
   and proceed to raw-target/normalizer localization.
4. If the local boundary is exact, classify `LOCAL_NOT_REPRODUCED`; then run a
   separate same-input tail construction probe. That probe may validate code
   mechanics but cannot establish the r35 Pathways cause.

## Decision table

| Observation | Verdict | Next action |
|---|---|---|
| A-B red, B-C exact | `LOCAL_REPRODUCED` | Compare processed target and normalizer before changing the tail. |
| A-B exact, B-C exact | `LOCAL_NOT_REPRODUCED` | Run only the synthetic construction/negative-control gate; retain P38.2 target requirement. |
| B-C red | `VOID_REGRESSION` | Stop; the one-host canonical baseline regressed. |
| Missing row, bad contract, disconnect | `INCONCLUSIVE` | Fix infrastructure or instrumentation; do not interpret numbers. |

## Exit gate

- One ordinary pre-alignment record is flushed and printed.
- The log attests four TPU devices, exact source/diff provenance, canonical
  path traces, and `STOP_BEFORE_BACKWARD`.
- Optimizer commits, checkpoints and W&B writes are zero.
- A one-bit negative control proves the record classifier rejects drift.
- A separate exact-production-shape tail control executes the same canonical
  callable inside two distinct outer JIT programs, compares all output
  elements, and detects an injected one-bit change. This is construction-only
  evidence and cannot establish a Pathways result.

## Rollback

Leave `CANON_P38_PRECHECK_ONLY` unset. The learner follows its existing path and
the new stop is unreachable.

## Result

- The signed GSM8K DP1xTP4 run observed 11,340 action tokens. Both
  `S_decode_vs_S_prefill` and `S_prefill_vs_T_old` differed by 0 of 45,360
  bytes. The classifier returned `LOCAL_NOT_REPRODUCED`.
- Canonical path evidence was present (`fixed_ar=168`, `fixed_embed=1`,
  `logprob_m=1`, shared-tail identity, four TPU devices, and overlay byte
  identity). Backward and optimizer marker counts were zero.
- The construction control compared 38,895,616 f32 outputs from one canonical
  tail invoked through two outer JIT programs. It found zero differing
  elements; its injected one-bit negative control found exactly one.
- These results validate the direct-attached construction only. They do not
  reproduce or resolve the 64-chip Pathways r35 carrier.

Artifacts:

- `/mnt/disks/tunix-data/logp_probe_1host/p38_onehost_0810_r2.raw.log`
  (`sha256=3a58aa9f1e37b4afdc30ac0cab317eae56545a0dd09c3b6acf60f3db33d97d81`)
- `/mnt/disks/tunix-data/logp_probe_1host/p38_onehost_0810_r2/pre_alignment.jsonl`
  (`sha256=cf857cffd87a2917deaac89a9bda1f47700a3e2bdffe8e6eadf9f9e61781345e`)
- `/mnt/disks/tunix-data/logp_probe_1host/p38_tail_0810_r1.result.json`
  (`sha256=4b5b27daf313223974f7428004ea49a903e14a6a501602c841e232f057b550f1`)
