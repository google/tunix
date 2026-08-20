# P38.2y2 — Qwen3-4B/32B TP8 fixed output-head extension

Status: implementation and pinned-image CPU gates complete; target not run.

## Goal

Reuse the output-head repair already validated on Qwen3-1.7B/8B without
pretending that a new model or TP width inherits that evidence. Register the
two production TP8 geometries used by DeepSWE:

| model | endpoint | hidden K | TP | local vocab N | fixed N |
|---|---|---:|---:|---:|---:|
| Qwen3-4B-Instruct-2507 | `tied_embed` | 2560 | 8 | 18992 | 19200 |
| Qwen3-32B | `untied_lm_head` | 5120 | 8 | 18992 | 19200 |

Both use `BM=128, BN=256, BK=256`. Request rows
`M={8,16,32,64,128,256}` execute one padded `M256` body; learner `M4096`
executes 16 ordered `M256` chunks with the existing fixed-order custom VJP.

## Implementation contract

1. `p38_fixed_lm_head.py` resolves an exact `(hidden, TP, endpoint)` registry
   entry. Cross-model endpoint substitution, hidden drift, and TP drift fail
   before compilation.
2. Qwen3-4B and Qwen3-32B model overlays register only the new output-axis
   padding `18992 -> 19200`; no layer-internal projection geometry changes.
3. P34, P44, and P46 renderers keep the intervention default-off. The only
   supported opt-in is `--fixed-lm-head`; the rendered env and JobSet label
   both record the choice.
4. `00_env.sh` admits the flag only on training stages with backward work.
   P46 evaluation and P44 rollout-only remain forbidden because they cannot
   produce the required forward+VJP receipt set.
5. `90_run.sh` classifies exact model-specific receipts. A TP8 run must report
   `TP=8`, `local_N=18992`, `fixed_N=19200`, the registered endpoint, the
   production-required request buckets M16/32/64/128/256, learner M4096, and
   the fixed-order VJP. M8 remains a supported construction bucket but is not
   required when the workload never compiles it.

## Local gates completed

- Focused unit/renderer/env suites: 60/60 tests passed.
- Qwen3-32B pinned image `sha256:418dc632...`: installed 34-file overlay
  manifest matched and emitted
  `P38_FIXED_LM_HEAD_EXACT_IMAGE_PASS ... model=qwen32b tp=8 K=5120 ... endpoint=untied_lm_head`.
- Qwen3-4B pinned image `sha256:418dc632...`: installed 34-file overlay
  manifest matched, the 42-test P44 suite passed, and emitted
  `P38_FIXED_LM_HEAD_EXACT_IMAGE_PASS ... model=qwen4b tp=8 K=2560 ... endpoint=tied_embed`.
- Backward-compatibility pinned-image gate for Qwen3-1.7B tied TP4 and
  Qwen3-8B untied TP4 also exited zero with both 34-file manifests and endpoint
  probes intact.

These are construction and delivery gates only. No TP8 Pallas kernel, full
model forward/VJP, Pathways run, or alignment boundary was executed on TPU.

## Target gates

Run one bounded update before full training for each model. Promotion requires:

1. exact installed-overlay manifest;
2. model-specific fixed-head primal and VJP receipt classifier PASS;
3. finite, nonzero gradients and existing reducer/optimizer gates;
4. exact B-C and no new hard numerical boundary;
5. rollback comparison remains available by rendering without
   `--fixed-lm-head`.

Qwen3-4B and Qwen3-32B are certified independently. A green result for one
does not promote the other.

## Rollback

Render the same workload without `--fixed-lm-head`. This emits
`CANON_P38_FIXED_LM_HEAD=0` and leaves every other recipe field unchanged.
