# P38.2y1 — tied output-head integration and executable receipts

Status: implementation locally gated; target not run.

## Why this phase exists

P38s23r3 and P38h1 are admitted evidence for Qwen3-8B's **untied**
`JaxLmHead` endpoint: their fixed-head receipts are present and their measured
forward boundaries are bitwise exact. They do not prove Qwen3-1.7B's tied
output-head endpoint.

P38y7 used Qwen3-1.7B with `tie_word_embeddings=true`. The pinned engine's
`compute_logits` calls `model.embed_tokens.decode(hidden)` for that model and
does not construct `JaxLmHead`. P38y7 set `CANON_P38_FIXED_LM_HEAD=1`, but its
complete returned log contains zero `CANON_P38_FIXED_LM_HEAD` primal or VJP
receipts. It therefore did not execute the P38s23 repair. P38y7 is useful
full-training alignment evidence, but it is not a fixed-head causal test.

The raw P38y7 boundaries through completed step 4 are:

| step | actions | A-B elements | A-B bytes | A-B max abs | B-C bytes |
|---:|---:|---:|---:|---:|---:|
| 0 | 190621 | 0 | 0 | 0 | 0 |
| 1 | 205214 | 2 | 2 | 5.7220458984375e-06 | 0 |
| 2 | 205057 | 5 | 7 | 8.182525634765625e-04 | 0 |
| 3 | 182361 | 41 | 74 | 1.0463905334472656e-01 | 0 |
| 4 | 188667 | 1 | 1 | 7.62939453125e-06 | 0 |

The old progress report's `7.629e-06` overall maximum is false; it reported
the final observed step rather than the maximum over observed steps.

## Repair

1. When the fixed-head flag is enabled, patch both output endpoints:
   `JaxLmHead.__call__` for untied models and `JaxEmbed.decode` for tied
   models.
2. The tied endpoint passes the embedding table `[V,D]` as its transposed
   `[D,V]` view to the same fixed Pallas body. Automatic differentiation
   returns the head cotangent in embedding orientation; P28.G5C retains
   responsibility for combining input-embedding and output-head cotangents in
   its already certified order.
3. Every primal and M4096 VJP receipt carries an explicit endpoint identity:
   `untied_lm_head`, `tied_embed`, or `direct_probe`.
4. The production postflight classifier fails closed unless the admitted
   endpoint emits request M16/32/64/128/256, learner M4096, exact
   M/K/N/BM/BN/BK/chunk geometry, and (outside serving-capture diagnostics)
   the fixed-order M4096 VJP receipt. A tied run must also contain the P28.G5C
   tied-adapter marker.

## Gates

- Focused CPU/unit: endpoint registration, tied wrapper transpose, classifier
  positives, and missing-M/missing-endpoint/wrong-endpoint/missing-VJP
  negatives.
- Pinned exact image: Qwen3-1.7B and Qwen3-8B overlays, 34 tests each. The
  Qwen3-1.7B gate invokes the patched `JaxEmbed.decode` object rather than
  trusting a source string.
- Real one-host v5p Qwen3-1.7B forward+VJP: required before publication when
  the local TPU is free. It was not run during implementation because another
  container owned the device.
- Target: one P38y8 64-TPU 200-step GSM8K full run.

## P38y8 decision table

| Observation | Verdict |
|---|---|
| receipt classifier fails | `INCONCLUSIVE_FIXED_HEAD_NOT_EXECUTED`; no numerical causal claim |
| tied receipts pass, all 200 steps A=B=C | P38.2y fixed-head full target PASS |
| tied receipts pass, A-B remains red, B-C exact | fixed head is insufficient for GSM8K; retain alignment-degraded training result and reopen the upstream/normalizer branch |
| any B-C, nonfinite, reducer, optimizer, or state-transition red | hard FAIL |
| infrastructure loss or exhausted restarts | INCONCLUSIVE |

The target changes only the previously bypassed tied output endpoint and the
receipt gate. Prefix cache remains off; evaluation remains off; P52 batched
reverse remains off. Do not combine another numerical intervention with
P38y8.

## Rollback

Revert the tied `JaxEmbed.decode` hook and endpoint-receipt classifier as one
change. Do not disable the existing Qwen3-8B untied fixed-head path or the
independent P47/P50 performance bundle.
