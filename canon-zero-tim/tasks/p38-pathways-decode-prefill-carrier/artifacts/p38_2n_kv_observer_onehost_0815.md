# P38.2n live-KV observer one-host rehearsal — 2026-08-15 UTC

Verdict: `PASS` for local wiring and neutrality. This is not a production
mechanism verdict because all three one-host alignment rounds were A-B exact.

Command:

```bash
bash \
  canon-zero-tim/tasks/p38-pathways-decode-prefill-carrier/scripts/run_p38_incident_onehost.sh \
  on p38_2n_kvobs_r6
```

The real Qwen3-8B DP1xTP4 v5p run completed three frozen-weight rounds with no
backward and no optimizer commit. The observer selected one naturally
single-active request per round and persisted exactly three live A records and
three exact-prefix clean-rescore B records:

| round | A target length | logical pages | fingerprint result |
|---|---:|---:|---|
| 0 | 1137 | 5 | A == B |
| 1 | 1077 | 5 | A == B |
| 2 | 1488 | 6 | A == B |

For every pair, token IDs, token-history SHA, target length, valid-token
extents, and provenance matched exactly. Integer aggregate-prefix cells and
fixed sample-prefix cells had zero differences. The classifier returned:

```text
status=PASS
classification=observer_pairs_valid_red_join_pending
records=6
pairs=3
red_joins=0
```

Important implementation findings from the rejected r1-r5 rehearsals:

1. `input_batch.num_prompt_logprobs` is consumed by prompt-logprob
   post-processing, so it cannot be discovered after sampling.
2. Prompt-logprob-only requests need not appear in sampled `output.req_ids`.
3. The clean B cache must be observed after `model_fn` materializes the final
   prompt chunk, but outside `maybe_forbid_compile`; compiling the observer
   inside that region stalls.

An AST regression test now rejects moving the clean-B hook under any `with`
context. Patch 16 remains default-off and the production renderer enables it
only for the stock P38 discriminator.

Evidence identities retained on the one-host disk:

```text
raw log sha256:        64f1cf4078a091d08e02bb2274526eb0c01d5ee0d420c36ca8d3a3c081c11ebb
classification sha256: 78b6a0fcf453f9548ceb79d368dd94c6177f2c44010495415907089410c8d900
classification: /mnt/disks/tunix-data/logp_probe_1host/
  p38_incident_p38_2n_kvobs_r6_on/p38_kv_observer.classification.json
```

Claim ceiling: the aggregate/sample table is a diagnostic bit-level
fingerprint, not a cryptographic hash or full-byte KV proof. Only a fingerprint
pair joined to a production A-B-red mismatch may select stale-content versus
decode-program-seam branches.
