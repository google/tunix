# T9a — bounded first-diff trajectory diagnostics

- Status: complete

## Motivation

Exact TiTO already reports the first unequal token and then fails. That is
enough to judge the contract, but not enough to reproduce a multi-turn drift:
the operator also needs to know which initial-prompt, assistant, or environment
segment produced the expected ledger and what token stream serving consumed.

## Preregistered design

- Add `CANON_P57_TOKEN_CONTINUITY_DEBUG=first-diff`, absent by default.
- Admit it only together with `CANON_P57_TOKEN_CONTINUITY=exact` on the exact
  P45 or M15 300-update DP8xTP8 full identity. Empty, `0`, another value,
  legacy/old selector, unselected arm, or neighboring workload is fatal.
- Renderer exposes `--token-continuity-debug`; it writes the debug key only to
  arms selected exact by the closed token-continuity enum.
- On the first exact mismatch per trajectory engine, atomically write one JSON
  replay capsule under `$CANON_STATE/token-continuity-first-diff/` and print
  its path/SHA. Also print the same complete data as bounded JSON chunks so a
  durable GCS worker log can reconstruct the capsule even when a failed pod's
  `/tmp` state disappears. The data contains the expected segment ledger and
  actual serving-consumed token stream; each record includes kind, turn,
  offset, length, SHA256, and integer token IDs. No free-form prompt, thought,
  model text, task payload, or secret is emitted. Token IDs remain reversible
  with the matching tokenizer and are therefore sensitive private evidence;
  raw chunks and capsules must not be committed or pasted into review.
- Ship a log extractor/verifier that rebuilds the JSON capsule, checks chunk,
  segment, metadata, and whole-stream hashes. It rejects extra segments or
  inconsistent workload/kind/turn attribution, writes the sensitive output as
  mode `0600`, refuses incomplete input, and requires an explicit capsule ID
  when multiple trajectories are interleaved in one worker log.
- Set the one-shot latch before printing. After the dump, execute the existing
  fatal mismatch with no warning/continue path.
- A successful exact run emits no debug dump even when the flag is enabled.

## Gates

1. Unit positive: a deliberately mismatching P45 and M15 fixture emits one
   reconstructable header/segment set, writes one capsule, and then raises the
   original fatal. Extraction from the log alone must reproduce identical
   actual/expected streams.
2. Unit negative: equal streams emit no debug record; repeated mismatch on one
   engine emits at most one dump; malformed/debug-only/neighbor identities fail.
3. Renderer structural A/B: without the debug CLI the manifests are unchanged;
   with it, only selected exact arms gain exactly one debug entry.
4. Classifier/report: legacy rejects the flag and any debug receipt; exact
   success permits the armed flag but requires zero emitted dumps.
5. Full P57/V1 host gates, flag audit, syntax/diff checks, then complete pinned
  image. Host/image green remains construction evidence; DP8xTP8 is target
  unverified.

## Result

Implemented and host-green. A deliberate mismatch produces one complete
verified capsule containing the actual serving token stream and the ordered
initial-prompt/assistant/environment expected ledger; the capsule survives a
local atomic round trip and is reconstructed byte-identically from raw log
chunks. Corrupt/missing chunks, equal streams, malformed or unscoped flags,
and debug receipts in a supposedly successful full run all fail. Interleaved
P45/M15 worker-log chunks require an explicit capsule ID and then reconstruct
the selected capsule exactly. Focused diagnostics are 13/13, P57 is 191/191,
V1 is 101/101, and the flag registry is 411/411. The complete immutable-image
gate passed against local image
`sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
with terminal fields `frozenlake_tito_impl=2`,
`frozenlake_tito_selector=closed`, `frozenlake_tito_summary=1`,
`frozenlake_tito_debug=1`, `frozenlake_tito_capsule_integrity=1`, and
`frozenlake_tito_default=legacy`. The raw local admission log is
`/tmp/p57_tito_pair_pinned_20260902_r2.log`, SHA256
`9bc28afb41ac0a7049eb66a2c65aa47abb912bdcf42ed4620603c961700446a3`;
it has not been copied to durable GCS. No TPU or Kubernetes target ran.
