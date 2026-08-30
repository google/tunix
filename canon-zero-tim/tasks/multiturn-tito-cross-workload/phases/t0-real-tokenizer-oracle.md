# T0 — Real-tokenizer transcript oracle

- Status: complete

## Finding

- Confirmed: shared reconstruction concatenates the actual first-turn serving
  prompt tail, sampled assistant IDs, and once-tokenized nonterminal environment
  IDs.
- Confirmed: current M15 unit tests use artificial integer arrays; they do not
  exercise a real Qwen3-8B tokenizer, disabled-thinking generation prompt, or
  FrozenLake user-role boundaries.
- Confirmed: Qwen assistant end-token normalization appends zero tokens, so an
  `n_append` counter omission is not the current Qwen mechanism.
- Hypothesis: a later FrozenLake rendered-text request may tokenize differently
  from the exact incremental sequence at a role/message boundary, but no real
  M15 token verdict currently exists.

## Execution

1. Add a read-only oracle that accepts a bounded trajectory transcript plus a
   tokenizer/parser identity and computes, for every later turn:
   - full rendered-chat token IDs;
   - incremental initial + assistant + environment token IDs;
   - lengths, SHA256, first mismatch, role and boundary offset.
2. Exercise Qwen DeepSWE tool/user and FrozenLake user-only transcripts with the
   correct thinking-mode setting.
3. Add a one-token deletion/substitution negative and prove the classifier
   reports the injected first mismatch.
4. Keep token content out of ordinary receipts; detailed bounded fixtures stay
   only in test data.

## Exit gate

- Command: focused unittest for the new oracle plus existing M15 and P58 token-
  continuity suites.
- Pass: both real-tokenizer geometries produce complete per-turn receipts; the
  poison negative is detected at the expected coordinate; existing DeepSWE and
  M15 tests remain green.
- Fail: stop before one-host code. A parser/tokenizer boundary that cannot be
  represented by the shared incremental ledger requires a helper repair first.

## Result

- Added `scripts/audit_tito_transcript.py` and a focused bounded-receipt test.
- Focused unittest: 3/3 PASS.
- Real DeepSWE input: Qwen3-4B-Instruct-2507, thinking enabled, persisted
  trajectory SHA256
  `30e44424f774f684e0d1cabdf0caf536a62da69adb54bdbdc02051c7f709f118`.
  The first later turn is equal; turns 2 through 11 are different. The first
  mismatch is token 2242 in every red turn. Audit SHA256
  `9b5235b80b57a6627387cc9848e2dcf819ebd76122e5998dcf40e22f54f98f97`.
  A bounded local decode placed token 2242 inside the second sampled assistant
  span, 171 tokens after that span began. The model emitted a malformed
  `command=view` parameter spelling: sampled transport used two tokens for the
  relevant text fragment, while full-text re-tokenization merged it into one.
  Thus the first red precedes the following environment-role wrapper and is a
  direct detokenize/re-tokenize non-involution, not a tool-vs-user role change.
- Synthetic FrozenLake input: Qwen3-8B, thinking disabled, user-only
  environment roles. Both later turns are exact. Audit SHA256
  `d8ed005febfb33d392d65766c0965fc9593316135456eb8305c5e67a942d5b51`.
  This is a parser/tokenizer mechanism check, not a real M15 verdict.
- Both audits caught the injected one-token poison at the registered
  coordinate. No raw token content is stored in the durable receipt.
- Exit: PASS for the oracle. The real M15 legacy observer remains required.
