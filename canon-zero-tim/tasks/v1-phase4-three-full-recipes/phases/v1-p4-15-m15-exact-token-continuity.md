# V1.P4.15 — M15 exact token continuity (TITO)

Status: active; user overrode the verify-first prerequisite and selected exact
TITO as the signed M15 full default. T1/T2 runtime, YAML/profile delivery, and
postflight hard gates are published/read back at `3fc7ef8b`; host and
post-rebase exact pinned-image construction pass. One-host and DP8xTP8 target
are not run.

## Problem and evidence boundary

DeepSWE P58 proved on a real Qwen3-4B DP1xTP4 carrier that rendering later
agent turns as text and tokenizing them again can give a different BPE sequence
for the same visible text. Its exact-token repair restored A=B=C on that
carrier. That claim does not transfer to M15, Qwen3-8B, or DP8xTP8.

The generic agentic trajectory engine still sends rendered chat text after the
first turn unless the exclusive P58 TP4 selector is active. It separately
retains the sampled assistant token IDs and tokenized environment token IDs
that form the trainer sequence. M15 is therefore structurally exposed to the
same decode/text/re-tokenize seam, but existing APC-off M15 controls have also
returned strict A=B=C and prior FrozenLake reds have independently localized
to serving checked-VMA leakage or APC/RPA. Retokenization is a hypothesis, not
the current M15 verdict.

## Intended outcome

Feed the accumulated integer token IDs back to vLLM by default for the exact
M15 full recipe and prove that serving consumed those same IDs on every later
turn. The user explicitly authorized this input change before a real M15
verify verdict; it remains target-unverified and cannot inherit P58 evidence.
Keep the already published finite A-B warning lane so a 300-update convergence
concept run can continue through residual decode/prefill program drift. TITO
never weakens token equality, B-C, nonfinite, backward, replica, or optimizer
gates.

## Shape and identity ledger

- workload: FrozenLake M15/main only;
- model/tokenizer: Qwen3-8B, exact published snapshot identity;
- target topology: rollout DP8xTP8 and trainer DP8xTP8;
- target producer: 32 prompts x 8 generations = 256 trajectories;
- horizon: 300 updates, no evaluation, no checkpoint;
- initial prompt tokens: the non-padding tail returned by serving on turn 0;
- continuation tokens: exact sampled assistant IDs, including the registered
  assistant-end tokens, followed by exact nonterminal environment IDs;
- actual serving prompt: the unpadded integer prompt consumed by vLLM for that
  turn;
- TITO invariant: actual serving prompt IDs equal the accumulated integer IDs
  in length and every element before sampling begins.

Token equality is independent of DP/TP topology at the host boundary, but
one-host evidence cannot certify the distinct DP8xTP8 serving executable or
its A-B values.

## Phases and gates

### T0 — observer-only token-continuity verifier

Add a default-off `verify` mode that leaves the existing text prompt and
sampling request unchanged. After each model call it compares the actual
unpadded prompt IDs returned by serving with the exact sequence reconstructed
from the trajectory. Persist only bounded metadata: turn, lengths, hashes,
first mismatch position, and the two token IDs at that position.

Exit gate:

- observer-neutral positive: verify-off and verify-on produce identical
  rollout tokens/logprobs for a fixed carrier;
- same-text/different-BPE positive makes the observer fire;
- malformed/missing assistant or nonterminal environment arrays fail closed;
- M15 supplies a durable `TOKEN_STREAM_EQUAL` or `TOKEN_STREAM_DIFFERENT`
  verdict before any TITO behavior is enabled.

Historical preregistration: if M15 were `TOKEN_STREAM_EQUAL` through the first
observed A-B red, TITO would have been exonerated before changing production
input. The user superseded that sequencing rule for the signed M15 full recipe
on 2026-08-30; the observation remains useful but no longer blocks exact mode.

### T1 — generic helper and exact-M15 selector

Land two separable concerns:

1. extract P58's reconstruction into a generic helper while keeping P58
   behavior and markers unchanged;
2. register an absence-sensitive enum selector for M15 with values
   `verify|exact`, globally absent. `verify` remains observational. The exact
   M15 Zero v1-hp profile and YAML default to `exact`, which supplies integer
   prompt IDs and disables chat-template reapplication only for that tuple.

The selector must traverse renderer -> profile -> `00_env.sh` -> authoritative
`env.sh` -> learner process -> runtime marker -> postflight. P45, GSM8K,
Native, IS, eval, diagnostics, and DeepSWE production are neighboring negative
controls. Do not overload the alignment-warning flag to select token input.

Exit gate: M15 exact mode makes the vLLM-consumed prompt equal the accumulated
IDs on every later turn; any token mismatch is fatal even though finite A-B
logprob drift is warning-only.

### T2 — host and pinned-image admission

Required coverage:

- identical visible text with two different BPE segmentations;
- multi-turn assistant/environment concatenation including end tokens;
- prompt padding removal and exact length accounting;
- missing arrays, negative IDs, caller overrides, and wrong mode values;
- exact M15 identity positive and P45/Native/IS/eval negatives;
- installed sampler preserves pre-tokenized input without re-encoding;
- all existing P58 token-continuity tests remain green;
- P57, V1, flag audit, syntax, diff hygiene, and the complete pinned-image
  terminal pass.

This is construction evidence only.

### T3 — one-host mechanism

Use the exact Qwen3-8B tokenizer and a real FrozenLake M15 task on one v5p host.
First run verify mode to classify the legacy text path. Then run exact mode
with the same bounded carrier, require prompt-token equality, strict A=B=C,
zero backward or an explicitly pre-registered backward-no-commit boundary,
and zero optimizer commits. Freeze unique run directories and raw logs.

One-host green proves token plumbing and observer neutrality, not DP8xTP8.

### T4 — DP8xTP8 full target

After separate commit/push, image, render, and launch approvals, start one
fresh M15 300-update full run with TITO exact and the existing M15 A-B warning
policy. Require:

- token equality hard PASS on every later turn;
- B-C, T-current/r, all nonfinite, gradient, replica, and optimizer faults
  remain fatal;
- every finite A-B warning and its direct w/wr/clip/TIS consequence is counted
  durably;
- complete first-update, P59/P67, timing, XProf/Perfetto, and 300-commit
  receipts.

The full run is `convergence-only / alignment-degraded` if the warning policy
is enabled, even when its observed A-B warning count is zero. A later strict
warning-off target is required to recover a Zero-TIM claim.

## Decision table

| Observation | Verdict | Next action |
|---|---|---|
| legacy prompt IDs differ and exact mode restores prompt equality plus A=B=C | TITO cause admitted for the bounded carrier | progress to target gate |
| legacy prompt IDs differ, exact prompt equality holds, finite A-B remains | token seam fixed; serving program still differs | allow concept run under existing A-B warning only |
| prompt IDs are equal through the first A-B red | TITO exonerated for that incident | stop; return to serving/RPA localization |
| exact mode prompt IDs differ | implementation RED | repair at current gate |
| any B-C, nonfinite, backward, replica, or optimizer red | numerical RED | stop; warning policy cannot admit it |

## Claim ceiling, downside, and rollback

TITO changes the actual model input and can change rollout tokens, context
length, rewards, cache behavior, and throughput. It is a numerical/data
identity repair, not a pure performance optimization. A mechanism green does
not transfer from P58, TP4, or one host to M15 DP8xTP8.

Rollback is the exact M15 selector and its profile/postflight admission. The
generic helper must leave the existing P58 path byte-for-byte equivalent. The
published M15 finite A-B warning lane and all failed evidence remain intact.
No rollback deletes a run directory.

## Result log

- 2026-08-30: user approved opening the M15 TITO phase while retaining a
  relaxed finite A-B gate. Pre-registered that only finite A-B and its direct
  ratio consequences remain warnings; token inequality, B-C, nonfinite,
  backward, replica, and optimizer faults remain fatal. No runtime code,
  commit, push, render, Kubernetes object, or TPU run existed at phase open.
- 2026-08-30: T0 observer implemented. The exact M15 admission parser and its
  deliberate rejection of `exact` are at
  `tunix/rl/agentic/token_continuity.py:24-101`; fail-closed integer token
  reconstruction/unpadding and bounded equality receipt are at
  `tunix/rl/agentic/token_continuity.py:104-258`. The trajectory engine reuses
  that helper for the existing P58 exact path at
  `tunix/rl/agentic/trajectory/trajectory_collect_engine.py:659-667`, while
  M15 verify observes only after the rendered-text model call at
  `tunix/rl/agentic/trajectory/trajectory_collect_engine.py:758-781` and
  rejects caller token overrides at lines 123-130. Verified by host M15 5/5,
  P57 181/181, V1 91/91, flag registry 409/409, syntax and diff hygiene; the
  complete pinned-image gate exited zero with `V1_HP_EXACT_IMAGE_PASS ...
  m15_token=1 ... manifests=3`, and the final affected-image rerun passed M15
  5/5 plus trajectory integration 3/3. Not verified on one-host v5p or real
  M15 DP8xTP8 because no TPU launch was approved. No M15
  `TOKEN_STREAM_EQUAL|DIFFERENT` target verdict exists, so `exact` remains
  unreachable and production profiles/renderers are unchanged.
- 2026-08-30: user explicitly overrode the verify-first prerequisite and
  selected exact TITO as the M15 full default. Local implementation admits
  `exact` only for the signed M15/main Zero v1-hp DP8xTP8 300-update,
  no-eval/no-checkpoint identity; the first turn establishes the actual
  serving prompt tail, and every later turn passes the reconstructed integer
  stream with chat-template reapplication disabled. Serving-returned prompt
  IDs are checked exactly and any mismatch is fatal. The M15 YAML/profile now
  write `CANON_M15_TOKEN_CONTINUITY=exact`; P45/GSM8K/Native/IS/eval and other
  profiles require the key absent. The full classifier requires a runtime env
  marker plus one or more exact/equal receipts and rejects verify, drift, or
  neighboring leakage. Focused helper 5/5 and renderer 14/14 passed locally.
  Full host gates passed: P57 181/181, V1 92/92, classifier 29/29, and flag
  audit 409/409. The complete immutable-image gate on
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`
  exited zero with `V1_HP_EXACT_IMAGE_PASS ... m15_tito_exact=1 ...
  manifests=3`; the raw console log was not durably saved, so this is an
  admission receipt rather than a signed artifact. Not verified on one-host
  or DP8xTP8 target because no TPU launch was approved. Commit, push, render,
  launch, and TPU remain unrun.
- 2026-08-30: rebased the exact-M15 CL over publication tip `18f29c56`, which
  contains generic DeepSWE TITO. The first post-rebase pinned-image run stopped
  at trajectory construction because one automatically merged mutual-exclusion
  check still referenced the removed `_p58_exact_token_continuity` field.
  Replaced it with `_deepswe_exact_token_continuity`, retained independent
  DeepSWE and M15 admissions, and added learner plus trajectory mutual-exclusion
  negatives. Post-fix P57 181/181, V1 92/92, flag audit 409/409, syntax, and
  diff hygiene pass. The complete pinned-image rerun exited zero with
  `V1_HP_EXACT_IMAGE_PASS ... m15_tito_exact=1 ... manifests=3`. Not verified
  on one-host or DP8xTP8 target. No render, launch, Kubernetes mutation, or TPU
  use occurred.
- 2026-08-30: fast-forward published runtime/delivery commit
  `3fc7ef8b93426d0b9ec6b1b9e133198f0b37aa45`. A post-push fetch returned the
  same SHA for local HEAD, FETCH_HEAD, and the remote-tracking branch. This is
  source publication only: no manifest render, launch, Kubernetes mutation,
  TPU use, one-host mechanism run, DP8xTP8 target, or optimizer update occurred.
