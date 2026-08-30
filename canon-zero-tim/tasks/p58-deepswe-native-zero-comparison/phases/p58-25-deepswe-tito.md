# P58.25 — DeepSWE token-in/token-out continuity

Status: implemented locally; host and digest-pinned exact-image construction
pass; target not run.

## Problem

DeepSWE is an interactive token-in/token-out workload.  The sampled assistant
token IDs from turn N must be part of the exact prompt token IDs sent to turn
N+1.  The historical multi-turn path instead rebuilt later prompts from chat
text.  Decode followed by tokenizer encode is not an identity operation for
all valid Qwen token streams, so a later prefill could consume different IDs
even when the rendered text looked identical.

P58.22 implemented exact continuity, but its admission was restricted to the
special direct-host Qwen3-4B TP4 Zero carrier.  P58 Native, Native+IS, target
DP8xTP8 Zero, Qwen3-32B, and other admitted DeepSWE profiles still took the
text/re-tokenize continuation path.  That scope boundary was the bug.

## Repair contract

- TiTO is a shared DeepSWE transport invariant, not a Zero-TIM treatment.
  Native, Native+IS, Zero, diagnostics, production, and one-host DeepSWE all
  use it.  Non-DeepSWE agentic workloads remain unchanged.
- The first-turn prompt is anchored to the rollout worker's actual prompt
  token IDs and valid prompt length.
- Each assistant segment uses the exact IDs returned by rollout.  It is never
  reconstructed from decoded assistant text.
- R2E-Gym returns environment observations as text, so each new observation
  is encoded exactly once with the signed chat parser.  Those resulting IDs
  are stored and reused; prior observations are never re-rendered.
- A continuation prompt is the integer concatenation
  `initial_prompt + assistant_0 + environment_0 + ...`.  It is sent to the
  rollout engine with `apply_chat_template=False`.
- Missing/non-integer assistant or environment tokens, inconsistent prompt
  length, caller-owned prompt token overrides, and overlapping production /
  one-host selectors fail closed.
- Native/Zero comparison differences remain numerical-algorithm differences
  only.  TiTO is deliberately identical in both arms.

The model's decoded text is still consumed by the DeepSWE action parser and
the environment still originates text.  Neither fact permits sampled token
IDs to be decoded and re-tokenized for the next model prompt.

## Executable evidence

Startup emits exactly one:

```text
[DEEPSWE.TITO] ADMISSION_PASS ... mode=token-in-token-out retokenize_sampled_tokens=0
```

Every exercised turn after turn zero emits an integer-stream receipt:

```text
[DEEPSWE.TITO] CONTINUATION turn=<n> prompt_tokens=<n> sha256=<64-hex>
```

P58 postflight requires one admission receipt and at least one continuation
receipt for ordinary Native/Zero training.  Pre-backward recorded diagnostics
retain the admission check but are exempt from the live continuation count.

## Gates and claim ceiling

- Python and Bash syntax plus diff hygiene: PASS.
- Selector, one-host, sampler, runtime-receipt, and negative contracts: PASS.
- Full local renderer contract: bare-host invocation is unavailable because
  `metrax` is absent; this invocation is not represented as a PASS.
- Digest-pinned exact-image gate: PASS on image ID
  `sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a`.
  It ran the real renderer/environment boundary and focused agentic tests,
  observed a `[DEEPSWE.TITO] CONTINUATION` receipt, and ended with
  `P58_EXACT_IMAGE_CPU_PASS ... regressions=1`.
- P34 static: PASS, ten suites.
- Flag registry: PASS, 409/409 names; P58.25 introduces no new flag name.
- Fresh real one-host / DP8xTP8 run: not run.

This implementation proves the source no longer reconstructs a DeepSWE
continuation from sampled assistant text.  It does not yet prove that TiTO
alone removes the K04 DP8xTP8 A-B residual; that requires a fresh target run
with the receipts above. Source commit/push was explicitly authorized for this
delivery. Launch and image publication remain separately unapproved.
