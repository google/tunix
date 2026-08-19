# P38.2y — GSM8K fixed-lm-head full-training integration

Status: LOCALLY GATED; TARGET NOT RUN. The Qwen3-8B TP4 fixed-tile lm-head has
passed the P38s23r3 forward target and the P38.2h actual-model
backward-no-commit target. This phase promotes the same construction to the
Qwen3-1.7B TP4 GSM8K full-training lane. No target GSM8K run has executed yet.

## Objective

Run one existing 200-step GSM8K full-training campaign with all of the
following in the same production profile:

- the fixed-tile Pallas lm-head primal and fixed-order custom VJP;
- resident optimizer state;
- `CANON_BATCHED_EVIDENCE=1`;
- `CANON_P28_BATCHED_REPORT=1`;
- the already unconditional P47a prompt-logprob removal;
- the existing GSM8K A-B warning-only policy and complete per-step evidence.

`CANON_P28_BATCHED_REVERSE=1` is excluded: its one-host gate is green, but its
DP16 grouped implementation and certification are still open.

## Shape ledger

Do not collapse these rows. The first two are workload geometry; the final
three are the lm-head hook's visible program geometry.

| Quantity | GSM8K DP16xTP4 | Fixed lm-head interpretation |
|---|---:|---|
| caller-global rollout capacity | 256 trajectories | scheduler capacity only |
| per-DP-rank live request capacity | 16 | request warmup/runtime semantic M is one of 8/16/32/64/128/256 |
| caller-visible learner M | 4096 | exactly 16 ascending M256 chunks |
| canonical-kernel M | 256 | unchanged for request and learner programs |
| semantic valid rows | exact request M or 4096 | request output is sliced; learner concatenates 16 chunks |

The model dimensions are:

| Model | global weight | TP4 local weight | fixed local call |
|---|---|---|---|
| Qwen3-1.7B | `[K2048,V151936]` | `[K2048,N37984]` | `[M256,K2048] @ [K2048,N38144]` |
| Qwen3-8B | `[K4096,V151936]` | `[K4096,N37984]` | `[M256,K4096] @ [K4096,N38144]` |

Both K values divide fixed `BK=256`; N37984 is padded by the model-pinned
contract to N38144, which divides fixed `BN=256`. `BM/BN/BK=128/256/256` and
the ascending learner dWeight accumulation order remain unchanged.

## Implementation contract

1. `CANON_P38_FIXED_LM_HEAD=1` accepts only hidden sizes 2048 and 4096,
   BF16 inputs/weights, vocab 151936, and TP4. Every other geometry fails
   closed.
2. GSM8K full is the only newly admitted production lane. Existing P38
   serving-capture and FrozenLake backward-no-commit rules remain unchanged.
3. The GSM8K renderer must carry the fixed-lm-head flag explicitly; relying on
   an implicit image default is forbidden.
4. The normal P38h compact base64 return is not emitted for a 200-step GSM8K
   run. Normal P33 reports and the official classifier remain authoritative.
5. The run must emit request and learner fixed-lm-head PATHTRACE receipts; a
   missing VJP receipt is fatal.

## Gate ladder

1. static/CPU: contracts for K2048/K4096, wrong-K/TP/dtype negatives, renderer
   intent-diff, shell syntax, Python compile, manifests;
2. pinned exact image: qwen1p7b and qwen8b overlays contain and activate the
   same hook under their exact model contracts;
3. direct one-host v5p with real Qwen3-1.7B weights (PASS,
   `p38y3_20260819`):
   - request buckets versus M256 are bitwise exact;
   - learner M4096 versus sixteen completed M256 calls is bitwise exact;
   - dHidden and dWeight equal the ascending chunk oracle;
   - repeat gradients are exact, finite, nonzero;
   - a normal-value negative changes exactly one element;
4. target: one 200-step DP16xTP4 GSM8K full run. This is TARGET NOT RUN until
   separately launched from a committed source SHA.

The one-host evidence was written under `/tmp` because the persistent evidence
disk was full before TPU execution. Its sealed SHA-256 values are:

- raw log: `6912b9cbc0f81dd681a735242de84be99b36ee9ddc168913677aa012de3f8cd1`;
- forward JSON: `9b328e96b0aa8b11c441cd40e7dedd1d99beaf7266a37b8bc9ab2868f46a5d3a`;
- VJP JSON: `4bd13dfebd19a6e7ae3953e992f54b50cc4e1e3bbc0f84f1524b940ec1d27b50`.

This is local construction evidence only; publication should archive the
small receipts before `/tmp` is reclaimed. The first two attempted labels did
not enter the numerical program (empty model snapshot, then full evidence
disk) and carry no scientific result.

## Target interpretation

The full run is intentionally warning-only for A-B so infrastructure or a
remaining carrier cannot discard training value. B-C, finiteness, replica
consensus, optimizer transaction, and state-transition failures remain fatal.

- all 200 steps A=B=C: cross-workload full-training validation of the fixed
  lm-head candidate;
- A-B warning but B-C exact: training may finish as alignment-degraded; the
  Qwen3-1.7B target has not closed zero-TIM;
- any B-C, gradient, reducer, optimizer, or state-transition red: target FAIL;
- infrastructure loss: INCONCLUSIVE.

This target can validate actual optimizer commits and sustained training. It
does not replace the Qwen3-8B FrozenLake full-training target or prove other
models/topologies.

## Rollback

Remove `CANON_P38_FIXED_LM_HEAD=1` from only the `gsm8k-full` rendered env. The
existing P47a/P50 performance optimizations and optimizer placement are
independent and remain enabled.
