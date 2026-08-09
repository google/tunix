# P35.2 r26 memory-space probe

Date: 2026-08-09 UTC

## Target observation

The source-pinned r26 run reached native arm A, reference logprob execution and two arm-B
metadata records. It then stopped before the immutable P35 report while comparing mapped trainer
weights with live engine weights:

```text
ValueError: memory_space of all inputs passed to `eq` must be the same.
Got one operand with type: uint8<host>[151936,2048,2]
and another operand with type: uint8[151936,2048,2]
```

Artifact:
`canon-zero-tim/debug_logs/p35_r26_gsm8k_envelope.raw.log`

SHA-256:
`3384f01a6864e549c3f8630653e9733465e8556e41a72a922897dbd657a54a0f`

This run emitted no complete P35 report and no classification. It is not a carrier verdict.

## One-host v5p reproduction

Resource: existing `aaron-v5p-node6`, `v5p-8`, four chips, state `READY`.

Image: `tunix_frozenlake_image:vllm-tpu0.25.0`.

JAX: `0.10.2`.

The probe placed equal bf16 arrays in otherwise identical shardings whose memory kinds were
`pinned_host` and `device`. The current comparison shape reproduced the r26 exception before any
numerical comparison completed:

```text
JAX 0.10.2 devices 4 memory pinned_host device
ValueError: memory_space of all inputs passed to `eq` must be the same.
Got one operand with type: uint8<host>[8,2] and another operand with type: uint8[8,2]
```

## Same-memory positive and negative controls

The host array was explicitly placed into the device `NamedSharding` before invoking the same
bitcast-to-uint8 equality reduction:

```text
MEMORY_BEFORE pinned_host device
MEMORY_AFTER device device
EXACT_EQUAL True
ONE_VALUE_NEGATIVE_CONTROL_EQUAL False
```

The positive control proves that an explicit same-memory placement admits exact comparison. The
negative control proves that the comparison still detects changed data; the placement does not
turn the gate into a constant pass.

## Classification and next gate

- Confirmed cause: the diagnostic passed two different JAX memory-space types to one `eq`.
- Not a numerical mismatch, model failure, OOM, precision issue or P35 carrier classification.
- The smallest repair is to normalize both leaves to one explicit device sharding before the
  exact bytewise reduction. The comparison must remain bitwise and retain signed-zero and one-bit
  negative controls.
- A target r27 remains required. The one-host probe validates the JAX placement contract; it does
  not validate centralized Pathways execution or classify A/B/C.

## Implemented repair and final local gates

`tunix/rl/canonical_qwen3_adapter.py` now inspects both explicit memory kinds before the exact
comparison. When exactly one operand is already in device memory, it places the other operand in
that existing device sharding and then runs the unchanged bitcast-to-uint8 reduction. Two
different explicit non-device memory kinds fail closed. The returned weight attestation records
the observed memory-kind pairs and the number of leaves normalized to device memory.

Added gates:

- host-left/device-right placement unit control;
- device-left/host-right placement unit control;
- incompatible non-device memory spaces rejected;
- one-host TPU direct mixed-memory rejection control;
- one-host TPU equal values in both operand orders;
- signed-zero and one-bit negative controls.

Final results:

```text
canonical_qwen3_adapter_test: 31 tests PASS, 5 skipped
P33/P35 CPU gate: PASS
exact image: qwen1p7b 10/10, qwen8b 10/10, manifest 29/29 each
one-host v5p: equal=1 reversed_equal=1 signed_zero_equal=0 one_bit_equal=0
git diff --check: PASS
```

The first one-host exact-code attempt was invalid because the stock image lacked the branch's
`canonical_logsoftmax` module. The final gate mounted the complete required canonical module set
read-only and exercised the actual modified helper. The invalid import attempt is not counted as
a numerical test.

Rollback: leave `CANON_P35_ENVELOPE` unset. No production path was changed by this probe.
