# P44.2 — Qwen3-4B TP8 engine adapter

- Status: passed

## Finding

- Confirmed: Qwen3-4B has hidden width 2560, intermediate width 9728, 32 attention heads, 8 KV heads, and 36 layers.
- Confirmed: At TP8 the seven local projection shapes include intermediate width 1216, which is not divisible by the historical fixed BK256 VJP chunk.
- Hypothesis: Selecting the model contract's declared BK in the canonical VJP replica will preserve Qwen3-1.7B/8B behavior, honor the existing Qwen3-32B BK128 declaration, and admit Qwen3-4B with BK64.

## Execution

1. Add a Qwen3-4B TP8 model environment and exact seven-site projection contract.
2. Add the RMSNorm overlay and model manifest.
3. Make the canonical VJP replica consume the model-local BK contract.
4. Run self-tests, manifest checks, and adjacent model overlay tests.

## Exit gate

- Command: `bash canon-zero-tim/tests/p44_deepswe_qwen4b_parity/run_cpu.sh`
- Pass: Qwen3-4B registry geometry, TP8 projection shapes, VJP chunk selection, and model manifest tests pass without changing P43/P34 gates.
- Fail: Keep P44.2 active and do not wire the model into a remote renderer.

## Result

Digest-pinned exact-image gate passed: 29/29 installed files matched, P44 CPU
tests passed 9/9, and the Qwen3-4B TP8 projection self-test passed 5/5.
