# Full-capture bitwise comparison

`compare_capture_bits.py` compares two independently sealed FrozenLake P61
captures. It proves equality only at the **backward-no-commit diagnostic
boundary**. It does not certify A/B/C alignment, Adam updates, performance,
capture-observer neutrality, or a final source revision by itself.

Before comparison, a reviewer fixes a JSON contract with exactly these fields:

- `schema`: `canon-p61-bitwise-contract-v1`.
- `gradient_leaves`: the complete parameter count (399 for the historical 8B
  capsule, not a universal model constant).
- `common`: the full run manifest excluding only `label`, `source_commit`,
  `source_diff_sha256`, and `runner_sha256`. Both runs must match it exactly.
- `left` and `right`: each contains `source_commit` (full 40-character SHA),
  `runtime_capture_root` (the absolute capture directory as seen by the producer
  process), and `files` (the following seven relative paths mapped to
  independently checked SHA256 values). The runtime capture root maps to the
  host run's `p61_numerical/`; review the launch mount/env contract rather than
  deriving this expected path from whatever appears in the log.

```text
run_manifest.json
raw.log
p61_numerical/algo_config.json
p61_numerical/model_before/manifest.json
p61_numerical/example/manifest.json
p61_numerical/logps/manifest.json
p61_numerical/gradient/manifest.json
```

The contract binds the clean source, image, model, input dataset, algorithm,
topology, selectors, replay capsule and its producer. Review the producer
against the actual capsule producer; it is not the measurement label. The
comparison tool does not generate or refresh its own expected hashes. A
contract generated solely from whichever candidate files happen to exist is
not independent provenance review.

Each tree manifest binds every payload file and raw array bytes. The tool checks
the complete file inventory, unique paths, index, shape, dtype, finite values,
full parameter/gradient coverage, and each byte digest. Equal norms are not
used. Signed zero and low-bit differences are differences. Both logs need
plain-VJP and checked-VMA receipts plus all four capture terminal markers;
VMA receipt count may not decrease. All completion and plain-VJP markers are
parsed before validating their values; malformed or conflicting duplicates
cannot disappear through an expected-value-only regex. Each manifest path
must match the reviewed runtime-to-artifact mapping exactly.

Run on CPU with two **physical**, independent run roots and a fresh output:

```bash
JAX_PLATFORMS=cpu python canon-zero-tim/tests/p61_backward/compare_capture_bits.py \
  --left-root /physical/old-plain-run \
  --right-root /physical/final-plain-run \
  --contract /reviewed/capture-contract.json \
  --output /fresh/bitwise-result.json
```

Exit 0 is `FULL_CAPTURE_BITWISE_EQUAL`; 1 is different or invalid evidence;
2 is `INCONCLUSIVE_NO_SIGNAL` (equal all-zero gradients). Existing output is
never overwritten. Tool and contract hashes are included in the receipt.
The no-commit capture's host microbatch accumulation is not interchangeable
with runtime optimizer accumulation. An observer-neutrality argument and the
separate training gates are still required before transferring admission.

Regression gate:

```bash
JAX_PLATFORMS=cpu python -m pytest -q canon-zero-tim/tests/p61_backward/test_compare_capture_bits.py canon-zero-tim/tests/p61_backward/test_compare_full_trees.py
```

The tests use complete small synthetic captures and rejection controls, not
hardware evidence. The older `compare_full_trees.py` retains its separate
DP4/TP1 committed-update contract unchanged.
