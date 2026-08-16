# Distributed shape contracts

Load this reference before changing a numerical boundary under DP/TP, a bucket, padding,
`PartitionSpec`, `shard_map`, processed logprobs, or a fixed-shape kernel.

## Contents

- Five independent quantities
- DP16 processed-logprob example
- Why the historical assertion failed
- Required tests for a shape change
- Other shape rules
- Scheduler and precompile ledger

## Five independent quantities

Always write all five values explicitly:

1. **Caller-global M**: logical rows at the distributed API boundary.
2. **Shard-local M**: rows one DP rank sees after sharding.
3. **Canonical-kernel M**: static rows compiled by the numerical kernel.
4. **Semantic valid rows**: rows retained for gather, loss, or update.
5. **Scheduler/precompile capacity**: per-rank request/token limits and the global buckets they
   cause the runner to compile.

These values may be equal in a DP1 probe and different in production. A DP1 test therefore does
not cover this contract.

## DP16 processed-logprob example

The current P33/P34 contract uses:

```text
DP size                     16
canonical kernel M          256 per rank
global padded prefill M     16 * 256 = 4096
global compact rollout M    256
compact rows per rank       256 / 16 = 16
```

Valid flows:

```text
rollout global [256,V]
  -- PartitionSpec("data", None) --> local [16,V]
  -- local zero pad -----------> local [256,V]
  -- canonical log-softmax ---> local [256,V]
  -- slice valid rows ---------> local [16,V]
  -- gather -------------------> global [256,...]

prefill global [4096,V]
  -- PartitionSpec("data", None) --> local [256,V]
  -- canonical log-softmax -------> local [256,V]
  -- gather ----------------------> global [4096,...]
```

Both paths compile the numerical log-softmax at local M256. Only distribution and valid-row
selection differ. An unregistered global M512 must fail before compilation.

## Why the historical assertion failed

The old wrapper asserted `caller-global M == 4096` before sharding. That is correct for full
prefill but rejects compact rollout M256. The kernel did not fail; the wrapper confused its
global API contract with the kernel's local static contract.

Do not fix this by changing the kernel to M16. That would create a second numerical program and
violate the fixed-M alignment principle. Do not accept arbitrary M either. Admit only registered
global shapes and map both to the same local kernel M.

## Required tests for a shape change

1. Force the intended device count and mesh, not a DP1 approximation.
2. Record the shape seen inside the mapped kernel for every valid caller shape.
3. Assert both valid paths see the same canonical-kernel M.
4. Assert output shape/sharding and retained values are exact.
5. Reject one adjacent unregistered global shape.
6. Run the neighboring recipe suite in the pinned image.
7. Re-run the real target boundary before claiming bitwise alignment.

## Other shape rules

- Derive global `MIN_TOKEN_BUCKET = DP * local_M`; do not copy the DP1 value.
- In the current TPU runner, `max_num_batched_tokens` and `max_num_seqs` are **per DP rank**. The
  runner multiplies them by `dp_size` before building global scheduler geometry. Keep both values
  separate from canonical kernel M.
- Keep `CANON_VJP2_MAX_SEQS` equal to the per-call differentiable sequence contract, not scheduler
  capacity or global batch size.
- Keep TP width separate from DP row sharding. Changing TP can change collective order even when M
  is unchanged.
- Compare semantic masks from producer metadata. Never infer validity from pad token id or array
  width.

## Scheduler and precompile ledger

For every recipe, record and test:

```text
DP size
canonical local M
global MIN_TOKEN_BUCKET
per-rank max_num_batched_tokens
per-rank max_num_seqs
expected global token buckets
expected global request capacity
expected backbone precompile count
```

The signed DP16/local-M256 example is:

```text
MIN_TOKEN_BUCKET=4096
max_num_batched_tokens=256 per rank
max_num_seqs=16 per rank       # global concurrency 256
Prepared token paddings=[4096]
worker0 backbone: num_tokens=4096, num_reqs=256
```

Passing `4096` and `256` as the two per-rank maxima is not equivalent. The runner multiplies them
again and historically produced token buckets `[4096,8192,16384,32768,65536]` and request
capacity 4096, compiling five full backbone shapes before training.

`max_num_batched_tokens=256` limits one scheduler step per rank; it does not cap prompt or context
length. Long prompts remain valid through chunked prefill. Do not infer scheduler buckets from
inner PATHTRACE dimensions: head flattening can turn token M256 into matrix M8192 or M32768.
Use the runner's prepared-token-padding and backbone-precompile lines as the authoritative record.

For separated rollout/trainer roles, compute this ledger from each role's own DP width and capacity.
Never multiply by total cluster devices or combine both roles into one scheduler geometry.

The current DeepSWE role contract uses DP16xTP8 independently for rollout and trainer. Its
per-rank scheduler request limit is 4, so the expected global request capacity is 64 while the
global token bucket remains 4096 and the local canonical kernel remains M256. Do not inherit the
P33 per-rank request limit of 16 merely because both profiles use DP16 and local M256.
