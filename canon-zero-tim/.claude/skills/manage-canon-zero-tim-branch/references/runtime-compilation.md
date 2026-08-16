# Runtime compilation contract

Load this reference before changing TPU scheduler limits, Pathways precompile behavior, or JAX
compilation-cache configuration. Compilation performance and numerical alignment share shape
inputs, but they have different evidence rules.

## Contents

- Keep five quantities separate
- Interpret compile output correctly
- Persistent JAX cache is performance-only
- Deliver Pathways compiler flags at the compiler boundary
- Preserve the Pathways session contract
- Required gates

## Keep five quantities separate

Record caller-global M, shard-local M, canonical-kernel M, semantic valid rows, and scheduler
capacity independently. Scheduler limits decide which whole-model executables are compiled;
padding inside the canonical adapter decides which fixed local numerical program processes valid
rows. Neither value substitutes for the other.

In the current TPU inference runner:

- `MIN_TOKEN_BUCKET` is global.
- `max_num_batched_tokens` is per DP rank and is multiplied by `dp_size`.
- `max_num_seqs` is per DP rank and is multiplied by `dp_size`.
- `get_token_paddings` enumerates every global backbone token shape to precompile.

For DP16 with canonical local M256 and global concurrency 256, register:

```text
MIN_TOKEN_BUCKET=4096
max_num_batched_tokens=256 per rank
max_num_seqs=16 per rank
expected token buckets=[4096]
expected global request capacity=256
expected backbone precompile count=1
```

The compact rollout still follows global256 -> local16 -> pad256 -> slice16. Full prefill follows
global4096 -> local256. Both reach the same canonical local-M256 kernel.

## Interpret compile output correctly

Use `Prepared token paddings` and worker backbone `num_tokens`/`num_reqs` as authoritative scheduler
geometry. Inner PATHTRACE M values may flatten token and head axes and are not additional token
buckets.

`max_num_batched_tokens=256` limits one scheduler step per rank, not the context length. Chunked
prefill can process a 4096-token prompt through repeated scheduler steps. Do not increase the
per-rank limit merely because the model accepts long contexts.

For DeepSWE or another split-role system, compute rollout and trainer ledgers independently from
their own DP widths. Do not use total cluster devices as either role's scheduler DP size.

## Persistent JAX cache is performance-only

Use `JAX_COMPILATION_CACHE_DIR` for the TPU persistent executable cache. The compilation cache is
enabled by default in the pinned JAX runtime. `JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHING` is an
invalid legacy variable. The recognized `JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=all` controls GPU
kernel/autotune cache options in this JAX version; it does not enable TPU executable caching.

Build a cache namespace from the complete executable contract:

```text
source SHA + image digest + JAX/jaxlib/libtpu versions + topology + role/profile
+ scheduler/shape ledger + precision/XLA flags
```

Use a registered nonzero minimum compile-time threshold unless the experiment intentionally wants
every tiny entry. Make cache pull/upload failures visible. Report direction, object count, bytes,
and return code without exposing credentials. Sync periodically and on `EXIT`; a hard numerical
gate may terminate the job after an expensive compile.

Enable `JAX_LOG_COMPILES=1` and `JAX_EXPLAIN_CACHE_MISSES=1` while validating cache behavior.
Existing files are not proof of a cache hit. A hit is not proof of numerical correctness.

## Deliver Pathways compiler flags at the compiler boundary

Pathways compiles TPU programs in the server-side proxy process. Set compiler `XLA_FLAGS` in the
`pathways-proxy` container environment. Do not use either of these as delivery evidence:

- `XLA_FLAGS` present only in the JAX client container;
- a raw `--xla_*` proxy argument accepted or rejected by the proxy's absl command parser.

Statically assert the rendered proxy environment, archive the proxy startup log, and run a target
way-count or production-boundary single-variable control. When comparing flag OFF and ON, pin the
source, image, topology, input digest, scheduler ledger, and cache namespace; otherwise the pair
does not establish causality. A client-side environment check is only a client contract.

## Preserve the Pathways session contract

Assume one live JAX client per Pathways slice unless the pinned release explicitly documents and
tests otherwise. Repeated short-lived clients can cancel or poison the session even when each
client only calls `jax.devices()`.

Prefer this sequence:

1. Use non-JAX proxy/RM/worker readiness markers to wait for the full registered slice.
2. Start the final long-lived training process once.
3. Before model work, initialize Pathways, require the exact device count, execute a tiny JIT, and
   keep that process alive for the workload.

For the current local-proxy head, use `JAX_PLATFORMS=proxy,cpu` and
`JAX_BACKEND_TARGET=grpc://localhost:29000`. Treat a missing URL scheme, partial device count,
socket close, or client cancellation as infrastructure `INCONCLUSIVE`. Archive proxy and resource
manager logs before deleting the JobSet.

Keep the reviewed `ENABLE_PATHWAYS_PERSISTENCE` setting consistent across workers. Persistence is
an operational setting, not a numerical gate and not permission to create a second client.

Require the runbook's cluster-scoped `PriorityClass` on all head and worker Pods. Verify its exact
name, value, and preemption policy read-only before apply. It reduces lower-priority preemption but
does not prevent maintenance, OOM, IFRT failure, or loss of an uncheckpointed run.

## Required gates

1. Static arithmetic test for global/per-rank units.
2. Exact-image call to the pinned runner's real bucket function; require exact list equality.
3. Negative control using the historical global-as-local configuration; require rejection.
4. Target log gate for bucket list, request capacity, and compile count.
5. Runtime gate rejecting unexpected larger-shape compiles or cache misses.
6. Warm/cold comparison preserving all A/B/C, THIRDPROG, gradient, and update gates.

Keep precompile enabled. Reducing the registered bucket family is a contract change; setting
`SKIP_JAX_PRECOMPILE` only moves compilation into training and cannot be claimed as an optimization
or numerical fix.
