# Zero-TIM kernels (experimental)

Batch-invariant, fixed-reduction-order kernels for **zero training-inference
mismatch** (Zero-TIM) in RL post-training.

In on-policy RL the learner recomputes the log-probability of every sampled
token and divides it by the sampler's value (the importance ratio). With a stock
stack the two differ in the last bits: the rollout engine
and the learner run *different programs* over the same weights, with different
batch sizes, padding, fusion and collective lowering. Zero-TIM removes every
such difference so that

```
A (sampler logprob) == B (engine re-scoring) == C (learner forward)
```

holds **bitwise** for every token, independent of batch composition, row
position, padding bucket, and of whether the program is decode, prefill,
training forward or forward+backward. The backward only has to be a correct
gradient of that exact forward.

This directory holds the numerical building blocks and a canonical Qwen3
benchmark suite.

## Root causes and fixes

*   **R1, TP reduction order** (`fixed_order_reduce`). The bf16 all-reduce that
    completes the contract-parallel projections (`o_proj`, `down_proj`) and the
    vocab-sharded embedding sums the TP partials in a row-position-dependent
    order, and bf16 addition is not associative. Casting the all-reduce to f32
    does not fix it. Fix: one association order, `((p0 + p1) + p2) + p3`, on
    every rank, for every row and in every program. The ring, all-gather and
    all-to-all forms are bitwise equal.
*   **R2, attention block sizes** (`rpa_canonical`). RPA v3 looks up tuned block
    sizes by shape and clamps them by the decode concurrency, so the
    online-softmax KV block differs between decode, prefill and batch sizes.
    Fix: pin `(bq, bkv, bq_c, bkv_c) = (128, 512, 128, 512)` for decode, prefill
    and mixed batches, and `MIN_TOKEN_BUCKET=256`.
*   **R3, third-program lowering** (XLA flag). XLA keeps bf16 intermediates in
    f32 differently in a forward-only and a forward+backward program. Fix:
    `--xla_allow_excess_precision=false`, plus the fixed-order embedding of R1.
*   **R4, differentiable attention** (`rpa_diff_chunked`, `rpa_diff`,
    `rpa_canonical`). The engine's Pallas attention has no VJP, but the learner
    must run the very same forward. Fix: a `custom_vjp` whose forward is the
    kernel verbatim and whose backward differentiates a pure-JAX replica and
    routes the paged KV-cache cotangent.
*   **R5, log-softmax** (`canonical_logsoftmax`). Over 151936 columns XLA picks
    the reduction tree per shape and program. Fix: a fixed three-stage Pallas
    log-softmax (1024-column tiles, left-to-right tile combine) with 8-aligned
    row buckets up to 256 rows.
*   **P22, projections and norms** (`pallas_*`, `padded_*`, `model_contracts`,
    `canonical_vjp`). `jnp.dot` picks its tiling and K split from M, so a row's
    value depends on the batch size; RMSNorm and SwiGLU fuse differently per
    program. Fix: Pallas kernels with a fixed contraction order (BK-blocked f32
    accumulation, one bf16 store), per-model tile and zero-padding contracts,
    and replica VJPs.
*   **P38, LM head** (`fixed_lm_head`). The head runs at M = 8..256 (decode
    buckets), 256 (scoring) and 2048/4096 (learner). Fix: one fixed `[256, K] @
    [K, V/TP]` Pallas program; small M is zero-padded, large M is mapped over
    256-row chunks with `lax.map`, and dW is accumulated over the chunks in
    ascending order with `lax.scan`.

## Modules

*   `pallas_matmul`:
    every output element is the sum over K blocks of BK=256, accumulated left to
    right in f32 and rounded to bf16 once. `block_m` and `block_n` are bitwise
    neutral; only `block_k` enters the numerics.
*   `pallas_rmsnorm`: f32
    sum of squares over BF=128 feature blocks strictly left to right.
    `canonical_rmsnorm` is the bit-exact pure-JAX statement.
*   `pallas_swiglu`:
    `silu(gate) * up` in one kernel with one bf16 store, so no program can fuse
    it differently.
*   `model_contracts`: per model and TP
    degree, the tiles, the admitted K/N/SwiGLU zero padding and the seven
    projection sites. Fails closed on any unadmitted width.
*   `padded_matmul`: M zero-padded to 128
    rows, K/N padded only to contract widths, wide-N tiles. Real elements keep
    their contraction order.
*   `padded_swiglu`: contract padding around
    the SwiGLU kernel.
*   `canonical_vjp`: the Pallas kernel is the
    primal; the VJP differentiates a K-block replica or one plain dot
    (`CANON_MATMUL_VJP_PLAIN`, default on).
*   `pallas_norm_matmul`: the RMSNorm
    prologue fused into the matmul, bitwise equal to the two-kernel chain.
*   `fixed_order_reduce`: ring, gather and scatter fixed-order TP sums, the
    fixed-order embedding lookup, the P59 rank-ordered f32 backward sum and the
    P59/P66/P67 context helpers.
*   `fixed_lm_head`: the fixed-shape LM head for
    the registered Qwen3 geometries; unregistered shapes fail closed.
*   `canonical_logsoftmax`: the
    fixed-program log-softmax and gathered log-probs, including the
    continue-decode row bucket.
*   `rpa_diff_chunked`:
    differentiable chunked-cache RPA and its ragged adapter.
*   `rpa_diff`: prefill-only
    differentiable RPA with zero KV-cache gradient.
*   `rpa_canonical`: pinned block sizes, forced-mixed distribution,
    differentiable wrapper selection and the P59 local-attention checks.
*   `test_utils`: CPU test helpers (testonly).
*   `bench/qwen3`: standalone canonical Qwen3 reference model
    (`qwen3_model_canon`) and per-kernel numeric/performance ablation suite
    (`qwen3_canon_benchmark`, `qwen3_model_canon_test`).

## Canonical environment switches

*   R1: `CANON_FIXED_AR=1`, `CANON_FIXED_AR_EMBED=1`.
*   R2: `CANON_RPA_D`, `CANON_RPA_P` and `CANON_RPA_M` set to `128,512,128,512`;
    `MIN_TOKEN_BUCKET=256`.
*   R3: `XLA_FLAGS=--xla_allow_excess_precision=false`.
*   R4: `CANON_RPA_VJP2=1`, `CANON_VJP2_MAX_SEQS=1`; `CANON_RPA_VJP` is
    deliberately unset.
*   R5: `CANON_LOGPROB_M=256`, `CANON_PROMPT_PROCESSED_LOGPROBS=1`,
    `CANON_PALLAS_LOGSOFTMAX=1`.
*   Pallas ops: `CANON_PALLAS_ALL_PROJ=1`, `CANON_PALLAS_ALL_RMSNORM=1`,
    `CANON_PALLAS_SWIGLU=1`, `CANON_PALLAS_MPAD=1`,
    `CANON_PALLAS_SWIGLU_MPAD=1`, `CANON_PALLAS_CANONICAL_VJP=1`.
*   Learner forward through the engine modules: `CANON_ENGINE_MODULE_C=1`.
*   Device order: `CANON_EXPECT_MODEL_MESH_IDS` asserts that rollout and trainer
    use the same TP device order, which the fixed rank-order sums require.

The high-performance profiles add `CANON_P38_FIXED_LM_HEAD=1` and
`CANON_P59_RANK_PARALLEL_BACKWARD=1`, plus `CANON_P67_P66_VMA_P59_ONLY=1` for
P59-only workloads. Optional: `CANON_FIXED_AR_GATHER` and
`CANON_FIXED_AR_SCATTER` (cheaper R1 collectives, bitwise equal to the ring),
`CANON_P66_P59_CHECK_VMA`, `CANON_CONTINUE_DECODE` and `CANON_LOGPROB_M_BUCKET`
(default on). The modules here read only the flags that change their numerics or
selection; the engine enable switches are the caller's business.

## Usage

```python
from tunix.experimental.zero_tim_kernel import canonical_logsoftmax
from tunix.experimental.zero_tim_kernel import canonical_vjp
from tunix.experimental.zero_tim_kernel import fixed_lm_head
from tunix.experimental.zero_tim_kernel import fixed_order_reduce
from tunix.experimental.zero_tim_kernel import model_contracts
from tunix.experimental.zero_tim_kernel import padded_matmul

contract = model_contracts.get_contract("qwen8b")  # Qwen3-8B at TP=4.

# Column-parallel projection: fixed tiles and padding, differentiable.
forward = lambda a, b: padded_matmul.matmul(a, b, contract=contract)
y = canonical_vjp.matmul(x, w, forward=forward, contract=contract)

# Contract-parallel projection (o_proj/down_proj) completed in a fixed order.
out = fixed_order_reduce.contract_parallel_projection(
    h, w_down, equation="mn,np->mp", mesh=mesh, tp_axis="model",
    local_matmul=forward)

# LM head and log-probs through one fixed program.
logits = fixed_lm_head.fixed_lm_head(
    hidden, lm_head, mesh=mesh, tp_axis="model",
    local_matmul=fixed_lm_head.canonical_local_matmul(contract),
    endpoint="untied_lm_head")
logprobs = canonical_logsoftmax.gathered_logprobs(logits, token_ids)
```

Attention: build the kernel call with `rpa_canonical.pinned_block_size_kwargs()`
and wrap it with `rpa_canonical.differentiable_rpa(kernel_fn, ...)` for
training.

## Caveats

*   The bitwise guarantees are properties of the Mosaic TPU compilation of the
    kernels (with `shape_invariant_numerics=True`). The CPU tests run Pallas in
    interpret mode, which proves invariance and exactness, not TPU bits.
*   All programs must run with `--xla_allow_excess_precision=false`.
*   `CANON_VJP2_MAX_SEQS` bounds how many sequences the ragged RPA backward
    unrolls (default 1): later sequences silently get **zero** gradient. The
    default exists because unrolling 64 sequences x 64 layers stalled
    compilation.
*   The modules are building blocks: bitwise A == B == C additionally needs the
    engine and the learner to route every site through them with the same
    contract, mesh order and device order.
*   Requires JAX >= 0.8 (`jax.shard_map`, `jax.typeof`, `jax.enable_x64`).

## Tests

The tests run on CPU with Pallas in interpret mode. They check what a CPU can
prove: same-shape batch invariance (a row inside a larger batch is bitwise the
row computed alone), exact answers on operands whose every partial sum is exact
in f32 (`test_utils.exact_bf16`), closeness to float64 references on random
operands, and the collective algebra (ring == gather == scatter, bitwise) on 4
CPU devices.

```shell
pytest tests/experimental/zero_tim_kernel/
```
