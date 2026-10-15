# Zero-TIM Kernels

Batch-invariant, fixed-reduction-order Pallas TPU kernels and model contracts
for **zero training-inference mismatch** (**Zero-TIM**) in
RL post-training (`tunix/zero_tim_kernel`).

In on-policy RL (GRPO, PPO), the learner recomputes the log-probability of every
sampled token and divides it by the sampler's log-probability (the importance
ratio). On a stock JAX/XLA stack, the two differ in the low-order bits because
the rollout engine and the learner compile and execute *different programs* over
the same weights—with different batch sizes, sequence lengths (`T=1` decode vs.
prefill), padding buckets, operator fusions, activation-checkpointing boundaries,
and collective reductions.

This implementation removes every source of numerical drift across the
entire Transformer forward and log-probability scoring pipeline so that:

```
A (sampler logprob) == B (engine re-scoring) == C (learner forward)
```

holds **bitwise** (`max |diff| == 0.0`) for every token, independent of batch
composition, row position, padding bucket, and whether the program is KV-cached
decode, cacheless prefill, training forward, or training `forward + backward`
(`jax.value_and_grad`). The backward pass only needs to be a valid gradient of
that exact forward primal.

---

## Summary of Root Causes (`R1`–`R5`)

*   **`R1` — Tensor-Parallel (TP) reduction order (`fixed_order_reduce`)**:
    `bfloat16` addition is non-associative. Stock XLA ring `all-reduce` (`psum`)
    sums TP partials in a rank- and row-position-dependent order across
    contract-parallel projections (`o_proj`, `down_proj`) and vocab-sharded
    embedding lookups (`embedder.encode`). Fix: enforce a single canonical
    association order `((p0 + p1) + p2) + p3` on every rank, for every row, in
    every program (`RING`, `GATHER`, and `SCATTER` modes are bitwise identical).
*   **`R2` — Attention block sizes and batch distribution (`rpa_canonical`, `canon_attention`)**:
    Stock Ragged Paged Attention (RPA v3) selects tuned block sizes by shape and
    clamps them by decode concurrency, while stock model code switches between
    dense `einsum` (`seq_len == 1`) and Flash/Splash attention (`seq_len > 1`),
    changing the online-softmax rescale points. Fix: pin
    `(bkv_p, bq_sz, bkv_compute_p, bkv_sz) = (128, 512, 128, 512)`, force the
    mixed batch distribution `[0, 0, B]`, and pad token rows to
    `MIN_TOKEN_BUCKET = 256` (or run the standalone fixed-tile `flash_fwd_pallas`
    kernel with 512-key blocks and 128-key sub-blocks).
*   **`R3` — Forward vs. Forward+Backward XLA excess precision (`fixed_order_reduce.hard_round`)**:
    Under the TPU default `--xla_allow_excess_precision=true`, XLA elides
    intermediate `f32 -> bf16` truncations differently when an activation is
    consumed by a backward graph or saved at a `jax.checkpoint` boundary. Fix:
    every Pallas kernel stores `bf16` explicitly at exit, and every XLA glue
    boundary (TP-sum add chains, RoPE `sin`/`cos` and rotations, residual adds)
    rounds explicitly via `jax.lax.reduce_precision(x, 8, 7)`
    (`hard_bf16_add`, `hard_round`, `hard_upcast`).
*   **`R4` — Differentiable inference attention (`rpa_diff_chunked`, `rpa_diff`, `rpa_canonical`)**:
    The inference engine's Pallas RPA v3 kernel has no native autodiff VJP rule,
    yet the learner must execute the exact same forward kernel. Fix: wrap the
    verbatim engine forward kernel in a `jax.custom_vjp` whose backward pass
    differentiates a pure-JAX replica per sequence window and routes the paged
    KV-cache cotangent through a collision-free written-slot mask.
*   **`R5` — Large-vocabulary log-softmax reduction tree (`canonical_logsoftmax`)**:
    Across large vocabularies (e.g., `V = 151,936`), XLA chooses different
    `max` and `sum(exp)` reduction trees depending on the row count `M` and TP
    sharding. Fix: a three-stage Pallas kernel over fixed `VOCAB_TILE = 1024`
    column tiles with strict left-to-right tile combination and 8-aligned row
    buckets up to `PRODUCTION_M = 256`.

---

## End-to-End Op-by-Op Architecture

Using Qwen3 (`tunix.models.qwen3.model.Qwen3` and its canonical counterpart
`bench/qwen3/qwen3_model_canon.py:Qwen3Canon`) as the reference Transformer
pipeline, the table below maps every operation in the model stem, `DecoderLayer`
(`Op 0`–`Op 11`), and model tail (`Op T1`–`Op T4`) to its `tunix/zero_tim_kernel`
implementation and unit tests.

| Stage | Op ID | Operation | Drift Source in Stock Stack | `tunix/zero_tim_kernel` Implementation | Unit Test Coverage | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stem** | **Op H1** | `embedder.encode` (Vocab-Parallel Lookup) | GSPMD `psum` order & `act_btd` TP sharding mismatch (`R1`, `R3`) | `fixed_order_reduce.py` (`fixed_order_embed_*`), `CanonEmbedder.encode` | `fixed_order_reduce_test.py`, `qwen3_model_canon_test.py` | Complete |
| **Layer** | **Op 0** | Remat / Checkpoint Boundaries (`NONE`/`BLOCK`/`DECODER`) | XLA excess-precision `f32 -> bf16` elision across checkpoint boundaries (`R3`) | `fixed_order_reduce.py` (`hard_round`, `hard_upcast`), `CanonDecoderLayer` | `fixed_order_reduce_test.py`, `qwen3_model_canon_test.py` | Complete |
| **Layer** | **Op 1** | `input_layernorm` (Pre-Attention RMSNorm) | Feature-axis `mean(x^2)` tree order & `div`/`rsqrt` fusion vary with `M` | `pallas_rmsnorm.py`, `pallas_norm_matmul.py`, `canonical_vjp.py` | `pallas_rmsnorm_test.py`, `pallas_norm_matmul_test.py`, `canonical_vjp_test.py` | Complete |
| **Layer** | **Op 2** | `q_proj`, `k_proj`, `v_proj` (Column-Parallel Matmul) | MXU tiling, K-split, and operand order vary with `M = B * T` | `pallas_matmul.py`, `padded_matmul.py`, `model_contracts.py`, `canonical_vjp.py` | `pallas_matmul_test.py`, `padded_matmul_test.py`, `model_contracts_test.py` | Complete (TP) |
| **Layer** | **Op 3** | `q_norm`, `k_norm` (Per-Head QK RMSNorm) | Shape-dependent fusion with QKV matmul and RoPE; weight sharding | `pallas_rmsnorm.py` (`BF = 128`), `_canon_rmsnorm_nd` | `pallas_rmsnorm_test.py`, `qwen3_model_canon_test.py` | Complete |
| **Layer** | **Op 4** | `apply_rope` (Rotary Position Embedding) | Asymmetric FMA selection, `sin`/`cos` excess precision, `T=1` vs `T=64` VPU layout | `apply_rope_canonical_qk` (`ROPE_PAD_ROWS=64`, FMA barriers, `hard_round`) | `qwen3_model_canon_test.py` | Complete |
| **Layer** | **Op 5** | KV Cache Update & Paged Layout | Unwritten cache slots (`index >= kv_len`) vs. cacheless trainer tensor | `rpa_diff_chunked.py`, `bench/qwen3/rpa_backend.py`, `bench/qwen3/canon_attention.py` | `rpa_diff_chunked_test.py`, `rpa_backend_test.py`, `canon_attention_test.py` | Complete |
| **Layer** | **Op 6** | Core Attention | Decode vs. prefill algorithm switch; dynamic RPA block sizes (`R2`, `R4`) | `rpa_canonical.py`, `rpa_diff_chunked.py`, `rpa_diff.py`, `bench/qwen3/canon_attention.py` | `rpa_canonical_test.py`, `rpa_diff_chunked_test.py`, `rpa_diff_test.py`, `canon_attention_test.py` | Complete |
| **Layer** | **Op 7** | `o_proj` (Row-Parallel Matmul + TP All-Reduce) | Local contraction varies with `M`; ring `all-reduce` is non-associative (`R1`) | `padded_matmul.py` + `fixed_order_reduce.py` (`RING` / `GATHER` / `SCATTER`) | `fixed_order_reduce_test.py`, `padded_matmul_test.py`, `qwen3_model_canon_test.py` | Complete |
| **Layer** | **Op 8** | Pre-MLP Residual Add (`x + attn_out`) | Upstream `f32` accumulator fused into residual add in `Fwd` vs `Fwd+Bwd` (`R3`) | `fixed_order_reduce.py` (`hard_upcast`, `hard_round`), `_residual_add` | `fixed_order_reduce_test.py`, `qwen3_model_canon_test.py` | Complete |
| **Layer** | **Op 9** | `post_attention_layernorm` (Post-Attention RMSNorm) | Same as **Op 1** (feature-axis reduction & fusion drift) | `pallas_rmsnorm.py`, `pallas_norm_matmul.py`, `canonical_vjp.py` | `pallas_rmsnorm_test.py`, `pallas_norm_matmul_test.py`, `canonical_vjp_test.py` | Complete |
| **Layer** | **Op 10A** | Dense `MLP`: `gate_proj`/`up_proj` $\to$ `SwiGLU` $\to$ `down_proj` | Matmul `M`-dependent tiling, `silu(gate) * up` epilogue/prologue fusion, TP sum (`R1`) | `padded_matmul.py`, `pallas_swiglu.py`, `padded_swiglu.py`, `fixed_order_reduce.py` | `pallas_swiglu_test.py`, `padded_swiglu_test.py`, `padded_matmul_test.py`, `fixed_order_reduce_test.py` | Complete (TP) |
| **Layer** | **Op 10B** | `MoELayer` (Router `top_k` + `megablox.gmm`) | Discrete router `top_k` flips & batch-dependent `gmm` group tiling | Falls back to stock `MoELayer` （dense layers only) | N/A | Follow-up |
| **Layer** | **Op 11** | Post-MLP Residual Add (`residual + mlp_out`) | Same as **Op 8** (`f32` fusion across layer boundary, `R3`) | `fixed_order_reduce.py` (`hard_upcast`, `hard_round`), `_residual_add` | `fixed_order_reduce_test.py`, `qwen3_model_canon_test.py` | Complete |
| **Tail** | **Op T1** | `final_norm` (Final Layer RMSNorm) | Fusion drift between final residual add (**Op 11**) and LM head (**Op T2**) | `pallas_rmsnorm.py`, `CanonRMSNorm` | `pallas_rmsnorm_test.py`, `qwen3_model_canon_test.py` | Complete |
| **Tail** | **Op T2** | `compute_final_logits` (Tied `embedder.decode` / Untied `lm_head`) | `M` spans `1..256` (decode) vs `1024..4096` (trainer); non-256 `V/TP` tail | `fixed_lm_head.py` (`FIXED_M = 256` pad/chunk program, explicit `bf16` output) | `fixed_lm_head_test.py`, `qwen3_model_canon_test.py` | Complete (TP) |
| **Tail** | **Op T3** | `log_softmax` & `token_logprobs` (`R5`) | `V = 151,936` reduction tree varies with `M` & TP sharding | `canonical_logsoftmax.py` (3-stage `VOCAB_TILE = 1024` Pallas kernel) | `canonical_logsoftmax_test.py`, `qwen3_model_canon_test.py` | Complete |
| **Tail** | **Op T4** | Token Sampling & Temperature Contract | Inconsistent `bf16`/`f32` cast or temperature division before `log_softmax` | `_forward_sample` & `next_token_logprobs` (`bench/qwen3/qwen3_model_canon.py`) | `qwen3_model_canon_test.py` | Complete |

---

## Detailed Op-by-Op Walkthrough

### 1. Model Stem

#### Op H1 — Vocab-Parallel Embedding Lookup (`embedder.encode`)
*   **Why Stock Drifts**: When the embedding table `emb_vd` is sharded along the
    vocabulary axis across TP ranks (`P('tp', fsdp)`), GSPMD lowers
    `input_embedding[(x,)]` to a local lookup with out-of-range tokens masked to
    zero followed by a cross-TP `all-reduce` (`psum`) or `reduce-scatter`. Under
    mixed precision (`param_dtype=float32`) or varying `act_btd` sharding specs
    (`None if is_sampling else 'tp'`), XLA reorders the `f32 -> bf16` cast and
    the cross-TP reduction between sampler and learner.
*   **Implementation (`fixed_order_reduce.py`, `CanonEmbedder.encode`)**:
    *   `fixed_order_reduce.fixed_order_embed_1d` and `fixed_order_embed_2d`
        cast local shard weights to `bf16` prior to lookup, mask out-of-range
        token IDs to exact `+0.0`, and reduce across TP in ascending rank order
        `((p0 + p1) + p2) + p3` with `hard_bf16_add`.
    *   The residual stream `act_btd` is kept replicated along the hidden
        dimension `D` (`P(None, None, None)`) in both sampling and training so
        downstream layer norms never split their feature reduction across TP.
*   **Tests**: `fixed_order_reduce_test.py`, `qwen3_model_canon_test.py`.

---

### 2. Transformer `DecoderLayer` (`Op 0`–`Op 11`)

#### Op 0 — Activation Rematerialization & Rounding Boundaries (`RematConfig`)
*   **Why Stock Drifts**: Sampler forward passes run without backward consumers,
    whereas learner passes wrap `Attention`, `MLP`, or `DecoderLayer` in
    `jax.checkpoint` (`RematConfig.NONE`, `BLOCK`, `DECODER`). With
    `--xla_allow_excess_precision=true`, XLA keeps `bf16` operations in `f32`
    registers inside fusions unless forced to materialize a `bf16` tensor at a
    checkpoint boundary (`R3`).
*   **Implementation (`fixed_order_reduce.py`, `CanonDecoderLayer`)**:
    *   Every Pallas kernel writes `bf16` outputs explicitly at kernel exit.
    *   `fixed_order_reduce.hard_round` and `hard_upcast` invoke
        `jax.lax.reduce_precision(x, 8, 7)` at every XLA glue boundary so
        `RematConfig.NONE`, `BLOCK`, `DECODER`, and `CHECKPOINT_ATTN_CORE`
        produce the exact same forward bits under both default and strict XLA
        precision flags.
*   **Tests**: `fixed_order_reduce_test.py`,
    `qwen3_model_canon_test.py` (`test_fwd_vs_fwdbwd_remat_modes`,
    `test_reduce_precision_glue_tpu_ab`).

#### Op 1 & Op 9 — Pre-Attention & Post-Attention RMSNorm (`input_layernorm`, `post_attention_layernorm`)
*   **Why Stock Drifts**: `jnp.mean(x_f32**2, axis=-1)` changes its VPU tree
    reduction order and `div` vs. `rsqrt` fusion when the flattened token row
    count `M = B * T` changes between decode (`T=1`) and prefill (`T=L`).
*   **Implementation (`pallas_rmsnorm.py`, `pallas_norm_matmul.py`, `canonical_vjp.py`)**:
    *   `pallas_rmsnorm.rmsnorm`: pads `M` to a multiple of `BM = 8`, accumulates
        sum-of-squares in `f32` over fixed `BF = 128` feature blocks strictly
        left to right (`i = 0, 1, ..., D/128 - 1`), multiplies by
        `lax.rsqrt(acc / D + eps)` and `bf16`-aligned `gamma` in `f32`, and
        stores `bf16` once.
    *   `pallas_norm_matmul.norm_matmul`: fuses the exact `pallas_rmsnorm`
        prologue into VMEM scratch inside the downstream projection matmul,
        eliminating one `[M, D]` HBM round-trip per norm site while remaining
        100% bitwise identical to `pallas_rmsnorm -> pallas_matmul`.
    *   `canonical_vjp.rmsnorm`: wraps the Pallas forward primal with a pure-JAX
        backward pass so `Fwd` and `Fwd+Bwd` share the exact same forward kernel.
*   **Tests**: `pallas_rmsnorm_test.py`, `pallas_norm_matmul_test.py`,
    `canonical_vjp_test.py`.

#### Op 2 — Q / K / V Linear Projections (`q_proj`, `k_proj`, `v_proj`)
*   **Why Stock Drifts**: Stock `jnp.einsum` / `jnp.dot` selects different MXU
    tile sizes, split-`K` factors, and operand orderings (`a @ b` vs.
    `(b.T @ a.T).T`, which are not bitwise equal on TPU MXU) depending on
    `M = B * T`.
*   **Implementation (`pallas_matmul.py`, `padded_matmul.py`, `model_contracts.py`, `canonical_vjp.py`)**:
    *   `pallas_matmul.matmul`: fixes the contraction order to `BK = 256` blocks
        accumulated left to right in `f32` with a single `bf16` store at the
        final `K` step (`BM = 128, BN = 128` are bitwise neutral).
    *   `model_contracts.py` & `padded_matmul.matmul`: enforces per-model,
        per-TP tile and zero-padding contracts (failing closed on unadmitted
        widths) and zero-pads `M` to multiples of 128 so `T=1` decode and `T=L`
        prefill run the exact same MXU instruction stream.
    *   `canonical_vjp.matmul`: binds the Pallas forward kernel as the primal
        with either a single-dot or `K`-blocked custom VJP backward pass.
*   **Tests**: `pallas_matmul_test.py`, `padded_matmul_test.py`,
    `model_contracts_test.py`, `canonical_vjp_test.py`.

#### Op 3 — Per-Head Q / K RMSNorm (`q_norm`, `k_norm`)
*   **Why Stock Drifts**: Normalizing each head of dimension `H = 128` in stock
    JAX fuses differently with upstream QKV projections and downstream RoPE
    between `T=1` decode and `T=L` prefill.
*   **Implementation (`pallas_rmsnorm.py`, `_canon_rmsnorm_nd`)**:
    *   Flattens `[B, T, N_local, H]` to `[B * T * N_local, H]` inside each TP
        shard via `jax.shard_map` and runs `pallas_rmsnorm.rmsnorm` (`BF = 128`
        covers `H = 128` in one feature block).
*   **Tests**: `pallas_rmsnorm_test.py`, `qwen3_model_canon_test.py`.

#### Op 4 — Rotary Position Embedding (`apply_rope`)
*   **Why Stock Drifts**: Even without a reduction axis, stock RoPE (`first * cos - second * sin`, `second * cos + first * sin`) drifts on TPU for three reasons:
    1.  XLA asymmetrically fuses one product into an FMA (`fmsub`/`fmadd`, single
        rounding) while keeping the other as `mul` (double rounding), and flips
        which term gets the FMA between `Fwd` and `Fwd+Bwd`;
    2.  `sin`/`cos` `f32 -> bf16` rounding is elided under excess precision;
    3.  TPU VPU sublane vectorization of `jnp.sin`/`jnp.cos` changes between
        `T=1` decode and `T=64` prefill unless the sequence dimension is padded
        to a fixed multiple of 64 rows.
*   **Implementation (`apply_rope_canonical_qk` in `bench/qwen3/qwen3_model_canon.py`)**:
    *   Pads the sequence dimension `T` up to a multiple of `ROPE_PAD_ROWS = 64`;
    *   Applies `fixed_order_reduce.hard_round` / `hard_upcast` to `sin`, `cos`,
        inputs, and outputs (`HARD_ROUNDING_ROPE`);
    *   Places `jax.lax.optimization_barrier` on each of the four products
        (`first_half * cos`, `second_half * sin`, `second_half * cos`,
        `first_half * sin`) before the addition/subtraction to prevent asymmetric
        FMA emission.
*   **Tests**: `qwen3_model_canon_test.py`
    (`test_rope_canon_eliminates_decode_prefill_and_fwdbwd_drift`,
    `test_apply_rope_canonical_qk_matches_separate_calls`).

#### Op 5 — KV Cache Update & Paged Layout (`rpa_backend`, `rpa_diff_chunked`, `canon_attention`)
*   **Why Stock Drifts**: The sampler writes into a pre-allocated dense or paged
    KV cache with unused trailing capacity (`index >= kv_len`), whereas the
    learner runs cacheless (`cache=None`) over `[B, T, KH, D]`. Any contribution
    from masked KV slots or unaligned KV blocks perturbs the online-softmax
    running state `(m, l, acc)`.
*   **Implementation (`rpa_diff_chunked.py`, `bench/qwen3/rpa_backend.py`, `bench/qwen3/canon_attention.py`)**:
    *   `rpa_backend` and `rpa_diff_chunked` map cacheless prefill, KV-cached
        sampler prefill, and single-token decode onto the same paged KV layout
        `[pages, page_size, KH, 2, D]` with explicit `kv_lens`.
    *   `canon_attention.flash_fwd_pallas` skips fully masked 128-key sub-blocks
        via a scalar-prefetched `block_map` and preserves `(m_prev, l_prev, acc_prev)`
        identically across masked trailing positions.
*   **Tests**: `rpa_diff_chunked_test.py`, `rpa_backend_test.py`,
    `canon_attention_test.py`.

#### Op 6 — Core Attention (`rpa_canonical`, `rpa_diff_chunked`, `rpa_diff`, `canon_attention`)
*   **Why Stock Drifts**:
    1.  Stock Qwen3 guards Flash/Splash attention with
        `if self.config.use_flash_attention and seq_len > 1:`, running a dense
        `einsum` + full-sequence `softmax` for `seq_len == 1` decode and
        `SplashAttention` for `seq_len > 1` prefill/training;
    2.  Stock RPA v3 picks different `bkv_sz` compute blocks for decode vs.
        prefill (`R2`) and lacks an autodiff VJP rule (`R4`).
*   **Implementation**:
    *   **Engine RPA v3 Path (`rpa_canonical.py`, `rpa_diff_chunked.py`, `rpa_diff.py`, `bench/qwen3/rpa_backend.py`)**:
        *   `rpa_canonical.pinned_block_size_kwargs`: pins
            `(bkv_p, bq_sz, bkv_compute_p, bkv_sz) = (128, 512, 128, 512)` across
            decode, prefill, and mixed batches.
        *   `rpa_canonical.force_mixed_distribution`: rewrites `distribution` to
            `[0, 0, total_seqs]` so 1-token decode executes the exact same mixed
            kernel path as multi-token prefill, padded to
            `CANONICAL_MIN_TOKEN_BUCKET = 256`.
        *   `rpa_diff_chunked` & `rpa_diff`: provide `jax.custom_vjp` wrappers
            whose forward pass invokes the engine's RPA v3 kernel verbatim and
            whose backward pass differentiates each sequence on its static
            `max_q_len` window with collision-free page-table cotangent routing.
    *   **Standalone Flash Pallas Path (`bench/qwen3/canon_attention.py`)**:
        *   `flash_fwd_pallas` & `flash_bwd_pallas`: fixed 128-row $\times$
            128-key `exp2` online-softmax tile program inside fixed 512-key
            blocks (`KV_BLOCK_SIZE = 512`, `KV_SUB_BLOCK = 128`), guaranteeing
            identical bits for any query length, batch size, or cache capacity.
*   **Tests**: `rpa_canonical_test.py`, `rpa_diff_chunked_test.py`,
    `rpa_diff_test.py`, `rpa_backend_test.py`, `rpa_backend_tpu_test.py`,
    `canon_attention_test.py`.

#### Op 7 — Attention Output Projection (`o_proj` — Row-Parallel + Cross-TP Reduction)
*   **Why Stock Drifts**: Each TP rank contracts its local head slice
    `N_local * H` and reduces across the `'tp'` axis via `lax.psum`. Stock XLA
    ring `all-reduce` rotates the accumulation start rank by row chunk (`R1`),
    and excess precision can keep intermediate partial sums in `f32` (`R3`).
*   **Implementation (`padded_matmul.py`, `fixed_order_reduce.py`)**:
    *   Computes each rank's local `bf16` partial via `padded_matmul.matmul`;
    *   Reduces across TP ranks in strict ascending rank order
        `((p0 + p1) + p2) + p3` via `fixed_order_reduce.fixed_order_tp_sum`
        (`RING`, `GATHER`, and `SCATTER` modes all round each step via
        `hard_bf16_add` and produce identical bits).
*   **Tests**: `fixed_order_reduce_test.py`, `padded_matmul_test.py`,
    `qwen3_model_canon_test.py`.

#### Op 8 & Op 11 — Residual Connections (`x + attn_out`, `residual + mlp_out`)
*   **Why Stock Drifts**: Under `--xla_allow_excess_precision=true`, XLA can fuse
    an upstream `f32` accumulator directly into `x + attn_out` or
    `residual + mlp_out` in a forward-only program while truncating to `bf16`
    first in `Fwd+Bwd` (`R3`).
*   **Implementation (`fixed_order_reduce.py`, `_residual_add`)**:
    *   Applies `hard_upcast` to both operands and `hard_round`
        (`lax.reduce_precision(x, 8, 7)`) to the sum at both residual sites.
*   **Tests**: `fixed_order_reduce_test.py`, `qwen3_model_canon_test.py`.

#### Op 10A — Dense MLP (`gate_proj` / `up_proj` $\to$ `SwiGLU` $\to$ `down_proj`)
*   **Op 10A.1 (`gate_proj`, `up_proj`)**:
    *   Uses `padded_matmul.matmul` (or fused `pallas_norm_matmul.norm_matmul`
        with **Op 9**) and `model_contracts.py` to pad `M` to 128 rows and pad
        non-256-multiple per-rank intermediate widths `F / TP` (e.g., Qwen3-4B
        `F = 9728` at `TP = 8` where `F / TP = 1216 -> 1280`) with zeros.
*   **Op 10A.2 (`SwiGLU`: `silu(gate) * up`)**:
    *   **Why Stock Drifts**: XLA fuses `silu(gate) * up` into either the
        `gate`/`up` matmul epilogue or the `down_proj` prologue differently
        between `Fwd` and `Fwd+Bwd`.
    *   **Implementation (`pallas_swiglu.py`, `padded_swiglu.py`, `canonical_vjp.py`)**:
        Standalone Pallas kernel with `allow_input_fusion=(False, False)` and
        fixed `(BM = 8, BN = 128)` tiles that computes
        `(gate_f32 * jax.nn.sigmoid(gate_f32)) * up_f32` in `f32` and stores
        `bf16` once, wrapped with contract padding (`padded_swiglu.swiglu`) and
        a custom VJP (`canonical_vjp.swiglu`).
*   **Op 10A.3 (`down_proj`)**:
    *   Uses `padded_matmul.matmul` (zero-padding the contracted `F / TP` axis
        to a multiple of `BK = 256` when needed) followed by
        `fixed_order_reduce` (`RING` / `GATHER` / `SCATTER`) across TP.
*   **Tests**: `pallas_swiglu_test.py`, `padded_swiglu_test.py`,
    `padded_matmul_test.py`, `fixed_order_reduce_test.py`.

#### Op 10B — Mixture-of-Experts (`MoELayer`)
*   **Scope**: This package targets dense Transformer layers; when
    `config.num_experts is not None`, `CanonDecoderLayer` currently falls back to
    stock `qwen3_stock.MoELayer`. Canonical MoE support (router `padded_matmul` +
    routed-expert ID replay + fixed-tile grouped matmul + `fixed_order_reduce`
    expert combine) is tracked for `v2`.

---

### 3. Model Tail & RL Scoring (`Op T1`–`Op T4`)

#### Op T1 — Final RMSNorm (`final_norm`)
*   **Why Stock Drifts**: Same feature-axis reduction and fusion drift as **Op 1**
    and **Op 9**, situated between the final layer's residual add (**Op 11**)
    and the LM head (**Op T2**).
*   **Implementation (`pallas_rmsnorm.py`, `CanonRMSNorm`)**:
    *   Executes `pallas_rmsnorm.rmsnorm` (`BF = 128`) with a local shard-map VJP
        wrapper that avoids redundant cross-TP collectives on replicated
        activations.
*   **Tests**: `pallas_rmsnorm_test.py`, `qwen3_model_canon_test.py`.

#### Op T2 — Final Logits Projection (`compute_final_logits`: Tied `embedder.decode` & Untied `lm_head`)
*   **Why Stock Drifts**:
    1.  The token row count `M = B * T` spans two orders of magnitude between
        decode (`M = 1..256`), prefill scoring (`M = 256`), and learner training
        (`M = 1024..4096`), causing XLA to pick different MXU tilings and
        split-`K` trees;
    2.  Per-rank vocabulary shards `V / TP` (e.g., `151,936 / 4 = 37,984`,
        `37,984 % 256 = 96`) are not multiples of 256;
    3.  Upcasting `logits` to `f32` inside `compute_final_logits` allows XLA to
        elide the `bf16` rounding of the matmul accumulator in one program while
        keeping it in another.
*   **Implementation (`fixed_lm_head.py`)**:
    *   Compiles a **single canonical Pallas program** of shape
        `[FIXED_M = 256, K] @ [K, round_up(V / TP, 256)]` with tiles
        `(BM = 128, BN = 256, BK = 256)`:
        *   **Small `M <= 256` (decode / scoring buckets)**: zero-pads `M` to
            `FIXED_M = 256`, runs the canonical kernel, and slices the first `M`
            rows.
        *   **Large `M > 256` (learner batches, e.g., `1024, 2048, 4096`)**:
            reshapes `x` into `(-1, 256, K)` and maps the exact same
            `FIXED_M = 256` kernel over chunks via `jax.lax.map`; the custom VJP
            accumulates `dW` across chunks in ascending order via `jax.lax.scan`.
    *   Returns explicit `bf16` logits so no `f32` logit copy is materialized in
        HBM and `bf16` rounding is never elided.
*   **Tests**: `fixed_lm_head_test.py`, `qwen3_model_canon_test.py`.

#### Op T3 — Canonical Log-Softmax & Gathered Token Log-Probs (`canonical_logsoftmax`)
*   **Why Stock Drifts**: Over `V = 151,936` columns, `jax.nn.log_softmax`
    compiles different `max` and `sum(exp)` reduction trees depending on `M` and
    TP sharding (`R5`). Furthermore, slicing `logits[:, :-1, :]` before scoring
    produces unaligned row counts `B * (T - 1)`.
*   **Implementation (`canonical_logsoftmax.py`)**:
    *   **Three-stage Pallas kernel**:
        1.  `partial_kernel`: partitions the vocabulary into fixed
            `VOCAB_TILE = 1024` tiles (grouped by `TILES_PER_GROUP = 8`, padding
            tail columns beyond `V` with `jnp.finfo(jnp.float32).min`) and
            computes per-tile `(tile_max, tile_sum)` in `f32`;
        2.  `combine_kernel`: combines all tile summaries strictly left to right
            into each row's scalar normalizer `global_max + log(global_sum)`;
        3.  `normalize_kernel` / `gather_kernel`: `log_softmax` writes the full
            `[M, V]` output, whereas `token_logprobs` / `gathered_logprobs`
            subtracts the row normalizer directly from the target token's logit
            without materializing the `[M, V]` `f32` log-probability tensor
            (100% bitwise equal to `log_softmax` + `take_along_axis`).
    *   **Row-Bucket Invariance**: All three stages operate independently across
        8-row blocks (`BM = 8`), so any 8-aligned bucket `M in {8, 16, ..., 256}`
        produces the exact same bits as `PRODUCTION_M = 256`.
    *   **Vocab-Parallel Admission Guard**: In `Qwen3Canon`, vocab-parallel Stage 1
        runs locally on each TP shard only when `(V / TP) % VOCAB_TILE == 0`
        (where shard boundaries align with 1024-column tiles, guaranteeing
        bitwise equality with the unsharded library call); otherwise it gathers
        the `bf16` shards before Stage 1 so the global 1024-column tile
        boundaries never shift.
*   **Tests**: `canonical_logsoftmax_test.py`, `qwen3_model_canon_test.py`.

#### Op T4 — Token Sampling & Temperature Contract (`_forward_sample`, `next_token_logprobs`)
*   **Why Stock Drifts**: Even when `compute_final_logits` is bitwise identical
    between sampler and learner, applying `logits / temperature` in different
    dtypes (`bf16` vs. `f32`) or slicing `logits[:, :-1, :]` before scoring
    causes `old_logprobs` and `new_logprobs` to diverge.
*   **Implementation (`bench/qwen3/qwen3_model_canon.py`)**:
    *   `_forward_sample` and `next_token_logprobs` score all `B * T` aligned
        rows through the exact same `compute_token_logprobs` contract over the
        `bf16` logits returned by `compute_final_logits`.
*   **Tests**: `qwen3_model_canon_test.py`
    (`test_ablation_end_to_end_rollout_vs_trainer`,
    `test_qwen3_canon_tp4_invariance`).

---

## Package Layout

*   `pallas_matmul.py`: Fixed-contraction-order (`BK = 256`) `bf16` Pallas
    matmul with `f32` accumulation and a single `bf16` output store.
*   `pallas_rmsnorm.py`: Fixed-feature-block (`BF = 128`) left-to-right `f32`
    sum-of-squares Pallas RMSNorm and its bit-exact pure-JAX counterpart
    `canonical_rmsnorm`.
*   `pallas_swiglu.py`: Single-kernel Pallas `silu(gate) * up` with input fusion
    disabled and a single `bf16` output store.
*   `pallas_norm_matmul.py`: Fused `pallas_rmsnorm` VMEM prologue +
    `pallas_matmul` in one custom call, bitwise equal to the two-kernel chain.
*   `model_contracts.py`: Per-model and per-TP tile and zero-padding contracts
    (`qwen0p6b`, `qwen1p7b`, `qwen4b`, `qwen8b`, `qwen32b`; `TP = 4` and
    `TP = 8`). Fails closed on unadmitted widths.
*   `padded_matmul.py`: Row zero-padding (`M` to 128) and contract-governed `K`/`N`
    zero-padding wrapper around `pallas_matmul`.
*   `padded_swiglu.py`: Contract-governed padding wrapper around `pallas_swiglu`.
*   `canonical_vjp.py`: `jax.custom_vjp` wrappers binding `pallas_matmul`,
    `pallas_rmsnorm`, and `pallas_swiglu` as forward primals with selectable
    `plain` or `K`-blocked backward passes.
*   `fixed_order_reduce.py`: Fixed rank-order (`((p0 + p1) + p2) + p3`) TP
    collectives (`RING`, `GATHER`, `SCATTER`), explicit `bf16` rounding helpers
    (`hard_bf16_add`, `hard_round`, `hard_upcast`, `plain_add`), fixed-order
    vocab-sharded embedding lookups (`fixed_order_embed_1d`,
    `fixed_order_embed_2d`), and rank-parallel backward helpers.
*   `fixed_lm_head.py`: Fixed `[FIXED_M = 256, K] @ [K, round_up(V/TP, 256)]`
    LM-head Pallas program with request zero-padding (`M <= 256`), learner
    chunk mapping (`M > 256`), and ordered `lax.scan` weight-gradient
    accumulation.
*   `canonical_logsoftmax.py`: Fixed three-stage Pallas `log_softmax` and
    `token_logprobs` / `gathered_logprobs` (`VOCAB_TILE = 1024`) with 8-aligned
    row buckets up to `PRODUCTION_M = 256`.
*   `rpa_canonical.py`: Pinned RPA v3 block sizes `(128, 512, 128, 512)`, forced
    mixed batch distribution, 256-row token bucket, and differentiable wrapper
    selection.
*   `rpa_diff_chunked.py`: Chunked-cache differentiable RPA v3 (`jax.custom_vjp`)
    and its multi-sequence ragged adapter (`make_diff_rpa_chunked_ragged`).
*   `rpa_diff.py`: Prefill-only differentiable RPA v3 wrapper with zero KV-cache
    cotangent.
*   `test_utils.py`: Shared CPU test helpers (`exact_bf16`, `cosine_similarity`).
*   `bench/qwen3/`:
    *   `qwen3_model_canon.py`: `Qwen3Canon`, a drop-in counterpart of
        `tunix.models.qwen3.model.Qwen3` sharing its parameter tree with 11
        per-kernel `CanonKernelConfig` switches for Leave-One-Out and Add-One-In
        ablations.
    *   `canon_attention.py`: Standalone fixed-tile Pallas Flash Attention
        forward (`flash_fwd_pallas`) and backward (`flash_bwd_pallas`) kernels.
    *   `rpa_backend.py`: Dense $\leftrightarrow$ paged adapter running
        `Qwen3Canon`'s cacheless prefill, KV-cached prefill, and decode attention
        programs on the `tpu_inference` RPA v3 kernel.
    *   `qwen3_canon_benchmark.py`: Numeric invariance suite, drift-regime
        diagnostics, and latency / HBM scaling benchmarks.

---

## Canonical Environment Switches

*   **`R1` (TP reduction & embedding)**: `CANON_FIXED_AR=1`,
    `CANON_FIXED_AR_EMBED=1`, optional `CANON_FIXED_AR_GATHER=1` or
    `CANON_FIXED_AR_SCATTER=1`.
*   **`R2` (RPA block sizes & bucket)**: `CANON_RPA_D=128,512,128,512`,
    `CANON_RPA_P=128,512,128,512`, `CANON_RPA_M=128,512,128,512`,
    `MIN_TOKEN_BUCKET=256`.
*   **`R3` (Rounding boundaries)**: Pinned in code by default (`hard_bf16_add`
    in `fixed_order_reduce` and `HARD_ROUNDING=ALL` in `Qwen3Canon`);
    `XLA_FLAGS=--xla_allow_excess_precision=false` is optional defense in depth.
*   **`R4` (Differentiable RPA)**: `CANON_RPA_VJP2=1`, `CANON_VJP2_MAX_SEQS=1`
    (or batch size `B` when unrolling multi-sequence ragged training batches).
*   **`R5` (Log-softmax)**: `CANON_LOGPROB_M=256`, `CANON_LOGPROB_M_BUCKET=1`,
    `CANON_PROMPT_PROCESSED_LOGPROBS=1`, `CANON_PALLAS_LOGSOFTMAX=1`.
*   **Projections, Norms & LM Head**: `CANON_PALLAS_ALL_PROJ=1`,
    `CANON_PALLAS_ALL_RMSNORM=1`, `CANON_PALLAS_SWIGLU=1`,
    `CANON_PALLAS_MPAD=1`, `CANON_PALLAS_SWIGLU_MPAD=1`,
    `CANON_PALLAS_CANONICAL_VJP=1`, `CANON_MATMUL_VJP_PLAIN=1`,
    `CANON_FIXED_LM_HEAD=1`.
*   **Rank-Parallel Backward & Device Order**: `CANON_RANK_PARALLEL_BACKWARD=1`,
    `CANON_RANK_PARALLEL_BACKWARD_CHECK_VMA=1`,
    `CANON_CHECK_VMA_RANK_PARALLEL_ONLY=1`, `CANON_EXPECT_MODEL_MESH_IDS`.

---

## Usage

```python
from tunix.zero_tim_kernel import canonical_logsoftmax
from tunix.zero_tim_kernel import canonical_vjp
from tunix.zero_tim_kernel import fixed_lm_head
from tunix.zero_tim_kernel import fixed_order_reduce
from tunix.zero_tim_kernel import model_contracts
from tunix.zero_tim_kernel import padded_matmul

contract = model_contracts.get_contract("qwen8b")  # Qwen3-8B at TP=4.

# Column-parallel projection (Op 2 / Op 10A.1): fixed tiles and padding, differentiable.
forward = lambda a, b: padded_matmul.matmul(a, b, contract=contract)
y = canonical_vjp.matmul(x, w, forward=forward, contract=contract)

# Contract-parallel projection (Op 7 o_proj / Op 10A.3 down_proj) completed in a fixed order.
out = fixed_order_reduce.contract_parallel_projection(
    h, w_down, equation="mn,np->mp", mesh=mesh, tp_axis="model",
    local_matmul=forward)

# LM head (Op T2) and gathered token log-probs (Op T3) through one fixed program.
logits = fixed_lm_head.fixed_lm_head(
    hidden, lm_head, mesh=mesh, tp_axis="model",
    local_matmul=fixed_lm_head.canonical_local_matmul(contract),
    endpoint="untied_lm_head")
logprobs = canonical_logsoftmax.gathered_logprobs(logits, token_ids)
```

For attention (**Op 6**), build the kernel call with
`rpa_canonical.pinned_block_size_kwargs()` and wrap it with
`rpa_canonical.differentiable_rpa(kernel_fn, ...)` for training.

---

## Caveats

*   The bitwise guarantees are properties of the Mosaic TPU compilation of the
    kernels (with `shape_invariant_numerics=True`). CPU tests execute Pallas in
    `interpret=True` mode, which verifies shape/batch invariance, exactness on
    representable inputs, and collective identities (`RING == GATHER == SCATTER`),
    not physical TPU instruction bits.
*   The modules pin the `bf16` rounding of their own reductions
    (`fixed_order_reduce.hard_bf16_add` by default). Any glue a caller writes
    *between* kernels must also round explicitly at `bf16` boundaries via
    `fixed_order_reduce.hard_round` / `hard_upcast` (`lax.reduce_precision(x, 8, 7)`)
    or run with `--xla_allow_excess_precision=false`.
*   `CANON_VJP2_MAX_SEQS` bounds how many sequences the ragged RPA backward
    unrolls (default `1`): sequences beyond `CANON_VJP2_MAX_SEQS` receive zero
    gradient unless `CANON_VJP2_MAX_SEQS` is set to at least the per-step batch
    size `B`.
*   Requires JAX >= 0.8 (`jax.shard_map`, `jax.typeof`, `jax.enable_x64`).

---

## Running Tests

```shell
pytest tests/zero_tim_kernel/
```
