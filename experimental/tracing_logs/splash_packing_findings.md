# Why sequence packing makes attention backward slower on TPU

Measured on a single-host v5p (4 chips), Qwen3-1.7B, `fsdp=4, tp=1`.
Raw numbers: `splash_microbench_RUN1_results.log`. Reproduce with
`bash experimental/bench_splash_v5p_docker.sh` (no model files, no vLLM,
random weights, seconds per case).

## TL;DR

Splash attention charges a packed row for its **entire causal area**, no matter
how many sequences are inside it. `segment_ids` does not make the kernel skip
cross-sequence blocks; it computes them and then zeroes them. So packing K
sequences into one row of length `B` costs `B²/2` — the same as one sequence of
length `B` — instead of the `K · (B/K)²/2` the sequences actually need.

That makes attention cost `rows × row_len²`, i.e. **linear in the packing
budget at a fixed token count**. Packing still halves the MLP/lm_head work (no
padding), which is why the end-to-end step time barely moved while attention
backward went 6ms → 16ms.

## The mechanism

Splash builds its block schedule from a **static** mask at trace time:

- `_process_mask(mask, block_shape, ...)` (`splash_attention_mask_info.py:518`)
  takes no segment information at all.
- The pallas grid width comes from that schedule:
  `grid_width = mask_info.data_next.shape[-1]`
  (`splash_attention_kernel.py:1141` forward, `:1476` backward).
- `segment_ids` is an ordinary kernel input, applied inside
  `_apply_mask_and_soft_cap` (`:596-677`) as `mask &= (q_ids == kv_ids)` — on
  blocks the kernel has already computed.

Segment boundaries are runtime data, so they cannot influence a compile-time
schedule. The only sparsity splash exploits is the static causal mask's upper
triangle.

## Evidence 1 — the segment sweep (the decisive one)

Same shape `[1, 8192]`, same tokens, only the segment structure varies:

| segments in the row | forward | backward |
|---|---|---|
| 1 | 4.48 ms | 11.84 ms |
| 2 | 4.48 ms | 11.84 ms |
| 4 | 4.48 ms | 11.84 ms |
| 8 | 4.48 ms | 11.84 ms |
| 16 | 4.48 ms | 11.84 ms |
| 32 | 4.48 ms | 11.84 ms |
| 32, deliberately uneven sizes | 4.48 ms | 11.84 ms |

Identical to the digit. If the kernel skipped cross-segment blocks, cost would
fall as the block-diagonal got sparser. It does not move at all.

Two corollaries: `segment_ids` cannot save compute, and **balancing segment
sizes across rows cannot help either** — the cost does not depend on how the
sequences are distributed.

## Evidence 2 — same sequences, three layouts

32 sequences of 1024 real tokens, laid out three ways. Every arm attends the
same tokens, so the *effective* attention work is identical (8 × 1024²/2 per
chip); they differ only in the padding and cross-segment area the kernel runs
through on top.

| layout (per chip) | effective | executed = rows·len²/2 | **measured bwd** |
|---|---|---|---|
| `[8, 2048]` unpacked | 1.00× | 1.00× | **1.00×** (2.34 ms) |
| `[2, 4096]` packed | 1.00× | 1.00× | **0.87×** (2.03 ms) |
| `[1, 8192]` packed | 1.00× | 2.00× | **3.15×** (7.36 ms) |
| `[1, 8192]`, segment_ids dropped | n/a | 2.00× | 3.21× (7.51 ms) |

Measured tracks **executed**, not effective. Dropping `segment_ids` entirely
changes almost nothing (3%), which prices the feature: it costs 3% and saves 0.

Independent corroboration on CPU, counting the blocks JAX's real `process_mask`
schedules (block size 256): unpacked 288 blocks, `[1, 8192]` causal 528 blocks
— the same 1.83× the timings show.

## Evidence 3 — why the end-to-end step time did not move

The full attention module (kernel + q/k/v/o projections), global shapes:

| layout | total bwd | = kernel + projections |
|---|---|---|
| `[32, 2048]` unpacked | 12.32 ms | 5.01 + 7.30 |
| `[8, 4096]` packed | **9.45 ms (0.77×)** | 4.40 + 5.05 |
| `[4, 8192]` packed | 23.26 ms (1.89×) | 11.84 + 11.42 |

The projections are linear in the token count, so packing (which removes the
padding) makes them cheaper. At a large budget that saving is cancelled by the
attention penalty — hence a flat end-to-end number hiding a 3× attention
regression underneath.

## What follows

**The packing budget is not "as large as fits".** At a fixed token count,
attention cost grows linearly with the budget while the MLP saving is already
fully realised at the smallest legal budget. The optimum is therefore the
smallest budget that still holds a maximal sequence
(`max_prompt_length + max_response_length`), which is also the unpacked row
length. Measured: that setting is 0.50× the unpacked attention (CPU block
count) and 0.77× end-to-end at the module level, while four times that budget
costs 1.83–2.36×.

**If a larger budget is wanted anyway** (fewer rows, fewer micro-steps), a
static band mask bounds the damage. `LocalMask(shape, window_size=(W, 0), 0)`
with `W ≥ L_max − 1` is a superset of the true block-diagonal, so `segment_ids`
still does the exact masking and the numerics are unchanged — but everything
beyond the band is dropped from the schedule at trace time. Block counts for
`[1, 8192]`: causal 528 → band(2048) 252, i.e. 1.83× → 0.88× of unpacked. It
never beats simply using the smallest budget (the optimum is `B = W`, where the
band degenerates to causal), but it turns the budget from a knob that must be
right into one that cannot be badly wrong. `tunix/models/gemma4/model.py:979`
already uses `LocalMask` for the sliding-window layers, so the mechanism is
established in this codebase.

**Going below `T · L_max / 2`** requires a mask that is block-diagonal at trace
time, which means the packer would have to emit a fixed, quantised segment
layout (otherwise the shape changes every step and everything recompiles).
That is a packer-level change, not a kernel one.

## Note for anyone comparing against GPU

This is a kernel-design difference, not a hardware limit. GPU flash-attention
`varlen` takes `cu_seqlens` as a runtime argument and schedules each segment
independently, so packing there really does cost `Σ Lᵢ²`. The TPU Pallas kernel
fixes its block schedule at compile time, so the same information cannot be
used the same way. Packing on TPU still buys the MLP/lm_head saving and the HBM
reduction (76 → 61 GB in our run); it just does not buy the attention saving.
