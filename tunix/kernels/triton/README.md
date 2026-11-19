# Branched vLLM Triton attention kernels

Branched copies of files from upstream
[`vllm/v1/attention/ops`](https://github.com/vllm-project/vllm/tree/1d2d83a07fd3180068917a031c28dcc83141d0be/vllm/v1/attention/ops)
pinned to commit `1d2d83a07fd3180068917a031c28dcc83141d0be`. These files
are unchanged from vLLM v0.26.0 through v0.29.0.

| File | Upstream |
| --- | --- |
| `triton_unified_attention.py` | `vllm/v1/attention/ops/triton_unified_attention.py` |
| `triton_attention_helpers.py` | `vllm/v1/attention/ops/triton_attention_helpers.py` |
| `triton_reshape_and_cache_flash.py` | `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py` |

The unified kernel is tunix's GPU paged attention path for prefill (with or
without a cached prefix) and decode. `triton_reshape_and_cache_flash` first
writes the new K/V into the paged cache, as vLLM's `triton_attn` backend does.
vLLM's dense `triton_prefill_attention` is deliberately not copied: vLLM only
uses it for cache-free encoder attention, and tunix uses splash attention for
dense cases.

## Local changes

Only import lines differ from upstream. Each is marked `# tunix: branched
import`:

*   `from vllm.triton_utils import tl, triton` becomes `import triton` /
    `import triton.language as tl` (all three files) so the kernels do not
    depend on `vllm` at import time.
*   `triton_unified_attention.py` and `triton_reshape_and_cache_flash.py` stub
    their top-level `torch` and `vllm.*` imports (used only by their unused
    PyTorch launchers) so importing the `@triton.jit` kernels does not require
    `torch` or `vllm`.
*   `triton_unified_attention.py` imports its helpers from this directory
    rather than from `vllm.v1.attention.ops`.

The kernel code itself is unchanged. Other adaptations for JAX (launchers,
slot mapping, KV cache layout) belong in
`tunix/models/paged_attention_vllm_gpu.py`, not here.

## Refreshing

1.  Re-copy the files from upstream
    [`vllm/v1/attention/ops`](https://github.com/vllm-project/vllm/tree/main/vllm/v1/attention/ops).
2.  Re-apply the import changes above.
3.  Check the result with `diff <upstream_vllm>/vllm/v1/attention/ops/<file>
    <file>`. Only the marked import lines should differ.
4.  Update the pinned commit in this README.

## Status

Tunix launches the `@triton.jit` kernels from JAX through `jax_triton` in
`tunix/models/paged_attention_vllm_gpu.py`.
