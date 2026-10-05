# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPU correctness tests for the vendored vLLM Triton unified attention kernel.

These exercise the verbatim copy in this directory through its own PyTorch
launcher, against a naive float32 PyTorch reference. They need a CUDA GPU with
`torch`, `triton` and `vllm` installed; elsewhere every test is skipped.

The copies import their helpers from the installed `vllm`, so install the vLLM
version the copies were taken from (see README.md) to avoid skew between the
copied kernels and the installed helpers.

Run from the directory that contains the `tunix` package:

  python -m pytest tunix/kernels/triton/vllm/vllm_triton_attention_test.py -v
"""

from __future__ import annotations

from collections.abc import Sequence
import math
import os

from absl.testing import absltest
from absl.testing import parameterized
import torch

_IMPORT_ERROR: Exception | None = None
try:
  from tunix.kernels.triton.vllm import triton_unified_attention  # pylint: disable=g-import-not-at-top
except Exception as e:  # pylint: disable=broad-exception-caught
  _IMPORT_ERROR = e

_DEVICE = "cuda"
# Mirrors vLLM's TritonAttentionBackend constants (backends/triton_attn.py).
_MIN_LAUNCH_GRID_SIZE_2D = 128
_NUM_PAR_SOFTMAX_SEGMENTS = 16


def _skip_reason() -> str | None:
  if not torch.cuda.is_available():
    if torch.version.cuda is None:
      return (
          f"torch {torch.__version__} is a CPU-only build (torch.version.cuda"
          " is None); install a CUDA build of torch"
      )
    return (
        f"torch {torch.__version__} (CUDA {torch.version.cuda}) sees no GPU:"
        f" device_count={torch.cuda.device_count()}, CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}; check the driver"
        " (nvidia-smi) and that the torch CUDA version is supported by it"
    )
  if _IMPORT_ERROR is not None:
    return f"vendored vLLM kernels failed to import: {_IMPORT_ERROR!r}"
  return None


def _cdiv(a: int, b: int) -> int:
  return (a + b - 1) // b


def _next_power_of_2(n: int) -> int:
  return 1 << (n - 1).bit_length()


def _tolerance(dtype: torch.dtype) -> dict[str, float]:
  return dict(atol=2e-2, rtol=2e-2) if dtype == torch.bfloat16 else dict(
      atol=1e-2, rtol=1e-2
  )


def _reference_attention(
    q: torch.Tensor,
    keys: Sequence[torch.Tensor],
    values: Sequence[torch.Tensor],
    q_lens: Sequence[int],
    *,
    scale: float,
    window_left: int | None = None,
    softcap: float = 0.0,
) -> torch.Tensor:
  """Causal GQA attention, one sequence at a time, in float32.

  Args:
    q: `[sum(q_lens), num_q_heads, head_dim]`. Each sequence's queries are its
      *last* `q_len` positions, i.e. they follow `kv_len - q_len` cached tokens.
    keys: per-sequence `[kv_len, num_kv_heads, head_dim]`, including the new
      tokens.
    values: like `keys`.
    q_lens: query count per sequence.
    scale: softmax scale.
    window_left: if set, a query at position p only sees keys at positions
      `>= p - window_left` (vLLM's `window_size[0]` convention).
    softcap: if > 0, logits become `softcap * tanh(logits / softcap)`.

  Returns:
    `[sum(q_lens), num_q_heads, head_dim]` in `q.dtype`.
  """
  outs = []
  start = 0
  for k, v, q_len in zip(keys, values, q_lens, strict=True):
    kv_len = k.shape[0]
    qi = q[start : start + q_len].float()
    start += q_len
    group = qi.shape[1] // k.shape[1]
    ki = k.float().repeat_interleave(group, dim=1)
    vi = v.float().repeat_interleave(group, dim=1)

    logits = torch.einsum("qhd,khd->hqk", qi, ki) * scale
    if softcap > 0:
      logits = softcap * torch.tanh(logits / softcap)

    q_pos = torch.arange(kv_len - q_len, kv_len, device=q.device)[:, None]
    k_pos = torch.arange(kv_len, device=q.device)[None, :]
    mask = k_pos <= q_pos
    if window_left is not None:
      mask &= (q_pos - k_pos) <= window_left
    logits = logits.masked_fill(~mask[None], float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    outs.append(torch.einsum("hqk,khd->qhd", probs, vi))
  return torch.cat(outs).to(q.dtype)


def _make_paged_cache(
    keys: Sequence[torch.Tensor],
    values: Sequence[torch.Tensor],
    *,
    block_size: int,
    extra_blocks: int = 7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Scatters per-sequence K/V into a shuffled paged cache.

  Uses vLLM's own layout: one `[num_blocks, num_kv_heads, block_size,
  2 * head_dim]` buffer, viewed as `(key_cache, value_cache)` of shape
  `[num_blocks, block_size, num_kv_heads, head_dim]` via transpose + split, so
  the caches are non-contiguous exactly as in `TritonAttentionImpl.forward`.
  Unused slots are filled with garbage so any over-read shows up in the output.

  Returns:
    `(key_cache, value_cache, block_table)`.
  """
  num_kv_heads, head_dim = keys[0].shape[1:]
  dtype = keys[0].dtype
  blocks_per_seq = [_cdiv(k.shape[0], block_size) for k in keys]
  num_blocks = sum(blocks_per_seq) + extra_blocks

  kv_cache = 100.0 * torch.randn(
      num_blocks, num_kv_heads, block_size, 2 * head_dim,
      dtype=dtype, device=_DEVICE,
  )
  key_cache, value_cache = kv_cache.transpose(1, 2).split(head_dim, dim=-1)

  perm = torch.randperm(num_blocks, device=_DEVICE).to(torch.int32)
  block_table = torch.zeros(
      len(keys), max(blocks_per_seq), dtype=torch.int32, device=_DEVICE
  )
  next_block = 0
  for i, (k, v) in enumerate(zip(keys, values, strict=True)):
    num_seq_blocks = blocks_per_seq[i]
    ids = perm[next_block : next_block + num_seq_blocks]
    next_block += num_seq_blocks
    block_table[i, :num_seq_blocks] = ids
    for j, block in enumerate(ids.tolist()):
      k_chunk = k[j * block_size : (j + 1) * block_size]
      v_chunk = v[j * block_size : (j + 1) * block_size]
      key_cache[block, : k_chunk.shape[0]] = k_chunk
      value_cache[block, : v_chunk.shape[0]] = v_chunk
  return key_cache, value_cache, block_table


def _random_inputs(
    q_lens: Sequence[int],
    kv_lens: Sequence[int],
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
  q = torch.randn(
      sum(q_lens), num_q_heads, head_dim, dtype=dtype, device=_DEVICE
  )
  keys = [
      torch.randn(n, num_kv_heads, head_dim, dtype=dtype, device=_DEVICE)
      for n in kv_lens
  ]
  values = [
      torch.randn(n, num_kv_heads, head_dim, dtype=dtype, device=_DEVICE)
      for n in kv_lens
  ]
  return q, keys, values


def _cu_seqlens(lens: Sequence[int]) -> torch.Tensor:
  cu = [0]
  for n in lens:
    cu.append(cu[-1] + n)
  return torch.tensor(cu, dtype=torch.int32, device=_DEVICE)


class TritonUnifiedAttentionTest(parameterized.TestCase):
  """`triton_unified_attention.unified_attention`: paged, mixed batches."""

  def setUp(self):
    super().setUp()
    if reason := _skip_reason():
      self.skipTest(reason)
    torch.manual_seed(0)

  def _run(
      self,
      ctx_and_q_lens: Sequence[tuple[int, int]],
      *,
      num_q_heads: int = 8,
      num_kv_heads: int = 2,
      head_dim: int = 128,
      block_size: int = 16,
      dtype: torch.dtype = torch.bfloat16,
      window_left: int | None = None,
      softcap: float = 0.0,
      use_3d_buffers: bool = False,
  ):
    q_lens = [q for _, q in ctx_and_q_lens]
    kv_lens = [c + q for c, q in ctx_and_q_lens]
    num_seqs = len(q_lens)
    q, keys, values = _random_inputs(
        q_lens, kv_lens,
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, dtype=dtype,
    )
    # The unified kernel reads K/V only from the cache: the new tokens must
    # already be written (vLLM does this in `do_kv_cache_update`).
    key_cache, value_cache, block_table = _make_paged_cache(
        keys, values, block_size=block_size
    )
    out = torch.full_like(q, float("nan"))
    scale = 1.0 / math.sqrt(head_dim)
    # As in TritonAttentionImpl.forward for an unquantized cache.
    descale = torch.ones((), dtype=torch.float32, device=_DEVICE).expand(
        num_seqs, num_kv_heads
    )

    segm_kwargs = {}
    if use_3d_buffers:
      seq_threshold_3d = _MIN_LAUNCH_GRID_SIZE_2D // num_kv_heads
      self.assertLessEqual(num_seqs, seq_threshold_3d)
      segm_shape = (seq_threshold_3d, num_q_heads, _NUM_PAR_SOFTMAX_SEGMENTS)
      segm_kwargs = dict(
          seq_threshold_3D=seq_threshold_3d,
          num_par_softmax_segments=_NUM_PAR_SOFTMAX_SEGMENTS,
          softmax_segm_output=torch.empty(
              *segm_shape, _next_power_of_2(head_dim),
              dtype=torch.float32, device=_DEVICE,
          ),
          softmax_segm_max=torch.empty(
              segm_shape, dtype=torch.float32, device=_DEVICE
          ),
          softmax_segm_expsum=torch.empty(
              segm_shape, dtype=torch.float32, device=_DEVICE
          ),
      )

    triton_unified_attention.unified_attention(
        q=q,
        k=key_cache,
        v=value_cache,
        out=out,
        cu_seqlens_q=_cu_seqlens(q_lens),
        max_seqlen_q=max(q_lens),
        seqused_k=torch.tensor(kv_lens, dtype=torch.int32, device=_DEVICE),
        max_seqlen_k=max(kv_lens),
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1) if window_left is None else (window_left, 0),
        block_table=block_table,
        softcap=softcap,
        q_descale=None,
        k_descale=descale,
        v_descale=descale,
        **segm_kwargs,
    )
    torch.cuda.synchronize()

    expected = _reference_attention(
        q, keys, values, q_lens,
        scale=scale, window_left=window_left, softcap=softcap,
    )
    torch.testing.assert_close(out, expected, **_tolerance(dtype))

  @parameterized.named_parameters(
      ("prefill_no_prefix", [(0, 1), (0, 17), (0, 128), (0, 300)]),
      ("prefill_with_prefix", [(64, 37), (100, 20), (513, 64), (15, 3)]),
      (
          "mixed_prefill_and_decode",
          [(0, 37), (64, 1), (100, 20), (513, 1), (15, 1), (200, 64)],
      ),
      ("decode_2d", [(n, 1) for n in (0, 1, 15, 16, 17, 255, 1000, 33)]),
  )
  def test_batches(self, ctx_and_q_lens):
    self._run(ctx_and_q_lens)

  @parameterized.named_parameters(
      ("few_seqs", [(n, 1) for n in (0, 16, 300, 1023, 5)]),
      ("long_context", [(4000, 1), (2047, 1)]),
  )
  def test_decode_3d_split_softmax(self, ctx_and_q_lens):
    self._run(ctx_and_q_lens, use_3d_buffers=True)

  @parameterized.product(
      heads=[(8, 2), (4, 4), (32, 8)],
      head_dim=[64, 128, 256],
      block_size=[16, 32],
      dtype=[torch.bfloat16, torch.float16],
  )
  def test_shapes(self, heads, head_dim, block_size, dtype):
    num_q_heads, num_kv_heads = heads
    self._run(
        [(0, 33), (70, 1), (129, 16), (5, 1)],
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
        head_dim=head_dim, block_size=block_size, dtype=dtype,
    )

  @parameterized.product(
      window_left=[None, 31, 127],
      softcap=[0.0, 30.0],
      use_3d_buffers=[False, True],
  )
  def test_window_and_softcap(self, window_left, softcap, use_3d_buffers):
    batch = (
        [(300, 1), (40, 1), (1000, 1)]
        if use_3d_buffers
        else [(0, 200), (300, 1), (40, 90)]
    )
    self._run(
        batch,
        window_left=window_left,
        softcap=softcap,
        use_3d_buffers=use_3d_buffers,
    )


if __name__ == "__main__":
  absltest.main()
