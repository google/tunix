#!/usr/bin/env python3
"""Phase E1 Diagnostic Probe: RPA Tail-Page VMEM Padding & Online Softmax Divergence.

Simulates 100 random decode calls over a 1226-token prefix in bfloat16:
- Sequence length: 1226 tokens
- Page size: 16 tokens -> 77 pages (Page 0..75: 16 tokens; Page 76: 10 tokens)
- Number of Q heads: 32, KV heads: 8, Head Dim: 128
"""

import math
import numpy as np


def softmax(x, axis=-1):
  e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
  return e_x / np.sum(e_x, axis=axis, keepdims=True)


def run_prefill_sdpa(q, k, v, sm_scale):
  seq_len, num_kv_heads, head_dim = k.shape
  num_q_heads = q.shape[1]
  heads_per_kv = num_q_heads // num_kv_heads

  k_rep = np.repeat(k, heads_per_kv, axis=1)
  v_rep = np.repeat(v, heads_per_kv, axis=1)

  q_t = np.transpose(q, (1, 0, 2))
  k_t = np.transpose(k_rep, (1, 2, 0))
  v_t = np.transpose(v_rep, (1, 0, 2))

  scores = np.matmul(q_t, k_t) * sm_scale
  attn_weights = softmax(scores, axis=-1)
  out = np.matmul(attn_weights, v_t)
  return np.transpose(out, (1, 0, 2))


def run_rpa_decode_online_softmax(
    q, k_pages, v_pages, seq_len, page_size, sm_scale, pad_noise=None, dtype=np.float32
):
  num_pages, _, num_kv_heads, head_dim = k_pages.shape
  num_q_heads = q.shape[1]
  heads_per_kv = num_q_heads // num_kv_heads

  m_prev = np.full((num_q_heads, 1, 1), -np.inf, dtype=dtype)
  l_prev = np.zeros((num_q_heads, 1, 1), dtype=dtype)
  o_prev = np.zeros((num_q_heads, 1, head_dim), dtype=dtype)

  q_t = np.transpose(q.astype(dtype), (1, 0, 2))

  processed_tokens = 0
  for p in range(num_pages):
    cur_page_k = k_pages[p].copy().astype(dtype)
    cur_page_v = v_pages[p].copy().astype(dtype)

    tokens_in_page = min(page_size, seq_len - processed_tokens)
    if tokens_in_page < page_size and pad_noise is not None:
      cur_page_k[tokens_in_page:page_size] = pad_noise[0].astype(dtype)
      cur_page_v[tokens_in_page:page_size] = pad_noise[1].astype(dtype)

    k_block_rep = np.repeat(cur_page_k, heads_per_kv, axis=1)
    v_block_rep = np.repeat(cur_page_v, heads_per_kv, axis=1)

    k_block_t = np.transpose(k_block_rep, (1, 2, 0))
    v_block_t = np.transpose(v_block_rep, (1, 0, 2))

    s_block = np.matmul(q_t, k_block_t) * dtype(sm_scale)

    if tokens_in_page < page_size:
      mask = np.arange(page_size) < tokens_in_page
      mask = mask.reshape(1, 1, page_size)
      s_block = np.where(mask, s_block, dtype(-1e30))

    s_rowmax = np.max(s_block, axis=-1, keepdims=True)
    m_curr = np.maximum(m_prev, s_rowmax)

    p_block = np.exp(s_block - m_curr)
    if tokens_in_page < page_size:
      p_block = np.where(mask, p_block, dtype(0.0))

    p_rowsum = np.sum(p_block, axis=-1, keepdims=True)
    exp_m_diff = np.exp(m_prev - m_curr)

    l_curr = exp_m_diff * l_prev + p_rowsum
    pv_block = np.matmul(p_block, v_block_t)
    o_curr = exp_m_diff * o_prev + pv_block

    m_prev = m_curr
    l_prev = l_curr
    o_prev = o_curr
    processed_tokens += tokens_in_page

  final_out = o_prev / l_prev
  return np.transpose(final_out, (1, 0, 2))


class BF16Simulator:
  @staticmethod
  def cast(x):
    f32_bytes = np.ascontiguousarray(x.astype(np.float32)).view(np.uint32)
    rounding = (f32_bytes >> 16) & 1
    bf16_uint32 = (f32_bytes + 0x7FFF + rounding) & 0xFFFF0000
    return bf16_uint32.view(np.float32)


def main():
  print("=== Phase E1 RPA Multi-Call Online Softmax Simulation (100 Calls) ===")
  np.random.seed(2026)

  seq_len = 1226
  page_size = 16
  num_pages = math.ceil(seq_len / page_size) # 77
  tail_tokens = seq_len % page_size # 10

  num_q_heads = 32
  num_kv_heads = 8
  head_dim = 128
  sm_scale = 1.0 / math.sqrt(head_dim)

  # Generate realistic KV cache
  k_flat = np.random.randn(seq_len, num_kv_heads, head_dim).astype(np.float32) * 0.8
  v_flat = np.random.randn(seq_len, num_kv_heads, head_dim).astype(np.float32) * 0.8

  k_pages = np.zeros((num_pages, page_size, num_kv_heads, head_dim), dtype=np.float32)
  v_pages = np.zeros((num_pages, page_size, num_kv_heads, head_dim), dtype=np.float32)
  for p in range(num_pages):
    start = p * page_size
    end = min(start + page_size, seq_len)
    count = end - start
    k_pages[p, :count] = k_flat[start:end]
    v_pages[p, :count] = v_flat[start:end]

  k_pages_bf16 = BF16Simulator.cast(k_pages)
  v_pages_bf16 = BF16Simulator.cast(v_pages)
  k_flat_bf16 = BF16Simulator.cast(k_flat)
  v_flat_bf16 = BF16Simulator.cast(v_flat)

  num_trials = 100
  total_mismatches = 0
  total_elements = num_trials * num_q_heads * head_dim
  max_diff_seen = 0.0
  trials_with_mismatch = 0

  for i in range(num_trials):
    q = np.random.randn(1, num_q_heads, head_dim).astype(np.float32) * 0.8
    q_bf16 = BF16Simulator.cast(q)

    ref_bf16 = BF16Simulator.cast(run_prefill_sdpa(q_bf16, k_flat_bf16, v_flat_bf16, sm_scale))
    rpa_bf16 = BF16Simulator.cast(run_rpa_decode_online_softmax(q_bf16, k_pages_bf16, v_pages_bf16, seq_len, page_size, sm_scale))

    mismatches = np.count_nonzero(ref_bf16 != rpa_bf16)
    diff = np.max(np.abs(ref_bf16 - rpa_bf16))
    if diff > max_diff_seen:
      max_diff_seen = diff
    if mismatches > 0:
      total_mismatches += mismatches
      trials_with_mismatch += 1

  print(f"Evaluated {num_trials} calls across {total_elements} output tensor elements:")
  print(f"- Calls with at least 1 differing element: {trials_with_mismatch} / {num_trials} ({trials_with_mismatch/num_trials*100:.1f}%)")
  print(f"- Total differing elements: {total_mismatches} / {total_elements} ({total_mismatches/total_elements*100:.2f}%)")
  print(f"- Max observed absolute difference: {max_diff_seen:.6f} (LSB level)")
  print(f"- Average differing elements per affected call: {total_mismatches / max(1, trials_with_mismatch):.1f} elements")


if __name__ == "__main__":
  main()
