# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark & per-kernel ablation suite for Stock Qwen3 vs Qwen3Canon.

Evaluates:
1. Forward sampling logprobs & token agreement (`Qwen3` vs `Qwen3Canon`).
2. Training-Inference Mismatch (TIM) numerical invariance metrics:
   - Path A (KV-cached autoregressive decode, T=1, S=cache_size) vs
     Path B (cacheless prefill scoring, T=L, S=L).
   - Batch-size invariance (`B=1` single sequence vs `B=4` batched sequence 0).
   - Path B (forward-only prefill) vs Path C (learner `value_and_grad` fwd+bwd).
3. Execution latency (ms) for KV-cached forward sampling, prefill scoring, and
   learner forward+backward across:
   - Stock `Qwen3` (`tunix.models.qwen3.model`)
   - `Qwen3Canon` (`all_disabled` baseline)
   - `Qwen3Canon` (`all_enabled` full Zero-TIM)
   - Leave-One-Out ablations (`CanonKernelConfig.without(k)`)
   - Add-One-In ablations (`CanonKernelConfig.only(k)`)
"""

from __future__ import annotations

from collections.abc import Sequence
import contextlib
import dataclasses
import json
import os
import time
from typing import Any

from absl import app
from absl import flags
from absl import logging
from flax import nnx
import jax
from jax import numpy as jnp
import numpy as np
from tunix.experimental.zero_tim_kernel import canonical_logsoftmax
from tunix.experimental.zero_tim_kernel.bench.qwen3 import qwen3_model_canon as qwen3_canon
from tunix.models.qwen3 import model as qwen3_stock

# B2 diagnostics (HBM attribution). When this environment variable is set,
# `run_performance_scaling_benchmark` compiles each stock / Zero-TIM perf
# program once more under the HLO module name `jit_hbmdump_<label>`. Paired
# with `XLA_FLAGS=--xla_dump_to=<dir> --xla_dump_hlo_module_re=hbmdump` (the
# TPU test sets both from `TEST_UNDECLARED_OUTPUTS_DIR`) XLA then writes the
# optimized HLO and the buffer assignment of exactly these programs, which is
# what attributes a `Temp HBM` figure to individual buffers. The measured
# programs are untouched: the extra compile is of the same jitted function.
HBM_DUMP_ENV = 'ZERO_TIM_HBM_DUMP'


def hbm_dump_enabled() -> bool:
  """Whether `HBM_DUMP_ENV` asks for the B2 HLO / buffer-assignment dump."""
  value = os.environ.get(HBM_DUMP_ENV, '').strip().lower()
  return value not in ('', '0', 'false', 'no')


HBM_DUMP_MODULE_PREFIX = 'hbmdump_'

# The production vocabulary of Qwen3 (`ModelConfig.qwen3_0p6b().vocab_size`,
# the library's `PRODUCTION_V`) and the same vocabulary padded to the
# library's own column grouping (`TILES_PER_GROUP=8` tiles of
# `VOCAB_TILE=1024` columns): `151936 -> 19 groups -> 155648 = 38 * 4096`.
# The padded size makes every rank's `V/TP` shard a whole number of
# 1024-column tiles for TP in {2, 4, 8}, which is what admits the
# vocab-parallel log-softmax path (`vocab_parallel_logsoftmax_admitted`);
# `151936` itself is not admitted at any of these degrees (`V/TP % 1024` =
# 192 / 96 / 560).
PRODUCTION_VOCAB = canonical_logsoftmax.PRODUCTION_V
_VOCAB_GROUP_COLUMNS = (
    canonical_logsoftmax.TILES_PER_GROUP * canonical_logsoftmax.VOCAB_TILE
)
PRODUCTION_VOCAB_ALIGNED = (
    -(-PRODUCTION_VOCAB // _VOCAB_GROUP_COLUMNS) * _VOCAB_GROUP_COLUMNS
)

MODEL_PRESETS: tuple[str, ...] = (
    'mini_qwen3',
    'qwen3_0p6b_2l',
    'qwen3_0p6b_2l_fullvocab',
    'qwen3_0p6b_2l_fullvocab_aligned',
    'qwen3_0p6b',
)

_MODEL_PRESET = flags.DEFINE_enum(
    'model_preset',
    'mini_qwen3',
    list(MODEL_PRESETS),
    'Model size preset to run for the benchmark and ablation suite.',
)
_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 4, 'Batch size B for batched forward sampling and scoring.'
)
_PROMPT_LEN = flags.DEFINE_integer(
    'prompt_len', 6, 'Prompt token length L_prompt.'
)
_MAX_NEW_TOKENS = flags.DEFINE_integer(
    'max_new_tokens', 4, 'Number of new tokens to decode autoregressively.'
)
_CACHE_SIZE = flags.DEFINE_integer(
    'cache_size', 32, 'KV cache size (>= prompt_len + max_new_tokens).'
)
_NUM_ITERS = flags.DEFINE_integer(
    'num_iters', 5, 'Number of timed iterations after warmup.'
)
_RUN_ABLATIONS = flags.DEFINE_bool(
    'run_ablations',
    True,
    'Whether to run full Leave-One-Out and Add-One-In per-kernel ablations.',
)
_INCLUDE_FWDBWD = flags.DEFINE_bool(
    'include_fwdbwd',
    True,
    'Whether to include learner value_and_grad fwd+bwd check and timing.',
)
_TP_SIZE = flags.DEFINE_integer(
    'tp_size',
    0,
    'Tensor-parallel degree (0 = auto: use all TPU devices if >=2 else 1).',
)
_OUTPUT_JSON = flags.DEFINE_string(
    'output_json',
    '',
    'Optional path to write benchmark and ablation results as JSON.',
)


@dataclasses.dataclass(slots=True, frozen=True)
class MemoryProfile:
  """Per-device HBM memory breakdown (MB) from XLA compiled memory_analysis.

  `temp_hbm_mb` / `total_hbm_mb` are PJRT's legacy `temp_size_in_bytes`
  numbers. On TPU `temp_size_in_bytes` is the HLO temporary block size *plus*
  its fragmentation although the block size already includes that
  fragmentation, so it double counts fragmentation and must not be read as
  live memory. They are kept for continuity with older reports; cross-run
  comparisons should use the buffer-assignment based metrics:

  * `arena_hbm_mb`: bytes of all non-indefinite (temporary) HBM buffer
    allocations, `total_allocation_bytes - indefinite_allocations` (both are
    default-memory-space, i.e. HBM, only). On TPU this is exactly the HLO
    temporary arena of the buffer assignment including its internal
    fragmentation.
  * `code_hbm_mb`: `generated_code_size_in_bytes` (TPU: GLOBAL + OVERLAYS
    blocks, i.e. per-call-site kernel code). The TPU "program HBM
    requirement" is `code_hbm_mb + arena_hbm_mb`.
  * `ba_peak_mb`: `peak_memory_in_bytes - indefinite_allocations`,
    informational only. PJRT's `peak_memory_in_bytes` is the buffer-assignment
    peak over *all* memory spaces (on TPU it includes the VMEM scoped buffers
    of every fusion / Mosaic kernel), so after subtracting the HBM-only
    indefinite allocations it is neither an HBM number nor comparable to
    `arena_hbm_mb` (it exceeds the arena for small programs). It is not
    reported in the scaling table; the per-memory-space heap-simulator peak
    is only available from an XLA dump (`--xla_dump_to`).
  """

  arg_hbm_mb: float = 0.0
  output_hbm_mb: float = 0.0
  temp_hbm_mb: float = 0.0
  alias_hbm_mb: float = 0.0
  total_hbm_mb: float = 0.0
  arena_hbm_mb: float = 0.0
  code_hbm_mb: float = 0.0
  ba_peak_mb: float = 0.0
  device_in_use_hbm_mb: float = 0.0
  device_peak_hbm_mb: float = 0.0


@dataclasses.dataclass(slots=True)
class BenchmarkRow:
  """Metrics for a single model / kernel ablation configuration."""

  name: str
  category: str
  # Logprob comparison against Stock Qwen3 & Full Qwen3Canon on identical tokens
  vs_stock_max_abs_diff: float
  vs_stock_mean_abs_diff: float
  vs_canon_max_abs_diff: float
  vs_canon_mean_abs_diff: float
  vs_canon_exact_match_pct: float
  # Greedy sampled token agreement rate vs Full Qwen3Canon
  sampled_token_match_pct: float
  # Zero-TIM Invariance checks (within the configuration itself)
  decode_vs_prefill_max_diff: float
  decode_vs_prefill_mean_diff: float
  decode_vs_prefill_exact_pct: float
  batch1_vs_batchn_max_diff: float
  batch1_vs_batchn_exact_pct: float
  fwd_vs_fwdbwd_max_diff: float
  fwd_vs_fwdbwd_exact_pct: float
  # Latency metrics (ms)
  sample_latency_ms: float
  prefill_latency_ms: float
  fwdbwd_latency_ms: float
  # Per-device HBM metrics (MB) from XLA compiled executable analysis
  sample_temp_hbm_mb: float = 0.0
  sample_total_hbm_mb: float = 0.0
  prefill_temp_hbm_mb: float = 0.0
  prefill_total_hbm_mb: float = 0.0
  fwdbwd_temp_hbm_mb: float = 0.0
  fwdbwd_total_hbm_mb: float = 0.0


def build_model_config(
    preset: str, *, tp_size: int = 1
) -> qwen3_stock.ModelConfig:
  """Builds a `ModelConfig` in `bfloat16` for benchmarking and ablations."""
  shd_cfg = (
      qwen3_canon.get_tp_sharding_config('tp')
      if tp_size > 1
      else qwen3_stock.ShardingConfig.get_default_sharding(is_sampling=True)
  )
  bf16_dtype = jnp.dtype(jnp.bfloat16)
  if preset == 'mini_qwen3':
    # Tile-aligned Qwen3 architecture (embed_dim=256, hidden_dim=512,
    # head_dim=128, vocab_size=2048) that exercises multi-tile Pallas reductions
    # across every kernel (BM=128, BN=128/256, BK=128, BF=128/256,
    # VOCAB_TILE=1024 x 2, bkv_csz=512) on CPU interpret mode or TPU. Under TP
    # the vocabulary grows to `1024 * tp_size` so every rank's `V/TP` shard is
    # a whole tile and the vocab-parallel log-softmax path
    # (`vocab_parallel_logsoftmax_admitted`) is exercised rather than its
    # gathered fallback.
    return qwen3_stock.ModelConfig(
        num_layers=2,
        vocab_size=max(2048, 1024 * tp_size),
        embed_dim=256,
        hidden_dim=max(512, 128 * tp_size),
        num_heads=max(4, 2 * tp_size),
        head_dim=128,
        num_kv_heads=max(2, tp_size),
        norm_eps=1e-6,
        rope_theta=1_000_000,
        use_tied_embedding=True,
        shd_config=shd_cfg,
        dtype=bf16_dtype,
        param_dtype=bf16_dtype,
    )
  if preset == 'qwen3_0p6b_2l':
    cfg = qwen3_stock.ModelConfig.qwen3_0p6b()
    cfg.num_layers = 2
    cfg.vocab_size = 8192 if tp_size > 1 else 2048
    cfg.shd_config = shd_cfg
    cfg.dtype = bf16_dtype
    cfg.param_dtype = bf16_dtype
    return cfg
  if preset in ('qwen3_0p6b_2l_fullvocab', 'qwen3_0p6b_2l_fullvocab_aligned'):
    # Two layers at the production vocabulary (`qwen3_0p6b_2l` shrinks it to
    # 8192 so that the suite runs in minutes), independent of `tp_size`. At
    # `V=151936` the vocab-parallel log-softmax path is not admitted for
    # TP in {2, 4, 8} (see `PRODUCTION_VOCAB`), so `compute_log_softmax` /
    # `compute_token_logprobs` take their gathered fallback and the fixed LM
    # head pads its `N=V/TP` columns (`37984 -> 38144` at TP=4). The
    # `_aligned` variant uses `PRODUCTION_VOCAB_ALIGNED` (`V/TP = 38 * 1024`
    # at TP=4; no LM-head padding), which admits the vocab-parallel path: the
    # pair measures what aligning the vocabulary to `1024 * TP` buys at
    # production scale. Both keep the 0.6B widths (`embed_dim=1024`,
    # `hidden_dim=3072`, 16 / 8 heads) and tied embeddings.
    cfg = qwen3_stock.ModelConfig.qwen3_0p6b()
    cfg.num_layers = 2
    cfg.vocab_size = (
        PRODUCTION_VOCAB_ALIGNED
        if preset.endswith('_aligned')
        else PRODUCTION_VOCAB
    )
    cfg.shd_config = shd_cfg
    cfg.dtype = bf16_dtype
    cfg.param_dtype = bf16_dtype
    return cfg
  if preset == 'qwen3_0p6b':
    cfg = qwen3_stock.ModelConfig.qwen3_0p6b()
    cfg.shd_config = shd_cfg
    cfg.dtype = bf16_dtype
    cfg.param_dtype = bf16_dtype
    return cfg
  raise ValueError(f'Unknown preset: {preset}')


def _stat_bytes(stats: Any, name: str) -> float:
  """Returns `stats.<name>` as float bytes (0.0 if absent / None)."""
  return float(getattr(stats, name, 0) or 0)


def _extract_memory_profile(fn, *args, **kwargs) -> MemoryProfile:
  """Extracts per-device HBM breakdown (MB) from `fn.lower(...).compile()`."""
  mb = 1024.0 * 1024.0
  arg_mb = out_mb = temp_mb = alias_mb = total_mb = 0.0
  arena_mb = code_mb = ba_peak_mb = 0.0
  try:
    compiled = fn.lower(*args, **kwargs).compile()
    stats = compiled.memory_analysis()
    if stats is not None:
      arg_mb = _stat_bytes(stats, 'argument_size_in_bytes') / mb
      out_mb = _stat_bytes(stats, 'output_size_in_bytes') / mb
      temp_mb = _stat_bytes(stats, 'temp_size_in_bytes') / mb
      alias_mb = _stat_bytes(stats, 'alias_size_in_bytes') / mb
      total_mb = max(0.0, arg_mb + out_mb + temp_mb - alias_mb)
      # Buffer-assignment based metrics (see `MemoryProfile`).
      indefinite_b = _stat_bytes(stats, 'indefinite_allocations')
      arena_mb = (
          max(0.0, _stat_bytes(stats, 'total_allocation_bytes') - indefinite_b)
          / mb
      )
      code_mb = _stat_bytes(stats, 'generated_code_size_in_bytes') / mb
      ba_peak_mb = (
          max(0.0, _stat_bytes(stats, 'peak_memory_in_bytes') - indefinite_b)
          / mb
      )
  except Exception:  # pylint: disable=broad-exception-caught
    pass

  dev_in_use_mb = dev_peak_mb = 0.0
  try:
    dev_stats = jax.devices()[0].memory_stats()
    if dev_stats:
      dev_in_use_mb = float(dev_stats.get('bytes_in_use', 0)) / mb
      dev_peak_mb = float(dev_stats.get('peak_bytes_in_use', 0)) / mb
  except Exception:  # pylint: disable=broad-exception-caught
    pass

  return MemoryProfile(
      arg_hbm_mb=arg_mb,
      output_hbm_mb=out_mb,
      temp_hbm_mb=temp_mb,
      alias_hbm_mb=alias_mb,
      total_hbm_mb=total_mb,
      arena_hbm_mb=arena_mb,
      code_hbm_mb=code_mb,
      ba_peak_mb=ba_peak_mb,
      device_in_use_hbm_mb=dev_in_use_mb,
      device_peak_hbm_mb=dev_peak_mb,
  )


def _dump_compiled_program(fn, label: str, *args, **kwargs) -> None:
  """B2: re-compiles `fn(*args, **kwargs)` as HLO module `jit_hbmdump_<label>`.

  No-op unless `HBM_DUMP_ENV` is set (see the module constant). `kwargs` must
  be the static (hashable) keyword arguments of `fn`.

  Args:
    fn: The jit-able callable to re-compile.
    label: Suffix of the dumped HLO module name.
    *args: Positional (array) arguments of `fn`.
    **kwargs: Static keyword arguments of `fn`.
  """
  if not hbm_dump_enabled():
    return

  def wrapper(*a, **k):
    return fn(*a, **k)

  wrapper.__name__ = wrapper.__qualname__ = HBM_DUMP_MODULE_PREFIX + label
  try:
    jax.jit(wrapper, static_argnames=tuple(kwargs)).lower(
        *args, **kwargs
    ).compile()
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.warning('HBM dump compile of %s failed: %s', label, e)


def _time_fn(
    fn, *args, num_iters: int = 5, **kwargs
) -> tuple[Any, float, MemoryProfile]:
  """Runs 1 warmup call, `num_iters` timed calls, and extracts HBM profile."""
  out = fn(*args, **kwargs)
  jax.block_until_ready(out)
  times_ms = []
  for _ in range(num_iters):
    t0 = time.perf_counter()
    res = fn(*args, **kwargs)
    jax.block_until_ready(res)
    times_ms.append((time.perf_counter() - t0) * 1000.0)
  mem_profile = _extract_memory_profile(fn, *args, **kwargs)
  return out, float(np.median(times_ms)), mem_profile


def _diff_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
  """Returns `(max_abs_diff, mean_abs_diff, exact_match_pct)`."""
  a_f32 = np.asarray(a, dtype=np.float32)
  b_f32 = np.asarray(b, dtype=np.float32)
  abs_diff = np.abs(a_f32 - b_f32)
  max_diff = float(np.max(abs_diff))
  mean_diff = float(np.mean(abs_diff))
  exact_pct = float(np.mean(a_f32 == b_f32) * 100.0)
  return max_diff, mean_diff, exact_pct


def evaluate_configuration(
    name: str,
    category: str,
    model: qwen3_stock.Qwen3 | qwen3_canon.Qwen3Canon,
    *,
    prompt_tokens: jax.Array,
    ref_full_tokens: jax.Array,
    stock_prefill_lps: np.ndarray,
    canon_full_prefill_lps: np.ndarray,
    canon_full_sampled_tokens: np.ndarray,
    max_new_tokens: int,
    cache_size: int,
    use_canon_logsoftmax: bool,
    num_iters: int,
    include_fwdbwd: bool = True,
) -> BenchmarkRow:
  """Evaluates one model/kernel configuration on numerics, TIM, HBM, and latency."""
  # 1. Forward sampling with KV cache (Path A: prefill + decode steps)
  (full_tokens, decode_step_lps, _), sample_ms, sample_mem = _time_fn(
      qwen3_canon.forward_sample_with_logprobs,
      model,
      prompt_tokens,
      max_new_tokens=max_new_tokens,
      cache_size=cache_size,
      use_canon_logsoftmax=use_canon_logsoftmax,
      num_iters=num_iters,
  )

  # 2. Cacheless prefill scoring on the model's own generated trajectory
  #    to check Decode (Path A) vs Prefill (Path B) self-consistency!
  l_prompt = int(prompt_tokens.shape[1])
  own_prefill_all_lps, prefill_ms, prefill_mem = _time_fn(
      qwen3_canon.score_sequence_prefill,
      model,
      full_tokens,
      use_canon_logsoftmax=use_canon_logsoftmax,
      num_iters=num_iters,
  )
  # Slice the generated token positions
  # `[l_prompt - 1 : l_prompt - 1 + max_new_tokens]`.
  own_prefill_decode_lps = np.asarray(
      own_prefill_all_lps[:, l_prompt - 1 :], dtype=np.float32
  )
  decode_vs_prefill_max, decode_vs_prefill_mean, decode_vs_prefill_exact = (
      _diff_stats(np.asarray(decode_step_lps), own_prefill_decode_lps)
  )

  # 3. Score on reference trajectory `ref_full_tokens` for apple-to-apple
  #    comparison against Stock Qwen3 and Full Qwen3Canon
  ref_prefill_lps = np.asarray(
      qwen3_canon.score_sequence_prefill(
          model,
          ref_full_tokens,
          use_canon_logsoftmax=use_canon_logsoftmax,
      ),
      dtype=np.float32,
  )
  vs_stock_max, vs_stock_mean, _ = _diff_stats(
      ref_prefill_lps, stock_prefill_lps
  )
  vs_canon_max, vs_canon_mean, vs_canon_exact = _diff_stats(
      ref_prefill_lps, canon_full_prefill_lps
  )

  # 4. Batch-size invariance: score `ref_full_tokens[:1]` (B=1) vs
  #    `ref_full_tokens[0]` from the B=4 batch
  b1_prefill_lps = np.asarray(
      qwen3_canon.score_sequence_prefill(
          model,
          ref_full_tokens[:1],
          use_canon_logsoftmax=use_canon_logsoftmax,
      ),
      dtype=np.float32,
  )[0]
  b1_vs_bn_max, _, b1_vs_bn_exact = _diff_stats(
      b1_prefill_lps, ref_prefill_lps[0]
  )

  # 5. Forward-only (Path B) vs Learner Forward+Backward (Path C)
  if include_fwdbwd:
    (fwdbwd_lps, _), fwdbwd_ms, fwdbwd_mem = _time_fn(
        qwen3_canon.score_sequence_fwd_bwd,
        model,
        ref_full_tokens,
        use_canon_logsoftmax=use_canon_logsoftmax,
        num_iters=num_iters,
    )
    fwd_vs_bwd_max, _, fwd_vs_bwd_exact = _diff_stats(
        ref_prefill_lps, np.asarray(fwdbwd_lps)
    )
  else:
    fwd_vs_bwd_max, fwd_vs_bwd_exact, fwdbwd_ms = 0.0, 100.0, 0.0
    fwdbwd_mem = MemoryProfile()

  gen_tokens_np = np.asarray(full_tokens[:, l_prompt:])
  token_match_pct = float(
      np.mean(gen_tokens_np == canon_full_sampled_tokens) * 100.0
  )

  return BenchmarkRow(
      name=name,
      category=category,
      vs_stock_max_abs_diff=vs_stock_max,
      vs_stock_mean_abs_diff=vs_stock_mean,
      vs_canon_max_abs_diff=vs_canon_max,
      vs_canon_mean_abs_diff=vs_canon_mean,
      vs_canon_exact_match_pct=vs_canon_exact,
      sampled_token_match_pct=token_match_pct,
      decode_vs_prefill_max_diff=decode_vs_prefill_max,
      decode_vs_prefill_mean_diff=decode_vs_prefill_mean,
      decode_vs_prefill_exact_pct=decode_vs_prefill_exact,
      batch1_vs_batchn_max_diff=b1_vs_bn_max,
      batch1_vs_batchn_exact_pct=b1_vs_bn_exact,
      fwd_vs_fwdbwd_max_diff=fwd_vs_bwd_max,
      fwd_vs_fwdbwd_exact_pct=fwd_vs_bwd_exact,
      sample_latency_ms=sample_ms,
      prefill_latency_ms=prefill_ms,
      fwdbwd_latency_ms=fwdbwd_ms,
      sample_temp_hbm_mb=sample_mem.temp_hbm_mb,
      sample_total_hbm_mb=sample_mem.total_hbm_mb,
      prefill_temp_hbm_mb=prefill_mem.temp_hbm_mb,
      prefill_total_hbm_mb=prefill_mem.total_hbm_mb,
      fwdbwd_temp_hbm_mb=fwdbwd_mem.temp_hbm_mb,
      fwdbwd_total_hbm_mb=fwdbwd_mem.total_hbm_mb,
  )


def _run_benchmark_suite_body(
    *,
    preset: str,
    batch_size: int,
    prompt_len: int,
    max_new_tokens: int,
    cache_size: int,
    num_iters: int,
    run_ablations: bool,
    include_baselines: bool,
    include_fwdbwd: bool,
    switches: Sequence[str],
    tp_size: int,
) -> list[BenchmarkRow]:
  """Inner body of `run_benchmark_suite` (executed inside mesh if `tp_size > 1`)."""
  config = build_model_config(preset, tp_size=tp_size)
  stock_model = qwen3_stock.Qwen3(config, rngs=nnx.Rngs(params=0))
  qwen3_canon.init_non_zero_mlp_weights(stock_model, seed=123)

  canon_model = qwen3_canon.Qwen3Canon(
      config,
      rngs=nnx.Rngs(params=0),
      canon_config=qwen3_canon.CanonKernelConfig.all_enabled(),
  )
  qwen3_canon.copy_weights(stock_model, canon_model)

  key = jax.random.PRNGKey(2026)
  prompt_tokens = jax.random.randint(
      key, (batch_size, prompt_len), 0, config.vocab_size, dtype=jnp.int32
  )

  # Generate the reference trajectory and logprobs with the full Qwen3Canon
  # and with stock Qwen3.
  canon_model.set_canon_config(qwen3_canon.CanonKernelConfig.all_enabled())
  ref_full_tokens, _, _ = qwen3_canon.forward_sample_with_logprobs(
      canon_model,
      prompt_tokens,
      max_new_tokens=max_new_tokens,
      cache_size=cache_size,
      use_canon_logsoftmax=True,
  )
  canon_full_sampled_tokens = np.asarray(ref_full_tokens[:, prompt_len:])
  canon_full_prefill_lps = np.asarray(
      qwen3_canon.score_sequence_prefill(
          canon_model,
          ref_full_tokens,
          use_canon_logsoftmax=True,
      ),
      dtype=np.float32,
  )
  stock_prefill_lps = np.asarray(
      qwen3_canon.score_sequence_prefill(
          stock_model,
          ref_full_tokens,
          use_canon_logsoftmax=False,
      ),
      dtype=np.float32,
  )

  rows: list[BenchmarkRow] = []

  def evaluate(name, category, model, *, use_canon_logsoftmax):
    return evaluate_configuration(
        name,
        category,
        model,
        prompt_tokens=prompt_tokens,
        ref_full_tokens=ref_full_tokens,
        stock_prefill_lps=stock_prefill_lps,
        canon_full_prefill_lps=canon_full_prefill_lps,
        canon_full_sampled_tokens=canon_full_sampled_tokens,
        max_new_tokens=max_new_tokens,
        cache_size=cache_size,
        use_canon_logsoftmax=use_canon_logsoftmax,
        num_iters=num_iters,
        include_fwdbwd=include_fwdbwd,
    )

  if include_baselines:
    # 1. Stock Qwen3 (`tunix.models.qwen3.model`)
    rows.append(
        evaluate(
            'stock_qwen3 (model.py)',
            'baseline',
            stock_model,
            use_canon_logsoftmax=False,
        )
    )

    # 2. Qwen3Canon with all switches disabled (verifies 0.0 diff vs stock
    #    Qwen3)
    canon_model.set_canon_config(qwen3_canon.CanonKernelConfig.all_disabled())
    rows.append(
        evaluate(
            'canon_all_disabled',
            'baseline',
            canon_model,
            use_canon_logsoftmax=False,
        )
    )

    # 3. Qwen3Canon with all switches enabled (Full Zero-TIM)
    canon_model.set_canon_config(qwen3_canon.CanonKernelConfig.all_enabled())
    rows.append(
        evaluate(
            'canon_all_enabled (Zero-TIM)',
            'baseline',
            canon_model,
            use_canon_logsoftmax=True,
        )
    )

  if not run_ablations:
    return rows

  # 4. Leave-One-Out ablations: disable one kernel at a time from Full Zero-TIM
  for switch_name in switches:
    cfg_loo = qwen3_canon.CanonKernelConfig.without(switch_name)
    canon_model.set_canon_config(cfg_loo)
    rows.append(
        evaluate(
            f'without_{switch_name}',
            'leave_one_out',
            canon_model,
            use_canon_logsoftmax=cfg_loo.use_canon_logsoftmax,
        )
    )

  # 5. Add-One-In ablations: enable ONLY one kernel at a time on top of stock
  for switch_name in switches:
    cfg_aoi = qwen3_canon.CanonKernelConfig.only(switch_name)
    canon_model.set_canon_config(cfg_aoi)
    rows.append(
        evaluate(
            f'only_{switch_name}',
            'add_one_in',
            canon_model,
            use_canon_logsoftmax=cfg_aoi.use_canon_logsoftmax,
        )
    )

  return rows


def run_benchmark_suite(
    *,
    preset: str = 'mini_qwen3',
    batch_size: int = 4,
    prompt_len: int = 6,
    max_new_tokens: int = 4,
    cache_size: int = 32,
    num_iters: int = 5,
    run_ablations: bool = True,
    include_baselines: bool = True,
    include_fwdbwd: bool = True,
    switches: Sequence[str] = qwen3_canon.ABLATION_SWITCH_NAMES,
    tp_size: int | None = None,
) -> list[BenchmarkRow]:
  """Runs the complete Stock vs Canon benchmark and per-kernel ablation suite.

  The drift columns of every row compare against the full `Qwen3Canon` and
  stock Qwen3 references, which are computed regardless of the row selection.

  Args:
    preset: Model preset (`MODEL_PRESETS`).
    batch_size: Prompt batch size `B`; the `B=1` column re-scores row 0 alone.
    prompt_len: Prompt length of the random prompt.
    max_new_tokens: Decode steps of the sampler.
    cache_size: KV-cache slots (`>= prompt_len + max_new_tokens`).
    num_iters: Timed iterations per program (after one warm-up).
    run_ablations: Append one Leave-One-Out and one Add-One-In row per entry of
      `switches`.
    include_baselines: Evaluate the three baseline rows (stock,
      `canon_all_disabled`, `canon_all_enabled`).
    include_fwdbwd: Run the learner fwd+bwd program (else its diff / latency
      columns read 0 and its exact-match column 100).
    switches: `CanonKernelConfig` switch names for the ablation rows.
    tp_size: Tensor-parallel size; `None` / `0` picks the TPU device count when
      it is 2, 4 or 8 and 1 otherwise.

  Returns:
    Rows in the order stock, `canon_all_disabled`, `canon_all_enabled` (if
    `include_baselines`), then `without_<switch>` for every switch, then
    `only_<switch>` for every switch (if `run_ablations`).
  """
  if tp_size is None or tp_size <= 0:
    num_devs = len(jax.devices())
    if jax.default_backend() == 'tpu' and num_devs in (2, 4, 8):
      tp_size = num_devs
    else:
      tp_size = 1

  if tp_size > 1:
    devices = np.asarray(jax.devices()[:tp_size]).reshape(1, tp_size)
    mesh = jax.sharding.Mesh(devices, ('fsdp', 'tp'))
  else:
    mesh = contextlib.nullcontext()
  with mesh:
    return _run_benchmark_suite_body(
        preset=preset,
        batch_size=batch_size,
        prompt_len=prompt_len,
        max_new_tokens=max_new_tokens,
        cache_size=cache_size,
        num_iters=num_iters,
        run_ablations=run_ablations,
        include_baselines=include_baselines,
        include_fwdbwd=include_fwdbwd,
        switches=switches,
        tp_size=tp_size,
    )


def format_markdown_report(rows: Sequence[BenchmarkRow]) -> str:
  """Formats benchmark & ablation rows into a Markdown table."""
  lines = [
      (
          '| Configuration | Category | `|LP - Stock|` Max |'
          ' `|LP - Canon|` Max | `|Decode - Prefill|` Max (Exact %) |'
          ' `|B=1 - B=N|` Max (Exact %) | `|Fwd - FwdBwd|` Max (Exact %) |'
          ' Sample (ms) | Prefill (ms) | Fwd+Bwd (ms) |'
          ' Sample HBM Temp/Tot (MB) | Prefill HBM Temp/Tot (MB) |'
          ' Fwd+Bwd HBM Temp/Tot (MB) |'
      ),
      (
          '| :--- | :--- | :---: | :---: | :---: | :---: | :---: |'
          ' :---: | :---: | :---: | :---: | :---: | :---: |'
      ),
  ]
  for r in rows:
    lines.append(
        f'| `{r.name}` | `{r.category}` | '
        f'`{r.vs_stock_max_abs_diff:.3e}` | '
        f'`{r.vs_canon_max_abs_diff:.3e}` | '
        f'`{r.decode_vs_prefill_max_diff:.3e}` '
        f'({r.decode_vs_prefill_exact_pct:.1f}%) | '
        f'`{r.batch1_vs_batchn_max_diff:.3e}` '
        f'({r.batch1_vs_batchn_exact_pct:.1f}%) | '
        f'`{r.fwd_vs_fwdbwd_max_diff:.3e}` '
        f'({r.fwd_vs_fwdbwd_exact_pct:.1f}%) | '
        f'`{r.sample_latency_ms:.2f}` | '
        f'`{r.prefill_latency_ms:.2f}` | '
        f'`{r.fwdbwd_latency_ms:.2f}` | '
        f'`{r.sample_temp_hbm_mb:.2f} / {r.sample_total_hbm_mb:.2f}` | '
        f'`{r.prefill_temp_hbm_mb:.2f} / {r.prefill_total_hbm_mb:.2f}` | '
        f'`{r.fwdbwd_temp_hbm_mb:.2f} / {r.fwdbwd_total_hbm_mb:.2f}` |'
    )
  return '\n'.join(lines)


@dataclasses.dataclass(slots=True)
class PerfScalingRow:
  """Performance (Latency & HBM) row across sequence lengths and layer depths.

  `category` is `'baseline'` for the stock / `canon_all_disabled` /
  `canon_all_enabled` rows and `'leave_one_out'` for the timing-only
  `without_<switch>` rows (see `run_performance_scaling_benchmark`).
  """

  workload: str
  config_name: str
  sample_latency_ms: float
  prefill_latency_ms: float
  prefill_tok_per_s: float
  fwdbwd_latency_ms: float
  fwdbwd_tok_per_s: float
  sample_arg_hbm_mb: float
  sample_temp_hbm_mb: float
  sample_total_hbm_mb: float
  prefill_arg_hbm_mb: float
  prefill_temp_hbm_mb: float
  prefill_total_hbm_mb: float
  fwdbwd_arg_hbm_mb: float
  fwdbwd_temp_hbm_mb: float
  fwdbwd_total_hbm_mb: float
  device_peak_hbm_mb: float
  category: str = 'baseline'
  # Buffer-assignment based HBM metrics (see `MemoryProfile`).
  sample_arena_hbm_mb: float = 0.0
  sample_code_hbm_mb: float = 0.0
  prefill_arena_hbm_mb: float = 0.0
  prefill_code_hbm_mb: float = 0.0
  fwdbwd_arena_hbm_mb: float = 0.0
  fwdbwd_code_hbm_mb: float = 0.0


@dataclasses.dataclass(slots=True, frozen=True)
class PerfWorkload:
  """One (depth, batch, prompt, decode) workload of the scaling benchmark."""

  label: str
  num_layers: int
  batch_size: int
  prompt_len: int
  max_new_tokens: int
  cache_size: int
  use_remat: bool = False


DEFAULT_PERF_WORKLOADS: tuple[PerfWorkload, ...] = (
    PerfWorkload('L=2, B=4, T=64 (48+16)', 2, 4, 48, 16, 128),
    PerfWorkload('L=2, B=4, T=256 (224+32)', 2, 4, 224, 32, 256),
    PerfWorkload('L=8, B=4, T=64 (48+16)', 8, 4, 48, 16, 128),
    PerfWorkload('L=8, B=4, T=256 (224+32)', 8, 4, 224, 32, 256),
    PerfWorkload('L=8, B=4, T=256 (Remat=BLOCK)', 8, 4, 224, 32, 256, True),
    # Larger batches: at `B=4` every program is launch-bound (the Leave-One-Out
    # rows attribute the Sample gap to ~3-4 us per kernel call site, not to any
    # one kernel); these rows show how the stock / canon gap scales once the
    # per-token work dominates.
    PerfWorkload('L=8, B=16, T=256 (224+32)', 8, 16, 224, 32, 256),
    PerfWorkload('L=8, B=32, T=256 (224+32)', 8, 32, 224, 32, 256),
)

# Workload(s) on which `run_performance_scaling_benchmark` additionally times
# the Leave-One-Out configurations (`without_<switch>`): the largest
# non-remat `B=4` workload, where per-kernel latency contributions are
# measurable above launch-overhead noise. The `B=32` workload -- where the
# stock / canon gap is a steady-state throughput gap (`~2x`, growing with the
# batch) that the `B=4` attribution does not transfer to -- gets the same
# Leave-One-Out treatment in `qwen3_model_canon_test.test_leave_one_out_b32_tpu`
# (its own process, so the TPU target's shards stay inside the test runner's
# 60-minute timeout).
DEFAULT_LEAVE_ONE_OUT_PERF_WORKLOADS: tuple[str, ...] = (
    'L=8, B=4, T=256 (224+32)',
)

# Learner row buckets of the production-vocabulary presets
# (`qwen3_0p6b_2l_fullvocab[_aligned]`): `B * T = 2048` and `4096` scoring
# rows, i.e. the two sides of the `FORWARD_TP_SUM_SCATTER_MIN_M` threshold,
# where the `[M, V]` logits (`4096 x 151936` bf16 = 1.2 GB per rank once
# gathered) rather than the per-call launch overhead set the stock / Zero-TIM
# HBM and latency gap. Timed by
# `qwen3_model_canon_test.test_production_vocab_hbm_latency_tpu` together with
# the `qwen3_0p6b_2l` (`V=8192`) control row of the larger workload.
PRODUCTION_VOCAB_PERF_WORKLOADS: tuple[PerfWorkload, ...] = (
    PerfWorkload('L=2, B=8, T=256 (224+32)', 2, 8, 224, 32, 256),
    PerfWorkload('L=2, B=16, T=256 (224+32)', 2, 16, 224, 32, 256),
)

# Sequence-length sweep of the `L=8, B=4` row of `DEFAULT_PERF_WORKLOADS`:
# `T = 256, 512, 1024, 2048` with the cache sized to the sequence, i.e. the
# Pallas attention forward visiting 1, 1, 2 and 4 fixed 512-key blocks (the
# sampler prefill and decode programs see the same block count as the
# scoring prefill). Timed, with the attention Leave-One-Out row on every
# workload, by
# `qwen3_model_canon_test.test_zero_tim_long_sequence_multi_block_tpu`.
LONG_SEQUENCE_PERF_WORKLOADS: tuple[PerfWorkload, ...] = (
    PerfWorkload('L=8, B=4, T=256 (224+32)', 8, 4, 224, 32, 256),
    PerfWorkload('L=8, B=4, T=512 (480+32)', 8, 4, 480, 32, 512),
    PerfWorkload('L=8, B=4, T=1024 (992+32)', 8, 4, 992, 32, 1024),
    PerfWorkload('L=8, B=4, T=2048 (2016+32)', 8, 4, 2016, 32, 2048),
)

_BASELINE_PERF_CONFIG_NAMES: tuple[str, ...] = (
    'stock_qwen3 (model.py)',
    'canon_all_disabled',
    'canon_all_enabled (Zero-TIM)',
)


def run_performance_scaling_benchmark(
    *,
    tp_size: int = 4,
    num_iters: int = 5,
    preset: str = 'qwen3_0p6b_2l',
    workloads: Sequence[PerfWorkload] = DEFAULT_PERF_WORKLOADS,
    leave_one_out_workloads: Sequence[str] = (
        DEFAULT_LEAVE_ONE_OUT_PERF_WORKLOADS
    ),
    leave_one_out_switches: Sequence[str] = qwen3_canon.ABLATION_SWITCH_NAMES,
    configs: Sequence[str] | None = None,
) -> list[PerfScalingRow]:
  """Benchmarks Latency and HBM across sequence lengths and model depths.

  For every workload the three baselines (stock, `canon_all_disabled`,
  `canon_all_enabled`) are timed and, when `HBM_DUMP_ENV` is set, re-compiled
  for the B2 HBM dump. For the workloads whose label is in
  `leave_one_out_workloads` the Leave-One-Out configurations
  `without_<switch>` (`switch` in `leave_one_out_switches`) are additionally
  timed; these rows carry `category='leave_one_out'`, are timing / HBM only
  (the Zero-TIM contract of each switch is checked by `run_benchmark_suite`)
  and are not dumped. `configs` (configuration names as they appear in the
  report) restricts the timed rows, e.g. to `canon_all_enabled (Zero-TIM)` for
  a same-process A/B of a glue switch the stock rows cannot see; `None` times
  every configuration.

  Args:
    tp_size: Tensor-parallel mesh size.
    num_iters: Timed calls per measurement (after one warm-up call).
    preset: `build_model_config` preset name.
    workloads: The `PerfWorkload`s to time.
    leave_one_out_workloads: Labels of the workloads that also get the
      Leave-One-Out rows.
    leave_one_out_switches: `CanonKernelConfig` switches to leave out.
    configs: Configuration names to time; `None` times every configuration.

  Returns:
    One `PerfScalingRow` per timed (workload, configuration) pair.
  """
  known_configs = set(_BASELINE_PERF_CONFIG_NAMES) | {
      f'without_{switch_name}' for switch_name in leave_one_out_switches
  }
  selected_configs = None if configs is None else set(configs)
  if selected_configs is not None and not selected_configs <= known_configs:
    raise ValueError(
        'Unknown perf configs'
        f' {sorted(selected_configs - known_configs)}; known:'
        f' {sorted(known_configs)}'
    )
  devices = np.asarray(jax.devices()[:tp_size]).reshape(1, tp_size)
  mesh = jax.sharding.Mesh(devices, ('fsdp', 'tp'))
  perf_rows: list[PerfScalingRow] = []
  loo_workloads = set(leave_one_out_workloads)

  with mesh:
    model_cache: dict[
        tuple[int, bool], tuple[qwen3_stock.Qwen3, qwen3_canon.Qwen3Canon]
    ] = {}
    for w in workloads:
      config = build_model_config(preset, tp_size=tp_size)
      config.num_layers = w.num_layers
      if w.use_remat:
        config.remat_config = qwen3_stock.RematConfig.BLOCK

      cache_key = (w.num_layers, w.use_remat)
      if cache_key not in model_cache:
        stock_model = qwen3_stock.Qwen3(config, rngs=nnx.Rngs(params=0))
        qwen3_canon.init_non_zero_mlp_weights(stock_model, seed=123)

        canon_model = qwen3_canon.Qwen3Canon(
            config,
            rngs=nnx.Rngs(params=0),
            canon_config=qwen3_canon.CanonKernelConfig.all_enabled(),
        )
        qwen3_canon.copy_weights(stock_model, canon_model)
        model_cache[cache_key] = (stock_model, canon_model)
      else:
        stock_model, canon_model = model_cache[cache_key]

      key = jax.random.PRNGKey(2026)
      prompt_tokens = jax.random.randint(
          key,
          (w.batch_size, w.prompt_len),
          0,
          config.vocab_size,
          dtype=jnp.int32,
      )
      total_len = w.prompt_len + w.max_new_tokens
      full_tokens = jax.random.randint(
          key, (w.batch_size, total_len), 0, config.vocab_size, dtype=jnp.int32
      )
      total_tokens = float(w.batch_size * total_len)

      # (config name, category, model, canon config | None, dump tag | None).
      stock_name, disabled_name, enabled_name = _BASELINE_PERF_CONFIG_NAMES
      specs: list[
          tuple[
              str,
              str,
              qwen3_stock.Qwen3 | qwen3_canon.Qwen3Canon,
              qwen3_canon.CanonKernelConfig | None,
              str | None,
          ]
      ] = [
          (stock_name, 'baseline', stock_model, None, 'stock'),
          (
              disabled_name,
              'baseline',
              canon_model,
              qwen3_canon.CanonKernelConfig.all_disabled(),
              None,
          ),
          (
              enabled_name,
              'baseline',
              canon_model,
              qwen3_canon.CanonKernelConfig.all_enabled(),
              'canon',
          ),
      ]
      if w.label in loo_workloads:
        specs.extend(
            (
                f'without_{switch_name}',
                'leave_one_out',
                canon_model,
                qwen3_canon.CanonKernelConfig.without(switch_name),
                None,
            )
            for switch_name in leave_one_out_switches
        )

      for cfg_name, category, mdl, ccfg, dump_tag in specs:
        if selected_configs is not None and cfg_name not in selected_configs:
          continue
        if ccfg is not None and isinstance(mdl, qwen3_canon.Qwen3Canon):
          mdl.set_canon_config(ccfg)
        use_cls = ccfg is not None and ccfg.use_canon_logsoftmax

        _, sample_ms, sample_mem = _time_fn(
            qwen3_canon.forward_sample_with_logprobs,
            mdl,
            prompt_tokens,
            max_new_tokens=w.max_new_tokens,
            cache_size=w.cache_size,
            use_canon_logsoftmax=use_cls,
            num_iters=num_iters,
        )
        _, prefill_ms, prefill_mem = _time_fn(
            qwen3_canon.score_sequence_prefill,
            mdl,
            full_tokens,
            use_canon_logsoftmax=use_cls,
            num_iters=num_iters,
        )
        _, fwdbwd_ms, fwdbwd_mem = _time_fn(
            qwen3_canon.score_sequence_fwd_bwd,
            mdl,
            full_tokens,
            use_canon_logsoftmax=use_cls,
            num_iters=num_iters,
        )
        prefill_tps = (
            total_tokens / (prefill_ms / 1000.0) if prefill_ms > 0 else 0.0
        )
        fwdbwd_tps = (
            total_tokens / (fwdbwd_ms / 1000.0) if fwdbwd_ms > 0 else 0.0
        )

        if dump_tag is not None:
          remat_tag = '_block' if w.use_remat else ''
          label = (
              f'{dump_tag}_L{w.num_layers}_B{w.batch_size}_T{total_len}'
              f'{remat_tag}'
          )
          _dump_compiled_program(
              qwen3_canon.forward_sample_with_logprobs,
              f'{label}_sample',
              mdl,
              prompt_tokens,
              max_new_tokens=w.max_new_tokens,
              cache_size=w.cache_size,
              use_canon_logsoftmax=use_cls,
          )
          _dump_compiled_program(
              qwen3_canon.score_sequence_prefill,
              f'{label}_prefill',
              mdl,
              full_tokens,
              use_canon_logsoftmax=use_cls,
          )
          _dump_compiled_program(
              qwen3_canon.score_sequence_fwd_bwd,
              f'{label}_fwdbwd',
              mdl,
              full_tokens,
              use_canon_logsoftmax=use_cls,
          )

        perf_rows.append(
            PerfScalingRow(
                workload=w.label,
                config_name=cfg_name,
                category=category,
                sample_latency_ms=sample_ms,
                prefill_latency_ms=prefill_ms,
                prefill_tok_per_s=prefill_tps,
                fwdbwd_latency_ms=fwdbwd_ms,
                fwdbwd_tok_per_s=fwdbwd_tps,
                sample_arg_hbm_mb=sample_mem.arg_hbm_mb,
                sample_temp_hbm_mb=sample_mem.temp_hbm_mb,
                sample_total_hbm_mb=sample_mem.total_hbm_mb,
                prefill_arg_hbm_mb=prefill_mem.arg_hbm_mb,
                prefill_temp_hbm_mb=prefill_mem.temp_hbm_mb,
                prefill_total_hbm_mb=prefill_mem.total_hbm_mb,
                fwdbwd_arg_hbm_mb=fwdbwd_mem.arg_hbm_mb,
                fwdbwd_temp_hbm_mb=fwdbwd_mem.temp_hbm_mb,
                fwdbwd_total_hbm_mb=fwdbwd_mem.total_hbm_mb,
                device_peak_hbm_mb=fwdbwd_mem.device_peak_hbm_mb,
                sample_arena_hbm_mb=sample_mem.arena_hbm_mb,
                sample_code_hbm_mb=sample_mem.code_hbm_mb,
                prefill_arena_hbm_mb=prefill_mem.arena_hbm_mb,
                prefill_code_hbm_mb=prefill_mem.code_hbm_mb,
                fwdbwd_arena_hbm_mb=fwdbwd_mem.arena_hbm_mb,
                fwdbwd_code_hbm_mb=fwdbwd_mem.code_hbm_mb,
            )
        )

  return perf_rows


def format_performance_scaling_report(rows: Sequence[PerfScalingRow]) -> str:
  """Formats `PerfScalingRow` into a Markdown table.

  The `Arg/Temp/Tot` columns are the legacy PJRT numbers (`Temp` double counts
  TPU fragmentation, see `MemoryProfile`); `Arena/Code` are the
  buffer-assignment based HLO temp arena and generated kernel code, which are
  the columns to compare across runs (their sum is the TPU program HBM
  requirement on top of the parameters).

  Args:
    rows: The rows returned by `run_performance_scaling_benchmark`.

  Returns:
    The Markdown table.
  """
  lines = [
      (
          '| Workload | Configuration | Category | Sample (ms) |'
          ' Prefill (ms / tok/s) | Fwd+Bwd (ms / tok/s) |'
          ' Sample HBM (Arg/Temp/Tot MB) | Prefill HBM (Arg/Temp/Tot MB) |'
          ' Fwd+Bwd HBM (Arg/Temp/Tot MB) |'
          ' Sample HBM (Arena/Code MB) |'
          ' Prefill HBM (Arena/Code MB) |'
          ' Fwd+Bwd HBM (Arena/Code MB) | Dev Peak HBM (MB) |'
      ),
      (
          '| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |'
          ' :---: | :---: | :---: | :---: | :---: |'
      ),
  ]
  for r in rows:
    lines.append(
        f'| `{r.workload}` | `{r.config_name}` | `{r.category}` | '
        f'`{r.sample_latency_ms:.2f}` | '
        f'`{r.prefill_latency_ms:.2f}` ({r.prefill_tok_per_s:,.0f}) | '
        f'`{r.fwdbwd_latency_ms:.2f}` ({r.fwdbwd_tok_per_s:,.0f}) | '
        f'`{r.sample_arg_hbm_mb:.2f} / {r.sample_temp_hbm_mb:.2f} /'
        f' {r.sample_total_hbm_mb:.2f}` | '
        f'`{r.prefill_arg_hbm_mb:.2f} / {r.prefill_temp_hbm_mb:.2f} /'
        f' {r.prefill_total_hbm_mb:.2f}` | '
        f'`{r.fwdbwd_arg_hbm_mb:.2f} / {r.fwdbwd_temp_hbm_mb:.2f} /'
        f' {r.fwdbwd_total_hbm_mb:.2f}` | '
        f'`{r.sample_arena_hbm_mb:.2f} / {r.sample_code_hbm_mb:.2f}` | '
        f'`{r.prefill_arena_hbm_mb:.2f} / {r.prefill_code_hbm_mb:.2f}` | '
        f'`{r.fwdbwd_arena_hbm_mb:.2f} / {r.fwdbwd_code_hbm_mb:.2f}` | '
        f'`{r.device_peak_hbm_mb:.2f}` |'
    )
  return '\n'.join(lines)


@dataclasses.dataclass(slots=True)
class DriftRegimeRow:
  """Diagnostic row comparing a specific TIM drift regime across models."""

  regime: str
  config_name: str
  max_abs_diff: float
  mean_abs_diff: float
  exact_match_pct: float


def run_drift_regime_diagnostics(
    *,
    preset: str = 'qwen3_0p6b_2l',
    batch_size: int = 4,
    prompt_len: int = 48,
    max_new_tokens: int = 16,
    cache_size: int = 128,
    tp_size: int = 4,
    configs: Sequence[str] | None = None,
    include_deep_stack: bool = True,
) -> list[DriftRegimeRow]:
  """Measures Decode-vs-Prefill, B=1-vs-B=N, and Fwd-vs-Fwd+Bwd sub-regimes.

  The regime labels carry the actual shapes (`N = batch_size`,
  `T = prompt_len + max_new_tokens`, `L = num_layers` of the preset); their
  numbered prefixes `1a.` / `1b.` / `1c.` / `2.` / `3a.` (and `3b.` for the
  deep stack) are stable and are what reports and tests key on.

  Args:
    preset: `build_model_config` preset name.
    batch_size: Batch size `N` of the B=1-vs-B=N regimes.
    prompt_len: Prompt length of the sampled sequences.
    max_new_tokens: Decode steps per sampled sequence.
    cache_size: KV-cache length of the sampler.
    tp_size: Tensor-parallel mesh size.
    configs: Configuration names of the model rows (regimes 1a-3a); `None` runs
      all four.
    include_deep_stack: Also run the model-free 24-layer MLP stack regime 3b.

  Returns:
    One `DriftRegimeRow` per (regime, configuration).
  """
  devices = np.asarray(jax.devices()[:tp_size]).reshape(1, tp_size)
  mesh = jax.sharding.Mesh(devices, ('fsdp', 'tp'))
  diag_rows: list[DriftRegimeRow] = []

  with mesh:
    config = build_model_config(preset, tp_size=tp_size)
    stock_model = qwen3_stock.Qwen3(config, rngs=nnx.Rngs(params=0))
    qwen3_canon.init_non_zero_mlp_weights(stock_model, seed=123)

    canon_model = qwen3_canon.Qwen3Canon(
        config,
        rngs=nnx.Rngs(params=0),
        canon_config=qwen3_canon.CanonKernelConfig.all_enabled(),
    )
    qwen3_canon.copy_weights(stock_model, canon_model)

    key = jax.random.PRNGKey(2026)
    prompt_tokens = jax.random.randint(
        key, (batch_size, prompt_len), 0, config.vocab_size, dtype=jnp.int32
    )

    model_specs = [
        ('stock_qwen3 (model.py)', stock_model, None, False),
        (
            'canon_all_disabled',
            canon_model,
            qwen3_canon.CanonKernelConfig.all_disabled(),
            False,
        ),
        (
            'only_use_fixed_order_reduce',
            canon_model,
            qwen3_canon.CanonKernelConfig.only('use_fixed_order_reduce'),
            False,
        ),
        (
            'canon_all_enabled (Zero-TIM)',
            canon_model,
            qwen3_canon.CanonKernelConfig.all_enabled(),
            True,
        ),
    ]
    if configs is not None:
      known = {spec[0] for spec in model_specs}
      unknown = set(configs) - known
      if unknown:
        raise ValueError(
            f'Unknown drift regime configs {sorted(unknown)}; known: '
            f'{sorted(known)}'
        )
      model_specs = [spec for spec in model_specs if spec[0] in set(configs)]

    n = batch_size
    t = prompt_len + max_new_tokens
    same_bn = f'Same B={n}==TP' if n == tp_size else f'Same B={n}'

    for label, mdl, ccfg, use_cls in model_specs:
      if ccfg is not None and isinstance(mdl, qwen3_canon.Qwen3Canon):
        mdl.set_canon_config(ccfg)

      # (A) B=1 KV-cached Decode vs B=1 Cacheless Prefill (pure T=1 vs T=t
      #     at B=1).
      b1_full, b1_dec_lps, _ = qwen3_canon.forward_sample_with_logprobs(
          mdl,
          prompt_tokens[:1],
          max_new_tokens=max_new_tokens,
          cache_size=cache_size,
          use_canon_logsoftmax=use_cls,
      )
      b1_pref_all = qwen3_canon.score_sequence_prefill(
          mdl, b1_full, use_canon_logsoftmax=use_cls
      )
      b1_pref_dec_lps = np.asarray(
          b1_pref_all[:, prompt_len - 1 :], dtype=np.float32
      )
      mx, mn, ex = _diff_stats(np.asarray(b1_dec_lps), b1_pref_dec_lps)
      diag_rows.append(
          DriftRegimeRow(
              f'1a. Decode(B=1,T=1) vs Prefill(B=1,T={t}) [Same B=1]',
              label,
              mx,
              mn,
              ex,
          )
      )

      # (B) B=1 KV-cached Decode vs B=N Cacheless Prefill (Rollout B=1 vs
      #     Learner B=N).
      bn_full, bn_dec_lps, _ = qwen3_canon.forward_sample_with_logprobs(
          mdl,
          prompt_tokens,
          max_new_tokens=max_new_tokens,
          cache_size=cache_size,
          use_canon_logsoftmax=use_cls,
      )
      # Put b1_full[0] into slot 0 of a B=N batch so the sequence-0 tokens are
      # identical.
      bn_with_b1_seq0 = bn_full.at[0].set(b1_full[0])
      bn_pref_all = qwen3_canon.score_sequence_prefill(
          mdl, bn_with_b1_seq0, use_canon_logsoftmax=use_cls
      )
      bn_pref_seq0_dec_lps = np.asarray(
          bn_pref_all[0:1, prompt_len - 1 :], dtype=np.float32
      )
      mx, mn, ex = _diff_stats(np.asarray(b1_dec_lps), bn_pref_seq0_dec_lps)
      diag_rows.append(
          DriftRegimeRow(
              f'1b. Decode(B=1,T=1) vs Prefill(B={n},T={t})'
              ' [Rollout vs Learner]',
              label,
              mx,
              mn,
              ex,
          )
      )

      # (C) B=N KV-cached Decode vs B=N Cacheless Prefill (Same B=N).
      bn_own_pref_all = qwen3_canon.score_sequence_prefill(
          mdl, bn_full, use_canon_logsoftmax=use_cls
      )
      bn_own_pref_dec = np.asarray(
          bn_own_pref_all[:, prompt_len - 1 :], dtype=np.float32
      )
      mx, mn, ex = _diff_stats(np.asarray(bn_dec_lps), bn_own_pref_dec)
      diag_rows.append(
          DriftRegimeRow(
              f'1c. Decode(B={n},T=1) vs Prefill(B={n},T={t}) [{same_bn}]',
              label,
              mx,
              mn,
              ex,
          )
      )

      # (D) Prefill(B=1, T=t) vs Prefill(B=N, T=t)
      mx, mn, ex = _diff_stats(
          np.asarray(b1_pref_all[0]), np.asarray(bn_pref_all[0])
      )
      diag_rows.append(
          DriftRegimeRow(
              f'2. Prefill(B=1,T={t}) vs Prefill(B={n},T={t})'
              ' [Batch Invariance]',
              label,
              mx,
              mn,
              ex,
          )
      )

      # (E) Fwd vs Fwd+Bwd at the preset's depth.
      fwdbwd_lps, _ = qwen3_canon.score_sequence_fwd_bwd(
          mdl, bn_with_b1_seq0, use_canon_logsoftmax=use_cls
      )
      mx, mn, ex = _diff_stats(np.asarray(bn_pref_all), np.asarray(fwdbwd_lps))
      diag_rows.append(
          DriftRegimeRow(
              f'3a. Fwd vs Fwd+Bwd (L={config.num_layers}, 3D [B={n},T={t},D])',
              label,
              mx,
              mn,
              ex,
          )
      )

    if not include_deep_stack:
      return diag_rows

    # (F) 24-layer 2D [T=256, D=512] TP=4 Transformer MLP+RMSNorm stack:
    #     XLA native 4-way all-reduce vs the Zero-TIM fixed-order reduction
    #     across jit(fwd) vs value_and_grad.
    pspec = jax.sharding.PartitionSpec
    mesh_1d = jax.sharding.Mesh(np.asarray(jax.devices()[:tp_size]), ('tp',))
    rng = np.random.default_rng(0)
    d_dim, f_dim, t_rows = 512, 2048, 256
    put = lambda a, s: jax.device_put(
        jnp.asarray(a, jnp.bfloat16), jax.sharding.NamedSharding(mesh_1d, s)
    )
    w_24 = [
        dict(
            g=put(rng.normal(size=(d_dim,)) * 0.1 + 1.0, pspec(None)),
            Wg=put(rng.normal(size=(d_dim, f_dim)) * 0.02, pspec(None, 'tp')),
            Wu=put(rng.normal(size=(d_dim, f_dim)) * 0.02, pspec(None, 'tp')),
            Wd=put(rng.normal(size=(f_dim, d_dim)) * 0.02, pspec('tp', None)),
        )
        for _ in range(24)
    ]
    x0_24 = put(rng.normal(size=(t_rows, d_dim)) * 0.5, pspec(None, None))

    def _rms(x, g):
      xf = x.astype(jnp.float32)
      return (
          (xf * jax.lax.rsqrt(jnp.mean(xf * xf, -1, keepdims=True) + 1e-6))
          * g.astype(jnp.float32)
      ).astype(jnp.bfloat16)

    for fixed_flag, tag in (
        (False, 'canon_all_disabled (XLA 4-way all-reduce, L=24)'),
        (True, 'use_fixed_order_reduce (Zero-TIM F4 tree, L=24)'),
    ):

      def _make_fwd(use_fixed: bool):
        def _fwd(x, w_list):
          for p in w_list:
            h = _rms(x, p['g'])
            gate = jax.nn.silu(
                jnp.einsum('td,df->tf', h, p['Wg']).astype(jnp.float32)
            ).astype(jnp.bfloat16)
            act = gate * jnp.einsum('td,df->tf', h, p['Wu'])
            if use_fixed:

              def _loc(a_loc, wd_loc):
                part = jnp.dot(a_loc, wd_loc).astype(jnp.bfloat16)
                return qwen3_canon._fixed_order_tp_sum_with_vjp(  # pylint: disable=protected-access
                    part,
                    'tp',
                    tp_size,
                    mode=qwen3_canon.fixed_order_reduce.RING,
                )

              m_out = jax.shard_map(
                  _loc,
                  mesh=mesh_1d,
                  in_specs=(pspec(None, 'tp'), pspec('tp', None)),
                  out_specs=pspec(None, None),
                  check_vma=False,
              )(act, p['Wd'])
            else:
              m_out = jnp.einsum('tf,fd->td', act, p['Wd'])
            x = x + m_out
          return x

        return _fwd

      fwd_fn = _make_fwd(fixed_flag)
      out_fwd = np.asarray(jax.device_get(jax.jit(fwd_fn)(x0_24, w_24)))

      def _loss_24(x, w_list, fwd=fwd_fn):
        y = fwd(x, w_list)
        return jnp.sum(y.astype(jnp.float32)), y

      (_, out_bwd_dev), _ = jax.jit(
          jax.value_and_grad(_loss_24, argnums=0, has_aux=True)
      )(x0_24, w_24)
      out_bwd = np.asarray(jax.device_get(out_bwd_dev))
      mx, mn, ex = _diff_stats(out_fwd, out_bwd)
      diag_rows.append(
          DriftRegimeRow(
              '3b. Fwd vs Fwd+Bwd (L=24 deep stack, 2D [T=256,D=512])',
              tag,
              mx,
              mn,
              ex,
          )
      )

  return diag_rows


def format_drift_regime_report(rows: Sequence[DriftRegimeRow]) -> str:
  """Formats `DriftRegimeRow` diagnostics into a Markdown table."""
  lines = [
      (
          '| Drift Regime | Configuration | Max Abs Diff |'
          ' Mean Abs Diff | Bitwise Exact (%) |'
      ),
      '| :--- | :--- | :---: | :---: | :---: |',
  ]
  for r in rows:
    lines.append(
        f'| `{r.regime}` | `{r.config_name}` | `{r.max_abs_diff:.3e}` | '
        f'`{r.mean_abs_diff:.3e}` | `{r.exact_match_pct:.1f}%` |'
    )
  return '\n'.join(lines)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  rows = run_benchmark_suite(
      preset=_MODEL_PRESET.value,
      batch_size=_BATCH_SIZE.value,
      prompt_len=_PROMPT_LEN.value,
      max_new_tokens=_MAX_NEW_TOKENS.value,
      cache_size=_CACHE_SIZE.value,
      num_iters=_NUM_ITERS.value,
      run_ablations=_RUN_ABLATIONS.value,
      include_fwdbwd=_INCLUDE_FWDBWD.value,
      tp_size=_TP_SIZE.value,
  )
  report = format_markdown_report(rows)
  print('\n=== Qwen3 Stock vs Qwen3Canon Benchmark & Ablation Report ===\n')
  print(report)

  if _OUTPUT_JSON.value:
    with open(_OUTPUT_JSON.value, 'w', encoding='utf-8') as f:
      json.dump([dataclasses.asdict(r) for r in rows], f, indent=2)


if __name__ == '__main__':
  app.run(main)
