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
import dataclasses
import json
import time
from typing import Any

from absl import app
from absl import flags
from flax import nnx
import jax
from jax import numpy as jnp
import numpy as np
from tunix.experimental.zero_tim_kernel.bench.qwen3 import qwen3_model_canon as qwen3_canon
from tunix.models.qwen3 import model as qwen3_stock

_MODEL_PRESET = flags.DEFINE_enum(
    'model_preset',
    'mini_qwen3',
    ['mini_qwen3', 'qwen3_0p6b_2l', 'qwen3_0p6b'],
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
    # VOCAB_TILE=1024 x 2, bkv_csz=16) on CPU interpret mode or TPU.
    return qwen3_stock.ModelConfig(
        num_layers=2,
        vocab_size=2048,
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
  if preset == 'qwen3_0p6b':
    cfg = qwen3_stock.ModelConfig.qwen3_0p6b()
    cfg.shd_config = shd_cfg
    cfg.dtype = bf16_dtype
    cfg.param_dtype = bf16_dtype
    return cfg
  raise ValueError(f'Unknown preset: {preset}')


def _time_fn(fn, *args, num_iters: int = 5, **kwargs) -> tuple[Any, float]:
  """Runs 1 warmup call (including compilation) and `num_iters` timed calls."""
  out = fn(*args, **kwargs)
  jax.block_until_ready(out)
  times_ms = []
  for _ in range(num_iters):
    t0 = time.perf_counter()
    res = fn(*args, **kwargs)
    jax.block_until_ready(res)
    times_ms.append((time.perf_counter() - t0) * 1000.0)
  return out, float(np.median(times_ms))


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
  """Evaluates one model/kernel configuration on numerics, TIM, and latency."""
  # 1. Forward sampling with KV cache (Path A: prefill + decode steps)
  (full_tokens, decode_step_lps, _), sample_ms = _time_fn(
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
  own_prefill_all_lps, prefill_ms = _time_fn(
      qwen3_canon.score_sequence_prefill,
      model,
      full_tokens,
      use_canon_logsoftmax=use_canon_logsoftmax,
      num_iters=num_iters,
  )
  # Slice generated token positions [l_prompt - 1 : l_prompt - 1 + max_new_tokens]
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
    (fwdbwd_lps, _), fwdbwd_ms = _time_fn(
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

  # Generate reference trajectory and logprobs with Full Qwen3Canon and Stock Qwen3
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

  # 1. Stock Qwen3 (`tunix.models.qwen3.model`)
  rows.append(
      evaluate_configuration(
          'stock_qwen3 (model.py)',
          'baseline',
          stock_model,
          prompt_tokens=prompt_tokens,
          ref_full_tokens=ref_full_tokens,
          stock_prefill_lps=stock_prefill_lps,
          canon_full_prefill_lps=canon_full_prefill_lps,
          canon_full_sampled_tokens=canon_full_sampled_tokens,
          max_new_tokens=max_new_tokens,
          cache_size=cache_size,
          use_canon_logsoftmax=False,
          num_iters=num_iters,
          include_fwdbwd=include_fwdbwd,
      )
  )

  # 2. Qwen3Canon with all switches disabled (verifies 0.0 diff vs stock Qwen3)
  cfg_off = qwen3_canon.CanonKernelConfig.all_disabled()
  canon_model.set_canon_config(cfg_off)
  rows.append(
      evaluate_configuration(
          'canon_all_disabled',
          'baseline',
          canon_model,
          prompt_tokens=prompt_tokens,
          ref_full_tokens=ref_full_tokens,
          stock_prefill_lps=stock_prefill_lps,
          canon_full_prefill_lps=canon_full_prefill_lps,
          canon_full_sampled_tokens=canon_full_sampled_tokens,
          max_new_tokens=max_new_tokens,
          cache_size=cache_size,
          use_canon_logsoftmax=False,
          num_iters=num_iters,
          include_fwdbwd=include_fwdbwd,
      )
  )

  # 3. Qwen3Canon with all switches enabled (Full Zero-TIM)
  cfg_on = qwen3_canon.CanonKernelConfig.all_enabled()
  canon_model.set_canon_config(cfg_on)
  rows.append(
      evaluate_configuration(
          'canon_all_enabled (Zero-TIM)',
          'baseline',
          canon_model,
          prompt_tokens=prompt_tokens,
          ref_full_tokens=ref_full_tokens,
          stock_prefill_lps=stock_prefill_lps,
          canon_full_prefill_lps=canon_full_prefill_lps,
          canon_full_sampled_tokens=canon_full_sampled_tokens,
          max_new_tokens=max_new_tokens,
          cache_size=cache_size,
          use_canon_logsoftmax=True,
          num_iters=num_iters,
          include_fwdbwd=include_fwdbwd,
      )
  )

  if not run_ablations:
    return rows

  # 4. Leave-One-Out ablations: disable one kernel at a time from Full Zero-TIM
  for switch_name in switches:
    cfg_loo = qwen3_canon.CanonKernelConfig.without(switch_name)
    canon_model.set_canon_config(cfg_loo)
    rows.append(
        evaluate_configuration(
            f'without_{switch_name}',
            'leave_one_out',
            canon_model,
            prompt_tokens=prompt_tokens,
            ref_full_tokens=ref_full_tokens,
            stock_prefill_lps=stock_prefill_lps,
            canon_full_prefill_lps=canon_full_prefill_lps,
            canon_full_sampled_tokens=canon_full_sampled_tokens,
            max_new_tokens=max_new_tokens,
            cache_size=cache_size,
            use_canon_logsoftmax=cfg_loo.use_canon_logsoftmax,
            num_iters=num_iters,
            include_fwdbwd=include_fwdbwd,
        )
    )

  # 5. Add-One-In ablations: enable ONLY one kernel at a time on top of stock
  for switch_name in switches:
    cfg_aoi = qwen3_canon.CanonKernelConfig.only(switch_name)
    canon_model.set_canon_config(cfg_aoi)
    rows.append(
        evaluate_configuration(
            f'only_{switch_name}',
            'add_one_in',
            canon_model,
            prompt_tokens=prompt_tokens,
            ref_full_tokens=ref_full_tokens,
            stock_prefill_lps=stock_prefill_lps,
            canon_full_prefill_lps=canon_full_prefill_lps,
            canon_full_sampled_tokens=canon_full_sampled_tokens,
            max_new_tokens=max_new_tokens,
            cache_size=cache_size,
            use_canon_logsoftmax=cfg_aoi.use_canon_logsoftmax,
            num_iters=num_iters,
            include_fwdbwd=include_fwdbwd,
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
    include_fwdbwd: bool = True,
    switches: Sequence[str] = qwen3_canon.ABLATION_SWITCH_NAMES,
    tp_size: int | None = None,
) -> list[BenchmarkRow]:
  """Runs the complete Stock vs Canon benchmark and per-kernel ablation suite."""
  if tp_size is None or tp_size <= 0:
    num_devs = len(jax.devices())
    if jax.default_backend() == 'tpu' and num_devs in (2, 4, 8):
      tp_size = num_devs
    else:
      tp_size = 1

  if tp_size > 1:
    devices = np.asarray(jax.devices()[:tp_size]).reshape(1, tp_size)
    mesh = jax.sharding.Mesh(devices, ('fsdp', 'tp'))
    with mesh:
      return _run_benchmark_suite_body(
          preset=preset,
          batch_size=batch_size,
          prompt_len=prompt_len,
          max_new_tokens=max_new_tokens,
          cache_size=cache_size,
          num_iters=num_iters,
          run_ablations=run_ablations,
          include_fwdbwd=include_fwdbwd,
          switches=switches,
          tp_size=tp_size,
      )
  return _run_benchmark_suite_body(
      preset=preset,
      batch_size=batch_size,
      prompt_len=prompt_len,
      max_new_tokens=max_new_tokens,
      cache_size=cache_size,
      num_iters=num_iters,
      run_ablations=run_ablations,
      include_fwdbwd=include_fwdbwd,
      switches=switches,
      tp_size=1,
  )


def format_markdown_report(rows: Sequence[BenchmarkRow]) -> str:
  """Formats benchmark & ablation rows into a Markdown table."""
  lines = [
      (
          '| Configuration | Category | `|LP - Stock|` Max |'
          ' `|LP - Canon|` Max | `|Decode - Prefill|` Max (Exact %) |'
          ' `|B=1 - B=N|` Max (Exact %) | `|Fwd - FwdBwd|` Max (Exact %) |'
          ' Sample (ms) | Prefill (ms) | Fwd+Bwd (ms) |'
      ),
      (
          '| :--- | :--- | :---: | :---: | :---: | :---: | :---: |'
          ' :---: | :---: | :---: |'
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
        f'`{r.fwdbwd_latency_ms:.2f}` |'
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
) -> list[DriftRegimeRow]:
  """Measures Decode-vs-Prefill, B=1-vs-B=4, and Fwd-vs-Fwd+Bwd sub-regimes."""
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

    for label, mdl, ccfg, use_cls in model_specs:
      if ccfg is not None and isinstance(mdl, qwen3_canon.Qwen3Canon):
        mdl.set_canon_config(ccfg)

      # (A) B=1 KV-cached Decode vs B=1 Cacheless Prefill (pure T=1 vs T=64 at B=1)
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
              '1a. Decode(B=1,T=1) vs Prefill(B=1,T=64) [Same B=1]',
              label,
              mx,
              mn,
              ex,
          )
      )

      # (B) B=1 KV-cached Decode vs B=4 Cacheless Prefill (Rollout B=1 vs Learner B=4)
      b4_full, b4_dec_lps, _ = qwen3_canon.forward_sample_with_logprobs(
          mdl,
          prompt_tokens,
          max_new_tokens=max_new_tokens,
          cache_size=cache_size,
          use_canon_logsoftmax=use_cls,
      )
      # Put b1_full[0] into slot 0 of a B=4 batch so sequence 0 tokens are identical
      b4_with_b1_seq0 = b4_full.at[0].set(b1_full[0])
      b4_pref_all = qwen3_canon.score_sequence_prefill(
          mdl, b4_with_b1_seq0, use_canon_logsoftmax=use_cls
      )
      b4_pref_seq0_dec_lps = np.asarray(
          b4_pref_all[0:1, prompt_len - 1 :], dtype=np.float32
      )
      mx, mn, ex = _diff_stats(np.asarray(b1_dec_lps), b4_pref_seq0_dec_lps)
      diag_rows.append(
          DriftRegimeRow(
              '1b. Decode(B=1,T=1) vs Prefill(B=4,T=64) [Rollout vs Learner]',
              label,
              mx,
              mn,
              ex,
          )
      )

      # (C) B=4 KV-cached Decode vs B=4 Cacheless Prefill (Same B=4==TP)
      b4_own_pref_all = qwen3_canon.score_sequence_prefill(
          mdl, b4_full, use_canon_logsoftmax=use_cls
      )
      b4_own_pref_dec = np.asarray(
          b4_own_pref_all[:, prompt_len - 1 :], dtype=np.float32
      )
      mx, mn, ex = _diff_stats(np.asarray(b4_dec_lps), b4_own_pref_dec)
      diag_rows.append(
          DriftRegimeRow(
              '1c. Decode(B=4,T=1) vs Prefill(B=4,T=64) [Same B=4==TP]',
              label,
              mx,
              mn,
              ex,
          )
      )

      # (D) Prefill(B=1, T=64) vs Prefill(B=4, T=64)
      mx, mn, ex = _diff_stats(
          np.asarray(b1_pref_all[0]), np.asarray(b4_pref_all[0])
      )
      diag_rows.append(
          DriftRegimeRow(
              '2. Prefill(B=1,T=64) vs Prefill(B=4,T=64) [Batch Invariance]',
              label,
              mx,
              mn,
              ex,
          )
      )

      # (E) Fwd vs Fwd+Bwd at L=2
      fwdbwd_lps, _ = qwen3_canon.score_sequence_fwd_bwd(
          mdl, b4_with_b1_seq0, use_canon_logsoftmax=use_cls
      )
      mx, mn, ex = _diff_stats(np.asarray(b4_pref_all), np.asarray(fwdbwd_lps))
      diag_rows.append(
          DriftRegimeRow(
              '3a. Fwd vs Fwd+Bwd (L=2, 3D [B=4,T=64,D])',
              label,
              mx,
              mn,
              ex,
          )
      )

    # (F) 24-layer 2D [T=256, D=512] TP=4 Transformer MLP+RMSNorm stack
    #     (reproducing `p19_minrepro_f4.py`: XLA native 4-way all-reduce vs
    #     Zero-TIM F4 fixed-order reduction across jit(fwd) vs value_and_grad)
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
                return qwen3_canon._tp_sum_with_replicated_bwd(  # pylint: disable=protected-access
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

      def _loss_24(x, w_list, _f=fwd_fn):
        y = _f(x, w_list)
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
