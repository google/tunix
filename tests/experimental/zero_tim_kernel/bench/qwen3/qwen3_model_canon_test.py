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

"""Tests for `qwen3_model_canon` and `qwen3_canon_benchmark`."""

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
from jax import numpy as jnp
import numpy as np
from tunix.experimental.zero_tim_kernel import test_utils
from tunix.experimental.zero_tim_kernel.bench.qwen3 import qwen3_canon_benchmark
from tunix.experimental.zero_tim_kernel.bench.qwen3 import qwen3_model_canon as qwen3_canon
from tunix.models.qwen3 import model as qwen3_stock


def setUpModule():
  test_utils.configure_cpu(num_devices=4)


def _make_test_config(
    *, use_tied_embedding: bool = True, tp_size: int = 1
) -> qwen3_stock.ModelConfig:
  shd_cfg = (
      qwen3_canon.get_tp_sharding_config('tp')
      if tp_size > 1
      else qwen3_stock.ShardingConfig.get_default_sharding(is_sampling=True)
  )
  return qwen3_stock.ModelConfig(
      num_layers=2,
      vocab_size=1024 if tp_size == 1 else 2048,
      embed_dim=256,
      hidden_dim=512,
      num_heads=max(4, 2 * tp_size),
      head_dim=128,
      num_kv_heads=max(2, tp_size),
      norm_eps=1e-6,
      rope_theta=1_000_000,
      use_tied_embedding=use_tied_embedding,
      shd_config=shd_cfg,
      dtype=jnp.dtype(jnp.bfloat16),
      param_dtype=jnp.dtype(jnp.bfloat16),
  )


class Qwen3ModelCanonTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('tied_embedding', True),
      ('untied_embedding', False),
  )
  def test_all_disabled_matches_stock_qwen3_bitwise(
      self, use_tied_embedding: bool
  ):
    cfg = _make_test_config(use_tied_embedding=use_tied_embedding)
    stock = qwen3_stock.Qwen3(cfg, rngs=nnx.Rngs(params=0))
    qwen3_canon.init_non_zero_mlp_weights(stock, seed=7)

    canon = qwen3_canon.Qwen3Canon(
        cfg,
        rngs=nnx.Rngs(params=0),
        canon_config=qwen3_canon.CanonKernelConfig.all_disabled(),
    )
    qwen3_canon.copy_weights(stock, canon)

    tokens = jax.random.randint(
        jax.random.PRNGKey(1), (3, 7), 0, cfg.vocab_size, dtype=jnp.int32
    )
    stock_lps = qwen3_canon.score_sequence_prefill(
        stock, tokens, use_canon_logsoftmax=False
    )
    canon_lps = qwen3_canon.score_sequence_prefill(
        canon, tokens, use_canon_logsoftmax=False
    )
    np.testing.assert_array_equal(np.asarray(stock_lps), np.asarray(canon_lps))

  def test_zero_tim_decode_vs_prefill_and_batch_and_fwdbwd_bitwise(self):
    cfg = _make_test_config(use_tied_embedding=True)
    canon = qwen3_canon.Qwen3Canon(
        cfg,
        rngs=nnx.Rngs(params=0),
        canon_config=qwen3_canon.CanonKernelConfig.all_enabled(),
    )
    qwen3_canon.init_non_zero_mlp_weights(canon, seed=42)

    prompt_tokens = jax.random.randint(
        jax.random.PRNGKey(99), (4, 5), 0, cfg.vocab_size, dtype=jnp.int32
    )
    max_new_tokens = 4
    cache_size = 32  # intentionally > 5 + 4 to test inactive KV blocks

    # Path A: KV-cached forward sampling
    full_tokens, decode_lps, prefill_prompt_lps = (
        qwen3_canon.forward_sample_with_logprobs(
            canon,
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            cache_size=cache_size,
            use_canon_logsoftmax=True,
        )
    )

    # Path B: Cacheless prefill over full_tokens [4, 9]
    prefill_all_lps = qwen3_canon.score_sequence_prefill(
        canon, full_tokens, use_canon_logsoftmax=True
    )
    # Prompt logprobs [0..3] and decode logprobs [4..7] must match bitwise!
    np.testing.assert_array_equal(
        np.asarray(prefill_prompt_lps),
        np.asarray(prefill_all_lps[:, :4]),
    )
    np.testing.assert_array_equal(
        np.asarray(decode_lps),
        np.asarray(prefill_all_lps[:, 4:]),
    )

    # Batch-size invariance: B=1 vs B=4 slice 0
    b1_lps = qwen3_canon.score_sequence_prefill(
        canon, full_tokens[:1], use_canon_logsoftmax=True
    )
    np.testing.assert_array_equal(
        np.asarray(b1_lps[0]),
        np.asarray(prefill_all_lps[0]),
    )

    # Path C: Forward+Backward inside value_and_grad must match Path B bitwise!
    fwdbwd_lps, grad_norm = qwen3_canon.score_sequence_fwd_bwd(
        canon, full_tokens, use_canon_logsoftmax=True
    )
    np.testing.assert_array_equal(
        np.asarray(fwdbwd_lps),
        np.asarray(prefill_all_lps),
    )
    self.assertTrue(np.isfinite(float(grad_norm)))
    self.assertGreater(float(grad_norm), 0.0)

  def test_benchmark_and_ablation_suite_runs(self):
    if jax.default_backend() == 'tpu' and len(jax.devices()) >= 4:
      rows = qwen3_canon_benchmark.run_benchmark_suite(
          preset='qwen3_0p6b_2l',
          batch_size=4,
          prompt_len=48,
          max_new_tokens=16,
          cache_size=128,
          num_iters=5,
          run_ablations=True,
          include_fwdbwd=True,
          tp_size=4,
      )
      report = qwen3_canon_benchmark.format_markdown_report(rows)
      print('\n=== REAL TPU (TP=4) BENCHMARK & ABLATION REPORT ===\n')
      print(report)
      print('\n=== END REAL TPU REPORT ===\n')
      self.assertLen(rows, 23)

      diag_rows = qwen3_canon_benchmark.run_drift_regime_diagnostics(
          preset='qwen3_0p6b_2l',
          batch_size=4,
          prompt_len=48,
          max_new_tokens=16,
          cache_size=128,
          tp_size=4,
      )
      diag_report = qwen3_canon_benchmark.format_drift_regime_report(diag_rows)
      print('\n=== REAL TPU (TP=4) DRIFT REGIME DIAGNOSTIC REPORT ===\n')
      print(diag_report)
      print('\n=== END DRIFT REGIME DIAGNOSTIC REPORT ===\n')
      for d in diag_rows:
        if (
            'canon_all_enabled' in d.config_name
            or 'Zero-TIM F4 tree' in d.config_name
        ):
          self.assertEqual(d.max_abs_diff, 0.0, msg=f'Failed 0 diff for {d}')
          self.assertEqual(d.exact_match_pct, 100.0, msg=f'Failed 100% for {d}')
    else:
      switches = (
          'use_canon_rmsnorm',
          'use_canon_attention',
          'use_fixed_order_reduce',
      )
      tp_size = 4 if len(jax.devices()) >= 4 else 1
      rows = qwen3_canon_benchmark.run_benchmark_suite(
          preset='mini_qwen3',
          batch_size=2,
          prompt_len=4,
          max_new_tokens=3,
          cache_size=16,
          num_iters=1,
          run_ablations=True,
          include_fwdbwd=True,
          switches=switches,
          tp_size=tp_size,
      )
      self.assertLen(rows, 9)

    by_name = {r.name: r for r in rows}

    # Verify all-disabled has 0 diff vs stock
    self.assertEqual(by_name['canon_all_disabled'].vs_stock_max_abs_diff, 0.0)

    # Verify full Zero-TIM achieves 0.0 Decode-vs-Prefill, B=1-vs-B=N, and Fwd-vs-FwdBwd diff
    full_row = by_name['canon_all_enabled (Zero-TIM)']
    self.assertEqual(full_row.decode_vs_prefill_max_diff, 0.0)
    self.assertEqual(full_row.decode_vs_prefill_exact_pct, 100.0)
    self.assertEqual(full_row.batch1_vs_batchn_max_diff, 0.0)
    self.assertEqual(full_row.batch1_vs_batchn_exact_pct, 100.0)
    self.assertEqual(full_row.fwd_vs_fwdbwd_max_diff, 0.0)
    self.assertEqual(full_row.fwd_vs_fwdbwd_exact_pct, 100.0)


if __name__ == '__main__':
  absltest.main()
