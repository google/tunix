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

"""The bench's attention programs on the real RPA v3 kernel (TPU, 4 devices).

`test_kernel_import` runs wherever tpu_inference is importable. The
measurements need `jax.default_backend() == 'tpu'` with at least 4 devices,
are env-gated and meant to run one at a time; each prints markdown tables.

* `ZERO_TIM_RPA_KERNEL=1` -> `test_rpa_kernel_invariance_tpu` (R1, kernel
  level; model-free random bf16 operands of the Qwen3-0.6B head geometry: 16
  query / 8 KV heads, head_dim 128).
    K1  The engine form (pinned blocks, forced mixed, 256-row bucket): the
        sampler's prefill-write and its 32 decode steps vs the cacheless
        scoring prefill, and the B=1 subset vs B=N, bitwise, over B in
        {1, 4, 8} x T in {256, 1024} x page_size in {16, 128}; one geometry
        re-run at TP=4 (the head-sharded `shard_map`). Asserted.
    K2  Leave-One-Out of the two call-site controls (`pin_block_sizes=False`,
        `force_mixed=False`, and `plain`: both off, the kernel's own decode /
        prefill split and block heuristic) on B=4 x T in {256, 1024}. Reported.
    K3  Cross-kernel on the same dense operands: RPA (engine form) vs the
        bench contract (`canon_attention` Pallas forward: bkv=512, exp2 online
        softmax in f32) and the f32 contract replica (`reference_kernel`).
        Reported. (No f32-accumulator RPA variant: the kernel's `out_dtype`
        must keep the query dtype's width, see `rpa_backend`.)
* `ZERO_TIM_RPA_BACKEND=1` -> `test_cross_engine_rpa_backend_tpu` (R2, model
  level; `Qwen3Canon` with `canon_attention.ATTENTION_FWD = RPA_FWD`): the
  drift-regime rows at T=256 and T=1024, the attention Leave-One-Out /
  Add-One-In suite rows, the TP=4 parameter-gradient gate vs stock
  (`CANON_VJP2_MAX_SEQS=4`), the perf rows (L=8: B=4 Sample / Prefill /
  Fwd+Bwd, B=32 Sample / Prefill -- the chunked VJP unrolls B sequences, so
  the B=32 learner program is not timed).
"""

import dataclasses
import functools
import os
import time
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax
from jax import numpy as jnp
import numpy as np
from tunix.experimental.zero_tim_kernel import rpa_diff_chunked
from tunix.experimental.zero_tim_kernel.bench.qwen3 import canon_attention
from tunix.experimental.zero_tim_kernel.bench.qwen3 import qwen3_canon_benchmark
from tunix.experimental.zero_tim_kernel.bench.qwen3 import qwen3_model_canon as qwen3_canon
from tunix.experimental.zero_tim_kernel.bench.qwen3 import rpa_backend
from tunix.models.qwen3 import model as qwen3_stock

R1_ENV = 'ZERO_TIM_RPA_KERNEL'
R2_ENV = 'ZERO_TIM_RPA_BACKEND'

# Qwen3-0.6B head geometry.
QH, KH, D = 16, 8, 128
SCALE = D**-0.5
DECODE_STEPS = 32
TP = 4

STOCK = 'stock_qwen3 (model.py)'
DISABLED = 'canon_all_disabled'
ENABLED = 'canon_all_enabled (Zero-TIM)'


def setUpModule():
  # Initialize the backend before tpu_inference is imported (its package
  # `__init__` appends to `XLA_FLAGS` / `LIBTPU_INIT_ARGS`).
  jax.devices()


def _skip_unless_measurement(test: absltest.TestCase, env: str) -> None:
  if os.environ.get(env) != '1':
    test.skipTest(f'set {env}=1 to run this measurement')
  if jax.default_backend() != 'tpu' or len(jax.devices()) < TP:
    test.skipTest(f'needs a TPU host with >= {TP} devices')


def _tp_mesh() -> jax.sharding.Mesh:
  devices = np.asarray(jax.devices()[:TP]).reshape(1, TP)
  return jax.sharding.Mesh(devices, ('fsdp', 'tp'))


def _operands(seed: int, b: int, t: int):
  kq, kk, kv = jax.random.split(jax.random.PRNGKey(seed), 3)
  q = jax.random.normal(kq, (b, t, QH, D), jnp.float32).astype(jnp.bfloat16)
  k = jax.random.normal(kk, (b, t, KH, D), jnp.float32).astype(jnp.bfloat16)
  v = jax.random.normal(kv, (b, t, KH, D), jnp.float32).astype(jnp.bfloat16)
  return q, k, v


def _timed(fn, *args, iters: int = 3):
  """`(fn(*args), median ms over iters)` after one warm-up call."""
  out = fn(*args)
  jax.block_until_ready(out)
  times = []
  for _ in range(iters):
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    times.append((time.perf_counter() - t0) * 1000.0)
  return out, float(np.median(times))


def _diff(got, ref) -> tuple[int, int, float]:
  """`(rows differing, rows compared, max abs diff)`; a row is `[H, D]`."""
  got = np.asarray(got, np.float32)
  ref = np.asarray(ref, np.float32)
  differs = np.any(got != ref, axis=(-1, -2))
  max_abs = float(np.max(np.abs(got - ref))) if differs.any() else 0.0
  return int(differs.sum()), int(differs.size), max_abs


@dataclasses.dataclass(frozen=True)
class _Row:
  """One (geometry, program) row of the R1 tables."""

  tag: str
  b: int
  t: int
  page_size: int
  program: str
  rows_differing: int
  rows_compared: int
  max_abs_diff: float
  ms: float

  def cells(self) -> str:
    return (
        f'| `{self.tag}` | {self.b} | {self.t} | {self.page_size} |'
        f' `{self.program}` | `{self.rows_differing}/{self.rows_compared}` |'
        f' `{self.max_abs_diff:.3e}` | `{self.ms:.3f}` |'
    )


_R1_HEADER = (
    '| Form | B | T | page | Program | Rows differing vs scoring | Max abs'
    ' diff | ms |\n| :--- | :-: | :-: | :-: | :--- | :-: | :-: | :-: |'
)


def _programs(tag, q, k, v, *, page_size, mesh=None, **knobs) -> list[_Row]:
  """Scoring prefill vs sampler prefill-write, decode steps and B=1 subset."""
  b, t = int(q.shape[0]), int(q.shape[1])
  prompt_len = t - DECODE_STEPS

  @jax.jit
  def scoring(q, k, v):
    return rpa_backend.attention_cacheless(
        q,
        k,
        v,
        scale=SCALE,
        mesh=mesh,
        page_size=page_size,
        differentiable=False,
        **knobs,
    )

  @jax.jit
  def cached(cache, q, k, v):
    return rpa_backend.attention_cached(
        cache,
        q,
        k,
        v,
        scale=SCALE,
        mesh=mesh,
        page_size=page_size,
        differentiable=False,
        **knobs,
    )

  ref, ms_scoring = _timed(scoring, q, k, v)
  cache = {
      'k': jnp.zeros((b, t, KH, D), q.dtype),
      'v': jnp.zeros((b, t, KH, D), q.dtype),
      'end_index': jnp.zeros((b,), jnp.int32),
  }
  (cache, prefill), ms_prefill = _timed(
      cached, cache, q[:, :prompt_len], k[:, :prompt_len], v[:, :prompt_len]
  )
  steps, step_ms = [], []
  for pos in range(prompt_len, t):
    s = slice(pos, pos + 1)
    (cache, out), ms = _timed(cached, cache, q[:, s], k[:, s], v[:, s])
    steps.append(out)
    step_ms.append(ms)
  decode = jnp.concatenate(steps, axis=1)
  one, ms_one = _timed(scoring, q[:1], k[:1], v[:1])
  ref_np = np.asarray(ref, np.float32)
  rows = []
  for program, got, want, ms in (
      (f'scoring prefill (T={t})', ref, ref, ms_scoring),
      (
          f'sampler prefill (T={prompt_len}, S={t})',
          prefill,
          ref_np[:, :prompt_len],
          ms_prefill,
      ),
      (
          f'decode x{DECODE_STEPS} (T=1, S={t})',
          decode,
          ref_np[:, prompt_len:],
          float(np.median(step_ms)),
      ),
      ('B=1 subset vs row 0', one[0], ref_np[0], ms_one),
  ):
    differing, compared, max_abs = _diff(got, want)
    rows.append(
        _Row(tag, b, t, page_size, program, differing, compared, max_abs, ms)
    )
  return rows


class RpaBackendTpuTest(absltest.TestCase):

  def test_kernel_import(self):
    kernel = rpa_backend.load_kernel()
    self.assertTrue(callable(kernel))
    self.assertIs(rpa_backend.resolve_kernel(), kernel)
    print(
        f'[rpa] kernel: {kernel!r} controls: {rpa_backend.describe()}',
        flush=True,
    )

  def test_rpa_kernel_invariance_tpu(self):
    _skip_unless_measurement(self, R1_ENV)
    rows: list[_Row] = []

    # K1: the engine form.
    for b in (1, 4, 8):
      for t in (256, 1024):
        for page_size in (16, 128):
          q, k, v = _operands(b * 1000 + t, b, t)
          rows += _programs('engine', q, k, v, page_size=page_size)
    k1_rows = list(rows)
    # K2: Leave-One-Out of the call-site controls, then both off (`plain`: the
    # decode steps take the kernel's decode pass and its decode block
    # heuristic, which the single-knob rows never reach together).
    for t in (256, 1024):
      q, k, v = _operands(4000 + t, 4, t)
      rows += _programs('no_pin', q, k, v, page_size=16, pin_block_sizes=False)
      rows += _programs(
          'no_force_mixed', q, k, v, page_size=16, force_mixed=False
      )
      rows += _programs(
          'plain',
          q,
          k,
          v,
          page_size=16,
          pin_block_sizes=False,
          force_mixed=False,
      )
    # K1 at TP=4: the head-sharded shard_map.
    mesh = _tp_mesh()
    q, k, v = _operands(4256, 4, 256)
    with mesh:
      tp_rows = _programs('engine_tp4', q, k, v, page_size=16, mesh=mesh)
      tp_scoring = jax.jit(
          functools.partial(
              rpa_backend.attention_cacheless,
              scale=SCALE,
              mesh=mesh,
              differentiable=False,
          )
      )(q, k, v)
    rows += tp_rows
    single = rpa_backend.attention_cacheless(
        q, k, v, scale=SCALE, differentiable=False
    )
    tp_vs_single = _diff(tp_scoring, single)
    print(
        '[rpa-r1] K1 / K2 / TP=4 rows\n'
        + _R1_HEADER
        + '\n'
        + '\n'.join(r.cells() for r in rows)
        + '\n\n[rpa-r1] TP=4 scoring vs single-device scoring (B=4, T=256):'
        f' rows differing {tp_vs_single[0]}/{tp_vs_single[1]}, max abs diff'
        f' {tp_vs_single[2]:.3e}',
        flush=True,
    )

    # K3: cross-kernel on the same dense operands (single device).
    k3_lines = [
        (
            '| B | T | Pair | Max abs diff | Mean abs diff | Exact % | ms (left'
            ' / right) |'
        ),
        '| :-: | :-: | :--- | :-: | :-: | :-: | :-: |',
    ]
    for b, t in ((1, 256), (4, 256), (4, 1024)):
      q, k, v = _operands(7000 + b * 10 + t, b, t)
      idx = jnp.arange(t)
      mask = jnp.broadcast_to((idx[:, None] >= idx[None, :])[None], (b, t, t))
      rpa_fn = jax.jit(
          functools.partial(
              rpa_backend.attention_cacheless, scale=SCALE, differentiable=False
          )
      )
      contract_fn = jax.jit(
          functools.partial(
              qwen3_canon.canonical_online_attention,
              scale=SCALE,
              attn_mask=mask,
          )
      )
      replica_fn = jax.jit(
          functools.partial(
              rpa_backend.attention_cacheless,
              scale=SCALE,
              differentiable=False,
              kernel=rpa_backend.reference_kernel,
          )
      )
      rpa_out, rpa_ms = _timed(rpa_fn, q, k, v)
      contract_out, contract_ms = _timed(contract_fn, q, k, v)
      with jax.default_matmul_precision('highest'):
        replica_out, replica_ms = _timed(replica_fn, q, k, v)
      outs = {
          'rpa (engine form)': (rpa_out, rpa_ms),
          'contract (canon_attention)': (contract_out, contract_ms),
          'f32 replica': (replica_out, replica_ms),
      }
      pairs = (
          ('rpa (engine form)', 'contract (canon_attention)'),
          ('rpa (engine form)', 'f32 replica'),
          ('contract (canon_attention)', 'f32 replica'),
      )
      for left, right in pairs:
        a, a_ms = outs[left]
        c, c_ms = outs[right]
        a = np.asarray(a, np.float32)
        c = np.asarray(c, np.float32)
        abs_diff = np.abs(a - c)
        exact = float(np.mean(a == c) * 100.0)
        k3_lines.append(
            f'| {b} | {t} | `{left}` vs `{right}` |'
            f' `{float(abs_diff.max()):.3e}` | `{float(abs_diff.mean()):.3e}`'
            f' | `{exact:.2f}` | `{a_ms:.3f} / {c_ms:.3f}` |'
        )
    print('[rpa-r1] K3 cross-kernel\n' + '\n'.join(k3_lines), flush=True)

    # The engine form's claim: one attention row per query, whichever program.
    for row in k1_rows + tp_rows:
      self.assertEqual(
          row.rows_differing,
          0,
          msg=(
              f'{row.tag} B={row.b} T={row.t} page={row.page_size}'
              f' {row.program}'
          ),
      )

  def test_cross_engine_rpa_backend_tpu(self):
    _skip_unless_measurement(self, R2_ENV)
    jax.clear_caches()
    env = {rpa_diff_chunked.MAX_SEQS_ENV: '4'}
    rpa_fwd = mock.patch.object(
        canon_attention, 'ATTENTION_FWD', canon_attention.RPA_FWD
    )
    try:
      with mock.patch.dict(os.environ, env), rpa_fwd:
        self._cross_engine_body()
    finally:
      jax.clear_caches()

  def _cross_engine_body(self):
    print(f'[rpa-r2] controls: {rpa_backend.describe()}', flush=True)
    # (a) Drift regimes at T=256 and T=1024.
    for prompt_len, cache_size in ((224, 256), (992, 1024)):
      drift = qwen3_canon_benchmark.run_drift_regime_diagnostics(
          preset='qwen3_0p6b_2l',
          batch_size=4,
          prompt_len=prompt_len,
          max_new_tokens=DECODE_STEPS,
          cache_size=cache_size,
          tp_size=TP,
          configs=(STOCK, DISABLED, ENABLED),
          include_deep_stack=False,
      )
      print(
          f'[rpa-r2] drift regimes T={cache_size}\n'
          + qwen3_canon_benchmark.format_drift_regime_report(drift),
          flush=True,
      )
      # The stock program (`canon_all_disabled`) drifts by design; the claim
      # is on the Zero-TIM program with the RPA backend as its attention.
      for row in drift:
        self.assertTrue(np.isfinite(row.max_abs_diff), msg=row)
        if row.config_name == ENABLED:
          self.assertEqual(row.max_abs_diff, 0.0, msg=row)

    # (b) The suite with the attention switch's Leave-One-Out / Add-One-In.
    suite = qwen3_canon_benchmark.run_benchmark_suite(
        preset='qwen3_0p6b_2l',
        batch_size=4,
        prompt_len=224,
        max_new_tokens=DECODE_STEPS,
        cache_size=256,
        num_iters=3,
        run_ablations=True,
        include_baselines=True,
        include_fwdbwd=True,
        switches=('use_canon_attention',),
        tp_size=TP,
    )
    print(
        '[rpa-r2] suite (attention Leave-One-Out / Add-One-In)\n'
        + qwen3_canon_benchmark.format_markdown_report(suite),
        flush=True,
    )
    self.assertEqual(
        [r.name for r in suite],
        [
            STOCK,
            DISABLED,
            ENABLED,
            'without_use_canon_attention',
            'only_use_canon_attention',
        ],
    )
    by_name = {r.name: r for r in suite}
    self.assertEqual(by_name[DISABLED].vs_stock_max_abs_diff, 0.0)
    enabled = by_name[ENABLED]
    for field in (
        'decode_vs_prefill_max_diff',
        'batch1_vs_batchn_max_diff',
        'fwd_vs_fwdbwd_max_diff',
    ):
      self.assertEqual(getattr(enabled, field), 0.0, msg=field)

    # (c) TP=4 parameter-gradient gate vs stock (the replica VJP).
    cfg = qwen3_canon_benchmark.build_model_config('qwen3_0p6b_2l', tp_size=TP)
    with _tp_mesh():
      stock = qwen3_stock.Qwen3(cfg, rngs=nnx.Rngs(params=0))
      qwen3_canon.init_non_zero_mlp_weights(stock, seed=7)
      canon = qwen3_canon.Qwen3Canon(
          cfg,
          rngs=nnx.Rngs(params=0),
          canon_config=qwen3_canon.CanonKernelConfig.all_enabled(),
      )
      qwen3_canon.copy_weights(stock, canon)
      tokens = jax.random.randint(
          jax.random.PRNGKey(1), (4, 256), 0, cfg.vocab_size, dtype=jnp.int32
      )
      g_stock = _param_grads(stock, tokens, use_canon_logsoftmax=False)
      g_canon = _param_grads(canon, tokens, use_canon_logsoftmax=True)
    ref = jax.tree_util.tree_leaves_with_path(g_stock)
    test = jax.tree_util.tree_leaves_with_path(g_canon)
    self.assertLen(test, len(ref))
    worst_ratio_dev, min_cosine = 0.0, 1.0
    for (p_ref, r), (p_test, t) in zip(ref, test):
      path = jax.tree_util.keystr(p_ref)
      self.assertEqual(path, jax.tree_util.keystr(p_test))
      r = np.asarray(r, np.float32).ravel()
      t = np.asarray(t, np.float32).ravel()
      norm_r, norm_t = float(np.linalg.norm(r)), float(np.linalg.norm(t))
      ratio = norm_t / norm_r
      cosine = float(np.dot(r, t) / (norm_r * norm_t + 1e-30))
      worst_ratio_dev = max(worst_ratio_dev, abs(ratio - 1.0))
      min_cosine = min(min_cosine, cosine)
    print(
        f'[rpa-r2] grad-check tp={TP} {len(ref)} leaves, worst |ratio-1| ='
        f' {worst_ratio_dev:.2e}, min cosine = {min_cosine:.6f}',
        flush=True,
    )
    # The project's gradient gate (`qwen3_model_canon_test`): measured
    # 3.32e-03 / 0.999487 on 4x v5p, TP=4.
    self.assertLess(worst_ratio_dev, 0.02)
    self.assertGreaterEqual(min_cosine, 0.999)

    # (d) Perf: L=8, B=4 (all three programs) and B=32 (Sample / Prefill).
    perf = qwen3_canon_benchmark.run_performance_scaling_benchmark(
        tp_size=TP,
        num_iters=3,
        preset='qwen3_0p6b',
        workloads=(
            qwen3_canon_benchmark.PerfWorkload(
                'L=8, B=4, T=256 (224+32)', 8, 4, 224, 32, 256
            ),
        ),
        leave_one_out_workloads=(),
        leave_one_out_switches=(),
    )
    print(
        '[rpa-r2] perf L=8, B=4 (RPA backend in the canon rows)\n'
        + qwen3_canon_benchmark.format_performance_scaling_report(perf),
        flush=True,
    )
    lines = [
        (
            '| Workload | Configuration | Sample (ms) | Prefill (ms) |'
            ' decode-vs-prefill max abs diff |'
        ),
        '| :--- | :--- | :-: | :-: | :-: |',
    ]
    cfg = qwen3_canon_benchmark.build_model_config('qwen3_0p6b', tp_size=TP)
    cfg.num_layers = 8
    with _tp_mesh():
      stock = qwen3_stock.Qwen3(cfg, rngs=nnx.Rngs(params=0))
      qwen3_canon.init_non_zero_mlp_weights(stock, seed=123)
      canon = qwen3_canon.Qwen3Canon(
          cfg,
          rngs=nnx.Rngs(params=0),
          canon_config=qwen3_canon.CanonKernelConfig.all_enabled(),
      )
      qwen3_canon.copy_weights(stock, canon)
      prompt = jax.random.randint(
          jax.random.PRNGKey(2026),
          (32, 224),
          0,
          cfg.vocab_size,
          dtype=jnp.int32,
      )
      for name, model, use_cls in (
          (STOCK, stock, False),
          (ENABLED, canon, True),
      ):
        (full_tokens, decode_lps, _), sample_ms, _ = (
            qwen3_canon_benchmark._time_fn(  # pylint: disable=protected-access
                qwen3_canon.forward_sample_with_logprobs,
                model,
                prompt,
                max_new_tokens=DECODE_STEPS,
                cache_size=256,
                use_canon_logsoftmax=use_cls,
                num_iters=3,
            )
        )
        prefill_lps, prefill_ms, _ = (
            qwen3_canon_benchmark._time_fn(  # pylint: disable=protected-access
                qwen3_canon.score_sequence_prefill,
                model,
                full_tokens,
                use_canon_logsoftmax=use_cls,
                num_iters=3,
            )
        )
        diff = float(
            np.max(
                np.abs(
                    np.asarray(decode_lps, np.float32)
                    - np.asarray(prefill_lps[:, 224 - 1 :], np.float32)
                )
            )
        )
        lines.append(
            f'| `L=8, B=32, T=256 (224+32)` | `{name}` | `{sample_ms:.2f}` |'
            f' `{prefill_ms:.2f}` | `{diff:.3e}` |'
        )
    print(
        '[rpa-r2] perf L=8, B=32 (Sample / Prefill)\n' + '\n'.join(lines),
        flush=True,
    )


def _param_grads(model, tokens: jax.Array, *, use_canon_logsoftmax: bool):
  """`d(-mean token logprob) / d params` of a cacheless causal prefill."""
  graphdef, params, other_state = nnx.split(model, nnx.Param, ...)
  b, l_total = tokens.shape
  positions = jnp.broadcast_to(
      jnp.arange(l_total, dtype=jnp.int32)[None, :], (b, l_total)
  )
  idx = jnp.arange(l_total, dtype=jnp.int32)
  causal_mask = jnp.broadcast_to(
      (idx[:, None] >= idx[None, :])[None, :, :], (b, l_total, l_total)
  )

  def loss_fn(param_state):
    m = nnx.merge(graphdef, param_state, other_state)
    logits, _ = m(tokens, positions, None, causal_mask)
    lps, _ = qwen3_canon.next_token_logprobs(
        logits,
        tokens,
        use_canon_logsoftmax=use_canon_logsoftmax,
        exact_residual_dtype=model.config.dtype,
    )
    return -jnp.mean(lps)

  return jax.jit(jax.grad(loss_fn))(params)


if __name__ == '__main__':
  absltest.main()
