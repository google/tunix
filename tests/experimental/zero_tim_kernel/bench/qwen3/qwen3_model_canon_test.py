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

import contextlib
import dataclasses
import functools
import hashlib
import os
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import nnx
import jax
from jax import numpy as jnp
import numpy as np
from tunix.experimental.zero_tim_kernel import canonical_logsoftmax
from tunix.experimental.zero_tim_kernel import fixed_order_reduce
from tunix.experimental.zero_tim_kernel import test_utils
from tunix.experimental.zero_tim_kernel.bench.qwen3 import canon_attention
from tunix.experimental.zero_tim_kernel.bench.qwen3 import qwen3_canon_benchmark
from tunix.experimental.zero_tim_kernel.bench.qwen3 import qwen3_model_canon as qwen3_canon
from tunix.models.qwen3 import model as qwen3_stock


def setUpModule():
  test_utils.configure_cpu(num_devices=4)
  # B2 diagnostics (TPU target only): XLA writes the optimized HLO and buffer
  # assignment of the `jit_hbmdump_*` perf programs compiled by
  # `qwen3_canon_benchmark._dump_compiled_program` into the test outputs.
  dump_root = os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR')
  if qwen3_canon_benchmark.hbm_dump_enabled() and dump_root:
    dump_dir = os.path.join(dump_root, 'xla_dump')
    os.environ['XLA_FLAGS'] = (
        f'{os.environ.get("XLA_FLAGS", "")} --xla_dump_to={dump_dir}'
        f' --xla_dump_hlo_module_re={qwen3_canon_benchmark.HBM_DUMP_MODULE_PREFIX}'
    ).strip()


def _make_test_config(
    *,
    use_tied_embedding: bool = True,
    tp_size: int = 1,
    vocab_size: int | None = None,
) -> qwen3_stock.ModelConfig:
  shd_cfg = (
      qwen3_canon.get_tp_sharding_config('tp')
      if tp_size > 1
      else qwen3_stock.ShardingConfig.get_default_sharding(is_sampling=True)
  )
  if vocab_size is None:
    # Under TP every rank's `V/TP` shard is one whole 1024-column tile so the
    # vocab-parallel log-softmax path is the one under test (see
    # `vocab_parallel_logsoftmax_admitted`).
    vocab_size = 1024 if tp_size == 1 else 1024 * tp_size
  return qwen3_stock.ModelConfig(
      num_layers=2,
      vocab_size=vocab_size,
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


def _tp_mesh(tp_size: int) -> jax.sharding.Mesh:
  """`('fsdp', 'tp')` mesh over the first `tp_size` devices.

  Skips the test only when `test_utils.configure_cpu` recorded why it could
  not provide the devices (a shared OSS test process); otherwise too few
  devices is an error, not a silent skip.

  Args:
    tp_size: Tensor-parallel degree; the mesh shape is `(1, tp_size)`.

  Returns:
    The `('fsdp', 'tp')` mesh.
  """
  return test_utils.cpu_mesh((1, tp_size), ('fsdp', 'tp'))


def _interpret() -> bool:
  return jax.default_backend() == 'cpu'


def _perf_workload(label: str) -> qwen3_canon_benchmark.PerfWorkload:
  """The `DEFAULT_PERF_WORKLOADS` entry with this label."""
  workloads = qwen3_canon_benchmark.DEFAULT_PERF_WORKLOADS
  (workload,) = [w for w in workloads if w.label == label]
  return workload


def _vs_stock_cell(
    r: qwen3_canon_benchmark.PerfScalingRow,
    stock: qwen3_canon_benchmark.PerfScalingRow,
    attr: str,
    fmt: str,
) -> str:
  value = getattr(r, attr)
  return f'`{value:{fmt}}` ({value - getattr(stock, attr):+{fmt}})'


def _perf_delta_table(
    rows: list[qwen3_canon_benchmark.PerfScalingRow],
) -> str:
  """Latency / arena of every perf row with its delta against the stock row.

  Args:
    rows: `run_performance_scaling_benchmark` rows; every workload must have its
      `stock_qwen3 (model.py)` row.

  Returns:
    A Markdown table, one line per row, cells `value (delta vs stock)`.
  """
  by_workload: dict[str, dict[str, qwen3_canon_benchmark.PerfScalingRow]] = {}
  for r in rows:
    by_workload.setdefault(r.workload, {})[r.config_name] = r
  lines = [
      (
          '| Workload | Configuration | Sample ms (vs stock) |'
          ' Prefill ms (vs stock) | Fwd+Bwd ms (vs stock) |'
          ' Prefill Arena MB (vs stock) | Fwd+Bwd Arena MB (vs stock) |'
      ),
      '| :--- | :--- | :---: | :---: | :---: | :---: | :---: |',
  ]
  for workload, configs in by_workload.items():
    stock = configs['stock_qwen3 (model.py)']
    for name, r in configs.items():
      cells = [
          _vs_stock_cell(r, stock, 'sample_latency_ms', '.2f'),
          _vs_stock_cell(r, stock, 'prefill_latency_ms', '.2f'),
          _vs_stock_cell(r, stock, 'fwdbwd_latency_ms', '.2f'),
          _vs_stock_cell(r, stock, 'prefill_arena_hbm_mb', '.0f'),
          _vs_stock_cell(r, stock, 'fwdbwd_arena_hbm_mb', '.0f'),
      ]
      lines.append(f'| `{workload}` | `{name}` | ' + ' | '.join(cells) + ' |')
  return '\n'.join(lines)


def _library_reference(
    logits_f32: np.ndarray, tokens: np.ndarray, *, interpret: bool
) -> tuple[np.ndarray, np.ndarray]:
  """Unsharded library `token_logprobs` / `log_softmax` for any row count.

  The library serves one row bucket (a multiple of 8 up to `PRODUCTION_M=256`)
  per call on TPU; rows are independent, so longer inputs are padded to a
  multiple of 8 and scored 256 rows at a time.

  Args:
    logits_f32: `[M, V]` float32 logits.
    tokens: `[M]` int32 token ids to score.
    interpret: run the Pallas kernels in interpret mode (CPU).

  Returns:
    `(token_logprobs, log_softmax)` numpy arrays for the `M` rows.
  """
  os.environ.setdefault(canonical_logsoftmax.ENV, '1')
  m = logits_f32.shape[0]
  align = canonical_logsoftmax.ROW_BUCKET_ALIGN
  mp = -(-m // align) * align
  logits_p = np.pad(logits_f32, ((0, mp - m), (0, 0)))
  tokens_p = np.pad(tokens, (0, mp - m))
  lps, ls = [], []
  for start in range(0, mp, canonical_logsoftmax.PRODUCTION_M):
    stop = min(start + canonical_logsoftmax.PRODUCTION_M, mp)
    chunk = jnp.asarray(logits_p[start:stop], jnp.float32)
    tok = jnp.asarray(tokens_p[start:stop], jnp.int32)
    lps.append(
        np.asarray(
            canonical_logsoftmax.token_logprobs(
                chunk, tok, interpret=interpret, strict_vocab=False
            )
        )
    )
    ls.append(
        np.asarray(
            canonical_logsoftmax.log_softmax(
                chunk, interpret=interpret, strict_vocab=False
            )
        )
    )
  return np.concatenate(lps)[:m], np.concatenate(ls)[:m]


def _prefill_logits(model, tokens: jax.Array) -> jax.Array:
  """Cacheless causal prefill logits `[B, L, V]`."""
  b, l_total = tokens.shape
  positions = jnp.broadcast_to(
      jnp.arange(l_total, dtype=jnp.int32)[None, :], (b, l_total)
  )
  idx = jnp.arange(l_total, dtype=jnp.int32)
  causal_mask = jnp.broadcast_to(
      (idx[:, None] >= idx[None, :])[None, :, :], (b, l_total, l_total)
  )
  logits, _ = model(tokens, positions, None, causal_mask)
  return logits


@functools.partial(jax.jit, static_argnames=('use_canon_logsoftmax',))
def _sliced_scoring(model, tokens: jax.Array, *, use_canon_logsoftmax: bool):
  """The pre-A1 scoring: slice `logits[:, :-1, :]` first, then score."""
  logits = _prefill_logits(model, tokens)
  return qwen3_canon.compute_token_logprobs(
      logits[:, :-1, :],
      tokens[:, 1:],
      use_canon_logsoftmax=use_canon_logsoftmax,
      exact_residual_dtype=model.config.dtype,
  )


_scoring_logits = jax.jit(_prefill_logits)


@dataclasses.dataclass(frozen=True)
class _RowDiff:
  """Rows of one program that differ from the scoring prefill's rows."""

  program: str
  rows_differing: int
  rows_compared: int
  max_abs_diff: float
  steps_differing: tuple[int, ...]  # positions / steps with any differing row

  def cells(self) -> str:
    shown = self.steps_differing[:16]
    steps = ','.join(map(str, shown)) or '-'
    if len(self.steps_differing) > len(shown):
      steps += f',... (+{len(self.steps_differing) - len(shown)})'
    return (
        f'`{self.program}` | `{self.rows_differing}/{self.rows_compared}` |'
        f' `{self.max_abs_diff:.3e}` | `{steps}`'
    )


def _attention_decode_vs_prefill(
    *,
    batch_size: int,
    seq_len: int,
    prompt_len: int,
    cache_size: int,
    num_heads: int = 16,
    num_kv_heads: int = 8,
    head_dim: int = 128,
    tp_size: int = 4,
    seed: int = 0,
    interpret: bool = False,
) -> list[_RowDiff]:
  """`canonical_online_attention` in the sampler's program shapes vs the scoring prefill.

  Model-free: random bf16 queries / keys / values of one `[B, seq_len]`
  sequence (Qwen3-0.6B head geometry, TP-sharded over the heads like the
  model) are attended three ways -- the scoring prefill (`T = S = seq_len`,
  causal), the sampler prefill (`T = prompt_len` rows against the
  `cache_size`-slot KV cache, slots past the prompt zero) and the decode step
  (`T = 1` against the cache, slots past the step's position zero) -- and the
  rows the programs share are compared bitwise. This isolates the attention
  core, the one Zero-TIM kernel whose program shape differs between the
  sampler (`T=1`, cache-sized `S`) and the scoring prefill (`T=S=L`); it is
  where the XLA online-softmax forward was caught drifting (`Tp=64` vs
  `Tp=256` programs) before the Pallas forward replaced it.

  Args:
    batch_size: Batch size of every program.
    seq_len: Sequence length of the scoring prefill (`prompt_len` plus decode
      steps).
    prompt_len: Rows of the sampler prefill; decode steps run at positions
      `prompt_len .. seq_len - 1`.
    cache_size: KV-cache slots of the sampler programs (`>= seq_len`).
    num_heads: Query heads (`QH`).
    num_kv_heads: KV heads (`KH`).
    head_dim: Head dimension.
    tp_size: Tensor-parallel degree of the mesh.
    seed: PRNG seed of the random operands.
    interpret: Run the Pallas attention kernel in interpret mode (CPU).

  Returns:
    `[_RowDiff(sampler prefill), _RowDiff(decode)]`; a row is one query token's
    `[QH, D]` output, `steps_differing` are prompt positions / decode steps.
  """
  scale = head_dim**-0.5
  kq, kk, kv = jax.random.split(jax.random.PRNGKey(seed), 3)
  q = jax.random.normal(
      kq, (batch_size, seq_len, num_heads, head_dim), jnp.float32
  ).astype(jnp.bfloat16)
  k = jax.random.normal(
      kk, (batch_size, seq_len, num_kv_heads, head_dim), jnp.float32
  ).astype(jnp.bfloat16)
  v = jax.random.normal(
      kv, (batch_size, seq_len, num_kv_heads, head_dim), jnp.float32
  ).astype(jnp.bfloat16)
  idx = jnp.arange(seq_len, dtype=jnp.int32)
  slots = jnp.arange(cache_size, dtype=jnp.int32)
  attend = functools.partial(
      qwen3_canon.canonical_online_attention,
      scale=scale,
      interpret=interpret,
  )

  def to_cache(x, last_filled):
    """`x` in a `cache_size`-slot cache, slots past `last_filled` zero."""
    x = jnp.pad(x, ((0, 0), (0, cache_size - seq_len), (0, 0), (0, 0)))
    keep = (slots <= last_filled)[None, :, None, None]
    return jnp.where(keep, x, jnp.zeros_like(x))

  @jax.jit
  def scoring_prefill(q, k, v):
    mask = jnp.broadcast_to(
        (idx[:, None] >= idx[None, :])[None], (batch_size, seq_len, seq_len)
    )
    return attend(q, k, v, attn_mask=mask)

  @jax.jit
  def sampler_prefill(q, k, v):
    prompt_pos = idx[:prompt_len]
    mask = jnp.broadcast_to(
        jnp.logical_and(
            slots[None, :] <= prompt_pos[:, None], slots[None, :] < prompt_len
        )[None],
        (batch_size, prompt_len, cache_size),
    )
    return attend(
        q[:, :prompt_len],
        to_cache(k, prompt_len - 1),
        to_cache(v, prompt_len - 1),
        attn_mask=mask,
    )

  @jax.jit
  def decode_step(q, k, v, pos):
    q_step = jax.lax.dynamic_slice_in_dim(q, pos, 1, axis=1)
    mask = jnp.broadcast_to(
        (slots <= pos)[None, None, :], (batch_size, 1, cache_size)
    )
    return attend(q_step, to_cache(k, pos), to_cache(v, pos), attn_mask=mask)

  def row_diff(program, got, ref, steps):
    got = np.asarray(got).astype(np.float32)
    ref = np.asarray(ref).astype(np.float32)
    differs = np.any(got != ref, axis=(-1, -2))  # [B, rows]
    return _RowDiff(
        program,
        int(differs.sum()),
        int(differs.size),
        float(np.max(np.abs(got - ref))) if differs.any() else 0.0,
        tuple(int(s) for s, d in zip(steps, differs.any(axis=0)) if d),
    )

  with _tp_mesh(tp_size):
    ref = np.asarray(scoring_prefill(q, k, v))  # [B, seq_len, QH, D]
    sampler = sampler_prefill(q, k, v)  # [B, prompt_len, QH, D]
    decode = jnp.concatenate(
        [
            decode_step(q, k, v, jnp.int32(pos))
            for pos in range(prompt_len, seq_len)
        ],
        axis=1,
    )  # [B, seq_len - prompt_len, QH, D]
  return [
      row_diff(
          f'sampler prefill (T={prompt_len}, S={cache_size})',
          sampler,
          ref[:, :prompt_len],
          range(prompt_len),
      ),
      row_diff(
          f'decode (T=1, S={cache_size})',
          decode,
          ref[:, prompt_len:],
          range(seq_len - prompt_len),
      ),
  ]


@dataclasses.dataclass(frozen=True)
class _StepDiff:
  """Sampler decode step vs the scoring prefill's row for the same token."""

  step: int
  position: int
  logit_rows_differing: int
  max_abs_logit_diff: float
  logprobs_differing: int
  max_abs_logprob_diff: float


def _decode_step_diffs(
    model,
    prompt_tokens: jax.Array,
    *,
    max_new_tokens: int,
    cache_size: int,
) -> tuple[list[_StepDiff], _RowDiff]:
  """Per-decode-step logits / logprobs of the sampler vs the scoring prefill.

  Runs `forward_sample_with_logits` (the sampler with its per-step logits
  exposed) and scores `full_tokens` with the cacheless prefill (logits via
  `_scoring_logits`, logprobs via `score_sequence_prefill`); both with
  `use_canon_logsoftmax=True`.

  Args:
    model: The model (`Qwen3Canon` in the configuration under test).
    prompt_tokens: `[B, L]` prompt.
    max_new_tokens: Decode steps.
    cache_size: KV-cache slots.

  Returns:
    `(per-step diffs, prompt-row diff)`: one `_StepDiff` per decode step (step
    0 is the first generated token, scored on the sampler prefill's last prompt
    row) and the `_RowDiff` of the sampler prefill's prompt logprobs against the
    scoring prefill's first `L - 1` rows.
  """
  _, prompt_len = prompt_tokens.shape
  full_tokens, dec_lps, prompt_lps, step_logits = (
      qwen3_canon.forward_sample_with_logits(
          model,
          prompt_tokens,
          max_new_tokens=max_new_tokens,
          cache_size=cache_size,
          use_canon_logsoftmax=True,
      )
  )
  pref_lps = np.asarray(
      qwen3_canon.score_sequence_prefill(
          model, full_tokens, use_canon_logsoftmax=True
      )
  )
  pref_logits = np.asarray(_scoring_logits(model, full_tokens)).astype(
      np.float32
  )
  step_logits = np.asarray(step_logits).astype(np.float32)
  dec_lps = np.asarray(dec_lps)
  prompt_lps = np.asarray(prompt_lps)
  steps = []
  for t in range(max_new_tokens):
    pos = prompt_len - 1 + t  # scoring row of generated token `t`
    d_logit = np.abs(step_logits[:, t] - pref_logits[:, pos])
    d_lp = np.abs(dec_lps[:, t] - pref_lps[:, pos])
    steps.append(
        _StepDiff(
            t,
            pos,
            int(np.any(d_logit > 0, axis=-1).sum()),
            float(d_logit.max()),
            int((d_lp > 0).sum()),
            float(d_lp.max()),
        )
    )
  d_prompt = np.abs(prompt_lps - pref_lps[:, : prompt_len - 1])
  prompt_rows = _RowDiff(
      f'sampler prefill prompt logprobs (rows 0..{prompt_len - 2})',
      int((d_prompt > 0).sum()),
      int(d_prompt.size),
      float(d_prompt.max()),
      tuple(int(s) for s in np.flatnonzero(np.any(d_prompt > 0, axis=0))),
  )
  return steps, prompt_rows


def _format_step_diffs(steps: list[_StepDiff], batch_size: int) -> str:
  lines = [
      (
          '| Step | Scoring row | Logit rows differing (of'
          f' {batch_size}) | Max abs logit diff | Logprobs differing (of'
          f' {batch_size}) | Max abs logprob diff |'
      ),
      '| :---: | :---: | :---: | :---: | :---: | :---: |',
  ]
  for s in steps:
    lines.append(
        f'| {s.step} | {s.position} | `{s.logit_rows_differing}` |'
        f' `{s.max_abs_logit_diff:.3e}` | `{s.logprobs_differing}` |'
        f' `{s.max_abs_logprob_diff:.3e}` |'
    )
  return '\n'.join(lines)


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

  def test_forward_sample_with_logits_is_the_sampler(self):
    """`forward_sample_with_logits` = `forward_sample_with_logprobs` + logits.

    Same tokens and logprobs, the `step_logits` argmax is the generated token,
    rescoring the stacked logits in one call reproduces the per-step logprobs
    (row-independent canonical log-softmax), and the `_decode_step_diffs`
    probe built on it reports no differing step or prompt row on CPU, where
    `test_zero_tim_decode_vs_prefill_and_batch_and_fwdbwd_bitwise` already
    proves the contract.
    """
    cfg = _make_test_config(use_tied_embedding=True)
    canon = qwen3_canon.Qwen3Canon(
        cfg,
        rngs=nnx.Rngs(params=0),
        canon_config=qwen3_canon.CanonKernelConfig.all_enabled(),
    )
    qwen3_canon.init_non_zero_mlp_weights(canon, seed=42)
    prompt_tokens = jax.random.randint(
        jax.random.PRNGKey(99), (3, 5), 0, cfg.vocab_size, dtype=jnp.int32
    )
    for max_new_tokens in (
        4,
        1,
    ):  # the scan body and the `max_new_tokens == 1` return
      kwargs = dict(
          max_new_tokens=max_new_tokens,
          cache_size=32,
          use_canon_logsoftmax=True,
      )
      tokens, lps, prompt_lps = qwen3_canon.forward_sample_with_logprobs(
          canon, prompt_tokens, **kwargs
      )
      tokens2, lps2, prompt_lps2, step_logits = (
          qwen3_canon.forward_sample_with_logits(canon, prompt_tokens, **kwargs)
      )
      np.testing.assert_array_equal(np.asarray(tokens2), np.asarray(tokens))
      np.testing.assert_array_equal(np.asarray(lps2), np.asarray(lps))
      np.testing.assert_array_equal(
          np.asarray(prompt_lps2), np.asarray(prompt_lps)
      )
      self.assertEqual(step_logits.shape, (3, max_new_tokens, cfg.vocab_size))
      self.assertEqual(step_logits.dtype, cfg.dtype)
      gen_tokens = tokens2[:, 5:]
      np.testing.assert_array_equal(
          np.asarray(jnp.argmax(step_logits, axis=-1)), np.asarray(gen_tokens)
      )
      rescored = qwen3_canon.compute_token_logprobs(
          step_logits,
          gen_tokens,
          use_canon_logsoftmax=True,
          exact_residual_dtype=cfg.dtype,
      )
      np.testing.assert_array_equal(np.asarray(rescored), np.asarray(lps2))

    steps, prompt_rows = _decode_step_diffs(
        canon, prompt_tokens, max_new_tokens=4, cache_size=32
    )
    self.assertLen(steps, 4)
    self.assertEqual([s.position for s in steps], [4, 5, 6, 7])
    for s in steps:
      self.assertEqual(s.logit_rows_differing, 0, msg=str(s))
      self.assertEqual(s.logprobs_differing, 0, msg=str(s))
      self.assertEqual(s.max_abs_logit_diff, 0.0, msg=str(s))
      self.assertEqual(s.max_abs_logprob_diff, 0.0, msg=str(s))
    self.assertEqual(prompt_rows.rows_compared, 3 * 4)
    self.assertEqual(prompt_rows.rows_differing, 0, msg=str(prompt_rows))
    self.assertEqual(prompt_rows.steps_differing, ())
    self.assertIn('| `0` |', _format_step_diffs(steps, 3))

  def test_attention_decode_vs_prefill_probe_tp4(self):
    """The model-free attention probe reports no differing row on CPU (TP=4).

    Same claim for the attention core alone that the model-level CPU test
    makes for the whole forward: the sampler-prefill and decode program
    shapes reproduce the scoring prefill's rows bitwise. With the Pallas
    forward this holds by construction (one fixed tile program for every
    `T`); the TPU harness measures it on the real kernel under both
    `--xla_allow_excess_precision` settings. Two geometries: one key block
    (`S=16`), and `S=520` in a 1024-slot cache, where the second 512-key
    block is live for the last eight positions only (so the scoring prefill
    rescales across blocks for those rows, the decode steps visit two
    blocks and the sampler prefill skips the fully masked second block).
    """
    for seq_len, prompt_len, cache_size in ((16, 12, 16), (520, 512, 1024)):
      diffs = _attention_decode_vs_prefill(
          batch_size=2,
          seq_len=seq_len,
          prompt_len=prompt_len,
          cache_size=cache_size,
          tp_size=4,
          interpret=_interpret(),
      )
      self.assertLen(diffs, 2)
      sampler, decode = diffs
      self.assertStartsWith(
          sampler.program, f'sampler prefill (T={prompt_len}, S={cache_size})'
      )
      self.assertEqual(sampler.rows_compared, 2 * prompt_len)
      self.assertEqual(decode.program, f'decode (T=1, S={cache_size})')
      self.assertEqual(decode.rows_compared, 2 * (seq_len - prompt_len))
      for d in diffs:
        self.assertEqual(d.rows_differing, 0, msg=str(d))
        self.assertEqual(d.max_abs_diff, 0.0, msg=str(d))
        self.assertEqual(d.steps_differing, (), msg=str(d))
        self.assertIn('`0/', d.cells())

  def test_vocab_parallel_logsoftmax_admitted(self):
    admitted = qwen3_canon.vocab_parallel_logsoftmax_admitted
    self.assertTrue(admitted(8192, 4))  # TPU bench: v_local = 2048
    self.assertTrue(admitted(4096, 4))  # CPU tests: v_local = 1024
    self.assertTrue(admitted(2048, 2))
    self.assertFalse(admitted(2048, 4))  # v_local = 512: half a tile
    self.assertFalse(admitted(4096, 1))  # no TP axis
    self.assertFalse(admitted(4096, 3))  # V not divisible by TP
    for tp_size in (2, 4, 8):
      # Qwen3's production vocabulary (151936) is never tile-aligned per rank.
      self.assertFalse(admitted(canonical_logsoftmax.PRODUCTION_V, tp_size))

  @parameterized.named_parameters(('m8', 8), ('m256', 256), ('m516', 516))
  def test_vocab_parallel_logsoftmax_matches_library_bitwise(self, m: int):
    """A0 contract: the TP-sharded normalizer is bitwise the library's.

    The bench's vocab-parallel program (Stage 1 per rank on `[m, V/TP]` with a
    `(row block, tile)` grid, one all-gather of tile summaries, Stage 2 in
    global tile order) must reproduce the unsharded library
    `canonical_logsoftmax` bit for bit, for bf16 and f32 operands, for
    `token_logprobs` and `log_softmax`, and for row counts the library serves
    as one bucket (8, 256) or several (516 = 256 + 256 + 4, padded to 8).
    """
    tp_size, vocab = 4, 4096
    self.assertTrue(
        qwen3_canon.vocab_parallel_logsoftmax_admitted(vocab, tp_size)
    )
    rng = np.random.default_rng(m)
    logits_bf16 = jnp.asarray(
        rng.standard_normal((m, vocab)) * 4.0, jnp.bfloat16
    )
    logits_f32 = logits_bf16.astype(jnp.float32)
    tokens = jnp.asarray(rng.integers(0, vocab, size=(m,)), jnp.int32)
    ref_lps, ref_ls = _library_reference(
        np.asarray(logits_f32), np.asarray(tokens), interpret=_interpret()
    )

    token_logprobs = jax.jit(
        functools.partial(
            qwen3_canon.compute_token_logprobs, use_canon_logsoftmax=True
        )
    )
    log_softmax = jax.jit(
        functools.partial(
            qwen3_canon.compute_log_softmax, use_canon_logsoftmax=True
        )
    )
    with _tp_mesh(tp_size):
      vp_lps_bf16 = token_logprobs(logits_bf16, tokens)
      vp_lps_f32 = token_logprobs(logits_f32, tokens)
      vp_ls_bf16 = log_softmax(logits_bf16)
    test_utils.assert_bitwise_equal(vp_lps_bf16, ref_lps)
    test_utils.assert_bitwise_equal(vp_lps_f32, ref_lps)
    test_utils.assert_bitwise_equal(vp_ls_bf16, ref_ls)

  def test_non_tile_aligned_shard_falls_back_to_library_bitwise(self):
    """A2 guard: `V/TP` that is not a whole tile uses the gathered library path."""
    tp_size, vocab, m = 4, 2048, 16  # v_local = 512
    self.assertFalse(
        qwen3_canon.vocab_parallel_logsoftmax_admitted(vocab, tp_size)
    )
    rng = np.random.default_rng(0)
    logits_bf16 = jnp.asarray(
        rng.standard_normal((m, vocab)) * 4.0, jnp.bfloat16
    )
    tokens = jnp.asarray(rng.integers(0, vocab, size=(m,)), jnp.int32)
    ref_lps, ref_ls = _library_reference(
        np.asarray(logits_bf16, np.float32),
        np.asarray(tokens),
        interpret=_interpret(),
    )
    mesh = _tp_mesh(tp_size)
    with mesh:
      lps = jax.jit(
          functools.partial(
              qwen3_canon.compute_token_logprobs, use_canon_logsoftmax=True
          )
      )(logits_bf16, tokens)
      ls = jax.jit(
          functools.partial(
              qwen3_canon.compute_log_softmax, use_canon_logsoftmax=True
          )
      )(logits_bf16)
      # The vocab-parallel program itself fails closed on such a shard.
      with self.assertRaisesRegex(ValueError, 'needs v_local'):
        qwen3_canon._vocab_parallel_token_logprobs(  # pylint: disable=protected-access
            logits_bf16,
            tokens,
            mesh=mesh,
            tp_axis='tp',
            tp_size=tp_size,
            interpret=_interpret(),
            exact_residual_dtype=None,
        )
    test_utils.assert_bitwise_equal(lps, ref_lps)
    test_utils.assert_bitwise_equal(ls, ref_ls)

  def test_row_complete_scoring_is_bitwise_tp4(self):
    """A1: scoring all `B*T` rows at once equals slicing `logits[:, :-1]` first.

    Also pins the sampler's merged first-token scoring (Path A) and the
    learner's row-complete scoring (Path C) to Path B under the vocab-parallel
    log-softmax at TP=4.
    """
    tp_size = 4
    cfg = _make_test_config(use_tied_embedding=True, tp_size=tp_size)
    with _tp_mesh(tp_size):
      canon = qwen3_canon.Qwen3Canon(
          cfg,
          rngs=nnx.Rngs(params=0),
          canon_config=qwen3_canon.CanonKernelConfig.all_enabled(),
      )
      qwen3_canon.init_non_zero_mlp_weights(canon, seed=3)
      tokens = jax.random.randint(
          jax.random.PRNGKey(5), (2, 9), 0, cfg.vocab_size, dtype=jnp.int32
      )
      row_complete = qwen3_canon.score_sequence_prefill(
          canon, tokens, use_canon_logsoftmax=True
      )
      sliced = _sliced_scoring(canon, tokens, use_canon_logsoftmax=True)
      fwdbwd_lps, grad_norm = qwen3_canon.score_sequence_fwd_bwd(
          canon, tokens, use_canon_logsoftmax=True
      )
      full_tokens, decode_lps, prompt_lps = (
          qwen3_canon.forward_sample_with_logprobs(
              canon,
              tokens[:, :5],
              max_new_tokens=1,
              cache_size=16,
              use_canon_logsoftmax=True,
          )
      )
      prefill_full = qwen3_canon.score_sequence_prefill(
          canon, full_tokens, use_canon_logsoftmax=True
      )
    test_utils.assert_bitwise_equal(row_complete, sliced)
    test_utils.assert_bitwise_equal(fwdbwd_lps, row_complete)
    self.assertTrue(np.isfinite(float(grad_norm)))
    self.assertGreater(float(grad_norm), 0.0)
    test_utils.assert_bitwise_equal(prompt_lps, prefill_full[:, :4])
    test_utils.assert_bitwise_equal(decode_lps, prefill_full[:, 4:])

  @parameterized.named_parameters(
      ('vocab_parallel_logsoftmax', 'VOCAB_PARALLEL_LOGSOFTMAX', False, 1e-3),
      ('checkpoint_attn_core', 'CHECKPOINT_ATTN_CORE', False, 1e-3),
      ('attention_bwd', 'ATTENTION_BWD', canon_attention.XLA_BWD, 2e-2),
      ('rope_fuse_qk', 'ROPE_FUSE_QK', False, 1e-3),
      ('hard_rounding', 'HARD_ROUNDING', 0, 1e-3),
  )
  def test_optimization_switch_is_forward_bitwise_neutral_tp4(
      self, switch: str, alt_value, grad_rtol: float
  ):
    """Each HBM/latency switch changes the program, never the logprob bits.

    `HARD_ROUNDING` is the rounding-point counterpart: switching every site
    group off (`0`, the plain `astype` glue) must reproduce the default's
    `reduce_precision` bits under the strict
    `--xla_allow_excess_precision=false`
    these tests run with.
    """
    module = canon_attention if switch == 'ATTENTION_BWD' else qwen3_canon
    self.assertNotEqual(
        getattr(module, switch), alt_value, msg=f'{switch} default'
    )
    tp_size = 4
    cfg = _make_test_config(use_tied_embedding=False, tp_size=tp_size)
    with _tp_mesh(tp_size):
      canon = qwen3_canon.Qwen3Canon(
          cfg,
          rngs=nnx.Rngs(params=0),
          canon_config=qwen3_canon.CanonKernelConfig.all_enabled(),
      )
      qwen3_canon.init_non_zero_mlp_weights(canon, seed=11)
      tokens = jax.random.randint(
          jax.random.PRNGKey(2), (2, 9), 0, cfg.vocab_size, dtype=jnp.int32
      )
      on_prefill = qwen3_canon.score_sequence_prefill(
          canon, tokens, use_canon_logsoftmax=True
      )
      on_lps, on_norm = qwen3_canon.score_sequence_fwd_bwd(
          canon, tokens, use_canon_logsoftmax=True
      )
      # The switches are read at trace time; drop the cached traces so the
      # patched value is seen, and again afterwards so no later test reuses a
      # program traced with the alternative value.
      try:
        with mock.patch.object(module, switch, alt_value):
          jax.clear_caches()
          off_prefill = qwen3_canon.score_sequence_prefill(
              canon, tokens, use_canon_logsoftmax=True
          )
          off_lps, off_norm = qwen3_canon.score_sequence_fwd_bwd(
              canon, tokens, use_canon_logsoftmax=True
          )
      finally:
        jax.clear_caches()
    test_utils.assert_bitwise_equal(off_prefill, on_prefill)
    test_utils.assert_bitwise_equal(off_lps, on_lps)
    # The backward programs differ (local vs gathered VJP, remat vs stored
    # activations, Pallas flash backward vs its dense XLA form) and may round
    # differently; the flash backwards round their MXU operands to bf16 like
    # the TPU does, so they get the whole-model gradient gate
    # (`|ratio - 1| <= 0.02`). `canon_attention.ATTENTION_FWD` is deliberately
    # not in this list: the two forwards are two contracts (different
    # association orders), not a neutral switch (see `canon_attention_test`).
    np.testing.assert_allclose(float(off_norm), float(on_norm), rtol=grad_rtol)

  def test_forward_tp_sum_scatter_threshold_is_bitwise_tp4(self):
    """`FORWARD_TP_SUM_SCATTER_MIN_M` changes the collectives, never the bits.

    `_forward_tp_sum_mode` picks SCATTER for contract-parallel sites with at
    least `min_m` rows (default 4096, Step 12) and `FORWARD_TP_SUM_MODE`
    below. Under the default every `o_proj` / `down_proj` site of a `[2, 8]`
    scoring (`M=16`) is GATHER; with the threshold at 0 every site is SCATTER
    (`M=16` is divisible by TP=4, so `fixed_order_reduce` takes its
    `all_to_all` form and not the gather fallback): the lowered program must
    gain `all_to_all` collectives while prefill / fwd+bwd logprobs stay
    bitwise identical (identical operands, identical association order).
    """
    self.assertEqual(qwen3_canon.FORWARD_TP_SUM_SCATTER_MIN_M, 4096)
    gather, scatter = fixed_order_reduce.GATHER, fixed_order_reduce.SCATTER
    mode = qwen3_canon._forward_tp_sum_mode  # pylint: disable=protected-access
    self.assertEqual(qwen3_canon.FORWARD_TP_SUM_MODE, gather)
    self.assertEqual(
        [mode(m) for m in (16, 256, 4095, 4096, 8192)],
        [gather, gather, gather, scatter, scatter],
    )
    with mock.patch.object(qwen3_canon, 'FORWARD_TP_SUM_SCATTER_MIN_M', None):
      self.assertEqual(mode(1 << 20), gather)

    tp_size = 4
    cfg = _make_test_config(use_tied_embedding=False, tp_size=tp_size)
    with _tp_mesh(tp_size):
      canon = qwen3_canon.Qwen3Canon(
          cfg,
          rngs=nnx.Rngs(params=0),
          canon_config=qwen3_canon.CanonKernelConfig.all_enabled(),
      )
      qwen3_canon.init_non_zero_mlp_weights(canon, seed=11)
      tokens = jax.random.randint(
          jax.random.PRNGKey(2), (2, 8), 0, cfg.vocab_size, dtype=jnp.int32
      )

      def run():
        lowered = qwen3_canon.score_sequence_prefill.lower(
            canon, tokens, use_canon_logsoftmax=True
        )
        prefill = qwen3_canon.score_sequence_prefill(
            canon, tokens, use_canon_logsoftmax=True
        )
        lps, norm = qwen3_canon.score_sequence_fwd_bwd(
            canon, tokens, use_canon_logsoftmax=True
        )
        return (
            lowered.as_text().count('all_to_all'),
            np.asarray(prefill),
            np.asarray(lps),
            float(norm),
        )

      gather_a2a, on_prefill, on_lps, on_norm = run()
      try:
        with mock.patch.object(qwen3_canon, 'FORWARD_TP_SUM_SCATTER_MIN_M', 0):
          jax.clear_caches()
          scatter_a2a, off_prefill, off_lps, off_norm = run()
      finally:
        jax.clear_caches()
    # The SCATTER form adds `all_to_all` collectives (one per contract-parallel
    # site before any lowering-level dedup of identical layers); the default
    # program is all-gather based.
    self.assertGreater(scatter_a2a, gather_a2a)
    test_utils.assert_bitwise_equal(off_prefill, on_prefill)
    test_utils.assert_bitwise_equal(off_lps, on_lps)
    # The custom VJP broadcasts the same cotangent in both forms; the forward
    # collectives only change how XLA fuses around the adds (same gate as the
    # other forward-neutral switches).
    np.testing.assert_allclose(off_norm, on_norm, rtol=1e-3)

  @parameterized.named_parameters(('tp4_mesh', 4), ('no_mesh', 1))
  def test_apply_rope_canonical_qk_matches_separate_calls(self, tp_size: int):
    """D1: one q+k RoPE program is bitwise the two separate programs.

    The reference is the Step-10 glue: `apply_rope_canonical` once per tensor
    with the 64-row padding. Checked for the merged program with the padding
    (`ROPE_FUSE_QK`), without it (`ROPE_PAD_ROWS=None`), and for the
    `ROPE_FUSE_QK=False` fallback, both inside a TP=4 mesh (one `shard_map`
    over `[B, L, QH/TP + KH/TP, H]`) and without a mesh. `L=5` is not a
    multiple of the padding so the pad / slice path is exercised.

    Same-shape equality is all this test can show for `ROPE_PAD_ROWS`: on TPU
    the padding is what keeps the `T=1` decode program bitwise the `T=L`
    prefill program (drift regimes 1a-1c reject `None`, see the switch's
    definition), which no per-call or CPU check can see.
    """
    self.assertTrue(qwen3_canon.ROPE_FUSE_QK)
    self.assertEqual(qwen3_canon.ROPE_PAD_ROWS, 64)
    rng = np.random.default_rng(0)
    b, l, qh, kh, h = 2, 5, 8, 4, 128
    query = jnp.asarray(rng.standard_normal((b, l, qh, h)), jnp.bfloat16)
    key = jnp.asarray(rng.standard_normal((b, l, kh, h)), jnp.bfloat16)
    positions = jnp.asarray(
        np.arange(l, dtype=np.int32)[None, :] + np.array([[0], [1000]]),
        jnp.int32,
    )
    rope = functools.partial(qwen3_canon.apply_rope_canonical, head_dim=h)
    rope_qk = functools.partial(qwen3_canon.apply_rope_canonical_qk, head_dim=h)
    mesh = _tp_mesh(tp_size) if tp_size > 1 else None
    context = mesh if mesh is not None else contextlib.nullcontext()

    def run(fuse_qk: bool, pad_rows: int | None):
      try:
        with mock.patch.object(
            qwen3_canon, 'ROPE_FUSE_QK', fuse_qk
        ), mock.patch.object(qwen3_canon, 'ROPE_PAD_ROWS', pad_rows):
          jax.clear_caches()
          with context:
            q_out, k_out = jax.jit(rope_qk)(query, key, positions)
        return np.asarray(q_out), np.asarray(k_out)
      finally:
        jax.clear_caches()

    with context:
      q_ref = np.asarray(jax.jit(rope)(query, positions))
      k_ref = np.asarray(jax.jit(rope)(key, positions))
    # The rotation is not a no-op and the reference is finite.
    self.assertTrue(np.all(np.isfinite(q_ref.astype(np.float32))))
    self.assertFalse(np.array_equal(q_ref, np.asarray(query)))
    for fuse_qk, pad_rows in ((True, 64), (True, None), (False, 64)):
      with self.subTest(fuse_qk=fuse_qk, pad_rows=pad_rows):
        q_out, k_out = run(fuse_qk, pad_rows)
        self.assertEqual(q_out.shape, (b, l, qh, h))
        self.assertEqual(k_out.shape, (b, l, kh, h))
        test_utils.assert_bitwise_equal(q_out, q_ref)
        test_utils.assert_bitwise_equal(k_out, k_ref)

  def test_reduce_precision_glue_tpu_ab(self):
    """Attributes the default-XLA-flag drift to the XLA-glue rounding points.

    With the plain `astype` glue (`HARD_ROUNDING = 0`) and the TPU default
    `--xla_allow_excess_precision=true` the `canon_all_enabled` forward is not
    one program: the `B=4, T=64` prefill drifts from every other program
    (regimes 1b / 1c / 2 non-zero) while under
    `--xla_allow_excess_precision=false` every regime is `0.000e+00`. The
    Pallas kernels materialize their bf16 outputs, so the suspects are the
    `astype(bf16)` boundaries of the XLA glue, which the flag lets XLA move
    (see `HARD_ROUNDING`). For every mask of (none, all, each single site
    group, each complement) this runs regimes 1a-3a and a `B=4, T=64` prefill
    / fwd+bwd scoring in one process and prints one table; the masks that
    restore `0.000e+00` under the default flags attribute the drift, the
    complements show whether one group suffices, and the pair `ROPE + TP_SUM`
    (the two groups the complement rows single out) measures directly whether
    that pair alone reproduces the strict-flag bits. Under the strict flag
    every mask must be bitwise the unmodified program (asserted); under the
    default flags the table is the result and only its shape is asserted.

    Result on 4x v5p (TP=4, `qwen3_0p6b_2l`): under the strict flag all
    masks share one prefill / fwd+bwd fingerprint; under the default flags
    `ROPE + TP_SUM`, `ALL`, `ALL - RESIDUAL` and `ALL - ATTENTION` reproduce
    exactly that fingerprint with every regime `0.000e+00`; `only TP_SUM`,
    `only RESIDUAL`, `ALL - ROPE` and `ALL - TP_SUM` also reach zero drift but
    with flag-dependent bits; `none`, `only ROPE` and `only ATTENTION` drift
    (1b / 1c / 2 `max |diff|` `1.6e-02 .. 3.1e-02`, `0-6%` bitwise-equal).
    That made `HARD_ROUNDING_ALL` the default; the harness is kept for re-runs
    on new XLA / JAX versions.

    TPU-only (the strict-flag neutrality of `HARD_ROUNDING = 0` against the
    default is proven on CPU by the neutrality test above); runs only when
    `ZERO_TIM_HARD_ROUNDING_AB=1` is set. `setUpModule` adds
    `--xla_allow_excess_precision=false` for the whole process, TPU included,
    unless `XLA_FLAGS` already sets the flag, so the default-flag table needs
    an explicit `XLA_FLAGS=--xla_allow_excess_precision=true` in the test
    environment; run it filtered to this method so the 11 masks stay inside
    the runner's timeout.
    """
    if jax.default_backend() != 'tpu' or len(jax.devices()) < 4:
      self.skipTest('TPU-only attribution; the neutrality check runs on CPU.')
    if not os.environ.get('ZERO_TIM_HARD_ROUNDING_AB'):
      self.skipTest('Set ZERO_TIM_HARD_ROUNDING_AB=1 (see docstring).')
    strict = test_utils.EXCESS_PRECISION_FLAG in os.environ.get('XLA_FLAGS', '')
    zero_tim = 'canon_all_enabled (Zero-TIM)'
    all_bits = qwen3_canon.HARD_ROUNDING_ALL
    groups = (
        ('ROPE', qwen3_canon.HARD_ROUNDING_ROPE),
        ('TP_SUM', qwen3_canon.HARD_ROUNDING_TP_SUM),
        ('RESIDUAL', qwen3_canon.HARD_ROUNDING_RESIDUAL),
        ('ATTENTION', qwen3_canon.HARD_ROUNDING_ATTENTION),
    )
    masks = [('none', 0), ('ALL', all_bits)]
    masks += [(f'only {name}', bit) for name, bit in groups]
    masks += [(f'ALL - {name}', all_bits & ~bit) for name, bit in groups]
    masks += [(
        'ROPE + TP_SUM',
        qwen3_canon.HARD_ROUNDING_ROPE | qwen3_canon.HARD_ROUNDING_TP_SUM,
    )]
    tp_size = 4
    config = qwen3_canon_benchmark.build_model_config(
        'qwen3_0p6b_2l', tp_size=tp_size
    )
    tokens = jax.random.randint(
        jax.random.PRNGKey(2026), (4, 64), 0, config.vocab_size, dtype=jnp.int32
    )
    diag_by_mask = {}
    scores_by_mask = {}
    try:
      for label, mask in masks:
        with mock.patch.object(qwen3_canon, 'HARD_ROUNDING', mask):
          jax.clear_caches()
          diag_by_mask[label] = (
              qwen3_canon_benchmark.run_drift_regime_diagnostics(
                  preset='qwen3_0p6b_2l',
                  batch_size=4,
                  prompt_len=48,
                  max_new_tokens=16,
                  cache_size=128,
                  tp_size=tp_size,
                  configs=(zero_tim,),
                  include_deep_stack=False,
              )
          )
          with _tp_mesh(tp_size):
            canon = qwen3_canon.Qwen3Canon(
                config,
                rngs=nnx.Rngs(params=0),
                canon_config=qwen3_canon.CanonKernelConfig.all_enabled(),
            )
            qwen3_canon.init_non_zero_mlp_weights(canon, seed=123)
            prefill = qwen3_canon.score_sequence_prefill(
                canon, tokens, use_canon_logsoftmax=True
            )
            lps, norm = qwen3_canon.score_sequence_fwd_bwd(
                canon, tokens, use_canon_logsoftmax=True
            )
            scores_by_mask[label] = (
                np.asarray(prefill),
                np.asarray(lps),
                float(norm),
            )
            del canon
    finally:
      jax.clear_caches()

    def cell(max_abs: float, exact_pct: float) -> str:
      return f'`{max_abs:.1e}` ({exact_pct:.0f}%)'

    def versus(a: np.ndarray, b: np.ndarray) -> str:
      a32 = a.astype(np.float32)
      b32 = b.astype(np.float32)
      return cell(
          float(np.max(np.abs(a32 - b32))),
          100.0 * float(np.mean(a32 == b32)),
      )

    regimes = [d.regime.split('.')[0] for d in diag_by_mask['none']]
    flags = 'strict' if strict else 'default'
    print(
        f'\n=== REAL TPU (TP=4) HARD ROUNDING A/B ({flags} XLA flags) ===\n',
        flush=True,
    )
    print(
        'Cells: `max |diff|` (bitwise-equal %). Regimes 1a-3a per mask;'
        ' `prefill` / `fwd+bwd` compare the `B=4, T=64` logprobs of the mask'
        ' against mask `none`.\n',
        flush=True,
    )
    header = (
        ['HARD_ROUNDING'] + regimes + ['prefill vs none', 'fwd+bwd vs none']
    )
    print('| ' + ' | '.join(header) + ' |', flush=True)
    print('|' + ' :---: |' * len(header), flush=True)
    base_prefill, base_lps, _ = scores_by_mask['none']
    zero_masks = []
    for label, _ in masks:
      rows = diag_by_mask[label]
      prefill, lps, _ = scores_by_mask[label]
      cells = [cell(d.max_abs_diff, d.exact_match_pct) for d in rows]
      cells.append(versus(prefill, base_prefill))
      cells.append(versus(lps, base_lps))
      print(f'| {label} | ' + ' | '.join(cells) + ' |', flush=True)
      if all(
          d.max_abs_diff == 0.0 and d.exact_match_pct == 100.0 for d in rows
      ):
        zero_masks.append(label)
    print(
        f'\nZero-drift masks ({flags} flags): {zero_masks or "none"}',
        flush=True,
    )
    # Cross-process fingerprints (the tokens and weights are deterministic):
    # the strict-flag run and the default-flag run of this test can be
    # compared bit for bit through these.
    print(
        '\n| HARD_ROUNDING | prefill sha256[:16] | fwd+bwd sha256[:16] |'
        ' grad norm |',
        flush=True,
    )
    print('| :---: | :---: | :---: | :---: |', flush=True)
    for label, _ in masks:
      prefill, lps, norm = scores_by_mask[label]
      digests = [
          hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
          for a in (prefill, lps)
      ]
      print(
          f'| {label} | `{digests[0]}` | `{digests[1]}` | `{norm:.6e}` |',
          flush=True,
      )
    print('\n=== END HARD ROUNDING A/B ===\n', flush=True)

    for label, _ in masks:
      rows = diag_by_mask[label]
      self.assertLen(rows, 5, msg=label)  # regimes 1a, 1b, 1c, 2, 3a
      self.assertSetEqual({d.config_name for d in rows}, {zero_tim}, msg=label)
      self.assertTrue(np.all(np.isfinite(scores_by_mask[label][0])), label)
    if strict:
      # The strict flag pins every `astype` already, so a hard rounding point
      # must be a no-op there: bitwise the unmodified program, zero drift.
      for label, _ in masks:
        prefill, lps, norm = scores_by_mask[label]
        test_utils.assert_bitwise_equal(prefill, base_prefill)
        test_utils.assert_bitwise_equal(lps, base_lps)
        np.testing.assert_allclose(norm, scores_by_mask['none'][2], rtol=1e-3)
      self.assertEqual(zero_masks, [label for label, _ in masks])
    else:
      # The property the default rests on: with every group pinned (and
      # already with the two attributed groups) the forward does not depend
      # on the flag. The rest of the table is the attribution and may move
      # with XLA versions.
      self.assertIn('ALL', zero_masks)
      self.assertIn('ROPE + TP_SUM', zero_masks)

  def test_rope_glue_switches_tpu_ab(self):
    """Same-process TPU A/B of the RoPE glue switches (D1 timing, D1b gate).

    Times the three baselines of the `L=8, B=4, T=256` workload under
    (a) the Step-10 glue (`ROPE_FUSE_QK=False`, 64-row padding), (b) the
    default merged q/k program, and (c) the merged program without padding,
    and runs drift regimes 1a-3a for (c). The stock rows do not depend on the
    switches and bound the run-to-run noise. `ROPE_PAD_ROWS=None` may only be
    adopted as default if its regimes are all `0.000e+00` AND its latency is
    not worse; the verdict is printed, the default (b) is asserted zero-drift
    by `test_benchmark_and_ablation_suite_runs_tpu`.

    Result on 4x v5p (Step 11): (a) -> (b) Sample `22.03 -> 21.32 ms`
    (stock `14.32 / 14.31`), Code `34.23 -> 30.68 MB`; (c) would save another
    `0.49 ms` but drifts (`1a-1c`: `max |diff| 1.6e-2 .. 3.1e-2`, `6-8%`
    bitwise), so `ROPE_PAD_ROWS` stays 64. The harness is kept for re-runs on
    new XLA / JAX versions and only runs when `ZERO_TIM_ROPE_AB=1` is set.
    """
    if jax.default_backend() != 'tpu' or len(jax.devices()) < 4:
      self.skipTest('TPU-only timing A/B; the bitwise checks run on CPU.')
    if not os.environ.get('ZERO_TIM_ROPE_AB'):
      self.skipTest(
          'Measured in Step 11 (see docstring); set ZERO_TIM_ROPE_AB=1.'
      )
    workload = _perf_workload('L=8, B=4, T=256 (224+32)')
    variants = (
        ('ROPE_FUSE_QK=False, ROPE_PAD_ROWS=64 (Step-10 glue)', False, 64),
        ('ROPE_FUSE_QK=True, ROPE_PAD_ROWS=64 (default)', True, 64),
        ('ROPE_FUSE_QK=True, ROPE_PAD_ROWS=None (D1b candidate)', True, None),
    )
    zero_tim = 'canon_all_enabled (Zero-TIM)'
    perf_by_variant = {}
    diag_rows = []
    try:
      for label, fuse_qk, pad_rows in variants:
        with mock.patch.object(
            qwen3_canon, 'ROPE_FUSE_QK', fuse_qk
        ), mock.patch.object(qwen3_canon, 'ROPE_PAD_ROWS', pad_rows):
          jax.clear_caches()
          perf_by_variant[label] = (
              qwen3_canon_benchmark.run_performance_scaling_benchmark(
                  tp_size=4,
                  num_iters=5,
                  workloads=(workload,),
                  leave_one_out_workloads=(),
              )
          )
          if pad_rows is None:
            diag_rows = qwen3_canon_benchmark.run_drift_regime_diagnostics(
                preset='qwen3_0p6b_2l',
                batch_size=4,
                prompt_len=48,
                max_new_tokens=16,
                cache_size=128,
                tp_size=4,
                configs=(zero_tim,),
                include_deep_stack=False,
            )
    finally:
      jax.clear_caches()

    print('\n=== REAL TPU (TP=4) ROPE GLUE A/B (D1 / D1b) ===\n', flush=True)
    for label, rows in perf_by_variant.items():
      print(f'#### {label}\n', flush=True)
      print(
          qwen3_canon_benchmark.format_performance_scaling_report(rows),
          flush=True,
      )
      print('', flush=True)
    print('#### Drift regimes with ROPE_PAD_ROWS=None\n', flush=True)
    print(
        qwen3_canon_benchmark.format_drift_regime_report(diag_rows), flush=True
    )
    d1b_zero = all(
        d.max_abs_diff == 0.0 and d.exact_match_pct == 100.0 for d in diag_rows
    )
    print(
        f'\nD1b drift verdict: {"ZERO-DRIFT" if d1b_zero else "DRIFTS"}'
        ' (adopt only if also not slower)',
        flush=True,
    )
    print('\n=== END ROPE GLUE A/B ===\n', flush=True)

    self.assertLen(diag_rows, 5)  # regimes 1a, 1b, 1c, 2, 3a
    self.assertSetEqual({d.config_name for d in diag_rows}, {zero_tim})
    for label, rows in perf_by_variant.items():
      self.assertLen(rows, 3, msg=label)
      self.assertEqual(rows[-1].config_name, zero_tim, msg=label)
      for r in rows:
        self.assertGreater(r.sample_latency_ms, 0.0, msg=f'{label}: {r}')
        self.assertGreater(r.prefill_latency_ms, 0.0, msg=f'{label}: {r}')
        self.assertGreater(r.fwdbwd_latency_ms, 0.0, msg=f'{label}: {r}')

  def test_forward_tp_sum_scatter_tpu_ab(self):
    """Same-process TPU A/B + bitwise guard of the SCATTER default (`B=32`).

    Times `canon_all_enabled` on the `L=8, B=32, T=256` workload under (a)
    forced GATHER (`FORWARD_TP_SUM_SCATTER_MIN_M=None`, the Step-11 program)
    and (b) the default (SCATTER at every contract-parallel site of at least
    4096 rows: the `B=32` prefill / Fwd+Bwd sites, `M = 7168 / 8192`; the
    decode steps, `M=32`, stay GATHER). The stock rows of the main scaling
    table bound the run-to-run noise. Both programs must produce bitwise
    identical prefill and Fwd+Bwd logprobs on the TPU: the CPU test proves
    the glue in interpret mode, but on TPU the collective form changes the
    fusion boundaries around the bf16 rank-ordered adds, and the Zero-TIM
    suite's `B=4, T=64` workloads never reach the threshold -- this test is
    the only place the default's SCATTER sites are checked against GATHER.

    Result on 4x v5p (Step 12): (a) -> (b) Sample `43.94 -> 41.82 ms`, Prefill
    `14.38 -> 11.73`, Fwd+Bwd `31.64 -> 29.16` (stock `21.19 / 7.08 / 15.29`),
    bitwise EQUAL, Code smaller in all three programs, Arena
    `+0.03 / +0.02 / +4.00 MB` (one `[M/TP, N]` bf16 shard in Fwd+Bwd);
    accepted as the default with the `+4 MB` as the known cost.
    """
    if jax.default_backend() != 'tpu' or len(jax.devices()) < 4:
      self.skipTest('TPU-only timing A/B; the bitwise check runs on CPU.')
    self.assertEqual(qwen3_canon.FORWARD_TP_SUM_SCATTER_MIN_M, 4096)
    workload = _perf_workload('L=8, B=32, T=256 (224+32)')
    zero_tim = 'canon_all_enabled (Zero-TIM)'
    variants = (
        ('FORWARD_TP_SUM_SCATTER_MIN_M=None (forced GATHER, Step-11)', None),
        ('FORWARD_TP_SUM_SCATTER_MIN_M=4096 (default, SCATTER M>=4096)', 4096),
    )
    tp_size = 4
    config = qwen3_canon_benchmark.build_model_config(
        'qwen3_0p6b_2l', tp_size=tp_size
    )
    config.num_layers = workload.num_layers
    full_tokens = jax.random.randint(
        jax.random.PRNGKey(2026),
        (workload.batch_size, workload.prompt_len + workload.max_new_tokens),
        0,
        config.vocab_size,
        dtype=jnp.int32,
    )
    perf_by_variant = {}
    outputs = {}
    try:
      for label, min_m in variants:
        with mock.patch.object(
            qwen3_canon, 'FORWARD_TP_SUM_SCATTER_MIN_M', min_m
        ), mock.patch.dict(
            # The B2 dump of this workload belongs to the main scaling table.
            os.environ,
            {qwen3_canon_benchmark.HBM_DUMP_ENV: ''},
        ):
          jax.clear_caches()
          perf_by_variant[label] = (
              qwen3_canon_benchmark.run_performance_scaling_benchmark(
                  tp_size=tp_size,
                  num_iters=5,
                  workloads=(workload,),
                  leave_one_out_workloads=(),
                  configs=(zero_tim,),
              )
          )
          with _tp_mesh(tp_size):
            canon = qwen3_canon.Qwen3Canon(
                config,
                rngs=nnx.Rngs(params=0),
                canon_config=qwen3_canon.CanonKernelConfig.all_enabled(),
            )
            qwen3_canon.init_non_zero_mlp_weights(canon, seed=123)
            prefill = qwen3_canon.score_sequence_prefill(
                canon, full_tokens, use_canon_logsoftmax=True
            )
            lps, norm = qwen3_canon.score_sequence_fwd_bwd(
                canon, full_tokens, use_canon_logsoftmax=True
            )
            outputs[label] = (np.asarray(prefill), np.asarray(lps), float(norm))
            del canon
    finally:
      jax.clear_caches()

    print('\n=== REAL TPU (TP=4) TP-SUM SCATTER A/B (B=32) ===\n', flush=True)
    for label, rows in perf_by_variant.items():
      print(f'#### {label}\n', flush=True)
      print(
          qwen3_canon_benchmark.format_performance_scaling_report(rows),
          flush=True,
      )
      print('', flush=True)
    items = list(outputs.items())
    self.assertLen(items, 2)
    gather_label, gather_out = items[0]
    scatter_label, scatter_out = items[1]
    bitwise = all(
        np.array_equal(a, b) for a, b in zip(gather_out[:2], scatter_out[:2])
    )
    print(
        f'SCATTER bitwise verdict: {"EQUAL" if bitwise else "DIFFERS"}'
        f' (prefill + Fwd+Bwd logprobs, grad norm {gather_out[2]:.6e} vs'
        f' {scatter_out[2]:.6e}); the default (b) must stay EQUAL to the'
        ' forced-GATHER program (a)',
        flush=True,
    )
    print('\n=== END TP-SUM SCATTER A/B ===\n', flush=True)

    self.assertEqual(gather_label, variants[0][0])
    self.assertEqual(scatter_label, variants[1][0])
    test_utils.assert_bitwise_equal(scatter_out[0], gather_out[0])
    test_utils.assert_bitwise_equal(scatter_out[1], gather_out[1])
    # Gradient bits are not part of the contract (the forward collectives may
    # change how XLA fuses the backward); same gate as the other switches.
    np.testing.assert_allclose(scatter_out[2], gather_out[2], rtol=1e-3)
    for label, rows in perf_by_variant.items():
      self.assertLen(rows, 1, msg=label)
      (r,) = rows
      self.assertEqual(r.config_name, zero_tim, msg=label)
      self.assertEqual(r.workload, workload.label, msg=label)
      self.assertGreater(r.sample_latency_ms, 0.0, msg=f'{label}: {r}')
      self.assertGreater(r.prefill_latency_ms, 0.0, msg=f'{label}: {r}')
      self.assertGreater(r.fwdbwd_latency_ms, 0.0, msg=f'{label}: {r}')

  def test_leave_one_out_b32_tpu(self):
    """Leave-One-Out attribution of the `B=32` stock / canon gap (Step 12).

    The main scaling table shows the Zero-TIM cost growing with the batch
    (`canon / stock` Sample latency `1.50x -> 1.81x -> 2.09x` at
    `B=4 / 16 / 32`): a steady-state throughput gap that the `B=4`
    Leave-One-Out rows -- which attribute a launch-bound gap to ~3-4 us per
    kernel call site -- do not transfer to. This times `canon_all_enabled`
    and every `without_<switch>` configuration on the `L=8, B=32, T=256`
    workload in one process, so each per-switch delta is read against a
    same-process baseline (the run-to-run noise is bounded by the stock rows
    of the main table, timed in the other shard). The rows are timing / HBM
    only -- the Zero-TIM contract of each switch is checked by
    `run_benchmark_suite`. It is its own test method so that, with
    `shard_count = 2`, each shard of the TPU target stays inside the test
    runner's 60-minute timeout (the sorted round-robin puts this method on
    the shard without the suite test).
    """
    if jax.default_backend() != 'tpu' or len(jax.devices()) < 4:
      self.skipTest('TPU-only timing attribution.')
    workload = _perf_workload('L=8, B=32, T=256 (224+32)')
    zero_tim = 'canon_all_enabled (Zero-TIM)'
    loo_configs = tuple(
        f'without_{switch}' for switch in qwen3_canon.ABLATION_SWITCH_NAMES
    )
    try:
      with mock.patch.dict(
          # The B2 dump of this workload belongs to the main scaling table.
          os.environ,
          {qwen3_canon_benchmark.HBM_DUMP_ENV: ''},
      ):
        rows = qwen3_canon_benchmark.run_performance_scaling_benchmark(
            tp_size=4,
            num_iters=5,
            workloads=(workload,),
            leave_one_out_workloads=(workload.label,),
            configs=(zero_tim,) + loo_configs,
        )
    finally:
      jax.clear_caches()

    print('\n=== REAL TPU (TP=4) LEAVE-ONE-OUT (B=32) REPORT ===\n', flush=True)
    print(
        qwen3_canon_benchmark.format_performance_scaling_report(rows),
        flush=True,
    )
    print('\n=== END LEAVE-ONE-OUT (B=32) REPORT ===\n', flush=True)

    self.assertLen(rows, 1 + len(loo_configs))
    self.assertEqual(rows[0].config_name, zero_tim)
    self.assertEqual(rows[0].category, 'baseline')
    self.assertSetEqual({r.config_name for r in rows[1:]}, set(loo_configs))
    self.assertSetEqual({r.category for r in rows[1:]}, {'leave_one_out'})
    for r in rows:
      self.assertEqual(r.workload, workload.label, msg=str(r))
      self.assertGreater(r.sample_latency_ms, 0.0, msg=str(r))
      self.assertGreater(r.prefill_latency_ms, 0.0, msg=str(r))
      self.assertGreater(r.fwdbwd_latency_ms, 0.0, msg=str(r))
      for prog in ('sample', 'prefill', 'fwdbwd'):
        arena = getattr(r, f'{prog}_arena_hbm_mb')
        code = getattr(r, f'{prog}_code_hbm_mb')
        temp = getattr(r, f'{prog}_temp_hbm_mb')
        self.assertGreater(arena, 0.0, msg=f'{prog}: {r}')
        self.assertGreater(code, 0.0, msg=f'{prog}: {r}')
        self.assertLessEqual(arena, temp + 1e-6, msg=f'{prog}: {r}')

  def _assert_benchmark_suite_rows(self, rows):
    """Stock parity and Zero-TIM invariance assertions shared by the suites."""
    by_name = {r.name: r for r in rows}

    # Verify all-disabled has 0 diff vs stock
    self.assertEqual(by_name['canon_all_disabled'].vs_stock_max_abs_diff, 0.0)

    # Verify full Zero-TIM achieves 0.0 Decode-vs-Prefill, B=1-vs-B=N and
    # Fwd-vs-FwdBwd diff.
    full_row = by_name['canon_all_enabled (Zero-TIM)']
    self.assertEqual(full_row.decode_vs_prefill_max_diff, 0.0)
    self.assertEqual(full_row.decode_vs_prefill_exact_pct, 100.0)
    self.assertEqual(full_row.batch1_vs_batchn_max_diff, 0.0)
    self.assertEqual(full_row.batch1_vs_batchn_exact_pct, 100.0)
    self.assertEqual(full_row.fwd_vs_fwdbwd_max_diff, 0.0)
    self.assertEqual(full_row.fwd_vs_fwdbwd_exact_pct, 100.0)

  def test_benchmark_and_ablation_suite_runs_tpu(self):
    """Full-size suite, drift-regime and scaling reports on 4 TPU devices."""
    if jax.default_backend() != 'tpu' or len(jax.devices()) < 4:
      self.skipTest('Requires 4 TPU devices; the mini suite covers CPU.')
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
    print('\n=== REAL TPU (TP=4) BENCHMARK & ABLATION REPORT ===\n', flush=True)
    print(report, flush=True)
    print('\n=== END REAL TPU REPORT ===\n', flush=True)
    # stock + all_disabled + all_enabled, then one Add-One-In and one
    # Leave-One-Out row per switch.
    self.assertLen(rows, 3 + 2 * len(qwen3_canon.ABLATION_SWITCH_NAMES))
    # Contract first: a perf-table failure below must not mask it.
    self._assert_benchmark_suite_rows(rows)

    diag_rows = qwen3_canon_benchmark.run_drift_regime_diagnostics(
        preset='qwen3_0p6b_2l',
        batch_size=4,
        prompt_len=48,
        max_new_tokens=16,
        cache_size=128,
        tp_size=4,
    )
    diag_report = qwen3_canon_benchmark.format_drift_regime_report(diag_rows)
    print(
        '\n=== REAL TPU (TP=4) DRIFT REGIME DIAGNOSTIC REPORT ===\n',
        flush=True,
    )
    print(diag_report, flush=True)
    print('\n=== END DRIFT REGIME DIAGNOSTIC REPORT ===\n', flush=True)
    for d in diag_rows:
      if (
          'canon_all_enabled' in d.config_name
          or 'Zero-TIM F4 tree' in d.config_name
      ):
        self.assertEqual(d.max_abs_diff, 0.0, msg=f'Failed 0 diff for {d}')
        self.assertEqual(d.exact_match_pct, 100.0, msg=f'Failed 100% for {d}')

    perf_rows = qwen3_canon_benchmark.run_performance_scaling_benchmark(
        tp_size=4,
        num_iters=5,
    )
    perf_report = qwen3_canon_benchmark.format_performance_scaling_report(
        perf_rows
    )
    print(
        '\n=== REAL TPU (TP=4) HBM & LATENCY SCALING REPORT ===\n', flush=True
    )
    print(perf_report, flush=True)
    print('\n=== END HBM & LATENCY SCALING REPORT ===\n', flush=True)
    # 3 baselines per workload + one timing-only Leave-One-Out row per
    # switch on the designated large workload.
    num_workloads = len(qwen3_canon_benchmark.DEFAULT_PERF_WORKLOADS)
    num_loo = len(
        qwen3_canon_benchmark.DEFAULT_LEAVE_ONE_OUT_PERF_WORKLOADS
    ) * len(qwen3_canon.ABLATION_SWITCH_NAMES)
    self.assertLen(perf_rows, 3 * num_workloads + num_loo)
    loo_rows = [r for r in perf_rows if r.category == 'leave_one_out']
    self.assertLen(loo_rows, num_loo)
    self.assertSetEqual(
        {r.workload for r in loo_rows},
        set(qwen3_canon_benchmark.DEFAULT_LEAVE_ONE_OUT_PERF_WORKLOADS),
    )
    for r in perf_rows:
      # Buffer-assignment based HBM metrics are populated on TPU. The HLO
      # temp arena is bounded by the legacy PJRT `temp_size_in_bytes`,
      # which double counts the arena's fragmentation.
      for prog in ('sample', 'prefill', 'fwdbwd'):
        arena = getattr(r, f'{prog}_arena_hbm_mb')
        code = getattr(r, f'{prog}_code_hbm_mb')
        temp = getattr(r, f'{prog}_temp_hbm_mb')
        self.assertGreater(arena, 0.0, msg=f'{prog}: {r}')
        self.assertGreater(code, 0.0, msg=f'{prog}: {r}')
        self.assertLessEqual(arena, temp + 1e-6, msg=f'{prog}: {r}')

  def test_production_vocab_zero_tim_tpu(self):
    """Zero-TIM contract at the production vocabulary and learner row buckets.

    Every other TPU table runs `qwen3_0p6b_2l` (`V=8192`, at most `B*T=1024`
    scoring rows). Qwen3's production vocabulary (`V=151936`, preset
    `qwen3_0p6b_2l_fullvocab`) takes four code paths those tables never reach
    on TPU: the partial last column group of the library log-softmax
    (`151936 = 18 * 8192 + 4480`), the gathered fallback of
    `compute_log_softmax` / `compute_token_logprobs` (`V/TP = 37984` is not a
    whole number of 1024-column tiles, so the vocab-parallel path is not
    admitted) with its 256-row `lax.map` chunks (16 at `M=4096`), the fixed
    LM head's `N = 37984 -> 38144` column padding, and decode (`M=B`) versus
    prefill (`M=B*T`) rows on opposite sides of the LM-head row chunking and
    of the `FORWARD_TP_SUM_SCATTER_MIN_M=4096` threshold.

    Runs drift regimes 1a-3a of the four standard configurations at `B=8` and
    `B=16` with `T=256 (224+32)` -- the learner row buckets `M=2048` and
    `M=4096` -- and the ablation suite at `B=16, T=256` restricted to the
    three switches whose contribution the vocabulary can change
    (`use_fixed_lm_head`, `use_canon_logsoftmax`, `use_fixed_order_reduce`:
    3 baselines + 3 Leave-One-Out + 3 Add-One-In rows). Every table is
    printed as soon as it exists, followed by sha256 fingerprints of the
    `canon_all_enabled` `B=16, T=256` prefill / fwd+bwd logprobs for
    cross-process comparison (e.g. against a strict-flag re-run). Asserted:
    every `canon_all_enabled` regime is `0.000e+00` / 100% at both row
    buckets and the suite's stock-parity / Zero-TIM invariants hold
    (`_assert_benchmark_suite_rows`); the Leave-One-Out / Add-One-In drifts
    are the measurement (each switch's contribution at the production
    vocabulary) and are not asserted.

    TPU-only; runs only when `ZERO_TIM_PRODUCTION_VOCAB=1` is set (about 100
    compiles, ~25 min on 4x v5p; run it filtered to this method).
    """
    if jax.default_backend() != 'tpu' or len(jax.devices()) < 4:
      self.skipTest('TPU-only production-vocabulary measurement.')
    if not os.environ.get('ZERO_TIM_PRODUCTION_VOCAB'):
      self.skipTest('Set ZERO_TIM_PRODUCTION_VOCAB=1 (see docstring).')
    preset = 'qwen3_0p6b_2l_fullvocab'
    tp_size = 4
    prompt_len, max_new_tokens, cache_size = 224, 32, 256
    total_len = prompt_len + max_new_tokens
    switches = (
        'use_fixed_lm_head',
        'use_canon_logsoftmax',
        'use_fixed_order_reduce',
    )
    zero_tim = 'canon_all_enabled (Zero-TIM)'
    regime_configs = {
        'stock_qwen3 (model.py)',
        'canon_all_disabled',
        'only_use_fixed_order_reduce',
        zero_tim,
    }
    config = qwen3_canon_benchmark.build_model_config(preset, tp_size=tp_size)
    self.assertEqual(config.vocab_size, qwen3_canon_benchmark.PRODUCTION_VOCAB)
    self.assertFalse(
        qwen3_canon.vocab_parallel_logsoftmax_admitted(
            config.vocab_size, tp_size
        )
    )
    print(
        f'\n=== REAL TPU (TP=4) PRODUCTION VOCABULARY (V={config.vocab_size})'
        ' DRIFT REGIMES ===\n',
        flush=True,
    )
    diag_by_batch = {}
    try:
      for batch_size in (8, 16):
        diag_rows = qwen3_canon_benchmark.run_drift_regime_diagnostics(
            preset=preset,
            batch_size=batch_size,
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
            cache_size=cache_size,
            tp_size=tp_size,
            include_deep_stack=False,
        )
        diag_by_batch[batch_size] = diag_rows
        print(
            f'\nB={batch_size}, T={total_len}'
            f' (M={batch_size * total_len} scoring rows):\n',
            flush=True,
        )
        print(
            qwen3_canon_benchmark.format_drift_regime_report(diag_rows),
            flush=True,
        )
      rows = qwen3_canon_benchmark.run_benchmark_suite(
          preset=preset,
          batch_size=16,
          prompt_len=prompt_len,
          max_new_tokens=max_new_tokens,
          cache_size=cache_size,
          num_iters=5,
          run_ablations=True,
          include_fwdbwd=True,
          switches=switches,
          tp_size=tp_size,
      )
      print(
          f'\n=== REAL TPU (TP=4) PRODUCTION VOCABULARY (V={config.vocab_size})'
          f' ABLATION REPORT (B=16, T={total_len}) ===\n',
          flush=True,
      )
      print(qwen3_canon_benchmark.format_markdown_report(rows), flush=True)
      # Cross-process fingerprints (deterministic tokens and weights).
      tokens = jax.random.randint(
          jax.random.PRNGKey(2026),
          (16, total_len),
          0,
          config.vocab_size,
          dtype=jnp.int32,
      )
      with _tp_mesh(tp_size):
        canon = qwen3_canon.Qwen3Canon(
            config,
            rngs=nnx.Rngs(params=0),
            canon_config=qwen3_canon.CanonKernelConfig.all_enabled(),
        )
        qwen3_canon.init_non_zero_mlp_weights(canon, seed=123)
        prefill = np.asarray(
            qwen3_canon.score_sequence_prefill(
                canon, tokens, use_canon_logsoftmax=True
            )
        )
        lps, norm = qwen3_canon.score_sequence_fwd_bwd(
            canon, tokens, use_canon_logsoftmax=True
        )
        lps = np.asarray(lps)
        norm = float(norm)
        del canon
    finally:
      jax.clear_caches()

    digests = [
        hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
        for a in (prefill, lps)
    ]
    print(
        f'\n| canon_all_enabled, B=16, T={total_len} | sha256[:16] |\n'
        '| :--- | :---: |\n'
        f'| prefill logprobs | `{digests[0]}` |\n'
        f'| fwd+bwd logprobs | `{digests[1]}` |\n'
        f'| fwd+bwd grad norm | `{norm:.6e}` |',
        flush=True,
    )
    print('\n=== END PRODUCTION VOCABULARY REPORT ===\n', flush=True)

    for batch_size, diag_rows in diag_by_batch.items():
      msg = f'B={batch_size}'
      self.assertLen(diag_rows, 5 * len(regime_configs), msg=msg)
      self.assertSetEqual(
          {d.config_name for d in diag_rows}, regime_configs, msg=msg
      )
      zero_tim_rows = [d for d in diag_rows if d.config_name == zero_tim]
      self.assertEqual(
          [d.regime[:3] for d in zero_tim_rows],
          ['1a.', '1b.', '1c.', '2. ', '3a.'],
          msg=msg,
      )
      self.assertIn(
          f'Prefill(B={batch_size},T={total_len})', zero_tim_rows[1].regime
      )
      for d in zero_tim_rows:
        self.assertEqual(d.max_abs_diff, 0.0, msg=f'{msg}: {d}')
        self.assertEqual(d.exact_match_pct, 100.0, msg=f'{msg}: {d}')
    self.assertLen(rows, 3 + 2 * len(switches))
    self._assert_benchmark_suite_rows(rows)
    self.assertTrue(np.all(np.isfinite(prefill)))
    self.assertTrue(np.all(np.isfinite(lps)))
    self.assertTrue(np.isfinite(norm))

  def test_production_vocab_hbm_latency_tpu(self):
    """HBM / latency of the LM head -> log-softmax at the production vocabulary.

    Times the three baselines and the `without_use_fixed_lm_head` /
    `without_use_canon_logsoftmax` Leave-One-Out configurations on the
    `PRODUCTION_VOCAB_PERF_WORKLOADS` (`L=2`, `B=8` and `B=16` at `T=256`:
    `M=2048` / `4096` scoring rows) for (a) `qwen3_0p6b_2l_fullvocab`
    (`V=151936`: gathered log-softmax fallback, LM head padded to 38144
    columns), (b) `qwen3_0p6b_2l_fullvocab_aligned` (`V=155648 = 38 * 4096`:
    vocab-parallel log-softmax, unpadded LM head) and (c) the `qwen3_0p6b_2l`
    (`V=8192`) control on the `B=16` workload, in one process. (a) against
    (c) isolates the vocabulary's share of the stock / Zero-TIM arena and
    latency gap: on the gathered path the canonical program materializes the
    replicated `[M, V]` logits in bf16 and f32 (`1.2 GB` + `2.5 GB` per rank
    at `M=4096`) plus the bf16 residual and f32 cotangent of the fwd+bwd
    program, where the stock model keeps its f32 logits `V/TP`-sharded; that
    gap bounds what a fused LM head + log-softmax can give back. (b) against
    (a) is what aligning the vocabulary to `1024 * TP` buys on today's
    kernels. The rows are timing / HBM only; the Zero-TIM contract at this
    vocabulary is `test_production_vocab_zero_tim_tpu`.

    TPU-only; runs only when `ZERO_TIM_PRODUCTION_VOCAB=1` is set (75
    compiles, ~20 min on 4x v5p; run it filtered to this method).
    """
    if jax.default_backend() != 'tpu' or len(jax.devices()) < 4:
      self.skipTest('TPU-only production-vocabulary measurement.')
    if not os.environ.get('ZERO_TIM_PRODUCTION_VOCAB'):
      self.skipTest('Set ZERO_TIM_PRODUCTION_VOCAB=1 (see docstring).')
    tp_size = 4
    workloads = qwen3_canon_benchmark.PRODUCTION_VOCAB_PERF_WORKLOADS
    loo_switches = ('use_fixed_lm_head', 'use_canon_logsoftmax')
    config_names = (
        'stock_qwen3 (model.py)',
        'canon_all_disabled',
        'canon_all_enabled (Zero-TIM)',
    ) + tuple(f'without_{switch}' for switch in loo_switches)
    tables = (
        ('qwen3_0p6b_2l_fullvocab', workloads),
        ('qwen3_0p6b_2l_fullvocab_aligned', workloads),
        ('qwen3_0p6b_2l', workloads[-1:]),
    )
    print(
        '\n=== REAL TPU (TP=4) PRODUCTION VOCABULARY HBM & LATENCY REPORT'
        ' ===\n',
        flush=True,
    )
    rows_by_preset = {}
    try:
      with mock.patch.dict(
          # The B2 dump belongs to the main scaling table.
          os.environ,
          {qwen3_canon_benchmark.HBM_DUMP_ENV: ''},
      ):
        for preset, preset_workloads in tables:
          rows = qwen3_canon_benchmark.run_performance_scaling_benchmark(
              tp_size=tp_size,
              num_iters=5,
              preset=preset,
              workloads=preset_workloads,
              leave_one_out_workloads=tuple(w.label for w in preset_workloads),
              leave_one_out_switches=loo_switches,
          )
          rows_by_preset[preset] = rows
          vocab = qwen3_canon_benchmark.build_model_config(
              preset, tp_size=tp_size
          ).vocab_size
          admitted = qwen3_canon.vocab_parallel_logsoftmax_admitted(
              vocab, tp_size
          )
          print(
              f'\n`{preset}` (V={vocab}, vocab-parallel log-softmax'
              f' admitted: {admitted}):\n',
              flush=True,
          )
          print(
              qwen3_canon_benchmark.format_performance_scaling_report(rows),
              flush=True,
          )
          print('\nDeltas against the stock row:\n', flush=True)
          print(_perf_delta_table(rows), flush=True)
          jax.clear_caches()
    finally:
      jax.clear_caches()
    print(
        '\n=== END PRODUCTION VOCABULARY HBM & LATENCY REPORT ===\n', flush=True
    )

    for preset, preset_workloads in tables:
      rows = rows_by_preset[preset]
      expected_rows = []
      for w in preset_workloads:
        expected_rows.extend((w.label, name) for name in config_names)
      self.assertEqual(
          [(r.workload, r.config_name) for r in rows], expected_rows, msg=preset
      )
      for r in rows:
        expected_category = (
            'leave_one_out'
            if r.config_name.startswith('without_')
            else 'baseline'
        )
        self.assertEqual(r.category, expected_category, msg=str(r))
        self.assertGreater(r.sample_latency_ms, 0.0, msg=str(r))
        self.assertGreater(r.prefill_latency_ms, 0.0, msg=str(r))
        self.assertGreater(r.fwdbwd_latency_ms, 0.0, msg=str(r))
        for prog in ('sample', 'prefill', 'fwdbwd'):
          arena = getattr(r, f'{prog}_arena_hbm_mb')
          code = getattr(r, f'{prog}_code_hbm_mb')
          temp = getattr(r, f'{prog}_temp_hbm_mb')
          self.assertGreater(arena, 0.0, msg=f'{prog}: {r}')
          self.assertGreater(code, 0.0, msg=f'{prog}: {r}')
          self.assertLessEqual(arena, temp + 1e-6, msg=f'{prog}: {r}')

  def test_decode_vs_prefill_attribution_tpu(self):
    """Attribution harness for a decode-vs-prefill drift at a new geometry.

    `test_production_vocab_zero_tim_tpu` found the `canon_all_enabled`
    KV-cached decode drifting from the cacheless prefill at `V=151936`,
    `T=256 (224+32)`, cache 256 under the TPU default flag (regime `1a`:
    `1.2e-04` / 96.9% bitwise at `B=1`; `1c`: `1.6e-02` / 63% at `B=8`,
    `3.1e-02` / 66% at `B=16`) while the prefill stayed batch-invariant and
    bitwise equal to fwd+bwd; the Zero-TIM suite had only ever checked the
    contract at `V=8192`, `T=64 (48+16)`, cache 128, `B<=4`. Three
    measurements, printed as tables and not asserted (the tables are the
    measurement; run it under both `--xla_allow_excess_precision` settings):

    1. `_attention_decode_vs_prefill`: the attention core alone, model-free,
       in the sampler-prefill and decode program shapes against the scoring
       prefill, at both geometries and `B in {1, 4, 8, 16}`.
    2. Drift regimes of `canon_all_enabled` at `B=16` for the three cells the
       production-vocabulary run did not cover: `V=8192` at the suite geometry
       (batch alone), `V=8192` at the item-2 geometry (geometry without the
       vocabulary) and `V=151936` at the suite geometry (vocabulary without
       the geometry).
    3. `_decode_step_diffs`: per decode step, the sampler's logits and
       logprobs against the scoring prefill's rows (`V=151936` at `B=1` and
       `B=16`, `V=8192` at `B=16`; item-2 geometry), preceded by the sampler
       prefill's prompt logprobs against the scoring prefill's prompt rows.

    TPU-only; runs only when `ZERO_TIM_DECODE_PREFILL_AB=1` is set (~40
    compiles, ~10 min on 4x v5p; run it filtered to this method).
    """
    if jax.default_backend() != 'tpu' or len(jax.devices()) < 4:
      self.skipTest('TPU-only attribution harness.')
    if not os.environ.get('ZERO_TIM_DECODE_PREFILL_AB'):
      self.skipTest('Set ZERO_TIM_DECODE_PREFILL_AB=1 (see docstring).')
    tp_size = 4
    zero_tim = 'canon_all_enabled (Zero-TIM)'
    # (label, prompt_len, max_new_tokens, cache_size)
    suite_geo = ('T=64 (48+16), cache 128 [suite]', 48, 16, 128)
    item2_geo = ('T=256 (224+32), cache 256 [item 2]', 224, 32, 256)
    regime_cells = (
        ('qwen3_0p6b_2l', suite_geo),
        ('qwen3_0p6b_2l', item2_geo),
        ('qwen3_0p6b_2l_fullvocab', suite_geo),
    )
    step_probes = (
        ('qwen3_0p6b_2l_fullvocab', 1),
        ('qwen3_0p6b_2l_fullvocab', 16),
        ('qwen3_0p6b_2l', 16),
    )
    attention_rows = []
    regime_tables = []
    step_tables = []
    try:
      for label, prompt_len, max_new_tokens, cache_size in (
          suite_geo,
          item2_geo,
      ):
        for batch_size in (1, 4, 8, 16):
          for d in _attention_decode_vs_prefill(
              batch_size=batch_size,
              seq_len=prompt_len + max_new_tokens,
              prompt_len=prompt_len,
              cache_size=cache_size,
              tp_size=tp_size,
          ):
            attention_rows.append((label, batch_size, d))
      jax.clear_caches()
      for preset, (
          label,
          prompt_len,
          max_new_tokens,
          cache_size,
      ) in regime_cells:
        rows = qwen3_canon_benchmark.run_drift_regime_diagnostics(
            preset=preset,
            batch_size=16,
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
            cache_size=cache_size,
            tp_size=tp_size,
            configs=(zero_tim,),
            include_deep_stack=False,
        )
        regime_tables.append((preset, label, rows))
        jax.clear_caches()
      _, prompt_len, max_new_tokens, cache_size = item2_geo
      for preset, batch_size in step_probes:
        config = qwen3_canon_benchmark.build_model_config(
            preset, tp_size=tp_size
        )
        prompt = jax.random.randint(
            jax.random.PRNGKey(2026),
            (batch_size, prompt_len),
            0,
            config.vocab_size,
            dtype=jnp.int32,
        )
        with _tp_mesh(tp_size):
          canon = qwen3_canon.Qwen3Canon(
              config,
              rngs=nnx.Rngs(params=0),
              canon_config=qwen3_canon.CanonKernelConfig.all_enabled(),
          )
          qwen3_canon.init_non_zero_mlp_weights(canon, seed=123)
          steps, prompt_rows = _decode_step_diffs(
              canon,
              prompt,
              max_new_tokens=max_new_tokens,
              cache_size=cache_size,
          )
          del canon
        step_tables.append((preset, batch_size, steps, prompt_rows))
        jax.clear_caches()
    finally:
      jax.clear_caches()

    row_header = (
        '| Geometry | B | Program | Rows differing / compared | Max abs diff |'
        ' Steps differing |\n| :--- | :---: | :--- | :---: | :---: | :--- |'
    )
    print(
        '\n=== REAL TPU (TP=4) DECODE-VS-PREFILL ATTRIBUTION ===\n', flush=True
    )
    print(
        '#### 1. Attention core alone (random operands, Qwen3-0.6B heads,'
        ' TP=4)\n',
        flush=True,
    )
    print(row_header, flush=True)
    for label, batch_size, d in attention_rows:
      print(f'| `{label}` | `{batch_size}` | {d.cells()} |', flush=True)
    for preset, label, rows in regime_tables:
      print(
          f'\n#### 2. Drift regimes of `{preset}`, B=16, {label}\n', flush=True
      )
      print(qwen3_canon_benchmark.format_drift_regime_report(rows), flush=True)
    for preset, batch_size, steps, prompt_rows in step_tables:
      print(
          f'\n#### 3. Decode steps of `{preset}`, B={batch_size},'
          f' {item2_geo[0]}\n',
          flush=True,
      )
      print(row_header, flush=True)
      print(
          f'| `{item2_geo[0]}` | `{batch_size}` | {prompt_rows.cells()} |\n',
          flush=True,
      )
      print(_format_step_diffs(steps, batch_size), flush=True)
    print('\n=== END DECODE-VS-PREFILL ATTRIBUTION ===\n', flush=True)

    self.assertLen(attention_rows, 2 * 4 * 2)
    for _, _, d in attention_rows:
      self.assertGreater(d.rows_compared, 0, msg=str(d))
      self.assertLessEqual(d.rows_differing, d.rows_compared, msg=str(d))
    self.assertLen(regime_tables, len(regime_cells))
    for _, _, rows in regime_tables:
      self.assertLen(rows, 5)  # regimes 1a, 1b, 1c, 2, 3a
      self.assertSetEqual({d.config_name for d in rows}, {zero_tim})
    self.assertLen(step_tables, len(step_probes))
    for _, batch_size, steps, prompt_rows in step_tables:
      self.assertLen(steps, item2_geo[2])
      for s in steps:
        self.assertLessEqual(s.logit_rows_differing, batch_size, msg=str(s))
        self.assertLessEqual(s.logprobs_differing, batch_size, msg=str(s))
      self.assertEqual(
          prompt_rows.rows_compared, batch_size * (item2_geo[1] - 1)
      )

  def test_zero_tim_long_sequence_multi_block_tpu(self):
    """Zero-TIM contract and cost once attention spans several KV blocks.

    Every other table keeps the keys inside one fixed 512-key block
    (`T <= 256`, cache <= 256): the block map of the Pallas attention forward
    never skips a fully masked key block, and no row is ever rescaled across
    blocks. The kernel tests cover the multi-block path on random operands
    (`s=600`, `t=384`); this is the model-level check at the Qwen3-0.6B head
    geometry in the sampler's program shapes, with the cache sized to the
    sequence (`T=1024`: 2 blocks, `T=2048`: 4 blocks) and once oversized
    (`T=1024` in a 2048-slot cache: the last two key blocks are fully masked
    in every program and must be exact no-ops). Three measurements, printed
    as tables:

    1. `_attention_decode_vs_prefill`: the attention core alone at the three
       geometries and `B in {1, 4}`; asserted: no differing row.
    2. Drift regimes 1a-3a of `qwen3_0p6b_2l` (`V=8192`) at `B=4`: the four
       standard configurations at `T=1024 (992+32)`, cache 1024 (the stock
       rows are the long-sequence baseline drift), `canon_all_enabled` alone
       at `T=2048 (2016+32)`, cache 2048 and at `T=1024` in the 2048-slot
       cache; asserted: every `canon_all_enabled` regime is `0.000e+00` /
       100%.
    3. Latency / HBM of `LONG_SEQUENCE_PERF_WORKLOADS` (`L=8, B=4`,
       `T = 256 .. 2048`, 1-4 key blocks) for the three baselines and the
       `without_use_canon_attention` Leave-One-Out row on every workload;
       timing only, not asserted beyond shape and positivity.

    TPU-only; runs only when `ZERO_TIM_LONG_SEQ=1` is set (~70 compiles, ~13
    min on 4x v5p; run it filtered to this method).
    """
    if jax.default_backend() != 'tpu' or len(jax.devices()) < 4:
      self.skipTest('TPU-only long-sequence measurement.')
    if not os.environ.get('ZERO_TIM_LONG_SEQ'):
      self.skipTest('Set ZERO_TIM_LONG_SEQ=1 (see docstring).')
    tp_size = 4
    preset = 'qwen3_0p6b_2l'
    batch_size = 4
    zero_tim = 'canon_all_enabled (Zero-TIM)'
    all_configs = (
        'stock_qwen3 (model.py)',
        'canon_all_disabled',
        'only_use_fixed_order_reduce',
        zero_tim,
    )
    # (label, prompt_len, max_new_tokens, cache_size)
    two_blocks = ('T=1024 (992+32), cache 1024 [2 KV blocks]', 992, 32, 1024)
    oversized = (
        'T=1024 (992+32), cache 2048 [2 of 4 KV blocks live]',
        992,
        32,
        2048,
    )
    four_blocks = ('T=2048 (2016+32), cache 2048 [4 KV blocks]', 2016, 32, 2048)
    geometries = (two_blocks, oversized, four_blocks)
    regime_cells = (
        (two_blocks, None),  # all four configurations
        (four_blocks, (zero_tim,)),
        (oversized, (zero_tim,)),
    )
    workloads = qwen3_canon_benchmark.LONG_SEQUENCE_PERF_WORKLOADS
    loo_switches = ('use_canon_attention',)
    perf_config_names = (
        'stock_qwen3 (model.py)',
        'canon_all_disabled',
        zero_tim,
    ) + tuple(f'without_{switch}' for switch in loo_switches)
    attention_rows = []
    regime_tables = []
    perf_rows = []
    try:
      for label, prompt_len, max_new_tokens, cache_size in geometries:
        for b in (1, batch_size):
          for d in _attention_decode_vs_prefill(
              batch_size=b,
              seq_len=prompt_len + max_new_tokens,
              prompt_len=prompt_len,
              cache_size=cache_size,
              tp_size=tp_size,
          ):
            attention_rows.append((label, b, d))
        jax.clear_caches()
      for (
          label,
          prompt_len,
          max_new_tokens,
          cache_size,
      ), configs in regime_cells:
        rows = qwen3_canon_benchmark.run_drift_regime_diagnostics(
            preset=preset,
            batch_size=batch_size,
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
            cache_size=cache_size,
            tp_size=tp_size,
            configs=configs,
            include_deep_stack=False,
        )
        regime_tables.append((label, configs, rows))
        jax.clear_caches()
      with mock.patch.dict(
          # The B2 dump belongs to the main scaling table.
          os.environ,
          {qwen3_canon_benchmark.HBM_DUMP_ENV: ''},
      ):
        perf_rows = qwen3_canon_benchmark.run_performance_scaling_benchmark(
            tp_size=tp_size,
            num_iters=5,
            preset=preset,
            workloads=workloads,
            leave_one_out_workloads=tuple(w.label for w in workloads),
            leave_one_out_switches=loo_switches,
        )
    finally:
      jax.clear_caches()

    row_header = (
        '| Geometry | B | Program | Rows differing / compared | Max abs diff |'
        ' Steps differing |\n| :--- | :---: | :--- | :---: | :---: | :--- |'
    )
    print(
        '\n=== REAL TPU (TP=4) LONG-SEQUENCE / MULTI-KV-BLOCK REPORT ===\n',
        flush=True,
    )
    print(
        '#### 1. Attention core alone (random operands, Qwen3-0.6B heads,'
        ' TP=4)\n',
        flush=True,
    )
    print(row_header, flush=True)
    for label, b, d in attention_rows:
      print(f'| `{label}` | `{b}` | {d.cells()} |', flush=True)
    for label, configs, rows in regime_tables:
      which = 'all configurations' if configs is None else zero_tim
      print(
          f'\n#### 2. Drift regimes of `{preset}`, B={batch_size}, {label},'
          f' {which}\n',
          flush=True,
      )
      print(qwen3_canon_benchmark.format_drift_regime_report(rows), flush=True)
    print(
        f'\n#### 3. Latency / HBM of `{preset}` across sequence lengths'
        ' (attention Leave-One-Out on every workload)\n',
        flush=True,
    )
    print(
        qwen3_canon_benchmark.format_performance_scaling_report(perf_rows),
        flush=True,
    )
    print('\nDeltas against the stock row:\n', flush=True)
    print(_perf_delta_table(perf_rows), flush=True)
    print('\n=== END LONG-SEQUENCE / MULTI-KV-BLOCK REPORT ===\n', flush=True)

    self.assertLen(attention_rows, len(geometries) * 2 * 2)
    for label, b, d in attention_rows:
      msg = f'{label}, B={b}: {d}'
      self.assertGreater(d.rows_compared, 0, msg=msg)
      self.assertEqual(d.rows_differing, 0, msg=msg)
      self.assertEqual(d.max_abs_diff, 0.0, msg=msg)
      self.assertEqual(d.steps_differing, (), msg=msg)
    self.assertLen(regime_tables, len(regime_cells))
    for label, configs, rows in regime_tables:
      expected = set(all_configs if configs is None else configs)
      self.assertLen(rows, 5 * len(expected), msg=label)
      self.assertSetEqual({d.config_name for d in rows}, expected, msg=label)
      zero_tim_rows = [d for d in rows if d.config_name == zero_tim]
      self.assertEqual(
          [d.regime[:3] for d in zero_tim_rows],
          ['1a.', '1b.', '1c.', '2. ', '3a.'],
          msg=label,
      )
      for d in zero_tim_rows:
        self.assertEqual(d.max_abs_diff, 0.0, msg=f'{label}: {d}')
        self.assertEqual(d.exact_match_pct, 100.0, msg=f'{label}: {d}')
    expected_rows = []
    for w in workloads:
      expected_rows.extend((w.label, name) for name in perf_config_names)
    self.assertEqual(
        [(r.workload, r.config_name) for r in perf_rows], expected_rows
    )
    for r in perf_rows:
      expected_category = (
          'leave_one_out'
          if r.config_name.startswith('without_')
          else 'baseline'
      )
      self.assertEqual(r.category, expected_category, msg=str(r))
      self.assertGreater(r.sample_latency_ms, 0.0, msg=str(r))
      self.assertGreater(r.prefill_latency_ms, 0.0, msg=str(r))
      self.assertGreater(r.fwdbwd_latency_ms, 0.0, msg=str(r))
      for prog in ('sample', 'prefill', 'fwdbwd'):
        arena = getattr(r, f'{prog}_arena_hbm_mb')
        temp = getattr(r, f'{prog}_temp_hbm_mb')
        self.assertGreater(arena, 0.0, msg=f'{prog}: {r}')
        self.assertLessEqual(arena, temp + 1e-6, msg=f'{prog}: {r}')

  @parameterized.named_parameters(
      ('rmsnorm', 'use_canon_rmsnorm'),
      ('attention', 'use_canon_attention'),
      ('fixed_order_reduce', 'use_fixed_order_reduce'),
  )
  def test_benchmark_and_ablation_suite_runs_mini(self, switch):
    """Mini-model suite rows of one Leave-One-Out / Add-One-In switch.

    With the interpret-mode Pallas kernels the suite rows are the slowest work
    in this file, and absltest assigns cases to CPU test shards round-robin
    by name, so only small cases keep every shard well inside the `large`
    timeout: one switch per case, and the three baseline rows evaluated once
    in `test_zero_tim_suite_baselines_mini` instead of once per case
    (`include_baselines=False`). A mixed program has no bitwise contract
    (it drifts from both baselines), so the drift columns are only checked
    to be well-formed.
    """
    if jax.default_backend() == 'tpu' and len(jax.devices()) >= 4:
      self.skipTest('Covered at full size by the TPU suite test.')
    tp_size = 4 if len(jax.devices()) >= 4 else 1
    rows = qwen3_canon_benchmark.run_benchmark_suite(
        preset='mini_qwen3',
        batch_size=2,
        prompt_len=4,
        max_new_tokens=3,
        cache_size=16,
        num_iters=1,
        run_ablations=True,
        include_baselines=False,
        include_fwdbwd=True,
        switches=(switch,),
        tp_size=tp_size,
    )
    self.assertEqual(
        [(r.name, r.category) for r in rows],
        [
            (f'without_{switch}', 'leave_one_out'),
            (f'only_{switch}', 'add_one_in'),
        ],
    )
    for r in rows:
      for name in (
          'vs_stock_max_abs_diff',
          'vs_canon_max_abs_diff',
          'decode_vs_prefill_max_diff',
          'batch1_vs_batchn_max_diff',
          'fwd_vs_fwdbwd_max_diff',
      ):
        self.assertTrue(np.isfinite(getattr(r, name)), msg=f'{name}: {r}')
      for name in (
          'vs_canon_exact_match_pct',
          'sampled_token_match_pct',
          'decode_vs_prefill_exact_pct',
          'batch1_vs_batchn_exact_pct',
          'fwd_vs_fwdbwd_exact_pct',
      ):
        self.assertBetween(getattr(r, name), 0.0, 100.0, msg=f'{name}: {r}')
      self.assertGreater(r.sample_latency_ms, 0.0, msg=str(r))
      self.assertGreater(r.prefill_latency_ms, 0.0, msg=str(r))
      self.assertGreater(r.fwdbwd_latency_ms, 0.0, msg=str(r))

  def test_zero_tim_suite_baselines_mini(self):
    """The suite's three baseline rows prove the Zero-TIM invariants on CPU.

    `canon_all_disabled` reproduces stock Qwen3 bitwise, and
    `canon_all_enabled` is bitwise identical across KV-cached decode,
    cacheless prefill and fwd+bwd scoring and across `B=1` / `B=2`
    (`_assert_benchmark_suite_rows`, shared with the TPU suite). The name
    sorts last so that adding this case moved no other case between the
    shards of either test target.
    """
    if jax.default_backend() == 'tpu' and len(jax.devices()) >= 4:
      self.skipTest('Covered at full size by the TPU suite test.')
    tp_size = 4 if len(jax.devices()) >= 4 else 1
    rows = qwen3_canon_benchmark.run_benchmark_suite(
        preset='mini_qwen3',
        batch_size=2,
        prompt_len=4,
        max_new_tokens=3,
        cache_size=16,
        num_iters=1,
        run_ablations=False,
        include_fwdbwd=True,
        tp_size=tp_size,
    )
    self.assertEqual(
        [(r.name, r.category) for r in rows],
        [
            ('stock_qwen3 (model.py)', 'baseline'),
            ('canon_all_disabled', 'baseline'),
            ('canon_all_enabled (Zero-TIM)', 'baseline'),
        ],
    )
    self._assert_benchmark_suite_rows(rows)

  @parameterized.named_parameters(
      *((name, name) for name in qwen3_canon.ABLATION_SWITCH_NAMES)
  )
  def test_ablation_switch_programs_run_mini(self, switch: str):
    """Every Add-One-In / Leave-One-Out program runs on the mini model.

    The mini suite above times three switches; this keeps the mixed stock /
    canonical programs of every switch -- KV-cached decode, cacheless prefill
    scoring and fwd+bwd scoring -- compiling and finite on CPU (TP=4,
    interpret-mode kernels), where the full ablation table is TPU-only. A
    mixed program has no bitwise contract (it is expected to drift from both
    baselines), so only shapes, finiteness and the log-probability range are
    checked; the three baselines are not re-run per case.
    """
    if jax.default_backend() == 'tpu' and len(jax.devices()) >= 4:
      self.skipTest('Covered at full size by the TPU suite test.')
    tp_size = 4 if len(jax.devices()) >= 4 else 1
    config = qwen3_canon_benchmark.build_model_config(
        'mini_qwen3', tp_size=tp_size
    )
    prompt = jax.random.randint(
        jax.random.PRNGKey(7), (2, 4), 0, config.vocab_size, dtype=jnp.int32
    )
    max_new_tokens = 2
    # Like `run_benchmark_suite`: a TP mesh only for TP > 1.
    mesh = _tp_mesh(tp_size) if tp_size > 1 else contextlib.nullcontext()
    with mesh:
      model = qwen3_canon.Qwen3Canon(
          config,
          rngs=nnx.Rngs(params=0),
          canon_config=qwen3_canon.CanonKernelConfig.all_enabled(),
      )
      qwen3_canon.init_non_zero_mlp_weights(model, seed=123)
      for label, cfg in (
          (f'only_{switch}', qwen3_canon.CanonKernelConfig.only(switch)),
          (f'without_{switch}', qwen3_canon.CanonKernelConfig.without(switch)),
      ):
        model.set_canon_config(cfg)
        use_cls = cfg.use_canon_logsoftmax
        full, decode_lps, _ = qwen3_canon.forward_sample_with_logprobs(
            model,
            prompt,
            max_new_tokens=max_new_tokens,
            cache_size=16,
            use_canon_logsoftmax=use_cls,
        )
        prefill_lps = qwen3_canon.score_sequence_prefill(
            model, full, use_canon_logsoftmax=use_cls
        )
        fwdbwd_lps, grad_norm = qwen3_canon.score_sequence_fwd_bwd(
            model, full, use_canon_logsoftmax=use_cls
        )
        full = np.asarray(full)
        self.assertEqual(full.shape, (2, 4 + max_new_tokens), msg=label)
        self.assertTrue(
            np.all((full >= 0) & (full < config.vocab_size)), msg=label
        )
        for name, lps, shape in (
            ('decode', decode_lps, (2, max_new_tokens)),
            ('prefill', prefill_lps, (2, 4 + max_new_tokens - 1)),
            ('fwd+bwd', fwdbwd_lps, (2, 4 + max_new_tokens - 1)),
        ):
          lps = np.asarray(lps, dtype=np.float32)
          self.assertEqual(lps.shape, shape, msg=f'{label}: {name}')
          self.assertTrue(np.all(np.isfinite(lps)), msg=f'{label}: {name}')
          # Log-probabilities: `x - logsumexp(x) <= 0` up to one rounding.
          self.assertLessEqual(float(lps.max()), 1e-3, msg=f'{label}: {name}')
        grad_norm = float(grad_norm)
        self.assertTrue(np.isfinite(grad_norm), msg=label)
        self.assertGreater(grad_norm, 0.0, msg=label)

  def test_performance_scaling_benchmark_rows_and_report(self):
    """Mini-model run of the scaling benchmark incl. Leave-One-Out rows."""
    if jax.default_backend() == 'tpu':
      self.skipTest('Covered at full size by the TPU suite test.')
    tp_size = 4 if len(jax.devices()) >= 4 else 1
    workload = qwen3_canon_benchmark.PerfWorkload(
        label='mini L=2, B=2, T=7 (4+3)',
        num_layers=2,
        batch_size=2,
        prompt_len=4,
        max_new_tokens=3,
        cache_size=16,
    )
    other = qwen3_canon_benchmark.PerfWorkload(
        label='mini L=1, B=2, T=7 (4+3)',
        num_layers=1,
        batch_size=2,
        prompt_len=4,
        max_new_tokens=3,
        cache_size=16,
    )
    rows = qwen3_canon_benchmark.run_performance_scaling_benchmark(
        tp_size=tp_size,
        num_iters=1,
        preset='mini_qwen3',
        workloads=(workload, other),
        leave_one_out_workloads=(workload.label,),
        # The log-softmax switch is the one whose Leave-One-Out row must run
        # the forward functions with `use_canon_logsoftmax=False`.
        leave_one_out_switches=('use_canon_logsoftmax',),
    )
    # 3 baselines per workload + 1 Leave-One-Out row on `workload` only.
    self.assertLen(rows, 3 * 2 + 1)
    self.assertEqual(
        [(r.workload, r.config_name, r.category) for r in rows],
        [
            (workload.label, 'stock_qwen3 (model.py)', 'baseline'),
            (workload.label, 'canon_all_disabled', 'baseline'),
            (workload.label, 'canon_all_enabled (Zero-TIM)', 'baseline'),
            (workload.label, 'without_use_canon_logsoftmax', 'leave_one_out'),
            (other.label, 'stock_qwen3 (model.py)', 'baseline'),
            (other.label, 'canon_all_disabled', 'baseline'),
            (other.label, 'canon_all_enabled (Zero-TIM)', 'baseline'),
        ],
    )
    for r in rows:
      self.assertGreater(r.sample_latency_ms, 0.0, msg=str(r))
      self.assertGreater(r.prefill_latency_ms, 0.0, msg=str(r))
      self.assertGreater(r.fwdbwd_latency_ms, 0.0, msg=str(r))
      self.assertGreater(r.prefill_tok_per_s, 0.0, msg=str(r))
      self.assertGreater(r.fwdbwd_tok_per_s, 0.0, msg=str(r))
      # The buffer-assignment metrics are only meaningful on TPU; on CPU they
      # must still be finite and non-negative.
      for name in (
          'sample_arena_hbm_mb',
          'sample_code_hbm_mb',
          'prefill_arena_hbm_mb',
          'prefill_code_hbm_mb',
          'fwdbwd_arena_hbm_mb',
          'fwdbwd_code_hbm_mb',
      ):
        value = getattr(r, name)
        self.assertTrue(np.isfinite(value), msg=f'{name}: {value}')
        self.assertGreaterEqual(value, 0.0, msg=f'{name}: {value}')

    report = qwen3_canon_benchmark.format_performance_scaling_report(rows)
    header = report.splitlines()[0]
    self.assertIn('| Category |', header)
    self.assertIn('Sample HBM (Arena/Code MB)', header)
    self.assertIn('Fwd+Bwd HBM (Arena/Code MB)', header)
    self.assertNotIn('Peak/', header)
    self.assertIn('`without_use_canon_logsoftmax` | `leave_one_out` |', report)
    # Header, separator, one line per row.
    self.assertLen(report.splitlines(), 2 + len(rows))

    # `configs` restricts the timed rows (used by the same-process TPU A/Bs to
    # time only the Zero-TIM configuration); unknown names fail up front.
    zero_tim = 'canon_all_enabled (Zero-TIM)'
    only_zero_tim = qwen3_canon_benchmark.run_performance_scaling_benchmark(
        tp_size=tp_size,
        num_iters=1,
        preset='mini_qwen3',
        workloads=(other,),
        leave_one_out_workloads=(),
        configs=(zero_tim,),
    )
    self.assertEqual(
        [(r.workload, r.config_name, r.category) for r in only_zero_tim],
        [(other.label, zero_tim, 'baseline')],
    )
    with self.assertRaisesRegex(ValueError, 'Unknown perf configs'):
      qwen3_canon_benchmark.run_performance_scaling_benchmark(
          tp_size=tp_size,
          num_iters=1,
          preset='mini_qwen3',
          workloads=(other,),
          leave_one_out_workloads=(),
          configs=('canon_all_enabled',),
      )

  def test_drift_regime_diagnostics_config_filter(self):
    """`configs` / `include_deep_stack` select exactly the requested rows."""
    if jax.default_backend() == 'tpu':
      self.skipTest('Covered at full size by the TPU suite and RoPE A/B tests.')
    tp_size = 4 if len(jax.devices()) >= 4 else 1
    zero_tim = 'canon_all_enabled (Zero-TIM)'
    rows = qwen3_canon_benchmark.run_drift_regime_diagnostics(
        preset='mini_qwen3',
        batch_size=2,
        prompt_len=4,
        max_new_tokens=3,
        cache_size=16,
        tp_size=tp_size,
        configs=(zero_tim,),
        include_deep_stack=False,
    )
    self.assertLen(rows, 5)
    self.assertSetEqual({r.config_name for r in rows}, {zero_tim})
    self.assertEqual(
        [r.regime[:3] for r in rows], ['1a.', '1b.', '1c.', '2. ', '3a.']
    )
    # The labels carry the shapes that were run (B=2, T=4+3, L=2).
    self.assertEqual(
        [r.regime for r in rows],
        [
            '1a. Decode(B=1,T=1) vs Prefill(B=1,T=7) [Same B=1]',
            '1b. Decode(B=1,T=1) vs Prefill(B=2,T=7) [Rollout vs Learner]',
            '1c. Decode(B=2,T=1) vs Prefill(B=2,T=7) [Same B=2]',
            '2. Prefill(B=1,T=7) vs Prefill(B=2,T=7) [Batch Invariance]',
            '3a. Fwd vs Fwd+Bwd (L=2, 3D [B=2,T=7,D])',
        ],
    )
    for r in rows:
      self.assertEqual(r.max_abs_diff, 0.0, msg=str(r))
      self.assertEqual(r.exact_match_pct, 100.0, msg=str(r))
    with self.assertRaisesRegex(ValueError, 'Unknown drift regime configs'):
      qwen3_canon_benchmark.run_drift_regime_diagnostics(
          preset='mini_qwen3',
          batch_size=2,
          prompt_len=4,
          max_new_tokens=3,
          cache_size=16,
          tp_size=tp_size,
          configs=('canon_all_enabled',),
      )

  @parameterized.named_parameters(
      ('fullvocab', 'qwen3_0p6b_2l_fullvocab', False),
      ('aligned', 'qwen3_0p6b_2l_fullvocab_aligned', True),
  )
  def test_production_vocab_presets(self, preset: str, aligned: bool):
    """The production-vocabulary presets have the geometry the TPU tables use.

    `qwen3_0p6b_2l_fullvocab` keeps Qwen3's `V=151936` at every TP; at
    TP=2/4/8 the vocab-parallel log-softmax is not admitted (gathered
    fallback) and the fixed LM head pads its `V/TP` columns (`37984 -> 38144`
    at TP=4). `qwen3_0p6b_2l_fullvocab_aligned` is the library's 8192-column
    padding of it (`155648 = 38 * 4096`): admitted at TP=2/4/8 and
    tile-aligned for the LM head. Both are the 0.6B architecture at two
    layers in bf16 and are registered with the `--model_preset` flag.
    """
    expected_vocab = (
        qwen3_canon_benchmark.PRODUCTION_VOCAB_ALIGNED
        if aligned
        else qwen3_canon_benchmark.PRODUCTION_VOCAB
    )
    self.assertIn(preset, qwen3_canon_benchmark.MODEL_PRESETS)
    self.assertEqual(
        qwen3_canon_benchmark.PRODUCTION_VOCAB,
        qwen3_stock.ModelConfig.qwen3_0p6b().vocab_size,
    )
    self.assertEqual(
        qwen3_canon_benchmark.PRODUCTION_VOCAB,
        canonical_logsoftmax.PRODUCTION_V,
    )
    self.assertEqual(qwen3_canon_benchmark.PRODUCTION_VOCAB_ALIGNED, 155648)
    for tp_size in (1, 2, 4, 8):
      msg = f'{preset}, tp={tp_size}'
      config = qwen3_canon_benchmark.build_model_config(preset, tp_size=tp_size)
      self.assertEqual(config.vocab_size, expected_vocab, msg=msg)
      self.assertEqual(config.num_layers, 2, msg=msg)
      self.assertEqual(
          (
              config.embed_dim,
              config.hidden_dim,
              config.num_heads,
              config.num_kv_heads,
              config.head_dim,
          ),
          (1024, 3072, 16, 8, 128),
          msg=msg,
      )
      self.assertTrue(config.use_tied_embedding, msg=msg)
      self.assertEqual(config.dtype, jnp.dtype(jnp.bfloat16), msg=msg)
      self.assertEqual(config.param_dtype, jnp.dtype(jnp.bfloat16), msg=msg)
      if tp_size > 1:
        self.assertEqual(
            qwen3_canon.vocab_parallel_logsoftmax_admitted(
                config.vocab_size, tp_size
            ),
            aligned,
            msg=msg,
        )
    # The fixed LM head geometry at the measured TP=4.
    contract = qwen3_canon.resolve_or_build_contract(
        qwen3_canon_benchmark.build_model_config(preset, tp_size=4), 4
    )
    local_vocab = expected_vocab // 4
    self.assertEqual(contract.block_n, 256)
    if aligned:
      self.assertEqual(local_vocab, 38 * 1024)
      self.assertNotIn(local_vocab, contract.matmul_n_padding)
    else:
      self.assertEqual(local_vocab, 37984)
      self.assertEqual(contract.matmul_n_padding[local_vocab], 38144)

  def test_extract_memory_profile_fields(self):
    """`_extract_memory_profile` populates every `MemoryProfile` field."""

    @jax.jit
    def f(x):
      return jnp.sum(jnp.sin(x) @ jnp.cos(x).T, axis=0)

    x = jnp.ones((64, 128), jnp.float32)
    profile = qwen3_canon_benchmark._extract_memory_profile(f, x)  # pylint: disable=protected-access
    for field in dataclasses.fields(profile):
      value = getattr(profile, field.name)
      self.assertIsInstance(value, float, msg=field.name)
      self.assertTrue(np.isfinite(value), msg=f'{field.name}: {value}')
      self.assertGreaterEqual(value, 0.0, msg=f'{field.name}: {value}')

    # (Positivity of arena / code on real programs is asserted by the TPU
    # suite test; a toy program may keep its temporaries in VMEM.)
    # A failing compile degrades to an all-zero profile instead of raising.
    @jax.jit
    def failing(x):
      del x
      raise ValueError('compile failure under test')

    zero = qwen3_canon_benchmark._extract_memory_profile(failing, x)  # pylint: disable=protected-access
    self.assertEqual(zero.temp_hbm_mb, 0.0)
    self.assertEqual(zero.arena_hbm_mb, 0.0)
    self.assertEqual(zero.code_hbm_mb, 0.0)
    self.assertEqual(zero.ba_peak_mb, 0.0)

  @parameterized.named_parameters(
      ('all_enabled', None, None, (2, 9)),
      ('all_enabled_m258', None, None, (2, 129)),
      (
          'without_use_canon_norm_matmul',
          'without',
          'use_canon_norm_matmul',
          (2, 9),
      ),
      ('only_use_fixed_order_reduce', 'only', 'use_fixed_order_reduce', (2, 9)),
      ('only_use_canon_qkv_proj', 'only', 'use_canon_qkv_proj', (2, 9)),
      ('only_use_canon_rmsnorm', 'only', 'use_canon_rmsnorm', (2, 9)),
      ('only_use_fixed_lm_head', 'only', 'use_fixed_lm_head', (2, 9)),
  )
  def test_param_grads_match_stock_tp4(
      self,
      factory: str | None,
      switch_name: str | None,
      token_shape: tuple[int, int],
  ):
    """Parameter gradients at TP=4 must match stock `Qwen3` leaf by leaf.

    Every `jax.shard_map` of the canonical model runs with `check_vma=False`
    and is differentiated from the outside, so each site has to agree with
    JAX's transpose convention (divide replicated output cotangents by the
    unmentioned axis sizes, `psum` replicated input cotangents). The forward
    Zero-TIM checks cannot see a wrong gradient scale; this test can: before
    the `_fixed_order_tp_sum_with_vjp` / `_column_parallel_local` fix the
    grad-norm ratios were `0.25x` (contract-parallel sites), `4.0x` (every
    RMSNorm gamma, the LM head's upstream) and up to `11x` (embedding) in
    exactly these configurations. The `m258` case (`B*T = 258 > 256` rows)
    additionally covers the row-chunked LM-head VJP and the un-chunked
    vocab-parallel log-softmax backward at more than one production bucket.
    """
    tp_size = 4
    cfg = _make_test_config(use_tied_embedding=False, tp_size=tp_size)
    if factory is None:
      canon_cfg = qwen3_canon.CanonKernelConfig.all_enabled()
    else:
      canon_cfg = getattr(qwen3_canon.CanonKernelConfig, factory)(switch_name)
    with _tp_mesh(tp_size):
      stock = qwen3_stock.Qwen3(cfg, rngs=nnx.Rngs(params=0))
      qwen3_canon.init_non_zero_mlp_weights(stock, seed=7)
      canon = qwen3_canon.Qwen3Canon(
          cfg, rngs=nnx.Rngs(params=0), canon_config=canon_cfg
      )
      qwen3_canon.copy_weights(stock, canon)
      tokens = jax.random.randint(
          jax.random.PRNGKey(1), token_shape, 0, cfg.vocab_size, dtype=jnp.int32
      )
      g_stock = _param_grads(stock, tokens, use_canon_logsoftmax=False)
      g_canon = _param_grads(
          canon, tokens, use_canon_logsoftmax=canon_cfg.use_canon_logsoftmax
      )

    ref = jax.tree_util.tree_leaves_with_path(g_stock)
    test = jax.tree_util.tree_leaves_with_path(g_canon)
    self.assertLen(test, len(ref))
    worst_ratio_dev = 0.0
    min_cosine = 1.0
    for (p_ref, r), (p_test, t) in zip(ref, test):
      path = jax.tree_util.keystr(p_ref)
      self.assertEqual(path, jax.tree_util.keystr(p_test))
      r = np.asarray(r, np.float32)
      t = np.asarray(t, np.float32)
      norm_ref = float(np.linalg.norm(r))
      norm_test = float(np.linalg.norm(t))
      self.assertGreater(norm_ref, 0.0, msg=path)
      ratio = norm_test / norm_ref
      cosine = float((r * t).sum() / (norm_ref * norm_test + 1e-30))
      worst_ratio_dev = max(worst_ratio_dev, abs(ratio - 1.0))
      min_cosine = min(min_cosine, cosine)
      # Canonical kernels differ from XLA's bf16 ops only at rounding level
      # (measured `|ratio - 1| <= 1e-3`, `cos >= 0.9999`); the bugs this test
      # guards against are integer factors of TP.
      self.assertAlmostEqual(
          ratio,
          1.0,
          delta=0.02,
          msg=f'{path}: grad-norm ratio canon/stock = {ratio:.4f}',
      )
      self.assertGreaterEqual(
          cosine, 0.999, msg=f'{path}: grad cosine = {cosine:.5f}'
      )
    print(
        f'[grad-check {self._testMethodName}] {len(ref)} leaves, worst'
        f' |ratio-1| = {worst_ratio_dev:.2e}, min cosine = {min_cosine:.6f}',
        flush=True,
    )


def _param_grads(model, tokens: jax.Array, *, use_canon_logsoftmax: bool):
  """Returns `d(-mean token logprob) / d params` for a cacheless prefill."""
  graphdef, params, other_state = nnx.split(model, nnx.Param, ...)

  def loss_fn(param_state):
    m = nnx.merge(graphdef, param_state, other_state)
    logits = _prefill_logits(m, tokens)
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
