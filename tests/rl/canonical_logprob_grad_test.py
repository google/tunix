"""canonical_logsoftmax.target_logprob_grad: one tiled pass equals the analytic softmax gradient."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CANON_PALLAS_LOGSOFTMAX", "1")  # the module's fail-closed contract

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tunix.rl import canonical_logsoftmax as cls


def _reference(logits, token_ids, cotangent):
  probabilities = jax.nn.softmax(logits, axis=-1)
  selected = jax.nn.one_hot(token_ids, logits.shape[-1], dtype=logits.dtype)
  return (selected - probabilities) * cotangent[:, None]


@pytest.mark.parametrize("rows,vocab", [(4, 4096), (2, 3000), (3, 8192 + 5)])
def test_fused_grad_matches_reference_in_interpret_mode(rows, vocab):
  key = jax.random.PRNGKey(rows * 7 + vocab)
  logits = jax.random.normal(key, (rows, vocab), dtype=jnp.float32) * 4.0
  token_ids = jax.random.randint(jax.random.PRNGKey(1), (rows,), 0, vocab)
  cotangent = jax.random.normal(jax.random.PRNGKey(2), (rows,), dtype=jnp.float32)
  got = cls.target_logprob_grad(logits, token_ids, cotangent, interpret=True)
  want = _reference(logits, token_ids, cotangent)
  assert got.shape == (rows, vocab) and got.dtype == logits.dtype
  np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-6)
  # softmax gradient rows sum to zero (up to rounding) and the selected column carries cot * (1 - p)
  np.testing.assert_allclose(np.asarray(got).sum(axis=-1), 0.0, atol=1e-4)
  p_sel = np.asarray(jax.nn.softmax(logits, axis=-1))[np.arange(rows), np.asarray(token_ids)]
  np.testing.assert_allclose(np.asarray(got)[np.arange(rows), np.asarray(token_ids)], np.asarray(cotangent) * (1 - p_sel), rtol=1e-5, atol=1e-6)


def test_output_dtype_follows_logits():
  logits = jax.random.normal(jax.random.PRNGKey(3), (4, 512), dtype=jnp.float32)
  out = cls.target_logprob_grad(logits, jnp.zeros((4,), jnp.int32), jnp.ones((4,), jnp.float32), interpret=True)
  assert out.shape == (4, 512) and out.dtype == jnp.float32


def _random_case(rows, vocab, seed):
  logits = jax.random.normal(jax.random.PRNGKey(seed), (rows, vocab), dtype=jnp.float32) * 4.0
  token_ids = jax.random.randint(jax.random.PRNGKey(seed + 1), (rows,), 0, vocab)
  cotangent = jax.random.normal(jax.random.PRNGKey(seed + 2), (rows,), dtype=jnp.float32)
  return logits, token_ids, cotangent


def _recording_grad(monkeypatch):
  seen = []
  real = cls.target_logprob_grad

  def record(logits, token_ids, cotangent, *, interpret=False):
    seen.append(tuple(int(d) for d in logits.shape))
    return real(logits, token_ids, cotangent, interpret=interpret)

  monkeypatch.setattr(cls, "target_logprob_grad", record)
  return seen


def test_row_plan_matches_the_forward_buckets():
  # the trainer's DP2 chunk: two 256-row programs, like one per DP rank in the forward
  assert cls.target_logprob_grad_row_plan(512) == ((256, 256), (256, 256))
  assert cls.target_logprob_grad_row_plan(256) == ((256, 256),)
  assert cls.target_logprob_grad_row_plan(16) == ((16, 16),)
  # a short tail runs at its own bucket (multiple of 8) with inert rows appended
  assert cls.target_logprob_grad_row_plan(300) == ((256, 256), (44, 48))
  assert cls.target_logprob_grad_row_plan(3) == ((3, 8),)


def test_rows_wrapper_splits_512_rows_into_two_256_row_programs(monkeypatch):
  seen = _recording_grad(monkeypatch)
  logits, token_ids, cotangent = _random_case(512, 2048, 11)
  got = cls.target_logprob_grad_rows(logits, token_ids, cotangent, interpret=True)
  assert seen == [(256, 2048), (256, 2048)]
  np.testing.assert_allclose(np.asarray(got), np.asarray(_reference(logits, token_ids, cotangent)), rtol=1e-5, atol=1e-6)


def test_rows_wrapper_pads_a_short_tail_and_discards_the_inert_rows(monkeypatch):
  seen = _recording_grad(monkeypatch)
  logits, token_ids, cotangent = _random_case(300, 1024, 17)
  got = cls.target_logprob_grad_rows(logits, token_ids, cotangent, interpret=True)
  assert seen == [(256, 1024), (48, 1024)]
  assert got.shape == (300, 1024)
  np.testing.assert_allclose(np.asarray(got), np.asarray(_reference(logits, token_ids, cotangent)), rtol=1e-5, atol=1e-6)


def test_rows_wrapper_passes_an_admitted_bucket_straight_through(monkeypatch):
  seen = _recording_grad(monkeypatch)
  logits, token_ids, cotangent = _random_case(16, 1024, 23)
  got = cls.target_logprob_grad_rows(logits, token_ids, cotangent, interpret=True)
  assert seen == [(16, 1024)]
  np.testing.assert_allclose(np.asarray(got), np.asarray(_reference(logits, token_ids, cotangent)), rtol=1e-5, atol=1e-6)


def test_real_kernel_runs_per_rank_inside_the_forward_style_shard_map():
  """The interpret-mode kernel itself, one row slice per data rank, under check_vma=False."""
  from jax.sharding import Mesh, PartitionSpec as P
  devices = jax.devices()
  if len(devices) < 2:
    pytest.skip("needs --xla_force_host_platform_device_count=2")
  mesh = Mesh(np.array(devices[:2]).reshape(2, 1), axis_names=("data", "model"))
  logits, token_ids, cotangent = _random_case(512, 2048, 29)

  def per_rank(l, i, c):
    assert l.shape == (256, 2048)  # the rank's own slice, never the global 512 rows
    return cls.target_logprob_grad_rows(l, i, c, interpret=True)

  mapped = jax.shard_map(per_rank, mesh=mesh, in_specs=(P("data", None), P("data"), P("data")), out_specs=P("data", None), check_vma=False)
  got = mapped(logits, token_ids, cotangent)
  assert got.shape == (512, 2048)
  np.testing.assert_allclose(np.asarray(got), np.asarray(_reference(logits, token_ids, cotangent)), rtol=1e-5, atol=1e-6)
