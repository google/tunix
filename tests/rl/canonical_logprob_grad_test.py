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
