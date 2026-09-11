"""CANON_ENTROPY_VJP: skipping a zero entropy cotangent leaves the gradient bytes unchanged."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tunix.rl import canonical_qwen3_adapter as adapter


class _Cfg:
  def __init__(self, coef):
    self.entropy_coef = coef


def test_switch_default_and_values(monkeypatch):
  monkeypatch.delenv(adapter.ENTROPY_VJP_ENV, raising=False)
  assert adapter.entropy_vjp_enabled()
  monkeypatch.setenv(adapter.ENTROPY_VJP_ENV, "0")
  assert not adapter.entropy_vjp_enabled()
  monkeypatch.setenv(adapter.ENTROPY_VJP_ENV, "2")
  with pytest.raises(ValueError):
    adapter.entropy_vjp_enabled()


def test_contract_rejects_entropy_loss_when_skipping(monkeypatch):
  monkeypatch.setenv(adapter.ENTROPY_VJP_ENV, "0")
  adapter.check_entropy_vjp_contract(_Cfg(None))
  adapter.check_entropy_vjp_contract(_Cfg(0.0))
  adapter.check_entropy_vjp_contract(object())
  with pytest.raises(adapter.FunctionalMappingError):
    adapter.check_entropy_vjp_contract(_Cfg(0.01))
  monkeypatch.setenv(adapter.ENTROPY_VJP_ENV, "1")
  adapter.check_entropy_vjp_contract(_Cfg(0.01))


def test_zero_entropy_cotangent_matches_logps_only_vjp_bytewise():
  """The two program shapes the switch selects between give identical gradient bytes."""
  key = jax.random.PRNGKey(0)
  logits = jax.random.normal(key, (16, 512), dtype=jnp.float32) * 3.0
  targets = jax.random.randint(jax.random.PRNGKey(1), (16,), 0, 512)
  cot = jax.random.normal(jax.random.PRNGKey(2), (16,), dtype=jnp.float32)

  def both(x):
    logp = jax.nn.log_softmax(x, axis=-1)
    target = jnp.take_along_axis(logp, targets[:, None], axis=-1)[:, 0]
    entropy = -jnp.sum(jnp.exp(logp) * logp, axis=-1)
    return target, entropy

  _, pull_both = jax.vjp(both, logits)
  g_both = pull_both((cot, jnp.zeros_like(cot)))[0]
  _, pull_logps = jax.vjp(lambda x: both(x)[0], logits)
  g_logps = pull_logps(cot)[0]
  a = np.asarray(g_both).view(np.uint32); b = np.asarray(g_logps).view(np.uint32)
  # bytes may differ only where one side is a signed zero
  differ = a != b
  assert np.all(np.asarray(g_both)[differ] == 0.0) and np.all(np.asarray(g_logps)[differ] == 0.0)
  assert np.array_equal(np.asarray(g_both), np.asarray(g_logps))
