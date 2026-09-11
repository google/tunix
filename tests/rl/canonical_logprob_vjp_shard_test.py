"""The fused target-logprob VJP runs per data rank inside a shard_map, like the forward."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")
os.environ.setdefault("CANON_PALLAS_LOGSOFTMAX", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from tunix.rl import canonical_logsoftmax as cls
from tunix.rl import canonical_qwen3_adapter as adapter


class _Tensors:
  def __init__(self, logprobs):
    self.logprobs = logprobs


def _compute_and_gather(logits, token_ids, max_logprobs):
  logp = jax.nn.log_softmax(logits, axis=-1)
  return _Tensors(jnp.take_along_axis(logp, token_ids[:, None], axis=-1))


def _reference(logits, token_ids, cotangent):
  probabilities = jax.nn.softmax(logits, axis=-1)
  selected = jax.nn.one_hot(token_ids, logits.shape[-1], dtype=logits.dtype)
  return (selected - probabilities) * cotangent[:, None]


def _stand_in(seen):
  # pure-jnp stand-in for the TPU kernel: records the local shape it is traced with
  def rows(logits, token_ids, cotangent, *, interpret=False):
    seen.append(tuple(int(d) for d in logits.shape))
    return _reference(logits, token_ids, cotangent)
  return rows


def _case(rows, vocab):
  logits = jax.random.normal(jax.random.PRNGKey(5), (rows, vocab), dtype=jnp.float32) * 3.0
  token_ids = jax.random.randint(jax.random.PRNGKey(6), (rows,), 0, vocab)
  cotangent = jax.random.normal(jax.random.PRNGKey(7), (rows,), dtype=jnp.float32)
  return logits, token_ids, cotangent


def _grad(fn, logits, token_ids, cotangent):
  # the trainer closes over the token ids and differentiates the logits only
  _, pull = jax.vjp(lambda x: fn(x, token_ids), logits)
  return pull(cotangent)[0]


def test_kernel_route_runs_one_row_slice_per_data_rank(monkeypatch):
  monkeypatch.setenv(adapter.LOGPROB_VJP_KERNEL_ENV, "1")
  devices = jax.devices()
  assert len(devices) >= 2, "needs --xla_force_host_platform_device_count=2"
  mesh = Mesh(np.array(devices[:2]).reshape(2, 1), axis_names=("data", "model"))
  seen = []
  monkeypatch.setattr(cls, "target_logprob_grad_rows", _stand_in(seen))
  fn = adapter._make_processed_target_logprob_vjp(_compute_and_gather, 1, mesh=mesh)
  logits, token_ids, cotangent = _case(512, 1024)
  got = _grad(fn, logits, token_ids, cotangent)
  # the trainer's DP2 chunk: each rank sees its own 256 rows, never the global 512
  assert seen == [(256, 1024)]
  np.testing.assert_allclose(np.asarray(got), np.asarray(_reference(logits, token_ids, cotangent)), rtol=1e-5, atol=1e-6)


def test_kernel_route_without_a_data_axis_sees_the_whole_chunk(monkeypatch):
  monkeypatch.setenv(adapter.LOGPROB_VJP_KERNEL_ENV, "1")
  seen = []
  monkeypatch.setattr(cls, "target_logprob_grad_rows", _stand_in(seen))
  fn = adapter._make_processed_target_logprob_vjp(_compute_and_gather, 1, mesh=None)
  logits, token_ids, cotangent = _case(512, 1024)
  got = _grad(fn, logits, token_ids, cotangent)
  assert seen == [(512, 1024)]
  np.testing.assert_allclose(np.asarray(got), np.asarray(_reference(logits, token_ids, cotangent)), rtol=1e-5, atol=1e-6)


def test_xla_route_is_untouched_when_the_switch_is_off(monkeypatch):
  monkeypatch.setenv(adapter.LOGPROB_VJP_KERNEL_ENV, "0")
  seen = []
  monkeypatch.setattr(cls, "target_logprob_grad_rows", _stand_in(seen))
  fn = adapter._make_processed_target_logprob_vjp(_compute_and_gather, 1, mesh=None)
  logits, token_ids, cotangent = _case(64, 1024)
  got = _grad(fn, logits, token_ids, cotangent)
  assert seen == []
  np.testing.assert_allclose(np.asarray(got), np.asarray(_reference(logits, token_ids, cotangent)), rtol=1e-5, atol=1e-6)
