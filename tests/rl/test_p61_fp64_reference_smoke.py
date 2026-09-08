"""Smoke test of the Phase 13 fp64 reference on a tiny Qwen3 (CPU, x64).

Pins the mechanics the 1.7B run relies on: the captured model_before fills
an abstract model leaf for leaf by manifest path; the microbatched
unreduced-sum gradient equals the whole-batch gradient (the loss is
separable over rows given the batch scale); the metrics helpers; and the
logps validation against a captured engine logps array.
"""

import importlib.util
import os
import pathlib
import types

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from tunix.models.qwen3 import model as qwen3_model
from tunix.rl import common
from tunix.rl.agentic import agentic_rl_learner as learner

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "canon-zero-tim/tests/p61_backward/fp64_reference.py"
)


def _load_script():
  spec = importlib.util.spec_from_file_location("fp64_reference", _SCRIPT)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _tiny_config():
  return qwen3_model.ModelConfig(
      num_layers=2, vocab_size=64, embed_dim=32, hidden_dim=48, num_heads=4,
      head_dim=8, num_kv_heads=2, rope_theta=10000, norm_eps=1e-6,
      use_tied_embedding=True,
  )


def _algo():
  return types.SimpleNamespace(
      beta=0.04, epsilon=0.2, epsilon_high=0.2, epsilon_c=None,
      loss_algo="grpo", loss_agg_mode="sequence-mean-token-mean",
      temperature=1.0, kl_loss_mode="low_var_kl", kl_clamp_value=None,
  )


def _example(rows=4, prompt=5, completion=6, vocab=64):
  key = jax.random.PRNGKey(0)
  k1, k2, k3, k4 = jax.random.split(key, 4)
  prompt_ids = jax.random.randint(k1, (rows, prompt), 1, vocab)
  completion_ids = jax.random.randint(k2, (rows, completion), 1, vocab)
  lengths = jnp.asarray([6, 4, 5, 3])[:rows]
  completion_mask = jnp.arange(completion)[None, :] < lengths[:, None]
  return common.TrainExample(
      prompt_ids=prompt_ids, prompt_mask=jnp.ones((rows, prompt), jnp.bool_),
      completion_ids=completion_ids, completion_mask=completion_mask,
      advantages=jax.random.normal(k3, (rows,), jnp.float64),
      ref_per_token_logps=-jnp.abs(jax.random.normal(k4, (rows, completion), jnp.float64)),
      old_per_token_logps=-0.5 * jnp.ones((rows, completion), jnp.float64),
  )


def test_fp64_reference_mechanics(tmp_path):
  script = _load_script()
  config = _tiny_config()
  model = qwen3_model.Qwen3(config, rngs=nnx.Rngs(3))
  root = tmp_path / "p61"
  learner._p61_capture_tree(str(root), "model_before", nnx.state(model, nnx.Param))  # pylint: disable=protected-access
  example = _example()
  learner._p61_capture_tree(str(root), "example", example)  # pylint: disable=protected-access
  learner._p61_write_algo_config(str(root), _AlgoDataclass(), pad_id=0, eos_id=2)  # pylint: disable=protected-access
  # The captured parameters fill the abstract model leaf for leaf, bitwise
  # (up to the f32 -> f64 widening).
  params = script.load_capture(root, "model_before")
  graphdef, state = script.build_model(config, params)
  filled = nnx.merge(graphdef, state)
  for path, variable in jax.tree_util.tree_flatten_with_path(
      nnx.state(filled, nnx.Param), is_leaf=script._is_variable  # pylint: disable=protected-access
  )[0]:
    want = params[jax.tree_util.keystr(path)]
    assert np.asarray(variable[...]).astype(np.float32).tobytes() == want.tobytes()
  # Microbatched gradient of the unreduced sum == whole-batch gradient.
  loaded = script.load_example(root)
  algo, pad_id, eos_id = script.load_algo_config(root)  # the dataclass carries temperature
  grad_fn = script.make_grad_fn(graphdef, algo, pad_id, eos_id)
  (whole_value, whole_logps), whole = grad_fn(state, loaded)
  # The whole pipeline computes in float64 once the script widened float32.
  assert whole_value.dtype == jnp.float64 and whole_logps.dtype == jnp.float64
  assert all(leaf.dtype == jnp.float64 for leaf in jax.tree.leaves(whole))
  parts = [grad_fn(state, script.example_rows(loaded, slice(i, i + 2))) for i in (0, 2)]
  summed = jax.tree.map(lambda a, b: a + b, parts[0][1], parts[1][1])
  assert abs(float(whole_value) - sum(float(p[0][0]) for p in parts)) < 1e-12 * max(1.0, abs(float(whole_value)))
  for a, b in zip(jax.tree.leaves(whole), jax.tree.leaves(summed)):
    assert np.allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-14)
  scale = script.batch_scale(loaded, algo)
  assert scale == pytest.approx(1.0 / 4)
  # Metrics and the logps validation.
  learner._p61_capture_tree(str(root), "logps", jnp.asarray(whole_logps, jnp.float32))  # pylint: disable=protected-access
  validation = script.validate_logps(np.asarray(whole_logps), root, loaded, 4)
  assert validation["action_tokens"] == 18 and validation["max_abs_diff"] < 1e-6
  same = script.metrics(np.asarray(whole_logps), np.asarray(whole_logps))
  assert same["rel_l2"] == 0.0 and same["one_minus_cos"] == 0.0
  moved = script.metrics(np.ones(4), np.asarray([1.0, 1.0, 1.0, 1.1]))
  assert 0.0 < moved["rel_l2"] < 0.06 and moved["one_minus_cos"] > 0.0
  assert script.leaf_group("['layers'][3]['attn']['q_proj']['w']") == "layer[3].attn"
  assert script.leaf_group("['embedder']['input_embedding']") == "embedder"


import dataclasses  # noqa: E402


@dataclasses.dataclass
class _AlgoDataclass:
  beta: float = 0.04
  epsilon: float = 0.2
  epsilon_high: float = 0.2
  epsilon_c: float | None = None
  loss_algo: str = "grpo"
  loss_agg_mode: str = "sequence-mean-token-mean"
  temperature: float = 1.0
  kl_loss_mode: str = "low_var_kl"
  kl_clamp_value: float | None = None
