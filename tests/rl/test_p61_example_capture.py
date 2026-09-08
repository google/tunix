"""P61 full-tree capture: the example, logps and loss-config captures
(tasks/v2_dispatch Phase 13).  Round trip through the manifest writer;
None fields are skipped; the config writer keeps scalars only and refuses
to overwrite."""

import json
import os
import pathlib
import types

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from tunix.rl import alignment
from tunix.rl import common
from tunix.rl.agentic import agentic_rl_learner as learner


def _example():
  return common.TrainExample(
      prompt_ids=jnp.arange(8, dtype=jnp.int32).reshape(2, 4),
      prompt_mask=jnp.ones((2, 4), jnp.bool_),
      completion_ids=jnp.arange(6, dtype=jnp.int32).reshape(2, 3),
      completion_mask=jnp.asarray([[1, 1, 0], [1, 0, 0]], jnp.bool_),
      advantages=jnp.asarray([0.5, -0.5], jnp.float32),
      ref_per_token_logps=None,
      old_per_token_logps=jnp.full((2, 3), -0.25, jnp.float32),
  )


def _read(capture_dir):
  manifest = json.loads((capture_dir / "manifest.json").read_text())
  return {
      leaf["path"]: np.load(capture_dir / leaf["file"], allow_pickle=False)
      for leaf in manifest["leaves"]
  }


def test_example_capture_round_trips_and_skips_none_fields(tmp_path):
  root = tmp_path / "p61"
  example = _example()
  manifest = learner._p61_capture_tree(str(root), "example", example)  # pylint: disable=protected-access
  paths = {leaf["path"] for leaf in manifest["leaves"]}
  # A flax struct dataclass flattens to attribute paths.
  assert ".ref_per_token_logps" not in paths
  assert {".prompt_ids", ".completion_mask", ".advantages",
          ".old_per_token_logps"} <= paths
  loaded = _read(root / "example")
  for name in ("prompt_ids", "prompt_mask", "completion_ids",
               "completion_mask", "advantages", "old_per_token_logps"):
    want = np.asarray(getattr(example, name))
    got = loaded[f".{name}"]
    assert got.dtype == want.dtype and got.tobytes() == want.tobytes(), name
  logps = jnp.full((2, 3), -1.5, jnp.float32)
  learner._p61_capture_tree(str(root), "logps", logps)  # pylint: disable=protected-access
  assert _read(root / "logps")[""].tobytes() == np.asarray(logps).tobytes()
  with pytest.raises(alignment.AlignmentGateError, match="unsupported"):
    learner._p61_capture_tree(str(root), "optimizer", logps)  # pylint: disable=protected-access


def test_algo_config_writer_keeps_scalars_and_refuses_overwrite(tmp_path):
  import dataclasses

  @dataclasses.dataclass
  class Config:
    beta: float = 0.04
    epsilon: float = 0.2
    loss_agg_mode: str = "sequence-mean-token-mean"
    temperature: float = 1.0
    epsilon_c: float | None = None
    metric_fns: tuple = ()
    nested: dict = dataclasses.field(default_factory=dict)

  root = tmp_path / "p61"
  record = learner._p61_write_algo_config(  # pylint: disable=protected-access
      str(root), Config(), pad_id=151643, eos_id=151645
  )
  on_disk = json.loads((root / "algo_config.json").read_text())
  assert on_disk == record
  assert on_disk["algo_config"] == {
      "beta": 0.04, "epsilon": 0.2, "epsilon_c": None,
      "loss_agg_mode": "sequence-mean-token-mean", "temperature": 1.0,
  }
  assert (on_disk["pad_id"], on_disk["eos_id"]) == (151643, 151645)
  with pytest.raises(FileExistsError):
    learner._p61_write_algo_config(  # pylint: disable=protected-access
        str(root), Config(), pad_id=1, eos_id=2
    )
  with pytest.raises(alignment.AlignmentGateError, match="absolute"):
    learner._p61_write_algo_config("relative/dir", Config(), pad_id=1, eos_id=2)  # pylint: disable=protected-access


def test_stock_gradient_capture_matches_a_whole_batch_gradient(tmp_path):
  """Phase 13b: the stock-path gradient written next to the P61 capture is
  the whole-batch gradient of the loss at the ratio-one point (microbatches
  summed, scaled by the batch scale) with one leaf per model parameter,
  plus the stock logps; model_after is the caller's business."""
  import types
  import jax
  from flax import nnx
  from tunix.models.qwen3 import model as qwen3_model
  from tunix.rl import algo_core
  from tunix.rl import common as rl_common

  config = qwen3_model.ModelConfig(
      num_layers=2, vocab_size=64, embed_dim=32, hidden_dim=48, num_heads=4,
      head_dim=8, num_kv_heads=2, rope_theta=10000, norm_eps=1e-6,
      use_tied_embedding=True,
  )
  model = qwen3_model.Qwen3(config, rngs=nnx.Rngs(5))
  key = jax.random.PRNGKey(1)
  rows, prompt, completion = 4, 5, 6
  example = common.TrainExample(
      prompt_ids=jax.random.randint(key, (rows, prompt), 1, 64),
      prompt_mask=jnp.ones((rows, prompt), jnp.bool_),
      completion_ids=jax.random.randint(jax.random.fold_in(key, 1), (rows, completion), 1, 64),
      completion_mask=(jnp.arange(completion)[None, :] < jnp.asarray([6, 4, 5, 3])[:, None]),
      advantages=jnp.asarray([0.5, -0.25, 1.0, -1.0], jnp.float32),
      ref_per_token_logps=-jnp.ones((rows, completion), jnp.float32),
      old_per_token_logps=-0.5 * jnp.ones((rows, completion), jnp.float32),
  )
  algo = types.SimpleNamespace(
      beta=0.04, epsilon=0.2, epsilon_high=0.2, epsilon_c=None,
      loss_algo="grpo", loss_agg_mode="sequence-mean-token-mean",
      temperature=1.0, kl_loss_mode="low_var_kl", kl_clamp_value=None,
  )
  root = tmp_path / "p61"
  learner._p61_capture_stock_gradient(  # pylint: disable=protected-access
      str(root), model, example, algo, pad_id=0, eos_id=2, rows_per_step=3
  )
  written = _read(root / "stock_gradient")
  params = {
      "['" + "']['".join(str(k.key) if hasattr(k, "key") else str(k.idx) for k in path) + "']": v
      for path, v in jax.tree_util.tree_flatten_with_path(
          nnx.state(model, nnx.Param), is_leaf=lambda x: isinstance(x, nnx.Variable)
      )[0]
  }
  assert len(written) == len(params) == 24
  # The direct whole-batch gradient at the same point.
  graphdef, state = nnx.split(model)
  at_one = example.replace(old_per_token_logps=None)

  def whole(params_state):
    logps, entropy = rl_common.compute_per_token_logps(
        graphdef, params_state, prompt_tokens=at_one.prompt_ids,
        completion_tokens=at_one.completion_ids, pad_id=0, eos_id=2,
        stop_gradient=False, return_entropy=True, temperature=1.0,
        canonical_actor=False, prompt_mask=at_one.prompt_mask,
        completion_mask=at_one.completion_mask,
    )
    out = algo_core.grpo_loss_from_precomputed_logps(logps, entropy, at_one, algo)
    return out.primary_loss.unreduced_sum * out.primary_loss.compute_scale()

  direct = jax.grad(whole)(state)
  for path, leaf in jax.tree_util.tree_flatten_with_path(
      direct, is_leaf=lambda x: isinstance(x, nnx.Variable)
  )[0]:
    key_str = jax.tree_util.keystr(path)
    value = leaf[...] if isinstance(leaf, nnx.Variable) else leaf
    assert np.allclose(written[key_str], np.asarray(value), rtol=1e-5, atol=1e-7), key_str
  logps = _read(root / "stock_logps")[""]
  assert logps.shape == (rows, completion)
