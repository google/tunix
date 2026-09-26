# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""fp32 master weights through the real MaxText engine (CPU, tiny model).

Everything here goes through the production entry points --
`maxtext_utils.build_maxtext_config` -> `maxtext_utils.create_maxtext_engine` --
and MaxText's own `get_optimizer`, `skip_step_on_spikes`,
`apply_trainable_parameters_mask`, `nnx.Optimizer` and update kernel. Nothing
in MaxText is mocked or reassigned. Skipped when maxtext is not importable.
"""

import os
import tempfile
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from tunix.utils import master_weights as mw_lib
from tunix.utils import maxtext_utils

try:
  from maxtext.optimizers import optimizers as mt_optimizers  # pylint: disable=g-import-not-at-top
  from maxtext.training_engine import maxtext_engine_compile  # pylint: disable=g-import-not-at-top
  from maxtext.utils import maxtext_utils as mt_maxtext_utils  # pylint: disable=g-import-not-at-top

  _MAXTEXT_ERROR = None
except Exception as e:  # pylint: disable=broad-except
  _MAXTEXT_ERROR = e

# A tiny `default` decoder. float32_gate_logits=True keeps logits_dense in fp32
# (common_types.get_weight_dtype), so the tree mixes bf16 and fp32 leaves the
# way Qwen3.5's does (norms / gates / A_log / dt_bias / conv1d in fp32).
_TINY = (
    "vocab_size=8 base_emb_dim=8 base_mlp_dim=16 base_num_decoder_layers=2"
    " base_num_query_heads=2 base_num_kv_heads=2 head_dim=4"
    " float32_gate_logits=True use_tokamax_gmm=false use_gmm_v2=false"
)
# Freezes the embedding the way the MLPerf recipe freezes the router
# ('^(?!.*routed_experts/gate/kernel).*'): negative lookahead on one path.
_FREEZE_EMBED = ["^(?!.*token_embedder).*"]
_BF16_TRAINABLE = 10  # every param but logits_dense (fp32) and token_embedder
_FP32 = "decoder/logits_dense/kernel"
_FROZEN = "token_embedder/embedding"


def _build(fp32_master_weights, *, skip=True, mask=None, extra="", lr=1e-2):
  n = jax.device_count()
  env = {"MAXTEXT_EXTRA_FLAGS": f"{_TINY} {extra}".strip()}
  with mock.patch.dict(os.environ, env):
    cfg = maxtext_utils.build_maxtext_config(
        model_name="default",
        worker_id="mw_engine_test",
        train_micro_batch_size=n,
        mesh_fsdp=n,
        num_devices=n,
        max_prompt_length=2,
        max_response_length=2,
        learning_rate=lr,
        base_output_directory=tempfile.mkdtemp(),
        trainable_parameters_mask=mask,
        adam_b2=0.999,
        adam_weight_decay=0.1,
        gradient_clipping_threshold=0.125,
        skip_step_on_spikes=skip,
        fp32_master_weights=fp32_master_weights,
    )
  mesh = maxtext_utils.create_maxtext_mesh(cfg)
  engine = maxtext_utils.create_maxtext_engine(
      cfg,
      mesh,
      wrap_with_tunix_adapter=False,  # the adapter needs an HF config; `default` has none
      log_shapes=False,
      fp32_master_weights=fp32_master_weights,
  )
  return cfg, mesh, engine


def _batch(cfg, seed=0):
  batch, seq = int(cfg.micro_batch_size_to_train_on), cfg.max_target_length
  rng = np.random.default_rng(seed)
  tokens = jnp.asarray(rng.integers(0, cfg.vocab_size, size=(batch, seq)), dtype=jnp.int32)
  targets = jnp.asarray(rng.integers(0, cfg.vocab_size, size=(batch, seq)), dtype=jnp.int32)
  positions = jnp.arange(seq, dtype=jnp.int32)[None, :].repeat(batch, axis=0)
  segmentation = jnp.ones((batch, seq), dtype=jnp.int32)
  return {
      "inputs": tokens,
      "targets": targets,
      "inputs_position": positions,
      "inputs_segmentation": segmentation,
      "targets_segmentation": segmentation,
      "decoder_input_tokens": tokens,
      "decoder_target_tokens": targets,
      "decoder_loss_weights": jnp.ones((batch, seq), dtype=jnp.float32),
      "decoder_positions": positions,
  }


def _step(engine, mesh, cfg, seed):
  with mesh:
    engine.fwd_bwd(_batch(cfg, seed))
    engine.update()


def _flat(tree):
  """{'a/b/c': leaf} over a pure tree, keeping MaskedNode leaves."""
  out = {}
  for path, leaf in jax.tree_util.tree_flatten_with_path(
      tree, is_leaf=lambda x: isinstance(x, optax.MaskedNode)
  )[0]:
    out["/".join(str(getattr(k, "key", getattr(k, "name", getattr(k, "idx", k)))) for k in path)] = leaf
  return out


def _params(engine):
  return {k: np.asarray(v) for k, v in _flat(nnx.to_pure_dict(nnx.state(engine.model, nnx.Param))).items()}


def _master(engine):
  states = mw_lib.find_master_states(nnx.as_pure(engine.optimizer.opt_state))
  assert len(states) == 1, len(states)
  return _flat(states[0].master)


def _bits_equal(a, b):
  a, b = np.asarray(a), np.asarray(b)
  return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(
      a.reshape(-1).view(np.uint8), b.reshape(-1).view(np.uint8)
  )


def _engine_is_skipped(engine):
  """Reads is_skipped exactly where MaxText's update kernel looks for it."""
  opt_state = nnx.to_pure_dict(nnx.state(engine.optimizer)).get("opt_state", {})
  return opt_state.get("is_skipped") if isinstance(opt_state, dict) else None


@absltest.skipIf(_MAXTEXT_ERROR is not None, f"maxtext not importable: {_MAXTEXT_ERROR!r}")
class MasterWeightsMaxTextEngineTest(absltest.TestCase):

  def test_skip_and_mask_wrap_master_and_update_keeps_dtypes(self):
    with self.assertLogs(level="INFO") as logs:
      cfg, mesh, engine = _build(True, skip=True, mask=_FREEZE_EMBED)
    self.assertIs(mt_optimizers.optax, optax)  # the scope is gone
    self.assertEqual(jnp.dtype(cfg.mu_dtype), jnp.float32)
    self.assertEqual(jnp.dtype(cfg.weight_dtype), jnp.bfloat16)
    self.assertIn(
        f"fp32 master weights: ON, {_BF16_TRAINABLE} bf16 leaves mastered, 1 fp32 leaves"
        " passthrough, 1 frozen",
        "\n".join(logs.output),
    )

    # mask(skip(master(adamw))): the wrapper sits inside both MaxText wrappers.
    opt_state = nnx.as_pure(engine.optimizer.opt_state)
    self.assertIsInstance(opt_state, optax.PartitionState)
    trainable = opt_state.inner_states["trainable"]
    self.assertIsInstance(trainable, optax.MaskedState)
    skip_state = trainable.inner_state
    self.assertEqual(set(skip_state), {"inner_state", "losses", "grad_norms", "count", "is_skipped"})
    self.assertIsInstance(skip_state["inner_state"], mw_lib.MasterWeightsState)

    params0 = _params(engine)
    master = _master(engine)
    self.assertLen(params0, _BF16_TRAINABLE + 2)
    for name, p in params0.items():
      if name in (_FP32, _FROZEN):
        self.assertIsInstance(master[name], optax.MaskedNode, name)
      else:
        self.assertEqual(p.dtype, jnp.bfloat16, name)
        self.assertEqual(master[name].dtype, jnp.float32, name)
        self.assertTrue(_bits_equal(master[name].astype(jnp.bfloat16), p), name)
    self.assertEqual(params0[_FP32].dtype, jnp.float32)
    # The frozen leaf has no moments either.
    adam = [
        x
        for x in jax.tree.leaves(opt_state, is_leaf=lambda x: isinstance(x, optax.ScaleByAdamState))
        if isinstance(x, optax.ScaleByAdamState)
    ]
    self.assertLen(adam, 1)
    for moments in (adam[0].mu, adam[0].nu):
      flat = _flat(moments)
      self.assertIsInstance(flat[_FROZEN], optax.MaskedNode)
      for name in params0:
        if name != _FROZEN:
          self.assertEqual(flat[name].dtype, jnp.float32, name)

    for seed in range(3):
      _step(engine, mesh, cfg, seed)

    params = _params(engine)
    master = _master(engine)
    self.assertEqual({k: v.dtype for k, v in params.items()}, {k: v.dtype for k, v in params0.items()})
    self.assertTrue(_bits_equal(params[_FROZEN], params0[_FROZEN]))
    self.assertFalse(_bits_equal(params[_FP32], params0[_FP32]))
    moved = 0
    for name, p in params.items():
      if name in (_FP32, _FROZEN):
        continue
      self.assertTrue(_bits_equal(master[name].astype(jnp.bfloat16), p), name)
      moved += int(not _bits_equal(p, params0[name]))
    self.assertGreater(moved, 0)
    self.assertFalse(bool(skip_state["is_skipped"]))

  def test_param_tree_seen_by_weight_sync_is_unchanged(self):
    _, _, off = _build(False, skip=True, mask=_FREEZE_EMBED)
    _, _, on = _build(True, skip=True, mask=_FREEZE_EMBED)
    p_off, p_on = _params(off), _params(on)
    self.assertEqual(sorted(p_off), sorted(p_on))
    for name in p_off:
      self.assertEqual((p_on[name].dtype, p_on[name].shape), (p_off[name].dtype, p_off[name].shape))
      self.assertTrue(_bits_equal(p_on[name], p_off[name]), name)  # same init, same bytes

  def test_is_skipped_stays_where_the_engine_reads_it(self):
    cfg, mesh, engine = _build(True, skip=True, mask=None)
    self.assertIsNotNone(_engine_is_skipped(engine))
    _step(engine, mesh, cfg, 0)
    self.assertIsNotNone(_engine_is_skipped(engine))
    self.assertFalse(bool(_engine_is_skipped(engine)))
    # With a trainable mask the skip state is one multi_transform level down,
    # with or without the master: the wrapper does not move it.
    _, _, masked_on = _build(True, skip=True, mask=_FREEZE_EMBED)
    _, _, masked_off = _build(False, skip=True, mask=_FREEZE_EMBED)
    self.assertIsNone(_engine_is_skipped(masked_on))
    self.assertIsNone(_engine_is_skipped(masked_off))

  def test_fp32_leaf_first_step_identical_to_flag_off(self):
    # OFF with mu_dtype=float32 too, so the only difference is the master.
    cfg, mesh, off = _build(False, skip=False, mask=_FREEZE_EMBED, extra="mu_dtype=float32")
    _, _, on = _build(True, skip=False, mask=_FREEZE_EMBED)
    _step(off, mesh, cfg, 0)
    _step(on, mesh, cfg, 0)
    p_off, p_on = _params(off), _params(on)
    self.assertTrue(_bits_equal(p_on[_FP32], p_off[_FP32]))
    self.assertTrue(_bits_equal(p_on[_FROZEN], p_off[_FROZEN]))
    master = _master(on)
    for name in p_on:
      if name not in (_FP32, _FROZEN):
        self.assertTrue(_bits_equal(master[name].astype(jnp.bfloat16), p_on[name]), name)

  def test_all_fp32_weights_bit_identical_to_flag_off(self):
    extra = "weight_dtype=float32"  # MAXTEXT_EXTRA_FLAGS comes last, so it wins
    cfg, mesh, off = _build(False, skip=True, mask=_FREEZE_EMBED, extra=f"{extra} mu_dtype=float32")
    with self.assertLogs(level="WARNING") as logs:
      _, _, on = _build(True, skip=True, mask=_FREEZE_EMBED, extra=extra)
    self.assertIn("no-op", "\n".join(logs.output))
    self.assertEqual(jnp.dtype(cfg.weight_dtype), jnp.float32)
    self.assertEqual(jax.tree.leaves(_master(on)), [])
    for seed in range(3):
      _step(off, mesh, cfg, seed)
      _step(on, mesh, cfg, seed)
    p_off, p_on = _params(off), _params(on)
    for name in p_off:
      self.assertTrue(_bits_equal(p_on[name], p_off[name]), name)

  def test_abstract_engine_builds_and_compiles_master_under_the_scope(self):
    # No trainable mask: MaxText's AbstractMaxTextEngine cannot trace a frozen
    # leaf's OptVariable(MaskedNode) with or without the master (pre-existing).
    cfg, mesh, _ = _build(False, skip=True, mask=None, extra="mu_dtype=float32")
    with mw_lib.install_in_maxtext_get_optimizer() as record:
      engine = maxtext_engine_compile.AbstractMaxTextEngine(cfg, mesh)
    self.assertIs(mt_optimizers.optax, optax)
    stats = mw_lib.verify_installed(record, engine.optimizer.opt_state)
    self.assertGreaterEqual(len(record.init_stats), 1)  # traced once per eval_shape
    self.assertEqual((stats.mastered, stats.passthrough, stats.frozen), (_BF16_TRAINABLE + 1, 1, 0))
    master = _master(engine)
    self.assertLen(jax.tree.leaves(master), _BF16_TRAINABLE + 1)
    self.assertTrue(all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(master)))
    compiled = engine.compile_kernels(maxtext_engine_compile.get_shaped_micro_batch(cfg))
    self.assertIn("update", compiled)

  def test_real_get_optimizer_on_router_like_tree(self):
    mask = ["^(?!.*routed_experts/gate/kernel).*"]
    cfg, _, _ = _build(False, skip=True, mask=mask, extra="mu_dtype=float32")
    schedule = mt_maxtext_utils.create_learning_rate_schedule(cfg)
    with mw_lib.install_in_maxtext_get_optimizer() as record:
      tx = mt_optimizers.get_optimizer(cfg, schedule)
    params = {
        "layers": {
            "wq": jnp.full((4, 4), 0.25, jnp.bfloat16),
            "norm": jnp.ones((4,), jnp.float32),
            "moe": {"routed_experts": {"gate": {"kernel": jnp.full((4, 2), 0.5, jnp.bfloat16)}}},
        }
    }
    state = tx.init(params)
    stats = mw_lib.verify_installed(record, state)
    self.assertEqual((stats.mastered, stats.passthrough, stats.frozen), (1, 1, 1))
    master = _flat(mw_lib.find_master_states(state)[0].master)
    self.assertIsInstance(master["layers/moe/routed_experts/gate/kernel"], optax.MaskedNode)
    self.assertIsInstance(master["layers/norm"], optax.MaskedNode)
    self.assertEqual(master["layers/wq"].dtype, jnp.float32)

  def test_raises_for_non_adamw(self):
    with self.assertRaisesRegex(ValueError, "opt_type=adamw"):
      _build(True, skip=True, mask=_FREEZE_EMBED, extra="opt_type=sgd")

  def test_raises_when_maxtext_builds_adamw_elsewhere(self):
    """If MaxText stops building adamw through its module-level optax, construction fails loudly."""

    def drifted_get_optimizer(config, learning_rate_schedule, model=None, mesh=None):
      del model, mesh
      return mt_optimizers.apply_trainable_parameters_mask(optax.adamw(learning_rate_schedule), config)

    with mock.patch.object(mt_optimizers, "get_optimizer", drifted_get_optimizer):
      with self.assertRaisesRegex(RuntimeError, "saw 0 call"):
        _build(True, skip=False, mask=_FREEZE_EMBED)
    self.assertIs(mt_optimizers.optax, optax)


if __name__ == "__main__":
  absltest.main()
