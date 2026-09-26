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

"""CPU tests for tunix.utils.master_weights."""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import types  # pylint: disable=g-import-not-at-top,g-bad-import-order

from absl.testing import absltest  # pylint: disable=g-import-not-at-top
from absl.testing import parameterized  # pylint: disable=g-import-not-at-top
from flax import nnx  # pylint: disable=g-import-not-at-top
import jax  # pylint: disable=g-import-not-at-top
import jax.numpy as jnp  # pylint: disable=g-import-not-at-top
import numpy as np  # pylint: disable=g-import-not-at-top
import optax  # pylint: disable=g-import-not-at-top

from tunix.utils import master_weights as mw_lib  # pylint: disable=g-import-not-at-top

MasterWeightsState = mw_lib.MasterWeightsState
find_master_states = mw_lib.find_master_states
master_weights = mw_lib.master_weights
materialize_params = mw_lib.materialize_params

# MaxText base.yml defaults for b1/b2/eps/wd; lr is the MLPerf recipe's.
LR, B1, B2, EPS, WD = 1e-6, 0.9, 0.95, 1e-8, 0.1
BF16, F32, F16 = jnp.bfloat16, jnp.float32, jnp.float16


def adamw(mu_dtype=None, weight_decay=WD, lr=LR):
  return optax.adamw(lr, b1=B1, b2=B2, eps=EPS, weight_decay=weight_decay, mu_dtype=mu_dtype)


def make_step(tx):
  """One optimizer step as nnx.Optimizer.update does it: tx.update then apply_updates."""

  @jax.jit
  def step(params, state, grads):
    updates, state = tx.update(grads, state, params)
    return optax.apply_updates(params, updates), state

  return step


def bits_equal(a, b):
  a, b = np.asarray(a), np.asarray(b)
  if a.dtype != b.dtype or a.shape != b.shape:
    return False
  return np.array_equal(a.reshape(-1).view(np.uint8), b.reshape(-1).view(np.uint8))


def invariant_ok(params, state):
  """Every master-backed leaf equals master.astype(leaf dtype) (numerically)."""
  ref = materialize_params(params, state)
  return all(
      bool(jnp.all(p == r)) and p.dtype == r.dtype
      for p, r in zip(jax.tree.leaves(params), jax.tree.leaves(ref))
  )


def rand_grads(key, params, scale=1.0):
  leaves, treedef = jax.tree.flatten(params)
  keys = jax.random.split(key, len(leaves))
  return treedef.unflatten([scale * jax.random.normal(k, p.shape, F32) for k, p in zip(keys, leaves)])


def _maxtext_optimizers():
  try:
    from maxtext.optimizers import optimizers  # pylint: disable=g-import-not-at-top
  except Exception:  # pylint: disable=broad-except
    return None
  return optimizers


class MasterWeightsTest(parameterized.TestCase):

  def test_bf16_leaf_moves_where_plain_adamw_is_frozen(self):
    """A bf16 leaf at 1.0 with lr 1e-6: plain adamw never moves it; the wrapper does, exactly."""
    params = {"w": jnp.ones((8,), BF16)}
    grads = {"w": jnp.full((8,), 0.3, F32)}  # grad_dtype=float32, as tunix sets it

    # MaxText today: weight_dtype=bf16, mu_dtype inherits it.
    plain = adamw(mu_dtype=BF16)
    st = plain.init(params)
    step = make_step(plain)
    p = params
    for _ in range(100):
      p, st = step(p, st, grads)
    self.assertTrue(bits_equal(p["w"], params["w"]))

    tx = master_weights(adamw())
    st = tx.init(params)

    @jax.jit
    def run(p, st):
      def body(carry, _):
        p, st = carry
        u, st = tx.update(grads, st, p)
        p = optax.apply_updates(p, u)
        exact = jnp.all(p["w"] == st.master["w"].astype(BF16))
        return (p, st), (p["w"][0], exact)

      return jax.lax.scan(body, (p, st), None, length=3000)

    (p, st), (w_hist, exact_hist) = run(params, st)
    w_hist = np.asarray(w_hist, np.float32)
    self.assertTrue(bool(np.all(np.asarray(exact_hist))))
    self.assertTrue(np.any(w_hist != 1.0))
    self.assertEqual(p["w"].dtype, BF16)

  def test_fp32_leaf_bit_identical_to_plain_adamw(self):
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    params = {
        "w_bf16": (0.02 * jax.random.normal(k1, (64, 32))).astype(BF16),
        "w_fp32": 0.02 * jax.random.normal(k2, (64, 32), F32),
        "norm_fp32": 1.0 + 0.01 * jax.random.normal(k3, (32,), F32),
    }
    tx = master_weights(adamw())
    ref_tx = adamw()
    ref_params = jax.tree.map(lambda x: x.astype(F32), params)
    st, ref_st = tx.init(params), ref_tx.init(ref_params)
    step, ref_step = make_step(tx), make_step(ref_tx)
    p, rp = params, ref_params
    for i in range(200):
      g = rand_grads(jax.random.PRNGKey(100 + i), params)
      p, st = step(p, st, g)
      rp, ref_st = ref_step(rp, ref_st, g)
    self.assertTrue(bits_equal(p["w_fp32"], rp["w_fp32"]))
    self.assertTrue(bits_equal(p["norm_fp32"], rp["norm_fp32"]))
    # The bf16 leaf's master is exactly fp32 adamw run on its fp32 copy.
    self.assertTrue(bits_equal(st.master["w_bf16"], rp["w_bf16"]))
    self.assertTrue(invariant_ok(p, st))

  def test_dtypes_preserved_and_master_only_below_fp32(self):
    params = {
        "a": jnp.full((4, 4), 0.5, BF16),
        "b": jnp.full((4,), 0.5, F32),
        "c": {"d": jnp.full((3,), 0.5, F16), "e": jnp.full((2, 2), 0.5, BF16)},
    }
    tx = master_weights(adamw())
    st = tx.init(params)
    step = make_step(tx)
    p = params
    for i in range(5):
      p, st = step(p, st, rand_grads(jax.random.PRNGKey(i), params))
    self.assertEqual(jax.tree.map(lambda x: x.dtype, p), jax.tree.map(lambda x: x.dtype, params))
    self.assertIsInstance(st.master["b"], optax.MaskedNode)
    for leaf in (st.master["a"], st.master["c"]["d"], st.master["c"]["e"]):
      self.assertEqual(leaf.dtype, F32)
    adam = st.inner_state[0]
    self.assertEqual(adam.nu["a"].dtype, F32)
    self.assertEqual(adam.nu["b"].dtype, F32)
    self.assertEqual(adam.nu["c"]["d"].dtype, F32)
    self.assertTrue(invariant_ok(p, st))

  def test_param_stats_and_on_init(self):
    params = {
        "a": jnp.zeros((4, 4), BF16),
        "b": jnp.zeros((4,), F32),
        "c": {"d": jnp.zeros((3,), F16), "frozen": optax.MaskedNode()},
        "i": jnp.zeros((2,), jnp.int32),
    }
    seen = []
    master_weights(adamw(), on_init=seen.append).init(params)
    self.assertLen(seen, 1)
    stats = seen[0]
    self.assertEqual(stats.mastered, 2)
    self.assertEqual(stats.mastered_dtypes, (("bfloat16", 1), ("float16", 1)))
    self.assertEqual(stats.mastered_elements, 19)
    self.assertEqual(stats.master_bytes, 76)
    self.assertEqual(stats.passthrough, 2)
    self.assertEqual(stats.frozen, 1)

  def test_update_requires_params(self):
    tx = master_weights(adamw())
    params = {"w": jnp.ones((2,), BF16)}
    st = tx.init(params)
    with self.assertRaisesRegex(ValueError, "needs `params`"):
      tx.update({"w": jnp.ones((2,), F32)}, st)

  def test_skipped_step_leaves_master_and_params(self):
    """MaxText's NaN skip: jnp.where(skip, old, new) over params AND optimizer state."""
    params = {"w": jnp.full((16,), 0.7, BF16), "n": jnp.full((16,), 1.0, F32)}
    tx = master_weights(adamw())

    @jax.jit
    def engine_update(params, opt_state, grads):
      grad_norm = optax.tree.norm(jax.tree.map(lambda g: g.astype(F32), grads))
      should_skip = jnp.logical_not(jnp.isfinite(grad_norm))
      safe = jax.tree.map(lambda g: jnp.where(should_skip, jnp.zeros_like(g), g), grads)
      u, new_opt = tx.update(safe, opt_state, params)
      new_params = optax.apply_updates(params, u)
      old, new = (params, opt_state), (new_params, new_opt)
      reverted = jax.tree.map(lambda n, o: jnp.where(should_skip, o, n), new, old)
      return reverted, new, should_skip

    st = tx.init(params)
    p = params
    for i in range(10):
      (p, st), _, _ = engine_update(p, st, rand_grads(jax.random.PRNGKey(i), params))
    nan_grads = {"w": jnp.full((16,), jnp.nan, F32), "n": jnp.ones((16,), F32)}
    (p2, st2), (_, st_unrev), skipped = engine_update(p, st, nan_grads)
    self.assertTrue(bool(skipped))
    for a, b in zip(jax.tree.leaves((p2, st2)), jax.tree.leaves((p, st))):
      self.assertTrue(bits_equal(a, b))
    # Weight decay alone would have moved the fp32 master: the where protects it.
    self.assertFalse(bits_equal(st_unrev.master["w"], st.master["w"]))
    (p3, st3), _, sk = engine_update(p2, st2, rand_grads(jax.random.PRNGKey(99), params))
    self.assertFalse(bool(sk))
    self.assertTrue(invariant_ok(p3, st3))

  @parameterized.parameters("clip_outside", "clip_inside")
  def test_wd0_and_clip_compose(self, where):
    k1, _ = jax.random.split(jax.random.PRNGKey(1))
    params = {"w": (0.05 * jax.random.normal(k1, (128,))).astype(BF16), "n": jnp.ones((8,), F32)}
    inner = adamw(weight_decay=0.0)
    if where == "clip_outside":
      tx = optax.chain(optax.clip_by_global_norm(1.0), master_weights(inner))
      get_mw = lambda st: st[1]
    else:
      tx = master_weights(optax.chain(optax.clip_by_global_norm(1.0), inner))
      get_mw = lambda st: st
    ref_tx = optax.chain(optax.clip_by_global_norm(1.0), adamw(weight_decay=0.0))
    ref_params = jax.tree.map(lambda x: x.astype(F32), params)
    st, rst = tx.init(params), ref_tx.init(ref_params)
    step, rstep = make_step(tx), make_step(ref_tx)
    p, rp = params, ref_params
    for i in range(100):
      g = rand_grads(jax.random.PRNGKey(200 + i), params, scale=50.0)  # clip active
      p, st = step(p, st, g)
      rp, rst = rstep(rp, rst, g)
    mw = get_mw(st)
    self.assertIsInstance(mw, MasterWeightsState)
    self.assertTrue(bits_equal(p["n"], rp["n"]))
    self.assertTrue(bits_equal(mw.master["w"], rp["w"]))
    self.assertTrue(invariant_ok(p, mw))

  def test_inner_mu_bf16(self):
    """mu_dtype=bf16 (MaxText's default under weight_dtype=bf16) still works, mu stays bf16."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(2))
    params = {
        "w": (0.02 * jax.random.normal(k1, (256,))).astype(BF16),
        "n": 0.02 * jax.random.normal(k2, (16,), F32),
    }
    tx = master_weights(adamw(mu_dtype=BF16))
    ref_tx = adamw(mu_dtype=BF16)
    st, rst = tx.init(params), ref_tx.init({"n": params["n"]})
    step, rstep = make_step(tx), make_step(ref_tx)
    p, rn = params, {"n": params["n"]}
    for i in range(200):
      g = rand_grads(jax.random.PRNGKey(300 + i), params)
      p, st = step(p, st, g)
      rn, rst = rstep(rn, rst, {"n": g["n"]})
      self.assertTrue(invariant_ok(p, st))
    adam = st.inner_state[0]
    self.assertEqual(adam.mu["w"].dtype, BF16)
    self.assertEqual(adam.mu["n"].dtype, BF16)
    self.assertEqual(adam.nu["w"].dtype, F32)
    self.assertEqual(st.master["w"].dtype, F32)
    self.assertTrue(bits_equal(p["n"], rn["n"]))
    self.assertEqual(mw_lib.adam_moment_dtypes(st), (("bfloat16",), ("float32",)))

  def test_exactness_sweep(self):
    """The additive encoding recovers new_master.astype(bf16) except in the >2^16 collapse corner."""
    n = 2_000_000
    kp, ke, ks, kr, kz = jax.random.split(jax.random.PRNGKey(3), 5)
    p = jnp.exp2(jax.random.uniform(kp, (n,), minval=-40.0, maxval=10.0)) * jnp.sign(
        jax.random.normal(ks, (n,))
    )
    p = p.astype(BF16)
    r = jnp.exp2(jax.random.uniform(ke, (n,), minval=-30.0, maxval=30.0)) * jnp.sign(
        jax.random.normal(kr, (n,))
    )
    small = jax.random.uniform(kz, (n,)) < 0.3
    r = jnp.where(small, 1.0 + 1e-3 * jax.random.normal(kz, (n,)), r)
    nm = p.astype(F32) * r

    @jax.jit
    def encodings(p, nm):
      t32 = nm.astype(BF16).astype(F32)
      ours = optax.apply_updates(p, t32 - p.astype(F32))
      bf16_sub = optax.apply_updates(p, nm.astype(BF16) - p)
      return nm.astype(BF16), ours, bf16_sub

    target, ours, bf16_sub = encodings(p, nm)
    target32 = np.asarray(target, np.float32)
    p32 = np.asarray(p, np.float32)
    ratio = np.abs(target32) / np.abs(p32)
    log2r = np.floor(np.log2(np.where(ratio > 0, ratio, np.nan)))
    m_ours = np.asarray(ours, np.float32) != target32
    m_bf16 = np.asarray(bf16_sub, np.float32) != target32
    if m_ours.any():
      self.assertLessEqual(np.nanmax(log2r[m_ours]), -17)
      rel_err = np.abs(np.asarray(ours, np.float32)[m_ours] - target32[m_ours]) / np.abs(p32[m_ours])
      self.assertLessEqual(rel_err.max(), 2.0**-24)
    exact_region = (np.nan_to_num(log2r, nan=0.0) >= -16) | (target32 == 0)
    self.assertFalse((m_ours & exact_region).any())
    # Emitting the difference in bf16 would miss even at realistic step sizes.
    realistic = np.abs(np.nan_to_num(log2r, nan=99)) <= 1
    self.assertGreater(m_bf16[realistic].sum(), 0)
    self.assertEqual(m_ours[realistic].sum(), 0)

  def test_nnx_optimizer_update_kernel_pattern(self):
    """nnx.Optimizer driven as MaxText's update kernel drives it: split/merge/apply/where."""

    class Tiny(nnx.Module):

      def __init__(self):
        self.w = nnx.Param(jnp.full((4, 4), 0.25, BF16))
        self.scale = nnx.Param(jnp.ones((4,), F32))

    class TrainState(nnx.Module):

      def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

      def apply_gradients(self, grads, **kw):
        self.optimizer.update(self.model, grads, **kw)

    model = Tiny()
    opt = nnx.Optimizer(model, master_weights(adamw(lr=1e-3)), wrt=nnx.Param)
    state = TrainState(model, opt)
    graphdef, state_pure = nnx.split(state)

    @jax.jit
    def update_kernel(state_pure, grads, force_skip):
      local = nnx.merge(graphdef, state_pure, copy=True)
      local.apply_gradients(grads)
      _, new = nnx.split(local)
      return jax.tree.map(lambda n, o: jnp.where(force_skip, o, n), new, state_pure)

    template = nnx.state(model, nnx.Param)

    def grads_for(i):
      k1, k2 = jax.random.split(jax.random.PRNGKey(400 + i))
      vals = {"w": jax.random.normal(k1, (4, 4), F32), "scale": jax.random.normal(k2, (4,), F32)}
      return jax.tree_util.tree_map_with_path(lambda path, x: vals[path[0].key], template)

    ref_tx = adamw(lr=1e-3)
    ref_p = {"scale": jnp.ones((4,), F32)}
    ref_st = ref_tx.init(ref_p)
    ref_step = make_step(ref_tx)
    sp = state_pure
    for i in range(20):
      g = grads_for(i)
      sp = update_kernel(sp, g, jnp.array(False))
      ref_p, ref_st = ref_step(ref_p, ref_st, {"scale": g["scale"][...]})
    before = sp
    sp = update_kernel(sp, grads_for(99), jnp.array(True))
    for a, b in zip(jax.tree.leaves(sp), jax.tree.leaves(before)):
      self.assertTrue(bits_equal(a, b))
    nnx.update(state, sp)
    w, scale = state.model.w[...], state.model.scale[...]
    self.assertEqual(w.dtype, BF16)
    self.assertEqual(scale.dtype, F32)
    mws = find_master_states(nnx.as_pure(state.optimizer.opt_state))
    self.assertLen(mws, 1)
    self.assertTrue(bool(jnp.all(w == mws[0].master["w"].astype(BF16))))
    self.assertIsInstance(mws[0].master["scale"], optax.MaskedNode)
    self.assertTrue(bits_equal(scale, ref_p["scale"]))
    self.assertFalse(bits_equal(w, jnp.full((4, 4), 0.25, BF16)))

  def test_maxtext_skip_on_spikes_and_trainable_mask(self):
    """mask(skip_step_on_spikes(master_weights(adamw))) with MaxText's own wrappers."""
    mt = _maxtext_optimizers()
    if mt is None:
      self.skipTest("maxtext is not importable")
    cfg = types.SimpleNamespace(trainable_parameters_mask=["^layers/"])
    base = master_weights(adamw(lr=1e-3))
    base = mt.skip_step_on_spikes(base, interval=8, scaling_factor=3.0)
    tx = mt.apply_trainable_parameters_mask(base, cfg)
    params = {
        "layers": {"w": jnp.full((8,), 0.3, BF16), "norm": jnp.ones((8,), F32)},
        "embed": jnp.full((8,), 0.1, BF16),  # frozen by the mask
    }
    st = tx.init(params)

    @jax.jit
    def step(p, st, g, loss, gn):
      u, st = tx.update(g, st, p, loss=loss, grad_norm=gn)
      return optax.apply_updates(p, u), st

    p = params
    skipped = []
    for i in range(16):
      g = rand_grads(jax.random.PRNGKey(500 + i), params)
      loss = jnp.float32(100.0 if i == 12 else 1.0 + 0.01 * i)  # spike at step 12
      prev_p, prev_st = p, st
      p, st = step(p, st, g, loss, jnp.float32(1.0))
      is_sk = bool(st.inner_states["trainable"].inner_state["is_skipped"])
      skipped.append(is_sk)
      mw = find_master_states(st)[0]
      self.assertTrue(invariant_ok(p["layers"], MasterWeightsState(mw.master["layers"], None)))
      if is_sk:
        prev_mw = find_master_states(prev_st)[0]
        self.assertTrue(bits_equal(mw.master["layers"]["w"], prev_mw.master["layers"]["w"]))
        self.assertTrue(bits_equal(p["layers"]["w"], prev_p["layers"]["w"]))
    self.assertTrue(skipped[12])
    self.assertEqual(sum(skipped), 1)
    self.assertTrue(bits_equal(p["embed"], params["embed"]))
    self.assertEqual(jax.tree.map(lambda x: x.dtype, p), jax.tree.map(lambda x: x.dtype, params))
    self.assertIsInstance(find_master_states(st)[0].master["embed"], optax.MaskedNode)

  def test_sharding_follows_params(self):
    devs = np.array(jax.devices()[:8])
    if devs.size < 8:
      self.skipTest("needs 8 host devices")
    mesh = jax.sharding.Mesh(devs.reshape(4, 2), ("fsdp", "expert"))
    P = jax.sharding.PartitionSpec
    sh_w = jax.sharding.NamedSharding(mesh, P("fsdp", "expert"))
    sh_n = jax.sharding.NamedSharding(mesh, P())
    params = {
        "w": jax.device_put(jnp.full((16, 8), 0.5, BF16), sh_w),
        "n": jax.device_put(jnp.ones((8,), F32), sh_n),
    }
    tx = master_weights(adamw(lr=1e-3))
    st = tx.init(params)
    self.assertEqual(st.master["w"].sharding, sh_w)
    self.assertEqual(st.inner_state[0].nu["w"].sharding, sh_w)
    replicated = jax.sharding.NamedSharding(mesh, P())

    def mesh_sharding(x):
      s = getattr(x, "sharding", None)
      return s if isinstance(s, jax.sharding.NamedSharding) and s.mesh == mesh else replicated

    state_sh = jax.tree.map(mesh_sharding, st)
    param_sh = jax.tree.map(mesh_sharding, params)
    st = jax.tree.map(jax.device_put, st, state_sh)
    step = jax.jit(
        lambda p, s, g: (lambda u_s: (optax.apply_updates(p, u_s[0]), u_s[1]))(tx.update(g, s, p)),
        in_shardings=(param_sh, state_sh, param_sh),
        out_shardings=(param_sh, state_sh),
    )
    g = jax.tree.map(lambda x, s: jax.device_put(jnp.ones(x.shape, F32), s), params, param_sh)
    p2, st2 = step(params, st, g)
    self.assertEqual(p2["w"].sharding, sh_w)
    self.assertEqual(st2.master["w"].sharding, sh_w)
    self.assertEqual(p2["w"].dtype, BF16)
    self.assertEqual(p2["n"].dtype, F32)
    self.assertTrue(invariant_ok(p2, st2))


def _fake_optimizers_module(n_adamw_calls=1):
  """A stand-in for maxtext.optimizers.optimizers: get_optimizer reads module-global optax."""
  mod = types.ModuleType("fake_maxtext_optimizers")
  mod.optax = optax

  def get_optimizer(lr):
    txs = [mod.optax.adamw(lr, b1=B1, b2=B2, mu_dtype=jnp.float32) for _ in range(n_adamw_calls)]
    return mod.optax.chain(mod.optax.identity(), *txs) if txs else mod.optax.sgd(lr)

  mod.get_optimizer = get_optimizer
  return mod


class InstallTest(absltest.TestCase):

  def test_proxy_wraps_adamw_only_and_restores(self):
    mod = _fake_optimizers_module()
    with mw_lib.install_in_maxtext_get_optimizer(mod) as record:
      self.assertIsNot(mod.optax, optax)
      self.assertIs(mod.optax.identity, optax.identity)
      self.assertIs(mod.optax.MaskedNode, optax.MaskedNode)
      tx = mod.get_optimizer(1e-3)
    self.assertIs(mod.optax, optax)
    self.assertEqual(record.adamw_calls, 1)
    self.assertEqual(record.adamw_kwargs["mu_dtype"], jnp.float32)
    params = {"w": jnp.ones((4,), BF16), "n": jnp.ones((2,), F32)}
    st = tx.init(params)
    stats = mw_lib.verify_installed(record, st)
    self.assertEqual((stats.mastered, stats.passthrough, stats.frozen), (1, 1, 0))
    # A tx built after the scope is plain adamw.
    self.assertEmpty(find_master_states(mod.get_optimizer(1e-3).init(params)))

  def test_restores_on_error_and_rejects_nesting(self):
    mod = _fake_optimizers_module()
    with self.assertRaisesRegex(ZeroDivisionError, ""):
      with mw_lib.install_in_maxtext_get_optimizer(mod):
        _ = 1 / 0
    self.assertIs(mod.optax, optax)
    with mw_lib.install_in_maxtext_get_optimizer(mod):
      with self.assertRaisesRegex(RuntimeError, "already active"):
        with mw_lib.install_in_maxtext_get_optimizer(_fake_optimizers_module()):
          pass
      self.assertIsNot(mod.optax, optax)
    self.assertIs(mod.optax, optax)

  def test_verify_raises_unless_exactly_one_adamw(self):
    params = {"w": jnp.ones((4,), BF16)}
    for n in (0, 2):
      mod = _fake_optimizers_module(n_adamw_calls=n)
      with mw_lib.install_in_maxtext_get_optimizer(mod) as record:
        st = mod.get_optimizer(1e-3).init(params)
      with self.assertRaisesRegex(RuntimeError, f"saw {n} call"):
        mw_lib.verify_installed(record, st)

  def test_verify_raises_when_state_lacks_master(self):
    mod = _fake_optimizers_module()
    with mw_lib.install_in_maxtext_get_optimizer(mod) as record:
      tx = mod.get_optimizer(1e-3)
    params = {"w": jnp.ones((4,), BF16)}
    with self.assertRaisesRegex(RuntimeError, "never initialized"):
      mw_lib.verify_installed(record, None)
    tx.init(params)
    with self.assertRaisesRegex(RuntimeError, "found 0"):
      mw_lib.verify_installed(record, adamw().init(params))

  def test_summary_line(self):
    stats = mw_lib.MasterWeightsStats(
        mastered=3,
        mastered_dtypes=(("bfloat16", 3),),
        mastered_elements=2**28,
        passthrough=2,
        frozen=1,
        master_bytes=2**30,
    )
    line = mw_lib.summary_line(stats, ("float32",), ("float32",))
    self.assertIn("fp32 master weights: ON, 3 bf16 leaves mastered", line)
    self.assertIn("2 fp32 leaves passthrough, 1 frozen", line)
    self.assertIn("1.00 GiB", line)
    self.assertIn("mu dtype=float32, nu dtype=float32", line)


if __name__ == "__main__":
  absltest.main()
