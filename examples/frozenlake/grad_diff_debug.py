# Compare v1 and v2 weight difference
from shutil import copy
from typing import Any, List
import numpy as np
import jax
import gc
import warnings

# JAX reports failed buffer donation as "Some donated buffers were not usable",
# once per location. Python dedupes repeated warnings by default, so a warning
# raised during the first step would be invisible if you only skim later output.
warnings.simplefilter("always")


import os
from flax import nnx
import jax
import optax

from tunix.rl import common
from tunix.sft import peft_trainer
from tunix.experimental.train import peft_trainer_v2
from tunix.tests import test_common as tc

# Set XLA flag to disable excess precision (run this BEFORE jax initializes devices)
os.environ['XLA_FLAGS'] = (
    os.environ.get('XLA_FLAGS', '') + ' --xla_allow_excess_precision=false'
).strip()

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, AutoProcessor

# # Optional: You can also set JAX matmul precision to 'highest' for maximum precision
# jax.config.update('jax_default_matmul_precision', 'highest')

_LEARNING_RATE = 1e-5
_BATCH_SIZE = 8
_SEQ_LEN = 256
# Steps discarded from steady-state timing (compilation + warmup).
_WARMUP_STEPS = 3
# Steps used to compute the average step time.
_BENCHMARK_STEPS = 20
# Toy transformer vocab is >= 128 (see peft_trainer_v2_test.py), so token ids
# are drawn from [0, 128).
_VOCAB_UPPER_BOUND = 128

tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")

def gen_model_input_fn(x: peft_trainer.TrainingInput):
  pad_mask = x.input_tokens != tokenizer.pad_token_id
  positions = common.build_positions_from_mask(pad_mask)
  attention_mask = common.make_causal_attn_mask(pad_mask)
  return {
      'input_tokens': x.input_tokens,
      'input_mask': x.input_mask,
      'positions': positions,
      'attention_mask': attention_mask,
  }

def _make_dataset(
    training_input_cls: Any,
    num_steps: int,
    batch_size: int,
    seq_len: int,
    vocab_upper_bound: int = _VOCAB_UPPER_BOUND,
) -> List[Any]:
  """Creates `num_steps` deterministic single-batch training inputs."""
  rng = np.random.default_rng(0)
  dataset = []
  for _ in range(num_steps):
    tokens = rng.integers(
        0, vocab_upper_bound, size=(batch_size, seq_len)
    ).astype(np.int32)
    dataset.append(
        training_input_cls(
            input_tokens=tokens,
            input_mask=np.ones((batch_size, seq_len), dtype=np.int32),
        )
    )
  return dataset


dataset = _make_dataset(
    peft_trainer.TrainingInput,
    num_steps=10,
    batch_size=_BATCH_SIZE,
    seq_len=_SEQ_LEN,
    )

if len(jax.devices()) == 8:
  mesh = jax.make_mesh((1, 2), ('fsdp', 'tp'), axis_types=(jax.sharding.AxisType.Auto,) * 2, devices=np.asarray(jax.devices())[:2])
else:
  mesh = jax.make_mesh((1,), ('fsdp',), axis_types=(jax.sharding.AxisType.Auto,))

from flax import nnx
from tunix.models.gemma4 import model as g4_model

def create_sharded_model(config, rngs, mesh):
  @nnx.jit
  def _init(rngs):
    model = g4_model.Gemma4(config, rngs=rngs, text_only=True)
    state = nnx.state(model)
    pspecs = nnx.get_partition_spec(state)
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state)
    return model

  with mesh:
    return _init(rngs)


tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
config = g4_model.ModelConfig.gemma4_e2b()
config.num_layers = 12
# Findings (gemma4_e2b, num_layers=12, 2 devices; logs at the bottom of this
# file). "identical" means bit-for-bit, checked with tc.tree_bit_checksum.
#
#   fp32, excess_precision=false, SPLIT path : 195/197 leaves identical;
#         layers 4 and 9 differ by <= 2 fp32 eps (2.1e-7 of tensor scale).
#         After one SGD step at lr=1e-5, 54-87 weights out of 6.3M move by
#         ~1 ULP.
#   fp32, excess_precision=false, FUSED path : identical.
#   bf16, excess_precision=false, 1 and 2 dev : identical.
#   bf16, excess_precision=true,  1 and 2 dev : grads differ by 1-2 bf16 ULP on
#         ~24.7% of elements; grad_norm differs 1.35e-4 rel. Independent of the
#         device count.
#
# Two separate mechanisms, and one flag only fixes one of them:
#
#   Excess precision. With --xla_allow_excess_precision=true (the default) XLA
#   may keep intermediates in fp32 and skip convert(bf16) nodes; how much it
#   keeps depends on fusion, and v1 (one module) fuses differently from v2's
#   split path (two modules). Setting the flag to false removes this class.
#   It is necessary for bf16 bit-identity.
#
#   Reduction order. The flag does NOT constrain summation trees, collective
#   algorithms, or the scheduling of a multi-term accumulation. That is what
#   remains in fp32, and its location is the tell: the ONLY leaves that differ
#   are layers 4 and 9, which for this config are the GLOBAL attention layers
#   (index % 5 == 4) and, with num_unshared = int(12 - 12*20/35) = 5, exactly
#   the KV-sharing origin/consumer pair. Layer 4's gradient accumulates its own
#   attention path plus the one flowing back through layer 9's shared KV --
#   two contributions from different depths, which XLA schedules differently
#   when the backward is compiled alone versus together with the update. Every
#   other layer has a purely local gradient path and comes out identical.
#
# So the flag is necessary but NOT sufficient. Only the fused path guarantees
# bit-identity, because it makes the two HLO modules identical and leaves XLA no
# choice. On the split path, compare with a relative tolerance -- roughly
# max|x-y| / max|x| < 1e-6 -- not with a ULP gate (see CAVEAT 4).
#
# CAVEAT 1: grad_norm is not a valid pass/fail gate. optax.global_norm sums
# ~4e8 squares in fp32, and XLA picks a different reduction tree per module,
# so the norm can differ while the arrays are bit-identical. Observed noise
# floor with identical grads: 1 to 6 fp32 ULP (6.7e-8 to 3.8e-7 rel). The real
# divergence above is 1.35e-4 rel, ~1000x higher, so the two are cleanly
# separable -- but use tc.assert_close_ulp / tc.tree_bit_checksum to decide.
#
# CAVEAT 2: numbers are NOT comparable across configurations. The norms range
# over 1815 / 1939 / 1843 / 1995 / 2094 and the losses differ too, because the
# initial weights differ between runs: param_dtype is passed straight into
# nnx.initializers.normal(dtype=...), and jax.random.normal consumes a
# different number of random bits for bf16 than for fp32 -- it draws a
# different set of numbers, not a rounded version of the same one. Changing the
# device count also changed the loss (12.531497 on 1 dev vs 12.530317 on 2 dev
# at bf16/false), which is far above fp32 reduction noise and so likewise
# implies different weights. Only within-run v1-vs-v2 comparisons are
# meaningful. To compare across configurations, initialize once, checkpoint,
# and restore the same weights everywhere.
#
# CAVEAT 3: observing the gradients changes them. Gradients are an internal
# temporary of the traced step; the only way to see them from Python is to make
# them a program output, which forces materialization and changes the fusion,
# scheduling and buffer assignment of the very backward pass being measured.
# Every hook added during this investigation moved the numbers:
#   - v2's original jax.debug.print / pytree_xor_checksum calls inhibited fusion.
#     They are why an earlier version of this comment claimed fp32 was
#     "bit-identical unconditionally" -- that result was an artifact of the
#     instrumentation, not a property of the trainers.
#   - A hook that parked v1's depth-1 gradients in the accumulator changed v1's
#     XOR from 1000145275 to 508432006.
#   - Forcing v2's accumulator to be preallocated changed v2's from 1000145275
#     to 295270455: it pins the gradient sharding to the parameter partition
#     spec, which changes the collective and hence the summation order.
# Both hooks have since been removed rather than left behind a flag, because a
# knob that silently changes the numbers is worse than no knob at all.
# Both hooks moved their side, and by different amounts, so an instrumented run
# compares two programs that neither matches the uninstrumented one. Draw
# conclusions only from quantities observable without a hook: the loss, the
# grad_norm the step already returns, and the weights after the step.
#
# CAVEAT 4: ULP distance is the wrong metric for gradients. It is scale-free,
# which is right for a tensor whose entries share a magnitude and wrong for one
# spanning tens of orders of magnitude: two entries that are both numerically
# zero (1e-38 vs 1.5e-38) sit millions of ULP apart while contributing nothing
# to the norm or to the update. The fp32 divergence above reported max ULP
# 6.3e6 on 78% of elements and was, in absolute terms, 2.1e-7 of the tensor
# scale. Use ULP for weights and activations; use max|x-y| / max|x| for
# gradients.
#
# config.param_dtype = jnp.bfloat16
# config.dtype = jnp.bfloat16

rngs = nnx.Rngs(0)
gemma = create_sharded_model(config, rngs, mesh)

# HBM hygiene: v2 must start from exactly the weights v1 started from, but a
# second device-resident copy must NOT be alive while v1 is being profiled --
# otherwise v1's peak carries a full parameter tree that has nothing to do with
# v1. So snapshot the initial weights to host memory (numpy) instead, remember
# each leaf's sharding, and rebuild the device arrays only after v1 is done.
# Host cost is one unsharded fp32 copy of the tree (~7 GiB here), which is
# cheap on a TPU host; device cost during v1 is zero.
def _snapshot_sharding(x):
  # Non-array leaves (if any) have no sharding to restore.
  return getattr(x, "sharding", None)


def _restore_to_device(host_value, sharding):
  return host_value if sharding is None else jax.device_put(host_value, sharding)


# NOTE: an earlier version of this script tried to print XLA's own
# argument/output/temp/alias accounting by lowering the jitted step functions
# (`compiled.memory_analysis()`). That does not work for `nnx.jit`: the wrapper
# has no `.lower`, and reaching the inner `jax.jit` bypasses nnx's update
# context ("No update context found for tag <JitWrapped ...>"). Read the same
# breakdown from the xprof Memory Viewer instead -- arguments and outputs are
# the heap-resident part, temp is the scratch XLA can overlap with activations.


def _live_device_gib():
  # nbytes is the *global* (all-shard) size, so this is the unsharded-equivalent
  # total, not the per-device figure the profiler reports. Divide by the number
  # of devices the arrays are sharded over for a per-device estimate.
  return sum(a.nbytes for a in jax.live_arrays()) / 2**30


graph, _init_state = nnx.split(gemma)
_init_shardings = jax.tree.map(_snapshot_sharding, _init_state)
_init_state_host = jax.device_get(_init_state)
del _init_state
gc.collect()

# Bit-exact fingerprint of the weights the two trainers start from. Print this
# in every run: if it changes between two runs you are comparing two different
# models, and any grad_norm difference between those runs says nothing about
# v1-vs-v2 (see CAVEAT 2 above).
_init_ck = tc.tree_bit_checksum(_init_state_host)
print(f"INIT weight checksum: {_init_ck}")

# Two comparisons with very different sensitivities, so they get different
# gates -- and, deliberately, different METRICS.
#
#   gradients: gated on ULP, threshold 0. The two trainers are supposed to
#     compute these identically, so any difference at all is a real divergence
#     in the compiled backward and worth seeing.
#
#   weights after one SGD step: gated on max|dw| / max|w|, NOT on ULP.
#     `w - lr*g` with lr=1e-5 discards all but ~9 bits of the update, and
#     rounding is a threshold operation, so an arbitrarily small difference in
#     `g` flips the result by a full ULP whenever the exact value sits near a
#     tie. Weights are an amplifier, not an independent signal.
#
# Why not ULP for weights: the two metrics normalise differently.
#
#     k ULP      -> relative to the ELEMENT's magnitude:  |dw| ~ k*eps*|w_elem|
#     |dw|/scale -> relative to the TENSOR's max
#     |dw|/scale = k * eps * (|w_elem| / max|w|)  <=  k * eps
#
# So k ULP only bounds the scale-relative difference; the actual value can be
# orders of magnitude smaller when the differing entries are not the large ones.
# Measured here: layers.9 reported max ULP 24 (upper bound 2.9e-6) while the
# actual max|dw|/max|w| was 1.8e-8 -- 160x below the bound, because those
# entries sit at 0.23 of the tensor max. A ULP gate therefore raises false
# alarms on exactly the entries that matter least.
#
# For reference, in fp32 (eps = 1.19e-7):
#     |dw|/scale  1e-7  <->    0.8 ULP   noise floor
#                 1e-6  <->    8.4 ULP   suspicious
#                 1e-4  <->  839   ULP   real divergence
_GRAD_MAX_ULP = 0
_WEIGHT_MAX_REL = 1e-6

# HOW TO COMPARE GRADIENTS: set gradient_accumulation_steps=2 on both configs.
#
# At depth 1 gradients are an internal temporary of the traced step and there is
# deliberately no hook to expose them -- adding one changes the graph being
# measured and the numbers stop meaning anything (CAVEAT 3). At depth > 1 the
# accumulator is a genuine part of the algorithm: both trainers write it with
# `add()` and read it with `get()`, so `trainer.grad_accumulator.grads` can be
# read after the step with no instrumentation at all, and both sides allocate it
# eagerly so the layouts match.
#
# At depth 1, compare the quantities that are observable without a hook: the
# loss, the grad_norm the step already returns, and the weights after the step.


def diagnose_leaf(path, x, y):
  """Explains *how* two versions of one tensor differ.

  Three questions, because each one alone can mislead:

  1. Are they bit-identical? (max ULP == 0)
  2. If not, is it the same multiset in a different arrangement? Sorting removes
     position from the picture, which catches the case where the two arrays were
     assembled from device shards under different shardings -- an elementwise
     diff then compares mismatched positions and reports an enormous difference
     for arrays holding exactly the same numbers.
  3. If the values really differ, *by how much in absolute terms*? ULP distance
     is scale-free, which is what you want for a tensor whose entries share a
     magnitude, and exactly what you do not want for a gradient spanning tens of
     orders of magnitude: two entries that are both numerically zero (say 1e-38
     vs 1.5e-38) sit millions of ULP apart while contributing nothing to the
     norm or to the weight update. So report the absolute difference and its
     size relative to the tensor's own scale, which is what actually decides
     whether the difference can affect training.
  """
  # ULP and the bitwise sort comparison must run in the STORAGE dtype: one ULP
  # of bfloat16 is 2^16 times one ULP of float32, so forcing float32 here would
  # inflate every bf16 distance by that factor and make the numbers meaningless.
  xs = np.asarray(x)
  ys = np.asarray(y)
  int_view = {1: np.int8, 2: np.int16, 4: np.int32, 8: np.int64}[xs.dtype.itemsize]
  d = tc.ulp_dist(xs, ys)
  max_ulp = int(d.max())
  if max_ulp == 0:
    print(f"[diag] {path}: shape={xs.shape} dtype={xs.dtype} -> bit-identical")
    return

  same_sorted = np.array_equal(
      np.sort(xs.ravel()).view(int_view),
      np.sort(ys.ravel()).view(int_view),
  )
  # float64 only for the magnitude arithmetic, so the check itself cannot
  # overflow or lose precision.
  x = xs.astype(np.float64)
  y = ys.astype(np.float64)
  scale = float(np.abs(x).max())
  absdiff = np.abs(x - y)
  worst = int(np.argmax(absdiff))
  bad = d > 0
  # Magnitude of the entries that disagree, versus the tensor's own scale.
  mag_of_differing = float(np.abs(x).ravel()[bad.ravel()].max()) if bad.any() else 0.0
  print(
      f"[diag] {path}: shape={xs.shape} dtype={xs.dtype}\n"
      f"        max ULP={max_ulp} violating={int(bad.sum())}/{d.size}"
      f"  sorted-identical={same_sorted}\n"
      f"        tensor scale max|g|={scale:.6g}\n"
      f"        max |x-y|={absdiff.max():.6g}"
      f"  (= {absdiff.max() / scale:.3e} of scale)"
      f"  at |x|={abs(float(x.ravel()[worst])):.6g}\n"
      f"        largest |x| among differing entries={mag_of_differing:.6g}"
      f"  (= {mag_of_differing / scale:.3e} of scale)"
  )


def _scale_rel(x, y):
  """max|x-y| / max|x| for one leaf, in float64."""
  x = np.asarray(x).astype(np.float64)
  y = np.asarray(y).astype(np.float64)
  scale = float(np.abs(x).max())
  if scale == 0.0:
    return 0.0
  return float(np.abs(x - y).max()) / scale


def compare(tag, a, b, *, max_ulp=None, max_rel=None):
  """Compares two pytrees, gating on ULP or on scale-relative difference.

  Exactly one of `max_ulp` / `max_rel` should be given. Use ULP when the two
  sides are supposed to be bit-identical and every last bit is signal; use
  `max_rel` (max|x-y| / max|x| per leaf) when small entries carry large relative
  error that does not matter -- see the note above the thresholds.
  """
  ck_a, ck_b = tc.tree_bit_checksum(a), tc.tree_bit_checksum(b)
  status = "identical" if ck_a == ck_b else "DIFFER"
  print(f"[cmp] {tag}: XOR v1={ck_a} v2={ck_b} {status}")
  if ck_a == ck_b:
    return

  pairs = [
      (pth, l1, l2)
      for (pth, l1), (_, l2) in zip(
          jax.tree_util.tree_flatten_with_path(a)[0],
          jax.tree_util.tree_flatten_with_path(b)[0],
      )
  ]

  if max_rel is not None:
    scored = sorted(
        ((_scale_rel(p, q), pth, p, q) for pth, p, q in pairs),
        key=lambda t: -t[0],
    )
    worst_rel, worst_path = scored[0][0], scored[0][1]
    over = [s for s in scored if s[0] > max_rel]
    verdict = "EXCEEDS" if over else "within"
    print(
        f"[cmp] {tag}: {verdict} max|dw|/max|w| = {max_rel:.0e}"
        f"  (worst leaf {jax.tree_util.keystr(worst_path)}: {worst_rel:.2e},"
        f" {len(over)}/{len(scored)} leaves over)"
    )
    for _, path, p, q in scored[:3]:
      diagnose_leaf(jax.tree_util.keystr(path), p, q)
    return

  try:
    jax.tree.map_with_path(
        lambda path, x, y: tc.assert_close_ulp(path, x, y, max_ulp=max_ulp),
        a,
        b,
    )
    print(f"[cmp] {tag}: within {max_ulp} ULP")
  except AssertionError as e:
    print(f"[cmp] {tag}: EXCEEDS {max_ulp} ULP\n{e}")
    # Report the worst few leaves in a position-independent way.
    worst = sorted(
        (
            (int(tc.ulp_dist(np.asarray(p), np.asarray(q)).max()), pth, p, q)
            for pth, p, q in pairs
        ),
        key=lambda t: -t[0],
    )[:3]
    for _, path, p, q in worst:
      diagnose_leaf(jax.tree_util.keystr(path), p, q)


def describe_grads(tag, grads_host, expected_norm=None):
  """Host-side sanity check on a gradient tree read out of an accumulator.

  Deliberately independent of XOR checksums and of any assumption about how nnx
  round-trips module state: it just adds up the numbers that actually came back.
  Compare `norm` against the grad_norm the trainer printed. If they agree, the
  accumulator really holds the step's gradients; if `norm` is absurd or the
  nonzero-leaf count is short, what came back is not the gradients and any
  comparison built on it is meaningless.
  """
  leaves = [np.asarray(l) for l in jax.tree_util.tree_leaves(grads_host)]
  nonzero = sum(1 for l in leaves if np.any(l != 0))
  finite = sum(1 for l in leaves if np.all(np.isfinite(l)))
  # float64 accumulation so the check itself cannot be the thing that is wrong.
  norm = float(np.sqrt(sum(float(np.sum(l.astype(np.float64) ** 2)) for l in leaves)))
  biggest = max((float(np.abs(l).max()) for l in leaves if l.size), default=0.0)
  print(
      f"[grads] {tag}: leaves={len(leaves)} nonzero={nonzero} finite={finite} "
      f"max|g|={biggest:.6g} norm={norm:.6f}"
      + (
          f" expected~{float(expected_norm):.6f}"
          f" rel={abs(norm - float(expected_norm)) / float(expected_norm):.3e}"
          if expected_norm is not None
          else ""
      )
  )
  return norm


with mesh:
  # v1
  optimizer_v1 = optax.inject_hyperparams(optax.sgd)(
      learning_rate=optax.constant_schedule(_LEARNING_RATE)
  )
  config_v1 = peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=8, gradient_accumulation_steps=4)
  trainer_v1 = peft_trainer.PeftTrainer(gemma, optimizer_v1, config_v1)
  trainer_v1 = trainer_v1.with_gen_model_input_fn(gen_model_input_fn)
  with jax.profiler.trace(log_dir="gs://linchai-bucket-dev/xprof/grad_diff"):
    trainer_v1.train(dataset, skip_jit=False)
    jax.effects_barrier()

  model_state_v1 = nnx.state(gemma)
  opt_state_v1 = nnx.state(trainer_v1.optimizer)
  # Empty at depth 1 (the fast path never touches the accumulator); holds the
  # accumulated gradients when gradient_accumulation_steps > 1, and v1's depth-1
  # branch never calls reset() -- the only reset() lives in `apply_updates`,
  # which runs on the accumulating nnx.cond path.
  grads_v1_host = jax.device_get(trainer_v1.grad_accumulator.grads)
  if jax.tree_util.tree_leaves(grads_v1_host):
    describe_grads("v1", grads_v1_host)
  # loss_v1 = trainer_v1.metrics_logger.get_metric("", "loss", "train")
  del gemma, trainer_v1, optimizer_v1
  gc.collect()

# `nnx.state()` returns a pytree that holds the device buffers themselves, so
# `del gemma` above frees nothing while model_state_v1 is alive -- v1's whole
# parameter tree would stay resident for all of v2's run and show up in v2's
# peak. Snapshot to host if the post-v1 weights are still wanted, then drop the
# device copies before v2 allocates anything.
model_state_v1_host = jax.device_get(model_state_v1)
opt_state_v1_host = jax.device_get(opt_state_v1)
del model_state_v1, opt_state_v1
gc.collect()

print(f"live device arrays after v1 teardown: {_live_device_gib():.2f} GiB "
      "(global, unsharded-equivalent)")

with mesh:
  # Rebuild v2's model on device from the host snapshot, restoring the original
  # per-leaf sharding. Identical bits to what v1 started from -- asserted below.
  gemma_v2 = nnx.merge(
      graph,
      jax.tree.map(_restore_to_device, _init_state_host, _init_shardings),
  )
  _ck_v2 = tc.tree_bit_checksum(nnx.state(gemma_v2))
  assert _ck_v2 == _init_ck, (
      f"v2 start weights differ from v1's: {_ck_v2} != {_init_ck}"
  )
  print(f"v2 start weight checksum: {_ck_v2} OK")

with mesh:
  # v2
  optimizer_v2 = optax.inject_hyperparams(optax.sgd)(
      learning_rate=optax.constant_schedule(_LEARNING_RATE)
  )
  config_v2 = peft_trainer_v2.TrainingConfig(eval_every_n_steps=2, max_steps=8, gradient_accumulation_steps=4)
  trainer_v2 = peft_trainer_v2.PeftTrainer(gemma_v2, optimizer_v2, config_v2)
  trainer_v2 = trainer_v2.with_gen_model_input_fn(gen_model_input_fn)

  # Memory-benchmark state: `train()`, which at depth 1 routes through the fused
  # single-executable step -- the configuration whose HBM profile we want. To
  # exercise the split path instead, call fwd_bwd(dataset[0]) then update().
  with jax.profiler.trace(log_dir="gs://linchai-bucket-dev/xprof/grad_diff"):
    trainer_v2.train(dataset, skip_jit=False, cache_nnx_graph=False)
    jax.effects_barrier()

  # Populated only at gradient_accumulation_steps > 1; see the note above the
  # comparison helpers for why depth 1 has no gradient hook.
  grads_v2_host = jax.device_get(trainer_v2.grad_accumulator.grads)
  if jax.tree_util.tree_leaves(grads_v2_host):
    describe_grads("v2", grads_v2_host)
    compare("grads", grads_v1_host, grads_v2_host, max_ulp=_GRAD_MAX_ULP)

  # Weights are observable without any hook, so this comparison is valid in
  # memory-benchmark state. Expect ~1e-8 of tensor scale in the KV-sharing
  # layers on the split path, and bit-identity on the fused path.
  compare(
      "weights after training",
      model_state_v1_host,
      jax.device_get(nnx.state(gemma_v2)),
      max_rel=_WEIGHT_MAX_REL,
  )

  model_state_v2 = nnx.state(gemma_v2)
  opt_state_v2 = nnx.state(trainer_v2.optimizer)
  # loss_v2 = trainer_v2.metrics_logger.get_metric("", "loss", "train")
  del gemma_v2, trainer_v2, optimizer_v2
  gc.collect()

# # Tolerate typical bfloat16 XLA computation graph re-association on TPUs
# assert_fn = lambda path, x, y: tc.assert_close(
#     path, x, y, atol=1e-5, rtol=1e-5
# )

# jax.tree.map_with_path(assert_fn, model_state_v1, model_state_v2)
# jax.tree.map_with_path(assert_fn, opt_state_v1, opt_state_v2)
# np.testing.assert_allclose(loss_v1, loss_v2, rtol=1e-5, atol=1e-5)
# print(f"Loss: {loss_v1}, {loss_v2}")


##debug log: 
"""

bf16 with xla_allow_excess_precision=false, single-device execution:
Training:   0%|                                                                                                                                                                | 0/1 [00:00<?, ?step/s]JIT TRACE v1 [train_step] value_and_grad output dtypes: {dtype(bfloat16)}
DEBUG v1: Grad Norm in train_step: 1939.4212646484375
Debug: train_loss:  12.531497 grad_norm:  1939.4213
Training: 100%|███████████████████████████████████████████████████████████████████████████████████| 1/1 [00:21<00:00, 21.32s/step, _train_loss=12.5, _train_perplexity=2.77e+5, _train_learning_rate=0]
/home/linchai_google_com/tunix/examples/frozenlake/grad_diff_debug.py:141: DeprecationWarning: `with mesh:` context manager has been deprecated. Please use `with jax.set_mesh(mesh):` instead.
  with mesh:
JIT TRACE v2 [fwd_bwd] value_and_grad output dtypes:  {dtype(bfloat16)}
JIT TRACE v2 [fwd_bwd] grad_accumulator.grads output dtypes:  {dtype(bfloat16)}
DEBUG v2: Raw Grad Norm in fwd_bwd: 1939.4212646484375
DEBUG v2: Norm AFTER set inside fwd_bwd: 1939.4212646484375
DEBUG v2 fwd_bwd Whole-Tree XOR: 11706
Debug: train_loss:  12.531497
JIT TRACE v2 [update_step] acc_grads output dtypes: {dtype(bfloat16)}
DEBUG v2 update slice: [2.5 0.255859 -2.76562 2.07812 2.45312]
DEBUG v2 update Whole-Tree XOR: 11706
DEBUG v2: Norm in update_step: 1939.4205322265625
compare grads between v1 and v2.update: 
"""

"""
bf16 with xla_allow_excess_precision=true single-device execution:
Training:   0%|                                                                                                                                                                | 0/1 [00:00<?, ?step/s]JIT TRACE v1 [train_step] value_and_grad output dtypes: {dtype(bfloat16)}
DEBUG v1: Grad Norm in train_step: 1995.14892578125
Debug: train_loss:  12.536115 grad_norm:  1995.1489
Training: 100%|███████████████████████████████████████████████████████████████████████████████████| 1/1 [00:21<00:00, 21.04s/step, _train_loss=12.5, _train_perplexity=2.78e+5, _train_learning_rate=0]
/home/linchai_google_com/tunix/examples/frozenlake/grad_diff_debug.py:141: DeprecationWarning: `with mesh:` context manager has been deprecated. Please use `with jax.set_mesh(mesh):` instead.
  with mesh:
JIT TRACE v2 [fwd_bwd] value_and_grad output dtypes:  {dtype(bfloat16)}
JIT TRACE v2 [fwd_bwd] grad_accumulator.grads output dtypes:  {dtype(bfloat16)}
DEBUG v2: Raw Grad Norm in fwd_bwd: 1994.87353515625
DEBUG v2: Norm AFTER set inside fwd_bwd: 1994.87353515625
DEBUG v2 fwd_bwd Whole-Tree XOR: 27483
Debug: train_loss:  12.536115
JIT TRACE v2 [update_step] acc_grads output dtypes: {dtype(bfloat16)}
DEBUG v2 update slice: [-1.14062 -0.644531 1.35156 -0.464844 -1.54688]
DEBUG v2 update Whole-Tree XOR: 27483
DEBUG v2: Norm in update_step: 1994.87158203125
compare grads between v1 and v2.update: 
Traceback (most recent call last):
  File "/home/linchai_google_com/tunix/examples/frozenlake/grad_diff_debug.py", line 160, in <module>
    jax.tree.map_with_path(
  File "/home/linchai_google_com/miniconda3/envs/tunix/lib/python3.12/site-packages/jax/_src/tree.py", line 425, in map_with_path
    return tree_util.tree_map_with_path(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/linchai_google_com/miniconda3/envs/tunix/lib/python3.12/site-packages/jax/_src/tree_util.py", line 1258, in tree_map_with_path
    return treedef.unflatten(f(*xs) for xs in zip(*all_keypath_leaves))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/linchai_google_com/miniconda3/envs/tunix/lib/python3.12/site-packages/jax/_src/tree_util.py", line 1258, in <genexpr>
    return treedef.unflatten(f(*xs) for xs in zip(*all_keypath_leaves))
                             ^^^^^^
  File "/home/linchai_google_com/tunix/examples/frozenlake/grad_diff_debug.py", line 161, in <lambda>
    lambda path, g1, g2: tc.assert_close(path, g1, g2, atol=1e-5, rtol=1e-5),
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/linchai_google_com/tunix/tunix/tests/test_common.py", line 69, in assert_close
    np.testing.assert_allclose(
  File "/home/linchai_google_com/miniconda3/envs/tunix/lib/python3.12/site-packages/numpy/testing/_private/utils.py", line 1711, in assert_allclose
    assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
  File "/home/linchai_google_com/miniconda3/envs/tunix/lib/python3.12/site-packages/numpy/testing/_private/utils.py", line 919, in assert_array_compare
    raise AssertionError(msg)
AssertionError: 
Not equal to tolerance rtol=1e-05, atol=1e-05
Mismatch at path: (DictKey(key='layers'), DictKey(key=0), DictKey(key='attn'), DictKey(key='attn_vec_einsum'), DictKey(key='w'), GetAttrKey(name='value'))
Mismatched elements: 776927 / 3145728 (24.7%)
Max absolute difference among violations: 0.00390625
Max relative difference among violations: 0.00775146
 ACTUAL: array([[[0.202148, -0.131836, -0.166016, ..., 0.211914, -0.0776367,
         -0.0737305],
        [-0.189453, -0.121582, -0.178711, ..., -0.15332, 0.097168,...
 DESIRED: array([[[0.201172, -0.131836, -0.166016, ..., 0.210938, -0.0776367,
         -0.0737305],
        [-0.188477, -0.121582, -0.178711, ..., -0.15332, 0.097168,...
"""

"""
bf16 with xla_allow_excess_precision=true, 2-device execution:
Training:   0%|                                                                                                                                                                | 0/1 [00:00<?, ?step/s]JIT TRACE v1 [train_step] value_and_grad output dtypes: {dtype(bfloat16)}
DEBUG v1: Grad Norm in train_step: 1842.9820556640625
Debug: train_loss:  12.536715 grad_norm:  1842.982
Training: 100%|███████████████████████████████████████████████████████████████████████████████████| 1/1 [00:22<00:00, 22.33s/step, _train_loss=12.5, _train_perplexity=2.78e+5, _train_learning_rate=0]
/home/linchai_google_com/tunix/examples/frozenlake/grad_diff_debug.py:141: DeprecationWarning: `with mesh:` context manager has been deprecated. Please use `with jax.set_mesh(mesh):` instead.
  with mesh:
JIT TRACE v2 [fwd_bwd] value_and_grad output dtypes:  {dtype(bfloat16)}
JIT TRACE v2 [fwd_bwd] grad_accumulator.grads output dtypes:  {dtype(bfloat16)}
DEBUG v2: Raw Grad Norm in fwd_bwd: 1842.7333984375
DEBUG v2: Norm AFTER set inside fwd_bwd: 1842.7333984375
DEBUG v2 fwd_bwd Whole-Tree XOR: 16781
Debug: train_loss:  12.536715
JIT TRACE v2 [update_step] acc_grads output dtypes: {dtype(bfloat16)}
DEBUG v2 update slice: [0.308594 -0.322266 0.550781 0.404297 -0.243164]
DEBUG v2: Norm in update_step: 1842.734130859375
DEBUG v2 update Whole-Tree XOR: 16781
compare grads between v1 and v2.update: 
Traceback (most recent call last):
  File "/home/linchai_google_com/tunix/examples/frozenlake/grad_diff_debug.py", line 160, in <module>
    jax.tree.map_with_path(
  File "/home/linchai_google_com/miniconda3/envs/tunix/lib/python3.12/site-packages/jax/_src/tree.py", line 425, in map_with_path
    return tree_util.tree_map_with_path(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/linchai_google_com/miniconda3/envs/tunix/lib/python3.12/site-packages/jax/_src/tree_util.py", line 1258, in tree_map_with_path
    return treedef.unflatten(f(*xs) for xs in zip(*all_keypath_leaves))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/linchai_google_com/miniconda3/envs/tunix/lib/python3.12/site-packages/jax/_src/tree_util.py", line 1258, in <genexpr>
    return treedef.unflatten(f(*xs) for xs in zip(*all_keypath_leaves))
                             ^^^^^^
  File "/home/linchai_google_com/tunix/examples/frozenlake/grad_diff_debug.py", line 161, in <lambda>
    lambda path, g1, g2: tc.assert_close(path, g1, g2, atol=1e-5, rtol=1e-5),
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/linchai_google_com/tunix/tunix/tests/test_common.py", line 69, in assert_close
    np.testing.assert_allclose(
  File "/home/linchai_google_com/miniconda3/envs/tunix/lib/python3.12/site-packages/numpy/testing/_private/utils.py", line 1711, in assert_allclose
    assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
  File "/home/linchai_google_com/miniconda3/envs/tunix/lib/python3.12/site-packages/numpy/testing/_private/utils.py", line 919, in assert_array_compare
    raise AssertionError(msg)
AssertionError: 
Not equal to tolerance rtol=1e-05, atol=1e-05
Mismatch at path: (DictKey(key='layers'), DictKey(key=1), DictKey(key='attn'), DictKey(key='kv_einsum'), DictKey(key='w'), GetAttrKey(name='value'))
Mismatched elements: 194083 / 786432 (24.7%)
Max absolute difference among violations: 0.0078125
Max relative difference among violations: 0.00775146
 ACTUAL: array([[[[0.165039, 0.263672, 0.123535, ..., 0.359375, 0.0878906,
          0.107422],
         [-0.28125, 0.484375, -0.219727, ..., 0.289062, 0.235352,...
 DESIRED: array([[[[0.165039, 0.263672, 0.123535, ..., 0.361328, 0.0883789,
          0.107422],
         [-0.283203, 0.484375, -0.21875, ..., 0.289062, 0.235352,...
         
The result doesn't match with single-device execution.
"""

"""
bf16 with xla_allow_excess_precision=false, 2-device execution:
Training:   0%|                                                                                                                                                                | 0/1 [00:00<?, ?step/s]JIT TRACE v1 [train_step] value_and_grad output dtypes: {dtype(bfloat16)}
DEBUG v1: Grad Norm in train_step: 1814.9761962890625
Debug: train_loss:  12.530317 grad_norm:  1814.9762
Training: 100%|███████████████████████████████████████████████████████████████████████████████████| 1/1 [00:22<00:00, 22.44s/step, _train_loss=12.5, _train_perplexity=2.77e+5, _train_learning_rate=0]
/home/linchai_google_com/tunix/examples/frozenlake/grad_diff_debug.py:141: DeprecationWarning: `with mesh:` context manager has been deprecated. Please use `with jax.set_mesh(mesh):` instead.
  with mesh:
JIT TRACE v2 [fwd_bwd] value_and_grad output dtypes:  {dtype(bfloat16)}
JIT TRACE v2 [fwd_bwd] grad_accumulator.grads output dtypes:  {dtype(bfloat16)}
DEBUG v2: Raw Grad Norm in fwd_bwd: 1814.97607421875
DEBUG v2: Norm AFTER set inside fwd_bwd: 1814.97607421875
DEBUG v2 fwd_bwd Whole-Tree XOR: 29441
Debug: train_loss:  12.530317
JIT TRACE v2 [update_step] acc_grads output dtypes: {dtype(bfloat16)}
DEBUG v2 update slice: [1.66406 -0.417969 -1.11719 1.16406 0.314453]
DEBUG v2: Norm in update_step: 1814.97607421875
DEBUG v2 update Whole-Tree XOR: 29441
compare grads between v1 and v2.update: 
"""

"""
fp32 with xla_allow_excess_precision=false, 2-device execution:
Training:   0%|                                                                                                                                                                | 0/1 [00:00<?, ?step/s]JIT TRACE v1 [train_step] value_and_grad output dtypes: {dtype('float32')}
DEBUG v1: Grad Norm in train_step: 2094.20654296875
Debug: train_loss:  12.544204 grad_norm:  2094.2065
Training: 100%|████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:20<00:00, 20.94s/step, _train_loss=12.5, _train_perplexity=2.8e+5, _train_learning_rate=0]
/home/linchai_google_com/tunix/examples/frozenlake/grad_diff_debug.py:141: DeprecationWarning: `with mesh:` context manager has been deprecated. Please use `with jax.set_mesh(mesh):` instead.
  with mesh:
JIT TRACE v2 [fwd_bwd] value_and_grad output dtypes:  {dtype('float32')}
JIT TRACE v2 [fwd_bwd] grad_accumulator.grads output dtypes:  {dtype('float32')}
DEBUG v2: Raw Grad Norm in fwd_bwd: 2094.20654296875
DEBUG v2: Norm AFTER set inside fwd_bwd: 2094.20654296875
DEBUG v2 fwd_bwd Whole-Tree XOR: 1000145275
Debug: train_loss:  12.544204
JIT TRACE v2 [update_step] acc_grads output dtypes: {dtype('float32')}
DEBUG v2 update Whole-Tree XOR: 1000145275
DEBUG v2 update slice: [-0.00412249  0.467074    0.9625872  -0.02641185  0.47519204]
DEBUG v2: Norm in update_step: 2094.20654296875
compare grads between v1 and v2.update: 
"""
"""
fp32 with xla_allow_excess_precision=true, 2-device execution:
Training:   0%|                                                                                                                                                                | 0/1 [00:00<?, ?step/s]JIT TRACE v1 [train_step] value_and_grad output dtypes: {dtype('float32')}
DEBUG v1: Grad Norm in train_step: 2094.20654296875
Debug: train_loss:  12.544204 grad_norm:  2094.2065
Training: 100%|████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:21<00:00, 21.71s/step, _train_loss=12.5, _train_perplexity=2.8e+5, _train_learning_rate=0]
/home/linchai_google_com/tunix/examples/frozenlake/grad_diff_debug.py:141: DeprecationWarning: `with mesh:` context manager has been deprecated. Please use `with jax.set_mesh(mesh):` instead.
  with mesh:
JIT TRACE v2 [fwd_bwd] value_and_grad output dtypes:  {dtype('float32')}
JIT TRACE v2 [fwd_bwd] grad_accumulator.grads output dtypes:  {dtype('float32')}
DEBUG v2: Raw Grad Norm in fwd_bwd: 2094.20654296875
DEBUG v2: Norm AFTER set inside fwd_bwd: 2094.20654296875
DEBUG v2 fwd_bwd Whole-Tree XOR: 1000145275
Debug: train_loss:  12.544204
JIT TRACE v2 [update_step] acc_grads output dtypes: {dtype('float32')}
DEBUG v2 update Whole-Tree XOR: 1000145275
DEBUG v2 update slice: [-0.00412249  0.467074    0.9625872  -0.02641185  0.47519204]
DEBUG v2: Norm in update_step: 2094.20654296875
compare grads between v1 and v2.update: 
"""





