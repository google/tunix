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

# # Set XLA flag to disable excess precision (run this BEFORE jax initializes devices)
# os.environ['XLA_FLAGS'] = (
#     os.environ.get('XLA_FLAGS', '') + ' --xla_allow_excess_precision=false'
# ).strip()

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
# Findings so far (gemma4_e2b, num_layers=12; full logs at the bottom of this file). 
# "matched" below means bit-for-bit identical, verified via the whole-tree XOR checksum, not via grad_norm -- see the caveat further down.
#
#   fp32, excess_precision=false, 2 dev : loss + grad_norm + grads all matched
#   fp32, excess_precision=true,  2 dev : loss + grad_norm + grads all matched
#   bf16, excess_precision=false, 1 dev : loss + grad_norm + grads all matched
#   bf16, excess_precision=false, 2 dev : loss + grad_norm + grads all matched
#   bf16, excess_precision=true,  1 dev : loss matched; grad_norm differs
#                                         1.4e-4 rel; grads differ by 1 bf16
#                                         ULP on 24.7% of elements
#   bf16, excess_precision=true,  2 dev : same as above (1.35e-4 rel), so the
#                                         divergence is independent of the
#                                         device count
#
# v1 and v2 are algorithmically equivalent -- fp32 is bit-identical unconditionally. 
# In bf16, --xla_allow_excess_precision=false is both necessary and sufficient for bit-identity.
# With excess precision allowed (the default) XLA may keep intermediates in fp32 and skip convert(bf16) nodes; how much it keeps depends on fusion,
# Different module => different fusion => different rounding points.
# So bf16 without the flag can only be compared to a ULP tolerance.
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
# config.param_dtype = jnp.bfloat16

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


def report_memory_analysis(tag, partial_fn, *extra_args):
  """Prints XLA's own memory accounting for an already-jitted step function.

  `alias_size_in_bytes` is the decisive number for the donation question: it is
  the volume of input buffers XLA actually managed to alias onto outputs. If a
  step declares `donate_argnames` for a gradient-tree-sized argument and this
  comes back ~0, donation silently degraded into a copy and the peak carries two
  trees instead of one.

  The argument/output/temp split also maps onto the profiler's heap-vs-stack
  view: arguments and outputs are the heap-resident part, temp is the scratch
  that XLA can overlap with activations.

  Best effort only -- lowering a jitted nnx function is version-sensitive, and a
  failure here must not take the benchmark down with it.
  """
  try:
    jitted = getattr(partial_fn, "func", partial_fn)
    bound = getattr(partial_fn, "args", ())
    compiled = jitted.lower(*bound, *extra_args).compile()
    m = compiled.memory_analysis()
    g = 2**30
    print(
        f"[mem] {tag}: "
        f"args={m.argument_size_in_bytes / g:.2f} "
        f"out={m.output_size_in_bytes / g:.2f} "
        f"temp={m.temp_size_in_bytes / g:.2f} "
        f"alias={m.alias_size_in_bytes / g:.2f} GiB"
    )
  except Exception as e:  # pylint: disable=broad-except
    print(f"[mem] {tag}: memory_analysis unavailable ({type(e).__name__}: {e})")


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

with mesh:
  # v1
  optimizer_v1 = optax.inject_hyperparams(optax.sgd)(
      learning_rate=optax.constant_schedule(_LEARNING_RATE)
  )
  config_v1 = peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=10)
  trainer_v1 = peft_trainer.PeftTrainer(gemma, optimizer_v1, config_v1)
  trainer_v1 = trainer_v1.with_gen_model_input_fn(gen_model_input_fn)
  with jax.profiler.trace(log_dir="gs://linchai-bucket-dev/xprof/grad_diff"):
    trainer_v1.train(dataset, skip_jit=False)
    jax.effects_barrier()

  # Executables exist only after the first step has compiled them.
  report_memory_analysis(
      "v1 train_step",
      trainer_v1._jitted_train_step_fn,  # pylint: disable=protected-access
      gen_model_input_fn(dataset[0]),
      jnp.array(True, dtype=jnp.bool_),
  )

  model_state_v1 = nnx.state(gemma)
  opt_state_v1 = nnx.state(trainer_v1.optimizer)
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
  config_v2 = peft_trainer_v2.TrainingConfig(eval_every_n_steps=2, max_steps=10)
  trainer_v2 = peft_trainer_v2.PeftTrainer(gemma_v2, optimizer_v2, config_v2)
  trainer_v2 = trainer_v2.with_gen_model_input_fn(gen_model_input_fn)
  # trainer_v2.fwd_bwd(dataset[0])
  # jax.effects_barrier()

  # # Expected max_ulp:
  # #   0 for fp32 (any flag), and for bf16 with
  # #     --xla_allow_excess_precision=false
  # #   1..2 for bf16 with excess precision allowed (the default)
  # # Raise it only if you have decided the extra rounding is acceptable; do not
  # # reach for a larger atol instead, which cannot express this bound.
  # _MAX_ULP = 0

  # NOTE: v2 still parks gradients in `grad_accumulator.grads`, but only between
  # fwd_bwd and update -- a non-persistent `reset()` drops them at the end of
  # update, so read them BEFORE calling update. v1's depth-1 fast path never
  # touches its accumulator at all, so it has no equivalent hook: to compare
  # gradients again, set gradient_accumulation_steps=2 so both trainers take the
  # accumulating path.
  # def compare_grads(tag, g1, g2):
  #   ck1, ck2 = tc.tree_bit_checksum(g1), tc.tree_bit_checksum(g2)
  #   print(f"compare grads between v1 and v2.{tag}: "
  #         f"XOR v1={ck1} v2={ck2} {'identical' if ck1 == ck2 else 'DIFFER'}")
  #   jax.tree.map_with_path(
  #       lambda path, a, b: tc.assert_close_ulp(path, a, b, max_ulp=_MAX_ULP),
  #       g1, g2,
  #   )

  # compare_grads("fwd_bwd", v1_grads, trainer_v2.grad_accumulator.grads)
  # trainer_v2.update()
  # jax.effects_barrier()
  
  with jax.profiler.trace(log_dir="gs://linchai-bucket-dev/xprof/grad_diff"):
    trainer_v2.train(dataset, skip_jit=False, cache_nnx_graph=False)
    jax.effects_barrier()

  # In the single-microstep regime the accumulator holds gradients only between
  # fwd_bwd and update: nothing is pre-allocated, and update's `reset()` drops
  # the buffer instead of zeroing it. So fwd_bwd's `args` should be ~one
  # parameter tree (the model; the accumulator arrives empty) and update's
  # `alias` should be ~one parameter tree (the donated accumulator reused for
  # the updated parameters). An `alias` of ~0 means donation degraded to a copy.
  report_memory_analysis(
      "v2 fwd_bwd",
      trainer_v2._jitted_fwd_bwd_step_fn,  # pylint: disable=protected-access
      gen_model_input_fn(dataset[0]),
  )
  report_memory_analysis(
      "v2 update",
      trainer_v2._jitted_update_step_fn,  # pylint: disable=protected-access
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





