# Compare v1 and v2 weight difference
from shutil import copy
from typing import Any, List
import numpy as np
import jax
import gc


import os
from flax import nnx
import jax
import optax

from tunix.rl import common
from tunix.sft import peft_trainer
from tunix.experimental.train import peft_trainer_v2
from tunix.tests import test_common as tc

# Set XLA flag to disable excess precision (run this BEFORE jax initializes devices)
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
    num_steps=8,
    batch_size=_BATCH_SIZE,
    seq_len=_SEQ_LEN,
    )

if len(jax.devices()) == 8:
  mesh = jax.make_mesh((1, 1), ('fsdp', 'tp'), axis_types=(jax.sharding.AxisType.Auto,) * 2, devices=np.asarray(jax.devices())[:1])
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
# default fp32 with xla_allow_excess_precision=false: ???
# dtype bf16 with xla_allow_excess_precision=false: loss matched grad_norm  matched, grads after fwd_bwd and in update all matched with v1
# default fp32 with xla_allow_excess_precision=true: grad_norm  ???
# dtype bf16 with xla_allow_excess_precision=true: grad_norm ???
config.param_dtype = jnp.bfloat16

rngs = nnx.Rngs(0)
gemma = create_sharded_model(config, rngs, mesh)

graph, state = nnx.split(gemma)
gemma_v2 = nnx.merge(
      graph,
      jax.tree.map(jnp.copy, state),
  )
with mesh:
  # v1
  optimizer_v1 = optax.inject_hyperparams(optax.sgd)(
      learning_rate=optax.constant_schedule(_LEARNING_RATE)
  )
  config_v1 = peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=1)
  trainer_v1 = peft_trainer.PeftTrainer(gemma, optimizer_v1, config_v1)
  trainer_v1 = trainer_v1.with_gen_model_input_fn(gen_model_input_fn)
  trainer_v1.train(dataset, skip_jit=False)
  jax.effects_barrier()

  model_state_v1 = nnx.state(gemma)
  opt_state_v1 = nnx.state(trainer_v1.optimizer)
  # loss_v1 = trainer_v1.metrics_logger.get_metric("", "loss", "train")
  del gemma, optimizer_v1
  gc.collect()

with mesh:
  # v2
  optimizer_v2 = optax.inject_hyperparams(optax.sgd)(
      learning_rate=optax.constant_schedule(_LEARNING_RATE)
  )
  config_v2 = peft_trainer_v2.TrainingConfig(eval_every_n_steps=2, max_steps=1)
  trainer_v2 = peft_trainer_v2.PeftTrainer(gemma_v2, optimizer_v2, config_v2)
  trainer_v2 = trainer_v2.with_gen_model_input_fn(gen_model_input_fn)
  trainer_v2.fwd_bwd(dataset[0])
  jax.effects_barrier()
  # print("compare grads between v1 and v2.fwd_bwd: ")
  # jax.tree.map_with_path(
  #   lambda path, g1, g2: tc.assert_close(path, g1, g2, atol=1e-5, rtol=1e-5),
  #   trainer_v1.grad_accumulator.grads,
  #   trainer_v2.grad_accumulator.grads,
  # )
  trainer_v2.update()
  jax.effects_barrier()
  print("compare grads between v1 and v2.update: ")
  jax.tree.map_with_path(
    lambda path, g1, g2: tc.assert_close(path, g1, g2, atol=1e-5, rtol=1e-5),
    trainer_v1.grad_accumulator.grads,
    trainer_v2.grad_accumulator.grads,
  )
  # trainer_v2.train(dataset, skip_jit=False, cache_nnx_graph=False)
  jax.effects_barrier()

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

bf16 with xla_allow_excess_precision=false:
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
bf16 with xla_allow_excess_precision=true:
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
