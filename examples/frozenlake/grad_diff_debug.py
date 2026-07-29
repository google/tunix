# Compare v1 and v2 weight difference
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
config.num_layers = 5

rngs = nnx.Rngs(0)
gemma = create_sharded_model(config, rngs, mesh)

with mesh:
  # v1
  model_v1 = create_sharded_model(config, rngs, mesh) 
  optimizer_v1 = optax.inject_hyperparams(optax.sgd)(
      learning_rate=optax.constant_schedule(_LEARNING_RATE)
  )
  config_v1 = peft_trainer.TrainingConfig(eval_every_n_steps=2, max_steps=1)
  trainer_v1 = peft_trainer.PeftTrainer(model_v1, optimizer_v1, config_v1)
  trainer_v1 = trainer_v1.with_gen_model_input_fn(gen_model_input_fn)
  trainer_v1.train(dataset, skip_jit=False)
  jax.effects_barrier()

  model_state_v1 = nnx.state(model_v1)
  opt_state_v1 = nnx.state(trainer_v1.optimizer)
  loss_v1 = trainer_v1.metrics_logger.get_metric("", "loss", "train")
  del model_v1, optimizer_v1
  gc.collect()

with mesh:
  # v2
  model_v2 = create_sharded_model(config, rngs, mesh)
  optimizer_v2 = optax.inject_hyperparams(optax.sgd)(
      learning_rate=optax.constant_schedule(_LEARNING_RATE)
  )
  config_v2 = peft_trainer_v2.TrainingConfig(eval_every_n_steps=2, max_steps=1)
  trainer_v2 = peft_trainer_v2.PeftTrainer(model_v2, optimizer_v2, config_v2)
  trainer_v2 = trainer_v2.with_gen_model_input_fn(gen_model_input_fn)
  trainer_v2.fwd_bwd(dataset[0])
  jax.effects_barrier()
  print("compare grads between v1 and v2.fwd_bwd: ")
  jax.tree.map_with_path(
    lambda path, g1, g2: tc.assert_close(path, g1, g2, atol=1e-5, rtol=1e-5),
    trainer_v1.grad_accumulator.grads,
    trainer_v2.grad_accumulator.grads,
  )
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

  model_state_v2 = nnx.state(model_v2)
  opt_state_v2 = nnx.state(trainer_v2.optimizer)
  loss_v2 = trainer_v2.metrics_logger.get_metric("", "loss", "train")
  del model_v2, trainer_v2, optimizer_v2
  gc.collect()

# # Tolerate typical bfloat16 XLA computation graph re-association on TPUs
# assert_fn = lambda path, x, y: tc.assert_close(
#     path, x, y, atol=1e-5, rtol=1e-5
# )

# jax.tree.map_with_path(assert_fn, model_state_v1, model_state_v2)
# jax.tree.map_with_path(assert_fn, opt_state_v1, opt_state_v2)
# np.testing.assert_allclose(loss_v1, loss_v2, rtol=1e-5, atol=1e-5)
# print(f"Loss: {loss_v1}, {loss_v2}")