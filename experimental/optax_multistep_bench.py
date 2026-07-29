"""optax.MultiSteps native-accumulation HBM/speed baseline.

Parity counterpart to `mem_repro_fix_accum.sh` (which measures our custom
`GradientAccumulator`). Runs a bare nnx training loop with the optimizer wrapped
in `optax.MultiSteps` -- bypassing PeftTrainer -- so we can measure optax's own
gradient accumulation under the same 4 states and compare HBM/wall.

Arms (mirror the custom-accumulator arms):
  optax_d1              plain adamw, no MultiSteps -> no accumulator
  optax_d4_fp32_accum   MultiSteps(k=4, accumulator_dtype=float32), bf16 moments
  optax_d4_bf16_accum   MultiSteps(k=4, accumulator_dtype=bfloat16), bf16 moments
  optax_d4_fp32_moments MultiSteps(k=4, accumulator_dtype=float32) + moments->fp32

Usage (real model, TPU):
  python3 experimental/optax_multistep_bench.py --arm_name optax_d4_fp32_accum \
      --grad_accum_steps 4 --model gemma4 --model_path <path>
Usage (local smoke, CPU):
  XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
  python3 experimental/optax_multistep_bench.py --arm_name optax_d4_fp32_accum \
      --grad_accum_steps 4 --model toy
"""

import argparse
import time

import jax
import jax.numpy as jnp
from flax import nnx
import optax

from experimental.compile_repro_sft import build_model_and_mesh
from experimental.compile_repro_sft import gen_model_input_fn
from tunix.sft import peft_trainer
from tunix.sft.peft_trainer import _cast_opt_state_floats


def parse_args():
  p = argparse.ArgumentParser()
  p.add_argument("--arm_name", required=True)
  p.add_argument("--grad_accum_steps", type=int, default=4)
  p.add_argument("--model", choices=["gemma4", "toy"], default="gemma4")
  p.add_argument("--model_path", default="")
  p.add_argument("--mesh_fsdp", type=int, default=2)
  p.add_argument("--mesh_tp", type=int, default=2)
  p.add_argument("--seq_len", type=int, default=2048)
  p.add_argument("--batch", type=int, default=4)
  return p.parse_args()


def build(args):
  """Returns (model, mesh, seq_len). Toy path is CPU-testable."""
  if args.model == "toy":
    from tunix.tests import test_common as tc  # pylint: disable=g-import-not-at-top

    mesh = jax.make_mesh(
        (args.mesh_fsdp, args.mesh_tp),
        ("fsdp", "tp"),
        axis_types=(jax.sharding.AxisType.Auto,) * 2,
    )
    m = tc.ToyTransformer(config=tc.ModelConfig(), rngs=nnx.Rngs(0))
    bf16 = jax.tree.map(
        lambda x: x.astype(jnp.bfloat16)
        if jnp.issubdtype(x.dtype, jnp.floating)
        else x,
        nnx.state(m, nnx.Param),
    )
    nnx.update(m, bf16)
    return m, mesh, 16
  m, mesh = build_model_and_mesh(args.model_path, args.mesh_fsdp, args.mesh_tp)
  return m, mesh, args.seq_len


def make_optimizer(model, arm, k):
  """Build the optax optimizer for one arm."""
  base = optax.adamw(1e-5)
  if arm == "optax_d1":
    tx = base  # no MultiSteps -> no accumulator (matches d1_default)
  else:
    accdt = jnp.bfloat16 if arm == "optax_d4_bf16_accum" else jnp.float32
    tx = optax.MultiSteps(base, every_k_schedule=k, accumulator_dtype=accdt)
  opt = nnx.Optimizer(model, tx, wrt=nnx.Param)
  if arm == "optax_d4_fp32_moments":
    # optax's adamw can only set mu's dtype (nu is always param dtype), so cast
    # all float opt-state leaves to fp32 -- mirrors our optimizer_state_dtype.
    _cast_opt_state_floats(opt, jnp.float32)
  return opt


def main():
  args = parse_args()
  m, mesh, seq = build(args)
  tok = jnp.ones((args.batch, seq), dtype=jnp.int32)
  mask = jnp.ones((args.batch, seq), dtype=jnp.int32)
  inputs = gen_model_input_fn(
      peft_trainer.TrainingInput(input_tokens=tok, input_mask=mask)
  )
  opt = make_optimizer(m, args.arm_name, args.grad_accum_steps)
  accdt = "bfloat16" if args.arm_name == "optax_d4_bf16_accum" else "float32"

  @nnx.jit
  def train_step(model, optimizer, inputs):
    def loss_fn(model):
      out = peft_trainer._default_loss_fn(model, **inputs)
      return out.primary_loss.compute()

    _, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)  # MultiSteps accumulates; emits every k

  n_steps = max(2, args.grad_accum_steps * 2)  # cover >=2 emit cycles
  t0 = time.perf_counter()
  with mesh:
    for _ in range(n_steps):
      train_step(m, opt, inputs)
    jax.block_until_ready(jax.tree_util.tree_leaves(nnx.state(m)))
  wall = time.perf_counter() - t0

  print(
      f"[[COMPILE_REPRO]] arm={args.arm_name} "
      f"mesh={args.mesh_fsdp}x{args.mesh_tp} train_wall_s={wall:.1f}",
      flush=True,
  )
  for d in jax.local_devices():
    s = d.memory_stats() or {}
    print(
        f"[[MEM]] arm={args.arm_name} steps={args.grad_accum_steps} "
        f"dtype={accdt} device={d.id} "
        f"peak_hbm_gb={s.get('peak_bytes_in_use', 0) / 1e9:.2f} "
        f"current_hbm_gb={s.get('bytes_in_use', 0) / 1e9:.2f} "
        f"limit_gb={s.get('bytes_limit', 0) / 1e9:.2f}",
        flush=True,
    )


if __name__ == "__main__":
  main()
