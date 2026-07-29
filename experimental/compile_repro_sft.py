"""Compile-time repro harness: pure SFT train_step, optax vs stream grad accum.

Stripped-down version of experimental/sampling_example.ipynb (cells 1/3/4):
same Gemma4-E2B config, mesh style and dummy data, but no sampler / cache /
processor and no RL. One variant per process (fresh process = fresh jit
cache), so the trainer's own first-step elapsed (~ XLA compile time) and the
final [[COMPILE_REPRO]] wall-clock line are directly comparable across runs.

Usage (deps present, e.g. inside the tunix_base_image container):
  python3 experimental/compile_repro_sft.py --grad_accum stream
  python3 experimental/compile_repro_sft.py --grad_accum optax

Every flag also has an env-var default:
  MODEL_PATH MESH_FSDP MESH_TP GRAD_ACCUM GRAD_ACCUM_STEPS MAX_STEPS
GRAD_ACCUM_STEPS defaults to unset -> TrainingConfig.gradient_accumulation_steps
= None (accumulation depth 1), matching the notebook.
"""

import argparse
import os
import time

from absl import logging as absl_logging
import jax
import jax.numpy as jnp
import optax

from tunix.models.gemma4 import model
from tunix.models.gemma4 import params_safetensors
from tunix.sft import peft_trainer
from tunix.sft import utils


def parse_args():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument(
      "--model_path",
      default=os.environ.get(
          "MODEL_PATH", "/mnt/workspace/models/google/gemma-4-e2b-it"
      ),
      help="Local dir with the Gemma4-E2B-it HF safetensors.",
  )
  p.add_argument(
      "--mesh_fsdp", type=int, default=int(os.environ.get("MESH_FSDP", "2"))
  )
  p.add_argument(
      "--mesh_tp", type=int, default=int(os.environ.get("MESH_TP", "2"))
  )
  p.add_argument(
      "--grad_accum",
      choices=["optax", "stream"],
      default=os.environ.get("GRAD_ACCUM", "stream"),
      help="TrainingConfig.grad_accum: the only knob that differs per variant.",
  )
  p.add_argument(
      "--grad_accum_steps",
      type=int,
      default=(
          int(os.environ["GRAD_ACCUM_STEPS"])
          if os.environ.get("GRAD_ACCUM_STEPS")
          else None
      ),
      help=(
          "TrainingConfig.gradient_accumulation_steps. Default None"
          " (notebook parity: accumulation depth 1)."
      ),
  )
  p.add_argument(
      "--gradient_accumulator_dtype",
      choices=["float32", "bfloat16"],
      default=os.environ.get("GRAD_ACCUM_DTYPE", "float32"),
      help="TrainingConfig.gradient_accumulator_dtype",
  )
  p.add_argument(
      "--max_steps", type=int, default=int(os.environ.get("MAX_STEPS", "3"))
  )
  return p.parse_args()


# Same as sampling_example.ipynb cell-3.
def gen_model_input_fn(x: peft_trainer.TrainingInput):
  pad_mask = x.input_tokens != 0
  positions = utils.build_positions_from_mask(pad_mask)
  attention_mask = utils.make_causal_attn_mask(pad_mask)
  return {
      "input_tokens": x.input_tokens,
      "input_mask": x.input_mask,
      "positions": positions,
      "attention_mask": attention_mask,
  }


def main():
  args = parse_args()
  # The measurement lines (first-step elapsed, "Compiled train_step cache
  # size", "Train loop finished in") are absl INFO logs.
  absl_logging.set_verbosity(absl_logging.INFO)

  # Record whether JAX_LOG_COMPILES actually took effect: the config option
  # name can drift across JAX versions, and a stale env var is ignored
  # silently. The primary signal (train_wall_s below) does not depend on it.
  print(
      f"[[COMPILE_REPRO]] env jax={jax.__version__} "
      f"log_compiles={getattr(jax.config, 'jax_log_compiles', '<no such option>')}",
      flush=True,
  )

  mesh_shape = [(args.mesh_fsdp, args.mesh_tp), ("fsdp", "tp")]
  mesh = jax.make_mesh(
      *mesh_shape, axis_types=(jax.sharding.AxisType.Auto,) * len(mesh_shape[0])
  )

  # E2B config: verbatim from sampling_example.ipynb cell-1 ("e2b" branch).
  config = model.ModelConfig.gemma4_e2b()
  config.vision_encoder.output_length = 70
  config.use_flash_attention = True
  config.flash_attention_block_size = 256
  config.remat_config = model.RematConfig.BLOCK
  m = params_safetensors.create_model_from_safe_tensors(
      args.model_path, config, mesh, dtype=jnp.bfloat16, text_only=True
  )

  accum_dtype = (
      jnp.float32
      if args.gradient_accumulator_dtype == "float32"
      else jnp.bfloat16
  )
  training_config = peft_trainer.TrainingConfig(
      eval_every_n_steps=10,
      max_steps=args.max_steps,
      grad_accum=args.grad_accum,
      gradient_accumulation_steps=args.grad_accum_steps,
      gradient_accumulator_dtype=accum_dtype,
  )
  trainer = peft_trainer.PeftTrainer(
      m, optax.adamw(1e-5), training_config
  ).with_gen_model_input_fn(gen_model_input_fn)

  ds = [
      peft_trainer.TrainingInput(
          input_tokens=jnp.ones((4, 2048), dtype=jnp.int32),
          input_mask=jnp.ones((4, 2048), dtype=jnp.int32),
      )
      for _ in range(args.max_steps + 2)
  ]

  # Optional xprof capture (env-gated). With NO warmup, the first train_step's
  # Python tracing + XLA compilation happen inside this window, so xprof can
  # attribute both phases. Default host_tracer already captures the XLA compile
  # TraceMe (compile blocks / passes) — the compilation view py-spy cannot give
  # (goal B). To also capture Python tracing (set_value/.sharding, goal A),
  # raise the python tracer level; the exact API varies across jax versions, so
  # set it via jax.profiler.ProfileOptions / start_trace against the installed
  # jax, and keep MAX_STEPS=1 so the instrumenting tracer does not overflow the
  # ~2GB trace file.
  xprof_dir = os.environ.get("PROFILE_XPROF")
  t0 = time.perf_counter()
  if xprof_dir:
    os.makedirs(xprof_dir, exist_ok=True)
    with mesh, jax.profiler.trace(xprof_dir):
      trainer.train(ds, None)
    print(f"[[COMPILE_REPRO]] xprof trace -> {xprof_dir}", flush=True)
  else:
    with mesh:
      trainer.train(ds, None)
  print(
      f"[[COMPILE_REPRO]] grad_accum={args.grad_accum} "
      f"mesh={args.mesh_fsdp}x{args.mesh_tp} "
      f"train_wall_s={time.perf_counter() - t0:.1f}",
      flush=True,
  )

  # Independent HBM cross-check for the xprof memory profile: the XLA
  # allocator's own peak counter. Only meaningful with
  # XLA_PYTHON_CLIENT_PREALLOCATE=false (otherwise it reports the pool size).
  gib = float(2**30)
  for d in jax.local_devices():
    try:
      s = d.memory_stats()
      print(
          f"[[MEM]] grad_accum={args.grad_accum} "
          f"steps={args.grad_accum_steps} dtype={args.gradient_accumulator_dtype} "
          f"device={d.id} "
          f"peak_hbm_gb={s.get('peak_bytes_in_use', 0) / gib:.2f} "
          f"current_hbm_gb={s.get('bytes_in_use', 0) / gib:.2f} "
          f"limit_gb={s.get('bytes_limit', 0) / gib:.2f}",
          flush=True,
      )
    except Exception as e:  # pylint: disable=broad-except
      print(f"[[MEM]] memory_stats unavailable on {d}: {e}", flush=True)


if __name__ == "__main__":
  main()
