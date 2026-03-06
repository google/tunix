import os
import sys

# Add google3 to sys.path if needed, or assume we run from google3 root
# Actually, we likely need to run this with adequate PYTHONPATH.

from absl import app
from flax import nnx
import jax
import jax.numpy as jnp
import qwix
from tunix.models.dummy_model_creator import create_dummy_model
from tunix.models.llama3 import model as llama_lib


def main(_):
  # Configuration
  rank = 64
  alpha = 64.0
  num_devices = min(jax.device_count(), 4)
  print(f"Num devices: {num_devices}")

  # Create mesh
  mesh = jax.make_mesh(
      (1, num_devices),
      ("fsdp", "tp"),
      devices=jax.devices()[:num_devices],
      axis_types=(jax.sharding.AxisType.Auto,) * 2,
  )

  model_config = llama_lib.ModelConfig.llama3p2_1b()

  with mesh:
    base_model = create_dummy_model(
        model_class=llama_lib.Llama3,
        config=model_config,
        mesh=mesh,
        dtype=jnp.bfloat16,
        random_seed=3,
    )

    # Apply LoRA to gate_proj
    lora_provider = qwix.LoraProvider(
        module_path=".*gate_proj",
        rank=rank,
        alpha=alpha,
    )

    model_input = base_model.get_model_input()
    # Pass input_tokens etc explicitly or unpack
    lora_model = qwix.apply_lora_to_model(
        base_model, lora_provider, **model_input, rngs=nnx.Rngs(params=0)
    )

    _, trainer_state = nnx.split(lora_model)

    flatten_trainer_state = nnx.to_flat_state(trainer_state)

    print("\nDumping keys in flatten_trainer_state:")
    found_gate_proj = False
    for keys, param in flatten_trainer_state:
      path = ".".join(str(key) for key in keys)
      if "gate_proj" in path:
        print(f"MATCH: {path}")
        found_gate_proj = True
      else:
        # Print a few non-matches just to see structure
        if "layers.0" in path:
          print(f"OTHER: {path}")

    if not found_gate_proj:
      print("\nFAIL: No gate_proj found in keys!")
    else:
      print("\nSUCCESS: Found gate_proj keys.")


if __name__ == "__main__":
  app.run(main)
