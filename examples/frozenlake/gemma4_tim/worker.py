"""Child process for an explicitly authorized direct-TPU run; not a launcher."""

import argparse
import json
from pathlib import Path

from examples.frozenlake.gemma4_tim import artifacts, recipe


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ("snapshot", "data", "output"):
    parser.add_argument("--" + name, required=True, type=Path)
  args = parser.parse_args()
  contract = json.loads((args.output / "contract.json").read_text())
  recipe.validate_resolved(contract)
  if contract["arm"] == "zero":
    raise recipe.RecipeError("Zero-TIM engine integration is not yet implemented")
  import jax
  devices = jax.devices()
  if (jax.process_count() != 1 or jax.local_device_count() != 4 or len(devices) != 4
      or any(d.platform != "tpu" or "v5p" not in d.device_kind.lower() for d in devices)):
    raise recipe.RecipeError("target execution requires one host with four local v5p devices")
  artifacts.write_json(args.output / "devices.json", {
      "jax": jax.__version__, "devices": [d.id for d in devices],
      "kinds": [d.device_kind for d in devices], "processes": jax.process_count(),
  })
  from examples.frozenlake.gemma4_tim.pipeline import GemmaPipeline
  pipeline = GemmaPipeline(contract, snapshot=args.snapshot, data=args.data, output=args.output)
  result = pipeline._run()
  artifacts.write_json(args.output / "worker-result.json", result)
  print("GEMMA4_E2B_WORKER_COMPLETE", flush=True)


if __name__ == "__main__":
  main()
