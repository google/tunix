"""CPU-safe plan entrypoint; execution is always an explicit operator action."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from examples.frozenlake.gemma4_tim import artifacts, recipe


class Once(argparse.Action):
  def __call__(self, parser, namespace, values, option_string=None):
    seen = getattr(namespace, "_seen", set())
    if self.dest in seen:
      parser.error("duplicate option: " + option_string)
    seen.add(self.dest)
    namespace._seen = seen
    setattr(namespace, self.dest, values)


def require_clean_source(expected):
  if not expected or not re.fullmatch(r"[0-9a-f]{40}", expected):
    raise recipe.RecipeError("execution requires an exact full source commit")
  def git(*args):
    return subprocess.check_output(["git", "-C", str(recipe.REPO), *args], text=True).strip()
  if git("rev-parse", "HEAD") != expected or git("status", "--porcelain", "--untracked-files=all"):
    raise recipe.RecipeError("execution requires the exact clean source commit")


def execute(args, contract):
  require_clean_source(args.source_sha)
  for name in ("snapshot", "data", "input_manifest", "input_sha", "output"):
    if getattr(args, name) is None:
      raise recipe.RecipeError("execution missing --" + name.replace("_", "-"))
  snapshot, data, manifest, output = map(Path, (args.snapshot, args.data, args.input_manifest, args.output))
  if not all(p.is_absolute() for p in (snapshot, data, manifest, output)):
    raise recipe.RecipeError("execution paths must be absolute")
  if output.resolve().is_relative_to(recipe.REPO):
    raise recipe.RecipeError("evidence output must be outside the clean source tree")
  inputs = artifacts.verify_inputs(manifest, args.input_sha, snapshot, data)
  recipe.require_versions()
  from examples.frozenlake.gemma4_tim import data_contract
  data_receipt = data_contract.validate_training_data(data)
  import importlib.util
  engine = Path(importlib.util.find_spec("tpu_inference").origin).parent / "models/jax/gemma4.py"
  if artifacts.file_sha(engine) != "5b93b3b3dd5d42374c0115a4523661ed584191fb0cb9fc9639ae2248b3896fbd":
    raise recipe.RecipeError("installed Gemma source drift")
  output.mkdir(mode=0o700, parents=False, exist_ok=False)
  artifacts.write_json(output / "contract.json", contract)
  artifacts.write_json(output / "inputs.json", inputs)
  artifacts.write_json(output / "data-validation.json", data_receipt)
  artifacts.write_json(output / "source.json", {"commit": args.source_sha, "clean": True})
  worker_args = [sys.executable, "-m", "examples.frozenlake.gemma4_tim.worker",
                 "--snapshot", str(snapshot), "--data", str(data), "--output", str(output)]
  with (output / "worker.log").open("xb") as log:
    result = subprocess.run(worker_args, cwd=recipe.REPO, stdout=log, stderr=subprocess.STDOUT,
                            check=False)
  # Worker and its atexit writers have exited before any file is hashed.
  result_file = output / "worker-result.json"
  verdict = {"status": "INCONCLUSIVE", "worker_rc": result.returncode,
             "claim": "worker-did-not-produce-complete-receipt"}
  if result.returncode == 0 and result_file.is_file():
    verdict = json.loads(result_file.read_text())
  sha = artifacts.seal_run(output, verdict)
  print(f"GEMMA4_E2B_SEALED status={verdict['status']} manifest_sha256={sha}", flush=True)
  return 0 if verdict["status"] == "GREEN" else 1


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--arm", required=True, choices=recipe.ARMS, action=Once)
  parser.add_argument("--stage", choices=recipe.STAGES, default="train", action=Once)
  parser.add_argument("--execute", action="store_true")
  for name in ("source-sha", "snapshot", "data", "input-manifest", "input-sha", "output"):
    parser.add_argument("--" + name, action=Once)
  args = parser.parse_args(argv)
  recipe.require_isolated_environment(os.environ)
  contract = recipe.resolve(args.arm, stage=args.stage)
  if args.execute:
    return execute(args, contract)
  print(json.dumps(contract, sort_keys=True, indent=2))
  print("GEMMA4_E2B_PLAN_PASS execution=0 target=NOT_RUN")


if __name__ == "__main__":
  sys.exit(main())
