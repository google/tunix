"""Durable offline gate runner. Never mounts TPU devices or uses the network."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from examples.frozenlake.gemma4_tim import artifacts, recipe


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--image", action="store_true")
  parser.add_argument("--changed-base", default="HEAD",
                      help="Flag-audit diff base; defaults to the available checkout HEAD")
  parser.add_argument("--bench", type=Path, help="Optional external three-lane T0 bench entrypoint")
  args = parser.parse_args()
  args.output.mkdir(mode=0o700, parents=False, exist_ok=False)
  commands = [
      ("recipe", [sys.executable, "-m", "unittest", "discover", "-s", "tests/rl", "-p", "gemma4_tim_recipe_test.py", "-v"]),
      ("artifacts", [sys.executable, "-m", "unittest", "discover", "-s", "tests/rl", "-p", "gemma4_tim_artifacts_test.py", "-v"]),
      ("diff", ["git", "diff", "--check"]),
      ("flags", [sys.executable, "canon-zero-tim/.claude/skills/manage-canon-flags/scripts/audit_flag_registry.py",
                  "--repo", ".", "--changed-base", args.changed_base]),
  ]
  if args.bench:
    commands.append(("three-lane-T0", [sys.executable, str(args.bench), "--tier", "T0"]))
  if args.image:
    image_prefix = [
        "sudo", "docker", "run", "--rm", "--network", "none", "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--tmpfs", "/tmp:rw,size=1g", "--cpus", "4", "--memory", "8g",
        "-e", "JAX_PLATFORMS=cpu", "-e", "XLA_FLAGS=--xla_force_host_platform_device_count=4",
        "-e", "OPENBLAS_NUM_THREADS=1", "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-e", "WANDB_MODE=offline", "-e", "WANDB_DIR=/tmp",
        "-e", "WANDB_CACHE_DIR=/tmp/wandb-cache",
        "-e", "PYTHONPATH=/workspace", "--mount",
        f"type=bind,src={REPO},dst=/workspace,readonly", "-w", "/workspace",
        "--entrypoint", "python", recipe.IMAGE_ID,
    ]
    commands.append(("image", image_prefix + [
        "-m", "unittest", "discover", "-s", "tests/rl", "-p", "gemma4_tim_*test.py", "-v",
    ]))
    for name, directory, pattern in (
        ("existing-grpo", "tests/rl/agentic", "agentic_grpo_learner_test.py"),
        ("existing-canonical", "tests/rl", "canonical_forward_test.py"),
        ("existing-e4b", "tests/rl", "gemma4_e4b_admission_test.py"),
    ):
      commands.append((name, image_prefix + ["-m", "unittest", "discover", "-s", directory, "-p", pattern, "-v"]))
  results = []
  for name, command in commands:
    print(f"GEMMA4_E2B_CPU_GATE_START gate={name}", flush=True)
    with (args.output / (name + ".log")).open("xb") as log:
      result = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=False)
    results.append({"gate": name, "rc": result.returncode})
    print(f"GEMMA4_E2B_CPU_GATE_RESULT gate={name} rc={result.returncode}", flush=True)
    if result.returncode:
      break
  # Bind every changed source file, including new untracked implementation.
  names = subprocess.check_output(["git", "ls-files", "--modified", "--others", "--exclude-standard"], cwd=REPO, text=True).splitlines()
  artifacts.write_json(args.output / "changed-source.json", {
      "base": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
      "files": {n: artifacts.file_sha(REPO / n) for n in sorted(set(names)) if (REPO / n).is_file()},
  })
  verdict = {"status": "GREEN" if all(r["rc"] == 0 for r in results) else "RED",
             "results": results, "claim": "CPU-construction-only", "tpu": "NOT_RUN"}
  sha = artifacts.seal_run(args.output, verdict)
  print(f"GEMMA4_E2B_CPU_GATE_SEALED status={verdict['status']} sha256={sha}", flush=True)
  return 0 if verdict["status"] == "GREEN" else 1


if __name__ == "__main__":
  sys.exit(main())
