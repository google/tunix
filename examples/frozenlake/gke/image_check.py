"""Sanity checks for the SC A/B image, run by cloudbuild.yaml on CPU.

Prints the key package versions and imports everything
examples/frozenlake/train_frozenlake_qwen3.py needs, so a missing dependency
fails the build instead of a TPU job. vLLM / tpu-inference are only reported:
they are imported lazily by the rollout engine and may need a TPU.
"""

import importlib
from importlib import metadata
import sys

sys.path.insert(0, "/app")

for pkg in (
    "jax", "jaxlib", "libtpu", "flax", "vllm", "tpu-inference",
    "pathwaysutils", "orbax-checkpoint", "wandb", "gcsfs", "gymnasium",
):
  try:
    print(f"{pkg} {metadata.version(pkg)}")
  except metadata.PackageNotFoundError:
    print(f"{pkg} not installed")

REQUIRED = [
    "absl.logging", "datasets", "flax.nnx", "fsspec", "gcsfs",
    "google.cloud.storage", "grain", "gymnasium", "huggingface_hub", "jax",
    "metrax.logging", "numpy", "optax", "orbax.checkpoint", "pandas",
    "pathwaysutils", "qwix", "tensorboardX", "transformers", "wandb",
    "tunix.cli.utils.data",
    "tunix.models.qwen3.model",
    "tunix.models.qwen3.params",
    "tunix.rl.agentic.agentic_grpo_learner",
    "tunix.rl.agentic.parser.chat_template_parser.parser",
    "tunix.rl.rl_cluster",
    "tunix.rl.rollout.base_rollout",
    "tunix.sft.metrics_logger",
    "tunix.sft.utils",
    "examples.frozenlake.agent",
    "examples.frozenlake.env",
]
OPTIONAL = ["vllm", "tpu_inference", "tunix.rl.rollout.vllm_rollout"]

failed = []
for name in REQUIRED:
  try:
    importlib.import_module(name)
  except Exception as e:  # pylint: disable=broad-except
    failed.append(f"{name}: {e!r}")
for name in OPTIONAL:
  try:
    importlib.import_module(name)
    print(f"optional import ok: {name}")
  except Exception as e:  # pylint: disable=broad-except
    print(f"optional import failed: {name}: {e!r}"[:300])

if failed:
  print("required imports FAILED:\n  " + "\n  ".join(failed))
  sys.exit(1)
print(f"all {len(REQUIRED)} required imports ok")
